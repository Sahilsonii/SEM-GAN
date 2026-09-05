"""
Open-set evaluation (stage 8, H6): does the detector know when it doesn't know?

Requires a detector trained closed-set (known_classes=(1,), pinhole only - see
train_detector.py --known-classes 1). PbI2 is then never trained on; it is the
unknown morphology, held out by yolo_export.build and recorded in every export
manifest's held_out_images.

Two measurements, because the corpus only supports one of them properly.

IMAGE level (the design in the plan): per-image MAX detection confidence,
positive = an image with at least one pinhole box, negative = a PbI2-only image.
AUROC, AUPR and FPR@95TPR. The catch, found by counting: this corpus has ONE
PbI2-only image in val and THREE in test. Those numbers are reported with n
beside them and a caveat string, and they support no claim at this size.

BOX level (added for Phase 2): run the detector at an operating threshold and
ask, for every ground-truth box, whether any detection overlaps it at IoU>=0.5.
On the 596 PbI2 boxes in test that is the false-alarm rate on the unknown
class; on the 735 pinhole boxes it is the recall on the known class. Hundreds
of samples instead of three. This is the number that means something.

Neither needs an EDL head - both work on any trained ultralytics checkpoint.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data"))

from eval.detection import iou_matrix, load_net, xywhn_to_xyxy  # noqa: E402

CURATED_IMAGES = ROOT / "data" / "curated" / "images"
SPLITS = ROOT / "data" / "splits"
OUT = ROOT / "outputs"
EXPORT = ROOT / "data" / "yolo" / "openset_probe"
PBI2, PINHOLE = 0, 1
MIN_UNKNOWN_FOR_CLAIM = 10


def _roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC via rank statistic - no sklearn dependency for this one number."""
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    sum_ranks_pos = ranks[labels == 1].sum()
    return float((sum_ranks_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _pr_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Average precision via the step-wise formula (same convention as COCO AP):
    AP = sum_n (R_n - R_{n-1}) * P_n. Plain trapz(precision, recall) undercounts
    the region before the first positive is reached - verified on a perfectly
    separable toy case, where it reported 0.75 instead of 1.0.
    """
    order = np.argsort(-scores)
    y = labels[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(y.sum(), 1)
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - recall_prev) * precision))


def _fpr_at_tpr(scores: np.ndarray, labels: np.ndarray, target_tpr: float = 0.95) -> float:
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    thresh = np.percentile(pos, 100 * (1 - target_tpr))   # score >= thresh keeps target_tpr of positives
    return float((neg >= thresh).mean())


def _ensure_export(split: str) -> dict:
    """The openset_probe export, rebuilt if missing or if it lacks the split.

    Training builds it without test. The manifest is the source of truth for
    which images are held-out unknowns, so a test-split evaluation needs the
    export rebuilt with include_test=True - the old code only checked that
    *a* manifest existed and would report n_unknown=0 on test forever.
    """
    path = EXPORT / "export_manifest.json"
    ok = path.exists()
    if ok and split == "test":
        ok = "test" in json.loads(path.read_text(encoding="utf-8"))["held_out_images"]
    if not ok:
        import yolo_export
        yolo_export.build(regime="openset_probe", known_classes=(PINHOLE,),
                          include_test=(split == "test"))
    return json.loads(path.read_text(encoding="utf-8"))


def _records(split: str) -> list[dict]:
    return json.loads((SPLITS / f"{split}.json").read_text(encoding="utf-8"))["records"]


def split_images(split: str):
    """(known image paths, unknown image paths) for the image-level score.

    Known = an image carrying at least one pinhole box. Backgrounds are NOT
    known positives: a background scores 0.0 by construction and labelling 51
    of them positive would put half the positives at the bottom of the ranking
    and crush AUROC for a reason that has nothing to do with open-set ability.
    """
    export = _ensure_export(split)
    known = [CURATED_IMAGES / r["file"] for r in _records(split)
             if any(b[0] == PINHOLE for b in r["boxes"])]
    unknown = [CURATED_IMAGES / f for f in export["held_out_images"].get(split, [])]
    return known, unknown


def image_level(net, split: str, conf_floor: float = 0.001) -> dict:
    known_files, unknown_files = split_images(split)

    def max_conf(paths):
        out = []
        for p in paths:
            img = cv2.imread(str(p))
            if img is None:
                continue
            r = net.predict(img, conf=conf_floor, verbose=False)[0]
            out.append(float(r.boxes.conf.max().item()) if len(r.boxes) else 0.0)
        return np.array(out, dtype=float)

    known, unknown = max_conf(known_files), max_conf(unknown_files)
    scores = np.concatenate([known, unknown])
    labels = np.concatenate([np.ones(len(known)), np.zeros(len(unknown))])
    res = {
        "n_known": int(len(known)), "n_unknown": int(len(unknown)),
        "known_mean_conf": round(float(known.mean()), 4) if len(known) else None,
        "unknown_mean_conf": round(float(unknown.mean()), 4) if len(unknown) else None,
        "auroc": round(_roc_auc(scores, labels), 4),
        "aupr": round(_pr_auc(scores, labels), 4),
        "fpr_at_95tpr": round(_fpr_at_tpr(scores, labels), 4),
    }
    if len(unknown) < MIN_UNKNOWN_FOR_CLAIM:
        res["caveat"] = (f"only {len(unknown)} unknown-only image(s) in {split}; "
                         f"AUROC/AUPR/FPR are not interpretable at this n - "
                         f"use the box-level numbers")
    return res


THRESHOLDS = (0.05, 0.10, 0.25)


def box_level(net, split: str, iou_thr: float = 0.5,
              conf_floor: float = 0.001, thresholds=THRESHOLDS) -> dict:
    """Per ground-truth-box score: the confidence of the best detection that
    overlaps it at IoU >= iou_thr, or 0 if nothing does.

    Every one of the 735 pinhole and 596 PbI2 boxes in test gets a score, so
    the open-set question becomes an AUROC over 1,331 samples - pinhole boxes
    positive, PbI2 boxes negative - instead of three images. Rates at a few
    thresholds are kept beside it because AUROC alone hides the operating
    point: a first draft reported a 0.0 false-alarm rate at conf=0.25 that
    looked like perfect rejection and was actually a detector that fired on
    2% of pinholes too. Silence is not discrimination.
    """
    scores = {PBI2: [], PINHOLE: []}
    n_images = 0
    for r in _records(split):
        if r["n_boxes"] == 0:
            continue
        img = cv2.imread(str(CURATED_IMAGES / r["file"]))
        if img is None:
            continue
        n_images += 1
        H, W = img.shape[:2]
        res = net.predict(img, conf=conf_floor, verbose=False)[0]
        if len(res.boxes):
            pred = res.boxes.xyxy.cpu().numpy().astype(np.float32)
            conf = res.boxes.conf.cpu().numpy().astype(np.float32)
        else:
            pred, conf = np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
        gt = np.stack([xywhn_to_xyxy(b, W, H) for b in r["boxes"]])
        ious = iou_matrix(pred, gt)                     # (n_pred, n_gt)
        for j, b in enumerate(r["boxes"]):
            over = ious[:, j] >= iou_thr if len(pred) else np.zeros(0, bool)
            scores[b[0]].append(float(conf[over].max()) if over.any() else 0.0)

    pin, pb = np.array(scores[PINHOLE]), np.array(scores[PBI2])
    allsc = np.concatenate([pin, pb])
    labels = np.concatenate([np.ones(len(pin)), np.zeros(len(pb))])
    out = {
        "iou_thr": iou_thr, "images": n_images,
        "n_pinhole_boxes": int(len(pin)), "n_pbi2_boxes": int(len(pb)),
        "box_auroc": round(_roc_auc(allsc, labels), 4),
        "box_fpr_at_95tpr": round(_fpr_at_tpr(allsc, labels, 0.95), 4),
        "box_fpr_at_50tpr": round(_fpr_at_tpr(allsc, labels, 0.50), 4),
        "at_threshold": {},
    }
    for t in thresholds:
        out["at_threshold"][str(t)] = {
            "known_box_recall": round(float((pin >= t).mean()), 4) if len(pin) else None,
            "unknown_box_false_alarm_rate": round(float((pb >= t).mean()), 4) if len(pb) else None,
        }
    return out


def evaluate(checkpoint: str, split: str = "val", save: bool = True,
             tag: str | None = None) -> dict:
    """tag distinguishes seeds in the output filename; without it every seed
    overwrites open_set_<split>.json."""
    net = load_net(checkpoint)
    result = {"checkpoint": checkpoint, "split": split,
              "image_level": image_level(net, split),
              "box_level": box_level(net, split)}
    if save:
        name = f"open_set_{split}{'_' + tag if tag else ''}.json"
        (OUT / name).write_text(json.dumps(result, indent=1), encoding="utf-8")
    im, bx = result["image_level"], result["box_level"]
    print(f"[open-set] image-level  known n={im['n_known']} mean_conf={im['known_mean_conf']} | "
          f"unknown(pbi2) n={im['n_unknown']} mean_conf={im['unknown_mean_conf']}")
    print(f"[open-set]   AUROC={im['auroc']}  AUPR={im['aupr']}  FPR@95TPR={im['fpr_at_95tpr']}"
          + (f"\n[open-set]   CAVEAT: {im['caveat']}" if "caveat" in im else ""))
    print(f"[open-set] box-level  {bx['n_pinhole_boxes']} pinhole vs {bx['n_pbi2_boxes']} PbI2 boxes: "
          f"AUROC={bx['box_auroc']}  FPR@95TPR={bx['box_fpr_at_95tpr']}  FPR@50TPR={bx['box_fpr_at_50tpr']}")
    for t, v in bx["at_threshold"].items():
        print(f"[open-set]   @{t}: known recall={v['known_box_recall']}  "
              f"unknown false-alarm={v['unknown_box_false_alarm_rate']}")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True,
                    help="path to a .pt trained with --known-classes 1")
    ap.add_argument("--split", default="val")
    ap.add_argument("--tag", default=None,
                    help="suffix for the output file, e.g. seed1")
    a = ap.parse_args()
    evaluate(a.checkpoint, split=a.split, tag=a.tag)
