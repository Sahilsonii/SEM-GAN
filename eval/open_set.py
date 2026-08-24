"""
Open-set evaluation (stage 8, H6): does the detector know when it doesn't know?

Requires a detector trained closed-set (known_classes=(1,), pinhole only - see
train_detector.py --known-classes 1). PbI2 is then never trained on; it is the
unknown morphology, held out by yolo_export.build and recorded in every export
manifest's held_out_images.

Signal used: per-image MAX detection confidence. A detector that has genuinely
learned "pinhole" rather than "anything textured" should fire confidently on
real pinhole images and stay quiet on PbI2-only images, since PbI2 (bright
particles) and pinholes (dark pits) are visually opposite. This does not
require an EDL head - it is available for any trained ultralytics checkpoint.

Metrics: AUROC and AUPR treating "is a known-class image" as positive, plus
FPR@95TPR - at the confidence threshold that keeps 95% of true pinhole
detections, what fraction of unknown-only images still fire a confident false
detection. FPR@95TPR is the operationally meaningful one: it is the false-alarm
rate an inspector would actually see at a usable recall.
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

CURATED_IMAGES = ROOT / "data" / "curated" / "images"
SPLITS = ROOT / "data" / "splits"
OUT = ROOT / "outputs"


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


def collect_scores(checkpoint: str, split: str = "val", conf_floor: float = 0.001):
    """Run a trained detector over known-class and held-out-unknown images."""
    from ultralytics import YOLO

    net = YOLO(checkpoint)
    export_path = ROOT / "data" / "yolo" / "openset_probe" / "export_manifest.json"
    if not export_path.exists():
        import yolo_export
        yolo_export.build(regime="openset_probe", known_classes=(1,))
    export = json.loads(export_path.read_text(encoding="utf-8"))

    known_dir = ROOT / "data" / "yolo" / "openset_probe" / "images" / split
    known_files = sorted(known_dir.glob("*")) if known_dir.exists() else []

    unknown_files = [CURATED_IMAGES / f for f in export["held_out_images"].get(split, [])]

    def max_conf(paths):
        scores = []
        for p in paths:
            img = cv2.imread(str(p))
            if img is None:
                continue
            r = net.predict(img, conf=conf_floor, verbose=False)[0]
            scores.append(float(r.boxes.conf.max().item()) if len(r.boxes) else 0.0)
        return np.array(scores, dtype=float)

    known_scores = max_conf(known_files)
    unknown_scores = max_conf(unknown_files)
    return known_scores, unknown_scores


def evaluate(checkpoint: str, split: str = "val", save: bool = True) -> dict:
    known, unknown = collect_scores(checkpoint, split)
    scores = np.concatenate([known, unknown])
    labels = np.concatenate([np.ones(len(known)), np.zeros(len(unknown))])

    result = {
        "checkpoint": checkpoint, "split": split,
        "n_known": len(known), "n_unknown": len(unknown),
        "known_mean_conf": round(float(known.mean()), 4) if len(known) else None,
        "unknown_mean_conf": round(float(unknown.mean()), 4) if len(unknown) else None,
        "auroc": round(_roc_auc(scores, labels), 4),
        "aupr": round(_pr_auc(scores, labels), 4),
        "fpr_at_95tpr": round(_fpr_at_tpr(scores, labels), 4),
    }
    if save:
        (OUT / f"open_set_{split}.json").write_text(json.dumps(result, indent=1),
                                                     encoding="utf-8")
    print(f"[open-set] known n={result['n_known']} mean_conf={result['known_mean_conf']} | "
          f"unknown(pbi2) n={result['n_unknown']} mean_conf={result['unknown_mean_conf']}")
    print(f"[open-set] AUROC={result['auroc']}  AUPR={result['aupr']}  "
          f"FPR@95TPR={result['fpr_at_95tpr']}")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True,
                    help="path to a .pt trained with --known-classes 1")
    ap.add_argument("--split", default="val")
    a = ap.parse_args()
    evaluate(a.checkpoint, split=a.split)
