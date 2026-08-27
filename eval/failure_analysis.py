"""
Failure analysis (plan section 15) - categories declared BEFORE looking.

The plan pre-declares the categories so the taxonomy cannot be reverse-fitted
to whatever the model happened to get wrong. Each miss and each false positive
is assigned to exactly one bucket, counted, and a few representative crops are
written to outputs/failures/<category>/ so the write-up can show them.

Runs on VAL. The frequency table is the deliverable; the crops are evidence.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.detection import iou_matrix, xywhn_to_xyxy
from eval.tiny_defect import assign_bin, load_bins

CURATED = ROOT / "data" / "curated" / "images"
SPLITS = ROOT / "data" / "splits"
OUT = ROOT / "outputs"
FAIL_DIR = OUT / "failures"

# Pre-declared: fixed before any prediction was inspected (plan section 15).
CATEGORIES = [
    "missed_sub_stride",       # GT in T1 (<8px) not detected - the known hard floor
    "missed_tiny",             # GT in T2 (8-16px) not detected
    "missed_low_contrast",     # GT missed and its local contrast is bottom-quartile
    "missed_dense_cluster",    # GT missed with many neighbours nearby
    "missed_other",            # GT missed, none of the above
    "fp_on_grain_boundary",    # false positive sitting on a Canny ridge
    "fp_high_contrast_spot",   # false positive on a locally bright/dark blob
    "fp_near_miss",            # false positive within 2x IoU-radius of a real GT
    "fp_other",
]


def _local_contrast(gray, box_xyxy):
    x0, y0, x1, y1 = [int(v) for v in box_xyxy]
    H, W = gray.shape
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x1), min(H, y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inner = gray[y0:y1, x0:x1].astype(np.float32).mean()
    pad = max(6, (x1 - x0))
    rx0, ry0 = max(0, x0 - pad), max(0, y0 - pad)
    rx1, ry1 = min(W, x1 + pad), min(H, y1 + pad)
    ring = gray[ry0:ry1, rx0:rx1].astype(np.float32).mean()
    return abs(inner - ring)


def analyse(checkpoint: str, conf: float = 0.10, iou_thr: float = 0.5,
            device: str = "0", crops_per_cat: int = 4) -> dict:
    from ultralytics import YOLO

    if FAIL_DIR.exists():
        shutil.rmtree(FAIL_DIR)
    for c in CATEGORIES:
        (FAIL_DIR / c).mkdir(parents=True, exist_ok=True)

    ref_px, bins = load_bins()
    net = YOLO(checkpoint)
    recs = [r for r in json.loads((SPLITS / "val.json").read_text(encoding="utf-8"))["records"]
            if r["n_boxes"] > 0]

    counts = Counter()
    saved = Counter()
    total_gt = total_fp = 0

    for r in recs:
        p = CURATED / r["file"]
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if im is None:
            continue
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape
        edges = cv2.dilate(cv2.Canny(gray, 30, 80), np.ones((5, 5), np.uint8))

        gt = r["boxes"]
        gt_xyxy = (np.stack([xywhn_to_xyxy(b, W, H) for b in gt])
                   if gt else np.zeros((0, 4), np.float32))
        res = net.predict(im, conf=conf, verbose=False, device=device)[0]
        pred, scores = [], []
        if len(res.boxes):
            for (cx, cy, bw, bh), c, s in zip(res.boxes.xywhn.cpu().numpy(),
                                              res.boxes.cls.cpu().numpy().astype(int),
                                              res.boxes.conf.cpu().numpy()):
                pred.append([int(c), float(cx), float(cy), float(bw), float(bh)])
                scores.append(float(s))
        pr_xyxy = (np.stack([xywhn_to_xyxy(b, W, H) for b in pred])
                   if pred else np.zeros((0, 4), np.float32))
        ious = iou_matrix(pr_xyxy, gt_xyxy)

        matched_gt = set()
        matched_pred = set()
        order = np.argsort(-np.array(scores)) if scores else []
        for pi in order:
            best, bj = 0.0, -1
            for gj in range(len(gt)):
                if gj in matched_gt or gt[gj][0] != pred[pi][0]:
                    continue
                if ious[pi, gj] > best:
                    best, bj = ious[pi, gj], gj
            if best >= iou_thr:
                matched_gt.add(bj)
                matched_pred.add(pi)

        # --- missed ground truth ---
        contrasts = [_local_contrast(gray, gt_xyxy[i]) for i in range(len(gt))]
        q1 = np.percentile(contrasts, 25) if contrasts else 0.0
        for gj, b in enumerate(gt):
            total_gt += 1
            if gj in matched_gt:
                continue
            bin_name = assign_bin(b[3], b[4], ref_px, bins)
            # neighbour density: how many other GT centres lie within 3 box-widths
            cx, cy = b[1] * W, b[2] * H
            near = sum(1 for o in gt
                       if o is not b and abs(o[1] * W - cx) < 3 * b[3] * W
                       and abs(o[2] * H - cy) < 3 * b[4] * H)
            if bin_name == "T1_sub_stride":
                cat = "missed_sub_stride"
            elif bin_name == "T2_tiny":
                cat = "missed_tiny"
            elif contrasts[gj] <= q1:
                cat = "missed_low_contrast"
            elif near >= 4:
                cat = "missed_dense_cluster"
            else:
                cat = "missed_other"
            counts[cat] += 1
            if saved[cat] < crops_per_cat:
                _save_crop(im, gt_xyxy[gj], FAIL_DIR / cat,
                           f"{Path(r['file']).stem}_gt{gj}", (0, 0, 255))
                saved[cat] += 1

        # --- false positives ---
        for pi in range(len(pred)):
            if pi in matched_pred:
                continue
            total_fp += 1
            x0, y0, x1, y1 = [int(v) for v in pr_xyxy[pi]]
            cxi, cyi = np.clip((x0 + x1) // 2, 0, W - 1), np.clip((y0 + y1) // 2, 0, H - 1)
            on_edge = edges[cyi, cxi] > 0
            ctr = _local_contrast(gray, pr_xyxy[pi])
            nearest = float(ious[pi].max()) if len(gt) else 0.0
            if nearest > 0.1:
                cat = "fp_near_miss"
            elif on_edge:
                cat = "fp_on_grain_boundary"
            elif ctr > 15:
                cat = "fp_high_contrast_spot"
            else:
                cat = "fp_other"
            counts[cat] += 1
            if saved[cat] < crops_per_cat:
                _save_crop(im, pr_xyxy[pi], FAIL_DIR / cat,
                           f"{Path(r['file']).stem}_fp{pi}", (0, 200, 255))
                saved[cat] += 1

    out = {"checkpoint": checkpoint, "conf_threshold": conf, "iou_threshold": iou_thr,
           "images": len(recs), "total_gt": total_gt, "total_false_positives": total_fp,
           "counts": {c: counts.get(c, 0) for c in CATEGORIES},
           "crops_dir": str(FAIL_DIR)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "failure_analysis.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

    print(f"[failures] {len(recs)} val images, {total_gt} GT, {total_fp} false positives")
    miss_tot = sum(counts.get(c, 0) for c in CATEGORIES if c.startswith("missed"))
    for c in CATEGORIES:
        n = counts.get(c, 0)
        denom = miss_tot if c.startswith("missed") else total_fp
        pct = (100 * n / denom) if denom else 0.0
        print(f"  {c:24} {n:>6}  ({pct:5.1f}% of {'misses' if c.startswith('missed') else 'FPs'})")
    print(f"  crops -> {FAIL_DIR}")
    return out


def _save_crop(im, xyxy, dest: Path, name: str, colour, pad: int = 40):
    H, W = im.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in xyxy]
    cx0, cy0 = max(0, x0 - pad), max(0, y0 - pad)
    cx1, cy1 = min(W, x1 + pad), min(H, y1 + pad)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    crop = im[cy0:cy1, cx0:cx1].copy()
    cv2.rectangle(crop, (x0 - cx0, y0 - cy0), (x1 - cx0, y1 - cy0), colour, 1)
    if crop.shape[0] < 60:
        crop = cv2.resize(crop, (crop.shape[1] * 3, crop.shape[0] * 3),
                          interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(dest / f"{name}.png"), crop)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--conf", type=float, default=0.10)
    a = ap.parse_args()
    analyse(a.checkpoint, conf=a.conf)
