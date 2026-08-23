"""
Detection metrics, scale-stratified.

Deliberately dependency-free (numpy only) so the classical baseline can be
scored with exactly the same code path as the deep detectors - if the two were
measured by different implementations, the comparison would be worthless.

Follows the standard COCO protocol: greedy matching of predictions to ground
truth in descending score order, one GT per prediction, AP by 101-point
interpolation of the precision-recall curve.
"""
from __future__ import annotations

import numpy as np

from eval.tiny_defect import assign_bin, load_bins


def xywhn_to_xyxy(box, w: int, h: int):
    _, cx, cy, bw, bh = box
    return np.array([(cx - bw / 2) * w, (cy - bh / 2) * h,
                     (cx + bw / 2) * w, (cy + bh / 2) * h], dtype=np.float32)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (N,4), b: (M,4) in xyxy -> (N,M)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return (inter / np.maximum(union, 1e-9)).astype(np.float32)


def _ap_from_pr(tp: np.ndarray, scores: np.ndarray, n_gt: int) -> float:
    """101-point interpolated AP."""
    if n_gt == 0 or len(tp) == 0:
        return 0.0
    order = np.argsort(-scores)
    tp = tp[order]
    ctp = np.cumsum(tp)
    cfp = np.cumsum(1 - tp)
    recall = ctp / n_gt
    precision = ctp / np.maximum(ctp + cfp, 1e-9)
    # make precision monotonically decreasing
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    grid = np.linspace(0, 1, 101)
    idx = np.searchsorted(recall, grid, side="left")
    out = np.where(idx < len(precision), precision[np.clip(idx, 0, len(precision) - 1)], 0.0)
    return float(out.mean())


def evaluate(samples: list[dict], iou_thresholds=None, score_key: str = "scores") -> dict:
    """Score a list of per-image {gt, pred, scores, wh} dicts.

    gt/pred are YOLO-normalised [cls, cx, cy, w, h]. Returns overall and
    per-scale-bin metrics.
    """
    if iou_thresholds is None:
        iou_thresholds = np.round(np.arange(0.50, 0.96, 0.05), 2)
    ref_px, bins = load_bins()
    bin_names = [b["name"] for b in bins]

    per_thr = {}
    for thr in iou_thresholds:
        recs, n_gt_total = [], 0
        n_gt_bin = {n: 0 for n in bin_names}
        for s in samples:
            w, h = s["wh"]
            gt = s["gt"]
            pred = s["pred"]
            scores = np.asarray(s.get(score_key, np.ones(len(pred))), dtype=np.float32)

            gt_xyxy = np.stack([xywhn_to_xyxy(b, w, h) for b in gt]) if gt else np.zeros((0, 4), np.float32)
            pr_xyxy = np.stack([xywhn_to_xyxy(b, w, h) for b in pred]) if pred else np.zeros((0, 4), np.float32)
            gt_bins = [assign_bin(b[3], b[4], ref_px, bins) for b in gt]
            n_gt_total += len(gt)
            for bn in gt_bins:
                n_gt_bin[bn] += 1

            ious = iou_matrix(pr_xyxy, gt_xyxy)
            taken = np.zeros(len(gt), bool)
            order = np.argsort(-scores) if len(scores) else []
            for pi in order:
                best, best_j = 0.0, -1
                for gj in range(len(gt)):
                    if taken[gj] or gt[gj][0] != pred[pi][0]:
                        continue
                    if ious[pi, gj] > best:
                        best, best_j = ious[pi, gj], gj
                hit = best >= thr
                if hit:
                    taken[best_j] = True
                recs.append((float(scores[pi]), 1.0 if hit else 0.0,
                             gt_bins[best_j] if hit else None,
                             assign_bin(pred[pi][3], pred[pi][4], ref_px, bins)))

            for gj in range(len(gt)):
                if not taken[gj]:
                    recs.append(None)  # placeholder: unmatched GT affects recall via n_gt

        recs = [r for r in recs if r is not None]
        if recs:
            sc = np.array([r[0] for r in recs], np.float32)
            tp = np.array([r[1] for r in recs], np.float32)
        else:
            sc = tp = np.zeros(0, np.float32)

        entry = {"ap": _ap_from_pr(tp, sc, n_gt_total),
                 "n_gt": n_gt_total,
                 "n_pred": len(recs),
                 "tp": float(tp.sum()),
                 "per_bin": {}}
        for bn in bin_names:
            sel = np.array([(r[2] == bn) or (r[1] == 0.0 and r[3] == bn) for r in recs], bool) \
                if recs else np.zeros(0, bool)
            entry["per_bin"][bn] = {
                "ap": _ap_from_pr(tp[sel], sc[sel], n_gt_bin[bn]) if sel.any() else 0.0,
                "n_gt": n_gt_bin[bn],
                "recall": float(tp[sel].sum() / n_gt_bin[bn]) if n_gt_bin[bn] else 0.0,
            }
        per_thr[float(thr)] = entry

    ap50 = per_thr[0.5]["ap"]
    ap5095 = float(np.mean([v["ap"] for v in per_thr.values()]))
    e50 = per_thr[0.5]
    precision = e50["tp"] / max(e50["n_pred"], 1)
    recall = e50["tp"] / max(e50["n_gt"], 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    return {
        "mAP50": round(ap50, 4),
        "mAP50_95": round(ap5095, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "n_gt": e50["n_gt"],
        "n_pred": e50["n_pred"],
        "per_bin_at50": {k: {kk: round(vv, 4) if isinstance(vv, float) else vv
                             for kk, vv in v.items()}
                         for k, v in e50["per_bin"].items()},
    }
