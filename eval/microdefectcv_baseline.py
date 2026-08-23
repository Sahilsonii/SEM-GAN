"""
R1 - MicroDefectCV as a zero-training classical baseline.

MicroDefectCV (PyPI, MIT) is prior work by the same author; its documentation
ships no quantitative benchmark. This module supplies one, scored through the
same eval/detection.py path as every deep detector so the comparison is real.

The interesting question is not whether classical CV wins overall - it probably
does not - but whether it wins on T1/T2. Morphological top-hat and black-hat
operate at full resolution with no stride floor, whereas a P3-based detector has
to represent a 6 px pinhole inside a single feature cell.

FIREWALL (plan 7.2): this module only ever READS. MicroDefectCV never enters a
training loss and never informs the renderer priors, so it stays an independent
evaluator of what the renderer produces.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.detection import evaluate

CURATED_IMAGES = ROOT / "data" / "curated" / "images"
SPLITS = ROOT / "data" / "splits"
OUT = ROOT / "outputs"

# MicroDefectCV mode -> which of our merged classes its detections represent
MODE_CLASS = {"pinhole": 1, "pbi2": 0}


def _contrast_score(gray: np.ndarray, bbox, cls: int) -> float:
    """Confidence proxy: how far the region departs from local background.

    MicroDefectCV emits no confidence, and scoring every box at 1.0 makes AP
    degenerate - the precision-recall curve collapses to a single point. Local
    contrast is the natural stand-in: pinholes are dark against their
    surroundings, PbI2 particles bright, and the magnitude of that departure is
    exactly what a human uses to judge a borderline detection.
    """
    x, y, w, h = bbox
    H, W = gray.shape
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inner = gray[y0:y1, x0:x1].astype(np.float32).mean()

    pad = max(4, max(w, h))
    rx0, ry0 = max(0, x0 - pad), max(0, y0 - pad)
    rx1, ry1 = min(W, x1 + pad), min(H, y1 + pad)
    ring = gray[ry0:ry1, rx0:rx1].astype(np.float32).mean()

    delta = (ring - inner) if cls == 1 else (inner - ring)   # pinhole dark, pbi2 bright
    return float(np.clip(delta / 64.0, 0.0, 1.0))


def predict_image(img_bgr: np.ndarray, modes=("pinhole", "pbi2"),
                  min_area: int = 4, sensitivity: float = 1.5) -> tuple:
    """Run MicroDefectCV once per class-specific mode and merge the detections.

    Images come from data/curated, where the FESEM banner has already been
    removed corpus-wide. Verified: detect_defects' own internal bar crop is then
    a no-op (input rows == output mask rows), so returned coordinates need no
    compensation.
    """
    from microdefectcv import detect_defects

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    H, W = img_bgr.shape[:2]

    boxes, scores = [], []
    for mode in modes:
        cls = MODE_CLASS[mode]
        try:
            res = detect_defects(img_bgr, mode=mode, min_area=min_area,
                                 sensitivity=sensitivity)
        except Exception:
            continue
        for det in res.get("detections", []):
            x, y, bw, bh = det["bbox"]
            if bw < 2 or bh < 2:
                continue
            boxes.append([cls, (x + bw / 2) / W, (y + bh / 2) / H, bw / W, bh / H])
            scores.append(_contrast_score(gray, det["bbox"], cls))
    return boxes, np.asarray(scores, np.float32)


def run_baseline(split: str = "val", limit: int | None = None,
                 min_area: int = 4, sensitivity: float = 1.5) -> dict:
    if split == "test":
        raise RuntimeError(
            "the test split is locked until the final evaluation stage; "
            "develop against 'val'")

    recs = json.loads((SPLITS / f"{split}.json").read_text(encoding="utf-8"))["records"]
    recs = [r for r in recs if r["n_boxes"] > 0]
    if limit:
        recs = recs[:limit]

    samples, t0 = [], time.time()
    for r in recs:
        img = cv2.imread(str(CURATED_IMAGES / r["file"]), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        pred, scores = predict_image(img, min_area=min_area, sensitivity=sensitivity)
        samples.append({"gt": r["boxes"], "pred": pred,
                        "scores": scores, "wh": (w, h)})
    elapsed = time.time() - t0

    metrics = evaluate(samples)
    metrics.update({
        "method": "microdefectcv",
        "split": split,
        "images": len(samples),
        "seconds_total": round(elapsed, 2),
        "seconds_per_image": round(elapsed / max(len(samples), 1), 3),
        "trainable_params": 0,
        "min_area": min_area,
        "sensitivity": sensitivity,
        "note": "zero-training classical baseline; score = local contrast proxy",
    })

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"microdefectcv_baseline_{split}.json").write_text(
        json.dumps(metrics, indent=1), encoding="utf-8")

    print(f"[R1] microdefectcv on {split}: {len(samples)} imgs, "
          f"{metrics['n_pred']} preds vs {metrics['n_gt']} gt")
    print(f"     mAP50={metrics['mAP50']:.4f}  mAP50-95={metrics['mAP50_95']:.4f}  "
          f"P={metrics['precision']:.3f}  R={metrics['recall']:.3f}  F1={metrics['f1']:.3f}")
    for bn, v in metrics["per_bin_at50"].items():
        print(f"     {bn:<14} n_gt={v['n_gt']:<5} recall={v['recall']:.3f}  AP={v['ap']:.4f}")
    print(f"     {metrics['seconds_per_image']:.3f} s/image on CPU, 0 trainable params")
    return metrics


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--min-area", type=int, default=4)
    ap.add_argument("--sensitivity", type=float, default=1.5)
    a = ap.parse_args()
    run_baseline(split=a.split, limit=a.limit, min_area=a.min_area,
                 sensitivity=a.sensitivity)


def sweep(split: str = "val", limit: int = 8,
          min_areas=(4, 12, 30, 60, 120), sensitivities=(1.0, 1.5, 2.5)) -> dict:
    """Tune MicroDefectCV on VAL before reporting it.

    A classical baseline run at whatever default happened to be in the signature
    is a straw man. It gets the same courtesy every deep model gets: a
    hyperparameter search on the validation split, with the test split untouched.
    """
    rows = []
    for ma in min_areas:
        for sn in sensitivities:
            m = run_baseline(split=split, limit=limit, min_area=ma, sensitivity=sn)
            rows.append({"min_area": ma, "sensitivity": sn, "mAP50": m["mAP50"],
                         "mAP50_95": m["mAP50_95"], "precision": m["precision"],
                         "recall": m["recall"], "n_pred": m["n_pred"]})
            print(f"  -> min_area={ma:<4} sens={sn:<4} mAP50={m['mAP50']:.4f} "
                  f"P={m['precision']:.3f} R={m['recall']:.3f} n_pred={m['n_pred']}")
    best = max(rows, key=lambda r: r["mAP50"])
    (OUT / f"microdefectcv_sweep_{split}.json").write_text(
        json.dumps({"rows": rows, "best": best}, indent=1), encoding="utf-8")
    print(f"\n[R1] best on {split}: min_area={best['min_area']} "
          f"sensitivity={best['sensitivity']} mAP50={best['mAP50']:.4f}")
    return best
