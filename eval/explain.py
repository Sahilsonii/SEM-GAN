"""
Explainability, EVALUATED rather than displayed (plan section 12).

The plan is explicit that a heatmap picture is not evidence. So this measures
three things instead:

  pointing game      does the attribution PEAK fall inside an expert box?
  explanation IoU    thresholded attribution vs the union of expert boxes
  background ratio   fraction of attribution mass OUTSIDE every box

The third is the one that matters most for this corpus. The images carried a
burned-in FESEM metadata banner (stripped at curation) and grain-boundary
texture that the failure analysis showed accounts for 47% of false positives.
Background-attribution ratio is what distinguishes "the model reads defects"
from "the model reads texture that correlates with defects".

Attribution is occlusion-based, not gradient-based: it needs no hooks into
Ultralytics internals, it is model-agnostic, and it measures what actually
changes the prediction. Cost is one forward pass per patch, so the stride is
coarse by default.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.detection import xywhn_to_xyxy

CURATED = ROOT / "data" / "curated" / "images"
SPLITS = ROOT / "data" / "splits"
OUT = ROOT / "outputs"


def occlusion_saliency(net, im, patch: int = 64, stride: int = 32,
                       conf: float = 0.05, device: str = "0") -> np.ndarray:
    """Drop in total detection confidence when each patch is occluded.

    High value = occluding here HURT the prediction = the model was using it.
    """
    H, W = im.shape[:2]
    base = net.predict(im, conf=conf, verbose=False, device=device)[0]
    base_score = float(base.boxes.conf.sum().item()) if len(base.boxes) else 0.0
    if base_score <= 0:
        return np.zeros((H, W), np.float32)

    sal = np.zeros((H, W), np.float32)
    cnt = np.zeros((H, W), np.float32)
    mean_val = float(im.mean())

    for y in range(0, H - 1, stride):
        for x in range(0, W - 1, stride):
            y1, x1 = min(H, y + patch), min(W, x + patch)
            occ = im.copy()
            occ[y:y1, x:x1] = mean_val          # grey-out, not black: black is
                                                # itself a defect-like signal here
            r = net.predict(occ, conf=conf, verbose=False, device=device)[0]
            s = float(r.boxes.conf.sum().item()) if len(r.boxes) else 0.0
            sal[y:y1, x:x1] += max(0.0, base_score - s)
            cnt[y:y1, x:x1] += 1.0
    return sal / np.maximum(cnt, 1e-6)


def evaluate(checkpoint: str, n_images: int = 6, patch: int = 64, stride: int = 32,
             conf: float = 0.05, device: str = "0", save_figs: bool = True) -> dict:
    from ultralytics import YOLO

    net = YOLO(checkpoint)
    recs = [r for r in json.loads((SPLITS / "val.json").read_text(encoding="utf-8"))["records"]
            if r["n_boxes"] > 0][:n_images]

    fig_dir = OUT / "figures" / "explain"
    if save_figs:
        fig_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for r in recs:
        im = cv2.imread(str(CURATED / r["file"]), cv2.IMREAD_COLOR)
        if im is None:
            continue
        H, W = im.shape[:2]
        sal = occlusion_saliency(net, im, patch, stride, conf, device)
        if sal.max() <= 0:
            continue

        gt_mask = np.zeros((H, W), bool)
        for b in r["boxes"]:
            x0, y0, x1, y1 = [int(v) for v in xywhn_to_xyxy(b, W, H)]
            gt_mask[max(0, y0):min(H, y1), max(0, x0):min(W, x1)] = True

        # pointing game: does the single strongest location land on a defect?
        py, px = np.unravel_index(int(np.argmax(sal)), sal.shape)
        hit = bool(gt_mask[py, px])

        # explanation IoU at the attribution's 90th percentile
        thr = np.percentile(sal, 90)
        expl = sal >= thr
        inter = np.logical_and(expl, gt_mask).sum()
        union = np.logical_or(expl, gt_mask).sum()
        iou = float(inter / union) if union else 0.0

        # how much attribution mass sits outside every annotated box
        total = float(sal.sum())
        bg_ratio = float(sal[~gt_mask].sum() / total) if total > 0 else 1.0
        gt_frac = float(gt_mask.mean())      # what fraction of the frame IS defect

        rows.append({"file": r["file"], "pointing_hit": hit,
                     "explanation_iou": round(iou, 4),
                     "background_attribution_ratio": round(bg_ratio, 4),
                     "gt_area_fraction": round(gt_frac, 4),
                     "attribution_concentration": round(
                         (1 - bg_ratio) / gt_frac if gt_frac > 0 else 0.0, 3)})

        if save_figs:
            hm = cv2.applyColorMap(
                (255 * sal / sal.max()).astype(np.uint8), cv2.COLORMAP_INFERNO)
            over = cv2.addWeighted(im, 0.6, hm, 0.4, 0)
            for b in r["boxes"]:
                x0, y0, x1, y1 = [int(v) for v in xywhn_to_xyxy(b, W, H)]
                cv2.rectangle(over, (x0, y0), (x1, y1), (0, 255, 0), 1)
            cv2.circle(over, (int(px), int(py)), 6, (255, 255, 255), 2)
            cv2.imwrite(str(fig_dir / f"{Path(r['file']).stem}_saliency.png"),
                        np.hstack([im, over]))

    if not rows:
        raise RuntimeError("no usable saliency maps - model detected nothing")

    pg = float(np.mean([r["pointing_hit"] for r in rows]))
    out = {
        "checkpoint": checkpoint, "n_images": len(rows),
        "patch": patch, "stride": stride,
        "pointing_game_accuracy": round(pg, 4),
        "mean_explanation_iou": round(float(np.mean([r["explanation_iou"] for r in rows])), 4),
        "mean_background_attribution_ratio": round(
            float(np.mean([r["background_attribution_ratio"] for r in rows])), 4),
        "mean_gt_area_fraction": round(
            float(np.mean([r["gt_area_fraction"] for r in rows])), 4),
        "mean_attribution_concentration": round(
            float(np.mean([r["attribution_concentration"] for r in rows])), 3),
        "per_image": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "explainability.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

    print(f"[explain] {len(rows)} val images, occlusion patch={patch} stride={stride}")
    print(f"  pointing game accuracy        {out['pointing_game_accuracy']:.3f}"
          f"   (peak attribution inside a defect box)")
    print(f"  mean explanation IoU          {out['mean_explanation_iou']:.4f}")
    print(f"  background attribution ratio  {out['mean_background_attribution_ratio']:.4f}")
    print(f"  defect area is only           {out['mean_gt_area_fraction']:.4f} of the frame")
    print(f"  concentration (x vs chance)   {out['mean_attribution_concentration']:.2f}"
          f"   (>1 = attends defects more than area alone would give)")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n-images", type=int, default=6)
    ap.add_argument("--patch", type=int, default=64)
    ap.add_argument("--stride", type=int, default=32)
    a = ap.parse_args()
    evaluate(a.checkpoint, n_images=a.n_images, patch=a.patch, stride=a.stride)
