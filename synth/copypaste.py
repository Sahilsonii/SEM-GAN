"""
Copy-paste augmentation pool: real defect crops on real train backgrounds.

This is the baseline the rendered pool should have been compared against from
the start (Ghiasi et al., "Simple Copy-Paste", CVPR 2021). It answers one
question the rendered pool cannot: does label-exact augmentation help this
task at all, when the defect *appearance* is real? Every crop is a human-
annotated defect from the train split; every canvas is a defect-free train
image; placement reuses the renderer's grain-boundary prior; blending is
Poisson (cv2.seamlessClone). The box written to the label is the pasted
rectangle, so the label is exact by construction, as with the renderer.

Two things the rendered pool got wrong are fixed here and in generate.py:
  * the per-image defect count is SAMPLED from the real prior, not set to its
    mean - every rendered image carried exactly 52 defects;
  * a fraction of images are backgrounds with no defects, because 69% of real
    train images are, and a pool with none teaches "defects everywhere".

Output layout matches generate.py exactly (images/, labels/, manifest.jsonl,
summary.json) so yolo_export.build(synth_pool="copypaste") needs no change.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from synth.generate import (CURATED_IMAGES, OUT_ROOT, _assert_no_holdout_leak,  # noqa: E402
                            _fit_long_side, _load_split, background_pool)
from synth.renderer import fit_priors, grain_boundary_affinity  # noqa: E402

MARGIN = 0.15          # context around the box that travels with the crop
MIN_SIDE_PX = 3        # below this a crop is a few pixels of noise
BOUNDARY_BIAS = 0.7    # same as renderer.sample_params


def source_crops(split: str = "train", render_px: int = 512) -> list[dict]:
    """Every annotated box in `split`, cut out with a margin and scaled to the
    canvas resolution. Kept in memory: ~2,600 small arrays."""
    crops = []
    for r in _load_split(split)["records"]:
        if not r["boxes"]:
            continue
        img = cv2.imread(str(CURATED_IMAGES / r["file"]), cv2.IMREAD_COLOR)
        if img is None:
            continue
        H, W = img.shape[:2]
        s = render_px / max(H, W)               # same scale the canvas gets
        for j, (c, cx, cy, w, h) in enumerate(r["boxes"]):
            bw, bh = w * W, h * H
            mx, my = bw * MARGIN, bh * MARGIN
            x0, y0 = int(max(0, cx * W - bw / 2 - mx)), int(max(0, cy * H - bh / 2 - my))
            x1, y1 = int(min(W, cx * W + bw / 2 + mx)), int(min(H, cy * H + bh / 2 + my))
            if x1 - x0 < 2 or y1 - y0 < 2:
                continue
            patch = img[y0:y1, x0:x1]
            ph, pw = max(1, round((y1 - y0) * s)), max(1, round((x1 - x0) * s))
            patch = cv2.resize(patch, (pw, ph), interpolation=cv2.INTER_AREA)
            box_w, box_h = bw * s, bh * s        # the label box, in canvas px
            if min(box_w, box_h) < MIN_SIDE_PX:
                continue
            crops.append({"patch": patch, "cls": int(c), "box_w": box_w,
                          "box_h": box_h, "source": r["file"], "index": j})
    return crops


def _place(canvas, patch, cx, cy):
    """Paste `patch` centred at (cx, cy). Poisson-blend when the region is big
    enough for it to mean anything; direct copy for the few-pixel crops where
    seamlessClone either throws or smears the defect into the background."""
    ph, pw = patch.shape[:2]
    H, W = canvas.shape[:2]
    x0, y0 = int(round(cx - pw / 2)), int(round(cy - ph / 2))
    x0, y0 = max(0, min(x0, W - pw)), max(0, min(y0, H - ph))
    if min(ph, pw) >= 8:
        try:
            # Elliptical mask, not the full rectangle. With a rectangle the
            # Poisson boundary sits on the crop's edge, where the source's
            # margin texture meets a different destination texture, and large
            # pastes showed a visible rectangular seam - a shortcut feature a
            # detector would learn. An inscribed ellipse keeps the boundary
            # inside the margin, where the blend has room to absorb the offset.
            mask = np.zeros((ph, pw), np.uint8)
            cv2.ellipse(mask, (pw // 2, ph // 2), (max(1, pw // 2 - 1), max(1, ph // 2 - 1)),
                        0, 0, 360, 255, -1)
            return cv2.seamlessClone(patch, canvas, mask,
                                     (x0 + pw // 2, y0 + ph // 2), cv2.NORMAL_CLONE), (x0, y0)
        except cv2.error:
            pass
    out = canvas.copy()
    out[y0:y0 + ph, x0:x0 + pw] = patch
    return out, (x0, y0)


def generate(n_images: int = 500, seed: int = 42, pool_name: str = "copypaste",
             render_px: int = 512, bg_fraction: float = 0.4,
             split: str = "train") -> dict:
    rng = np.random.default_rng(seed)
    backgrounds = background_pool(split)
    crops = source_crops(split, render_px)
    by_cls = {c: [k for k in crops if k["cls"] == c] for c in (0, 1)}
    if not backgrounds or not all(by_cls.values()):
        raise RuntimeError("need defect-free backgrounds and crops of both classes")
    priors = fit_priors(split)

    out_dir = OUT_ROOT / pool_name
    if out_dir.exists():
        import shutil
        shutil.rmtree(out_dir)
    (out_dir / "images").mkdir(parents=True)
    (out_dir / "labels").mkdir()

    n_boxes = n_bg = 0
    used_groups: set[str] = set()
    with (out_dir / "manifest.jsonl").open("w", encoding="utf-8") as mf:
        for i in tqdm(range(n_images), desc=f"copypaste[{pool_name}]", unit="img",
                      dynamic_ncols=True):
            bg = backgrounds[i % len(backgrounds)]
            used_groups.add(bg["group"])
            canvas = cv2.imread(str(CURATED_IMAGES / bg["file"]), cv2.IMREAD_COLOR)
            if canvas is None:
                continue
            canvas = _fit_long_side(canvas, render_px)
            H, W = canvas.shape[:2]

            boxes, params = [], []
            if rng.random() >= bg_fraction:
                k = int(min(60, max(1, rng.choice(priors["counts"]))))
                ridges = grain_boundary_affinity(cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY))
                for _ in range(k):
                    c = int(rng.integers(0, 2))                  # balanced by class
                    src = by_cls[c][int(rng.integers(len(by_cls[c])))]
                    if len(ridges) and rng.random() < BOUNDARY_BIAS:
                        x, y = ridges[int(rng.integers(len(ridges)))]
                        cx = float(np.clip(x + rng.normal(0, render_px * 0.015), 0, W - 1))
                        cy = float(np.clip(y + rng.normal(0, render_px * 0.015), 0, H - 1))
                    else:
                        cx, cy = rng.uniform(0, W), rng.uniform(0, H)
                    canvas, (x0, y0) = _place(canvas, src["patch"], cx, cy)
                    ph, pw = src["patch"].shape[:2]
                    # the LABEL is the annotated box, centred in the margin crop
                    bx = (x0 + pw / 2) / W
                    by = (y0 + ph / 2) / H
                    boxes.append([c, bx, by, src["box_w"] / W, src["box_h"] / H])
                    params.append({"kind": "pbi2" if c == 0 else "pinhole",
                                   "cx": bx, "cy": by, "source": src["source"],
                                   "source_box": src["index"]})
            else:
                n_bg += 1

            stem = f"cp_{pool_name}_{i:05d}"
            cv2.imwrite(str(out_dir / "images" / f"{stem}.jpg"), canvas,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            (out_dir / "labels" / f"{stem}.txt").write_text(
                "".join("%d %.6f %.6f %.6f %.6f\n" % tuple(b) for b in boxes),
                encoding="utf-8")
            mf.write(json.dumps({
                "stem": stem, "background_file": bg["file"],
                "background_group": bg["group"], "source_split": split,
                "render_px": render_px, "n_defects": len(boxes), "params": params,
            }) + "\n")
            n_boxes += len(boxes)

    _assert_no_holdout_leak(used_groups)
    summary = {"pool": pool_name, "method": "copy-paste, Poisson blend",
               "images": n_images, "backgrounds": n_bg, "boxes": n_boxes,
               "source_crops": len(crops), "backgrounds_used": len(used_groups),
               "seed": seed, "render_px": render_px, "bg_fraction": bg_fraction}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"[copypaste] '{pool_name}': {n_images} images ({n_bg} backgrounds), "
          f"{n_boxes} boxes from {len(crops)} real crops, {len(used_groups)} train "
          f"backgrounds, leak check passed")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pool", default="copypaste")
    ap.add_argument("--render-px", type=int, default=512)
    ap.add_argument("--bg-fraction", type=float, default=0.4,
                    help="share of images with no defects (real train: 0.69)")
    a = ap.parse_args()
    generate(n_images=a.n, seed=a.seed, pool_name=a.pool,
             render_px=a.render_px, bg_fraction=a.bg_fraction)
