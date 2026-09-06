"""
Composition-first synthesis: two pools that reuse REAL defect layouts.

Every pool so far (rendered, refined, copy-paste) scattered defects over
backgrounds a human had labelled clean, at counts drawn independently of
where they landed. Phase 2 showed those pools are neutral at any budget and
that appearance is not the cause (RESULTS.md §17). What all of them share is
the composition, so these two pools change only that:

  layout_transfer   the exact box layout of one real annotated train image
                    (positions, sizes, classes), reproduced on a clean train
                    background with class- and size-matched real crops.
                    Real composition, clean-background pixels.

  augment_defective a real annotated train image as the canvas, labels kept,
                    plus ~20% extra class-matched crops placed near existing
                    defects. Real composition AND real defective-film context
                    (Ghiasi et al. 2021, as actually practised).

Both write the layout generate.py / copypaste.py use, so data/yolo_export.py
and train_detector.py need no change:
  data/synthetic/<pool>/{images,labels}/<stem>.{jpg,txt}, manifest.jsonl,
  summary.json.  Every canvas group is asserted to be in TRAIN.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from synth.copypaste import _place, source_crops                     # noqa: E402
from synth.generate import (CURATED_IMAGES, OUT_ROOT, _assert_no_holdout_leak,  # noqa: E402
                            _fit_long_side, _load_split, background_pool)

JITTER = 0.015          # layout_transfer: sd of position jitter, fraction of render_px
EXTRA_FRACTION = 0.20   # augment_defective: extra crops as a share of the image's own count
NEAR = (1.0, 2.0)       # augment_defective: distance to a random existing box, in units of
                        # that box's nearest-neighbour distance


def layouts(split: str = "train") -> list[dict]:
    """Every annotated image in `split` as a layout: normalised boxes plus the
    source pixel size, so crop sizes can be matched in canvas pixels."""
    out = []
    for r in _load_split(split)["records"]:
        if not r["boxes"]:
            continue
        img = cv2.imread(str(CURATED_IMAGES / r["file"]), cv2.IMREAD_COLOR)
        if img is None:
            continue
        H, W = img.shape[:2]
        out.append({"file": r["file"], "group": r["group"], "W": W, "H": H,
                    "boxes": [list(b) for b in r["boxes"]]})
    return out


def _index_crops(crops):
    """Per class, crops sorted by log box area for nearest-size lookup."""
    idx = {}
    for c in (0, 1):
        ks = [k for k in crops if k["cls"] == c]
        areas = np.log([k["box_w"] * k["box_h"] for k in ks])
        order = np.argsort(areas)
        idx[c] = ([ks[i] for i in order], areas[order])
    return idx


def _nearest_crop(idx, c, box_w, box_h, rng, spread=5):
    """A crop of class c whose box area is close to (box_w, box_h): one of the
    `spread` nearest by log area, chosen at random so a layout box does not
    always pull the same crop."""
    ks, areas = idx[c]
    j = int(np.searchsorted(areas, np.log(max(box_w * box_h, 1e-6))))
    lo, hi = max(0, j - spread), min(len(ks), j + spread)
    return ks[int(rng.integers(lo, hi))]


def _open_pool(pool_name):
    out_dir = OUT_ROOT / pool_name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "images").mkdir(parents=True)
    (out_dir / "labels").mkdir()
    return out_dir


def _emit(out_dir, mf, stem, canvas, boxes, meta):
    cv2.imwrite(str(out_dir / "images" / f"{stem}.jpg"), canvas,
                [cv2.IMWRITE_JPEG_QUALITY, 95])
    (out_dir / "labels" / f"{stem}.txt").write_text(
        "".join("%d %.6f %.6f %.6f %.6f\n" % tuple(b) for b in boxes), encoding="utf-8")
    mf.write(json.dumps({"stem": stem, "n_defects": len(boxes), **meta}) + "\n")


def _paste(canvas, src, cx, cy, c):
    """Paste crop `src` centred at (cx, cy); return the exact label row."""
    H, W = canvas.shape[:2]
    canvas, (x0, y0) = _place(canvas, src["patch"], cx, cy)
    ph, pw = src["patch"].shape[:2]
    return canvas, [c, (x0 + pw / 2) / W, (y0 + ph / 2) / H, src["box_w"] / W, src["box_h"] / H]


def generate_layout_transfer(n_images=500, seed=42, pool_name="layout_transfer",
                             render_px=512, bg_fraction=0.4, split="train") -> dict:
    rng = np.random.default_rng(seed)
    backgrounds = background_pool(split)
    lays = layouts(split)
    idx = _index_crops(source_crops(split, render_px))
    if not backgrounds or not lays or not all(len(v[0]) for v in idx.values()):
        raise RuntimeError("need backgrounds, annotated layouts and crops of both classes")

    out_dir = _open_pool(pool_name)
    n_boxes = n_bg = 0
    used = set()
    with (out_dir / "manifest.jsonl").open("w", encoding="utf-8") as mf:
        for i in tqdm(range(n_images), desc=f"layout_transfer[{pool_name}]", unit="img",
                      dynamic_ncols=True):
            bg = backgrounds[i % len(backgrounds)]
            used.add(bg["group"])
            canvas = _fit_long_side(cv2.imread(str(CURATED_IMAGES / bg["file"])), render_px)
            H, W = canvas.shape[:2]
            boxes, meta = [], {"background_file": bg["file"], "background_group": bg["group"],
                               "source_split": split, "render_px": render_px}
            if rng.random() >= bg_fraction:
                lay = lays[int(rng.integers(len(lays)))]
                flip = bool(rng.random() < 0.5)
                s = render_px / max(lay["W"], lay["H"])       # source px -> canvas px
                meta.update({"layout_file": lay["file"], "flipped": flip})
                for c, cx, cy, w, h in lay["boxes"]:
                    if flip:
                        cx = 1.0 - cx
                    src = _nearest_crop(idx, int(c), w * lay["W"] * s, h * lay["H"] * s, rng)
                    px = float(np.clip(cx * W + rng.normal(0, JITTER * render_px), 0, W - 1))
                    py = float(np.clip(cy * H + rng.normal(0, JITTER * render_px), 0, H - 1))
                    canvas, row = _paste(canvas, src, px, py, int(c))
                    boxes.append(row)
            else:
                n_bg += 1
            _emit(out_dir, mf, f"lt_{pool_name}_{i:05d}", canvas, boxes, meta)
            n_boxes += len(boxes)

    _assert_no_holdout_leak(used)
    summary = {"pool": pool_name, "method": "layout transfer: real layouts on clean backgrounds",
               "images": n_images, "backgrounds": n_bg, "boxes": n_boxes,
               "layouts": len(lays), "backgrounds_used": len(used), "seed": seed,
               "render_px": render_px, "bg_fraction": bg_fraction, "jitter": JITTER}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"[compose] '{pool_name}': {n_images} images ({n_bg} backgrounds), {n_boxes} boxes "
          f"from {len(lays)} real layouts, leak check passed")
    return summary


def generate_augment_defective(n_images=500, seed=42, pool_name="augment_defective",
                               render_px=512, split="train") -> dict:
    rng = np.random.default_rng(seed)
    lays = layouts(split)
    idx = _index_crops(source_crops(split, render_px))
    if not lays or not all(len(v[0]) for v in idx.values()):
        raise RuntimeError("need annotated canvases and crops of both classes")

    out_dir = _open_pool(pool_name)
    n_boxes = n_extra = 0
    used = set()
    with (out_dir / "manifest.jsonl").open("w", encoding="utf-8") as mf:
        for i in tqdm(range(n_images), desc=f"augment_defective[{pool_name}]", unit="img",
                      dynamic_ncols=True):
            lay = lays[i % len(lays)]
            used.add(lay["group"])
            canvas = _fit_long_side(cv2.imread(str(CURATED_IMAGES / lay["file"])), render_px)
            H, W = canvas.shape[:2]
            boxes = [list(b) for b in lay["boxes"]]          # the real labels, verbatim
            # local spacing: each box's nearest-neighbour distance, canvas px
            pts = np.array([[b[1] * W, b[2] * H] for b in boxes])
            if len(pts) > 1:
                d = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
                np.fill_diagonal(d, np.inf)
                nn = d.min(1)
            else:
                nn = np.full(len(pts), 0.05 * render_px)
            classes = [int(b[0]) for b in boxes]
            k = max(1, round(EXTRA_FRACTION * len(boxes)))
            for _ in range(k):
                j = int(rng.integers(len(lay["boxes"])))    # anchor on a REAL defect, not
                                                             # on an extra appended above
                c = classes[j]                               # same class as its neighbour
                b = lay["boxes"][j]
                src = _nearest_crop(idx, c, b[3] * lay["W"] * render_px / max(lay["W"], lay["H"]),
                                    b[4] * lay["H"] * render_px / max(lay["W"], lay["H"]), rng)
                r = float(nn[j] * rng.uniform(*NEAR))
                th = float(rng.uniform(0, 2 * np.pi))
                px = float(np.clip(pts[j, 0] + r * np.cos(th), 0, W - 1))
                py = float(np.clip(pts[j, 1] + r * np.sin(th), 0, H - 1))
                canvas, row = _paste(canvas, src, px, py, c)
                boxes.append(row)
            n_extra += k
            _emit(out_dir, mf, f"ad_{pool_name}_{i:05d}", canvas, boxes,
                  {"background_file": lay["file"], "background_group": lay["group"],
                   "source_split": split, "render_px": render_px,
                   "n_real_boxes": len(lay["boxes"]), "n_extra": k})
            n_boxes += len(boxes)

    _assert_no_holdout_leak(used)
    summary = {"pool": pool_name, "method": "copy-paste onto real defective images",
               "images": n_images, "backgrounds": 0, "boxes": n_boxes, "extra_boxes": n_extra,
               "canvases": len(lays), "backgrounds_used": len(used), "seed": seed,
               "render_px": render_px, "extra_fraction": EXTRA_FRACTION}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"[compose] '{pool_name}': {n_images} images on {len(lays)} real defective canvases, "
          f"{n_boxes} boxes ({n_extra} pasted), leak check passed")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", choices=["layout_transfer", "augment_defective"], required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--render-px", type=int, default=512)
    ap.add_argument("--bg-fraction", type=float, default=0.4, help="layout_transfer only")
    a = ap.parse_args()
    if a.pool == "layout_transfer":
        generate_layout_transfer(a.n, a.seed, render_px=a.render_px, bg_fraction=a.bg_fraction)
    else:
        generate_augment_defective(a.n, a.seed, render_px=a.render_px)
