"""
Stage 2 - build a synthetic pool from TRAIN backgrounds only.

The previous pipeline drew its canvases from every background in the corpus,
including the ones that ended up in val and test. That alone invalidated any
synth-to-real claim. Here the background pool is read from data/splits/train.json
and a hard assertion refuses to proceed if a val/test group ever appears.

Each generated image ships a YOLO label file whose boxes came from the renderer's
own masks, plus a manifest line recording every parameter used - so any image can
be regenerated exactly, and the severity ladder used for counterfactual probing
is reproducible.
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

from synth.renderer import fit_priors, render, sample_params

CURATED_IMAGES = ROOT / "data" / "curated" / "images"
SPLITS = ROOT / "data" / "splits"
OUT_ROOT = ROOT / "data" / "synthetic"


def _load_split(name: str) -> dict:
    return json.loads((SPLITS / (name + ".json")).read_text(encoding="utf-8"))


def background_pool(split: str = "train") -> list[dict]:
    """Defect-free images from `split`, used as canvases."""
    recs = _load_split(split)["records"]
    return [r for r in recs if r["n_boxes"] == 0]


def _assert_no_holdout_leak(used_groups: set[str]) -> None:
    """Refuse to emit a pool built on any val/test specimen."""
    for holdout in ("val", "test"):
        bad = used_groups & set(_load_split(holdout)["groups"])
        if bad:
            raise AssertionError(
                f"LEAK: {len(bad)} {holdout} groups used as synthetic canvases: "
                f"{sorted(bad)[:5]}")


def generate(n_images: int = 200, render_px: int = 512, seed: int = 42,
             pool_name: str = "controlled", severity=None,
             defects_per_image=None, split: str = "train",
             pbi2_fraction: float = 0.0) -> dict:
    """Build a synthetic pool.

    pbi2_fraction defaults to 0: the closed-set task is pinhole detection and
    PbI2 is the held-out unknown morphology, so synthesising PbI2 into the
    training pool would defeat the open-set experiment. Raise it only to build a
    deliberate unknown-morphology probe.
    """
    backgrounds = background_pool(split)
    if not backgrounds:
        raise RuntimeError(f"no defect-free backgrounds in split '{split}'")

    priors = fit_priors(split)
    rng = np.random.default_rng(seed)

    out_dir = OUT_ROOT / pool_name
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"

    used_groups: set[str] = set()
    n_boxes = 0
    with manifest_path.open("w", encoding="utf-8") as mf:
        pbar = tqdm(range(n_images), desc=f"render[{pool_name}]", unit="img",
                    dynamic_ncols=True)
        for i in pbar:
            bg = backgrounds[i % len(backgrounds)]
            used_groups.add(bg["group"])

            img = cv2.imread(str(CURATED_IMAGES / bg["file"]), cv2.IMREAD_COLOR)
            if img is None:
                continue
            if img.shape[:2] != (render_px, render_px):
                img = cv2.resize(img, (render_px, render_px), interpolation=cv2.INTER_AREA)

            k = (int(rng.choice(priors["counts"])) if defects_per_image is None
                 else int(defects_per_image))
            k = max(1, min(k, 60))
            # canvases come from data/curated, where the FESEM banner is already
            # gone - no placement restriction is needed here any more. bg_gray
            # is passed so placement is biased toward this REAL background's
            # own grain boundaries, per the literature cited in renderer.py
            # (both defect classes are reported as grain-boundary phenomena).
            bg_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            params = sample_params(priors, k, rng, render_px=render_px,
                                   severity=severity, pbi2_fraction=pbi2_fraction,
                                   bg_gray=bg_gray)
            res = render(img, params, seed=int(rng.integers(0, 2**31 - 1)))
            if not res["boxes"]:
                continue

            stem = f"syn_{pool_name}_{i:05d}"
            cv2.imwrite(str(out_dir / "images" / f"{stem}.jpg"), res["image"],
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(out_dir / "masks" / f"{stem}.png"), res["mask"])
            (out_dir / "labels" / f"{stem}.txt").write_text(
                "\n".join(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}"
                          for b in res["boxes"]) + "\n", encoding="utf-8")

            mf.write(json.dumps({
                "stem": stem,
                "background_file": bg["file"],
                "background_group": bg["group"],
                "source_split": split,
                "render_px": render_px,
                "n_defects": len(res["boxes"]),
                "params": res["params"],
            }) + "\n")
            n_boxes += len(res["boxes"])
            pbar.set_postfix({"boxes": n_boxes, "bg_used": len(used_groups)})

    _assert_no_holdout_leak(used_groups)

    summary = {
        "pool": pool_name,
        "images": len(list((out_dir / "images").glob("*.jpg"))),
        "boxes": n_boxes,
        "backgrounds_used": len(used_groups),
        "source_split": split,
        "seed": seed,
        "render_px": render_px,
        "severity": severity,
        "pbi2_fraction": pbi2_fraction,
        "placement": "full frame (canvases are pre-cropped; no banner exists)",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"[synth] pool '{pool_name}': {summary['images']} images, {n_boxes} boxes, "
          f"{len(used_groups)} train backgrounds, leak check passed")
    return summary


def severity_ladder(rungs=(0.0, 0.25, 0.5, 0.75, 1.0), render_px: int = 512,
                    seed: int = 0, n_defects: int = 12) -> dict:
    """Counterfactual probe (N4): one fixed background, only severity varies."""
    backgrounds = background_pool("train")
    bg = backgrounds[0]
    priors = fit_priors("train")
    img = cv2.imread(str(CURATED_IMAGES / bg["file"]), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (render_px, render_px), interpolation=cv2.INTER_AREA)

    # identical layout on every rung - only severity changes
    params = sample_params(priors, n_defects, np.random.default_rng(seed),
                           render_px=render_px,
                           bg_gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    for p in params:
        p.size_px = max(p.size_px, 12.0)

    out_dir = OUT_ROOT / "counterfactual"
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)

    rows = []
    base_mean = float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean())
    for sev in rungs:
        if sev <= 0.0:
            res = {"image": img.copy(),
                   "mask": np.zeros(img.shape[:2], np.uint8),
                   "boxes": []}                      # clean rung: no defect at all
        else:
            for p in params:
                p.severity = float(sev)
            res = render(img, params, seed=seed)

        stem = f"cf_sev{int(sev*100):03d}"
        cv2.imwrite(str(out_dir / "images" / f"{stem}.jpg"), res["image"],
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(out_dir / "masks" / f"{stem}.png"), res["mask"])
        (out_dir / "labels" / f"{stem}.txt").write_text(
            "".join("%d %.6f %.6f %.6f %.6f" % tuple(b) + "\n"
                    for b in res["boxes"]),
            encoding="utf-8")

        m = res["mask"] > 0
        gray = cv2.cvtColor(res["image"], cv2.COLOR_BGR2GRAY).astype(np.float32)
        # severity controls CONTRAST; the layout is identical on every rung, so
        # mask area is constant by construction and is not the quantity to track
        contrast = float(base_mean - gray[m].mean()) if m.any() else 0.0
        rows.append({"severity": sev, "stem": stem, "n_boxes": len(res["boxes"]),
                     "mask_area_ratio": float(m.mean()),
                     "mean_contrast_vs_canvas": round(contrast, 3)})

    (out_dir / "ladder.json").write_text(
        json.dumps({"background": bg["file"], "rungs": rows}, indent=1), encoding="utf-8")
    cons = [r["mean_contrast_vs_canvas"] for r in rows]
    print(f"[synth] counterfactual ladder on {Path(bg['file']).name}: "
          f"contrast {cons[0]:.1f} -> {cons[-1]:.1f} over {len(rows)} rungs "
          f"(layout identical, only severity varies)")
    return {"background": bg["file"], "rungs": rows}


def generate_balanced(per_class: int = 5000, render_px: int = 512, seed: int = 42,
                      pool_name: str = "bulk") -> dict:
    """Generate a pool with a TARGET COUNT per class rather than a total.

    Runs the renderer twice, once per class, forcing pbi2_fraction to 0 or 1 so
    each pass is single-class, then merges the manifests. This is the "N per
    class" mode - two separate n_images=N calls rather than one call with a
    fraction, because a fraction only hits the target in expectation and drifts
    on small pools.
    """
    from synth.renderer import fit_priors

    priors = fit_priors("train")
    n_pinhole_bg = len(background_pool("train"))
    if n_pinhole_bg == 0:
        raise RuntimeError("no defect-free train backgrounds available")

    out_dir = OUT_ROOT / pool_name
    if out_dir.exists():
        import shutil
        shutil.rmtree(out_dir)

    pbi2_summary = generate(n_images=per_class, render_px=render_px, seed=seed,
                            pool_name=f"{pool_name}_pbi2", pbi2_fraction=1.0,
                            defects_per_image=max(1, int(round(priors["counts"].mean()))))
    pinhole_summary = generate(n_images=per_class, render_px=render_px, seed=seed + 1,
                               pool_name=f"{pool_name}_pinhole", pbi2_fraction=0.0,
                               defects_per_image=max(1, int(round(priors["counts"].mean()))))

    # merge both single-class pools into one pool directory + manifest
    out_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("images", "labels", "masks"):
        (out_dir / sub).mkdir(exist_ok=True)
    merged_lines = []
    for src_name in (f"{pool_name}_pbi2", f"{pool_name}_pinhole"):
        src = OUT_ROOT / src_name
        for sub in ("images", "labels", "masks"):
            for f in (src / sub).glob("*"):
                (out_dir / sub / f.name).write_bytes(f.read_bytes())
        merged_lines += (src / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    (out_dir / "manifest.jsonl").write_text(
        "\n".join(merged_lines) + "\n", encoding="utf-8")

    summary = {
        "pool": pool_name, "per_class": per_class,
        "pbi2_images": pbi2_summary["images"], "pinhole_images": pinhole_summary["images"],
        "total_images": pbi2_summary["images"] + pinhole_summary["images"],
        "total_boxes": pbi2_summary["boxes"] + pinhole_summary["boxes"],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"[synth] balanced pool '{pool_name}': {summary['total_images']} images "
          f"({pbi2_summary['images']} pbi2 + {pinhole_summary['images']} pinhole), "
          f"{summary['total_boxes']} boxes")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--pool", default="controlled")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--render-px", type=int, default=512)
    ap.add_argument("--pbi2-fraction", type=float, default=0.0,
                    help="0 keeps PbI2 out of training pools (it is the open-set unknown)")
    ap.add_argument("--balanced-per-class", type=int, default=None,
                    help="generate this many images PER CLASS instead of --n total")
    ap.add_argument("--ladder", action="store_true", help="also build the severity ladder")
    a = ap.parse_args()
    if a.balanced_per_class:
        generate_balanced(per_class=a.balanced_per_class, render_px=a.render_px,
                          seed=a.seed, pool_name=a.pool)
    else:
        generate(n_images=a.n, pool_name=a.pool, seed=a.seed, render_px=a.render_px,
                 pbi2_fraction=a.pbi2_fraction)
    if a.ladder:
        severity_ladder(render_px=a.render_px)
