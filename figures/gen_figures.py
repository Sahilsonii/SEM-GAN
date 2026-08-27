"""
Figures for the SYNTHETIC DATA GENERATION half of the project.

Everything here is CPU-only (numpy / opencv / matplotlib) so it can run while
the GPU is training.

Two of these figures exist specifically as regression evidence for bugs that
invalidated an earlier synthetic pool, and they are the reason this file reports
medians numerically rather than leaving them to be eyeballed:

  fig_aspect_distribution   synthetic box aspect was pinned at 1.000 (a
                            cv2.resize to a square) while real was 0.786
  fig_size_distribution     a mask-threshold bug shrank emitted boxes to
                            35-66% of the sampled size

Sizes are in PIXELS ONLY. JPEG re-encoding destroyed the FESEM pixel-size
headers in all 440 source images and no TIFs survive, so there is no
nm-per-pixel calibration for this corpus and no axis here may claim one.
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt

from figures.style import (BIN_COLORS, BIN_LABELS, C, apply, despine, note,
                           save)

CURATED = ROOT / "data" / "curated" / "images"
SPLITS = ROOT / "data" / "splits"
SYNTH = ROOT / "data" / "synthetic"
OUT = ROOT / "outputs"

DET_PX = 640                       # detector input; all sizes are reported here
BIN_EDGES = [8, 16, 32]            # pre-registered, configs/tiny_defect_bins.yaml
POOLS = ["controlled", "refined", "refined_nofft"]
POOL_LABEL = {"controlled": "renderer only", "refined": "refined (FFT on)",
              "refined_nofft": "refined (FFT off)"}
POOL_COLOR = {"controlled": C["renderer"], "refined": C["fft_on"],
              "refined_nofft": C["fft_off"]}
CLS_NAME = {0: "PbI2", 1: "pinhole"}


# ----------------------------------------------------------------- loading ---
def real_boxes(split: str = "train") -> list[tuple[int, float, float]]:
    """(class, side_px_at_640, aspect w/h) for every expert box in the split."""
    p = SPLITS / f"{split}.json"
    if not p.exists():
        return []
    out = []
    for r in json.loads(p.read_text(encoding="utf-8"))["records"]:
        for b in r.get("boxes", []):
            cls, _, _, w, h = b
            if w <= 0 or h <= 0:
                continue
            out.append((int(cls), math.sqrt(w * h) * DET_PX, w / h))
    return out


def synth_manifest(pool: str) -> list[dict]:
    p = SYNTH / pool / "manifest.jsonl"
    if not p.exists():
        return []
    recs = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            recs.append(json.loads(line))
    return recs


def synth_params(pool: str) -> list[dict]:
    """Flattened defect params, each carrying its render_px so sizes can be put
    on the same 640 px footing as the real boxes."""
    out = []
    for rec in synth_manifest(pool):
        rpx = rec.get("render_px") or DET_PX
        for prm in rec.get("params", []):
            d = dict(prm)
            d["render_px"] = rpx
            d["side_640"] = prm["size_px"] * (DET_PX / rpx)
            out.append(d)
    return out


def synth_label_boxes(pool: str, limit: int = 600) -> list[tuple[int, float, float]]:
    """Read the EMITTED yolo labels, not the sampled theta.

    This is the distinction that mattered: theta said one thing and the label
    written to disk said another, because the mask threshold shrank the box. The
    label is what the detector actually trains on, so the label is what gets
    plotted.
    """
    ld = SYNTH / pool / "labels"
    if not ld.is_dir():
        return []
    out = []
    for f in _spread(sorted(ld.glob("*.txt")), limit):
        for line in f.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cls, _, _, w, h = int(parts[0]), *[float(v) for v in parts[1:5]]
            if w > 0 and h > 0:
                out.append((cls, math.sqrt(w * h) * DET_PX, w / h))
    return out


def synth_geometry(limit: int = 600) -> tuple[list, str]:
    """Synthetic box geometry, drawn ONCE across pools.

    The refiner rewrites texture inside the mask and never touches the label,
    so every pool built from the same renderer output carries byte-identical
    boxes. Plotting three overlapping curves hid that invariant instead of
    stating it; this collapses them and reports if the assumption ever breaks
    (which would itself be a bug worth seeing).
    """
    per = {p: synth_label_boxes(p, limit) for p in POOLS}
    per = {k: v for k, v in per.items() if v}
    if not per:
        return [], ""
    keys = list(per)
    base = per[keys[0]]
    same = [k for k in keys if per[k] == base]
    if len(same) == len(keys):
        return base, (f"synthetic, all {len(keys)} pools identical "
                      f"(refiner preserves geometry)")
    print(f"  NOTE geometry differs across pools; matching {same} only - "
          f"the refiner is not supposed to move a box")
    return base, f"synthetic ({keys[0]})"


def _bin_of(side: float) -> int:
    return int(np.searchsorted(BIN_EDGES, side, side="right"))


def _spread(items: list, cap: int) -> list:
    """Even stride through a sorted list, NOT the first `cap` entries.

    Pool filenames sort as syn_<pool>_pbi2_* before syn_<pool>_pinhole_*
    ("b" < "i"), so a prefix of 600 files was entirely PbI2 and every synthetic
    distribution in this file was silently single-class. A stride keeps both
    classes in proportion and stays deterministic.
    """
    if cap >= len(items):
        return list(items)
    step = len(items) / cap
    return [items[int(i * step)] for i in range(cap)]


# ----------------------------------------------------------------- figures ---
def fig_size_distribution() -> None:
    real = real_boxes("train")
    if not real:
        print("  skip size_distribution: no train.json")
        return
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    bins = np.logspace(np.log10(2), np.log10(300), 45)

    r = np.array([s for _, s, _ in real])
    ax.hist(r, bins=bins, density=True, histtype="stepfilled", alpha=0.45,
            color=C["real"], label=f"real expert boxes (n={len(r):,})")

    geom, glab = synth_geometry()
    if geom:
        s = np.array([v for _, v, _ in geom])
        ax.hist(s, bins=bins, density=True, histtype="step", linewidth=1.8,
                color=C["synth"], label=f"{glab} (n={len(s):,})")

    for e in BIN_EDGES:
        ax.axvline(e, color="#7a7a7a", ls="--", lw=0.8, zorder=0)
    ax.set_xscale("log")
    top = ax.get_ylim()[1]
    ax.set_ylim(0, top * 1.14)                 # headroom so T-labels clear the legend
    for x, t in zip([4.2, 11, 22, 62], ["T1", "T2", "T3", "T4"]):
        ax.text(x, top * 1.04, t, ha="center", fontsize=8, color="#5a5a5a")
    ax.set_xlabel("defect side length at 640 px detector input  (sqrt(w*h), pixels)")
    ax.set_ylabel("density")
    ax.set_title("Did the renderer reproduce the real defect scale distribution?")
    ax.legend(loc="upper right")
    despine(ax)
    note(ax, "Sizes are pixels at the 640 px detector input. No nm calibration "
             "exists for this corpus (JPEG re-encoding destroyed the FESEM "
             "pixel-size headers). Dashed lines are the pre-registered bin edges.")
    save(fig, "size_distribution", "generation")


def fig_scale_bin_balance() -> None:
    real = real_boxes("train")
    if not real:
        print("  skip scale_bin_balance: no train.json")
        return
    sources = [("real", np.array([_bin_of(s) for _, s, _ in real]), C["real"])]
    geom, glab = synth_geometry()
    if geom:
        sources.append((glab, np.array([_bin_of(s) for _, s, _ in geom]),
                        C["synth"]))

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    w = 0.8 / len(sources)
    x = np.arange(4)
    for i, (lab, b, col) in enumerate(sources):
        frac = [float((b == k).mean()) for k in range(4)]
        pos = x + (i - (len(sources) - 1) / 2) * w
        ax.bar(pos, frac, w, color=col, label=f"{lab} (n={len(b):,})")
        for p, f in zip(pos, frac):
            ax.text(p, f + 0.012, f"{f*100:.0f}", ha="center", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels([BIN_LABELS[k] for k in BIN_LABELS])
    ax.set_ylabel("fraction of defects")
    ax.set_title("Defect-scale balance per pool, against the pre-registered bins")
    ax.legend(ncol=2)
    despine(ax)
    save(fig, "scale_bin_balance", "generation")


def fig_aspect_distribution() -> None:
    real = real_boxes("train")
    if not real:
        print("  skip aspect_distribution: no train.json")
        return
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), sharey=True)
    bins = np.linspace(0.2, 3.0, 40)

    for ax, cls in zip(axes, (0, 1)):
        r = np.array([a for c, _, a in real if c == cls])
        drew = False
        if len(r):
            ax.hist(r, bins=bins, density=True, histtype="stepfilled",
                    alpha=0.45, color=C["real"], label=f"real (n={len(r):,})")
            ax.axvline(np.median(r), color=C["real"], lw=1.4)
            ax.text(np.median(r), ax.get_ylim()[1] * 0.88,
                    f" real med {np.median(r):.3f}", fontsize=7, color=C["real"])
            drew = True
        geom, glab = synth_geometry()
        lab = [a for c, _, a in geom if c == cls]
        if lab:
            s = np.array(lab)
            ax.hist(s, bins=bins, density=True, histtype="step", linewidth=1.7,
                    color=C["synth"], label=f"synthetic (n={len(s):,})")
            ax.axvline(np.median(s), color=C["synth"], lw=1.2, ls=":")
            ax.text(np.median(s), ax.get_ylim()[1] * 0.78,
                    f" synth med {np.median(s):.3f}", fontsize=7,
                    color=C["synth"])
            drew = True
        ax.axvline(1.0, color="#9a9a9a", lw=0.8, ls="--", zorder=0)
        ax.set_title(CLS_NAME[cls])
        ax.set_xlabel("box aspect ratio  w / h")
        despine(ax)
        if drew:
            ax.legend(fontsize=6.5)
    axes[0].set_ylabel("density")
    fig.suptitle("Regression evidence: emitted box aspect, synthetic vs real",
                 y=1.02, fontsize=10, fontweight="semibold")
    note(axes[0], "A mass exactly at 1.000 (grey dashed) was the old bug: a "
                  "cv2.resize to a square forced every synthetic box square "
                  "while real boxes sat near 0.79. Medians drawn as vertical "
                  "lines.")
    save(fig, "aspect_distribution", "generation")


def fig_severity_and_morphology() -> None:
    per_cls_sev: dict[str, list[float]] = defaultdict(list)
    per_cls_morph: dict[str, Counter] = defaultdict(Counter)
    total = 0
    for pool in ["controlled"]:               # theta is identical across pools
        for prm in synth_params(pool):
            k = prm.get("kind", "?")
            per_cls_sev[k].append(prm.get("severity", np.nan))
            per_cls_morph[k][prm.get("morphology", "?")] += 1
            total += 1
    if not total:
        print("  skip severity_and_morphology: no manifest params")
        return

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))
    ax = axes[0]
    for k, col in (("pbi2", C["synth"]), ("pinhole", C["fft_on"])):
        v = np.array([x for x in per_cls_sev.get(k, []) if np.isfinite(x)])
        if len(v):
            ax.hist(v, bins=30, density=True, histtype="step", linewidth=1.6,
                    color=col, label=f"{k} (n={len(v):,})")
    ax.set_xlabel("sampled severity  (dimensionless simulation control)")
    ax.set_ylabel("density")
    ax.set_title("Severity sampling")
    ax.legend()
    despine(ax)
    note(ax, "severity is a normalised RENDERER CONTROL, not a physical depth. "
             "It has no nm or volt interpretation.")

    ax = axes[1]
    kinds = [k for k in ("pbi2", "pinhole") if per_cls_morph.get(k)]
    morphs = sorted({m for k in kinds for m in per_cls_morph[k]})
    y = np.arange(len(morphs))
    h = 0.8 / max(1, len(kinds))
    for i, (k, col) in enumerate(zip(kinds, (C["synth"], C["fft_on"]))):
        counts = [per_cls_morph[k].get(m, 0) for m in morphs]
        ax.barh(y + (i - (len(kinds) - 1) / 2) * h, counts, h, color=col,
                label=f"{k} (n={sum(counts):,})")
    ax.set_yticks(y)
    ax.set_yticklabels(morphs)
    ax.set_xlabel("defects rendered")
    ax.set_title("Morphology mix")
    ax.legend()
    despine(ax)
    save(fig, "severity_and_morphology", "generation")


def _contrast_samples(items, img_root: Path, cap: int = 400) -> np.ndarray:
    """abs(mean inside box - mean of a surrounding ring), in grey levels."""
    vals, n = [], 0
    for fname, boxes in _spread(items, min(len(items), 200)) if items else []:
        if n >= cap:
            break
        im = cv2.imread(str(img_root / fname), cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        H, W = im.shape
        for b in boxes:
            if n >= cap:
                break
            _, cx, cy, w, h = b
            bw, bh = max(2, int(w * W)), max(2, int(h * H))
            x0, y0 = int(cx * W - bw / 2), int(cy * H - bh / 2)
            x1, y1 = x0 + bw, y0 + bh
            px0, py0 = max(0, x0 - bw), max(0, y0 - bh)
            px1, py1 = min(W, x1 + bw), min(H, y1 + bh)
            if x1 <= x0 + 1 or y1 <= y0 + 1:
                continue
            inner = im[max(0, y0):min(H, y1), max(0, x0):min(W, x1)]
            outer = im[py0:py1, px0:px1].astype(np.float64).copy()
            if inner.size == 0 or outer.size == 0:
                continue
            oy0, ox0 = max(0, y0) - py0, max(0, x0) - px0
            ring = outer.copy()
            ring[oy0:oy0 + inner.shape[0], ox0:ox0 + inner.shape[1]] = np.nan
            if np.all(np.isnan(ring)):
                continue
            vals.append(abs(float(inner.mean()) - float(np.nanmean(ring))))
            n += 1
    return np.array(vals)


def fig_defect_contrast() -> None:
    real_items = []
    p = SPLITS / "train.json"
    if p.exists():
        for r in json.loads(p.read_text(encoding="utf-8"))["records"]:
            if r.get("boxes"):
                real_items.append((r["file"], r["boxes"]))

    groups, labels, colors = [], [], []
    rv = _contrast_samples(real_items, CURATED)
    if len(rv):
        groups.append(rv); labels.append(f"real\nn={len(rv)}"); colors.append(C["real"])

    for pool in POOLS:
        ld, idir = SYNTH / pool / "labels", SYNTH / pool / "images"
        if not ld.is_dir() or not idir.is_dir():
            continue
        items = []
        for f in _spread(sorted(ld.glob("*.txt")), 120):
            boxes = []
            for line in f.read_text(encoding="utf-8").splitlines():
                q = line.split()
                if len(q) >= 5:
                    boxes.append([int(q[0])] + [float(v) for v in q[1:5]])
            img = next((f.stem + e for e in (".jpg", ".png")
                        if (idir / (f.stem + e)).exists()), None)
            if img and boxes:
                items.append((img, boxes))
        v = _contrast_samples(items, idir)
        if len(v):
            groups.append(v)
            labels.append(f"{POOL_LABEL[pool]}\nn={len(v)}")
            colors.append(POOL_COLOR[pool])

    if not groups:
        print("  skip defect_contrast: nothing readable")
        return

    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    bp = ax.boxplot(groups, patch_artist=True, showfliers=False, widths=0.55,
                    medianprops=dict(color="black", linewidth=1.4))
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col); patch.set_alpha(0.65)
    for i, g in enumerate(groups, start=1):
        ax.text(i, np.median(g), f"  {np.median(g):.1f}", va="center",
                fontsize=7.5, fontweight="semibold")
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("|mean(inside box) - mean(surrounding ring)|  (grey levels)")
    ax.set_title("Defect conspicuity: the renderer over-marks, the refiner corrects")
    despine(ax)
    note(ax, "Medians annotated. The renderer makes defects far more "
             "conspicuous than real FESEM; the refiner pulls them back toward "
             "the real distribution. Outliers hidden for legibility.")
    save(fig, "defect_contrast", "generation")


def _radial_spectrum(gray: np.ndarray, nbins: int = 64) -> np.ndarray:
    g = gray.astype(np.float64)
    g = (g - g.mean()) / (g.std() + 1e-8)
    n = min(g.shape)
    g = g[:n, :n] * np.outer(np.hanning(n), np.hanning(n))
    ps = np.abs(np.fft.fftshift(np.fft.fft2(g))) ** 2
    c = n // 2
    yy, xx = np.indices((n, n))
    r = np.sqrt((yy - c) ** 2 + (xx - c) ** 2)
    rb = (r / r.max() * (nbins - 1)).astype(int)
    out = np.zeros(nbins)
    for k in range(nbins):
        m = rb == k
        out[k] = ps[m].mean() if m.any() else np.nan
    return np.log10(out + 1e-12)


def _sample_spectra(img_dir: Path, files: list[str], cap: int = 40) -> np.ndarray:
    acc = []
    for f in _spread(files, cap):
        im = cv2.imread(str(img_dir / f), cv2.IMREAD_GRAYSCALE)
        if im is not None and min(im.shape) >= 64:
            acc.append(_radial_spectrum(im))
    return np.nanmean(np.vstack(acc), axis=0) if acc else np.array([])


def fig_radial_power_spectrum() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    drew = 0

    p = SPLITS / "train.json"
    if p.exists():
        files = [r["file"] for r in json.loads(p.read_text(encoding="utf-8"))["records"]]
        s = _sample_spectra(CURATED, files)
        if s.size:
            ax.plot(np.linspace(0, 1, len(s)), s, color=C["real"], lw=2.0,
                    label=f"real (n={min(40, len(files))})")
            drew += 1

    for pool in POOLS:
        idir = SYNTH / pool / "images"
        if not idir.is_dir():
            continue
        files = sorted(q.name for q in idir.iterdir()
                       if q.suffix.lower() in (".jpg", ".png"))
        s = _sample_spectra(idir, files)
        if s.size:
            ax.plot(np.linspace(0, 1, len(s)), s, color=POOL_COLOR[pool], lw=1.5,
                    label=f"{POOL_LABEL[pool]} (n={min(40, len(files))})")
            drew += 1

    if not drew:
        print("  skip radial_power_spectrum: no images")
        plt.close(fig)
        return
    ax.set_xlabel("normalised radial spatial frequency  (0 = DC, 1 = Nyquist)")
    ax.set_ylabel("log10 mean power")
    ax.set_title("Radial power spectrum - the axis the FFT discriminator targets")
    ax.legend()
    despine(ax)
    note(ax, "This is the H2 evidence axis: the Fourier discriminator branch is "
             "supposed to move the synthetic spectrum toward real. Judge H2 "
             "against downstream detection too - see the training figures.")
    save(fig, "radial_power_spectrum", "generation")


def fig_domain_gap_levels() -> None:
    p = OUT / "domain_gap.json"
    if not p.exists():
        print("  skip domain_gap_levels: outputs/domain_gap.json missing")
        return
    d = json.loads(p.read_text(encoding="utf-8"))

    # Structure is discovered rather than assumed: find pool -> {level: value}.
    pools: dict[str, dict[str, float]] = {}
    def _harvest(node, path=()):
        if isinstance(node, dict):
            nums = {k: v for k, v in node.items() if isinstance(v, (int, float))}
            if nums and len(path) >= 1:
                pools.setdefault(path[-1], {}).update(nums)
            for k, v in node.items():
                _harvest(v, path + (k,))
    _harvest(d)
    pools = {k: v for k, v in pools.items() if len(v) >= 2}
    if not pools:
        print(f"  skip domain_gap_levels: could not locate levels in {list(d)}")
        return

    levels = sorted({lv for v in pools.values() for lv in v})
    levels = [lv for lv in levels if any(lv in v for v in pools.values())][:6]
    ncol = min(4, len(levels))
    nrow = int(np.ceil(len(levels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.0 * nrow))
    axes = np.atleast_1d(axes).ravel()

    names = list(pools)
    cols = [POOL_COLOR.get(n, C["baseline"]) for n in names]
    for ax, lv in zip(axes, levels):
        vals = [pools[n].get(lv, np.nan) for n in names]
        ax.bar(range(len(names)), vals, color=cols, width=0.6)
        for i, v in enumerate(vals):
            if np.isfinite(v):
                ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=6.5)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([POOL_LABEL.get(n, n) for n in names], rotation=30,
                           ha="right", fontsize=6.5)
        ax.set_title(lv, fontsize=8.5)
        despine(ax)
    for ax in axes[len(levels):]:
        ax.set_visible(False)
    fig.suptitle("Domain gap to real, per level - separate axes on purpose",
                 y=1.01, fontsize=10, fontweight="semibold")
    fig.text(0.0, -0.04, "Levels differ in magnitude by orders of magnitude; a "
                         "shared y-axis would flatten the frequency level, which "
                         "is the one H2 is about. Lower is closer to real.",
             fontsize=7, color="#5a5a5a")
    save(fig, "domain_gap_levels", "generation")


def fig_sample_grid(n_rows: int = 4) -> None:
    cols = [("controlled", "renderer only"), ("refined", "refined (FFT on)"),
            ("refined_nofft", "refined (FFT off)")]
    cols = [(p, lab) for p, lab in cols if (SYNTH / p / "images").is_dir()]
    if not cols:
        print("  skip sample_grid: no synthetic image dirs")
        return

    stems = sorted(q.stem for q in (SYNTH / cols[0][0] / "images").iterdir()
                   if q.suffix.lower() in (".jpg", ".png"))
    shared = [s for s in stems
              if all(any((SYNTH / p / "images" / (s + e)).exists()
                         for e in (".jpg", ".png")) for p, _ in cols)]
    picks = _spread(shared, n_rows) if shared else _spread(stems, n_rows)
    if not picks:
        print("  skip sample_grid: no stems")
        return

    fig, axes = plt.subplots(len(picks), len(cols),
                             figsize=(3.0 * len(cols), 3.0 * len(picks)))
    axes = np.atleast_2d(axes)
    for i, stem in enumerate(picks):
        for j, (pool, lab) in enumerate(cols):
            ax = axes[i, j]
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            f = next((SYNTH / pool / "images" / (stem + e) for e in (".jpg", ".png")
                      if (SYNTH / pool / "images" / (stem + e)).exists()), None)
            im = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) if f else None
            if im is None:
                ax.text(0.5, 0.5, "missing", ha="center", va="center")
                continue
            ax.imshow(im, cmap="gray")
            lp = SYNTH / pool / "labels" / f"{stem}.txt"
            if lp.exists():
                H, W = im.shape
                for line in lp.read_text(encoding="utf-8").splitlines():
                    q = line.split()
                    if len(q) < 5:
                        continue
                    _, cx, cy, w, h = int(q[0]), *[float(v) for v in q[1:5]]
                    ax.add_patch(plt.Rectangle(
                        ((cx - w / 2) * W, (cy - h / 2) * H), w * W, h * H,
                        fill=False, edgecolor="#39ff88", lw=0.7))
            if i == 0:
                ax.set_title(lab, fontsize=9)
            if j == 0:
                ax.set_ylabel(stem[-12:], fontsize=6.5)
    fig.suptitle("Same rendered layout through each pool, with the emitted labels",
                 y=1.005, fontsize=10, fontweight="semibold")
    fig.text(0.0, -0.01, "Boxes are the labels written to disk, not the sampled "
                         "theta - the label is what the detector trains on.",
             fontsize=7, color="#5a5a5a")
    save(fig, "sample_grid", "generation")


def main() -> None:
    apply()
    print("[gen] generation figures -> outputs/figures/generation/")
    for fn in (fig_size_distribution, fig_scale_bin_balance,
               fig_aspect_distribution, fig_severity_and_morphology,
               fig_defect_contrast, fig_radial_power_spectrum,
               fig_domain_gap_levels, fig_sample_grid):
        print(f"- {fn.__name__}")
        try:
            fn()
        except Exception as exc:            # one bad figure must not kill the set
            print(f"  FAILED {fn.__name__}: {type(exc).__name__}: {exc}")

    # numbers worth quoting, printed so they are not eyeballed off the charts
    real = real_boxes("train")
    if real:
        print("\nkey medians (side length at 640 px / aspect w:h):")
        for cls in (0, 1):
            a = [x for c, _, x in real if c == cls]
            s = [x for c, x, _ in real if c == cls]
            if a:
                print(f"  real {CLS_NAME[cls]:8s} n={len(a):5d} "
                      f"side {np.median(s):6.2f} px  aspect {np.median(a):.3f}")
        for pool in POOLS:
            lab = synth_label_boxes(pool)
            for cls in (0, 1):
                a = [x for c, _, x in lab if c == cls]
                s = [x for c, x, _ in lab if c == cls]
                if a:
                    print(f"  {POOL_LABEL[pool]:20s} {CLS_NAME[cls]:8s} "
                          f"n={len(a):5d} side {np.median(s):6.2f} px  "
                          f"aspect {np.median(a):.3f}")


if __name__ == "__main__":
    main()
