#!/usr/bin/env python3
"""
Publication figures for MODEL TRAINING and DETECTION RESULTS.

CPU ONLY. Nothing in here imports torch or ultralytics and nothing loads a
checkpoint: every number is parsed out of CSV/JSON that earlier runs already
wrote to disk. A training job owns the only (4 GB) GPU, so touching it here
would be a good way to kill it.

Run:  py -3.10 figures/train_figures.py

House rule enforced throughout: no mean is drawn without its spread or its n,
no band is drawn where n == 1, every bar carries its numeric value, and a
metric that is not interpretable (PbI2 AP, 5 training images) is marked as such
inside the figure rather than left to the caption.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))          # so `figures.style` resolves when run as a script
from figures.style import (apply, save, despine, note, C,        # noqa: E402
                           BINS, BIN_LABELS, BIN_COLORS)

OUT = ROOT / "outputs"
EXP = ROOT / "experiments"
MASTER = OUT / "master_results.csv"
TEST_JSON = OUT / "final_test_results.json"
BASELINE = OUT / "microdefectcv_baseline_val.json"

# Locked test split: three seeds per arm, keyed exactly as final_test_results.json
REAL_KEYS = ["real_only_s1", "real_only_s2", "real_only_s42"]
SYNTH_KEYS = ["scale005_s1", "scale005_s2", "scale005_s42"]

# `regime` does not encode the synthetic fraction reliably (real_only rows carry
# synth_ratio=1.0, which is the REAL fraction), so the ladder x-axis comes from
# the regime name instead.
REGIME_RATIO = {"real_only": 0.0, "scale_002": 0.02, "scale_005": 0.05,
                "scale_010": 0.10, "scale_025": 0.25}

# The two regimes the argument rests on, matched by seed.
CURVE_RUNS = [f"real_only_yolo11s_seed{s}" for s in (1, 2, 42)] + \
             [f"scale_005_yolo11s_seed{s}" for s in (1, 2, 42)]
REPRESENTATIVE = "scale_005_yolo11s_seed42"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _skip(name: str, *paths) -> bool:
    """True (and a printed reason) if any input this figure needs is absent."""
    gone = [Path(p) for p in paths if not Path(p).exists()]
    if gone:
        print(f"  [skip] {name}: missing " + ", ".join(str(p) for p in gone))
    return bool(gone)


def msn(vals) -> tuple[float, float, int]:
    """mean, sample sd (nan when n<2 - never fake a spread), n."""
    a = np.asarray(list(vals), dtype=float)
    return float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else float("nan"), int(a.size)


def load_master() -> pd.DataFrame:
    df = pd.read_csv(MASTER)
    return df.drop_duplicates("exp_id", keep="last").reset_index(drop=True)


def load_curve(exp_id: str) -> pd.DataFrame | None:
    p = EXP / exp_id / "run" / "results.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df.columns = [c.strip() for c in df.columns]   # ultralytics pads its header
    return df


def per_bin(entry: dict) -> dict:
    """per_bin_at50 re-keyed onto style.BINS.

    style.py calls the top bin T4_medium_plus; configs/tiny_defect_bins.yaml and
    every JSON on disk call it T4_medium_up. style.py is not mine to edit, so the
    alias lives here.
    """
    pb = entry.get("per_bin_at50", {}) or {}
    out = {}
    for b in BINS:
        k = b if b in pb else b.replace("_plus", "_up")
        if k in pb:
            out[b] = pb[k]
    return out


def pct(a: float, b: float) -> str:
    if not a:
        return "n/a (0 base)"
    return f"{100.0 * (b - a) / a:+.0f}%"


# --------------------------------------------------------------------------- #
# 1. headline: locked test split, real-only vs +5% synthetic
# --------------------------------------------------------------------------- #
def fig_headline_test():
    if _skip("fig_headline_test", TEST_JSON):
        return
    d = json.load(open(TEST_JSON))
    if not all(k in d for k in REAL_KEYS + SYNTH_KEYS):
        print("  [skip] fig_headline_test: final_test_results.json lacks the "
              "3 + 3 seed keys it needs")
        return

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.4))
    off = np.array([-0.13, 0.0, 0.13])
    n_gt = d[REAL_KEYS[0]].get("n_gt")
    n_img = d[REAL_KEYS[0]].get("images")

    for ax, metric, label in zip(axes, ("mAP50", "mAP50_95"),
                                 ("mAP@0.5", "mAP@0.5:0.95")):
        real = [d[k][metric] for k in REAL_KEYS]
        syn = [d[k][metric] for k in SYNTH_KEYS]
        rm, rs, rn = msn(real)
        sm, ss, sn = msn(syn)

        ax.bar([0], [rm], 0.52, yerr=[rs], color=C["real"], label="real only",
               error_kw=dict(ecolor=C["ink"], capsize=5, lw=1.0))
        ax.bar([1], [sm], 0.52, yerr=[ss], color=C["synth"],
               label="real + 5% synthetic",
               error_kw=dict(ecolor=C["ink"], capsize=5, lw=1.0))
        ax.set_ylim(0, max(syn) * 1.6)

        # every seed drawn, so the reader sees the actual three numbers
        for x, vals in ((0, real), (1, syn)):
            ax.scatter(x + off, vals, s=26, zorder=5, facecolor="white",
                       edgecolor=C["ink"], linewidth=0.9)
        ax.scatter([], [], s=26, facecolor="white", edgecolor=C["ink"],
                   linewidth=0.9, label="individual seeds (1, 2, 42)")

        # the seed ranges do not overlap - show the gap rather than assert it
        gap_lo, gap_hi = max(real), min(syn)
        ax.axhspan(gap_lo, gap_hi, color=C["accent"], alpha=0.07, zorder=0)
        ax.axhline(gap_lo, color=C["real"], lw=0.8, ls=":", zorder=1)
        ax.axhline(gap_hi, color=C["synth"], lw=0.8, ls=":", zorder=1)
        # This sat at x=0.5 in DATA coords - i.e. in the gap between the two
        # bars - so it printed straight over them. Parked in reserved margin.
        ax.annotate(f"seed ranges\ndisjoint\n\nworst +5%\n{gap_hi:.4f}\n"
                    f"beats best\nreal-only\n{gap_lo:.4f}",
                    (1.44, (gap_lo + gap_hi) / 2), ha="left", va="center",
                    fontsize=6.4, color="#4a4a4a", linespacing=1.3)
        ax.set_xlim(-0.55, 2.3)          # reserve the right margin for it

        for x, m, s in ((0, rm, rs), (1, sm, ss)):
            ax.annotate(f"{m:.4f}", (x, m + (s if np.isfinite(s) else 0)),
                        xytext=(0, 8), textcoords="offset points", ha="center",
                        fontsize=8, fontweight="semibold")
            # pinned to y=0 this rendered half outside the axes; place it inside
            # the bar at a fraction of the bar's own height instead
            ax.annotate(f"sd {s:.4f}\nn=3 seeds", (x, m * 0.08),
                        ha="center", va="bottom", fontsize=6.8, color="white")

        t, p = stats.ttest_ind(syn, real, equal_var=False)
        ax.set_title(f"{label}   {sm / rm:.2f}$\\times$\n"
                     f"Welch t={t:.2f}, p={p:.4f}")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["real only", "+5% synthetic"])
        ax.set_ylabel(label)
        despine(ax)

    axes[0].legend(loc="upper left", fontsize=7)

    # Three very long caption lines made savefig(bbox="tight") widen the canvas
    # to fit them, which flung the two axes to the outer edges. Wrap instead.
    import textwrap
    fig.tight_layout(rect=(0, 0.19, 1, 1))
    fig.text(0.012, 0.155, "\n".join(textwrap.wrap(
        f"LOCKED TEST SPLIT ({n_img} images, {n_gt} ground-truth boxes), never used for "
        f"tuning or model selection. Bars = mean of n=3 seeds (1, 2, 42); error bars = "
        f"sample sd (ddof=1); open dots are the three individual seed values. Welch "
        f"two-sided t-test, unequal variance, n=3 per arm - at three seeds a p-value is a "
        f"weak instrument, and it is quoted only because the two seed ranges do not "
        f"overlap at all (shaded band). Ratios are ratios of the seed means.  "
        f"CONFOUND: both arms ran 100 epochs, but the synthetic arm has 660 training "
        f"images against 160, so it took 8,300 gradient steps against 2,000. This "
        f"comparison is not step-matched and a step-matched baseline is outstanding.",
        158)), fontsize=6.6, color="#5a5a5a", va="top", ha="left")
    save(fig, "headline_test_real_vs_synth", subdir="training")


# --------------------------------------------------------------------------- #
# 2. synthetic-ratio scaling ladder (validation)
# --------------------------------------------------------------------------- #
def fig_scaling_ladder():
    if _skip("fig_scaling_ladder", MASTER):
        return
    df = load_master()
    df = df[(df["model"] == "yolo11s") & (df["regime"].isin(REGIME_RATIO))]
    if df.empty:
        print("  [skip] fig_scaling_ladder: no yolo11s ladder regimes in "
              "master_results.csv")
        return
    df = df.assign(ratio=df["regime"].map(REGIME_RATIO))

    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.9))
    for ax, metric, label in zip(axes, ("mAP50", "mAP50_95"),
                                 ("mAP@0.5", "mAP@0.5:0.95")):
        # The curve SHAPE is seed 42 only: 2/10/25% were never repeated, so the
        # line through them is a single draw and is drawn dashed/open to say so.
        s42 = df[df["seed"] == 42].sort_values("ratio")
        ax.plot(s42["ratio"] * 100, s42[metric], ls="--", lw=1.1,
                color=C["baseline"], zorder=2,
                label="seed 42 only (curve shape, n=1 per point)")
        ax.scatter(s42["ratio"] * 100, s42[metric], s=34, facecolor="white",
                   edgecolor=C["baseline"], linewidth=1.1, zorder=3)

        # where a point IS replicated, put mean +- sd on top of it
        for ratio, g in df.groupby("ratio"):
            m, s, n = msn(g[metric])
            x = ratio * 100
            if n > 1:
                ax.errorbar([x], [m], yerr=[s], fmt="o", ms=6,
                            color=C["synth"] if ratio > 0 else C["real"],
                            ecolor=C["ink"], capsize=4, lw=1.0, zorder=5,
                            label=None)
            v = g[g["seed"] == 42][metric]
            y = float(v.iloc[0]) if len(v) else m
            ax.annotate(f"{y:.4f}", (x, y), xytext=(0, 10),
                        textcoords="offset points", ha="center", fontsize=7)
            ax.annotate(f"n={n}", (x, y), xytext=(0, -14),
                        textcoords="offset points", ha="center", fontsize=7,
                        color="#5a5a5a")

        best = s42.loc[s42[metric].idxmax()]
        ax.annotate(f"optimum ({best['ratio'] * 100:.0f}%)",
                    (best["ratio"] * 100, best[metric]), xytext=(16, 20),
                    textcoords="offset points", fontsize=7.5, color=C["accent"],
                    arrowprops=dict(arrowstyle="->", color=C["accent"], lw=0.9))
        ax.set_xlabel("synthetic images added (% of real train set)")
        ax.set_ylabel(f"{label} (val)")
        ax.set_title(f"{label} vs synthetic ratio")
        ax.set_xticks([0, 2, 5, 10, 25])
        ax.set_ylim(0, df[metric].max() * 1.4)
        despine(ax)

    ns = {int(r * 100): len(g) for r, g in df.groupby("ratio")}
    axes[0].legend(loc="lower right", fontsize=7)
    note(axes[0],
         "VALIDATION split, from outputs/master_results.csv (deduped on exp_id). Seeds per point: "
         + ", ".join(f"{k}%: n={v}" for k, v in sorted(ns.items())) + ".\n"
         "THE DASHED LINE IS ONE SEED (42). Only the points labelled n=3 are replicated and only "
         "those carry an error bar (sample sd, ddof=1). No confidence band is\n"
         "drawn anywhere: at n=1 there is nothing to be confident about. The shape of this curve "
         "- the apparent 5% peak and the 10-25% plateau - is a single-draw\n"
         "observation, and the only replicated evidence for an optimum is the 0% vs 5% contrast.")
    fig.tight_layout()
    save(fig, "scaling_ladder_val", subdir="training")


# --------------------------------------------------------------------------- #
# 3. training curves for the key runs
# --------------------------------------------------------------------------- #
def fig_training_curves():
    have = {e: c for e, c in ((e, load_curve(e)) for e in CURVE_RUNS) if c is not None}
    if not have:
        print("  [skip] fig_training_curves: no experiments/*/run/results.csv for "
              f"any of {CURVE_RUNS}")
        return
    for e in CURVE_RUNS:
        if e not in have:
            print(f"  [warn] fig_training_curves: no results.csv for {e}, omitted")

    ls_by_seed = {1: "-", 2: "--", 42: ":"}
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.9))

    for ax, col, label in ((axes[0], "metrics/mAP50(B)", "mAP@0.5"),
                           (axes[1], "metrics/mAP50-95(B)", "mAP@0.5:0.95")):
        for e, c in have.items():
            seed = int(e.rsplit("seed", 1)[1])
            ax.plot(c["epoch"], c[col], lw=1.1,
                    color=C["synth"] if e.startswith("scale") else C["real"],
                    ls=ls_by_seed.get(seed, "-"), label=e)
        ax.set_xlabel("epoch")
        ax.set_ylabel(f"{label} (val)")
        ax.set_title(f"{label} per epoch")
        despine(ax)
    axes[0].legend(fontsize=6.3, loc="upper left")

    # third panel: the three TRAINING losses for the seed-42 pair only. Six runs
    # times three losses is unreadable; log y because cls_loss starts an order up.
    ax = axes[2]
    loss_ls = {"train/box_loss": "-", "train/cls_loss": "--", "train/dfl_loss": ":"}
    pair = [e for e in ("real_only_yolo11s_seed42", "scale_005_yolo11s_seed42")
            if e in have]
    for e in pair:
        colour = C["synth"] if e.startswith("scale") else C["real"]
        for col, ls in loss_ls.items():
            ax.plot(have[e]["epoch"], have[e][col], lw=1.0, color=colour, ls=ls,
                    label=f"{e.split('_yolo11s_')[0]} {col.split('/')[1]}")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss (log scale)")
    ax.set_title(f"training losses, seed 42 pair (n={len(pair)} runs)")
    ax.legend(fontsize=6.2, ncol=2)
    despine(ax)

    lens = {e: int(c["epoch"].max()) for e, c in have.items()}
    if len(set(lens.values())) > 1:
        clip = ("Runs differ in length (" +
                ", ".join(f"{e.split('_yolo11s_')[0]}/s{e.rsplit('seed', 1)[1]}: {n} ep"
                          for e, n in lens.items()) +
                "); each line is drawn to its OWN final epoch - nothing is truncated, "
                "padded or extrapolated.")
    else:
        clip = (f"All {len(have)} runs shown ran {next(iter(set(lens.values())))} epochs; "
                f"no line is truncated, padded or extrapolated.")
    n_real = sum(1 for e in have if not e.startswith("scale"))
    note(axes[0],
         f"Per-epoch VALIDATION metrics from experiments/<exp_id>/run/results.csv. n={len(have)} "
         f"runs: {n_real} real-only + {len(have) - n_real} +5% synthetic, seeds matched\n"
         f"pairwise (1, 2, 42). EVERY LINE IS ONE RUN - nothing here is seed-averaged and no band "
         f"is drawn. Colour = regime, dash pattern = seed. The reported\n"
         f"checkpoint is best.pt, so the final-epoch value of a curve is not the reported number. "
         f"{clip}")
    fig.tight_layout()
    save(fig, "training_curves", subdir="training")


# --------------------------------------------------------------------------- #
# 4. train vs val loss for one representative run (overfitting check)
# --------------------------------------------------------------------------- #
def fig_loss_curves():
    c = load_curve(REPRESENTATIVE)
    if c is None:
        print(f"  [skip] fig_loss_curves: missing experiments/{REPRESENTATIVE}"
              f"/run/results.csv")
        return

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.7))
    best_ep = int(c.loc[c["metrics/mAP50(B)"].idxmax(), "epoch"])
    last = int(c["epoch"].max())
    lines = []
    for ax, kind in zip(axes, ("box", "cls", "dfl")):
        tr, va = f"train/{kind}_loss", f"val/{kind}_loss"
        ax.plot(c["epoch"], c[tr], lw=1.2, color=C["real"], label="train")
        ax.plot(c["epoch"], c[va], lw=1.2, color=C["synth"], label="val")
        vmin_ep = int(c.loc[c[va].idxmin(), "epoch"])
        ax.axvline(best_ep, color=C["baseline"], lw=0.8, ls=":")
        ax.axvline(vmin_ep, color=C["accent"], lw=0.8, ls="--")
        ax.annotate(f"val min @ ep {vmin_ep}/{last}\n({c[va].min():.3f})",
                    (vmin_ep, ax.get_ylim()[1]), xytext=(4, -8),
                    textcoords="offset points", fontsize=6.8, color=C["accent"],
                    va="top")
        ax.set_xlabel("epoch")
        ax.set_ylabel(f"{kind} loss")
        ax.set_title(f"{kind}: train vs val")
        ax.legend(fontsize=7)
        despine(ax)
        lines.append(f"{kind}: val min ep {vmin_ep}/{last}, tail rise "
                     f"{float(c[va].iloc[-1] - c[va].min()):+.3f}")

    note(axes[0],
         f"ONE representative run: {REPRESENTATIVE} (n=1 run - this is not an average and no "
         f"spread is claimed; the other five runs are in training_curves.png).\n"
         f"Dotted grey line = best-mAP50 epoch ({best_ep}/{last}); dashed red = val-loss minimum. "
         + "; ".join(lines) + ".\n"
         "A val minimum well before the last epoch with a rising tail is the overfitting "
         "signature. The reported checkpoint is best.pt, not last.pt, so training\n"
         "past that point does not by itself invalidate the reported metric.")
    fig.tight_layout()
    save(fig, "loss_curves_representative", subdir="training")


# --------------------------------------------------------------------------- #
# 5. per-bin test results: where the gain actually lives
# --------------------------------------------------------------------------- #
def fig_scale_bins():
    if _skip("fig_scale_bins", TEST_JSON):
        return
    d = json.load(open(TEST_JSON))
    have_r = [k for k in REAL_KEYS if k in d]
    have_s = [k for k in SYNTH_KEYS if k in d]
    if not (have_r and have_s):
        print("  [skip] fig_scale_bins: final_test_results.json lacks both arms")
        return

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.3))
    x = np.arange(len(BINS))
    w = 0.36
    n_gt = {}

    for ax, field, label in ((axes[0], "ap", "AP@0.5"),
                             (axes[1], "recall", "recall@0.5")):
        top = 0.0
        for i, b in enumerate(BINS):
            rv = [per_bin(d[k])[b][field] for k in have_r if b in per_bin(d[k])]
            sv = [per_bin(d[k])[b][field] for k in have_s if b in per_bin(d[k])]
            if not rv or not sv:
                print(f"  [warn] fig_scale_bins: bin {b} absent from the test JSON")
                continue
            n_gt[b] = per_bin(d[have_r[0]])[b].get("n_gt")
            rm, rs, rn = msn(rv)
            sm, ss, sn = msn(sv)
            top = max(top, rm + (rs if np.isfinite(rs) else 0),
                      sm + (ss if np.isfinite(ss) else 0))
            ax.bar(x[i] - w / 2, rm, w, yerr=rs, color=BIN_COLORS[i], alpha=0.42,
                   edgecolor=BIN_COLORS[i], lw=1.1,
                   error_kw=dict(ecolor=C["ink"], capsize=3.5, lw=0.9))
            ax.bar(x[i] + w / 2, sm, w, yerr=ss, color=BIN_COLORS[i],
                   edgecolor=BIN_COLORS[i], lw=1.1,
                   error_kw=dict(ecolor=C["ink"], capsize=3.5, lw=0.9))
            ax.annotate(f"{rm:.3f}", (x[i] - w / 2, rm), xytext=(0, 10),
                        textcoords="offset points", ha="center", fontsize=6.8)
            ax.annotate(f"{sm:.3f}", (x[i] + w / 2, sm), xytext=(0, 10),
                        textcoords="offset points", ha="center", fontsize=6.8)
            ax.annotate(pct(rm, sm), (x[i], max(rm, sm)), xytext=(0, 30),
                        textcoords="offset points", ha="center", fontsize=8.5,
                        fontweight="semibold",
                        color=C["fft_on"] if sm >= rm else C["synth"])
        ax.set_ylim(0, top * 1.55 if top else 1)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{BIN_LABELS[b]}\nn_gt={n_gt.get(b, '?')}" for b in BINS])
        ax.set_ylabel(f"{label} (test)")
        ax.set_title(f"{label} per pre-registered scale bin")
        despine(ax)

    axes[0].legend(handles=[Patch(facecolor="#8a8a8a", alpha=0.42, label="real only"),
                            Patch(facecolor="#8a8a8a", label="real + 5% synthetic")],
                   loc="upper left", fontsize=7.5)
    note(axes[0],
         f"LOCKED TEST SPLIT ({d[have_r[0]].get('images')} images, {d[have_r[0]].get('n_gt')} GT "
         f"boxes). Bars = mean of n={len(have_r)} seeds per arm (1, 2, 42); error bars = sample sd "
         f"(ddof=1); % = change\nin the seed mean. Bins are pre-registered on YOLO head strides "
         f"(configs/tiny_defect_bins.yaml), not fitted to these results. BOTH panels are shown "
         f"deliberately:\nby AP the gain shrinks monotonically with defect size but stays POSITIVE "
         f"at T4, while by RECALL it goes NEGATIVE at T4 - the 'gain vanishes on large\ndefects' "
         f"claim holds for recall, not for AP. Per-bin n_gt is small (T4 especially), so a "
         f"single-bin difference is a noisy quantity.")
    fig.tight_layout()
    save(fig, "scale_bins_test", subdir="training")


# --------------------------------------------------------------------------- #
# 6. zero-training classical baseline vs the detector
# --------------------------------------------------------------------------- #
def fig_classical_vs_deep():
    if _skip("fig_classical_vs_deep", BASELINE, TEST_JSON):
        return
    base = json.load(open(BASELINE))
    d = json.load(open(TEST_JSON))
    bb = per_bin(base)
    have_s = [k for k in SYNTH_KEYS if k in d]
    if not bb or not have_s:
        print("  [skip] fig_classical_vs_deep: no per-bin baseline or no +5% arm")
        return

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.3))
    x = np.arange(len(BINS))
    w = 0.36
    for ax, field, label in ((axes[0], "ap", "AP@0.5"),
                             (axes[1], "recall", "recall@0.5")):
        top = 0.0
        for i, b in enumerate(BINS):
            if b not in bb:
                print(f"  [warn] fig_classical_vs_deep: bin {b} absent from baseline")
                continue
            cv = float(bb[b][field])
            sv = [per_bin(d[k])[b][field] for k in have_s if b in per_bin(d[k])]
            sm, ss, sn = msn(sv) if sv else (float("nan"), float("nan"), 0)
            top = max(top, cv, (sm if np.isfinite(sm) else 0)
                      + (ss if np.isfinite(ss) else 0))
            ax.bar(x[i] - w / 2, cv, w, color=C["baseline"], edgecolor=C["ink"], lw=0.8)
            ax.bar(x[i] + w / 2, sm, w, yerr=ss, color=BIN_COLORS[i],
                   edgecolor=BIN_COLORS[i], lw=1.1,
                   error_kw=dict(ecolor=C["ink"], capsize=3.5, lw=0.9))
            # a true zero is an invisible bar; say it in words instead
            if cv == 0:
                ax.annotate("0.000\nnothing found\nat all",
                            (x[i] - w / 2, 0.0), xytext=(-14, 34),
                            textcoords="offset points", ha="center", fontsize=7,
                            color=C["synth"], fontweight="semibold",
                            arrowprops=dict(arrowstyle="-|>", color=C["synth"], lw=1.0))
            else:
                ax.annotate(f"{cv:.3f}", (x[i] - w / 2, cv), xytext=(0, 8),
                            textcoords="offset points", ha="center", fontsize=6.8)
            ax.annotate(f"{sm:.3f}", (x[i] + w / 2, sm), xytext=(0, 10),
                        textcoords="offset points", ha="center", fontsize=6.8)
        ax.set_ylim(0, top * 1.6 if top else 1)
        ax.set_xticks(x)
        ax.set_xticklabels([BIN_LABELS[b] for b in BINS])
        ax.set_ylabel(label)
        ax.set_title(f"MicroDefectCV vs YOLO11s + 5%: {label}")
        despine(ax)

    axes[0].legend(handles=[
        Patch(facecolor=C["baseline"],
              label=f"MicroDefectCV, 0 trainable params, {base.get('seconds_per_image')} s/img "
                    f"(val, n=1)"),
        Patch(facecolor=BIN_COLORS[2],
              label=f"YOLO11s + 5% synthetic (test, mean of n={len(have_s)} seeds)")],
        loc="upper left", fontsize=7)
    note(axes[0],
         f"SPLITS DIFFER - indicative, not a matched comparison. MicroDefectCV is the zero-training "
         f"classical baseline on the VALIDATION split ({base.get('images')} images,\n"
         f"{base.get('n_gt')} GT boxes; n=1 deterministic run, so there is no seed variance to "
         f"show and none is drawn). YOLO11s+5% is the LOCKED TEST split "
         f"({d[have_s[0]].get('images')} images,\n{d[have_s[0]].get('n_gt')} GT boxes; mean of "
         f"n={len(have_s)} seeds, error bars = sample sd). No per-bin YOLO numbers exist on val in "
         f"any artefact on disk, so a like-for-like per-bin\ncomparison is not available without "
         f"re-running inference. MicroDefectCV scores exactly 0.000 AP and 0.000 recall in T1: it "
         f"finds nothing whatsoever below one\nP3 stride cell, which is precisely the size regime "
         f"this work is about.")
    fig.tight_layout()
    save(fig, "classical_vs_deep_bins", subdir="training")


# --------------------------------------------------------------------------- #
# 7. what the accuracy costs
# --------------------------------------------------------------------------- #
def fig_efficiency():
    if _skip("fig_efficiency", MASTER):
        return
    df = load_master()
    if df.empty:
        print("  [skip] fig_efficiency: master_results.csv has no rows")
        return

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    smin, smax = float(df["train_seconds"].min()), float(df["train_seconds"].max())

    def size(sec: float) -> float:
        f = 0.0 if smax == smin else (float(sec) - smin) / (smax - smin)
        return 40 + 380 * f

    for _, r in df.iterrows():
        ax.scatter(r["params_M"], r["mAP50"], s=size(r["train_seconds"]),
                   color=C["real"] if r["regime"] == "real_only" else C["synth"],
                   alpha=0.72, edgecolor=C["ink"], lw=0.7, zorder=3)
        ax.annotate(f"{r['exp_id']}\n{r['model']}, {r['params_M']:.2f} M, "
                    f"{r['train_seconds']:.0f} s, mAP50={r['mAP50']:.4f}",
                    (r["params_M"], r["mAP50"]), xytext=(13, -2),
                    textcoords="offset points", fontsize=6.3, va="center")

    if BASELINE.exists():
        b = json.load(open(BASELINE))
        ax.scatter([0.0], [b["mAP50"]], s=70, marker="D", color=C["baseline"],
                   edgecolor=C["ink"], lw=0.8, zorder=3)
        ax.annotate(f"MicroDefectCV (classical)\n0 params, 0 s training, "
                    f"{b['seconds_per_image']} s/img, mAP50={b['mAP50']:.4f}",
                    (0.0, b["mAP50"]), xytext=(10, -2), textcoords="offset points",
                    fontsize=6.3, va="center")
    else:
        print("  [warn] fig_efficiency: microdefectcv_baseline_val.json missing, "
              "zero-parameter point omitted")

    models = sorted(df["model"].unique())
    ax.set_xlim(-1.6, float(df["params_M"].max()) * 1.75)
    ax.set_ylim(0, max(float(df["mAP50"].max()) * 1.3, 0.02))
    ax.set_xlabel("trainable parameters (M)")
    ax.set_ylabel("mAP@0.5 (val)")
    ax.set_title("accuracy vs model size; marker area = wall-clock training seconds")
    ax.legend(handles=[Patch(facecolor=C["real"], label="real only"),
                       Patch(facecolor=C["synth"], label="real + synthetic"),
                       Patch(facecolor=C["baseline"], label="classical baseline")],
              loc="upper left", fontsize=7.5)
    despine(ax)
    note(ax,
         f"One dot per EXPERIMENT (n={len(df)} runs, deduped on exp_id keeping the last "
         f"occurrence) - each dot is a single run, not a seed mean, so no spread is\n"
         f"drawn. master_results.csv currently holds only {len(models)} detector architecture "
         f"({', '.join(models)}, {df['params_M'].iloc[0]:.2f} M params), so the parameter axis has "
         f"exactly one\nnon-zero value: the vertical spread is seed and synthetic ratio, NOT "
         f"capacity. Marker area is linear over {smin:.0f}-{smax:.0f} s of training and is not "
         f"comparable to the\nbaseline diamond (zero training, but {json.load(open(BASELINE))['seconds_per_image'] if BASELINE.exists() else '?'} s per image at inference). "
         f"mAP50 is the validation split for both methods, so the accuracy axis is like-for-like.")
    fig.tight_layout()
    save(fig, "efficiency_params_vs_map", subdir="training")


# --------------------------------------------------------------------------- #
# 8. per-class AP, with the uninterpretable class marked as such
# --------------------------------------------------------------------------- #
def fig_per_class_ap():
    if _skip("fig_per_class_ap", MASTER):
        return
    df = load_master()
    rows = []
    for e in df["exp_id"]:
        p = EXP / e / "metrics.json"
        if not p.exists():
            print(f"  [warn] fig_per_class_ap: no experiments/{e}/metrics.json, omitted")
            continue
        pc = json.load(open(p)).get("per_class") or {}
        if pc:
            rows.append((e, pc))
        else:
            print(f"  [warn] fig_per_class_ap: {e}/metrics.json has no per_class, omitted")
    if not rows:
        print("  [skip] fig_per_class_ap: no metrics.json carries a per_class dict")
        return

    classes = sorted({c for _, pc in rows for c in pc})
    fig, ax = plt.subplots(figsize=(max(7.8, 1.15 * len(rows)), 4.6))
    x = np.arange(len(rows))
    w = 0.8 / len(classes)
    bad = {}

    for j, cls in enumerate(classes):
        vals = [pc.get(cls, {}).get("AP50", np.nan) for _, pc in rows]
        interp = [bool(pc.get(cls, {}).get("interpretable", True)) for _, pc in rows]
        n_img = sorted({pc.get(cls, {}).get("train_images") for _, pc in rows}
                       - {None})
        xs = x + (j - (len(classes) - 1) / 2) * w
        ok = all(interp)
        if not ok:
            bad[cls] = n_img
        ax.bar(xs, vals, w * 0.9,
               color=BIN_COLORS[j % len(BIN_COLORS)] if ok else "#c9ccce",
               edgecolor=C["ink"] if ok else C["synth"],
               hatch=None if ok else "////", lw=0.8,
               label=f"{cls} (train images {n_img})"
                     + ("" if ok else " - NOT INTERPRETABLE"))
        for xi, v, iv in zip(xs, vals, interp):
            if np.isfinite(v):
                ax.annotate(f"{v:.4f}", (xi, v), xytext=(0, 4),
                            textcoords="offset points", ha="center", va="bottom",
                            fontsize=6.4, rotation=90,
                            color=C["ink"] if iv else C["synth"])

    ax.set_ylim(0, ax.get_ylim()[1] * 1.45)
    for cls, n in bad.items():
        ax.annotate(f"{cls}: AP@0.5 = 0.0000 in every run, {n} training image(s). "
                    f"metrics.json flags interpretable=False.\nThis is an EMPTY measurement, not "
                    f"evidence of a hard class: the hatched grey bars are placeholders\nand must "
                    f"not be read as a result of any kind.",
                    (0.5, 0.97), xycoords="axes fraction", ha="center", va="top",
                    fontsize=7.2, color=C["synth"],
                    bbox=dict(boxstyle="round,pad=0.35", fc="#fdf3f1",
                              ec=C["synth"], lw=0.8))
    ax.set_xticks(x)
    ax.set_xticklabels([e.replace("_yolo11s_", "\n") for e, _ in rows], fontsize=6.6)
    ax.set_ylabel("per-class AP@0.5 (val)")
    ax.set_title(f"per-class AP@0.5 by experiment (n={len(rows)} runs, 1 run per group)")
    ax.legend(loc="upper left", fontsize=7)
    despine(ax)
    note(ax,
         f"n={len(rows)} runs, ONE run per bar group - no seed averaging, no spread claimed; "
         f"validation split, from experiments/<exp_id>/metrics.json per_class.\n"
         f"Interpretability is taken from the file itself (per_class[cls]['interpretable'] and "
         f"['train_images']), not from a judgement made in this script. A class with a\n"
         f"handful of training images cannot produce a meaningful AP either way, so its bar is "
         f"hatched and greyed rather than plotted as though it were a measurement.")
    fig.tight_layout()
    save(fig, "per_class_ap50", subdir="training")


# --------------------------------------------------------------------------- #
# sanity check against the documented headline numbers
# --------------------------------------------------------------------------- #
DOCUMENTED = {
    "test real_only mAP50 mean": 0.0688, "test real_only mAP50 sd": 0.0105,
    "test scale005 mAP50 mean": 0.1333, "test scale005 mAP50 sd": 0.0128,
    "test mAP50 ratio": 1.94, "test mAP50 welch p": 0.0028,
    "test real_only mAP50_95 mean": 0.0261, "test scale005 mAP50_95 mean": 0.0521,
    "test mAP50_95 ratio": 2.00, "test mAP50_95 welch p": 0.0065,
    "test T1 AP50 real": 0.039, "test T1 AP50 synth": 0.044,
    "test T2 AP50 real": 0.316, "test T2 AP50 synth": 0.398,
    "test T3 AP50 real": 0.384, "test T3 AP50 synth": 0.409,
    "test T4 AP50 real": 0.315, "test T4 AP50 synth": 0.302,
    "val ladder 0%": 0.0583, "val ladder 2%": 0.1020, "val ladder 5%": 0.1249,
    "val ladder 10%": 0.1010, "val ladder 25%": 0.1002,
}


def sanity() -> None:
    """Print parsed vs documented for every headline number. A DISAGREE line here
    is a finding to chase in the source data, not a plotting bug: the figures
    always plot the parsed value."""
    print("\nsanity check (parsed vs documented):")
    got, extra = {}, {}
    if TEST_JSON.exists():
        d = json.load(open(TEST_JSON))
        for metric in ("mAP50", "mAP50_95"):
            r = [d[k][metric] for k in REAL_KEYS if k in d]
            s = [d[k][metric] for k in SYNTH_KEYS if k in d]
            if not (r and s):
                continue
            rm, rs, _ = msn(r)
            sm, ss, _ = msn(s)
            got[f"test real_only {metric} mean"] = rm
            got[f"test scale005 {metric} mean"] = sm
            got[f"test {metric} ratio"] = sm / rm
            got[f"test {metric} welch p"] = float(
                stats.ttest_ind(s, r, equal_var=False)[1])
            if metric == "mAP50":
                got["test real_only mAP50 sd"] = rs
                got["test scale005 mAP50 sd"] = ss
        for i, b in enumerate(BINS, start=1):
            for tag, keys in (("real", REAL_KEYS), ("synth", SYNTH_KEYS)):
                for field, slot in (("ap", got), ("recall", extra)):
                    v = [per_bin(d[k])[b][field] for k in keys
                         if k in d and b in per_bin(d[k])]
                    if v:
                        slot[f"test T{i} {'AP50' if field == 'ap' else 'recall'} {tag}"] = \
                            msn(v)[0]
    if MASTER.exists():
        df = load_master()
        df = df[(df["model"] == "yolo11s") & (df["seed"] == 42)]
        for regime, ratio in REGIME_RATIO.items():
            v = df[df["regime"] == regime]["mAP50"]
            if len(v):
                got[f"val ladder {int(ratio * 100)}%"] = float(v.iloc[0])

    for k, doc in DOCUMENTED.items():
        if k not in got:
            print(f"  ??       {k:30s} documented {doc:<8} parsed MISSING")
            continue
        tol = 0.006 if "ratio" in k else 0.0006
        ok = abs(got[k] - doc) <= tol
        line = f"  {'OK      ' if ok else 'DISAGREE'} {k:30s} documented {doc:<8} " \
               f"parsed {got[k]:.4f}"
        alt = k.replace("AP50", "recall")
        if not ok and alt in extra and abs(extra[alt] - doc) <= tol:
            line += f"   <- documented value matches per-bin RECALL ({extra[alt]:.4f}), not AP"
        print(line)


def main() -> None:
    apply()
    print(f"writing figures under {OUT / 'figures' / 'training'}")
    for fn in (fig_headline_test, fig_scaling_ladder, fig_training_curves,
               fig_loss_curves, fig_scale_bins, fig_classical_vs_deep,
               fig_efficiency, fig_per_class_ap):
        print(f"- {fn.__name__}")
        fn()
    sanity()


if __name__ == "__main__":
    main()
