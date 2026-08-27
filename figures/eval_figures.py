"""
Publication figures for the EVALUATION DIAGNOSTICS block: calibration,
robustness, failure taxonomy, counterfactual severity response, explainability.

Reads only the pre-computed JSON under outputs/. NO model loading, NO CUDA -
the GPU is busy and none of these figures need it.

Every panel carries its n. Several of these analyses rest on 3-21 samples and
the figures say so on their face rather than in a caption that gets dropped.

Run:  py -3.10 figures/eval_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt

from figures.style import (apply, save, despine, note, C, BINS, BIN_LABELS,
                           BIN_COLORS)

OUT = ROOT / "outputs"


def load(name: str):
    """Return parsed JSON or None (with a warning) so one missing input cannot
    take the whole figure set down."""
    p = OUT / f"{name}.json"
    if not p.exists():
        print(f"  [skip] {p.relative_to(ROOT)} missing")
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def stamp_n(ax, text: str, loc: str = "upper left") -> None:
    """n printed inside the axes, always, no exceptions."""
    x, ha = (0.02, "left") if "left" in loc else (0.98, "right")
    y, va = (0.97, "top") if "upper" in loc else (0.03, "bottom")
    ax.text(x, y, text, transform=ax.transAxes, fontsize=7.5, ha=ha, va=va,
            color="#333", bbox=dict(fc="white", ec="#cccccc", lw=0.6,
                                    boxstyle="round,pad=0.3", alpha=0.92))


# --------------------------------------------------------------------------- #
# 1. reliability diagram
# --------------------------------------------------------------------------- #
def fig_reliability(cal) -> None:
    rel = [b for b in cal["ece"]["reliability"] if b["n"] > 0]
    conf = np.array([b["conf"] for b in rel])
    acc = np.array([b["acc"] for b in rel])
    ns = np.array([b["n"] for b in rel], dtype=float)
    n_tot = cal["n"]
    n_bins = cal["ece"]["n_bins"]
    low2 = ns[:2].sum() / ns.sum() * 100.0

    fig = plt.figure(figsize=(5.6, 6.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.1], hspace=0.10)
    ax = fig.add_subplot(gs[0])
    axb = fig.add_subplot(gs[1], sharex=ax)

    lim = float(max(conf.max(), acc.max())) * 1.10
    ax.plot([0, lim], [0, lim], ls="--", lw=1.0, c=C["ink"],
            label="perfect calibration")
    # above the diagonal = accuracy exceeds confidence = UNDER-confident
    ax.fill_between([0, lim], [0, lim], [lim, lim], color=C["fft_on"],
                    alpha=0.08, lw=0)
    ax.fill_between([0, lim], [0, 0], [0, lim], color=C["synth"],
                    alpha=0.08, lw=0)
    ax.text(0.05 * lim, 0.92 * lim, "UNDER-confident\n(accuracy > confidence)",
            fontsize=7.5, color=C["fft_on"], va="top", fontweight="semibold")
    ax.text(0.96 * lim, 0.06 * lim, "OVER-confident\n(confidence > accuracy)",
            fontsize=7.5, color=C["synth"], va="bottom", ha="right")

    ax.plot(conf, acc, "-", lw=1.2, color=C["real"], zorder=3)
    ax.scatter(conf, acc, s=24 + 200 * ns / ns.max(), color=C["real"],
               edgecolor="white", lw=0.8, zorder=4,
               label="occupied bin (area proportional to n)")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.set_ylabel("observed accuracy (TP fraction)")
    ax.set_title("Confidence calibration: the model is mildly under-confident")
    ax.legend(loc="lower right")
    despine(ax)
    ax.tick_params(labelbottom=False)

    gap = float(cal["accuracy"] - cal["mean_confidence"])
    ax.text(0.97, 0.56,
            f"ECE      = {cal['ece']['ece']:.4f}\n"
            f"Brier    = {cal['brier']:.4f}\n"
            f"accuracy = {cal['accuracy']:.4f}\n"
            f"mean conf= {cal['mean_confidence']:.4f}\n"
            f"gap      = {gap:+.4f}  (under)",
            transform=ax.transAxes, fontsize=7.5, ha="right", va="top",
            family="monospace",
            bbox=dict(fc="#f7f7f7", ec="#bbbbbb", lw=0.7,
                      boxstyle="round,pad=0.45"))
    stamp_n(ax, f"n = {n_tot} detections, {n_bins} bins "
                f"({len(rel)} occupied)", loc="upper left")

    # bin populations - the top bins carry almost nothing and must look like it
    w = 1.0 / n_bins * 0.85
    axb.bar(conf, ns, width=w, color=C["baseline"], edgecolor="white", lw=0.5)
    for x, v in zip(conf, ns):
        axb.text(x, v + ns.max() * 0.05, f"{int(v)}", ha="center",
                 fontsize=6.5, color="#444")
    axb.set_ylim(0, ns.max() * 1.30)
    axb.set_ylabel("detections\nin bin")
    axb.set_xlabel("mean predicted confidence in bin")
    despine(axb)
    note(axb,
         "Bin population, not just bin position, decides how much a point means: "
         f"the two highest occupied bins hold {int(ns[-2])} and {int(ns[-1])} "
         "detections.\n"
         "Under-confidence is the safer direction for an inspection tool - it errs "
         "toward flagging for review rather than passing a defect - but ECE is\n"
         f"small here largely because {low2:.0f}% of detections sit in the two "
         "lowest bins, where confidence and accuracy are both near zero and so "
         "agree cheaply.")
    save(fig, "calibration_reliability", subdir="diagnostics")


# --------------------------------------------------------------------------- #
# 2. risk-coverage
# --------------------------------------------------------------------------- #
def fig_risk_coverage(cal) -> None:
    rc = cal["risk_coverage"]
    cov = np.asarray(rc["coverage"], dtype=float)
    risk = np.asarray(rc["risk"], dtype=float)
    n_tot = cal["n"]

    fig, ax = plt.subplots(figsize=(5.9, 4.5))
    ax.plot(cov, risk, lw=1.6, color=C["real"], label="selective risk")
    ax.fill_between(cov, 0, risk, color=C["real"], alpha=0.10, lw=0)
    ax.axhline(risk[-1], ls=":", lw=1.0, color=C["baseline"])
    ax.text(0.015, risk[-1] + 0.02,
            f"full-coverage risk = {risk[-1]:.4f}  (= 1 - accuracy)",
            fontsize=7.5, color="#444")

    # where does abstention actually buy anything
    for c_target in (0.10, 0.25, 0.50):
        i = min(int(np.searchsorted(cov, c_target)), len(cov) - 1)
        ax.plot([cov[i]], [risk[i]], "o", ms=4.5, color=C["accent"], zorder=5)
        ax.annotate(f"{cov[i] * 100:.0f}% coverage\nrisk {risk[i]:.3f}",
                    (cov[i], risk[i]), textcoords="offset points",
                    xytext=(7, -18), fontsize=7, color=C["accent"])

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("coverage (fraction of detections retained, "
                  "highest confidence first)")
    ax.set_ylabel("risk (error rate among retained detections)")
    ax.set_title("Risk-coverage: the score ranks better than it calibrates")
    ax.text(0.98, 0.07, f"AURC = {rc['aurc']:.4f}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9, family="monospace",
            bbox=dict(fc="#f7f7f7", ec="#bbbbbb", lw=0.7,
                      boxstyle="round,pad=0.45"))
    stamp_n(ax, f"n = {n_tot} detections, {len(cov)} curve points (val split)",
            loc="upper left")
    ax.legend(loc="lower right")
    despine(ax)
    note(ax,
         f"Full per-detection sweep from the JSON ({len(cov)} points) - nothing "
         "interpolated or invented. Risk falls as coverage shrinks, so ranking by\n"
         f"confidence is informative, but AURC {rc['aurc']:.4f} against a "
         f"full-coverage floor of {risk[-1]:.4f} is a weak selective classifier: "
         "even after discarding\n90% of detections the majority of those retained "
         "are still wrong. Left-hand end is noisy by construction (1-10 "
         "detections).")
    save(fig, "calibration_risk_coverage", subdir="diagnostics")


# --------------------------------------------------------------------------- #
# 3. robustness - the central diagnostic
# --------------------------------------------------------------------------- #
def fig_robustness_heatmap(rob) -> None:
    rows = rob["rows"]
    base = rob["baseline"]
    perts = list(dict.fromkeys(r["perturbation"] for r in rows))
    by_p = {p: [r for r in rows if r["perturbation"] == p] for p in perts}
    ncol = max(len(v) for v in by_p.values())
    over = set(rob.get("overconfident", []))

    dmap = np.full((len(perts), ncol), np.nan)
    dcnf = np.full((len(perts), ncol), np.nan)
    mags = np.empty((len(perts), ncol), dtype=object)
    for i, p in enumerate(perts):
        for j, r in enumerate(by_p[p]):
            dmap[i, j] = r["delta_mAP_pct"]
            dcnf[i, j] = r["delta_conf_pct"]
            mags[i, j] = r["magnitude"]

    fig = plt.figure(figsize=(11.6, 9.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.30], hspace=0.45,
                          wspace=0.24)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    for ax, M, cmap, vmin, vmax, ttl in (
            (ax1, dmap, "RdYlGn", -100, 0, "delta mAP50 (%) vs clean"),
            (ax2, dcnf, "coolwarm_r", -60, 60,
             "delta mean confidence (%) vs clean")):
        im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(ncol))
        ax.set_xticklabels([f"magnitude {k + 1}" for k in range(ncol)])
        ax.set_yticks(range(len(perts)))
        ax.set_yticklabels([p.replace("_", " ") for p in perts])
        ax.set_title(ttl)
        ax.grid(False)
        for i in range(len(perts)):
            for j in range(ncol):
                if np.isnan(M[i, j]):
                    continue
                tag = f"{perts[i]}@{mags[i, j]}"
                is_o = tag in over
                ax.text(j, i, f"{M[i, j]:+.1f}\n({mags[i, j]:g})",
                        ha="center", va="center", fontsize=6.8,
                        fontweight="bold" if is_o else "normal", color="#111")
                if is_o:
                    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1,
                                               fill=False, ec=C["synth"],
                                               lw=2.0, zorder=5))
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
        stamp_n(ax, f"n = {len(rows)} conditions; red box = overconfident",
                loc="lower right")

    # ---- the scatter that makes silent failure unmissable -----------------
    tags, is_over, lmap, lcnf = [], [], [], []
    for p in perts:
        for r in by_p[p]:
            tags.append(f"{p}@{r['magnitude']:g}")
            is_over.append(f"{p}@{r['magnitude']}" in over)
            lmap.append(-r["delta_mAP_pct"])       # loss, positive = worse
            lcnf.append(-r["delta_conf_pct"])      # loss, negative = conf ROSE
    is_over = np.array(is_over)
    lmap = np.array(lmap)
    lcnf = np.array(lcnf)

    ax3.axhline(0, lw=0.8, color="#999")
    ax3.plot([-10, 110], [-10, 110], ls="--", lw=1.1, color=C["ink"],
             label="well-behaved: confidence falls as much as mAP")
    ax3.fill_between([-10, 110], [-30, -30], [-10, 110], color=C["synth"],
                     alpha=0.07, lw=0)
    ax3.text(78, 3, "SILENT-FAILURE ZONE\nmAP collapses, confidence does not",
             fontsize=9, color=C["synth"], fontweight="semibold",
             ha="center", va="bottom")

    cmap_p = {p: BIN_COLORS[k % len(BIN_COLORS)] for k, p in enumerate(perts)}
    for k, t in enumerate(tags):
        p = t.rsplit("@", 1)[0]
        ax3.scatter(lmap[k], lcnf[k], s=110 if is_over[k] else 68,
                    marker="X" if is_over[k] else "o", color=cmap_p[p],
                    edgecolor=C["synth"] if is_over[k] else "white",
                    lw=1.8 if is_over[k] else 0.8, zorder=4)
    for p in perts:
        ax3.scatter([], [], color=cmap_p[p], s=60, edgecolor="white",
                    label=p.replace("_", " "))
    ax3.scatter([], [], marker="X", s=110, color="#ffffff",
                edgecolor=C["synth"], lw=1.8,
                label=f"overconfident ({len(over)}/{len(rows)}: confidence ROSE)")

    for k, t in enumerate(tags):
        if lmap[k] > 70 or lcnf[k] < -6:
            ax3.annotate(t, (lmap[k], lcnf[k]), textcoords="offset points",
                         xytext=(8, 4), fontsize=6.8, color="#333")
    k_n = tags.index("noise@0.02")
    ax3.annotate("noise@0.02: mAP -96.1%, confidence only -13.5%\n"
                 "the model does not know it has failed",
                 (lmap[k_n], lcnf[k_n]), textcoords="offset points",
                 xytext=(-230, 46), fontsize=8.5, color=C["synth"],
                 fontweight="semibold",
                 arrowprops=dict(arrowstyle="->", color=C["synth"], lw=1.2))

    ax3.set_xlabel("mAP50 loss vs clean baseline (%)   ---->   worse")
    ax3.set_ylabel("mean-confidence loss (%)\n(negative = confidence went UP)")
    ax3.set_title("Confidence does not track degradation: points far below the "
                  "diagonal are failures the model reports as normal")
    ax3.set_xlim(-8, 112)
    ax3.set_ylim(-24, 64)
    ax3.legend(loc="upper left", ncol=2, fontsize=7.5)
    despine(ax3)
    stamp_n(ax3, f"n = {len(rows)} conditions (7 perturbations x 3 magnitudes); "
                 f"baseline mAP50 {base['mAP50']:.4f}, mean conf "
                 f"{base['mean_conf']:.4f}", loc="lower right")
    note(ax3,
         "One evaluation pass per condition on the val split - no repeats, so no "
         "error bars: read the shape, not the third decimal.\n"
         "The JSON's own 'confidence_tracks_degradation' flag is direction-only, "
         "and it calls noise@0.02 'tracking' because confidence did fall (by "
         "13.5%,\nwhile mAP fell 96.1%). The diagonal here is the magnitude test "
         "that flag does not apply. JPEG is harmless at every magnitude tested "
         "(<=4.2% mAP loss).")
    save(fig, "robustness_confidence_vs_degradation", subdir="diagnostics")


# --------------------------------------------------------------------------- #
# 4. failure taxonomy
# --------------------------------------------------------------------------- #
def fig_failure_taxonomy(fa) -> None:
    c = fa["counts"]
    miss_keys = ["missed_sub_stride", "missed_tiny", "missed_low_contrast",
                 "missed_dense_cluster", "missed_other"]
    fp_keys = ["fp_near_miss", "fp_on_grain_boundary", "fp_high_contrast_spot",
               "fp_other"]
    # T1/T2 size wording comes from the pre-registered bin labels, so the
    # taxonomy and the scale figures cannot drift apart.
    t1 = BIN_LABELS[BINS[0]].split("\n")[1]
    t2 = BIN_LABELS[BINS[1]].split("\n")[1]
    miss_lab = {"missed_sub_stride": f"sub-stride (T1, {t1})",
                "missed_tiny": f"tiny (T2, {t2})",
                "missed_low_contrast": "low contrast",
                "missed_dense_cluster": "dense cluster",
                "missed_other": "other"}
    fp_lab = {"fp_near_miss": "near-miss (IoU > 0.1 on a real defect)",
              "fp_on_grain_boundary": "on grain boundary",
              "fp_high_contrast_spot": "high-contrast spot",
              "fp_other": "other"}

    n_miss = sum(c[k] for k in miss_keys)
    n_fp = sum(c[k] for k in fp_keys)
    sub16 = c["missed_sub_stride"] + c["missed_tiny"]

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(12.0, 4.5))
    fig.subplots_adjust(wspace=0.42)

    for ax, keys, labs, tot, cols, ttl in (
            (axa, miss_keys, miss_lab, n_miss, BIN_COLORS + [C["baseline"]],
             "Missed defects by category"),
            (axb, fp_keys, fp_lab, n_fp,
             [C["real"], C["renderer"], C["synth"], C["baseline"]],
             "False positives by category")):
        vals = [c[k] for k in keys]
        y = np.arange(len(keys))[::-1]
        ax.barh(y, vals, color=cols[:len(keys)], edgecolor="white", lw=0.6,
                height=0.68)
        for yy, v in zip(y, vals):
            ax.text(v + tot * 0.012, yy, f"{v}   ({v / tot * 100:.1f}%)",
                    va="center", fontsize=8.2, color="#222")
        ax.set_yticks(y)
        ax.set_yticklabels([labs[k] for k in keys], fontsize=8.5)
        ax.set_xlim(0, max(vals) * 1.34)
        ax.set_xlabel("count")
        ax.set_title(ttl)
        despine(ax)

    stamp_n(axa, f"n = {n_miss} misses of {fa['total_gt']} GT boxes "
                 f"({fa['images']} images)", loc="lower right")
    stamp_n(axb, f"n = {n_fp} false positives ({fa['images']} images)",
            loc="lower right")
    axa.text(0.97, 0.44,
             f"{sub16 / n_miss * 100:.1f}% of misses are sub-16 px\n"
             f"(T1 + T2 = {sub16}/{n_miss}): a resolution\nlimit, not a semantic "
             f"one",
             transform=axa.transAxes, ha="right", va="top", fontsize=8,
             color=C["synth"], fontweight="semibold",
             bbox=dict(fc="#fdf3f1", ec=C["synth"], lw=0.7,
                       boxstyle="round,pad=0.4"))
    axb.text(0.97, 0.44,
             f"{c['fp_near_miss'] / n_fp * 100:.1f}% of FPs are near-misses on\n"
             f"genuine defects (IoU > 0.1 but under\nthe 0.5 match gate): "
             f"localisation\nerror, not hallucination",
             transform=axb.transAxes, ha="right", va="top", fontsize=8,
             color=C["real"], fontweight="semibold",
             bbox=dict(fc="#f0f4f8", ec=C["real"], lw=0.7,
                       boxstyle="round,pad=0.4"))
    note(axa,
         f"Nine pre-declared categories, assigned at conf {fa['conf_threshold']} "
         f"/ IoU {fa['iou_threshold']} over {fa['images']} images. Categories are "
         "mutually exclusive and assigned by a fixed\npriority order, so a small "
         "dense low-contrast defect is counted once under the first rule that "
         "fires - the boundary between adjacent categories is\nsofter than the "
         "counts look. Percentages are of misses (left) and of false positives "
         "(right), not of all GT.")
    save(fig, "failure_taxonomy", subdir="diagnostics")


# --------------------------------------------------------------------------- #
# 5. counterfactual severity ladder
# --------------------------------------------------------------------------- #
def fig_counterfactual(cf) -> None:
    rungs = cf["rungs"]
    sev = np.array([r["severity"] for r in rungs])
    ndet = np.array([r["n_detections"] for r in rungs], dtype=float)
    contr = np.array([r["gt_contrast"] for r in rungs])
    # mean_conf is stored as 0.0 where nothing was detected. That is "undefined",
    # not "zero confidence", and must not be drawn as a data point on the floor.
    conf = np.array([r["mean_conf"] if r["n_detections"] > 0 else np.nan
                     for r in rungs])

    mono = cf["monotonicity"]["n_detections"]
    rho = mono["spearman_rho"]
    exact_p = mono.get("exact_p")

    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    fig.subplots_adjust(right=0.78)
    ax2 = ax.twinx()
    ax3 = ax.twinx()
    ax3.spines["right"].set_position(("axes", 1.17))

    x = np.arange(len(sev))
    ax.axvspan(1.5, 2.5, color=C["renderer"], alpha=0.12, lw=0)
    ax.bar(x, ndet, width=0.5, color=C["real"], edgecolor="white", lw=0.6,
           zorder=3)
    for xx, v in zip(x, ndet):
        ax.text(xx, v + max(ndet) * 0.03, f"{int(v)}", ha="center", fontsize=8.5,
                color=C["real"], fontweight="semibold", zorder=4)

    ax2.plot(x, conf, "o-", color=C["synth"], lw=1.6, ms=6, zorder=5)
    for xx, v in zip(x, conf):
        if not np.isnan(v):
            ax2.text(xx, v + 0.014, f"{v:.4f}", ha="center", fontsize=8,
                     color=C["synth"], fontweight="semibold")
    ax3.plot(x, contr, "s--", color=C["renderer"], lw=1.3, ms=5, zorder=2)
    for xx, v in zip(x, contr):
        ax3.text(xx, v + max(contr) * 0.035, f"{v:.1f}", ha="center",
                 fontsize=7.5, color="#8a6220")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{s:.2f}" for s in sev])
    ax.set_xlabel("injected severity")
    ax.set_ylabel("detections (count)", color=C["real"])
    ax2.set_ylabel("mean confidence", color=C["synth"])
    ax3.set_ylabel("GT contrast (grey levels)", color=C["renderer"])
    ax.set_ylim(0, max(ndet) * 1.42)
    ax2.set_ylim(0, 0.36)
    ax3.set_ylim(0, max(contr) * 1.32)
    ax2.grid(False)
    ax3.grid(False)
    ax.set_title("Counterfactual severity ladder on one fixed background")

    ax.annotate(f"clean canvas:\n0 detections,\n"
                f"{cf['false_positives_on_clean_rung']} false alarms",
                (0, 0), textcoords="offset points", xytext=(6, 30),
                fontsize=7.5, color=C["fft_on"], fontweight="semibold")
    ax.text(2.0, max(ndet) * 1.30, "detection threshold\nlies in this gap",
            ha="center", fontsize=8, color="#8a6220", fontweight="semibold")
    ax.text(0.42, 0.70,
            f"Spearman rho = {rho:.4f}"
            + (f"\nreported exact p = {exact_p:.3f}" if exact_p is not None
               else ""),
            transform=ax.transAxes, ha="center", va="top", fontsize=8.5,
            family="monospace",
            bbox=dict(fc="#f7f7f7", ec="#bbbbbb", lw=0.7,
                      boxstyle="round,pad=0.4"))
    stamp_n(ax, f"n = {len(rungs)} rungs, 1 background, "
                f"{rungs[-1]['gt_boxes']} injected defects per non-clean rung",
            loc="upper left")

    handles = [plt.Rectangle((0, 0), 1, 1, color=C["real"]),
               plt.Line2D([], [], color=C["synth"], marker="o", lw=1.6),
               plt.Line2D([], [], color=C["renderer"], marker="s", ls="--",
                          lw=1.3)]
    ax.legend(handles, ["detections", "mean confidence", "GT contrast"],
              loc="upper left", bbox_to_anchor=(0.0, 0.85), fontsize=7.5)
    despine(ax)
    p_txt = f"{exact_p:.3f}" if exact_p is not None else "n/a"
    note(ax,
         "Five rungs is the entire sample. At n = 5 a two-sided exact permutation "
         "p-value cannot fall below 2/120 = 0.017, so p is bounded by rung count, "
         "not by\neffect size; the reported exact p here is "
         f"{p_txt}, further limited because the three zero-detection rungs are "
         "tied. Direction is the claim, not significance.\n"
         "Mean confidence is undefined (not zero) below severity 0.75 - the JSON "
         "stores 0.0 there and those markers are deliberately absent rather than "
         "plotted on the floor.")
    save(fig, "counterfactual_severity", subdir="diagnostics")


# --------------------------------------------------------------------------- #
# 6. explainability - a NOT-INTERPRETABLE result, drawn as one
# --------------------------------------------------------------------------- #
def fig_explainability(ex) -> None:
    per = ex["per_image"]
    n = len(per)
    names = [Path(p["file"]).parent.name.replace("_", " ") + "\n"
             + Path(p["file"]).name for p in per]
    gt = np.array([p["gt_area_fraction"] for p in per])
    bg = np.array([p["background_attribution_ratio"] for p in per])
    iou = np.array([p["explanation_iou"] for p in per])
    conc = np.array([p["attribution_concentration"] for p in per])
    hit = np.array([bool(p["pointing_hit"]) for p in per])
    hits = int(hit.sum())

    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.6))
    fig.subplots_adjust(hspace=0.72, wspace=0.30, top=0.855, bottom=0.14)
    (a, b), (cc, d) = axes

    fig.text(0.5, 0.972,
             "Occlusion attribution: NOT INTERPRETABLE AS REPORTED",
             ha="center", fontsize=13, fontweight="bold", color=C["synth"])
    fig.text(0.5, 0.935,
             f"n = {n} images. Per-image GT defect area spans "
             f"{gt.min() * 100:.1f}% to {gt.max() * 100:.1f}% of the frame, so "
             "'background attribution ratio' measures annotation density, not "
             "model attention.",
             ha="center", fontsize=8.5, color="#444")

    # (a) the four summary scalars
    labs = ["pointing game\naccuracy", "mean explanation\nIoU",
            "mean background\nattribution ratio",
            "mean attribution\nconcentration"]
    vals = [ex["pointing_game_accuracy"], ex["mean_explanation_iou"],
            ex["mean_background_attribution_ratio"],
            ex["mean_attribution_concentration"]]
    cols = [C["real"], C["real"], C["synth"], C["baseline"]]
    a.bar(range(4), vals, color=cols, edgecolor="white", lw=0.6, width=0.62)
    for i, v in enumerate(vals):
        a.text(i, v + max(vals) * 0.035, f"{v:.4g}", ha="center", fontsize=8.5,
               fontweight="semibold", color="#222")
    a.set_xticks(range(4))
    a.set_xticklabels(labs, fontsize=7.5)
    a.set_ylim(0, max(vals) * 1.28)
    a.set_title(f"Summary metrics (each a mean over {n} images)")
    stamp_n(a, f"n = {n} images; pointing game = {hits}/{n} hits", "upper left")
    despine(a)
    note(a, f"Pointing-game accuracy {ex['pointing_game_accuracy']:.4f} is "
            f"{hits} of {n}. One image flipping moves it by {1 / n * 100:.0f} "
            f"points, so the number has no\nusable precision and no interval "
            f"worth quoting. These bars are shown only because the scalars are "
            f"what the JSON reports.")

    # (b) the confound, per image
    y = np.arange(n)[::-1]
    b.barh(y + 0.18, gt, height=0.32, color=C["fft_off"], edgecolor="white",
           lw=0.5, label="GT defect area fraction")
    b.barh(y - 0.18, bg, height=0.32, color=C["synth"], edgecolor="white",
           lw=0.5, label="background attribution ratio")
    for yy, g, bb in zip(y, gt, bg):
        b.text(g + 0.02, yy + 0.18, f"{g:.4f}", va="center", fontsize=7.5)
        b.text(bb + 0.02, yy - 0.18, f"{bb:.4f}", va="center", fontsize=7.5)
    b.set_yticks(y)
    b.set_yticklabels(names, fontsize=7)
    b.set_xlim(0, 1.34)
    b.set_xlabel("fraction of frame")
    b.set_title("Why it is confounded: the two bars are near-complements")
    b.legend(loc="lower right", fontsize=7.5)
    stamp_n(b, f"n = {n} images", "upper right")
    despine(b)

    # (c) the same point as a scatter against the 1 - x identity
    cc.plot([0, 1], [1, 0], ls="--", lw=1.1, color=C["ink"],
            label="bg ratio = 1 - GT area (pure annotation density)")
    cc.scatter(gt, bg, s=95, color=C["synth"], edgecolor="white", lw=0.9,
               zorder=4)
    for g, bb, nm in zip(gt, bg, names):
        cc.annotate(nm.replace("\n", " "), (g, bb), textcoords="offset points",
                    xytext=(-4, -17), fontsize=6.8, color="#333", ha="left")
    resid = np.abs(bg - (1.0 - gt))
    cc.set_xlim(-0.06, 1.18)
    cc.set_ylim(-0.06, 1.18)
    cc.set_xlabel("GT defect area fraction")
    cc.set_ylabel("background attribution ratio")
    cc.set_title(f"All {n} images sit on the identity line")
    cc.legend(loc="upper center", fontsize=7.5)
    stamp_n(cc, f"n = {n} images; max |residual| = {resid.max():.4f}",
            "lower left")
    despine(cc)
    note(cc, "If this metric measured attention, points would scatter off the "
             f"line. They do not - the largest departure is {resid.max():.4f}. "
             "The metric is\nreporting how much of the frame is annotated as "
             "defect, which is a property of the labels, not of the model.")

    # (d) per-image values - no mean drawn over n=3
    w = 0.34
    xx = np.arange(n)
    d2 = d.twinx()
    d.bar(xx - w / 2, iou, width=w, color=C["real"], edgecolor="white", lw=0.5)
    d2.bar(xx + w / 2, conc, width=w, color=C["renderer"], edgecolor="white",
           lw=0.5)
    for i, v in enumerate(iou):
        d.text(i - w / 2, v + max(iou) * 0.05, f"{v:.4f}", ha="center",
               fontsize=7.5, color=C["real"])
    for i, v in enumerate(conc):
        d2.text(i + w / 2, v + max(conc) * 0.03, f"{v:.3f}", ha="center",
                fontsize=7.5, color="#8a6220")
    for i, h_ in enumerate(hit):
        d.text(i, -max(iou) * 0.16, "pointing HIT" if h_ else "pointing miss",
               ha="center", fontsize=7,
               color=C["fft_on"] if h_ else C["synth"], fontweight="semibold")
    d.set_xticks(xx)
    d.set_xticklabels(names, fontsize=7)
    d.set_ylim(-max(iou) * 0.28, max(iou) * 1.45)
    d2.set_ylim(0, max(conc) * 1.30)
    d2.grid(False)
    d.set_ylabel("explanation IoU", color=C["real"])
    d2.set_ylabel("attribution concentration", color=C["renderer"])
    d.set_title("Per image, so the spread is visible")
    stamp_n(d, f"n = {n} images (patch {ex['patch']}, stride {ex['stride']})",
            "upper right")
    despine(d)
    note(d, "The only pointing-game hit is the image whose annotated defect "
            f"covers {gt.max() * 100:.1f}% of the frame, where a hit is close to "
            "unavoidable. Explanation IoU is\n"
            f"<= {iou.max():.2f} everywhere. Treat this panel as a pipeline smoke "
            "test, not as evidence about what the detector attends to: it needs a "
            "frame-area-matched\nimage set and a far larger n before any claim "
            "here is supportable.")
    save(fig, "explainability_occlusion", subdir="diagnostics")


# --------------------------------------------------------------------------- #
def main() -> None:
    apply()
    print("eval diagnostics figures -> outputs/figures/diagnostics/")
    jobs = [
        ("calibration_val", fig_reliability),
        ("calibration_val", fig_risk_coverage),
        ("robustness", fig_robustness_heatmap),
        ("failure_analysis", fig_failure_taxonomy),
        ("counterfactual", fig_counterfactual),
        ("explainability", fig_explainability),
    ]
    for src, fn in jobs:
        d = load(src)
        if d is None:
            print(f"  [skip] {fn.__name__}: outputs/{src}.json not found")
            continue
        try:
            fn(d)
        except Exception as e:            # one bad figure must not kill the set
            print(f"  [FAIL] {fn.__name__}: {type(e).__name__}: {e}")
    print("done")


if __name__ == "__main__":
    main()
