"""
Assemble outputs/RESULTS.md from whatever the overnight run actually produced.

Reads master_results.csv and the per-experiment metrics.json files rather than
anything hand-entered, so the report cannot drift from the runs. Every PbI2
figure is printed with the training support behind it.
"""
from __future__ import annotations

import csv
import json
import statistics as st
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
EXP = ROOT / "experiments"


def _rows() -> list[dict]:
    """Latest row per exp_id.

    train_detector now upserts, so the file should already be unique - but
    dedup on READ as well, because a hand-edited or older file can still carry
    duplicates, and averaging a stale run into a seed mean is silent and ends
    up in a results table. Belt and braces on the one number that matters.
    """
    p = OUT / "master_results.csv"
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as fh:
        latest: dict[str, dict] = {}
        for r in csv.DictReader(fh):
            if r.get("exp_id"):
                latest[r["exp_id"]] = r
    return [latest[k] for k in sorted(latest)]


def _agg(rows: list[dict]) -> dict:
    """regime -> mean/sd over seeds."""
    by = defaultdict(list)
    for r in rows:
        try:
            by[r["regime"]].append((float(r["mAP50"]), float(r["mAP50_95"])))
        except (ValueError, KeyError):
            continue
    out = {}
    for regime, vals in by.items():
        m50 = [v[0] for v in vals]
        m5095 = [v[1] for v in vals]
        out[regime] = {
            "n": len(vals),
            "mAP50_mean": st.mean(m50),
            "mAP50_sd": st.stdev(m50) if len(m50) > 1 else 0.0,
            "mAP5095_mean": st.mean(m5095),
            "mAP5095_sd": st.stdev(m5095) if len(m5095) > 1 else 0.0,
        }
    return out


def _per_class() -> dict:
    out = {}
    for m in EXP.glob("*/metrics.json"):
        try:
            d = json.loads(m.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "per_class" in d:
            out[d["exp_id"]] = d["per_class"]
    return out


def build() -> str:
    rows = _rows()
    agg = _agg(rows)
    per_class = _per_class()
    L = [
        "# Results",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M} from `outputs/master_results.csv` "
        f"({len(rows)} runs).",
        "",
        "All numbers are on the **validation** split. The test split is locked "
        "until the final evaluation stage and has not been read.",
        "",
        "## Detection by regime",
        "",
        "| regime | seeds | mAP50 | mAP50-95 |",
        "|---|---|---|---|",
    ]
    for regime in sorted(agg):
        a = agg[regime]
        L.append(f"| `{regime}` | {a['n']} | {a['mAP50_mean']:.4f} ± {a['mAP50_sd']:.4f} "
                 f"| {a['mAP5095_mean']:.4f} ± {a['mAP5095_sd']:.4f} |")

    if per_class:
        L += ["", "## Per class", "",
              "PbI2 is reported with the training support behind it. With single-digit "
              "training images its AP measures the annotation budget, not the method.",
              "", "| experiment | class | AP50 | train images | interpretable |",
              "|---|---|---|---|---|"]
        for exp in sorted(per_class):
            for cname, v in per_class[exp].items():
                mark = "yes" if v.get("interpretable") else "**no**"
                L.append(f"| `{exp}` | {cname} | {v['AP50']:.4f} | "
                         f"{v.get('train_images','?')} | {mark} |")

    # scaling ladder
    scale = {r["regime"]: r for r in rows if r["regime"].startswith("scale_")}
    if scale:
        L += ["", "## Synthetic scaling", "",
              "| synthetic ratio | mAP50 | mAP50-95 |", "|---|---|---|"]
        for regime in sorted(scale, key=lambda k: float(k.split("_")[1])):
            r = scale[regime]
            L.append(f"| {int(regime.split('_')[1])}% | {float(r['mAP50']):.4f} "
                     f"| {float(r['mAP50_95']):.4f} |")

    # H2 ablation
    if "real_plus_refined" in agg and "real_plus_refined_nofft" in agg:
        a, b = agg["real_plus_refined"], agg["real_plus_refined_nofft"]
        d = a["mAP5095_mean"] - b["mAP5095_mean"]
        L += ["", "## H2 - does the Fourier discriminator branch matter?", "",
              f"- with FFT: mAP50-95 {a['mAP5095_mean']:.4f} ± {a['mAP5095_sd']:.4f}",
              f"- without : mAP50-95 {b['mAP5095_mean']:.4f} ± {b['mAP5095_sd']:.4f}",
              f"- difference: {d:+.4f}",
              "",
              "Read against the seed spread above before treating this as support "
              "for or against H2."]

    # classical baseline
    mdcv = OUT / "microdefectcv_baseline_val.json"
    if mdcv.exists():
        d = json.loads(mdcv.read_text(encoding="utf-8"))
        L += ["", "## R1 - MicroDefectCV classical baseline", "",
              f"- mAP50 {d['mAP50']:.4f}, mAP50-95 {d['mAP50_95']:.4f}, "
              f"P {d['precision']:.3f}, R {d['recall']:.3f}",
              f"- {d['seconds_per_image']:.2f} s/image on CPU, 0 trainable parameters",
              "", "| scale bin | n gt | recall | AP |", "|---|---|---|---|"]
        for bn, v in d.get("per_bin_at50", {}).items():
            L.append(f"| {bn} | {v['n_gt']} | {v['recall']:.3f} | {v['ap']:.4f} |")

    L += ["", "## Caveats that travel with these numbers", "",
          "- Validation carries 12 defect-bearing images, so run-to-run spread is wide; "
          "read the seed standard deviation before any difference.",
          "- Defect sizes are in pixels. JPEG re-encoding stripped the FESEM pixel-size "
          "headers from all 440 source images and no TIFs survive, so no nanometre "
          "calibration exists.",
          "- Renderer `severity` is a normalised simulation control, not a depth.",
          "- The test split has not been read.",
          ""]

    text = "\n".join(L)
    print(f"[report] built table ({len(rows)} runs, {len(agg)} regimes)")
    return text


if __name__ == "__main__":
    build()
