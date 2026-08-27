"""
Counterfactual severity probing (contribution N4).

data/synthetic/counterfactual holds one FIXED background rendered at five
severity rungs, with the defect layout held IDENTICAL across rungs - only
severity varies. So this asks something a normal test set cannot:

    as the visual evidence for a defect gets monotonically stronger, does the
    detector's response move monotonically too?

A detector that is genuinely reading defect contrast should show confidence and
detection count rising with severity. One that has latched onto layout, texture
priors, or position would stay flat. Spearman rho with a permutation test on 5
rungs, because n=5 makes the parametric p-value meaningless.

The rung-0 image is a genuinely clean canvas (no defect drawn at all), so it
doubles as a false-positive probe: anything detected there is a false alarm.
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import permutations
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CF = ROOT / "data" / "synthetic" / "counterfactual"
OUT = ROOT / "outputs"


def _exact_spearman_p(x: list[float], y: list[float]) -> tuple[float, float]:
    """Spearman rho plus an EXACT permutation p-value.

    With 5 rungs there are only 120 orderings, so the exact test is cheap and
    correct where the asymptotic p-value is not.
    """
    from scipy import stats
    rho, _ = stats.spearmanr(x, y)
    if np.isnan(rho):
        return float("nan"), float("nan")
    obs = abs(rho)
    hits = tot = 0
    for perm in permutations(range(len(y))):
        r, _ = stats.spearmanr(x, [y[i] for i in perm])
        if not np.isnan(r):
            tot += 1
            if abs(r) >= obs - 1e-12:
                hits += 1
    return float(rho), float(hits / tot) if tot else float("nan")


def probe(checkpoint: str, conf: float = 0.01, device: str = "0") -> dict:
    from ultralytics import YOLO

    ladder_path = CF / "ladder.json"
    if not ladder_path.exists():
        raise RuntimeError(f"no counterfactual ladder at {ladder_path} - run stage 3")
    ladder = json.loads(ladder_path.read_text(encoding="utf-8"))

    net = YOLO(checkpoint)
    rows = []
    for rung in ladder["rungs"]:
        img_p = CF / "images" / f"{rung['stem']}.jpg"
        if not img_p.exists():
            continue
        im = cv2.imread(str(img_p))
        res = net.predict(im, conf=conf, verbose=False, device=device)[0]
        n = len(res.boxes)
        confs = res.boxes.conf.cpu().numpy() if n else np.array([])
        # predicted area as a fraction of the frame, summed over detections
        area = 0.0
        if n:
            wh = res.boxes.xywhn.cpu().numpy()
            area = float((wh[:, 2] * wh[:, 3]).sum())
        rows.append({
            "severity": rung["severity"],
            "gt_boxes": rung["n_boxes"],
            "gt_mask_area": rung.get("mask_area_ratio", 0.0),
            "gt_contrast": rung.get("mean_contrast_vs_canvas", 0.0),
            "n_detections": n,
            "mean_conf": round(float(confs.mean()), 4) if n else 0.0,
            "max_conf": round(float(confs.max()), 4) if n else 0.0,
            "pred_area": round(area, 5),
        })

    sev = [r["severity"] for r in rows]
    out = {"checkpoint": checkpoint, "conf_threshold": conf, "rungs": rows,
           "monotonicity": {}}
    for key in ("n_detections", "mean_conf", "max_conf", "pred_area"):
        rho, p = _exact_spearman_p(sev, [r[key] for r in rows])
        out["monotonicity"][key] = {"spearman_rho": round(rho, 4) if rho == rho else None,
                                    "exact_p": round(p, 4) if p == p else None}

    clean = next((r for r in rows if r["severity"] <= 0.0), None)
    if clean:
        out["false_positives_on_clean_rung"] = clean["n_detections"]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "counterfactual.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

    print(f"[N4] {Path(checkpoint).parent.parent.parent.name}")
    print(f"  {'sev':>5} {'gt_box':>7} {'gt_contr':>9} {'ndet':>5} {'meanconf':>9} {'predarea':>9}")
    for r in rows:
        print(f"  {r['severity']:>5.2f} {r['gt_boxes']:>7} {r['gt_contrast']:>9.1f} "
              f"{r['n_detections']:>5} {r['mean_conf']:>9.4f} {r['pred_area']:>9.5f}")
    print("  monotonic response (Spearman rho, exact p over 120 permutations):")
    for k, v in out["monotonicity"].items():
        print(f"    {k:14} rho={v['spearman_rho']}  p={v['exact_p']}")
    if clean:
        print(f"  false positives on the CLEAN rung: {clean['n_detections']}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--conf", type=float, default=0.01)
    a = ap.parse_args()
    probe(a.checkpoint, conf=a.conf)
