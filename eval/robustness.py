"""
Robustness sweep (plan section 14).

Perturb the VALIDATION images and record (detection quality down, confidence
behaviour). The property being tested is not "does mAP survive" - it will not -
but whether the model's confidence FALLS as the image degrades. A detector that
stays confident while its accuracy collapses is the dangerous failure mode for
an inspection tool, because nothing downstream can tell that its output stopped
being trustworthy.

Uses val, never test: this is a diagnostic, not a headline number.
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

from eval.detection import evaluate as score_detections

CURATED = ROOT / "data" / "curated" / "images"
SPLITS = ROOT / "data" / "splits"
OUT = ROOT / "outputs"


# --------------------------------------------------------- perturbations ----

def p_identity(im, _): return im


def p_brightness(im, m):
    return np.clip(im.astype(np.float32) * (1.0 + m), 0, 255).astype(np.uint8)


def p_contrast(im, m):
    mu = im.mean()
    return np.clip((im.astype(np.float32) - mu) * (1.0 + m) + mu, 0, 255).astype(np.uint8)


def p_noise(im, m):
    rng = np.random.default_rng(0)          # fixed: perturbation must be reproducible
    return np.clip(im.astype(np.float32) + rng.normal(0, m * 255, im.shape),
                   0, 255).astype(np.uint8)


def p_blur(im, m):
    k = max(3, int(m * 20) | 1)
    return cv2.GaussianBlur(im, (k, k), 0)


def p_jpeg(im, m):
    q = int(np.clip(100 - m * 100, 5, 95))
    ok, enc = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else im


def p_downscale(im, m):
    """Resolution loss, then back up - simulates a lower-magnification capture."""
    h, w = im.shape[:2]
    s = max(0.15, 1.0 - m)
    small = cv2.resize(im, (max(1, int(w * s)), max(1, int(h * s))),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


PERTURBATIONS = {
    "brightness_up": (p_brightness, [0.2, 0.4, 0.6]),
    "brightness_down": (p_brightness, [-0.2, -0.4, -0.6]),
    "contrast_down": (p_contrast, [-0.3, -0.5, -0.7]),
    "noise": (p_noise, [0.02, 0.05, 0.10]),
    "blur": (p_blur, [0.2, 0.4, 0.6]),
    "jpeg": (p_jpeg, [0.3, 0.6, 0.85]),
    "downscale": (p_downscale, [0.25, 0.5, 0.7]),
}


# ---------------------------------------------------------------- driver ----

def _val_records():
    recs = json.loads((SPLITS / "val.json").read_text(encoding="utf-8"))["records"]
    return [r for r in recs if r["n_boxes"] > 0]


def _run(net, records, fn, mag, conf, device):
    samples, confs = [], []
    for r in records:
        im = cv2.imread(str(CURATED / r["file"]), cv2.IMREAD_COLOR)
        if im is None:
            continue
        im = fn(im, mag) if fn is not p_identity else im
        h, w = im.shape[:2]
        res = net.predict(im, conf=conf, verbose=False, device=device)[0]
        pred, sc = [], []
        if len(res.boxes):
            xywhn = res.boxes.xywhn.cpu().numpy()
            cls = res.boxes.cls.cpu().numpy().astype(int)
            cf = res.boxes.conf.cpu().numpy()
            for (cx, cy, bw, bh), c, s in zip(xywhn, cls, cf):
                pred.append([int(c), float(cx), float(cy), float(bw), float(bh)])
                sc.append(float(s))
            confs.extend(sc)
        samples.append({"gt": r["boxes"], "pred": pred,
                        "scores": np.array(sc, np.float32), "wh": (w, h)})
    m = score_detections(samples)
    return m["mAP50"], (float(np.mean(confs)) if confs else 0.0), len(confs)


def sweep(checkpoint: str, conf: float = 0.05, device: str = "0") -> dict:
    from ultralytics import YOLO

    net = YOLO(checkpoint)
    records = _val_records()
    base_map, base_conf, base_n = _run(net, records, p_identity, 0, conf, device)
    print(f"[robust] baseline mAP50={base_map:.4f}  mean_conf={base_conf:.4f}  "
          f"dets={base_n}  ({len(records)} val defect images)")

    rows = []
    for name, (fn, mags) in PERTURBATIONS.items():
        for mag in mags:
            mp, mc, nd = _run(net, records, fn, mag, conf, device)
            d_map = (mp / base_map - 1) * 100 if base_map else 0.0
            d_conf = (mc / base_conf - 1) * 100 if base_conf else 0.0
            # The desired behaviour: when mAP falls, confidence falls TOO, and by
            # an amount that is not trivial next to the accuracy loss.
            #
            # This was `(d_map >= -5) or (d_conf < 0)` - direction only - which
            # passed noise@0.02 as well-behaved because confidence fell 13.5%
            # while mAP fell 96.1%. A flag that reports the single worst failure
            # mode in the sweep as fine is worse than no flag. Requiring the
            # confidence drop to reach a third of the accuracy drop separates
            # "noticed" from "technically moved in the right direction".
            rose = d_conf > 0
            ok = (d_map >= -5) or (d_conf <= d_map / 3.0)
            rows.append({"perturbation": name, "magnitude": mag,
                         "mAP50": round(mp, 4), "mean_conf": round(mc, 4),
                         "delta_mAP_pct": round(d_map, 1),
                         "delta_conf_pct": round(d_conf, 1),
                         "confidence_tracks_degradation": bool(ok),
                         "confidence_rose": bool(rose)})
            flag = ("" if ok else
                    "   <- CONFIDENCE ROSE while mAP fell" if rose else
                    "   <- SILENT FAILURE: mAP collapsed, confidence barely moved")
            print(f"  {name:16} m={mag:<6} mAP50={mp:.4f} ({d_map:+6.1f}%)  "
                  f"conf={mc:.4f} ({d_conf:+6.1f}%){flag}")

    bad = [r for r in rows if not r["confidence_tracks_degradation"]]
    rose = [r for r in bad if r["confidence_rose"]]
    silent = [r for r in bad if not r["confidence_rose"]]
    tag = lambda rs: [f"{r['perturbation']}@{r['magnitude']}" for r in rs]
    out = {"checkpoint": checkpoint, "conf_threshold": conf,
           "baseline": {"mAP50": round(base_map, 4), "mean_conf": round(base_conf, 4)},
           "rows": rows,
           "criterion": ("well-behaved if mAP loss <= 5% OR the confidence drop "
                         "reaches a third of the mAP drop; direction alone is "
                         "not enough"),
           "n_unnoticed_cases": len(bad),
           "n_confidence_rose": len(rose),
           "n_silent_failure": len(silent),
           "confidence_rose": tag(rose),
           "silent_failure": tag(silent),
           # kept so older readers of this file do not break
           "n_overconfident_cases": len(bad),
           "overconfident": tag(bad)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "robustness.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"[robust] {len(bad)}/{len(rows)} conditions where accuracy fell but "
          f"confidence did not follow")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--conf", type=float, default=0.05)
    a = ap.parse_args()
    sweep(a.checkpoint, conf=a.conf)
