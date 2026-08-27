"""
Calibration (stage 8, hypothesis H8 in the plan).

Uncertainty must be validated, not just reported - a mean-uncertainty number on
its own proves nothing. This module answers: when the detector is confident, is
it usually right? And does confidence correlate with correctness at all?

Works on any list of (confidence, is_correct) pairs, so it is agnostic to
whether confidence came from softmax, an EDL Dirichlet u = K/S, or a detector's
box score - whatever the upstream caller supplies.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def expected_calibration_error(conf: np.ndarray, correct: np.ndarray,
                                n_bins: int = 15) -> dict:
    """ECE with equal-width confidence bins, plus the reliability diagram data."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)

    ece, bins = 0.0, []
    for b in range(n_bins):
        mask = bin_ids == b
        n = int(mask.sum())
        if n == 0:
            bins.append({"bin": b, "n": 0, "conf": None, "acc": None})
            continue
        avg_conf = float(conf[mask].mean())
        avg_acc = float(correct[mask].mean())
        ece += (n / len(conf)) * abs(avg_conf - avg_acc)
        bins.append({"bin": b, "n": n, "conf": avg_conf, "acc": avg_acc})

    return {"ece": round(float(ece), 4), "n_bins": n_bins, "reliability": bins}


def brier_score(conf: np.ndarray, correct: np.ndarray) -> float:
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    return round(float(np.mean((conf - correct) ** 2)), 4)


def risk_coverage(conf: np.ndarray, correct: np.ndarray) -> dict:
    """Selective risk at each coverage level, sorted by descending confidence.

    Coverage = fraction of predictions kept; risk = error rate among those kept.
    A well-calibrated, useful detector should show risk falling monotonically
    as coverage drops - the model's most confident predictions should be its
    most reliable ones. AURC (area under the risk-coverage curve) summarises
    this as one number; lower is better.
    """
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    order = np.argsort(-conf)
    err = 1.0 - correct[order]
    cum_err = np.cumsum(err) / (np.arange(len(err)) + 1)
    coverage = (np.arange(len(err)) + 1) / len(err)
    trapezoid = getattr(np, "trapezoid", np.trapz)
    aurc = float(trapezoid(cum_err, coverage))
    return {"aurc": round(aurc, 4),
            "coverage": coverage.tolist(), "risk": cum_err.tolist()}


def calibration_report(conf, correct, out_path: str | Path | None = None) -> dict:
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    report = {
        "n": len(conf),
        "mean_confidence": round(float(conf.mean()), 4),
        "accuracy": round(float(correct.mean()), 4),
        "ece": expected_calibration_error(conf, correct),
        "brier": brier_score(conf, correct),
        "risk_coverage": risk_coverage(conf, correct),
    }
    if out_path:
        Path(out_path).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"[calibration] n={report['n']}  acc={report['accuracy']:.3f}  "
          f"mean_conf={report['mean_confidence']:.3f}  ECE={report['ece']['ece']:.4f}  "
          f"Brier={report['brier']:.4f}  AURC={report['risk_coverage']['aurc']:.4f}")
    return report


if __name__ == "__main__":
    # smoke test: a synthetic well-calibrated vs overconfident model
    rng = np.random.default_rng(0)
    n = 2000
    true_p = rng.uniform(0, 1, n)
    correct = rng.uniform(0, 1, n) < true_p

    print("well-calibrated (conf == true_p):")
    calibration_report(true_p, correct)

    print("\noverconfident (conf pushed toward 1):")
    overconf = np.clip(true_p * 1.4, 0, 1)
    calibration_report(overconf, correct)


# --------------------------------------------------------------- driver ----

def from_checkpoint(checkpoint: str, split: str = "val", conf: float = 0.05,
                    iou_thr: float = 0.5, device: str = "0") -> dict:
    """Per-detection (confidence, correct) pairs from a trained detector.

    Correctness is IoU>=iou_thr against a same-class ground-truth box, greedily
    matched in descending confidence - the same convention as eval/detection.py,
    so a detection counted as a hit here is a hit there too.

    Refuses the test split: calibration is a diagnostic and must not consume
    the locked set.
    """
    import json
    from pathlib import Path

    import cv2
    from ultralytics import YOLO

    from eval.detection import iou_matrix, xywhn_to_xyxy

    if split == "test":
        raise RuntimeError("calibration is a diagnostic - do not read the locked "
                           "test split; use 'val'")

    root = Path(__file__).resolve().parents[1]
    recs = json.loads((root / "data" / "splits" / f"{split}.json")
                      .read_text(encoding="utf-8"))["records"]
    recs = [r for r in recs if r["n_boxes"] > 0]
    net = YOLO(checkpoint)

    confs, corrects = [], []
    for r in recs:
        im = cv2.imread(str(root / "data" / "curated" / "images" / r["file"]),
                        cv2.IMREAD_COLOR)
        if im is None:
            continue
        H, W = im.shape[:2]
        gt = r["boxes"]
        gt_xyxy = (np.stack([xywhn_to_xyxy(b, W, H) for b in gt])
                   if gt else np.zeros((0, 4), np.float32))
        res = net.predict(im, conf=conf, verbose=False, device=device)[0]
        if not len(res.boxes):
            continue
        pred, sc = [], []
        for (cx, cy, bw, bh), c, s in zip(res.boxes.xywhn.cpu().numpy(),
                                          res.boxes.cls.cpu().numpy().astype(int),
                                          res.boxes.conf.cpu().numpy()):
            pred.append([int(c), float(cx), float(cy), float(bw), float(bh)])
            sc.append(float(s))
        pr_xyxy = np.stack([xywhn_to_xyxy(b, W, H) for b in pred])
        ious = iou_matrix(pr_xyxy, gt_xyxy)
        taken = set()
        for pi in np.argsort(-np.array(sc)):
            best, bj = 0.0, -1
            for gj in range(len(gt)):
                if gj in taken or gt[gj][0] != pred[pi][0]:
                    continue
                if ious[pi, gj] > best:
                    best, bj = ious[pi, gj], gj
            hit = best >= iou_thr
            if hit:
                taken.add(bj)
            confs.append(sc[pi])
            corrects.append(1.0 if hit else 0.0)

    out_path = Path(__file__).resolve().parents[1] / "outputs" / f"calibration_{split}.json"
    rep = calibration_report(np.array(confs), np.array(corrects), out_path=out_path)
    rep["checkpoint"] = checkpoint
    rep["split"] = split
    out_path.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    return rep
