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
