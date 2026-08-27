"""
Domain gap between real and synthetic, at four levels (contribution N2).

The question this exists to answer is NOT "how realistic do the synthetic
images look" - it is "does a smaller domain gap predict better downstream
detection". Every level below returns a single scalar distance from a synthetic
pool to the REAL TRAIN distribution, so the four can be regressed against the
mAP50 each pool produced.

  L1 pixel      intensity histogram distance (Wasserstein)
  L2 frequency  radially-averaged log power spectrum (L1)
  L3 morphology MicroDefectCV descriptor distributions (Wasserstein)
  L4 feature    DINOv2 embedding distance to the real centroid

FIREWALL (plan 7.2): L3 uses MicroDefectCV, which is never in any training
loss and never informed the renderer priors (those came from expert boxes).
That independence is the whole point - it is why L3 is evidence rather than a
measurement of the renderer's own settings reflected back.
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

from eval.spectrum import compute_radial_power_spectrum

CURATED = ROOT / "data" / "curated" / "images"
SPLITS = ROOT / "data" / "splits"
SYNTH = ROOT / "data" / "synthetic"
OUT = ROOT / "outputs"


# ---------------------------------------------------------------- helpers ----

def _wasserstein_1d(a: np.ndarray, b: np.ndarray) -> float:
    """1-D Wasserstein distance without a scipy dependency at import time."""
    a = np.sort(np.asarray(a, float))
    b = np.sort(np.asarray(b, float))
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    q = np.linspace(0, 1, 256)
    return float(np.abs(np.quantile(a, q) - np.quantile(b, q)).mean())


def _real_train_images(limit: int | None = None) -> list[Path]:
    recs = json.loads((SPLITS / "train.json").read_text(encoding="utf-8"))["records"]
    ps = [CURATED / r["file"] for r in recs]
    return ps[:limit] if limit else ps


def _pool_images(pool: str, limit: int | None = None) -> list[Path]:
    ps = sorted((SYNTH / pool / "images").glob("*.jpg"))
    return ps[:limit] if limit else ps


def _gray(p: Path, size=(512, 352)) -> np.ndarray | None:
    im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    return None if im is None else cv2.resize(im, size, interpolation=cv2.INTER_AREA)


# ------------------------------------------------------------------ levels ----

def l1_pixel(real: list[Path], synth: list[Path]) -> float:
    """Intensity-histogram distance. Cheapest and least discriminative level."""
    def pool(ps):
        v = []
        for p in ps:
            g = _gray(p)
            if g is not None:
                v.append(g.reshape(-1)[::37])       # subsample; full res is overkill
        return np.concatenate(v) if v else np.array([])
    return _wasserstein_1d(pool(real), pool(synth))


def l2_frequency(real: list[Path], synth: list[Path]) -> float:
    """L1 between mean radially-averaged log power spectra.

    This is the level the refiner's FFT discriminator branch is supposed to
    move, so it is also the natural read-out for H2.
    """
    def mean_rpsd(ps):
        acc, n = None, 0
        for p in ps:
            g = _gray(p)
            if g is None:
                continue
            r = compute_radial_power_spectrum(g.astype(np.float32))
            r = np.log(r + 1e-8)
            acc = r if acc is None else acc[:len(r)] + r[:len(acc)]
            n += 1
        return None if acc is None else acc / n
    a, b = mean_rpsd(real), mean_rpsd(synth)
    if a is None or b is None:
        return float("nan")
    k = min(len(a), len(b))
    return float(np.abs(a[:k] - b[:k]).mean())


def l3_morphology(real: list[Path], synth: list[Path]) -> float:
    """Wasserstein distance between MicroDefectCV descriptor distributions.

    The independent evaluator: it never saw the renderer's parameters.
    """
    try:
        from microdefectcv import detect_defects
    except Exception:
        return float("nan")

    def descriptors(ps):
        counts, ratios, areas = [], [], []
        for p in ps:
            im = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if im is None:
                continue
            try:
                r = detect_defects(im, mode="auto", min_area=20)
            except Exception:
                continue
            counts.append(r.get("defect_count", 0))
            ratios.append(r.get("defect_area_ratio", 0.0))
            cs = [cv2.contourArea(c) for c in r.get("contours", [])]
            areas.extend(cs)
        return np.array(counts, float), np.array(ratios, float), np.array(areas, float)

    rc, rr, ra = descriptors(real)
    sc, sr, sa = descriptors(synth)
    # normalise each descriptor by the real scale so they are comparable, then
    # average - one composite morphology distance
    parts = []
    for r_, s_ in ((rc, sc), (rr, sr), (ra, sa)):
        if len(r_) and len(s_):
            scale = np.median(np.abs(r_)) or 1.0
            parts.append(_wasserstein_1d(r_ / scale, s_ / scale))
    return float(np.mean(parts)) if parts else float("nan")


def l4_feature(real: list[Path], synth: list[Path], device: str | None = None) -> float:
    """Cosine distance between mean DINOv2 embeddings.

    Uses the existing MicroscopyFoundationBenchmark wrapper. Reported last and
    weighted least: an ImageNet-era encoder has never seen an electron
    micrograph, so it is the least trustworthy of the four levels here.
    """
    try:
        import torch

        from eval.foundation_features import MicroscopyFoundationBenchmark
    except Exception:
        return float("nan")

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    try:
        bench = MicroscopyFoundationBenchmark(device=dev)
    except Exception:
        return float("nan")

    def centroid(ps):
        feats = []
        with torch.no_grad():
            for p in ps:
                im = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if im is None:
                    continue
                im = cv2.resize(im, (224, 224))
                t = torch.from_numpy(im.astype(np.float32) / 255.0)
                t = t.permute(2, 0, 1)[None].to(dev)
                try:
                    feats.append(bench.extract_foundation_features(t).cpu())
                except Exception:
                    continue
        if not feats:
            return None
        f = torch.cat(feats, 0).mean(0)
        return torch.nn.functional.normalize(f, dim=-1)

    a, b = centroid(real), centroid(synth)
    if a is None or b is None:
        return float("nan")
    return float(1.0 - torch.dot(a, b))


# ----------------------------------------------------------------- driver ----

def measure(pool: str, n_real: int = 60, n_synth: int = 60,
            skip_l4: bool = False) -> dict:
    real = _real_train_images(n_real)
    synth = _pool_images(pool, n_synth)
    if not synth:
        raise RuntimeError(f"no images in pool '{pool}'")
    out = {
        "pool": pool, "n_real": len(real), "n_synth": len(synth),
        "L1_pixel": round(l1_pixel(real, synth), 5),
        "L2_frequency": round(l2_frequency(real, synth), 5),
        "L3_morphology": round(l3_morphology(real, synth), 5),
        "L4_feature": float("nan") if skip_l4 else round(l4_feature(real, synth), 5),
    }
    print(f"[gap] {pool:16} L1={out['L1_pixel']:.4f}  L2={out['L2_frequency']:.4f}  "
          f"L3={out['L3_morphology']:.4f}  L4={out['L4_feature']}")
    return out


def regress(pools_to_map: dict, n_real: int = 60, n_synth: int = 60,
            skip_l4: bool = False) -> dict:
    """Does a smaller gap predict better detection? Spearman per level.

    pools_to_map maps pool name -> observed mAP50 for a detector trained on it.
    A NEGATIVE correlation is the hypothesis (smaller gap -> higher mAP).
    """
    from scipy import stats

    gaps = {p: measure(p, n_real, n_synth, skip_l4) for p in pools_to_map}
    res = {"gaps": gaps, "mAP50": pools_to_map, "spearman": {}}
    if len(pools_to_map) >= 3:
        for lvl in ("L1_pixel", "L2_frequency", "L3_morphology", "L4_feature"):
            x = [gaps[p][lvl] for p in pools_to_map]
            y = [pools_to_map[p] for p in pools_to_map]
            if any(np.isnan(x)):
                res["spearman"][lvl] = None
                continue
            rho, pv = stats.spearmanr(x, y)
            res["spearman"][lvl] = {"rho": round(float(rho), 4),
                                    "p": round(float(pv), 4)}
    else:
        res["note"] = (f"only {len(pools_to_map)} pools - need >=3 for a "
                       f"correlation; gaps reported without regression")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "domain_gap.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", default="controlled,refined,refined_nofft")
    ap.add_argument("--n-real", type=int, default=60)
    ap.add_argument("--n-synth", type=int, default=60)
    ap.add_argument("--skip-l4", action="store_true",
                    help="skip the DINOv2 level (slow / needs timm weights)")
    a = ap.parse_args()
    for p in a.pools.split(","):
        measure(p.strip(), a.n_real, a.n_synth, a.skip_l4)
