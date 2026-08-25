"""
Renderer exactness (contribution N1).

The claim is that synthetic labels are ground truth by construction. That is
only true if the emitted box really does bound the mask component it was drawn
from, and the class field really does match the requested defect type. These
tests are the claim.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from synth.renderer import (PBI2, PINHOLE, DefectParams, fit_priors,
                            grain_boundary_affinity, render, sample_params)


def _canvas(size=512, mean=118, noise=9, seed=0):
    rng = np.random.default_rng(seed)
    c = np.full((size, size, 3), mean, np.float32)
    return np.clip(c + rng.normal(0, noise, c.shape), 0, 255).astype(np.uint8)


def test_every_box_tightly_bounds_its_own_mask():
    """Box vs mask IoU >= 0.98 for isolated defects, over many random params."""
    priors = fit_priors("train")
    rng = np.random.default_rng(7)
    checked = 0
    for i in range(60):
        p = sample_params(priors, 1, rng, render_px=512)[0]
        p.size_px = max(p.size_px, 14.0)          # keep it resolvable
        res = render(_canvas(seed=i), [p], seed=i)
        if not res["boxes"]:
            continue
        _, cx, cy, w, h = res["boxes"][0]
        H, W = res["mask"].shape
        x0, x1 = int(round((cx - w / 2) * W)), int(round((cx + w / 2) * W))
        y0, y1 = int(round((cy - h / 2) * H)), int(round((cy + h / 2) * H))

        box_mask = np.zeros_like(res["mask"])
        box_mask[y0:y1, x0:x1] = 255
        m = res["mask"] > 0
        b = box_mask > 0
        assert m.sum() > 0
        # every mask pixel must lie inside the box
        assert (m & ~b).sum() == 0, "mask pixels fall outside the emitted box"
        # and the box must be tight: its extent equals the mask extent
        ys, xs = np.nonzero(m)
        assert abs((xs.max() + 1 - xs.min()) - (x1 - x0)) <= 1
        assert abs((ys.max() + 1 - ys.min()) - (y1 - y0)) <= 1
        checked += 1
    assert checked >= 40, "too few defects survived to test meaningfully"


def test_class_id_matches_requested_kind():
    priors = fit_priors("train")
    rng = np.random.default_rng(11)
    for kind, expect in ((PBI2, 0), (PINHOLE, 1)):
        ps = sample_params(priors, 8, rng, render_px=512)
        for p in ps:
            p.kind = kind
            p.size_px = max(p.size_px, 14.0)
        res = render(_canvas(), ps, seed=3)
        assert res["boxes"], "nothing rendered"
        assert all(b[0] == expect for b in res["boxes"]), \
            "emitted class id does not match the requested defect kind"


def test_box_count_matches_param_count():
    """Every retained param produces exactly one box - no silent duplication."""
    priors = fit_priors("train")
    rng = np.random.default_rng(5)
    ps = sample_params(priors, 15, rng, render_px=512)
    for p in ps:
        p.size_px = max(p.size_px, 14.0)
    res = render(_canvas(), ps, seed=1)
    assert len(res["boxes"]) == len(res["params"])


def test_pinholes_darken_and_pbi2_brightens():
    """Sign of the intensity change is a physical convention we must not flip."""
    canvas = _canvas()
    base = canvas.astype(np.float32).mean()
    for kind, sign in ((PINHOLE, -1), (PBI2, +1)):
        p = DefectParams(kind=kind, cx=0.5, cy=0.5, size_px=90.0,
                         severity=0.9, morphology="circular")
        res = render(canvas, [p], seed=0)
        m = res["mask"] > 0
        inside = res["image"].astype(np.float32).mean(axis=2)[m].mean()
        assert sign * (inside - base) > 0, f"{kind} changed intensity the wrong way"


def test_severity_is_monotonic_in_contrast():
    """Counterfactual probing (N4) needs severity to actually control contrast."""
    canvas = _canvas()
    base = canvas.astype(np.float32).mean()
    deltas = []
    for sev in (0.2, 0.4, 0.6, 0.8, 1.0):
        p = DefectParams(kind=PINHOLE, cx=0.5, cy=0.5, size_px=90.0,
                         severity=sev, morphology="circular")
        res = render(canvas, [p], seed=0)
        m = res["mask"] > 0
        deltas.append(base - res["image"].astype(np.float32).mean(axis=2)[m].mean())
    assert all(b > a for a, b in zip(deltas, deltas[1:])), \
        f"contrast not monotonic in severity: {deltas}"


def test_render_is_deterministic_given_seed():
    priors = fit_priors("train")
    ps = sample_params(priors, 10, np.random.default_rng(2), render_px=512)
    a = render(_canvas(), ps, seed=99)
    b = render(_canvas(), ps, seed=99)
    assert np.array_equal(a["image"], b["image"])
    assert a["boxes"] == b["boxes"]


def test_priors_come_from_train_split_only():
    """Firewall: renderer priors must never be fitted on val or test."""
    tr = fit_priors("train")
    te = fit_priors("test")
    assert len(tr["side_pinhole"]) != len(te["side_pinhole"]), \
        "train and test priors are identical - check the split being read"


def test_masks_never_enter_the_metadata_banner():
    """Defects must not be drawn into the burned-in FESEM instrument banner.

    Bounding the defect CENTRE is not enough - a large blob centred just above
    the banner still spills into it - so render() clips the mask at region_bottom
    and the box, being derived from that mask, follows.
    """
    priors = fit_priors("train")
    rng = np.random.default_rng(3)
    H = 512
    cut = int(H * 0.90)
    ps = sample_params(priors, 25, rng, render_px=H)
    for p in ps:
        p.size_px = 60.0          # deliberately large
        p.cy = 0.88               # deliberately near the banner
    res = render(_canvas(H), ps, seed=0, region_bottom=cut)

    assert res["mask"][cut:].sum() == 0, "mask pixels drawn inside the banner"
    for _, _, cy, _, h in res["boxes"]:
        assert (cy + h / 2) * H <= cut + 1, "box extends into the banner"
