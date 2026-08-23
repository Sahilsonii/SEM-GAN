"""
Parametric defect renderer (contribution N1).

The point of this module is that the label is not inferred. We decide where a
defect goes and how big it is, then we draw it there; the mask IS the drawing
and the box IS the extent of that mask. Nothing downstream has to guess.

This replaces the previous generator, which took only a binary rectangle mask,
had no class input at all, and wrote np.random.choice([0,1,2]) into the label
file - making the class statistically independent of the pixels.

FIREWALL (plan 7.2): size/density priors are fitted to EXPERT BOXES on the
train split only. MicroDefectCV is never consulted here - it has to stay an
independent evaluator of what this module produces.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data" / "splits"

PINHOLE, PBI2 = "pinhole", "pbi2"
MORPHOLOGIES = ("circular", "irregular", "elongated", "clustered")


@dataclass
class DefectParams:
    """One defect. Every field is a SIMULATION CONTROL, not a physical quantity.

    severity in particular is a normalised [0,1] drawing parameter. It is not a
    depth in nm and must never be reported as one - the FESEM pixel-size headers
    did not survive JPEG re-encoding, so no calibration exists for this corpus.
    """
    kind: str = PINHOLE
    cx: float = 0.5                # normalised centre
    cy: float = 0.5
    size_px: float = 12.0          # equivalent square side at render resolution
    severity: float = 0.6          # [0,1] contrast depth of the defect
    morphology: str = "circular"
    aspect: float = 1.0            # >1 elongates along angle
    angle: float = 0.0             # radians


def fit_priors(split: str = "train") -> dict:
    """Empirical size/count distribution from human annotations on a split."""
    recs = json.loads((SPLITS / (split + ".json")).read_text(encoding="utf-8"))
    sides = {0: [], 1: []}
    per_image = []
    for r in recs["records"]:
        if not r["boxes"]:
            continue
        per_image.append(len(r["boxes"]))
        for c, _, _, w, h in r["boxes"]:
            sides[c].append(math.sqrt(w * h))          # normalised side
    return {
        "side_pbi2": np.array(sides[0]) if sides[0] else np.array([0.01]),
        "side_pinhole": np.array(sides[1]) if sides[1] else np.array([0.01]),
        "counts": np.array(per_image) if per_image else np.array([10]),
    }


def _blob_mask(h: int, w: int, p: DefectParams, rng: np.random.Generator) -> np.ndarray:
    """Sub-pixel-accurate soft mask for one defect, in [0,1]."""
    cx, cy = p.cx * w, p.cy * h
    r = max(p.size_px / 2.0, 1.0)
    ry = r / math.sqrt(p.aspect)
    rx = r * math.sqrt(p.aspect)

    pad = int(max(rx, ry) * 3) + 4
    x0c, y0c = max(0, int(cx) - pad), max(0, int(cy) - pad)
    x1c, y1c = min(w, int(cx) + pad), min(h, int(cy) + pad)
    out = np.zeros((h, w), np.float32)
    if x1c <= x0c or y1c <= y0c:
        return out

    yy, xx = np.mgrid[y0c:y1c, x0c:x1c].astype(np.float32)
    dx, dy = xx - cx, yy - cy
    ca, sa = math.cos(-p.angle), math.sin(-p.angle)
    rxx, ryy = dx * ca - dy * sa, dx * sa + dy * ca
    rad = np.sqrt((rxx / rx) ** 2 + (ryy / ry) ** 2)

    if p.morphology in ("irregular", "clustered"):
        # Low-order Fourier perturbation of the boundary. Real pinholes are not
        # discs, and a perfect ellipse is the tell-tale of synthetic data.
        theta = np.arctan2(ryy, rxx)
        wob = np.zeros_like(theta)
        for k in (2, 3, 5):
            wob += rng.uniform(0.06, 0.18) * np.sin(k * theta + rng.uniform(0, 2 * np.pi))
        rad = rad / (1.0 + wob)

    local = np.clip(1.0 - rad, 0.0, 1.0) ** 0.65      # soft edge
    out[y0c:y1c, x0c:x1c] = local
    return out


def _apply(img: np.ndarray, soft: np.ndarray, p: DefectParams,
           rng: np.random.Generator) -> np.ndarray:
    """Composite one defect. Pinholes darken; PbI2 particles brighten."""
    m = soft[..., None]
    local_mean = float(img.mean())

    if p.kind == PINHOLE:
        floor = local_mean * (1.0 - 0.85 * p.severity)
        target = np.full_like(img, floor)
        # Rim brightening: secondary-electron edge effect at a pit boundary.
        rim = np.clip(soft - cv2.erode(soft, np.ones((3, 3), np.uint8)), 0, 1)[..., None]
        img = img * (1 - m) + target * m + rim * 22.0 * p.severity
    else:
        ceil = min(255.0, local_mean + (255.0 - local_mean) * (0.35 + 0.6 * p.severity))
        target = np.full_like(img, ceil)
        grain = rng.normal(0, 5.0 * p.severity, img.shape).astype(np.float32)
        img = img * (1 - m) + target * m + grain * m

    return np.clip(img, 0, 255)


def render(canvas: np.ndarray, params: list, seed: int = 0,
           mask_threshold: float = 0.5) -> dict:
    """Draw params onto canvas.

    Returns exact per-defect masks, YOLO boxes and class ids. The box is derived
    from the mask it was drawn with, so box/mask/class agreement is a property
    of the code rather than something to be validated after the fact.
    """
    rng = np.random.default_rng(seed)
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    h, w = canvas.shape[:2]

    img = canvas.astype(np.float32)
    union = np.zeros((h, w), np.float32)
    boxes, kept = [], []

    for p in params:
        soft = _blob_mask(h, w, p, rng)
        binary = soft >= mask_threshold
        if int(binary.sum()) < 4:                  # sub-resolution, skip
            continue
        img = _apply(img, soft, p, rng)
        union = np.maximum(union, soft)

        ys, xs = np.nonzero(binary)
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        cls = 0 if p.kind == PBI2 else 1
        boxes.append([cls, ((x0 + x1) / 2) / w, ((y0 + y1) / 2) / h,
                      (x1 - x0) / w, (y1 - y0) / h])
        kept.append(p)

    return {
        "image": img.astype(np.uint8),
        "mask": (union >= mask_threshold).astype(np.uint8) * 255,
        "soft_mask": union,
        "boxes": boxes,
        "params": [asdict(p) for p in kept],
    }


def sample_params(priors: dict, n: int, rng: np.random.Generator,
                  render_px: int = 512, severity=None,
                  pbi2_fraction: float = 0.25) -> list:
    """Draw n defect parameter sets from the expert-box priors."""
    out = []
    for _ in range(n):
        kind = PBI2 if rng.random() < pbi2_fraction else PINHOLE
        pool = priors["side_pbi2"] if kind == PBI2 else priors["side_pinhole"]
        side_norm = float(rng.choice(pool))
        morph = str(rng.choice(MORPHOLOGIES, p=[0.35, 0.4, 0.15, 0.1]))
        if morph == "elongated":
            aspect = float(rng.uniform(2.0, 4.5))
        else:
            aspect = float(rng.uniform(0.9, 1.3))
        sev = float(rng.uniform(0.35, 0.95)) if severity is None else float(severity)
        margin = 0.04
        out.append(DefectParams(
            kind=kind,
            cx=float(rng.uniform(margin, 1 - margin)),
            cy=float(rng.uniform(margin, 1 - margin)),
            size_px=max(3.0, side_norm * render_px),
            severity=sev,
            morphology=morph,
            aspect=aspect,
            angle=float(rng.uniform(0, math.pi)),
        ))
    return out


if __name__ == "__main__":
    pri = fit_priors("train")
    rng = np.random.default_rng(0)
    canvas = np.full((512, 512, 3), 118, np.uint8)
    canvas = np.clip(canvas + rng.normal(0, 9, canvas.shape), 0, 255).astype(np.uint8)
    res = render(canvas, sample_params(pri, 12, rng), seed=0)
    print("[renderer] priors: pinhole n=%d median_side=%.5f | pbi2 n=%d median_side=%.5f"
          % (len(pri["side_pinhole"]), float(np.median(pri["side_pinhole"])),
             len(pri["side_pbi2"]), float(np.median(pri["side_pbi2"]))))
    print("[renderer] rendered %d defects, mask covers %.3f%% of image"
          % (len(res["boxes"]), res["mask"].mean() / 255 * 100))
