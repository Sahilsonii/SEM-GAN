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

PLACEMENT AND MORPHOLOGY ARE LITERATURE-GROUNDED, NOT ARBITRARY
-----------------------------------------------------------------
Read (open access, top to bottom) before writing this version:

  1. Majewski, Ronsin & Harting, "Morphology Formation Pathways in
     Solution-Processed Perovskite Thin Films", arXiv:2509.04175 (2025).
     Pinholes are gaps left between adjacent crystal grains that fail to
     coalesce before the solvent finishes evaporating - their shape is
     therefore set by the packing geometry of the surrounding grains, and
     "pinhole-free and flat films" require the evaporation rate to dominate
     the crystal growth rate. Pinholes are a GRAIN-BOUNDARY phenomenon by
     construction, not a random pit on an otherwise flat grain.
  2. IntechOpen ch. 60167, "High-Quality Perovskite Film Preparations for
     Efficient Perovskite Solar Cells". SEM of an untreated one-step film:
     "small grain size and many pinholes BETWEEN THE GRAIN BOUNDARIES" and
     "bright portions at the grain boundaries... likely to be less
     conductive PbI2". Both defect classes are named as boundary phenomena
     in the same figure.
  3. ACS Omega 10.1021/acsomega.0c04483, PbI2 precursor engineering. At a
     1:1 PbI2:Pb(Ac)2 ratio (i.e. high excess) the residual PbI2 morphology
     "deteriorates and forms needle-shaped chunks" - excess severity shifts
     PbI2 from compact platelets toward elongated needles, it does not just
     scale a fixed shape up.
  4. ScienceDirect S0925838821023094, platelet-like PbI2 films from
     water-processed precursor. PbI2's native crystal habit (CdI2-type
     hexagonal layered structure) is platelet/plate-like, not circular -
     "circular" is a convenience approximation for pinhole pits, not for
     PbI2 grains.
  5. RSC Advances d4ra07942f (2025) and Nature Sci. Rep. s41598-017-04690-w /
     PMC5498566, thermal and moisture degradation of MAPbI3. Below ~200 C,
     MAPbI3 decomposes to volatile HI/CH3NH2 (or via hydrolysis with H2O)
     leaving solid, crystalline PbI2 behind - PbI2 is the persistent solid
     residue of BOTH the excess-precursor pathway and the degradation
     pathway, which is why it reads as bright/compositionally distinct
     against the surrounding perovskite in SEM regardless of which pathway
     produced it.
  6. ScienceDirect S0167577X20306844 / S0013468620300888, PbI2 grain-boundary
     passivation. Confirms placement again independently: PbI2 is
     deliberately engineered to sit AT grain boundaries (moderate amounts
     passivate them; excess amounts insulate them), not distributed
     uniformly across grain interiors.

Two changes follow directly and are new in this version:

  * `grain_boundary_affinity()` detects boundary-like ridges in the ACTUAL
    real background canvas (Canny + dilation, the same primitive already
    used by interpret/boundary_index.py) and `sample_params(..., bg_gray=...)`
    biases placement toward those ridges for BOTH classes, uniform-random
    placement remaining as the (now literature-contradicted) fallback when no
    canvas is available. This is derived from the real photographed grain
    texture already in the corpus, not a fabricated grain lattice.
  * Morphology selection is no longer a fixed distribution independent of
    kind and severity. PbI2 shifts from platelet/compact toward elongated as
    severity (a stand-in for excess-precursor ratio) rises, per finding 3.
    Pinhole morphology favours irregular/clustered (cusped, polygonal void
    shapes set by neighbouring-grain packing) over smooth "circular", per
    finding 1.
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
# "faceted" and "boundary_film" are PbI2-specific and literature-driven; see the
# module docstring and Science abp8873 figs. S1/S6.
MORPHOLOGIES = ("circular", "irregular", "elongated", "clustered",
                "faceted", "boundary_film")
PBI2_MORPHOLOGIES = ("faceted", "boundary_film", "elongated", "clustered")
PINHOLE_MORPHOLOGIES = ("circular", "irregular", "clustered", "elongated")


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

    theta = np.arctan2(ryy, rxx)

    if p.morphology == "faceted":
        # PbI2 has the CdI2-type hexagonal layered structure, so residual
        # crystals present as ANGULAR faceted platelets, not smooth blobs -
        # directly visible as the bright angular/polygonal patches in
        # Science abp8873 fig. S6B (aged film, 4 um scale bar) and as the
        # bright flake cluster EDS-confirmed at 26.9 wt% Pb / 58.2 wt% I in
        # its fig. S1. A polygonal radius profile reproduces that habit; a
        # Fourier-perturbed ellipse cannot make a straight edge or a corner.
        n_faces = int(rng.integers(5, 8))          # hexagonal-ish
        phase = float(rng.uniform(0, 2 * np.pi))
        # distance to a regular n-gon boundary, normalised: flat between
        # vertices, so the resulting mask has real facets and corners
        wedge = np.cos(np.pi / n_faces) / np.maximum(
            np.cos(((theta + phase) % (2 * np.pi / n_faces)) - np.pi / n_faces), 1e-6)
        rad = rad * wedge
        # slight per-facet irregularity so every platelet is not identical
        rad = rad * (1.0 + 0.06 * np.sin(3 * theta + phase))

    elif p.morphology in ("irregular", "clustered"):
        # Low-order Fourier perturbation of the boundary. Pinholes are
        # inter-granular voids whose outline is set by the packing of the
        # grains around them (arXiv:2509.04175), so they are cusped and
        # lobed - a perfect ellipse is the tell-tale of synthetic data.
        wob = np.zeros_like(theta)
        for k in (2, 3, 5):
            wob += rng.uniform(0.06, 0.18) * np.sin(k * theta + rng.uniform(0, 2 * np.pi))
        rad = rad / (1.0 + wob)

    # Faceted platelets have crisper edges than solvent-shaped voids: PbI2
    # crystal facets are atomically abrupt, whereas a pinhole rim is a
    # rounded grain shoulder.
    edge = 1.6 if p.morphology == "faceted" else 0.65

    # The emitted box is derived from the mask thresholded at MASK_THRESHOLD, so
    # the soft falloff silently shrinks it: (1-rad)**edge crosses 0.5 at
    # rad = 1 - 0.5**(1/edge), i.e. 0.656 of nominal for edge=0.65 and only
    # 0.352 for edge=1.6. Defect sizes are sampled from the real expert-box
    # distribution, so emitting 35-66% of the sampled size means the synthetic
    # size prior does not match the real one it was fitted to - and it skews the
    # stride-anchored tiny-defect bins. Rescaling rad puts the threshold
    # crossing back at the nominal radius for any edge exponent.
    rad_at_threshold = 1.0 - 0.5 ** (1.0 / edge)
    local = np.clip(1.0 - rad * rad_at_threshold, 0.0, 1.0) ** edge
    out[y0c:y1c, x0c:x1c] = local
    return out


def _boundary_film_mask(h: int, w: int, p: DefectParams, gb_mask: np.ndarray,
                        rng: np.random.Generator) -> np.ndarray:
    """Thin bright PbI2 film decorating a grain boundary.

    In Science abp8873 figs. S1 and S6 the residual PbI2 does not only appear
    as compact crystals - it also traces the grain boundaries as a THIN BRIGHT
    NETWORK, which is the morphology the passivation literature is describing
    when it says PbI2 "sits at grain boundaries" (refs 2, 6). Rendering that as
    a compact blob would be wrong; it is a ridge following the boundary path.

    gb_mask is the dilated Canny ridge map of the real background, so the film
    follows THIS image's actual grain boundaries rather than an invented path.
    """
    out = np.zeros((h, w), np.float32)
    if gb_mask is None or not gb_mask.any():
        return out

    cx, cy = int(p.cx * w), int(p.cy * h)
    reach = max(int(p.size_px * 1.5), 6)
    x0, y0 = max(0, cx - reach), max(0, cy - reach)
    x1, y1 = min(w, cx + reach), min(h, cy + reach)
    if x1 <= x0 or y1 <= y0:
        return out

    patch = (gb_mask[y0:y1, x0:x1] > 0).astype(np.float32)
    if not patch.any():
        return out

    # keep only the boundary segment connected to this defect's neighbourhood,
    # tapered by distance from the centre so the film is a local decoration
    # rather than lighting up every boundary in the crop
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(reach, 1)
    taper = np.clip(1.0 - dist, 0.0, 1.0) ** 0.8
    film = patch * taper
    # thin it slightly and soften, so it reads as a film rather than a band
    film = cv2.GaussianBlur(film, (0, 0), sigmaX=0.8)
    out[y0:y1, x0:x1] = film
    return out


def _apply(img: np.ndarray, soft: np.ndarray, p: DefectParams,
           rng: np.random.Generator) -> np.ndarray:
    """Composite one defect. Pinholes darken; PbI2 particles brighten."""
    m = soft[..., None]
    local_mean = float(img.mean())

    if p.kind == PINHOLE:
        # A pinhole is a VOID through the absorber, not a shallow dimple:
        # Agarwal & Nair (arXiv:1704.06605) model pinholes explicitly as
        # regions where perovskite is absent between ETL and HTL, and the
        # reverse-bias failure work attributes shorting to the perovskite
        # "failing to provide complete insulation" at pinhole sites. So a
        # severe pinhole exposes the substrate and goes near-black, rather
        # than merely dimming toward a fraction of the local mean.
        floor = local_mean * (1.0 - 0.92 * p.severity)
        target = np.full_like(img, floor)
        # Rim brightening: secondary-electron edge effect at a pit boundary.
        rim = np.clip(soft - cv2.erode(soft, np.ones((3, 3), np.uint8)), 0, 1)[..., None]
        img = img * (1 - m) + target * m + rim * 22.0 * p.severity
    else:
        ceil = min(255.0, local_mean + (255.0 - local_mean) * (0.35 + 0.6 * p.severity))
        target = np.full_like(img, ceil)
        if p.morphology == "boundary_film":
            # a thin decorating film is brighter than the perovskite but much
            # smoother than a bulk crystal - little internal texture
            grain = rng.normal(0, 2.0 * p.severity, img.shape).astype(np.float32)
        else:
            grain = rng.normal(0, 5.0 * p.severity, img.shape).astype(np.float32)
        img = img * (1 - m) + target * m + grain * m

    return np.clip(img, 0, 255)


def render(canvas: np.ndarray, params: list, seed: int = 0,
           mask_threshold: float = 0.5, region_bottom: int | None = None) -> dict:
    """Draw params onto canvas.

    Returns exact per-defect masks, YOLO boxes and class ids. The box is derived
    from the mask it was drawn with, so box/mask/class agreement is a property
    of the code rather than something to be validated after the fact.

    region_bottom is the first row of the FESEM metadata banner. Masks are
    clipped there, so no synthetic defect - and therefore no synthetic box - can
    extend into instrument text. Bounding the defect centre is not sufficient:
    a large blob centred just above the banner would still spill into it.
    """
    rng = np.random.default_rng(seed)
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    h, w = canvas.shape[:2]
    cut = h if region_bottom is None else max(1, min(int(region_bottom), h))

    img = canvas.astype(np.float32)
    union = np.zeros((h, w), np.float32)
    boxes, kept = [], []

    # grain-boundary ridges of THIS canvas, needed only if a boundary_film
    # defect is present - computed once, lazily, since Canny is not free
    gb_mask = None
    if any(getattr(p, "morphology", "") == "boundary_film" for p in params):
        gray_c = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        gb_mask = cv2.dilate(cv2.Canny(gray_c, 30, 80), np.ones((2, 2), np.uint8))

    for p in params:
        if p.morphology == "boundary_film":
            soft = _boundary_film_mask(h, w, p, gb_mask, rng)
            if not (soft >= mask_threshold).any():
                # no boundary ridge here - fall back to a compact platelet
                # rather than silently emitting nothing
                soft = _blob_mask(h, w, DefectParams(**{**asdict(p),
                                                        "morphology": "faceted"}), rng)
        else:
            soft = _blob_mask(h, w, p, rng)
        if cut < h:
            soft[cut:] = 0.0
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


def grain_boundary_affinity(gray: np.ndarray, canny_lo: int = 30, canny_hi: int = 80,
                            dilate: int = 5) -> np.ndarray:
    """Boundary-like ridge coordinates in a REAL background image.

    Both defect classes are reported as grain-boundary phenomena in the
    literature (see module docstring, refs 1-2 and 5-6) - this is what lets
    sample_params place defects there instead of uniformly at random. The
    detector is a plain Canny edge map, dilated to a placement corridor; it
    finds *ridges* in the photographed grain texture, not defects, so using it
    to seed placement does not leak anything about where real defects were
    annotated.

    Returns an (N, 2) array of (x, y) pixel coordinates; empty if none found.
    """
    edges = cv2.Canny(gray, canny_lo, canny_hi)
    if dilate > 1:
        edges = cv2.dilate(edges, np.ones((dilate, dilate), np.uint8))
    ys, xs = np.nonzero(edges)
    return np.stack([xs, ys], axis=1) if len(xs) else np.zeros((0, 2), np.int64)


def sample_params(priors: dict, n: int, rng: np.random.Generator,
                  render_px: int = 512, severity=None,
                  pbi2_fraction: float = 0.25, cy_max: float = 1.0,
                  bg_gray: np.ndarray | None = None,
                  boundary_bias: float = 0.7) -> list:
    """Draw n defect parameter sets from the expert-box priors.

    cy_max bounds vertical placement to the imaging region. Every image in this
    corpus carries a burned-in FESEM metadata banner over the bottom ~8.2% of
    its height (see data/sem_bar.py); rendering defects into it would teach the
    detector that pinholes occur inside instrument text.

    bg_gray, when given, is the real background canvas this batch of defects
    will be rendered onto. With probability boundary_bias each defect is
    centred on a detected grain-boundary ridge (jittered) rather than placed
    uniformly at random - literature refs 1-2 and 5-6 in the module docstring
    report both pinholes and PbI2 as concentrating at grain boundaries, not
    scattered across flat grain interiors. Falls back to uniform placement
    when bg_gray is None (e.g. the synthetic canvas in this module's __main__
    smoke test, or any caller that has not been updated to pass a canvas).
    """
    boundary_px = (grain_boundary_affinity(bg_gray) if bg_gray is not None
                  else np.zeros((0, 2), np.int64))

    # side_norm is sqrt(w_norm * h_norm), so converting it to pixels needs the
    # canvas GEOMETRIC MEAN sqrt(W*H), not one edge length. On a square canvas
    # they coincide, which is why this was invisible until backgrounds stopped
    # being force-resized to square: with a 512x352 canvas, render_px=512 would
    # oversize every defect by sqrt(512/352) = 1.21x.
    if bg_gray is not None:
        H_c, W_c = bg_gray.shape[:2]
        size_ref = math.sqrt(float(H_c) * float(W_c))
    else:
        size_ref = float(render_px)

    out = []
    for _ in range(n):
        kind = PBI2 if rng.random() < pbi2_fraction else PINHOLE
        pool = priors["side_pbi2"] if kind == PBI2 else priors["side_pinhole"]
        side_norm = float(rng.choice(pool))
        sev = float(rng.uniform(0.35, 0.95)) if severity is None else float(severity)

        # Morphology depends on kind AND severity, not a fixed distribution -
        # see module docstring refs 1, 3-4. PbI2's native habit is
        # platelet/plate-like (ref 4); it degrades toward needle-shaped
        # chunks as excess/severity rises (ref 3). Pinholes are inter-granular
        # voids shaped by neighbouring-grain packing, so irregular/clustered
        # (cusped) beats smooth "circular" (ref 1).
        if kind == PBI2:
            # Faceted platelets dominate (CdI2 hexagonal habit, Science
            # abp8873 fig. S6B); a boundary-decorating film is the second
            # major morphology (figs. S1/S6, and the passivation literature);
            # elongated needle-chunks grow in at high excess (ACS Omega
            # 10.1021/acsomega.0c04483 reports needles at 50:50 PbI2:Pb(Ac)2).
            p_needle = float(np.clip(0.05 + 0.40 * sev, 0.05, 0.45))
            rest = 1.0 - p_needle
            probs_map = {"faceted": rest * 0.50, "boundary_film": rest * 0.35,
                        "elongated": p_needle, "clustered": rest * 0.15}
            choices = PBI2_MORPHOLOGIES
        else:
            # Inter-granular voids: cusped/lobed outlines set by the packing of
            # surrounding grains (arXiv:2509.04175), rarely a clean disc.
            probs_map = {"circular": 0.15, "irregular": 0.45,
                        "clustered": 0.30, "elongated": 0.10}
            choices = PINHOLE_MORPHOLOGIES
        probs = np.array([probs_map[c] for c in choices], dtype=float)
        morph = str(rng.choice(choices, p=probs / probs.sum()))

        if morph == "elongated":
            aspect = float(rng.uniform(2.0, 4.5))
        else:
            aspect = float(rng.uniform(0.9, 1.3))

        margin = 0.04
        cy_hi = max(margin + 1e-3, min(1.0, cy_max) - margin)
        if len(boundary_px) and rng.random() < boundary_bias:
            H, W = bg_gray.shape[:2]
            px, py = boundary_px[rng.integers(0, len(boundary_px))]
            jitter = render_px * 0.015
            cx = float(np.clip((px + rng.normal(0, jitter)) / W, margin, 1 - margin))
            cy = float(np.clip((py + rng.normal(0, jitter)) / H, margin, cy_hi))
        else:
            cx = float(rng.uniform(margin, 1 - margin))
            cy = float(rng.uniform(margin, cy_hi))

        out.append(DefectParams(
            kind=kind,
            cx=cx,
            cy=cy,
            size_px=max(3.0, side_norm * size_ref),
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
