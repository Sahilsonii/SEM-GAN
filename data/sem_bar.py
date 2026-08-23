"""
FESEM metadata-bar detection.

Every image in this corpus carries a burned-in instrument banner along the
bottom (EHT, WD, Signal A, Mag, scale bar, lab name). It is not specimen data
and it is a real confound:

  * a detector can learn it as a context cue that is present in every image;
  * saliency maps will point at high-contrast glyphs rather than defects;
  * the renderer would otherwise paste synthetic defects on top of it, teaching
    the model that pinholes appear inside text.

Bar rows are strongly bimodal - near-black background with saturated white
glyphs - so the share of mid-tone pixels collapses to roughly zero, which is
what we detect. Note the very last row or two are often image content again,
so we look for the topmost row of the low-mid-tone block rather than scanning
up from the bottom edge.
"""
from __future__ import annotations

import cv2
import numpy as np

SEARCH_FRACTION = 0.25      # only look in the bottom quarter
MID_LO, MID_HI = 40, 215    # what counts as a mid-tone pixel
COLLAPSE = 0.25             # bar rows keep < 25% of the body's mid-tone share


def find_bar_top(gray: np.ndarray) -> int:
    """Return the first row index belonging to the metadata bar, or H if none.

    Everything from this row downwards should be treated as non-imaging.
    """
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    f = gray.astype(np.float32)

    mid = ((f > MID_LO) & (f < MID_HI)).mean(axis=1)
    body = float(np.median(mid[: int(h * 0.70)]))
    if body <= 1e-6:
        return h

    start = int(h * (1.0 - SEARCH_FRACTION))
    is_bar = mid[start:] < COLLAPSE * body
    if not is_bar.any():
        return h

    # topmost row of the LAST contiguous run of bar rows
    idx = np.flatnonzero(is_bar)
    runs, run = [], [idx[0]]
    for a, b in zip(idx, idx[1:]):
        if b == a + 1:
            run.append(b)
        else:
            runs.append(run)
            run = [b]
    runs.append(run)
    longest = max(runs, key=len)
    if len(longest) < 4:                 # too thin to be a banner
        return h
    return start + int(longest[0])


def imaging_region(gray: np.ndarray) -> tuple[int, int]:
    """(top, bottom) row bounds of the usable specimen area."""
    return 0, find_bar_top(gray)


def crop_bar(img: np.ndarray) -> np.ndarray:
    """Drop the metadata bar. Returns a view; shape changes."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return img[: find_bar_top(g)]


def bar_fraction(gray: np.ndarray) -> float:
    h = gray.shape[0]
    return (h - find_bar_top(gray)) / h


if __name__ == "__main__":
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    recs = json.loads((root / "data" / "curated" / "curated.json")
                      .read_text(encoding="utf-8"))["records"]
    fracs = []
    for r in recs:
        im = cv2.imread(str(root / "data" / "raw_snapshot" / "images" / r["file"]),
                        cv2.IMREAD_GRAYSCALE)
        if im is not None:
            fracs.append(bar_fraction(im))
    a = np.array(fracs)
    print("[sem_bar] n=%d  detected in %d (%.1f%%)  median=%.4f p90=%.4f max=%.4f"
          % (len(a), (a > 0.005).sum(), 100 * (a > 0.005).mean(),
             np.median(a), np.percentile(a, 90), a.max()))
