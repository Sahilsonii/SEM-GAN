"""
Shared figure style. Imported by every figures/*.py so the whole set looks like
one document rather than three.

seaborn is NOT installed and is not worth adding - matplotlib covers everything
here, and a new dependency on a working 4 GB CUDA env is a bad trade for nicer
defaults. Anything seaborn-shaped is done by hand below.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # headless: these run in a pipeline, not a notebook
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "outputs" / "figures"

# One palette for the whole report. REAL vs SYNTH is the comparison that recurs
# most, so those two are the most separable pair.
C = {
    "real":     "#1b3a5c",
    "synth":    "#c1584b",
    "renderer": "#e0a458",
    "fft_on":   "#2a7f62",
    "fft_off":  "#9b6a9e",
    "baseline": "#8a9299",
    "accent":   "#c1584b",
    "grid":     "#d8dbdd",
    "ink":      "#1a1a1a",
}
# Defect-scale bins are pre-registered (configs/tiny_defect_bins.yaml); keep the
# same order and colour everywhere they appear.
BINS = ["T1_sub_stride", "T2_tiny", "T3_small", "T4_medium_plus"]
BIN_LABELS = {"T1_sub_stride": "T1\n<8 px", "T2_tiny": "T2\n8-16 px",
              "T3_small": "T3\n16-32 px", "T4_medium_plus": "T4\n>=32 px"}
BIN_COLORS = ["#5c3a4e", "#8c4a56", "#c1584b", "#e0a458"]


def apply() -> None:
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "semibold",
        "axes.labelsize": 9,
        "axes.edgecolor": "#4a4a4a",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": C["grid"],
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.facecolor": "white",
    })


def despine(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def save(fig, name: str, subdir: str = "") -> Path:
    """Write under outputs/figures[/subdir]/<name>.png and return the path."""
    out = FIG / subdir if subdir else FIG
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{name}.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"  [fig] {p.relative_to(ROOT)}")
    return p


def note(ax, text: str) -> None:
    """Caveat printed INTO the figure. A chart that travels without its n or its
    units invites exactly the over-reading this project keeps having to undo."""
    ax.text(0.0, -0.22, text, transform=ax.transAxes, fontsize=7,
            color="#5a5a5a", va="top", ha="left", wrap=True)
