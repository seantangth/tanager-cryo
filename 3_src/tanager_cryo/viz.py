"""Figure style and builders.

Colour follows one rule per job. Categorical hues are assigned in fixed slot order and
never cycled; only the first three slots are used, which is the set that clears the
all-pairs colour-vision-deficiency and normal-vision separation floors. Magnitude is
always a single hue light-to-dark -- never a rainbow, which would invent structure the
data does not have and is unreadable to a dichromat. Text never wears a series colour.
"""

from __future__ import annotations

import matplotlib as mpl
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# --- palette -----------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8880"

# Categorical, fixed order. Three slots only: past three, the yellow/orange pair
# fails the all-pairs floors, so a fourth series folds to "Other" or facets instead.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")

# Single-hue sequential ramp for magnitude.
SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
            "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SEQ_ORANGE_ANCHORS = ["#fde4d8", "#f8b593", "#f08a5c", "#eb6834", "#c14f22", "#8f3915"]

# Diverging: warm/cool poles with a neutral grey midpoint.
DIV_ANCHORS = ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec", "#f2a6a6", "#d03b3b", "#7d1f1f"]

STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}

CMAP_SEQ = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
CMAP_SEQ_ALT = LinearSegmentedColormap.from_list("seq_orange", SEQ_ORANGE_ANCHORS)
CMAP_DIV = LinearSegmentedColormap.from_list("div_blue_red", DIV_ANCHORS)


def use_style() -> None:
    """Apply the figure style: recessive axes and grid, ink-coloured text."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": INK_MUTED,
            "axes.linewidth": 0.8,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "medium",
            "axes.titlelocation": "left",
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "grid.color": "#e6e5e1",
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "font.size": 9,
            "font.family": "sans-serif",
            "lines.linewidth": 2.0,
            "lines.markersize": 4.5,
            "figure.dpi": 130,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
        }
    )


def robust_limits(a: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> tuple[float, float]:
    """Percentile limits, so a handful of outliers cannot flatten a map's contrast."""
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return 0.0, 1.0
    return float(np.percentile(finite, lo)), float(np.percentile(finite, hi))


def annotate_units(ax, text: str) -> None:
    """Put the unit under the title in muted ink, rather than crowding the title itself.

    The title is pushed up to make room, so the two never collide regardless of how long
    either string is.
    """
    ax.set_title(ax.get_title(), pad=22)
    ax.text(
        0.0, 1.012, text, transform=ax.transAxes,
        color=INK_MUTED, fontsize=8, va="bottom", ha="left",
    )
