"""Thesis figure style: LaTeX-matching serif, validated ordinal palette, vector output.

Marker size is an ORDERED magnitude (36 -> 149 mm), not arbitrary identity, so the size
groups get a SEQUENTIAL single-hue ramp (light = small, dark = large) rather than four
arbitrary hues. That encodes the ordering, is colourblind-safe by construction (one hue,
monotone lightness), and survives greyscale printing -- which a thesis may well be.

Palette validated with the dataviz validator (ordinal mode): lightness monotone, adjacent
dL gaps >= 0.06, light-end contrast 2.44:1 vs surface, single hue (3 deg spread).
Distinct marker shapes are a deliberate SECONDARY encoding so identity survives pure B&W.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter

GHOST_COLOR = "#c9c8c4"

# Sequential blue ramp, light -> dark = small -> large marker.
SIZE_COLORS = {35.6: "#6da7ec", 44.4: "#2a78d6", 74.7: "#184f95", 149.4: "#0d366b"}
SIZE_MARKERS = {35.6: "o", 44.4: "s", 74.7: "^", 149.4: "D"}

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#d9d8d4"


def apply():
    """Thesis rcParams. Serif to match LaTeX body text; recessive grid/axes."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Nimbus Roman"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.edgecolor": TEXT_SECONDARY,
        "axes.labelcolor": TEXT_PRIMARY,
        "text.color": TEXT_PRIMARY,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.8,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.8,
        # Small: the curve carries the trend, the marker only pins where a bin sits.
        # Oversized markers hide the line and collide once several series overlap.
        "lines.markersize": 3,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "legend.frameon": False,
    })


# Single-column and double-column widths for a typical LaTeX thesis (inches).
COL_W = 3.4
WIDE_W = 5.6


def log_px_ticks(ax):
    """Explicit, plainly-labelled ticks on a log-scaled apparent-px axis.

    The default log formatter only shows decade labels (10^1, 10^2), which cannot
    answer "is that 21 px or 40 px?". Replace with an explicit tick set and disable
    minor ticks so no stray unlabelled/scientific ticks slip back in.
    """
    # No 200: the largest observed apparent size is ~170 px, and on a log axis a 200
    # tick sits close enough to 150 that the two labels collide into "150200".
    ticks = [10, 15, 21, 30, 50, 75, 100, 150]
    ax.set_xscale("log")
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.minorticks_off()
    ax.xaxis.set_minor_formatter(NullFormatter())


def linear_range_ticks(ax, step=0.5, max_val=5.0):
    """Dense, plainly-labelled ticks on a linear range (m) axis."""
    ticks = np.arange(0, max_val + step / 2, step)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])


def save(fig, stem, outdir="results"):
    """Write BOTH vector PDF (for LaTeX) and PNG (for quick viewing)."""
    import os
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, f"{stem}.pdf"))
    fig.savefig(os.path.join(outdir, f"{stem}.png"))
    plt.close(fig)
