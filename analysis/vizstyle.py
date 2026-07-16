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
        "lines.markersize": 5,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "legend.frameon": False,
    })


# Single-column and double-column widths for a typical LaTeX thesis (inches).
COL_W = 3.4
WIDE_W = 5.6


def save(fig, stem, outdir="results"):
    """Write BOTH vector PDF (for LaTeX) and PNG (for quick viewing)."""
    import os
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, f"{stem}.pdf"))
    fig.savefig(os.path.join(outdir, f"{stem}.png"))
    plt.close(fig)
