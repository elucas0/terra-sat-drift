"""Shared palette and matplotlib style for the triplets dataset report figures.

Palette provenance: the three sensor colours are slots 1-3 of a pre-validated
categorical palette (blue / orange / aqua). That three-slot subset is the one
documented as clearing the all-pairs colour-vision-deficiency and
normal-vision separation gates in both light and dark modes, which is what
scatter and small-multiple forms require. Do not extend it to a fourth
generated hue -- fold extra categories into "other" or facet instead.

Land-cover figures deliberately use the official ESA WorldCover v200 legend
colours rather than the categorical palette: 11 classes exceed what any
categorical palette can separate, and the WorldCover legend is the domain
reference encoding readers already know (and matches the mask visualisations
elsewhere in the codebase). Class identity is always carried by an axis label
or direct label as well, never by colour alone.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# --- sensor identity (categorical slots 1-3) --------------------------------
SENSOR_COLOR = {
    "real": "#2a78d6",  # slot 1, blue   -- real PhiSat-2, the deployment domain
    "s2b": "#eb6834",   # slot 2, orange -- Sentinel-2B, the teacher's domain
    "sim": "#1baf7a",   # slot 3, aqua   -- PhiSat-2 simulated from Sentinel-2
}
SENSOR_LABEL = {"real": "PhiSat-2 (real)", "s2b": "Sentinel-2B", "sim": "PhiSat-2 (simulated)"}

# Full 8-slot categorical order, for line and bar charts comparing several runs.
# This fixed order is the colour-vision-safety mechanism, not cosmetic: it clears
# the adjacent-pair separation gates (lines, bars, stacks) in that sequence. It
# does NOT clear the all-pairs gates that scatter and small-multiple forms need
# -- for those use only the first three slots, as SENSOR_COLOR does. Never extend
# past slot 8 with a generated hue: fold extra series into "other" or facet.
CATEGORICAL = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]

# --- magnitude: one hue, light to dark -------------------------------------
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#eef4fb", "#2a78d6", "#10315a"])

# --- ink -------------------------------------------------------------------
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e3e3e0"

BAND_NAMES = ["Blue", "Green", "Red", "RE1", "RE2", "RE3", "NIR"]

# Official ESA WorldCover v200 legend, keyed by the product's class code.
WC_COLOR = {
    10: "#006400", 20: "#ffbb22", 30: "#ffff4c", 40: "#f096ff",
    50: "#fa0000", 60: "#b4b4b4", 70: "#f0f0f0", 80: "#0064c8",
    90: "#0096a0", 95: "#00cf75", 100: "#fae6a0",
}
WC_NAME = {
    10: "Tree cover", 20: "Shrubland", 30: "Grassland", 40: "Cropland",
    50: "Built-up", 60: "Bare / sparse veg.", 70: "Snow and ice",
    80: "Permanent water", 90: "Herbaceous wetland", 95: "Mangroves",
    100: "Moss and lichen",
}

# Köppen-Geiger main groups. NOTE: assumes the Beck et al. (2018) 1-30 legend
# ordering (1-3 = A, 4-7 = B, 8-16 = C, 17-28 = D, 29-30 = E, 0 = no data),
# which the manifest's int8 range is consistent with but which is not recorded
# anywhere in this repo. Verify against whatever produced `koppen_zone` before
# quoting these in the thesis.
KOPPEN_GROUPS = [
    ("A  Tropical", range(1, 4)),
    ("B  Arid", range(4, 8)),
    ("C  Temperate", range(8, 17)),
    ("D  Continental", range(17, 29)),
    ("E  Polar", range(29, 31)),
    ("no data", range(0, 1)),
]


def setup():
    """Applies a recessive, print-oriented style: thin marks, muted grid, no
    top/right spines, text in ink tokens rather than series colours."""
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "medium",
        "axes.labelsize": 9,
        "axes.labelcolor": INK_SOFT,
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "text.color": INK,
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 2.0,
        "lines.markersize": 4,
        "figure.titlesize": 11,
    })


def title(ax, text, subtitle=None):
    """Left-aligned title with an optional muted subtitle carrying the units or
    the sample size -- the two things a reader needs and charts usually omit.

    Both are drawn as axes-relative text rather than via `set_title`, so the
    subtitle sits below the title instead of colliding with it.
    """
    ax.text(0.0, 1.13 if subtitle else 1.03, text, transform=ax.transAxes,
            fontsize=10, color=INK, va="bottom", fontweight="medium")
    if subtitle:
        ax.text(0.0, 1.02, subtitle, transform=ax.transAxes,
                fontsize=8, color=INK_SOFT, va="bottom")


def save(fig, outdir: Path, name: str):
    """Writes PNG (for drafts) and PDF (vector, for LaTeX inclusion)."""
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {name}.png / .pdf")
