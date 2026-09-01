"""Grid of the Phi-sat-2 degradation ladder: one row per factor, one column per level."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, rasterio, sys
from rasterio.windows import Window

ROOT = "/shared/home/elucas/datasets/sen1floods11_ladder"
SCENE = sys.argv[1] if len(sys.argv) > 1 else "Nigeria_81933"
CROP  = int(sys.argv[2]) if len(sys.argv) > 2 else 256
OUT   = sys.argv[3] if len(sys.argv) > 3 else "/tmp/grid.png"

FACTORS = [("snr", "Sensor noise\n(SNR)", {1: "SNR 40-80", 2: "SNR 20-40", 3: "SNR 10-20", 4: "SNR 5-10"}),
           ("psf", "Optical blur\n(PSF)",  {1: r"$\sigma$=0.5 px", 2: r"$\sigma$=1.0 px", 3: r"$\sigma$=2.0 px", 4: r"$\sigma$=4.0 px"}),
           ("misalign", "Band\nmisalignment", {1: r"$\sigma$=1.25 px", 2: r"$\sigma$=2.5 px", 3: r"$\sigma$=5 px", 4: r"$\sigma$=10 px"})]
SEVERITY = {1: 0.125, 2: 0.25, 3: 0.5, 4: 1.0}
OFF = (1077 - CROP) // 2
WIN = Window(OFF, OFF, CROP, CROP)

def read_rgb(variant):
    p = f"{ROOT}/{variant}/v1.1/data/flood_events/HandLabeled/S2Hand/simulated_L1C_{SCENE}_S2Hand.tif"
    with rasterio.open(p) as s:
        a = s.read(window=WIN).astype(np.float32)
    return a[[2, 1, 0]].transpose(1, 2, 0)          # B04,B03,B02 -> RGB

clean = read_rgb("clean")
# One stretch for the whole figure, fixed on the clean scene, so that what changes
# between cells is the degradation and not the display normalisation.
lo = np.percentile(clean, 2, axis=(0, 1))
hi = np.percentile(clean, 98, axis=(0, 1))
show = lambda a: np.clip((a - lo) / (hi - lo), 0, 1)

fig, axes = plt.subplots(3, 5, figsize=(13.6, 9.0))
for r, (fac, label, params) in enumerate(FACTORS):
    for c in range(5):
        ax = axes[r, c]
        ax.imshow(show(clean if c == 0 else read_rgb(f"{fac}_l{c}")))
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("0.7"); sp.set_linewidth(0.7)
        # Parameter inside the panel: below it, the label would read as belonging to
        # the row underneath.
        txt = "no degradation" if c == 0 else params[c]
        ax.text(0.035, 0.035, txt, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=8.6, color="white",
                bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=2.2))
        if r == 0:
            ax.set_title("Undegraded" if c == 0 else f"L{c}  ·  severity {SEVERITY[c]:g}",
                         fontsize=10.5, pad=6)
    axes[r, 0].set_ylabel(label, fontsize=11.5, labelpad=12)

for r in range(3):   # grey the repeated reference column
    for sp in axes[r, 0].spines.values():
        sp.set_edgecolor("0.35"); sp.set_linewidth(1.2)

fig.suptitle(f"$\\Phi$-sat-2 degradation ladder — {SCENE.replace('_', ' ')}, "
             f"{CROP}$\\times${CROP} px at 4.75 m", fontsize=13, y=0.975)
fig.text(0.5, 0.022, "Each row varies one factor in isolation; every other stage of the simulator is held fixed. "
                     "Level 4 is the full simulator configuration and each step down halves the perturbation. "
                     "A single contrast stretch, fixed on the undegraded scene, is applied to every panel.",
         ha="center", fontsize=8.6, color="0.3")
fig.tight_layout(rect=(0, 0.05, 1, 0.955))
fig.subplots_adjust(wspace=0.045, hspace=0.11)
fig.savefig(OUT, dpi=160, bbox_inches="tight", facecolor="white")
print("wrote", OUT)
