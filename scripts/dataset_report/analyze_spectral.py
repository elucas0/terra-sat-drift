"""
Spectral characterisation of the triplets dataset: PhiSat-2 vs Sentinel-2.

Samples patches from the triplets HDF5 and quantifies the radiometric domain gap
between the co-registered sensors -- the gap the domain-adaptation work has to
close. Also emits the per-domain sqrt-space statistics needed by
``dataset_paired_triplets_lulc.DOMAIN_STATS``.

    python scripts/dataset_report/analyze_spectral.py [--n-patches 600] [--outdir DIR]

Reading is the bottleneck (the HDF5 holds 259k x 8 x 256 x 256 int16 per domain),
so patches are sampled, indices sorted for sequential access, and pixels
subsampled within each patch.
"""

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_style import (BAND_NAMES, GRID, INK, INK_SOFT, SENSOR_COLOR,
                        SENSOR_LABEL, SEQ, save, setup, title)

H5 = "/shared/projects/phisat2/data/processed/triplets_v1/phisat2_s2b_dataset_v1.h5"
MANIFEST = "/shared/projects/phisat2/data/processed/worldcover_all_clean_v1/worldcover_all_clean_manifest_v1.csv"
BAD_PRODUCT_IDS = [
    1294, 1296, 1342, 1385, 1397, 1420, 1460, 1497, 1647, 1854, 2223, 2246,
    2259, 2373, 2631, 2640, 2743, 2834, 2853, 3374, 3619, 4071, 4693, 4813,
    4942, 2352, 2882, 3322, 3914, 4702, 1333, 1466, 1615, 2460, 2729, 2763,
]

# Per the dataset v1 specification, `real` and `sim` store
#   [0] PAN, [1] Blue, [2] Green, [3] Red, [4] RE1, [5] RE2, [6] RE3, [7] NIR
# and `s2b` stores B02, B03, B04, B05, B06, B07, B08 -- i.e. the same
# Blue..NIR sequence. So slicing [1:8] and taking all of s2b puts both on the
# same band order, which `fig_band_correspondence` re-verifies from the data.
# PAN is dropped because the models do not consume it.
VIEWS = {"real": ("real/images", slice(1, 8)),
         "s2b": ("s2b/images", slice(0, 7)),
         "sim": ("sim/images", slice(1, 8))}


def sample_pixels(n_patches: int, px_per_patch: int, max_cloud: float, seed: int):
    """Samples co-registered pixels.

    Returns ``(raw, aligned, idx)``: ``raw`` keeps every stored band (8 for
    real/sim including PAN, 7 for s2b) so the band assignment can be verified
    from the data; ``aligned`` is the Blue..NIR stack the models consume.
    """
    df = pd.read_csv(MANIFEST, low_memory=False)
    df = df[~df["product_id"].isin(BAD_PRODUCT_IDS)]
    if max_cloud is not None:
        df = df[df["thick_cloud_pct"] + df["thin_cloud_pct"] <= max_cloud]
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(df["patch_index"].values,
                             size=min(n_patches, len(df)), replace=False))

    out = {v: [] for v in VIEWS}
    with h5py.File(H5, "r") as f:
        for i, pi in enumerate(idx):
            flat = rng.integers(0, 256 * 256, size=px_per_patch)
            for view, (key, _) in VIEWS.items():
                patch = f[key][int(pi)].astype(np.float32)             # (nb, 256, 256)
                out[view].append(patch.reshape(patch.shape[0], -1)[:, flat].T)
            if (i + 1) % 100 == 0:
                print(f"    {i + 1}/{len(idx)} patches read")
    raw = {v: np.concatenate(a, axis=0) for v, a in out.items()}
    aligned = {v: raw[v][:, VIEWS[v][1]] for v in VIEWS}
    return raw, aligned, idx


def fig_band_correspondence(raw, outdir, n_px=20000, seed=0):
    """Verifies from the data which Sentinel-2 band each stored PhiSat-2 band matches.

    Rank correlation over co-registered pixels needs no assumption about gain,
    offset or calibration -- only that the two views see the same ground. It
    independently confirms the documented layout: the diagonal should light up
    for stored bands 1-7, and stored band 0 should correlate broadly with the
    visible bands while falling away toward NIR, which is the panchromatic
    signature.

    The two panels together are also the cleanest statement of how much harder
    the real domain is than the simulated one -- compare their diagonals.
    """
    from scipy.stats import spearmanr

    rng = np.random.default_rng(seed)
    sel = rng.choice(len(raw["s2b"]), size=min(n_px, len(raw["s2b"])), replace=False)
    s2b = raw["s2b"][sel]
    stored = ["PAN", "Blue", "Green", "Red", "RE1", "RE2", "RE3", "NIR"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))
    for ax, view in zip(axes, ["sim", "real"]):
        a = raw[view][sel]
        m = np.array([[spearmanr(a[:, b], s2b[:, j]).statistic for j in range(7)]
                      for b in range(a.shape[1])])
        im = ax.imshow(m, cmap=SEQ, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(7)); ax.set_xticklabels(BAND_NAMES)
        ax.set_yticks(range(a.shape[1]))
        ax.set_yticklabels([f"[{b}] {stored[b]}" for b in range(a.shape[1])])
        ax.set_xlabel("Sentinel-2B band")
        ax.grid(False)
        for b in range(a.shape[1]):
            j = int(np.argmax(m[b]))
            ax.text(j, b, f"{m[b, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if m[b, j] > 0.55 else INK)
        title(ax, f"{SENSOR_LABEL[view]} vs Sentinel-2B", "Spearman rho; label = row maximum")
    cb = fig.colorbar(im, ax=axes, pad=0.02, aspect=30)
    cb.set_label("rank correlation", fontsize=8, color=INK_SOFT)
    cb.outline.set_visible(False)
    save(fig, outdir, "fig_band_correspondence")


# ---------------------------------------------------------------------------
def fig_band_boxplots(data, outdir):
    """The headline spectral comparison, per band and per sensor.

    A log y-axis is used because the two sensors' digital numbers differ by more
    than an order of magnitude; on a linear axis the PhiSat-2 boxes collapse to
    the baseline. The scale difference is itself one of the findings.
    """
    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    views = ["real", "s2b", "sim"]
    width = 0.24
    for k, view in enumerate(views):
        pos = np.arange(7) + (k - 1) * width
        bp = ax.boxplot([data[view][:, b] for b in range(7)], positions=pos,
                        widths=width * 0.85, showfliers=False, patch_artist=True,
                        medianprops=dict(color="white", linewidth=1.2),
                        whiskerprops=dict(color=SENSOR_COLOR[view], linewidth=0.9),
                        capprops=dict(color=SENSOR_COLOR[view], linewidth=0.9))
        for box in bp["boxes"]:
            box.set(facecolor=SENSOR_COLOR[view], edgecolor="white", linewidth=0.6)
        ax.plot([], [], color=SENSOR_COLOR[view], linewidth=6, label=SENSOR_LABEL[view])
    ax.set_yscale("log")
    ax.set_xticks(np.arange(7))
    ax.set_xticklabels(BAND_NAMES)
    ax.set_ylabel("raw digital number (log scale)")
    ax.set_xlim(-0.6, 6.6)
    # Legend below the axes: inside the plot it collides with the NIR boxes.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3)
    title(ax, "Per-band value distributions by sensor",
          f"boxes = IQR, whiskers = 1.5 IQR, outliers hidden; "
          f"{len(data['real']):,} sampled pixels")
    fig.tight_layout()
    save(fig, outdir, "fig_band_boxplots")


def fig_normalized_boxplots(data, outdir):
    """The same comparison after each sensor is standardised by its *own*
    statistics -- i.e. what the network actually receives.

    This is the figure that shows how much of the gap per-domain normalisation
    removes and how much survives it. Residual differences in spread and shape
    here are the part that representation-level domain adaptation must handle.
    """
    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    width = 0.24
    for k, view in enumerate(["real", "s2b", "sim"]):
        x = np.sqrt(np.maximum(data[view], 0))
        z = (x - x.mean(axis=0)) / x.std(axis=0)
        pos = np.arange(7) + (k - 1) * width
        bp = ax.boxplot([z[:, b] for b in range(7)], positions=pos,
                        widths=width * 0.85, showfliers=False, patch_artist=True,
                        medianprops=dict(color="white", linewidth=1.2),
                        whiskerprops=dict(color=SENSOR_COLOR[view], linewidth=0.9),
                        capprops=dict(color=SENSOR_COLOR[view], linewidth=0.9))
        for box in bp["boxes"]:
            box.set(facecolor=SENSOR_COLOR[view], edgecolor="white", linewidth=0.6)
        ax.plot([], [], color=SENSOR_COLOR[view], linewidth=6, label=SENSOR_LABEL[view])
    ax.axhline(0, color=INK_SOFT, linewidth=0.8, linestyle="--")
    ax.set_xticks(np.arange(7))
    ax.set_xticklabels(BAND_NAMES)
    ax.set_ylabel("z-score of sqrt(DN), per sensor")
    ax.set_xlim(-0.6, 6.6)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3)
    title(ax, "After per-sensor normalisation",
          "statistics measured on this sample, not the hardcoded DOMAIN_STATS; "
          "residual spread differences are what domain adaptation must close")
    fig.tight_layout()
    save(fig, outdir, "fig_band_boxplots_normalized")


def fig_signature(data, outdir):
    """Spectral signature *shape*, with the absolute scale divided out.

    Each sensor's per-band median vector is divided by the mean of that vector,
    so every curve averages to 1.0 and only the across-band shape is compared.
    (Dividing by the sensor's overall mean instead would confound shape with
    distribution skew -- PhiSat-2's DN histogram is strongly right-skewed, mean
    ~322 against median ~59 in Blue, so its curve would sit low for a reason
    that has nothing to do with spectral response.)

    Sentinel-2B and the simulated view coincide almost exactly, so the simulated
    curve is dashed on top of the solid Sentinel-2 curve -- otherwise one hides
    the other entirely and looks like a missing series.

    Reading the result: the real curve's shape differs sharply from Sentinel-2's
    (a Red maximum and a NIR minimum, against Sentinel-2's monotonic rise into
    the red edge). This is *not* a band-ordering error -- band correspondence is
    confirmed independently in `fig_band_correspondence`. It is that `real` is
    uncalibrated level-1 digital number, where per-band magnitude reflects
    detector gain and integration time rather than surface reflectance, whereas
    s2b and sim are scaled reflectance. The dataset specification additionally
    notes the two platforms' spectral response functions are not identical. Both
    effects are per-band and multiplicative, which is exactly what the pipeline's
    per-band standardisation is there to absorb.
    """
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    x = np.arange(7)
    styles = {"real": dict(linewidth=2.2), "s2b": dict(linewidth=3.0),
              "sim": dict(linewidth=1.8, linestyle=(0, (4, 2)))}
    for view in ["real", "s2b", "sim"]:
        med = np.median(data[view], axis=0)
        ax.plot(x, med / med.mean(), color=SENSOR_COLOR[view], marker="o",
                label=SENSOR_LABEL[view], **styles[view])
    # No IQR band here: PhiSat-2's distribution is wide enough that its band
    # spans 0.5-7.7 and flattens every median curve into the baseline. Spread is
    # the boxplot figure's job; this figure is only about across-band shape.
    ax.axhline(1.0, color=INK_SOFT, linewidth=0.7, linestyle=":")
    ax.set_xticks(x)
    ax.set_xticklabels(BAND_NAMES)
    ax.set_ylabel("median DN / mean of the median vector")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    title(ax, "Spectral signature shape", "absolute scale divided out; 1.0 = the sensor's own band average")
    fig.tight_layout()
    save(fig, outdir, "fig_spectral_signature")


def fig_sensor_scatter(data, outdir, max_cloud):
    """Per-band Sentinel-2 vs PhiSat-2 density for co-registered pixels.

    Because the pixels are the same ground at (near) the same time, this is an
    empirical estimate of the cross-sensor transfer function: the slope says how
    much of the gap is a plain linear gain, and the correlation says how much of
    the variance is shared at all.

    Both Pearson and Spearman are reported. The relation is visibly non-linear
    and heteroscedastic, so Pearson alone understates the association; Spearman
    is rank-based and does not assume linearity. Expect these numbers to move a
    lot with the cloud filter -- bright cloud is correlated across sensors and
    inflates agreement, while cloud present in only one view destroys it.
    """
    from scipy.stats import spearmanr

    fig, axes = plt.subplots(2, 4, figsize=(12, 5.8))
    rows = []
    for b, ax in enumerate(axes.ravel()[:7]):
        xr, ys = data["real"][:, b], data["s2b"][:, b]
        ax.hexbin(xr, ys, gridsize=45, bins="log", cmap=SEQ, mincnt=1, linewidths=0)
        slope, icpt = np.polyfit(xr, ys, 1)
        r = np.corrcoef(xr, ys)[0, 1]
        rho = spearmanr(xr, ys).statistic
        xs = np.linspace(xr.min(), xr.max(), 20)
        ax.plot(xs, slope * xs + icpt, color=SENSOR_COLOR["s2b"], linewidth=1.4)
        ax.set_title(f"{BAND_NAMES[b]}", loc="left", fontsize=9, color=INK)
        ax.text(0.04, 0.95, f"r {r:.2f}   rho {rho:.2f}\nslope {slope:.1f}",
                transform=ax.transAxes, fontsize=7.5, color=INK_SOFT, va="top")
        if b >= 3:
            ax.set_xlabel("PhiSat-2 DN")
        if b % 4 == 0:
            ax.set_ylabel("Sentinel-2B DN")
        rows.append({"band": BAND_NAMES[b], "slope": slope, "intercept": icpt,
                     "pearson_r": r, "spearman_rho": rho})
    axes.ravel()[7].axis("off")
    cloud = "no cloud filter" if max_cloud is None else f"cloud <= {max_cloud:g}%"
    fig.text(0.008, 1.0, "Co-registered pixel agreement, Sentinel-2B vs PhiSat-2",
             ha="left", va="top", fontsize=11, color=INK)
    fig.text(0.008, 0.965, f"colour = pixel density (log); orange = OLS fit; {cloud}",
             ha="left", va="top", fontsize=8, color=INK_SOFT)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    save(fig, outdir, "fig_sensor_scatter")
    return pd.DataFrame(rows)


def fig_patch_examples(idx, outdir, n=4, seed=0):
    """RGB previews of the same ground across the three views -- the qualitative
    counterpart to the statistics above."""
    rng = np.random.default_rng(seed)
    picks = rng.choice(idx, size=n, replace=False)
    fig, axes = plt.subplots(n, 3, figsize=(6.6, 2.25 * n), squeeze=False)
    with h5py.File(H5, "r") as f:
        for r, pi in enumerate(picks):
            for c, view in enumerate(["real", "sim", "s2b"]):
                key, bands = VIEWS[view]
                img = f[key][int(pi)][bands].astype(np.float32)[[2, 1, 0]]  # R,G,B
                lo, hi = np.percentile(img, [2, 98], axis=(1, 2), keepdims=True)
                rgb = np.clip((img - lo) / np.clip(hi - lo, 1e-6, None), 0, 1)
                axes[r][c].imshow(rgb.transpose(1, 2, 0))
                axes[r][c].set_xticks([]); axes[r][c].set_yticks([])
                for s in axes[r][c].spines.values():
                    s.set_visible(False)
                if r == 0:
                    axes[r][c].set_title(SENSOR_LABEL[view], fontsize=9, color=INK)
            axes[r][0].set_ylabel(f"patch {pi}", fontsize=7.5, color=INK_SOFT)
    fig.suptitle("Co-registered views, independently contrast-stretched", x=0.02, ha="left")
    fig.tight_layout()
    save(fig, outdir, "fig_patch_examples")


# ---------------------------------------------------------------------------
def stats_tables(data, outdir):
    """Per-band statistics, plus ready-to-paste sqrt-space DOMAIN_STATS entries.

    The sqrt-space mean/std/clip triple matches the convention in
    ``dataset_paired_triplets_lulc.DOMAIN_STATS`` (sqrt -> clip -> z-score), so
    the printed block can be pasted straight in. `clip` is taken as the 99.9th
    percentile of sqrt(DN).

    ``pct_zero`` and ``pct_at_4095`` quantify floor and ceiling clipping. The
    4095 column is only meaningful for the real PhiSat-2 view, which is 12-bit
    digital number; ``s2b`` and ``sim`` are scaled reflectance on a different
    range and their values simply sit above 4095 much of the time.
    """
    rows, domain_stats = [], {}
    for view, d in data.items():
        x = np.sqrt(np.maximum(d, 0))
        domain_stats[view] = {
            "mean": [round(float(v), 4) for v in x.mean(axis=0)],
            "std": [round(float(v), 4) for v in x.std(axis=0)],
            "clip": round(float(np.percentile(x, 99.9)), 3),
        }
        for b in range(7):
            v = d[:, b]
            rows.append({
                "sensor": view, "band": BAND_NAMES[b],
                "mean": v.mean(), "std": v.std(), "min": v.min(),
                "p2": np.percentile(v, 2), "median": np.median(v),
                "p98": np.percentile(v, 98), "max": v.max(),
                "sqrt_mean": x[:, b].mean(), "sqrt_std": x[:, b].std(),
                "pct_zero": 100.0 * (v <= 0).mean(),
                "pct_at_4095": 100.0 * (v >= 4095).mean(),
            })
    tbl = pd.DataFrame(rows).round(3)
    tbl.to_csv(outdir / "spectral_stats.csv", index=False)
    (outdir / "domain_stats_measured.json").write_text(json.dumps(domain_stats, indent=2))
    print("\n" + tbl.to_string(index=False))
    print("\n  wrote spectral_stats.csv / domain_stats_measured.json")
    return domain_stats


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-patches", type=int, default=600)
    p.add_argument("--pixels-per-patch", type=int, default=800)
    p.add_argument("--max-cloud", type=float, default=5.0,
                   help="Drop patches whose thick+thin cloud %% exceeds this. Cloud is "
                        "a confounder for a *sensor* comparison and has its own figure "
                        "in analyze_manifest.py; pass a large value to disable.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=Path,
                   default=Path("/shared/home/elucas/scratch/terra-sat-drift/outputs/dataset_report"))
    args = p.parse_args()

    setup()
    args.outdir.mkdir(parents=True, exist_ok=True)
    print(f"Sampling {args.n_patches} patches x {args.pixels_per_patch} px "
          f"(max_cloud={args.max_cloud})...")
    raw, data, idx = sample_pixels(args.n_patches, args.pixels_per_patch,
                                   args.max_cloud, args.seed)
    print(f"  sampled {len(data['real']):,} pixels per view\n")

    fig_band_correspondence(raw, args.outdir)
    fig_band_boxplots(data, args.outdir)
    fig_normalized_boxplots(data, args.outdir)
    fig_signature(data, args.outdir)
    fit = fig_sensor_scatter(data, args.outdir, args.max_cloud)
    fig_patch_examples(idx, args.outdir)
    fit.round(4).to_csv(args.outdir / "sensor_transfer_fit.csv", index=False)
    print("  wrote sensor_transfer_fit.csv")
    stats_tables(data, args.outdir)


if __name__ == "__main__":
    main()
