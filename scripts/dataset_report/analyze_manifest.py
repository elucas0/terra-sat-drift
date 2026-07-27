"""
Dataset characterisation from the triplets manifest (CSV only -- runs in seconds).

Produces the figures and headline numbers needed to describe the PhiSat-2 /
Sentinel-2 triplets dataset in a thesis: acquisition time lag, cloud cover,
land-cover class prior, geographic and climate coverage, and split composition.

    python scripts/dataset_report/analyze_manifest.py [--outdir DIR]
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_style import (GRID, INK, INK_SOFT, KOPPEN_GROUPS, SENSOR_COLOR, SEQ,
                        WC_COLOR, WC_NAME, save, setup, title)

MANIFEST = "/shared/projects/phisat2/data/processed/worldcover_all_clean_v1/worldcover_all_clean_manifest_v1.csv"

# Excluded upstream for radiometric/geometric quality; kept identical to the
# dataset classes so every number here describes the data actually trained on.
BAD_PRODUCT_IDS = [
    1294, 1296, 1342, 1385, 1397, 1420, 1460, 1497, 1647, 1854, 2223, 2246,
    2259, 2373, 2631, 2640, 2743, 2834, 2853, 3374, 3619, 4071, 4693, 4813,
    4942, 2352, 2882, 3322, 3914, 4702, 1333, 1466, 1615, 2460, 2729, 2763,
]
WC_CODES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]


def load():
    df = pd.read_csv(MANIFEST, low_memory=False)
    n_raw = len(df)
    df = df[~df["product_id"].isin(BAD_PRODUCT_IDS)].reset_index(drop=True)
    df["date_phi"] = pd.to_datetime(df["date_phi"], errors="coerce")
    df["date_s2b"] = pd.to_datetime(df["date_s2b"], errors="coerce")
    return df, n_raw


def splits(df):
    """Reproduces the seed-42 80/10/10 patch-level split used by the datasets."""
    np.random.seed(42)
    idx = np.arange(len(df))
    np.random.shuffle(idx)
    a, b = int(0.8 * len(idx)), int(0.9 * len(idx))
    return {"train": idx[:a], "val": idx[a:b], "test": idx[b:]}


# ---------------------------------------------------------------------------
def fig_acquisition(df, outdir):
    """Time lag between the paired acquisitions, and when they were acquired.

    The lag matters because the two views are treated as the *same* scene: any
    real change within the gap is a labelling error the model cannot win against.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.2))

    d = df["patch_delta_days"].dropna()
    # Half-day bins: the lag is date-quantised (Sentinel-2 dates are daily), so
    # arbitrary bin counts straddle day boundaries and produce a comb artefact.
    ax1.hist(d, bins=np.arange(0, d.max() + 0.5, 0.5),
             color=SENSOR_COLOR["real"], edgecolor="white", linewidth=0.3)
    med = d.median()
    ax1.axvline(med, color=INK, linewidth=1.2, linestyle="--")
    ax1.annotate(f"median {med:.1f} d", xy=(med, ax1.get_ylim()[1] * 0.94),
                 xytext=(6, 0), textcoords="offset points", fontsize=8, color=INK)
    ax1.set_xlabel("|PhiSat-2 date - Sentinel-2 date|  (days)")
    ax1.set_ylabel("patches")
    title(ax1, "Acquisition time lag", f"n = {len(d):,} patches, max {d.max():.1f} d")

    # A timeline, not a month-of-year aggregate: the collection spans an uneven
    # number of years per calendar month, so aggregating months would imply a
    # seasonal pattern that is really just uneven coverage.
    ts = df["date_phi"].dt.to_period("M").value_counts().sort_index()
    x = ts.index.to_timestamp()
    ax2.bar(x, ts.values, width=22, color=SENSOR_COLOR["real"],
            edgecolor="white", linewidth=0.3)
    ax2.set_xlabel("month of PhiSat-2 acquisition")
    ax2.set_ylabel("patches")
    for lab in ax2.get_xticklabels():
        lab.set_rotation(30)
        lab.set_ha("right")
    title(ax2, "Temporal coverage",
          f"{df['date_phi'].min():%b %Y} - {df['date_phi'].max():%b %Y}, "
          f"{len(ts)} months")

    fig.tight_layout()
    save(fig, outdir, "fig_acquisition")


def fig_clouds(df, outdir):
    """Cumulative cloud cover. An ECDF is used rather than a histogram because
    both variables are strongly zero-inflated with a long tail, which a
    histogram renders as one spike."""
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for col, color, label in [("thick_cloud_pct", SENSOR_COLOR["real"], "Thick cloud"),
                              ("thin_cloud_pct", SENSOR_COLOR["s2b"], "Thin cloud")]:
        v = np.sort(df[col].dropna().values)
        ax.plot(v, np.arange(1, len(v) + 1) / len(v) * 100, color=color, label=label)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("cloud cover in patch (%)")
    ax.set_ylabel("patches at or below (%)")
    ax.legend(loc="lower right")
    frac = (df[["thick_cloud_pct", "thin_cloud_pct"]].sum(axis=1) < 5).mean() * 100
    title(ax, "Cloud contamination", f"{frac:.0f}% of patches below 5% total cloud")
    fig.tight_layout()
    save(fig, outdir, "fig_clouds")


def fig_landcover(df, outdir):
    """Class prior two ways: by pixel (what the loss sees) and by dominant class
    (how the patches were sampled). The pixel prior is the one that governs class
    imbalance in segmentation."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.6))

    frac = np.array([df[f"wc_frac_{c}"].mean() * 100 for c in WC_CODES])
    order = np.argsort(frac)
    y = np.arange(len(order))
    ax1.barh(y, frac[order], color=[WC_COLOR[WC_CODES[i]] for i in order],
             edgecolor=GRID, linewidth=0.6)
    ax1.set_yticks(y)
    ax1.set_yticklabels([WC_NAME[WC_CODES[i]] for i in order])
    for yi, v in zip(y, frac[order]):
        ax1.text(v + 0.4, yi, f"{v:.1f}", va="center", fontsize=7.5, color=INK_SOFT)
    ax1.set_xlabel("share of all labelled pixels (%)")
    ax1.grid(axis="y", visible=False)
    title(ax1, "Class prior by pixel", f"{df['wc_nodata_frac'].mean()*100:.2f}% unlabelled")

    cnt = df["wc_dominant_class"].value_counts()
    cnt = cnt[[c for c in WC_CODES if c in cnt.index]]
    o = np.argsort(cnt.values)
    y = np.arange(len(o))
    ax2.barh(y, cnt.values[o], color=[WC_COLOR[cnt.index[i]] for i in o],
             edgecolor=GRID, linewidth=0.6)
    ax2.set_yticks(y)
    ax2.set_yticklabels([WC_NAME[cnt.index[i]] for i in o])
    ax2.set_xlabel("patches where the class dominates")
    ax2.grid(axis="y", visible=False)
    title(ax2, "Dominant class per patch",
          f"median dominance {df['wc_dominant_frac'].median()*100:.0f}% of the patch")

    fig.tight_layout()
    save(fig, outdir, "fig_landcover")


def fig_geography(df, outdir):
    """Where the patches are, and which climates they cover -- the two axes a
    reviewer asks about when a model claims to generalise."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.6),
                                   gridspec_kw={"width_ratios": [1.7, 1]})

    hb = ax1.hexbin(df["center_lon"], df["center_lat"], gridsize=(70, 35),
                    bins="log", cmap=SEQ, mincnt=1, linewidths=0)
    ax1.set_xlim(-180, 180)
    ax1.set_ylim(-90, 90)
    ax1.set_xticks(range(-180, 181, 60))
    ax1.set_yticks(range(-90, 91, 30))
    ax1.set_xlabel("longitude")
    ax1.set_ylabel("latitude")
    cb = fig.colorbar(hb, ax=ax1, pad=0.02, aspect=30)
    cb.set_label("patches per cell", fontsize=8, color=INK_SOFT)
    cb.outline.set_visible(False)
    title(ax1, "Geographic coverage",
          f"{df['product_id'].nunique():,} PhiSat-2 products, "
          f"lat {df['center_lat'].min():.0f} to {df['center_lat'].max():.0f}")

    labels, counts = [], []
    for name, rng in KOPPEN_GROUPS:
        labels.append(name)
        counts.append(int(df["koppen_zone"].isin(list(rng)).sum()))
    y = np.arange(len(labels))[::-1]
    ax2.barh(y, counts, color=SENSOR_COLOR["real"], edgecolor="white", linewidth=0.3)
    ax2.set_yticks(y)
    ax2.set_yticklabels(labels)
    for yi, v in zip(y, counts):
        ax2.text(v + max(counts) * 0.01, yi, f"{v/len(df)*100:.0f}%",
                 va="center", fontsize=7.5, color=INK_SOFT)
    ax2.set_xlabel("patches")
    ax2.grid(axis="y", visible=False)
    title(ax2, "Köppen-Geiger climate group", "legend assumed: Beck et al. 2018")

    fig.tight_layout()
    save(fig, outdir, "fig_geography")


def fig_splits(df, outdir):
    """Split composition, and the scene-overlap problem in the current split.

    The right panel is the caveat: the split is drawn over *patches*, but patches
    come in large per-product blocks from one acquisition, so the same scene lands
    in train and test. Class balance (left) looks fine; independence does not.
    """
    sp = splits(df)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.4))

    width = 0.27
    x = np.arange(len(WC_CODES))
    for i, (name, idx) in enumerate(sp.items()):
        sub = df.iloc[idx]
        v = [sub[f"wc_frac_{c}"].mean() * 100 for c in WC_CODES]
        ax1.bar(x + (i - 1) * width, v, width * 0.92,
                color=list(SENSOR_COLOR.values())[i], label=f"{name} (n={len(idx):,})",
                edgecolor="white", linewidth=0.4)
    ax1.set_xticks(x)
    ax1.set_xticklabels([WC_NAME[c].split()[0] for c in WC_CODES], rotation=45, ha="right")
    ax1.set_ylabel("share of labelled pixels (%)")
    ax1.legend(loc="upper right")
    title(ax1, "Class prior is stable across splits", "seed-42 patch-level 80/10/10")

    per_prod = df.groupby("product_id").size()
    ax2.hist(per_prod, bins=40, color=SENSOR_COLOR["s2b"], edgecolor="white", linewidth=0.3)
    ax2.set_xlabel("patches per PhiSat-2 product")
    ax2.set_ylabel("products")
    train_p, test_p = set(df.iloc[sp["train"]]["product_id"]), set(df.iloc[sp["test"]]["product_id"])
    shared = len(train_p & test_p)
    title(ax2, "...but scenes are shared across splits",
          f"{shared:,} of {len(test_p):,} test products also occur in train")
    fig.tight_layout()
    save(fig, outdir, "fig_splits")


# ---------------------------------------------------------------------------
def summary(df, n_raw, outdir):
    sp = splits(df)
    train_p = set(df.iloc[sp["train"]]["product_id"])
    test_p = set(df.iloc[sp["test"]]["product_id"])
    per_prod = df.groupby("product_id").size()
    s = {
        "patches_in_manifest": int(n_raw),
        "patches_after_bad_product_filter": int(len(df)),
        "products": int(df["product_id"].nunique()),
        "patches_per_product_median": int(per_prod.median()),
        "patches_per_product_max": int(per_prod.max()),
        "delta_days_median": round(float(df["patch_delta_days"].median()), 2),
        "delta_days_mean": round(float(df["patch_delta_days"].mean()), 2),
        "delta_days_max": round(float(df["patch_delta_days"].max()), 2),
        "pct_patches_under_5pct_cloud": round(
            float((df[["thick_cloud_pct", "thin_cloud_pct"]].sum(axis=1) < 5).mean() * 100), 1),
        "unlabelled_pixel_pct": round(float(df["wc_nodata_frac"].mean() * 100), 3),
        "date_range": [str(df["date_phi"].min().date()), str(df["date_phi"].max().date())],
        "split_sizes": {k: int(len(v)) for k, v in sp.items()},
        "test_products_also_in_train": [int(len(train_p & test_p)), int(len(test_p))],
        "class_prior_pct_by_pixel": {
            WC_NAME[c]: round(float(df[f"wc_frac_{c}"].mean() * 100), 3) for c in WC_CODES},
    }
    (outdir / "manifest_summary.json").write_text(json.dumps(s, indent=2))
    print("\n" + json.dumps(s, indent=2))
    print("\n  wrote manifest_summary.json")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", type=Path,
                   default=Path("/shared/home/elucas/scratch/terra-sat-drift/outputs/dataset_report"))
    args = p.parse_args()

    setup()
    df, n_raw = load()
    print(f"Loaded {len(df):,} patches ({n_raw - len(df):,} dropped as bad products)\n")
    fig_acquisition(df, args.outdir)
    fig_clouds(df, args.outdir)
    fig_landcover(df, args.outdir)
    fig_geography(df, args.outdir)
    fig_splits(df, args.outdir)
    summary(df, n_raw, args.outdir)


if __name__ == "__main__":
    main()
