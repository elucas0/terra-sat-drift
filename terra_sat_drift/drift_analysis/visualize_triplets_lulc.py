#!/usr/bin/env python
"""Visualise the co-registered triplets dataset with LULC overlays + cross-sensor error.

The ``phisat2_s2b_dataset_v1.h5`` triplets store the *same* ground patch as seen
by two sensors -- real PhiSat-2 (``real``) and Sentinel-2B (``s2b``) -- plus a
shared WorldCover label map. This script produces, per sampled patch:

    Real RGB | S2B RGB | LULC mask | Real+mask | S2B+mask | approximation error

and, aggregated over a larger sample, a per-class bar chart of the cross-sensor
**approximation error** -- how well Sentinel-2B approximates real PhiSat-2, broken
down by land-cover class.

What "approximation error" means here
-------------------------------------
The two sensors sit on radiometric scales that differ by more than an order of
magnitude, so differencing raw digital numbers is meaningless. Both views are
first standardised with their *own* per-band statistics (the exact normalisation
``PhisatPairedLULCDataset`` feeds the encoder: sqrt -> clip -> per-band z-score),
which removes the first-order per-band mean/variance offset. The residual
per-pixel L1 distance across bands,

    err(u, v) = mean_c | real_norm[c, u, v] - s2b_norm[c, u, v] | ,

is therefore the part of the gap that alignment of the marginals cannot explain
-- PSF, band response, view geometry, co-registration slop -- i.e. the error a
sensor-to-sensor approximation still carries after radiometric matching.

Reuses ``PhisatPairedLULCDataset`` (same split/normalisation as training) and the
shared ``plot_utils`` / ``constants`` so the colours and stretch match every other
figure in the repo.

Example
-------
    python visualize_triplets_lulc.py --num-samples 6 --split test \
        --output-dir ./triplet_viz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: this is a file-producing script
import matplotlib.pyplot as plt
import numpy as np
import torch

# The dataset code lives in the `dataset` package one level up (terra_sat_drift/).
# Add it to the path so `dataset.*` imports resolve when the script is run from
# anywhere, mirroring how the drift_analysis notebooks bootstrap themselves.
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from dataset.constants import WC_CLASS_COLORS, WC_CLASS_DISPLAY_NAMES  # noqa: E402
from dataset.dataset_paired_triplets_lulc import PhisatPairedLULCDataset  # noqa: E402
from dataset.plot_utils import labels_to_rgb, legend_handles, stretch_rgb  # noqa: E402

DEFAULT_IMAGES = "/shared/projects/phisat2/data/processed/triplets_v1/phisat2_s2b_dataset_v1.h5"
DEFAULT_LABELS = "/shared/projects/phisat2/data/processed/worldcover_all_clean_v1/worldcover_all_clean_labels_v1.h5"
DEFAULT_MANIFEST = "/shared/projects/phisat2/data/processed/worldcover_all_clean_v1/worldcover_all_clean_manifest_v1.csv"


def _to_numpy(x) -> np.ndarray:
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Alpha-blend the LULC colour map over an (H, W, 3) RGB composite.

    Only labelled pixels are tinted; ignore (-1) pixels keep the underlying
    imagery, so the overlay reads as "colour on top of the scene" rather than a
    black wash over unlabelled ground.
    """
    mask_rgb = labels_to_rgb(mask)
    valid = (mask >= 0)[..., None]
    blended = np.where(valid, (1.0 - alpha) * rgb + alpha * mask_rgb, rgb)
    return np.clip(blended, 0.0, 1.0)


def approximation_error(img_target: np.ndarray, img_source: np.ndarray) -> np.ndarray:
    """Per-pixel mean-over-bands L1 distance between the two normalised views."""
    return np.abs(img_target - img_source).mean(axis=0)


def plot_sample(sample: dict, alpha: float, error_vmax: float | None = None) -> plt.Figure:
    """Six-panel per-patch figure: two sensors, mask, two overlays, error map."""
    img_real = _to_numpy(sample["image_target"])   # (7, H, W) normalised (real)
    img_s2b = _to_numpy(sample["image_source"])     # (7, H, W) normalised (s2b)
    mask = _to_numpy(sample["mask"])

    rgb_real = stretch_rgb(img_real)
    rgb_s2b = stretch_rgb(img_s2b)
    err = approximation_error(img_real, img_s2b)
    vmax = error_vmax if error_vmax is not None else float(np.percentile(err, 99))

    labelled = mask >= 0
    mean_err = float(err[labelled].mean()) if labelled.any() else float(err.mean())

    panels = [
        ("PhiSat-2 (real)", rgb_real, None),
        ("Sentinel-2B", rgb_s2b, None),
        ("LULC mask", labels_to_rgb(mask), None),
        ("Real + LULC", overlay_mask(rgb_real, mask, alpha), None),
        ("S2B + LULC", overlay_mask(rgb_s2b, mask, alpha), None),
        (f"Approx. error (mean {mean_err:.2f})", err, "error"),
    ]

    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.6))
    err_im = None
    for ax, (title, arr, kind) in zip(axes, panels):
        if kind == "error":
            err_im = ax.imshow(arr, cmap="inferno", vmin=0.0, vmax=vmax)
        else:
            ax.imshow(arr)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    if err_im is not None:
        cbar = fig.colorbar(err_im, ax=axes[-1], fraction=0.046, pad=0.04)
        cbar.set_label("|Δ| (z-score units)", fontsize=9)

    fig.legend(
        handles=legend_handles(mask),
        loc="lower center",
        ncol=min(6, max(1, len(legend_handles(mask)))),
        bbox_to_anchor=(0.5, -0.02),
        frameon=True,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


def per_class_error(dataset: PhisatPairedLULCDataset, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean/std cross-sensor approximation error per LULC class over ``n`` patches.

    Uses running sums so memory stays flat regardless of ``n``.
    """
    n_classes = len(WC_CLASS_DISPLAY_NAMES)
    count = np.zeros(n_classes, dtype=np.float64)
    err_sum = np.zeros(n_classes, dtype=np.float64)
    err_sq = np.zeros(n_classes, dtype=np.float64)

    n = min(n, len(dataset))
    for i in range(n):
        sample = dataset[i]
        err = approximation_error(
            _to_numpy(sample["image_target"]), _to_numpy(sample["image_source"])
        ).ravel()
        mask = _to_numpy(sample["mask"]).ravel()
        for c in range(n_classes):
            sel = mask == c
            if not sel.any():
                continue
            e = err[sel]
            count[c] += e.size
            err_sum[c] += e.sum()
            err_sq[c] += np.square(e).sum()

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, err_sum / count, np.nan)
        var = np.where(count > 0, err_sq / count - mean**2, np.nan)
    std = np.sqrt(np.clip(var, 0.0, None))
    return mean, std, count


def plot_per_class_error(mean: np.ndarray, std: np.ndarray, count: np.ndarray, n: int) -> plt.Figure:
    """Bar chart of mean approximation error per LULC class, sorted worst-first."""
    present = count > 0
    idx = np.where(present)[0]
    order = idx[np.argsort(-np.nan_to_num(mean[idx]))]

    labels = [WC_CLASS_DISPLAY_NAMES[c] for c in order]
    colors = [WC_CLASS_COLORS[c] for c in order]
    heights = mean[order]
    errs = std[order]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(
        range(len(order)), heights, yerr=errs, capsize=4,
        color=colors, edgecolor="#52514e", linewidth=0.6, error_kw={"ecolor": "#333", "alpha": 0.6},
    )
    overall = np.nansum(mean[order] * count[order]) / np.nansum(count[order])
    ax.axhline(overall, color="#d62728", linestyle="--", linewidth=1.2,
               label=f"pixel-weighted mean = {overall:.2f}")

    for bar, c in zip(bars, order):
        frac = 100.0 * count[c] / count.sum()
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{frac:.0f}%", ha="center", va="bottom", fontsize=8, color="#333")

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Mean approximation error  |Δ| (z-score units)")
    ax.set_title(f"Cross-sensor (S2B → real PhiSat-2) approximation error by land cover "
                 f"({n} patches)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    return fig


def build_dataset(args) -> PhisatPairedLULCDataset:
    return PhisatPairedLULCDataset(
        h5_images_path=args.images,
        h5_labels_path=args.labels,
        manifest_path=args.manifest,
        split=args.split,
        transform=None,             # raw geometry: overlays must stay pixel-aligned
        target_domain="real",
        source_domain="s2b",
    )


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", default=DEFAULT_IMAGES, help="triplets HDF5 path")
    p.add_argument("--labels", default=DEFAULT_LABELS, help="WorldCover labels HDF5 path")
    p.add_argument("--manifest", default=DEFAULT_MANIFEST, help="manifest CSV path")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--num-samples", type=int, default=4,
                   help="number of patches to render as overlay figures")
    p.add_argument("--indices", type=int, nargs="*", default=None,
                   help="explicit dataset indices to render (overrides --num-samples selection)")
    p.add_argument("--error-samples", type=int, default=500,
                   help="patches aggregated for the per-class error bar chart")
    p.add_argument("--alpha", type=float, default=0.5, help="mask overlay opacity")
    p.add_argument("--seed", type=int, default=0, help="seed for random patch selection")
    p.add_argument("--output-dir", default="./triplet_viz", help="where figures are written")
    p.add_argument("--dpi", type=int, default=140)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(args)
    print(f"Loaded '{args.split}' split: {len(dataset)} patches (real vs s2b).")

    if args.indices is not None:
        indices = [i for i in args.indices if 0 <= i < len(dataset)]
    else:
        rng = np.random.default_rng(args.seed)
        k = min(args.num_samples, len(dataset))
        indices = sorted(rng.choice(len(dataset), size=k, replace=False).tolist())

    for i in indices:
        sample = dataset[i]
        fig = plot_sample(sample, alpha=args.alpha)
        path = out / f"triplet_overlay_{i:06d}.png"
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {path}")

    # mean, std, count = per_class_error(dataset, args.error_samples)
    # n_used = min(args.error_samples, len(dataset))
    # fig = plot_per_class_error(mean, std, count, n_used)
    # err_path = out / "class_approximation_error.png"
    # fig.savefig(err_path, dpi=args.dpi, bbox_inches="tight")
    # plt.close(fig)
    # print(f"  wrote {err_path}")

    # print("\nPer-class approximation error (mean |Δ|, z-score units):")
    # order = np.argsort(-np.nan_to_num(mean))
    # for c in order:
    #     if count[c] > 0:
    #         print(f"  {WC_CLASS_DISPLAY_NAMES[c]:<26s} {mean[c]:.3f} ± {std[c]:.3f}"
    #               f"   ({100 * count[c] / count.sum():4.1f}% of pixels)")


if __name__ == "__main__":
    main()
