"""
Paired (co-registered) two-sensor LULC dataset built on the triplets HDF5.

``phisat2_s2b_dataset_v1.h5`` stores three spatially co-registered views of the
same 256x256 ground patch under a *shared* first index:

    real/images  (N, 8, 256, 256)  int16   real PhiSat-2 L1
    sim/images   (N, 8, 256, 256)  int16   PhiSat-2 simulated from Sentinel-2
    s2b/images   (N, 7, 256, 256)  int16   Sentinel-2B

WorldCover labels live in a second HDF5 and are addressed by the manifest's
``label_h5_index``. Because the three views share the patch index and are
pixel-aligned, one label map is valid for all of them.

The existing ``PhisatRealLULCDataset`` / ``PhisatS2LULCDataset`` each expose one
domain at a time, which is enough for plain supervised training but cannot
supply the *positive pairs* a cross-sensor contrastive objective needs. This
dataset returns both views of the same patch plus their shared mask.
"""

from typing import Optional

import albumentations as A
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .constants import WC_CLASS_MAPPING
from .plot_utils import labels_to_rgb, legend_handles, stretch_rgb

# Product ids excluded upstream for radiometric/geometric quality reasons.
# Kept identical to the single-domain datasets so splits stay comparable.
BAD_PRODUCT_IDS = [
    1294, 1296, 1342, 1385, 1397, 1420, 1460, 1497, 1647, 1854, 2223, 2246,
    2259, 2373, 2631, 2640, 2743, 2834, 2853, 3374, 3619, 4071, 4693, 4813,
    4942, 2352, 2882, 3322, 3914, 4702, 1333, 1466, 1615, 2460, 2729, 2763,
]

# Per-domain radiometric statistics, in Blue..NIR order, measured on sqrt-space
# digital numbers. Each domain is standardised with its *own* statistics: this
# removes the first-order (per-band mean/variance) part of the covariate shift
# in the input space, so the representation-level losses are left to handle the
# harder, non-affine part of the gap (PSF, band response, view geometry).
DOMAIN_STATS: dict[str, dict] = {
    "real": {
        "mean": np.array([14.5305, 14.4030, 15.4191, 13.6231, 14.2143, 14.7041, 13.1745], dtype=np.float32),
        "std": np.array([10.6197, 9.4811, 9.0923, 10.5712, 10.4277, 10.3784, 9.7216], dtype=np.float32),
        "clip": 38.729,
        "clip_mode": "smooth",
        "h5_key": "real/images",
        "band_slice": slice(1, 8),  # 8 stored bands -> Blue..NIR
    },
    "s2b": {
        "mean": np.array([49.0215, 48.4241, 49.2270, 51.1619, 55.4031, 57.3537, 56.7685], dtype=np.float32),
        "std": np.array([6.5464, 6.9918, 9.1444, 8.3999, 7.9740, 8.3373, 8.4429], dtype=np.float32),
        "clip": 100.0,
        "clip_mode": "hard",
        "h5_key": "s2b/images",
        "band_slice": slice(0, 7),
    },
    # The simulated view is Sentinel-2 pushed through the PhiSat-2 sensor model,
    # so it keeps Sentinel-2's *radiometric* scale (scaled reflectance) and only
    # takes on PhiSat-2's spatial characteristics. Its statistics are therefore
    # close to s2b's, not to real PhiSat-2's -- clip is 100.0, not 38.729.
    # Values from the dataset v1 specification, bands 1..7 of the stored 8.
    "sim": {
        "mean": np.array([49.0253, 48.4297, 49.2364, 51.1648, 55.4065, 57.3572, 56.7808], dtype=np.float32),
        "std": np.array([6.5203, 6.9570, 9.0981, 8.3858, 7.9555, 8.3155, 8.3664], dtype=np.float32),
        "clip": 100.0,
        # The simulator emits small negative values as a processing artefact;
        # `normalize_domain` floors at 0 before the sqrt, so a hard clip is
        # sufficient here (matching the s2b treatment).
        "clip_mode": "hard",
        "h5_key": "sim/images",
        "band_slice": slice(1, 8),
    },
}


def smooth_clip(x: np.ndarray, clip_value: float, softness: float = 6.0) -> np.ndarray:
    """Smoothly saturates ``x`` toward ``clip_value`` instead of hard-clipping.

    Identical to ``PhisatRealLULCDataset.smooth_clip``; duplicated here so the
    paired dataset has no import dependency on the single-domain classes.
    """
    beta = 4.0 / max(softness, 1e-6)
    z = beta * (np.asarray(clip_value, dtype=np.float64) - x)
    softplus = np.logaddexp(0.0, z) / beta
    return clip_value - softplus


def normalize_domain(img_chw: np.ndarray, stats: dict, softness: float = 6.0) -> np.ndarray:
    """sqrt -> (smooth or hard) clip -> per-band z-score, using ``stats``."""
    x = np.sqrt(np.maximum(img_chw, 0))
    if stats["clip_mode"] == "smooth":
        x = smooth_clip(x, stats["clip"], softness=softness)
    else:
        x = np.clip(x, None, stats["clip"])
    return ((x - stats["mean"][:, None, None]) / stats["std"][:, None, None]).astype(np.float32)


def build_paired_transform(train: bool = True) -> Optional[A.Compose]:
    """Geometric-only augmentation applied *identically* to both sensor views.

    This is not cosmetic. The contrastive objectives rely on the two views
    being the same ground location, and the pixel-level objective additionally
    relies on pixel (u, v) meaning the same place in both views. Any transform
    sampled independently per view would silently destroy that correspondence
    and turn true positives into misaligned pairs. Albumentations'
    ``additional_targets`` mechanism guarantees one sampled transform is
    replayed across ``image``, ``image_source`` and ``mask``.

    Photometric augmentation is deliberately omitted: the sensor difference is
    itself the radiometric perturbation we want the encoder to become invariant
    to, and synthetic jitter on top of it would confound the measurement.
    """
    if not train:
        return None
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ],
        additional_targets={"image_source": "image"},
    )


class PhisatPairedLULCDataset(Dataset):
    """Co-registered (target-sensor, source-sensor, shared-label) triples.

    Args:
        h5_images_path: path to the triplets HDF5.
        h5_labels_path: path to the WorldCover labels HDF5.
        manifest_path: manifest CSV mapping ``patch_index`` -> ``label_h5_index``.
        split: ``"train"`` | ``"val"`` | ``"test"``. Uses the same seed-42
            shuffle and 80/10/10 proportions as the single-domain datasets, so
            a paired run is directly comparable with an existing single-domain
            run.
        transform: joint albumentations pipeline (see ``build_paired_transform``).
        max_samples: optional cap, applied after splitting.
        target_domain: sensor the student must ultimately work on (``"real"``
            or ``"sim"``).
        source_domain: sensor the teacher is trusted on (``"s2b"``).
        domain_stats: optional override of ``DOMAIN_STATS``.

    Returns per item:
        ``image_target``  (7, H, W) float32, normalised with target statistics
        ``image_source``  (7, H, W) float32, normalised with source statistics
        ``mask``          (H, W)    int64, WorldCover mapped to 0..10, -1 = ignore
        ``patch_index``   scalar int64, the shared co-registration key
    """

    def __init__(
        self,
        h5_images_path: str,
        h5_labels_path: str,
        manifest_path: str,
        split: str = "train",
        transform: A.Compose | None = None,
        max_samples: Optional[int] = None,
        target_domain: str = "real",
        source_domain: str = "s2b",
        domain_stats: Optional[dict] = None,
    ):
        # `source_domain=None` puts the dataset in single-domain mode: only the
        # target view is read and returned. That is what the no-KD baseline
        # needs, and routing it through this class rather than a separate one
        # means the baseline inherits the *identical* split, normalisation and
        # augmentation as the KD runs -- which is the whole point of a control.
        if source_domain is not None and target_domain == source_domain:
            raise ValueError("target_domain and source_domain must differ for paired training.")

        stats_table = domain_stats if domain_stats is not None else DOMAIN_STATS
        for name in (target_domain, source_domain):
            if name is not None and name not in stats_table:
                raise ValueError(f"Unknown domain {name!r}; known: {sorted(stats_table)}")

        self.h5_images_path = h5_images_path
        self.h5_labels_path = h5_labels_path
        self.transform = transform
        self.target_domain = target_domain
        self.source_domain = source_domain
        self.paired = source_domain is not None
        self.target_stats = stats_table[target_domain]
        self.source_stats = stats_table[source_domain] if self.paired else None

        df = pd.read_csv(manifest_path)
        df = df[~df["product_id"].isin(BAD_PRODUCT_IDS)].reset_index(drop=True)

        np.random.seed(42)
        indices = np.arange(len(df))
        np.random.shuffle(indices)

        train_end = int(0.8 * len(indices))
        val_end = int(0.9 * len(indices))
        if split == "train":
            curr_indices = indices[:train_end]
        elif split == "val":
            curr_indices = indices[train_end:val_end]
        else:
            curr_indices = indices[val_end:]

        self.df = df.iloc[curr_indices].reset_index(drop=True)
        if max_samples:
            self.df = self.df.head(max_samples)

        # Remap WorldCover codes once, as a lookup table, instead of looping
        # over 11 classes for every sample.
        self._wc_lut = np.full(256, -1, dtype=np.int64)
        for code, target in WC_CLASS_MAPPING.items():
            self._wc_lut[int(code)] = target

        self.h5_images = None
        self.h5_labels = None

    def __len__(self) -> int:
        return len(self.df)

    def _read_view(self, patch_idx: int, stats: dict) -> np.ndarray:
        raw = self.h5_images[stats["h5_key"]][patch_idx][stats["band_slice"]].astype(np.float32)
        return normalize_domain(raw, stats)

    def __getitem__(self, idx: int) -> dict:
        if self.h5_images is None:
            # Opened lazily so each DataLoader worker gets its own handle.
            self.h5_images = h5py.File(self.h5_images_path, "r")
            self.h5_labels = h5py.File(self.h5_labels_path, "r")

        row = self.df.iloc[idx]
        patch_idx = int(row["patch_index"])
        label_h5_idx = int(row["label_h5_index"])

        img_t = self._read_view(patch_idx, self.target_stats)
        img_s = self._read_view(patch_idx, self.source_stats) if self.paired else None

        mask = self.h5_labels["worldcover/labels"][label_h5_idx].astype(np.int64)
        mask = self._wc_lut[np.clip(mask, 0, 255)]

        if self.transform is not None:
            # Albumentations works in HWC; the HDF5 stores CHW. An unused
            # `additional_target` is simply ignored, so the same Compose serves
            # both modes and the sampled augmentation is identical either way.
            kwargs = {"image": np.ascontiguousarray(img_t.transpose(1, 2, 0)), "mask": mask}
            if self.paired:
                kwargs["image_source"] = np.ascontiguousarray(img_s.transpose(1, 2, 0))
            out = self.transform(**kwargs)
            img_t = out["image"].transpose(2, 0, 1)
            mask = out["mask"]
            if self.paired:
                img_s = out["image_source"].transpose(2, 0, 1)

        sample = {
            "image_target": torch.as_tensor(np.ascontiguousarray(img_t)).float(),
            "mask": torch.as_tensor(np.ascontiguousarray(mask)).long(),
            "patch_index": torch.as_tensor(patch_idx).long(),
        }
        if self.paired:
            sample["image_source"] = torch.as_tensor(np.ascontiguousarray(img_s)).float()
        return sample

    DOMAIN_LABELS = {
        "real": "PhiSat-2 (real)",
        "sim": "PhiSat-2 (simulated)",
        "s2b": "Sentinel-2B",
    }

    def plot(self, sample: dict, suptitle: str | None = None, show_axes: bool = False):
        """Plots the two co-registered views side by side with the label mask.

        Both views are stretched independently, so the panels show the *content*
        each sensor sees rather than their radiometric offset -- the two domains
        sit on scales that differ by more than an order of magnitude, and a shared
        stretch would render the PhiSat-2 view almost black.

        Accepts an optional "prediction" entry, so the same method serves for
        eyeballing model output.
        """
        def _np(x):
            if x is None:
                return None
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().numpy()
            return x[0] if x.ndim == 4 or (x.ndim == 3 and x.shape[0] == 1 and x.shape[-1] != 3) else x

        img_t = _np(sample["image_target"])
        img_s = _np(sample.get("image_source"))   # absent in single-domain mode
        mask = _np(sample["mask"])
        prediction = _np(sample.get("prediction"))
        if mask.ndim == 3:
            mask = mask[0]
        if prediction is not None and prediction.ndim == 3:
            prediction = prediction[0]

        panels = [(self.DOMAIN_LABELS.get(self.target_domain, self.target_domain),
                   stretch_rgb(img_t))]
        if img_s is not None:
            panels.append((self.DOMAIN_LABELS.get(self.source_domain, self.source_domain),
                           stretch_rgb(img_s)))
        panels.append(("Mask", labels_to_rgb(mask)))
        if prediction is not None:
            panels.append(("Prediction", labels_to_rgb(prediction)))

        fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5))
        for ax, (name, arr) in zip(np.atleast_1d(axes), panels):
            ax.imshow(arr)
            ax.set_title(name)
            if not show_axes:
                ax.axis("off")

        if suptitle:
            fig.suptitle(suptitle)

        maps = [mask] if prediction is None else [mask, prediction]
        fig.legend(
            handles=legend_handles(*maps),
            loc="center left",
            bbox_to_anchor=(0.84, 0.5),
            title="Class names",
            frameon=True,
        )
        fig.tight_layout(rect=(0, 0, 0.82, 1))
        return fig
