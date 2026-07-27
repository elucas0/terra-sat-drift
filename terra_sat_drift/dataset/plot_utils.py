"""Shared plotting helpers for the WorldCover LULC datasets.

These live here rather than on each Dataset because the `plot` methods were
copy-pasted between the dataset classes, and the legend bug they all carried
(swatches coloured from a different normalisation than the panels, and labelled
with WorldCover *codes* instead of class names) had to be fixed in each copy.
"""

import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch

from .constants import NO_LABEL_COLOR, WC_CLASS_COLORS, WC_CLASS_DISPLAY_NAMES

_PALETTE = np.array([to_rgb(c) for c in WC_CLASS_COLORS], dtype=np.float32)
_NO_LABEL = np.array(to_rgb(NO_LABEL_COLOR), dtype=np.float32)


def labels_to_rgb(label_map: np.ndarray) -> np.ndarray:
    """Maps an (H, W) label map to an (H, W, 3) float image via the WorldCover legend.

    A direct lookup rather than `imshow(cmap=..., vmin=-1, vmax=10)`, so a class
    index always gets the same colour whichever classes a patch happens to
    contain, and so `legend_handles` can be built from the same table.
    Out-of-range and ignore_index (-1) pixels take the no-label colour.
    """
    label_map = np.asarray(label_map)
    out = np.tile(_NO_LABEL, (*label_map.shape, 1))
    valid = (label_map >= 0) & (label_map < len(_PALETTE))
    out[valid] = _PALETTE[label_map[valid]]
    return out


def legend_handles(*label_maps: np.ndarray) -> list[Patch]:
    """Legend entries for the classes present in the given label maps.

    Listing only the classes actually present keeps a two-class patch from
    carrying an eleven-entry legend. The swatch colours come from the same table
    `labels_to_rgb` paints with, so a swatch cannot disagree with the image.
    """
    present = np.unique(np.concatenate([np.unique(m) for m in label_maps]))
    handles = [
        Patch(facecolor=WC_CLASS_COLORS[i], edgecolor="#52514e", linewidth=0.5,
              label=WC_CLASS_DISPLAY_NAMES[i])
        for i in range(len(WC_CLASS_COLORS)) if i in present
    ]
    if np.any(present < 0):
        handles.insert(0, Patch(facecolor=NO_LABEL_COLOR, edgecolor="#52514e",
                                linewidth=0.5, label="No label"))
    return handles


def stretch_rgb(image: np.ndarray, rgb_indices=(2, 1, 0),
                low: float = 2.0, high: float = 98.0) -> np.ndarray:
    """Builds a contrast-stretched (H, W, 3) composite from a (C, H, W) array.

    Always works on a copy: `.numpy()` on a CPU tensor returns a view, so
    stretching in place corrupted the caller's sample. A percentile stretch is
    used rather than min-max because a single saturated pixel (the real PhiSat-2
    red-edge bands reach the 12-bit ceiling of 4095) flattens the composite.
    """
    rgb = np.asarray(image)[list(rgb_indices)].astype(np.float32)
    lo = np.percentile(rgb, low, axis=(1, 2), keepdims=True)
    hi = np.percentile(rgb, high, axis=(1, 2), keepdims=True)
    rgb = np.clip((rgb - lo) / np.clip(hi - lo, 1e-6, None), 0, 1)
    return np.transpose(rgb, (1, 2, 0))
