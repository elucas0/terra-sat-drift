import os
import torch
import numpy as np
import h5py
import pandas as pd
import logging
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import cv2
import albumentations as A
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from torch.utils.data import Dataset, DataLoader

from .constants import WC_CLASS_MAPPING

class PhisatS2LULCDataset(Dataset):
    """Dataset for pre-training on Sentinel-2 data with WorldCover labels."""
    def __init__(
        self, 
        h5_images_path: str,
        h5_labels_path: str,
        manifest_path: str,
        split: str = "train",
        transform: A.Compose | None = None,
        max_samples: Optional[int] = None
    ):
        self.h5_images_path = h5_images_path
        self.h5_labels_path = h5_labels_path
        self.transform = transform
        
        # Load and filter manifest
        df = pd.read_csv(manifest_path)
        
        # Filter out bad products if any (context mentioned BAD_PRODUCT_IDS)
        BAD_PRODUCT_IDS = [1294,1296,1342,1385,1397,1420,1460,1497,1647,1854,2223,2246,2259,2373,2631,2640,2743,2834,2853,3374,3619,4071,4693,4813,4942,2352,2882,3322,3914,4702,1333,1466,1615,2460,2729,2763]
        # Blue, Green, Red, RE1, RE2, RE3, NIR — s2b ordering
        self.S2B_MEAN = np.array([49.0215, 48.4241, 49.2270, 51.1619, 55.4031, 57.3537, 56.7685], dtype=np.float32)
        self.S2B_STD  = np.array([6.5464, 6.9918, 9.1444, 8.3999, 7.9740, 8.3373, 8.4429], dtype=np.float32)
        self.S2B_CLIP = 100.0
        
        df = df[~df['product_id'].isin(BAD_PRODUCT_IDS)].reset_index(drop=True)
        
        # Simple split based on positional index
        np.random.seed(42)
        indices = np.arange(len(df))
        np.random.shuffle(indices)
        
        train_end = int(0.8 * len(indices))
        val_end = int(0.9 * len(indices))
        
        if split == "train":
            curr_indices = indices[:train_end]
        elif split == "val":
            curr_indices = indices[train_end:val_end]
        else: # test
            curr_indices = indices[val_end:]
            
        self.df = df.iloc[curr_indices].reset_index(drop=True)
        
        if max_samples:
            self.df = self.df.head(max_samples)
            
        self.h5_images = None
        self.h5_labels = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.h5_images is None:
            self.h5_images = h5py.File(self.h5_images_path, 'r')
            self.h5_labels = h5py.File(self.h5_labels_path, 'r')
            
        row = self.df.iloc[idx]
        patch_idx = int(row['patch_index'])
        label_h5_idx = int(row['label_h5_index'])
        
        img = self.h5_images['s2b/images'][patch_idx].astype(np.float32)   # Blue..NIR
        img = self.sqrt_clip_zscore(img)                                             # sqrt -> clip -> z-score
        #img = np.transpose(img, (1, 2, 0))                                       # CHW -> HWC for albumentations

        mask = self.h5_labels['worldcover/labels'][label_h5_idx].astype(np.int64)
        new_mask = np.full_like(mask, -1)
        for val, target in WC_CLASS_MAPPING.items():
            new_mask[mask == int(val)] = target
        mask = new_mask

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img, mask = augmented['image'], augmented['mask']

        return {"image": {"S2L1C": img}, "mask": torch.as_tensor(mask).long()}

    def sqrt_clip_zscore(self, img_chw) -> np.ndarray:
        """img_chw: (C,H,W) raw tensor/array. mean/std/clip: per-band arrays, same normalization for every domain."""
        if isinstance(img_chw, torch.Tensor):
            img_chw = img_chw.detach().cpu().numpy()
        img = np.sqrt(np.maximum(img_chw, 0))
        img = np.clip(img, None, self.S2B_CLIP)
        return ((img - self.S2B_MEAN[:, None, None]) / self.S2B_STD[:, None, None]).astype(np.float32)
        
    def plot(self, sample, suptitle: str | None = None, show_axes: bool = False):
        if "image" in sample:
            image = sample["image"]
            if isinstance(image, dict):
                image = image.get("S2L1C", next(iter(image.values())))
        elif "S2L1C" in sample:
            image = sample["S2L1C"]
        else:
            raise KeyError("Expected 'image' or 'S2L1C' in sample for plotting.")

        mask = sample["mask"]
        prediction = sample.get("prediction")

        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
        if isinstance(prediction, torch.Tensor):
            prediction = prediction.detach().cpu().numpy()

        # Handle potential batch dimension in plotting path.
        if image.ndim == 4:
            image = image[0]
        if mask.ndim == 3:
            mask = mask[0]
        if prediction is not None and prediction.ndim == 3:
            prediction = prediction[0]

        # Normalize each channel individually to [0, 1] for visualization
        for i in range(image.shape[0]):
            c_min = image[i].min()
            c_max = image[i].max()
            if c_max > c_min:
                image[i] = (image[i] - c_min) / (c_max - c_min)
          
        # Transpose to HWC for plotting      
        image = np.transpose(image[[2, 1, 0]], (1, 2, 0))
        has_prediction = prediction is not None
        ncols = 3 if has_prediction else 2
        fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))
        if ncols == 2:
            axes = [axes[0], axes[1]]

        class_items = sorted(WC_CLASS_MAPPING.items(), key=lambda item: item[1])
        class_handles = [Patch(facecolor=plt.get_cmap("tab20")(idx / max(len(class_items) - 1, 1)), label=name)
                         for name, idx in class_items]
        if np.any(mask == -1):
            class_handles = [Patch(facecolor="black", label="No label")] + class_handles

        axes[0].imshow(image)
        axes[0].set_title("Image")

        axes[1].imshow(mask, cmap="tab20", vmin=-1, vmax=max(WC_CLASS_MAPPING.values()))
        axes[1].set_title("Mask")

        if has_prediction:
            axes[2].imshow(prediction, cmap="tab20", vmin=-1, vmax=max(WC_CLASS_MAPPING.values()))
            axes[2].set_title("Prediction")

        for ax in axes:
            if not show_axes:
                ax.axis("off")

        if suptitle:
            fig.suptitle(suptitle)

        fig.legend(
            handles=class_handles,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            title="Class Names",
            frameon=True,
        )

        fig.tight_layout(rect=(0, 0, 0.82, 1))
        return fig