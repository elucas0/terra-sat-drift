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
from torch.utils.data import Dataset, DataLoader

from .constants import WC_CLASS_MAPPING

class PhisatRealLULCDataset(Dataset):
    """Dataset for pre-training on real PhiSat-2 data with WorldCover labels."""
    def __init__(
        self, 
        h5_images_path: str,
        h5_labels_path: str,
        manifest_path: str,
        split: str = "train",
        transform: Optional[A.Compose] = None,
        max_samples: Optional[int] = None
    ):
        self.h5_images_path = h5_images_path
        self.h5_labels_path = h5_labels_path
        self.transform = transform
        
        # Load and filter manifest
        df = pd.read_csv(manifest_path)
        
        # Filter out bad products if any (context mentioned BAD_PRODUCT_IDS)
        BAD_PRODUCT_IDS = [1294,1296,1342,1385,1397,1420,1460,1497,1647,1854,2223,2246,2259,2373,2631,2640,2743,2834,2853,3374,3619,4071,4693,4813,4942,2352,2882,3322,3914,4702,1333,1466,1615,2460,2729,2763]
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
        
        img = self.h5_images['real/images'][patch_idx][:7].astype(np.float32)
        img = np.transpose(img, (1, 2, 0)) # To HWC
        
        # Initialize to -1 to properly ignore non-categorized NoData pixels
        mask = self.h5_labels['worldcover/labels'][label_h5_idx].astype(np.int64)
        new_mask = np.full_like(mask, -1)
        
        for val, target in WC_CLASS_MAPPING.items():
            new_mask[mask == int(val)] = target
            
        if self.transform:
            augmented = self.transform(image=img, mask=new_mask)
            img = augmented['image']
            mask = augmented['mask']
            
        # Modalité S2L1C explicite pour TerraMind
        return {"image": {"S2L1C": img}, "mask": mask.long()}