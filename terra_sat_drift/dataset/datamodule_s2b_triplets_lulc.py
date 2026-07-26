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

from terratorch.tasks import SemanticSegmentationTask
from terratorch.datamodules import GenericNonGeoSegmentationDataModule

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger

import cv2
import albumentations as A
from torch.utils.data import DataLoader

import lightning.pytorch as pl

from dataset.dataset_s2b_triplets_lulc import PhisatS2LULCDataset

class PhisatS2LULCDataModule(pl.LightningDataModule):
    def __init__(
        self,
        h5_images_path: str,
        h5_labels_path: str,
        manifest_path: str,
        batch_size: int,
        num_workers: int,
        train_transform: A.Compose | None,
        val_transform: A.Compose | None,
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        self.h5_images_path = h5_images_path
        self.h5_labels_path = h5_labels_path
        self.manifest_path = manifest_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.max_samples = max_samples

    def setup(self, stage: Optional[str] = None):
        train_max_samples = self.max_samples
        val_max_samples = max(1, self.max_samples // 10) if self.max_samples else None

        if stage in (None, "fit"):
            self.train_dataset = PhisatS2LULCDataset(
                self.h5_images_path,
                self.h5_labels_path,
                self.manifest_path,
                "train",
                self.train_transform,
                train_max_samples,
            )
            self.val_dataset = PhisatS2LULCDataset(
                self.h5_images_path,
                self.h5_labels_path,
                self.manifest_path,
                "val",
                self.val_transform,
                val_max_samples,
            )
            self.test_dataset = PhisatS2LULCDataset(
                self.h5_images_path,
                self.h5_labels_path,
                self.manifest_path,
                "test",
                self.val_transform,
                val_max_samples,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )