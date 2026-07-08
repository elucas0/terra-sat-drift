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
from torch.utils.data import Dataset, DataLoader

from dataset.phisat_lulc import PhisatRealLULCDataset
from dataset.constants import WC_CLASS_MAPPING

def main(): 
    parser = argparse.ArgumentParser(description="Pre-train TerraMind Tiny on Real LULC data")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone", type=str, default="terramind_v1_tiny")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    seed_everything(42)

    # Paths
    h5_images_path = "/shared/projects/phisat2/data/processed/triplets_v1/phisat2_s2b_dataset_v1.h5"
    h5_labels_path = "/shared/projects/phisat2/data/processed/worldcover_all_clean_v1/worldcover_all_clean_labels_v1.h5"
    manifest_path = "/shared/projects/phisat2/data/processed/worldcover_all_clean_v1/worldcover_all_clean_manifest_v1.csv"
    
    output_dir = Path("/shared/home/elucas/scratch/terra-sat-drift/outputs/terramind_lulc_pretrain")
    output_dir.mkdir(parents=True, exist_ok=True)

    target_size = (256, 256) 

    # Attention: Si les données réelles ne nécessitent pas de np.sqrt (ex: si elles sont en DN brut 0-10000), 
    # il faut retirer A.Lambda(np.sqrt) et recalculer les `mean` et `std` sur la distribution des données réelles.
    train_transform = A.Compose([
        A.RandomCrop(height=target_size[0], width=target_size[1]),
        A.Lambda(image=lambda x, **kwargs: np.sqrt(np.maximum(x, 0)).astype(np.float32)),
        A.Normalize(
            mean=[49.0, 48.0, 49.0, 51.0, 55.0, 57.0, 56.0], 
            std=[7.0, 7.0, 9.0, 8.0, 8.0, 8.0, 8.0],
            max_pixel_value=1.0 
        ), 
        A.pytorch.ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.CenterCrop(height=target_size[0], width=target_size[1]),
        A.Lambda(image=lambda x, **kwargs: np.sqrt(np.maximum(x, 0)).astype(np.float32)),
        A.Normalize(
            mean=[49.0, 48.0, 49.0, 51.0, 55.0, 57.0, 56.0], 
            std=[7.0, 7.0, 9.0, 8.0, 8.0, 8.0, 8.0],
            max_pixel_value=1.0 
        ), 
        A.pytorch.ToTensorV2()
    ])

    train_ds = PhisatRealLULCDataset(h5_images_path, h5_labels_path, manifest_path, "train", train_transform, args.max_samples)
    val_ds = PhisatRealLULCDataset(h5_images_path, h5_labels_path, manifest_path, "val", val_transform, args.max_samples // 10 if args.max_samples else None)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model definition
    BACKBONE_NECK_INDICES = {
        "tiny": [1, 3, 4, 5],
        "small": [1, 3, 4, 5],
        "base": [2, 5, 8, 11],
    }
    
    # We use 7 bands: B, G, R, RE1, RE2, RE3, NIR
    backbone_bands = {
        "S2L1C": {
            "B02": 0, "B03": 1, "B04": 2, 
            "B05": 3, "B06": 4, "B07": 5, "B08": 6
        }
    }

    model_args = {
        "backbone": args.backbone,
        "backbone_pretrained": True,
        "backbone_modalities": ["S2L1C"],
        "backbone_bands": backbone_bands,
        "necks": [
            {"name": "SelectIndices", "indices": BACKBONE_NECK_INDICES["tiny"]},
            {"name": "ReshapeTokensToImage", "remove_cls_token": False},
            {"name": "LearnedInterpolateToPyramidal"},
        ],
        "decoder": "UNetDecoder",
        "decoder_channels": [256, 128, 64, 32],
        "head_dropout": 0.1,
        "num_classes": 11, # 11 WorldCover classes
    }

    task = SemanticSegmentationTask(
        model_factory="EncoderDecoderFactory",
        model_args=model_args,
        lr=args.lr,
        ignore_index=-1,
        optimizer="AdamW",
        optimizer_hparams={"weight_decay": 0.05},
        class_names=list(WC_CLASS_MAPPING.keys()),
    )

    logger = WandbLogger(project="terra-sat-drift", name=f"pretrain_lulc_real_{args.backbone}")
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="best-val_mIoU",
        monitor="val/mIoU",
        mode="max",
        save_top_k=1,
    )

    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback, EarlyStopping(monitor="val/mIoU", patience=10, mode="max")],
        precision="16-mixed"
    )

    trainer.fit(task, train_loader, val_loader)

if __name__ == "__main__":
    main()