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

from dataset.datamodule_s2b_triplets_lulc import PhisatS2LULCDataModule
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
    
    output_dir = Path(f"/shared/home/elucas/scratch/terra-sat-drift/outputs/{args.backbone}_pretrain_s2b_lulc")
    output_dir.mkdir(parents=True, exist_ok=True)

    datamodule = PhisatS2LULCDataModule(
        h5_images_path=h5_images_path,
        h5_labels_path=h5_labels_path,
        manifest_path=manifest_path,
        batch_size=args.batch_size,
        num_workers=16,
        train_transform=None,
        val_transform=None,
        max_samples=args.max_samples,
    )
    

    # Model definition
    BACKBONE_NECK_INDICES = {
        "tiny": [2, 5, 8, 11],
        "small": [2, 5, 8, 11],
        "base": [2, 5, 8, 11],
        "large": [5, 11, 17, 23],
    }
    
    backbone_bands = {
        "S2L1C": ["BLUE", "GREEN", "RED", "RED_EDGE_1", "RED_EDGE_2", "RED_EDGE_3", "NIR_BROAD"]
    }

    model_args = {
        "backbone": args.backbone,
        "backbone_pretrained": True,
        "backbone_modalities": ["S2L1C"],
        "backbone_bands": backbone_bands,
        "necks": [
            {"name": "SelectIndices", "indices": BACKBONE_NECK_INDICES[args.backbone.split("_")[-1]]},
            {"name": "ReshapeTokensToImage", "remove_cls_token": False},
            {"name": "LearnedInterpolateToPyramidal"},
        ],
        "decoder": "UNetDecoder",
        "decoder_channels": [256, 128, 64, 32],
        "head_dropout": 0.1,
        "num_classes": len(WC_CLASS_MAPPING), 
    }

    task = SemanticSegmentationTask(
        model_factory="EncoderDecoderFactory",
        model_args=model_args,
        lr=args.lr,
        ignore_index=-1,
        optimizer="AdamW",
        optimizer_hparams={"weight_decay": 0.05},
        class_names=list(WC_CLASS_MAPPING.keys()),
        freeze_backbone=False,
        freeze_decoder=False,
        plot_on_val=True,
    )

    logger = WandbLogger(project="encoder-lulc", name=f"pretrain_lulc_s2b_{args.backbone}")
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="best-val_mIoU",
        monitor="val/mIoU",
        mode="max",
        save_top_k=3,
    )

    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback, EarlyStopping(monitor="val/mIoU", patience=10, mode="max")],
    )

    trainer.fit(task, datamodule=datamodule)
    
    datamodule.setup(stage="test")
    trainer.test(task, datamodule=datamodule)

if __name__ == "__main__":
    main()