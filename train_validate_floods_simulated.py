import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import logging
import argparse
import json
from pathlib import Path
from typing import Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

from terratorch.tasks import SemanticSegmentationTask
from terratorch import BACKBONE_REGISTRY
from terratorch.datamodules import GenericNonGeoSegmentationDataModule
from terratorch.datamodules.sen1floods11 import Sen1Floods11NonGeoDataModule

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger

import cv2
import albumentations as A


from terra_sat_drift.data_simulation.phisat2_constants import S2_BANDS_NAMES, S2_BANDS, S2_PAN_BANDS

if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    root_dir=Path("/shared/home/elucas/datasets/sen1floods11_simulated")

    # Load band statistics from JSON file
    stats_file = f"{root_dir}/v1.1/spectral_statistics.json"
    with open(stats_file, 'r') as f:
        stats = json.load(f)

    # Extract means and stds for all bands
    means = list([stats[f"band_{i+1}"]["mean"] for i in range(8)])
    stds = list([stats[f"band_{i+1}"]["std"] for i in range(8)])
    TARGET_SIZE = (1088, 1088)  # Must be divisible by patch size (16)

    def preprocess_mask(mask, **kwargs):
        clean_mask = np.full(mask.shape, -1, dtype=np.int64)
        clean_mask[mask == 0] = 0
        clean_mask[mask == 1] = 1
        return clean_mask

    transform = [
        A.Resize(width=TARGET_SIZE[0], height=TARGET_SIZE[1], interpolation=cv2.INTER_NEAREST),
        A.Lambda(mask=preprocess_mask),
        A.pytorch.ToTensorV2(),
    ]

    datamodule = GenericNonGeoSegmentationDataModule(
        batch_size=8,
        data_root=root_dir,
        
        # We use the same roots for train/val/test and select samples via the given split files
        train_data_root=root_dir / "v1.1/data/flood_events/HandLabeled/S2Hand",
        train_label_data_root=Path("/shared/home/elucas/datasets/sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand"),
        val_data_root=root_dir / "v1.1/data/flood_events/HandLabeled/S2Hand",
        val_label_data_root=Path("/shared/home/elucas/datasets/sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand"),
        test_data_root=root_dir / "v1.1/data/flood_events/HandLabeled/S2Hand",
        test_label_data_root=Path("/shared/home/elucas/datasets/sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand"),

        # Split files
        train_split=root_dir / "v1.1/splits/flood_handlabeled/flood_train_data.txt",
        val_split=root_dir / "v1.1/splits/flood_handlabeled/flood_valid_data.txt",
        test_split=root_dir / "v1.1/splits/flood_handlabeled/flood_test_data.txt",
        
        train_transform=transform,
        val_transform=transform,
        test_transform=transform,
        means=means,
        stds=stds,
        dataset_bands=[0, 1, 2, 3, 4, 5, 6, 7],
        output_bands=[0, 1, 2, 3, 4, 5, 6, 7],
        num_workers=4,
        download=False,
        use_metadata=True,
        rgb_indices=[2, 1, 0],
        num_classes=2,
    )
    
    logger = logging.getLogger(__name__)

    BACKBONE_SIZES = {"tiny", "small", "base", "large"}
    BACKBONE_NECK_INDICES = {
        "tiny": [1, 3, 4, 5],
        "small": [1, 3, 4, 5],
        "base": [2, 5, 8, 11],
        "large": [5, 11, 17, 23],
    }

    SIM_BACKBONE_BANDS = {
        "S2L1C": {
            "B02": 0, "B03": 1, "B04": 2, "PAN": 3, 
            "B08": 4, "B05": 5, "B06": 6, "B07": 7
        }
    }

    backbone_size = "tiny"
    backbone_name = f"terramind_v1_{backbone_size}"
    neck_indices = BACKBONE_NECK_INDICES[backbone_size]

    model_args = {
        "backbone": backbone_name,
        "backbone_pretrained": True,
        "backbone_modalities": ["S2L1C"],
        "backbone_bands": SIM_BACKBONE_BANDS,
        # Necks
        "necks": [
            {"name": "SelectIndices", "indices": neck_indices},
            {"name": "ReshapeTokensToImage", "remove_cls_token": False},
            {"name": "LearnedInterpolateToPyramidal"},
        ],
        # Decoder
        "decoder": "UNetDecoder",
        "decoder_channels": [256, 128, 64, 32],
        # Head
        "head_dropout": 0.1,
        "num_classes": 2,
    }

    task = SemanticSegmentationTask(
        model_factory="EncoderDecoderFactory",
        model_args=model_args,
        lr=2e-5,
        ignore_index=-1,
        plot_on_val=False,
        loss="ce",
        optimizer="AdamW",
        optimizer_hparams={"weight_decay": 0.05},
        class_names=["background", "flood"],
        freeze_backbone=False,
    )
    # task = SemanticSegmentationTask.load_from_checkpoint(checkpoint_path="/shared/home/elucas/terra-sat-drift/outputs/terramind_sen1floods_simulated/terramind_v1_tiny_simulated/checkpoints/best-val_mIoU-v6.ckpt")

    
    experiment_name = f"terramind_v1_{backbone_size}_simulated_no_normalization"
    output_dir="/shared/home/elucas/terra-sat-drift/outputs/terramind_sen1floods_simulated"
    logger = WandbLogger(project="terra-sat-drift", name=experiment_name, save_dir=root_dir)

    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir + f"/{experiment_name}/checkpoints",
        filename="best-val_mIoU",
        monitor="val/mIoU",
        mode="max",
        save_top_k=3,
        verbose=True,
    )

    early_stopping = EarlyStopping(
        monitor="val/mIoU",
        patience=30,
        mode="max",
        verbose=True,
    )

    trainer = Trainer(
        max_epochs=100,
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
        log_every_n_steps=10,
        enable_progress_bar=True,
    )
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    datamodule.setup(stage="fit")

    trainer.fit(model=task, datamodule=datamodule)
    
    datamodule.setup(stage="test")
    trainer.test(model=task, dataloaders=datamodule)