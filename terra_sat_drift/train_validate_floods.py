import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

from terratorch.tasks import SemanticSegmentationTask
from terratorch import BACKBONE_REGISTRY
from terratorch.datamodules.sen1floods11 import Sen1Floods11NonGeoDataModule

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger

from data_simulation.phisat2_constants import S2_BANDS_NAMES, S2_BANDS, S2_PAN_BANDS


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

root_dir="/shared/home/elucas/datasets/sen1floods11"

# Random crop for training
# TODO: simulated settings, change them to base one
train_transform = [
    A.RandomCrop(width=TARGET_SIZE[0], height=TARGET_SIZE[1]),
    # A.Normalize(
    #     mean=S2L1C_means, 
    #     std=S2L1C_stds, 
    #     max_pixel_value=10000.0 
    # ),
    A.Lambda(mask=preprocess_mask),
    A.pytorch.ToTensorV2(),
]

# Center crop for deterministic validation/testing
val_test_transform = [
    A.Resize(width=1077, height=1077, interpolation=cv2.INTER_NEAREST),
    A.CenterCrop(width=TARGET_SIZE[0], height=TARGET_SIZE[1]),
    # A.Normalize(
    #     mean=S2L1C_means, 
    #     std=S2L1C_stds, 
    #     max_pixel_value=10000.0 
    # ),
    A.Lambda(mask=preprocess_mask),
    A.pytorch.ToTensorV2(),
]

datamodule = Sen1Floods11NonGeoDataModule(
    batch_size=8,
    data_root=root_dir,
    bands=S2_BANDS_NAMES,
    train_transform=train_transform,
    val_transform=val_test_transform,
    test_transform=val_test_transform,
    num_workers=4,
    download=False,
    use_metadata=True,
    rgb_indices=[3, 2, 1],
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

backbone_size = "base"
backbone_name = f"terramind_v1_{backbone_size}"
neck_indices = BACKBONE_NECK_INDICES[backbone_size]

model_args = {
    "backbone": backbone_name,
    "backbone_pretrained": True,
    "backbone_modalities": ["S2L1C"],
    "backbone_bands": {"S2L1C": S2_BANDS},
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

#task = SemanticSegmentationTask.load_from_checkpoint(checkpoint_path="/scratch/elucas/terra-sat-drift/outputs/terramind_sen1floods/terramind_v1_base/checkpoints/best-val_mIoU.ckpt")

experiment_name = f"terramind_v1_{backbone_size}"
output_dir="/shared/home/elucas/scratch/terra-sat-drift/outputs/terramind_sen1floods"
logger = WandbLogger(project="terra-sat-drift", name=experiment_name, save_dir=root_dir, offline=True)

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
    patience=10,
    mode="max",
    verbose=True,
)

trainer = Trainer(
    max_epochs=50,
    callbacks=[checkpoint_callback, early_stopping],
    logger=logger,
    log_every_n_steps=10,
    enable_progress_bar=True,
    accelerator="gpu", 
    devices=1
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

datamodule.setup(stage="fit")
trainer.fit(model=task, datamodule=datamodule)

datamodule.setup(stage="test")
trainer.test(model=task, dataloaders=datamodule)