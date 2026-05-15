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
from lightning.pytorch.utilities.combined_loader import CombinedLoader

import cv2
import albumentations as A


from terra_sat_drift.data_simulation.phisat2_constants import S2_BANDS_NAMES, S2_BANDS, S2_PAN_BANDS

class MMDLoss(torch.nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma

    def gaussian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(source.size(0)) + int(target.size(0))
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bw) for bw in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size(0))
        kernels = self.gaussian_kernel(source, target, self.kernel_mul, self.kernel_num, self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        return torch.mean(XX + YY - 2 * XY)

class DomainAdaptationTask(SemanticSegmentationTask):
    def __init__(self, mmd_weight=0.1, **kwargs):
        super().__init__(**kwargs)
        self.mmd_loss_fn = MMDLoss()
        self.mmd_weight = mmd_weight

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        source_batch = batch["source"]
        target_batch = batch["target"]

        # 1. Sélection des données
        # Source (S2 Clean) : Déjà 7 bandes
        x_source = source_batch["image"] 
        
        # Target (Simulated) : On retire la PAN (index 0)
        # Les bandes 1 à 7 correspondent aux multispectrales de S2
        x_target = target_batch["image"][:, 1:, :, :] 
        y_target = target_batch["mask"]

        # 2. Extraction des caractéristiques via le backbone
        # On récupère le token [CLS] (index 0) de la dernière couche
        feat_source = self.model.model.backbone(x_source)[-1][:, 0, :]
        feat_target = self.model.model.backbone(x_target)[-1][:, 0, :]

        # 3. Calcul des pertes
        y_hat_target = self.forward(x_target)
        seg_loss = self.loss(y_hat_target, y_target)
        
        # Alignement des domaines sur les 7 bandes multispectrales
        mmd_dist = self.mmd_loss_fn(feat_source, feat_target)

        total_loss = seg_loss + self.mmd_weight * mmd_dist

        self.log("train/mmd_dist", mmd_dist)
        return total_loss

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
    TARGET_SIZE = (1078, 1078)

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
    
    dm_source = Sen1Floods11NonGeoDataModule(
        data_root="/shared/home/elucas/datasets/sen1floods11",
        bands=S2_BANDS_NAMES, # 7 bandes
        train_transform=transform,
        val_transform=transform,
        test_transform=transform,
        batch_size=8, num_workers=4
    )

    dm_target = GenericNonGeoSegmentationDataModule(
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

    task = DomainAdaptationTask(
        model_factory="EncoderDecoderFactory",
        model_args=model_args,
        mmd_weight=0.2,
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

    
    experiment_name = f"terramind_v1_{backbone_size}_simulated"
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
    )
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    dm_source.setup("fit")
    dm_target.setup("fit")
    
    combined_loader = CombinedLoader(
        {"source": dm_source.train_dataloader(), "target": dm_target.train_dataloader()},
        mode="min_size" # S'arrête quand le plus petit dataset est épuisé
    )
    
    trainer.fit(model=task, train_dataloaders=combined_loader, val_dataloaders=dm_target.val_dataloader())
    # trainer.test(model=task, dataloaders=datamodule)