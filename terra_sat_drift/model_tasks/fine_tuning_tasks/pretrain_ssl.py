import os
import torch
import numpy as np
import h5py
import pandas as pd
import logging
import argparse
import json
import sys
import math
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import lightning.pytorch as pl
import torch.nn.functional as F
from torch import nn
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger

import cv2
import albumentations as A
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl


from  terratorch.models.encoder_decoder_factory import EncoderDecoderFactory


from dataset.datamodule_triplets_lulc import PhisatRealLULCDataModule
from dataset.constants import WC_CLASS_MAPPING


def _strip_compile_prefix(state_dict):
    """Removes the '_orig_mod.' prefix if the model was saved with torch.compile()"""
    return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

class SSLPretrainModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        in_channels: int = 7,
        lr: float = 1e-4,
        weight_decay: float = 0.05,
        patch_size: int = 16,
        mask_ratio: float = 0.6,
        masking_strategy: str = "block",   # "random" | "block"
    ) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.masking_strategy = masking_strategy

        print(
            f"SSLPretrainModule | patch_size={patch_size} "
            f"| mask_ratio={mask_ratio} | masking={masking_strategy}"
        )

        self.mask_token = nn.Parameter(torch.zeros(1, in_channels, 1, 1))
        self.save_hyperparameters(ignore=["model"])

    def forward(self, image_dict: dict) -> torch.Tensor:
        out = self.model(image_dict)
        # Handle TerraTorch return dictionaries gracefully
        if isinstance(out, dict):
            return out.get('out', list(out.values())[0])
        return out

    def generate_mask(self, batch_size: int, h: int, w: int, device: torch.device) -> torch.Tensor:
        """Return a (B, 1, H, W) binary mask — 1 = masked, 0 = visible."""
        if self.masking_strategy == "block":
            return self._block_mask(batch_size, h, w, device)
        return self._random_mask(batch_size, h, w, device)

    def _random_mask(self, batch_size: int, h: int, w: int, device: torch.device) -> torch.Tensor:
        grid_h, grid_w = h // self.patch_size, w // self.patch_size
        num_patches = grid_h * grid_w
        num_masked  = int(num_patches * self.mask_ratio)

        noise       = torch.rand(batch_size, num_patches, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)

        binary_mask = torch.zeros(batch_size, num_patches, device=device)
        binary_mask.scatter_(1, ids_shuffle[:, :num_masked], 1.0)

        binary_mask = binary_mask.view(batch_size, 1, grid_h, grid_w)
        return F.interpolate(binary_mask, size=(h, w), mode="nearest")

    def _block_mask(self, batch_size: int, h: int, w: int, device: torch.device) -> torch.Tensor:
        grid_h  = h // self.patch_size
        grid_w  = w // self.patch_size
        target  = int(grid_h * grid_w * self.mask_ratio)
        max_bh  = max(2, grid_h // 3)
        max_bw  = max(2, grid_w // 3)

        masks = []
        for _ in range(batch_size):
            m = torch.zeros(grid_h, grid_w)
            for _ in range(200):
                bh   = torch.randint(1, max_bh + 1, (1,)).item()
                bw   = torch.randint(1, max_bw + 1, (1,)).item()
                top  = torch.randint(0, grid_h - bh + 1, (1,)).item()
                left = torch.randint(0, grid_w - bw + 1, (1,)).item()
                m[top: top + bh, left: left + bw] = 1.0
                if m.sum() >= target:
                    break
            masks.append(m)

        binary_mask = torch.stack(masks).unsqueeze(1).to(device)
        return F.interpolate(binary_mask, size=(h, w), mode="nearest")

    @staticmethod
    def _masked_band_loss(reconstruction, target, mask):
        loss_map   = F.mse_loss(reconstruction, target, reduction="none")  
        masked_sum = (loss_map * mask).sum()
        n_elements = mask.sum() * target.shape[1]
        return masked_sum / (n_elements + 1e-8)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss, _, _, _ = self._shared_step(batch, "train")
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        loss, image, masked_image, reconstruction = self._shared_step(batch, "val")
        if batch_idx == 0:
            self._visualize_and_save(image, masked_image, reconstruction, self.current_epoch)

    def test_step(self, batch: dict, batch_idx: int) -> None:
        loss, image, masked_image, reconstruction = self._shared_step(batch, "test")
        if batch_idx == 0:
            self._visualize_and_save(image, masked_image, reconstruction, "test")

    def _shared_step(self, batch: dict, prefix: str) -> tuple:
        # Generic multi-modal mapping: Extract the explicit Modality Tensor
        image = batch["image"]["S2L1C"]
        B, C, H, W = image.shape

        mask           = self.generate_mask(B, H, W, image.device)
        masked_image   = image * (1 - mask) + self.mask_token * mask
        
        # TerraTorch backbone expects a Dict input
        reconstruction = self({"S2L1C": masked_image})

        loss = self._masked_band_loss(reconstruction, image, mask)
        self.log(
            f"{prefix}/loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=B,
        )
        return loss, image, masked_image, reconstruction

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    def on_load_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["state_dict"] = _strip_compile_prefix(
            checkpoint.get("state_dict", {})
        )

    @staticmethod
    def _percentile_stretch(
        t: torch.Tensor, ref_t: torch.Tensor = None, lo: float = 2.0, hi: float = 98.0
    ) -> np.ndarray:
        if ref_t is None:
            ref_t = t
            
        flat = ref_t.reshape(-1).float()
        v_lo = torch.quantile(flat, lo / 100.0)
        v_hi = torch.quantile(flat, hi / 100.0)
        
        return ((t.float() - v_lo) / (v_hi - v_lo + 1e-6)).clamp(0, 1).numpy()

    @staticmethod
    def _to_falsecolor(
        t: torch.Tensor,
        ref_t: torch.Tensor = None,
        rgb_idx: tuple[int, int, int] = (2, 1, 0), # B04(R), B03(G), B02(B)
    ) -> np.ndarray:
        if ref_t is None:
            ref_t = t
            
        C = t.shape[0]
        idx = [c for c in rgb_idx if c < C]
        
        if len(idx) < 3:
            gray = SSLPretrainModule._percentile_stretch(t[0], ref_t[0])
            return np.stack([gray, gray, gray], axis=-1)
            
        channels = [SSLPretrainModule._percentile_stretch(t[c], ref_t[c]) for c in idx]
        return np.stack(channels, axis=-1)

    def _visualize_and_save(
        self,
        image: torch.Tensor,
        masked_image: torch.Tensor,
        reconstruction: torch.Tensor,
        epoch_idx,
        max_samples: int = 5,
    ) -> None:
        
        n = min(max_samples, image.shape[0])
        fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n), squeeze=False)
        fig.suptitle(
            f"Reconstruction Epoch {epoch_idx}\n"
            f"strategy={self.masking_strategy}   ratio={self.mask_ratio}",
            fontsize=14,
        )
        
        col_titles = ["Original", "Masked Input", "Reconstruction"]
        for ax, title in zip(axes[0], col_titles):
            ax.set_title(title, fontsize=11)

        for i in range(n):
            orig  = image[i].detach().cpu()
            mskd  = masked_image[i].detach().cpu()
            recon = reconstruction[i].detach().cpu()

            axes[i, 0].imshow(self._to_falsecolor(orig, ref_t=orig))
            axes[i, 1].imshow(self._to_falsecolor(mskd, ref_t=orig))
            axes[i, 2].imshow(self._to_falsecolor(recon, ref_t=orig))

            for ax in axes[i]:
                ax.axis("off")

        save_dir = Path(self.logger.save_dir) if self.logger else Path(self.trainer.default_root_dir)
        os.makedirs(save_dir / "reconstructions", exist_ok=True)
        path = save_dir / "reconstructions" / f"epoch_{epoch_idx}.png"
        
        plt.tight_layout()
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)

# ------------------------------------------------------------------ #
#  Main Routine                                                        #
# ------------------------------------------------------------------ #

def main(): 
    parser = argparse.ArgumentParser(description="SSL Pre-training TerraMind on Real PhiSat-2 data")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--backbone", type=str, default="terramind_v1_tiny")
    parser.add_argument("--mask-ratio", type=float, default=0.6)
    parser.add_argument("--mask-strategy", type=str, default="block", choices=["random", "block"])
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    seed_everything(42)

    # Paths
    h5_images_path = "/shared/projects/phisat2/data/processed/triplets_v1/phisat2_s2b_dataset_v1.h5"
    h5_labels_path = "/shared/projects/phisat2/data/processed/worldcover_all_clean_v1/worldcover_all_clean_labels_v1.h5"
    manifest_path = "/shared/projects/phisat2/data/processed/worldcover_all_clean_v1/worldcover_all_clean_manifest_v1.csv"
    
    output_dir = Path(f"/shared/home/elucas/scratch/terra-sat-drift/outputs/{args.backbone}_ssl_pretrain_real")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sentinel-2 L1C global pre-training statistics expected by TerraMind
    S2L1C_MEANS = (2137.385, 2018.788, 2082.986, 2295.651, 2854.537, 3122.849, 3040.560)
    S2L1C_STDS = (1675.806, 1557.708, 1833.702, 1823.738, 1733.977, 1732.131, 1679.732)


    # Note: Images are natively 256x256 in the H5 file.
    train_transform = A.Compose([
        A.Normalize(
            mean=S2L1C_MEANS, 
            std=S2L1C_STDS,
            max_pixel_value=1.0 
        ), 
        A.pytorch.ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Normalize(
            mean=S2L1C_MEANS, 
            std=S2L1C_STDS,
            max_pixel_value=1.0 
        ), 
        A.pytorch.ToTensorV2()
    ])

    datamodule = PhisatRealLULCDataModule(
        h5_images_path=h5_images_path,
        h5_labels_path=h5_labels_path,
        manifest_path=manifest_path,
        batch_size=args.batch_size,
        num_workers=4,
        train_transform=train_transform,
        val_transform=val_transform,
        max_samples=args.max_samples,
    )

    BACKBONE_NECK_INDICES = {
        "tiny": [1, 3, 4, 5],
        "small": [1, 3, 4, 5],
        "base": [2, 5, 8, 11],
    }
    
    SIM_BACKBONE_BANDS = {
        "S2L1C": {
            "B02": 0, "B03": 1, "B04": 2, 
            "B05": 3, "B06": 4, "B07": 5, "B08": 6
        }
    }

    model_args = {
        "backbone": args.backbone,
        "backbone_pretrained": True,
        "backbone_modalities": ["S2L1C"],
        "backbone_bands": SIM_BACKBONE_BANDS,
        "necks": [
            {"name": "SelectIndices", "indices": BACKBONE_NECK_INDICES["tiny"]},
            {"name": "ReshapeTokensToImage", "remove_cls_token": False},
            {"name": "LearnedInterpolateToPyramidal"},
        ],
        "decoder": "UNetDecoder",
        "decoder_channels": [256, 128, 64, 32],
        "head_dropout": 0.1,
        # CRITICAL SSL PARAMETER: Reconstruct 7 bands instead of doing segmentation (num_classes=11)
        "num_classes": 7, 
    }

    # Build the model using TerraTorch factory (Segmenter acts as a dense pixel-wise regressor)
    model = EncoderDecoderFactory().build_model(
        task="segmentation", 
        **model_args
    )

    # Initialize the custom SSL Module wrapper
    task = SSLPretrainModule(
        model=model,
        in_channels=7,
        lr=args.lr,
        mask_ratio=args.mask_ratio,
        masking_strategy=args.mask_strategy
    )

    logger = WandbLogger(project="encoder-lulc", name=f"ssl_pretrain_real_{args.backbone}")
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="best-val_loss",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
    )

    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback],
        precision="16-mixed"
    )

    trainer.fit(task, datamodule=datamodule)
    
    # Run test pipeline after training
    datamodule.setup(stage="test")
    trainer.test(task, datamodule=datamodule)

if __name__ == "__main__":
    main()