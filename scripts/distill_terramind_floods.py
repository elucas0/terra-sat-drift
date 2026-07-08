import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from terratorch.tasks import SemanticSegmentationTask
from terratorch.datamodules import GenericNonGeoSegmentationDataModule

from lightning.pytorch import Trainer, LightningModule
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Define the Distillation Task
class KnowledgeDistillationSegmentationTask(LightningModule):
    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        lr: float = 1e-4,
        temperature: float = 3.0,
        alpha: float = 0.5,
        ignore_index: int = -1,
    ):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.lr = lr
        self.temperature = temperature
        self.alpha = alpha
        self.ignore_index = ignore_index
        
        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

    def training_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["mask"]

        with torch.no_grad():
            teacher_logits = self.teacher(x)
        
        student_logits = self.student(x)

        # Standard Cross Entropy Loss with ground truth
        loss_ce = F.cross_entropy(student_logits, y, ignore_index=self.ignore_index)

        # Distillation Loss (KL Divergence)
        # Soften probabilities with temperature
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        student_log_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        
        loss_kd = F.kl_div(student_log_soft, teacher_soft, reduction='batchmean') * (self.temperature ** 2)

        loss = self.alpha * loss_ce + (1 - self.alpha) * loss_kd

        self.log("train/loss", loss, prog_bar=True)
        self.log("train/loss_ce", loss_ce)
        self.log("train/loss_kd", loss_kd)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["mask"]
        
        logits = self.student(x)
        loss = F.cross_entropy(logits, y, ignore_index=self.ignore_index)
        
        # Simple IoU calculation for logging
        preds = torch.argmax(logits, dim=1)
        valid_mask = (y != self.ignore_index)
        iou = self._calculate_iou(preds[valid_mask], y[valid_mask])
        
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/mIoU", iou, prog_bar=True)
        return loss

    def _calculate_iou(self, preds, target):
        if target.numel() == 0: return 0.0
        ious = []
        for cls in range(self.student.num_classes if hasattr(self.student, 'num_classes') else 2):
            intersection = ((preds == cls) & (target == cls)).sum().float()
            union = ((preds == cls) | (target == cls)).sum().float()
            if union > 0:
                ious.append(intersection / union)
        return torch.tensor(ious).mean() if ious else torch.tensor(0.0)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.student.parameters(), lr=self.lr)

if __name__ == "__main__":
    # Parameters
    root_dir = Path("/shared/home/elucas/datasets/sen1floods11_simulated_alt_v1")
    label_root = Path("/shared/home/elucas/datasets/sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand")
    TARGET_SIZE = (1024, 1024) 
    batch_size = 4
    
    # Load stats
    stats_file = root_dir / "v1.1/spectral_statistics.json"
    with open(stats_file, 'r') as f:
        stats = json.load(f)
    means = [stats[f"band_{i+1}"]["mean"] for i in range(8)]
    stds = [stats[f"band_{i+1}"]["std"] for i in range(8)]

    def preprocess_mask(mask, **kwargs):
        clean_mask = np.full(mask.shape, -1, dtype=np.int64)
        clean_mask[mask == 0] = 0
        clean_mask[mask == 1] = 1
        return clean_mask

    transform = A.Compose([
        A.Resize(width=TARGET_SIZE[0], height=TARGET_SIZE[1], interpolation=cv2.INTER_NEAREST),
        A.Lambda(mask=preprocess_mask),
        ToTensorV2(),
    ])

    datamodule = GenericNonGeoSegmentationDataModule(
        batch_size=batch_size,
        data_root=root_dir,
        train_data_root=root_dir / "v1.1/data/flood_events/HandLabeled/S2Hand",
        train_label_data_root=label_root,
        val_data_root=root_dir / "v1.1/data/flood_events/HandLabeled/S2Hand",
        val_label_data_root=label_root,
        test_data_root=root_dir / "v1.1/data/flood_events/HandLabeled/S2Hand",
        test_label_data_root=label_root,
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
        num_classes=2,
    )

    # Teacher Model Setup (TerraMind Base)
    teacher_model_args = {
        "backbone": "terramind_v1_base",
        "backbone_pretrained": True,
        "backbone_modalities": ["S2L1C"],
        "backbone_bands": {"S2L1C": {"B02": 0, "B03": 1, "B04": 2, "PAN": 3, "B08": 4, "B05": 5, "B06": 6, "B07": 7}},
        "necks": [
            {"name": "SelectIndices", "indices": [2, 5, 8, 11]},
            {"name": "ReshapeTokensToImage", "remove_cls_token": False},
            {"name": "LearnedInterpolateToPyramidal"},
        ],
        "decoder": "UNetDecoder",
        "decoder_channels": [256, 128, 64, 32],
        "num_classes": 2,
    }

    teacher_task = SemanticSegmentationTask(
        model_factory="EncoderDecoderFactory",
        model_args=teacher_model_args,
    )
    # Note: User might want to load specific weights here
    # teacher_task = SemanticSegmentationTask.load_from_checkpoint("path/to/checkpoint")

    # Student Model Setup (MobileNetV2)
    student_model_args = {
        "backbone": "mobilenetv2_100",
        "backbone_pretrained": True,
        "backbone_indices": [1, 2, 3, 5], # Adjust based on mobilenetv2 layers
        "decoder": "UNetDecoder",
        "decoder_channels": [128, 64, 32, 16],
        "num_classes": 2,
    }
    
    student_task = SemanticSegmentationTask(
        model_factory="EncoderDecoderFactory",
        model_args=student_model_args,
    )

    # Distillation Wrapper
    distill_task = KnowledgeDistillationSegmentationTask(
        teacher=teacher_task.model,
        student=student_task.model,
        lr=1e-4,
        temperature=3.0,
        alpha=0.4
    )

    experiment_name = "distillation_terramind_to_mobilenet_simulated_alt_v1"
    logger = WandbLogger(project="terra-sat-drift", name=experiment_name)

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"outputs/{experiment_name}",
        filename="best-val_mIoU",
        monitor="val/mIoU",
        mode="max",
        save_top_k=1,
    )

    trainer = Trainer(
        max_epochs=50,
        callbacks=[checkpoint_callback, EarlyStopping(monitor="val/mIoU", patience=10, mode="max")],
        logger=logger,
        devices=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
    )

    trainer.fit(distill_task, datamodule=datamodule)
