"""Training utilities for fine-tuning TerraMind on segmentation datasets."""

from typing import Optional, Dict, Any
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from terratorch import BACKBONE_REGISTRY
from terratorch.tasks.segmentation_tasks import SemanticSegmentationTask


class TerraMindTrainer:
    """Fine-tune TerraMind backbone on custom segmentation datasets.
    
    Manages model instantiation, training setup, freezing/unfreezing components,
    and checkpoint management for SemanticSegmentationTask.
    """

    BAND_NAMES = ["B02", "B03", "B04", "B08", "B05", "B06", "B07"]
    BACKBONE_SIZES = {"tiny", "small", "base", "large"}
    BACKBONE_NECK_INDICES = {
        "tiny": [1, 3, 4, 5],
        "small": [1, 3, 4, 5],
        "base": [2, 5, 8, 11],
        "large": [5, 11, 17, 23],
    }

    def __init__(
        self,
        num_classes: int = 2,
        backbone_size: str = "base",
        device: torch.device | None = None,
        class_names: list[str] | None = None,
        learning_rate: float = 0.001,
        freeze_backbone: bool = False,
        freeze_decoder: bool = False,
    ) -> None:
        """Initialize TerraMindTrainer for fine-tuning.

        Args:
            num_classes: Number of output classes for segmentation (default: 2 for binary).
            backbone_size: TerraMind model size: 'tiny', 'small', 'base', or 'large'.
            device: Optional torch device. If omitted, CUDA is used when available.
            class_names: Optional list of class names for logging.
            learning_rate: Learning rate for optimizer.
            freeze_backbone: Freeze the backbone during training (feature extraction mode).
            freeze_decoder: Freeze the decoder during training.

        Raises:
            ValueError: If backbone_size is not in BACKBONE_SIZES.
        """
        if backbone_size not in self.BACKBONE_SIZES:
            raise ValueError(
                f"backbone_size must be one of {self.BACKBONE_SIZES}, got {backbone_size!r}"
            )
        
        self.num_classes = num_classes
        self.backbone_size = backbone_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.learning_rate = learning_rate
        self.freeze_backbone = freeze_backbone
        self.freeze_decoder = freeze_decoder
        
        self.model = self._build_model()

    def _build_model(self) -> SemanticSegmentationTask:
        """Create and configure the TerraTorch SemanticSegmentationTask model for training.
        
        Instantiates the model in training mode with proper configuration for fine-tuning.
        """
        backbone_name = f"terramind_v1_{self.backbone_size}"
        neck_indices = self.BACKBONE_NECK_INDICES[self.backbone_size]

        # Build TerraMind backbone with pretrained weights
        terramind_backbone = BACKBONE_REGISTRY.build(
            backbone_name,
            pretrained=True,
            modalities=["S2L1C"],
            bands={"S2L1C": self.BAND_NAMES},
        )
        if terramind_backbone is None:
            raise RuntimeError(f"Failed to initialize TerraMind backbone: {backbone_name}")
        terramind_backbone.to(self.device)

        model_args = {
            "backbone": terramind_backbone,
            "backbone_pretrained": True,
            "backbone_modalities": ["S2L1C"],
            "backbone_bands": {"S2L1C": self.BAND_NAMES},
            "decoder": "UNetDecoder",
            "decoder_channels": [256, 128, 64, 32],
            "necks": [
                {"name": "SelectIndices", "indices": neck_indices},
                {"name": "ReshapeTokensToImage", "remove_cls_token": False},
                {"name": "LearnedInterpolateToPyramidal"},
            ],
            "num_classes": self.num_classes,
        }

        # Create task with training settings
        model = SemanticSegmentationTask(
            model_factory="EncoderDecoderFactory",
            model_args=model_args,
            class_names=self.class_names,
            lr=self.learning_rate,
            freeze_backbone=self.freeze_backbone,
            freeze_decoder=self.freeze_decoder,
        )
        model.to(self.device)
        model.train()
        return model

    def get_model(self) -> SemanticSegmentationTask:
        """Get the underlying SemanticSegmentationTask model.
        
        Returns:
            SemanticSegmentationTask instance
        """
        return self.model

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters (feature extraction mode)."""
        if hasattr(self.model.model, "backbone"):
            for param in self.model.model.backbone.parameters():
                param.requires_grad = False
        elif hasattr(self.model, "backbone"):
            for param in self.model.backbone.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters (fine-tuning mode)."""
        if hasattr(self.model.model, "backbone"):
            for param in self.model.model.backbone.parameters():
                param.requires_grad = True
        elif hasattr(self.model, "backbone"):
            for param in self.model.backbone.parameters():
                param.requires_grad = True

    def freeze_decoder(self) -> None:
        """Freeze all decoder parameters."""
        if hasattr(self.model.model, "decoder"):
            for param in self.model.model.decoder.parameters():
                param.requires_grad = False
        elif hasattr(self.model, "decoder"):
            for param in self.model.decoder.parameters():
                param.requires_grad = False

    def unfreeze_decoder(self) -> None:
        """Unfreeze all decoder parameters."""
        if hasattr(self.model.model, "decoder"):
            for param in self.model.model.decoder.parameters():
                param.requires_grad = True
        elif hasattr(self.model, "decoder"):
            for param in self.model.decoder.parameters():
                param.requires_grad = True

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters.
        
        Args:
            trainable_only: If True, count only trainable parameters
            
        Returns:
            Number of parameters
        """
        if trainable_only:
            return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.model.parameters())

    def save_checkpoint(self, path: Path | str) -> None:
        """Save model checkpoint.
        
        Args:
            path: Path to save checkpoint to
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load_checkpoint(self, path: Path | str) -> None:
        """Load model checkpoint.
        
        Args:
            path: Path to checkpoint file
        """
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)

    def get_optimizer(
        self,
        optimizer_name: str = "adam",
        **kwargs: Any,
    ) -> torch.optim.Optimizer:
        """Create optimizer for model parameters.
        
        Args:
            optimizer_name: Name of optimizer ('adam', 'sgd', 'adamw')
            **kwargs: Additional arguments passed to optimizer
            
        Returns:
            Optimizer instance
        """
        default_kwargs = {"lr": self.learning_rate}
        default_kwargs.update(kwargs)
        
        if optimizer_name.lower() == "adam":
            return torch.optim.Adam(self.model.parameters(), **default_kwargs)
        elif optimizer_name.lower() == "adamw":
            return torch.optim.AdamW(self.model.parameters(), **default_kwargs)
        elif optimizer_name.lower() == "sgd":
            return torch.optim.SGD(self.model.parameters(), **default_kwargs)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

    def get_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler_name: str = "cosine",
        **kwargs: Any,
    ) -> torch.optim.lr_scheduler.LRScheduler:
        """Create learning rate scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            scheduler_name: Name of scheduler ('cosine', 'step', 'exponential')
            **kwargs: Additional arguments passed to scheduler
            
        Returns:
            LR scheduler instance
        """
        if scheduler_name.lower() == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
        elif scheduler_name.lower() == "step":
            return torch.optim.lr_scheduler.StepLR(optimizer, **kwargs)
        elif scheduler_name.lower() == "exponential":
            return torch.optim.lr_scheduler.ExponentialLR(optimizer, **kwargs)
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")
