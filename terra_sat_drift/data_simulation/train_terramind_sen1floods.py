"""End-to-end training script for TerraMind on Sen1Floods11 with Phisat-2 augmentation."""

import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from terratorch.datamodules.sen1floods11 import Sen1Floods11NonGeoDataModule
from terratorch.tasks import SemanticSegmentationTask
from terratorch import BACKBONE_REGISTRY

from phisat2_constants import S2_BANDS_NAMES, S2_BANDS

from phisat2_albumentations import create_phisat2_transform

logger = logging.getLogger(__name__)

BACKBONE_SIZES = {"tiny", "small", "base", "large"}
BACKBONE_NECK_INDICES = {
    "tiny": [1, 3, 4, 5],
    "small": [1, 3, 4, 5],
    "base": [2, 5, 8, 11],
    "large": [5, 11, 17, 23],
}


class TerraMindSen1FloodsTrainer:
    """Orchestrator for TerraMind fine-tuning on Sen1Floods11 with staged training."""

    def __init__(
        self,
        backbone_size: str = "small",
        learning_rate: float = 1e-4,
        batch_size: int = 16,
        max_epochs: int = 50,
        output_dir: str = "./outputs",
        num_workers: int = 4,
        use_amp: bool = True,
    ):
        """Initialize trainer.
        
        Args:
            backbone_size: ["tiny", "small", "base", "large"]
            learning_rate: Learning rate for fine-tuning
            batch_size: Batch size for training
            max_epochs: Maximum training epochs
            output_dir: Directory for checkpoints and logs
            num_workers: Number of data loading workers
            use_amp: Whether to use automatic mixed precision
        """
        self.backbone_size = backbone_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.output_dir = Path(output_dir)
        self.num_workers = num_workers
        self.use_amp = use_amp
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_datamodule(
        self,
        root_dir: str,
        phisat_config: Dict[str, Any] | None = None,
    ) -> Sen1Floods11NonGeoDataModule:
        """Create Sen1Floods11 datamodule with optional Phisat-2 augmentation.
        
        Args:
            root_dir: Path to Sen1Floods11 dataset root
            phisat_config: Configuration for Phisat-2 transforms
            
        Returns:
            Configured Sen1Floods11NonGeoDataModule
        """
        phisat_transform = None
        if phisat_config:
            phisat_transform = create_phisat2_transform(phisat_config)
            
        return Sen1Floods11NonGeoDataModule(
            data_root=root_dir,
            bands=S2_BANDS_NAMES,
            train_transform=phisat_transform,
            val_transform=phisat_transform,
            test_transform=phisat_transform,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            download=False,
            use_metadata=True
        )

    def create_model(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
    ) -> SemanticSegmentationTask:
        """Create SemanticSegmentationTask with TerraMind backbone.
        
        Args:
            num_classes: Number of output classes
            pretrained: Whether to load pretrained TerraMind weights
            
        Returns:
            Configured SemanticSegmentationTask
        """
        
        backbone_name = f"terramind_v1_{self.backbone_size}"
        neck_indices = BACKBONE_NECK_INDICES[self.backbone_size]

        model_args = {
            "backbone": backbone_name,
            "backbone_pretrained": pretrained,
            "backbone_modalities": ["S2L1C"],
            "backbone_bands": {"S2L1C": S2_BANDS},
            "decoder": "UNetDecoder",
            "decoder_channels": [256, 128, 64, 32],
            "necks": [
                {"name": "SelectIndices", "indices": neck_indices},
                {"name": "ReshapeTokensToImage", "remove_cls_token": False},
                {"name": "LearnedInterpolateToPyramidal"},
            ],
            "num_classes": num_classes,
        }
        
        return SemanticSegmentationTask(
            model_factory="EncoderDecoderFactory",
            model_args=model_args,
            lr=self.learning_rate,
            ignore_index=-1,
            plot_on_val=0,
            freeze_backbone=True,  # Start with frozen backbone for stage 1
        )

    def train(
        self,
        datamodule: Sen1Floods11NonGeoDataModule,
        model: SemanticSegmentationTask,
        stage_1_epochs: int = 15,
        disable_stage_2: bool = True,
    ) -> Tuple[Trainer, SemanticSegmentationTask]:
        """Execute staged fine-tuning on Sen1Floods11.
        
        Stage 1: Train head/decoder only (backbone frozen)
        Stage 2: Fine-tune entire model (backbone unfrozen)
        
        Args:
            datamodule: Configured datamodule
            model: SemanticSegmentationTask to train
            stage_1_epochs: Number of epochs for stage 1 (frozen backbone)
            disable_stage_2: If True, skip stage 2 (full fine-tuning)
            
        Returns:
            Trained model
        """
        logger.info("Starting staged fine-tuning on Sen1Floods11")

        # Stage 1: Train only head/decoder (backbone frozen)
        logger.info(f"Stage 1: Training head (epochs 0-{stage_1_epochs})")
        self._freeze_backbone(model)
                
        trainer_stage1 = self._create_trainer(
            stage=1,
            max_epochs=stage_1_epochs,
        )
        trainer_stage1.fit(model, datamodule)
        
        # if not disable_stage_2:
        #     # Stage 2: Fine-tune entire model (backbone unfrozen)
        #     logger.info(
        #         f"Stage 2: Fine-tuning full model (epochs {stage_1_epochs}-{self._max_epochs})"
        #     )
        #     self._unfreeze_backbone(model)
            
        #     trainer_stage2 = self._create_trainer(
        #         stage=2,
        #         max_epochs=self._max_epochs - stage_1_epochs,
        #     )
        #     trainer_stage2.fit(model, datamodule=datamodule, ckpt_path="last")

        return trainer_stage1, model

    def _freeze_backbone(self, model: SemanticSegmentationTask) -> None:
        """Freeze all backbone parameters."""
        if hasattr(model, "backbone"):
            for param in model.backbone.parameters():
                param.requires_grad = False
            logger.info("Backbone frozen")

    def _unfreeze_backbone(self, model: SemanticSegmentationTask) -> None:
        """Unfreeze all backbone parameters."""
        if hasattr(model, "backbone"):
            for param in model.backbone.parameters():
                param.requires_grad = True
            logger.info("Backbone unfrozen")

    def _create_trainer(
        self,
        stage: int,
        max_epochs: int,
    ) -> Trainer:
        """Create PyTorch Lightning trainer with callbacks.
        
        Args:
            stage: Training stage (1 or 2)
            max_epochs: Maximum epochs for this stage
            
        Returns:
            Configured pl.Trainer
        """
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.output_dir / f"checkpoints_stage{stage}",
            filename="best-val_mIoU",
            monitor="val/mIoU",
            mode="max",
            save_top_k=3,
            verbose=True,
        )

        early_stopping = EarlyStopping(
            monitor="val/mIoU",
            patience=5,
            mode="max",
            verbose=True,
        )

        logger_tb = TensorBoardLogger(
            save_dir=self.output_dir,
            name=f"stage{stage}",
            version=0,
        )

        return Trainer(
            max_epochs=max_epochs,
            callbacks=[checkpoint_callback, early_stopping],
            logger=logger_tb,
            log_every_n_steps=10,
            enable_progress_bar=True,
        )


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(
        description="Train TerraMind on Sen1Floods11 with optional Phisat-2 augmentation"
    )
    
    parser.add_argument(
        "--data_root",
        type=str,
        default="datasets/sen1floods11",
        help="Path to Sen1Floods11 dataset root directory",
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/terramind_sen1floods",
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--backbone_size",
        type=str,
        default="small",
        choices=["tiny", "small", "base", "large"],
        help="TerraMind backbone size",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Maximum training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for fine-tuning",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--no_amp",
        action="store_true",
        help="Disable automatic mixed precision",
    )
    
    # Phisat-2 augmentation arguments
    parser.add_argument(
        "--apply_band_misalignment",
        action="store_true",
        help="Apply band misalignment transform",
    )
    parser.add_argument(
        "--apply_pan_band",
        action="store_true",
        help="Create panchromatic band",
    )
    parser.add_argument(
        "--apply_psf",
        action="store_true",
        help="Apply PSF kernel convolution",
    )
    parser.add_argument(
        "--apply_snr",
        action="store_true",
        help="Apply SNR noise",
    )
    parser.add_argument(
        "--processing_level",
        type=str,
        default="L1A",
        choices=["L1A", "L1B"],
        help="Processing level for band misalignment",
    )
    
    # External executable arguments
    parser.add_argument(
        "--psf_executable",
        type=str,
        default="./executables/phisat2_unix.bin",
        help="Path to PSF executable binary (optional)",
    )
    parser.add_argument(
        "--snr_executable",
        type=str,
        default=None,
        help="Path to SNR executable binary (optional)",
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Create trainer
    trainer = TerraMindSen1FloodsTrainer(
        backbone_size=args.backbone_size,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        use_amp=not args.no_amp,
    )
    
    # Build phisat configuration
    phisat_config = {
        "apply_radiance_calculation": True,
        "apply_band_misalignment": True,
        "apply_pan_band": True,
        "apply_psf": True,
        "apply_snr": True,
        "processing_level": "L1A",
        "psf_executable": "./executables/phisat2_unix.bin",
        "snr_executable": "./executables/phisat2_unix.bin",
    }
    
    # # Add executables if provided
    # if args.psf_executable:
    #     phisat_config["psf_executable"] = args.psf_executable
    #     logger.info(f"Using PSF executable: {args.psf_executable}")
    
    # if args.snr_executable:
    #     phisat_config["snr_executable"] = args.snr_executable
    #     logger.info(f"Using SNR executable: {args.snr_executable}")
    
    logger.info(f"Phisat-2 config: {phisat_config}")
    
    # Create datamodule
    datamodule = trainer.create_datamodule(
        root_dir=args.data_root,
        phisat_config=phisat_config if any(phisat_config.values()) else None,
    )
    datamodule.setup(stage="fit")
    
    # Create model
    model = trainer.create_model(num_classes=2, pretrained=True)
    
    # Train
    trainer, trained_model = trainer.train(
        datamodule=datamodule,
        model=model,
        stage_1_epochs=20,
        disable_stage_2=False,
    )
    logger.info(f"Training complete!")
    
    # Save final model
    final_ckpt = Path(args.output_dir) / "best-val_mIoU.ckpt"
    trainer.test(trained_model, datamodule=datamodule, ckpt_path=str(final_ckpt))


if __name__ == "__main__":
    main()
