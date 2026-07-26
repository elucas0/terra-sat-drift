import os
import argparse
from pathlib import Path
import warnings

import torch
import albumentations as A
import lightning.pytorch as pl
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger

from terratorch.models.encoder_decoder_factory import EncoderDecoderFactory

# Import your custom modules
from student_mobilenet import create_student_model
from model_tasks.kd_module import KDSegmentationModule
from dataset.datamodule_triplets_lulc import PhisatRealLULCDataModule
from dataset.constants import WC_CLASS_MAPPING

warnings.filterwarnings('ignore')

def build_teacher_model(backbone="terramind_v1_tiny", num_classes=11, ckpt_path=None):
    """Instantiates the heavy TerraMind Teacher Model and loads fine-tuned weights"""
    BACKBONE_NECK_INDICES = {
        "tiny": [2, 5, 8, 11],
        "small": [2, 5, 8, 11],
        "base": [2, 5, 8, 11],
        "large": [5, 11, 17, 23],
    }
    
    # Extract size identifier (e.g., 'tiny' from 'terramind_v1_tiny')
    size = backbone.split('_')[-1]
    
    backbone_bands = {
        "S2L1C": ["BLUE", "GREEN", "RED", "RED_EDGE_1", "RED_EDGE_2", "RED_EDGE_3", "NIR_BROAD"]
    }
    
    model_args = {
        "backbone": backbone,
        "backbone_pretrained": False, # Will load from fine-tuned checkpoint anyway
        "backbone_modalities": ["S2L1C"],
        "backbone_bands": backbone_bands,
        "necks": [
            {"name": "SelectIndices", "indices": BACKBONE_NECK_INDICES[size]},
            {"name": "ReshapeTokensToImage", "remove_cls_token": False},
            {"name": "LearnedInterpolateToPyramidal"},
        ],
        "decoder": "UNetDecoder",
        "decoder_channels": [256, 128, 64, 32],
        "head_dropout": 0.1,
        "num_classes": num_classes,
    }
    
    teacher = EncoderDecoderFactory().build_model(
        task="segmentation", 
        **model_args
    )
    
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"Loading Teacher weights from: {ckpt_path}")
        # Assuming a standard PyTorch Lightning checkpoint
        state_dict = torch.load(ckpt_path, map_location="cpu")['state_dict']
        # Remove 'model.' prefix if it exists (depends on how SemanticSegmentationTask saves it)
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        teacher.load_state_dict(state_dict, strict=False)
    else:
        print("WARNING: No teacher checkpoint provided. Distilling an untrained teacher!")
        
    return teacher

def main():
    parser = argparse.ArgumentParser(description="Knowledge Distillation for EO LULC")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone", type=str, default="terramind_v1_tiny", help="Teacher backbone")
    parser.add_argument("--teacher-ckpt", type=str, default=None, help="Path to fine-tuned teacher checkpoint")
    parser.add_argument("--alpha", type=float, default=0.4, help="Weight for Teacher MSE loss (0.0 to 1.0)")
    parser.add_argument("--use-pseudo-labels", action="store_true", help="If passed, ignores GT and uses Teacher predictions")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-samples-to-log", type=int, default=4, help="How many val samples to show in the qualitative plot")
    parser.add_argument("--log-every-n-epochs", type=int, default=1, help="How often (in val epochs) to log the qualitative plot")
    args = parser.parse_args()

    seed_everything(42)
    output_dir = Path(f"/shared/home/elucas/scratch/terra-sat-drift/outputs/kd_lulc_{args.backbone}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 1. DataModule Setup (Real PhiSat-2 Triplets)
    # ---------------------------------------------------------
    h5_images_path = "/shared/projects/phisat2/data/processed/triplets_v1/phisat2_s2b_dataset_v1.h5"
    h5_labels_path = "/shared/projects/phisat2/data/processed/worldcover_all_clean_v1/worldcover_all_clean_labels_v1.h5"
    manifest_path = "/shared/projects/phisat2/data/processed/worldcover_all_clean_v1/worldcover_all_clean_manifest_v1.csv"

    datamodule = PhisatRealLULCDataModule(
        h5_images_path=h5_images_path,
        h5_labels_path=h5_labels_path,
        manifest_path=manifest_path,
        batch_size=args.batch_size,
        num_workers=16,
        train_transform=None,
        val_transform=None,
        max_samples=args.max_samples,
    )

    # ---------------------------------------------------------
    # 2. Model Initialization
    # ---------------------------------------------------------
    num_classes = len(WC_CLASS_MAPPING) # WorldCover LULC
    
    print(f"Building Teacher Model ({args.backbone})...")
    teacher = build_teacher_model(backbone=args.backbone, num_classes=num_classes, ckpt_path=args.teacher_ckpt)
    
    print("Building Student Model (UNet)...")
    # 7 bands input (B, G, R, RE1, RE2, RE3, NIR)
    student = create_student_model(in_channels=7, num_classes=num_classes, pretrained=False)

    # ---------------------------------------------------------
    # 3. Knowledge Distillation Task
    # ---------------------------------------------------------
    kd_task = KDSegmentationModule(
        student_model=student,
        teacher_model=teacher,
        num_classes=num_classes,
        lr=args.lr,
        alpha=args.alpha,
        use_pseudo_labels=args.use_pseudo_labels,
        ignore_index=-1,
        num_samples_to_log=args.num_samples_to_log,
        log_every_n_epochs=args.log_every_n_epochs,
        class_names=list(WC_CLASS_MAPPING.keys()),
        rgb_band_indices=(2, 1, 0),
    )

    # ---------------------------------------------------------
    # 4. Trainer Setup
    # ---------------------------------------------------------
    logger = WandbLogger(project="kd-eo", name=f"kd_unet_student_{args.backbone}_alpha{args.alpha}")
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="best-kd-model-{epoch:02d}-{val/loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
    )

    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback, EarlyStopping(monitor="val/loss", patience=10, mode="min")],
        #precision="16-mixed"
    )

    # ---------------------------------------------------------
    # 5. Start Training
    # ---------------------------------------------------------
    print(f"Starting Distillation... (Alpha: {args.alpha}, Pseudo-Labels: {args.use_pseudo_labels})")
    trainer.fit(kd_task, datamodule=datamodule)
    
    datamodule.setup(stage="test")
    trainer.test(kd_task, datamodule=datamodule)

if __name__ == "__main__":
    main()