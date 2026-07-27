"""
Train a sensor-invariant CNN student by cross-sensor contrastive distillation
from the TerraMind ViT teacher on co-registered PhiSat-2 / Sentinel-2 patches.

Run from the ``terra_sat_drift`` package directory, like ``kd_lulc.py``:

    python kd_lulc_contrastive.py --teacher-ckpt /path/to/teacher.ckpt

The method and its references are documented in
``model_tasks/kd_contrastive_module.py`` and ``docs/cross_sensor_kd.md``.

Suggested ablation ladder (each line adds one term; all share the same splits,
so mean IoU on the PhiSat-2 branch and ``val/domain_gap_iou`` are comparable):

    --w-kd 1 --w-instance 0 --w-pixel 0 --kd-mode mse   # ~ the old baseline
    --w-kd 1 --w-instance 0 --w-pixel 0                 # + temperature KL
    --w-kd 1 --w-instance 0.5 --w-pixel 0               # + instance contrast
    --w-kd 1 --w-instance 0.5 --w-pixel 0.1             # + pixel contrast
    --w-kd 1 --w-instance 0.5 --w-pixel 0.1 --use-dsbn  # + domain-specific BN
"""

import argparse
import os
import warnings
from pathlib import Path

import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from terratorch.models.encoder_decoder_factory import EncoderDecoderFactory

from dataset.constants import WC_CLASS_MAPPING, WC_CLASS_NAMES
from dataset.datamodule_paired_triplets_lulc import PhisatPairedLULCDataModule
from model_tasks.kd_contrastive_module import CrossSensorKDModule
from student_mobilenet import create_student_model

warnings.filterwarnings("ignore")

H5_IMAGES = "/shared/projects/phisat2/data/processed/triplets_v1/phisat2_s2b_dataset_v1.h5"
H5_LABELS = "/shared/projects/phisat2/data/processed/worldcover_all_clean_v1/worldcover_all_clean_labels_v1.h5"
MANIFEST = "/shared/projects/phisat2/data/processed/worldcover_all_clean_v1/worldcover_all_clean_manifest_v1.csv"


def build_teacher_model(backbone="terramind_v1_tiny", num_classes=11, ckpt_path=None):
    """Instantiates the TerraMind teacher and loads fine-tuned weights.

    Unchanged from ``kd_lulc.build_teacher_model``. The teacher is only ever
    evaluated on the Sentinel-2 view, which is the domain it was pretrained and
    fine-tuned on -- that is the point of the cross-modal setup.
    """
    BACKBONE_NECK_INDICES = {
        "tiny": [2, 5, 8, 11],
        "small": [2, 5, 8, 11],
        "base": [2, 5, 8, 11],
        "large": [5, 11, 17, 23],
    }
    size = backbone.split("_")[-1]
    backbone_bands = {
        "S2L1C": ["BLUE", "GREEN", "RED", "RED_EDGE_1", "RED_EDGE_2", "RED_EDGE_3", "NIR_BROAD"]
    }
    model_args = {
        "backbone": backbone,
        "backbone_pretrained": False,
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
    teacher = EncoderDecoderFactory().build_model(task="segmentation", **model_args)

    if ckpt_path and os.path.exists(ckpt_path):
        print(f"Loading teacher weights from: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        missing, unexpected = teacher.load_state_dict(state_dict, strict=False)
        # Reported rather than swallowed: a silently mismatched teacher looks
        # exactly like a method that does not work.
        print(f"  missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
    else:
        print("WARNING: no teacher checkpoint provided. Distilling an untrained teacher!")

    return teacher


def parse_args():
    p = argparse.ArgumentParser(description="Cross-sensor contrastive KD for EO LULC")
    # data / schedule
    p.add_argument("--batch-size", type=int, default=8,
                   help="Patches per batch. Each patch costs two student forwards, and "
                        "in-batch negatives scale with this, so prefer the largest that fits.")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--target-domain", type=str, default="real", choices=["real", "sim"],
                   help="Sensor the student must work on at deployment.")
    p.add_argument("--source-domain", type=str, default="s2b", choices=["s2b"],
                   help="Sensor the teacher is trusted on.")
    p.add_argument("--no-augment", action="store_true",
                   help="Disable the joint geometric augmentation.")
    # teacher / student
    p.add_argument("--backbone", type=str, default="terramind_v1_tiny")
    p.add_argument("--teacher-ckpt", type=str, default=None)
    # loss weights
    p.add_argument("--w-task-target", type=float, default=1.0)
    p.add_argument("--w-task-source", type=float, default=1.0)
    p.add_argument("--w-kd", type=float, default=1.0)
    p.add_argument("--w-instance", type=float, default=0.5, help="Cross-sensor NT-Xent weight.")
    p.add_argument("--w-pixel", type=float, default=0.1, help="Semantic-guided pixel contrast weight.")
    p.add_argument("--w-crd", type=float, default=0.0, help="CRD-style teacher/student contrast weight.")
    # distillation
    p.add_argument("--kd-mode", type=str, default="kl", choices=["kl", "mse"])
    p.add_argument("--kd-temperature", type=float, default=4.0)
    p.add_argument("--kd-labeled-only", action="store_true",
                   help="Restrict distillation to labelled pixels (default: all pixels).")
    p.add_argument("--no-distill-source", action="store_true",
                   help="Distil only the PhiSat-2 branch, not the Sentinel-2 branch.")
    # contrastive
    p.add_argument("--instance-temperature", type=float, default=0.1)
    p.add_argument("--pixel-temperature", type=float, default=0.1)
    p.add_argument("--cross-view-negatives-only", action="store_true",
                   help="Use CLIP/CROMA negatives (N-1) instead of SimCLR's (2N-2).")
    p.add_argument("--proj-dim", type=int, default=128)
    p.add_argument("--pixel-proj-dim", type=int, default=64)
    p.add_argument("--pixel-feat-size", type=int, default=64)
    p.add_argument("--proto-momentum", type=float, default=0.999)
    p.add_argument("--max-pixels-per-class", type=int, default=128)
    p.add_argument("--contrastive-warmup-epochs", type=int, default=1)
    # architecture
    p.add_argument("--use-dsbn", action="store_true",
                   help="Domain-specific BatchNorm in the student (Chang et al. 2019).")
    # logging
    p.add_argument("--num-samples-to-log", type=int, default=4)
    p.add_argument("--log-every-n-epochs", type=int, default=1)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--precision", type=str, default="32-true")
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(42)

    tag = args.run_name or (
        f"xsensor_kd_{args.backbone}_kd{args.w_kd}_inst{args.w_instance}"
        f"_pix{args.w_pixel}{'_dsbn' if args.use_dsbn else ''}"
    )
    output_dir = Path("/shared/home/elucas/scratch/terra-sat-drift/outputs") / tag
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 1. Paired datamodule (co-registered PhiSat-2 / Sentinel-2)
    # ---------------------------------------------------------
    import albumentations as A

    datamodule = PhisatPairedLULCDataModule(
        h5_images_path=H5_IMAGES,
        h5_labels_path=H5_LABELS,
        manifest_path=MANIFEST,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_transform=A.Compose([]) if args.no_augment else None,
        val_transform=None,
        max_samples=args.max_samples,
        target_domain=args.target_domain,
        source_domain=args.source_domain,
    )

    # ---------------------------------------------------------
    # 2. Models
    # ---------------------------------------------------------
    num_classes = len(WC_CLASS_MAPPING)

    print(f"Building teacher ({args.backbone})...")
    teacher = build_teacher_model(
        backbone=args.backbone, num_classes=num_classes, ckpt_path=args.teacher_ckpt
    )

    print("Building student (UNet)...")
    student = create_student_model(in_channels=7, num_classes=num_classes, pretrained=False)

    # ---------------------------------------------------------
    # 3. Task
    # ---------------------------------------------------------
    task = CrossSensorKDModule(
        student_model=student,
        teacher_model=teacher,
        num_classes=num_classes,
        lr=args.lr,
        weight_decay=args.weight_decay,
        w_task_target=args.w_task_target,
        w_task_source=args.w_task_source,
        w_kd=args.w_kd,
        w_instance=args.w_instance,
        w_pixel=args.w_pixel,
        w_crd=args.w_crd,
        kd_mode=args.kd_mode,
        kd_temperature=args.kd_temperature,
        kd_on_unlabeled=not args.kd_labeled_only,
        distill_source_branch=not args.no_distill_source,
        instance_temperature=args.instance_temperature,
        pixel_temperature=args.pixel_temperature,
        cross_view_negatives_only=args.cross_view_negatives_only,
        proj_dim=args.proj_dim,
        pixel_proj_dim=args.pixel_proj_dim,
        pixel_feat_size=args.pixel_feat_size,
        proto_momentum=args.proto_momentum,
        max_pixels_per_class=args.max_pixels_per_class,
        contrastive_warmup_epochs=args.contrastive_warmup_epochs,
        use_dsbn=args.use_dsbn,
        ignore_index=-1,
        student_in_channels=7,
        rgb_band_indices=(2, 1, 0),
        num_samples_to_log=args.num_samples_to_log,
        log_every_n_epochs=args.log_every_n_epochs,
        class_names=WC_CLASS_NAMES,
    )

    # ---------------------------------------------------------
    # 4. Trainer
    # ---------------------------------------------------------
    logger = WandbLogger(project="kd-eo", name=tag, save_dir=str(output_dir))

    callbacks = [
        # Selected on target-domain IoU rather than total loss: the loss mixes
        # five terms with a ramping schedule, so it is not a stable model-
        # selection signal, and PhiSat-2 IoU is the quantity of interest.
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="best-{epoch:02d}-{val/student_target_iou:.4f}",
            monitor="val/student_target_iou",
            mode="max",
            save_top_k=3,
            save_last=True,
            auto_insert_metric_name=False,
        ),
        EarlyStopping(monitor="val/student_target_iou", mode="max", patience=12),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=logger,
        callbacks=callbacks,
        default_root_dir=str(output_dir),
        precision=args.precision,
    )

    print(
        f"Starting cross-sensor distillation | kd={args.w_kd} ({args.kd_mode}) "
        f"instance={args.w_instance} pixel={args.w_pixel} crd={args.w_crd} dsbn={args.use_dsbn}"
    )
    trainer.fit(task, datamodule=datamodule)

    datamodule.setup(stage="test")
    trainer.test(task, datamodule=datamodule)


if __name__ == "__main__":
    main()
