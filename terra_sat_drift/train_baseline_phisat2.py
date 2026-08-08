"""
No-KD baseline: train the student U-Net from scratch directly on one sensor.

This is the control for the cross-sensor KD experiments. Same architecture, same
seed-42 split, same per-domain normalisation, same augmentation, same task loss,
same optimiser and schedule, same metric names -- the only difference is that
there is no teacher, no distillation and no second sensor.

    # the baseline to compare against the DSBN KD run
    python terra_sat_drift/train_baseline_phisat2.py \
        --task-loss focal --class-weights inverse_sqrt \
        --max-samples 10000 --epochs 50

    # same thing on the full training split
    python terra_sat_drift/train_baseline_phisat2.py \
        --task-loss focal --class-weights inverse_sqrt --epochs 50

`--target-domain s2b` trains on Sentinel-2 instead, which gives the other useful
control: how far a Sentinel-2-trained model falls over when evaluated on PhiSat-2.

Note on a fair comparison: the KD runs apply a task loss to *both* sensor views of
each patch, so per epoch they take twice the gradient steps' worth of supervision
from the same labels. This baseline sees one view. That asymmetry is inherent to
the comparison -- it is what "no triplets" means -- but it is worth stating
rather than discovering later.
"""

import argparse
import os
from pathlib import Path

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from dataset.constants import WC_CLASS_MAPPING, WC_CLASS_NAMES, WC_CLASS_PIXEL_FREQ
from dataset.datamodule_paired_triplets_lulc import PhisatPairedLULCDataModule
from model_tasks.baseline_module import SupervisedSegmentationModule
from model_tasks.losses.segmentation import class_weights
from student_mobilenet import create_student_model

DATA = Path("/shared/projects/phisat2/data/processed")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target-domain", type=str, default="real",
                   choices=["real", "sim", "s2b"],
                   help="Sensor to train and evaluate on.")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--max-samples", type=int, default=None,
                   help="Cap on training patches. The KD runs so far used 10000 "
                        "(5%% of the split); match it for a like-for-like comparison.")
    p.add_argument("--val-max-samples", type=int, default=1000,
                   help="Validation patches. Fixed, NOT derived from --max-samples: "
                        "1000 is the smallest size containing all 11 classes. Keep it "
                        "constant across runs you intend to compare.")
    p.add_argument("--test-max-samples", type=int, default=None,
                   help="Test patches; default None = the full 25,323. Test runs once, "
                        "so it can afford to be thorough (snow/ice on 255 patches "
                        "instead of 14).")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--task-loss", type=str, default="ce", choices=["ce", "focal"])
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--class-weights", type=str, default="none",
                   choices=["none", "inverse", "inverse_sqrt", "effective"])
    p.add_argument("--class-weight-beta", type=float, default=0.999)
    p.add_argument("--num-samples-to-log", type=int, default=4)
    p.add_argument("--log-every-n-epochs", type=int, default=1)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--test-ckpt", type=str, default="best", choices=["best", "last"],
                   help="Which weights to run the test split on. The KD runs "
                        "effectively used 'last' (they pass no ckpt_path), so use "
                        "'last' for a strictly like-for-like number.")
    p.add_argument("--output-root", type=Path,
                   default=Path("/shared/home/elucas/scratch/terra-sat-drift/outputs"))
    p.add_argument("--tag", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    num_classes = len(WC_CLASS_MAPPING)

    tag = args.tag or (
        f"baseline_nokd_{args.target_domain}_{args.task_loss}"
        f"_{args.class_weights}"
        f"{'_n' + str(args.max_samples) if args.max_samples else '_full'}"
    )
    output_dir = args.output_root / tag
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run tag: {tag}\nOutput:  {output_dir}")

    # --- data: single-domain mode (source_domain=None) ------------------
    dm = PhisatPairedLULCDataModule(
        h5_images_path=str(DATA / "triplets_v1/phisat2_s2b_dataset_v1.h5"),
        h5_labels_path=str(DATA / "worldcover_all_clean_v1/worldcover_all_clean_labels_v1.h5"),
        manifest_path=str(DATA / "worldcover_all_clean_v1/worldcover_all_clean_manifest_v1.csv"),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        val_max_samples=args.val_max_samples,
        test_max_samples=args.test_max_samples,
        augment=not args.no_augment,
        target_domain=args.target_domain,
        source_domain=None,
    )

    # --- model ----------------------------------------------------------
    print("Building student U-Net (random init, no teacher)...")
    model = create_student_model(in_channels=7, num_classes=num_classes, pretrained=False)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params/1e6:.2f}M parameters")

    task = SupervisedSegmentationModule(
        model=model,
        num_classes=num_classes,
        lr=args.lr,
        weight_decay=args.weight_decay,
        ignore_index=-1,
        task_loss=args.task_loss,
        focal_gamma=args.focal_gamma,
        # Passed as a plain list, not a numpy array: `save_hyperparameters`
        # stores this in the checkpoint, and PyTorch >= 2.6 loads checkpoints
        # with weights_only=True by default, which refuses to unpickle numpy
        # objects -- making the checkpoint unloadable by `ckpt_path="best"`.
        class_weights=(None if args.class_weights == "none"
                       else class_weights(WC_CLASS_PIXEL_FREQ, scheme=args.class_weights,
                                          beta=args.class_weight_beta).tolist()),
        class_names=WC_CLASS_NAMES,
        num_samples_to_log=args.num_samples_to_log,
        log_every_n_epochs=args.log_every_n_epochs,
        rgb_band_indices=(2, 1, 0),
    )

    logger = WandbLogger(project="kd-eo", name=tag, save_dir=str(output_dir))
    callbacks = [
        # Same monitor and filename pattern as the KD runs, so the two families
        # of checkpoints and W&B panels line up without special-casing.
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="best-{epoch:02d}-{val/student_target_iou:.4f}",
            monitor="val/student_target_iou", mode="max",
            save_top_k=3, save_last=True, auto_insert_metric_name=False,
        ),
        EarlyStopping(monitor="val/student_target_iou", mode="max", patience=args.patience),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=logger,
        callbacks=callbacks,
        default_root_dir=str(output_dir),
        log_every_n_steps=50,
    )
    trainer.fit(task, datamodule=dm)

    # NOTE on comparability: kd_lulc_contrastive.py calls `trainer.test(task,
    # datamodule=...)` with no ckpt_path, which in Lightning 2.6 evaluates the
    # *in-memory* (final-epoch) weights, not the best checkpoint. So the test
    # numbers recorded for the KD runs are final-epoch numbers. Use
    # `--test-ckpt last` here to reproduce that convention, or the default
    # `best` for the more standard (and more favourable) protocol -- just do not
    # mix the two in one table.
    trainer.test(task, datamodule=dm,
                 ckpt_path=None if args.test_ckpt == "last" else "best")


if __name__ == "__main__":
    main()
