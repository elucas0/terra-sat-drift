"""Domain-gap probe on the Φ-sat-2 degradation ladder, fine-tuning TerraMind end to end.

The counterpart to ``finetune_floods_student.py``: same experiment, different model. Where
that script freezes a distilled MobileNet encoder and trains a decoder on top, this
fine-tunes a pretrained TerraMind ViT with a UNet decoder, nothing frozen. Comparing the
two degradation curves separates "how much does this perturbation destroy" from "how much
does this particular representation care".

Model, datamodule and optimisation follow ``train_validate_floods_simulated.py`` -- the
TerraMind backbone with ``SelectIndices`` / ``ReshapeTokensToImage`` /
``LearnedInterpolateToPyramidal`` necks into a ``UNetDecoder``, dice loss, class weights
[0.3, 0.7], AdamW at 2e-5 with ReduceLROnPlateau -- with three deliberate changes:

* ``--backbone-size base`` by default, where the reference script used ``tiny``.
* **No centre crop on val and test.** The reference script cropped a 512x512 window out
  of the 1077x1077 scene, which scores the model on 23% of each image and makes the
  metric depend on where the flood happens to sit. Evaluation here runs on the full
  scene. TerraMind interpolates its position embeddings, so 1077x1077 is accepted
  directly (verified: matching output shape, ~1.3 GiB at batch 1) and no resampling to a
  patch-divisible size is needed. Training still uses random 512 crops.
* ``--test-roots``, so one fine-tuned checkpoint can be scored against every rung of the
  ladder in a single process -- the domain-gap measurement.

Note the metric consequence of dropping the crop: full-scene IoU is not comparable to the
reference script's centre-crop IoU, and neither is comparable to the student probe's
256-crop numbers. Compare *drops from the clean baseline* across models, not absolute IoU.

Examples::

    # fine-tune on the clean rung, then score every ladder variant
    python terra_sat_drift/finetune_floods_terramind.py \\
        --sim-root /shared/home/elucas/datasets/sen1floods11_ladder/clean \\
        --test-roots /shared/home/elucas/datasets/sen1floods11_ladder/*

    # batch_probe_ladder.py --model terramind drives the whole sweep
"""

from __future__ import annotations

import argparse
import gc
import json
import warnings
from pathlib import Path

import albumentations as A
import cv2
import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from terratorch.datamodules import GenericNonGeoSegmentationDataModule
from terratorch.tasks import SemanticSegmentationTask

warnings.filterwarnings("ignore")

DEFAULT_SIM_ROOT = Path("/shared/home/elucas/datasets/sen1floods11_ladder/clean")
LABEL_ROOT = Path("/shared/home/elucas/datasets/sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand")
IMG_SUBDIR = "data/flood_events/HandLabeled/S2Hand"
SPLIT_SUBDIR = "splits/flood_handlabeled"

# The simulated scenes are 1077x1077 at 4.75 m; the hand labels are 512x512 at 10 m, so
# both are put on the 1077 grid before anything else happens.
SCENE_SIZE = 1077

# Band 3 is panchromatic and dropped, leaving Blue, Green, Red, RE1, RE2, RE3, NIR.
DATASET_BANDS = [0, 1, 2, 3, 4, 5, 6, 7]
OUTPUT_BANDS = [0, 1, 2, 4, 5, 6, 7]
MEANS = [2137.385, 2018.788, 2082.986, 2295.651, 2854.537, 3122.849, 3040.560]
STDS = [1675.806, 1557.708, 1833.702, 1823.738, 1733.977, 1732.131, 1679.732]

BACKBONE_NECK_INDICES = {
    "tiny": [2, 5, 8, 11],
    "small": [2, 5, 8, 11],
    "base": [2, 5, 8, 11],
    "large": [5, 11, 17, 23],
}
BACKBONE_BANDS = {
    "S2L1C": ["BLUE", "GREEN", "RED", "RED_EDGE_1", "RED_EDGE_2", "RED_EDGE_3", "NIR_BROAD"]
}
CLASS_NAMES = ["background", "flood"]


def resolve_sim_root(path: Path | str) -> Path:
    """Accept either a dataset root or its ``v1.1`` directory."""
    path = Path(path)
    return path / "v1.1" if (path / "v1.1").is_dir() else path


def variant_name(sim_root: Path) -> str:
    return sim_root.parent.name if sim_root.name == "v1.1" else sim_root.name


def preprocess_mask(mask, **kwargs):
    """Sen1Floods11 labels to a binary flood mask, preserving the -1 no-data marker.

    The reference script mapped everything that was not class 1 to 0, which silently
    turns unobserved pixels into confident "not flood" and inflates the background IoU.
    ``ignore_index=-1`` only works if -1 survives to the loss, so it is kept here.
    """
    clean = np.full(mask.shape, 0, dtype=np.int64)
    clean[mask == 1] = 1
    clean[mask == -1] = -1
    return clean


def build_datamodule(sim_root: Path, crop: int, batch_size: int, num_workers: int,
                     eval_full_scene: bool = True) -> GenericNonGeoSegmentationDataModule:
    resize = A.Resize(width=SCENE_SIZE, height=SCENE_SIZE, interpolation=cv2.INTER_NEAREST)
    train_transform = [
        resize,
        A.RandomCrop(width=crop, height=crop),
        A.Lambda(mask=preprocess_mask),
        A.pytorch.ToTensorV2(),
    ]
    # No centre crop: val and test see the whole scene.
    val_test_transform = [resize]
    if not eval_full_scene:
        val_test_transform.append(A.CenterCrop(width=crop, height=crop))
    val_test_transform += [A.Lambda(mask=preprocess_mask), A.pytorch.ToTensorV2()]

    img_dir = sim_root / IMG_SUBDIR
    splits = sim_root / SPLIT_SUBDIR
    return GenericNonGeoSegmentationDataModule(
        batch_size=batch_size,
        num_workers=num_workers,
        num_classes=2,
        train_data_root=img_dir,
        val_data_root=img_dir,
        test_data_root=img_dir,
        train_label_data_root=LABEL_ROOT,
        val_label_data_root=LABEL_ROOT,
        test_label_data_root=LABEL_ROOT,
        train_split=splits / "flood_train_data.txt",
        val_split=splits / "flood_valid_data.txt",
        test_split=splits / "flood_test_data.txt",
        img_grep="*_S2Hand.tif",
        label_grep="*_LabelHand.tif",
        train_transform=train_transform,
        val_transform=val_test_transform,
        test_transform=val_test_transform,
        dataset_bands=DATASET_BANDS,
        output_bands=OUTPUT_BANDS,
        means=MEANS,
        stds=STDS,
        rgb_indices=[2, 1, 0],
        no_label_replace=-1,
        no_data_replace=0,
    )


def build_task(args) -> SemanticSegmentationTask:
    model_args = {
        "backbone": f"terramind_v1_{args.backbone_size}",
        "backbone_pretrained": not args.no_pretrained,
        "backbone_modalities": ["S2L1C"],
        "backbone_bands": BACKBONE_BANDS,
        "necks": [
            {"name": "SelectIndices", "indices": BACKBONE_NECK_INDICES[args.backbone_size]},
            {"name": "ReshapeTokensToImage", "remove_cls_token": False},
            {"name": "LearnedInterpolateToPyramidal"},
        ],
        "decoder": "UNetDecoder",
        "decoder_channels": [256, 128, 64, 32],
        "head_dropout": 0.1,
        "num_classes": 2,
    }
    return SemanticSegmentationTask(
        model_factory="EncoderDecoderFactory",
        model_args=model_args,
        lr=args.lr,
        scheduler="ReduceLROnPlateau",
        scheduler_hparams={"factor": 0.5, "patience": 5},
        ignore_index=-1,
        plot_on_val=False,
        loss=args.loss,
        optimizer="AdamW",
        optimizer_hparams={"weight_decay": args.weight_decay},
        class_names=CLASS_NAMES,
        freeze_backbone=args.freeze_backbone,
        freeze_decoder=False,
        class_weights=[0.3, 0.7],
    )


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sim-root", type=Path, default=DEFAULT_SIM_ROOT,
                   help="Dataset to fine-tune on (dataset root or its v1.1 directory)")
    p.add_argument("--test-roots", nargs="+", type=Path, default=None,
                   help="Also evaluate the best checkpoint on each of these roots' test split. "
                        "This is the domain-gap measurement.")
    p.add_argument("--backbone-size", default="base", choices=sorted(BACKBONE_NECK_INDICES))
    p.add_argument("--no-pretrained", action="store_true",
                   help="Random backbone init; the control for how much pretraining contributes.")
    p.add_argument("--freeze-backbone", action="store_true",
                   help="Train the decoder only. Default is end-to-end fine-tuning, as in "
                        "train_validate_floods_simulated.py.")
    p.add_argument("--crop", type=int, default=512, help="Training crop size")
    p.add_argument("--eval-crop", action="store_true",
                   help="Restore the reference script's centre crop on val/test instead of "
                        "evaluating the full scene.")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--eval-batch-size", type=int, default=None,
                   help="Batch size for the full-scene val/test passes (default: --batch-size). "
                        "Lower it if 1077x1077 evaluation runs out of memory.")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--loss", default="dice", choices=["dice", "ce", "focal", "jaccard"])
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--eval-workers", type=int, default=0,
                   help="Dataloader workers for the cross-condition evaluations. Defaults to 0 "
                        "(load in the main process): one run builds a fresh datamodule per test "
                        "root, and successive worker pools across that many trainer.test calls "
                        "have deadlocked here, stalling a run indefinitely. Costs ~1 min per test "
                        "set instead of ~35 s, which is worth it for a sweep that must finish.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--metrics-out", type=Path, default=None)
    p.add_argument("--output-root", type=Path,
                   default=Path("/shared/home/elucas/scratch/terra-sat-drift/outputs"))
    p.add_argument("--tag", type=str, default=None)
    p.add_argument("--wandb-project", default="terra-sat-drift")
    p.add_argument("--no-wandb", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)

    sim_root = resolve_sim_root(args.sim_root)
    tag = args.tag or (f"floods_terramind_{args.backbone_size}_{variant_name(sim_root)}"
                       f"_s{args.seed}")
    output_dir = args.output_root / tag
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_full_scene = not args.eval_crop
    print(f"Run tag: {tag}\nOutput:  {output_dir}\nData:    {sim_root}\n"
          f"Eval:    {'full scene ' + str(SCENE_SIZE) + 'x' + str(SCENE_SIZE) if eval_full_scene else f'centre crop {args.crop}'}")

    datamodule = build_datamodule(sim_root, args.crop, args.batch_size, args.num_workers,
                                  eval_full_scene=eval_full_scene)
    task = build_task(args)

    if args.no_wandb:
        from lightning.pytorch.loggers import CSVLogger
        logger = CSVLogger(save_dir=str(output_dir), name="csv")
    else:
        logger = WandbLogger(project=args.wandb_project, name=tag, save_dir=str(output_dir))

    checkpoint = ModelCheckpoint(
        dirpath=output_dir / "checkpoints", filename="best-{epoch:02d}-{val/mIoU:.4f}",
        monitor="val/mIoU", mode="max", save_top_k=1, save_last=True,
        auto_insert_metric_name=False)
    callbacks = [
        checkpoint,
        EarlyStopping(monitor="val/mIoU", mode="max", patience=args.patience),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1, logger=logger, callbacks=callbacks,
        default_root_dir=str(output_dir), log_every_n_steps=10,
    )

    trainer.fit(model=task, datamodule=datamodule)
    test_metrics = trainer.test(model=task, datamodule=datamodule, ckpt_path="best")

    # Domain gap: the same checkpoint against every other degradation condition. A fresh
    # datamodule per root keeps the test dataloader pointed at one dataset at a time,
    # which matters because the task accumulates metrics per test epoch.
    cross_condition = {}
    if args.test_roots:
        best_path = checkpoint.best_model_path
        eval_bs = args.eval_batch_size or args.batch_size
        print(f"\nCross-condition evaluation of {best_path} on {len(args.test_roots)} test sets")
        for root in args.test_roots:
            root = resolve_sim_root(root)
            name = variant_name(root)
            dm = build_datamodule(root, args.crop, eval_bs, args.eval_workers,
                                  eval_full_scene=eval_full_scene)
            res = trainer.test(model=task, datamodule=dm, ckpt_path=best_path)
            cross_condition[name] = {k: float(v) for k, v in (res[0] if res else {}).items()}
            miou = cross_condition[name].get("test/mIoU", float("nan"))
            print(f"  {name:<14} mIoU {miou:.4f}", flush=True)
            # Drop the datamodule and its dataloaders before building the next one, so
            # worker pools cannot accumulate across the 13 evaluations.
            del dm
            gc.collect()

    if args.metrics_out:
        record = {
            "tag": tag,
            "model": f"terramind_v1_{args.backbone_size}",
            "sim_root": str(sim_root),
            "seed": args.seed,
            "scratch": args.no_pretrained,
            "freeze": "backbone" if args.freeze_backbone else "none",
            "eval_full_scene": eval_full_scene,
            "epochs_run": int(trainer.current_epoch),
            "best_ckpt": str(checkpoint.best_model_path),
            "best_val_iou": float(checkpoint.best_model_score)
            if checkpoint.best_model_score is not None else None,
            "test": {k: float(v) for k, v in (test_metrics[0] if test_metrics else {}).items()},
            "cross_condition": cross_condition,
        }
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_out.write_text(json.dumps(record, indent=2) + "\n")
        print(f"Metrics written to {args.metrics_out}")


if __name__ == "__main__":
    main()
