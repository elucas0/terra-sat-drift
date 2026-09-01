"""Frozen-encoder probe of the PhiSat-2 foundation-model student on real PhiSat-2 LULC.

Takes the published HydraNet / Phi2FM `PhisatNet` student -- pretrained on
*simulated* PhiSat-2 imagery -- keeps its encoder and bottleneck frozen, and
trains a new decoder on *real* PhiSat-2, so the result can sit next to the no-KD
U-Net baseline and the TerraMind runs in the land-cover table.

It is `train_baseline_phisat2.py` that this is matched to, exactly: same split,
same normalisation, same loss, same schedule, same `SupervisedSegmentationModule`
and therefore the same metric keys. The TerraMind runs are *not* automatically
comparable -- they go through terratorch's `SemanticSegmentationTask`, which logs
`test/mIoU` rather than `test/student_target_iou`, and `pretrain_lulc.py` trains
on the Sentinel-2 view rather than PhiSat-2. Check which TerraMind run is being
put in the table and on which domain before lining the numbers up.

Everything that is not the intervention is held identical to the no-KD baseline:
seed-42 80/10/10 split, the same `BAD_PRODUCT_IDS` filter, the same per-domain
sqrt/clip/z-score normalisation, the same geometric augmentation, the same focal
/ class-weighted task loss, the same optimiser and ReduceLROnPlateau schedule,
the same early-stopping monitor, and the same metric key names. The differences
are stated in one place -- here:

  * **8 channels, not 7.** The baselines take Blue..NIR; this model's stem was
    trained on eight bands including panchromatic, so it gets all eight, in the
    order its stem expects (`real8`; see `dataset.dataset_paired_triplets_lulc`).
    Dropping PAN to match the baselines exactly is `--in-channels 7`, which also
    discards the pretrained stem weights -- an ablation, not the headline run.
  * **A higher learning rate.** Training a decoder on frozen features is not the
    same optimisation problem as training a network end to end, and 1e-3 is the
    rate this repo's other frozen-encoder probes use. `--lr 1e-4` reproduces the
    baseline's rate if a single number across the table matters more.
  * **Far fewer trainable parameters**: 0.15M of the model's 0.35M, with 0.20M
    frozen, against 9.85M trained end to end in the U-Net baseline. That gap is
    the point of the comparison, not a confound, but it means an epoch buys less
    here and the early-stopping patience is doing more of the work.

Usage::

    # the headline run: frozen pretrained encoder, new decoder, real PhiSat-2
    python terra_sat_drift/train_phisatnet_lulc.py \\
        --task-loss focal --class-weights inverse_sqrt --max-samples 10000 --epochs 50

    # the control that says what the pretraining is worth: same protocol, no weights
    python terra_sat_drift/train_phisatnet_lulc.py --no-pretrained \\
        --task-loss focal --class-weights inverse_sqrt --max-samples 10000 --epochs 50

    # let the frozen encoder's BatchNorm statistics adapt to the target sensor
    python terra_sat_drift/train_phisatnet_lulc.py --bn-mode adapt ...

    # full fine-tune, for the top of the ladder
    python terra_sat_drift/train_phisatnet_lulc.py --freeze none --lr 1e-4 ...

Match `--max-samples` to whatever the baseline and TerraMind rows used; the
existing baseline runs in `outputs/` are `n1000`, `n5000` and `n10000`.
"""

import argparse
import json
from pathlib import Path

import lightning.pytorch as pl
import torch
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from dataset.constants import WC_CLASS_MAPPING, WC_CLASS_NAMES, WC_CLASS_PIXEL_FREQ
from dataset.datamodule_paired_triplets_lulc import PhisatPairedLULCDataModule
from model_tasks.baseline_module import SupervisedSegmentationModule
from model_tasks.losses.segmentation import class_weights
from phisatnet_student import create_phisatnet_student, resolve_checkpoint

DATA = Path("/shared/projects/phisat2/data/processed")
DEFAULT_WEIGHTS_DIR = Path("/shared/home/elucas/scratch/terra-sat-drift/weights/hydranet")


class FrozenEncoderSegmentationModule(SupervisedSegmentationModule):
    """The baseline task, with the optimiser given only the parameters that move.

    `SupervisedSegmentationModule` hands `self.model.parameters()` to AdamW.
    Frozen parameters never receive a gradient so AdamW skips them, which is
    harmless but makes the logged parameter count and any future weight-decay
    accounting lie about what is being optimised. Filtering here keeps the run
    record honest without touching the shared module the baseline uses.
    """

    def configure_optimizers(self):
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable:
            raise ValueError("Nothing to train: every parameter is frozen.")
        opt = torch.optim.AdamW(trainable, lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode=self.lr_monitor_mode, factor=0.5, patience=5)
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "monitor": self.lr_monitor}}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # --- what to train on ---------------------------------------------------
    p.add_argument("--target-domain", type=str, default="real8",
                   choices=["real8", "sim8", "s2b8"],
                   help="Sensor to train and evaluate on. Only the 8-channel "
                        "domains are valid here: they carry the band permutation "
                        "the PhiSat-2 student's stem expects. 's2b8' fills the "
                        "panchromatic channel with a proxy -- a control, not a "
                        "like-for-like input.")
    p.add_argument("--max-samples", type=int, default=10000,
                   help="Cap on training patches. Match the baseline row you "
                        "intend to compare against (10000 for the KD runs).")
    p.add_argument("--val-max-samples", type=int, default=1000,
                   help="Validation patches. Fixed, NOT derived from --max-samples: "
                        "1000 is the smallest size containing all 11 classes.")
    p.add_argument("--test-max-samples", type=int, default=None,
                   help="Test patches; default None = the full 25,323.")
    p.add_argument("--no-augment", action="store_true")

    # --- which pretrained student -------------------------------------------
    p.add_argument("--checkpoint-task", type=str, default="lc",
                   help="Published checkpoint task. 'lc' is the same 11-class "
                        "WorldCover problem on simulated PhiSat-2, so its encoder "
                        "is the closest available starting point.")
    p.add_argument("--checkpoint-training", type=str, default="finetuning",
                   choices=["finetuning", "linear_probing"])
    p.add_argument("--checkpoint-nshots", type=int, default=5000)
    p.add_argument("--checkpoint-datetime", type=int, default=None,
                   help="Pin a release date (e.g. 20260108). Default: the latest. "
                        "Releases are separate trainings, not reruns -- their "
                        "encoders differ -- so pin this for anything reproducible.")
    p.add_argument("--weights-dir", type=Path, default=DEFAULT_WEIGHTS_DIR)
    p.add_argument("--hydranet-src", type=Path, default=None,
                   help="`src` directory of the hydranet-phisat2 checkout.")
    p.add_argument("--no-pretrained", action="store_true",
                   help="Random init instead of the published weights. This is the "
                        "control that measures what the pretraining is worth; run it.")

    # --- what is frozen and what is rebuilt ---------------------------------
    p.add_argument("--freeze", type=str, default="encoder+bottleneck",
                   choices=["encoder+bottleneck", "encoder", "none"])
    p.add_argument("--bn-mode", type=str, default="frozen", choices=["frozen", "adapt"],
                   help="'frozen' keeps the frozen encoder's BatchNorm in eval mode. "
                        "'adapt' lets its running statistics track the target sensor, "
                        "which is a domain-adaptation intervention in its own right.")
    p.add_argument("--warm-start-decoder", action="store_true",
                   help="Start from the published decoder instead of a new one. "
                        "That is a warm start, not the 'new decoder head' protocol.")
    p.add_argument("--gamma-fill", type=str, default="ones", choices=["ones", "model_init"],
                   help="Fill for the layer-scale gammas the checkpoints omit. 'ones' "
                        "reproduces training-time behaviour; 'model_init' reproduces "
                        "hydranet.load_student, which scales them by 1e-6.")
    p.add_argument("--in-channels", type=int, default=8, choices=[7, 8],
                   help="7 drops the panchromatic band to match the baselines' input "
                        "exactly, at the cost of the pretrained stem weights.")

    # --- optimisation -------------------------------------------------------
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3,
                   help="1e-3 is this repo's frozen-encoder probe rate; the end-to-end "
                        "baseline uses 1e-4.")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--task-loss", type=str, default="ce", choices=["ce", "focal"])
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--class-weights", type=str, default="none",
                   choices=["none", "inverse", "inverse_sqrt", "effective"])
    p.add_argument("--class-weight-beta", type=float, default=0.999)
    p.add_argument("--seed", type=int, default=42,
                   help="Seeds init and augmentation order. The train/val/test split "
                        "is separately fixed at 42 inside the dataset, so changing "
                        "this varies the run, never the data.")

    # --- bookkeeping --------------------------------------------------------
    p.add_argument("--num-samples-to-log", type=int, default=4)
    p.add_argument("--log-every-n-epochs", type=int, default=1)
    p.add_argument("--test-ckpt", type=str, default="best", choices=["best", "last"],
                   help="Which weights the test split runs on. The KD runs recorded "
                        "final-epoch numbers ('last'); do not mix the two in a table.")
    p.add_argument("--output-root", type=Path,
                   default=Path("/shared/home/elucas/scratch/terra-sat-drift/outputs"))
    p.add_argument("--tag", type=str, default=None)
    p.add_argument("--metrics-out", type=Path, default=None,
                   help="Write a JSON run record here (same shape as the probe "
                        "ladder's), for collecting rows into a table.")
    p.add_argument("--wandb-project", type=str, default="kd-eo")
    p.add_argument("--no-wandb", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed, workers=True)
    num_classes = len(WC_CLASS_MAPPING)

    init = "scratch" if args.no_pretrained else (
        f"{args.checkpoint_task}-{args.checkpoint_training[:2]}{args.checkpoint_nshots}")
    # "probe" only when something is actually frozen. `--freeze none` is a full
    # fine-tune answering a different question, and a directory called
    # `phisatnet_probe_..._none_...` reads as a probe at a glance in a results
    # listing, which is how a row ends up mislabelled in a table.
    kind = "ft" if args.freeze == "none" else "probe"
    tag = args.tag or (
        f"phisatnet_{kind}_{args.target_domain}_{init}"
        f"_{args.freeze.replace('+', '-')}_{args.task_loss}_{args.class_weights}"
        f"{'_n' + str(args.max_samples) if args.max_samples else '_full'}"
        f"_s{args.seed}"
    )
    output_dir = args.output_root / tag
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run tag: {tag}\nOutput:  {output_dir}")

    # --- data: single-domain mode (source_domain=None), identical to baseline ---
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

    # --- model --------------------------------------------------------------
    checkpoint = None
    if not args.no_pretrained:
        args.weights_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = resolve_checkpoint(
            task=args.checkpoint_task,
            training=args.checkpoint_training,
            n_shots=args.checkpoint_nshots,
            datetime=args.checkpoint_datetime,
            weights_dir=str(args.weights_dir),
            hydranet_src=args.hydranet_src,
        )

    print("Building PhiSat-2 student (frozen encoder, new decoder)...")
    model = create_phisatnet_student(
        num_classes=num_classes,
        checkpoint=checkpoint,
        in_channels=args.in_channels,
        freeze=args.freeze,
        reinit_decoder=not args.warm_start_decoder,
        gamma_fill=args.gamma_fill,
        bn_mode=args.bn_mode,
        hydranet_src=args.hydranet_src,
    )

    task = FrozenEncoderSegmentationModule(
        model=model,
        num_classes=num_classes,
        lr=args.lr,
        weight_decay=args.weight_decay,
        ignore_index=-1,
        task_loss=args.task_loss,
        focal_gamma=args.focal_gamma,
        # A plain list, not a numpy array: `save_hyperparameters` puts this in
        # the checkpoint, and torch >= 2.6 loads with weights_only=True, which
        # refuses to unpickle numpy objects -- making `ckpt_path="best"` fail.
        class_weights=(None if args.class_weights == "none"
                       else class_weights(WC_CLASS_PIXEL_FREQ, scheme=args.class_weights,
                                          beta=args.class_weight_beta).tolist()),
        class_names=WC_CLASS_NAMES,
        num_samples_to_log=args.num_samples_to_log,
        log_every_n_epochs=args.log_every_n_epochs,
        # The 8-channel contract puts Blue, Green, Red in the first three slots,
        # so the baseline's RGB indices still hold.
        rgb_band_indices=(2, 1, 0),
    )

    if args.no_wandb:
        from lightning.pytorch.loggers import CSVLogger
        logger = CSVLogger(save_dir=str(output_dir), name="csv")
    else:
        logger = WandbLogger(project=args.wandb_project, name=tag, save_dir=str(output_dir))

    callbacks = [
        # Same monitor and filename pattern as the baseline and KD runs, so the
        # checkpoints and W&B panels line up without special-casing.
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
    test_metrics = trainer.test(task, datamodule=dm,
                                ckpt_path=None if args.test_ckpt == "last" else "best")

    best = callbacks[0]
    record = {
        "tag": tag,
        "model": "phisatnet_student",
        "target_domain": args.target_domain,
        "in_channels": args.in_channels,
        "pretrained": not args.no_pretrained,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "checkpoint_spec": {
            "task": args.checkpoint_task, "training": args.checkpoint_training,
            "n_shots": args.checkpoint_nshots, "datetime": args.checkpoint_datetime,
        },
        "freeze": args.freeze,
        "bn_mode": args.bn_mode,
        "gamma_fill": args.gamma_fill,
        "reinit_decoder": not args.warm_start_decoder,
        "seed": args.seed,
        # The whole optimisation and data configuration, not just the parts that
        # vary today. A row in a comparison table is only defensible if the run
        # record can be diffed against the row next to it -- and the first thing
        # that went wrong here was a run that differed from the baseline in its
        # task loss without that being visible in its record.
        "max_samples": args.max_samples,
        "val_max_samples": args.val_max_samples,
        "test_max_samples": args.test_max_samples,
        "augment": not args.no_augment,
        "task_loss": args.task_loss,
        "focal_gamma": args.focal_gamma,
        "class_weights": args.class_weights,
        "class_weight_beta": args.class_weight_beta,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "epochs_requested": args.epochs,
        "patience": args.patience,
        "test_ckpt": args.test_ckpt,
        "epochs_run": int(trainer.current_epoch),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_params": sum(p.numel() for p in model.parameters()),
        "best_ckpt": str(best.best_model_path),
        "best_val_iou": float(best.best_model_score) if best.best_model_score is not None else None,
        "load_report": {k: v for k, v in getattr(model, "load_report", {}).items()
                        if k != "missing"},
        "test": {k: float(v) for k, v in (test_metrics[0] if test_metrics else {}).items()},
    }
    metrics_out = args.metrics_out or (output_dir / "metrics.json")
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"\nMetrics written to {metrics_out}")
    print(f"  test mIoU (aggregate) : {record['test'].get('test/student_target_iou')}")
    print(f"  test mIoU (fixed denom): {record['test'].get('test/student_target_iou_fixed')}")


if __name__ == "__main__":
    main()
