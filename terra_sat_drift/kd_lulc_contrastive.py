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

from dataset.constants import WC_CLASS_MAPPING, WC_CLASS_NAMES, WC_CLASS_PIXEL_FREQ
from model_tasks.losses.segmentation import class_weights
from dataset.datamodule_paired_triplets_lulc import PhisatPairedLULCDataModule
from model_tasks.kd_contrastive_module import CrossSensorKDModule
from student_mobilenet import create_student_model
from phisatnet_student import create_phisatnet_student, resolve_checkpoint

warnings.filterwarnings("ignore")

H5_IMAGES = "/shared/projects/phisat2/data/processed/triplets_v1/phisat2_s2b_dataset_v1.h5"
H5_LABELS = "/shared/projects/phisat2/data/processed/worldcover_all_clean_v1/worldcover_all_clean_labels_v1.h5"
MANIFEST = "/shared/projects/phisat2/data/processed/worldcover_all_clean_v1/worldcover_all_clean_manifest_v1.csv"

# Per-student input contract. The UNet student and the TerraMind teacher both
# consume the same seven S2L1C bands, so nothing has to be reordered between
# them. PhisatNet does not: its stem was built for eight bands in the PhilEO
# order (Blue, Green, Red, NIR, RE1, RE2, RE3, PAN), so the source view is
# served as `s2b8` and the teacher takes a gather back to its own seven-band
# order. Both models see the same pixels either way.
STUDENT_SPECS = {
    "unet": {
        "in_channels": 7,
        "target_domain": "real",
        "source_domain": "s2b",
        "teacher_band_indices": None,
    },
    "phisatnet": {
        "in_channels": 8,
        "target_domain": "real8",
        "source_domain": "s2b8",
        # s2b8 is [B, G, R, NIR, RE1, RE2, RE3, PAN]; the teacher declares
        # [BLUE, GREEN, RED, RED_EDGE_1, RED_EDGE_2, RED_EDGE_3, NIR_BROAD].
        "teacher_band_indices": [0, 1, 2, 4, 5, 6, 3],
    },
}


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
    p.add_argument("--val-max-samples", type=int, default=1000,
                   help="Validation patches. Fixed, NOT derived from --max-samples: "
                        "1000 is the smallest size containing all 11 classes. Keep it "
                        "constant across runs you intend to compare.")
    p.add_argument("--test-max-samples", type=int, default=None,
                   help="Test patches; default None = the full 25,323. Test runs once, "
                        "so it can afford to be thorough (snow/ice on 255 patches "
                        "instead of 14).")
    p.add_argument("--target-domain", type=str, default=None,
                   choices=["real", "sim", "real8", "sim8"],
                   help="Sensor the student must work on at deployment. Defaults to "
                        "whichever variant matches --student's channel contract.")
    p.add_argument("--source-domain", type=str, default=None, choices=["s2b", "s2b8"],
                   help="Sensor the teacher is trusted on. Defaults per --student.")
    p.add_argument("--no-augment", action="store_true",
                   help="Disable the joint geometric augmentation.")
    # teacher / student
    p.add_argument("--backbone", type=str, default="terramind_v1_tiny")
    p.add_argument("--teacher-ckpt", type=str, default=None)
    p.add_argument("--student", type=str, default="unet", choices=["unet", "phisatnet"],
                   help="Architecture to distil into. 'unet' is the 9.85M student the "
                        "published runs use; 'phisatnet' is the 0.35M PhiSat-2 model, "
                        "which also switches the data to its 8-band contract.")
    p.add_argument("--student-pretrained", action="store_true",
                   help="PhisatNet only: start from a published checkpoint instead of a "
                        "random initialisation. OFF by default, because the UNet student "
                        "has no pretrained weights available and a random init is what "
                        "makes the two students' conditions differ only in architecture.")
    p.add_argument("--student-ckpt-task", type=str, default="lc")
    p.add_argument("--student-ckpt-training", type=str, default="finetuning",
                   choices=["finetuning", "linear_probing"])
    p.add_argument("--student-ckpt-nshots", type=int, default=5000)
    p.add_argument("--student-ckpt-datetime", type=int, default=None,
                   help="Pin a release date; the published encoders differ between "
                        "releases, so leave this set for anything reproducible.")
    p.add_argument("--student-weights-dir", type=str,
                   default="/shared/home/elucas/scratch/terra-sat-drift/weights/hydranet")
    # loss weights
    p.add_argument("--w-task-target", type=float, default=1.0)
    p.add_argument("--w-task-source", type=float, default=1.0)
    p.add_argument("--w-kd", type=float, default=1.0)
    p.add_argument("--w-instance", type=float, default=0.5, help="Cross-sensor NT-Xent weight.")
    p.add_argument("--w-pixel", type=float, default=0.1, help="Semantic-guided pixel contrast weight.")
    p.add_argument("--w-crd", type=float, default=0.0, help="CRD-style teacher/student contrast weight.")
    # task loss (class imbalance)
    p.add_argument("--task-loss", type=str, default="ce", choices=["ce", "focal"],
                   help="Focal down-weights already-confident pixels so gradient mass "
                        "moves to the rare classes plain CE never learns.")
    p.add_argument("--focal-gamma", type=float, default=2.0,
                   help="Focusing parameter; 0 == weighted CE, 2.0 is the RetinaNet default.")
    p.add_argument("--class-weights", type=str, default="none",
                   choices=["none", "inverse", "inverse_sqrt", "effective"],
                   help="Per-class weights from the measured WorldCover pixel frequencies. "
                        "'inverse_sqrt' is the recommended start; 'inverse' is ~60x "
                        "moss-vs-tree here and destabilises easily.")
    p.add_argument("--class-weight-beta", type=float, default=0.999,
                   help="beta for --class-weights effective (Cui et al. 2019).")
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
    p.add_argument("--label-smoothing", type=float, default=0.0,
                   help="Label smoothing for instance contrastive loss (default 0.0 = none).")
    # architecture
    p.add_argument("--concat-batch", action="store_true",
                   help="Forward both sensors as one concatenated batch so a SHARED "
                        "BatchNorm sees a mixed batch. This is the control for --use-dsbn: "
                        "with separate passes and shared BatchNorm the two domains are "
                        "normalised by their own batch statistics in training but by a "
                        "blend of both at inference, and this removes that inconsistency. "
                        "Mutually exclusive with --use-dsbn.")
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

    spec = STUDENT_SPECS[args.student]
    # Explicit flags win, but a domain whose channel count the student cannot
    # consume is a silent corruption rather than an error, so it is refused here.
    target_domain = args.target_domain or spec["target_domain"]
    source_domain = args.source_domain or spec["source_domain"]
    eight_band = {"real8", "sim8", "s2b8"}
    if (target_domain in eight_band) != (spec["in_channels"] == 8):
        raise SystemExit(
            f"--student {args.student} takes {spec['in_channels']} channels, which does "
            f"not match --target-domain {target_domain}. Use "
            f"{spec['target_domain']}/{spec['source_domain']}, or leave both unset."
        )
    if (source_domain in eight_band) != (spec["in_channels"] == 8):
        raise SystemExit(
            f"--student {args.student} takes {spec['in_channels']} channels, which does "
            f"not match --source-domain {source_domain}."
        )

    # The teacher checkpoint goes in the tag. Several teachers live side by side
    # under one directory as `best-val_mIoU.ckpt`, `-v6`, and so on, and they are
    # not the same model: measured on the test split they differ by ~0.07 mIoU.
    # A run tagged only by its loss weights gives no way to tell from the output
    # directory which teacher produced it, and a student distilled from the wrong
    # one looks like a worse method rather than a worse experiment.
    teacher_tag = Path(args.teacher_ckpt).stem if args.teacher_ckpt else "noteacher"
    tag = args.run_name or (
        f"xsensor_kd_{args.student}_{args.backbone}_kd{args.w_kd}_inst{args.w_instance}"
        f"_pix{args.w_pixel}{'_dsbn' if args.use_dsbn else ''}"
        f"{'_pre' if (args.student == 'phisatnet' and args.student_pretrained) else ''}"
        # The label budget too, for the same reason as the teacher: the
        # label-scarcity arm and the full-budget arm are different experiments
        # and must not land in one directory.
        f"_n{args.max_samples if args.max_samples else 'full'}"
        f"_T-{teacher_tag}"
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
        val_max_samples=args.val_max_samples,
        test_max_samples=args.test_max_samples,
        target_domain=target_domain,
        source_domain=source_domain,
    )

    # ---------------------------------------------------------
    # 2. Models
    # ---------------------------------------------------------
    num_classes = len(WC_CLASS_MAPPING)

    print(f"Building teacher ({args.backbone})...")
    teacher = build_teacher_model(
        backbone=args.backbone, num_classes=num_classes, ckpt_path=args.teacher_ckpt
    )

    if args.student == "unet":
        print("Building student (UNet)...")
        student = create_student_model(in_channels=7, num_classes=num_classes,
                                       pretrained=False)
    else:
        checkpoint = None
        if args.student_pretrained:
            checkpoint = resolve_checkpoint(
                task=args.student_ckpt_task,
                training=args.student_ckpt_training,
                n_shots=args.student_ckpt_nshots,
                datetime=args.student_ckpt_datetime,
                weights_dir=args.student_weights_dir,
            )
        print(f"Building student (PhisatNet, "
              f"{'pretrained' if checkpoint else 'random init'})...")
        student = create_phisatnet_student(
            num_classes=num_classes,
            checkpoint=checkpoint,
            in_channels=spec["in_channels"],
            # Distillation trains the whole student, so nothing is frozen and the
            # BatchNorm mode is inert -- DSBN, if enabled, is applied by the task
            # module after this returns.
            freeze="none",
            reinit_decoder=not args.student_pretrained,
        )
    n_params = sum(q.numel() for q in student.parameters())
    print(f"  student parameters: {n_params/1e6:.3f}M")

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
        task_loss=args.task_loss,
        focal_gamma=args.focal_gamma,
        # `.tolist()`, not the raw numpy array: `save_hyperparameters` stores this
        # in every checkpoint, and torch >= 2.6 loads with weights_only=True by
        # default, which refuses to unpickle numpy objects. Passing the array
        # makes every checkpoint this script writes unloadable via `ckpt_path=`,
        # which is why the earlier KD runs could only ever be tested on their
        # final-epoch weights. `train_baseline_phisat2.py` already carried this
        # fix; the KD path did not.
        class_weights=(
            None if args.class_weights == "none"
            else class_weights(WC_CLASS_PIXEL_FREQ, scheme=args.class_weights,
                               beta=args.class_weight_beta).tolist()
        ),
        kd_mode=args.kd_mode,
        kd_temperature=args.kd_temperature,
        label_smoothing=args.label_smoothing,
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
        concat_batch=args.concat_batch,
        ignore_index=-1,
        student_in_channels=spec["in_channels"],
        teacher_band_indices=spec["teacher_band_indices"],
        # Blue, Green, Red are the first three channels in every domain served
        # here, 7-band and 8-band alike.
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
