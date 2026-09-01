"""
Downstream validation of the distilled student: flood segmentation on simulated
PhiSat-2 Sen1Floods11, with the encoder frozen.

This is a transfer probe, not a new model. The student's encoder is frozen and
only the decoder (and a freshly initialised 2-class head) are trained, so the
score measures how much flood-relevant structure the LULC-distilled
representation already contains. It is the protocol used for the segmentation
benchmarks in X-STARS [Marsocci25] -- freeze the pretrained encoder, fine-tune a
decoder -- and it answers a different question from end-to-end fine-tuning.

    # default probe: encoder frozen, decoder + head trained
    python terra_sat_drift/finetune_floods_student.py \
        --student-ckpt outputs/xsensor_kd_terramind_v1_base_kd1.0_inst0.0_pix0.0_dsbn/checkpoints/best-89-0.4607.ckpt

    # strict linear probe: everything frozen except the 1x1 head (66 parameters)
    python terra_sat_drift/finetune_floods_student.py --freeze backbone ...

    # control: same architecture, random initialisation, nothing frozen
    python terra_sat_drift/finetune_floods_student.py --scratch --freeze none ...

    # one rung of the degradation ladder; batch_probe_ladder.py sweeps all of them
    python terra_sat_drift/finetune_floods_student.py \
        --sim-root /shared/home/elucas/datasets/sen1floods11_ladder/psf_l3

Run the ``--scratch`` control as well. Without it the probe score is
uninterpretable: a decoder with 5.1M trainable parameters can reach a
respectable flood IoU on its own, so only the difference against a randomly
initialised encoder isolates what the distillation actually contributed.

Data notes, all verified against the files rather than assumed:

* The simulated scenes are 8-band and 1077x1077 at 4.75 m; the hand labels come
  from the *original* Sen1Floods11 release and are 512x512 at 10 m. Labels are
  therefore upsampled (nearest) to the image grid rather than the image being
  downsampled, which keeps the input at the 4.75 m scale the student was
  distilled at.
* Band index 3 is panchromatic (its median sits between Red and RE1) and is
  dropped, leaving Blue, Green, Red, RE1, RE2, RE3, NIR -- the order the student
  consumes.
* Pixel values are scaled reflectance, the same radiometric family as the
  Sentinel-2 branch of the triplets, so ``DOMAIN_STATS["sim"]`` is the correct
  normalisation and the source DSBN branch is the matching one.
* Label -1 is Sen1Floods11's no-data marker and is kept as ``ignore_index``.
  Note the earlier ``train_validate_floods_simulated.py`` mapped it to class 0
  instead, which counts unobserved pixels as confidently "not water".
"""

import argparse
import json
import warnings
from pathlib import Path

import albumentations as A
import lightning.pytorch as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset

from dataset.dataset_paired_triplets_lulc import DOMAIN_STATS, normalize_domain
from model_tasks.baseline_module import SupervisedSegmentationModule
from model_tasks.domain_adaptation.dsbn import convert_to_dsbn, set_domain
from model_tasks.losses.segmentation import class_weights
from student_mobilenet import create_student_model

warnings.filterwarnings("ignore")

DEFAULT_SIM_ROOT = Path("/shared/home/elucas/datasets/sen1floods11_simulated_alt_v5/v1.1")
LABEL_ROOT = Path("/shared/home/elucas/datasets/sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand")
IMG_SUBDIR = "data/flood_events/HandLabeled/S2Hand"
SPLIT_SUBDIR = "splits/flood_handlabeled"


def resolve_sim_root(path: Path | str) -> Path:
    """Accept either a dataset root or its ``v1.1`` directory.

    The degradation-ladder variants are laid out exactly like the ``alt_v*``
    datasets, so pointing at either level should work.
    """
    path = Path(path)
    return path / "v1.1" if (path / "v1.1").is_dir() else path

PAN_INDEX = 3                       # verified from per-band medians
BANDS = [0, 1, 2, 4, 5, 6, 7]       # -> Blue, Green, Red, RE1, RE2, RE3, NIR
CLASS_NAMES = ["not_water", "water"]
SOURCE_DOMAIN_ID, TARGET_DOMAIN_ID = 0, 1


class SimulatedFloodDataset(Dataset):
    """Simulated PhiSat-2 Sen1Floods11 scenes paired with the original hand labels."""

    # The encoder downsamples by 16, so a full scene has to be padded up to a multiple
    # of it. The padding is excluded from the metric (see ``_pad_to_multiple``).
    DOWNSAMPLE = 16

    def __init__(self, split: str, stats: dict, crop: int = 256, train: bool = True,
                 sim_root: Path | str = DEFAULT_SIM_ROOT, full_scene: bool = False):
        sim_root = resolve_sim_root(sim_root)
        self.img_dir = sim_root / IMG_SUBDIR
        split_file = sim_root / SPLIT_SUBDIR / f"flood_{split}_data.txt"
        self.ids = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
        missing = [i for i in self.ids if not (self.img_dir / f"simulated_L1C_{i}_S2Hand.tif").exists()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} scenes in {split_file.name} have no image, e.g. {missing[:3]}")
        self.stats = stats
        self.train = train
        # Evaluating the whole scene instead of a centre crop, so that this probe and the
        # TerraMind probe score the identical set of labelled pixels and their domain gaps
        # can be compared directly rather than only in relative terms.
        self.full_scene = full_scene and not train
        # Geometric augmentation only: the simulator has already applied the
        # radiometric perturbation this probe is meant to be robust to, and
        # photometric jitter on top would confound it.
        if train:
            eval_ops = [A.RandomCrop(crop, crop), A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5)]
        elif self.full_scene:
            eval_ops = []  # whole scene; padded to a stride multiple in __getitem__
        else:
            eval_ops = [A.CenterCrop(crop, crop)]
        self.transform = A.Compose(eval_ops)
        # 34% of Sen1Floods11 label pixels are no-data, and at crop 256 roughly
        # one training crop in six lands entirely inside a no-data region. Such
        # a crop carries no supervision at all, so resample it rather than spend
        # a batch slot on it. Validation and test keep the deterministic centre
        # crop, since resampling them would change the evaluation set.
        self.crop_retries = 10 if train else 0

    def _pad_to_multiple(self, image: np.ndarray, mask: np.ndarray):
        """Pad a full scene up to the encoder's stride, without adding scored pixels.

        The image is reflected so the convolutions near the border see plausible
        content; the mask is padded with the ignore value, so the padded region
        contributes to neither the loss nor the IoU. The pixels actually scored are
        therefore exactly the scene's own, which is what makes this metric comparable
        to the TerraMind probe's full-scene number.
        """
        h, w = mask.shape
        ph = (-h) % self.DOWNSAMPLE
        pw = (-w) % self.DOWNSAMPLE
        if not ph and not pw:
            return image, mask
        image = np.pad(image, ((0, ph), (0, pw), (0, 0)), mode="reflect")
        mask = np.pad(mask, ((0, ph), (0, pw)), mode="constant", constant_values=-1)
        return image, mask

    def class_frequencies(self) -> np.ndarray:
        """Valid-pixel class counts over the split, for optional class weighting."""
        counts = np.zeros(2, dtype=np.int64)
        for sid in self.ids:
            with rasterio.open(LABEL_ROOT / f"{sid}_LabelHand.tif") as src:
                m = src.read(1)
            counts += np.bincount(m[m >= 0].ravel(), minlength=2)
        return counts

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> dict:
        sid = self.ids[idx]
        with rasterio.open(self.img_dir / f"simulated_L1C_{sid}_S2Hand.tif") as src:
            img = src.read().astype(np.float32)[BANDS]           # (7, H, W)
        with rasterio.open(LABEL_ROOT / f"{sid}_LabelHand.tif") as src:
            mask = src.read(1).astype(np.int64)                  # (512, 512)

        # Labels are at the original 10 m grid; bring them up to the simulated
        # 4.75 m grid so the input keeps the scale the student was trained on.
        if mask.shape != img.shape[1:]:
            m = torch.as_tensor(mask)[None, None].float()
            mask = torch.nn.functional.interpolate(
                m, size=img.shape[1:], mode="nearest")[0, 0].long().numpy()

        img = normalize_domain(img, self.stats)                  # sqrt -> clip -> z-score
        hwc = np.ascontiguousarray(img.transpose(1, 2, 0))
        out = self.transform(image=hwc, mask=mask)
        for _ in range(self.crop_retries):                       # see crop_retries above
            if (out["mask"] >= 0).any():
                break
            out = self.transform(image=hwc, mask=mask)
        if self.full_scene:
            out = dict(out)
            out["image"], out["mask"] = self._pad_to_multiple(out["image"], out["mask"])
        return {
            "image_target": torch.as_tensor(np.ascontiguousarray(out["image"].transpose(2, 0, 1))).float(),
            "mask": torch.as_tensor(np.ascontiguousarray(out["mask"])).long(),
            # Carried under the key the base module already forwards to the
            # qualitative plot, so the figure can name the scene it drew.
            "patch_index": torch.as_tensor(idx).long(),
        }


class ProbeStudent(nn.Module):
    """Selects the DSBN branch and holds the frozen submodules in eval mode.

    The eval-mode part matters and is easy to miss: ``requires_grad_(False)``
    stops the weights being updated but does **not** stop BatchNorm from
    updating its running statistics, so a nominally frozen encoder would still
    drift towards the flood dataset. Overriding ``train()`` re-applies eval to
    the frozen modules every time Lightning switches the model to train mode.
    """

    def __init__(self, student: nn.Module, domain_id: int, frozen: list[nn.Module], is_dsbn: bool):
        super().__init__()
        self.student = student
        self.domain_id = domain_id
        self.is_dsbn = is_dsbn
        self._frozen = frozen
        for module in frozen:
            for p in module.parameters():
                p.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        for module in self._frozen:
            module.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_dsbn:
            set_domain(self.student, self.domain_id)
        return self.student(x)


# Flood palette. Water reuses the WorldCover permanent-water blue so these
# figures sit alongside the LULC ones, and land is a recessive neutral rather
# than a second saturated hue -- the two-class fallback palette assigns two
# near-identical light blues, which is unreadable for a water mask.
FLOOD_COLORS = {"nodata": "#000000", "land": "#e3e3e0", "water": "#0064c8"}
# Error map: correct pixels stay recessive so the mistakes are what the eye
# lands on. Missed water and false alarms get distinct hue *and* lightness, and
# both are named in the legend, so identity is never carried by colour alone.
ERROR_COLORS = {"nodata": "#000000", "correct_land": "#e3e3e0",
                "correct_water": "#0064c8", "missed_water": "#e34948",
                "false_alarm": "#eda100"}


class FloodProbeModule(SupervisedSegmentationModule):
    """Only optimises the parameters that were left trainable.

    Also replaces the inherited qualitative plot: for a two-class task the
    generic palette is unusable, and an explicit error map is far more
    informative than a bare prediction mask when the metric of interest is
    water IoU.
    """

    def __init__(self, *args, scene_ids: list[str] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.scene_ids = scene_ids or []

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode=self.lr_monitor_mode, factor=0.5, patience=5)
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "monitor": self.lr_monitor}}

    # ------------------------------------------------------------------
    @staticmethod
    def _mask_to_rgb(mask: np.ndarray) -> np.ndarray:
        from matplotlib.colors import to_rgb
        out = np.full((*mask.shape, 3), to_rgb(FLOOD_COLORS["nodata"]), dtype=np.float32)
        out[mask == 0] = to_rgb(FLOOD_COLORS["land"])
        out[mask == 1] = to_rgb(FLOOD_COLORS["water"])
        return out

    @staticmethod
    def _error_to_rgb(mask: np.ndarray, pred: np.ndarray) -> np.ndarray:
        from matplotlib.colors import to_rgb
        out = np.full((*mask.shape, 3), to_rgb(ERROR_COLORS["nodata"]), dtype=np.float32)
        valid = mask >= 0
        out[valid & (mask == 0) & (pred == 0)] = to_rgb(ERROR_COLORS["correct_land"])
        out[valid & (mask == 1) & (pred == 1)] = to_rgb(ERROR_COLORS["correct_water"])
        out[valid & (mask == 1) & (pred == 0)] = to_rgb(ERROR_COLORS["missed_water"])
        out[valid & (mask == 0) & (pred == 1)] = to_rgb(ERROR_COLORS["false_alarm"])
        return out

    def _log_qualitative(self, out: dict, split: str = "val") -> None:
        if self.logger is None:
            return
        from matplotlib.patches import Patch

        batch = out["image"].shape[0]
        n = min(self.num_samples_to_log, batch)
        rows = np.random.default_rng(self.current_epoch + 1).choice(batch, n, replace=False)

        fig, axes = plt.subplots(n, 4, figsize=(11, 2.9 * n), squeeze=False)
        for r, i in enumerate(rows):
            m = out["mask"][i].cpu().numpy()
            p = out["pred"][i].cpu().numpy()
            panels = [
                ("Simulated $\\Phi$-sat-2", self._to_rgb(out["image"][i])),
                ("Ground truth", self._mask_to_rgb(m)),
                ("Prediction", self._mask_to_rgb(p)),
                ("Errors", self._error_to_rgb(m, p)),
            ]
            for c, (name, arr) in enumerate(panels):
                axes[r][c].imshow(arr)
                axes[r][c].axis("off")
                if r == 0:
                    axes[r][c].set_title(name, fontsize=9)
            # Per-scene water IoU makes the row self-describing: a scene with no
            # water at all is not a failure, and the number says which is which.
            valid = m >= 0
            inter = int((valid & (m == 1) & (p == 1)).sum())
            union = int((valid & ((m == 1) | (p == 1))).sum())
            sid = (self.scene_ids[int(out["patch_index"][i])]
                   if out.get("patch_index") is not None and self.scene_ids else "")
            iou = f"water IoU {inter/union:.2f}" if union else "no water in crop"
            axes[r][0].text(0.02, 0.98, f"{sid}\n{iou}".strip(),
                            transform=axes[r][0].transAxes, va="top", fontsize=6,
                            color="white",
                            bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=1.5))

        handles = [Patch(facecolor=ERROR_COLORS["correct_water"], edgecolor="#52514e",
                         linewidth=0.5, label="water, correct"),
                   Patch(facecolor=ERROR_COLORS["missed_water"], edgecolor="#52514e",
                         linewidth=0.5, label="water, missed"),
                   Patch(facecolor=ERROR_COLORS["false_alarm"], edgecolor="#52514e",
                         linewidth=0.5, label="false alarm"),
                   Patch(facecolor=ERROR_COLORS["correct_land"], edgecolor="#52514e",
                         linewidth=0.5, label="land, correct"),
                   Patch(facecolor=ERROR_COLORS["nodata"], edgecolor="#52514e",
                         linewidth=0.5, label="no data (ignored)")]
        fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=8)
        fig.suptitle(f"{split} - epoch {self.current_epoch}", x=0.01, ha="left", fontsize=10)
        fig.tight_layout(rect=(0, 0.05, 1, 0.97))

        if WandbLogger is not None and isinstance(self.logger, WandbLogger):
            import wandb
            self.logger.experiment.log({f"{split}/predictions": wandb.Image(fig),
                                        "epoch": self.current_epoch})
        elif hasattr(self.logger, "experiment") and hasattr(self.logger.experiment, "add_figure"):
            self.logger.experiment.add_figure(f"{split}/predictions", fig, self.global_step)
        else:
            d = Path(getattr(self.trainer, "default_root_dir", ".")) / "qualitative"
            d.mkdir(parents=True, exist_ok=True)
            fig.savefig(d / f"{split}_epoch{self.current_epoch}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)


def build_model(args):
    student = create_student_model(in_channels=7, num_classes=11, pretrained=False)
    is_dsbn = False

    if not args.scratch:
        ck = torch.load(args.student_ckpt, map_location="cpu", weights_only=False)
        sd = {k[len("student."):]: v for k, v in ck["state_dict"].items() if k.startswith("student.")}
        is_dsbn = any(".bns." in k for k in sd)
        print(f"checkpoint epoch {ck.get('epoch')} | DSBN student: {is_dsbn}")
        if is_dsbn:
            convert_to_dsbn(student, num_domains=2)
        missing, unexpected = student.load_state_dict(sd, strict=True)
        print(f"  loaded {len(sd)} tensors; missing={list(missing)} unexpected={list(unexpected)}")
        del ck, sd
    else:
        print("--scratch: random initialisation, no distilled weights loaded")

    # Fresh 2-class head; the 11-class LULC head is discarded.
    student.head = nn.Conv2d(student.decoder_out_channels, 2, kernel_size=1)

    frozen = {"encoder": [student.backbone.encoder],
              "backbone": [student.backbone],
              "none": []}[args.freeze]
    model = ProbeStudent(student, args.dsbn_domain, frozen, is_dsbn)

    total = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"freeze={args.freeze}: {train_p/1e6:.3f}M trainable of {total/1e6:.3f}M "
          f"({100*train_p/total:.1f}%)")
    return model


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--student-ckpt", type=str,
                   default="outputs/xsensor_kd_terramind_v1_base_kd1.0_inst0.0_pix0.0_dsbn"
                           "/checkpoints/best-89-0.4607.ckpt")
    p.add_argument("--scratch", action="store_true",
                   help="Random init instead of the distilled weights; the control that makes "
                        "the probe score interpretable.")
    p.add_argument("--freeze", type=str, default="encoder",
                   choices=["encoder", "backbone", "none"],
                   help="'encoder' trains decoder+head (the standard transfer protocol); "
                        "'backbone' is a strict linear probe on the 1x1 head; "
                        "'none' fine-tunes everything.")
    p.add_argument("--dsbn-domain", type=int, default=SOURCE_DOMAIN_ID, choices=[0, 1],
                   help="DSBN branch. The simulated scenes are scaled reflectance, the same "
                        "radiometric family as the Sentinel-2 branch, so 0 (source) matches "
                        "their statistics. Try 1 to test the PhiSat-2 branch.")
    p.add_argument("--crop", type=int, default=256,
                   help="Training crop size. 256 at 4.75 m covers the same ground as the "
                        "distillation patches, so the frozen features are used at the scale they "
                        "were learned. Also the val/test centre-crop size unless "
                        "--eval-full-scene is given.")
    p.add_argument("--eval-full-scene", action="store_true",
                   help="Evaluate val/test on the whole 1077x1077 scene instead of a centre crop, "
                        "padded up to the encoder stride with the pad labelled ignore. This is "
                        "what makes the score comparable to the TerraMind probe, which evaluates "
                        "full scenes; the centre-crop default scores only 23%% of each image.")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--eval-batch-size", type=int, default=None,
                   help="Batch size for val/test (default: --batch-size). Lower it for "
                        "full-scene evaluation if memory is tight.")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Higher than distillation's 1e-4: only a small head/decoder is training.")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--task-loss", type=str, default="ce", choices=["ce", "focal"])
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--class-weights", type=str, default="none",
                   choices=["none", "inverse", "inverse_sqrt"],
                   help="Water is only ~10%% of labelled pixels, so unweighted CE can settle "
                        "on predicting 'not water' almost everywhere. Frequencies are measured "
                        "from the training split at startup rather than hardcoded.")
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--domain-stats", type=str, default="sim", choices=["sim", "s2b", "real"],
                   help="Normalisation statistics. 'sim' matches these scenes' reflectance scale.")
    p.add_argument("--sim-root", type=Path, default=DEFAULT_SIM_ROOT,
                   help="Simulated dataset root, either the dataset directory or its v1.1 "
                        "sub-directory. Point this at a degradation-ladder variant "
                        "(e.g. .../sen1floods11_ladder/psf_l3) to probe one isolated factor.")
    p.add_argument("--test-roots", nargs="+", type=Path, default=None,
                   help="After training, evaluate the best checkpoint on the test split of each "
                        "of these roots as well. This is the domain-gap measurement: train on one "
                        "degradation condition, test on the others. Results are keyed by variant "
                        "name under 'cross_condition' in --metrics-out.")
    p.add_argument("--seed", type=int, default=42,
                   help="Seed for pl.seed_everything. Repeat a config under several seeds to "
                        "separate a real degradation effect from run-to-run variance.")
    p.add_argument("--metrics-out", type=Path, default=None,
                   help="Write the final val/test metrics to this JSON file, for batch "
                        "aggregation (see batch_probe_ladder.py).")
    p.add_argument("--output-root", type=Path,
                   default=Path("/shared/home/elucas/scratch/terra-sat-drift/outputs"))
    p.add_argument("--tag", type=str, default=None)
    p.add_argument("--wandb-project", type=str, default="terra-sat-drift")
    p.add_argument("--no-wandb", action="store_true",
                   help="Log to a local CSV instead of Weights & Biases.")
    return p.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)

    tag = args.tag or (f"floods_probe_{'scratch' if args.scratch else 'distilled'}"
                       f"_freeze-{args.freeze}_dom{args.dsbn_domain}")
    output_dir = args.output_root / tag
    output_dir.mkdir(parents=True, exist_ok=True)
    sim_root = resolve_sim_root(args.sim_root)
    print(f"Run tag: {tag}\nOutput:  {output_dir}\nData:    {sim_root}")

    stats = DOMAIN_STATS[args.domain_stats]
    loaders, datasets = {}, {}
    eval_bs = args.eval_batch_size or args.batch_size
    for split, train in (("train", True), ("valid", False), ("test", False)):
        ds = SimulatedFloodDataset(split, stats, crop=args.crop, train=train, sim_root=sim_root,
                                   full_scene=args.eval_full_scene)
        datasets[split] = ds
        loaders[split] = DataLoader(
            ds, batch_size=args.batch_size if train else eval_bs, shuffle=train,
            num_workers=args.num_workers, pin_memory=True, drop_last=train,
            persistent_workers=args.num_workers > 0)
        print(f"  {split:6s} {len(ds):4d} scenes")

    weights = None
    if args.class_weights != "none":
        counts = datasets["train"].class_frequencies()
        weights = class_weights(counts, scheme=args.class_weights).tolist()
        share = 100 * counts / counts.sum()
        print(f"  train label prior: not_water {share[0]:.1f}%, water {share[1]:.1f}% "
              f"-> weights {[round(w, 3) for w in weights]}")

    task = FloodProbeModule(
        model=build_model(args),
        num_classes=2,
        lr=args.lr,
        weight_decay=args.weight_decay,
        ignore_index=-1,
        task_loss=args.task_loss,
        focal_gamma=args.focal_gamma,
        class_weights=weights,
        class_names=CLASS_NAMES,
        rgb_band_indices=(2, 1, 0),
        num_samples_to_log=4,
        scene_ids=datasets["valid"].ids,
    )

    if args.no_wandb:
        from lightning.pytorch.loggers import CSVLogger
        logger = CSVLogger(save_dir=str(output_dir), name="csv")
    else:
        logger = WandbLogger(project=args.wandb_project, name=tag, save_dir=str(output_dir))
    callbacks = [
        ModelCheckpoint(dirpath=output_dir / "checkpoints",
                        filename="best-{epoch:02d}-{val/student_target_iou:.4f}",
                        monitor="val/student_target_iou", mode="max",
                        save_top_k=3, save_last=True, auto_insert_metric_name=False),
        EarlyStopping(monitor="val/student_target_iou", mode="max", patience=args.patience),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1, logger=logger, callbacks=callbacks,
        default_root_dir=str(output_dir), log_every_n_steps=10,
    )
    trainer.fit(task, train_dataloaders=loaders["train"], val_dataloaders=loaders["valid"])
    test_metrics = trainer.test(task, dataloaders=loaders["test"], ckpt_path="best")

    # Domain gap: the same trained checkpoint against other degradation conditions.
    # Each root is evaluated in its own trainer.test call rather than as a multi-dataloader
    # test, because the module keeps one metric object per stage and would otherwise
    # accumulate every test set into a single score.
    cross_condition = {}
    if args.test_roots:
        best_path = callbacks[0].best_model_path
        print(f"\nCross-condition evaluation of {best_path} on {len(args.test_roots)} test sets")
        for root in args.test_roots:
            root = resolve_sim_root(root)
            name = root.parent.name if root.name == "v1.1" else root.name
            ds = SimulatedFloodDataset("test", stats, crop=args.crop, train=False, sim_root=root,
                                       full_scene=args.eval_full_scene)
            dl = DataLoader(ds, batch_size=eval_bs, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
            res = trainer.test(task, dataloaders=dl, ckpt_path=best_path)
            cross_condition[name] = {k: float(v) for k, v in (res[0] if res else {}).items()}
            print(f"  {name:<14} water IoU "
                  f"{cross_condition[name].get('test_per_class/IoU_water', float('nan')):.4f}")

    if args.metrics_out:
        best = callbacks[0]
        record = {
            "tag": tag,
            "sim_root": str(sim_root),
            "seed": args.seed,
            "scratch": args.scratch,
            "freeze": args.freeze,
            "dsbn_domain": args.dsbn_domain,
            "eval_full_scene": args.eval_full_scene,
            "epochs_run": int(trainer.current_epoch),
            "best_ckpt": str(best.best_model_path),
            "best_val_iou": float(best.best_model_score) if best.best_model_score is not None else None,
            "test": {k: float(v) for k, v in (test_metrics[0] if test_metrics else {}).items()},
            "cross_condition": cross_condition,
        }
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_out.write_text(json.dumps(record, indent=2) + "\n")
        print(f"Metrics written to {args.metrics_out}")


if __name__ == "__main__":
    main()
