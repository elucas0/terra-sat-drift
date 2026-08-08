"""
Supervised segmentation baseline: the student U-Net trained directly on one
sensor, with no teacher and no distillation.

This is the control the cross-sensor KD results have to be read against. If
training the same architecture directly on labelled PhiSat-2 matches or beats the
distilled student, then the teacher, the triplets and the domain-adaptation
machinery are not earning their keep, and the honest conclusion is that the task
is simply supervised segmentation with a class-imbalance problem.

To keep the comparison fair, everything that is not the intervention is held
identical to ``kd_contrastive_module``: architecture, per-domain normalisation,
seed-42 split, geometric augmentation, task loss (focal / class weights),
optimiser, LR schedule, early-stopping monitor, and **metric key names** -- the
last of these so ``scripts/analyze_kd_runs.py`` reads these runs unchanged.

Metrics are logged as ``{stage}/student_target_iou`` etc. even though there is no
"student" and no "target" here; the names are deliberately borrowed so the two
families of runs land on the same axes in W&B and in the comparison script.
"""

from pathlib import Path
from typing import Optional, Sequence

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchmetrics
from lightning.pytorch.loggers import WandbLogger
from torchmetrics import ClasswiseWrapper
from torchmetrics.classification import MulticlassF1Score, MulticlassJaccardIndex

from .losses.segmentation import focal_ce_loss

try:  # imported two ways in this repo; see kd_module for the same shim
    from ..dataset.plot_utils import build_palette, labels_to_rgb
except ImportError:  # pragma: no cover
    from dataset.plot_utils import build_palette, labels_to_rgb


class SupervisedSegmentationModule(pl.LightningModule):
    """Plain supervised segmentation on a single sensor."""

    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        ignore_index: int = -1,
        task_loss: str = "ce",
        focal_gamma: float = 2.0,
        class_weights: Optional[Sequence[float]] = None,
        lr_monitor: str = "val/student_target_iou",
        lr_monitor_mode: str = "max",
        rgb_band_indices: Sequence[int] = (2, 1, 0),
        num_samples_to_log: int = 4,
        log_every_n_epochs: int = 1,
        class_names: Optional[list[str]] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.ignore_index = ignore_index
        self.lr_monitor = lr_monitor
        self.lr_monitor_mode = lr_monitor_mode
        self.rgb_band_indices = tuple(rgb_band_indices)
        self.num_samples_to_log = num_samples_to_log
        self.log_every_n_epochs = max(log_every_n_epochs, 1)
        self.class_names = class_names

        if task_loss not in ("ce", "focal"):
            raise ValueError(f"task_loss must be 'ce' or 'focal', got {task_loss!r}")
        self.task_loss = task_loss
        self.focal_gamma = focal_gamma
        self.register_buffer(
            "class_weight",
            torch.as_tensor(list(class_weights), dtype=torch.float32)
            if class_weights is not None else torch.empty(0),
            persistent=True,
        )
        if self.class_weight.numel() and self.class_weight.numel() != num_classes:
            raise ValueError(
                f"class_weights has {self.class_weight.numel()} entries, expected {num_classes}")

        self.metrics = nn.ModuleDict()
        for stage in ("train", "val", "test"):
            self.metrics[f"{stage}_acc"] = torchmetrics.Accuracy(
                task="multiclass", num_classes=num_classes, ignore_index=ignore_index)
            self.metrics[f"{stage}_iou"] = torchmetrics.JaccardIndex(
                task="multiclass", num_classes=num_classes, ignore_index=ignore_index)
            self.metrics[f"{stage}_f1"] = MulticlassF1Score(
                num_classes=num_classes, ignore_index=ignore_index, average="macro")
            self.metrics[f"{stage}_iou_per_class"] = ClasswiseWrapper(
                MulticlassJaccardIndex(num_classes=num_classes, ignore_index=ignore_index,
                                       average=None),
                labels=self.class_names,
                prefix=f"{stage}_per_class/IoU_",
            )
            self.register_buffer(f"_support_{stage}", torch.zeros(num_classes), persistent=False)

    # ------------------------------------------------------------------
    def forward(self, x):
        return self.model(x)

    def _task_loss(self, logits, mask):
        alpha = self.class_weight if self.class_weight.numel() else None
        return focal_ce_loss(
            logits, mask,
            gamma=self.focal_gamma if self.task_loss == "focal" else 0.0,
            alpha=alpha, ignore_index=self.ignore_index,
        )

    def _shared_step(self, batch, prefix="train"):
        x = batch["image_target"]
        mask = batch["mask"].long()
        logits = self.model(x)
        if logits.shape[-2:] != mask.shape[-2:]:
            logits = torch.nn.functional.interpolate(
                logits, size=mask.shape[-2:], mode="bilinear", align_corners=False)

        loss = self._task_loss(logits, mask)
        self.log(f"{prefix}/loss", loss, prog_bar=True, sync_dist=True)
        self.log(f"{prefix}/loss_task_target", loss, sync_dist=True)

        for name, key in (("acc", "acc"), ("iou", "iou"), ("f1", "f1")):
            self.metrics[f"{prefix}_{key}"](logits, mask)
            # Borrowed key names, so the KD comparison script reads these runs.
            self.log(f"{prefix}/student_target_{name}", self.metrics[f"{prefix}_{key}"],
                     on_step=False, on_epoch=True, prog_bar=(name == "iou"))
        self.metrics[f"{prefix}_iou_per_class"].update(logits, mask)
        with torch.no_grad():
            valid = mask[mask != self.ignore_index]
            if valid.numel():
                buf = getattr(self, f"_support_{prefix}")
                buf += torch.bincount(valid.flatten(), minlength=self.num_classes).to(buf.dtype)

        return {"loss": loss, "image": x, "mask": mask, "pred": logits.argmax(1),
                "patch_index": batch.get("patch_index")}

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")["loss"]

    def on_validation_epoch_start(self):
        """Vary which batch gets plotted; the val loader is not shuffled, so a
        fixed index would show the same handful of patches every epoch."""
        n = getattr(self.trainer, "num_val_batches", None) if self.trainer else None
        if isinstance(n, (list, tuple)):
            n = n[0] if n else 0
        self._plot_batch_idx = (
            int(np.random.default_rng(self.current_epoch).integers(0, n))
            if isinstance(n, int) and n > 0 else 0)

    def validation_step(self, batch, batch_idx):
        out = self._shared_step(batch, "val")
        if (batch_idx == getattr(self, "_plot_batch_idx", 0)
                and self.trainer is not None and self.trainer.is_global_zero
                and self.current_epoch % self.log_every_n_epochs == 0):
            self._log_qualitative(out)
        return out["loss"]

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")["loss"]

    def _epoch_end(self, prefix):
        metric = self.metrics[f"{prefix}_iou_per_class"]
        per_class = metric.compute()
        self.log_dict(per_class, sync_dist=True)
        # Constant-denominator macro mIoU; see kd_contrastive_module._epoch_end
        # for why the aggregate `student_target_iou` is not safe to compare
        # across models when a class can be absent from the eval split.
        self.log(f"{prefix}/student_target_iou_fixed",
                 torch.stack(list(per_class.values())).mean(), sync_dist=True)
        metric.reset()
        support = getattr(self, f"_support_{prefix}")
        names = self.class_names or [str(i) for i in range(self.num_classes)]
        self.log_dict({f"{prefix}_per_class/pixels_{n}": support[i] for i, n in enumerate(names)},
                      reduce_fx="sum", sync_dist=True)
        support.zero_()

    def on_train_epoch_end(self):
        self._epoch_end("train")

    def on_validation_epoch_end(self):
        self._epoch_end("val")

    def on_test_epoch_end(self):
        self._epoch_end("test")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode=self.lr_monitor_mode, factor=0.5, patience=5)
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "monitor": self.lr_monitor}}

    # ------------------------------------------------------------------
    def _labels_to_rgb(self, label_map):
        """Shared colour table, so these figures match the KD runs' exactly."""
        return labels_to_rgb(label_map, build_palette(self.num_classes))

    def _to_rgb(self, img):
        idx = [i for i in self.rgb_band_indices if i < img.shape[0]]
        rgb = img[idx].detach().float().cpu().numpy().transpose(1, 2, 0)
        lo, hi = np.percentile(rgb, 2, (0, 1), keepdims=True), np.percentile(rgb, 98, (0, 1), keepdims=True)
        return np.clip((rgb - lo) / np.clip(hi - lo, 1e-6, None), 0, 1)

    def _log_qualitative(self, out, split="val"):
        if self.logger is None:
            return
        b = out["image"].shape[0]
        n = min(self.num_samples_to_log, b)
        rows = np.random.default_rng(self.current_epoch + 1).choice(b, n, replace=False)
        fig, axes = plt.subplots(n, 3, figsize=(8, 2.7 * n), squeeze=False)
        for r, i in enumerate(rows):
            for c, (title, arr) in enumerate((
                ("PhiSat-2", self._to_rgb(out["image"][i])),
                ("Ground truth", self._labels_to_rgb(out["mask"][i].cpu().numpy())),
                ("Prediction", self._labels_to_rgb(out["pred"][i].cpu().numpy())),
            )):
                axes[r][c].imshow(arr)
                axes[r][c].axis("off")
                if r == 0:
                    axes[r][c].set_title(title, fontsize=9)
            if out.get("patch_index") is not None:
                axes[r][0].text(0.02, 0.98, f"#{int(out['patch_index'][i])}",
                                transform=axes[r][0].transAxes, va="top", fontsize=6,
                                color="white",
                                bbox=dict(facecolor="black", alpha=0.5, edgecolor="none", pad=1.5))
        fig.suptitle(f"{split} — epoch {self.current_epoch}")
        fig.tight_layout()
        if WandbLogger is not None and isinstance(self.logger, WandbLogger):
            import wandb
            self.logger.experiment.log({f"{split}/predictions": wandb.Image(fig),
                                        "epoch": self.current_epoch})
        elif hasattr(self.logger, "experiment") and hasattr(self.logger.experiment, "add_figure"):
            self.logger.experiment.add_figure(f"{split}/predictions", fig, self.global_step)
        else:
            d = Path(getattr(self.trainer, "default_root_dir", ".")) / "qualitative"
            d.mkdir(parents=True, exist_ok=True)
            fig.savefig(d / f"{split}_epoch{self.current_epoch}.png")
        plt.close(fig)
