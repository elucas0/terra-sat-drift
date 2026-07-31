from pathlib import Path
from typing import Optional, Sequence

import matplotlib

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import lightning.pytorch as pl

from matplotlib.colors import to_rgb
from torchmetrics import ClasswiseWrapper
from torchmetrics.classification import MulticlassF1Score, MulticlassJaccardIndex
from lightning.pytorch.loggers import WandbLogger

try:
    from ..dataset.constants import NO_LABEL_COLOR, WC_CLASS_COLORS
except ImportError:
    try:
        from dataset.constants import NO_LABEL_COLOR, WC_CLASS_COLORS
    except ImportError:
        NO_LABEL_COLOR, WC_CLASS_COLORS = "#000000", None


class KDSegmentationModule(pl.LightningModule):
    def __init__(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        num_classes: int,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        alpha: float = 0.4,
        use_pseudo_labels: bool = False,
        ignore_index: int = -1,
        rgb_band_indices: Sequence[int] = (2, 1, 0),
        num_samples_to_log: int = 4,
        log_every_n_epochs: int = 1,
        class_names: Optional[list[str]] = None,
        class_colors: Optional[Sequence[Sequence[float]]] = None,
    ):
        """
        Knowledge Distillation Module for Earth Observation Segmentation.

        Args:
            student_model: The lightweight model to train (e.g., terratorch UNet backbone + head).
            teacher_model: The heavy pre-trained model (e.g., TerraMind).
            num_classes: Number of segmentation classes. Needed for metrics and
                         for coloring the qualitative prediction plots.
            lr: Learning rate.
            weight_decay: Weight decay for optimizer.
            alpha: Weighting factor between task loss and distillation loss.
                   0.0 = Only Task Loss, 1.0 = Only Distillation Loss.
            use_pseudo_labels: If True, implements Strategy B (Teacher predictions as GT).
                               If False, implements Strategy A (Real GT + Teacher Logits).
            ignore_index: Index to ignore in the Cross-Entropy loss / metrics.
            rgb_band_indices: Which channel indices of the input tensor correspond to
                               (R, G, B), used only to build a display composite for
                               the qualitative validation plots. Defaults to (2, 1, 0)
                               matching a (B, G, R, RE1, RE2, RE3, NIR) band order.
            num_samples_to_log: How many samples from the first validation batch to
                                 include in the qualitative image/mask/prediction plot.
            log_every_n_epochs: Log the qualitative plot every N validation epochs.
            class_names: Optional list of class names (index-aligned), used to label
                         per-class info if you extend the plots later.
            class_colors: Optional list of (R, G, B) floats in [0, 1], index-aligned
                          with class ids. If omitted, a matplotlib qualitative colormap
                          is used to generate one automatically.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["student_model", "teacher_model"])

        self.student = student_model
        self.teacher = teacher_model
        self.lr = lr
        self.weight_decay = weight_decay
        self.alpha = alpha
        self.use_pseudo_labels = use_pseudo_labels
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.rgb_band_indices = tuple(rgb_band_indices)
        self.num_samples_to_log = num_samples_to_log
        self.log_every_n_epochs = max(log_every_n_epochs, 1)
        self.class_names = class_names
        self.class_colors = self._build_palette(num_classes, class_colors)

        # Freeze the teacher model completely
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        # --- Metrics ----------------------------------------------------
        # Separate metric objects per (stage, model) so Lightning correctly
        # accumulates/resets them over the epoch, and so the student's own
        # performance is visible next to the teacher's as a reference.
        self.metrics = nn.ModuleDict()
        for stage in ("train", "val", "test"):
            for who in ("student", "teacher"):
                self.metrics[f"{stage}_{who}_acc"] = torchmetrics.Accuracy(
                    task="multiclass", num_classes=num_classes, ignore_index=ignore_index
                )
                self.metrics[f"{stage}_{who}_iou"] = torchmetrics.JaccardIndex(
                    task="multiclass", num_classes=num_classes, ignore_index=ignore_index
                )
                self.metrics[f"{stage}_{who}_f1"] = MulticlassF1Score(
                    num_classes=num_classes, ignore_index=ignore_index, average="macro"
                )
                # Logged at epoch end via `compute()` + `log_dict()` + `reset()`
                # (see the `on_*_epoch_end` hooks below), matching how
                # terratorch's own SemanticSegmentationTask logs its
                # ClasswiseWrapper metrics: `.compute()` on a ClasswiseWrapper
                # returns a dict of per-class tensors, and Lightning's
                # `self.log`/`self.log_dict` cannot log a Metric object whose
                # `compute()` returns a dict directly (only plain tensors).
                self.metrics[f"{stage}_{who}_iou_per_class"] = ClasswiseWrapper(
                    MulticlassJaccardIndex(num_classes=num_classes, ignore_index=ignore_index, average=None),
                    labels=self.class_names,
                    prefix=f"{stage}/{who}_IoU_",
                )

    # ------------------------------------------------------------------
    # Core forward / step logic
    # ------------------------------------------------------------------
    def forward(self, x):
        return self.student(x)

    def _extract_tensor(self, batch, key="image"):
        """Extracts the image tensor, handling TerraTorch's dict formats."""
        x = batch[key]
        if isinstance(x, dict):
            # Assume S2L1C is the modality, fallback to the first available
            x = x.get("S2L1C", next(iter(x.values())))
        return x

    def _unwrap_logits(self, model_out):
        """Extracts the raw tensor from HuggingFace/TerraTorch ModelOutput or dict."""
        if hasattr(model_out, "logits"):
            return model_out.logits
        if hasattr(model_out, "output"):
            return model_out.output
        if hasattr(model_out, "out"):
            return model_out.out
        if isinstance(model_out, dict):
            return model_out.get("out", next(iter(model_out.values())))
        return model_out

    @staticmethod
    def _align_spatial(tensor: torch.Tensor, target_hw, mode: str = "bilinear") -> torch.Tensor:
        if tensor.shape[-2:] == tuple(target_hw):
            return tensor
        kwargs = {"align_corners": False} if mode in ("bilinear", "bicubic") else {}
        return F.interpolate(tensor, size=target_hw, mode=mode, **kwargs)

    def _shared_step(self, batch, batch_idx, prefix="train"):
        images = self._extract_tensor(batch, "image")
        masks = batch.get("mask", None)

        # 1. Get Teacher Logits (No gradients needed)
        with torch.no_grad():
            self.teacher.eval()
            # TerraTorch models might expect a dict
            teacher_input = {"S2L1C": images} if isinstance(batch.get("image"), dict) else images
            teacher_out = self.teacher(teacher_input)
            teacher_logits = self._unwrap_logits(teacher_out)

        # 2. Get Student Logits
        student_logits = self.student(images)

        # Teacher and student heads can output at different resolutions (e.g.
        # the teacher's decoder upsamples less aggressively than the
        # student's). Align both to the ground-truth resolution when masks
        # are available (so loss/metrics are computed at full resolution),
        # otherwise align to the student's native resolution.
        ref_hw = masks.shape[-2:] if masks is not None else student_logits.shape[-2:]
        teacher_logits = self._align_spatial(teacher_logits, ref_hw)
        student_logits = self._align_spatial(student_logits, ref_hw)

        # 3. Compute L_teacher (Equation 2: MSE of logits)
        l_teacher = F.mse_loss(student_logits, teacher_logits)

        # 4. Compute L_task (Cross-Entropy)
        if self.use_pseudo_labels:
            # Strategy B: Use Teacher's predictions as Ground Truth (Equation 4)
            pseudo_labels = torch.argmax(teacher_logits.detach(), dim=1)
            l_task = F.cross_entropy(student_logits, pseudo_labels, ignore_index=self.ignore_index)

            # Purely diagnostic: how is the student doing against the *real*
            # labels, even though it isn't being trained on them directly?
            # This is not backpropagated.
            if masks is not None:
                with torch.no_grad():
                    student_ce_vs_gt = F.cross_entropy(
                        student_logits, masks.long(), ignore_index=self.ignore_index
                    )
                self.log(f"{prefix}/student_ce_vs_gt", student_ce_vs_gt, sync_dist=True)
        else:
            # Strategy A: Use Real Ground Truth Labels (Equation 1)
            if masks is None:
                raise ValueError("Ground Truth masks are required when `use_pseudo_labels=False`.")
            l_task = F.cross_entropy(student_logits, masks.long(), ignore_index=self.ignore_index)

        # 5. Total Loss (Equation 3 / Equation 5)
        loss = (1 - self.alpha) * l_task + self.alpha * l_teacher

        # --- Logging: total loss + the two components that make it up ---
        self.log(f"{prefix}/loss", loss, prog_bar=True, sync_dist=True)
        self.log(f"{prefix}/student_task_loss", l_task, prog_bar=True, sync_dist=True)
        self.log(f"{prefix}/distill_loss", l_teacher, sync_dist=True)

        student_pred = student_logits.argmax(dim=1)
        teacher_pred = teacher_logits.argmax(dim=1)

        # --- Logging: student vs. teacher accuracy / mIoU against real GT ---
        if masks is not None:
            masks_long = masks.long()
            self.metrics[f"{prefix}_student_acc"](student_logits, masks_long)
            self.metrics[f"{prefix}_student_iou"](student_logits, masks_long)
            self.metrics[f"{prefix}_student_f1"](student_logits, masks_long)
            self.metrics[f"{prefix}_student_iou_per_class"].update(student_logits, masks_long)
            self.metrics[f"{prefix}_teacher_acc"](teacher_logits, masks_long)
            self.metrics[f"{prefix}_teacher_iou"](teacher_logits, masks_long)
            self.metrics[f"{prefix}_teacher_f1"](teacher_logits, masks_long)
            self.metrics[f"{prefix}_teacher_iou_per_class"].update(teacher_logits, masks_long)

            self.log(f"{prefix}/student_acc", self.metrics[f"{prefix}_student_acc"], on_step=False, on_epoch=True)
            self.log(f"{prefix}/student_iou", self.metrics[f"{prefix}_student_iou"], on_step=False, on_epoch=True)
            self.log(f"{prefix}/student_f1", self.metrics[f"{prefix}_student_f1"], on_step=False, on_epoch=True)
            self.log(f"{prefix}/teacher_acc", self.metrics[f"{prefix}_teacher_acc"], on_step=False, on_epoch=True)
            self.log(f"{prefix}/teacher_iou", self.metrics[f"{prefix}_teacher_iou"], on_step=False, on_epoch=True)
            self.log(f"{prefix}/teacher_f1", self.metrics[f"{prefix}_teacher_f1"], on_step=False, on_epoch=True)

        return {
            "loss": loss,
            "images": images,
            "masks": masks,
            "student_pred": student_pred,
            "teacher_pred": teacher_pred,
        }

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, prefix="train")["loss"]

    def on_validation_epoch_start(self) -> None:
        """Picks which validation batch to plot this epoch.

        Hardcoding batch 0 (as this did) plots the identical patches every epoch,
        because the validation loader is not shuffled. Seeded by epoch so the
        choice varies but stays reproducible and agrees across DDP ranks.
        """
        n = getattr(self.trainer, "num_val_batches", None) if self.trainer else None
        if isinstance(n, (list, tuple)):
            n = n[0] if n else 0
        self._plot_batch_idx = (
            int(np.random.default_rng(self.current_epoch).integers(0, n))
            if isinstance(n, int) and n > 0 else 0
        )

    def validation_step(self, batch, batch_idx):
        outputs = self._shared_step(batch, batch_idx, prefix="val")

        should_plot = (
            batch_idx == getattr(self, "_plot_batch_idx", 0)
            and self.trainer is not None
            and self.trainer.is_global_zero
            and self.current_epoch % self.log_every_n_epochs == 0
        )
        if should_plot:
            self._log_qualitative_predictions(
                images=outputs["images"],
                masks=outputs["masks"],
                teacher_pred=outputs["teacher_pred"],
                student_pred=outputs["student_pred"],
                split="val",
            )

        return outputs["loss"]

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, prefix="test")["loss"]

    def _log_and_reset_iou_per_class(self, prefix: str) -> None:
        for who in ("student", "teacher"):
            metric = self.metrics[f"{prefix}_{who}_iou_per_class"]
            self.log_dict(metric.compute(), sync_dist=True)
            metric.reset()

    def on_train_epoch_end(self) -> None:
        self._log_and_reset_iou_per_class("train")

    def on_validation_epoch_end(self) -> None:
        self._log_and_reset_iou_per_class("val")

    def on_test_epoch_end(self) -> None:
        self._log_and_reset_iou_per_class("test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.student.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
            },
        }

    # ------------------------------------------------------------------
    # Qualitative visualization: image / GT / teacher pred / student pred
    # ------------------------------------------------------------------
    @staticmethod
    def _build_palette(num_classes: int, class_colors: Optional[Sequence[Sequence[float]]]) -> np.ndarray:
        """Colour table for the qualitative plots, index-aligned with class ids.

        Defaults to the official ESA WorldCover legend when the class count
        matches that task, so the training-time plots use the same colours as
        the datasets' own `plot` methods and the dataset report figures. This
        module is task-generic, so any other class count (e.g. 2-class floods)
        falls back to a qualitative colormap.
        """
        if class_colors is not None:
            return np.asarray(class_colors, dtype=float)
        if WC_CLASS_COLORS is not None and num_classes == len(WC_CLASS_COLORS):
            return np.asarray([to_rgb(c) for c in WC_CLASS_COLORS], dtype=float)
        cmap = plt.get_cmap("tab20" if num_classes <= 20 else "gist_ncar")
        colors = [cmap(i / max(num_classes - 1, 1))[:3] for i in range(num_classes)]
        return np.asarray(colors, dtype=float)

    def _to_rgb(self, img: torch.Tensor) -> np.ndarray:
        """Builds a contrast-stretched RGB composite from a (C, H, W) tensor, for display only."""
        c = img.shape[0]
        idx = [i for i in self.rgb_band_indices if i < c]
        if len(idx) < 3:
            idx = [0, 0, 0] if c == 1 else [0, min(1, c - 1), min(2, c - 1)]
        rgb = img[idx, :, :].detach().float().cpu().numpy().transpose(1, 2, 0)
        lo = np.percentile(rgb, 2, axis=(0, 1), keepdims=True)
        hi = np.percentile(rgb, 98, axis=(0, 1), keepdims=True)
        rgb = np.clip((rgb - lo) / np.clip(hi - lo, 1e-6, None), 0, 1)
        return rgb

    def _labels_to_rgb(self, label_map: np.ndarray) -> np.ndarray:
        """Maps a (H, W) integer label map to an (H, W, 3) RGB image using self.class_colors."""
        label_map = label_map.astype(int)
        h, w = label_map.shape
        # Ignore/out-of-range pixels take the same no-label colour the datasets'
        # own `plot` uses, so the two sets of figures can be read side by side.
        out = np.full((h, w, 3), to_rgb(NO_LABEL_COLOR), dtype=np.float32)
        valid = (label_map >= 0) & (label_map < self.num_classes)
        out[valid] = self.class_colors[label_map[valid]]
        return out

    def _log_qualitative_predictions(self, images, masks, teacher_pred, student_pred, split="val"):
        if self.logger is None:
            return

        n = min(self.num_samples_to_log, images.shape[0])
        images = images[:n]
        student_pred = student_pred[:n].detach().cpu()
        teacher_pred = teacher_pred[:n].detach().cpu()
        has_gt = masks is not None
        if has_gt:
            masks = masks[:n].detach().cpu()

        n_cols = 4 if has_gt else 3
        fig, axes = plt.subplots(n, n_cols, figsize=(3 * n_cols, 3 * n), squeeze=False)
        col_titles = ["Image"] + (["Ground Truth"] if has_gt else []) + ["Teacher", "Student"]

        for row in range(n):
            col = 0
            axes[row][col].imshow(self._to_rgb(images[row]))
            if row == 0:
                axes[row][col].set_title(col_titles[col])
            axes[row][col].axis("off")
            col += 1

            if has_gt:
                axes[row][col].imshow(self._labels_to_rgb(masks[row].numpy()))
                if row == 0:
                    axes[row][col].set_title(col_titles[col])
                axes[row][col].axis("off")
                col += 1

            axes[row][col].imshow(self._labels_to_rgb(teacher_pred[row].numpy()))
            if row == 0:
                axes[row][col].set_title(col_titles[col])
            axes[row][col].axis("off")
            col += 1

            axes[row][col].imshow(self._labels_to_rgb(student_pred[row].numpy()))
            if row == 0:
                axes[row][col].set_title(col_titles[col])
            axes[row][col].axis("off")

        fig.suptitle(f"{split} predictions — epoch {self.current_epoch}")
        fig.tight_layout()

        self._log_figure(fig, key=f"{split}/predictions")
        plt.close(fig)

    def _log_figure(self, fig, key: str):
        logger = self.logger
        if WandbLogger is not None and isinstance(logger, WandbLogger):
            import wandb

            logger.experiment.log({key: wandb.Image(fig), "epoch": self.current_epoch})
        elif hasattr(logger, "experiment") and hasattr(logger.experiment, "add_figure"):
            # e.g. TensorBoardLogger
            logger.experiment.add_figure(key, fig, global_step=self.global_step)
        else:
            # Fallback so the plot isn't silently lost with an unsupported logger.
            out_dir = Path(getattr(self.trainer, "default_root_dir", ".")) / "qualitative"
            out_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_dir / f"{key.replace('/', '_')}_epoch{self.current_epoch}.png")