"""
Cross-sensor contrastive knowledge distillation: ViT GeoFM teacher -> CNN student,
trained to be sensor-invariant on co-registered PhiSat-2 / Sentinel-2 patches.

What this changes relative to ``kd_module.KDSegmentationModule``
---------------------------------------------------------------
The baseline module distils a single logit map into the student with an MSE
loss, on one domain at a time. It has two structural problems for this project:

1. *The teacher is queried out of its domain.* TerraMind is pretrained on
   Sentinel-2. Running it on PhiSat-2 imagery means distilling a degraded
   teacher, so the student inherits the teacher's own domain-shift error.
2. *MSE on logits cannot produce domain invariance.* It constrains only the
   final class scores, and only pointwise. Nothing in the objective says that
   the same location seen by two sensors should have the same internal
   representation, which is precisely the property being asked for.

This module addresses both by exploiting the fact that the triplets dataset is
*co-registered*: for every patch we hold a Sentinel-2 view and a PhiSat-2 view
of the same ground, pixel-aligned, plus one shared WorldCover label map. That
turns positive pairs into free supervision and makes the following four
published objectives directly applicable.

The objective
-------------
Let x_s be the Sentinel-2 view, x_t the PhiSat-2 view, y the shared label map,
S the student and T the frozen teacher.

  L = w_task * [ CE(S(x_t), y) + CE(S(x_s), y) ]                        (task)
    + w_kd   * [ KD(S(x_t), T(x_s)) + KD(S(x_s), T(x_s)) ]              (1)
    + w_inst * NT-Xent( g(S_enc(x_s)), g(S_enc(x_t)) )                  (2)
    + w_pix  * PixelProto( h(S_dec(x_t)), h(S_dec(x_s)), y )            (3)
    + w_crd  * InfoNCE( g(S_enc(x_.)), g_T(T_enc(x_s)) )                (4)

(1) **Cross-modal supervision transfer.** The teacher is evaluated only on the
    Sentinel-2 view, where it is trustworthy, and its soft dense predictions
    supervise the student on *both* views. Transferring a teacher's supervision
    onto a paired, unlabelled-in-practice second modality is exactly the
    construction of Gupta, Hoffman & Malik, "Cross Modal Distillation for
    Supervision Transfer", CVPR 2016. The distillation itself is the
    temperature-scaled KL of Hinton, Vinyals & Dean (2015), with the direct
    logit-MSE variant of Kim et al. (IJCAI 2021) kept as an option -- the
    baseline's loss is recoverable as ``kd_mode="mse"``.

(2) **Instance-level cross-sensor contrast.** Symmetric NT-Xent (SimCLR, Chen
    et al. ICML 2020) where the two "views" are two sensors rather than two
    augmentations. This is the EO-standard reading of contrastive learning used
    by SeCo (Mañas et al. ICCV 2021, seasonal views) and CROMA (Fuller et al.
    NeurIPS 2023, aligned radar/optical). It is the term that directly enforces
    "same latent representation despite covariate shift", while in-batch
    negatives block the constant solution.

(3) **Semantic-guided pixel contrast.** SePiCo (Xie et al. TPAMI 2023):
    per-pixel embeddings are pulled to an EMA class centroid shared across both
    domains and pushed from the other centroids, in the multi-positive spirit of
    SupCon (Khosla et al. NeurIPS 2020). Term (2) aligns whole patches; a
    segmentation encoder additionally needs pixel-level, *class-conditional*
    alignment, which marginal methods such as MMD or adversarial DA provably
    cannot deliver. It also removes the false-negative pathology of plain
    instance contrast on land cover, where many distinct patches are the same
    class.

(4) **Contrastive representation distillation** (CRD, Tian, Krishnan & Isola,
    ICLR 2020), optional. Transfers the teacher's relational structure rather
    than its pointwise scores -- the appropriate mechanism when teacher and
    student are different architectures (ViT vs CNN) with incomparable feature
    layouts. Off by default (``w_crd=0``) since (1)-(3) carry the method.

All contrastive weights are linearly ramped in over the first epochs (the
ramp-up schedule of Laine & Aila, ICLR 2017), because the class prototypes need
a few hundred steps of feature statistics before their anchors mean anything.
"""

from pathlib import Path
from typing import Optional, Sequence

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from lightning.pytorch.loggers import WandbLogger
from torchmetrics import ClasswiseWrapper
from torchmetrics.classification import MulticlassF1Score, MulticlassJaccardIndex

from .domain_adaptation.dsbn import convert_to_dsbn, has_dsbn, set_domain
from .losses.contrastive import (
    PixelProjectionHead,
    ProjectionMLP,
    PrototypePixelContrast,
    crd_style_loss,
    dense_kd_loss,
    nt_xent_cross_domain,
    rbf_mmd2,
)
from .losses.segmentation import focal_ce_loss

# Imported two ways in this repo (package-relative, and flat with terra_sat_drift
# on sys.path from kd_lulc_contrastive.py); only one root resolves in each case.
try:
    from ..dataset.plot_utils import build_palette, labels_to_rgb
except ImportError:  # pragma: no cover - depends on how the caller set sys.path
    from dataset.plot_utils import build_palette, labels_to_rgb

# Domain branch ids used for DSBN routing.
SOURCE_DOMAIN_ID = 0  # Sentinel-2, the teacher's home domain
TARGET_DOMAIN_ID = 1  # PhiSat-2, the deployment domain


class CrossSensorKDModule(pl.LightningModule):
    """Distils a ViT GeoFM into a sensor-invariant CNN on co-registered pairs.

    Expects batches from ``PhisatPairedLULCDataModule``:
    ``image_target`` (B, 7, H, W), ``image_source`` (B, 7, H, W), ``mask`` (B, H, W).

    Args:
        student_model: CNN student exposing ``forward(x, return_features=True)``
            with keys ``logits`` / ``bottleneck`` / ``decoder`` (see
            ``student_mobilenet.UNetStudent``).
        teacher_model: frozen ViT-based segmentation model (TerraMind).
        num_classes: number of semantic classes.
        lr / weight_decay: AdamW settings for the student and the projection heads.
        w_task_target / w_task_source: cross-entropy weights per branch. Both
            are supervised because the WorldCover label map is shared by
            construction; set ``w_task_source=0`` to train the task head on the
            target sensor only.
        w_kd: weight of the dense distillation term (1).
        w_instance: weight of the cross-sensor NT-Xent term (2).
        w_pixel: weight of the semantic-guided pixel contrast term (3).
        w_crd: weight of the CRD-style teacher/student contrast term (4).
        kd_mode: ``"kl"`` (Hinton) or ``"mse"`` (the baseline's loss).
        kd_temperature: T for ``kd_mode="kl"``.
        kd_on_unlabeled: distil at pixels whose label is ``ignore_index``. True
            by default -- unlabelled pixels are where distillation adds
            supervision the task loss cannot provide.
        distill_source_branch: also distil the student's Sentinel-2 branch
            toward the teacher, not only its PhiSat-2 branch.
        instance_temperature / pixel_temperature: contrastive temperatures tau.
        cross_view_negatives_only: use the CLIP/CROMA negative set (N-1
            opposite-view) instead of SimCLR's (2N-2 in-batch).
        proj_dim / proj_hidden_dim: instance projection head geometry.
        pixel_proj_dim / pixel_proj_hidden_dim: pixel projection head geometry.
        pixel_feat_size: resolution the decoder feature map is resampled to
            before the pixel contrast, bounding its memory cost.
        proto_momentum: EMA coefficient of the class prototypes.
        max_pixels_per_class: class-balanced sampling cap per class per step.
        contrastive_warmup_epochs: length of the linear ramp-up for terms (2)-(4).
        use_dsbn: convert the student's BatchNorm to domain-specific BatchNorm.
        lr_monitor: metric the ReduceLROnPlateau scheduler tracks. Defaults to
            target-domain IoU rather than total loss, because the total mixes
            five terms under a ramping schedule and so is not a stable signal.
            Pass ``None`` to train without a scheduler (e.g. no val loader).
        lr_monitor_mode: ``"max"`` or ``"min"``, matching ``lr_monitor``.
        teacher_input_key: modality key the teacher expects; the image is passed
            as ``{key: tensor}``. Set to ``None`` to pass a bare tensor.
        teacher_encoder_attr: attribute holding the teacher's encoder, hooked to
            obtain teacher embeddings for term (4).
        ignore_index: label value excluded from losses and metrics.
        student_in_channels: input band count, used to probe feature dimensions.
        rgb_band_indices / num_samples_to_log / log_every_n_epochs /
        class_names / class_colors: qualitative-logging controls, as in
        ``KDSegmentationModule``.
    """

    def __init__(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        num_classes: int,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        # --- loss weights -------------------------------------------------
        w_task_target: float = 1.0,
        w_task_source: float = 1.0,
        w_kd: float = 1.0,
        w_instance: float = 0.5,
        w_pixel: float = 0.1,
        w_crd: float = 0.0,
        # --- task loss ------------------------------------------------------
        task_loss: str = "ce",
        focal_gamma: float = 2.0,
        class_weights: Optional[Sequence[float]] = None,
        # --- distillation -------------------------------------------------
        kd_mode: str = "kl",
        kd_temperature: float = 4.0,
        kd_on_unlabeled: bool = True,
        distill_source_branch: bool = True,
        # --- contrastive --------------------------------------------------
        instance_temperature: float = 0.1,
        pixel_temperature: float = 0.1,
        cross_view_negatives_only: bool = False,
        proj_dim: int = 128,
        proj_hidden_dim: int = 512,
        pixel_proj_dim: int = 64,
        pixel_proj_hidden_dim: int = 128,
        pixel_feat_size: int = 64,
        proto_momentum: float = 0.999,
        max_pixels_per_class: int = 128,
        contrastive_warmup_epochs: int = 1,
        label_smoothing: float = 0.0,
        # --- architecture / plumbing --------------------------------------
        use_dsbn: bool = False,
        lr_monitor: Optional[str] = "val/student_target_iou",
        lr_monitor_mode: str = "max",
        teacher_input_key: Optional[str] = "S2L1C",
        teacher_encoder_attr: str = "encoder",
        ignore_index: int = -1,
        student_in_channels: int = 7,
        rgb_band_indices: Sequence[int] = (2, 1, 0),
        num_samples_to_log: int = 4,
        log_every_n_epochs: int = 1,
        class_names: Optional[list[str]] = None,
        class_colors: Optional[Sequence[Sequence[float]]] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["student_model", "teacher_model"])

        self.student = student_model
        self.teacher = teacher_model
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay

        self.w_task_target = w_task_target
        self.w_task_source = w_task_source
        self.w_kd = w_kd
        self.w_instance = w_instance
        self.w_pixel = w_pixel
        self.w_crd = w_crd

        self.kd_mode = kd_mode
        self.kd_temperature = kd_temperature
        if task_loss not in ("ce", "focal"):
            raise ValueError(f"task_loss must be 'ce' or 'focal', got {task_loss!r}")
        self.task_loss = task_loss
        self.focal_gamma = focal_gamma
        # Registered as a buffer so it follows the module across devices and is
        # captured in the checkpoint -- reloading a run must not silently revert
        # to unweighted loss. Empty tensor means "no weighting".
        self.register_buffer(
            "class_weight",
            torch.as_tensor(list(class_weights), dtype=torch.float32)
            if class_weights is not None else torch.empty(0),
            persistent=True,
        )
        if self.class_weight.numel() and self.class_weight.numel() != num_classes:
            raise ValueError(
                f"class_weights has {self.class_weight.numel()} entries, "
                f"expected num_classes={num_classes}"
            )
        self.kd_on_unlabeled = kd_on_unlabeled
        self.distill_source_branch = distill_source_branch

        self.instance_temperature = instance_temperature
        self.cross_view_negatives_only = cross_view_negatives_only
        self.pixel_feat_size = pixel_feat_size
        self.contrastive_warmup_epochs = max(contrastive_warmup_epochs, 0)
        self.label_smoothing = label_smoothing

        self.lr_monitor = lr_monitor
        self.lr_monitor_mode = lr_monitor_mode
        self.teacher_input_key = teacher_input_key
        self.ignore_index = ignore_index
        self.rgb_band_indices = tuple(rgb_band_indices)
        self.num_samples_to_log = num_samples_to_log
        self.log_every_n_epochs = max(log_every_n_epochs, 1)
        self.class_names = class_names
        self.class_colors = self._build_palette(num_classes, class_colors)

        # Freeze the teacher completely.
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        if use_dsbn:
            convert_to_dsbn(self.student, num_domains=2)
        self.use_dsbn = has_dsbn(self.student)

        # --- projection heads -------------------------------------------
        bottleneck_dim, decoder_dim = self._probe_student_dims(student_in_channels)
        self.instance_proj = ProjectionMLP(bottleneck_dim, proj_hidden_dim, proj_dim)
        self.pixel_proj = PixelProjectionHead(decoder_dim, pixel_proj_hidden_dim, pixel_proj_dim)
        self.pixel_contrast = PrototypePixelContrast(
            num_classes=num_classes,
            dim=pixel_proj_dim,
            temperature=pixel_temperature,
            momentum=proto_momentum,
            ignore_index=ignore_index,
            max_pixels_per_class=max_pixels_per_class,
        )

        # --- teacher embedding hook (term 4 only) ------------------------
        self._teacher_feat: Optional[torch.Tensor] = None
        self.teacher_proj: Optional[nn.Module] = None
        self._teacher_proj_dim = proj_dim
        self._teacher_proj_hidden = proj_hidden_dim
        if self.w_crd > 0:
            self._register_teacher_hook(teacher_encoder_attr)

        # --- metrics -----------------------------------------------------
        # Per (stage, branch) so the target-domain and source-domain scores of
        # the *same* student are visible side by side. Their difference is the
        # end-to-end measure of how much domain gap survives.
        self.metrics = nn.ModuleDict()
        for stage in ("train", "val", "test"):
            for who in ("student_target", "student_source", "teacher_source"):
                self.metrics[f"{stage}_{who}_acc"] = torchmetrics.Accuracy(
                    task="multiclass", num_classes=num_classes, ignore_index=ignore_index
                )
                self.metrics[f"{stage}_{who}_iou"] = torchmetrics.JaccardIndex(
                    task="multiclass", num_classes=num_classes, ignore_index=ignore_index
                )
                self.metrics[f"{stage}_{who}_f1"] = MulticlassF1Score(
                    num_classes=num_classes, ignore_index=ignore_index, average="macro"
                )
            # Logged under a "<stage>_per_class/" section of their own rather than
            # mixed into "<stage>/". All 11 classes were always being logged, but
            # buried among the ~30 other keys in the same section they are easy to
            # miss in the W&B workspace, which only auto-renders a limited number
            # of panels per section.
            self.metrics[f"{stage}_student_target_iou_per_class"] = ClasswiseWrapper(
                MulticlassJaccardIndex(num_classes=num_classes, ignore_index=ignore_index, average=None),
                labels=self.class_names,
                prefix=f"{stage}_per_class/IoU_",
            )
            # Per-class pixel counts. torchmetrics reports IoU 0.0 both for "the
            # model got this class entirely wrong" and for "this class did not
            # occur", which are very different things for the rare classes here
            # (moss/lichen, mangroves and snow are each well under 1% of pixels).
            # The support tells the two apart.
            self.register_buffer(f"_support_{stage}", torch.zeros(num_classes), persistent=False)

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    def _probe_student_dims(self, in_channels: int) -> tuple[int, int]:
        """Runs one dummy forward to read off bottleneck / decoder widths.

        Cheaper and less error-prone than asking the caller to keep channel
        counts in sync with the student's configuration. Done in eval mode under
        ``no_grad`` so BatchNorm running statistics are untouched.
        """
        was_training = self.student.training
        self.student.eval()
        try:
            with torch.no_grad():
                # 64 is divisible by the UNet's total downsample factor of 16.
                out = self.student(torch.zeros(2, in_channels, 64, 64), return_features=True)
        except Exception as exc:  # pragma: no cover - depends on student impl
            raise RuntimeError(
                "Could not probe student feature dimensions. The student must accept "
                "forward(x, return_features=True) and return a dict with 'bottleneck' "
                "and 'decoder' entries."
            ) from exc
        finally:
            if was_training:
                self.student.train()
        return out["bottleneck"].shape[1], out["decoder"].shape[1]

    def _register_teacher_hook(self, attr: str) -> None:
        """Captures the teacher encoder's output for the CRD-style term."""
        target = getattr(self.teacher, attr, None)
        if target is None:
            raise AttributeError(
                f"w_crd > 0 but the teacher has no attribute {attr!r} to hook. Pass "
                "teacher_encoder_attr=<name of the encoder submodule>, or set w_crd=0."
            )

        def hook(_module, _inp, output):
            self._teacher_feat = output

        target.register_forward_hook(hook)

    def _lazy_teacher_proj(self, dim: int, device, dtype) -> nn.Module:
        """Builds the teacher-side projection on first use.

        The teacher's embedding width is only known once a real batch has passed
        through it (TerraMind's neck configuration determines it), so this head
        cannot be constructed in ``__init__``. It is registered as a submodule
        and added to the live optimizer so it still trains.
        """
        if self.teacher_proj is None:
            self.teacher_proj = ProjectionMLP(
                dim, self._teacher_proj_hidden, self._teacher_proj_dim
            ).to(device=device, dtype=dtype)
            try:
                optimizers = self.optimizers()
                optimizers = optimizers if isinstance(optimizers, list) else [optimizers]
                if optimizers and optimizers[0] is not None:
                    optimizers[0].add_param_group({"params": self.teacher_proj.parameters()})
            except RuntimeError:
                # No Trainer attached (unit tests, manual forward passes). The head
                # still works, it just is not optimised -- which is fine outside fit.
                pass
        return self.teacher_proj

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Inference path: student on the target (PhiSat-2) domain."""
        if self.use_dsbn:
            set_domain(self.student, TARGET_DOMAIN_ID)
        return self.student(x)

    def _student_forward(self, x: torch.Tensor, domain_id: int) -> dict:
        if self.use_dsbn:
            set_domain(self.student, domain_id)
        return self.student(x, return_features=True)

    @torch.no_grad()
    def _teacher_forward(self, x: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        self.teacher.eval()
        self._teacher_feat = None
        inp = {self.teacher_input_key: x} if self.teacher_input_key else x
        out = self.teacher(inp)
        return self._unwrap_logits(out), self._teacher_feat

    @staticmethod
    def _unwrap_logits(model_out):
        """Extracts the raw tensor from a HuggingFace/TerraTorch ModelOutput or dict."""
        for attr in ("logits", "output", "out"):
            if hasattr(model_out, attr):
                return getattr(model_out, attr)
        if isinstance(model_out, dict):
            return model_out.get("out", next(iter(model_out.values())))
        return model_out

    @staticmethod
    def _align_spatial(t: torch.Tensor, target_hw, mode: str = "bilinear") -> torch.Tensor:
        if t.shape[-2:] == tuple(target_hw):
            return t
        kwargs = {"align_corners": False} if mode in ("bilinear", "bicubic") else {}
        return F.interpolate(t, size=target_hw, mode=mode, **kwargs)

    @staticmethod
    def _pool(feat: torch.Tensor) -> torch.Tensor:
        """Global average pool a (B, C, H, W) or (B, N, D) feature to (B, C)."""
        if feat.dim() == 4:
            return F.adaptive_avg_pool2d(feat, 1).flatten(1)
        if feat.dim() == 3:
            return feat.mean(dim=1)
        return feat

    def _ramp(self) -> float:
        """Linear ramp-up factor in [0, 1] for the contrastive terms.

        Ramp-up as in Laine & Aila (ICLR 2017): the prototype anchors and the
        projection heads are meaningless at step 0, so applying their full
        weight immediately injects noise into the encoder.
        """
        if self.contrastive_warmup_epochs == 0:
            return 1.0
        return float(min(1.0, (self.current_epoch + 1) / (self.contrastive_warmup_epochs + 1)))

    def _task_loss(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Supervised segmentation loss, cross-entropy or focal.

        `focal_ce_loss` with gamma=0 is exactly weighted cross-entropy, so both
        settings share one code path and `class_weights` applies either way.
        """
        alpha = self.class_weight if self.class_weight.numel() else None
        return focal_ce_loss(
            logits, mask,
            gamma=self.focal_gamma if self.task_loss == "focal" else 0.0,
            alpha=alpha, ignore_index=self.ignore_index,
        )

    # ------------------------------------------------------------------
    # Shared step
    # ------------------------------------------------------------------
    def _shared_step(self, batch: dict, batch_idx: int, prefix: str = "train") -> dict:
        x_t = batch["image_target"]
        x_s = batch["image_source"]
        mask = batch["mask"].long()
        ref_hw = mask.shape[-2:]

        # --- forward passes ---------------------------------------------
        out_t = self._student_forward(x_t, TARGET_DOMAIN_ID)
        out_s = self._student_forward(x_s, SOURCE_DOMAIN_ID)
        teacher_logits, teacher_feat = self._teacher_forward(x_s)

        logits_t = self._align_spatial(out_t["logits"], ref_hw)
        logits_s = self._align_spatial(out_s["logits"], ref_hw)
        teacher_logits = self._align_spatial(teacher_logits, ref_hw).detach()

        losses: dict[str, torch.Tensor] = {}
        zero = logits_t.sum() * 0.0

        # --- task loss ---------------------------------------------------
        l_task_t = self._task_loss(logits_t, mask)
        l_task_s = self._task_loss(logits_s, mask) if self.w_task_source > 0 else zero
        losses["task_target"] = l_task_t
        losses["task_source"] = l_task_s

        # --- (1) cross-modal dense distillation --------------------------
        if self.w_kd > 0:
            valid = None if self.kd_on_unlabeled else (mask != self.ignore_index)
            kd_t = dense_kd_loss(
                logits_t, teacher_logits,
                temperature=self.kd_temperature, valid_mask=valid, mode=self.kd_mode,
            )
            kd_s = (
                dense_kd_loss(
                    logits_s, teacher_logits,
                    temperature=self.kd_temperature, valid_mask=valid, mode=self.kd_mode,
                )
                if self.distill_source_branch
                else zero
            )
            losses["kd_target"] = kd_t
            losses["kd_source"] = kd_s
        else:
            losses["kd_target"] = zero
            losses["kd_source"] = zero

        # --- projections used by (2), (3), (4) ---------------------------
        z_t = self.instance_proj(self._pool(out_t["bottleneck"]))
        z_s = self.instance_proj(self._pool(out_s["bottleneck"]))

        # --- (2) instance-level cross-sensor contrast --------------------
        losses["instance"] = (
            nt_xent_cross_domain(
                z_s, z_t,
                temperature=self.instance_temperature,
                cross_view_negatives_only=self.cross_view_negatives_only,
                label_smoothing=self.label_smoothing
            )
            if self.w_instance > 0
            else zero
        )

        # --- (3) semantic-guided cross-domain pixel contrast -------------
        if self.w_pixel > 0:
            size = (self.pixel_feat_size, self.pixel_feat_size)
            e_t = self.pixel_proj(self._align_spatial(out_t["decoder"], size))
            e_s = self.pixel_proj(self._align_spatial(out_s["decoder"], size))
            mask_small = F.interpolate(
                mask.unsqueeze(1).float(), size=size, mode="nearest"
            ).squeeze(1).long()
            losses["pixel"] = self.pixel_contrast(e_t, mask_small, e_s, mask_small)
        else:
            losses["pixel"] = zero

        # --- (4) CRD-style teacher/student contrast ----------------------
        if self.w_crd > 0 and teacher_feat is not None:
            t_vec = self._pool(
                teacher_feat[-1] if isinstance(teacher_feat, (list, tuple)) else teacher_feat
            ).detach()
            proj = self._lazy_teacher_proj(t_vec.shape[1], t_vec.device, t_vec.dtype)
            z_teacher = proj(t_vec)
            losses["crd"] = 0.5 * (
                crd_style_loss(z_t, z_teacher, self.instance_temperature)
                + crd_style_loss(z_s, z_teacher, self.instance_temperature)
            )
        else:
            losses["crd"] = zero

        # --- total -------------------------------------------------------
        ramp = self._ramp() if prefix == "train" else 1.0
        total = (
            self.w_task_target * losses["task_target"]
            + self.w_task_source * losses["task_source"]
            + self.w_kd * (losses["kd_target"] + losses["kd_source"])
            + ramp * self.w_instance * losses["instance"]
            + ramp * self.w_pixel * losses["pixel"]
            + ramp * self.w_crd * losses["crd"]
        )

        # --- logging -----------------------------------------------------
        self.log(f"{prefix}/loss", total, prog_bar=True, sync_dist=True)
        for name, value in losses.items():
            self.log(f"{prefix}/loss_{name}", value, sync_dist=True)
        if prefix == "train":
            self.log("train/contrastive_ramp", ramp, sync_dist=True)

        self._log_metrics(prefix, logits_t, logits_s, teacher_logits, mask)
        self._log_domain_gap(prefix, z_t, z_s)

        return {
            "loss": total,
            "image_target": x_t,
            "image_source": x_s,
            "mask": mask,
            "pred_target": logits_t.argmax(dim=1),
            "pred_source": logits_s.argmax(dim=1),
            "pred_teacher": teacher_logits.argmax(dim=1),
            # Carried through so the qualitative plot can name the patches it
            # drew; makes it possible to check that the selection really varies
            # and to go back to a specific patch in the dataset.
            "patch_index": batch.get("patch_index"),
        }

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def _log_metrics(self, prefix, logits_t, logits_s, teacher_logits, mask) -> None:
        pairs = {
            "student_target": logits_t,
            "student_source": logits_s,
            "teacher_source": teacher_logits,
        }
        for who, logits in pairs.items():
            for metric in ("acc", "iou", "f1"):
                key = f"{prefix}_{who}_{metric}"
                self.metrics[key](logits, mask)
                self.log(f"{prefix}/{who}_{metric}", self.metrics[key], on_step=False, on_epoch=True)
        self.metrics[f"{prefix}_student_target_iou_per_class"].update(logits_t, mask)

        with torch.no_grad():
            valid = mask[mask != self.ignore_index]
            if valid.numel():
                buf = getattr(self, f"_support_{prefix}")
                buf += torch.bincount(valid.flatten(), minlength=self.num_classes).to(buf.dtype)

    def _log_domain_gap(self, prefix: str, z_t: torch.Tensor, z_s: torch.Tensor) -> None:
        """Reports how sensor-invariant the latent space actually is.

        These are diagnostics, never optimised. Read them together:

        ``cos_paired``    cosine similarity between the two sensor views of the
                          *same* patch. Should rise toward 1.
        ``cos_unpaired``  mean similarity between views of *different* patches.
                          Must stay well below ``cos_paired``: if both approach
                          1 the encoder has collapsed to a constant and the
                          alignment is vacuous.
        ``align_margin``  cos_paired - cos_unpaired, the quantity that actually
                          matters.
        ``retrieval_top1`` fraction of PhiSat-2 patches whose nearest Sentinel-2
                          neighbour in the batch is its own co-registered pair.
                          A direct, interpretable read of cross-sensor
                          correspondence.
        ``mmd2``          multi-kernel MMD between the two domains' embedding
                          clouds (Gretton et al. 2012): residual *marginal* gap.
        """
        with torch.no_grad():
            n = z_t.shape[0]
            sim = z_t @ z_s.t()
            cos_paired = sim.diagonal().mean()
            if n > 1:
                off = ~torch.eye(n, dtype=torch.bool, device=sim.device)
                cos_unpaired = sim[off].mean()
                retrieval = (sim.argmax(dim=1) == torch.arange(n, device=sim.device)).float().mean()
            else:
                cos_unpaired = torch.zeros((), device=sim.device)
                retrieval = torch.ones((), device=sim.device)

            self.log(f"{prefix}/cos_paired", cos_paired, sync_dist=True)
            self.log(f"{prefix}/cos_unpaired", cos_unpaired, sync_dist=True)
            self.log(f"{prefix}/align_margin", cos_paired - cos_unpaired, sync_dist=True)
            self.log(f"{prefix}/retrieval_top1", retrieval, prog_bar=(prefix == "val"), sync_dist=True)
            self.log(f"{prefix}/mmd2", rbf_mmd2(z_t.float(), z_s.float()), sync_dist=True)
            self.log(
                f"{prefix}/prototypes_ready",
                self.pixel_contrast.proto_ready.float().sum(),
                sync_dist=True,
            )

    def _epoch_end(self, prefix: str) -> None:
        metric = self.metrics[f"{prefix}_student_target_iou_per_class"]
        per_class = metric.compute()
        self.log_dict(per_class, sync_dist=True)

        # Fixed-denominator macro mIoU: the mean over *all* num_classes, with
        # absent classes counted as 0.
        #
        # `torchmetrics.JaccardIndex` (logged as student_target_iou) drops
        # classes with zero union from its macro average, so its denominator
        # depends on which classes the model happens to predict. Two models
        # evaluated on the same data can therefore be averaged over different
        # numbers of classes -- observed at eval size 100, where snow/ice was
        # absent, one model was scored over 10 classes and the other over 11,
        # and the resulting mIoU were not comparable. This metric has a constant
        # denominator and is the safe one for cross-model comparison.
        self.log(f"{prefix}/student_target_iou_fixed",
                 torch.stack(list(per_class.values())).mean(), sync_dist=True)
        metric.reset()

        support = getattr(self, f"_support_{prefix}")
        names = self.class_names or [str(i) for i in range(self.num_classes)]
        self.log_dict(
            {f"{prefix}_per_class/pixels_{n}": support[i] for i, n in enumerate(names)},
            reduce_fx="sum", sync_dist=True,
        )
        support.zero_()

        # The headline domain-invariance number: how much worse the student is
        # on PhiSat-2 than on the Sentinel-2 view of the very same patches.
        # Any residual value is domain gap that survived training.
        iou_t = self.metrics[f"{prefix}_student_target_iou"].compute()
        iou_s = self.metrics[f"{prefix}_student_source_iou"].compute()
        self.log(f"{prefix}/domain_gap_iou", iou_s - iou_t, prog_bar=(prefix == "val"), sync_dist=True)

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, prefix="train")["loss"]

    def on_validation_epoch_start(self) -> None:
        self._plot_batch_idx = self._pick_plot_batch()

    def _pick_plot_batch(self) -> int:
        """Chooses which validation batch to render this epoch.

        This used to be hardcoded to batch 0, and the validation loader is not
        shuffled, so every epoch logged the identical handful of patches. The
        choice is seeded by the epoch number so it still varies from epoch to
        epoch while staying reproducible across runs and identical on every rank
        (important under DDP, where all ranks must agree on which batch is the
        one being plotted).
        """
        n = getattr(self.trainer, "num_val_batches", None) if self.trainer else None
        if isinstance(n, (list, tuple)):
            n = n[0] if n else 0
        if not isinstance(n, int) or n <= 0:  # unknown / inf (IterableDataset)
            return 0
        return int(np.random.default_rng(self.current_epoch).integers(0, n))

    def validation_step(self, batch, batch_idx):
        out = self._shared_step(batch, batch_idx, prefix="val")
        should_plot = (
            batch_idx == getattr(self, "_plot_batch_idx", 0)
            and self.trainer is not None
            and self.trainer.is_global_zero
            and self.current_epoch % self.log_every_n_epochs == 0
        )
        if should_plot:
            self._log_qualitative(out, split="val")
        return out["loss"]

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, prefix="test")["loss"]

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self):
        params = [
            {"params": [p for p in self.student.parameters() if p.requires_grad]},
            {"params": self.instance_proj.parameters()},
            {"params": self.pixel_proj.parameters()},
        ]
        optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        if not self.lr_monitor:
            return optimizer
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=self.lr_monitor_mode, factor=0.5, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": self.lr_monitor},
        }

    # ------------------------------------------------------------------
    # Qualitative visualisation
    # ------------------------------------------------------------------
    @staticmethod
    def _build_palette(num_classes: int, class_colors) -> np.ndarray:
        return build_palette(num_classes, class_colors)

    def _to_rgb(self, img: torch.Tensor) -> np.ndarray:
        c = img.shape[0]
        idx = [i for i in self.rgb_band_indices if i < c]
        if len(idx) < 3:
            idx = [0, 0, 0] if c == 1 else [0, min(1, c - 1), min(2, c - 1)]
        rgb = img[idx, :, :].detach().float().cpu().numpy().transpose(1, 2, 0)
        lo = np.percentile(rgb, 2, axis=(0, 1), keepdims=True)
        hi = np.percentile(rgb, 98, axis=(0, 1), keepdims=True)
        return np.clip((rgb - lo) / np.clip(hi - lo, 1e-6, None), 0, 1)

    def _labels_to_rgb(self, label_map: np.ndarray) -> np.ndarray:
        # Shared with the datasets' `plot` and the no-KD baseline, so the
        # validation figures of every training path use one colour scheme.
        return labels_to_rgb(label_map.astype(int), self.class_colors)

    def _log_qualitative(self, out: dict, split: str = "val") -> None:
        """Plots both sensor views with the student's prediction on each.

        Laid out so the two student columns can be compared directly: where they
        disagree on the same ground truth, the encoder is still sensor-dependent.
        """
        if self.logger is None:
            return

        batch_size = out["image_target"].shape[0]
        n = min(self.num_samples_to_log, batch_size)
        # Random rows rather than the first n, so a large batch does not always
        # surface the same corner of it. Seeded by epoch, as in _pick_plot_batch.
        rows = np.random.default_rng(self.current_epoch + 1).choice(
            batch_size, size=n, replace=False)
        cols = [
            ("PhiSat-2", lambda i: self._to_rgb(out["image_target"][i])),
            ("Sentinel-2", lambda i: self._to_rgb(out["image_source"][i])),
            ("Ground truth", lambda i: self._labels_to_rgb(out["mask"][i].cpu().numpy())),
            ("Teacher (S2)", lambda i: self._labels_to_rgb(out["pred_teacher"][i].cpu().numpy())),
            ("Student (PhiSat-2)", lambda i: self._labels_to_rgb(out["pred_target"][i].cpu().numpy())),
            ("Student (S2)", lambda i: self._labels_to_rgb(out["pred_source"][i].cpu().numpy())),
        ]

        fig, axes = plt.subplots(n, len(cols), figsize=(2.6 * len(cols), 2.6 * n), squeeze=False)
        for row, sample_idx in enumerate(rows):
            for col, (title, render) in enumerate(cols):
                ax = axes[row][col]
                ax.imshow(render(int(sample_idx)))
                if row == 0:
                    ax.set_title(title, fontsize=9)
                ax.axis("off")
            # Drawn as an overlay rather than a ylabel, which `axis("off")` hides.
            if out.get("patch_index") is not None:
                axes[row][0].text(
                    0.02, 0.98, f"#{int(out['patch_index'][sample_idx])}",
                    transform=axes[row][0].transAxes, va="top", fontsize=6,
                    color="white", bbox=dict(facecolor="black", alpha=0.5,
                                             edgecolor="none", pad=1.5))
        fig.suptitle(f"{split} — epoch {self.current_epoch}")
        fig.tight_layout()
        self._log_figure(fig, key=f"{split}/predictions")
        plt.close(fig)

    def _log_figure(self, fig, key: str) -> None:
        logger = self.logger
        if WandbLogger is not None and isinstance(logger, WandbLogger):
            import wandb

            logger.experiment.log({key: wandb.Image(fig), "epoch": self.current_epoch})
        elif hasattr(logger, "experiment") and hasattr(logger.experiment, "add_figure"):
            logger.experiment.add_figure(key, fig, global_step=self.global_step)
        else:
            out_dir = Path(getattr(self.trainer, "default_root_dir", ".")) / "qualitative"
            out_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_dir / f"{key.replace('/', '_')}_epoch{self.current_epoch}.png")
