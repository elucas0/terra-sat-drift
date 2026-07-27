"""
Contrastive / distillation objectives for cross-sensor domain-invariant learning.

Every loss in this file is an implementation of a published objective; the
docstrings name the paper each one comes from and flag any deviation from the
original formulation.

References
----------
[Hinton15]  Hinton, Vinyals & Dean. "Distilling the Knowledge in a Neural
            Network." NeurIPS Deep Learning Workshop, 2015. arXiv:1503.02531
[Kim21]     Kim, Park, Bengio et al. "Comparing Kullback-Leibler Divergence and
            Mean Squared Error Loss in Knowledge Distillation." IJCAI 2021.
            arXiv:2105.08919
[Chen20]    Chen, Kornblith, Norouzi & Hinton. "A Simple Framework for
            Contrastive Learning of Visual Representations" (SimCLR / NT-Xent).
            ICML 2020. arXiv:2002.05709
[Oord18]    van den Oord, Li & Vinyals. "Representation Learning with
            Contrastive Predictive Coding" (InfoNCE). arXiv:1807.03748
[Fuller23]  Fuller, Millard & Green. "CROMA: Remote Sensing Representations
            with Contrastive Radar-Optical Masked Autoencoders." NeurIPS 2023.
[Manas21]   Mañas, Lacoste, Giro-i-Nieto et al. "Seasonal Contrast:
            Unsupervised Pre-Training from Uncurated Remote Sensing Data."
            ICCV 2021. arXiv:2103.16607
[Khosla20]  Khosla, Teterwak, Wang et al. "Supervised Contrastive Learning"
            (SupCon). NeurIPS 2020. arXiv:2004.11362
[Xie23]     Xie, Li, Li et al. "SePiCo: Semantic-Guided Pixel Contrast for
            Domain Adaptive Semantic Segmentation." TPAMI 2023.
            arXiv:2204.08808
[Zhang21]   Zhang, Wang, Mao et al. "Prototypical Pseudo Label Denoising and
            Target Structure Learning for Domain Adaptive Semantic
            Segmentation" (ProDA). CVPR 2021. arXiv:2101.10979
[Tian20]    Tian, Krishnan & Isola. "Contrastive Representation Distillation"
            (CRD). ICLR 2020. arXiv:1910.10699
[Gretton12] Gretton, Borgwardt, Rasch et al. "A Kernel Two-Sample Test"
            (MMD). JMLR 2012.
"""

from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

_EPS = 1e-8


# ----------------------------------------------------------------------------
# Projection heads (SimCLR-style; discarded after training)
# ----------------------------------------------------------------------------
class ProjectionMLP(nn.Module):
    """Two-layer projection head on a pooled feature vector.

    SimCLR [Chen20] showed that the contrastive loss should be applied to a
    projection g(h) rather than to the backbone feature h itself, and that the
    projection is discarded at evaluation time. Same construction here.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class PixelProjectionHead(nn.Module):
    """1x1-conv projection head for dense (per-pixel) embeddings.

    The 1x1-conv "dense projection head" is the construction used by dense
    contrastive methods (DenseCL, Wang et al. CVPR 2021) and by the pixel
    contrast branch of SePiCo [Xie23].
    """

    def __init__(self, in_dim: int, hidden_dim: int = 128, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=1)


# ----------------------------------------------------------------------------
# 1. Dense knowledge distillation (replaces the naive MSE-on-logits)
# ----------------------------------------------------------------------------
def dense_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 4.0,
    valid_mask: Optional[torch.Tensor] = None,
    mode: str = "kl",
) -> torch.Tensor:
    """Pixel-wise knowledge distillation between two dense logit maps.

    ``mode="kl"`` is the classical temperature-scaled KL of [Hinton15], applied
    independently at every pixel and rescaled by T^2 so that gradient
    magnitudes stay comparable across temperatures. ``mode="mse"`` is the
    direct logit-matching objective, which [Kim21] shows is the T -> inf limit
    of the KL form and is often the stronger of the two; it is what the naive
    baseline in ``kd_module.py`` used.

    Args:
        student_logits: (B, C, H, W).
        teacher_logits: (B, C, H, W), already detached.
        temperature: softmax temperature T (``mode="kl"`` only).
        valid_mask: (B, H, W) bool, True where the pixel should contribute.
            Used to drop ``ignore_index`` pixels.
        mode: ``"kl"`` or ``"mse"``.
    """
    if mode == "mse":
        per_pixel = F.mse_loss(student_logits, teacher_logits, reduction="none").mean(dim=1)
    elif mode == "kl":
        log_p_s = F.log_softmax(student_logits / temperature, dim=1)
        log_p_t = F.log_softmax(teacher_logits / temperature, dim=1)
        p_t = log_p_t.exp()
        per_pixel = (p_t * (log_p_t - log_p_s)).sum(dim=1) * (temperature ** 2)
    else:
        raise ValueError(f"Unknown dense KD mode: {mode!r} (expected 'kl' or 'mse').")

    if valid_mask is None:
        return per_pixel.mean()
    valid_mask = valid_mask.to(per_pixel.dtype)
    return (per_pixel * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)


# ----------------------------------------------------------------------------
# 2. Instance-level cross-sensor contrast (the domain-invariance driver)
# ----------------------------------------------------------------------------
def nt_xent_cross_domain(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    temperature: float = 0.1,
    cross_view_negatives_only: bool = False,
) -> torch.Tensor:
    """Symmetric NT-Xent over two co-registered sensor views of the same patch.

    This is exactly the NT-Xent loss of SimCLR [Chen20] / InfoNCE [Oord18],
    with one change of interpretation that is itself standard practice in
    Earth observation: the two "views" of a sample are not two random
    augmentations of one image, but the *same ground location observed by two
    different sensors*. Positives therefore come for free from co-registration
    and no longer depend on hand-designed augmentations. SeCo [Manas21] uses
    the same trick with seasonal views of a location; CROMA [Fuller23] uses it
    with spatially aligned radar/optical pairs.

    Minimising this loss forces the encoder to map a PhiSat-2 patch and its
    co-registered Sentinel-2 patch to the same point on the unit sphere, i.e.
    to become invariant to the sensor (the covariate shift), while the
    in-batch negatives prevent the trivial constant solution.

    Args:
        z_a: (N, D) L2-normalised embeddings of view A (e.g. Sentinel-2).
        z_b: (N, D) L2-normalised embeddings of view B (e.g. PhiSat-2).
        temperature: InfoNCE temperature tau.
        cross_view_negatives_only: if True use only the N-1 opposite-view
            negatives (the CLIP / CROMA formulation); if False use all 2N-2
            in-batch negatives (the original SimCLR formulation, default).

    Returns:
        Scalar loss.
    """
    n = z_a.shape[0]
    if n < 2:
        return z_a.sum() * 0.0  # keeps the graph alive, contributes nothing

    if cross_view_negatives_only:
        logits = z_a @ z_b.t() / temperature  # (N, N)
        targets = torch.arange(n, device=z_a.device)
        return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets))

    z = torch.cat([z_a, z_b], dim=0)  # (2N, D)
    sim = z @ z.t() / temperature
    self_mask = torch.eye(2 * n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(self_mask, float("-inf"))
    # positive of row i<N is row i+N, and vice versa
    targets = torch.cat(
        [torch.arange(n, 2 * n, device=z.device), torch.arange(0, n, device=z.device)]
    )
    return F.cross_entropy(sim, targets)


# ----------------------------------------------------------------------------
# 3. Semantic-guided cross-domain pixel contrast
# ----------------------------------------------------------------------------
class PrototypePixelContrast(nn.Module):
    """Centroid-aware cross-domain pixel contrast, following SePiCo [Xie23].

    Instance-level alignment (``nt_xent_cross_domain``) makes the *global*
    embedding of a patch sensor-invariant, but a segmentation encoder also
    needs its *per-pixel* features to be sensor-invariant and
    class-discriminative. SePiCo's centroid-aware pixel contrast does this by
    pulling every pixel embedding toward the centroid ("prototype") of its own
    semantic class and pushing it away from all other class centroids. The
    prototypes are shared between the two domains and updated from both, so
    the same class in PhiSat-2 and in Sentinel-2 is pulled to *one* shared
    anchor -- that is what removes the class-conditional part of the domain
    gap, which pure marginal alignment (MMD, adversarial DA) cannot reach.

    Two further properties matter for this dataset:

    * It solves the false-negative problem. Plain instance contrast treats two
      different forest patches as negatives, which is actively wrong in EO
      where land cover is highly redundant. Using labels to define positives is
      the SupCon [Khosla20] correction, and the loss below is SupCon's
      multi-positive form collapsed onto class centroids.
    * Prototypes are maintained as an exponential moving average, as in ProDA
      [Zhang21] and SePiCo, so the anchors stay stable under the small batch
      sizes forced by 256x256x7 imagery.

    The loss for a sampled pixel embedding f with label y is

        L = -log  exp(f . phi_y / tau) / sum_c exp(f . phi_c / tau)

    i.e. a cross-entropy over cosine similarities to the C class prototypes.

    Args:
        num_classes: number of semantic classes C.
        dim: dimensionality D of the pixel embedding space.
        temperature: tau.
        momentum: EMA coefficient for the prototype update.
        ignore_index: label value to exclude.
        max_pixels_per_class: cap on sampled pixels per class per forward, to
            keep the loss class-balanced (SePiCo stresses class balance) and
            the memory bounded.
    """

    def __init__(
        self,
        num_classes: int,
        dim: int,
        temperature: float = 0.1,
        momentum: float = 0.999,
        ignore_index: int = -1,
        max_pixels_per_class: int = 128,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dim = dim
        self.temperature = temperature
        self.momentum = momentum
        self.ignore_index = ignore_index
        self.max_pixels_per_class = max_pixels_per_class

        self.register_buffer("prototypes", torch.zeros(num_classes, dim))
        self.register_buffer("proto_ready", torch.zeros(num_classes, dtype=torch.bool))

    # -- sampling ---------------------------------------------------------
    def _sample_pixels(self, embed: torch.Tensor, labels: torch.Tensor):
        """Flattens a dense embedding map and draws a class-balanced subset.

        Args:
            embed: (B, D, h, w) L2-normalised pixel embeddings.
            labels: (B, h, w) integer labels at the same resolution.

        Returns:
            (M, D) embeddings and (M,) labels.
        """
        b, d, h, w = embed.shape
        feats = embed.permute(0, 2, 3, 1).reshape(-1, d)
        flat_labels = labels.reshape(-1)

        keep = (flat_labels != self.ignore_index) & (flat_labels >= 0) & (flat_labels < self.num_classes)
        feats, flat_labels = feats[keep], flat_labels[keep]
        if feats.numel() == 0:
            return feats, flat_labels

        chosen = []
        for c in torch.unique(flat_labels):
            idx = (flat_labels == c).nonzero(as_tuple=True)[0]
            if idx.numel() > self.max_pixels_per_class:
                perm = torch.randperm(idx.numel(), device=idx.device)[: self.max_pixels_per_class]
                idx = idx[perm]
            chosen.append(idx)
        chosen = torch.cat(chosen)
        return feats[chosen], flat_labels[chosen]

    # -- prototype maintenance -------------------------------------------
    @torch.no_grad()
    def update_prototypes(self, feats: torch.Tensor, labels: torch.Tensor) -> None:
        """EMA-updates the shared class centroids from detached pixel features.

        Call this with the concatenation of *both* domains' pixels so that each
        prototype is a genuinely domain-agnostic anchor.
        """
        if feats.numel() == 0:
            return
        feats = feats.detach().float()
        labels = labels.detach()

        if dist.is_available() and dist.is_initialized():
            feats, labels = _all_gather_cat(feats), _all_gather_cat(labels)

        for c in torch.unique(labels):
            c_int = int(c)
            mean = F.normalize(feats[labels == c].mean(dim=0), dim=0)
            if not bool(self.proto_ready[c_int]):
                self.prototypes[c_int] = mean
                self.proto_ready[c_int] = True
            else:
                updated = self.momentum * self.prototypes[c_int] + (1.0 - self.momentum) * mean
                self.prototypes[c_int] = F.normalize(updated, dim=0)

    # -- loss --------------------------------------------------------------
    def forward(
        self,
        embed_a: torch.Tensor,
        labels_a: torch.Tensor,
        embed_b: Optional[torch.Tensor] = None,
        labels_b: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes the pixel-to-prototype contrastive loss over both domains.

        Args:
            embed_a / labels_a: dense embeddings (B, D, h, w) and labels
                (B, h, w) for domain A.
            embed_b / labels_b: same for domain B; optional.

        Returns:
            Scalar loss (0 on the first steps, while prototypes initialise).
        """
        feats, labs = self._sample_pixels(embed_a, labels_a)
        if embed_b is not None:
            f_b, l_b = self._sample_pixels(embed_b, labels_b)
            feats = torch.cat([feats, f_b], dim=0)
            labs = torch.cat([labs, l_b], dim=0)

        if feats.numel() == 0:
            return embed_a.sum() * 0.0

        # Snapshot the prototypes: the loss must use the anchors as they were
        # *before* this batch's update, and the EMA write below is in-place on
        # the buffer, which would otherwise invalidate the saved tensor that
        # backward needs for the matmul.
        ready = self.proto_ready.clone()
        protos = self.prototypes.detach().clone().to(feats.dtype)

        loss = embed_a.sum() * 0.0
        usable = ready[labs]
        if usable.any() and int(ready.sum()) >= 2:
            logits = feats[usable] @ protos.t() / self.temperature
            logits = logits.masked_fill(~ready.unsqueeze(0), float("-inf"))
            loss = F.cross_entropy(logits, labs[usable])

        self.update_prototypes(feats, labs)
        return loss


# ----------------------------------------------------------------------------
# 4. Teacher/student contrastive distillation (CRD-style)
# ----------------------------------------------------------------------------
def crd_style_loss(
    z_student: torch.Tensor,
    z_teacher: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Contrastive distillation between student and teacher embeddings.

    Motivation from CRD [Tian20]: matching only the marginal output
    distribution (logit KD) discards the *structural* knowledge in the
    teacher's representation -- which inputs the teacher considers similar. CRD
    instead maximises a lower bound on the mutual information between student
    and teacher representations by making (student_i, teacher_i) a positive
    pair and (student_i, teacher_j) negatives. This matters here because the
    teacher is a ViT and the student a CNN: the two feature spaces are not
    dimensionally or spatially comparable, so relational/contrastive transfer
    is more appropriate than pointwise feature regression.

    Deviation from the original: CRD draws a large number of negatives from a
    memory bank and uses the NCE estimator with an explicit partition-function
    correction. This implementation uses the in-batch symmetric InfoNCE
    variant instead (as in CLIP-style cross-encoder alignment), which needs no
    memory bank and no dataset-size constant, at the cost of fewer negatives.

    Args:
        z_student: (N, D) L2-normalised student embeddings.
        z_teacher: (N, D) L2-normalised teacher embeddings (detached).
        temperature: InfoNCE temperature.
    """
    n = z_student.shape[0]
    if n < 2:
        return z_student.sum() * 0.0
    logits = z_student @ z_teacher.detach().t() / temperature
    targets = torch.arange(n, device=z_student.device)
    return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets))


# ----------------------------------------------------------------------------
# 5. MMD -- kept as a *diagnostic* of the residual domain gap
# ----------------------------------------------------------------------------
def rbf_mmd2(
    x: torch.Tensor,
    y: torch.Tensor,
    sigmas: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0),
) -> torch.Tensor:
    """Biased estimate of the squared multi-kernel MMD [Gretton12] between two
    embedding sets.

    Reported (not optimised) in the training module as a scale-free measure of
    how much marginal distribution gap survives between the two sensors.
    """
    def _k(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        d2 = torch.cdist(a, b, p=2) ** 2
        out = torch.zeros_like(d2)
        for s in sigmas:
            out = out + torch.exp(-d2 / (2.0 * s ** 2))
        return out / len(sigmas)

    n, m = x.shape[0], y.shape[0]
    if n < 2 or m < 2:
        return torch.zeros((), device=x.device)
    mmd2 = _k(x, x).mean() + _k(y, y).mean() - 2.0 * _k(x, y).mean()
    return mmd2.clamp_min(0.0)


def _all_gather_cat(t: torch.Tensor) -> torch.Tensor:
    """Gathers equal-length tensors across ranks; falls back to the local one.

    Ranks can hold different numbers of sampled pixels, in which case a plain
    all_gather would deadlock, so sizes are exchanged first and mismatches make
    this a no-op.
    """
    world = dist.get_world_size()
    local_n = torch.tensor([t.shape[0]], device=t.device)
    sizes = [torch.zeros_like(local_n) for _ in range(world)]
    dist.all_gather(sizes, local_n)
    if len({int(s) for s in sizes}) != 1:
        return t
    gathered = [torch.zeros_like(t) for _ in range(world)]
    dist.all_gather(gathered, t.contiguous())
    return torch.cat(gathered, dim=0)
