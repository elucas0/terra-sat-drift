"""Task losses for imbalanced semantic segmentation.

Motivated by the observed failure mode on the WorldCover triplets: under plain
cross-entropy the student never predicts snow/ice, mangroves or moss/lichen at
all (IoU exactly 0.0 on *train* as well as val, so it is a failure to learn, not
overfitting), and shrubland reaches only ~0.08 despite being ~6% of pixels.
Because the reported mIoU is a macro average, those three dead classes are 1.2%
of the pixels but 27% of the metric.
"""

import warnings
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F


def focal_ce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    alpha: Optional[torch.Tensor] = None,
    ignore_index: int = -1,
) -> torch.Tensor:
    """Multi-class focal loss [Lin+ 2017, RetinaNet], averaged over valid pixels.

    ``FL = -alpha_c (1 - p_t)^gamma log(p_t)``, which down-weights pixels the
    model already classifies confidently so that gradient mass moves to the hard
    and rare ones. ``gamma=0`` reduces exactly to (weighted) cross-entropy.

    Computed from log-probabilities and an explicit gather rather than the common
    ``pt = exp(-ce)`` shortcut, because that identity only holds when there are no
    class weights -- with ``alpha`` set, ``ce = -alpha_c log p_t`` and ``exp(-ce)``
    is not ``p_t``.

    Note on scale: for ``gamma > 0`` the loss is strictly smaller than CE (the
    modulating factor is <= 1), typically by a factor of several once training
    settles. Since this term is summed against the distillation and contrastive
    terms with fixed weights, switching CE -> focal *implicitly raises* the
    relative weight of those other terms. Re-check ``w_kd`` against
    ``w_task_target`` after changing gamma.

    Note on ``alpha`` normalisation: this averages over valid pixels, whereas
    ``F.cross_entropy(weight=...)`` divides by the sum of the weights of the
    target pixels. Ours keeps the loss magnitude independent of which classes
    happen to land in a batch, so the task-vs-distillation balance does not drift
    when a rare-class patch appears; the two therefore differ by a batch-dependent
    constant, and ``gamma=0`` matches plain CE exactly only when ``alpha is None``.

    Args:
        logits: (B, C, H, W) raw scores.
        target: (B, H, W) int64 class ids, may contain ``ignore_index``.
        gamma: focusing parameter; 2.0 is the value used throughout [Lin+ 2017].
        alpha: optional (C,) per-class weights, on any device.
        ignore_index: label id to exclude from the average.
    """
    valid = target != ignore_index
    if not bool(valid.any()):
        return logits.sum() * 0.0

    # ignore pixels carry a negative id; clamp so gather stays in range, then
    # remove their contribution with the mask.
    safe = target.clamp_min(0)
    logp = F.log_softmax(logits, dim=1)
    logpt = logp.gather(1, safe.unsqueeze(1)).squeeze(1)
    loss = -((1.0 - logpt.exp()) ** gamma) * logpt

    if alpha is not None:
        loss = loss * alpha.to(device=logits.device, dtype=loss.dtype)[safe]

    loss = loss * valid
    return loss.sum() / valid.sum().clamp_min(1)


def class_weights(
    frequencies: Sequence[float],
    scheme: str = "inverse_sqrt",
    beta: float = 0.999,
    n_total: float = 1e4,
    normalize: bool = True,
) -> np.ndarray:
    """Per-class weights from class frequencies.

    Schemes:
        ``none``          all ones.
        ``inverse``       w ∝ 1/f. Strongest correction and the least stable: on
                          this dataset it gives moss/lichen ~60x the weight of
                          tree cover, which tends to trade the common classes away.
        ``inverse_sqrt``  w ∝ 1/sqrt(f). The usual compromise (~8x here), and the
                          recommended starting point.
        ``effective``     Effective number of samples [Cui+ 2019],
                          w ∝ (1-beta)/(1-beta^n_c) with n_c = f_c * n_total.

    On ``effective``: the scheme was designed for *sample* counts, and it only
    separates classes while ``n_c`` is within about ``1/(1-beta)``. Feeding it
    raw pixel counts makes it a silent no-op -- this dataset has ~1.7e10 labelled
    pixels, so every class saturates and all weights come out ~1.0. ``n_total``
    therefore defaults to an effective *independent-sample* count rather than a
    pixel count, on the grounds that pixels inside a 256x256 patch are strongly
    correlated. A warning is emitted if the resulting weights are near-uniform,
    which means ``beta`` is too low for the ``n_total`` given.

    Weights are normalised to mean 1 by default so that swapping schemes does not
    silently rescale the task loss relative to the distillation term.
    """
    f = np.asarray(frequencies, dtype=np.float64)
    f = f / f.sum()

    if scheme == "none":
        w = np.ones_like(f)
    elif scheme == "inverse":
        w = 1.0 / np.clip(f, 1e-12, None)
    elif scheme == "inverse_sqrt":
        w = 1.0 / np.sqrt(np.clip(f, 1e-12, None))
    elif scheme == "effective":
        n_c = np.clip(f * n_total, 1.0, None)
        w = (1.0 - beta) / np.clip(1.0 - np.power(beta, n_c), 1e-12, None)
    else:
        raise ValueError(f"Unknown class-weight scheme {scheme!r}")

    if normalize:
        w = w * len(w) / w.sum()

    if scheme != "none" and w.max() / max(w.min(), 1e-12) < 1.05:
        warnings.warn(
            f"class_weights(scheme={scheme!r}) produced near-uniform weights "
            f"(max/min = {w.max() / max(w.min(), 1e-12):.3f}); it will have no "
            f"practical effect. For 'effective', raise beta or lower n_total "
            f"(currently beta={beta}, n_total={n_total:g}).",
            stacklevel=2,
        )
    return w.astype(np.float32)
