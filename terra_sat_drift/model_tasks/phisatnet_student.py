"""The PhiSat-2 foundation-model student (HydraNet / Phi2FM `PhisatNet`) wired up
as a frozen encoder with a freshly trained decoder.

The published student is a small U-Net: a ConvNeXt-block encoder at three
resolutions, a bottleneck, a mirrored decoder with skip connections, and a 1x1
classifier. It was trained on *simulated* PhiSat-2 imagery (Sentinel-2 pushed
through the PhiSat-2 sensor model). This module keeps its encoder and bottleneck,
throws the decoder away, and rebuilds it from scratch so it can be trained on
*real* PhiSat-2 -- the same frozen-encoder probe protocol used for TerraMind
elsewhere in this repo, applied to the sensor-specific model instead.

Three details are easy to get wrong and expensive to discover later:

**Channel order.** The stem is `Conv2d(8, 16, 1)` and its columns are in the
PhilEO order `B02, B03, B04, B08, B05, B06, B07, PAN`. The triplets HDF5 stores
PhiSat-2 as `PAN, Blue, Green, Red, RE1, RE2, RE3, NIR`. The ``real8`` / ``sim8``
/ ``s2b8`` domains in ``dataset.dataset_paired_triplets_lulc`` do the reorder;
this model must be fed one of those, never the plain 7-band ones.

**Missing layer-scale parameters.** None of the published checkpoints carry the
`*.convnext_block.gamma` tensors, because they were trained with layer scale
disabled -- so the residual branch was applied at full strength. `PhisatNet`
constructs `gamma` at its ConvNeXt default of `1e-6`, and
`hydranet.loading.load_student` fills the missing keys from *that* default,
which scales every residual branch in the network down by a factor of a million
and leaves an encoder that is very nearly the identity function. Loading here
fills them with **ones** instead, matching the reference notebook
(`MISSING_CONVNEXT_GAMMA = "ones"` in `notebooks/decoders/4_land_cover_classification.ipynb`).
``--gamma-fill model_init`` reproduces the other behaviour if you want to see it.

**BatchNorm in a "frozen" encoder.** `requires_grad = False` stops the weights
moving but does *not* stop BatchNorm running statistics from tracking whatever
comes through in training mode. Since the whole point here is a domain shift
between what the encoder was trained on and what it now sees, that is not a
detail: leaving BN to adapt is a real (and often effective) domain-adaptation
intervention, and it would otherwise happen silently under a "frozen encoder"
label. ``bn_mode`` makes the choice explicit and ``"frozen"`` is the default.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn

# The `hydranet` package is a sibling checkout rather than an installed
# dependency (it pins python 3.9 and torch 2.4, neither of which this project
# uses; the model code itself is plain torch and runs fine here).
DEFAULT_HYDRANET_SRC = Path("/shared/home/elucas/hydranet-phisat2/src")

# Channel contract of the published checkpoints, kept next to the model that
# depends on it so a reader does not have to go to the dataset to find it.
PHISATNET_BAND_ORDER = ("B02", "B03", "B04", "B08", "B05", "B06", "B07", "PAN")

ENCODER_PREFIXES = ("encoders.", "pools.")
BOTTLENECK_PREFIXES = ("bottleneck.",)
DECODER_PREFIXES = ("upsamplers.", "decoders.", "final_conv.")


def _ensure_hydranet_importable(hydranet_src: Optional[Path] = None) -> None:
    try:
        import hydranet.models.student  # noqa: F401
        return
    except ImportError:
        pass
    src = Path(hydranet_src or DEFAULT_HYDRANET_SRC)
    if not (src / "hydranet" / "models" / "student.py").is_file():
        raise ImportError(
            f"Could not import `hydranet` and no package found at {src}. "
            "Point --hydranet-src at the `src` directory of the hydranet-phisat2 checkout."
        )
    sys.path.insert(0, str(src))
    import hydranet.models.student  # noqa: F401


def resolve_checkpoint(
    task: str = "lc",
    training: str = "finetuning",
    n_shots: int = 5000,
    datetime: Optional[int] = None,
    weights_dir: Optional[str] = None,
    hydranet_src: Optional[Path] = None,
) -> Path:
    """Local path to one published student checkpoint, downloading if needed.

    `datetime=None` takes the most recent release for that combination. The
    releases are *not* reruns of one model: their encoders differ by up to ~2.0
    in max absolute weight difference, so which one is used is a real
    experimental choice and belongs in the run record, not in a default nobody
    reads. `scripts/audit_hydranet_checkpoints.py` prints the differences.
    """
    _ensure_hydranet_importable(hydranet_src)
    from huggingface_hub import hf_hub_download
    from hydranet.weights import HF_REPO, _load_catalog

    df = _load_catalog()
    sel = df[(df.training == training) & (df.model == "student") & (df.task == task)
             & (df.n_shots == float(n_shots))]
    if datetime is not None:
        sel = sel[sel.datetime == int(datetime)]
    if len(sel) == 0:
        combos = (df[df.model == "student"].groupby(["training", "task"])["n_shots"]
                  .unique().to_dict())
        raise ValueError(
            f"No student checkpoint for training={training!r} task={task!r} "
            f"n_shots={n_shots} datetime={datetime}. Available: {combos}"
        )
    row = sel.sort_values("datetime").iloc[-1]
    return Path(hf_hub_download(repo_id=HF_REPO, filename=row.file_path,
                                repo_type="dataset", local_dir=weights_dir))


def _remap_checkpoint(state: dict, model: nn.Module, gamma_fill: str = "ones") -> dict:
    """Published checkpoint -> `PhisatNet` state dict, plus the missing gammas.

    Returns ``(patched_state, filled_gamma_keys)``; the caller reports what did
    and did not match.
    """
    patched = dict(state)

    # The published head is called `classifier`; `PhisatNet` calls it `final_conv`.
    for suffix in ("weight", "bias"):
        src, dst = f"classifier.{suffix}", f"final_conv.{suffix}"
        if dst not in patched and src in patched:
            patched[dst] = patched.pop(src)

    model_state = model.state_dict()
    filled = []
    for key in model_state:
        if key.endswith(".convnext_block.gamma") and key not in patched:
            patched[key] = (torch.ones_like(model_state[key]) if gamma_fill == "ones"
                            else model_state[key].clone())
            filled.append(key)
    return patched, filled


class PhisatNetSegmenter(nn.Module):
    """`PhisatNet` with the head resized for this task and the parts labelled.

    `forward(x, return_features=True)` returns the same dict shape as
    ``student_mobilenet.UNetStudent`` (``logits`` / ``bottleneck`` / ``decoder``),
    so this drops into the KD and contrastive modules unchanged if it is ever
    used as a student there rather than as a probe.
    """

    def __init__(self, net: nn.Module, frozen_prefixes: Sequence[str] = (),
                 bn_mode: str = "frozen"):
        super().__init__()
        self.net = net
        self.frozen_prefixes = tuple(frozen_prefixes)
        self.bn_mode = bn_mode
        self.in_channels = net.n_channels
        self.num_classes = net.n_classes

    # -- freezing ------------------------------------------------------------
    def _frozen_modules(self):
        for name, module in self.net.named_children():
            if any(f"{name}.".startswith(p) for p in self.frozen_prefixes):
                yield name, module

    def freeze(self) -> dict:
        """Freezes the configured prefixes; returns a parameter-count summary."""
        for p in self.net.parameters():
            p.requires_grad = True
        for _, module in self._frozen_modules():
            for p in module.parameters():
                p.requires_grad = False
        self._apply_bn_mode()
        total = sum(p.numel() for p in self.net.parameters())
        trainable = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}

    def _apply_bn_mode(self) -> None:
        """Puts frozen submodules in eval mode when BN is meant to stay frozen."""
        if self.training and self.bn_mode == "frozen":
            for _, module in self._frozen_modules():
                module.eval()

    def train(self, mode: bool = True):
        """Keeps frozen BatchNorm in eval mode unless BN adaptation was asked for.

        Without this, `model.train()` would put the frozen encoder's BatchNorm
        layers into training mode and let their running statistics drift onto the
        target sensor -- an unannounced domain adaptation inside a run labelled
        "frozen encoder".
        """
        super().train(mode)
        self._apply_bn_mode()
        return self

    # -- forward -------------------------------------------------------------
    def forward(self, x: torch.Tensor, return_features: bool = False):
        # Re-asserted per call, not just in `train()`, because the training loop
        # is not obliged to route mode changes through this module. Lightning
        # 2.6 never calls `.train()` during `fit` at all -- it relies on
        # `nn.Module`'s default of True -- and its evaluation loop restores
        # per-submodule modes directly by name after every validation pass. Both
        # bypass the `train()` override above, and the symptom is silent: the run
        # still trains, it just quietly adapts BatchNorm inside a "frozen
        # encoder" experiment. One cheap flag-set per forward removes the whole
        # class of problem.
        self._apply_bn_mode()
        if not return_features:
            return self.net(x)

        net = self.net
        skips = []
        current = x
        for i in range(net.depth):
            current = net.encoders[i](current)
            skips.append(current)
            if i < net.depth - 1:
                current = net.pools[i](current)
        current = net.pools[-1](current)
        bottleneck = net.bottleneck(current)

        current = bottleneck
        for i in range(net.depth):
            current = net.upsamplers[i](current)
            current = torch.cat([current, skips[net.depth - 1 - i]], dim=1)
            current = net.decoders[i](current)
        return {"logits": net.final_conv(current), "bottleneck": bottleneck,
                "decoder": current}


def create_phisatnet_student(
    num_classes: int,
    checkpoint: Optional[Path] = None,
    in_channels: int = 8,
    freeze: str = "encoder+bottleneck",
    reinit_decoder: bool = True,
    gamma_fill: str = "ones",
    bn_mode: str = "frozen",
    hydranet_src: Optional[Path] = None,
    verbose: bool = True,
) -> PhisatNetSegmenter:
    """Builds the student, loads a published checkpoint, freezes and resets parts.

    Args:
        num_classes: output classes; the 1x1 head is rebuilt at this width even
            when the checkpoint head already has it, so the head is never
            inherited by accident.
        checkpoint: path from `resolve_checkpoint`. `None` gives random init,
            which is the control that says how much the pretrained encoder is
            actually worth.
        in_channels: 8 for the native contract. Any other value rebuilds the stem
            from scratch, discarding pretrained stem weights -- it is supported so
            a 7-band ablation is possible, not because it is a good idea.
        freeze: ``"encoder+bottleneck"``, ``"encoder"``, or ``"none"``.
        reinit_decoder: `True` trains a new decoder (the point of this script);
            `False` starts from the published one, which is a warm start rather
            than a new head.
        gamma_fill: ``"ones"`` (correct, see module docstring) or ``"model_init"``.
        bn_mode: ``"frozen"`` or ``"adapt"``.
    """
    _ensure_hydranet_importable(hydranet_src)
    from hydranet.models.student import create_phisatnet

    net = create_phisatnet(config="checkpoint", n_channels=in_channels,
                           n_classes=num_classes)

    report = {"checkpoint": str(checkpoint) if checkpoint else None}
    if checkpoint is not None:
        raw = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
        patched, filled = _remap_checkpoint(raw, net, gamma_fill=gamma_fill)

        if in_channels != 8:
            # The stem is the only shape-dependent tensor; drop it so
            # load_state_dict reports it as missing rather than erroring.
            for suffix in ("weight", "bias"):
                patched.pop(f"encoders.0.channel_proj.{suffix}", None)
        # The head is rebuilt below in every case, so a checkpoint head of a
        # different width is not an error worth stopping for.
        if patched.get("final_conv.weight") is not None and \
                patched["final_conv.weight"].shape[0] != num_classes:
            patched.pop("final_conv.weight", None)
            patched.pop("final_conv.bias", None)

        missing, unexpected = net.load_state_dict(patched, strict=False)
        stem_keys = {f"encoders.0.channel_proj.{s}" for s in ("weight", "bias")}
        encoder_missing = [k for k in missing
                           if k.startswith(ENCODER_PREFIXES + BOTTLENECK_PREFIXES)
                           and not (in_channels != 8 and k in stem_keys)]
        if encoder_missing:
            raise RuntimeError(
                "Encoder/bottleneck weights missing from the checkpoint, so the "
                f"'pretrained encoder' would be partly random: {encoder_missing[:8]}"
            )
        report.update(missing=list(missing), unexpected=list(unexpected),
                      gamma_filled=len(filled), gamma_fill=gamma_fill)
        if verbose:
            print(f"Loaded {checkpoint}")
            print(f"  layer-scale gammas filled with {gamma_fill}: {len(filled)}")
            print(f"  missing keys: {len(missing)}  unexpected keys: {len(unexpected)}")
            if unexpected:
                print(f"    unexpected: {sorted(unexpected)[:6]}")
    elif verbose:
        print("No checkpoint given: PhisatNet is randomly initialised (control run).")

    # A fresh head every time, and a fresh decoder when asked for one. Done after
    # loading so nothing from the checkpoint survives in these parts.
    net.final_conv = nn.Conv2d(net.channels[0], num_classes, kernel_size=1)
    net.n_classes = num_classes
    if reinit_decoder:
        for module in (net.upsamplers, net.decoders):
            for sub in module.modules():
                if hasattr(sub, "reset_parameters"):
                    sub.reset_parameters()
        # Layer scale at 1 rather than ConvNeXt's 1e-6: the checkpoints were
        # trained with layer scale disabled, so a fresh decoder that starts at 1
        # behaves like the one being replaced instead of starting as a near-linear
        # projection that has to climb back out.
        for name, param in net.named_parameters():
            if name.startswith("decoders.") and name.endswith(".convnext_block.gamma"):
                nn.init.ones_(param)

    prefixes = {
        "encoder+bottleneck": ENCODER_PREFIXES + BOTTLENECK_PREFIXES,
        "encoder": ENCODER_PREFIXES,
        "none": (),
    }
    if freeze not in prefixes:
        raise ValueError(f"freeze must be one of {sorted(prefixes)}, got {freeze!r}")

    model = PhisatNetSegmenter(net, frozen_prefixes=prefixes[freeze], bn_mode=bn_mode)
    counts = model.freeze()
    model.load_report = report
    if verbose:
        print(f"  freeze={freeze} bn_mode={bn_mode} reinit_decoder={reinit_decoder}")
        print(f"  parameters: {counts['total']/1e6:.3f}M total, "
              f"{counts['trainable']/1e6:.3f}M trainable, "
              f"{counts['frozen']/1e6:.3f}M frozen")
    return model
