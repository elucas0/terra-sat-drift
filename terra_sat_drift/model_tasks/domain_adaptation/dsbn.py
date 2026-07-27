"""
Domain-Specific Batch Normalization (DSBN).

Reference
---------
Chang, You, Seo, Kwak & Han. "Domain-Specific Batch Normalization for
Unsupervised Domain Adaptation." CVPR 2019. arXiv:1906.03950

Why this exists in this project
-------------------------------
The student is a BatchNorm-heavy UNet that has to consume two sensors. With a
single shared BatchNorm there is a subtle but real train/test inconsistency:
during training the two domains are forwarded in separate passes, so each pass
is normalised by *its own* domain's batch statistics, but the running averages
accumulated for inference are a single blend of both domains. The encoder
therefore never sees at test time the normalisation it was optimised under.

DSBN removes that inconsistency by giving each domain its own normalisation
statistics and affine parameters while *all convolutional weights stay shared*.
Chang et al. show this both fixes the discrepancy and does part of the alignment
work itself, since matching first and second moments per domain is exactly what
whitening the covariate shift means.

Trade-off to be aware of: the deployed model is no longer literally a single
set of parameters -- you must select the target-domain branch at inference
(``set_domain(model, 1)``). If a strictly single-branch encoder is required,
leave DSBN off and rely on the representation-level losses alone.
"""

import torch
import torch.nn as nn


class DomainSpecificBatchNorm2d(nn.Module):
    """Holds one ``BatchNorm2d`` per domain and routes by the active domain id."""

    def __init__(self, num_features: int, num_domains: int = 2, **bn_kwargs):
        super().__init__()
        self.num_domains = num_domains
        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(num_features, **bn_kwargs) for _ in range(num_domains)]
        )
        # Plain python attribute, not a buffer: it is control flow, not state to
        # checkpoint, and keeping it out of the state dict means a DSBN model
        # stays loadable regardless of which branch was last active.
        self._domain = 0

    def set_domain(self, domain: int) -> None:
        if not 0 <= domain < self.num_domains:
            raise ValueError(f"domain {domain} out of range for {self.num_domains} domains")
        self._domain = domain

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bns[self._domain](x)


def convert_to_dsbn(module: nn.Module, num_domains: int = 2) -> nn.Module:
    """Recursively replaces every ``nn.BatchNorm2d`` with a DSBN block in place.

    Existing BatchNorm parameters and running statistics are copied into every
    domain branch, so conversion leaves the network's function unchanged at the
    moment it happens.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            dsbn = DomainSpecificBatchNorm2d(
                child.num_features,
                num_domains=num_domains,
                eps=child.eps,
                momentum=child.momentum,
                affine=child.affine,
                track_running_stats=child.track_running_stats,
            )
            for bn in dsbn.bns:
                bn.load_state_dict(child.state_dict())
            setattr(module, name, dsbn)
        else:
            convert_to_dsbn(child, num_domains=num_domains)
    return module


def set_domain(module: nn.Module, domain: int) -> None:
    """Activates a domain branch across every DSBN block in ``module``.

    A no-op on models that were never converted, so callers do not need to
    branch on whether DSBN is enabled.
    """
    for m in module.modules():
        if isinstance(m, DomainSpecificBatchNorm2d):
            m.set_domain(domain)


def has_dsbn(module: nn.Module) -> bool:
    """True if ``module`` contains at least one DSBN block."""
    return any(isinstance(m, DomainSpecificBatchNorm2d) for m in module.modules())
