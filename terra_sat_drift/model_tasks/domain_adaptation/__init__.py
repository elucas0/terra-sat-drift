"""Domain adaptation utilities (domain-specific normalization, MMD baselines)."""

from .dsbn import DomainSpecificBatchNorm2d, convert_to_dsbn, has_dsbn, set_domain

__all__ = [
    "DomainSpecificBatchNorm2d",
    "convert_to_dsbn",
    "has_dsbn",
    "set_domain",
]
