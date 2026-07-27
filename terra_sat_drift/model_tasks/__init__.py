"""Model service tasks for classification and segmentation."""

from .kd_contrastive_module import CrossSensorKDModule
from .kd_module import KDSegmentationModule

__all__ = [
    "CrossSensorKDModule",
    "KDSegmentationModule",
]
