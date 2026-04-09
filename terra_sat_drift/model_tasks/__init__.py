"""Model service tasks for classification and segmentation."""

from .classification_service import TerraMindClassifier
from .segmentation_service import TerraMindSegmenter

__all__ = [
    "TerraMindClassifier",
    "TerraMindSegmenter",
]
