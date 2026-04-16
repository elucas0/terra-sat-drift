"""Model loading and inference helpers for TerraMind semantic segmentation.

Backward compatibility module that re-exports from specialized submodules:
- segmentation_inference: TerraMindSegmenter for inference operations
- segmentation_training: TerraMindTrainer for fine-tuning utilities
"""

from __future__ import annotations

from .segmentation_inference import TerraMindSegmenter

__all__ = ["TerraMindSegmenter"]
