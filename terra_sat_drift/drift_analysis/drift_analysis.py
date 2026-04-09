"""Drift analysis classes for spectral, embedding, and prediction stability.

This module provides a unified interface for drift analysis across different tasks.
For task-specific analysis, use the specialized analyzers:
- ClassificationDriftAnalyzer: for classification tasks
- SegmentationDriftAnalyzer: for segmentation tasks
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..model_tasks import TerraMindClassifier, TerraMindSegmenter
from .spectral_utils import SpectralAnalyzer, EmbeddingAnalyzer
from .classification_drift_analyzer import ClassificationDriftAnalyzer
from .segmentation_drift_analyzer import SegmentationDriftAnalyzer


class DriftAnalyzer:
    """Unified drift analyzer supporting both classification and segmentation tasks.
    
    For backwards compatibility and convenience, this class delegates to specialized analyzers
    based on the task type. Use ClassificationDriftAnalyzer or SegmentationDriftAnalyzer
    directly for clearer, task-specific code.
    """

    def __init__(
        self, 
        classifier: TerraMindClassifier,
        segmenter: TerraMindSegmenter | None = None,
    ) -> None:
        """Initialize analyzer with classifier and optional segmenter.
        
        Args:
            classifier: TerraMindClassifier instance for classification tasks.
            segmenter: Optional TerraMindSegmenter instance for segmentation tasks.
                      If None, segmentation-related methods cannot be used.
        """
        self.classifier = classifier
        self.segmenter = segmenter
        
        # Initialize task-specific analyzers
        self.classification_analyzer = ClassificationDriftAnalyzer(classifier)
        if segmenter:
            self.segmentation_analyzer = SegmentationDriftAnalyzer(classifier, segmenter)
        else:
            self.segmentation_analyzer = None

    def extract_spectral_statistics(self, tif_path: str | Path) -> dict:
        """Extract robust per-band statistics for one TIFF file."""
        return SpectralAnalyzer.extract_spectral_statistics(tif_path)

    def compare_spectral_signature(self, file1_tif: str | Path, file2_tif: str | Path) -> dict:
        """Compare band statistics and summarize spectral drift."""
        return SpectralAnalyzer.compare_spectral_signature(file1_tif, file2_tif)

    def compare_stability(self, clean_tif: str | Path, simulated_tif: str | Path) -> dict:
        """Compare encoder embedding stability between two inputs."""
        return EmbeddingAnalyzer.compare_stability(self.classifier, clean_tif, simulated_tif)

    def compare_class_predictions(self, file1_tif: str | Path, file2_tif: str | Path) -> dict:
        """Measure prediction drift, including class flips and top-k consistency."""
        return self.classification_analyzer.compare_class_predictions(file1_tif, file2_tif)

    def analyze_drift_comprehensive(self, file1_tif: str | Path, file2_tif: str | Path, task: str = "classification") -> dict:
        """Combine spectral, embedding, and classifier/segmentation drift for one file pair.
        
        Args:
            file1_tif: Path to first TIFF file.
            file2_tif: Path to second TIFF file.
            task: Type of task - "classification" or "segmentation". Determines which drift metrics to compute.
        """
        if task == "classification":
            return self.classification_analyzer.analyze_drift(file1_tif, file2_tif)
        elif task == "segmentation":
            raise ValueError("For segmentation drift, use analyze_drift_with_segmentation() with ground truth.")
        else:
            raise ValueError(f"Unknown task: {task}. Supported: 'classification', 'segmentation'")

    def compare_directory_comprehensive(
        self,
        dir1: Path,
        dir2: Path,
        file1_pattern: str = "BANDS_RES-GRID",
        file2_pattern: str = "PHISAT2-BANDS-GRID",
        suffix: str = ".tiff",
    ) -> list[dict]:
        """Run comprehensive drift analysis over matched files in two folders."""
        return self.classification_analyzer.compare_directory(
            dir1, dir2, file1_pattern, file2_pattern, suffix
        )

    def analyze_class_flips(self, results: list[dict]) -> dict:
        """Aggregate class flip statistics across multiple comparisons."""
        return ClassificationDriftAnalyzer.analyze_class_flips(results)

    def compare_directory(
        self,
        raw_dir: Path,
        simulated_dir: Path,
        file1_pattern: str = "BANDS_RES-GRID",
        file2_pattern: str = "PHISAT2-BANDS-GRID",
        suffix: str = ".tiff",
    ) -> list[dict]:
        """Compute embedding stability metrics for all matched files in two folders."""
        return self.classification_analyzer.analyze_embedding_directory(
            raw_dir, simulated_dir, file1_pattern, file2_pattern, suffix
        )

    def compare_segmentation(
        self, raw_tif: str | Path, simulated_tif: str | Path, ground_truth_mask: np.ndarray
    ) -> dict:
        """Compare binary segmentation predictions between raw and simulated imagery."""
        if self.segmentation_analyzer is None:
            raise RuntimeError(
                "Segmenter not available. Initialize DriftAnalyzer with a "
                "TerraMindSegmenter to use segmentation methods."
            )
        return self.segmentation_analyzer.compare_segmentation(raw_tif, simulated_tif, ground_truth_mask)

    def analyze_drift_with_segmentation(
        self, raw_tif: str | Path, simulated_tif: str | Path, ground_truth_mask: np.ndarray
    ) -> dict:
        """Combine spectral, embedding, classification and segmentation drift.
        
        Performs comprehensive drift analysis including the new binary segmentation task.
        
        Args:
            raw_tif: Path to raw S2 TIFF.
            simulated_tif: Path to simulated Φ-sat-2 TIFF.
            ground_truth_mask: Binary ground truth mask.
            
        Returns:
            Dictionary with all drift analyses combined.
        """
        if self.segmentation_analyzer is None:
            raise RuntimeError(
                "Segmenter not available. Initialize DriftAnalyzer with a "
                "TerraMindSegmenter to use segmentation methods."
            )
        return self.segmentation_analyzer.analyze_drift(raw_tif, simulated_tif, ground_truth_mask)

    @staticmethod
    def analyze_segmentation_drift(results: list[dict]) -> dict:
        """Aggregate segmentation drift statistics across multiple comparisons."""
        return SegmentationDriftAnalyzer.analyze_segmentation_drift(results)
