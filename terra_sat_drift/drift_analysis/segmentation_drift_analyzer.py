"""Segmentation-specific drift analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..model_tasks import TerraMindSegmenter
from .spectral_utils import SpectralAnalyzer, EmbeddingAnalyzer


class SegmentationDriftAnalyzer:
    """Compute drift metrics for semantic segmentation tasks."""

    def __init__(self, classifier, segmenter: TerraMindSegmenter) -> None:
        """Initialize segmentation drift analyzer.
        
        Args:
            classifier: TerraMindClassifier instance (used for embedding analysis).
            segmenter: TerraMindSegmenter instance for segmentation predictions.
        """
        self.classifier = classifier
        self.segmenter = segmenter

    def compare_segmentation(
        self, raw_tif: str | Path, simulated_tif: str | Path, ground_truth_mask: np.ndarray
    ) -> dict:
        """Compare binary segmentation predictions between raw and simulated imagery.
        
        Performs segmentation on both raw and simulated images and compares predictions
        against ground truth, measuring segmentation drift and consistency.
        
        Args:
            raw_tif: Path to raw S2 TIFF file.
            simulated_tif: Path to simulated Φ-sat-2 TIFF file.
            ground_truth_mask: Binary ground truth mask (height, width, values 0 or 1).
            
        Returns:
            Dictionary with segmentation drift metrics including IoU, Dice, and accuracy drift.
            
        Raises:
            RuntimeError: If segmenter is not available.
        """
        if self.segmenter is None:
            raise RuntimeError(
                "Segmenter not available. Initialize with a TerraMindSegmenter."
            )
        
        # Segment raw image
        raw_seg = self.segmenter.segment_image(raw_tif)
        raw_pred = raw_seg["segmentation"]
        raw_metrics = self.segmenter.compute_segmentation_metrics(raw_pred, ground_truth_mask)
        
        # Segment simulated image
        simulated_seg = self.segmenter.segment_image(simulated_tif)
        simulated_pred = simulated_seg["segmentation"]
        simulated_metrics = self.segmenter.compute_segmentation_metrics(
            simulated_pred, ground_truth_mask
        )
        
        # Compute prediction agreement (Dice between two predictions)
        pred_agreement = self.segmenter.compute_segmentation_metrics(
            raw_pred, simulated_pred
        )
        
        # Drift metrics
        iou_drift = simulated_metrics["iou"] - raw_metrics["iou"]
        dice_drift = simulated_metrics["dice"] - raw_metrics["dice"]
        accuracy_drift = simulated_metrics["accuracy"] - raw_metrics["accuracy"]
        f1_drift = simulated_metrics["f1_score"] - raw_metrics["f1_score"]
        
        return {
            "file_pair": {
                "raw": str(raw_tif),
                "simulated": str(simulated_tif),
            },
            "raw_segmentation_metrics": raw_metrics,
            "simulated_segmentation_metrics": simulated_metrics,
            "prediction_agreement": {
                "dice": pred_agreement["dice"],
                "iou": pred_agreement["iou"],
            },
            "drift_metrics": {
                "iou_drift": iou_drift,
                "dice_drift": dice_drift,
                "accuracy_drift": accuracy_drift,
                "f1_drift": f1_drift,
            }
        }

    def analyze_drift(
        self, raw_tif: str | Path, simulated_tif: str | Path, ground_truth_mask: np.ndarray
    ) -> dict:
        """Analyze complete segmentation drift including spectral, embedding, and segmentation.
        
        Args:
            raw_tif: Path to raw S2 TIFF.
            simulated_tif: Path to simulated Φ-sat-2 TIFF.
            ground_truth_mask: Binary ground truth mask.
            
        Returns:
            Dictionary with spectral, embedding, and segmentation drift metrics.
        """
        return {
            "spectral_drift": SpectralAnalyzer.compare_spectral_signature(raw_tif, simulated_tif),
            "embedding_drift": EmbeddingAnalyzer.compare_stability(self.classifier, raw_tif, simulated_tif),
            "segmentation_drift": self.compare_segmentation(raw_tif, simulated_tif, ground_truth_mask),
            "task": "segmentation",
            "file_pair": {
                "raw": str(raw_tif),
                "simulated": str(simulated_tif),
            },
        }

    @staticmethod
    def analyze_segmentation_drift(results: list[dict]) -> dict:
        """Aggregate segmentation drift statistics across multiple comparisons.
        
        Args:
            results: List of segmentation drift comparison dictionaries.
            
        Returns:
            Dictionary with aggregated segmentation metrics.
        """
        if not results:
            return {
                "total_pairs": 0,
                "avg_raw_iou": 0.0,
                "avg_simulated_iou": 0.0,
                "avg_iou_drift": 0.0,
                "avg_prediction_agreement": 0.0,
            }
        
        raw_ious = [r["raw_segmentation_metrics"]["iou"] for r in results]
        sim_ious = [r["simulated_segmentation_metrics"]["iou"] for r in results]
        iou_drifts = [r["drift_metrics"]["iou_drift"] for r in results]
        agreements = [r["prediction_agreement"]["iou"] for r in results]
        
        raw_dices = [r["raw_segmentation_metrics"]["dice"] for r in results]
        sim_dices = [r["simulated_segmentation_metrics"]["dice"] for r in results]
        dice_drifts = [r["drift_metrics"]["dice_drift"] for r in results]
        
        raw_f1s = [r["raw_segmentation_metrics"]["f1_score"] for r in results]
        sim_f1s = [r["simulated_segmentation_metrics"]["f1_score"] for r in results]
        f1_drifts = [r["drift_metrics"]["f1_drift"] for r in results]
        
        return {
            "total_pairs": len(results),
            "raw_segmentation": {
                "avg_iou": float(np.mean(raw_ious)),
                "avg_dice": float(np.mean(raw_dices)),
                "avg_f1": float(np.mean(raw_f1s)),
                "std_iou": float(np.std(raw_ious)),
                "std_dice": float(np.std(raw_dices)),
                "std_f1": float(np.std(raw_f1s)),
            },
            "simulated_segmentation": {
                "avg_iou": float(np.mean(sim_ious)),
                "avg_dice": float(np.mean(sim_dices)),
                "avg_f1": float(np.mean(sim_f1s)),
                "std_iou": float(np.std(sim_ious)),
                "std_dice": float(np.std(sim_dices)),
                "std_f1": float(np.std(sim_f1s)),
            },
            "drift": {
                "avg_iou_drift": float(np.mean(iou_drifts)),
                "max_iou_drift": float(np.max(np.abs(iou_drifts))),
                "avg_dice_drift": float(np.mean(dice_drifts)),
                "max_dice_drift": float(np.max(np.abs(dice_drifts))),
                "avg_f1_drift": float(np.mean(f1_drifts)),
                "max_f1_drift": float(np.max(np.abs(f1_drifts))),
            },
            "prediction_agreement": {
                "avg_iou": float(np.mean(agreements)),
                "min_iou": float(np.min(agreements)),
            },
        }
