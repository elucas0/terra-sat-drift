"""Classification-specific drift analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..model_tasks import TerraMindClassifier
from .spectral_utils import SpectralAnalyzer, EmbeddingAnalyzer


class ClassificationDriftAnalyzer:
    """Compute drift metrics for classification tasks."""

    def __init__(self, classifier: TerraMindClassifier) -> None:
        """Initialize classification drift analyzer.
        
        Args:
            classifier: TerraMindClassifier instance for classification predictions.
        """
        self.classifier = classifier

    def compare_class_predictions(self, file1_tif: str | Path, file2_tif: str | Path) -> dict:
        """Measure prediction drift, including class flips and top-k consistency.
        
        Args:
            file1_tif: Path to first TIFF file.
            file2_tif: Path to second TIFF file.
            
        Returns:
            Dictionary with class predictions, flips, and consistency metrics.
        """
        pred1 = self.classifier.get_class_prediction(file1_tif)
        pred2 = self.classifier.get_class_prediction(file2_tif)

        class_flip = pred1["predicted_class"] != pred2["predicted_class"]
        prob_change = pred2["predicted_probability"] - pred1["predicted_probability"]
        top3_overlap = len(
            set([c for c, _ in pred1["top_3_classes"]])
            & set([c for c, _ in pred2["top_3_classes"]])
        )

        return {
            "file1": str(file1_tif),
            "file2": str(file2_tif),
            "pred1_class": pred1["predicted_class"],
            "pred1_probability": pred1["predicted_probability"],
            "pred2_class": pred2["predicted_class"],
            "pred2_probability": pred2["predicted_probability"],
            "class_flip": class_flip,
            "probability_change": prob_change,
            "top3_consistency": top3_overlap / 3,
            "pred1_top3": pred1["top_3_classes"],
            "pred2_top3": pred2["top_3_classes"],
        }

    def analyze_drift(self, file1_tif: str | Path, file2_tif: str | Path) -> dict:
        """Analyze complete classification drift including spectral, embedding, and predictions.
        
        Args:
            file1_tif: Path to first TIFF file.
            file2_tif: Path to second TIFF file.
            
        Returns:
            Dictionary with spectral, embedding, and classification drift metrics.
        """
        return {
            "spectral_drift": SpectralAnalyzer.compare_spectral_signature(file1_tif, file2_tif),
            "embedding_drift": EmbeddingAnalyzer.compare_stability(self.classifier, file1_tif, file2_tif),
            "class_drift": self.compare_class_predictions(file1_tif, file2_tif),
            "task": "classification",
            "file_pair": {
                "file1": str(file1_tif),
                "file2": str(file2_tif),
            },
        }

    def compare_directory(
        self,
        dir1: Path,
        dir2: Path,
        file1_pattern: str = "BANDS_RES-GRID",
        file2_pattern: str = "PHISAT2-BANDS-GRID",
        suffix: str = ".tiff",
    ) -> list[dict]:
        """Compute drift metrics for all matched files in two folders.
        
        Args:
            dir1: Directory containing first set of images.
            dir2: Directory containing second set of images.
            file1_pattern: Pattern to match in first directory files.
            file2_pattern: Pattern to match in second directory files.
            suffix: File suffix to search for.
            
        Returns:
            List of drift analysis results for each file pair.
        """
        results = []
        for file1_path in sorted(dir1.glob(f"*{suffix}")):
            sim_name = file1_path.name.replace(file1_pattern, file2_pattern)
            file2_path = dir2 / sim_name

            if not file2_path.exists():
                print(f"[WARN] No counterpart for {file1_path.name}, skipping.")
                continue

            results.append(self.analyze_drift(file1_path, file2_path))
        return results

    def analyze_embedding_directory(
        self,
        raw_dir: Path,
        simulated_dir: Path,
        file1_pattern: str = "BANDS_RES-GRID",
        file2_pattern: str = "PHISAT2-BANDS-GRID",
        suffix: str = ".tiff",
    ) -> list[dict]:
        """Compute embedding stability metrics for all matched files in two folders.
        
        Args:
            raw_dir: Directory containing raw images.
            simulated_dir: Directory containing simulated images.
            file1_pattern: Pattern to match in raw directory files.
            file2_pattern: Pattern to match in simulated directory files.
            suffix: File suffix to search for.
            
        Returns:
            List of embedding stability results for each file pair.
        """
        results = []
        for raw_path in sorted(raw_dir.glob(f"*{suffix}")):
            sim_name = raw_path.name.replace(file1_pattern, file2_pattern)
            sim_path = simulated_dir / sim_name

            if not sim_path.exists():
                print(f"[WARN] No simulated counterpart for {raw_path.name}, skipping.")
                continue

            stats = EmbeddingAnalyzer.compare_stability(self.classifier, raw_path, sim_path)
            stats["file"] = raw_path.name
            results.append(stats)
        return results

    @staticmethod
    def analyze_class_flips(results: list[dict]) -> dict:
        """Aggregate class flip statistics across multiple comparisons.
        
        Args:
            results: List of classification drift results.
            
        Returns:
            Dictionary with aggregated class flip statistics.
        """
        class_flips = [r for r in results if r["class_drift"]["class_flip"]]
        probabilities_changes = [r["class_drift"]["probability_change"] for r in results]
        top3_consistencies = [r["class_drift"]["top3_consistency"] for r in results]

        return {
            "total_pairs": len(results),
            "class_flips_count": len(class_flips),
            "class_flip_rate": len(class_flips) / len(results) if results else 0,
            "avg_probability_change": float(np.mean(probabilities_changes)) if probabilities_changes else 0,
            "max_probability_change": float(np.max(np.abs(probabilities_changes))) if probabilities_changes else 0,
            "avg_top3_consistency": float(np.mean(top3_consistencies)) if top3_consistencies else 0,
            "flipped_pairs": [
                {
                    "file1": f["class_drift"]["file1"],
                    "file2": f["class_drift"]["file2"],
                    "class_from": f["class_drift"]["pred1_class"],
                    "class_to": f["class_drift"]["pred2_class"],
                    "prob_change": f["class_drift"]["probability_change"],
                }
                for f in class_flips
            ],
        }
