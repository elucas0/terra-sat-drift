"""Drift analysis classes for spectral, embedding, and prediction stability."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F

from .model_service import TerraMindClassifier


class DriftAnalyzer:
    """Compute multiple drift metrics between pairs of Earth observation TIFFs."""

    def __init__(self, classifier: TerraMindClassifier) -> None:
        """Initialize analyzer with a configured classifier service."""
        self.classifier = classifier

    def extract_spectral_statistics(self, tif_path: str | Path) -> dict:
        """Extract robust per-band statistics for one TIFF file."""
        with rasterio.open(tif_path) as src:
            img = src.read().astype(np.float32)
            img = self.classifier._drop_panchromatic_if_needed(img)

        stats = {}
        for band_idx, band_name in enumerate(self.classifier.BAND_NAMES):
            band_data = img[band_idx, :, :]
            valid_data = band_data[band_data > 0]

            if len(valid_data) > 0:
                stats[band_name] = {
                    "mean": float(np.mean(valid_data)),
                    "std": float(np.std(valid_data)),
                    "min": float(np.min(valid_data)),
                    "max": float(np.max(valid_data)),
                    "median": float(np.median(valid_data)),
                    "percentile_25": float(np.percentile(valid_data, 25)),
                    "percentile_75": float(np.percentile(valid_data, 75)),
                }
            else:
                stats[band_name] = {
                    key: 0.0
                    for key in [
                        "mean",
                        "std",
                        "min",
                        "max",
                        "median",
                        "percentile_25",
                        "percentile_75",
                    ]
                }

        return stats

    def compare_spectral_signature(self, file1_tif: str | Path, file2_tif: str | Path) -> dict:
        """Compare band statistics and summarize spectral drift."""
        stats1 = self.extract_spectral_statistics(file1_tif)
        stats2 = self.extract_spectral_statistics(file2_tif)

        drift_analysis = {
            "file1": str(file1_tif),
            "file2": str(file2_tif),
            "bands": {},
        }

        all_mean_diffs = []
        all_std_changes = []

        for band_name in stats1.keys():
            s1 = stats1[band_name]
            s2 = stats2[band_name]

            mean_diff = s2["mean"] - s1["mean"]
            mean_diff_pct = (mean_diff / s1["mean"]) * 100 if s1["mean"] != 0 else 0
            std_change = s2["std"] - s1["std"]
            std_change_pct = (std_change / s1["std"]) * 100 if s1["std"] != 0 else 0

            all_mean_diffs.append(abs(mean_diff))
            all_std_changes.append(abs(std_change))

            drift_analysis["bands"][band_name] = {
                "file1_mean": s1["mean"],
                "file2_mean": s2["mean"],
                "mean_difference": mean_diff,
                "mean_diff_percent": mean_diff_pct,
                "file1_std": s1["std"],
                "file2_std": s2["std"],
                "std_change": std_change,
                "std_change_percent": std_change_pct,
                "range_change": (s2["max"] - s2["min"]) - (s1["max"] - s1["min"]),
            }

        drift_analysis["summary"] = {
            "avg_mean_difference": float(np.mean(all_mean_diffs)),
            "max_mean_difference": float(np.max(all_mean_diffs)),
            "avg_std_change": float(np.mean(all_std_changes)),
            "max_std_change": float(np.max(all_std_changes)),
        }
        return drift_analysis

    def compare_stability(self, clean_tif: str | Path, simulated_tif: str | Path) -> dict:
        """Compare encoder embedding stability between two inputs."""
        clean_input = self.classifier.load_tif_for_model(clean_tif)
        simu_input = self.classifier.load_tif_for_model(simulated_tif)

        pixel_diff = torch.abs(clean_input - simu_input).max().item()

        feat_clean_layers = self.classifier.extract_embeddings(clean_input)
        feat_simu_layers = self.classifier.extract_embeddings(simu_input)

        cosine_sims = []
        mse_errors = []
        mae_errors = []

        for i in range(min(len(feat_clean_layers), len(feat_simu_layers))):
            feat_clean = feat_clean_layers[i]
            feat_simu = feat_simu_layers[i]

            if feat_clean.shape != feat_simu.shape:
                continue

            cosine_sims.append(F.cosine_similarity(feat_clean, feat_simu, dim=1).mean().item())
            mse_errors.append(F.mse_loss(feat_clean, feat_simu).item())
            mae_errors.append(F.l1_loss(feat_clean, feat_simu).item())

        return {
            "cosine_similarity": sum(cosine_sims) / len(cosine_sims) if cosine_sims else 0.0,
            "mse_error": sum(mse_errors) / len(mse_errors) if mse_errors else 0.0,
            "mae_error": sum(mae_errors) / len(mae_errors) if mae_errors else 0.0,
            "pixel_max_diff": pixel_diff,
            "layer_cosine_similarities": cosine_sims,
            "layer_mse_errors": mse_errors,
            "layer_mae_errors": mae_errors,
        }

    def compare_class_predictions(self, file1_tif: str | Path, file2_tif: str | Path) -> dict:
        """Measure prediction drift, including class flips and top-k consistency."""
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

    def analyze_drift_comprehensive(self, file1_tif: str | Path, file2_tif: str | Path) -> dict:
        """Combine spectral, embedding, and classifier drift for one file pair."""
        return {
            "spectral_drift": self.compare_spectral_signature(file1_tif, file2_tif),
            "embedding_drift": self.compare_stability(file1_tif, file2_tif),
            "class_drift": self.compare_class_predictions(file1_tif, file2_tif),
            "file_pair": {
                "file1": str(file1_tif),
                "file2": str(file2_tif),
            },
        }

    def compare_directory_comprehensive(
        self,
        dir1: Path,
        dir2: Path,
        file1_pattern: str = "BANDS_RES-GRID",
        file2_pattern: str = "PHISAT2-BANDS-GRID",
        suffix: str = ".tiff",
    ) -> list[dict]:
        """Run comprehensive drift analysis over matched files in two folders."""
        results = []
        for file1_path in sorted(dir1.glob(f"*{suffix}")):
            sim_name = file1_path.name.replace(file1_pattern, file2_pattern)
            file2_path = dir2 / sim_name

            if not file2_path.exists():
                print(f"[WARN] No counterpart for {file1_path.name}, skipping.")
                continue

            results.append(self.analyze_drift_comprehensive(file1_path, file2_path))
        return results

    @staticmethod
    def analyze_class_flips(results: list[dict]) -> dict:
        """Aggregate class flip statistics across multiple comparisons."""
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

    def compare_directory(self, raw_dir: Path, simulated_dir: Path, suffix: str = ".tiff") -> list[dict]:
        """Compute embedding stability metrics for all matched files in two folders."""
        results = []
        for raw_path in sorted(raw_dir.glob(f"*{suffix}")):
            sim_name = raw_path.name.replace("BANDS_RES-GRID", "PHISAT2-BANDS-GRID")
            sim_path = simulated_dir / sim_name

            if not sim_path.exists():
                print(f"[WARN] No simulated counterpart for {raw_path.name}, skipping.")
                continue

            stats = self.compare_stability(raw_path, sim_path)
            stats["file"] = raw_path.name
            results.append(stats)
        return results
