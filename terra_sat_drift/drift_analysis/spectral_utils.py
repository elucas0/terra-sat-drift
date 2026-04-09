"""Shared spectral and embedding utilities for drift analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F

from ..model_tasks import TerraMindClassifier


class SpectralAnalyzer:
    """Shared utility for spectral and embedding drift analysis."""

    @staticmethod
    def extract_spectral_statistics(tif_path: str | Path) -> dict:
        """Extract robust per-band statistics for one TIFF file.
        
        Args:
            tif_path: Path to Sentinel-2 TIFF file.
            
        Returns:
            Dictionary with band-wise statistics (mean, std, min, max, median, percentiles).
        """
        band_indices = [1, 2, 3, 7, 4, 5, 6]
        with rasterio.open(tif_path) as src:
            img = src.read().astype(np.float32)
            img = img[band_indices, :, :]

        # Normalize Sentinel-2 data to 0-1 range
        img = np.clip(img / 10000.0, 0.0, 1.0)

        stats = {}
        for band_idx, band_name in enumerate(band_indices):
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

    @staticmethod
    def compare_spectral_signature(file1_tif: str | Path, file2_tif: str | Path) -> dict:
        """Compare band statistics and summarize spectral drift.
        
        Args:
            file1_tif: Path to first TIFF file.
            file2_tif: Path to second TIFF file.
            
        Returns:
            Dictionary with spectral drift analysis for each band and summary statistics.
        """
        stats1 = SpectralAnalyzer.extract_spectral_statistics(file1_tif)
        stats2 = SpectralAnalyzer.extract_spectral_statistics(file2_tif)

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


class EmbeddingAnalyzer:
    """Utility for embedding stability analysis."""

    @staticmethod
    def compare_stability(classifier: TerraMindClassifier, clean_tif: str | Path, simulated_tif: str | Path) -> dict:
        """Compare encoder embedding stability between two inputs.
        
        Args:
            classifier: TerraMindClassifier instance.
            clean_tif: Path to clean reference image.
            simulated_tif: Path to simulated image.
            
        Returns:
            Dictionary with cosine similarity and error metrics across layers.
        """
        clean_input = classifier.load_tif_for_model(clean_tif)
        simu_input = classifier.load_tif_for_model(simulated_tif)

        pixel_diff = torch.abs(clean_input - simu_input).max().item()

        feat_clean_layers = classifier.extract_embeddings(clean_input)
        feat_simu_layers = classifier.extract_embeddings(simu_input)

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
