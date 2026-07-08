"""Human-readable report formatting for drift analysis results."""

from __future__ import annotations

from pathlib import Path


class DriftReportPrinter:
    """Print terminal-friendly summaries for comprehensive drift analysis."""

    @staticmethod
    def print_drift_report(drift_analysis: dict, verbose: bool = False) -> None:
        """Pretty-print spectral, embedding, and class-drift details."""
        print(f"\n{'=' * 80}")
        print("SPECTRAL DRIFT ANALYSIS")
        print(f"{'=' * 80}")
        print(f"File 1: {Path(drift_analysis['spectral_drift']['file1']).name}")
        print(f"File 2: {Path(drift_analysis['spectral_drift']['file2']).name}")
        print()

        summary = drift_analysis["spectral_drift"]["summary"]
        print(f"Average mean difference across bands: {summary['avg_mean_difference']:.4f}")
        print(f"Maximum mean difference (worst band): {summary['max_mean_difference']:.4f}")
        print(f"Average std change across bands: {summary['avg_std_change']:.4f}")
        print(f"Maximum std change (worst band): {summary['max_std_change']:.4f}")

        if verbose:
            print("\nPer-band analysis:")
            for band_name, metrics in drift_analysis["spectral_drift"]["bands"].items():
                print(f"\n  {band_name}:")
                print(
                    "    Mean: "
                    f"{metrics['file1_mean']:.4f} -> {metrics['file2_mean']:.4f} "
                    f"(Delta {metrics['mean_difference']:+.4f}, {metrics['mean_diff_percent']:+.2f}%)"
                )
                print(
                    "    Std:  "
                    f"{metrics['file1_std']:.4f} -> {metrics['file2_std']:.4f} "
                    f"(Delta {metrics['std_change']:+.4f}, {metrics['std_change_percent']:+.2f}%)"
                )

        print(f"\n{'=' * 80}")
        print("EMBEDDING-BASED DRIFT (TerraMind Encoder)")
        print(f"{'=' * 80}")
        emb = drift_analysis["embedding_drift"]
        print(f"Cosine Similarity (avg across encoder layers): {emb['cosine_similarity']:.4f}")
        print(f"MSE Error: {emb['mse_error']:.6f}")
        print(f"MAE Error: {emb['mae_error']:.6f}")
        print(f"Pixel-level max difference: {emb['pixel_max_diff']:.6f}")

        if verbose:
            print("\nPer-layer cosine similarities:")
            for i, sim in enumerate(emb["layer_cosine_similarities"], 1):
                print(f"  Layer {i:2d}: {sim:.4f}")

        print(f"\n{'=' * 80}")
        print("CLASSIFICATION-BASED DRIFT (Class Flip Assessment)")
        print(f"{'=' * 80}")
        clf = drift_analysis["class_drift"]

        class_flip_indicator = "CLASS FLIP DETECTED" if clf["class_flip"] else "No class flip"
        print(class_flip_indicator)
        print("\nFile 1 Prediction:")
        print(f"  Predicted class: {clf['pred1_class']} (confidence: {clf['pred1_probability']:.4f})")
        print(f"  Top 3 classes: {clf['pred1_top3']}")

        print("\nFile 2 Prediction:")
        print(f"  Predicted class: {clf['pred2_class']} (confidence: {clf['pred2_probability']:.4f})")
        print(f"  Top 3 classes: {clf['pred2_top3']}")

        print("\nPrediction Stability:")
        print(f"  Probability change: {clf['probability_change']:+.4f}")
        print(f"  Top-3 consistency: {clf['top3_consistency']:.2%}")
        print(f"{'=' * 80}\n")
