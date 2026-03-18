"""End-to-end execution pipeline for drift experiments."""

from __future__ import annotations

from pathlib import Path

from .drift_analysis import DriftAnalyzer
from .model_service import TerraMindClassifier
from .reporting import DriftReportPrinter


class DriftPipeline:
    """Coordinate model initialization and drift-analysis example runs."""

    def __init__(self, num_classes: int = 10, backbone_size: str = "large") -> None:
        """Create the classifier and analyzer used across examples.

        Args:
            num_classes: Number of output classes for the classification head.
            backbone_size: TerraMind model size: 'tiny', 'small', 'base', or 'large'.
        """
        self.classifier = TerraMindClassifier(num_classes=num_classes, backbone_size=backbone_size)
        self.analyzer = DriftAnalyzer(self.classifier)
        self.report_printer = DriftReportPrinter()

    def validate(self) -> bool:
        """Validate model setup and print warning if validation fails."""
        print("Validating ClassificationTask setup...")
        valid = self.classifier.validate_setup()
        if not valid:
            print("Warning: Model validation failed. Predictions may not work correctly.")
        return valid

    def run_examples(self) -> None:
        """Execute the original three demonstration scenarios."""
        print("\n" + "=" * 80)
        print("EXAMPLE 1: Single file pair drift analysis with classification")
        print("=" * 80)

        raw_file = Path(
            "tiff_folder/raw_tiff_update/"
            "642220-5068670_32631_BANDS_RES-GRID_0_2025-07-15T10-48-25_000.tiff"
        )
        sim_file = Path(
            "tiff_folder/simulated_tiff/"
            "642220-5068670_32631_PHISAT2-BANDS-GRID_0_2025-07-15T10-48-25_000.tiff"
        )

        if raw_file.exists() and sim_file.exists():
            drift = self.analyzer.analyze_drift_comprehensive(raw_file, sim_file)
            self.report_printer.print_drift_report(drift, verbose=True)

        print("\n" + "=" * 80)
        print("EXAMPLE 2: Batch comparison with class flip detection")
        print("=" * 80)

        results = self.analyzer.compare_directory(
            raw_dir=Path("tiff_folder/simulated_custom_tiff"),
            simulated_dir=Path("tiff_folder/simulated_custom_l2_tiff"),
        )
        print(f"\nProcessed {len(results)} image pairs.")

        if results:
            avg_cos_sim = sum(r["cosine_similarity"] for r in results) / len(results)
            avg_mse = sum(r["mse_error"] for r in results) / len(results)
            avg_mae = sum(r["mae_error"] for r in results) / len(results)
            avg_pixel_diff = sum(r["pixel_max_diff"] for r in results) / len(results)
            print(f"\n{'=' * 60}")
            print(f"Average pixel-level max difference: {avg_pixel_diff:.6f}")
            print(f"Average cosine similarity (across all 12 layers): {avg_cos_sim:.4f}")
            print(f"Average MSE error: {avg_mse:.6f}")
            print(f"Average MAE error: {avg_mae:.6f}")
            print(f"{'=' * 60}")

            if results[0]["layer_cosine_similarities"]:
                num_layers = len(results[0]["layer_cosine_similarities"])
                print(f"\nPer-layer cosine similarity statistics ({num_layers} layers):")
                for layer_idx in range(num_layers):
                    layer_cos_sims = [
                        r["layer_cosine_similarities"][layer_idx]
                        for r in results
                        if layer_idx < len(r["layer_cosine_similarities"])
                    ]
                    if layer_cos_sims:
                        avg_layer_cos_sim = sum(layer_cos_sims) / len(layer_cos_sims)
                        print(f"  Layer {layer_idx + 1:2d}: {avg_layer_cos_sim:.4f}")

        print("\n" + "=" * 80)
        print("EXAMPLE 3: Classification-based drift analysis (class flips)")
        print("=" * 80)

        comp_results = self.analyzer.compare_directory_comprehensive(
            dir1=Path("tiff_folder/raw_tiff_update"),
            dir2=Path("tiff_folder/simulated_custom_l2_tiff"),
            file1_pattern="BANDS_RES-GRID",
            file2_pattern="PHISAT2-BANDS-GRID",
            suffix=".tiff",
        )

        if comp_results:
            class_flip_analysis = self.analyzer.analyze_class_flips(comp_results)
            print("\nClass Flip Analysis:")
            print(f"  Total pairs analyzed: {class_flip_analysis['total_pairs']}")
            print(f"  Class flips detected: {class_flip_analysis['class_flips_count']}")
            print(f"  Class flip rate: {class_flip_analysis['class_flip_rate']:.2%}")
            print(
                "  Average probability change: "
                f"{class_flip_analysis['avg_probability_change']:+.4f}"
            )
            print(f"  Max probability change: {class_flip_analysis['max_probability_change']:.4f}")
            print(
                "  Average top-3 consistency: "
                f"{class_flip_analysis['avg_top3_consistency']:.2%}"
            )

            if class_flip_analysis["flipped_pairs"]:
                print("\n  Flipped predictions:")
                for flip in class_flip_analysis["flipped_pairs"]:
                    print(f"    {Path(flip['file1']).name}")
                    print(
                        f"      Class {flip['class_from']} -> {flip['class_to']} "
                        f"(Delta prob: {flip['prob_change']:+.4f})"
                    )
