"""End-to-end execution pipeline for drift experiments."""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

import numpy as np

from .drift_analysis import DriftAnalyzer
from ..model_tasks import TerraMindClassifier, TerraMindSegmenter
from ..reporting import DriftReportPrinter
from ..sen1floods_drift_loader import Sen1FloodsDriftLoader



class DriftPipeline:
    """Coordinate model initialization and drift-analysis example runs."""

    def __init__(
        self, 
        num_classes: int = 10, 
        backbone_size: str = "large",
        enable_segmentation: bool = True,
    ) -> None:
        """Create the classifier, segmenter, and analyzer used across examples.

        Args:
            num_classes: Number of output classes for the classification head.
            backbone_size: TerraMind model size: 'tiny', 'small', 'base', or 'large'.
            enable_segmentation: Whether to initialize segmentation model (default: True).
                                Set to False to save memory if only doing classification.
        """
        self.classifier = TerraMindClassifier(
            num_classes=num_classes, 
            backbone_size=backbone_size
        )
        
        self.segmenter = None
        if enable_segmentation:
            # For segmentation, typically use binary (2 classes: background, foreground)
            self.segmenter = TerraMindSegmenter(
                num_classes=2,
                backbone_size=backbone_size
            )
        
        self.analyzer = DriftAnalyzer(self.classifier, segmenter=self.segmenter)
        self.report_printer = DriftReportPrinter()

    def run_examples(
        self,
        embedding_raw_dir: str | Path = "tiff_folder/simulated_custom_tiff",
        embedding_simulated_dir: str | Path = "tiff_folder/simulated_custom_l2_tiff",
        classification_raw_dir: str | Path = "tiff_folder/raw_tiff_update",
        classification_simulated_dir: str | Path = "tiff_folder/simulated_custom_l2_tiff",
        file1_pattern: str = "BANDS_RES-GRID",
        file2_pattern: str = "PHISAT2-BANDS-GRID",
        suffix: str = ".tiff",
    ) -> None:
        """Execute the original three demonstration scenarios.

        Args:
            embedding_raw_dir: Source directory for embedding-based pair comparisons.
            embedding_simulated_dir: Target directory for embedding-based pair comparisons.
            classification_raw_dir: Source directory for classification-based pair comparisons.
            classification_simulated_dir: Target directory for classification-based pair comparisons.
            file1_pattern: Substring to replace in source file names.
            file2_pattern: Substring used in target file names.
            suffix: File extension to include when scanning directories.
        """
        experiment_results: dict = {
            "metadata": {
                "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "embedding_raw_dir": str(embedding_raw_dir),
                "embedding_simulated_dir": str(embedding_simulated_dir),
                "classification_raw_dir": str(classification_raw_dir),
                "classification_simulated_dir": str(classification_simulated_dir),
                "file1_pattern": file1_pattern,
                "file2_pattern": file2_pattern,
                "suffix": suffix,
            },
            "example_1": {},
            "example_2": {},
            "example_3": {},
        }

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
            experiment_results["example_1"] = {
                "status": "ok",
                "raw_file": str(raw_file),
                "simulated_file": str(sim_file),
                "drift": drift,
            }
        else:
            experiment_results["example_1"] = {
                "status": "skipped",
                "reason": "example files not found",
                "raw_file": str(raw_file),
                "simulated_file": str(sim_file),
            }

        print("\n" + "=" * 80)
        print("EXAMPLE 2: Batch comparison with class flip detection")
        print("=" * 80)

        results = self.analyzer.compare_directory(
            raw_dir=Path(embedding_raw_dir),
            simulated_dir=Path(embedding_simulated_dir),
            file1_pattern=file1_pattern,
            file2_pattern=file2_pattern,
            suffix=suffix,
        )
        print(f"\nProcessed {len(results)} image pairs.")

        example_2_summary: dict = {
            "status": "ok",
            "pairs_processed": len(results),
            "avg_cosine_similarity": None,
            "avg_mse_error": None,
            "avg_mae_error": None,
            "avg_pixel_max_diff": None,
            "per_layer_avg_cosine_similarity": [],
        }

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

            example_2_summary.update(
                {
                    "avg_cosine_similarity": avg_cos_sim,
                    "avg_mse_error": avg_mse,
                    "avg_mae_error": avg_mae,
                    "avg_pixel_max_diff": avg_pixel_diff,
                }
            )

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
                        example_2_summary["per_layer_avg_cosine_similarity"].append(avg_layer_cos_sim)

        experiment_results["example_2"] = example_2_summary

        print("\n" + "=" * 80)
        print("EXAMPLE 3: Classification-based drift analysis (class flips)")
        print("=" * 80)

        comp_results = self.analyzer.compare_directory_comprehensive(
            dir1=Path(classification_raw_dir),
            dir2=Path(classification_simulated_dir),
            file1_pattern=file1_pattern,
            file2_pattern=file2_pattern,
            suffix=suffix,
        )

        if comp_results:
            class_flip_analysis = self.analyzer.analyze_class_flips(comp_results)

            class_flip_summary = {
                "total_pairs": class_flip_analysis["total_pairs"],
                "class_flips_count": class_flip_analysis["class_flips_count"],
                "class_flip_rate": class_flip_analysis["class_flip_rate"],
                "avg_probability_change": class_flip_analysis["avg_probability_change"],
                "max_probability_change": class_flip_analysis["max_probability_change"],
                "avg_top3_consistency": class_flip_analysis["avg_top3_consistency"],
            }

            experiment_results["example_3"] = {
                "status": "ok",
                "pairs_processed": len(comp_results),
                "class_flip_analysis": class_flip_summary,
            }
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
        else:
            experiment_results["example_3"] = {
                "status": "ok",
                "pairs_processed": 0,
                "class_flip_analysis": None,
            }

        output_dir = Path("experiments")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"drift_results_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        with output_file.open("w", encoding="utf-8") as handle:
            json.dump(experiment_results, handle, indent=2)

        print(f"\nSaved experiment results to: {output_file}")

    def run_simulation_experiments(
        self,
        simulation_config,
        raw_s2_source_dir: str | Path = "tiff_folder/raw_s2_cache",
        simulated_output_dir: str | Path = "tiff_folder/simulated_s2",
        comparison_pairs: list | None = None,
        metadata_path: str | None = None,
    ) -> None:
        """Execute drift analysis with on-the-fly Φ-sat-2 simulation from cached S2 data.

        This workflow:
        1. Loads raw S2 .tiff files from cache
        2. Applies configurable simulation steps (radiance, PAN, misalignment, SNR, PSF)
        3. Runs drift analysis comparing raw vs. simulated
        4. Exports results to JSON with experiment metadata

        Args:
            simulation_config: SimulationConfig instance defining processing steps.
            raw_s2_source_dir: Directory containing cached raw S2 L1C .tiff files.
            simulated_output_dir: Directory to save simulated Φ-sat-2 outputs.
            comparison_pairs: Optional list of (raw_file, simulated_file) tuples to analyze.
                If None, performs batch analysis on all pairs.
            metadata_path: Optional path to Sen1Floods metadata GeoJSON for acquisition date extraction.
        """
        from .simulation_pipeline import SimulationPipeline

        sim_pipeline = SimulationPipeline(simulation_config)

        print("\n" + "=" * 80)
        print("STEP 1: Apply on-the-fly Φ-sat-2 simulation pipeline")
        print("=" * 80)

        sim_results = sim_pipeline.batch_simulate_from_source_dir(
            source_dir=raw_s2_source_dir, pattern="*.tiff", metadata_path=metadata_path
        )

        experiment_results: dict = {
            "metadata": {
                "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "simulation_steps": simulation_config.steps.as_dict(),
                "raw_s2_source_dir": str(raw_s2_source_dir),
                "simulated_output_dir": str(simulated_output_dir),
                "simulation_successful": len(sim_results["successful"]),
                "simulation_failed": len(sim_results["failed"]),
            },
            "simulation_results": sim_results,
            "drift_analysis": {},
        }

        if not sim_results["successful"]:
            print("⚠ No successful simulations. Skipping drift analysis.")
            self._save_experiment_results(experiment_results, "simulation_only")
            return

        print("\n" + "=" * 80)
        print("STEP 2: Analyze drift between raw and simulated pairs")
        print("=" * 80)

        # Build comparison pairs from simulation output if not provided
        if comparison_pairs is None:
            comparison_pairs = []
            simulated_dir = Path(simulated_output_dir)
            raw_dir = Path(raw_s2_source_dir)

            for sim_file in sorted(simulated_dir.glob("simulated_*.tiff")):
                # Extract original filename from "simulated_<original>" pattern
                original_name = sim_file.name.replace("simulated_", "", 1)
                raw_file = raw_dir / original_name
                if raw_file.exists():
                    comparison_pairs.append((raw_file, sim_file))

        print(f"Analyzing {len(comparison_pairs)} raw-vs-simulated pairs...")

        drift_results: list = []
        for idx, (raw_file, sim_file) in enumerate(comparison_pairs, 1):
            try:
                drift = self.analyzer.analyze_drift_comprehensive(raw_file, sim_file)
                self.report_printer.print_drift_report(drift, verbose=False)

                drift_results.append(
                    {
                        "pair_index": idx,
                        "raw_file": str(raw_file),
                        "simulated_file": str(sim_file),
                        "drift": drift,
                    }
                )
                print(f"  ✓ Pair {idx}/{len(comparison_pairs)}: {raw_file.name}")
            except Exception as exc:
                print(f"  ✗ Pair {idx}/{len(comparison_pairs)}: {raw_file.name} - {exc}")

        # Aggregate statistics
        print("\n" + "=" * 80)
        print("STEP 3: Aggregate drift statistics")
        print("=" * 80)

        if drift_results:
            # Extract embedding-level metrics
            embedding_sims = [
                r["drift"].get("embedding_comparison", {}).get("cosine_similarity", 0)
                for r in drift_results
                if "embedding_comparison" in r["drift"]
            ]
            avg_embedding_sim = (
                sum(embedding_sims) / len(embedding_sims) if embedding_sims else None
            )

            # Extract classification metrics
            class_flip_rate = None
            avg_prob_change = None
            if drift_results[0]["drift"].get("classification_comparison"):
                class_results = [
                    r["drift"].get("classification_comparison", {}) for r in drift_results
                ]
                flips = sum(1 for cr in class_results if cr.get("class_flipped"))
                class_flip_rate = flips / len(class_results) if class_results else 0
                prob_changes = [
                    cr.get("probability_change_magnitude", 0) for cr in class_results
                ]
                avg_prob_change = (
                    sum(prob_changes) / len(prob_changes) if prob_changes else 0
                )

            summary = {
                "total_pairs_analyzed": len(drift_results),
                "avg_embedding_cosine_similarity": avg_embedding_sim,
                "class_flip_rate": class_flip_rate,
                "avg_probability_change": avg_prob_change,
            }

            print(f"Total pairs analyzed: {summary['total_pairs_analyzed']}")
            if avg_embedding_sim is not None:
                print(f"Average embedding cosine similarity: {avg_embedding_sim:.4f}")
            if class_flip_rate is not None:
                print(f"Class flip rate: {class_flip_rate:.2%}")
            if avg_prob_change is not None:
                print(f"Average probability change: {avg_prob_change:+.4f}")

            experiment_results["drift_analysis"] = {
                "summary": summary,
                "detailed_results": drift_results,
            }

        self._save_experiment_results(experiment_results, "simulation_with_analysis")

    def run_sen1floods_drift_analysis(
        self,
        sen1floods_root: str | Path,
        simulated_dir: str | Path,
        split: str = "train",
        num_samples: int | None = None,
        sample_indices: list | None = None,
    ) -> None:
        """Execute drift analysis on Sen1Floods11 dataset with binary segmentation.
        
        Loads paired raw S2 and simulated Φ-sat-2 images from Sen1Floods11 dataset
        along with ground truth water masks, then analyzes spectral, embedding,
        classification, and segmentation drift.
        
        Args:
            sen1floods_root: Root path to Sen1Floods11 dataset.
            simulated_dir: Directory with simulated Φ-sat-2 files.
            split: Dataset split ('train', 'valid', or 'test').
            num_samples: Limit analysis to first N pairs (None = all).
            sample_indices: Specific indices to analyze (overrides num_samples).
        """
        print("SEN1FLOODS DRIFT ANALYSIS WITH BINARY SEGMENTATION")
        
        # Initialize loader
        try:
            loader = Sen1FloodsDriftLoader(
                sen1floods_root=sen1floods_root,
                simulated_dir=simulated_dir,
                split=split,
            )
        except FileNotFoundError as e:
            print(f"Error initializing loader: {e}")
            return
        
        # Get split info
        split_info = loader.get_split_info()
        print(f"\nLoaded {split_info['total_pairs']} pairs from {split} split")
        print(f"  - Pairs with simulated data: {split_info['pairs_with_simulated']}")
        print(f"  - Pairs without simulated: {split_info['pairs_without_simulated']}")
        
        # Determine which pairs to analyze
        if sample_indices is not None:
            indices = sample_indices
        elif num_samples is not None:
            indices = list(range(min(num_samples, len(loader))))
        else:
            indices = list(range(len(loader)))
        
        print(f"\nAnalyzing {len(indices)} pairs...")
        
        # Results structure
        experiment_results = {
            "metadata": {
                "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "dataset": "sen1floods11",
                "split": split,
                "total_pairs_loaded": len(loader),
                "pairs_analyzed": len(indices),
                "sen1floods_root": str(sen1floods_root),
                "simulated_dir": str(simulated_dir),
            },
            "spectral_analysis": {},
            "embedding_analysis": {},
            "classification_analysis": {},
            "segmentation_analysis": {},
            "detailed_results": [],
        }
        
        # Analyze each pair
        drift_results = []
        segmentation_results = []
        spectral_results = []
        
        for idx in indices:
            try:
                # Load pair
                pair_data = loader.load_pair(idx)
                pair_info = loader.get_pair(idx)
                
                simulated_s2 = pair_data["simulated_s2"]
                mask = pair_data["mask"]
                
                # Skip if no simulated data
                if simulated_s2 is None:
                    location = loader.get_location_from_filename(idx)
                    print(f"No simulated data for pair {idx} (location: {location}), skipping...")
                    continue
                
                # Run comprehensive drift analysis with segmentation
                comprehensive_drift = self.analyzer.analyze_drift_with_segmentation(
                    raw_tif=pair_info["raw_s2"],
                    simulated_tif=pair_info["simulated_s2"],
                    ground_truth_mask=mask,
                )
                
                drift_results.append(comprehensive_drift)
                segmentation_results.append(comprehensive_drift["segmentation_drift"])
                spectral_results.append(comprehensive_drift["spectral_drift"])
                
            except Exception as e:
                print(f"  ✗ Error analyzing pair {idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Aggregate results
        print("\n" + "=" * 80)
        print("AGGREGATED ANALYSIS")
        print("=" * 80)
        
        if drift_results:
            # Segmentation analysis
            seg_summary = self.analyzer.analyze_segmentation_drift(segmentation_results)
            print(f"\nSegmentation Drift Summary:")
            print(f"  Total pairs: {seg_summary['total_pairs']}")
            print(f"  Raw segmentation avg IoU: {seg_summary['raw_segmentation']['avg_iou']:.4f}")
            print(f"  Simulated avg IoU: {seg_summary['simulated_segmentation']['avg_iou']:.4f}")
            print(f"  Avg IoU drift: {seg_summary['drift']['avg_iou_drift']:+.4f}")
            print(f"  Prediction agreement (avg IoU): {seg_summary['prediction_agreement']['avg_iou']:.4f}")
            
            experiment_results["segmentation_analysis"] = seg_summary
            
            # Spectral analysis
            if spectral_results:
                avg_mean_diffs = [
                    r["summary"]["avg_mean_difference"] for r in spectral_results
                ]
                avg_std_changes = [
                    r["summary"]["avg_std_change"] for r in spectral_results
                ]
                
                print(f"\nSpectral Drift Summary:")
                print(f"  Avg mean difference: {np.mean(avg_mean_diffs):.4f}")
                print(f"  Avg std change: {np.mean(avg_std_changes):.4f}")
                
                experiment_results["spectral_analysis"] = {
                    "avg_mean_difference": float(np.mean(avg_mean_diffs)),
                    "max_mean_difference": float(np.max(avg_mean_diffs)),
                    "avg_std_change": float(np.mean(avg_std_changes)),
                    "max_std_change": float(np.max(avg_std_changes)),
                }
            
            # Store detailed results
            experiment_results["detailed_results"] = drift_results
        else:
            print("No valid pairs analyzed")
        
        # Save results
        self._save_experiment_results(experiment_results, "sen1floods_drift_analysis")

    def _save_experiment_results(self, results: dict, experiment_type: str) -> None:
        """Save experiment results to timestamped JSON file.

        Args:
            results: Dictionary of results to save.
            experiment_type: Experiment label for filename.
        """
        output_dir = Path("experiments")
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"drift_results_{experiment_type}_{timestamp}.json"

        with output_file.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)

        print(f"\nSaved experiment results to: {output_file}")
