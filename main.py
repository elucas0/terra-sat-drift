"""Entry point for TerraSat drift analysis experiments."""

import argparse
import json
from pathlib import Path

from terra_sat_drift.pipeline import DriftPipeline
from terra_sat_drift.simulation_config import SimulationConfig, SimulationSteps


def main() -> None:
    """Run model validation and drift-analysis examples with optional on-the-fly simulation."""
    parser = argparse.ArgumentParser(
        description="TerraSat drift analysis using configurable TerraMind backbones with optional Φ-sat-2 simulation.",
    )

    # Model configuration
    parser.add_argument(
        "--backbone-size",
        choices=["tiny", "small", "base", "large"],
        default="large",
        help="TerraMind backbone size (default: large). Smaller models are faster but less accurate.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=10,
        help="Number of classification classes (default: 10).",
    )

    # Simulation mode
    parser.add_argument(
        "--mode",
        choices=["default", "simulation"],
        default="default",
        help="Execution mode: 'default' runs bundled examples, 'simulation' runs on-the-fly Φ-sat-2 synthesis.",
    )

    # Default mode: directory-based comparisons
    parser.add_argument(
        "--embedding-raw-dir",
        default="tiff_folder/raw_tiff_update",
        help="[Default mode] Source folder for embedding-based comparisons.",
    )
    parser.add_argument(
        "--embedding-simulated-dir",
        default="tiff_folder/simulated_custom_l2_tiff",
        help="[Default mode] Target folder for embedding-based comparisons.",
    )
    parser.add_argument(
        "--classification-raw-dir",
        default="tiff_folder/raw_tiff_update",
        help="[Default mode] Source folder for classification-based comparisons.",
    )
    parser.add_argument(
        "--classification-simulated-dir",
        default="tiff_folder/simulated_custom_l2_tiff",
        help="[Default mode] Target folder for classification-based comparisons.",
    )
    parser.add_argument(
        "--file1-pattern",
        default="BANDS_RES-GRID",
        help="[Default mode] Substring in source filenames used for counterpart matching.",
    )
    parser.add_argument(
        "--file2-pattern",
        default="PHISAT2-BANDS-GRID",
        help="[Default mode] Replacement substring in target filenames used for counterpart matching.",
    )
    parser.add_argument(
        "--suffix",
        default=".tiff",
        help="[Default mode] File suffix to scan when matching files.",
    )

    # Simulation mode: on-the-fly synthesis
    parser.add_argument(
        "--simulation-config",
        type=str,
        help="[Simulation mode] Path to JSON file with SimulationConfig (steps, parameters).",
    )
    parser.add_argument(
        "--raw-s2-cache",
        default="tiff_folder/raw_s2_cache",
        help="[Simulation mode] Directory containing cached raw S2 L1C .tiff files.",
    )
    parser.add_argument(
        "--simulated-output-dir",
        default="tiff_folder/simulated_dynamic",
        help="[Simulation mode] Directory to save simulated Φ-sat-2 outputs.",
    )

    # Simulation step toggles (for quick overrides without JSON config)
    parser.add_argument(
        "--enable-radiance",
        action="store_true",
        default=True,
        help="[Simulation mode] Include radiance conversion step.",
    )
    parser.add_argument(
        "--enable-panchromatic",
        action="store_true",
        default=True,
        help="[Simulation mode] Include panchromatic band synthesis.",
    )
    parser.add_argument(
        "--enable-misalignment",
        action="store_true",
        default=True,
        help="[Simulation mode] Include band misalignment simulation.",
    )
    parser.add_argument(
        "--enable-snr",
        action="store_true",
        default=True,
        help="[Simulation mode] Include SNR noise simulation.",
    )
    parser.add_argument(
        "--enable-psf",
        action="store_true",
        default=True,
        help="[Simulation mode] Include PSF filtering simulation.",
    )
    parser.add_argument(
        "--enable-reflectance",
        action="store_true",
        default=True,
        help="[Simulation mode] Include reflectance conversion (for L1C output).",
    )

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = DriftPipeline(num_classes=args.num_classes, backbone_size=args.backbone_size)
    pipeline.validate()

    if args.mode == "default":
        # Original bundled examples workflow
        print(
            f"Starting DriftPipeline (default mode) with backbone_size={args.backbone_size}, "
            f"num_classes={args.num_classes}"
            f"\nEmbedding comparison: {args.embedding_raw_dir} -> {args.embedding_simulated_dir}"
            f"\nClassification comparison: {args.classification_raw_dir} -> {args.classification_simulated_dir}"
        )
        pipeline.run_examples(
            embedding_raw_dir=args.embedding_raw_dir,
            embedding_simulated_dir=args.embedding_simulated_dir,
            classification_raw_dir=args.classification_raw_dir,
            classification_simulated_dir=args.classification_simulated_dir,
            file1_pattern=args.file1_pattern,
            file2_pattern=args.file2_pattern,
            suffix=args.suffix,
        )

    elif args.mode == "simulation":
        # On-the-fly simulation + drift analysis workflow
        print(
            f"Starting DriftPipeline (simulation mode) with backbone_size={args.backbone_size}, "
            f"num_classes={args.num_classes}"
        )

        # Load or build simulation config
        if args.simulation_config:
            print(f"Loading simulation config from: {args.simulation_config}")
            sim_config = SimulationConfig.from_json(args.simulation_config)
        else:
            # Build config from CLI args
            sim_config = SimulationConfig(
                steps=SimulationSteps(
                    radiance=args.enable_radiance,
                    add_panchromatic=args.enable_panchromatic,
                    band_misalignment=args.enable_misalignment,
                    snr_simulation=args.enable_snr,
                    psf_filtering=args.enable_psf,
                    reflectance_conversion=args.enable_reflectance,
                ),
                s2_source_dir=args.raw_s2_cache,
                output_dir=args.simulated_output_dir,
            )

        print(f"Simulation steps: {sim_config.steps.as_dict()}")
        print(f"S2 cache directory: {args.raw_s2_cache}")
        print(f"Simulated output directory: {args.simulated_output_dir}")

        pipeline.run_simulation_experiments(
            simulation_config=sim_config,
            raw_s2_source_dir=args.raw_s2_cache,
            simulated_output_dir=args.simulated_output_dir,
        )


if __name__ == "__main__":
    main()