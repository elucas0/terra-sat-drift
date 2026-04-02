"""Batch simulation script for Sen1Floods11 S2 files using the simulation pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional
import json

from sen1floods11_loader import Sen1Floods11S2Loader
from simulation_pipeline import SimulationPipeline
from simulation_config import SimulationConfig, SimulationSteps


def setup_logging(output_dir: Path, verbose: bool = False) -> None:
    """Setup logging configuration."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "simulation.log"

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    logging.info(f"Logging to {log_file}")


def simulate_sen1floods_s2(
    dataset_root: Path | str,
    output_dir: Path | str,
    split: str = "train",
    dataset_type: str = "HandLabeled",
    max_files: Optional[int] = None,
    simulation_steps: Optional[dict] = None,
    processing_level: str = "L1C",
    phisat2_exec_path: Optional[str] = None,
    sh_config_path: Optional[str] = None,
    verbose: bool = False,
) -> dict:
    """Simulate Sen1Floods11 S2 files through Φ-sat-2 pipeline.

    Args:
        dataset_root: Path to sen1floods dataset root (e.g., datasets/sen1floods11).
        output_dir: Directory to save simulated files.
        split: Dataset split - train, valid, or test.
        dataset_type: HandLabeled or WeaklyLabeled.
        max_files: Maximum number of files to process. If None, processes all.
        simulation_steps: Dict with step names and boolean flags. If None, uses defaults.
        processing_level: L1A, L1B, or L1C.
        phisat2_exec_path: Path to phisat2 binary if available.
        sh_config_path: Path to Sentinel Hub config file for AWS metadata fetching.
        verbose: Enable verbose logging.

    Returns:
        Results dictionary with 'successful', 'failed', and 'metadata'.
    """
    # Setup
    setup_logging(Path(output_dir), verbose)
    logger = logging.getLogger(__name__)

    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info(f"Starting Sen1Floods11 S2 Simulation")
    logger.info(f"Dataset root: {dataset_root}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Split: {split}, Type: {dataset_type}")
    logger.info(f"Max files: {max_files}")
    logger.info("=" * 80)

    # Load S2 files
    try:
        loader = Sen1Floods11S2Loader(
            root_path=dataset_root,
            split=split,
            dataset_type=dataset_type,
        )
        s2_files = loader.get_s2_files()
        logger.info(f"Loaded {len(s2_files)} S2 files from {split} split")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

    # Limit to max_files if specified
    if max_files is not None:
        s2_files = s2_files[:max_files]
        logger.info(f"Limited to {len(s2_files)} files for processing")

    # Configure simulation steps
    if simulation_steps is None:
        simulation_steps = {
            "radiance": True,
            "add_panchromatic": True,
            "band_misalignment": True,
            "snr_simulation": True,
            "psf_filtering": True,
            "reflectance_conversion": True if processing_level == "L1C" else False,
        }

    steps_obj = SimulationSteps(**simulation_steps)

    # Create output subdirectory for this split
    split_output_dir = output_dir / f"simulated_{split}"
    split_output_dir.mkdir(parents=True, exist_ok=True)

    # Create simulation config and pipeline
    config = SimulationConfig(
        steps=steps_obj,
        s2_source_dir=str(output_dir),  # Dummy, not used for single files
        output_dir=str(split_output_dir),
        phisat2_exec_path=phisat2_exec_path,
        sh_config_path=sh_config_path,
        processing_level=processing_level,
    )
    pipeline = SimulationPipeline(config)

    logger.info(f"Simulation steps: {steps_obj.as_dict()}")
    logger.info(f"Processing level: {processing_level}")

    # Process files
    results = {
        "successful": [],
        "failed": [],
        "metadata": {
            "split": split,
            "dataset_type": dataset_type,
            "processing_level": processing_level,
            "total_files": len(s2_files),
            "simulation_steps": steps_obj.as_dict(),
        },
    }
    
    geojson = loader.load_geojson_metadata()

    for i, s2_file in enumerate(s2_files, 1):
        try:
            # Create output filename
            output_filename = f"simulated_{s2_file.name}"
            output_file = split_output_dir / output_filename

            logger.info(f"[{i}/{len(s2_files)}] Processing: {s2_file.name}")

            # Run simulation
            success = pipeline.simulate_single_file(s2_file, output_file, geojson)

            if success:
                results["successful"].append(str(output_file))
                logger.info(f"✓ Success: {s2_file.name} -> {output_filename}")
            else:
                results["failed"].append(str(s2_file))
                logger.warning(f"✗ Failed: {s2_file.name}")

        except Exception as e:
            logger.error(f"Exception processing {s2_file.name}: {e}")
            results["failed"].append(str(s2_file))
            continue

    # Log summary
    logger.info("=" * 80)
    logger.info("Simulation Complete")
    logger.info(f"Successful: {len(results['successful'])} / {len(s2_files)}")
    logger.info(f"Failed: {len(results['failed'])} / {len(s2_files)}")
    logger.info("=" * 80)

    # Save results to JSON
    results_file = output_dir / f"simulation_results_{split}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_file}")

    return results


def main():
    """Command-line interface for batch simulation."""
    parser = argparse.ArgumentParser(
        description="Batch simulate Sen1Floods11 S2 files through Φ-sat-2 pipeline"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default="datasets/sen1floods11",
        help="Path to sen1floods dataset (default: datasets/sen1floods11)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="tiff_folder/simulated_sen1floods",
        help="Output directory for simulated files",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "valid", "test"],
        default="train",
        help="Dataset split to process",
    )
    parser.add_argument(
        "--dataset-type",
        choices=["HandLabeled", "WeaklyLabeled"],
        default="HandLabeled",
        help="Type of dataset to use",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of files to process",
    )
    parser.add_argument(
        "--processing-level",
        choices=["L1A", "L1B", "L1C"],
        default="L1C",
        help="Processing level for output",
    )
    parser.add_argument(
        "--disable-radiance",
        action="store_true",
        help="Disable radiance conversion",
    )
    parser.add_argument(
        "--disable-pan",
        action="store_true",
        help="Disable panchromatic band addition",
    )
    parser.add_argument(
        "--disable-misalignment",
        action="store_true",
        help="Disable band misalignment",
    )
    parser.add_argument(
        "--disable-snr",
        action="store_true",
        help="Disable SNR simulation",
    )
    parser.add_argument(
        "--disable-psf",
        action="store_true",
        help="Disable PSF filtering",
    )
    parser.add_argument(
        "--disable-reflectance",
        action="store_true",
        help="Disable reflectance conversion",
    )
    parser.add_argument(
        "--phisat2-path",
        type=str,
        default=None,
        help="Path to phisat2 binary if available",
    )
    parser.add_argument(
        "--shub-config",
        type=str,
        default="sentinelhub/config.json",
        help="Path to Sentinel Hub config file for AWS metadata fetching",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Build simulation steps from arguments
    simulation_steps = {
        "radiance": not args.disable_radiance,
        "add_panchromatic": not args.disable_pan,
        "band_misalignment": not args.disable_misalignment,
        "snr_simulation": not args.disable_snr,
        "psf_filtering": not args.disable_psf,
        "reflectance_conversion": not args.disable_reflectance and args.processing_level == "L1C",
    }

    # Run simulation
    results = simulate_sen1floods_s2(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        split=args.split,
        dataset_type=args.dataset_type,
        max_files=args.max_files,
        simulation_steps=simulation_steps,
        processing_level=args.processing_level,
        phisat2_exec_path=args.phisat2_path,
        sh_config_path=args.shub_config,
        verbose=args.verbose,
    )

    return 0 if len(results["failed"]) == 0 else 1


if __name__ == "__main__":
    exit(main())
