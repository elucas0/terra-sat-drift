"""Batch simulation script for HLS burn scar data using the simulation pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

from phisat2_constants import ProcessingLevels

from simulation_pipeline import simulate_with_executor
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


def simulate_hls_burn_scars(
    dataset_root: Path | str,
    output_dir: Path | str,
    split: str = "train",
    max_files: Optional[int] = None,
    simulation_steps: Optional[dict] = None,
    processing_level: ProcessingLevels = ProcessingLevels.L1C,
    verbose: bool = False,
):
    """Simulate HLS burn scar data through Φ-sat-2 pipeline using the dataset loader.

    Args:
        dataset_root: Path to HLS burn scars dataset root.
        output_dir: Directory to save simulated files.
        split: Dataset split - train, valid, or test.
        max_files: Maximum number of files to process. If None, processes all.
        simulation_steps: Dict with step names and boolean flags. If None, uses defaults.
        processing_level: L1A, L1B, or L1C.
        verbose: Enable verbose logging.

    """
    # Setup
    setup_logging(Path(output_dir), verbose)
    logger = logging.getLogger(__name__)

    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info(f"Starting HLS Burn Scars Simulation")
    logger.info(f"Dataset root: {dataset_root}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Split: {split}")
    logger.info(f"Max files: {max_files}")
    logger.info("=" * 80)

    # Load dataset
    try:
        # Try to load HLS dataset - adjust based on available dataset class
        # This is a flexible loader that can work with different HLS dataset implementations
        try:
            from terratorch.datasets import HLSBurnScarsNonGeo
            dataset = HLSBurnScarsNonGeo(
                data_root=str(dataset_root),
                split=split,
                use_metadata=True,  # Enable metadata loading for location and temporal info
            )
        except ImportError:
            # Fallback: try generic HLS dataset
            from torchgeo.datasets import HLS
            dataset = HLS(
                root=str(dataset_root),
                split=split,
                crs=None,
                res=None,
            )
        
        num_samples = len(dataset)
        logger.info(f"Loaded {num_samples} samples from {split} split")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        logger.error("Make sure HLS burn scars dataset is available at the specified root")
        raise

    # Limit to max_files if specified
    num_samples_to_process = num_samples
    if max_files is not None:
        num_samples_to_process = min(max_files, num_samples)
        logger.info(f"Limited to {num_samples_to_process} samples for processing")

    # Configure simulation steps
    if simulation_steps is None:
        simulation_steps = {
            "radiance": True,
            "add_panchromatic": True,
            "band_misalignment": True,
            "snr_simulation": True,
            "psf_filtering": True,
            "reflectance_conversion": True
        }

    steps_obj = SimulationSteps(**simulation_steps)

    # Create simulation config and pipeline
    config = SimulationConfig(
        steps=steps_obj,
        processing_level=processing_level,
        phisat2_exec_path="/shared/home/elucas/terra-sat-drift/executables/phisat2_unix.bin",
        snr_psf_method="executable",  # "alternative" or "executable"
    )

    logger.info(f"Simulation steps: {steps_obj.as_dict()}")
    logger.info(f"Processing level: {processing_level.value}")

    
    simulate_with_executor(
        config=config, 
        dataset=dataset,
        num_samples=num_samples_to_process,
        output_dir=str(output_dir),
        logs_folder=str(Path(output_dir) / "logs"),
        workers=4,
        save_logs=True,
        verbose=verbose,
        logger=logger,
    )

    # Log summary
    logger.info("=" * 80)
    logger.info("Simulation Complete")
    logger.info(f"Logs saved to: {Path(output_dir) / 'logs'}")
    logger.info("=" * 80)

def main():
    """Command-line interface for batch simulation."""
    parser = argparse.ArgumentParser(
        description="Batch simulate HLS burn scar data through Φ-sat-2 pipeline"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default="/shared/home/elucas/datasets/hls_burn_scars",
        help="Path to HLS burn scars dataset (default: /shared/home/elucas/datasets/hls_burn_scars)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="/shared/home/elucas/datasets/hls_burn_scars_simulated",
        help="Output directory for simulated files",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "valid", "test"],
        default="train",
        help="Dataset split to process",
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
        "reflectance_conversion": not args.disable_reflectance,
    }

    # Run simulation
    simulate_hls_burn_scars(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        split=args.split,
        max_files=args.max_files,
        simulation_steps=simulation_steps,
        processing_level=ProcessingLevels[args.processing_level],
        verbose=args.verbose,
    )

if __name__ == "__main__":
    exit(main())
