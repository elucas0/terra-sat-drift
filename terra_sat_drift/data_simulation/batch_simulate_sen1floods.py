"""Batch simulation script for Sen1Floods11 S2 files using the simulation pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional
import json
import numpy as np


from phisat2_constants import ProcessingLevels

from simulation_pipeline import simulate_with_executor
from simulation_config import SimulationConfig, SimulationSteps
from terratorch.datasets import Sen1Floods11NonGeo


def setup_logging(output_dir: Path, verbose: bool = False) -> None:
    """Setup logging configuration."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "simulation.log"

    level = logging.DEBUG if verbose else logging.INFO
    
    # Create handlers and set their levels explicitly
    # This prevents root logger level changes (e.g. from eo-learn) 
    # from causing DEBUG messages to be emitted by these handlers.
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[file_handler, stream_handler],
    )
    logging.info(f"Logging to {log_file}")


def simulate_sen1floods_s2(
    dataset_root: Path | str,
    output_dir: Path | str,
    split: str = "train",
    max_files: Optional[int] = None,
    simulation_steps: Optional[dict] = None,
    processing_level: ProcessingLevels = ProcessingLevels.L1C,
    verbose: bool = False,
    config: Optional[SimulationConfig] = None,
    workers: int = 1,
):
    """Simulate Sen1Floods11 S2 files through Φ-sat-2 pipeline using the dataset loader.

    Args:
        dataset_root: Path to sen1floods dataset root (e.g., datasets/sen1floods11).
        output_dir: Directory to save simulated files.
        split: Dataset split - train, valid, or test.
        max_files: Maximum number of files to process. If None, processes all.
        simulation_steps: Dict with step names and boolean flags. If None, uses defaults.
            Ignored when ``config`` is given.
        processing_level: L1A, L1B, or L1C. Ignored when ``config`` is given.
        verbose: Enable verbose logging.
        config: Fully specified simulation configuration. When None (the default), the
            historical hard-coded configuration is rebuilt from ``simulation_steps`` and
            ``processing_level``. Callers sweeping the noise parameters (see
            ``degradation_ladder.py``) pass their own instance instead.
        workers: Number of parallel EOExecutor workers.

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
    logger.info(f"Split: {split}")
    logger.info(f"Max files: {max_files}")
    logger.info("=" * 80)
    
    bands = ["B02", "B03", "B04", "B05", "B06", "B07", "B08"]
    bands_names = ["BLUE", "GREEN", "RED", "RED_EDGE_1", "RED_EDGE_2", "RED_EDGE_3", "NIR_BROAD"]

    # Load dataset
    try:
        dataset = Sen1Floods11NonGeo(
            data_root=str(dataset_root),
            split=split,
            bands=bands_names,
            constant_scale=1.0,  # No scaling, as we will apply radiance conversion in the pipeline
            use_metadata=True,  # Enable metadata loading for location and temporal info
        )
        num_samples = len(dataset)
        logger.info(f"Loaded {num_samples} samples from {split} split")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
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

    if config is None:
        steps_obj = SimulationSteps(**simulation_steps)
        snr_values = [5, 10]

        # Create simulation config and pipeline
        config = SimulationConfig(
            bands_names=bands,
            source_resolution=10.0,
            steps=steps_obj,
            processing_level=processing_level,
            # phisat2_exec_path="/shared/home/elucas/scratch/terra-sat-drift/executables/phisat2_unix.bin",
            snr_psf_method="alternative",  # "alternative" or "executable"
            misalignment_std_sea=6,
            misalignment_std_land=3,
            snr_values=snr_values,
            psf_kernel_sigma=4.0,
            radiance_reference=10000,
        )
    else:
        steps_obj = config.steps
        processing_level = config.processing_level

    logger.info(f"Simulation steps: {steps_obj.as_dict()}")
    logger.info(f"Processing level: {processing_level.value}")

    
    simulate_with_executor(
        config=config, 
        dataset=dataset,
        num_samples=num_samples_to_process,
        output_dir=output_dir,
        logs_folder=output_dir / "v1.1/logs",
        workers=workers,
        save_logs=True,
        verbose=verbose,
        logger=logger,
    )
    
    config.save_json(output_dir / "v1.1/simulation_config.json")

    # Log summary
    logger.info("=" * 80)
    logger.info("Simulation Complete")
    logger.info(f"Logs saved to: {output_dir / 'v1.1/logs'}")
    logger.info("=" * 80)

def main():
    """Command-line interface for batch simulation."""
    parser = argparse.ArgumentParser(
        description="Batch simulate Sen1Floods11 S2 files through Φ-sat-2 pipeline"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default="/shared/home/elucas/datasets/sen1floods11",
        help="Path to sen1floods dataset (default: datasets/sen1floods11)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="/shared/home/elucas/datasets/sen1floods11_simulated_alt_v2",
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
    simulate_sen1floods_s2(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        split=args.split,
        #dataset_type=args.dataset_type,
        max_files=args.max_files,
        simulation_steps=simulation_steps,
        processing_level=ProcessingLevels[args.processing_level],
        verbose=args.verbose,
    )

if __name__ == "__main__":
    exit(main())
