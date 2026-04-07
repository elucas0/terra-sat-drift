from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Any, Type
import json
from pangaea4sec.pangaea.datasets.base import RawGeoFMDataset
from sen1floods11_loader import Sen1Floods11S2Loader
from simulation_pipeline import SimulationPipeline
from simulation_config import SimulationConfig, SimulationSteps

from pangaea4sec.pangaea.datasets.sen1floods11 import Sen1Floods11

# Dataset registry for easy selection and extensibility
DATASET_REGISTRY = {
    "sen1floods11": Sen1Floods11,
    # "other_dataset": OtherDataset,
}


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

class PanGaeaDatasetLoader(RawGeoFMDataset):
    """Flexible loader for pangaea datasets registered in DATASET_REGISTRY."""
    
    def __init__(self, dataset_name: str, config: dict):
        """Initialize pangaea dataset loader.
        
        Args:
            dataset_name: Name of dataset (must be in DATASET_REGISTRY)
            config: Configuration dictionary with dataset parameters
        """
        if dataset_name not in DATASET_REGISTRY:
            available = ", ".join(DATASET_REGISTRY.keys())
            raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {available}")
        
        self.dataset_name = dataset_name
        self.dataset_class = DATASET_REGISTRY[dataset_name]
        self.config = config
        self._dataset = None
        self._initialize_dataset()
    
    def _initialize_dataset(self):
        """Initialize the pangaea dataset."""
        try:
            self._dataset = self.dataset_class(**self.config)
        except (TypeError, RuntimeError) as e:
            class_name = self.dataset_class.__name__ if hasattr(self.dataset_class, '__name__') else self.dataset_name
            raise RuntimeError(
                f"Failed to initialize {class_name} with provided config. "
                f"Error: {e}"
            ) from e


def simulate_dataset(
    loader: PanGaeaDatasetLoader,
    output_dir: Path | str,
    split: str = "train",
    max_files: Optional[int] = None,
    simulation_steps: Optional[dict] = None,
    processing_level: str = "L1C",
    phisat2_exec_path: Optional[str] = None,
    verbose: bool = False,
) -> dict:
    """Simulate optical files from any dataset through Φ-sat-2 pipeline.

    Args:
        loader: DatasetLoader instance providing optical files and metadata.
        output_dir: Directory to save simulated files.
        split: Dataset split - train, valid, or test.
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

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info(f"Starting Dataset Simulation")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Split: {split}")
    logger.info("=" * 80)

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
        processing_level=processing_level,
    )
    pipeline = SimulationPipeline(config)
    
    dataset = loader._dataset
    
    if dataset is None:
        raise RuntimeError("Dataset not initialized properly in loader.")
    
    optical_files = dataset.s2_image_list
    if max_files:
        optical_files = optical_files[:max_files]

    logger.info(f"Simulation steps: {steps_obj.as_dict()}")
    logger.info(f"Processing level: {processing_level}")

    # Process files
    results = {
        "successful": [],
        "failed": [],
        "metadata": {
            "split": split,
            "processing_level": processing_level,
            "total_files": len(optical_files),
            "simulation_steps": steps_obj.as_dict(),
        },
    }

    for i, optical_file in enumerate(optical_files, 1):
        try:
            optical_path = Path(optical_file)
            # Create output filename
            output_filename = f"simulated_{optical_path.name}"
            output_file = split_output_dir / output_filename
            
            dataset_item = dataset[i-1] 
            metadata = dataset.metadata if hasattr(dataset, 'metadata') else None

            logger.info(f"[{i}/{len(optical_files)}] Processing: {optical_path.name}")

            # Run simulation
            success = pipeline.simulate_single_file(optical_path, output_file, metadata)

            if success:
                results["successful"].append(str(output_file))
                logger.info(f"✓ Success: {optical_path.name} -> {output_filename}")
            else:
                results["failed"].append(str(optical_file))
                logger.warning(f"✗ Failed: {optical_path.name}")

        except Exception as e:
            logger.error(f"Exception processing {optical_file}: {e}")
            results["failed"].append(str(optical_file))
            continue

    # Log summary
    logger.info("=" * 80)
    logger.info("Simulation Complete")
    logger.info(f"Successful: {len(results['successful'])} / {len(optical_files)}")
    logger.info(f"Failed: {len(results['failed'])} / {len(optical_files)}")
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
        description="Batch simulate satellite imagery through Φ-sat-2 pipeline using any pangaea dataset"
    )
    
    # Dataset selection
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_REGISTRY.keys()),
        default="sen1floods11",
        help=f"Dataset to use for simulation (default: sen1floods11, available: {', '.join(DATASET_REGISTRY.keys())})",
    )
    parser.add_argument(
        "--use-pangaea",
        action="store_true",
        default=True,
        help="Use pangaea dataset loader instead of custom loader (requires pangaea-bench)",
    )
    
    # Dataset-specific arguments
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default="datasets/sen1floods11",
        help="Path to dataset root (default: datasets/sen1floods11)",
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
        help="Dataset split to process (default: train)",
    )
    parser.add_argument(
        "--dataset-type",
        choices=["HandLabeled", "WeaklyLabeled"],
        default="HandLabeled",
        help="Type of dataset to use (for Sen1Floods11, default: HandLabeled)",
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
        help="Processing level for output (default: L1C)",
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
        default="executables",
        help="Path to phisat2 binary if available",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    simulation_steps = {
        "radiance": not args.disable_radiance,
        "add_panchromatic": not args.disable_pan,
        "band_misalignment": not args.disable_misalignment,
        "snr_simulation": not args.disable_snr,
        "psf_filtering": not args.disable_psf,
        "reflectance_conversion": not args.disable_reflectance and args.processing_level == "L1C",
    }

    try:
        if args.use_pangaea:
            pangaea_config = {
                "split": args.split,
                "dataset_name": args.dataset.lower(),
                "multi_modal": True,
                "multi_temporal": 1,
                "root_path": str(args.dataset_root),
                "classes": ["non_flood", "flood"],
                "num_classes": 2,
                "ignore_index": 0,
                "img_size": 512,
                "bands": {"optical": ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]},
                "distribution": [50, 50],
                "data_mean": {"optical": [0.0] * 11},
                "data_std": {"optical": [1.0] * 11},
                "data_min": {"optical": [0.0] * 11},
                "data_max": {"optical": [10000.0] * 11},
                "download_url": "",
                "auto_download": False,
                "gcs_bucket": "",
            }
            
            loader = PanGaeaDatasetLoader(
                dataset_name=args.dataset,
                config=pangaea_config
            )
            print(f"✓ Loaded {args.dataset} using PanGaea loader")
      
        # Run simulation
        results = simulate_dataset(
            loader=loader,
            output_dir=args.output_dir,
            split=args.split,
            max_files=args.max_files,
            simulation_steps=simulation_steps,
            processing_level=args.processing_level,
            phisat2_exec_path=args.phisat2_path,
            verbose=args.verbose,
        )

        return 0 if len(results["failed"]) == 0 else 1
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
