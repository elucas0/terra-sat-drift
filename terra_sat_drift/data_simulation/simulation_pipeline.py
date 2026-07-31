"""Simulation pipeline wrapping phisat2_utils tasks for on-the-fly Φ-sat-2 synthesis."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import json
import cv2
import logging
from datetime import datetime, timedelta
from typing import Optional

from eolearn.core.eonode import linearly_connect_tasks
from eolearn.core.eoworkflow import EOWorkflow
from eolearn.core.eoexecution import EOExecutor
from eolearn.core.constants import FeatureType
from eolearn.core import EOPatch, EOTask
from eolearn.io.raster_io import ExportToTiffTask
from simulation_config import SimulationConfig
import pandas as pd
from eolearn.core.core_tasks import RemoveFeatureTask

from phisat2_utils import (  
    AddPANBandTask,  
    BandMisalignmentTask,  
    CalculateRadianceTask,  
    CalculateReflectanceTask,  
    PhisatCalculationTask,
    AlternativePhisatCalculationTask,
    ResamplingTask,
    LoadS2FileTask,
    LoadS2DatasetSampleTask
    
)
from utils.psf_utils import get_psf_kernels_dict
from phisat2_constants import ProcessingLevels
from torchgeo.datasets import NonGeoDataset

def _load_metadata_from_file(tiff_path: Path | str, metadata_dir: Path) -> dict:
    """Load metadata JSON file associated with a TIFF file.
    
    Extracts the ID (last part) from the TIFF filename and looks for matching metadata.
    For example: S2B_Crop_10m_2091.tif -> 2091_S2B_metadata.json
    
    Args:
        tiff_path: Path to the TIFF file
        metadata_path: Path to the metadata file
        
    Returns:
        Dictionary with metadata (earth_sun_dist, solar_irradiances, sun_zenith_angles),
        or raises FileNotFoundError if no metadata file found.
    """
    tiff_path = Path(tiff_path)
    stem = tiff_path.stem
    
    # Extract the ID (last string) by splitting on underscores
    parts = stem.split('_')
    if not parts:
        print(f"Warning: Could not extract ID from filename {tiff_path.name}")
        raise FileNotFoundError("Could not extract ID from TIFF filename")
    
    id_str = parts[0]
    
    metadata_path_ = metadata_dir / f"{id_str}_S2B_metadata.json"
    
    if metadata_path_.exists():
        try:
            with open(metadata_path_, 'r') as f:
                metadata = json.load(f)
            return metadata
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse JSON metadata {metadata_dir}: {e}")
        except Exception as e:
            print(f"Warning: Error loading metadata file {metadata_dir}: {e}")
    
    print(f"Warning: No metadata file found for {tiff_path.name} (ID: {id_str})")
    raise FileNotFoundError("No metadata file found")

def simulate_with_executor(
    config: SimulationConfig,
    dataset: NonGeoDataset,
    num_samples: int,
    output_dir: str | Path,
    logs_folder: str | Path,
    workers: int = 4,
    save_logs: bool = True,
    verbose: bool = False,
    day_of_year_base: int = 0,
    logger: Optional[logging.Logger] = None,
) -> EOExecutor:
    """Execute parallel simulation using EOExecutor with dataset samples.

    Iterates through the Sen1Floods11NonGeo dataset using __getitem__, processes
    each sample through the simulation pipeline, and exports results using EOExecutor.

    Args:
        config: SimulationConfig instance defining processing steps and parameters.
        dataset: Sen1Floods11NonGeo dataset instance.
        num_samples: Number of samples to process from the dataset.
        output_dir: Directory to save simulated files.
        logs_folder: Directory to save execution logs.
        workers: Number of parallel workers for execution.
        save_logs: Whether to save execution logs.
        verbose: Enable verbose logging.
        day_of_year_base: Whether the dataset's ``temporal_coords`` day-of-year is
            0-based (0, the TerraTorch default, e.g. Sen1Floods11NonGeo which stores
            ``date.dayofyear - 1``) or 1-based (1, e.g. FireScarsNonGeo which stores
            the Julian day straight from the filename).
        logger: Optional logger instance for output.

    Returns:
        EOExecutor instance with execution results.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    output_dir = Path(output_dir)
    logs_folder = Path(logs_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_folder.mkdir(parents=True, exist_ok=True)

    logger.info(f"Processing {num_samples} samples from dataset")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Logs folder: {logs_folder}")
    
    # Load dataset samples
    exec_args = []
    for sample_idx in range(min(num_samples, len(dataset))):
        try:
            # Load sample using __getitem__
            sample = dataset[sample_idx]
            file_name = Path(dataset.image_files[sample_idx]).stem
            
            # Get file path after root without filename
            file_path = Path(dataset.image_files[sample_idx]).parent.relative_to(dataset.data_root)
            
            logger.info(f"Processing sample {sample_idx + 1}/{num_samples}")
            
            # Extract image data from sample
            image = sample.get("image")
            mask = sample.get("mask")  # Not used in simulation but can be included in metadata if needed for export
            # Extract metadata if available
            temporal_coords = sample.get("temporal_coords")
            location_coords = sample.get("location_coords")
            
            if image is None:
                logger.warning(f"Sample {sample_idx}: No image data found, skipping")
                continue
            
            # Determine acquisition date
            acquisition_date = None
            if temporal_coords is not None:
                try:
                    # temporal_coords is [year, day_of_year], where day_of_year uses
                    # the dataset's own convention (see day_of_year_base). Note that
                    # strptime's "%j" is always 1-based, so it cannot be used directly
                    # for the 0-based datasets without shifting every date back a day.
                    year = int(temporal_coords[0, 0].item())
                    day_of_year = int(temporal_coords[0, 1].item())
                    acquisition_date = datetime(year, 1, 1) + timedelta(
                        days=day_of_year - day_of_year_base
                    )
                except (ValueError, IndexError, AttributeError):
                    raise ValueError(f"Invalid temporal coordinates format for sample {sample_idx}")
            else:
                raise ValueError(f"No temporal coordinates found for sample {sample_idx}, cannot determine acquisition date")
                        # Create output filename
            output_filename = f"{file_path}/simulated_{config.processing_level.name}_{file_name}.tif"
            output_path = output_dir / output_filename
            
            exec_args.append({
                "image": image,
                "mask": mask,
                "output_tiff_path": str(output_path),
                "metadata": None,
                "acquisition_date": acquisition_date,
                "location_coords": location_coords,
            })
            
            logger.info(f"Prepared sample {sample_idx} for simulation")
            
        except Exception as e:
            logger.error(f"Error processing sample {sample_idx}: {e}")
            if verbose:
                import traceback
                logger.error(traceback.format_exc())
            continue

    if not exec_args:
        raise ValueError(f"No valid samples processed from dataset")

    logger.info(f"\nPrepared {len(exec_args)} execution arguments")

    # Build individual task nodes to be connected with linearly_connect_tasks
    task_list = list[EOTask]()

    # Task 1: Load S2 data from dataset sample (using __getitem__)
    task_list.append(LoadS2DatasetSampleTask())

    # Name of the feature produced by the most recent enabled step. Each step reads
    # this and updates it, so disabling any step cannot silently misroute the chain.
    current_feature = "S2_BANDS"

    def _discard(feature_name: str) -> None:
        """Drop a consumed intermediate to free memory, never the loaded input."""
        if feature_name != "S2_BANDS":
            task_list.append(RemoveFeatureTask([(FeatureType.DATA, feature_name)]))

    # Task 2: Radiance conversion (if enabled)
    if config.steps.radiance:
        task_list.append(
            CalculateRadianceTask(
                (FeatureType.DATA, current_feature),
                (FeatureType.DATA, "S2_RADIANCE"),
            )
        )
        current_feature = "S2_RADIANCE"

    # Task 3: Add panchromatic band (if enabled)
    if config.steps.add_panchromatic:
        task_list.append(
            AddPANBandTask(
                (FeatureType.DATA, current_feature),
                (FeatureType.DATA, "BANDS-RAD-PAN"),
            )
        )
        current_feature = "BANDS-RAD-PAN"

    # Task 4: Spatial resampling (if any processing step requires it)
    if config.steps.add_panchromatic or config.steps.band_misalignment:
        task_list.append(ResamplingTask(current_feature, config))
        _discard(current_feature)
        current_feature = f"{current_feature}_RES"

    # Task 5: Band misalignment (if enabled)
    if config.steps.band_misalignment:
        task_list.append(
            BandMisalignmentTask(
                (FeatureType.DATA, current_feature),
                (FeatureType.DATA, "S2_MISALIGNED"),
                processing_level=ProcessingLevels.L1C,
                std_sea=config.misalignment_std_sea,
                std_land=config.misalignment_std_land,
                interpolation_method=cv2.INTER_NEAREST,
            )
        )
        # Remove resampled input feature to free memory
        _discard(current_feature)
        current_feature = "S2_MISALIGNED"

    # Tasks 6 & 7: SNR and PSF. The two are independent so either can be isolated;
    # whichever run are chained SNR -> PSF.
    apply_snr = config.steps.snr_simulation
    apply_psf = config.steps.psf_filtering

    if apply_snr and config.snr_values is None:
        logger.warning(
            "snr_simulation is enabled but config.snr_values is None - skipping the SNR stage"
        )
        apply_snr = False

    if apply_snr or apply_psf:
        if config.snr_psf_method == "executable" and config.phisat2_exec_path:
            # The executable exposes SNR and PSF as two separate invocations.
            if apply_snr:
                task_list.append(
                    PhisatCalculationTask(
                        input_feature=(FeatureType.DATA, current_feature),
                        output_feature=(FeatureType.DATA, "L_out_SNR"),
                        executable=config.phisat2_exec_path,
                        calculation="SNR",
                    )
                )
                current_feature = "L_out_SNR"
            if apply_psf:
                task_list.append(
                    PhisatCalculationTask(
                        input_feature=(FeatureType.DATA, current_feature),
                        output_feature=(FeatureType.DATA, "L_out_PSF"),
                        executable=config.phisat2_exec_path,
                        calculation="PSF",
                    )
                )
                current_feature = "L_out_PSF"
        elif config.snr_psf_method == "alternative":
            # One task runs both stages, each gated by its own flag.
            psf_kernels = get_psf_kernels_dict(
                sigma=config.psf_kernel_sigma,
                bands=config.bands_names,
                size=2 * int(np.ceil(3 * config.psf_kernel_sigma)) + 1,
            )
            task_list.append(
                AlternativePhisatCalculationTask(
                    input_feature=(FeatureType.DATA, current_feature),
                    snr_feature=(FeatureType.DATA, "L_out_SNR"),
                    psf_feature=(FeatureType.DATA, "L_out_PSF"),
                    snr_values=config.snr_values,
                    psf_kernel=psf_kernels,
                    l_ref=config.radiance_reference,
                    apply_snr=apply_snr,
                    apply_psf=apply_psf,
                )
            )
            produced = "L_out_PSF" if apply_psf else "L_out_SNR"
            _discard(current_feature)
            if apply_snr and apply_psf:
                # SNR output was only an intermediate on the way to PSF
                _discard("L_out_SNR")
            current_feature = produced
        else:
            logger.warning(
                f"snr_psf_method={config.snr_psf_method!r} is not usable "
                "(the executable backend also needs phisat2_exec_path) - "
                "skipping the SNR and PSF stages"
            )

    # Task 8: Reflectance conversion (if L1C and enabled)
    if config.steps.reflectance_conversion and config.processing_level.value == ProcessingLevels.L1C.value:
        task_list.append(
            CalculateReflectanceTask(
                (FeatureType.DATA, current_feature),
                (FeatureType.DATA, "S2_REFLECTANCE"),
                processing_level=config.processing_level,
            )
        )
        # Remove the consumed intermediate to free memory before export
        _discard(current_feature)
        current_feature = "S2_REFLECTANCE"

    # Export whatever the last enabled step produced
    export_feature = current_feature
    logger.info(f"Simulation chain output feature: {export_feature}")

    # Task 9: Export to TIFF (final task)
    task_list.append(ExportToTiffTask(
        feature=(FeatureType.DATA, export_feature),
        image_dtype=np.float32,
        # This path is note actually used but EOExecutor requires it to be set.
        path=f"{output_dir}/v1.1/data/flood_events/HandLabeled/S2Hand",
    ))

    # Connect all tasks with linearly_connect_tasks
    nodes = linearly_connect_tasks(*task_list)
    workflow = EOWorkflow(nodes)
    
    # Prepare execution kwargs
    execution_kwargs = [
        {
            nodes[0]: {
                "image": args["image"],
                "mask": args["mask"],
                "bands_names": config.bands_names,
                "source_resolution": config.source_resolution,
                "location_coords": args["location_coords"],
                "metadata": args["metadata"],
                "acquisition_date": args["acquisition_date"]
            },
            nodes[-1]: {
                "filename": args["output_tiff_path"],
            }
        }
        for args in exec_args
    ]
    
    # Create and configure EOExecutor
    # If not verbose, we add a filter to EOExecutor to suppress DEBUG logs in the 
    # execution-specific log files, since EOExecutor hardcodes DEBUG level for them.
    logs_filter = None
    if not verbose:
        logs_filter = logging.Filter()
        logs_filter.filter = lambda record: record.levelno >= logging.INFO

    executor = EOExecutor(
        workflow=workflow,
        execution_kwargs=execution_kwargs,
        save_logs=save_logs,
        logs_folder=str(logs_folder),
        # logs_filter=logs_filter,
    )

    logger.info(f"\nStarting parallel execution with {workers} workers...")
    executor.run(workers=workers)

    logger.info(f"\nExecution complete. Total tasks: {len(exec_args)}")
    return executor
