"""Simulation pipeline wrapping phisat2_utils tasks for on-the-fly Φ-sat-2 synthesis."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import json
import cv2
import logging
from datetime import datetime
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
                    # temporal_coords is [year, day_of_year]
                    year = int(temporal_coords[0, 0].item())
                    day_of_year = int(temporal_coords[0, 1].item())
                    acquisition_date = datetime.strptime(f"{year}:{day_of_year}", "%Y:%j")
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

    # Task 2: Radiance conversion (if enabled)
    if config.steps.radiance:
        task_list.append(
            CalculateRadianceTask(
                (FeatureType.DATA, "S2_BANDS"),
                (FeatureType.DATA, "S2_RADIANCE"),
            )
        )

    # Task 3: Add panchromatic band (if enabled)
    if config.steps.add_panchromatic:
        current_input = "S2_RADIANCE" if config.steps.radiance else "S2_BANDS"
        task_list.append(
            AddPANBandTask(
                (FeatureType.DATA, current_input),
                (FeatureType.DATA, "BANDS-RAD-PAN"),
            )
        )

    # Task 4: Spatial resampling (if any processing step requires it)
    if config.steps.add_panchromatic or config.steps.band_misalignment:
        pan_feature = "BANDS-RAD-PAN" if config.steps.add_panchromatic else (
            "S2_RADIANCE" if config.steps.radiance else "S2_BANDS"
        )
        task_list.append(ResamplingTask(pan_feature, config))
        task_list.append(RemoveFeatureTask([(FeatureType.DATA, pan_feature)]))

    # Task 5: Band misalignment (if enabled)
    if config.steps.band_misalignment:
        pan_feature_res = (
            "BANDS-RAD-PAN_RES" if config.steps.add_panchromatic else (
                "S2_RADIANCE_RES" if config.steps.radiance else "S2_BANDS_RES"
            )
        )
        task_list.append(
            BandMisalignmentTask(
                (FeatureType.DATA, pan_feature_res),
                (FeatureType.DATA, "S2_MISALIGNED"),
                processing_level=ProcessingLevels.L1C,
                std_sea=config.misalignment_std_sea,
                std_land=config.misalignment_std_land,
                interpolation_method=cv2.INTER_NEAREST,
            )
        )
        # Remove resampled input feature to free memory
        task_list.append(RemoveFeatureTask([(FeatureType.DATA, pan_feature_res)]))

    # Task 6: SNR simulation (if enabled)
    if config.steps.snr_simulation:
        input_feature = "S2_MISALIGNED" if config.steps.band_misalignment else (
            "BANDS-RAD-PAN_RES" if config.steps.add_panchromatic else (
                "S2_RADIANCE_RES" if config.steps.radiance else "S2_BANDS_RES"
            )
        )
        
        if config.snr_psf_method == "executable" and config.phisat2_exec_path:
            task_list.append(
                PhisatCalculationTask(
                    input_feature=(FeatureType.DATA, input_feature),
                    output_feature=(FeatureType.DATA, "L_out_SNR"),
                    executable=config.phisat2_exec_path,
                    calculation="SNR",
                )
            )
        elif config.snr_psf_method == "alternative" and config.snr_values is not None:
            # Preparing PSF kernels for AlternativePhisatCalculationTask
            # Note: AlternativePhisatCalculationTask performs both SNR and PSF in one execute() call
            psf_kernels = get_psf_kernels_dict(
                sigma=config.psf_kernel_sigma,
                bands=config.bands_names,
                size=7
            )
            task_list.append(
                AlternativePhisatCalculationTask(
                    input_feature=(FeatureType.DATA, input_feature),
                    snr_feature=(FeatureType.DATA, "L_out_SNR"),
                    psf_feature=(FeatureType.DATA, "L_out_PSF"),
                    snr_values=config.snr_values,
                    psf_kernel=psf_kernels,
                    l_ref=config.radiance_reference,
                )
            )

    # Task 7: PSF filtering (if enabled and using executable - skipped for alternative as it's handled in Task 6)
    if (config.steps.psf_filtering and 
        config.snr_psf_method == "executable" and 
        config.phisat2_exec_path):
        
        input_feature = "L_out_SNR" if config.steps.snr_simulation else (
            "S2_MISALIGNED" if config.steps.band_misalignment else (
                "BANDS-RAD-PAN_RES" if config.steps.add_panchromatic else (
                    "S2_RADIANCE_RES" if config.steps.radiance else "S2_BANDS_RES"
                )
            )
        )
        task_list.append(
            PhisatCalculationTask(
                input_feature=(FeatureType.DATA, input_feature),
                output_feature=(FeatureType.DATA, "L_out_PSF"),
                executable=config.phisat2_exec_path,
                calculation="PSF",
            )
        )
        
    # Task 8: Reflectance conversion (if L1C and enabled)
    if config.steps.reflectance_conversion and config.processing_level.value == ProcessingLevels.L1C.value:
        input_feature = "L_out_PSF" if ((config.steps.psf_filtering and config.snr_psf_method == "executable") or (config.steps.snr_simulation and config.snr_psf_method == "alternative")) else (
            "L_out_SNR" if config.steps.snr_simulation else (
                "S2_MISALIGNED" if config.steps.band_misalignment else (
                    "BANDS-RAD-PAN_RES" if config.steps.add_panchromatic else (
                        "S2_RADIANCE_RES" if config.steps.radiance else "S2_BANDS_RES"
                    )
                )
            )
        )
        task_list.append(
            CalculateReflectanceTask(
                (FeatureType.DATA, input_feature),
                (FeatureType.DATA, "S2_REFLECTANCE"),
                processing_level=config.processing_level,
            )
        )
        # Remove intermediate processing outputs to free memory before export
        features_to_remove = []
        if (config.steps.psf_filtering and config.snr_psf_method == "executable") or (config.steps.snr_simulation and config.snr_psf_method == "alternative"):
            features_to_remove.append((FeatureType.DATA, "L_out_PSF"))
        if config.steps.snr_simulation:
            features_to_remove.append((FeatureType.DATA, "L_out_SNR"))
        if config.steps.band_misalignment and not config.steps.snr_simulation and not (config.steps.psf_filtering and config.snr_psf_method == "executable"):
            features_to_remove.append((FeatureType.DATA, "S2_MISALIGNED"))
        if features_to_remove:
            task_list.append(RemoveFeatureTask(features_to_remove))
        
    # Determine which feature to export
    if config.steps.reflectance_conversion and config.processing_level.value == ProcessingLevels.L1C.value:
        export_feature = "S2_REFLECTANCE"
    elif (config.steps.psf_filtering and config.snr_psf_method == "executable") or (config.steps.snr_simulation and config.snr_psf_method == "alternative"):
        export_feature = "L_out_PSF"
    elif config.steps.snr_simulation:
        export_feature = "L_out_SNR"
    else:
        export_feature = "S2_BANDS"

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
