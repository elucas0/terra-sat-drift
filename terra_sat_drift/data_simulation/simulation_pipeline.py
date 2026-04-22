"""Simulation pipeline wrapping phisat2_utils tasks for on-the-fly Φ-sat-2 synthesis."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from datetime import datetime
import numpy as np
import json
import rasterio
import cv2

from sentinelhub.geometry import BBox
from sentinelhub.constants import CRS

from eolearn.core.eotask import EOTask
from eolearn.core.eodata import EOPatch
from eolearn.core.eonode import linearly_connect_tasks
from eolearn.core.eoworkflow import EOWorkflow
from eolearn.core.eoexecution import EOExecutor
from eolearn.io.raster_io import ExportToTiffTask
from eolearn.core.constants import FeatureType
from eolearn.core.core_tasks import MapFeatureTask
from eolearn.features.utils import spatially_resize_image as resize_images
from simulation_config import SimulationConfig

from phisat2_utils import (  
    AddPANBandTask,  
    AddMetadataTask,
    BandMisalignmentTask,  
    CalculateRadianceTask,  
    CalculateReflectanceTask,  
    PhisatCalculationTask,
)
from phisat2_constants import S2_RESOLUTION, PHISAT2_RESOLUTION, ProcessingLevels  


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


def _create_bbox_from_rasterio(src) -> BBox:
    """Create BBox object from rasterio source.
        
    Args:
        src: Rasterio source object with geospatial info.
       
    Returns:
        BBox object for the raster extent.
    """
    bounds = src.bounds
    bbox = BBox(
        bbox=(bounds.left, bounds.bottom, bounds.right, bounds.top),
        crs=CRS(src.crs)
    )
    return bbox


class LoadS2FileTask(EOTask):
    """Load S2 TIFF file and create EOPatch with metadata."""

    def execute(self, *, s2_tiff_path: str, metadata: dict, **kwargs) -> EOPatch:
        """Load S2 TIFF and initialize EOPatch."""
        s2_path = Path(s2_tiff_path)
        
        with rasterio.open(s2_path) as src:
            s2_data = src.read().astype(np.float32)
            try:
                bbox = _create_bbox_from_rasterio(src)
            except Exception as e:
                print(f"Warning: Could not extract bbox: {e}")
                bbox = None

        s2_data = np.transpose(s2_data, (1, 2, 0))
        eopatch = EOPatch(bbox=bbox, timestamps=[datetime.now()])
        eopatch[FeatureType.DATA, "S2_BANDS"] = s2_data[np.newaxis, :, :, :]

        # Add metadata to EOPatch
        try:
            add_meta_task = AddMetadataTask()
            eopatch = add_meta_task.execute(eopatch, metadata)
        except Exception as e:
            print(f"Warning: Failed to fetch metadata: {e}")

        return eopatch


class ResamplingTask(EOTask):
    """Spatially resample bands to Φ-sat-2 pixel size."""

    def __init__(self, pan_feature: str, config: SimulationConfig):
        self.pan_feature = pan_feature
        self.config = config

    def execute(self, eopatch: EOPatch) -> EOPatch:
        features_to_resize = {
            FeatureType.DATA: [self.pan_feature],
        }
        if "sunZenithAngles" in eopatch.data:
            features_to_resize[FeatureType.DATA].append("sunZenithAngles")

        s2_data_shape = eopatch.data[self.pan_feature].shape
        new_size = (
            int((s2_data_shape[1] * S2_RESOLUTION) / PHISAT2_RESOLUTION),
            int((s2_data_shape[2] * S2_RESOLUTION) / PHISAT2_RESOLUTION),
        )

        for feature_type in features_to_resize.keys():
            for feature in features_to_resize[feature_type]:
                resize_task = MapFeatureTask(
                    (feature_type, feature),
                    (feature_type, f"{feature}_RES"),
                    resize_images,
                    new_size=new_size,
                    resize_method="nearest",
                )
                eopatch = resize_task(eopatch)

        return eopatch

def simulate_with_executor(
    config: SimulationConfig,
    source_dir: Path,
    output_dir: Path,
    metadata_dir: Path,
    logs_folder: str,
    pattern: str,
    workers: int = 4,
    save_logs: bool = True,
) -> EOExecutor:
    """Execute parallel simulation using EOExecutor with decomposed task nodes.

    Creates an EOWorkflow by composing individual simulation tasks with linearly_connect_tasks
    for parallel execution across multiple workers for all S2 .tiff files in the source directory.

    Args:
        config: SimulationConfig instance defining processing steps and parameters.
        source_dir: Directory containing raw S2 .tiff files. If None, uses config.s2_source_dir.
        pattern: Glob pattern for .tiff files.
        workers: Number of parallel workers for execution.
        save_logs: Whether to save execution logs.

    Returns:
        EOExecutor instance with execution results.
    """
    # Collect all TIFF files and their metadata
    tiff_files = list(source_dir.glob(pattern))
    tiff_files = tiff_files[:5]  # Limit to first 10 files for testing
    
    print(f"Found {len(tiff_files)} TIFF files to process")

    # Create execution arguments for each file
    exec_args = []
    for s2_file in tiff_files:
        try:
            metadata = _load_metadata_from_file(s2_file, metadata_dir)
            output_file = output_dir / f"simulated_{config.processing_level.name}_{s2_file.name}"

            exec_args.append({
                "s2_tiff_path": str(s2_file),
                "output_tiff_path": str(output_file),
                "metadata": metadata,
            })
        except FileNotFoundError as e:
            print(f"Skipping {s2_file.name}: {e}")
            continue

    if not exec_args:
        raise ValueError(f"No valid files with metadata found in {source_dir}")

    print(f"\nPrepared {len(exec_args)} execution arguments")

    # Build individual task nodes to be connected with linearly_connect_tasks
    task_list = []

    # Task 1: Load S2 TIFF file and create EOPatch with metadata
    task_list.append(LoadS2FileTask())

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
                std_sea=6,
                interpolation_method=cv2.INTER_NEAREST,
            )
        )

    # Task 6: SNR simulation (if enabled and using executable)
    if (config.steps.snr_simulation and 
        config.snr_psf_method == "executable" and 
        config.phisat2_exec_path):
        
        input_feature = "S2_MISALIGNED" if config.steps.band_misalignment else (
            "BANDS-RAD-PAN_RES" if config.steps.add_panchromatic else (
                "S2_RADIANCE_RES" if config.steps.radiance else "S2_BANDS_RES"
            )
        )
        task_list.append(
            PhisatCalculationTask(
                input_feature=(FeatureType.DATA, input_feature),
                output_feature=(FeatureType.DATA, "L_out_SNR"),
                executable=config.phisat2_exec_path,
                calculation="SNR",
            )
        )

    # Task 7: PSF filtering (if enabled and using executable)
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
    if config.steps.reflectance_conversion and config.processing_level == "L1C":
        input_feature = "L_out_PSF" if config.steps.psf_filtering else (
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
        
    # Determine which feature to export
    if config.steps.reflectance_conversion and config.processing_level == "L1C":
        export_feature = "S2_REFLECTANCE"
    elif config.steps.psf_filtering:
        export_feature = "L_out_PSF"
    elif config.steps.snr_simulation:
        export_feature = "L_out_SNR"
    else:
        export_feature = "S2_BANDS"

    # Task 9: Export to TIFF (final task)
    task_list.append(ExportToTiffTask(
        feature=(FeatureType.DATA, export_feature),
        folder=str(output_dir),
        image_dtype=np.float32
    ))

    # Connect all tasks with linearly_connect_tasks
    nodes = linearly_connect_tasks(*task_list)
    workflow = EOWorkflow(nodes)

    # Prepare execution kwargs
    execution_kwargs = [
        {
            nodes[0]: {
                "s2_tiff_path": args["s2_tiff_path"],
                "metadata": args["metadata"],
            },
            nodes[-1]: {
                "filename": args["output_tiff_path"],
            }
        }
        for args in exec_args
    ]
    
    # Create and configure EOExecutor
    executor = EOExecutor(
        workflow=workflow,
        execution_kwargs=execution_kwargs,
        save_logs=save_logs,
        logs_folder=logs_folder,
    )

    print(f"\nStarting parallel execution with {workers} workers...")
    executor.run(workers=workers)

    print(f"\nExecution complete. Total tasks: {len(exec_args)}")
    return executor
