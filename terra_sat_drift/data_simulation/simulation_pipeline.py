"""Simulation pipeline wrapping phisat2_utils tasks for on-the-fly Φ-sat-2 synthesis."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import json
import cv2
from datetime import datetime

from eolearn.core.eonode import linearly_connect_tasks
from eolearn.core.eoworkflow import EOWorkflow
from eolearn.core.eoexecution import EOExecutor
from eolearn.io.raster_io import ExportToTiffTask
from eolearn.core.constants import FeatureType
from simulation_config import SimulationConfig
import pandas as pd
from eolearn.core.core_tasks import RemoveFeatureTask

from phisat2_utils import (  
    AddPANBandTask,  
    BandMisalignmentTask,  
    CalculateRadianceTask,  
    CalculateReflectanceTask,  
    PhisatCalculationTask,
    ResamplingTask,
    LoadS2FileTask
    
)
from phisat2_constants import ProcessingLevels  

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

def _get_acquisition_date_from_country(s2_tiff_path: Path | str, metadata: dict) -> Optional[datetime]:
    """Extract acquisition date from Sen1Floods11 metadata by matching bbox coordinates.
        
     Matches the center point of the bbox against the geometry polygons in the metadata.
        
    Args:
        s2_tiff_path: Path to the S2 .tiff file.
        metadata: Loaded geojson metadata dictionary.
            
    Returns:
        datetime object with the acquisition date, or None if no match found.
        """
    if metadata is None or "features" not in metadata:
        print("Warning: No valid metadata provided for acquisition date extraction")
        
    try:
        # Get the location name from the path
        location_name = Path(s2_tiff_path).stem.split("_")[0].lower()
            
        print(location_name)
        # Search through features to find location of the acquisition
        for feature in metadata.get("features", []):
            properties = feature.get("properties", {})
            location_property = properties.get("location", "").lower()
            if location_property == location_name:
                s2_date_str = properties.get("s2_date")
                if s2_date_str:
                    # Parse date string (format: "YYYY/MM/DD")
                    date_obj = datetime.strptime(s2_date_str, "%Y/%m/%d")
                    location = properties.get("location", "Unknown")
                    return date_obj
            elif location_property == "cambodia" and location_name == "mekong":
                # Special case for Cambodia where location name is inconsistent
                s2_date_str = properties.get("s2_date")
                if s2_date_str:
                    date_obj = datetime.strptime(s2_date_str, "%Y/%m/%d")
                    return date_obj
        print(f"Warning: No matching metadata found for this location: {location_name}")
        return None
    except Exception as e:
        print(f"Warning: Error matching path {s2_tiff_path} to metadata: {e}")
        return None


def simulate_with_executor(
    config: SimulationConfig,
    output_dir: str,
    logs_folder: str,
    source_dir: Path | None = None,
    tiff_files: list[Path] | None = None,
    pattern: str | None = None,
    metadata_file: dict | None = None,
    metadata_dir: Path | None = None,
    workers: int = 4,
    save_logs: bool = True,
    status_csv: Path | str | None = None,
) -> EOExecutor:
    """Execute parallel simulation using EOExecutor with decomposed task nodes.

    Creates an EOWorkflow by composing individual simulation tasks with linearly_connect_tasks
    for parallel execution across multiple workers for all S2 .tiff files in the source directory.

    Args:
        config: SimulationConfig instance defining processing steps and parameters.
        source_dir: Directory containing raw S2 .tiff files.
        pattern: Glob pattern for .tiff files.
        workers: Number of parallel workers for execution.
        save_logs: Whether to save execution logs.
        status_csv: Path to CSV file with simulation status. If provided, only FAILED products are processed.

    Returns:
        EOExecutor instance with execution results.
    """    
    # Load CSV and filter for FAILED products if provided
    failed_product_ids = set()
    if status_csv:
        status_csv = Path(status_csv)
        if status_csv.exists():
            df = pd.read_csv(status_csv)
            failed_products = df[df['simulation_status'] == 'FAILED']
            failed_product_ids = set(failed_products['product_id'].astype(str).values)
            print(f"Loaded {len(failed_product_ids)} FAILED products from {status_csv}")
        else:
            print(f"Warning: Status CSV not found at {status_csv}")
    
    input_tiff_files = []
    # Collect all TIFF files using the source directory and pattern, or from provided list
    if source_dir is not None:
        # Source dir pattern to filter files
        if pattern is not None:
            input_tiff_files = list(source_dir.glob(pattern))
        else:
            input_tiff_files = list(source_dir.glob("*.tif"))
    elif tiff_files is not None:
        input_tiff_files.extend(tiff_files)
    else :
        raise ValueError("Either source_dir or tiff_files must be provided")
        
    # Filter files by failed product IDs if CSV was provided
    if failed_product_ids:
        input_tiff_files = [f for f in input_tiff_files if any(pid in f.name for pid in failed_product_ids)]
        print(f"Filtered to {len(input_tiff_files)} FAILED product TIFF files")
    else:
        print(f"Found {len(input_tiff_files)} TIFF files to process")

    # Sort files by product ID descently for consistent processing order
    # input_tiff_files.sort(key=lambda f: int(f.stem.split('_')[0]), reverse=True)    
    
    # Create execution arguments for each file
    exec_args = []
    metadata = None
    acquisition_date = None
    for s2_file in input_tiff_files:
        try:
            s2_file_name =  s2_file.name
            # Load metadata for this file (when metadata for radiance is already present)
            if metadata_dir:
                metadata = _load_metadata_from_file(s2_file, metadata_dir)
                # We don't need to put the date of the acquisition but EOPatch requires a timestamp
                acquisition_date = datetime.now()
            elif metadata_file:
                acquisition_date = _get_acquisition_date_from_country(s2_file, metadata_file)
            else:
                raise ValueError("Either metadata_file or metadata_dir must be provided to load metadata for simulation")

            output_file = f"{output_dir}/simulated_{config.processing_level.name}_{s2_file_name}"

            exec_args.append({
                "s2_tiff_path": str(s2_file),
                "output_tiff_path": output_file,
                "metadata": metadata,
                "acquisition_date": acquisition_date
            })
            print(exec_args)
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
        task_list.append(RemoveFeatureTask([(FeatureType.DATA, "S2_BANDS")]))

    # Task 3: Add panchromatic band (if enabled)
    if config.steps.add_panchromatic:
        current_input = "S2_RADIANCE" if config.steps.radiance else "S2_BANDS"
        task_list.append(
            AddPANBandTask(
                (FeatureType.DATA, current_input),
                (FeatureType.DATA, "BANDS-RAD-PAN"),
            )
        )
        if config.steps.radiance:
            task_list.append(RemoveFeatureTask([(FeatureType.DATA, "S2_RADIANCE")]))
        else:
            task_list.append(RemoveFeatureTask([(FeatureType.DATA, "S2_BANDS")]))

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
                std_sea=6,
                interpolation_method=cv2.INTER_NEAREST,
            )
        )
        # Remove resampled input feature to free memory
        task_list.append(RemoveFeatureTask([(FeatureType.DATA, pan_feature_res)]))

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
        # Remove SNR output if it exists, and remove misalignment feature to free memory
        features_to_remove = []
        if config.steps.snr_simulation:
            features_to_remove.append((FeatureType.DATA, "L_out_SNR"))
        if config.steps.band_misalignment:
            features_to_remove.append((FeatureType.DATA, "S2_MISALIGNED"))
        if features_to_remove:
            task_list.append(RemoveFeatureTask(features_to_remove))

    # Task 8: Reflectance conversion (if L1C and enabled)
    if config.steps.reflectance_conversion and config.processing_level.value == ProcessingLevels.L1C.value:
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
        # Remove intermediate processing outputs to free memory before export
        features_to_remove = []
        if config.steps.psf_filtering:
            features_to_remove.append((FeatureType.DATA, "L_out_PSF"))
        if config.steps.snr_simulation and not config.steps.psf_filtering:
            features_to_remove.append((FeatureType.DATA, "L_out_SNR"))
        if config.steps.band_misalignment and not config.steps.snr_simulation:
            features_to_remove.append((FeatureType.DATA, "S2_MISALIGNED"))
        if features_to_remove:
            task_list.append(RemoveFeatureTask(features_to_remove))
        
    # Determine which feature to export
    if config.steps.reflectance_conversion and config.processing_level.value == ProcessingLevels.L1C.value:
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
        path=str(output_dir),
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
                "acquisition_date": args["acquisition_date"]
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
