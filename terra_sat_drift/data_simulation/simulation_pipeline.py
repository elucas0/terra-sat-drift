"""Simulation pipeline wrapping phisat2_utils tasks for on-the-fly Φ-sat-2 synthesis."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from datetime import datetime
import numpy as np
import json
import rasterio
from scipy.ndimage import gaussian_filter
import cv2
import numpy as np

from sentinelhub.geometry import BBox
from sentinelhub.constants import CRS

from eolearn.core.eodata import EOPatch
from eolearn.core.eonode import linearly_connect_tasks
from eolearn.core.eoworkflow import EOWorkflow
from eolearn.core.core_tasks import RemoveFeatureTask
from eolearn.io.raster_io import ExportToTiffTask
from eolearn.core.constants import FeatureType
from eolearn.core.core_tasks import MapFeatureTask
from eolearn.features.utils import spatially_resize_image as resize_images
from simulation_config import SimulationConfig, SimulationSteps

from tqdm import tqdm
from tqdm import tqdm
from phisat2_utils import (  
    AddPANBandTask,  
    AddMetadataTask,
    BandMisalignmentTask,  
    CalculateRadianceTask,  
    CalculateReflectanceTask,  
    AlternativePhisatCalculationTask,
    PhisatCalculationTask,
    
)
from phisat2_constants import S2_RESOLUTION, PHISAT2_RESOLUTION, ProcessingLevels  

class SimulationPipeline:
    """Orchestrates Φ-sat-2 on-the-fly simulation from cached S2 L1C .tiff files.
    
    This pipeline wraps the phisat2_utils tasks to enable:
    - Configurable simulation steps (radiance, PAN, misalignment, SNR, PSF, L1C)
    - Batch processing with EOExecutor
    - Caching of raw S2 data for scaled experiments
    """

    def __init__(self, config: SimulationConfig) -> None:
        """Initialize pipeline with simulation configuration.

        Args:
            config: SimulationConfig instance defining processing steps and parameters.
        """
        self.config = config
        output_dir = Path(self.config.output_dir) if isinstance(self.config.output_dir, str) else self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

    def _load_metadata_from_file(self, tiff_path: Path | str) -> dict:
        """Load metadata JSON file associated with a TIFF file.
        
        Extracts the ID (last part) from the TIFF filename and looks for matching metadata.
        For example: S2B_Crop_10m_2091.tif -> 2091_S2B_metadata.json
        
        Args:
            tiff_path: Path to the TIFF file
            
        Returns:
            Dictionary with metadata (earth_sun_dist, solar_irradiances, sun_zenith_angles),
            or raises FileNotFoundError if no metadata file found.
        """
        tiff_path = Path(tiff_path)
        stem = tiff_path.stem
        parent_dir = tiff_path.parent
        
        # Extract the ID (last string) by splitting on underscores
        parts = stem.split('_')
        if not parts:
            print(f"Warning: Could not extract ID from filename {tiff_path.name}")
            raise FileNotFoundError("Could not extract ID from TIFF filename")
        
        id_str = parts[-1]
        satellite = parts[0] if len(parts) > 0 else ""
        
        # Try common naming patterns with the extracted ID
        possible_metadata_paths = [
            parent_dir / f"{id_str}_{satellite}_metadata.json",
            parent_dir / f"{id_str}_metadata.json",
            parent_dir / f"{id_str}.json",
            parent_dir / f"{stem}_metadata.json",
            parent_dir / f"{stem}.json",
        ]
        
        for metadata_path in possible_metadata_paths:
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    print(f"✓ Loaded metadata from {metadata_path.name}")
                    return metadata
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse JSON metadata {metadata_path}: {e}")
                except Exception as e:
                    print(f"Warning: Error loading metadata file {metadata_path}: {e}")
        
        print(f"Warning: No metadata file found for {tiff_path.name} (ID: {id_str})")
        raise FileNotFoundError("No metadata file found")

    def _create_bbox_from_rasterio(self, src) -> BBox:
        """Create BBox object from rasterio source.
            
        Args:
            src: Rasterio source object with geospatial info.
           
        Returns:
            BBox object for the raster extent.
        """
        # Get the bounds from rasterio
        bounds = src.bounds
            
        # Create BBox (left, bottom, right, top)
        bbox = BBox(
            bbox=(bounds.left, bounds.bottom, bounds.right, bounds.top),
                crs=CRS(src.crs)
        )
        return bbox

    def simulate_single_file(
        self, s2_tiff_path: Path | str, output_tiff_path: Path | str, metadata: dict
    ) -> bool:
        """Apply simulation pipeline to a single S2 .tiff file using phisat2_utils tasks.

        This creates an EOPatch from the raw S2 .tiff, applies configurable simulation
        steps via phisat2_utils tasks, then extracts and saves the result.

        Args:
            s2_tiff_path: Path to raw S2 .tiff (7 or 8 bands).
            output_tiff_path: Path to save simulated Φ-sat-2 .tiff.

        Returns:
            True if successful, False otherwise.
        """
        try:
            s2_tiff_path = Path(s2_tiff_path)
            output_tiff_path = Path(output_tiff_path)
            output_tiff_path.parent.mkdir(parents=True, exist_ok=True)

            #  Step 0: Load raw S2 and create EOPatch 
            with rasterio.open(s2_tiff_path) as src:
                s2_data = src.read().astype(np.float32)  # Shape: (batch, height, width, bands)
                profile = src.profile.copy()
                
                # Extract bbox from geospatial metadata (while file is still open)
                try:
                    bbox = self._create_bbox_from_rasterio(src)
                except Exception as e:
                    print(f"Warning: Could not extract bbox from {s2_tiff_path}: {e}")
                    bbox = None

            # Select "B02", "B03", "B04", "B08", "B05", "B06", "B07"
            # band_indices = [1, 2, 3, 7, 4, 5, 6]
            # s2_data = s2_data[band_indices, :, :]
            
            # Transpose from (bands, height, width) to (height, width, bands)
            s2_data = np.transpose(s2_data, (1, 2, 0))
            
            # Create EOPatch
            eopatch = EOPatch(bbox=bbox, timestamps=[datetime.now()])
            
            # Shape: (time, height, width, bands)
            eopatch[FeatureType.DATA, "S2_BANDS"] = s2_data[np.newaxis, :, :, :]

            # Add metadata to EOPatch: Solar irradiance, Earth-Sun distance, Sun zenith angles
            try:
                add_meta_task = AddMetadataTask()
                eopatch = add_meta_task.execute(eopatch, metadata)
            except Exception as e:
                print(f"Warning: Failed to fetch metadata: {e}")
                print("Skipping radiance conversion - metadata required")
                self.config.steps.radiance = False
                
            #  Radiance conversion
            if self.config.steps.radiance:
                radiance_task = CalculateRadianceTask(
                    (FeatureType.DATA, "S2_BANDS"),
                    (FeatureType.DATA, "S2_RADIANCE"),
                )
                eopatch = radiance_task.execute(eopatch)
                current_feature = "S2_RADIANCE"
            else:
                current_feature = "S2_BANDS"
            
            #  Add panchromatic band
            if self.config.steps.add_panchromatic:
                pan_task = AddPANBandTask(
                    (FeatureType.DATA, current_feature),
                    (FeatureType.DATA, "BANDS-RAD-PAN"),
                )
                eopatch = pan_task.execute(eopatch)
                current_feature = "BANDS-RAD-PAN"
                
            # Spatial resampling to Φ-sat-2 pixel size
            if eopatch.data["sunZenithAngles"] is not None:
                features_to_resize = {
                    FeatureType.DATA: [current_feature, "sunZenithAngles"],
                }
            else:
                features_to_resize = {
                    FeatureType.DATA: [current_feature],
                }
            
            NEW_SIZE = (int((s2_data.shape[0] * S2_RESOLUTION) / PHISAT2_RESOLUTION), int((s2_data.shape[1] * S2_RESOLUTION) / PHISAT2_RESOLUTION))

            resize_task_list = []

            for feature_type in tqdm(features_to_resize.keys()):
                for feature in tqdm(features_to_resize[feature_type]):
                    resize_task_list.append(
                        MapFeatureTask(
                            (feature_type, feature),
                            (feature_type, f"{feature}_RES"),
                            resize_images,
                            new_size=NEW_SIZE,
                            resize_method="nearest",
                        )
                    )
                    eopatch = resize_task_list[-1](eopatch)
                    
            current_feature = f"{current_feature}_RES"
            
            # Band misalignment
            if self.config.steps.band_misalignment:
                processing_level = ProcessingLevels[self.config.processing_level]
                misalign_task = BandMisalignmentTask(
                    (FeatureType.DATA, current_feature),
                    (FeatureType.DATA, "S2_MISALIGNED"),
                    processing_level=processing_level,
                    std_sea=6,
                    interpolation_method=cv2.INTER_NEAREST,
                )
                eopatch = misalign_task.execute(eopatch)
                current_feature = "S2_MISALIGNED"

            # SNR + PSF simulation 
            if self.config.steps.snr_simulation or self.config.steps.psf_filtering:
                # Check which method to use for SNR/PSF calculation
                use_executable = (
                    self.config.snr_psf_method == "executable"
                    and hasattr(self.config, 'phisat2_exec_path')
                    and self.config.phisat2_exec_path
                )
                
                if use_executable and self.config.phisat2_exec_path:
                    # SNR simulation using executable
                    if self.config.steps.snr_simulation:
                        snr_task = PhisatCalculationTask(
                            input_feature=(FeatureType.DATA, current_feature),
                            output_feature=(FeatureType.DATA, "L_out_SNR"),
                            executable=self.config.phisat2_exec_path,
                            calculation="SNR",
                        )
                        eopatch = snr_task.execute(eopatch)
                        current_feature = "L_out_SNR"
                    
                    # PSF filtering using executable
                    if self.config.steps.psf_filtering:
                        psf_task = PhisatCalculationTask(
                            input_feature=(FeatureType.DATA, current_feature),
                            output_feature=(FeatureType.DATA, "L_out_PSF"),
                            executable=self.config.phisat2_exec_path,
                            calculation="PSF",
                        )
                        eopatch = psf_task.execute(eopatch)
                        current_feature = "L_out_PSF"
                elif self.config.snr_values:
                    # Use AlternativePhisatCalculationTask

                    # Create PSF kernels (Gaussian approximation)
                    psf_kernel = self._create_psf_kernels(self.config.psf_kernel_sigma)

                    snr_psf_task = AlternativePhisatCalculationTask(
                        input_feature=(FeatureType.DATA, current_feature),
                        snr_feature=(FeatureType.DATA, "S2_NOISY") if self.config.steps.snr_simulation else (FeatureType.DATA, current_feature),
                        snr_values=self.config.snr_values,
                        l_ref=self.config.radiance_reference,
                        psf_feature=(FeatureType.DATA, "S2_PSF"),
                        psf_kernel=psf_kernel,
                    )
                    eopatch = snr_psf_task.execute(eopatch)
                    current_feature = "S2_PSF"

            # Reflectance conversion (if L1C) 
            if self.config.steps.reflectance_conversion and self.config.processing_level == "L1C":
                reflectance_task = CalculateReflectanceTask(
                    (FeatureType.DATA, current_feature),
                    (FeatureType.DATA, "S2_REFLECTANCE"),
                    processing_level=ProcessingLevels.L1C,
                )
                eopatch = reflectance_task.execute(eopatch)
                current_feature = "S2_REFLECTANCE"
                
            # Build conditional list of features to remove based on simulation configuration
            features_to_remove = [
                (FeatureType.DATA, "S2_BANDS"),  # Always remove original S2 bands
            ]
            
            # Add conditional removals based on enabled steps
            if self.config.steps.radiance:
                features_to_remove.append((FeatureType.DATA, "S2_RADIANCE"))
            
            if self.config.steps.add_panchromatic:
                features_to_remove.extend([
                    (FeatureType.DATA, "BANDS-RAD-PAN"),
                    (FeatureType.DATA, "BANDS-RAD-PAN_RES"),
                ])
            
            if self.config.steps.band_misalignment:
                features_to_remove.append((FeatureType.DATA, "S2_MISALIGNED"))
            
            if (self.config.snr_psf_method == "alternative" and 
                (self.config.steps.snr_simulation or self.config.steps.psf_filtering)):
                if self.config.steps.snr_simulation:
                    features_to_remove.append((FeatureType.DATA, "S2_NOISY"))
                if self.config.steps.psf_filtering:
                    features_to_remove.append((FeatureType.DATA, "S2_PSF"))
            
            if "sunZenithAngles" in eopatch.data:
                features_to_remove.append((FeatureType.DATA, "sunZenithAngles"))
                
            remove_feature_task = RemoveFeatureTask(features_to_remove)
            eopatch = remove_feature_task.execute(eopatch)
            
            casting_task = MapFeatureTask(
                (FeatureType.DATA, current_feature),
                (FeatureType.DATA, current_feature),
                np.float32
            )
            eopatch = casting_task(eopatch)
              
            export_task = ExportToTiffTask(
                feature=(FeatureType.DATA, current_feature),
                folder=str(output_tiff_path),
            )
            export_task.execute(eopatch)

            return True

        except Exception as exc:
            print(f"Error simulating {s2_tiff_path}: {exc}")
            import traceback
            traceback.print_exc()
            return False

    def _create_psf_kernels(self, sigma) -> dict:
        """Create PSF kernels for all Φ-sat-2 bands using Gaussian approximation.
        
        Returns:
            Dictionary mapping band names to 7x7 PSF kernels.
        """

        kernel_bands = ["B1", "B2", "B3", "B0", "B7", "B4", "B5", "B6"]
        psf_kernels = {}

        for band in kernel_bands:
            # Create a 7x7 Gaussian kernel with sigma parameter
            kernel = np.zeros((7, 7))
            kernel[3, 3] = 1
            kernel = gaussian_filter(kernel, sigma)
            # Normalize
            kernel = kernel / kernel.sum()
            psf_kernels[band] = kernel

        return psf_kernels

    def batch_simulate_from_source_dir(
        self, source_dir: Optional[Path | str] = None, pattern: str = "*.tif",
    ) -> dict:
        """Apply simulation to all S2 .tiff files in a source directory.

        Args:
            source_dir: Directory containing raw S2 .tiff files. If None, uses config.s2_source_dir.
            pattern: Glob pattern for .tiff files.

        Returns:
            Dictionary with results: {"successful": [...], "failed": [...]}
        """
        source_dir = Path(source_dir or self.config.s2_source_dir)
        output_dir = Path(self.config.output_dir) if isinstance(self.config.output_dir, str) else self.config.output_dir
        results = {"successful": [], "failed": []}
        
        for s2_file in sorted(source_dir.glob(pattern)):
            # Load metadata file associated with this TIFF
            metadata = self._load_metadata_from_file(s2_file)
            
            output_file = output_dir / f"simulated_{self.config.processing_level}_{s2_file.name}"
            success = self.simulate_single_file(s2_file, output_file, metadata)

            if success:
                results["successful"].append(str(output_file))
                print(f"✓ Simulated: {s2_file.name}")
            else:
                results["failed"].append(str(s2_file))
                print(f"✗ Failed: {s2_file.name}")

        print(f"\nSummary: {len(results['successful'])} successful, {len(results['failed'])} failed")
        return results
