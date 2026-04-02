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
from skimage.transform import resize

from sentinelhub.geometry import BBox
from sentinelhub.constants import CRS

from eolearn.core.eodata import EOPatch
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
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
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

    def _load_sen1floods_metadata(self, metadata_path: str):
        """Load Sen1Floods11 metadata GeoJSON file.
        
        Args:
            metadata_path: Path to the metadata GeoJSON file.
            
        Returns:
            Dictionary with geojson data or None if file not found.
        """
        try:
            path = Path(metadata_path).expanduser().resolve()
            if not path.exists():
                print(f"Warning: Metadata file not found at {path}")
                return None
            
            with open(path, "r") as f:
                metadata = json.load(f)
            return metadata
        except Exception as e:
            print(f"Warning: Could not load metadata file: {e}")
            return None

    def _get_acquisition_date_from_country(self, s2_tiff_path: Path | str, metadata: dict) -> Optional[datetime]:
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
            band_indices = [1, 2, 3, 7, 4, 5, 6]
            s2_data = s2_data[band_indices, :, :]
            
            TARGET_10M_SIZE = (512, 512)
            if s2_data.shape[1:] != TARGET_10M_SIZE:
                s2_data = resize(s2_data, (len(band_indices), *TARGET_10M_SIZE), 
                                order=1, preserve_range=True, anti_aliasing=True)
            
            # Transpose from (bands, height, width) to (height, width, bands)
            s2_data = np.transpose(s2_data, (1, 2, 0))
            print(f"Loaded S2 data with shape {s2_data.shape}")
            
            # Create EOPatch
            eopatch = EOPatch(bbox=bbox)
            
            # Extract acquisition date from metadata if available using the tiff file country name
            acquisition_date = None
            try:
                acquisition_date = self._get_acquisition_date_from_country(s2_tiff_path, metadata)
            except Exception as e:
                print(f"Warning: Could not extract acquisition date: {e}")
            
            # Use acquisition date or fall back to current datetime
            eopatch.timestamp = [acquisition_date if acquisition_date else datetime.now()]
            
            # Shape: (time, height, width, bands)
            eopatch[FeatureType.DATA, "S2_BANDS"] = s2_data[np.newaxis, :, :, :]

            # Fetch metadata: Solar irradiance and Earth-Sun distance
            try:
                add_meta_task = AddMetadataTask()
                eopatch = add_meta_task.execute(eopatch)
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
            
            NEW_SIZE = (int(round(s2_data.shape[0] * (S2_RESOLUTION / PHISAT2_RESOLUTION))), int(round(s2_data.shape[1] * (S2_RESOLUTION / PHISAT2_RESOLUTION))))

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

            #  Step 4: SNR + PSF simulation 
            if self.config.steps.snr_simulation or self.config.steps.psf_filtering:
                # Check which method to use for SNR/PSF calculation
                use_executable = (
                    self.config.snr_psf_method == "executable"
                    and hasattr(self.config, 'phisat2_exec_path')
                    and self.config.phisat2_exec_path
                )
                
                if use_executable and self.config.phisat2_exec_path:
                    # Step 4a: SNR simulation using executable
                    if self.config.steps.snr_simulation:
                        snr_task = PhisatCalculationTask(
                            input_feature=(FeatureType.DATA, current_feature),
                            output_feature=(FeatureType.DATA, "L_out_SNR"),
                            executable=self.config.phisat2_exec_path,
                            calculation="SNR",
                        )
                        eopatch = snr_task.execute(eopatch)
                        current_feature = "L_out_SNR"
                    
                    # Step 4b: PSF filtering using executable
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

            #  Step 5: Reflectance conversion (if L1C) 
            if self.config.steps.reflectance_conversion and self.config.processing_level == "L1C":
                reflectance_task = CalculateReflectanceTask(
                    (FeatureType.DATA, current_feature),
                    (FeatureType.DATA, "S2_REFLECTANCE"),
                    processing_level=ProcessingLevels.L1C,
                )
                eopatch = reflectance_task.execute(eopatch)
                current_feature = "S2_REFLECTANCE"

            #  Extract result and save 
            # Remove time dimension: (time, height, width, bands) -> (height, width, bands)
            output_data = eopatch[FeatureType.DATA, current_feature][0]
            # Transpose to rasterio format: (bands, height, width)
            output_data = np.transpose(output_data, (2, 0, 1))
            
            profile.update(
                count=output_data.shape[0],
                dtype=output_data.dtype,
            )
            with rasterio.open(output_tiff_path, "w", **profile) as dst:
                dst.write(output_data)

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
        self, source_dir: Optional[Path | str] = None, pattern: str = "*.tiff", metadata_path: Optional[str] = None
    ) -> dict:
        """Apply simulation to all S2 .tiff files in a source directory.

        Args:
            source_dir: Directory containing raw S2 .tiff files. If None, uses config.s2_source_dir.
            pattern: Glob pattern for .tiff files.
            metadata_path: Optional path to Sen1Floods metadata GeoJSON file for date extraction.

        Returns:
            Dictionary with results: {"successful": [...], "failed": [...]}
        """
        source_dir = Path(source_dir or self.config.s2_source_dir)
        results = {"successful": [], "failed": []}
        
        # Try to load metadata if path provided, otherwise None
        metadata = None
        if metadata_path:
            metadata = self._load_sen1floods_metadata(metadata_path)
        
        for s2_file in sorted(source_dir.glob(pattern)):
            output_file = self.config.output_dir / f"simulated_{s2_file.name}"
            success = self.simulate_single_file(s2_file, output_file, metadata)

            if success:
                results["successful"].append(str(output_file))
                print(f"✓ Simulated: {s2_file.name}")
            else:
                results["failed"].append(str(s2_file))
                print(f"✗ Failed: {s2_file.name}")

        print(f"\nSummary: {len(results['successful'])} successful, {len(results['failed'])} failed")
        return results
