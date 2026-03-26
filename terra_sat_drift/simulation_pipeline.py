"""Simulation pipeline wrapping phisat2_utils tasks for on-the-fly Φ-sat-2 synthesis."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from datetime import datetime
import numpy as np
import json
from shapely.geometry import Point, shape

from eolearn.core.eodata import EOPatch
from eolearn.core.constants import FeatureType
from sentinelhub import BBox, CRS
from phisat2_utils import (  
    AddPANBandTask,  
    AddMetadataTask,
    BandMisalignmentTask,  
    CalculateRadianceTask,  
    CalculateReflectanceTask,  
    AlternativePhisatCalculationTask,
)
from phisat2_constants import ProcessingLevels  

class SimulationPipeline:
    """Orchestrates Φ-sat-2 on-the-fly simulation from cached S2 L1C .tiff files.
    
    This pipeline wraps the phisat2_utils tasks to enable:
    - Configurable simulation steps (radiance, PAN, misalignment, SNR, PSF, L1C)
    - Batch processing with EOExecutor
    - Caching of raw S2 data for scaled experiments
    """

    def __init__(self, config) -> None:
        """Initialize pipeline with simulation configuration.

        Args:
            config: SimulationConfig instance defining processing steps and parameters.
        """
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Sentinel Hub config for metadata fetching
        self._init_sh_config()

    def _create_bbox_from_rasterio(self, src) -> BBox:
        """Create BBox object from rasterio source.
        
        Args:
            src: Rasterio source object with geospatial info.
            
        Returns:
            BBox object for the raster extent.
        """
        # Get the bounds from rasterio
        bounds = src.bounds  # (left, bottom, right, top)
        
        # Determine CRS (default to WGS84 if not specified)
        crs = src.crs if src.crs else CRS.WGS84
        crs_epsg = crs.to_epsg() if crs else 4326
        
        # Create BBox (left, bottom, right, top)
        bbox = BBox(
            bbox=(bounds.left, bounds.bottom, bounds.right, bounds.top),
            crs=CRS(crs_epsg)
        )
        return bbox

    def _init_sh_config(self) -> None:
        """Initialize Sentinel Hub configuration for metadata fetching.
        
        Attempts to load SHConfig from:
        1. Provided config path (self.config.sh_config_path) - expands ~ and relative paths
        2. Default environment configuration
        3. None if no config available
        """
        from sentinelhub import SHConfig
        
        self.sh_config = None
        
        if self.config.sh_config_path:
            try:
                # Expand path (handle ~ and relative paths)
                config_path = Path(self.config.sh_config_path).expanduser().resolve()
                
                if not config_path.exists():
                    print(f"Warning: Sentinel Hub config file not found at {config_path}")
                    print("Attempting to use default SHConfig...")
                else:
                    self.sh_config = SHConfig()
                    # Read json config and update SHConfig
                    with open(config_path, "r") as f:
                        sh_config_dict = json.load(f)
                    for key, value in sh_config_dict.items():
                        setattr(self.sh_config, key, value)
                    print(f"✓ Loaded Sentinel Hub config from {config_path}")
            except Exception as e:
                print(f"Warning: Could not load SHConfig from {self.config.sh_config_path}: {e}")
                print("AddMetadataTask will not be able to fetch from AWS")

    def _load_sen1floods_metadata(self, metadata_path: str = "datasets/sen1floods11/v1.1/Sen1Floods11_Metadata.geojson"):
        """Load Sen1Floods11 metadata GeoJSON file.
        
        Args:
            metadata_path: Path to the metadata GeoJSON file.
            
        Returns:
            Dictionary with geojson data or None if file not found.
        """
        try:
            metadata_path = Path(metadata_path).expanduser().resolve()
            if not metadata_path.exists():
                print(f"Warning: Metadata file not found at {metadata_path}")
                return None
            
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            return metadata
        except Exception as e:
            print(f"Warning: Could not load metadata file: {e}")
            return None

    def _get_acquisition_date_from_bbox(self, bbox: BBox, metadata: dict) -> Optional[datetime]:
        """Extract acquisition date from Sen1Floods11 metadata by matching bbox coordinates.
        
        Matches the center point of the bbox against the geometry polygons in the metadata.
        
        Args:
            bbox: BBox object from the raster file.
            metadata: Loaded geojson metadata dictionary.
            
        Returns:
            datetime object with the acquisition date, or None if no match found.
        """
        if metadata is None or "features" not in metadata:
            return None
        
        try:
            # Get center point of bbox as Shapely Point (bbox.middle returns a tuple)
            center_coords = bbox.middle
            center_point = Point(center_coords[0], center_coords[1])
            
            # Search through features to find matching geometry
            for feature in metadata.get("features", []):
                geometry = feature.get("geometry")
                properties = feature.get("properties", {})
                
                if geometry is None:
                    continue
                
                try:
                    # Convert geojson geometry to shapely shape
                    geom_shape = shape(geometry)
                    
                    # Check if center point is within this geometry
                    if geom_shape.contains(center_point):
                        s2_date_str = properties.get("s2_date")
                        if s2_date_str:
                            # Parse date string (format: "YYYY/MM/DD")
                            date_obj = datetime.strptime(s2_date_str, "%Y/%m/%d")
                            location = properties.get("location", "Unknown")
                            print(f"✓ Found acquisition date {date_obj.date()} for location {location}")
                            return date_obj
                except Exception as e:
                    continue
            
            print("Warning: No matching metadata found for this location")
            return None
        except Exception as e:
            print(f"Warning: Error matching bbox to metadata: {e}")
            return None

    def simulate_single_file(
        self, s2_tiff_path: Path | str, output_tiff_path: Path | str
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
            import rasterio
            import traceback

            s2_tiff_path = Path(s2_tiff_path)
            output_tiff_path = Path(output_tiff_path)
            output_tiff_path.parent.mkdir(parents=True, exist_ok=True)

            # ===== Step 0: Load raw S2 and create EOPatch =====
            with rasterio.open(s2_tiff_path) as src:
                s2_data = src.read().astype(np.float32)  # Shape: (batch, height, width, bands)
                profile = src.profile.copy()
                metadata = src.tags()
                
                # Extract bbox from geospatial metadata (while file is still open)
                try:
                    bbox = self._create_bbox_from_rasterio(src)
                except Exception as e:
                    print(f"Warning: Could not extract bbox from {s2_tiff_path}: {e}")
                    bbox = None

            # Select "B02", "B03", "B04", "B08", "B05", "B06", "B07"
            band_indices = [1, 2, 3, 7, 4, 5, 6]  # Assuming original order is B01-B08
            s2_data = s2_data[band_indices, :, :]
            
            # Transpose from (bands, height, width) to (height, width, bands)
            s2_data = np.transpose(s2_data, (1, 2, 0))
            
            # Create EOPatch with single timestamp
            eopatch = EOPatch()
            eopatch.bbox = bbox
            
            # Extract acquisition date from metadata if available
            acquisition_date = None
            if eopatch.bbox is not None:
                try:
                    metadata = self._load_sen1floods_metadata(metadata_path="terra-sat-drift/datasets/sen1floods11/v1.1/Sen1Floods11_Metadata.geojson")
                    acquisition_date = self._get_acquisition_date_from_bbox(eopatch.bbox, metadata)
                except Exception as e:
                    print(f"Warning: Could not extract acquisition date: {e}")
            
            # Use acquisition date or fall back to current datetime
            eopatch.timestamp = [acquisition_date if acquisition_date else datetime.now()]
            
            # Shape: (time, height, width, bands)
            eopatch[FeatureType.DATA, "S2_BANDS"] = s2_data[np.newaxis, :, :, :]

            # ===== AddMetadataTask: Fetch solar irradiance and Earth-Sun distance =====
            if eopatch.bbox is not None:
                try:
                    print(f"Fetching AWS metadata for {s2_tiff_path.name}...")
                    add_metadata_task = AddMetadataTask(config=self.sh_config)
                    eopatch = add_metadata_task.execute(eopatch)
                    print(f"✓ Metadata fetched successfully")
                except Exception as e:
                    print(f"Warning: Failed to fetch metadata from AWS: {e}")
                    print("Skipping radiance conversion, PAN addition, requires metadata")
                    self.config.steps.radiance = False
                    self.config.steps.add_panchromatic = False  # PAN task also requires metadata
            else:
                print("Warning: No bbox available. Skipping AWS metadata fetch and radiance conversion")
                self.config.steps.radiance = False


            # ===== Step 1: Radiance conversion (if enabled) =====
            if self.config.steps.radiance:
                radiance_task = CalculateRadianceTask(
                    (FeatureType.DATA, "S2_BANDS"),
                    (FeatureType.DATA, "S2_RADIANCE"),
                )
                eopatch = radiance_task.execute(eopatch)
                current_feature = "S2_RADIANCE"
            else:
                current_feature = "S2_BANDS"
            
            # ===== Step 2: Add panchromatic band (if enabled) =====
            if self.config.steps.add_panchromatic:
                pan_task = AddPANBandTask(
                    (FeatureType.DATA, current_feature),
                    (FeatureType.DATA, "S2_WITH_PAN"),
                )
                eopatch = pan_task.execute(eopatch)
                current_feature = "S2_WITH_PAN"

            # ===== Step 3: Band misalignment (if enabled) =====
            if self.config.steps.band_misalignment:
                processing_level = ProcessingLevels[self.config.processing_level]
                misalign_task = BandMisalignmentTask(
                    (FeatureType.DATA, current_feature),
                    (FeatureType.DATA, "S2_MISALIGNED"),
                    processing_level=processing_level,
                    std_sea=self.config.misalignment_std_sea,
                )
                eopatch = misalign_task.execute(eopatch)
                current_feature = "S2_MISALIGNED"

            # ===== Step 4: SNR + PSF simulation =====
            if self.config.steps.snr_simulation or self.config.steps.psf_filtering:
                snr_values = self.config.snr_values or {
                    "B02": 15,
                    "B03": 15,
                    "B04": 15,
                    # "PAN": 10,
                    "B08": 20,
                    "B05": 15,
                    "B06": 15,
                    "B07": 15,
                }

                # Create PSF kernels (Gaussian approximation)
                psf_kernel = self._create_psf_kernels()

                snr_psf_task = AlternativePhisatCalculationTask(
                    input_feature=(FeatureType.DATA, current_feature),
                    snr_feature=(FeatureType.DATA, "S2_NOISY") if self.config.steps.snr_simulation else (FeatureType.DATA, current_feature),
                    snr_values=snr_values,
                    l_ref=self.config.radiance_reference,
                    psf_feature=(FeatureType.DATA, "S2_PSF"),
                    psf_kernel=psf_kernel,
                )
                eopatch = snr_psf_task.execute(eopatch)
                current_feature = "S2_PSF"

            # ===== Step 5: Reflectance conversion (if L1C) =====
            if self.config.steps.reflectance_conversion and self.config.processing_level == "L1C":
                reflectance_task = CalculateReflectanceTask(
                    (FeatureType.DATA, current_feature),
                    (FeatureType.DATA, "S2_REFLECTANCE"),
                    processing_level=ProcessingLevels.L1C,
                )
                eopatch = reflectance_task.execute(eopatch)
                current_feature = "S2_REFLECTANCE"

            # ===== Extract result and save =====
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
                # Preserve metadata
                if metadata:
                    dst.update_tags(**metadata)

            return True

        except Exception as exc:
            print(f"Error simulating {s2_tiff_path}: {exc}")
            import traceback
            traceback.print_exc()
            return False

    def _create_psf_kernels(self) -> dict:
        """Create PSF kernels for all Φ-sat-2 bands using Gaussian approximation.
        
        Returns:
            Dictionary mapping band names to 7x7 PSF kernels.
        """
        from scipy.ndimage import gaussian_filter

        kernel_bands = ["B1", "B2", "B3", "B7", "B4", "B5", "B6"]
        psf_kernels = {}

        for band in kernel_bands:
            # Create a 7x7 Gaussian kernel with sigma parameter
            kernel = np.zeros((7, 7))
            kernel[3, 3] = 1
            kernel = gaussian_filter(kernel, sigma=self.config.psf_kernel_sigma)
            # Normalize
            kernel = kernel / kernel.sum()
            psf_kernels[band] = kernel

        return psf_kernels

    def batch_simulate_from_source_dir(
        self, source_dir: Optional[Path | str] = None, pattern: str = "*.tiff"
    ) -> dict:
        """Apply simulation to all S2 .tiff files in a source directory.

        Args:
            source_dir: Directory containing raw S2 .tiff files. If None, uses config.s2_source_dir.
            pattern: Glob pattern for .tiff files.

        Returns:
            Dictionary with results: {"successful": [...], "failed": [...]}
        """
        source_dir = Path(source_dir or self.config.s2_source_dir)
        results = {"successful": [], "failed": []}

        for s2_file in sorted(source_dir.glob(pattern)):
            output_file = self.config.output_dir / f"simulated_{s2_file.name}"
            success = self.simulate_single_file(s2_file, output_file)

            if success:
                results["successful"].append(str(output_file))
                print(f"✓ Simulated: {s2_file.name}")
            else:
                results["failed"].append(str(s2_file))
                print(f"✗ Failed: {s2_file.name}")

        print(f"\nSummary: {len(results['successful'])} successful, {len(results['failed'])} failed")
        return results
