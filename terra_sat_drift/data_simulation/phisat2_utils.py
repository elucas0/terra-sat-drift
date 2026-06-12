import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy.ndimage import convolve

import cv2
import geopandas as gpd
import numpy as np
import shapely.geometry
import shapely.ops
from cv2 import warpAffine
from eolearn.features.utils import ResizeMethod, spatially_resize_image
from eolearn.core import EOPatch, EOTask, FeatureType
from eolearn.io import ExportToTiffTask
from simulation_config import SimulationConfig
from phisat2_constants import (
    BBOX_SIZE_CROPPED,
    CROP_SIZE,
    L1A_RAND_MEAN,
    L1A_RAND_STD,
    L1A_RELATIVE_SHIFTS,
    PAN_WEIGHTS,
    PHISAT2_RESOLUTION,
    S2_RESOLUTION,
    S2_BANDS,
    S2_PAN_BANDS,
    WORLD_GDF,
    ProcessingLevels,
)
from sentinelhub import (
    BBox,
    DataCollection,
    SentinelHubCatalog,
    SHConfig,
    parse_time,
    pixel_to_utm,
    CRS,
)
import rasterio
import requests
from sunpy.coordinates import sun
from astropy import units as u


class AlternativePhisatCalculationTask(EOTask):
    
    def __init__(
        self,
        input_feature: Tuple[FeatureType, str],
        snr_feature: Tuple[FeatureType, str],
        snr_values: List[int],
        l_ref: float,
        psf_feature: Tuple[FeatureType, str],
        psf_kernel: Dict[str, np.array],
    ):
        """Wapper task to simulate dummy SNR and PSF noise

        :param input_feature: Input feature holding radiance values.
        :param snr_feature: Output feature with SNR simulated (dummy) values.
        :param psf_feature: Output feature with PSF simulated (dummy) values.
        :param snr_values: A list of SNR values range for Sentinel-2 bands ["B02", "B03", "B04", "PAN", "B08", "B05", "B06", "B07"].
        :param psf_kernel: A dictionary of PSF 7x7 kernels for PhiSat bands bands ["B1", "B2", "B3", "B0", "B7", "B4", "B5", "B6"].
        :param l_ref: Spectral Radiance at Aperture (W/m^2/sr/um), a reference radiance used to generate the specific SNR
        """
        self.input_feature = input_feature
        self.snr_feature = snr_feature
        self.psf_feature = psf_feature
        self.snr_values = snr_values
        self.psf_kernel = psf_kernel
        self.l_ref = l_ref

    def add_psf(self, eopatch):
        bands_names = eopatch.meta_info.get("bands_names", S2_BANDS)
        
        # if there is a pan band in bands_names add use the same kernel as first band
        if "PAN" in bands_names:
            self.psf_kernel["PAN"] = self.psf_kernel.get(bands_names[0])
        
        convolved_data = np.concatenate(
            [
                np.concatenate(
                    [
                        convolve(
                            _data[..., band],
                            self.psf_kernel[kernel_band],
                            mode="mirror",
                        )[..., np.newaxis]
                        for band, kernel_band in enumerate(
                            bands_names
                        )
                    ],
                    axis=-1,
                )[np.newaxis, ...]
                for _data in eopatch[self.snr_feature]
            ],
            axis=0,
        )
        eopatch[self.psf_feature] = convolved_data

    def add_snr(self, eopatch):
        rng = np.random.default_rng() # Initialize local RNG
        
        bands_names = eopatch.meta_info.get("bands_names", S2_BANDS)
        
        # Use rng.integers instead of np.random.randint
        base_snr = rng.integers(self.snr_values[0], self.snr_values[1])
        snr_bands_values = {band: base_snr for band in bands_names}
        
        radiances = eopatch[self.input_feature]
        
        # Use rng.normal instead of np.random.normal
        random_noise = rng.normal(size=radiances.shape)

        snr = np.array(
            [
                snr_bands_values[band]
                for band in bands_names
            ]
        )
        snr = np.reshape(snr, (1, 1, 1, len(snr)))  # t, h, w, d

        noisy_radiances = radiances + self.l_ref * random_noise / snr
        
        noisy_radiances = np.clip(noisy_radiances, a_min=0.0, a_max=None)
        
        eopatch[self.snr_feature] = noisy_radiances
        
    def execute(self, eopatch: EOPatch) -> EOPatch:
        self.add_snr(eopatch)
        self.add_psf(eopatch)

        return eopatch


class SCLCloudTask(EOTask):
    def __init__(self, scl_feature: Tuple[FeatureType, str]):
        """Extract cloud-related info from the provided SCL layer in separate features and deletes the SCL feature.

        :param scl_feature: Name of feature in EOPatch holding the scene classification mask.
        """
        self.scl_feature = self.parse_feature(scl_feature)
        self.scl_cloud_feature = (FeatureType.MASK, "SCL_CLOUD")
        self.scl_cloud_shadow_feature = (FeatureType.MASK, "SCL_CLOUD_SHADOW")
        self.scl_cirrus_feature = (FeatureType.MASK, "SCL_CIRRUS")

    def execute(self, eopatch: EOPatch) -> EOPatch:
        scl = eopatch[self.scl_feature]
        eopatch[self.scl_cloud_feature] = ((scl == 8) | (scl == 9)).astype(np.uint8)
        eopatch[self.scl_cloud_shadow_feature] = (scl == 3).astype(np.uint8)
        eopatch[self.scl_cirrus_feature] = (scl == 10).astype(np.uint8)
        del eopatch[self.scl_feature]

        return eopatch


class AddPANBandTask(EOTask):
    def __init__(
        self,
        input_feature: Tuple[FeatureType, str],
        output_feature: Tuple[FeatureType, str],
    ):
        """Calculate pan-chromatic band as weighted average of other bands.

        :param input_feature: Input feature holding the Sentinel-2 bands used to calculate the pan-chromatic band.
        :param output_feature: Output feature holding Sentinel-2 bands and pan-chromatic band.
            The pan-chromatic band is inserted at index 3.
        """
        self.input_feature = self.parse_feature(
            input_feature, allowed_feature_types=[FeatureType.DATA]
        )
        self.output_feature = self.parse_feature(
            output_feature, allowed_feature_types=[FeatureType.DATA]
        )

    def execute(self, eopatch: EOPatch) -> EOPatch:
        bands = eopatch[self.input_feature]
        bands_names = eopatch.meta_info.get("bands_names", S2_BANDS)

        # Create a mapping from band name to its weight based on standard S2_BANDS
        weight_map = {band: weight for band, weight in zip(S2_BANDS, PAN_WEIGHTS)}

        # Get weights for the actual bands present in the input. 
        # Missing bands or bands not in S2_BANDS get 0.0 weight.
        current_weights = np.array([weight_map.get(band, 0.0) for band in bands_names])

        # Ensure current_weights matches the number of bands in the input feature
        if len(current_weights) != bands.shape[-1]:
            # If mismatch, pad with zeros or truncate (this shouldn't happen if bands_names is correct)
            if len(current_weights) < bands.shape[-1]:
                current_weights = np.pad(current_weights, (0, bands.shape[-1] - len(current_weights)))
            else:
                current_weights = current_weights[: bands.shape[-1]]

        weight_sum = np.sum(current_weights)
        if weight_sum > 0:
            pan_band = np.sum(bands * (current_weights / weight_sum), axis=-1)
        else:
            # Fallback if no weights match: simple average
            pan_band = np.mean(bands, axis=-1)

        pan_index = S2_PAN_BANDS.index("PAN")
        new_bands = np.insert(bands, pan_index, pan_band, axis=-1)

        eopatch[self.output_feature] = new_bands

        # Update bands_names in meta_info to include the new PAN band
        new_bands_names = list(bands_names)
        new_bands_names.insert(pan_index, "PAN")
        eopatch.meta_info["bands_names"] = new_bands_names

        return eopatch


class AddMetadataTask(EOTask):
    def __init__(self, config: Optional[SHConfig] = None):
        """Download Sentinel-2 metadata necessary to compute radiances

        :param config: Optional Sentinel Hub configuration file.
        """
        self.config = config

    def execute(self, eopatch: EOPatch, metadata: dict) -> EOPatch:
        """Add metadata (Earth-Sun distance, solar irradiances, sun zenith angles) to EOPatch.
        
        Args:
            eopatch: EOPatch to add metadata to
            metadata: Optional dict with metadata. If provided, should contain:
                    - earth_sun_dist: float
                    - solar_irradiances: dict with band names as keys
                    - sun_zenith_angles: 2D numpy array
                    
        Returns:
            EOPatch with added scalar and data features
        """
        # Add bands_names to meta_info
        eopatch.meta_info["bands_names"] = S2_BANDS
        # Initialize scalar fields
        dim = len(eopatch.timestamp) if eopatch.timestamp else 1
        eopatch.scalar["earth_sun_dist"] = np.ones((dim, 1))
        for s2_band in S2_BANDS:
            eopatch.scalar[f"sol_irr_{s2_band}"] = np.ones((dim, 1))


        return self._process_json_metadata(eopatch, metadata, dim)

    def _process_json_metadata(self, eopatch: EOPatch, metadata: dict, dim: int) -> EOPatch:
        """Process metadata from JSON file.
        
        Args:
            eopatch: EOPatch to update
            metadata: Dict with earth_sun_dist, solar_irradiances, sun_zenith_angles
            dim: Time dimension size
            
        Returns:
            Updated EOPatch
        """
        try:
            # Extract Earth-Sun distance
            if "earth_sun_dist" in metadata:
                earth_sun_dist = metadata["earth_sun_dist"]
                eopatch.scalar["earth_sun_dist"][:] = earth_sun_dist
            
            # Extract solar irradiances
            if "solar_irradiances" in metadata:
                solar_irradiances = metadata["solar_irradiances"]
                for s2_band in S2_BANDS:
                    # Try both B02 and B2 formats
                    irr_value = None
                    if s2_band in solar_irradiances:
                        irr_value = solar_irradiances[s2_band]
                    elif s2_band[1:] in solar_irradiances:  # Try without leading 'B'
                        irr_value = solar_irradiances[s2_band[1:]]
                    
                    if irr_value is not None:
                        eopatch.scalar[f"sol_irr_{s2_band}"][:] = irr_value
            
            # Extract and add sun zenith angles
            if "sun_zenith_angles" in metadata:
                sun_zenith_angles = np.array(metadata["sun_zenith_angles"], dtype=np.float32)
                
                # Get target shape from existing band data
                target_h, target_w = None, None
                for feature in eopatch.data.keys():
                    data_shape = eopatch[FeatureType.DATA, feature].shape
                    if len(data_shape) >= 3:
                        target_h, target_w = data_shape[1:3]
                        break
                
                if target_h is not None and target_w is not None:
                    # Resize sun zenith angles to match band resolution if needed
                    if sun_zenith_angles.shape != (target_h, target_w):
                        sun_zenith_angles = cv2.resize(
                            sun_zenith_angles, 
                            (target_w, target_h), 
                            interpolation=cv2.INTER_LINEAR
                        )
                    
                    # Add to eopatch with proper 4D shape (time, height, width, channels)
                    eopatch[FeatureType.DATA, "sunZenithAngles"] = sun_zenith_angles[np.newaxis, :, :, np.newaxis]
            
            return eopatch
            
        except Exception as e:
            print(f"Warning: Error processing JSON metadata: {e}")
            return eopatch

def get_shifts_l1a() -> List[Tuple[float, float]]:
    """Compute random shifts for L1A level"""
    rng = np.random.default_rng() # Initialize local RNG

    mis_amplitude = rng.normal(
        L1A_RAND_MEAN, L1A_RAND_STD, size=(len(S2_PAN_BANDS),)
    ) + np.array(L1A_RELATIVE_SHIFTS)
    mis_angle = rng.uniform(low=0, high=2 * np.pi, size=(len(S2_PAN_BANDS),))

    shifts = (mis_amplitude * (np.cos(mis_angle), np.sin(mis_angle))).T

    shifts[0, :] = np.array([0.0, 0.0])
    shifts = np.flip(np.cumsum(shifts, axis=0), axis=0)

    return [tuple(s) for s in shifts]


def get_shifts_l1b(rand_std: int) -> List[Tuple[float, float]]:
    """Compute random shifts for L1B level"""
    rng = np.random.default_rng() # Initialize local RNG

    mis_amplitude = rng.normal(0, rand_std, size=(len(S2_PAN_BANDS),))
    mis_angle = rng.uniform(low=0, high=2 * np.pi, size=(len(S2_PAN_BANDS),))

    shifts = (mis_amplitude * (np.cos(mis_angle), np.sin(mis_angle))).T
    shifts[2, :] = np.array([0.0, 0.0])

    return [tuple(s) for s in shifts.tolist()]


class BandMisalignmentTask(EOTask):
    def __init__(
        self,
        input_feature: Tuple[FeatureType, str],
        output_feature: Tuple[FeatureType, str],
        processing_level: ProcessingLevels,
        std_sea: int = 6,
        std_land: int = 1,
        interpolation_method: int = cv2.INTER_LINEAR,
    ):
        """Task for simulating L1A or L1B band misalignment

        :param input_feature: Input feature holding the bands to misalign.
        :param output_feature: Output feature with misaligned bands according to processing level.
        :param processing_level: Processing level that defines which band misalignment method is applied.
        :param std_sea: Standard deviation for AOIs over sea. Defaults to 6.
        :param interpolation_method: Interpolation method used in misalignment. Defaults to INTER_LINEAR.
        """
        self.input_feature = self.parse_feature(input_feature)
        self.output_feature = self.parse_feature(output_feature)
        self.processing_level = processing_level
        self.std_sea = std_sea
        self.std_land = std_land
        self.interpolation_method = interpolation_method

    def execute(self, eopatch: EOPatch) -> EOPatch:
        eopatch[self.output_feature] = eopatch[self.input_feature].copy()
        shift_dict = {}

        patch_geom = eopatch.bbox.transform(WORLD_GDF.crs.to_epsg()).geometry
        is_in_water = not WORLD_GDF.intersects(patch_geom).any()
        rand_std = self.std_sea if is_in_water else self.std_land

        for ts_idx, eop_ts in enumerate(eopatch[self.input_feature]):
            warp_matrix = np.eye(3)[:2, :]

            if self.processing_level.value == ProcessingLevels.L1A.value:
                shift_vectors = get_shifts_l1a()
            else:
                shift_vectors = get_shifts_l1b(rand_std)

            shift_dict[ts_idx] = shift_vectors

            bands_shifted = []

            for b_idx in range(eop_ts.shape[-1]):
                eop_ts_band = eop_ts[..., b_idx]
                warp_matrix[:, 2] = shift_vectors[b_idx]

                band = warpAffine(
                    src=eop_ts_band,
                    M=warp_matrix,
                    dsize=(eop_ts_band.shape[1], eop_ts_band.shape[0]),  # (width, height) format for OpenCV
                    flags=self.interpolation_method,
                    borderMode=cv2.BORDER_REFLECT_101,
                )
                bands_shifted.append(band)

            eopatch[self.output_feature][ts_idx] = np.moveaxis(
                np.array(bands_shifted), 0, -1
            )

        eopatch.meta_info["Shifts"] = shift_dict
        return eopatch


class CropTask(EOTask):
    def __init__(self, features_to_crop: List[Tuple[FeatureType, str]]):
        """Remove pixels at the boundary that contain misalignment artefacts

        :param features_to_crop: List o features that will be cropped in-place.
        """
        self.features_to_crop = self.parse_features(features_to_crop)

    def execute(self, eopatch: EOPatch) -> EOPatch:
        bbox_size = eopatch.bbox.upper_right[0] - eopatch.bbox.lower_left[0]

        if bbox_size < BBOX_SIZE_CROPPED:
            return eopatch

        crop_h, crop_w = CROP_SIZE
        for feature in self.features_to_crop:
            eopatch[feature] = eopatch[feature][:, crop_h:-crop_h, crop_w:-crop_w, :]

        eopatch.bbox = eopatch.bbox.buffer(
            (-crop_h * PHISAT2_RESOLUTION, -crop_w * PHISAT2_RESOLUTION),
            relative=False,
        )

        return eopatch


class CalculateRadianceTask(EOTask):
    def __init__(
        self,
        input_feature: Tuple[FeatureType, str],
        output_feature: Tuple[FeatureType, str],
    ):
        """Calculate radiances from reflectances using solar irradiance and distance Earth-Sun.

        :param input_feature: Input feature holding reflectance values. Expected 7 bands.
        :param output_feature: Output feature with radiance values.
        """
        self.input_feature = self.parse_feature(input_feature)
        self.output_feature = self.parse_feature(output_feature)

    def execute(self, eopatch: EOPatch) -> EOPatch:
        assert "bands_names" in eopatch.meta_info, "bands_names not found in eopatch.meta_info"
        bands_names = eopatch.meta_info["bands_names"]
        
        assert all(
            [
                isinstance(eopatch.scalar[f"sol_irr_{band}"], np.ndarray)
                for band in bands_names
                if band != "PAN"
            ]
        )
        assert isinstance(eopatch.scalar["earth_sun_dist"], np.ndarray)

        assert isinstance (eopatch.data["sunZenithAngles"], np.ndarray)
            
        factor = (
            np.cos(np.radians(eopatch.data["sunZenithAngles"]))
            * eopatch.scalar["earth_sun_dist"][:, np.newaxis, np.newaxis, :]
            / np.pi
        )
        
        # We don't have the irradiance for the PAN band, so we compute it as weighted sum
        if "PAN" in bands_names and "sol_irr_PAN" not in eopatch.scalar:
            sol_irr_pan = 0.0
            weight_dict = dict(zip(S2_BANDS, PAN_WEIGHTS))
            for band in bands_names:
                if band in weight_dict:
                    sol_irr_pan += eopatch.scalar[f"sol_irr_{band}"] * weight_dict[band]
            eopatch.scalar["sol_irr_PAN"] = sol_irr_pan

        solar_irradiances = np.concatenate(
            [eopatch.scalar[f"sol_irr_{band}"][:, :, np.newaxis] for band in bands_names],
            axis=-1,
        )

        radiances = (
            eopatch[self.input_feature]
            * factor
            * solar_irradiances[:, :, np.newaxis, :]
        )
        eopatch[self.output_feature] = radiances
        return eopatch


class CalculateReflectanceTask(EOTask):
    def __init__(
        self,
        input_feature: Tuple[FeatureType, str],
        output_feature: Tuple[FeatureType, str],
        processing_level: ProcessingLevels,
    ):
        """Calculate reflectances from radiances using solar irradiance and distance
        Earth-Sun if L1C level is requested. Otherwise radiance are returned.

        :param input_feature: Input feature holding radiance values. Expected 8 bands.
        :param output_feature: Output feature with radiances (for L1A and L1B) or reflectance (for L1C) values.
        :param processing_level: Processin g level desired. If L1A or L1B no conversion to reflectances is applied.
        """
        self.input_feature = self.parse_feature(input_feature)
        self.output_feature = self.parse_feature(output_feature)
        self.processing_level = processing_level

    def execute(self, eopatch: EOPatch) -> EOPatch:
        assert "bands_names" in eopatch.meta_info, "bands_names not found in eopatch.meta_info"
        bands_names = eopatch.meta_info["bands_names"]
        
        if self.processing_level.value == ProcessingLevels.L1C.value:
            bands_names = eopatch.meta_info.get("bands_names", bands_names)
            assert all(
                [
                    isinstance(eopatch.scalar[f"sol_irr_{band}"], np.ndarray)
                    for band in bands_names
                    if band != "PAN"
                ]
            )
            assert isinstance(eopatch.scalar["earth_sun_dist"], np.ndarray)

            # We don't have the irradiance for the PAN band, so we compute it as weighted sum
            sol_irr_pan = 0.0
            weight_dict = dict(zip(S2_BANDS, PAN_WEIGHTS))
            for band in bands_names:
                if band in weight_dict:
                    sol_irr_pan += eopatch.scalar[f"sol_irr_{band}"] * weight_dict[band]

            eopatch.scalar["sol_irr_PAN"] = sol_irr_pan

            factor = (
                np.cos(np.radians(eopatch.data["sunZenithAngles_RES"]))
                * eopatch.scalar["earth_sun_dist"][:, np.newaxis, np.newaxis, :]
                / np.pi
            )
            solar_irradiances = np.concatenate(
                [
                    eopatch.scalar[f"sol_irr_{band}"][:, :, np.newaxis]
                    for band in bands_names
                ],
                axis=-1,
            )

            reflectances = eopatch[self.input_feature] / (
                factor * solar_irradiances[:, :, np.newaxis, :]
            )
            eopatch[self.output_feature] = reflectances
        else:
            eopatch[self.output_feature] = eopatch[self.input_feature]
        return eopatch


class PhisatCalculationTask(EOTask):
    def __init__(
        self,
        input_feature: Tuple[FeatureType, str],
        output_feature: Tuple[FeatureType, str],
        executable: str,
        calculation: str,
    ):
        """Wapper task to simulate SNR and PSF noise using an external executable binary.

        :param input_feature: Input feature holding radiance values.
        :param output_feature: Output feature with SNR/PSF simulated values.
        :param executable: Path to executable binary.
        :param calculation: Which simulation to execute, i.e. "SNR" or "PSF"
        """
        self.input_feature = input_feature
        self.output_feature = output_feature
        self.executable = executable
        self.calculation = calculation

    def execute(self, eopatch: EOPatch) -> EOPatch:
        with tempfile.TemporaryDirectory(
            prefix="phisat2_calc", suffix=self.calculation
        ) as temp_dir:
            input_npy = os.path.join(temp_dir, "input.npy")
            output_npy = os.path.join(temp_dir, "output.npy")

            data = eopatch[self.input_feature]
            bands_names = eopatch.meta_info.get("bands_names", [])

            # Re-align bands to match S2_PAN_BANDS (8 bands) expected by the executable
            # S2_PAN_BANDS = ["B02", "B03", "B04", "PAN", "B08", "B05", "B06", "B07"]
            aligned_data = np.zeros((*data.shape[:-1], len(S2_PAN_BANDS)), dtype=np.float32)
            
            # Map existing bands to their positions in S2_PAN_BANDS
            for i, band in enumerate(S2_PAN_BANDS):
                if band in bands_names:
                    idx = bands_names.index(band)
                    aligned_data[..., i] = data[..., idx]
                elif band == "B08" and "B8A" in bands_names:
                    # Fallback for B8A if B08 is missing
                    idx = bands_names.index("B8A")
                    aligned_data[..., i] = data[..., idx]
                # Other missing bands will stay as 0.0

            print(f"[{self.calculation}] Aligned input feature {self.input_feature} from {data.shape} to {aligned_data.shape}")

            # temporary write input feature to numpy
            np.save(input_npy, aligned_data)

            # run exec
            try:
                result = subprocess.run(
                    f"{self.executable} {self.calculation} {input_npy} {output_npy}".split(
                        " "
                    ),
                    capture_output=True,
                    text=True,
                    check=True,
                )
                if result.stdout:
                    print(f"[{self.calculation}] stdout: {result.stdout}")
            except subprocess.CalledProcessError as e:
                print(f"[{self.calculation}] Command failed with exit code {e.returncode}")
                print(f"[{self.calculation}] stderr: {e.stderr}")
                print(f"[{self.calculation}] stdout: {e.stdout}")
                raise

            # read temp output
            output_data = np.load(output_npy)
            eopatch[self.output_feature] = output_data
            
            # Update meta_info with the new band names (the 8 bands expected/produced)
            eopatch.meta_info["bands_names"] = S2_PAN_BANDS

        # return eopatch with new feature
        return eopatch


class GriddingTask(EOTask):
    def __init__(
        self,
        raster_feature: Tuple[FeatureType, str],
        data_stack_feature: Tuple[FeatureType, str],
        grid_feature: Tuple[FeatureType, str],
        size: int,
        overlap: float,
        resolution: float,
        time_index: int = 0,
    ):
        """Split the AOI into smaller image chips to create an AI-ready dataset.

        :param raster_feature: A data feature to crop
        :param grid_feature: A vector feature where cropped grid is saved at.
        :param data_stack_feature: A data feature where output stack of data is stored.
        :param size: A size of of images to crop out of input image.
        :param overlap: Overlap between sub-images extracted.
        :param resolution: Resolution on which task is running.
        """
        self.raster_feature = raster_feature
        self.grid_feature = grid_feature
        self.data_stack_feature = data_stack_feature
        self.size = size
        self.overlap = overlap
        self.resolution = resolution
        self.time_index = time_index

    def _grid_data(self, data: np.array) -> Tuple[List[np.array], Dict]:
        height, width, bands = data.shape
        stride = int(self.size * (1 - self.overlap))

        gridded_data = []
        stats = defaultdict(list)
        for x in range(0, width, stride):
            for y in range(0, height, stride):
                x2, y2 = min(x + self.size, width), min(y + self.size, height)
                x1, y1 = max(0, x2 - self.size), max(0, y2 - self.size)

                if x1 == x2 or y1 == y2:
                    continue

                data_slice = data[y1:y2, x1:x2, ...]
                gridded_data.append(data_slice)

                polygon = shapely.geometry.box(x1, y1, x2, y2)
                stats["pixel_geometry"].append(polygon)

        gridded_data = (
            np.stack(gridded_data, axis=0)
            if gridded_data
            else np.zeros((0, self.size, self.size, bands), dtype=data.dtype)
        )
        return gridded_data, stats

    def execute(self, eopatch: EOPatch) -> EOPatch:
        data = eopatch[self.raster_feature][self.time_index]
        gridded_data, stats = self._grid_data(data)

        eopatch[self.data_stack_feature] = gridded_data

        if self.grid_feature:
            crop_grid_gdf = self.calculate_crop_grid(eopatch, stats)
            eopatch[self.grid_feature] = crop_grid_gdf

        return eopatch

    def calculate_crop_grid(self, eopatch: EOPatch, stats: Dict) -> gpd.GeoDataFrame:
        transform = eopatch.bbox.get_transform_vector(self.resolution, self.resolution)

        def pixel_to_utm_transformer(column, row):
            return pixel_to_utm(row, column, transform=transform)

        utm_polygons = [
            shapely.ops.transform(pixel_to_utm_transformer, polygon)
            for polygon in stats["pixel_geometry"]
        ]
        crop_grid_gdf = gpd.GeoDataFrame(
            stats, geometry=utm_polygons, crs=eopatch.bbox.crs.pyproj_crs()
        )
        return crop_grid_gdf


def get_extent(eopatch: EOPatch) -> Tuple[float, float, float, float]:
    """Calculate the extent (bounds) of the patch.
    :param eopatch: EOPatch for which the extent is calculated.
    :return: The list of EOPatch bounds (min_x, max_x, min_y, max_y)
    """
    return (
        eopatch.bbox.min_x,
        eopatch.bbox.max_x,
        eopatch.bbox.min_y,
        eopatch.bbox.max_y,
    )


class ExportGridToTiff(EOTask):
    def __init__(
        self,
        data_stack_feature: Tuple[FeatureType, str],
        grid_feature: Tuple[FeatureType, str],
        out_folder: str,
        time_index: int = 0,
    ):
        """Export the image chips as GeoTiff files.

        :param grid_feature: A vector feature where cropped grid is saved at.
        :param data_stack_feature: A data feature where output stack of data is stored.
        :param out_folder: Path to folder where tiff will be saved.
        :time_index: Timestamp index in EOPatch from which the grid was created.
        """
        self.grid_feature = grid_feature
        self.data_stack_feature = data_stack_feature
        self.out_folder = out_folder
        self.time_index = time_index

        self.export_task = ExportToTiffTask(
            feature=(FeatureType.DATA, "TEMP_CELL"),
            folder=self.out_folder,
            date_indices=[self.time_index],
        )

    def execute(self, eopatch: EOPatch) -> EOPatch:
        gridded_data = eopatch[self.data_stack_feature]
        gdf = eopatch[self.grid_feature]

        for n_row, row in gdf.iterrows():
            temp_eop = EOPatch(
                data={"TEMP_CELL": gridded_data[[n_row]]},
                bbox=BBox(row.geometry, crs=eopatch.bbox.crs),
                timestamp=[eopatch.timestamp[self.time_index]],
            )

            timestamp_str = eopatch.timestamp[self.time_index].strftime(
                "%Y-%m-%dT%H-%M-%S"
            )
            bbox_str = f"{int(eopatch.bbox.middle[0])}-{int(eopatch.bbox.middle[1])}"
            utm_crs_str = f"{eopatch.bbox.crs.epsg}"
            self.export_task(
                temp_eop,
                filename=f"{bbox_str}_{utm_crs_str}_{self.grid_feature[1]}_{timestamp_str}_{n_row:03d}.tiff",
            )
            del temp_eop

        return eopatch


class LoadS2FileTask(EOTask):
    """Load S2 TIFF file and create EOPatch with metadata."""

    def execute(self, *, s2_tiff_path: str, metadata: dict | None, acquisition_date: datetime, **kwargs) -> EOPatch:
        """Load S2 TIFF and initialize EOPatch."""
        s2_path = Path(s2_tiff_path)
        
        with rasterio.open(s2_path) as src:
            s2_data = src.read()
            try:
                bbox = _create_bbox_from_rasterio(src)
            except Exception as e:
                print(f"Warning: Could not extract bbox: {e}")
                bbox = None
                
        # Select "B02", "B03", "B04", "B08", "B05", "B06", "B07"
        band_indices = [1, 2, 3, 7, 4, 5, 6]
        s2_data = s2_data[band_indices, :, :]
        s2_data = np.transpose(s2_data, (1, 2, 0))
        eopatch = EOPatch(bbox=bbox, timestamps=[acquisition_date])
        eopatch.meta_info["bands_names"] = S2_BANDS
        eopatch[FeatureType.DATA, "S2_BANDS"] = s2_data[np.newaxis, :, :, :]

        # Add metadata to EOPatch
        if metadata is not None:
            try:
                add_meta_task = AddMetadataTask()
                eopatch = add_meta_task.execute(eopatch, metadata)
            except Exception as e:
                print(f"Warning: Failed to add metadata from file: {e}")
        else:
            try:
                add_meta_task = FetchMetadataTask()
                eopatch = add_meta_task.execute(eopatch, bands_names=S2_BANDS)
            except Exception as e:
                print(f"Warning: Failed to fetch metadata from API: {e}")

        return eopatch

class LoadS2DatasetSampleTask(EOTask):
    """Load S2 data from a dataset sample and fetch metadata using FetchMetadataTask.
    
    This task creates an EOPatch directly from image data provided through
    execution kwargs, uses lat/lon coordinates to fetch S2 acquisition metadata
    via the Copernicus API, and enables direct use of dataset samples via __getitem__.
    """

    def __init__(self, config: Optional[SHConfig] = None):
        """Initialize with optional SentinelHub config.
        
        Args:
            config: Optional SHConfig for metadata fetching.
        """
        self.config = config
        self.fetch_metadata_task = FetchMetadataTask(config=config)

    def execute(
        self,
        *,
        image: np.ndarray,
        bands_names: List[str],
        metadata: dict | None,
        acquisition_date: datetime,
        location_coords: np.ndarray | None = None,
        **kwargs
    ) -> EOPatch:
        """Create EOPatch from dataset sample and fetch S2 metadata via lat/lon point.
        
        Args:
            image: Image array from dataset sample (H, W, C or C, H, W)
            bands_names: List of band names
            metadata: Optional metadata dictionary
            acquisition_date: Acquisition date for the EOPatch timestamp
            location_coords: Optional location coordinates as torch.Tensor [lat, lon]
            **kwargs: Additional arguments (ignored)
            
        Returns:
            EOPatch with S2_BANDS feature and fetched metadata
        """
        # Convert torch.Tensor to numpy array if needed
        if hasattr(image, 'cpu'):  # Check if it's a torch.Tensor
            image = image.cpu().detach().numpy()
        elif not isinstance(image, np.ndarray):
            image = np.asarray(image)
        
        # Handle channel ordering
        if image.ndim == 3:
            image = np.transpose(image, (1, 2, 0))
            # Add image data as S2_BANDS (add time dimension: (T, H, W, C))
            image = image[np.newaxis, :, :, :]  # Add time dimension
        
        # Create EOPatch with bbox from lat/lon point
        eopatch = EOPatch(bbox=None, timestamps=[acquisition_date])
        eopatch.meta_info["bands_names"] = bands_names
                
        eopatch[FeatureType.DATA, "S2_BANDS"] = image
        
        # Fetch metadata using lat/lon if location_coords are provided
        if location_coords is not None:
            try:
                # Convert location_coords to lat/lon values
                if hasattr(location_coords, 'item'):
                    # torch.Tensor case
                    lat = float(location_coords[0].item())
                    lon = float(location_coords[1].item())
                else:
                    # numpy array case
                    lat = float(location_coords[0])
                    lon = float(location_coords[1])
                
                # Calculate buffer in degrees based on image size and S2_RESOLUTION (10m)
                # 1 degree latitude is approx 111320 meters
                # 1 degree longitude is approx 111320 * cos(lat) meters
                h, w = image.shape[1:3]
                
                # To ensure square pixels in degrees (which is required by some TIFF readers and simplified processing)
                # we use the same degree-per-meter conversion for both dimensions based on the latitude.
                # However, usually for geo-referenced imagery in WGS84, we want constant physical size.
                # If the goal is square pixels in degrees, we should use a single conversion factor.
                # If the goal is square pixels in meters (4.75m x 4.75m), the degree sizes must differ.
                
                # Use a single degree-to-meter conversion factor (at the equator/approximate)
                # to ensure that the degree-based BBox is perfectly square if the pixel grid is square.
                # This ensures PixelSize_X == PixelSize_Y in degree units.
                deg_per_meter = 1.0 / 111320.0
                buffer_lat = (h * S2_RESOLUTION / 2.0) * deg_per_meter
                buffer_lon = (w * S2_RESOLUTION / 2.0) * deg_per_meter
                
                bbox = BBox(
                    [lon - buffer_lon, lat - buffer_lat, lon + buffer_lon, lat + buffer_lat],
                    crs=CRS.WGS84
                )
                
                # Update eopatch with bbox for metadata fetching
                eopatch.bbox = bbox
                
                # Fetch metadata from Copernicus API using FetchMetadataTask
                eopatch = self.fetch_metadata_task.execute(eopatch, bands_names=bands_names)
                
            except Exception as e:
                # Log warning but continue without metadata
                print(f"Warning: Failed to fetch S2 metadata for location {location_coords}: {e}")
    
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
        
        new_size_cv2 = (new_size[0], new_size[1]) # cv2 uses (width, height)

        for feat_name in features_to_resize[FeatureType.DATA]:
            data = eopatch[FeatureType.DATA, feat_name]
            
            resampled_list = []
            for t in range(data.shape[0]):
                resampled_img = spatially_resize_image(
                    data[t],
                    new_size=new_size_cv2,
                    resize_method=ResizeMethod.NEAREST,
                ).astype(np.float32)
                if resampled_img.ndim == 2:
                    resampled_img = resampled_img[..., np.newaxis].astype(np.float32)
                resampled_list.append(resampled_img)
            
            eopatch[FeatureType.DATA, f"{feat_name}_RES"] = np.array(resampled_list)

        return eopatch

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

class FetchMetadataTask(EOTask):
    def __init__(
        self,
        input_feature: Tuple[FeatureType, str] = (FeatureType.DATA, "S2_BANDS"),
        config: Optional[SHConfig] = None,
    ):
        """Download Sentinel-2 metadata necessary to compute radiances

        :param input_feature: Input feature to get the shape from.
        :param config: Optional Sentinel Hub configuration file.
        """
        self.input_feature = self.parse_feature(input_feature)
        self.config = config

    def execute(self, eopatch: EOPatch, bands_names: Optional[List[str]] = None, **kwargs) -> EOPatch:
        if bands_names is not None:
            eopatch.meta_info["bands_names"] = bands_names
        else:
            bands_names = eopatch.meta_info.get("bands_names")
        
        if bands_names is None:
            raise ValueError("bands_names not found in eopatch.meta_info and not provided to FetchMetadataTask")

        if not all([eopatch, eopatch.bbox, eopatch.timestamps]):
            raise ValueError(
                "FetchMetadataTask needs eopatch to have bbox and temporal data!"
            )

        # Query Copernicus catalogue for Sentinel-2 L2A products
        sensor = "SENTINEL-2"
        bbox = eopatch.bbox
        
        # Convert bbox to WGS84 polygon string (minx, miny, maxx, maxy)
        area = f"POLYGON(({bbox.min_x} {bbox.min_y},{bbox.min_x} {bbox.max_y},{bbox.max_x} {bbox.max_y},{bbox.max_x} {bbox.min_y},{bbox.min_x} {bbox.min_y}))"
        
        start_date = eopatch.timestamps[0].strftime("%Y-%m-%d")
        end_date = eopatch.timestamps[-1].strftime("%Y-%m-%d")
        prod_type = "S2MSI2A"

        # Build OData query for Copernicus catalogue
        query_url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=Collection/Name eq '"+sensor+"' \
            and OData.CSC.Intersects(area=geography'SRID=4326;"+area+"') \
            and ContentDate/Start gt "+start_date+"T00:00:00.000Z \
            and ContentDate/Start lt "+end_date+"T23:59:59.000Z \
            and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' \
            and att/OData.CSC.StringAttribute/Value eq '"+prod_type+"')"

        # Fetch all products from Copernicus catalogue
        json_out = requests.get(query_url).json()
        products_list = json_out.get('value', [])

        # Handle pagination
        next_link = json_out.get('@odata.nextLink', None)
        while next_link:
            json_next = requests.get(next_link).json()
            products_list.extend(json_next.get('value', []))
            next_link = json_next.get('@odata.nextLink', None)
            
        # Initialize scalar fields
        dim = len(eopatch.timestamps)
        eopatch.scalar["earth_sun_dist"] = np.ones((dim, 1))
            
        for s2_band in bands_names:
            eopatch.scalar[f"sol_irr_{s2_band}"] = np.ones((dim, 1))
        
        # Handle the case when no products are found in the Copernicus catalogue for the given area and time range
        if len(products_list) == 0:
            solar_irradiances = {
                "B01": 1895.0,   # Coastal aerosol
                "B02": 1941.0,   # Blue
                "B03": 1822.0,   # Green
                "B04": 1610.0,   # Red
                "B05": 1519.0,   # Vegetation Red Edge
                "B06": 1447.0,   # Vegetation Red Edge
                "B07": 1387.0,   # Vegetation Red Edge
                "B08": 1034.0,   # NIR
                "B8A": 955.0,   # Vegetation Red Edge
                "B11": 245.0,   # SWIR
                "B12": 85.0,    # SWIR
            }
            for s2_band in bands_names:
                if s2_band in solar_irradiances:
                    eopatch.scalar[f"sol_irr_{s2_band}"] = np.full(
                        (dim, 1), solar_irradiances[s2_band]
                    )

            eopatch.scalar["earth_sun_dist"] = np.full(
                (dim, 1), sun.earth_distance(eopatch.timestamps[0]).to(u.au).value
            )
            # Assume sun Zenith angle of 30 degrees for all pixels and timestamps
            sun_zenith_angles = np.full(
                (
                    eopatch[self.input_feature].shape[1],
                    eopatch[self.input_feature].shape[2],
                ),
                30.0,
                dtype=np.float32,
            )
            eopatch[FeatureType.DATA, "sunZenithAngles"] = sun_zenith_angles[
                np.newaxis, :, :, np.newaxis
            ]
            raise ValueError(
                "No products found in Copernicus catalogue for the given area and time range, using default metadata values."
            )
        else:
            print(f"Found {len(products_list)} products in Copernicus catalogue for the given area and time range.")
            # Extract metadata from each product
            for tile_idx, (tile, timestamp) in enumerate(zip(products_list, eopatch.timestamps)):
                metadata = self._fetch_metadata_from_copernicus(tile)

                # Extract Earth-Sun distance and solar irradiance from metadata
                earth_sun_dist, solar_irradiances = self._extract_irradiance_data(metadata, timestamp, bands_names)
                
                eopatch.scalar["earth_sun_dist"][tile_idx] = earth_sun_dist
                print(f"Earth-Sun distance for tile {tile_idx}: {earth_sun_dist} AU")

                # Populate solar irradiance for each band
                for s2_band in bands_names:
                    if s2_band in solar_irradiances:
                        eopatch.scalar[f"sol_irr_{s2_band}"][tile_idx] = solar_irradiances[s2_band]

            # Extract and add sun zenith angles from metadata
            for tile_idx, tile in enumerate(products_list):
                safe_path = tile.get("S3Path", "").rstrip("/")
                sun_zenith_angles = self._parse_sun_zenith_angles(safe_path)
                # Add sun zenith angles to the data feature on first occurrence
                # Get target shape from existing band data
                for feature in eopatch.data.keys():
                    if eopatch[FeatureType.DATA, feature].ndim == 4:
                        target_h, target_w = eopatch[FeatureType.DATA, feature].shape[1:3]
                        # Resize to match band resolution
                        sun_zenith_angles = cv2.resize(sun_zenith_angles, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                        break
                            
                    # Add to eopatch with proper 4D shape (time, height, width, channels)
                eopatch[FeatureType.DATA, "sunZenithAngles"] = sun_zenith_angles[np.newaxis, :, :, np.newaxis]
                break

        return eopatch

    @staticmethod
    def _fetch_metadata_from_copernicus(product: Dict) -> Dict:
        """Fetch metadata for a product using the complete S3 path from Copernicus API.
        
        :param product: Product dictionary from Copernicus API
        :return: Metadata dictionary with XML root and product info
        """
        # Get the complete S3 path from the product
        s3_path = product.get("S3Path", "")
        product_id = product.get("Id", "")
        
        if not s3_path:
            raise ValueError(f"No S3Path found for product {product_id}")
        
        local_safe_path = s3_path.rstrip("/")
        
        if not os.path.exists(local_safe_path):
            raise FileNotFoundError(
                f"SAFE product not found at {local_safe_path} for product {product_id}"
            )
        print(f"Metadata file found for product {product_id} at {local_safe_path}")
        # Construct the metadata file path
        metadata_file = os.path.join(local_safe_path, "MTD_MSIL2A.xml")
        if not os.path.exists(metadata_file):
            # Try L1C metadata file instead
            metadata_file = os.path.join(local_safe_path, "MTD_MSIL1C.xml")
        
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(
                f"Metadata file not found at {local_safe_path} (tried MTD_MSIL2A.xml and MTD_MSIL1C.xml)"
            )
        
        # Parse the XML file
        tree = ET.parse(metadata_file)
        root = tree.getroot()
        
        metadata_info = {
            "product_id": product_id,
            "safe_path": local_safe_path,
            "xml_root": root,
            "product": product
        }
        
        return metadata_info

    @staticmethod
    def _extract_irradiance_data(metadata: Dict, timestamp: datetime, bands_names: List[str]) -> Tuple[float, Dict[str, float]]:
        """Extract Earth-Sun distance and solar irradiance from Sentinel-2 metadata XML.
        
        :param metadata: Metadata dictionary containing XML root and product info
        :param timestamp: Timestamp of the product
        :param bands_names: List of band names
        :return: Tuple of (earth_sun_distance, solar_irradiances_dict)
        """
        xml_root = metadata.get("xml_root")
        if not xml_root:
            raise ValueError("No XML root found in metadata dictionary")
        
        # Calculate Earth-Sun distance using sunpy
        earth_sun_dist = sun.earth_distance(timestamp).to(u.au).value
        
        solar_irradiances = {}
        
        # Extract solar irradiance from Solar_Irradiance_List
        band_id_to_band = {
            0: "B01", 1: "B02", 2: "B03", 3: "B04", 4: "B05", 5: "B06", 6: "B07", 7: "B08",
            8: "B8A", 9: "B09", 10: "B10", 11: "B11", 12: "B12"
        }
        
        irradiance_list = xml_root.find(".//Solar_Irradiance_List")
        if irradiance_list is not None:
            for irradiance_elem in irradiance_list.findall("SOLAR_IRRADIANCE"):
                band_id = irradiance_elem.get("bandId")
                if band_id is not None:
                    try:
                        band_id_int = int(band_id)
                        if band_id_int in band_id_to_band and band_id_to_band[band_id_int] in bands_names:
                            irradiance_value = float(irradiance_elem.text)
                            solar_irradiances[band_id_to_band[band_id_int]] = irradiance_value
                    except (ValueError, TypeError):
                        pass
        
        # Use standard reference values if extraction failed
        if not solar_irradiances:
            print('Could not extract solar irradiance from metadata, using standard reference values for all bands.')
            solar_irradiances = {
                "B01": 1895.0,   # Coastal aerosol
                "B02": 1941.0,   # Blue
                "B03": 1822.0,   # Green
                "B04": 1610.0,   # Red
                "B05": 1519.0,   # Vegetation Red Edge
                "B06": 1447.0,   # Vegetation Red Edge
                "B07": 1387.0,   # Vegetation Red Edge
                "B08": 1034.0,   # NIR
                "B8A": 955.0,   # Vegetation Red Edge
                "B11": 245.0,   # SWIR
                "B12": 85.0,    # SWIR
            }
        
        return earth_sun_dist, solar_irradiances

    @staticmethod
    def _parse_sun_zenith_angles(safe_path: str) -> Optional[np.ndarray]:
        """Extract sun zenith angles from MTD_TL.xml file in SAFE product.
        
        :param safe_path: Path to the SAFE product directory
        :return: 2D array of sun zenith angles or None if extraction fails
        """
        try:
            # Find MTD_TL.xml in GRANULE subdirectory
            granule_dir = os.path.join(safe_path, "GRANULE")
            if not os.path.exists(granule_dir):
                print(f"GRANULE directory not found in {safe_path}")
                return None
            
            # Get the first tile directory (assuming there's one)
            tile_dirs = [d for d in os.listdir(granule_dir) if os.path.isdir(os.path.join(granule_dir, d))]
            if not tile_dirs:
                print(f"No tile directories found in {granule_dir}")
                return None
            
            mtd_tl_path = os.path.join(granule_dir, tile_dirs[0], "MTD_TL.xml")
            if not os.path.exists(mtd_tl_path):
                print(f"MTD_TL.xml not found at {mtd_tl_path}")
                return None
            
            # Parse XML
            tree = ET.parse(mtd_tl_path)
            root = tree.getroot()
            
            # Extract sun zenith angles from Sun_Angles_Grid
            zenith_grid = root.find(".//Sun_Angles_Grid/Zenith")
            if zenith_grid is None:
                print("Zenith element not found in Sun_Angles_Grid")
                return None
            
            values_list = zenith_grid.find("Values_List")
            if values_list is None:
                print("Values_List element not found in Zenith grid")
                return None
            
            values_rows = []
            for values_elem in values_list.findall("VALUES"):
                if values_elem.text:
                    row_values = [float(v) for v in values_elem.text.strip().split()]
                    values_rows.append(row_values)
            
            # Convert to numpy array
            sun_zenith_angles = np.array(values_rows)
            
            return sun_zenith_angles
            
        except Exception as e:
            print(f"Error parsing sun zenith angles: {str(e)}")
            return None