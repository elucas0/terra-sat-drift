"""Custom Albumentations transforms for Phisat-2 satellite simulation on Sentinel-2 data.

Implements methods from phisat2_utils for on-the-fly Sentinel-2 to Φ-sat-2 simulation:
- Band misalignment (L1A/L1B processing levels)
- Panchromatic band creation with proper weighting
- PSF kernel convolution (via external executable or pure Python)
- SNR noise simulation (via external executable or pure Python)
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple, cast
from scipy.ndimage import convolve

import albumentations as A
import cv2
import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.pytorch import ToTensorV2

from phisat2_constants import (
    L1A_RELATIVE_SHIFTS, 
    L1A_RAND_MEAN, 
    L1A_RAND_STD, 
    PAN_WEIGHTS, 
    S2_BANDS,
    S2_PAN_BANDS
)

from sunpy.coordinates import sun
from astropy import units as u
import xml.etree.ElementTree as ET

class ProcessingLevels:
    """Processing levels for band misalignment simulation."""
    L1A = "L1A"
    L1B = "L1B"
    
@property
def targets_as_params(self):
    return self._targets_as_params

def _get_shifts_l1a() -> List[Tuple[float, float]]:
    """Compute random shifts for L1A level processing.
    
    Returns:
        List of (shift_x, shift_y) tuples for each band
    """
    mis_amplitude = np.random.normal(
        L1A_RAND_MEAN, L1A_RAND_STD, size=(len(S2_PAN_BANDS),)
    ) + np.array(L1A_RELATIVE_SHIFTS)
    mis_angle = np.random.uniform(low=0, high=2 * np.pi, size=(len(S2_PAN_BANDS),))

    shifts = (mis_amplitude * (np.cos(mis_angle), np.sin(mis_angle))).T

    shifts[0, :] = np.array([0.0, 0.0])
    shifts = np.flip(np.cumsum(shifts, axis=0), axis=0)

    return [tuple(s) for s in shifts]


def _get_shifts_l1b(rand_std: int = 1) -> List[Tuple[float, float]]:
    """Compute random shifts for L1B level processing.
    
    Args:
        rand_std: Standard deviation for random shift amplitude
        
    Returns:
        List of (shift_x, shift_y) tuples for each band
    """
    mis_amplitude = np.random.normal(0, rand_std, size=(len(S2_PAN_BANDS),))
    mis_angle = np.random.uniform(low=0, high=2 * np.pi, size=(len(S2_PAN_BANDS),))

    shifts = (mis_amplitude * (np.cos(mis_angle), np.sin(mis_angle))).T
    shifts[2, :] = np.array([0.0, 0.0])

    return [tuple(s) for s in shifts.tolist()]


class BandMisalignmentTransform(ImageOnlyTransform):
    """Simulate band misalignment artifacts in multispectral imagery.
    
    Applies different sub-pixel shifts to different bands using warpAffine
    to simulate misalignment that occurs in real satellite systems.
    Uses methods from phisat2_utils BandMisalignmentTask.
    """

    def __init__(
        self,
        processing_level: str = ProcessingLevels.L1A,
        std_sea: int = 6,
        interpolation_method: int = cv2.INTER_LINEAR,
        always_apply: bool = False,
    ):
        """Initialize band misalignment transform.
        
        Args:
            processing_level: "L1A" or "L1B" processing level
            std_sea: Standard deviation for shifts over sea areas
            interpolation_method: OpenCV interpolation method (default: LINEAR)
            always_apply: Whether to always apply the transform
            p: Probability of applying the transform
        """
        super().__init__(always_apply)
        self.processing_level = processing_level
        self.std_sea = std_sea
        self.interpolation_method = interpolation_method

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        """Apply band misalignment using warpAffine."""
        if img.ndim != 3:
            return img
        
        # Expects (H, W, C) format - transpose to (C, H, W) for processing
        if img.shape[2] == len(S2_PAN_BANDS):  # Has 8 bands including PAN
            img_band_first = np.transpose(img, (2, 0, 1))
        elif img.shape[2] == 7:  # Regular 7 bands
            img_band_first = np.transpose(img, (2, 0, 1))
        else:
            return img
        
        # Get shift vectors based on processing level
        if self.processing_level == ProcessingLevels.L1A:
            shift_vectors = _get_shifts_l1a()
        else:
            shift_vectors = _get_shifts_l1b(self.std_sea)
        
        result = []
        for b_idx, (shift_x, shift_y) in enumerate(shift_vectors[:img_band_first.shape[0]]):
            band = img_band_first[b_idx]
            
            if abs(shift_x) < 1e-6 and abs(shift_y) < 1e-6:
                # No shift needed
                result.append(band)
            else:
                # Apply warpAffine transformation
                warp_matrix = np.array([
                    [1.0, 0.0, shift_x],
                    [0.0, 1.0, shift_y]
                ], dtype=np.float32)
                
                h, w = band.shape
                warped = cv2.warpAffine(
                    band.astype(np.float32),
                    warp_matrix,
                    (w, h),
                    flags=self.interpolation_method,
                    borderMode=cv2.BORDER_REFLECT
                )
                result.append(warped.astype(img.dtype))
        
        # Transpose back to (H, W, C)
        result_img = np.transpose(np.array(result), (1, 2, 0))
        return result_img

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("processing_level", "std_sea", "interpolation_method")




class PanBandTransform(ImageOnlyTransform):
    """Create panchromatic band as weighted average of Sentinel-2 bands.
    
    Implements the AddPANBandTask from phisat2_utils - computes a Pan band
    from a weighted sum of the 7 S2 bands and inserts it at the correct position.
    
    Expects input to have 7 bands (B02, B03, B04, B08, B05, B06, B07).
    """

    def __init__(
        self,
        pan_weights: list[float] | None = None,
        always_apply: bool = False,
    ):
        """Initialize PAN band creation transform.
        
        Args:
            pan_weights: Weights for each band in computing PAN. 
                        Defaults to [0.216, 0.287, 0.257, 0.0, 0.123, 0.117, 0.0]
            always_apply: Whether to always apply
            p: Probability of applying
        """
        super().__init__(always_apply)
        self.pan_weights = pan_weights or PAN_WEIGHTS

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        """Create PAN band from weighted average."""
        # Handle both (H, W, C) and (C, H, W) formats
        if img.ndim == 3:
            # If last dimension is 7 (bands), assume (H, W, C)
            if img.shape[2] == 7:
                bands = img
                # Compute PAN as weighted average
                pan_band = np.sum(bands * np.array(self.pan_weights) / sum(self.pan_weights), axis=2)
                # Insert PAN at position 3 to match S2_PAN_BANDS order
                # Original: [B02, B03, B04, B08, B05, B06, B07]
                # Result: [B02, B03, B04, PAN, B08, B05, B06, B07]
                result = np.insert(bands, 3, pan_band, axis=2)
                return result
            elif img.shape[0] == 7:
                # (C, H, W) format
                # Compute PAN as weighted average
                pan_band = np.sum(img * np.array(self.pan_weights)[:, np.newaxis, np.newaxis] / sum(self.pan_weights), axis=0)
                # Insert at index 3
                result = np.insert(img, 3, pan_band[np.newaxis, :, :], axis=0)
                return result
        
        return img

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("pan_weights",)


class PSFTransform(ImageOnlyTransform):
    """Apply Point Spread Function (PSF) kernel convolution to bands.
    
    Can use either an external executable (like PhisatCalculationTask) or
    pure Python convolution with Gaussian kernels.
    """

    def __init__(
        self,
        executable: Optional[str] = None,
        psf_kernels: Optional[Dict[int, np.ndarray]] = None,
        always_apply: bool = False,
    ):
        """Initialize PSF transform.
        
        Args:
            executable: Path to PSF executable binary. If provided, uses subprocess.
                       If None, uses pure Python convolution with psf_kernels.
            psf_kernels: Dictionary mapping band indices to PSF kernel arrays.
                        Used only if executable is None.
            always_apply: Whether to always apply
            p: Probability of applying
        """
        super().__init__(always_apply)
        self.executable = executable
        self.psf_kernels = psf_kernels or {}
        
        if executable is not None and not os.path.exists(executable):
            raise FileNotFoundError(f"PSF executable not found: {executable}")

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        """Apply PSF via executable or pure Python convolution."""
        if self.executable:
            return self._apply_via_executable(img)
        else:
            return self._apply_via_python(img)

    def _apply_via_executable(self, img: np.ndarray) -> np.ndarray:
        """Apply PSF using external executable (PhisatCalculationTask style).
        
        Creates temporary directory, writes input to numpy file,
        runs executable, reads and returns output.
        """
        if self.executable is None:
            raise RuntimeError("PSF executable path is not set")
        
        with tempfile.TemporaryDirectory(prefix="phisat2_psf_") as temp_dir:
            input_npy = os.path.join(temp_dir, "input.npy")
            output_npy = os.path.join(temp_dir, "output.npy")
            
            # Write input to temporary numpy file
            np.save(input_npy, img)
            
            # Run executable
            try:
                subprocess.run(
                    [cast(str, self.executable), "PSF", input_npy, output_npy],
                    check=True,
                    capture_output=True,
                )
            except subprocess.CalledProcessError as e:
                raise RuntimeError(
                    f"PSF executable failed: {e.stderr.decode()}"
                )
            
            # Read and return output
            if os.path.exists(output_npy):
                return np.load(output_npy).astype(img.dtype)
            else:
                raise RuntimeError(f"PSF executable did not produce output file")

    def _apply_via_python(self, img: np.ndarray) -> np.ndarray:
        """Apply PSF using pure Python convolution."""
        if not self.psf_kernels:
            return img
        
        result = img.copy().astype(np.float32)
        
        # Handle (H, W, C) format
        if img.ndim == 3 and img.shape[2] > 1:
            for band_idx in self.psf_kernels.keys():
                if band_idx < img.shape[2]:
                    kernel = self.psf_kernels[band_idx]
                    result[..., band_idx] = convolve(
                        result[..., band_idx],
                        kernel,
                        mode="mirror"
                    )
        # Handle (C, H, W) format
        elif img.ndim == 3 and img.shape[0] > 1:
            for band_idx in self.psf_kernels.keys():
                if band_idx < img.shape[0]:
                    kernel = self.psf_kernels[band_idx]
                    result[band_idx] = convolve(
                        result[band_idx],
                        kernel,
                        mode="mirror"
                    )
        
        return result.astype(img.dtype)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("executable", "psf_kernels")


class SNRNoiseTransform(ImageOnlyTransform):
    """Add SNR (Signal-to-Noise Ratio) noise to simulate sensor noise.
    
    Can use either an external executable (like PhisatCalculationTask) or
    pure Python noise addition.
    """

    def __init__(
        self,
        executable: Optional[str] = None,
        snr_values: Optional[Dict[int, float]] = None,
        l_ref: float = 0.1,
        always_apply: bool = False,
    ):
        """Initialize SNR noise transform.
        
        Args:
            executable: Path to SNR executable binary. If provided, uses subprocess.
                       If None, uses pure Python noise addition with snr_values.
            snr_values: Dictionary mapping band indices to SNR values (dB).
                       Used only if executable is None.
            l_ref: Reference spectral radiance (W/m^2/sr/um) for noise scaling.
            always_apply: Whether to always apply
            p: Probability of applying
        """
        super().__init__(always_apply)
        self.executable = executable
        self.snr_values = snr_values or {}
        self.l_ref = l_ref
        
        if executable is not None and not os.path.exists(executable):
            raise FileNotFoundError(f"SNR executable not found: {executable}")

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        """Apply SNR noise via executable or pure Python."""
        if self.executable:
            return self._apply_via_executable(img)
        else:
            return self._apply_via_python(img)

    def _apply_via_executable(self, img: np.ndarray) -> np.ndarray:
        """Apply SNR noise using external executable (PhisatCalculationTask style).
        
        Creates temporary directory, writes input to numpy file,
        runs executable, reads and returns output.
        """
        if self.executable is None:
            raise RuntimeError("SNR executable path is not set")
        
        with tempfile.TemporaryDirectory(prefix="phisat2_snr_") as temp_dir:
            input_npy = os.path.join(temp_dir, "input.npy")
            output_npy = os.path.join(temp_dir, "output.npy")
            
            # Write input to temporary numpy file
            np.save(input_npy, img)
            
            # Run executable
            try:
                subprocess.run(
                    [cast(str, self.executable), "SNR", input_npy, output_npy],
                    check=True,
                    capture_output=True,
                )
            except subprocess.CalledProcessError as e:
                raise RuntimeError(
                    f"SNR executable failed: {e.stderr.decode()}"
                )
            
            # Read and return output
            if os.path.exists(output_npy):
                return np.load(output_npy).astype(img.dtype)
            else:
                raise RuntimeError(f"SNR executable did not produce output file")

    def _apply_via_python(self, img: np.ndarray) -> np.ndarray:
        """Apply SNR noise using pure Python."""
        if not self.snr_values:
            return img
        
        result = img.copy().astype(np.float32)
        random_noise = np.random.normal(0, 1, size=result.shape).astype(np.float32)
        
        # Handle (H, W, C) format
        if result.ndim == 3 and result.shape[2] > 1:
            for band_idx, snr in self.snr_values.items():
                if band_idx < result.shape[2]:
                    noise_scale = self.l_ref / snr
                    result[..., band_idx] = result[..., band_idx] + noise_scale * random_noise[..., band_idx]
        # Handle (C, H, W) format
        elif result.ndim == 3 and result.shape[0] > 1:
            for band_idx, snr in self.snr_values.items():
                if band_idx < result.shape[0]:
                    noise_scale = self.l_ref / snr
                    result[band_idx] = result[band_idx] + noise_scale * random_noise[band_idx]
        
        # Clip to valid range
        result = np.clip(result, img.min(), img.max())
        return result.astype(img.dtype)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("executable", "snr_values", "l_ref")


class CalculateRadianceTransform(ImageOnlyTransform):
    """Calculate radiances from reflectances using solar irradiance and Earth-Sun distance.
    
    This transform uses metadata fetched based on location and temporal coordinates
    from the dataset to compute radiance values from reflectance values.
    Requires location_coords and temporal_coords to be passed via targets_as_params.
    
    Implements similar functionality to CalculateRadianceTask from phisat2_utils.
    """

    def __init__(
        self,
        solar_irradiances: Optional[Dict[str, float]] = None,
        earth_sun_distance: Optional[float] = None,
        sun_zenith_angle: Optional[float] = None,
        always_apply: bool = False,
    ):
        """Initialize radiance calculation transform.
        
        Args:
            solar_irradiances: Dictionary mapping band names to solar irradiance values.
                             If None, will be fetched from metadata using temporal/location coords.
            earth_sun_distance: Earth-Sun distance in AU.
                               If None, will be estimated from temporal coordinates.
            sun_zenith_angle: Sun zenith angle in degrees.
                             If None, will be fetched from metadata.
            always_apply: Whether to always apply
            p: Probability of applying
        """
        super().__init__(always_apply)
        self.solar_irradiances = solar_irradiances
        self.earth_sun_distance = earth_sun_distance
        self.sun_zenith_angle = sun_zenith_angle
        self._targets_as_params = ["location_coords", "temporal_coords"]

    @property
    def targets_as_params(self) -> List[str]:
        """Return the list of targets that will be passed as params."""
        return self._targets_as_params

    @staticmethod
    def _fetch_solar_irradiance_from_metadata(
        metadata_dict: Dict[str, Any]
    ) -> Dict[str, float]:
        """Extract solar irradiances from metadata.
        
        Args:
            metadata_dict: Dictionary containing XML metadata or solar irradiance values
            
        Returns:
            Dictionary mapping band names to solar irradiance values
        """
        # Default solar irradiance values for Sentinel-2 bands (W/m^2/um)
        default_irradiances = {
            "B02": 1941.0,   # Blue
            "B03": 1822.0,   # Green
            "B04": 1610.0,   # Red
            "B05": 1519.0,   # Vegetation Red Edge
            "B06": 1447.0,   # Vegetation Red Edge
            "B07": 1387.0,   # Vegetation Red Edge
            "B08": 1034.0,   # NIR
            "B8A": 955.0,    # Vegetation Red Edge
        }
        
        # If XML metadata is available, try to parse it
        if "xml_root" in metadata_dict:
            try:
                xml_root = metadata_dict["xml_root"]
                irradiance_list = xml_root.find(".//Solar_Irradiance_List")
                if irradiance_list is not None:
                    band_id_to_band = {
                        1: "B02", 2: "B03", 3: "B04", 4: "B05", 5: "B06", 6: "B07", 7: "B08"
                    }
                    for irradiance_elem in irradiance_list.findall("SOLAR_IRRADIANCE"):
                        band_id = irradiance_elem.get("bandId")
                        if band_id is not None:
                            try:
                                band_id_int = int(band_id)
                                irradiance_value = float(irradiance_elem.text)
                                if band_id_int in band_id_to_band:
                                    default_irradiances[band_id_to_band[band_id_int]] = irradiance_value
                            except (ValueError, TypeError):
                                pass
            except Exception:
                pass
        
        # If solar irradiances are directly provided in metadata, use those
        if "solar_irradiances" in metadata_dict:
            return metadata_dict["solar_irradiances"]
        
        return default_irradiances

    @staticmethod
    def _estimate_earth_sun_distance(temporal_coords: np.ndarray) -> float:
        """Estimate Earth-Sun distance from temporal coordinates.
        
        Args:
            temporal_coords: Array of shape (n_timesteps, 2) with (year, day_of_year)
            
        Returns:
            Earth-Sun distance in AU
        """
        try:
            if len(temporal_coords.shape) == 2 and temporal_coords.shape[0] > 0:
                from datetime import datetime
                from sunpy.coordinates import sun as sun_coords
                from astropy import units as u_astropy
                
                year = int(temporal_coords[0, 0])
                day_of_year = int(temporal_coords[0, 1])
                # Create datetime from year and day of year
                date = datetime.strptime(f"{year} {day_of_year}", "%Y %j")
                # Convert to ISO format string for sunpy
                date_str = date.isoformat()
                earth_sun_dist = sun_coords.earth_distance(date_str).to(u_astropy.au).value
                return earth_sun_dist
        except Exception:
            print("Failed to estimate Earth-Sun distance from temporal coordinates, defaulting to 1 AU")
        
        # Default to 1 AU
        return 1.0

    @staticmethod
    def _fetch_sun_zenith_angle_from_metadata(
        metadata_dict: Dict[str, Any],
        location_coords: Optional[np.ndarray] = None
    ) -> float:
        """Extract sun zenith angle from metadata.
        
        Args:
            metadata_dict: Dictionary containing metadata information
            location_coords: Location coordinates (lat, lon) for interpolation
            
        Returns:
            Sun zenith angle in degrees
        """
        # If sun zenith angle is directly in metadata, use it
        if "sun_zenith_angle" in metadata_dict:
            return float(metadata_dict["sun_zenith_angle"])
        
        # If zenith angle array is available, interpolate at location
        if "sun_zenith_angles" in metadata_dict:
            zenith_array = metadata_dict["sun_zenith_angles"]
            if location_coords is not None and zenith_array.ndim == 2:
                # Interpolate at center (simple approach)
                center_y, center_x = zenith_array.shape[0] // 2, zenith_array.shape[1] // 2
                return float(zenith_array[center_y, center_x])
        
        # Default to 0 (sun at zenith)
        return 0.0

    def apply(
        self,
        img: np.ndarray,
        location_coords: Optional[np.ndarray] = None,
        temporal_coords: Optional[np.ndarray] = None,
        **params: Any
    ) -> np.ndarray:
        """Calculate radiances from reflectances.
        
        Args:
            img: Input image with reflectance values in (H, W, C) format
            location_coords: Location coordinates (lat, lon) from dataset
            temporal_coords: Temporal coordinates (year, day_of_year) from dataset
            **params: Additional parameters from albumentations
            
        Returns:
            Image with radiance values
        """
        # Prepare metadata dictionary from available data
        metadata_dict = {}
        
        # Fetch or use provided solar irradiances
        if self.solar_irradiances is not None:
            metadata_dict["solar_irradiances"] = self.solar_irradiances
        
        # Estimate Earth-Sun distance from temporal coordinates if not provided
        if self.earth_sun_distance is not None:
            earth_sun_dist = self.earth_sun_distance
        elif temporal_coords is not None:
            earth_sun_dist = self._estimate_earth_sun_distance(temporal_coords)
        else:
            earth_sun_dist = 1.0  # Default to 1 AU
        
        # Fetch sun zenith angle from metadata if not provided
        if self.sun_zenith_angle is not None:
            sun_zenith = self.sun_zenith_angle
        else:
            sun_zenith = self._fetch_sun_zenith_angle_from_metadata(
                metadata_dict, location_coords
            )
        
        # Extract solar irradiances
        solar_irradiances = self._fetch_solar_irradiance_from_metadata(metadata_dict)
        
        # Prepare solar irradiance array for the image bands
        # Assuming 7 bands: B02, B03, B04, B05, B06, B07, B08 (or 8 with PAN)
        num_bands = img.shape[2] if img.ndim == 3 else img.shape[0]
        
        # Map bands to irradiance values
        band_order = ["B02", "B03", "B04", "B05", "B06", "B07", "B08"]  # First 7 standard bands
        if num_bands > 7:
            band_order.insert(3, "PAN")  # PAN band at position 3 if present
        
        irradiance_values = np.array(
            [solar_irradiances.get(band, solar_irradiances["B02"]) for band in band_order[:num_bands]]
        )
        
        # Calculate radiance: L = reflectance * cos(zenith) * distance^2 / pi * irradiance
        # More precisely: L = reflectance * irradiance * cos(zenith) / pi * distance^2
        cos_zenith = np.cos(np.radians(sun_zenith))
        factor = cos_zenith * (earth_sun_dist ** 2) / np.pi
        
        # Apply the transformation
        if img.ndim == 3:
            # (H, W, C) format
            radiance = img.astype(np.float32) * factor * irradiance_values[np.newaxis, np.newaxis, :]
        else:
            # (C, H, W) format
            radiance = img.astype(np.float32) * factor * irradiance_values[:, np.newaxis, np.newaxis]
        
        return radiance.astype(img.dtype)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("solar_irradiances", "earth_sun_distance", "sun_zenith_angle")


def _create_gaussian_psf_kernels(num_bands: int = 8, sigma: float = 1.0) -> Dict[int, np.ndarray]:
    """Create Gaussian PSF kernels for multispectral bands.
    
    Args:
        num_bands: Number of bands to create kernels for
        sigma: Standard deviation of Gaussian kernel
        
    Returns:
        Dictionary mapping band indices to 7x7 PSF kernel arrays
    """
    size = 7
    x = np.linspace(-size // 2, size // 2, size)
    y = np.linspace(-size // 2, size // 2, size)
    X, Y = np.meshgrid(x, y)
    
    # Create Gaussian kernel
    kernel = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    
    # Return same kernel for all bands (can be customized per band)
    return {i: kernel for i in range(num_bands)}


def create_phisat2_transform(
    phisat_config: Dict[str, Any] | None = None,
) -> A.Compose:
    """Create an Albumentations Compose pipeline with Phisat-2 simulation transforms.
    
    Simulates various degradation effects seen in Φ-sat-2 satellite imagery compared
    to original Sentinel-2 data using methods from phisat2_utils. 
    Can be used as train_transform in terratorch datamodules.
    
    Args:
        phisat_config: Configuration dictionary with keys:
            - apply_radiance_calculation (bool): Calculate radiances from reflectances
            - apply_band_misalignment (bool): Apply band misalignment
            - apply_pan_band (bool): Create panchromatic band
            - apply_psf (bool): Apply PSF kernel convolution
            - apply_snr (bool): Apply SNR noise
            - processing_level (str): "L1A" or "L1B" for misalignment
            - psf_sigma (float): Sigma for Gaussian PSF kernels
            - psf_executable (str): Path to PSF executable (optional)
            - snr_values (dict): Band index -> SNR value mapping
            - snr_executable (str): Path to SNR executable (optional)
            - l_ref (float): Reference radiance for SNR scaling
            - solar_irradiances (dict): Band name -> solar irradiance mapping
            - earth_sun_distance (float): Earth-Sun distance in AU
            - sun_zenith_angle (float): Sun zenith angle in degrees
            
    Returns:
        albumentations.Compose pipeline ready for use with terratorch datamodules
        
    Example:
        >>> config = {
        ...     "apply_radiance_calculation": True,
        ...     "apply_band_misalignment": True,
        ...     "apply_pan_band": True,
        ...     "apply_psf": True,
        ...     "apply_snr": True,
        ...     "processing_level": "L1A",
        ...     "psf_executable": "/path/to/psf_executable",
        ...     "snr_executable": "/path/to/snr_executable",
        ... }
        >>> transform = create_phisat2_transform(config)
        >>> # Use with terratorch datamodule:
        >>> datamodule = Sen1Floods11DataModule(
        ...     root_dir="/path/to/sen1floods11",
        ...     train_transform=transform
        ... )
    """
    transforms = []
    
    if phisat_config is None:
        print("No Phisat-2 configuration provided, returning empty Compose")
        return A.Compose([])
    
    # Radiance calculation from reflectances, uses location/temporal coords
    if phisat_config.get("apply_radiance_calculation", False):
        transforms.append(
            CalculateRadianceTransform(
                solar_irradiances=phisat_config.get("solar_irradiances", None),
                earth_sun_distance=phisat_config.get("earth_sun_distance", None),
                sun_zenith_angle=phisat_config.get("sun_zenith_angle", None),
            )
        )
    
    # Band misalignment (L1A/L1B), simulates sensor artifacts
    if phisat_config.get("apply_band_misalignment", False):
        transforms.append(
            BandMisalignmentTransform(
                processing_level=phisat_config.get("processing_level", ProcessingLevels.L1A),
                std_sea=phisat_config.get("std_sea", 6),
            )
        )
    
    # PAN band creation, adds weighted average of bands
    if phisat_config.get("apply_pan_band", False):
        transforms.append(
            PanBandTransform()
        )
    
    # PSF kernel convolution, simulates optical degradation
    if phisat_config.get("apply_psf", False):
        psf_executable = phisat_config.get("psf_executable", None)
        
        if psf_executable is None:
            # Use pure Python PSF with Gaussian kernels
            psf_sigma = phisat_config.get("psf_sigma", 1.0)
            psf_kernels = _create_gaussian_psf_kernels(
                num_bands=8 if phisat_config.get("apply_pan_band", False) else 7,
                sigma=psf_sigma
            )
            transforms.append(
                PSFTransform(
                    psf_kernels=psf_kernels,
                )
            )
        else:
            # Use external executable for PSF
            transforms.append(
                PSFTransform(
                    executable=psf_executable,
                )
            )
    
    # SNR noise, simulates sensor noise
    if phisat_config.get("apply_snr", False):
        snr_executable = phisat_config.get("snr_executable", None)
        
        if snr_executable is None:
            # Use pure Python SNR noise
            # Default SNR values for S2 bands (typical values from literature)
            default_snr = {
                0: 100,  # B02
                1: 105,  # B03
                2: 106,  # B04
                3: 90,   # PAN (if added)
                4: 150,  # B08
                5: 180,  # B05
                6: 190,  # B06
                7: 195,  # B07
            }
            snr_values = phisat_config.get("snr_values", default_snr)
            
            transforms.append(
                SNRNoiseTransform(
                    snr_values=snr_values,
                    l_ref=phisat_config.get("l_ref", 0.1),
                )
            )
        else:
            # Use external executable for SNR
            transforms.append(
                SNRNoiseTransform(
                    executable=snr_executable,
                    l_ref=phisat_config.get("l_ref", 0.1),
                )
            )
            
        transforms.append(ToTensorV2())
    
    return A.Compose(transforms)
