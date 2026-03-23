"""Simulation pipeline wrapping phisat2_utils tasks for on-the-fly Φ-sat-2 synthesis."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from datetime import datetime
import numpy as np

from eolearn.core.eodata import EOPatch
from eolearn.core.constants import FeatureType
from .phisat2_utils import (  
    AddPANBandTask,  
    BandMisalignmentTask,  
    CalculateRadianceTask,  
    CalculateReflectanceTask,  
    AlternativePhisatCalculationTask,
)
from .phisat2_constants import ProcessingLevels  

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
                s2_data = src.read().astype(np.float32)  # Shape: (bands, height, width)
                profile = src.profile.copy()
                metadata = src.tags()

            # Ensure exactly 7 S2 bands (drop panchromatic if 8)
            if s2_data.shape[0] == 8:
                s2_data = s2_data[[0, 1, 2, 4, 5, 6, 7], :, :]
            elif s2_data.shape[0] != 7:
                raise ValueError(f"Expected 7 or 8 bands, got {s2_data.shape[0]}")

            # Create EOPatch with single timestamp
            eopatch = EOPatch()
            eopatch.timestamp = [datetime.now()]  # Dummy timestamp
            eopatch[FeatureType.DATA, "S2_BANDS"] = s2_data[np.newaxis, :, :, :]  # Add time dimension

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

            # ===== Step 4: SNR + PSF simulation (if enabled) =====
            if self.config.steps.snr_simulation or self.config.steps.psf_filtering:
                snr_values = self.config.snr_values or {
                    "B02": 15,
                    "B03": 15,
                    "B04": 15,
                    "PAN": 10,
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
            output_data = eopatch[FeatureType.DATA, current_feature][0]  # Remove time dimension
            
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

        kernel_bands = ["B1", "B2", "B3", "B0", "B7", "B4", "B5", "B6"]
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
