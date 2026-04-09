"""Configuration for Φ-sat-2 simulation pipeline experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import json


@dataclass
class SimulationSteps:
    """Control which simulation steps are applied."""

    radiance: bool = True
    add_panchromatic: bool = True
    band_misalignment: bool = True
    snr_simulation: bool = True
    psf_filtering: bool = True
    reflectance_conversion: bool = True

    def as_dict(self) -> dict:
        return {
            "radiance": self.radiance,
            "add_panchromatic": self.add_panchromatic,
            "band_misalignment": self.band_misalignment,
            "snr_simulation": self.snr_simulation,
            "psf_filtering": self.psf_filtering,
            "reflectance_conversion": self.reflectance_conversion,
        }


@dataclass
class SimulationConfig:
    """Configuration for a single simulation experiment."""

    # Simulation control
    steps: SimulationSteps = field(default_factory=SimulationSteps)

    # Input data
    s2_source_dir: Path | str = "tiff_folder/raw_s2_cache"
    output_dir: Path | str = "tiff_folder/simulated_l1c"

    # Processing parameters
    phisat2_exec_path: Optional[str] = None  # Path to phisat2 binary if using SNR/PSF tasks
    sh_config_path: Optional[str] = None  # Path to Sentinel Hub config for metadata fetching
    snr_psf_method: str = "alternative"  # "alternative" for Python implementation or "executable" for binary
    cell_size: int = 256
    grid_overlap: float = 0.0
    processing_level: str = "L1C"  # L1A, L1B, or L1C

    # Band misalignment parameters
    misalignment_std_land: float = 1.0
    misalignment_std_sea: float = 6.0

    # SNR/PSF parameters (for alternative Python-based simulation)
    snr_values: Optional[dict] = None  # e.g., {"B02": 15, "B03": 15, ...}
    psf_kernel_sigma: float = 1.0
    radiance_reference: float = 100.0

    def __post_init__(self) -> None:
        """Convert string paths to Path objects."""
        self.s2_source_dir = Path(self.s2_source_dir)
        self.output_dir = Path(self.output_dir)

    @classmethod
    def from_dict(cls, config_dict: dict) -> SimulationConfig:
        """Load configuration from dictionary."""
        steps_dict = config_dict.pop("steps", {})
        steps = SimulationSteps(**steps_dict) if steps_dict else SimulationSteps()
        return cls(steps=steps, **config_dict)

    @classmethod
    def from_json(cls, json_path: Path | str) -> SimulationConfig:
        """Load configuration from JSON file."""
        with open(json_path, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def save_json(self, output_path: Path | str) -> None:
        """Save configuration to JSON file."""
        config_dict = {
            "steps": self.steps.as_dict(),
            "s2_source_dir": str(self.s2_source_dir),
            "output_dir": str(self.output_dir),
            "phisat2_exec_path": self.phisat2_exec_path,
            "sh_config_path": self.sh_config_path,
            "snr_psf_method": self.snr_psf_method,
            "cell_size": self.cell_size,
            "grid_overlap": self.grid_overlap,
            "processing_level": self.processing_level,
            "misalignment_std_land": self.misalignment_std_land,
            "misalignment_std_sea": self.misalignment_std_sea,
            "snr_values": self.snr_values,
            "psf_kernel_sigma": self.psf_kernel_sigma,
            "radiance_reference": self.radiance_reference,
        }
        with open(output_path, "w") as f:
            json.dump(config_dict, f, indent=2)
