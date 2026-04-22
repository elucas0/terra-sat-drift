"""Configuration for Φ-sat-2 simulation pipeline experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import json
from phisat2_constants import ProcessingLevels

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

    # Processing parameters
    phisat2_exec_path: Optional[str] = None  # Path to phisat2 binary if using SNR/PSF tasks
    snr_psf_method: str = "executable"  # "alternative" for Python implementation or "executable" for binary
    processing_level: ProcessingLevels = ProcessingLevels.L1C  # L1A, L1B, or L1C

    # Band misalignment parameter
    misalignment_std_sea: int = 6

    # SNR/PSF parameters (for alternative Python-based simulation)
    snr_values: Optional[dict] = None  # e.g., {"B02": 15, "B03": 15, ...}
    psf_kernel_sigma: float = 1.0
    radiance_reference: float = 100.0
    
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
            "phisat2_exec_path": self.phisat2_exec_path,
            "snr_psf_method": self.snr_psf_method,
            "processing_level": self.processing_level,
            "misalignment_std_sea": self.misalignment_std_sea,
            "snr_values": self.snr_values,
            "psf_kernel_sigma": self.psf_kernel_sigma,
            "radiance_reference": self.radiance_reference,
        }
        with open(output_path, "w") as f:
            json.dump(config_dict, f, indent=2)
