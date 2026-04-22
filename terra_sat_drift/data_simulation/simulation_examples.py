"""Example usage of simulation workflow."""

from pathlib import Path
from phisat2_constants import ProcessingLevels
from simulation_pipeline import simulate_with_executor
from simulation_config import SimulationConfig, SimulationSteps
from sentinelhub.exceptions import SHDeprecationWarning
import warnings

def example_5_simulation_pipeline():
    """Example 5: Simulate a single S2 file using the pipeline."""
    print("\n" + "=" * 80)
    print("Example 5: Simulating S2 file with pipeline")
    print("=" * 80)
    # Configure pipeline
    steps = SimulationSteps(
        radiance=True,
        add_panchromatic=True,
        band_misalignment=True,
        snr_simulation=True,
        psf_filtering=True,
        reflectance_conversion=True,
    )

    config = SimulationConfig(
        steps=steps,
        processing_level=ProcessingLevels.L1C,
        phisat2_exec_path="executables/phisat2_unix.bin",  # Use with snr_psf_method="executable"
        snr_psf_method="executable",  # "alternative" or "executable"
    )

    # Print the steps that are on True
    print(f"Simulating with steps: {[step for step, enabled in steps.__dict__.items() if enabled]} to level {config.processing_level}")

    # Note: Set snr_psf_method to:
    #   - "alternative" to use Python-based AlternativePhisatCalculationTask (default)
    #   - "executable" to use compiled phisat2 binary (requires phisat2_exec_path to be set)

    executor = simulate_with_executor(
        config=config, 
        source_dir=Path("/shared/projects/phisat2/data/interim/s2b_croped"),
        output_dir=Path("/shared/projects/phisat2/data/interim/s2b_simulated"),
        metadata_dir=Path("/shared/projects/phisat2/data/interim/s2b_merged"),
        logs_folder="/shared/projects/phisat2/data/index/logs",
        pattern="*_s2b_cropped.tif",
        workers=1,
        save_logs=True,
    )
    print(f"Execution stats: {executor.general_stats}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=SHDeprecationWarning)
    warnings.filterwarnings("ignore")
    example_5_simulation_pipeline()
