"""Example usage of Sen1Floods11S2 loader for simulation workflow."""

from pathlib import Path
from simulation_pipeline import SimulationPipeline
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
        output_dir="tiff_folder/simulated_s2",
        processing_level="L1A",
        phisat2_exec_path="executables/phisat2_unix.bin",  # Use with snr_psf_method="executable"
        snr_psf_method="executable",  # "alternative" or "executable"
    )

    pipeline = SimulationPipeline(config)
    # Print the steps that are on True
    print(f"Simulating with steps: {[step for step, enabled in steps.__dict__.items() if enabled]}")

    # Note: Set snr_psf_method to:
    #   - "alternative" to use Python-based AlternativePhisatCalculationTask (default)
    #   - "executable" to use compiled phisat2 binary (requires phisat2_exec_path to be set)

    # Note: Uncomment to actually run the simulation
    success = pipeline.batch_simulate_from_source_dir(
        source_dir="tiff_folder/s2b_cropped", pattern="*.tif"
    )
    print(f"Success: {success}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=SHDeprecationWarning)
    warnings.filterwarnings("ignore")
    example_5_simulation_pipeline()

    print("\n" + "=" * 80)
    print("Examples completed!")
    print("=" * 80)
