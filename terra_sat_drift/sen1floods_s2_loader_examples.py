"""Example usage of Sen1Floods11S2 loader for simulation workflow."""

from pathlib import Path
from sen1floods11_loader import Sen1Floods11S2Loader
from simulation_pipeline import SimulationPipeline
from simulation_config import SimulationConfig, SimulationSteps
from sentinelhub.exceptions import SHDeprecationWarning
import warnings

def example_1_load_s2_files():
    """Example 1: Load S2 file paths from sen1floods dataset."""
    print("=" * 80)
    print("Example 1: Loading S2 files from Sen1Floods11")
    print("=" * 80)

    # Initialize loader for training split
    loader = Sen1Floods11S2Loader(
        root_path="datasets/sen1floods11",
        split="train",
        dataset_type="HandLabeled",
    )

    print(f"Total S2 files in train split: {len(loader)}")

    # Get all file paths
    s2_files = loader.get_s2_files()
    print(f"First 5 S2 files:")
    for i, f in enumerate(s2_files[:5], 1):
        print(f"  {i}. {f.name}")

    # Get file metadata for the first file
    print(f"\nMetadata for first file:")
    metadata = loader.load_s2_metadata(0)
    print(f"  Path: {metadata['path']}")
    print(f"  Shape: {metadata['shape']}")
    print(f"  Bands: {metadata['count']}")
    print(f"  Data type: {metadata['dtype']}")
    print(f"  Location: {loader.get_location_from_filename(0)}")


def example_2_load_single_file():
    """Example 2: Load and inspect a single S2 file."""
    print("\n" + "=" * 80)
    print("Example 2: Loading single S2 file")
    print("=" * 80)

    loader = Sen1Floods11S2Loader(
        root_path="datasets/sen1floods11",
        split="train",
    )

    # Load first S2 file as numpy array
    print("Loading first S2 file...")
    s2_data = loader.load_s2_file(0)
    print(f"S2 data shape: {s2_data.shape}")
    print(f"S2 data dtype: {s2_data.dtype}")
    print(f"S2 data min/max: {s2_data.min():.2f} / {s2_data.max():.2f}")


def example_3_batch_loading():
    """Example 3: Batch loading multiple files."""
    print("\n" + "=" * 80)
    print("Example 3: Batch loading")
    print("=" * 80)

    loader = Sen1Floods11S2Loader(
        root_path="datasets/sen1floods11",
        split="train",
    )

    # Load first 5 files
    indices = [0, 1, 2, 3, 4]
    print(f"Loading batch of {len(indices)} files...")
    batch = loader.load_batch(indices, return_paths=True)

    print(f"Batch data count: {len(batch['data'])}")
    for i, (idx, path) in enumerate(batch['paths']):
        print(f"  {i+1}. Index {idx}: {Path(path).name} - Shape: {batch['data'][i].shape}")


def example_4_iterate_files():
    """Example 4: Iterating over S2 files."""
    print("\n" + "=" * 80)
    print("Example 4: Iterating over files")
    print("=" * 80)

    loader = Sen1Floods11S2Loader(
        root_path="datasets/sen1floods11",
        split="valid",
    )

    print(f"Iterating over validation split ({len(loader)} files)...")
    for i, file_path in enumerate(loader):
        if i >= 3:  # Just show first 3
            print(f"  ... and {len(loader) - 3} more files")
            break
        print(f"  {i+1}. {file_path.name}")


def example_5_simulation_pipeline():
    """Example 5: Simulate a single S2 file using the pipeline."""
    print("\n" + "=" * 80)
    print("Example 5: Simulating S2 file with pipeline")
    print("=" * 80)
    # Load S2 file
    loader = Sen1Floods11S2Loader(
        root_path="datasets/sen1floods11",
        split="train",
    )
    s2_file = loader.get_s2_files()[0]

    geojson = loader.load_geojson_metadata()

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
        processing_level="L1C",
        sh_config_path="sh_config.json",
        phisat2_exec_path="executables/phisat2_unix.bin",  # Use with snr_psf_method="executable"
        snr_psf_method="executable",  # "alternative" or "executable"
    )

    pipeline = SimulationPipeline(config)

    # Simulate
    output_file = Path("tiff_folder/simulated_s2") / f"simulated_{s2_file.name}"
    print(f"Input: {s2_file.name}")
    print(f"Output: {output_file.name}")
    # Print the steps that are on True
    print(f"Simulating with steps: {[step for step, enabled in steps.__dict__.items() if enabled]}")

    # Note: Set snr_psf_method to:
    #   - "alternative" to use Python-based AlternativePhisatCalculationTask (default)
    #   - "executable" to use compiled phisat2 binary (requires phisat2_exec_path to be set)

    # Note: Uncomment to actually run the simulation
    success = pipeline.simulate_single_file(s2_file, output_file, metadata=geojson)
    print(f"Success: {success}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=SHDeprecationWarning)
    warnings.filterwarnings("ignore")
    # Run examples
    # example_1_load_s2_files()
    # example_2_load_single_file()
    # example_3_batch_loading()
    # example_4_iterate_files()
    example_5_simulation_pipeline()

    print("\n" + "=" * 80)
    print("Examples completed!")
    print("=" * 80)
