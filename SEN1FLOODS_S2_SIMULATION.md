# Sen1Floods11 S2 Simulation Guide

This guide explains how to use the Sen1Floods11 S2 loader and batch simulation scripts to prepare Sentinel-2 data from the Sen1Floods11 dataset for Φ-sat-2 simulation.

## Dataset Structure

The Sen1Floods11 dataset is organized as follows:

```
datasets/sen1floods11/
└── v1.1/
    ├── data/
    │   └── flood_events/
    │       └── HandLabeled/
    │           ├── S1Hand/          # Sentinel-1 SAR data
    │           ├── S2Hand/          # Sentinel-2 optical data (TARGET FOR SIMULATION)
    │           ├── LabelHand/       # Flood labels/masks
    │           ├── JRCWaterHand/    # JRC water masks
    │           └── S1OtsuLabelHand/ # Otsu labels
    ├── splits/
    │   └── flood_handlabeled/
    │       ├── flood_train_data.csv         # Train split
    │       ├── flood_valid_data.csv         # Validation split
    │       ├── flood_test_data.csv          # Test split
    │       └── flood_bolivia_data.csv       # Bolivia-specific split
    └── Sen1Floods11_Metadata.geojson        # Metadata with dates and locations
```

## Components

### 1. `sen1floods11_loader.py`

**Class: `Sen1Floods11S2Loader`**

A data loader that extracts and manages S2 file paths from the Sen1Floods11 dataset.

#### Features:
- Load S2 files from any split (train, val, test)
- Iterate over files or get complete file lists
- Load individual files as numpy arrays
- Batch loading support
- Metadata extraction (shape, dtype, profile)
- Location name extraction from filenames

#### Usage:

```python
from terra_sat_drift.sen1floods11_loader import Sen1Floods11S2Loader

# Initialize loader
loader = Sen1Floods11S2Loader(
    root_path="datasets/sen1floods11",
    split="train",                    # train, valid, or test
    dataset_type="HandLabeled"        # HandLabeled or WeaklyLabeled
)

# Get all S2 file paths
s2_files = loader.get_s2_files()
print(f"Total S2 files: {len(s2_files)}")

# Load a single file
s2_data = loader.load_s2_file(0)  # Shape: (bands, height, width)
print(f"S2 shape: {s2_data.shape}")

# Get metadata
metadata = loader.load_s2_metadata(0)
print(f"Bands: {metadata['count']}")

# Batch loading
batch = loader.load_batch([0, 1, 2, 3, 4], return_paths=True)
print(f"Loaded {len(batch['data'])} files")

# Iterate over files
for s2_file in loader:
    print(s2_file.name)
```

### 2. `batch_simulate_sen1floods.py`

**Function: `simulate_sen1floods_s2()`**

Orchestrates batch simulation of S2 files through the Φ-sat-2 pipeline.

#### Features:
- Batch process all S2 files from a split
- Configurable simulation steps
- Progress tracking and logging
- Error handling with detailed reports
- Results saved to JSON format
- Support for all processing levels (L1A, L1B, L1C)

#### Usage as Python API:

```python
from terra_sat_drift.batch_simulate_sen1floods import simulate_sen1floods_s2

results = simulate_sen1floods_s2(
    dataset_root="datasets/sen1floods11",
    output_dir="tiff_folder/simulated_sen1floods",
    split="train",                    # train, valid, or test
    dataset_type="HandLabeled",
    max_files=10,                     # Limit to 10 files (optional)
    simulation_steps={
        "radiance": True,
        "add_panchromatic": True,
        "band_misalignment": True,
        "snr_simulation": True,
        "psf_filtering": True,
        "reflectance_conversion": True,
    },
    processing_level="L1C",
    verbose=True,
)

print(f"Successful: {len(results['successful'])}")
print(f"Failed: {len(results['failed'])}")
```

#### Command-Line Usage:

```bash
# Basic usage (process entire train split)
python -m terra_sat_drift.batch_simulate_sen1floods \
    --dataset-root datasets/sen1floods11 \
    --output-dir tiff_folder/simulated_sen1floods \
    --split train

# Process validation split with limited files
python -m terra_sat_drift.batch_simulate_sen1floods \
    --dataset-root datasets/sen1floods11 \
    --output-dir tiff_folder/simulated_sen1floods \
    --split valid \
    --max-files 50

# Process with specific simulation steps disabled
python -m terra_sat_drift.batch_simulate_sen1floods \
    --dataset-root datasets/sen1floods11 \
    --output-dir tiff_folder/simulated_sen1floods \
    --split test \
    --processing-level L1C \
    --disable-snr \
    --disable-psf \
    --verbose

# Help
python -m terra_sat_drift.batch_simulate_sen1floods --help
```

#### Command-Line Arguments:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset-root` | Path | `datasets/sen1floods11` | Path to sen1floods dataset |
| `--output-dir` | Path | `tiff_folder/simulated_sen1floods` | Output directory for simulated files |
| `--split` | str | `train` | Dataset split (train/val/valid/test) |
| `--dataset-type` | str | `HandLabeled` | Dataset type |
| `--max-files` | int | None | Max files to process (None = all) |
| `--processing-level` | str | `L1C` | Output level (L1A/L1B/L1C) |
| `--disable-radiance` | flag | - | Disable radiance conversion |
| `--disable-pan` | flag | - | Disable PAN band addition |
| `--disable-misalignment` | flag | - | Disable band misalignment |
| `--disable-snr` | flag | - | Disable SNR simulation |
| `--disable-psf` | flag | - | Disable PSF filtering |
| `--disable-reflectance` | flag | - | Disable reflectance conversion |
| `--phisat2-path` | str | None | Path to phisat2 binary |
| `-v, --verbose` | flag | - | Enable verbose logging |

## Examples

### Example 1: Load and inspect S2 files

```python
from terra_sat_drift.sen1floods11_loader import Sen1Floods11S2Loader

loader = Sen1Floods11S2Loader("datasets/sen1floods11", split="train")

# Check what we have
print(f"Training set: {len(loader)} images")
for i, path in enumerate(loader.get_s2_files()[:5]):
    print(f"  {i}: {path.name}")

# Load and check first file
data = loader.load_s2_file(0)
print(f"Shape: {data.shape}, Dtype: {data.dtype}")
print(f"Value range: {data.min():.1f} to {data.max():.1f}")
```

### Example 2: Run S2 simulation on a subset

```python
from terra_sat_drift.batch_simulate_sen1floods import simulate_sen1floods_s2

# Simulate first 10 files from validation set with minimal steps
results = simulate_sen1floods_s2(
    dataset_root="datasets/sen1floods11",
    output_dir="tiff_folder/test_sim",
    split="valid",
    max_files=10,
    simulation_steps={
        "radiance": True,
        "add_panchromatic": False,  # Skip PAN
        "band_misalignment": False,  # Skip misalignment
        "snr_simulation": False,     # Skip SNR
        "psf_filtering": False,      # Skip PSF
        "reflectance_conversion": False,
    },
    processing_level="L1A",
    verbose=True,
)

print(f"Results: {len(results['successful'])} OK, {len(results['failed'])} failed")
```

### Example 3: Full pipeline with all steps

```bash
# Simulate entire training set with all processing steps
python -m terra_sat_drift.batch_simulate_sen1floods \
    --dataset-root datasets/sen1floods11 \
    --output-dir tiff_folder/simulated_train \
    --split train \
    --processing-level L1C \
    --verbose

# Then process validation and test
python -m terra_sat_drift.batch_simulate_sen1floods \
    --dataset-root datasets/sen1floods11 \
    --output-dir tiff_folder/simulated_val \
    --split valid

python -m terra_sat_drift.batch_simulate_sen1floods \
    --dataset-root datasets/sen1floods11 \
    --output-dir tiff_folder/simulated_test \
    --split test
```

## Output Structure

After simulation, the output directory will contain:

```
tiff_folder/simulated_sen1floods/
├── simulated_train/              # Simulated files for train split
│   ├── simulated_Bolivia_103757_S2Hand.tif
│   ├── simulated_Bolivia_129334_S2Hand.tif
│   └── ...
├── simulated_valid/              # Simulated files for validation split
│   └── ...
├── simulated_test/               # Simulated files for test split
│   └── ...
├── simulation_results_train.json
├── simulation_results_valid.json
├── simulation_results_test.json
└── simulation.log
```

## Results Format

Each `simulation_results_*.json` file contains:

```json
{
  "successful": [
    "/path/to/simulated_file1.tif",
    "/path/to/simulated_file2.tif"
  ],
  "failed": [
    "/path/to/failed_file1.tif"
  ],
  "metadata": {
    "split": "train",
    "dataset_type": "HandLabeled",
    "processing_level": "L1C",
    "total_files": 200,
    "simulation_steps": {
      "radiance": true,
      "add_panchromatic": true,
      "band_misalignment": true,
      "snr_simulation": true,
      "psf_filtering": true,
      "reflectance_conversion": true
    }
  }
}
```

## Performance Notes

- **File I/O**: Each S2 file is typically 100-500 MB depending on resolution
- **Processing time**: ~5-15 seconds per file on GPU (depending on simulation steps)
- **Memory**: ~2-4 GB RAM for batch processing
- **Disk space**: Simulated output is similar size to input

## Troubleshooting

### Missing S2 files
If you see warnings about missing S2 files, ensure the dataset is properly structured with S2Hand directory containing all S2Hand.tif files.

### Out of memory
Reduce `max_files` parameter or process splits separately with smaller batches.

### Slow performance
Disable unnecessary simulation steps or check GPU availability for accelerated processing.

## Integration with Simulation Pipeline

The scripts use the `SimulationPipeline` class which wraps Φ-sat-2 processing tasks:

1. **Radiance conversion**: DN → radiance (if enabled)
2. **PAN band addition**: Add synthetic panchromatic band (if enabled)
3. **Band misalignment**: Simulate spectral registration errors (if enabled)
4. **SNR + PSF**: Add noise and apply point-spread function (if enabled)
5. **Reflectance conversion**: Radiance → reflectance for L1C (if enabled)

Each step can be independently controlled via `SimulationSteps` configuration.
