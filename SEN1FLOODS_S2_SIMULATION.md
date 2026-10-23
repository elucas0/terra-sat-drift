# Sen1Floods11 S2 Simulation Guide

This guide explains how to simulate Φ-sat-2 observations from Sentinel-2 data in the **Sen1Floods11** dataset using the `terra_sat_drift.data_simulation` pipeline.

The simulation synthesises the spatial, spectral, and radiometric characteristics of the onboard Φ-sat-2 imager: spatial resampling to 4.75 m resolution, panchromatic (PAN) band synthesis, inter-band spatial misalignment, sensor noise (SNR), and optical blur (PSF).

---

## Dataset Structure

The pipeline expects the standard **Sen1Floods11 v1.1** directory layout:

```text
datasets/sen1floods11/
└── v1.1/
    ├── data/
    │   └── flood_events/
    │       └── HandLabeled/
    │           ├── S1Hand/          # Sentinel-1 SAR imagery
    │           ├── S2Hand/          # Sentinel-2 L1C optical GeoTIFFs (source for simulation)
    │           ├── LabelHand/       # Ground-truth flood masks (10 m)
    │           ├── JRCWaterHand/    # JRC permanent water masks
    │           └── S1OtsuLabelHand/ # Otsu thresholded masks
    ├── splits/
    │   └── flood_handlabeled/
    │       ├── flood_train_data.csv         # Train split filenames
    │       ├── flood_valid_data.csv         # Validation split filenames
    │       ├── flood_test_data.csv          # Test split filenames
    │       └── flood_bolivia_data.csv       # Bolivia regional test split
    └── Sen1Floods11_Metadata.geojson        # Scene coordinates and acquisition dates
```

---

## Key Components

### 1. `batch_simulate_sen1floods.py`

Located at: [`terra_sat_drift/data_simulation/batch_simulate_sen1floods.py`](file:///home/elouann/terra-sat-drift/terra_sat_drift/data_simulation/batch_simulate_sen1floods.py)

Orchestrates batch simulation over a chosen split (`train`, `valid`, or `test`).
It loads samples via `terratorch.datasets.Sen1Floods11NonGeo`, builds the `eo-learn` task graph via `simulate_with_executor`, and executes parallel synthesis.

#### Python API:

```python
from pathlib import Path
from terra_sat_drift.data_simulation.phisat2_constants import ProcessingLevels
from terra_sat_drift.data_simulation.batch_simulate_sen1floods import simulate_sen1floods_s2

simulate_sen1floods_s2(
    dataset_root="/path/to/sen1floods11",
    output_dir="/path/to/simulated_sen1floods",
    split="train",                              # "train", "val", "valid", or "test"
    max_files=10,                               # Optional int to limit samples (None = all)
    processing_level=ProcessingLevels.L1C,       # ProcessingLevels.L1A, L1B, or L1C
    simulation_steps={
        "spatial_resampling": True,
        "radiance": True,
        "add_panchromatic": True,
        "band_misalignment": True,
        "snr_simulation": True,
        "psf_filtering": True,
        "reflectance_conversion": True,
    },
    workers=4,
    verbose=True,
)
```

#### Parameters:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `dataset_root` | `Path \| str` | *required* | Root path containing the `v1.1/` dataset folder |
| `output_dir` | `Path \| str` | *required* | Destination root where simulated dataset structure is mirrored |
| `split` | `str` | `"train"` | Dataset split (`"train"`, `"valid"`, `"val"`, or `"test"`) |
| `max_files` | `int \| None` | `None` | Max samples to process (`None` processes all in split) |
| `simulation_steps` | `dict \| None` | `None` | Dict mapping step names to boolean flags (defaults to all `True`) |
| `processing_level` | `ProcessingLevels` | `ProcessingLevels.L1C` | Output radiometric level (`L1A`, `L1B`, or `L1C`) |
| `verbose` | `bool` | `False` | Enable debug logging |
| `config` | `SimulationConfig \| None` | `None` | Pre-configured `SimulationConfig` object (used by sweeps like `degradation_ladder.py`) |
| `workers` | `int` | `1` | Number of parallel worker threads in `EOExecutor` |

#### Command-Line Usage:

```bash
# Basic simulation of the validation split (runs from project root)
python terra_sat_drift/data_simulation/batch_simulate_sen1floods.py \
    --dataset-root /path/to/datasets/sen1floods11 \
    --output-dir /path/to/output/simulated_sen1floods \
    --split valid \
    --processing-level L1C \
    --verbose

# Quick test with only 5 files and no noise simulation
python terra_sat_drift/data_simulation/batch_simulate_sen1floods.py \
    --dataset-root /path/to/datasets/sen1floods11 \
    --output-dir /path/to/output/simulated_sen1floods \
    --split test \
    --max-files 5 \
    --disable-snr \
    --disable-psf
```

#### CLI Options:

| Option | Type | Default | Description |
|---|---|---|---|
| `--dataset-root` | `Path` | `/shared/home/elucas/datasets/sen1floods11` | Source dataset root |
| `--output-dir` | `Path` | `/shared/home/elucas/datasets/sen1floods11_simulated_alt_v2` | Output directory |
| `--split` | `choice` | `"train"` | Split: `train`, `val`, `valid`, `test` |
| `--max-files` | `int` | `None` | Maximum number of files to process |
| `--processing-level` | `choice` | `"L1C"` | Target level: `L1A`, `L1B`, `L1C` |
| `--disable-radiance` | flag | `False` | Skip reflectance-to-radiance conversion |
| `--disable-pan` | flag | `False` | Skip PAN band generation |
| `--disable-misalignment`| flag | `False` | Skip inter-band shift simulation |
| `--disable-snr` | flag | `False` | Skip Gaussian SNR noise |
| `--disable-psf` | flag | `False` | Skip PSF kernel blur |
| `--disable-reflectance` | flag | `False` | Skip final radiance-to-reflectance conversion |
| `-v, --verbose` | flag | `False` | Verbose logging |

---

### 2. `degradation_ladder.py`

Located at: [`terra_sat_drift/data_simulation/degradation_ladder.py`](file:///home/elouann/terra-sat-drift/terra_sat_drift/data_simulation/degradation_ladder.py)

Generates systematic noise ladder datasets to evaluate model drift per isolated degradation factor across 4 severity levels:
- **`snr`**: Signal-to-noise ratio sweeps (`[40, 80]` down to `[5, 10]`)
- **`psf`**: Point Spread Function kernel sigma (`0.5` up to `4.0`)
- **`misalign`**: Inter-band registration standard deviation (`1.25` up to `10.0`)
- **`joint`**: All three factors simultaneously

Each variant shares the identical deterministic backbone (radiance, PAN band, 4.75 m spatial resampling, L1C reflectance conversion) and differs *only* by the isolated factor under test.

```bash
# Dry run to inspect the planned ladder without executing
python terra_sat_drift/data_simulation/degradation_ladder.py --dry-run

# Run full ladder across train, val, and test splits with 4 workers
python terra_sat_drift/data_simulation/degradation_ladder.py \
    --dataset-root /path/to/sen1floods11 \
    --output-root /path/to/ladder_output \
    --splits train val test \
    --workers 4
```

---

### 3. Utility Scripts (`data_simulation/utils/`)

* **[`sen1floods11_loader.py`](file:///home/elouann/terra-sat-drift/terra_sat_drift/data_simulation/utils/sen1floods11_loader.py)**: Standalone `Sen1Floods11S2Loader` class for loading S2 files directly from split CSVs without requiring `terratorch` or multi-modal datasets. Useful for lightweight data inspection, plotting, and quick statistics.
* **[`resample_labels.py`](file:///home/elouann/terra-sat-drift/terra_sat_drift/data_simulation/utils/resample_labels.py)**: Resamples original 10 m flood masks to Φ-sat-2's 4.75 m grid using nearest-neighbor interpolation to preserve class IDs.
* **[`psf_utils.py`](file:///home/elouann/terra-sat-drift/terra_sat_drift/data_simulation/utils/psf_utils.py)**: Helper to construct Gaussian blur kernels for multi-spectral bands.
* **[`fix_sen1floods_splits.py`](file:///home/elouann/terra-sat-drift/terra_sat_drift/data_simulation/utils/fix_sen1floods_splits.py)**: Strips modal suffixes from split text files to obtain clean root scene identifiers.

---

## Output Structure

The simulation mirrors the Sen1Floods11 v1.1 structure so that the output can be loaded directly by standard dataset loaders (e.g., `Sen1Floods11NonGeo(data_root=output_dir)`):

```text
<output_dir>/
└── v1.1/
    ├── simulation_config.json                         # Serialized SimulationConfig
    ├── logs/                                         # Execution log files
    │   └── ...
    └── data/
        └── flood_events/
            └── HandLabeled/
                └── S2Hand/                            # Simulated GeoTIFFs
                    ├── simulated_L1C_Bolivia_103757_S2Hand.tif
                    ├── simulated_L1C_Bolivia_129334_S2Hand.tif
                    └── ...
```

### Simulated Image Characteristics

* **Dimensions**: 1077 × 1077 pixels (resampled from 512 × 512 at 10 m)
* **Spatial Resolution**: 4.75 m (native Φ-sat-2 GSD)
* **Bands (8 total)**:
  `[0: Blue, 1: Green, 2: Red, 3: PAN, 4: RE1, 5: RE2, 6: RE3, 7: NIR]`
* **Data Type**: `float32` (reflectance values scaled 0–10,000)

---

## Simulation Pipeline Tasks (`simulation_pipeline.py`)

The pipeline constructs an `eo-learn` `EOWorkflow` chaining the following tasks:

1. **`LoadS2DatasetSampleTask`**: Ingests image tensor, spatial bbox, and temporal coordinates from the sample.
2. **`CalculateRadianceTask`**: Converts reflectances to physical radiances based on earth-sun distance and solar zenith angle.
3. **`AddPANBandTask`**: Computes synthetic panchromatic channel via weighted spectral combination and inserts it at index 3.
4. **`ResamplingTask`**: Resamples from 10 m input resolution to 4.75 m grid using nearest-neighbor interpolation.
5. **`BandMisalignmentTask`**: Simulates inter-band optical registration shifts via affine transformations (accounting for land vs. sea variance).
6. **`AlternativePhisatCalculationTask`**: Adds Gaussian sensor noise according to target SNR range and applies 2D Gaussian PSF blur.
7. **`CalculateReflectanceTask`**: Re-converts radiances back to reflectance for L1C outputs.
8. **`ExportToTiffTask`**: Exports multi-band GeoTIFF to the output path. Intermediate features are automatically pruned to conserve memory.
