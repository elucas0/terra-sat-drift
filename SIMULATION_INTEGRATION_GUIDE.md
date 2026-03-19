# Φ-sat-2 Simulation Integration Guide

This guide explains how to use the new on-the-fly simulation pipeline integrated into terra-sat-drift. This enables **dynamic generation of simulated Φ-sat-2 imagery from cached Sentinel-2 L1C data** during drift analysis experiments.

## Architecture Overview

```
[Raw S2 L1C .tiff Cache] 
         ↓
[Configurable Simulation Pipeline]
    - Radiance conversion
    - Panchromatic band synthesis
    - Band misalignment
    - SNR noise simulation
    - PSF filtering
    - L1C reflectance conversion
         ↓
[Simulated Φ-sat-2 .tiff Output]
         ↓
[Drift Analysis vs. Raw]
    - Embedding similarity
    - Class prediction stability
    - Class flip detection
         ↓
[Timestamped JSON Results]
```

## Quick Start

### 1. Prepare Raw S2 Cache Directory

Place your raw Sentinel-2 L1C .tiff files in a cache directory:

```bash
mkdir -p tiff_folder/raw_s2_cache
# Copy your S2 L1C .tiffs here (7 or 8 bands)
```

**Expected format**: GeoTIFF files with 7 Sentinel-2 bands (B02, B03, B04, B08, B05, B06, B07) or 8 bands (with panchromatic).

### 2. Run Simulation Mode with Defaults

```bash
python main.py --mode simulation \
  --backbone-size large \
  --raw-s2-cache tiff_folder/raw_s2_cache \
  --simulated-output-dir tiff_folder/simulated_dynamic
```

This will:
1. Load all `.tiff` files from the cache directory
2. Apply all simulation steps in sequence
3. Save simulated Φ-sat-2 outputs to `simulated_dynamic/`
4. Analyze drift between raw and simulated pairs
5. Export results to `experiments/drift_results_simulation_with_analysis_YYYYMMDD_HHMMSS.json`

### 3. Selective Simulation Steps

To skip certain processing steps (e.g., band misalignment for baseline comparison):

```bash
python main.py --mode simulation \
  --backbone-size large \
  --enable-radiance \
  --enable-panchromatic \
  --enable-misalignment \
  --enable-snr \
  --disable-psf  # Skip PSF filtering
  --enable-reflectance
```

**Available toggles**:
- `--enable-radiance` / `--disable-radiance` (default: enabled)
- `--enable-panchromatic` / `--disable-panchromatic` (default: enabled)
- `--enable-misalignment` / `--disable-misalignment` (default: enabled)
- `--enable-snr` / `--disable-snr` (default: enabled)
- `--enable-psf` / `--disable-psf` (default: enabled)
- `--enable-reflectance` / `--disable-reflectance` (default: enabled)

## Advanced: Configuration Files

For complex experiments with many parameter variations, use JSON configuration files.

### Create a Config File

```json
{
  "steps": {
    "radiance": true,
    "add_panchromatic": true,
    "band_misalignment": true,
    "snr_simulation": true,
    "psf_filtering": true,
    "reflectance_conversion": true
  },
  "s2_source_dir": "tiff_folder/raw_s2_cache",
  "output_dir": "tiff_folder/simulated_dynamic",
  "processing_level": "L1C",
  "misalignment_std_land": 1.0,
  "misalignment_std_sea": 6.0,
  "psf_kernel_sigma": 1.0,
  "radiance_reference": 100.0,
  "phisat2_exec_path": null,
  "snr_values": {
    "B02": 15,
    "B03": 15,
    "B04": 15,
    "PAN": 10,
    "B08": 20,
    "B05": 15,
    "B06": 15,
    "B07": 15
  }
}
```

Save as `experiments/config_baseline.json` and run:

```bash
python main.py --mode simulation \
  --backbone-size large \
  --simulation-config experiments/config_baseline.json
```

### Parameter Sweep Example

Create multiple config files for different SNR levels:

```bash
# config_snr_high.json - Less noise
{"snr_values": {"B02": 30, "B03": 30, "B04": 30, "PAN": 20, ...}}

# config_snr_low.json - More noise  
{"snr_values": {"B02": 10, "B03": 10, "B04": 10, "PAN": 5, ...}}
```

Then run batch:

```bash
for config in experiments/config_snr_*.json; do
  python main.py --mode simulation \
    --backbone-size large \
    --simulation-config "$config"
done
```

## Simulation Pipeline Details

### Step 1: Radiance Conversion
Converts reflectance → radiance using Sentinel-2 solar irradiance values.
- Input: S2 reflectance [0-1]
- Output: Radiance [0-300] (typical range)

### Step 2: Panchromatic Band
Synthesizes panchromatic band as weighted combination of spectral bands.
- Weights: [0.05, 0.05, 0.15, 0.2, 0.25, 0.15, 0.15]
- Inserted at band index 3
- Output: 8-band imagery

### Step 3: Band Misalignment
Simulates satellite jitter causing inter-band shifts (L1B/L1C model).
- Red band (index 2) is reference
- Other bands: random amplitude N(0, std_land) + random angle
- Applied via `cv2.warpAffine` with nearest-neighbor interpolation

### Step 4: SNR Simulation
Adds thermal/readout noise based on signal-to-noise ratios.
- For each band: `noise = N(0, signal_std / SNR)`
- Default SNR values calibrated to typical Sentinel-2 spec
- Customizable via JSON config

### Step 5: PSF Filtering
Simulates sensor optics via Gaussian filtering.
- Default kernel sigma: 1.0 pixel
- Approximates Module Transfer Function (MTF) loss
- Alternative: Can use external `phisat2_unix.bin` for higher fidelity (future enhancement)

### Step 6: Reflectance Conversion
Converts radiance → reflectance for L1C output.
- Output: Reflectance [0-1] matching S2 L1C format

## Output Format

### Simulation Output Directory
```
tiff_folder/simulated_dynamic/
├── simulated_S2_image_001.tiff
├── simulated_S2_image_002.tiff
└── simulated_S2_image_003.tiff
```

### Results JSON
```json
{
  "metadata": {
    "timestamp_utc": "2026-03-19T15:30:45Z",
    "simulation_steps": {
      "radiance": true,
      "add_panchromatic": true,
      ...
    },
    "simulation_successful": 5,
    "simulation_failed": 0
  },
  "simulation_results": {
    "successful": ["simulated_file_1.tiff", ...],
    "failed": []
  },
  "drift_analysis": {
    "summary": {
      "total_pairs_analyzed": 5,
      "avg_embedding_cosine_similarity": 0.8234,
      "class_flip_rate": 0.05,
      "avg_probability_change": 0.0234
    },
    "detailed_results": [
      {
        "pair_index": 1,
        "raw_file": "...",
        "simulated_file": "...",
        "drift": {
          "spectral_comparison": {...},
          "embedding_comparison": {...},
          "classification_comparison": {...}
        }
      },
      ...
    ]
  }
}
```

## Integration with phisat2 Utilities

The simulation pipeline currently uses **pure Python implementations** for:
- Radiance/reflectance conversion
- PAN band synthesis
- Band misalignment
- SNR simulation (Gaussian noise)
- PSF filtering (Gaussian blur)

### Future: External Binary Support
To use the high-fidelity phisat2_unix.bin for SNR/PSF:

```json
{
  "phisat2_exec_path": "phisat-2/executables/phisat2_unix.bin",
  ...
}
```

(Not yet integrated; requires EOTask wrapper)

## Scaling: Batch Processing Multiple S2 Images

### Scenario: Process 100 Sentinel-2 images

```bash
# Organize cache directory
ls tiff_folder/raw_s2_cache/
# s2_image_001.tiff
# s2_image_002.tiff
# ...
# s2_image_100.tiff

# Run simulation once (processes all)
python main.py --mode simulation \
  --backbone-size large \
  --raw-s2-cache tiff_folder/raw_s2_cache
```

The pipeline will:
1. Iterate over all `.tiff` files
2. Simulate each one independently
3. Analyze drift for each raw-simulated pair
4. Aggregate statistics across all 100 pairs
5. Save comprehensive results to single JSON file

**Performance**: ~30-60 sec per image (depending on image size and backbone size).

### Parallel Execution (Future Enhancement)

Once you have many simulation outputs, you can use EOExecutor for parallel drift analysis:

```python
from terra_sat_drift import DriftPipeline

pipeline = DriftPipeline(backbone_size="small")  # Faster for parallel
# Process 100 pairs in parallel across 4 CPU cores
```

## Troubleshooting

### Empty simulation output
**Problem**: `simulated_dynamic/` is empty after running simulation mode.

**Causes**:
- Raw S2 cache directory not found
- .tiff files have wrong number of bands (not 7 or 8)
- Rasterio not installed

**Fix**:
```bash
ls tiff_folder/raw_s2_cache/  # Verify files exist
python -c "import rasterio; print(rasterio.__version__)"  # Check rasterio
```

### Band misalignment looks wrong
**Problem**: Simulated output looks like it has missing patches.

**Explanation**: Band misalignment uses nearest-neighbor interpolation, which may leave zero-padded borders. This is intentional (matches L1A processing). To disable:
```bash
python main.py --mode simulation --disable-misalignment
```

### High memory usage with large images
**Problem**: Out of memory on batch processing.

**Solution**: Process images in smaller batches or use smaller backbone:
```bash
python main.py --mode simulation --backbone-size small
```

## Example Workflows

### Workflow 1: Baseline simulation →  drift analysis
```bash
# Generate baseline
python main.py --mode simulation \
  --backbone-size large \
  --raw-s2-cache tiff_folder/raw_s2_cache \
  --simulated-output-dir tiff_folder/simulated_baseline

# Results in: experiments/drift_results_simulation_with_analysis_*.json
```

### Workflow 2: Ablation study (step-by-step disable)
```bash
# Full pipeline
python main.py --mode simulation --simulation-config experiments/config_full.json

# Without misalignment
python main.py --mode simulation --simulation-config experiments/config_no_misalignment.json

# Without SNR
python main.py --mode simulation --simulation-config experiments/config_no_snr.json

# Compare results to identify which step most affects drift metrics
```

### Workflow 3: Robustness across backbone sizes
```bash
for backbone in tiny small base large; do
  python main.py --mode simulation \
    --backbone-size "$backbone" \
    --raw-s2-cache tiff_folder/raw_s2_cache \
    --simulated-output-dir "tiff_folder/simulated_${backbone}"
done

# Results in: 4 independent experiment JSONs tracking drift by backbone size
```

## Tips & Best Practices

1. **Cache consistently**: Always use the same raw S2 cache for ablation studies to isolate simulation effects.

2. **Use smaller backbones for quick tests**: `tiny` and `small` run ~10x faster than `large`, good for parameter tuning.

3. **Save configs for reproducibility**: Version-control your config JSONs alongside results.

4. **Monitor memory**: Large images (>10k × 10k) may require small backbone or batch processing.

5. **Verify simulation outputs**: Check a few simulated GeoTIFFs in QGIS to validate visual appearance.

## API Reference

### SimulationConfig

```python
from terra_sat_drift import SimulationConfig

config = SimulationConfig(
    steps=SimulationSteps(
        radiance=True,
        add_panchromatic=True,
        band_misalignment=True,
        snr_simulation=True,
        psf_filtering=True,
        reflectance_conversion=True
    ),
    s2_source_dir="tiff_folder/raw_s2_cache",
    output_dir="tiff_folder/simulated_dynamic",
    psf_kernel_sigma=1.0,
    misalignment_std_land=1.0
)

# Save for later
config.save_json("experiments/my_config.json")

# Load from file
config = SimulationConfig.from_json("experiments/my_config.json")
```

### SimulationPipeline

```python
from terra_sat_drift import SimulationPipeline

pipeline = SimulationPipeline(config)

# Simulate single file
success = pipeline.simulate_single_file(
    "tiff_folder/raw_s2_cache/s2_001.tiff",
    "tiff_folder/simulated_dynamic/simulated_s2_001.tiff"
)

# Batch process all files
results = pipeline.batch_simulate_from_source_dir(
    source_dir="tiff_folder/raw_s2_cache",
    pattern="*.tiff"
)
print(f"Successful: {len(results['successful'])}, Failed: {len(results['failed'])}")
```

### DriftPipeline (Extended)

```python
from terra_sat_drift import DriftPipeline, SimulationConfig

pipeline = DriftPipeline(num_classes=10, backbone_size="large")
pipeline.validate()

# New method: run simulation + drift analysis
pipeline.run_simulation_experiments(
    simulation_config=config,
    raw_s2_source_dir="tiff_folder/raw_s2_cache",
    simulated_output_dir="tiff_folder/simulated_dynamic"
)
```

## Next Steps

1. **Prepare S2 cache**: Copy your Sentinel-2 L1C .tiff files to `tiff_folder/raw_s2_cache/`
2. **Test simulation mode**: Run quick test with 1-2 images
3. **Review output JSON**: Verify drift metrics make sense
4. **Run ablation study**: Disable each step to understand impact
5. **Scale up**: Process all your S2 imagery for large-scale analysis
