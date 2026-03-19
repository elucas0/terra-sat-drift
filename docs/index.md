# TerraSat Drift Documentation

This repository evaluates drift between paired TIFF files using a TerraMind backbone and exports reproducible experiment results as JSON files.

## Scope

- Compare spectral signatures across 7 Sentinel-2 bands.
- Compare embedding stability from TerraMind encoder features.
- Compare classification stability (class flips and confidence changes).
- Export run summaries to `experiments/drift_results_YYYYMMDD_HHMMSS.json`.

## Documentation Map

- `Getting Started`: environment setup and first run.
- `CLI Reference`: all runtime parameters.
- `Experiment Tracking`: workflow to record and interpret experiments.
- `Results JSON Format`: fields written by the pipeline.

## Quick Start

```bash
python main.py --backbone-size large
```

After execution, check `experiments/` for the generated JSON report.
