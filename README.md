# TerraSat Drift

TerraSat Drift is a reproducible pipeline for analyzing drift between paired Earth observation TIFF files using TerraMind backbones from `terratorch`.

It compares inputs at three levels:

- Spectral drift: band-level summary statistics across S2 bands.
- Embedding drift: encoder feature similarity and reconstruction-style error metrics.
- Classification drift: class stability, confidence changes, and class-flip rates.

Each run exports a timestamped JSON report in `experiments/`.

## Repository Layout

```text
.
├── main.py                      # CLI entrypoint
├── terra_sat_drift/
│   ├── model_service.py         # TerraMind model creation and inference helpers
│   ├── drift_analysis.py        # Drift computations for single pairs and folders
│   ├── reporting.py             # Console report formatting
│   └── pipeline.py              # End-to-end orchestration and JSON export
├── tiff_folder/                 # Input TIFF folders
├── experiments/                 # Generated drift_results_*.json files
├── docs/                        # MkDocs project documentation
└── mkdocs.yml                   # Docs site config
```

## Requirements

- Python `>=3.11`
- Dependencies listed in `pyproject.toml`:
	- `terratorch>=1.2.4`
	- `torchgeo>=0.7.0`

## Installation

Using `venv` + `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

If you use `uv`, install from the project root with your preferred `uv` workflow.

## Usage

Run with defaults:

```bash
python main.py
```

Run with explicit backbone and folder settings:

```bash
python main.py \
	--backbone-size large \
	--num-classes 10 \
	--embedding-raw-dir tiff_folder/raw_tiff_update \
	--embedding-simulated-dir tiff_folder/simulated_custom_l2_tiff \
	--classification-raw-dir tiff_folder/raw_tiff_update \
	--classification-simulated-dir tiff_folder/simulated_custom_tiff \
	--file1-pattern BANDS_RES-GRID \
	--file2-pattern PHISAT2-BANDS-GRID \
	--suffix .tiff
```

### CLI Parameters

- `--backbone-size {tiny,small,base,large}`
- `--num-classes INT`
- `--embedding-raw-dir PATH`
- `--embedding-simulated-dir PATH`
- `--classification-raw-dir PATH`
- `--classification-simulated-dir PATH`
- `--file1-pattern TEXT`
- `--file2-pattern TEXT`
- `--suffix TEXT`

## Output Reports

After each run, a JSON report is written to:

- `experiments/drift_results_YYYYMMDD_HHMMSS.json`

Report structure includes:

- `metadata`: run configuration and matching rules.
- `example_1`: detailed single-pair drift result.
- `example_2`: folder-level embedding summary metrics.
- `example_3`: folder-level classification summary metrics.

## Backbone Notes

The project supports TerraMind sizes:

- `tiny`
- `small`
- `base`
- `large`

Each size uses the corresponding encoder neck indices internally.

## Documentation Site

Project docs are under `docs/` and built with MkDocs.

Serve locally:

```bash
mkdocs serve
```

Build static site:

```bash
mkdocs build
```

Key docs pages:

- `docs/getting-started.md`
- `docs/cli-reference.md`
- `docs/experiments.md`
- `docs/results-json-format.md`