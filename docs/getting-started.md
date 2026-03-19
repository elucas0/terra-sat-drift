# Getting Started

## Prerequisites

- Python 3.11+
- Project dependencies installed from `pyproject.toml`
- Input TIFF folders available under `tiff_folder/`

## Install

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Run the Drift Pipeline

Minimal run:

```bash
python main.py
```

Example with explicit configuration:

```bash
python main.py \
  --backbone-size large \
  --embedding-raw-dir tiff_folder/raw_tiff_update \
  --embedding-simulated-dir tiff_folder/simulated_custom_l2_tiff \
  --classification-raw-dir tiff_folder/raw_tiff_update \
  --classification-simulated-dir tiff_folder/simulated_custom_tiff
```

## View Documentation Locally

```bash
mkdocs serve
```

Then open the local URL shown in the terminal.

## Build Static Docs

```bash
mkdocs build
```
