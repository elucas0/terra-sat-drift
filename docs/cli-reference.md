# CLI Reference

Run from repository root:

```bash
python main.py [OPTIONS]
```

## Core Model Options

- `--backbone-size {tiny,small,base,large}`
: Selects the TerraMind variant. Default: `large`.
- `--num-classes INT`
: Number of classes for the classification head. Default: `10`.

## Folder Comparison Options

- `--embedding-raw-dir PATH`
: Source folder used in embedding comparison batch. Default: `tiff_folder/raw_tiff_update`.
- `--embedding-simulated-dir PATH`
: Target folder used in embedding comparison batch. Default: `tiff_folder/simulated_custom_l2_tiff`.
- `--classification-raw-dir PATH`
: Source folder used in classification comparison batch. Default: `tiff_folder/raw_tiff_update`.
- `--classification-simulated-dir PATH`
: Target folder used in classification comparison batch. Default: `tiff_folder/simulated_custom_l2_tiff`.

## Pair-Matching Options

- `--file1-pattern TEXT`
: Substring expected in source filenames. Default: `BANDS_RES-GRID`.
- `--file2-pattern TEXT`
: Replacement substring used to locate target filenames. Default: `PHISAT2-BANDS-GRID`.
- `--suffix TEXT`
: File suffix used for scans. Default: `.tiff`.

## Example Commands

Large backbone with default paths:

```bash
python main.py --backbone-size large
```

Small backbone with custom classification target folder:

```bash
python main.py \
  --backbone-size small \
  --classification-simulated-dir tiff_folder/simulated_tiff
```
