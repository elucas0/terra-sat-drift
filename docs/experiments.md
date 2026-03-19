# Experiment Tracking

This project writes one JSON report per run to `experiments/`.

## Output Location

- Directory: `experiments/`
- File pattern: `drift_results_YYYYMMDD_HHMMSS.json`

## Recommended Workflow

1. Run `python main.py` with explicit parameters.
2. Confirm the output path printed at the end of the run.
3. Add a short note in your commit, PR, or lab notebook with:
   - command used
   - generated JSON filename
   - key summary metrics
4. Keep all generated JSON files under version control only when they are part of a reproducible report.

## Current Experiment Log

| JSON File | Timestamp (UTC) | Embedding Pairs | Class Flip Rate | Notes |
| --- | --- | ---: | ---: | --- |
| `experiments/drift_results_20260318_124626.json` | 2026-03-18T12:44:43Z | 256 | 0.00% | Legacy export format with per-file arrays in folder comparisons. |
| `experiments/drift_results_20260318_130314.json` | 2026-03-18T13:01:45Z | 256 | 0.00% | Summary-only folder metrics format. |

## Quick Comparison Checklist

- `example_2.avg_cosine_similarity` is close to `1.0`.
- `example_2.avg_mse_error` and `example_2.avg_mae_error` stay low.
- `example_3.class_flip_analysis.class_flip_rate` remains near `0.0`.
- `example_3.class_flip_analysis.avg_top3_consistency` stays high.

## Notes on Format Evolution

- Older runs can contain full per-file arrays (`results`) for folder comparisons.
- Current runs store summary-only folder outputs to keep reports lightweight.
