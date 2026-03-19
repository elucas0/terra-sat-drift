# Results JSON Format

Each run exports a single JSON document with this top-level layout:

```json
{
  "metadata": { "...": "..." },
  "example_1": { "...": "..." },
  "example_2": { "...": "..." },
  "example_3": { "...": "..." }
}
```

## `metadata`

- `timestamp_utc`: export timestamp.
- `embedding_raw_dir`, `embedding_simulated_dir`.
- `classification_raw_dir`, `classification_simulated_dir`.
- `file1_pattern`, `file2_pattern`.
- `suffix`.

## `example_1` (single pair detailed analysis)

- `status`: `ok` or `skipped`.
- `raw_file`, `simulated_file`.
- `drift` (when status is `ok`):
  - `spectral_drift` (per-band stats and summary)
  - `embedding_drift` (similarity and error metrics)
  - `class_drift` (class prediction stability)

## `example_2` (folder embedding summary)

Current format stores summary metrics only:

- `status`
- `pairs_processed`
- `avg_cosine_similarity`
- `avg_mse_error`
- `avg_mae_error`
- `avg_pixel_max_diff`
- `per_layer_avg_cosine_similarity` (array)

## `example_3` (folder classification summary)

- `status`
- `pairs_processed`
- `class_flip_analysis`:
  - `total_pairs`
  - `class_flips_count`
  - `class_flip_rate`
  - `avg_probability_change`
  - `max_probability_change`
  - `avg_top3_consistency`

## Backward Compatibility

Some older files can include verbose per-file arrays under `example_2.results` or `example_3.results`. Consumers should ignore unknown fields and prefer summary keys when present.
