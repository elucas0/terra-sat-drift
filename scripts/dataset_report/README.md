# Triplets dataset report

Two scripts that characterise the PhiSat-2 / Sentinel-2B `triplets_v1` dataset and
write thesis-ready figures (PNG at 300 dpi + PDF for LaTeX) plus machine-readable
stats.

```bash
# manifest only -- runs in seconds
python scripts/dataset_report/analyze_manifest.py

# samples the HDF5 -- a few minutes at the default 500 patches
python scripts/dataset_report/analyze_spectral.py --n-patches 500
```

Both default to `outputs/dataset_report/`. `plot_style.py` holds the shared
palette and style; figure colours are documented there, including why land-cover
figures use the official ESA WorldCover legend instead of the categorical palette.

## Outputs

| File | What it shows |
|---|---|
| `fig_acquisition` | PhiSat-2/Sentinel-2 time lag; acquisition timeline |
| `fig_clouds` | ECDF of thick and thin cloud cover |
| `fig_landcover` | Class prior by pixel and by dominant class |
| `fig_geography` | Global patch density; Köppen-Geiger climate groups |
| `fig_splits` | Class balance across splits; patches per product |
| `fig_band_correspondence` | Rank correlation of each stored band against each Sentinel-2 band |
| `fig_band_boxplots` | **Per-band value distribution per sensor** (log DN) |
| `fig_band_boxplots_normalized` | The same after per-sensor standardisation |
| `fig_spectral_signature` | Across-band shape with absolute scale divided out |
| `fig_sensor_scatter` | Per-band co-registered pixel agreement + OLS fit |
| `fig_patch_examples` | RGB previews of the three views of the same ground |
| `manifest_summary.json` | Headline numbers |
| `spectral_stats.csv` | Per-band stats incl. floor/ceiling clipping fractions |
| `domain_stats_measured.json` | sqrt-space mean/std/clip, in `DOMAIN_STATS` form |
| `sensor_transfer_fit.csv` | Per-band slope, intercept, Pearson r, Spearman rho |

## What the figures found

Numbers below are from 253,228 patches (manifest) and 400,000 pixels sampled from
500 patches at ≤5% cloud (spectral).

**The dataset.** 253,228 patches over 1,299 PhiSat-2 products, Aug 2024 – Apr
2026, latitude −54 to +76. Median acquisition lag **6.8 days** (max 14.9). 73% of
patches are below 5% total cloud. Land cover is well spread across the five major
classes (grassland 20.8%, tree cover 20.0%, bare/sparse 16.2%, built-up 14.4%,
cropland 13.4% of labelled pixels) with a long tail of rare classes
(moss/lichen 0.3%, mangroves 0.5%, snow/ice 0.5%). Only 0.2% of pixels are
unlabelled. Climate coverage is **arid-biased: 41% Köppen group B**, which is
consistent with the large bare/sparse share.

**Band assignment is correct, and now verified from the data.**
`fig_band_correspondence` recovers the documented layout without assuming it: for
the simulated view every stored band 1–7 matches its Sentinel-2 counterpart at
ρ = 0.98–1.00, and stored band 0 correlates broadly across the visible bands
while falling away toward NIR — the panchromatic signature. So the `[1:8]` slice
used throughout the codebase does yield Blue, Green, Red, RE1, RE2, RE3, NIR.

**The real domain gap is much larger than the simulated one.** Same figure, right
panel: real PhiSat-2 agrees with Sentinel-2 at only **ρ ≈ 0.47–0.61**, against
the simulation's ρ ≈ 1.00. Per-band Pearson r on raw values is 0.12–0.58
(`sensor_transfer_fit.csv`). Consequence for the thesis: the simulated view is
radiometrically *almost exactly* Sentinel-2, so training or validating on `sim`
tests very little of what makes `real` hard.

**The simulated view keeps Sentinel-2's radiometry.** Its sqrt-space means match
s2b's to ~0.01 (49.24 vs 49.24, 48.72 vs 48.72, …) while its standard deviation
is slightly *lower* in every band — the signature of a spatial blur applied to
Sentinel-2 with radiometry untouched. It therefore needs Sentinel-2-like
normalisation statistics (`clip = 100.0`), **not** real-PhiSat-2 ones
(`clip = 38.729`); an earlier provisional entry in `DOMAIN_STATS["sim"]` had this
wrong by roughly 25× in the mean and has been corrected.

**Scale and shape differ sharply between real and Sentinel-2.** Real PhiSat-2 DNs
are ~25× lower with a far wider relative spread (IQR spanning over a decade,
against Sentinel-2's tight box). The across-band *shape* also differs: real peaks
in Red and bottoms in NIR, where Sentinel-2 rises monotonically into the red edge.
This is not a band error — `real` is uncalibrated level-1 DN, where per-band
magnitude reflects detector gain rather than reflectance, and the dataset
specification notes the two platforms' spectral response functions differ.

**Per-band standardisation removes location and scale, not shape.** In
`fig_band_boxplots_normalized` the three sensors' boxes align on zero, but real
PhiSat-2 stays visibly right-skewed (median below zero, upper whisker to +3.5)
where s2b and sim are near-symmetric within ±2. That residual, non-affine
difference is the part input normalisation cannot reach — the case for
representation-level domain adaptation.

**Red-edge bands saturate.** In the real view, **2.06% of RE1 and 2.15% of RE2
pixels sit at the 12-bit ceiling of 4095**, and RE2/RE3 have 0.30%/0.36% of pixels
at zero. Saturated pixels carry no gradient-useful signal; `spectral_stats.csv`
has the per-band fractions.

**The simulator under-models PhiSat-2's blur.** In `fig_patch_examples`
co-registration is visibly tight across all three views, but the real view is
consistently *softer* than the simulated one, most clearly on urban texture —
i.e. the simulated MTF is sharper than the real sensor's.

## Caveat on the current split

`fig_splits` reproduces the seed-42 patch-level 80/10/10 split used by the dataset
classes. The class prior is stable across splits, but the split is drawn over
*patches* while patches come in large per-product blocks (median 209 per product,
max 713) from a single acquisition: **all 1,287 test products also appear in
train**. Test metrics on this split therefore measure interpolation within seen
scenes, not generalisation to new ones. A product-level (`product_id`) or
geographic split is the fix if the thesis needs a generalisation claim.

## Notes and limitations

- Köppen groups assume the Beck et al. v3 (1991–2020) 1–30 legend documented for
  this dataset, with 0 treated as no data (11,225 patches).
- The spectral figures default to `--max-cloud 5`, because cloud is correlated
  across sensors and inflates agreement. Pass a large value to disable; cloud has
  its own figure in the manifest script.
- `s2b` is bicubic-upsampled from 10 m to the 4.75 m grid, so it is intrinsically
  smoother than a native-resolution sensor would be.
- Sampling is random over patches; increase `--n-patches` for tighter estimates.
  Statistics printed by the script are sample estimates, not the whole-dataset
  values in the dataset specification.
