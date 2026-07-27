# Cross-sensor contrastive knowledge distillation

Distilling the TerraMind ViT GeoFM into a CNN student that is **invariant to the
sensor** — same latent representation for a PhiSat-2 patch and its co-registered
Sentinel-2 patch — using the `triplets_v1` dataset.

Implementation: [kd_contrastive_module.py](../terra_sat_drift/model_tasks/kd_contrastive_module.py),
[losses/contrastive.py](../terra_sat_drift/model_tasks/losses/contrastive.py),
[dataset_paired_triplets_lulc.py](../terra_sat_drift/dataset/dataset_paired_triplets_lulc.py).
Training entry point: [kd_lulc_contrastive.py](../terra_sat_drift/kd_lulc_contrastive.py).

## 1. What was wrong with the naive baseline

[kd_module.py](../terra_sat_drift/model_tasks/kd_module.py) minimises
`MSE(student_logits, teacher_logits)` on one domain at a time. Two structural
problems:

1. **The teacher is queried outside its domain.** TerraMind is pretrained on
   Sentinel-2. Running it on PhiSat-2 imagery distils a *degraded* teacher, so
   the student inherits the teacher's own domain-shift error on top of its own.
2. **Logit MSE cannot produce domain invariance.** It constrains only the final
   class scores, pointwise. Nothing in the objective states that the same ground
   location seen through two sensors should have the same internal
   representation — which is the property actually wanted.

## 2. The structural fact the method exploits

`phisat2_s2b_dataset_v1.h5` stores three **spatially co-registered** views under
a shared patch index (259,150 patches, 256×256):

| key | shape | content |
|---|---|---|
| `real/images` | (N, 8, 256, 256) | real PhiSat-2 L1 |
| `sim/images` | (N, 8, 256, 256) | PhiSat-2 simulated from Sentinel-2 |
| `s2b/images` | (N, 7, 256, 256) | Sentinel-2B |

Because the views are pixel-aligned and share one WorldCover label map, positive
pairs are **free**: no augmentation heuristic has to invent them. This is the
condition that makes the four published objectives below directly applicable.

## 3. The objective

With `x_s` the Sentinel-2 view, `x_t` the PhiSat-2 view, `y` the shared label
map, `S` the student and `T` the frozen teacher:

```
L =  w_task · [ CE(S(x_t), y) + CE(S(x_s), y) ]                      task
  +  w_kd   · [ KD(S(x_t), T(x_s)) + KD(S(x_s), T(x_s)) ]            (1)
  +  w_inst · NT-Xent( g(S_enc(x_s)), g(S_enc(x_t)) )                (2)
  +  w_pix  · PixelProto( h(S_dec(x_t)), h(S_dec(x_s)), y )          (3)
  +  w_crd  · InfoNCE( g(S_enc(x_·)), g_T(T_enc(x_s)) )              (4)
```

`g` and `h` are projection heads, discarded at inference (SimCLR).

### (1) Cross-modal supervision transfer

The teacher is evaluated **only on the Sentinel-2 view**, where it is
trustworthy, and its soft dense predictions supervise the student on *both*
views. Transferring a teacher's supervision onto a paired second modality is the
construction of **Gupta, Hoffman & Malik, "Cross Modal Distillation for
Supervision Transfer", CVPR 2016**. Distillation itself is the
temperature-scaled KL of **Hinton, Vinyals & Dean (2015)**; the direct logit-MSE
variant analysed by **Kim et al., IJCAI 2021** is kept as `kd_mode="mse"`, which
recovers the old baseline's loss exactly.

Distillation runs on *all* pixels by default, including unlabelled ones — that
is where distillation adds supervision the task loss cannot
(`--kd-labeled-only` restricts it).

### (2) Instance-level cross-sensor contrast — the invariance driver

Symmetric NT-Xent (**SimCLR**, Chen et al. ICML 2020; **InfoNCE**, van den Oord
et al. 2018) where the two "views" are **two sensors** rather than two
augmentations. This is the standard EO reading of contrastive learning:
**SeCo** (Mañas et al., ICCV 2021) uses seasonal views of a location, **CROMA**
(Fuller et al., NeurIPS 2023) uses spatially aligned radar/optical pairs.

This is the term that directly enforces *"same representation despite covariate
shift"*. The in-batch negatives are what stop the trivial constant solution.
Default is SimCLR's 2N−2 negative set; `--cross-view-negatives-only` switches to
the CLIP/CROMA N−1 set.

### (3) Semantic-guided pixel contrast

**SePiCo** (Xie et al., TPAMI 2023). Per-pixel embeddings are pulled toward an
EMA class centroid **shared across both domains** and pushed away from the other
centroids, in the multi-positive spirit of **SupCon** (Khosla et al., NeurIPS
2020). Prototypes are EMA-maintained as in **ProDA** (Zhang et al., CVPR 2021).

Two reasons this term matters and (2) alone is not enough:

- Term (2) aligns whole patches. A segmentation encoder additionally needs
  pixel-level, **class-conditional** alignment, which marginal-alignment methods
  (MMD, adversarial DA — including the existing
  [train_encoder_mmd.py](../terra_sat_drift/model_tasks/domain_adaptation/train_encoder_mmd.py))
  cannot deliver: matching marginals permits class-to-class mismatch.
- It removes the **false-negative pathology** of plain instance contrast on land
  cover. In EO, many distinct patches are the same class ("tree cover" dominates
  much of this manifest), and instance contrast wrongly pushes them apart. Using
  labels to define positives is the SupCon correction.

### (4) Contrastive representation distillation (optional, off by default)

**CRD** (Tian, Krishnan & Isola, ICLR 2020). Logit KD matches only the marginal
output distribution and discards the teacher's *relational* structure — which
inputs it considers similar. That structure is the appropriate transfer channel
here because teacher and student are **different architectures** (ViT vs CNN)
with incomparable feature layouts, so pointwise feature regression does not
apply.

> **Deviation flagged in code:** the original CRD draws many negatives from a
> memory bank with an NCE partition-function correction. The implementation uses
> the in-batch symmetric InfoNCE variant instead — no memory bank, fewer
> negatives.

### Ramp-up

Contrastive weights are linearly ramped over the first epochs (**Laine & Aila,
ICLR 2017**): the class prototypes and projection heads are meaningless at step
0, so full weight immediately injects noise into the encoder.

## 4. Supporting design decisions

**Per-domain input normalisation.** Each domain is standardised with its own
measured statistics (`DOMAIN_STATS`), removing the first-order (per-band
mean/variance) part of the shift in input space, so the representation losses
handle the harder non-affine part (PSF, band response, view geometry).

**Co-registration-safe augmentation.** Geometric augmentation is sampled *once*
and replayed across both views and the mask via albumentations
`additional_targets`. An independently sampled transform per view would silently
turn true positives into misaligned pairs and break term (3) entirely.
Photometric augmentation is deliberately omitted — the sensor difference *is*
the radiometric perturbation being studied, and synthetic jitter would confound
the measurement.

**Model selection tracks target-domain IoU, not total loss.** The checkpoint
callback, early stopping and the `ReduceLROnPlateau` scheduler all monitor
`val/student_target_iou`. The total loss blends five terms under a ramp-up
schedule, so it decreases for reasons unrelated to model quality and is not a
stable selection signal. Override via the module's `lr_monitor` /
`lr_monitor_mode` (pass `lr_monitor=None` to train without a scheduler, e.g.
when running without a validation loader).

**Domain-specific BatchNorm** (`--use-dsbn`, **Chang et al., CVPR 2019**). The
UNet student is BatchNorm-heavy. With shared BN there is a train/test
inconsistency: during training each domain's forward pass is normalised by its
*own* batch statistics, but inference uses a single blended running average, so
the encoder never sees at test time the normalisation it was optimised under.
DSBN gives each domain its own statistics and affine parameters while all
convolutional weights stay shared.

> **Trade-off:** with DSBN the deployed model is no longer literally one
> parameter set — you must select the target branch at inference
> (`set_domain(model, 1)`). Leave it off if a strictly single-branch encoder is
> required.

## 5. How to tell whether it worked

Marginal-gap numbers alone are not evidence, so the module logs a diagnostic set
that must be read **together**:

| metric | meaning | want |
|---|---|---|
| `cos_paired` | similarity of the two sensor views of the *same* patch | → 1 |
| `cos_unpaired` | similarity across *different* patches | stay low |
| `align_margin` | `cos_paired − cos_unpaired` | **the number that matters** |
| `retrieval_top1` | fraction of PhiSat-2 patches whose nearest S2 neighbour in-batch is its own pair | → 1 |
| `mmd2` | multi-kernel MMD between the domains' embedding clouds (Gretton et al. 2012) | ↓ |
| `student_target_iou` | task performance on PhiSat-2 | ↑ |
| `domain_gap_iou` | `source_iou − target_iou`, same patches | → 0 |

**`cos_paired` alone is not evidence of invariance.** If `cos_paired` and
`cos_unpaired` both approach 1 the encoder has collapsed to a constant and the
alignment is vacuous — that is why `align_margin` and `retrieval_top1` are
logged next to it. Watch these on the very first runs.

`domain_gap_iou` is the headline end-to-end number: how much worse the student is
on PhiSat-2 than on the Sentinel-2 view of the *very same patches*. Any residual
value is domain gap that survived training.

## 6. Ablation ladder

Each line adds one term. All configurations share the seed-42 80/10/10 split
used by the single-domain datasets, so results are directly comparable with
existing runs.

```bash
cd terra_sat_drift

# ~ the old baseline: logit MSE only
python kd_lulc_contrastive.py --w-kd 1 --w-instance 0 --w-pixel 0 --kd-mode mse
# + temperature-scaled KL, teacher queried on S2
python kd_lulc_contrastive.py --w-kd 1 --w-instance 0 --w-pixel 0
# + instance-level cross-sensor contrast
python kd_lulc_contrastive.py --w-kd 1 --w-instance 0.5 --w-pixel 0
# + semantic-guided pixel contrast
python kd_lulc_contrastive.py --w-kd 1 --w-instance 0.5 --w-pixel 0.1
# + domain-specific BatchNorm
python kd_lulc_contrastive.py --w-kd 1 --w-instance 0.5 --w-pixel 0.1 --use-dsbn
```

Add `--teacher-ckpt <path>` throughout — without it the script warns and distils
an untrained teacher.

### Weight scale note

The NT-Xent gradient scales roughly as 1/τ, so with the default `τ = 0.1` the
instance term's raw gradient magnitude is ~10× larger than the task term's
(measured: ≈46 vs ≈0.9 in an isolated-term gradient probe). That is why
`w_instance` defaults to 0.5 rather than 1.0. If you change `--instance-temperature`,
rescale `--w-instance` in the opposite direction.

### Batch size

Both contrastive terms draw negatives from within the batch, so `--batch-size`
sets the number of negatives and is a *method* hyperparameter here, not just a
memory knob. Each patch also costs two student forward passes. Prefer the largest
batch that fits; incomplete trailing batches are dropped for this reason.

## 7. Known gaps

- **`sim` domain statistics are provisional.** `DOMAIN_STATS["sim"]` currently
  reuses the real-PhiSat-2 values. Run
  [calculate_spectral_statistics.py](../scripts/calculate_spectral_statistics.py)
  over `sim/images` and replace them before drawing conclusions from a
  `--target-domain sim` experiment.
- **Prototype EMA is not all-gathered across ranks under unequal pixel counts.**
  `_all_gather_cat` degrades to rank-local statistics rather than deadlocking.
  Single-GPU training (the current setup) is unaffected.
- The `triplets_v1` manifest carries `thick_cloud_pct` / `thin_cloud_pct` and a
  `delta_days` between acquisitions. Neither is currently used for filtering.
  Cloud disagreement between the two views is a genuine source of *false*
  positive pairs, and filtering on those columns is the obvious next lever if
  alignment plateaus.
