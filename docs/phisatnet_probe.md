# Probing the PhiSat-2 foundation-model student on real PhiSat-2 LULC

Taking the published HydraNet / Phi2FM `PhisatNet` student — a sensor-specific
model pretrained on **simulated** PhiSat-2 imagery — freezing its encoder, and
training a new decoder on **real** PhiSat-2 WorldCover labels. The result is a
third row for the land-cover table, next to the aligned TerraMind run and the
no-KD U-Net baseline.

Implementation: [phisatnet_student.py](../terra_sat_drift/phisatnet_student.py),
[dataset_paired_triplets_lulc.py](../terra_sat_drift/dataset/dataset_paired_triplets_lulc.py).
Training entry point: [train_phisatnet_lulc.py](../terra_sat_drift/train_phisatnet_lulc.py).
Supporting scripts: [audit_hydranet_checkpoints.py](../scripts/audit_hydranet_checkpoints.py),
[sweep_phisatnet_encoders.py](../scripts/sweep_phisatnet_encoders.py).

The model itself lives in a sibling checkout, `hydranet-phisat2`, and is imported
by path rather than installed — it pins python 3.9 and torch 2.4, neither of
which this project uses, while the model code is plain torch and runs fine here.

## 1. Why this row is worth having

The other two rows in the land-cover table are both *general* models adapted to
PhiSat-2: TerraMind is a Sentinel-2-pretrained GeoFM, and the U-Net baseline is
trained from scratch. `PhisatNet` is the third possibility — a model that was
already built for this sensor, but for a *simulation* of it. So the row answers a
question neither of the others can: **does sensor-specific pretraining on
simulated PhiSat-2 transfer to the real thing, and by how much?**

The zero-shot number below suggests the honest headline answer is "not on its
own": the pretrained land-cover head, run unchanged on real PhiSat-2, scores at
chance. What the probe measures is whether the *features* survive even though the
predictions do not.

## 2. Protocol

Everything that is not the intervention is held identical to
[train_baseline_phisat2.py](../terra_sat_drift/train_baseline_phisat2.py): the
seed-42 80/10/10 split, the `BAD_PRODUCT_IDS` filter, per-domain
sqrt → clip → z-score normalisation, geometric-only augmentation, the focal /
class-weighted task loss, AdamW with `ReduceLROnPlateau`, the early-stopping
monitor, and the metric key names. The same `SupervisedSegmentationModule`
computes the metrics, so `test/student_target_iou` means the same thing in all
three rows.

Three things do differ, and any table carrying this row should say so:

| | U-Net baseline | This probe |
|---|---|---|
| input bands | 7 (Blue…NIR) | 8 (adds panchromatic) |
| trainable parameters | 9.85M | 0.15M of 0.35M |
| learning rate | 1e-4 | 1e-3 |

The extra band is the model's own contract, not a thumb on the scale — but it is
still an input the baseline does not get. `--in-channels 7` drops it, at the cost
of discarding the pretrained stem weights; run it as an ablation if a reviewer
asks. The learning rate is this repo's frozen-encoder probe rate (the same 1e-3
the flood probes use); `--lr 1e-4` matches the baseline if one number across the
table matters more.

## 3. Three things that are easy to get wrong

### Channel order

The stem is `Conv2d(8, 16, 1)` and its columns are in the PhilEO order
`B02, B03, B04, B08, B05, B06, B07, PAN`, documented in the hydranet repo's own
validated WorldFloods notebook. `triplets_v1` stores PhiSat-2 as
`PAN, Blue, Green, Red, RE1, RE2, RE3, NIR`. These are **not** the same order,
and feeding the stored order routes PAN into the column trained for Blue and NIR
into the column trained for RE1.

The `real8` / `sim8` / `s2b8` domains in `DOMAIN_STATS` carry the permutation
`[1, 2, 3, 7, 4, 5, 6, 0]` and the matching per-band statistics, so the reorder
happens once, in the dataset, and nothing downstream has to know. **This model
must be fed one of those domains**, never the plain 7-band `real` / `sim` / `s2b`.
Sentinel-2 has no panchromatic band at all, so `s2b8` fills that channel with the
mean of the standardised visible bands — a cross-sensor control, not a
like-for-like input.

### Missing layer-scale parameters

None of the published checkpoints carry their `*.convnext_block.gamma` tensors:
they were trained with layer scale disabled, so every residual branch ran at full
strength. `PhisatNet` constructs `gamma` at ConvNeXt's default of `1e-6`, and
`hydranet.loading.load_student` fills the missing keys from *that* — which scales
every residual branch down by a factor of a million and hands back an encoder
that is very nearly the identity function.

`create_phisatnet_student` fills them with **ones** instead, matching the
reference notebook's `MISSING_CONVNEXT_GAMMA = "ones"`. `--gamma-fill model_init`
reproduces the other behaviour. **Do not load these checkpoints with
`hydranet.load_student` for this purpose.**

### BatchNorm in a "frozen" encoder

`requires_grad = False` stops the weights moving; it does not stop BatchNorm
running statistics from tracking whatever passes through in training mode. Since
the whole point here is a domain shift, that is not a detail — letting BN adapt
is a real domain-adaptation intervention that would otherwise happen silently
under a "frozen encoder" label.

Overriding `train()` is not enough to prevent it. Lightning 2.6 never calls
`.train()` during `fit` — it relies on `nn.Module`'s default of `True` — and its
evaluation loop restores per-submodule modes by name after every validation
pass. Both bypass a container's `train()` override, and the first version of this
code hit exactly that: the frozen encoder's running statistics moved on every
step. `PhisatNetSegmenter` therefore re-asserts the mode at the top of every
`forward`, which no framework behaviour can undo. `--bn-mode adapt` turns the
adaptation on deliberately, and is worth running as its own row.

## 4. Choosing an encoder

The catalog offers ten land-cover student checkpoints, and they are not reruns of
one model. `audit_hydranet_checkpoints.py` found:

- All ten have the assumed architecture: 8-channel stem, 11-class head, no gammas.
- The five `finetuning` releases share an encoder to within 0.02 max absolute
  weight difference. The `linear_probing` releases sit ~0.35 away, and three of
  them (n100 / n500 / n1000, 20260109) are bit-identical in body *and* decoder,
  differing only in head and BN statistics. So there are **two** encoder families
  to choose between, not ten.
- **Zero-shot is at chance for every candidate**: mIoU 0.0008–0.0413, under both
  gamma fills and under every plausible band permutation. Chance here is 0.0376
  macro mIoU for a uniform-random predictor over these eleven class frequencies
  (0.0492 for a prior-matched one) — *not* 1/11, which is the random pixel
  accuracy and a different quantity. Even the best candidate is level with
  uniform-random. The simulated-to-real gap erases the signal before training
  starts, so the audit cannot pick an encoder — it can only establish the floor
  the probe has to beat.

The choice therefore has to come from a short probe, which is what
`sweep_phisatnet_encoders.py` runs: the same protocol on a smaller budget, one
subprocess per cell, resumable, with a random-init control included so the sweep
also says what the pretraining is worth.

## 5. Running it

```bash
# 1. what is in the catalog, and the zero-shot floor
python scripts/audit_hydranet_checkpoints.py --zero-shot --gamma-fill both

# 2. rank the encoder families on a short budget (includes the scratch control)
python scripts/sweep_phisatnet_encoders.py --epochs 10 --max-samples 2000

# 3. the run that gets reported, with the winner's release pinned
python terra_sat_drift/train_phisatnet_lulc.py \
    --checkpoint-training finetuning --checkpoint-nshots 5000 \
    --checkpoint-datetime 20260108 \
    --task-loss focal --class-weights inverse_sqrt \
    --max-samples 10000 --epochs 50

# 4. the control that says what the pretrained encoder is worth
python terra_sat_drift/train_phisatnet_lulc.py --no-pretrained \
    --task-loss focal --class-weights inverse_sqrt \
    --max-samples 10000 --epochs 50
```

Match `--max-samples` to whatever the row being compared against used; the
baseline runs already in `outputs/` are `n1000`, `n5000` and `n10000`.

Every run writes `metrics.json` into its output directory (or wherever
`--metrics-out` points) with the test metrics, the checkpoint spec, the freeze
and BN settings, and the trainable-parameter count — enough to reconstruct which
row is which months later.

## 6. Putting the row in the table

`test/student_target_iou` and the per-class `test_per_class/IoU_*` keys match the
baseline's exactly, so those two rows line up directly.

The TerraMind rows need checking first. They go through terratorch's
`SemanticSegmentationTask`, which logs `test/mIoU` rather than
`test/student_target_iou`, and `pretrain_lulc.py` trains on the **Sentinel-2**
view (`PhisatS2LULCDataModule`), not PhiSat-2 — its output directory is named
`terramind_v1_base_pretrain_s2b_lulc` for that reason. Confirm which TerraMind
run is going in the table, and on which domain, before putting the numbers side
by side.

Two more things to check:

- **`best` versus `last`.** The KD runs recorded final-epoch test numbers because
  they called `trainer.test` with no `ckpt_path`. This script defaults to `best`.
  Use `--test-ckpt last` for a strictly like-for-like number against those, and
  do not mix the two conventions in one table.
- **Which mIoU.** `test/student_target_iou` is `torchmetrics`' aggregate, whose
  denominator changes when a class is absent from the evaluation split;
  `test/student_target_iou_fixed` is the constant-denominator macro average. Both
  are logged. Use the same one for every row.

## 7. What to state when reporting

- The encoder was pretrained on simulated PhiSat-2, not real PhiSat-2, and the
  two are on different radiometric scales. Each domain is standardised with its
  own statistics, which is the closest available alignment and the same
  normalisation the other rows use, but it is not the checkpoint's training-time
  preprocessing.
- The zero-shot score is at chance. The probe measures whether the features
  transfer, not whether the model does.
- Which checkpoint release the encoder came from. It is a real experimental
  choice, and `--checkpoint-datetime` is what pins it.
