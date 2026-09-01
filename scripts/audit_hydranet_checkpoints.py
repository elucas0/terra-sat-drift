"""Audit the published HydraNet / Phi2FM PhiSat-2 student checkpoints before probing them.

`train_phisatnet_lulc.py` needs one checkpoint to take an encoder from, and the
catalog offers around thirty for the land-cover task alone. They are not reruns
of one model -- their encoders differ by up to ~2 in max absolute weight
difference -- so the choice is an experimental decision that should be made from
evidence and then pinned with `--checkpoint-datetime`.

This script answers three questions:

1. **What is actually in the catalog** for a task, and do the checkpoints have
   the architecture this project assumes (8-channel stem, 11-class head)?
2. **How much do they differ** from each other, weights and BatchNorm statistics
   separately, so "which release" can be seen to matter (or not).
3. **Which one transfers best, before any training at all.** With `--zero-shot`,
   each candidate is run as-is on real PhiSat-2 patches. The land-cover head is
   already the same 11 WorldCover classes in the same ascending-code order, so
   its predictions are directly scorable. That number is worth having in its own
   right: it is the "no adaptation at all" row beneath the frozen-encoder probe.

The zero-shot pass also puts a number on the layer-scale question. The
checkpoints omit their `*.convnext_block.gamma` tensors;
`hydranet.loading.load_student` fills them from `PhisatNet`'s ConvNeXt default of
1e-6, which turns every residual branch off. `--gamma-fill both` scores each
candidate under 1e-6 and under 1 so there is at least a measurement alongside the
argument -- though see below for how little that measurement can carry.

What the first run of this script found, so the columns are not over-read:

  * All ten land-cover checkpoints have the assumed architecture -- 8-channel
    stem, 11-class head -- and none carries layer-scale gammas.
  * The five `finetuning` releases share an encoder to within 0.02 max absolute
    weight difference, so which shot count they came from barely matters. The
    `linear_probing` releases sit ~0.35 away from them, and three of them
    (n100/n500/n1000, 20260109) are *bit-identical* in body and decoder and
    differ only in their head and BatchNorm statistics. So there are two encoder
    families to choose between, not ten.
  * **Zero-shot is at chance for every candidate** (mIoU 0.0008-0.0413), under
    both gamma fills, and -- checked separately -- under every plausible band
    permutation. The reference point is 0.0376: the macro mIoU a uniform-random
    predictor scores on these eleven class frequencies. It is not 1/11 = 0.09,
    which is the random *pixel accuracy* -- a different quantity, and the one
    that is easy to reach for by mistake. The simulated-to-real PhiSat-2 gap
    is large enough to erase the signal before training starts. Read the
    zero-shot column as the floor the probe has to beat, and choose the encoder
    with `sweep_phisatnet_encoders.py` instead: it cannot be chosen from here.
  * `gamma_fill=ones` beats `model_init` in 8 of 10 candidates, which is
    consistent with the checkpoints having been trained with layer scale off but
    is far too weak on its own -- everything is at chance. The argument for
    `ones` is the training-time one in `phisatnet_student`, not this table.

A caveat that applies to every zero-shot number here: these checkpoints were
trained on PhilEO-Bench reflectance standardised with its own statistics, and
real PhiSat-2 is on a different radiometric scale entirely. The inputs are
standardised with the real domain's own per-band statistics, which is the closest
available alignment and the same normalisation the probe and the baselines use,
but it is not the checkpoint's training-time preprocessing. Read the zero-shot
column as a floor, not as the model's ceiling.

Usage::

    # catalog and weight audit only, no data and no GPU needed
    python scripts/audit_hydranet_checkpoints.py

    # add zero-shot scores on 500 real PhiSat-2 validation patches
    python scripts/audit_hydranet_checkpoints.py --zero-shot --n-samples 500

    # widen to specific candidates
    python scripts/audit_hydranet_checkpoints.py --zero-shot \\
        --candidates finetuning:5000 linear_probing:5000 linear_probing:50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "terra_sat_drift"))

from phisatnet_student import (  # noqa: E402
    BOTTLENECK_PREFIXES,
    DECODER_PREFIXES,
    ENCODER_PREFIXES,
    _ensure_hydranet_importable,
    create_phisatnet_student,
    resolve_checkpoint,
)

DATA = Path("/shared/projects/phisat2/data/processed")
DEFAULT_WEIGHTS_DIR = Path("/shared/home/elucas/scratch/terra-sat-drift/weights/hydranet")
DEFAULT_OUT = Path("/shared/home/elucas/scratch/terra-sat-drift/outputs/hydranet_audit")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", type=str, default="lc")
    p.add_argument("--candidates", type=str, nargs="*", default=None,
                   help="`training:nshots[:datetime]` triples. Default: the latest "
                        "release of every (training, n_shots) combination for the task.")
    p.add_argument("--weights-dir", type=Path, default=DEFAULT_WEIGHTS_DIR)
    p.add_argument("--hydranet-src", type=Path, default=None)
    p.add_argument("--zero-shot", action="store_true",
                   help="Score each candidate on real PhiSat-2 with no training.")
    p.add_argument("--domains", type=str, nargs="+", default=["real8"],
                   choices=["real8", "sim8", "s2b8"])
    p.add_argument("--split", type=str, default="val", choices=["val", "test"])
    p.add_argument("--n-samples", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--gamma-fill", type=str, default="ones",
                   choices=["ones", "model_init", "both"])
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def list_candidates(task: str, spec: list[str] | None, hydranet_src) -> list[dict]:
    _ensure_hydranet_importable(hydranet_src)
    from hydranet.weights import _load_catalog

    df = _load_catalog()
    df = df[(df.model == "student") & (df.task == task)]
    if len(df) == 0:
        raise SystemExit(f"No student checkpoints for task {task!r}.")

    if spec:
        out = []
        for item in spec:
            parts = item.split(":")
            training, n_shots = parts[0], int(parts[1])
            dt = int(parts[2]) if len(parts) > 2 else None
            out.append({"training": training, "n_shots": n_shots, "datetime": dt})
        return out

    latest = (df.sort_values("datetime").groupby(["training", "n_shots"]).tail(1)
              .sort_values(["training", "n_shots"]))
    return [{"training": r.training, "n_shots": int(r.n_shots), "datetime": int(r.datetime)}
            for r in latest.itertuples()]


def weight_summary(path: Path) -> dict:
    state = torch.load(str(path), map_location="cpu", weights_only=True)
    head = next((k for k in ("classifier.weight", "final_conv.weight") if k in state), None)
    stem = state.get("encoders.0.channel_proj.weight")
    return {
        "n_tensors": len(state),
        "head_key": head,
        "head_classes": int(state[head].shape[0]) if head else None,
        "stem_in_channels": int(stem.shape[1]) if stem is not None else None,
        "gamma_keys": sum(1 for k in state if k.endswith("gamma")),
        "_state": state,
    }


def pairwise_deltas(states: dict[str, dict]) -> pd.DataFrame:
    """Max absolute difference between each pair, weights and BN stats apart.

    `num_batches_tracked` is excluded: it is a step counter in the thousands and
    dominates any max-absolute comparison while saying nothing about the model.
    """
    names = list(states)
    ref = states[names[0]]
    groups = {
        "enc_w": [k for k in ref if k.startswith(ENCODER_PREFIXES + BOTTLENECK_PREFIXES)
                  and "running_" not in k and "num_batches" not in k],
        "enc_bn": [k for k in ref if k.startswith(ENCODER_PREFIXES + BOTTLENECK_PREFIXES)
                   and "running_" in k],
        "dec_w": [k for k in ref if k.startswith(DECODER_PREFIXES)
                  and "running_" not in k and "num_batches" not in k],
    }
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            row = {"a": a, "b": b}
            for group, keys in groups.items():
                shared = [k for k in keys if k in states[a] and k in states[b]
                          and states[a][k].shape == states[b][k].shape]
                row[f"max_delta_{group}"] = (
                    max(float((states[a][k] - states[b][k]).abs().max()) for k in shared)
                    if shared else float("nan"))
            rows.append(row)
    return pd.DataFrame(rows)


@torch.no_grad()
def zero_shot_score(checkpoint: Path, domain: str, args, gamma_fill: str) -> dict:
    """Runs one checkpoint, unmodified, over a slice of one domain's split."""
    import torchmetrics
    from torch.utils.data import DataLoader
    from torchmetrics.classification import MulticlassJaccardIndex

    from dataset.constants import WC_CLASS_MAPPING
    from dataset.dataset_paired_triplets_lulc import PhisatPairedLULCDataset

    num_classes = len(WC_CLASS_MAPPING)
    model = create_phisatnet_student(
        num_classes=num_classes, checkpoint=checkpoint, freeze="none",
        reinit_decoder=False, gamma_fill=gamma_fill, hydranet_src=args.hydranet_src,
        verbose=False,
    )
    # `create_phisatnet_student` always rebuilds the head, so put the checkpoint's
    # own back: a zero-shot score of a random head measures nothing.
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    head_w = state.get("classifier.weight", state.get("final_conv.weight"))
    head_b = state.get("classifier.bias", state.get("final_conv.bias"))
    if head_w is None or head_w.shape[0] != num_classes:
        return {"skipped": f"head has {None if head_w is None else head_w.shape[0]} classes"}
    model.net.final_conv.weight.copy_(head_w)
    model.net.final_conv.bias.copy_(head_b)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    ds = PhisatPairedLULCDataset(
        h5_images_path=str(DATA / "triplets_v1/phisat2_s2b_dataset_v1.h5"),
        h5_labels_path=str(DATA / "worldcover_all_clean_v1/worldcover_all_clean_labels_v1.h5"),
        manifest_path=str(DATA / "worldcover_all_clean_v1/worldcover_all_clean_manifest_v1.csv"),
        split=args.split, transform=None, max_samples=args.n_samples,
        target_domain=domain, source_domain=None,
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    iou = torchmetrics.JaccardIndex(task="multiclass", num_classes=num_classes,
                                    ignore_index=-1).to(device)
    acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes,
                                ignore_index=-1).to(device)
    per_class = MulticlassJaccardIndex(num_classes=num_classes, ignore_index=-1,
                                       average=None).to(device)
    for batch in dl:
        x = batch["image_target"].to(device, non_blocking=True)
        y = batch["mask"].long().to(device, non_blocking=True)
        logits = model(x)
        iou.update(logits, y)
        acc.update(logits, y)
        per_class.update(logits, y)
    pc = per_class.compute().cpu()
    return {"miou": float(iou.compute()), "acc": float(acc.compute()),
            "miou_fixed": float(pc.mean()), "n_patches": len(ds)}


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.weights_dir.mkdir(parents=True, exist_ok=True)

    candidates = list_candidates(args.task, args.candidates, args.hydranet_src)
    print(f"Auditing {len(candidates)} '{args.task}' student checkpoints\n")

    rows, states = [], {}
    for cand in candidates:
        path = resolve_checkpoint(task=args.task, weights_dir=str(args.weights_dir),
                                  hydranet_src=args.hydranet_src, **cand)
        summary = weight_summary(path)
        name = f"{cand['training']}/n{cand['n_shots']}/{cand['datetime'] or 'latest'}"
        states[name] = summary.pop("_state")
        rows.append({"name": name, **cand, "path": str(path), **summary})
    table = pd.DataFrame(rows)
    print(table[["name", "n_tensors", "head_key", "head_classes",
                 "stem_in_channels", "gamma_keys"]].to_string(index=False))

    bad = table[(table.stem_in_channels != 8)]
    if len(bad):
        print(f"\n!! {len(bad)} checkpoint(s) do not have the assumed 8-channel stem.")
    if (table.gamma_keys == 0).all():
        print("\nNo checkpoint carries layer-scale gammas: they were trained with layer "
              "scale off, so filling them with ones reproduces training-time behaviour.")

    deltas = pairwise_deltas(states)
    print("\nPairwise max |difference| (num_batches_tracked excluded):")
    print(deltas.to_string(index=False, float_format=lambda v: f"{v:.3e}"))
    deltas.to_csv(args.out / f"{args.task}_pairwise_deltas.csv", index=False)

    if args.zero_shot:
        fills = ["ones", "model_init"] if args.gamma_fill == "both" else [args.gamma_fill]
        print(f"\nZero-shot on {args.split} split, {args.n_samples} patches, "
              f"domains {args.domains}, gamma fills {fills}")
        zs = []
        for row in table.itertuples():
            for domain in args.domains:
                for fill in fills:
                    res = zero_shot_score(Path(row.path), domain, args, fill)
                    zs.append({"name": row.name, "domain": domain, "gamma_fill": fill, **res})
                    if "skipped" in res:
                        print(f"  {row.name:34s} {domain:6s} {fill:10s} skipped: {res['skipped']}")
                    else:
                        print(f"  {row.name:34s} {domain:6s} {fill:10s} "
                              f"mIoU {res['miou']:.4f}  fixed {res['miou_fixed']:.4f}  "
                              f"acc {res['acc']:.4f}")
        zs_df = pd.DataFrame(zs)
        zs_df.to_csv(args.out / f"{args.task}_zero_shot.csv", index=False)
        if "miou" in zs_df:
            best = zs_df.dropna(subset=["miou"]).sort_values("miou", ascending=False).head(1)
            if len(best):
                b = best.iloc[0]
                print(f"\nBest zero-shot: {b['name']} on {b['domain']} "
                      f"(gamma_fill={b['gamma_fill']}) at mIoU {b['miou']:.4f}")
                print("Pin it with --checkpoint-training/--checkpoint-nshots/"
                      "--checkpoint-datetime in train_phisatnet_lulc.py.")

    table.drop(columns=["path"]).to_csv(args.out / f"{args.task}_checkpoints.csv", index=False)
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
