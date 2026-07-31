"""
Compare cross-sensor KD runs from their local W&B logs (no network needed).

    python scripts/analyze_kd_runs.py outputs/xsensor_kd_* [--outdir DIR]

Reads the `.wandb` transaction log in each run directory, so it works on runs
that were never synced. Where a directory holds several attempts (a crash and a
restart both write there), the one with the most history is used.

Writes a comparison figure plus a CSV of the headline numbers, and prints a
significance test on the per-epoch validation metrics -- these runs are noisy
enough epoch to epoch that a single best-checkpoint number is easy to over-read.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent / "dataset_report"))
from plot_style import CATEGORICAL, INK, INK_SOFT, save, setup, title  # noqa: E402

# Line and grouped-bar charts use the adjacent pairlist, so the full 8-slot
# order is available here (unlike scatter, which is capped at three).
RUN_COLORS = CATEGORICAL
WARMUP = 8  # epochs to drop before comparing; both runs are still ramping before this


def read_history(wandb_file: Path) -> pd.DataFrame:
    """Scalar history from a local .wandb log (keys live in `nested_key`)."""
    from wandb.proto import wandb_internal_pb2 as pb
    from wandb.sdk.internal import datastore

    ds = datastore.DataStore()
    ds.open_for_scan(str(wandb_file))
    rows = []
    while True:
        try:
            data = ds.scan_data()
        except Exception:
            break
        if data is None:
            break
        rec = pb.Record()
        try:
            rec.ParseFromString(data)
        except Exception:
            continue
        if rec.WhichOneof("record_type") != "history":
            continue
        row = {}
        for it in rec.history.item:
            key = ".".join(it.nested_key) if it.nested_key else it.key
            try:
                v = json.loads(it.value_json)
            except Exception:
                continue
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                row[key] = v
        if row:
            rows.append(row)
    df = pd.DataFrame(rows)
    return df.groupby("epoch", as_index=False).max(numeric_only=True) if "epoch" in df else df


def _label(run_dir: Path, summary: dict, cfg_path: Path) -> str:
    """Label from the loss weights, plus the task loss when it is not plain CE.

    Several settings share one output directory (the directory name only encodes
    the contrastive weights), so the label has to come from the config too or two
    runs that differ in task loss look identical.
    """
    m = re.search(r"inst([\d.]+)_pix([\d.]+)", run_dir.name)
    parts = [f"inst={m.group(1)}, pix={m.group(2)}"] if m else [run_dir.name]
    try:
        import yaml
        cfg = yaml.safe_load(cfg_path.read_text())
        get = lambda k: (cfg.get(k, {}) or {}).get("value") if isinstance(cfg.get(k), dict) else cfg.get(k)
        if get("task_loss") == "focal":
            parts.append(f"focal γ={get('focal_gamma')}")
        if get("class_weights"):
            parts.append("weighted")
        if get("kd_mode") and get("kd_mode") != "kl":
            parts.append(f"kd={get('kd_mode')}")
        if get("w_crd"):
            parts.append(f"crd={get('w_crd')}")
        if get("use_dsbn"):
            parts.append("DSBN")
    except Exception:
        pass
    return ", ".join(parts)


def load_run(path: Path):
    """Returns (label, history, summary).

    `path` may be an output directory (the longest attempt inside it is used, so
    a crashed restart does not win) or a single `run-*` directory.
    """
    if list(path.glob("run-*.wandb")):
        files, run_dir = sorted(path.glob("run-*.wandb")), path.parents[1]
    else:
        files, run_dir = sorted(path.glob("wandb/run-*/run-*.wandb")), path
    best = None
    for f in files:
        df = read_history(f)
        if best is None or len(df) > len(best[0]):
            summary = json.loads((f.parent / "files" / "wandb-summary.json").read_text())
            best = (df, summary, f.parent)
    if best is None:
        raise SystemExit(f"no .wandb log under {path}")
    label = _label(run_dir, best[1], best[2] / "files" / "config.yaml")
    print(f"  {label}\n    -> {best[2].name}, {len(best[0])} epochs")
    return label, best[0], best[1]


# ---------------------------------------------------------------------------
def _line(ax, runs, metric, ylabel, subtitle=None, name=None, best="max"):
    """`best` marks the best epoch: "max", "min" for lower-is-better metrics
    (domain gap, MMD), or None. Marking the max on a lower-is-better panel would
    highlight its *worst* epoch."""
    for (label, df, _), color in zip(runs, RUN_COLORS):
        if metric not in df:
            continue
        d = df[["epoch", metric]].dropna()
        ax.plot(d["epoch"], d[metric], color=color, label=label)
        if best:
            i = d[metric].idxmax() if best == "max" else d[metric].idxmin()
            ax.plot(d.loc[i, "epoch"], d.loc[i, metric], "o", color=color, ms=6,
                    markeredgecolor="white", markeredgewidth=1.2)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    title(ax, name or metric, subtitle)


def figure(runs, outdir):
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))

    _line(axes[0][0], runs, "val/student_target_iou", "mIoU",
          "dot = best epoch", "PhiSat-2 mIoU (the goal)")
    _line(axes[0][1], runs, "val/domain_gap_iou", "IoU(S2) - IoU(PhiSat-2)",
          "dot = lowest; lower is more domain-invariant", "IoU domain gap", best="min")
    _line(axes[0][2], runs, "val/mmd2", "MMD²",
          "dot = lowest; lower = closer feature distributions",
          "Distribution gap (MMD²)", best="min")

    # Alignment vs uniformity, the mechanism behind the contrastive term:
    # colour identifies the run, dash identifies which cosine.
    ax = axes[1][0]
    for (label, df, _), color in zip(runs, RUN_COLORS):
        for metric, ls, tag in (("val/cos_paired", "-", "paired"),
                                ("val/cos_unpaired", (0, (4, 2)), "unpaired")):
            if metric in df:
                d = df[["epoch", metric]].dropna()
                ax.plot(d["epoch"], d[metric], color=color, linestyle=ls,
                        label=f"{label} — {tag}", linewidth=2.0 if ls == "-" else 1.6)
    ax.axhline(0, color=INK_SOFT, linewidth=0.7, linestyle=":")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cosine similarity")
    title(ax, "Alignment vs uniformity",
          "solid = same patch across sensors; dashed = different patches")

    _line(axes[1][1], runs, "val/retrieval_top1", "top-1 accuracy",
          "cross-sensor instance matching", "Retrieval top-1")
    _line(axes[1][2], runs, "val/student_source_iou", "mIoU",
          "same student, Sentinel-2 input", "Sentinel-2 mIoU")

    for ax in axes.ravel():
        ax.legend(fontsize=7)
    fig.tight_layout()
    save(fig, outdir, "fig_kd_comparison")


def per_class_figure(runs, outdir):
    """Per-class IoU. Key names changed mid-campaign, so both are accepted."""
    classes = ["tree_cover", "shrubland", "grassland", "cropland", "built_up",
               "bare_sparse_veg", "snow_ice", "water", "herbaceous_wetland",
               "mangroves", "moss_lichen"]

    def get(summary, c):
        for k in (f"val_per_class/IoU_{c}", f"val/student_target_IoU_{c}"):
            if k in summary:
                return summary[k]
        return np.nan

    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    x = np.arange(len(classes))
    width = 0.8 / len(runs)
    for i, ((label, _, summary), color) in enumerate(zip(runs, RUN_COLORS)):
        vals = [get(summary, c) for c in classes]
        ax.bar(x + (i - (len(runs) - 1) / 2) * width, vals, width * 0.9,
               color=color, label=label, edgecolor="white", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", " ") for c in classes], rotation=30, ha="right")
    ax.set_ylabel("val IoU")
    ax.legend()
    # These come from the run summary, i.e. the FINAL epoch -- not the best
    # checkpoint that the reported test numbers use. Rare classes swing hard
    # between epochs (snow/ice ranged 0.00-0.45 within one run), so a class can
    # read ~0 here and still score well at test. Say so on the figure.
    dead = [c for c in classes if all(not (get(s, c) > 0.01) for _, _, s in runs)]
    title(ax, "Per-class IoU on PhiSat-2",
          "final epoch on val (not the best checkpoint); rare classes are volatile"
          + (f" — at ~0 for every run: {', '.join(dead)}" if dead else ""))
    fig.tight_layout()
    save(fig, outdir, "fig_kd_per_class")


def table(runs, outdir):
    metrics = ["val/student_target_iou", "val/student_source_iou", "val/domain_gap_iou",
               "val/mmd2", "val/retrieval_top1", "val/cos_paired", "val/cos_unpaired"]
    rows = []
    for label, df, summary in runs:
        d = df[df["epoch"] >= WARMUP]
        r = {"run": label, "epochs": int(df["epoch"].max()) + 1,
             "best_val_target_iou": df["val/student_target_iou"].max(),
             "best_epoch": int(df.loc[df["val/student_target_iou"].idxmax(), "epoch"]),
             "test_target_iou": summary.get("test/student_target_iou", np.nan),
             "test_source_iou": summary.get("test/student_source_iou", np.nan),
             "teacher_source_iou": summary.get("test/teacher_source_iou", np.nan)}
        for m in metrics:
            if m in d:
                r[f"{m.split('/')[1]}_mean"] = d[m].mean()
                r[f"{m.split('/')[1]}_sd"] = d[m].std()
        rows.append(r)
    tbl = pd.DataFrame(rows).round(4)
    tbl.to_csv(outdir / "kd_runs_summary.csv", index=False)
    print("\n" + tbl.to_string(index=False))

    if len(runs) == 2:
        print(f"\nWelch t-test on per-epoch validation metrics (epoch >= {WARMUP}), "
              f"n={len(runs[0][1][runs[0][1].epoch >= WARMUP])} vs "
              f"{len(runs[1][1][runs[1][1].epoch >= WARMUP])}:")
        (la, da, _), (lb, db, _) = runs
        for m in metrics:
            if m not in da or m not in db:
                continue
            a = da[da.epoch >= WARMUP][m].dropna()
            b = db[db.epoch >= WARMUP][m].dropna()
            p = stats.ttest_ind(a, b, equal_var=False).pvalue
            flag = "  <-- significant" if p < 0.01 else ""
            print(f"  {m:28s} {a.mean():+.4f} vs {b.mean():+.4f}   p={p:.4g}{flag}")
        print(f"\n  ({la} vs {lb})")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--outdir", type=Path,
                   default=Path("/shared/home/elucas/scratch/terra-sat-drift/outputs/kd_analysis"))
    args = p.parse_args()

    setup()
    if len(args.runs) > len(RUN_COLORS):
        raise SystemExit(
            f"{len(args.runs)} runs but only {len(RUN_COLORS)} validated colour slots. "
            "Compare fewer at a time rather than generating a 9th hue -- zip() would "
            "otherwise drop the extras from the figures silently."
        )
    print("Loading runs:")
    runs = [load_run(d) for d in args.runs]
    figure(runs, args.outdir)
    per_class_figure(runs, args.outdir)
    table(runs, args.outdir)


if __name__ == "__main__":
    main()
