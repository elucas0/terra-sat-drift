"""Pick the PhiSat-2 student encoder to probe with, by probing several of them.

`audit_hydranet_checkpoints.py` shows the published checkpoints fall into two
groups whose encoders genuinely differ, and that zero-shot scores sit at chance
for all of them -- the domain gap between simulated and real PhiSat-2 is large
enough to swamp any difference before training. So the audit cannot choose, and
a short frozen-decoder probe can: it is the same protocol as the real run, just
on a smaller budget, and it costs little because only ~0.15M parameters move.

Each cell is one `train_phisatnet_lulc.py` run in its own subprocess, so a crash
in one cannot take down the sweep, and cells whose metrics JSON already exists
are skipped -- rerun the same command and it continues where it stopped.

Read the output as a ranking, not as final numbers: the whole point of a short
budget is that it is short. Take the winner, then rerun it at full budget with
`--checkpoint-datetime` pinned, and report *that*.

Examples::

    # what would run
    python scripts/sweep_phisatnet_encoders.py --dry-run

    # the default sweep: the two encoder families plus the random-init control
    python scripts/sweep_phisatnet_encoders.py

    # everything in the catalog, longer budget
    python scripts/sweep_phisatnet_encoders.py --all-candidates \\
        --epochs 15 --max-samples 5000

    # re-aggregate the table from existing records without training anything
    python scripts/sweep_phisatnet_encoders.py --collect-only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = _REPO / "terra_sat_drift" / "train_phisatnet_lulc.py"
DEFAULT_RESULTS = Path("/shared/home/elucas/scratch/terra-sat-drift/outputs/phisatnet_encoder_sweep")

# The two families the audit separates, at both ends of the shot ladder, plus the
# control. Within the `finetuning` family the encoders differ by <0.02 in max
# absolute weight difference, so one from each end is enough to see whether the
# ladder position matters at all.
DEFAULT_CANDIDATES = [
    "finetuning:5000:20260108",
    "finetuning:50:20260108",
    "linear_probing:5000:20260108",
    "linear_probing:1000:20260109",
    "scratch",
]

METRIC_COLUMNS = {
    "test_miou": "test/student_target_iou",
    "test_miou_fixed": "test/student_target_iou_fixed",
    "test_acc": "test/student_target_acc",
    "test_f1": "test/student_target_f1",
    "test_loss": "test/loss",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--candidates", nargs="*", default=None,
                   help="`training:nshots[:datetime]` triples, or the literal "
                        "`scratch` for the random-init control.")
    p.add_argument("--all-candidates", action="store_true",
                   help="Every (training, n_shots) combination in the catalog, latest release.")
    p.add_argument("--seeds", type=int, nargs="+", default=[42],
                   help="Repeats per candidate. One seed ranks; three separates a "
                        "real difference from run-to-run noise.")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--max-samples", type=int, default=2000)
    p.add_argument("--val-max-samples", type=int, default=1000)
    p.add_argument("--test-max-samples", type=int, default=2000,
                   help="Capped for the sweep. The full split belongs to the final run.")
    p.add_argument("--task-loss", type=str, default="focal", choices=["ce", "focal"])
    p.add_argument("--class-weights", type=str, default="inverse_sqrt")
    p.add_argument("--target-domain", type=str, default="real8")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                   help="Everything after this is passed through to the training script.")
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--collect-only", action="store_true")
    return p.parse_args()


def catalog_candidates() -> list[str]:
    sys.path.insert(0, str(_REPO / "terra_sat_drift"))
    from phisatnet_student import _ensure_hydranet_importable

    _ensure_hydranet_importable()
    from hydranet.weights import _load_catalog

    df = _load_catalog()
    df = df[(df.model == "student") & (df.task == "lc")]
    latest = df.sort_values("datetime").groupby(["training", "n_shots"]).tail(1)
    return [f"{r.training}:{int(r.n_shots)}:{int(r.datetime)}"
            for r in latest.sort_values(["training", "n_shots"]).itertuples()] + ["scratch"]


def cell_name(candidate: str, seed: int) -> str:
    return f"{candidate.replace(':', '-')}__s{seed}"


def build_command(candidate: str, seed: int, record: Path, args) -> list[str]:
    cmd = [sys.executable, str(TRAIN_SCRIPT),
           "--target-domain", args.target_domain,
           "--epochs", str(args.epochs),
           "--max-samples", str(args.max_samples),
           "--val-max-samples", str(args.val_max_samples),
           "--test-max-samples", str(args.test_max_samples),
           "--task-loss", args.task_loss,
           "--class-weights", args.class_weights,
           "--lr", str(args.lr),
           "--batch-size", str(args.batch_size),
           "--num-workers", str(args.num_workers),
           "--patience", str(args.patience),
           "--seed", str(seed),
           "--metrics-out", str(record),
           "--tag", f"phisatnet_sweep_{cell_name(candidate, seed)}",
           "--no-wandb"]
    if candidate == "scratch":
        cmd.append("--no-pretrained")
    else:
        parts = candidate.split(":")
        cmd += ["--checkpoint-training", parts[0], "--checkpoint-nshots", parts[1]]
        if len(parts) > 2:
            cmd += ["--checkpoint-datetime", parts[2]]
    return cmd + list(args.extra)


def collect(results_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted((results_dir / "runs").glob("*.json")):
        try:
            rec = json.loads(path.read_text())
        except json.JSONDecodeError:
            print(f"  skipping unreadable record: {path.name}")
            continue
        spec = rec.get("checkpoint_spec", {})
        row = {
            "cell": path.stem,
            "pretrained": rec.get("pretrained"),
            "ckpt_training": None if not rec.get("pretrained") else spec.get("training"),
            "ckpt_nshots": None if not rec.get("pretrained") else spec.get("n_shots"),
            "ckpt_datetime": None if not rec.get("pretrained") else spec.get("datetime"),
            "seed": rec.get("seed"),
            "epochs_run": rec.get("epochs_run"),
            "trainable_params": rec.get("trainable_params"),
            "best_val_iou": rec.get("best_val_iou"),
        }
        test = rec.get("test", {})
        row.update({col: test.get(key) for col, key in METRIC_COLUMNS.items()})
        rows.append(row)
    df = pd.DataFrame(rows)
    # Nullable ints, so the control row's missing checkpoint spec does not turn
    # shot counts and release dates into "5000.0000" / "20260108.0000".
    for col in ("ckpt_nshots", "ckpt_datetime", "seed", "epochs_run", "trainable_params"):
        if col in df:
            df[col] = df[col].astype("Int64")
    return df


def _md_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    """Markdown table without pulling in `tabulate`, which is not in this env."""
    def cell(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return ""
        return format(v, floatfmt) if isinstance(v, float) else str(v)

    header = list(df.columns)
    rows = [[cell(v) for v in row] for row in df.itertuples(index=False)]
    return "\n".join(["| " + " | ".join(header) + " |",
                      "| " + " | ".join("---" for _ in header) + " |",
                      *["| " + " | ".join(r) + " |" for r in rows]])


def write_summary(df: pd.DataFrame, results_dir: Path, args) -> None:
    if df.empty:
        print("No records to summarise yet.")
        return
    df = df.sort_values("test_miou", ascending=False)
    df.to_csv(results_dir / "encoder_sweep_results.csv", index=False)

    group = ["pretrained", "ckpt_training", "ckpt_nshots", "ckpt_datetime"]
    agg = (df.groupby(group, dropna=False)["test_miou"]
             .agg(["count", "mean", "std", "max"]).reset_index()
             .sort_values("mean", ascending=False))

    lines = [
        "# PhiSat-2 student encoder sweep",
        "",
        f"Short frozen-decoder probes: {args.epochs} epochs, {args.max_samples} training "
        f"patches, {args.target_domain}, {args.task_loss} loss / {args.class_weights} "
        "class weights.",
        "",
        "**These are ranking numbers, not results.** Rerun the winner at full budget "
        "with its release date pinned before reporting anything.",
        "",
        _md_table(agg),
        "",
        "## Every cell",
        "",
        _md_table(df),
        "",
    ]
    (results_dir / "encoder_sweep_summary.md").write_text("\n".join(lines))
    print("\n" + agg.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    best = df.iloc[0]
    if best["pretrained"]:
        print(f"\nBest cell: {best['cell']} at test mIoU {best['test_miou']:.4f}")
        print(f"  Rerun at full budget with: --checkpoint-training {best['ckpt_training']} "
              f"--checkpoint-nshots {int(best['ckpt_nshots'])} "
              f"--checkpoint-datetime {int(best['ckpt_datetime'])}")
    else:
        print(f"\nBest cell is the random-init control ({best['test_miou']:.4f}). "
              "At this budget the pretrained encoders are not earning their keep -- "
              "check the budget before concluding that they never do.")


def main():
    args = parse_args()
    results_dir = args.results_dir
    (results_dir / "runs").mkdir(parents=True, exist_ok=True)
    (results_dir / "logs").mkdir(parents=True, exist_ok=True)

    if args.collect_only:
        write_summary(collect(results_dir), results_dir, args)
        return

    candidates = (catalog_candidates() if args.all_candidates
                  else (args.candidates or DEFAULT_CANDIDATES))
    cells = [(c, s) for c in candidates for s in args.seeds]
    pending = [(c, s) for c, s in cells
               if not (results_dir / "runs" / f"{cell_name(c, s)}.json").exists()]

    print(f"{len(cells)} cells ({len(candidates)} candidates x {len(args.seeds)} seeds); "
          f"{len(pending)} to run, {len(cells) - len(pending)} already done.")
    for c, s in pending:
        print(f"  {cell_name(c, s)}")
    if args.dry_run:
        return

    for i, (candidate, seed) in enumerate(pending, 1):
        name = cell_name(candidate, seed)
        record = results_dir / "runs" / f"{name}.json"
        log = results_dir / "logs" / f"{name}.log"
        cmd = build_command(candidate, seed, record, args)
        print(f"\n[{i}/{len(pending)}] {name}\n  {' '.join(cmd)}")
        start = time.time()
        with log.open("w") as fh:
            proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                  cwd=str(_REPO))
        elapsed = time.time() - start
        if proc.returncode != 0:
            print(f"  FAILED after {elapsed/60:.1f} min (exit {proc.returncode}); see {log}")
        elif not record.exists():
            print(f"  finished but wrote no record; see {log}")
        else:
            miou = json.loads(record.read_text()).get("test", {}).get(
                "test/student_target_iou")
            score = f", test mIoU {miou:.4f}" if miou is not None else ""
            print(f"  done in {elapsed/60:.1f} min{score}")

    write_summary(collect(results_dir), results_dir, args)
    print(f"\nWritten to {results_dir}")


if __name__ == "__main__":
    main()
