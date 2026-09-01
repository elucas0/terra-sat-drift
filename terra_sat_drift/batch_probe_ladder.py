"""Sweep the frozen-encoder flood probe across the Φ-sat-2 degradation ladder.

``degradation_ladder.py`` produces one dataset per isolated noise factor and level;
this runs ``finetune_floods_student.py`` on each of them and collects the results into a
single table, so downstream IoU can be plotted against noise severity per factor.

Two experiments share the machinery. By default each variant is trained *and* tested on
itself, which measures matched-condition difficulty: how much flood-relevant signal a
factor destroys when the model is allowed to adapt to it. With ``--train-variant`` the
probe instead trains once on that condition and evaluates the resulting checkpoint on
every variant's test split, which is the domain gap: how far a model falls when the input
degrades underneath it. The two answer different questions and the matched-condition run
can score *above* clean, because there the perturbation acts as training augmentation.

The matrix is ``variants x seeds x init``. Each cell is one training run in its own
subprocess -- Lightning, CUDA and W&B state are all per-process, and a crashed cell
cannot take down the sweep. Cells whose metrics JSON already exists are skipped, so the
sweep is resumable; rerun the same command and it continues where it stopped.

On seeds: repeated runs of an identical config on this probe have landed several IoU
points apart, which is the same order as the degradation effect being measured. A single
run per variant therefore cannot separate the two. ``--seeds`` defaults to three so each
point on the curve carries a spread; drop to one only for a quick look.

Results land in ``<results-dir>/``:

    runs/<cell>.json         one record per cell, written by the probe itself
    ladder_probe_results.csv one row per cell, for plotting
    ladder_probe_summary.md  mean +- spread per variant, ready to paste into notes
    logs/<cell>.log          full training log for that cell

Examples::

    # what would run, with a time estimate
    python terra_sat_drift/batch_probe_ladder.py --dry-run

    # smoke test: 2 epochs per cell, one seed, into a throwaway results dir
    python terra_sat_drift/batch_probe_ladder.py --epochs 2 --seeds 42 \\
        --variants clean psf_l4 --results-dir /tmp/probe_smoke

    # the real sweep: 13 variants x 3 seeds, distilled encoder
    python terra_sat_drift/batch_probe_ladder.py

    # add the random-init control on the ladder endpoints
    python terra_sat_drift/batch_probe_ladder.py --init scratch \\
        --variants clean snr_l4 psf_l4 misalign_l4

    # domain gap: train on the clean rung, evaluate that checkpoint on every variant
    python terra_sat_drift/batch_probe_ladder.py --train-variant clean --seeds 42 43 44 \\
        --results-dir .../outputs/ladder_probe_xc

    # re-aggregate the table from existing run records without training anything
    python terra_sat_drift/batch_probe_ladder.py --collect-only
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent

DEFAULT_LADDER_ROOT = Path("/shared/home/elucas/datasets/sen1floods11_ladder")
DEFAULT_RESULTS_DIR = Path("/shared/home/elucas/scratch/terra-sat-drift/outputs/ladder_probe")
DEFAULT_OUTPUT_ROOT = Path("/shared/home/elucas/scratch/terra-sat-drift/outputs")

# Ladder order: the clean baseline first, then each factor from mild to v6-strength.
FACTOR_ORDER = ("snr", "psf", "misalign", "joint")
LEVELS = (1, 2, 3, 4)

# Metrics pulled out of each run record into the table. The student probe and the
# TerraMind task name the same quantities differently, so each column lists the candidate
# keys in preference order and the first one present wins.
METRIC_KEYS = {
    "test_iou": ["test/student_target_iou", "test/mIoU"],
    "test_iou_fixed": ["test/student_target_iou_fixed"],
    "test_iou_water": ["test_per_class/IoU_water", "test/IoU_flood"],
    "test_iou_not_water": ["test_per_class/IoU_not_water", "test/IoU_background"],
    "test_loss": ["test/loss"],
}


def pick_metric(test: dict, candidates: list[str]):
    for key in candidates:
        if key in test:
            return test[key]
    return None


PROBE_SCRIPTS = {
    "student": _HERE / "finetune_floods_student.py",
    "terramind": _HERE / "finetune_floods_terramind.py",
}

# Defaults that differ per model: the student probe trains a small head at a high LR,
# TerraMind fine-tunes end to end at the rate the reference script settled on.
MODEL_DEFAULTS = {
    "student": {"lr": 1e-3, "patience": 15, "crop": 256, "seconds_per_run": 800},
    # TerraMind measured at ~3.3 min/epoch cold (data-loading bound: every sample is a
    # 37 MB tif off NFS), plus ~34 s per full-scene test pass.
    "terramind": {"lr": 2e-5, "patience": 10, "crop": 512, "seconds_per_run": 5400},
}



def discover_variants(ladder_root: Path) -> list[str]:
    """Ladder variants present on disk, in ladder order."""
    present = {p.name for p in ladder_root.iterdir()
               if p.is_dir() and (p / "v1.1" / "splits").is_dir()}
    ordered = [v for v in ["clean"] + [f"{f}_l{l}" for f in FACTOR_ORDER for l in LEVELS]
               if v in present]
    return ordered + sorted(present - set(ordered))


def parse_variant(name: str) -> tuple[str, int]:
    """('psf_l3') -> ('psf', 3); the clean baseline is level 0 of every factor."""
    if name == "clean":
        return "none", 0
    factor, _, level = name.rpartition("_l")
    return (factor, int(level)) if factor and level.isdigit() else (name, -1)


def severity_of(variant: str, ladder_root: Path) -> Optional[float]:
    """Severity recorded by the generator, so the table cannot drift from the data."""
    meta = ladder_root / variant / "variant.json"
    if meta.is_file():
        try:
            return float(json.loads(meta.read_text())["severity"])
        except (KeyError, ValueError, json.JSONDecodeError):
            pass
    return None


@dataclass
class Cell:
    variant: str
    seed: int
    init: str  # "distilled" | "scratch"
    cross: bool = False  # train on `variant`, also evaluate on every other variant

    @property
    def name(self) -> str:
        prefix = "xc_" if self.cross else ""
        return f"{prefix}{self.variant}__{self.init}__s{self.seed}"


def build_cells(variants: list[str], seeds: list[int], inits: list[str],
                cross: bool = False) -> list[Cell]:
    return [Cell(v, s, i, cross) for v in variants for i in inits for s in seeds]


def freezes_backbone(args: argparse.Namespace) -> bool:
    """Whether the TerraMind ViT is frozen.

    The student distinguishes 'encoder' (train decoder + head) from 'backbone' (train the
    1x1 head only); TerraMind's ViT is simply its encoder, so both requests mean the same
    thing there -- freeze the ViT, train the UNet decoder and head. Only ``--freeze none``
    fine-tunes end to end.
    """
    return args.freeze in ("encoder", "backbone")


def probe_command(cell: Cell, args: argparse.Namespace, metrics_path: Path) -> list[str]:
    if args.model == "student":
        tag = f"ladder_{cell.name}"
    else:
        # The freeze mode has to be in the tag: the same cell run frozen and fine-tuned
        # would otherwise write checkpoints into one directory.
        mode = "frozen" if freezes_backbone(args) else "ft"
        tag = f"ladder_terramind_{args.backbone_size}_{mode}_{cell.name}"
    # Flags both probe scripts understand.
    cmd = [
        sys.executable, str(PROBE_SCRIPTS[args.model]),
        "--sim-root", str(args.ladder_root / cell.variant),
        "--seed", str(cell.seed),
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--batch-size", str(args.batch_size),
        "--crop", str(args.crop),
        "--lr", str(args.lr),
        "--num-workers", str(args.num_workers),
        # A tag per cell keeps checkpoints from different variants out of one directory.
        "--tag", tag,
        "--output-root", str(args.output_root),
        "--metrics-out", str(metrics_path),
        "--wandb-project", args.wandb_project,
    ]

    if args.model == "student":
        cmd += [
            "--freeze", args.freeze,
            "--dsbn-domain", str(args.dsbn_domain),
            "--task-loss", args.task_loss,
            "--class-weights", args.class_weights,
            "--domain-stats", args.domain_stats,
        ]
        # "scratch" means a randomly initialised encoder in both models, but the flag
        # that requests it differs.
        cmd += ["--scratch"] if cell.init == "scratch" else ["--student-ckpt", str(args.student_ckpt)]
        if not args.eval_crop:
            # Match the TerraMind probe, which evaluates whole scenes.
            cmd.append("--eval-full-scene")
        if args.eval_batch_size:
            cmd += ["--eval-batch-size", str(args.eval_batch_size)]
    else:
        cmd += ["--backbone-size", args.backbone_size, "--loss", args.tm_loss]
        if args.eval_batch_size:
            cmd += ["--eval-batch-size", str(args.eval_batch_size)]
        if args.eval_crop:
            cmd.append("--eval-crop")
        if freezes_backbone(args):
            cmd.append("--freeze-backbone")
        if cell.init == "scratch":
            cmd.append("--no-pretrained")

    if args.no_wandb:
        cmd.append("--no-wandb")
    if cell.cross:
        cmd += ["--test-roots"] + [str(args.ladder_root / v) for v in args.test_variants]
    return cmd


def run_cell(cell: Cell, args: argparse.Namespace) -> tuple[str, float]:
    """Train one cell. Returns (status, seconds)."""
    metrics_path = args.results_dir / "runs" / f"{cell.name}.json"
    if metrics_path.is_file() and not args.force:
        return "skipped", 0.0

    log_dir = args.results_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{cell.name}.log"

    started = time.time()
    with open(log_path, "a") as log_file:
        log_file.write(f"\n{'=' * 80}\n=== {cell.name} @ {time.ctime(started)}\n{'=' * 80}\n")
        log_file.flush()
        proc = subprocess.run(
            probe_command(cell, args, metrics_path),
            cwd=str(_HERE.parent), stdout=log_file, stderr=subprocess.STDOUT,
        )
    elapsed = time.time() - started

    if proc.returncode != 0:
        return f"failed (exit {proc.returncode}, see {log_path})", elapsed
    if not metrics_path.is_file():
        return f"failed (no metrics written, see {log_path})", elapsed
    return "done", elapsed


# --------------------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------------------


def collect(args: argparse.Namespace) -> list[dict]:
    """Read every run record into flat rows, ordered along the ladder.

    One row per (run, test condition). A matched-condition run contributes a single row
    where ``train_variant == variant``; a cross-condition run contributes one row per
    test set it was evaluated on, which is what makes the domain gap readable as
    ``variant`` varying at fixed ``train_variant``.
    """
    rows = []
    for path in sorted((args.results_dir / "runs").glob("*.json")):
        try:
            rec = json.loads(path.read_text())
        except json.JSONDecodeError:
            print(f"  ! unreadable record: {path}", flush=True)
            continue
        sim_root = Path(rec["sim_root"])
        train_variant = sim_root.parent.name if sim_root.name == "v1.1" else sim_root.name

        model = rec.get("model", "student")
        # "scratch" means a random encoder in both models, but the informative label
        # differs: the student's alternative is a distilled encoder, TerraMind's is a
        # pretrained one.
        if model.startswith("terramind"):
            init_label = "random" if rec.get("scratch") else "pretrained"
        else:
            init_label = "scratch" if rec.get("scratch") else "distilled"

        def make_row(variant: str, test: dict) -> dict:
            factor, level = parse_variant(variant)
            row = {
                "train_variant": train_variant,
                "variant": variant,
                "factor": factor,
                "level": level,
                "severity": severity_of(variant, args.ladder_root),
                "init": init_label,
                "seed": rec.get("seed"),
                "freeze": rec.get("freeze"),
                "epochs_run": rec.get("epochs_run"),
                "best_val_iou": rec.get("best_val_iou"),
            }
            row["model"] = model
            for short, keys in METRIC_KEYS.items():
                row[short] = pick_metric(test, keys)
            return row

        cross = rec.get("cross_condition") or {}
        if cross:
            rows += [make_row(v, t) for v, t in cross.items()]
            # Keep the matched point if the training condition was not itself a test set.
            if train_variant not in cross:
                rows.append(make_row(train_variant, rec.get("test", {})))
        else:
            rows.append(make_row(train_variant, rec.get("test", {})))

    order = {v: i for i, v in enumerate(["clean"] + [f"{f}_l{l}" for f in FACTOR_ORDER for l in LEVELS])}
    rows.sort(key=lambda r: (r["init"] != "distilled", order.get(r["train_variant"], 99),
                             order.get(r["variant"], 99),
                             r["seed"] if r["seed"] is not None else 0))
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarise(rows: list[dict], metric: str = "test_iou_water") -> list[dict]:
    """Mean and spread of ``metric`` per (init, variant), across seeds."""
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for r in rows:
        groups.setdefault((r["init"], r["train_variant"], r["variant"]), []).append(r)

    out = []
    for (init, train_variant, variant), items in groups.items():
        vals = [i[metric] for i in items if i.get(metric) is not None]
        overall = [i["test_iou"] for i in items if i.get("test_iou") is not None]
        if not vals:
            continue
        factor, level = parse_variant(variant)
        out.append({
            "init": init,
            "train_variant": train_variant,
            "variant": variant,
            "factor": factor,
            "level": level,
            "severity": items[0]["severity"],
            "n_seeds": len(vals),
            "mean": statistics.mean(vals),
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals),
            "max": max(vals),
            "mean_test_iou": statistics.mean(overall) if overall else None,
            # Kept per seed so the delta against the baseline can be paired. Absolute
            # IoU swings with how well a given training run happened to go; the drop
            # between two test sets scored by the *same* checkpoint does not.
            "by_seed": {i["seed"]: i[metric] for i in items if i.get(metric) is not None},
        })
    order = {v: i for i, v in enumerate(["clean"] + [f"{f}_l{l}" for f in FACTOR_ORDER for l in LEVELS])}
    out.sort(key=lambda r: (r["init"] != "distilled", order.get(r["train_variant"], 99),
                            order.get(r["variant"], 99)))
    return out


def write_summary(rows: list[dict], summary: list[dict], path: Path, metric: str) -> str:
    lines = [
        "# Flood probe across the Φ-sat-2 degradation ladder",
        "",
        f"Metric: `{METRIC_KEYS.get(metric, metric)}` on the test split "
        f"(mean +- sd over seeds); `mean_test_iou` is the two-class mean IoU.",
        "",
        "| init | trained on | tested on | factor | severity | n | mean | sd | min | max | mean mIoU |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    # Baseline: same training condition, tested on the clean rung.
    baseline = {(s["init"], s["train_variant"]): s["mean"]
                for s in summary if s["variant"] == "clean"}
    baseline_by_seed = {(s["init"], s["train_variant"]): s["by_seed"]
                        for s in summary if s["variant"] == "clean"}
    for s in summary:
        sev = "-" if s["severity"] is None else f"{s['severity']:.3f}"
        miou = "-" if s["mean_test_iou"] is None else f"{s['mean_test_iou']:.4f}"
        lines.append(
            f"| {s['init']} | `{s['train_variant']}` | `{s['variant']}` | {s['factor']} | {sev} | "
            f"{s['n_seeds']} | **{s['mean']:.4f}** | {s['std']:.4f} | {s['min']:.4f} | "
            f"{s['max']:.4f} | {miou} |"
        )

    # Drop from the clean baseline is the number the ladder exists to produce. When the
    # training condition is held fixed this is the domain gap; when it tracks the test
    # condition it is the matched-condition difficulty.
    deltas = [s for s in summary
              if s["variant"] != "clean" and (s["init"], s["train_variant"]) in baseline]
    if deltas:
        lines += ["", "## Drop from the clean test set", "",
                  "Deltas are **paired**: for each seed the drop is measured against that same "
                  "seed's clean score, and the spread reported is the spread of those per-seed "
                  "drops. Comparing against the unpaired sd of the absolute IoU would understate "
                  "significance badly here, because absolute IoU varies with how well an "
                  "individual training run went while the drop between two test sets scored by "
                  "the same checkpoint does not.",
                  "",
                  "| init | trained on | tested on | severity | delta | sd of delta | vs. sd |",
                  "|---|---|---|---|---|---|---|"]
        for s in deltas:
            key = (s["init"], s["train_variant"])
            sev = "-" if s["severity"] is None else f"{s['severity']:.3f}"
            base_seeds = baseline_by_seed.get(key, {})
            paired = [s["by_seed"][sd] - base_seeds[sd]
                      for sd in s["by_seed"] if sd in base_seeds]

            if paired:
                d = statistics.mean(paired)
                sd = statistics.stdev(paired) if len(paired) > 1 else 0.0
            else:  # no shared seeds; fall back to the difference of means
                d, sd = s["mean"] - baseline[key], 0.0

            if len(paired) < 2:
                verdict, sd_txt = "n/a (1 seed)", "-"
            elif sd == 0:
                verdict, sd_txt = "exact", "0.0000"
            else:
                ratio = abs(d) / sd
                verdict = f"{ratio:.1f}x" + ("" if ratio >= 2 else "  (within noise)")
                sd_txt = f"{sd:.4f}"
            lines.append(f"| {s['init']} | `{s['train_variant']}` | `{s['variant']}` | {sev} | "
                         f"{d:+.4f} | {sd_txt} | {verdict} |")

    lines += ["", f"Rows: {len(rows)} runs. Generated {time.strftime('%Y-%m-%d %H:%M')}.", ""]
    text = "\n".join(lines)
    path.write_text(text)
    return text


def aggregate(args: argparse.Namespace, quiet: bool = False) -> None:
    rows = collect(args)
    if not rows:
        if not quiet:
            print("No run records found yet.", flush=True)
        return
    csv_path = args.results_dir / "ladder_probe_results.csv"
    md_path = args.results_dir / "ladder_probe_summary.md"
    write_csv(rows, csv_path)
    text = write_summary(rows, summarise(rows, args.metric), md_path, args.metric)
    if not quiet:
        print("\n" + text, flush=True)
        print(f"CSV:     {csv_path}\nSummary: {md_path}", flush=True)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the frozen-encoder flood probe on every degradation-ladder variant.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="student", choices=sorted(PROBE_SCRIPTS),
                   help="'student' probes the distilled MobileNet with a frozen encoder; "
                        "'terramind' fine-tunes the TerraMind ViT end to end. Each brings its "
                        "own lr/patience/crop defaults.")
    p.add_argument("--backbone-size", default="base", help="TerraMind size (terramind model only)")
    p.add_argument("--tm-loss", default="dice", help="Loss for the TerraMind task")
    p.add_argument("--eval-batch-size", type=int, default=None,
                   help="Batch size for the full-scene val/test passes")
    p.add_argument("--eval-crop", action="store_true",
                   help="Score a centre crop on val/test instead of the whole scene. Both models "
                        "evaluate full scenes by default, which is what makes their domain gaps "
                        "comparable as absolute numbers and not merely as relative drops.")
    p.add_argument("--ladder-root", type=Path, default=DEFAULT_LADDER_ROOT,
                   help="Directory holding the ladder variants (default: %(default)s)")
    p.add_argument("--variants", nargs="+", default=None,
                   help="Variants to probe (default: every variant found under --ladder-root)")
    p.add_argument("--train-variant", default=None,
                   help="Domain-gap mode: train only on this variant, then evaluate the resulting "
                        "checkpoint on every variant's test split. Without it each variant is "
                        "trained and tested on itself (matched-condition difficulty).")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44],
                   help="Seeds per variant (default: %(default)s). Repeats are what make the "
                        "curve interpretable against run-to-run variance.")
    p.add_argument("--init", nargs="+", default=["distilled"], choices=["distilled", "scratch"],
                   help="Encoder initialisation(s) to run (default: distilled only)")
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
                   help="Where the probe writes checkpoints, one sub-directory per cell")
    p.add_argument("--student-ckpt", type=str,
                   default="outputs/xsensor_kd_terramind_v1_base_kd1.0_inst0.0_pix0.0_dsbn"
                           "/checkpoints/best-89-0.4607.ckpt")
    p.add_argument("--metric", type=str, default="test_iou_water", choices=list(METRIC_KEYS),
                   help="Metric summarised per variant (default: %(default)s -- the water class "
                        "is the one that actually moves)")

    # Forwarded verbatim to the probe.
    p.add_argument("--freeze", default="encoder", choices=["encoder", "backbone", "none"])
    p.add_argument("--dsbn-domain", type=int, default=0, choices=[0, 1])
    p.add_argument("--epochs", type=int, default=100)
    # None means "take the default for --model"; see MODEL_DEFAULTS.
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--crop", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--task-loss", default="ce", choices=["ce", "focal"])
    p.add_argument("--class-weights", default="none",
                   choices=["none", "inverse", "inverse_sqrt"])
    p.add_argument("--domain-stats", default="sim", choices=["sim", "s2b", "real"])
    p.add_argument("--wandb-project", default="terra-sat-drift")
    p.add_argument("--no-wandb", action="store_true")

    p.add_argument("--force", action="store_true", help="Re-run cells that already have results")
    p.add_argument("--dry-run", action="store_true", help="Print the matrix and exit")
    p.add_argument("--collect-only", action="store_true",
                   help="Rebuild the table from existing records without training")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    args.ladder_root = args.ladder_root.resolve()
    args.results_dir = args.results_dir.resolve()
    (args.results_dir / "runs").mkdir(parents=True, exist_ok=True)

    defaults = MODEL_DEFAULTS[args.model]
    for key in ("lr", "patience", "crop"):
        if getattr(args, key) is None:
            setattr(args, key, defaults[key])
    seconds_per_run = defaults["seconds_per_run"]

    if args.collect_only:
        aggregate(args)
        return 0

    if not args.ladder_root.is_dir():
        raise SystemExit(f"Ladder root not found: {args.ladder_root}")
    variants = args.variants or discover_variants(args.ladder_root)
    unknown = [v for v in variants if not (args.ladder_root / v / "v1.1").is_dir()]
    if unknown:
        raise SystemExit(f"No such variant(s) under {args.ladder_root}: {unknown}")

    if args.train_variant:
        if not (args.ladder_root / args.train_variant / "v1.1").is_dir():
            raise SystemExit(f"No such training variant: {args.train_variant}")
        # Every variant becomes a test set; only the one training condition is trained.
        args.test_variants = variants
        cells = build_cells([args.train_variant], args.seeds, args.init, cross=True)
    else:
        args.test_variants = []
        cells = build_cells(variants, args.seeds, args.init)

    pending = [c for c in cells
               if args.force or not (args.results_dir / "runs" / f"{c.name}.json").is_file()]

    if args.train_variant:
        print(f"Domain-gap mode: train on '{args.train_variant}', test on {len(variants)} variants")
        print(f"{len(args.seeds)} seeds x {len(args.init)} init = {len(cells)} training runs "
              f"({len(pending)} pending), each followed by {len(variants)} evaluations")
    else:
        print(f"{len(variants)} variants x {len(args.seeds)} seeds x {len(args.init)} init "
              f"= {len(cells)} runs ({len(pending)} pending, "
              f"{len(cells) - len(pending)} already done)")
    print(f"model:    {args.model}"
          + (f" ({args.backbone_size})" if args.model == "terramind" else f" (freeze={args.freeze})"))
    print(f"variants: {', '.join(variants)}")
    print(f"seeds:    {args.seeds}   init: {args.init}   "
          f"lr: {args.lr}   crop: {args.crop}   patience: {args.patience}")
    print(f"estimated: ~{len(pending) * seconds_per_run / 3600:.1f} h "
          f"at ~{seconds_per_run / 60:.0f} min/run (early stopping)")
    print(f"results:  {args.results_dir}\n", flush=True)

    if args.dry_run:
        for c in cells:
            state = "done" if (args.results_dir / "runs" / f"{c.name}.json").is_file() else "pending"
            print(f"  {c.name:<38} {state}")
        return 0

    # One GPU, so cells run one at a time.
    failures = []
    for i, cell in enumerate(cells, 1):
        status, elapsed = run_cell(cell, args)
        mark = {"done": "ok", "skipped": "--"}.get(status, "FAILED")
        print(f"[{i}/{len(cells)}] {mark:>6}  {cell.name}  ({elapsed / 60:.1f} min)"
              f"{'' if status in ('done', 'skipped') else '  ' + status}", flush=True)
        if status not in ("done", "skipped"):
            failures.append((cell.name, status))
        # Refresh the table as we go, so the curve is readable mid-sweep.
        aggregate(args, quiet=True)

    print(f"\nSweep finished: {len(cells) - len(failures)} ok, {len(failures)} failed.", flush=True)
    for name, status in failures:
        print(f"  {name}: {status}", flush=True)

    aggregate(args)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
