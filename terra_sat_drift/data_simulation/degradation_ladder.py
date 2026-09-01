"""Generate a Φ-sat-2 degradation ladder: one dataset per isolated noise factor and level.

The existing simulated datasets (``sen1floods11_simulated_alt_v1..v6``) ramp every noise
factor at once, so a drift measured against them cannot be attributed to any single
factor. This script sweeps the factors one at a time.

Every variant shares the same deterministic backbone -- radiance conversion, PAN band,
spatial resampling to 4.75 m, reflectance conversion -- so all outputs land on an
identical grid (1077x1077, 8 bands, float32) and differ *only* by the factor under test.

Three factors are swept over four levels each, on a common severity axis defined
relative to the ``alt_v6`` settings (severity 1.0 == v6):

    level        severity   snr_values   psf_kernel_sigma   misalignment_std
    L1           0.125      [40, 80]     0.5                1.25
    L2           0.25       [20, 40]     1.0                2.5
    L3           0.5        [10, 20]     2.0                5.0
    L4           1.0        [5, 10]      4.0                10.0

Severity halves the perturbation magnitude at each step down: SNR noise amplitude scales
as 1/SNR, so the four SNR ranges give relative noise of 1/8, 1/4, 1/2, 1; the PSF sigma
and misalignment standard deviation follow the same doubling.

Variants produced:

    clean                       backbone only, no noise -- the level-0 rung of every ladder
    snr_l1..snr_l4              SNR only
    psf_l1..psf_l4              PSF blur only
    misalign_l1..misalign_l4    band misalignment only
    joint_l1..joint_l4          all three together (optional, --factors joint);
                                joint_l4 reproduces the alt_v6 configuration

Each variant is written as a self-contained, v6-style dataset root::

    <output-root>/<variant>/
        variant.json                                    factor, level, severity, params
        simulation.log
        v1.1/simulation_config.json                     SimulationConfig dump
        v1.1/splits/...                                 copied from the source dataset
        v1.1/Sen1Floods11_Metadata.geojson              copied from the source dataset
        v1.1/data/flood_events/HandLabeled/S2Hand/      simulated_L1C_<stem>_S2Hand.tif
        v1.1/data/flood_events/HandLabeled/LabelHand    symlink to the source labels

so it can be handed straight to ``Sen1Floods11NonGeo(data_root=<variant>)`` or to the
drift loader's ``simulated_dir``.

Each (variant, split) runs in its own subprocess: eo-learn's executor, the logging setup
and the peak memory are all per-process, and a failure in one variant cannot take down
the rest of the sweep. Completed work is detected from the files on disk, so the sweep is
resumable -- rerun the same command and it picks up where it stopped.

Examples::

    # inspect the matrix, disk and time budget without running anything
    python degradation_ladder.py --dry-run

    # smoke test: 2 samples per (variant, split) into a throwaway root
    python degradation_ladder.py --max-files 2 --output-root /tmp/ladder_smoke

    # the real sweep
    python degradation_ladder.py --jobs 2 --workers 4

    # just the PSF ladder, top two levels
    python degradation_ladder.py --factors psf --levels 3 4
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

# The simulation modules import each other flat (``from phisat2_constants import ...``),
# so the package directory has to be importable regardless of the caller's cwd.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


DEFAULT_DATASET_ROOT = Path("/shared/home/elucas/datasets/sen1floods11")
DEFAULT_OUTPUT_ROOT = Path("/shared/home/elucas/datasets/sen1floods11_ladder")

# Splits as named by the CLI / terratorch, mapped to the split-file stems on disk.
SPLIT_FILE_STEMS = {"train": "train", "val": "valid", "test": "test"}
ALL_SPLITS = ("train", "val", "test")

SPLIT_DIR = "v1.1/splits"
S2_DIR = "v1.1/data/flood_events/HandLabeled/S2Hand"
LABEL_DIR = "v1.1/data/flood_events/HandLabeled/LabelHand"
METADATA_FILE = "v1.1/Sen1Floods11_Metadata.geojson"

# Bands requested from the source dataset, and the Φ-sat-2 band ids they map to.
BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08"]
BANDS_NAMES = ["BLUE", "GREEN", "RED", "RED_EDGE_1", "RED_EDGE_2", "RED_EDGE_3", "NIR_BROAD"]

LEVELS = (1, 2, 3, 4)
FACTORS = ("snr", "psf", "misalign", "joint")
ISOLATED_FACTORS = ("snr", "psf", "misalign")

# Severity relative to the alt_v6 reference configuration (level 4 == v6).
SEVERITY = {1: 0.125, 2: 0.25, 3: 0.5, 4: 1.0}
# rng.integers(lo, hi) picks one SNR per patch; noise amplitude scales as 1 / SNR.
SNR_RANGES = {1: [40, 80], 2: [20, 40], 3: [10, 20], 4: [5, 10]}
# Gaussian PSF sigma in resampled (4.75 m) pixels.
PSF_SIGMAS = {1: 0.5, 2: 1.0, 3: 2.0, 4: 4.0}
# Per-band random shift std in resampled pixels, applied to sea and land alike so the
# factor stays a single scalar (v6 also used one value for both).
MISALIGN_STDS = {1: 1.25, 2: 2.5, 3: 5.0, 4: 10.0}

# Held constant across the whole ladder.
RADIANCE_REFERENCE = 10000
SOURCE_RESOLUTION = 10.0
SNR_PSF_METHOD = "alternative"
PROCESSING_LEVEL = "L1C"

# Observed on alt_v6: ~35 MB per output tif, ~2.1 s per sample with a single worker.
MB_PER_FILE = 35.0
SECONDS_PER_FILE = 2.1


@dataclass
class Variant:
    """One rung of the ladder: which factor is active, at which level."""

    name: str
    factor: str  # "none" | "snr" | "psf" | "misalign" | "joint"
    level: int  # 0 for the clean baseline
    severity: float
    snr: bool
    psf: bool
    misalign: bool
    snr_values: Optional[list[int]]
    psf_kernel_sigma: float
    misalignment_std: float
    description: str = ""

    def steps(self) -> dict:
        return {
            "spatial_resampling": True,
            "radiance": True,
            "add_panchromatic": True,
            "band_misalignment": self.misalign,
            "snr_simulation": self.snr,
            "psf_filtering": self.psf,
            "reflectance_conversion": True,
        }

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "factor": self.factor,
            "level": self.level,
            "severity": self.severity,
            "description": self.description,
            "active_steps": [k for k, v in self.steps().items() if v],
            "parameters": {
                "snr_values": self.snr_values,
                "psf_kernel_sigma": self.psf_kernel_sigma,
                "misalignment_std_sea": self.misalignment_std,
                "misalignment_std_land": self.misalignment_std,
                "radiance_reference": RADIANCE_REFERENCE,
                "snr_psf_method": SNR_PSF_METHOD,
                "processing_level": PROCESSING_LEVEL,
            },
        }


def make_variant(factor: str, level: int) -> Variant:
    """Build the variant for ``factor`` at ``level``.

    Parameters belonging to an inactive factor are pinned to inert values (``None`` SNR
    range, unit PSF sigma, zero misalignment std) so a saved ``simulation_config.json``
    never suggests a perturbation that was not applied.
    """
    if factor == "none":
        return Variant(
            name="clean",
            factor="none",
            level=0,
            severity=0.0,
            snr=False,
            psf=False,
            misalign=False,
            snr_values=None,
            psf_kernel_sigma=1.0,
            misalignment_std=0,
            description="Deterministic backbone only (radiance, PAN, 4.75 m resampling, reflectance)",
        )

    if factor not in FACTORS:
        raise ValueError(f"Unknown factor {factor!r}, expected one of {FACTORS}")
    if level not in LEVELS:
        raise ValueError(f"Unknown level {level!r}, expected one of {LEVELS}")

    use_snr = factor in ("snr", "joint")
    use_psf = factor in ("psf", "joint")
    use_mis = factor in ("misalign", "joint")

    if factor == "snr":
        desc = f"SNR only, base SNR drawn from {SNR_RANGES[level]}"
    elif factor == "psf":
        desc = f"PSF blur only, Gaussian sigma {PSF_SIGMAS[level]} px @ 4.75 m"
    elif factor == "misalign":
        desc = f"Band misalignment only, shift std {MISALIGN_STDS[level]} px @ 4.75 m"
    else:
        desc = f"All three factors at severity {SEVERITY[level]}"
        if level == 4:
            desc += " (reproduces the alt_v6 configuration)"

    return Variant(
        name=f"{factor}_l{level}",
        factor=factor,
        level=level,
        severity=SEVERITY[level],
        snr=use_snr,
        psf=use_psf,
        misalign=use_mis,
        snr_values=list(SNR_RANGES[level]) if use_snr else None,
        psf_kernel_sigma=PSF_SIGMAS[level] if use_psf else 1.0,
        misalignment_std=MISALIGN_STDS[level] if use_mis else 0,
        description=desc,
    )


def build_variants(
    factors: Iterable[str],
    levels: Iterable[int],
    include_clean: bool = True,
) -> list[Variant]:
    variants: list[Variant] = []
    if include_clean:
        variants.append(make_variant("none", 0))
    for factor in factors:
        for level in sorted(levels):
            variants.append(make_variant(factor, level))
    return variants


def build_config(variant: Variant):
    """Turn a variant into the SimulationConfig the pipeline consumes."""
    from phisat2_constants import ProcessingLevels
    from simulation_config import SimulationConfig, SimulationSteps

    return SimulationConfig(
        bands_names=BANDS,
        source_resolution=SOURCE_RESOLUTION,
        steps=SimulationSteps(**variant.steps()),
        processing_level=ProcessingLevels[PROCESSING_LEVEL],
        phisat2_exec_path=None,
        snr_psf_method=SNR_PSF_METHOD,
        misalignment_std_sea=variant.misalignment_std,
        misalignment_std_land=variant.misalignment_std,
        snr_values=variant.snr_values,
        psf_kernel_sigma=variant.psf_kernel_sigma,
        radiance_reference=RADIANCE_REFERENCE,
    )


# --------------------------------------------------------------------------------------
# Progress / completion tracking
# --------------------------------------------------------------------------------------


def split_stems(dataset_root: Path, split: str) -> list[str]:
    """Sample stems belonging to ``split``, in the order the dataset loader yields them.

    ``Sen1Floods11NonGeo`` sorts the S2 file glob and then filters it by the split file,
    so an alphabetical sort of the stems reproduces that order -- which is what
    ``--max-files`` truncates.
    """
    split_file = dataset_root / SPLIT_DIR / "flood_handlabeled" / f"flood_{SPLIT_FILE_STEMS[split]}_data.txt"
    stems = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
    return sorted(stems)


def expected_outputs(dataset_root: Path, split: str, max_files: Optional[int]) -> list[str]:
    stems = split_stems(dataset_root, split)
    if max_files is not None:
        stems = stems[:max_files]
    return [f"simulated_{PROCESSING_LEVEL}_{stem}_S2Hand.tif" for stem in stems]


def missing_outputs(
    variant_dir: Path, dataset_root: Path, split: str, max_files: Optional[int]
) -> list[str]:
    s2_dir = variant_dir / S2_DIR
    if not s2_dir.is_dir():
        return expected_outputs(dataset_root, split, max_files)
    present = {p.name for p in s2_dir.glob("*_S2Hand.tif") if p.stat().st_size > 0}
    return [name for name in expected_outputs(dataset_root, split, max_files) if name not in present]


# --------------------------------------------------------------------------------------
# Per-variant execution (child process)
# --------------------------------------------------------------------------------------


def run_one(
    variant: Variant,
    split: str,
    dataset_root: Path,
    output_root: Path,
    max_files: Optional[int],
    workers: int,
    verbose: bool,
) -> None:
    """Simulate a single (variant, split). Runs in a dedicated subprocess."""
    from batch_simulate_sen1floods import simulate_sen1floods_s2

    variant_dir = output_root / variant.name
    variant_dir.mkdir(parents=True, exist_ok=True)
    write_variant_metadata(variant, variant_dir)

    simulate_sen1floods_s2(
        dataset_root=dataset_root,
        output_dir=variant_dir,
        split=split,
        max_files=max_files,
        config=build_config(variant),
        verbose=verbose,
        workers=workers,
    )

    materialise_dataset_root(variant_dir, dataset_root)


def write_variant_metadata(variant: Variant, variant_dir: Path) -> None:
    payload = variant.as_dict()
    payload["ladder"] = {
        "severity_axis": SEVERITY,
        "snr_ranges": SNR_RANGES,
        "psf_sigmas": PSF_SIGMAS,
        "misalignment_stds": MISALIGN_STDS,
        "reference": "severity 1.0 == sen1floods11_simulated_alt_v6",
    }
    (variant_dir / "variant.json").write_text(json.dumps(payload, indent=2) + "\n")


def materialise_dataset_root(variant_dir: Path, dataset_root: Path) -> None:
    """Add the splits, metadata and label links that make a variant loadable on its own."""
    src_splits = dataset_root / SPLIT_DIR
    dst_splits = variant_dir / SPLIT_DIR
    if src_splits.is_dir():
        shutil.copytree(src_splits, dst_splits, dirs_exist_ok=True)

    # Needed by Sen1Floods11NonGeo(use_metadata=True); the simulation does not alter it.
    src_meta = dataset_root / METADATA_FILE
    dst_meta = variant_dir / METADATA_FILE
    if src_meta.is_file() and not dst_meta.exists():
        dst_meta.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_meta, dst_meta)

    # Labels are untouched by the simulation, so link rather than copy them.
    src_labels = dataset_root / LABEL_DIR
    dst_labels = variant_dir / LABEL_DIR
    if src_labels.is_dir() and not dst_labels.exists():
        dst_labels.parent.mkdir(parents=True, exist_ok=True)
        try:
            dst_labels.symlink_to(src_labels, target_is_directory=True)
        except OSError:
            pass  # a filesystem without symlinks is not worth failing the run over


# --------------------------------------------------------------------------------------
# Sweep orchestration (parent process)
# --------------------------------------------------------------------------------------


@dataclass
class RunResult:
    variant: str
    split: str
    status: str  # "done" | "skipped" | "failed"
    seconds: float = 0.0
    n_files: int = 0
    message: str = ""


def child_command(
    variant: Variant,
    split: str,
    args: argparse.Namespace,
) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--run-one",
        variant.name,
        "--split",
        split,
        "--dataset-root",
        str(args.dataset_root),
        "--output-root",
        str(args.output_root),
        "--workers",
        str(args.workers),
    ]
    if args.max_files is not None:
        cmd += ["--max-files", str(args.max_files)]
    if args.verbose:
        cmd.append("--verbose")
    return cmd


def execute_pair(variant: Variant, split: str, args: argparse.Namespace) -> RunResult:
    variant_dir = args.output_root / variant.name
    missing = missing_outputs(variant_dir, args.dataset_root, split, args.max_files)
    expected = len(expected_outputs(args.dataset_root, split, args.max_files))

    if not missing and not args.force:
        return RunResult(variant.name, split, "skipped", n_files=expected,
                         message="all outputs already present")

    started = time.time()
    log_dir = args.output_root / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{variant.name}.{split}.log"

    with open(log_path, "a") as log_file:
        log_file.write(f"\n{'=' * 80}\n=== {variant.name} / {split} @ {time.ctime(started)}\n{'=' * 80}\n")
        log_file.flush()
        proc = subprocess.run(
            child_command(variant, split, args),
            cwd=str(_HERE),
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )

    elapsed = time.time() - started
    still_missing = missing_outputs(variant_dir, args.dataset_root, split, args.max_files)
    produced = expected - len(still_missing)

    if proc.returncode != 0:
        return RunResult(variant.name, split, "failed", elapsed, produced,
                         f"exit code {proc.returncode}, see {log_path}")
    if still_missing:
        return RunResult(variant.name, split, "failed", elapsed, produced,
                         f"{len(still_missing)} outputs missing, see {log_path}")
    return RunResult(variant.name, split, "done", elapsed, produced)


def format_plan(variants: list[Variant], splits: list[str], args: argparse.Namespace) -> str:
    per_variant = sum(len(expected_outputs(args.dataset_root, s, args.max_files)) for s in splits)
    total_files = per_variant * len(variants)

    header = (
        f"{'variant':<14} {'factor':<9} {'sev':>6}  {'snr_values':<12} "
        f"{'psf_sigma':>9} {'mis_std':>8}  files"
    )
    lines = [header, "-" * len(header)]
    for v in variants:
        lines.append(
            f"{v.name:<14} {v.factor:<9} {v.severity:>6.3f}  "
            f"{str(v.snr_values) if v.snr else '-':<12} "
            f"{v.psf_kernel_sigma if v.psf else '-':>9} "
            f"{v.misalignment_std if v.misalign else '-':>8}  {per_variant}"
        )

    hours = total_files * SECONDS_PER_FILE / 3600 / max(1, args.jobs * max(1, args.workers) // 2)
    lines += [
        "-" * len(header),
        f"{len(variants)} variants x {len(splits)} splits ({', '.join(splits)}) = {total_files} tifs",
        f"estimated disk: {total_files * MB_PER_FILE / 1024:.0f} GiB"
        f"   (~{per_variant * MB_PER_FILE / 1024:.1f} GiB per variant)",
        f"estimated wall time: ~{hours:.1f} h at jobs={args.jobs}, workers={args.workers}",
        f"output root: {args.output_root}",
    ]
    return "\n".join(lines)


def write_manifest(path: Path, variants: list[Variant], splits: list[str],
                   args: argparse.Namespace, results: list[RunResult]) -> None:
    by_variant: dict[str, dict] = {}
    for r in results:
        by_variant.setdefault(r.variant, {})[r.split] = {
            "status": r.status,
            "n_files": r.n_files,
            "seconds": round(r.seconds, 1),
            "message": r.message,
        }

    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "generator": str(Path(__file__).resolve()),
        "dataset_root": str(args.dataset_root),
        "output_root": str(args.output_root),
        "splits": splits,
        "max_files": args.max_files,
        "reference_dataset": "/shared/home/elucas/datasets/sen1floods11_simulated_alt_v6",
        "severity_axis": {
            "definition": "1.0 == the alt_v6 configuration; each step down halves the perturbation",
            "levels": SEVERITY,
            "snr_ranges": SNR_RANGES,
            "psf_sigmas": PSF_SIGMAS,
            "misalignment_stds": MISALIGN_STDS,
        },
        "constant_backbone": ["radiance", "add_panchromatic", "spatial_resampling", "reflectance_conversion"],
        "variants": [dict(v.as_dict(), runs=by_variant.get(v.name, {})) for v in variants],
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2) + "\n")
    tmp.replace(path)


def sweep(variants: list[Variant], splits: list[str], args: argparse.Namespace) -> int:
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "ladder_manifest.json"
    pairs = [(v, s) for v in variants for s in splits]
    results: list[RunResult] = []
    total = len(pairs)

    def announce(result: RunResult, index: int) -> None:
        mark = {"done": "ok", "skipped": "--", "failed": "FAILED"}[result.status]
        detail = f" ({result.message})" if result.message else ""
        print(
            f"[{index}/{total}] {mark:>6}  {result.variant}/{result.split}"
            f"  {result.n_files} files in {result.seconds / 60:.1f} min{detail}",
            flush=True,
        )

    print(f"Running {total} (variant, split) jobs with jobs={args.jobs}, workers={args.workers}\n", flush=True)

    if args.jobs <= 1:
        for i, (variant, split) in enumerate(pairs, 1):
            result = execute_pair(variant, split, args)
            results.append(result)
            announce(result, i)
            write_manifest(manifest_path, variants, splits, args, results)
    else:
        # Threads only supervise subprocesses, so the GIL is not in the way.
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(execute_pair, v, s, args) for v, s in pairs]
            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                results.append(result)
                announce(result, i)
                write_manifest(manifest_path, variants, splits, args, results)

    write_manifest(manifest_path, variants, splits, args, results)

    failed = [r for r in results if r.status == "failed"]
    done = [r for r in results if r.status == "done"]
    skipped = [r for r in results if r.status == "skipped"]
    print(
        f"\nSweep finished: {len(done)} run, {len(skipped)} already present, {len(failed)} failed."
        f"\nManifest: {manifest_path}",
        flush=True,
    )
    if failed:
        print("Failures:", flush=True)
        for r in failed:
            print(f"  {r.variant}/{r.split}: {r.message}", flush=True)
    return 1 if failed else 0


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one simulated dataset per isolated Φ-sat-2 noise factor and level.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT,
                        help="Source Sen1Floods11 root (default: %(default)s)")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
                        help="Directory receiving one sub-directory per variant (default: %(default)s)")
    parser.add_argument("--factors", nargs="+", default=list(ISOLATED_FACTORS), choices=list(FACTORS),
                        help="Factors to sweep (default: the three isolated ones; add 'joint' for the combined ladder)")
    parser.add_argument("--levels", nargs="+", type=int, default=list(LEVELS), choices=list(LEVELS),
                        help="Severity levels to generate (default: all four)")
    parser.add_argument("--splits", nargs="+", default=list(ALL_SPLITS), choices=list(ALL_SPLITS),
                        help="Dataset splits to simulate (default: all three)")
    parser.add_argument("--no-clean", action="store_true",
                        help="Skip the noise-free baseline variant")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Only simulate the first N samples of each split (smoke tests)")
    parser.add_argument("--workers", type=int, default=4,
                        help="EOExecutor workers inside each variant run (default: %(default)s)")
    parser.add_argument("--jobs", type=int, default=1,
                        help="Variant runs to execute concurrently (default: %(default)s)")
    parser.add_argument("--force", action="store_true",
                        help="Re-simulate variants whose outputs are already on disk")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the variant matrix, disk and time budget, then exit")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose child logging")
    parser.add_argument("--run-one", metavar="VARIANT", default=None,
                        help=argparse.SUPPRESS)  # internal: simulate a single variant/split
    parser.add_argument("--split", default=None, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def resolve_variant(name: str) -> Variant:
    if name == "clean":
        return make_variant("none", 0)
    factor, _, level = name.rpartition("_l")
    if not factor or not level.isdigit():
        raise SystemExit(f"Cannot parse variant name {name!r} (expected 'clean' or '<factor>_l<level>')")
    return make_variant(factor, int(level))


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    args.dataset_root = args.dataset_root.resolve()
    args.output_root = args.output_root.resolve()

    # Child mode: one variant, one split, in this process.
    if args.run_one:
        if not args.split:
            raise SystemExit("--run-one requires --split")
        run_one(
            variant=resolve_variant(args.run_one),
            split=args.split,
            dataset_root=args.dataset_root,
            output_root=args.output_root,
            max_files=args.max_files,
            workers=args.workers,
            verbose=args.verbose,
        )
        return 0

    variants = build_variants(args.factors, args.levels, include_clean=not args.no_clean)
    splits = [s for s in ALL_SPLITS if s in args.splits]

    print(format_plan(variants, splits, args) + "\n", flush=True)
    if args.dry_run:
        return 0

    return sweep(variants, splits, args)


if __name__ == "__main__":
    raise SystemExit(main())
