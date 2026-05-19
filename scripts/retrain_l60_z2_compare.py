#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import shlex
import statistics
import subprocess
import sys
from pathlib import Path

import torch


DEFAULT_BETAS = "0.01,0.02,0.05,0.1,0.2,0.3,0.4,0.5,0.6"


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.replace(",", " ").split()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrain L=60 SEDD checkpoints with train-time Z2 symmetry and compare A1 across seeds."
    )
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--length", type=int, default=60)
    parser.add_argument("--snapshots-dir", type=Path, default=Path("snapshots"))
    parser.add_argument("--train-root", type=Path, default=Path("runs/length_sweep_L60_z2_seeds"))
    parser.add_argument("--a1-root", type=Path, default=Path("runs/a1_L60_z2_seed_compare"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--level-kind", choices=["beta", "tau", "ell"], default="ell")
    parser.add_argument("--level-min", type=float, default=0.001)
    parser.add_argument("--level-max", type=float, default=2.0)
    parser.add_argument("--sample-kind", choices=["beta", "tau", "ell"], default="ell")
    parser.add_argument("--sample-min", type=float, default=0.001)
    parser.add_argument("--sample-max", type=float, default=2.0)
    parser.add_argument("--val-batches", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--betas", default=DEFAULT_BETAS)
    parser.add_argument("--num-records", type=int, default=256)
    parser.add_argument("--num-posterior-samples", type=int, default=8)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--sweeps-per-step", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-train", action="store_true", help="Reuse existing seed checkpoints.")
    parser.add_argument("--skip-a1", action="store_true", help="Only retrain checkpoints; do not compute A1.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    return parser.parse_args()


def run_command(command: list[str], dry_run: bool) -> None:
    print(shlex.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def train_command(args: argparse.Namespace, seed: int, output_dir: Path) -> list[str]:
    command = [
        sys.executable,
        "scripts/train_lengths.py",
        "--lengths",
        str(args.length),
        "--snapshots-dir",
        str(args.snapshots_dir),
        "--output-dir",
        str(output_dir),
        "--objective",
        "sedd",
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--hidden-channels",
        str(args.hidden_channels),
        "--level-kind",
        args.level_kind,
        "--level-min",
        str(args.level_min),
        "--level-max",
        str(args.level_max),
        "--sample-kind",
        args.sample_kind,
        "--sample-min",
        str(args.sample_min),
        "--sample-max",
        str(args.sample_max),
        "--seed",
        str(seed),
        "--num-workers",
        str(args.num_workers),
        "--device",
        args.device,
        "--z2-symmetrize-train",
    ]
    if args.val_batches is not None:
        command += ["--val-batches", str(args.val_batches)]
    return command


def a1_command(args: argparse.Namespace, seed_train_dir: Path, seed_a1_dir: Path) -> list[str]:
    return [
        sys.executable,
        "scripts/compute_a1_sweep.py",
        "--lengths",
        str(args.length),
        "--betas",
        args.betas,
        "--checkpoints-dir",
        str(seed_train_dir / "checkpoints"),
        "--snapshots-dir",
        str(args.snapshots_dir),
        "--output-dir",
        str(seed_a1_dir),
        "--num-records",
        str(args.num_records),
        "--num-posterior-samples",
        str(args.num_posterior_samples),
        "--steps",
        str(args.steps),
        "--sweeps-per-step",
        str(args.sweeps_per_step),
        "--device",
        args.device,
        "--z2-symmetrize",
    ]


def read_seed_rows(seed: int, path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing A1 result for seed={seed}: {path}")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["seed"] = str(seed)
    return rows


def write_combined(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["seed"] + [name for name in rows[0] if name != "seed"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_seed_summary(path: Path, rows: list[dict[str, str]]) -> None:
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault((row["length"], row["beta"]), []).append(row)

    summary_rows: list[dict[str, float | int | str]] = []
    for (length, beta), group in sorted(groups.items(), key=lambda item: (int(item[0][0]), float(item[0][1]))):
        values = [float(row["a1"]) for row in group]
        reported_stderr = [float(row["a1_stderr"]) for row in group]
        seed_std = statistics.stdev(values) if len(values) > 1 else 0.0
        seed_stderr = seed_std / (len(values) ** 0.5) if len(values) > 1 else 0.0
        summary_rows.append(
            {
                "length": length,
                "beta": beta,
                "num_seeds": len(values),
                "a1_seed_mean": statistics.mean(values),
                "a1_seed_std": seed_std,
                "a1_seed_stderr": seed_stderr,
                "mean_reported_a1_stderr": statistics.mean(reported_stderr),
            }
        )

    fieldnames = [
        "length",
        "beta",
        "num_seeds",
        "a1_seed_mean",
        "a1_seed_std",
        "a1_seed_stderr",
        "mean_reported_a1_stderr",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def main() -> None:
    args = parse_args()
    seeds = parse_int_list(args.seeds)

    for seed in seeds:
        seed_train_dir = args.train_root / f"seed{seed}"
        seed_a1_dir = args.a1_root / f"seed{seed}"
        if not args.skip_train:
            run_command(train_command(args, seed, seed_train_dir), args.dry_run)
        if not args.skip_a1:
            run_command(a1_command(args, seed_train_dir, seed_a1_dir), args.dry_run)

    if args.skip_a1 or args.dry_run:
        return

    rows: list[dict[str, str]] = []
    for seed in seeds:
        rows.extend(read_seed_rows(seed, args.a1_root / f"seed{seed}" / "a1_sweep.csv"))

    combined_path = args.a1_root / "a1_compare.csv"
    summary_path = args.a1_root / "a1_seed_summary.csv"
    write_combined(combined_path, rows)
    write_seed_summary(summary_path, rows)
    print(f"Saved combined A1 comparison to {combined_path}", flush=True)
    print(f"Saved seed summary to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
