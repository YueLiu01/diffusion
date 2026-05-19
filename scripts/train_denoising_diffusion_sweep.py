#!/usr/bin/env python
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import torch


DEFAULT_BETAS = "0.01,0.02,0.05,0.1,0.2,0.3,0.4,0.5,0.6"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train noise-conditioned denoiser checkpoints and compute direct A1 estimates."
    )
    parser.add_argument("--lengths", default="20,40,60,80,100")
    parser.add_argument("--betas", default=DEFAULT_BETAS)
    parser.add_argument("--snapshots-dir", type=Path, default=Path("snapshots"))
    parser.add_argument("--train-dir", type=Path, default=Path("runs/denoising_diffusion_length_sweep"))
    parser.add_argument("--a1-dir", type=Path, default=Path("runs/denoising_diffusion_a1_sweep"))
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
    parser.add_argument("--num-records", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-a1", action="store_true")
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_command(command: list[str], dry_run: bool) -> None:
    print(shlex.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def train_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "scripts/train_lengths.py",
        "--lengths",
        args.lengths,
        "--snapshots-dir",
        str(args.snapshots_dir),
        "--output-dir",
        str(args.train_dir),
        "--objective",
        "denoiser",
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
        str(args.seed),
        "--num-workers",
        str(args.num_workers),
        "--device",
        args.device,
        "--z2-antisymmetrize-train",
    ]


def a1_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "scripts/compute_a1_sweep.py",
        "--kind",
        "denoiser",
        "--lengths",
        args.lengths,
        "--betas",
        args.betas,
        "--checkpoints-dir",
        str(args.train_dir / "checkpoints"),
        "--checkpoint-pattern",
        "denoiser_L{L}.pt",
        "--snapshots-dir",
        str(args.snapshots_dir),
        "--output-dir",
        str(args.a1_dir),
        "--num-records",
        str(args.num_records),
        "--device",
        args.device,
        "--z2-antisymmetrize",
    ]


def plot_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "scripts/plot_fig3_scaling.py",
        "--input",
        str(args.a1_dir / "a1_sweep.csv"),
        "--output",
        str(args.a1_dir / "fig3_scaling.png"),
        "--title",
        "Denoising diffusion finite-size scaling",
    ]


def main() -> None:
    args = parse_args()
    if not args.skip_train:
        run_command(train_command(args), args.dry_run)
    if not args.skip_a1:
        run_command(a1_command(args), args.dry_run)
    if not args.skip_plot:
        run_command(plot_command(args), args.dry_run)


if __name__ == "__main__":
    main()
