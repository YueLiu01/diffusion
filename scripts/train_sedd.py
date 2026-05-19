#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sedd.data import SpinSnapshotDataset
from sedd.models import DilatedConvNet
from sedd.noise import make_level_sampler
from sedd.training import save_checkpoint, train_epochs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a SEDD log-ratio model on clean Ising snapshots.")
    parser.add_argument("--snapshots", required=True, type=Path)
    parser.add_argument("--output", default=Path("checkpoints/sedd.pt"), type=Path)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--level-kind", choices=["beta", "tau", "ell"], default="ell")
    parser.add_argument("--level-min", type=float, default=0.05)
    parser.add_argument("--level-max", type=float, default=1.0)
    parser.add_argument(
        "--sample-kind",
        choices=["beta", "tau", "ell"],
        default=None,
        help="Noise coordinate to sample uniformly. Defaults to --level-kind.",
    )
    parser.add_argument("--sample-min", type=float, default=None, help="Lower bound in --sample-kind units.")
    parser.add_argument("--sample-max", type=float, default=None, help="Upper bound in --sample-kind units.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = SpinSnapshotDataset(args.snapshots, max_samples=args.max_samples)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    length = dataset.snapshots.shape[1]
    model = DilatedConvNet(length=length, hidden_channels=args.hidden_channels, output_activation="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    sample_kind = args.sample_kind or args.level_kind
    sample_min = args.level_min if args.sample_min is None else args.sample_min
    sample_max = args.level_max if args.sample_max is None else args.sample_max
    sampler = make_level_sampler(
        sample_kind=sample_kind,
        sample_min=sample_min,
        sample_max=sample_max,
        output_kind=args.level_kind,
    )
    train_epochs(
        model,
        loader,
        optimizer,
        sampler,
        epochs=args.epochs,
        device=torch.device(args.device),
        objective="sedd",
        level_kind=args.level_kind,
    )
    save_checkpoint(
        args.output,
        model,
        optimizer,
        vars(args) | {"length": length, "sample_kind": sample_kind, "sample_min": sample_min, "sample_max": sample_max},
    )
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
