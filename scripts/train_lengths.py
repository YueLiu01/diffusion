#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sedd.data import SpinSnapshotDataset
from sedd.models import DilatedConvNet
from sedd.noise import make_level_sampler
from sedd.training import save_checkpoint, train_epochs_with_validation


DEFAULT_LENGTHS = (20, 40, 60, 80, 100)


def parse_lengths(value: str) -> list[int]:
    return [int(item) for item in value.replace(",", " ").split()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SEDD/denoiser models for several Ising chain lengths.")
    parser.add_argument("--lengths", default=",".join(str(length) for length in DEFAULT_LENGTHS))
    parser.add_argument("--snapshots-dir", type=Path, default=Path("snapshots"))
    parser.add_argument("--snapshot-pattern", default="Ising_snapshotsL{L}.npy")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/length_sweep"))
    parser.add_argument("--objective", choices=["sedd", "denoiser"], default="sedd")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=0.2)
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
    parser.add_argument("--val-batches", type=int, default=None, help="Optional cap on validation batches per epoch.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def split_dataset(dataset: SpinSnapshotDataset, val_fraction: float, seed: int) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("--val-fraction must be between 0 and 1")
    val_size = max(1, int(round(len(dataset) * val_fraction)))
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise ValueError("Not enough samples for a nonempty train/validation split")
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)


def write_history_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["length", "epoch", "train_loss", "val_loss", "train_steps", "checkpoint"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_histories(histories: dict[int, list[dict[str, float | int | str]]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for length, rows in histories.items():
        epochs = [int(row["epoch"]) for row in rows]
        train = [float(row["train_loss"]) for row in rows]
        val = [float(row["val_loss"]) for row in rows]
        axes[0].plot(epochs, train, marker="o", linewidth=1.5, label=f"L={length}")
        axes[1].plot(epochs, val, marker="o", linewidth=1.5, label=f"L={length}")
    axes[0].set_title("Training loss")
    axes[1].set_title("Validation loss")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    lengths = parse_lengths(args.lengths)
    sample_kind = args.sample_kind or args.level_kind
    sample_min = args.level_min if args.sample_min is None else args.sample_min
    sample_max = args.level_max if args.sample_max is None else args.sample_max

    metrics_dir = args.output_dir / "metrics"
    checkpoint_dir = args.output_dir / "checkpoints"
    plot_dir = args.output_dir / "plots"
    all_rows: list[dict[str, float | int | str]] = []
    histories: dict[int, list[dict[str, float | int | str]]] = {}

    for length in lengths:
        snapshot_path = args.snapshots_dir / args.snapshot_pattern.format(L=length)
        if not snapshot_path.exists():
            raise FileNotFoundError(f"Missing snapshot file for L={length}: {snapshot_path}")

        dataset = SpinSnapshotDataset(snapshot_path, max_samples=args.max_samples)
        train_set, val_set = split_dataset(dataset, args.val_fraction, seed=args.seed + length)
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=args.num_workers,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers,
        )

        output_activation = "none" if args.objective == "sedd" else "tanh"
        model = DilatedConvNet(length=length, hidden_channels=args.hidden_channels, output_activation=output_activation)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        sampler = make_level_sampler(
            sample_kind=sample_kind,
            sample_min=sample_min,
            sample_max=sample_max,
            output_kind=args.level_kind,
        )

        print(
            f"Training L={length}: train={len(train_set)} val={len(val_set)} "
            f"objective={args.objective} sample={sample_kind}[{sample_min}, {sample_max}]",
            flush=True,
        )
        history = train_epochs_with_validation(
            model,
            train_loader,
            val_loader,
            optimizer,
            sampler,
            epochs=args.epochs,
            device=device,
            objective=args.objective,
            level_kind=args.level_kind,
            val_batches=args.val_batches,
        )

        checkpoint_path = checkpoint_dir / f"{args.objective}_L{length}.pt"
        config = vars(args) | {
            "length": length,
            "sample_kind": sample_kind,
            "sample_min": sample_min,
            "sample_max": sample_max,
            "train_samples": len(train_set),
            "val_samples": len(val_set),
        }
        save_checkpoint(checkpoint_path, model, optimizer, config)

        length_rows: list[dict[str, float | int | str]] = []
        for row in history:
            metric_row = dict(row)
            metric_row["length"] = length
            metric_row["checkpoint"] = str(checkpoint_path)
            length_rows.append(metric_row)
            print(
                f"L={length} epoch={row['epoch']} "
                f"train_loss={float(row['train_loss']):.6f} val_loss={float(row['val_loss']):.6f}",
                flush=True,
            )

        histories[length] = length_rows
        all_rows.extend(length_rows)
        write_history_csv(metrics_dir / f"losses_L{length}.csv", length_rows)

    write_history_csv(metrics_dir / "losses_all.csv", all_rows)
    with (metrics_dir / "losses_all.json").open("w") as handle:
        json.dump(all_rows, handle, indent=2)
    plot_histories(histories, plot_dir / "loss_curves.png")
    print(f"Saved metrics to {metrics_dir}")
    print(f"Saved checkpoints to {checkpoint_dir}")
    print(f"Saved plot to {plot_dir / 'loss_curves.png'}")


if __name__ == "__main__":
    main()
