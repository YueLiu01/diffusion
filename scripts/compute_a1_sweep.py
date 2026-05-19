#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sedd.checkpoint import load_checkpoint
from sedd.data import load_ising_snapshots
from sedd.models import DilatedConvNet
from sedd.observables import a1_from_denoiser, a1_from_sedd_sampler, records_from_clean


DEFAULT_BETAS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
DEFAULT_LENGTHS = (20, 40, 60, 80, 100)


def parse_float_list(value: str) -> list[float]:
    return [float(item) for item in value.replace(",", " ").split()]


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.replace(",", " ").split()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute A1(beta) for trained length-sweep checkpoints.")
    parser.add_argument("--lengths", default=",".join(str(length) for length in DEFAULT_LENGTHS))
    parser.add_argument("--betas", default=",".join(str(beta) for beta in DEFAULT_BETAS))
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("runs/length_sweep/checkpoints"))
    parser.add_argument("--checkpoint-pattern", default="sedd_L{L}.pt")
    parser.add_argument("--snapshots-dir", type=Path, default=Path("snapshots"))
    parser.add_argument("--snapshot-pattern", default="Ising_snapshotsL{L}.npy")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/a1_sweep"))
    parser.add_argument("--kind", choices=["sedd", "denoiser"], default="sedd")
    parser.add_argument("--num-records", type=int, default=32)
    parser.add_argument("--num-posterior-samples", type=int, default=2)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--sweeps-per-step", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_model(checkpoint_path: Path, kind: str, device: torch.device) -> tuple[DilatedConvNet, dict]:
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    length = int(config["length"])
    output_activation = "tanh" if kind == "denoiser" else "none"
    model = DilatedConvNet(
        length=length,
        hidden_channels=int(config.get("hidden_channels", 64)),
        output_activation=output_activation,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, config


def write_rows(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "length",
        "beta",
        "tau",
        "a1",
        "kind",
        "checkpoint",
        "num_records",
        "num_posterior_samples",
        "steps",
        "sweeps_per_step",
        "model_level",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    lengths = parse_int_list(args.lengths)
    betas = parse_float_list(args.betas)
    rows: list[dict[str, float | int | str]] = []

    for length in lengths:
        checkpoint_path = args.checkpoints_dir / args.checkpoint_pattern.format(L=length)
        snapshot_path = args.snapshots_dir / args.snapshot_pattern.format(L=length)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint for L={length}: {checkpoint_path}")
        if not snapshot_path.exists():
            raise FileNotFoundError(f"Missing snapshots for L={length}: {snapshot_path}")

        model, config = load_model(checkpoint_path, args.kind, device)
        model_level = str(config.get("level_kind", "ell"))
        z = load_ising_snapshots(snapshot_path, max_samples=args.num_records).to(device)

        for beta in betas:
            s = records_from_clean(z, beta=beta)
            if args.kind == "denoiser":
                a1 = a1_from_denoiser(model, s, beta=beta, model_level=model_level)
            else:
                a1 = a1_from_sedd_sampler(
                    model,
                    s,
                    num_samples=args.num_posterior_samples,
                    beta0=beta,
                    steps=args.steps,
                    sweeps_per_step=args.sweeps_per_step,
                    model_level=model_level,
                )
            tau = float(torch.tanh(torch.tensor(2.0 * beta)).item())
            row: dict[str, float | int | str] = {
                "length": length,
                "beta": beta,
                "tau": tau,
                "a1": a1,
                "kind": args.kind,
                "checkpoint": str(checkpoint_path),
                "num_records": int(z.shape[0]),
                "num_posterior_samples": args.num_posterior_samples,
                "steps": args.steps,
                "sweeps_per_step": args.sweeps_per_step,
                "model_level": model_level,
            }
            rows.append(row)
            print(f"L={length} beta={beta:g} tau={tau:.6f} A1={a1:.8f}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.output_dir / "a1_sweep.csv", rows)
    with (args.output_dir / "a1_sweep.json").open("w") as handle:
        json.dump(rows, handle, indent=2)
    print(f"Saved A1 results to {args.output_dir / 'a1_sweep.csv'}")


if __name__ == "__main__":
    main()
