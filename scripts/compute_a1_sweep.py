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
from sedd.models import DilatedConvNet, Z2SymmetrizedScore
from sedd.observables import records_from_clean
from sedd.sampling import posterior_mean_from_sampler


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
    parser.add_argument(
        "--z2-symmetrize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For SEDD checkpoints, enforce global spin-flip invariance of the log-ratio score at inference.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_model(checkpoint_path: Path, kind: str, device: torch.device, z2_symmetrize: bool) -> tuple[torch.nn.Module, dict]:
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
    if kind == "sedd" and z2_symmetrize:
        model = Z2SymmetrizedScore(model).to(device)
        model.eval()
    return model, config


def write_rows(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "length",
        "beta",
        "tau",
        "a1",
        "a1_stderr",
        "a1_std",
        "kind",
        "checkpoint",
        "num_records",
        "num_posterior_samples",
        "steps",
        "sweeps_per_step",
        "model_level",
        "z2_symmetrize",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def denoiser_a1_stats(model: DilatedConvNet, s: torch.Tensor, level: torch.Tensor) -> tuple[float, float, float]:
    means = model(s, level)
    per_record = torch.mean(means.square(), dim=1)
    return summarize(per_record)


@torch.no_grad()
def sedd_a1_stats(
    model: DilatedConvNet,
    s: torch.Tensor,
    num_samples: int,
    beta: float,
    model_level: str,
    steps: int,
    sweeps_per_step: int,
) -> tuple[float, float, float]:
    mean1 = posterior_mean_from_sampler(
        model,
        s,
        num_samples,
        beta0=beta,
        steps=steps,
        sweeps_per_step=sweeps_per_step,
        model_level=model_level,
    )
    mean2 = posterior_mean_from_sampler(
        model,
        s,
        num_samples,
        beta0=beta,
        steps=steps,
        sweeps_per_step=sweeps_per_step,
        model_level=model_level,
    )
    per_record = torch.mean(mean1 * mean2, dim=1)
    return summarize(per_record)


def summarize(per_record: torch.Tensor) -> tuple[float, float, float]:
    values = per_record.detach().float().cpu()
    mean = float(values.mean())
    if values.numel() <= 1:
        return mean, 0.0, 0.0
    std = float(values.std(unbiased=True))
    stderr = std / float(values.numel() ** 0.5)
    return mean, stderr, std


def model_level_tensor(model_level: str, beta: float, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    tau = torch.tanh(torch.tensor(2.0 * beta, device=device, dtype=dtype))
    if model_level == "beta":
        value = torch.tensor(beta, device=device, dtype=dtype)
    elif model_level == "tau":
        value = tau
    elif model_level == "ell":
        value = -0.5 * torch.log(tau.clamp(max=1.0 - 1e-6))
    else:
        raise ValueError("model_level must be 'beta', 'tau', or 'ell'")
    return value.expand(batch_size, 1)


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

        model, config = load_model(checkpoint_path, args.kind, device, args.z2_symmetrize)
        model_level = str(config.get("level_kind", "ell"))
        z = load_ising_snapshots(snapshot_path, max_samples=args.num_records).to(device)

        for beta in betas:
            s = records_from_clean(z, beta=beta)
            if args.kind == "denoiser":
                level = model_level_tensor(model_level, beta, s.shape[0], device, s.dtype)
                a1, a1_stderr, a1_std = denoiser_a1_stats(model, s, level)
            else:
                a1, a1_stderr, a1_std = sedd_a1_stats(
                    model,
                    s,
                    args.num_posterior_samples,
                    beta,
                    model_level,
                    args.steps,
                    args.sweeps_per_step,
                )
            tau = float(torch.tanh(torch.tensor(2.0 * beta)).item())
            row: dict[str, float | int | str] = {
                "length": length,
                "beta": beta,
                "tau": tau,
                "a1": a1,
                "a1_stderr": a1_stderr,
                "a1_std": a1_std,
                "kind": args.kind,
                "checkpoint": str(checkpoint_path),
                "num_records": int(z.shape[0]),
                "num_posterior_samples": args.num_posterior_samples,
                "steps": args.steps,
                "sweeps_per_step": args.sweeps_per_step,
                "model_level": model_level,
                "z2_symmetrize": bool(args.z2_symmetrize and args.kind == "sedd"),
            }
            rows.append(row)
            print(
                f"L={length} beta={beta:g} tau={tau:.6f} "
                f"A1={a1:.8f} stderr={a1_stderr:.8f}",
                flush=True,
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.output_dir / "a1_sweep.csv", rows)
    with (args.output_dir / "a1_sweep.json").open("w") as handle:
        json.dump(rows, handle, indent=2)
    print(f"Saved A1 results to {args.output_dir / 'a1_sweep.csv'}")


if __name__ == "__main__":
    main()
