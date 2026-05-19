#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sedd.checkpoint import load_checkpoint
from sedd.data import load_ising_snapshots
from sedd.models import DilatedConvNet
from sedd.observables import a1_from_denoiser, a1_from_sedd_sampler, records_from_clean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate A1 = L^{-1} sum_i E_s[m_i(s)^2].")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--snapshots", required=True, type=Path)
    parser.add_argument("--kind", choices=["denoiser", "sedd"], required=True)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--num-records", type=int, default=512)
    parser.add_argument("--num-posterior-samples", type=int, default=16)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--sweeps-per-step", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    config = checkpoint["config"]
    length = int(config["length"])
    output_activation = "tanh" if args.kind == "denoiser" else "none"
    model = DilatedConvNet(
        length=length,
        hidden_channels=int(config.get("hidden_channels", 64)),
        output_activation=output_activation,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    z = load_ising_snapshots(args.snapshots, max_samples=args.num_records).to(device)
    s = records_from_clean(z, beta=args.beta)
    if args.kind == "denoiser":
        value = a1_from_denoiser(model, s, beta=args.beta, model_level=config.get("level_kind", "ell"))
    else:
        value = a1_from_sedd_sampler(
            model,
            s,
            num_samples=args.num_posterior_samples,
            beta0=args.beta,
            steps=args.steps,
            sweeps_per_step=args.sweeps_per_step,
            model_level=config.get("level_kind", "ell"),
        )
    print(f"A1(beta={args.beta}) = {value:.8f}")


if __name__ == "__main__":
    main()
