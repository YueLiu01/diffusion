#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Fig. 3-style finite-size scaling collapse for A1/A2 data.")
    parser.add_argument("--input", type=Path, default=Path("runs/a1_sweep/a1_sweep.csv"))
    parser.add_argument("--output", type=Path, default=Path("runs/a1_sweep/fig3_scaling.png"))
    parser.add_argument("--delta", type=float, default=1.0 / 8.0, help="Ising spin scaling dimension.")
    parser.add_argument("--beta-max", type=float, default=0.3, help="Keep beta <= beta-max, as in Fig. 3.")
    parser.add_argument("--include-all-beta", action="store_true", help="Disable the beta-max filter.")
    parser.add_argument("--title", default="Finite-size scaling collapse")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, float]]:
    rows = []
    with path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "length": float(row["length"]),
                    "beta": float(row["beta"]),
                    "a1": float(row["a1"]),
                    "a1_stderr": float(row.get("a1_stderr", 0.0) or 0.0),
                }
            )
    return rows


def scaled_values(rows: list[dict[str, float]], delta: float) -> list[dict[str, float]]:
    denom = 1.0 - 2.0 * delta
    if denom <= 0:
        raise ValueError("Require delta < 1/2 for the Fig. 3 scaling form")
    y_exp = 4.0 * delta / denom
    x_exp = 2.0 / denom
    scaled = []
    for row in rows:
        beta = row["beta"]
        length = row["length"]
        beta_y = beta**y_exp
        x = (length * beta**x_exp) ** 0.5
        scaled.append(
            {
                **row,
                "x": x,
                "y": row["a1"] / beta_y,
                "yerr": row["a1_stderr"] / beta_y,
            }
        )
    return scaled


def plot(rows: list[dict[str, float]], output: Path, title: str, delta: float) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    lengths = sorted({int(row["length"]) for row in rows})
    fig, ax = plt.subplots(figsize=(6.2, 4.8), constrained_layout=True)

    for length in lengths:
        subset = sorted([row for row in rows if int(row["length"]) == length], key=lambda item: item["x"])
        x = [row["x"] for row in subset]
        y = [row["y"] for row in subset]
        yerr = [row["yerr"] for row in subset]
        ax.errorbar(x, y, yerr=yerr, marker="o", linestyle="-", linewidth=1.3, capsize=2.5, label=f"L={length}")

    ax.set_xscale("log")
    ax.set_xlabel(r"$[L\beta^{2/(1-2\Delta)}]^{1/2}$")
    ax.set_ylabel(r"$A_1 / \beta^{4\Delta/(1-2\Delta)}$")
    ax.set_title(title + rf" ($\Delta={delta:g}$)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    if not args.include_all_beta:
        rows = [row for row in rows if row["beta"] <= args.beta_max]
    if not rows:
        print("No rows left after filtering; nothing to plot.", file=sys.stderr)
        raise SystemExit(1)
    scaled = scaled_values(rows, args.delta)
    plot(scaled, args.output, args.title, args.delta)
    print(f"Saved Fig. 3-style scaling plot to {args.output}")


if __name__ == "__main__":
    main()
