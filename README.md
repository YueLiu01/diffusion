# Diffusion2 SEDD implementation

This repository implements the SEDD formulation in `notes.md` for diagonal weak-measurement protocols. Clean snapshots `z` are loaded from `snapshots/*.npy`, converted to spins in `{-1,+1}`, corrupted on the fly as `x ~ q_tau(x|z)`, and used to train models that take only `(x, beta)` or `(x, tau)` as input.

## Train

```bash
python scripts/train_sedd.py --snapshots snapshots/Ising_snapshotsL100.npy --epochs 20 --output checkpoints/sedd_L100.pt
python scripts/train_denoiser.py --snapshots snapshots/Ising_snapshotsL100.npy --epochs 20 --output checkpoints/denoiser_L100.pt
```

To train the default length sweep `L = 20, 40, 60, 80, 100`, save per-epoch training/validation losses, plot the curves, and write one checkpoint per length:

```bash
python scripts/train_lengths.py \
  --snapshots-dir snapshots \
  --output-dir runs/length_sweep \
  --epochs 20 \
  --level-kind beta \
  --sample-kind ell \
  --sample-min 0.01 \
  --sample-max 2.0
```

This writes metrics to `runs/length_sweep/metrics/`, checkpoints to `runs/length_sweep/checkpoints/`, and `runs/length_sweep/plots/loss_curves.png`.

The SEDD model outputs per-site log ratios `u_i(x, level) ~= log p_tau(F_i x) / p_tau(x)`. The direct denoiser baseline outputs posterior means `f_i(x, level) ~= E[z_i | x, level]`.

By default, `--level-kind beta` samples beta uniformly in `[--level-min, --level-max]`, and `--level-kind tau` samples tau uniformly. To sample the noise scale uniformly in diffusion time `ell = -0.5 * log(tau)`,

```bash
python scripts/train_sedd.py \
  --snapshots snapshots/Ising_snapshotsL100.npy \
  --level-kind beta \
  --sample-kind ell \
  --sample-min 0.01 \
  --sample-max 2.0
```

Equivalently, `ell = -0.5 * log(tanh(2 * beta))`. The model still receives beta because `--level-kind beta`; only the training distribution over noise levels changes. Use `sample-min > 0` to avoid the singular `tau = 1` endpoint.

Some reference values for this convention:

| `ell` | `tau = exp(-2 * ell)` | `beta = 0.5 * atanh(tau)` |
|---:|---:|---:|
| 0.001 | 0.998002 | 1.726939 |
| 0.01 | 0.980199 | 1.151301 |
| 0.05 | 0.904837 | 0.749141 |
| 0.1 | 0.818731 | 0.576478 |
| 0.25 | 0.606531 | 0.351707 |
| 0.5 | 0.367879 | 0.192984 |
| 1.0 | 0.135335 | 0.068085 |
| 1.5 | 0.049787 | 0.024914 |
| 2.0 | 0.018316 | 0.009159 |

## Estimate one-point nonlinear observable

```bash
python scripts/estimate_a1.py --kind denoiser --checkpoint checkpoints/denoiser_L100.pt --snapshots snapshots/Ising_snapshotsL100.npy --beta 0.5
python scripts/estimate_a1.py --kind sedd --checkpoint checkpoints/sedd_L100.pt --snapshots snapshots/Ising_snapshotsL100.npy --beta 0.5
```

The SEDD estimator uses two independent posterior sample batches to avoid directly squaring a noisy Monte Carlo posterior mean.

## Small-chain validation

For small systems, `sedd.validation` can compute posterior means and noisy-distribution log ratios exactly under the empirical snapshot prior:

```python
from sedd.validation import empirical_posterior_mean, empirical_noisy_log_ratios
```

## Implementation rule

The score network is always called as:

```python
u = model(x, beta)  # or model(x, tau)
```

The clean snapshot `z` appears only in the target ratio:

```python
a = (1 - tau * x * z) / (1 + tau * x * z)
loss = (exp(u) - a * u).mean()
```
