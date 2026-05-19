# Diffusion2 SEDD implementation

This repository implements the SEDD formulation in `notes.md` for diagonal weak-measurement protocols. Clean snapshots `z` are loaded from `snapshots/*.npy`, converted to spins in `{-1,+1}`, corrupted on the fly as `x ~ q_tau(x|z)`, and used to train models that take only `(x, beta)` or `(x, tau)` as input.

## Train

```bash
python scripts/train_sedd.py --snapshots snapshots/Ising_snapshotsL100.npy --epochs 20 --output checkpoints/sedd_L100.pt
python scripts/train_denoiser.py --snapshots snapshots/Ising_snapshotsL100.npy --epochs 20 --output checkpoints/denoiser_L100.pt
```

The SEDD model outputs per-site log ratios `u_i(x, level) ~= log p_tau(F_i x) / p_tau(x)`. The direct denoiser baseline outputs posterior means `f_i(x, level) ~= E[z_i | x, level]`.

By default, `--level-kind beta` samples beta uniformly in `[--level-min, --level-max]`, and `--level-kind tau` samples tau uniformly. To sample the noise scale uniformly in log-SNR,

```bash
python scripts/train_sedd.py \
  --snapshots snapshots/Ising_snapshotsL100.npy \
  --level-kind beta \
  --sample-kind ell \
  --sample-min -4 \
  --sample-max 4
```

Here `ell = log(tau^2 / (1 - tau^2))`. The model still receives beta because `--level-kind beta`; only the training distribution over noise levels changes.

Some reference values for this convention:

| `ell` | `tau = sqrt(sigmoid(ell))` | `beta = 0.5 * atanh(tau)` |
|---:|---:|---:|
| -6 | 0.049725 | 0.024883 |
| -4 | 0.134113 | 0.067463 |
| -2 | 0.345258 | 0.180025 |
| 0 | 0.707107 | 0.440687 |
| 2 | 0.938508 | 0.862691 |
| 4 | 0.990966 | 1.348847 |
| 6 | 0.998763 | 1.846883 |

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
