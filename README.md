# Diffusion2 SEDD implementation

This repository implements the SEDD formulation in `notes.md` for diagonal weak-measurement protocols. Clean snapshots `z` are loaded from `snapshots/*.npy`, converted to spins in `{-1,+1}`, corrupted on the fly as `x ~ q_tau(x|z)`, and used to train models that take only `(x, ell)` as input by default.

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
  --level-kind ell \
  --sample-kind ell \
  --sample-min 0.01 \
  --sample-max 2.0
```

This writes metrics to `runs/length_sweep/metrics/`, checkpoints to `runs/length_sweep/checkpoints/`, and `runs/length_sweep/plots/loss_curves.png`.

The SEDD model outputs per-site log ratios `u_i(x, level) ~= log p_tau(F_i x) / p_tau(x)`. The direct denoiser baseline outputs posterior means `f_i(x, level) ~= E[z_i | x, level]`.
For SEDD training, `--z2-symmetrize-train` is on by default and trains the even score `0.5 * (u(x, level) + u(-x, level))`. Use `--no-z2-symmetrize-train` only for diagnostics.
For denoising-diffusion training, use `--objective denoiser`; `--z2-antisymmetrize-train` is on by default and trains the odd posterior mean `0.5 * (f(x, level) - f(-x, level))`.

By default, `--level-kind ell` samples diffusion time uniformly in `[--level-min, --level-max]` and feeds `ell` directly to the network. To be explicit:

```bash
python scripts/train_sedd.py \
  --snapshots snapshots/Ising_snapshotsL100.npy \
  --level-kind ell \
  --sample-kind ell \
  --sample-min 0.01 \
  --sample-max 2.0
```

Equivalently, `ell = -0.5 * log(tau) = -0.5 * log(tanh(2 * beta))`. With `--level-kind ell`, the model call is `model(x, ell)`. Use `sample-min > 0` to avoid the singular `tau = 1` endpoint.

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

To compute `A1(beta)` for the length-sweep checkpoints and the default beta grid `0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6`:

```bash
python scripts/compute_a1_sweep.py \
  --checkpoints-dir runs/length_sweep/checkpoints \
  --snapshots-dir snapshots \
  --output-dir runs/a1_sweep
```

This writes `runs/a1_sweep/a1_sweep.csv` and `runs/a1_sweep/a1_sweep.json`.
Increase `--num-records`, `--num-posterior-samples`, `--steps`, and `--sweeps-per-step` for less noisy final estimates.

To make a Fig. 3-style finite-size scaling plot from an A1 sweep:

```bash
python scripts/plot_fig3_scaling.py \
  --input runs/a1_sweep/a1_sweep.csv \
  --output runs/a1_sweep/fig3_scaling.png
```

By default this uses the Ising value `Delta=1/8` and keeps `beta <= 0.3`.

## Denoising diffusion workflow

To bypass SEDD ratio sampling and train the noise-conditioned posterior-mean model directly:

```bash
python scripts/train_denoising_diffusion_sweep.py --device cuda
```

This trains `denoiser_L{L}.pt` checkpoints, computes A1 directly from `f_theta(s, beta)^2`, and writes:

```text
runs/denoising_diffusion_length_sweep/metrics/losses_all.csv
runs/denoising_diffusion_length_sweep/plots/loss_curves.png
runs/denoising_diffusion_a1_sweep/a1_sweep.csv
runs/denoising_diffusion_a1_sweep/fig3_scaling.png
```

This path is closer to image-diffusion `x0` prediction: the model is trained to denoise corrupted snapshots at random noise levels, but A1 evaluation does not use reverse MCMC.

## Caveats and checks

Endpoint cutoff:
Do not use `--level-min 0` with `--level-kind ell`. Since `tau = exp(-2 ell)`, `ell = 0` gives the deterministic clean endpoint `tau = 1`, where the SEDD target sits on a singular/full-support boundary. Use a small positive cutoff such as `0.001` or `0.01`.

Coordinate roles:
Training uses `ell` as the neural-network input by default, but corruption, target construction, and posterior sampling are still defined by the physical binary-channel parameter `tau`. Converting between schedule coordinates is fine as long as the network is called with the level it was trained on, e.g. `model(x, ell)`.

Reverse schedule:
Different schedules in `tau`, `ell`, or other monotone coordinates can change sampler accuracy at fixed compute. The current sampler stores a `tau` schedule and converts to `ell` before calling an `ell`-conditioned model. Check convergence by comparing `--steps 32,64,96,128` on representative beta and length values.

Posterior samples:
`--num-posterior-samples M` means `M` reverse samples for each of two independent posterior-mean batches, so the SEDD `A1` estimator uses `2M` generated posterior samples per measurement record. `M=1` is only for smoke tests and is too noisy for `A1`. Use `M=8` for a first real sweep and increase to `16` or more for final selected points if runtime allows.

Error bars:
`compute_a1_sweep.py` reports `a1_stderr`, the standard error over measurement records, and `a1_std`, the sample standard deviation over records. These error bars do not fully capture systematic sampler bias from too few reverse steps, too few sweeps, or imperfect model ratios.

Z2 score symmetry:
For the spin-flip-symmetric Ising data, the SEDD log-ratio score should obey `u(x, ell) = u(-x, ell)`. This symmetry is important at small beta: small symmetry-breaking errors in the learned score can produce a spuriously large `A1` when the true value should be close to zero. The CNN is trained with spin-flip augmentation, but augmentation does not enforce exact symmetry. Therefore the SEDD A1 scripts enforce Z2 symmetry at inference by default:

```python
u_sym = 0.5 * (model(x, ell) + model(-x, ell))
```

This inference-time symmetrization means existing checkpoints can be reused; retraining is not required just to fix the small-beta A1 issue. Disable it only for diagnostics with `--no-z2-symmetrize`.
New SEDD training also enables train-time Z2 symmetrization by default through `--z2-symmetrize-train`, so future checkpoints should satisfy the same even-score constraint during optimization as well as during A1 inference.

To retrain and compare the suspicious `L=60` case across seeds:

```bash
python scripts/retrain_l60_z2_compare.py --device cuda
```

This trains seeds `0,1,2` into `runs/length_sweep_L60_z2_seeds/`, computes A1 into `runs/a1_L60_z2_seed_compare/`, and writes `a1_compare.csv` plus `a1_seed_summary.csv`.

Runtime:
The current SEDD posterior sampler loops over posterior samples, reverse steps, sweeps, and lattice sites. Runtime scales roughly linearly with `num_records`, `num_posterior_samples`, `steps`, `sweeps_per_step`, number of beta values, and length. Batching posterior samples is a future optimization.

Checkpoint loading:
Older checkpoints may contain pickled `Path` objects in their config. Use `sedd.checkpoint.load_checkpoint` instead of raw `torch.load`; it handles the compatibility issue with newer PyTorch versions where `weights_only=True` is the default.

## Small-chain validation

For small systems, `sedd.validation` can compute posterior means and noisy-distribution log ratios exactly under the empirical snapshot prior:

```python
from sedd.validation import empirical_posterior_mean, empirical_noisy_log_ratios
```

## Implementation rule

With the default training settings, the score network is called as:

```python
u = model(x, ell)
```

The clean snapshot `z` appears only in the target ratio:

```python
a = (1 - tau * x * z) / (1 + tau * x * z)
loss = (exp(u) - a * u).mean()
```
