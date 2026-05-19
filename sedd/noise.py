from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def beta_to_tau(beta: torch.Tensor) -> torch.Tensor:
    """Map weak-measurement strength beta to channel correlation tau."""
    return torch.tanh(2.0 * beta)


def tau_to_beta(tau: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    tau = tau.clamp(-1.0 + eps, 1.0 - eps)
    return 0.5 * torch.atanh(tau)


def ell_to_tau(ell: torch.Tensor) -> torch.Tensor:
    """Map diffusion time ell = -0.5 log(tau) to tau."""
    return torch.exp(-2.0 * ell)


def tau_to_ell(tau: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Map tau in (0, 1] to diffusion time ell = -0.5 log(tau)."""
    tau = tau.clamp(eps, 1.0 - eps)
    return -0.5 * torch.log(tau)


def _convert_tau_to_level(tau: torch.Tensor, output_kind: str) -> torch.Tensor:
    if output_kind == "tau":
        return tau
    if output_kind == "beta":
        return tau_to_beta(tau)
    raise ValueError("output_kind must be 'beta' or 'tau'")


@dataclass(frozen=True)
class UniformBetaSampler:
    beta_min: float
    beta_max: float

    def __call__(self, batch_size: int, device: torch.device | None = None) -> torch.Tensor:
        beta = torch.rand(batch_size, 1, device=device)
        return self.beta_min + (self.beta_max - self.beta_min) * beta


@dataclass(frozen=True)
class UniformTauSampler:
    tau_min: float
    tau_max: float

    def __call__(self, batch_size: int, device: torch.device | None = None) -> torch.Tensor:
        tau = torch.rand(batch_size, 1, device=device)
        return self.tau_min + (self.tau_max - self.tau_min) * tau


@dataclass(frozen=True)
class UniformEllSampler:
    """Uniformly sample diffusion time ell and return beta or tau for the model."""

    ell_min: float
    ell_max: float
    output_kind: str = "beta"

    def __post_init__(self) -> None:
        if self.ell_min < 0.0:
            raise ValueError("ell_min must be nonnegative because tau = exp(-2 ell)")
        if self.ell_max <= self.ell_min:
            raise ValueError("ell_max must be greater than ell_min")

    def __call__(self, batch_size: int, device: torch.device | None = None) -> torch.Tensor:
        ell = torch.rand(batch_size, 1, device=device)
        ell = self.ell_min + (self.ell_max - self.ell_min) * ell
        return _convert_tau_to_level(ell_to_tau(ell), self.output_kind)


def make_level_sampler(
    sample_kind: str,
    sample_min: float,
    sample_max: float,
    output_kind: str,
) -> UniformBetaSampler | UniformTauSampler | UniformEllSampler:
    """Build a sampler for training levels.

    output_kind controls what the model receives: beta or tau. sample_kind
    controls the distribution used to draw the underlying noise scale.
    """
    if sample_kind == "beta":
        if output_kind != "beta":
            return _ConvertedBetaSampler(sample_min, sample_max, output_kind)
        return UniformBetaSampler(sample_min, sample_max)
    if sample_kind == "tau":
        if output_kind != "tau":
            return _ConvertedTauSampler(sample_min, sample_max, output_kind)
        return UniformTauSampler(sample_min, sample_max)
    if sample_kind == "ell":
        return UniformEllSampler(sample_min, sample_max, output_kind=output_kind)
    raise ValueError("sample_kind must be 'beta', 'tau', or 'ell'")


@dataclass(frozen=True)
class _ConvertedBetaSampler:
    beta_min: float
    beta_max: float
    output_kind: str

    def __call__(self, batch_size: int, device: torch.device | None = None) -> torch.Tensor:
        beta = UniformBetaSampler(self.beta_min, self.beta_max)(batch_size, device)
        return _convert_tau_to_level(beta_to_tau(beta), self.output_kind)


@dataclass(frozen=True)
class _ConvertedTauSampler:
    tau_min: float
    tau_max: float
    output_kind: str

    def __call__(self, batch_size: int, device: torch.device | None = None) -> torch.Tensor:
        tau = UniformTauSampler(self.tau_min, self.tau_max)(batch_size, device)
        return _convert_tau_to_level(tau, self.output_kind)


def expand_level(level: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Broadcast a scalar noise level to match a spin batch."""
    if level.ndim == 0:
        level = level.view(1, 1)
    elif level.ndim == 1:
        level = level[:, None]
    return level.to(device=like.device, dtype=like.dtype)


def corrupt_spins(z: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """Sample x ~ q_tau(x | z) for z in {-1, +1}."""
    tau = expand_level(tau, z)
    p_same = (1.0 + tau) / 2.0
    same = torch.rand_like(z) < p_same
    return torch.where(same, z, -z)


def sedd_target_ratio(x: torch.Tensor, z: torch.Tensor, tau: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Known target q_tau(F_i x | z) / q_tau(x | z) for the binary channel."""
    tau = expand_level(tau, x)
    xz = x * z
    numerator = (1.0 - tau * xz).clamp_min(eps)
    denominator = (1.0 + tau * xz).clamp_min(eps)
    return numerator / denominator


def random_global_flip(*spins: torch.Tensor, p: float = 0.5) -> tuple[torch.Tensor, ...]:
    """Apply the same global Z2 flip to each tensor in a batch with probability p."""
    if not spins:
        return ()
    batch = spins[0].shape[0]
    device = spins[0].device
    signs = torch.where(
        torch.rand(batch, 1, device=device) < p,
        torch.full((batch, 1), -1.0, device=device),
        torch.ones(batch, 1, device=device),
    )
    return tuple(spin * signs.to(dtype=spin.dtype) for spin in spins)


def random_cyclic_shift(*spins: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Apply one random periodic shift per sample to each spin batch."""
    if not spins:
        return ()
    batch, length = spins[0].shape
    shifts = torch.randint(length, (batch,), device=spins[0].device)
    shifted = []
    for spin in spins:
        rows = [torch.roll(spin[i], int(shifts[i].item()), dims=0) for i in range(batch)]
        shifted.append(torch.stack(rows, dim=0))
    return tuple(shifted)


def geometric_tau_schedule(tau0: float, steps: int, tau_max: float = 0.999) -> torch.Tensor:
    """Monotone schedule from observed tau0 toward the near-clean endpoint."""
    if not 0.0 < tau0 <= tau_max < 1.0:
        raise ValueError("Require 0 < tau0 <= tau_max < 1")
    if steps < 2:
        raise ValueError("steps must be at least 2")
    log_start = math.log(tau0)
    log_end = math.log(tau_max)
    return torch.exp(torch.linspace(log_start, log_end, steps))
