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
