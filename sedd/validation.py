from __future__ import annotations

import torch
from torch import nn

from .noise import beta_to_tau, corrupt_spins, expand_level, tau_to_ell


def empirical_prior(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return unique clean states and empirical probabilities."""
    states, counts = torch.unique(z, dim=0, return_counts=True)
    probs = counts.to(dtype=z.dtype, device=z.device) / counts.sum()
    return states, probs


def _log_channel_prob(x: torch.Tensor, z_states: torch.Tensor, tau: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    tau = expand_level(tau, x)
    factors = (1.0 + tau[:, None, :] * x[:, None, :] * z_states[None, :, :]) / 2.0
    return torch.log(factors.clamp_min(eps)).sum(dim=-1)


@torch.no_grad()
def empirical_posterior_mean(
    clean_snapshots: torch.Tensor,
    x: torch.Tensor,
    beta: float | None = None,
    tau: float | None = None,
) -> torch.Tensor:
    """Exact E_empirical[z | x, tau] under the empirical clean-snapshot prior."""
    if (beta is None) == (tau is None):
        raise ValueError("Provide exactly one of beta or tau")
    states, prior = empirical_prior(clean_snapshots)
    tau_value = torch.tanh(torch.tensor(2.0 * float(beta), device=x.device, dtype=x.dtype)) if beta is not None else torch.tensor(float(tau), device=x.device, dtype=x.dtype)
    tau_batch = tau_value.expand(x.shape[0], 1)
    logits = torch.log(prior.clamp_min(1e-12))[None, :] + _log_channel_prob(x, states, tau_batch)
    weights = torch.softmax(logits, dim=1)
    return weights @ states


@torch.no_grad()
def empirical_noisy_log_ratios(
    clean_snapshots: torch.Tensor,
    x: torch.Tensor,
    beta: float | None = None,
    tau: float | None = None,
) -> torch.Tensor:
    """Exact log p_tau(F_i x) / p_tau(x) under the empirical clean-snapshot prior."""
    if (beta is None) == (tau is None):
        raise ValueError("Provide exactly one of beta or tau")
    states, prior = empirical_prior(clean_snapshots)
    tau_value = torch.tanh(torch.tensor(2.0 * float(beta), device=x.device, dtype=x.dtype)) if beta is not None else torch.tensor(float(tau), device=x.device, dtype=x.dtype)
    tau_batch = tau_value.expand(x.shape[0], 1)
    base_logits = torch.log(prior.clamp_min(1e-12))[None, :] + _log_channel_prob(x, states, tau_batch)
    base_logp = torch.logsumexp(base_logits, dim=1)
    ratios = []
    for site in range(x.shape[1]):
        flipped = x.clone()
        flipped[:, site] *= -1.0
        flip_logits = torch.log(prior.clamp_min(1e-12))[None, :] + _log_channel_prob(flipped, states, tau_batch)
        ratios.append(torch.logsumexp(flip_logits, dim=1) - base_logp)
    return torch.stack(ratios, dim=1)


@torch.no_grad()
def one_point_calibration(
    model: nn.Module,
    clean_snapshots: torch.Tensor,
    beta: float,
    model_level: str = "ell",
    max_samples: int | None = None,
) -> tuple[float, float]:
    """Check E_s[m_i(s)^2] = E_{z,s}[z_i m_i(s)] for a posterior-mean model."""
    z = clean_snapshots[:max_samples] if max_samples is not None else clean_snapshots
    level = torch.full((z.shape[0], 1), beta, device=z.device, dtype=z.dtype)
    tau = beta_to_tau(level)
    s = corrupt_spins(z, tau)
    if model_level == "beta":
        model_input_level = level
    elif model_level == "tau":
        model_input_level = tau
    elif model_level == "ell":
        model_input_level = tau_to_ell(tau)
    else:
        raise ValueError("model_level must be 'beta', 'tau', or 'ell'")
    means = model(s, model_input_level)
    left = torch.mean(means**2)
    right = torch.mean(z * means)
    return float(left.cpu()), float(right.cpu())
