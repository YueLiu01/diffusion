from __future__ import annotations

import torch
from torch import nn

from .noise import beta_to_tau, corrupt_spins, tau_to_beta, tau_to_ell
from .sampling import posterior_mean_from_sampler


@torch.no_grad()
def records_from_clean(z: torch.Tensor, beta: float | None = None, tau: float | None = None) -> torch.Tensor:
    if (beta is None) == (tau is None):
        raise ValueError("Provide exactly one of beta or tau")
    level = torch.full((z.shape[0], 1), float(beta if beta is not None else tau), device=z.device, dtype=z.dtype)
    tau_tensor = beta_to_tau(level) if beta is not None else level
    return corrupt_spins(z, tau_tensor)


@torch.no_grad()
def a1_from_denoiser(
    model: nn.Module,
    s: torch.Tensor,
    beta: float | None = None,
    tau: float | None = None,
    model_level: str = "ell",
) -> float:
    """Estimate L^{-1} sum_i E_s[m_i(s)^2] from a direct posterior-mean model."""
    if (beta is None) == (tau is None):
        raise ValueError("Provide exactly one of beta or tau")
    if model_level == "beta":
        if beta is None:
            beta_tensor = tau_to_beta(torch.tensor(float(tau)))
            level_value = float(beta_tensor.item())
        else:
            level_value = float(beta)
    elif model_level == "tau":
        if tau is None:
            tau_tensor = torch.tanh(torch.tensor(2.0 * float(beta)))
            level_value = float(tau_tensor.item())
        else:
            level_value = float(tau)
    elif model_level == "ell":
        if tau is None:
            tau_tensor = torch.tanh(torch.tensor(2.0 * float(beta)))
        else:
            tau_tensor = torch.tensor(float(tau))
        level_value = float(tau_to_ell(tau_tensor).item())
    else:
        raise ValueError("model_level must be 'beta', 'tau', or 'ell'")
    level = torch.full((s.shape[0], 1), level_value, device=s.device, dtype=s.dtype)
    means = model(s, level)
    return float(torch.mean(means**2).cpu())


@torch.no_grad()
def a1_from_sedd_sampler(
    model: nn.Module,
    s: torch.Tensor,
    num_samples: int,
    beta0: float | None = None,
    tau0: float | None = None,
    **sampler_kwargs,
) -> float:
    """Bias-reduced A1 estimate using two independent posterior sample batches."""
    mean1 = posterior_mean_from_sampler(model, s, num_samples, beta0=beta0, tau0=tau0, **sampler_kwargs)
    mean2 = posterior_mean_from_sampler(model, s, num_samples, beta0=beta0, tau0=tau0, **sampler_kwargs)
    return float(torch.mean(mean1 * mean2).cpu())


@torch.no_grad()
def disconnected_two_point(means: torch.Tensor, max_distance: int | None = None) -> torch.Tensor:
    """Return site-averaged E_j[m_j m_{j+r}] for r = 0..max_distance."""
    length = means.shape[1]
    if max_distance is None:
        max_distance = length // 2
    values = []
    for distance in range(max_distance + 1):
        values.append(torch.mean(means * torch.roll(means, shifts=-distance, dims=1)))
    return torch.stack(values)
