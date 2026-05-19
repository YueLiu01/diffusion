from __future__ import annotations

import torch
from torch import nn

from .noise import geometric_tau_schedule, sedd_target_ratio, tau_to_beta, tau_to_ell


@torch.no_grad()
def annealed_posterior_sample(
    model: nn.Module,
    s: torch.Tensor,
    beta0: float | None = None,
    tau0: float | None = None,
    steps: int = 32,
    sweeps_per_step: int = 2,
    tau_max: float = 0.999,
    model_level: str = "ell",
    log_ratio_clip: float = 20.0,
) -> torch.Tensor:
    """Sample approximate z ~ p(z | s, tau0) by annealed single-site Metropolis.

    At intermediate tau >= tau0 the target bridge is proportional to
    p_tau(x) q_{tau0 / tau}(s | x). The learned SEDD ratio estimates the first
    factor, and the known channel ratio supplies the conditioning factor.
    """
    if (beta0 is None) == (tau0 is None):
        raise ValueError("Provide exactly one of beta0 or tau0")
    if tau0 is None:
        tau0 = float(torch.tanh(torch.tensor(2.0 * beta0)).item())
    if s.ndim != 2:
        raise ValueError("Expected s with shape [B, L]")
    device = s.device
    x = s.clone()
    schedule = geometric_tau_schedule(tau0, steps=steps, tau_max=tau_max).to(device=device, dtype=s.dtype)
    model.eval()
    batch, length = x.shape
    batch_index = torch.arange(batch, device=device)

    for tau in schedule[1:]:
        tau_batch = tau.expand(batch, 1)
        alpha = (tau0 / float(tau.item()))
        alpha_batch = torch.full((batch, 1), alpha, device=device, dtype=s.dtype)
        for _ in range(sweeps_per_step):
            for site in torch.randperm(length, device=device):
                level = tau_batch
                if model_level == "beta":
                    level = tau_to_beta(tau_batch)
                elif model_level == "ell":
                    level = tau_to_ell(tau_batch)
                elif model_level != "tau":
                    raise ValueError("model_level must be 'beta', 'tau', or 'ell'")
                log_ratio = model(x, level).clamp(-log_ratio_clip, log_ratio_clip)
                learned = torch.exp(log_ratio[batch_index, site])
                channel = sedd_target_ratio(x, s, alpha_batch)[batch_index, site]
                accept_prob = torch.minimum(torch.ones_like(learned), learned * channel)
                accept = torch.rand(batch, device=device) < accept_prob
                x[accept, site] *= -1.0
    return x


@torch.no_grad()
def posterior_mean_from_sampler(
    model: nn.Module,
    s: torch.Tensor,
    num_samples: int,
    beta0: float | None = None,
    tau0: float | None = None,
    **sampler_kwargs,
) -> torch.Tensor:
    samples = []
    for _ in range(num_samples):
        samples.append(annealed_posterior_sample(model, s, beta0=beta0, tau0=tau0, **sampler_kwargs))
    return torch.stack(samples, dim=0).mean(dim=0)
