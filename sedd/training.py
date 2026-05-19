from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader

from .noise import beta_to_tau, corrupt_spins, random_cyclic_shift, random_global_flip, sedd_target_ratio


def sedd_loss(
    model: nn.Module,
    z: torch.Tensor,
    level: torch.Tensor,
    level_kind: str = "beta",
    eps: float = 1e-6,
    logu_clip: float | None = 20.0,
    augment_z2: bool = True,
    augment_shift: bool = True,
) -> torch.Tensor:
    """Compute the score-entropy loss for a clean snapshot minibatch."""
    if level_kind == "beta":
        tau = beta_to_tau(level)
        model_level = level
    elif level_kind == "tau":
        tau = level
        model_level = level
    else:
        raise ValueError("level_kind must be 'beta' or 'tau'")
    x = corrupt_spins(z, tau)
    if augment_z2:
        z, x = random_global_flip(z, x)
    if augment_shift:
        z, x = random_cyclic_shift(z, x)
    target = sedd_target_ratio(x, z, tau, eps=eps)
    log_ratio = model(x, model_level)
    if logu_clip is not None:
        log_ratio = log_ratio.clamp(-logu_clip, logu_clip)
    return (torch.exp(log_ratio) - target * log_ratio).mean()


def denoiser_loss(
    model: nn.Module,
    z: torch.Tensor,
    level: torch.Tensor,
    level_kind: str = "beta",
    augment_z2: bool = True,
    augment_shift: bool = True,
) -> torch.Tensor:
    """MSE loss for f_phi(x, level) ~= E[z | x, level]."""
    tau = beta_to_tau(level) if level_kind == "beta" else level
    x = corrupt_spins(z, tau)
    if augment_z2:
        z, x = random_global_flip(z, x)
    if augment_shift:
        z, x = random_cyclic_shift(z, x)
    pred = model(x, level)
    return torch.mean((pred - z) ** 2)


def objective_loss(
    model: nn.Module,
    z: torch.Tensor,
    level: torch.Tensor,
    objective: str = "sedd",
    level_kind: str = "beta",
    augment_z2: bool = True,
    augment_shift: bool = True,
) -> torch.Tensor:
    if objective == "sedd":
        return sedd_loss(
            model,
            z,
            level,
            level_kind=level_kind,
            augment_z2=augment_z2,
            augment_shift=augment_shift,
        )
    if objective == "denoiser":
        return denoiser_loss(
            model,
            z,
            level,
            level_kind=level_kind,
            augment_z2=augment_z2,
            augment_shift=augment_shift,
        )
    raise ValueError("objective must be 'sedd' or 'denoiser'")


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    level_sampler: Callable[[int, torch.device | None], torch.Tensor],
    device: torch.device,
    objective: str = "sedd",
    level_kind: str = "beta",
    max_batches: int | None = None,
) -> float:
    """Evaluate stochastic validation loss on held-out clean snapshots."""
    model.eval()
    total = 0.0
    count = 0
    for batch_index, z in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        z = z.to(device=device, dtype=torch.float32)
        level = level_sampler(z.shape[0], device)
        loss = objective_loss(
            model,
            z,
            level,
            objective=objective,
            level_kind=level_kind,
            augment_z2=False,
            augment_shift=False,
        )
        batch_size = int(z.shape[0])
        total += float(loss.cpu()) * batch_size
        count += batch_size
    if count == 0:
        raise ValueError("Validation loader produced no batches")
    return total / count


def train_epochs(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    level_sampler: Callable[[int, torch.device | None], torch.Tensor],
    epochs: int,
    device: torch.device,
    objective: str = "sedd",
    level_kind: str = "beta",
    grad_clip: float = 1.0,
    log_every: int = 50,
) -> list[float]:
    """Run a compact training loop and return per-step losses."""
    losses: list[float] = []
    model.to(device)
    model.train()
    step = 0
    for epoch in range(epochs):
        for z in loader:
            z = z.to(device=device, dtype=torch.float32)
            level = level_sampler(z.shape[0], device)
            loss = objective_loss(model, z, level, objective=objective, level_kind=level_kind)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            value = float(loss.detach().cpu())
            losses.append(value)
            if log_every > 0 and step % log_every == 0:
                print(f"epoch={epoch + 1} step={step} loss={value:.6f}", flush=True)
            step += 1
    return losses


def train_epochs_with_validation(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    level_sampler: Callable[[int, torch.device | None], torch.Tensor],
    epochs: int,
    device: torch.device,
    objective: str = "sedd",
    level_kind: str = "beta",
    grad_clip: float = 1.0,
    val_batches: int | None = None,
) -> list[dict[str, float | int]]:
    """Train and return one metric row per epoch."""
    history: list[dict[str, float | int]] = []
    model.to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        count = 0
        steps = 0
        for z in train_loader:
            z = z.to(device=device, dtype=torch.float32)
            level = level_sampler(z.shape[0], device)
            loss = objective_loss(model, z, level, objective=objective, level_kind=level_kind)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            batch_size = int(z.shape[0])
            total += float(loss.detach().cpu()) * batch_size
            count += batch_size
            steps += 1
        if count == 0:
            raise ValueError("Training loader produced no batches")
        train_loss = total / count
        val_loss = evaluate_loss(
            model,
            val_loader,
            level_sampler,
            device,
            objective=objective,
            level_kind=level_kind,
            max_batches=val_batches,
        )
        row: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_steps": steps,
        }
        history.append(row)
    return history


def save_checkpoint(path: str | Path, model: nn.Module, optimizer: torch.optim.Optimizer, config: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": config,
        },
        path,
    )
