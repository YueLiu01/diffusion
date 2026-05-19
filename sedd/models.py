from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class FourierLevelEmbedding(nn.Module):
    """Small Fourier embedding for scalar beta/tau inputs."""

    def __init__(self, dim: int = 16, max_frequency: float = 16.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("Embedding dimension must be even")
        frequencies = torch.logspace(0.0, math.log10(max_frequency), dim // 2)
        self.register_buffer("frequencies", frequencies)

    def forward(self, level: torch.Tensor) -> torch.Tensor:
        if level.ndim == 1:
            level = level[:, None]
        angles = level * self.frequencies[None, :] * math.pi
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class CircularConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        total_pad = self.dilation * (self.kernel_size - 1)
        left = total_pad // 2
        right = total_pad - left
        return self.conv(F.pad(x, (left, right), mode="circular"))


class ResidualDilatedBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.conv = CircularConv1d(channels, channels, kernel_size, dilation=dilation)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.conv(x))


class DilatedConvNet(nn.Module):
    """Translation-covariant periodic 1D network for spin chains.

    The network sees only the corrupted configuration x and a scalar noise level.
    It can be used as a SEDD log-ratio model or as a direct posterior-mean denoiser.
    """

    def __init__(
        self,
        length: int,
        hidden_channels: int = 64,
        kernel_size: int = 3,
        level_embedding_dim: int = 16,
        residual_blocks: int | None = None,
        output_activation: str = "none",
    ):
        super().__init__()
        if residual_blocks is None:
            residual_blocks = max(1, math.ceil(math.log2(max(2, length - 1))) - 1)
        self.length = length
        self.output_activation = output_activation
        self.level_embedding = FourierLevelEmbedding(level_embedding_dim)
        in_channels = 1 + level_embedding_dim
        self.stem = nn.Sequential(
            CircularConv1d(in_channels, hidden_channels, kernel_size, dilation=1),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(
            *[
                ResidualDilatedBlock(hidden_channels, kernel_size, dilation=2**i)
                for i in range(residual_blocks)
            ]
        )
        self.readout = nn.Conv1d(hidden_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, level: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected x with shape [B, L], got {tuple(x.shape)}")
        if x.shape[1] != self.length:
            raise ValueError(f"Expected chain length {self.length}, got {x.shape[1]}")
        if level.ndim == 1:
            level = level[:, None]
        emb = self.level_embedding(level.to(device=x.device, dtype=x.dtype))
        emb = emb[:, :, None].expand(-1, -1, self.length)
        h = torch.cat([x[:, None, :], emb], dim=1)
        y = self.readout(self.blocks(self.stem(h))).squeeze(1)
        if self.output_activation == "tanh":
            return torch.tanh(y)
        if self.output_activation == "none":
            return y
        raise ValueError(f"Unknown output activation {self.output_activation!r}")


class Z2SymmetrizedScore(nn.Module):
    """Enforce global spin-flip invariance for SEDD log-ratio scores."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, level: torch.Tensor) -> torch.Tensor:
        return 0.5 * (self.model(x, level) + self.model(-x, level))
