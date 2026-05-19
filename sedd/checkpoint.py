from __future__ import annotations

import pathlib
import sys
import types
from pathlib import Path
from typing import Any

import torch
from torch import nn


def _install_pathlib_compat() -> None:
    """Allow loading checkpoints pickled with newer pathlib internals."""
    if "pathlib._local" in sys.modules:
        return
    module = types.ModuleType("pathlib._local")
    module.PosixPath = pathlib.PosixPath
    module.WindowsPath = pathlib.WindowsPath
    sys.modules["pathlib._local"] = module


def sanitize_config(value: Any) -> Any:
    """Convert checkpoint config values to JSON-like Python objects."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): sanitize_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_config(item) for item in value]
    return value


def load_checkpoint(path: str | Path, map_location: str | torch.device | None = None) -> dict[str, Any]:
    _install_pathlib_compat()
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def save_checkpoint(path: str | Path, model: nn.Module, optimizer: torch.optim.Optimizer, config: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": sanitize_config(config),
        },
        path,
    )
