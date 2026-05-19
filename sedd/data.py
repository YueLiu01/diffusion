from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def spins_to_pm_one(array: np.ndarray) -> np.ndarray:
    """Return spins encoded as {-1, +1} from either {0, 1} or {-1, +1} data."""
    values = np.unique(array)
    if np.all(np.isin(values, [-1, 1])):
        return array.astype(np.float32, copy=False)
    if np.all(np.isin(values, [0, 1])):
        return (2 * array - 1).astype(np.float32, copy=False)
    raise ValueError(f"Expected binary spins in {{0,1}} or {{-1,+1}}, found {values[:8]}")


def load_ising_snapshots(path: str | Path, max_samples: int | None = None) -> torch.Tensor:
    """Load an Ising snapshot file and convert it to a float tensor in {-1, +1}."""
    array = np.load(Path(path))
    if max_samples is not None:
        array = array[:max_samples]
    return torch.from_numpy(spins_to_pm_one(array))


class SpinSnapshotDataset(Dataset[torch.Tensor]):
    """Dataset of clean projective snapshots z sampled from p0(z)."""

    def __init__(self, path: str | Path, max_samples: int | None = None):
        self.snapshots = load_ising_snapshots(path, max_samples=max_samples)

    def __len__(self) -> int:
        return int(self.snapshots.shape[0])

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.snapshots[index]
