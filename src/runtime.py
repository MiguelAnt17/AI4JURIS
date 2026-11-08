"""Seeding and device helpers."""

import os
import random

import numpy as np
import torch

SEED = 42


def set_all_seeds(seed: int = SEED) -> None:
    """Seed all RNGs and enable best-effort determinism."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def get_device() -> torch.device:
    """Pick the device (mps > cuda > cpu)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
