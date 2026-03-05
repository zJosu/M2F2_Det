"""
Reproducibility utilities.

Ensures deterministic behaviour across runs with the same seed.
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seed for full reproducibility.

    Sets seeds for Python, NumPy, and PyTorch (CPU + CUDA).
    Also enables deterministic cuDNN behaviour.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic behaviour (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(preference: str = "auto") -> torch.device:
    """Resolve compute device.

    Args:
        preference: One of "auto", "cuda", "cpu".

    Returns:
        torch.device instance.
    """
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)
