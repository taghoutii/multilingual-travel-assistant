"""Seed every random number generator in use (numpy and torch only if installed)."""

from __future__ import annotations

import importlib
import os
import random


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        np = importlib.import_module("numpy")
    except ImportError:
        pass
    else:
        np.random.seed(seed)
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
