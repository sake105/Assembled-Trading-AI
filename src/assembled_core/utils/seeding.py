"""Central seed management for reproducible runs.

Usage: call set_global_seed(42) at the top of any training/backtest script.
"""

from __future__ import annotations
import os
import random
import logging

log = logging.getLogger(__name__)


def set_global_seed(seed: int = 42) -> None:
    """Set seeds for random, numpy, torch (if available), and PYTHONHASHSEED."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    log.debug("[seeding] global seed set to %d", seed)
