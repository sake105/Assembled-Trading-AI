"""Global random state utilities for deterministic backtests and experiments.

This module centralizes seeding of Python's random module and NumPy's RNG.
It is intended to be used by backtests, ML workflows, and playbooks to ensure
that runs are reproducible when a seed is provided.

Note:
    - set_global_seed() modifies global process-wide RNG state.
    - seed_context() is a convenience wrapper that calls set_global_seed()
      at the beginning of a block; it does NOT restore previous state.
"""

from __future__ import annotations

import os
import random
from contextlib import contextmanager
from typing import Iterator

import numpy as np

import logging

logger = logging.getLogger(__name__)


def set_global_seed(seed: int) -> None:
    """Set global RNG state for Python and NumPy.

    Args:
        seed: Integer seed value to set.

    This function:
        - Sets PYTHONHASHSEED in the environment.
        - Seeds Python's random module.
        - Seeds NumPy's global RNG.

    It is safe to call multiple times; later calls simply overwrite the state.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    logger.info("Global random seed set to %d", seed)


@contextmanager
def seed_context(seed: int) -> Iterator[None]:
    """Context manager that sets a deterministic global seed for a block.

    Args:
        seed: Integer seed value to set.

    Note:
        This context manager intentionally does NOT restore the previous RNG
        state after the block. The goal is that, given the same seed and the
        same sequence of operations, the entire process remains reproducible.
    """
    set_global_seed(seed)
    yield
