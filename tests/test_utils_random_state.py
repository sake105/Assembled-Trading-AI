"""Tests for global random state utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.mark.advanced
def test_set_global_seed_deterministic_sequences():
    """set_global_seed should make random sequences reproducible."""
    from src.assembled_core.utils import set_global_seed
    import random

    set_global_seed(123)
    seq1_py = [random.random() for _ in range(5)]
    seq1_np = np.random.rand(5)

    set_global_seed(123)
    seq2_py = [random.random() for _ in range(5)]
    seq2_np = np.random.rand(5)

    assert seq1_py == seq2_py
    assert np.allclose(seq1_np, seq2_np)


@pytest.mark.advanced
def test_seed_context_reproducible_blocks():
    """seed_context should produce identical sequences for same seed."""
    from src.assembled_core.utils import seed_context
    import random

    with seed_context(7):
        seq1_py = [random.random() for _ in range(3)]
        seq1_np = np.random.rand(3)

    with seed_context(7):
        seq2_py = [random.random() for _ in range(3)]
        seq2_np = np.random.rand(3)

    assert seq1_py == seq2_py
    assert np.allclose(seq1_np, seq2_np)
