"""Tests for global random seed reproducibility (backlog item 13 + 75)."""

import random

import numpy as np

from src.assembled_core.utils.random_state import set_global_seed


def test_numpy_reproducible_after_same_seed():
    set_global_seed(42)
    first = np.random.rand(5).tolist()

    set_global_seed(42)
    second = np.random.rand(5).tolist()

    assert first == second, "NumPy random output must be identical after same seed"


def test_python_random_reproducible_after_same_seed():
    set_global_seed(99)
    first = [random.random() for _ in range(5)]

    set_global_seed(99)
    second = [random.random() for _ in range(5)]

    assert first == second, "Python random output must be identical after same seed"


def test_different_seeds_produce_different_output():
    set_global_seed(1)
    a = np.random.rand(5).tolist()

    set_global_seed(2)
    b = np.random.rand(5).tolist()

    assert a != b, "Different seeds must produce different NumPy random output"
