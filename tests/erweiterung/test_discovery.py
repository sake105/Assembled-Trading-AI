"""Tests for erweiterung.discovery."""

from __future__ import annotations

import random

import numpy as np
import pandas as pd

from erweiterung.discovery import genetic_programming as gp


def test_random_tree_creation():
    rng = random.Random(42)
    t = gp.random_tree(depth=2, features=["a", "b"], rng=rng)
    s = t.to_str()
    assert isinstance(s, str)


def test_evaluate_tree():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    leaf_a = gp.GPNode(op="leaf", leaf_name="a")
    leaf_b = gp.GPNode(op="leaf", leaf_name="b")
    tree = gp.GPNode(op="+", children=[leaf_a, leaf_b])
    out = gp.evaluate_tree(tree, df)
    assert (out == [5, 7, 9]).all()


def test_fitness_ic():
    # fitness_ic requires >= 30 obs (drop guard); use 50 for a clean test
    n = 50
    s = pd.Series(np.linspace(1, n, n))
    r = pd.Series(np.linspace(n, 1, n))  # perfect negative correlation
    ic = gp.fitness_ic(s, r)
    # We use abs() in fitness, so should be near 1
    assert ic > 0.9


def test_gp_search_runs():
    rng_np = np.random.default_rng(0)
    n = 100
    df = pd.DataFrame(
        {
            "a": rng_np.normal(0, 1, n),
            "b": rng_np.normal(0, 1, n),
            "c": rng_np.normal(0, 1, n),
        }
    )
    target = pd.Series(df["a"] + 0.5 * df["b"] + rng_np.normal(0, 0.1, n))
    results = gp.gp_search(
        df, target, n_generations=5, population_size=20, tree_depth=2
    )
    assert len(results) > 0
    best_tree, best_fit = results[0]
    assert best_fit >= 0
