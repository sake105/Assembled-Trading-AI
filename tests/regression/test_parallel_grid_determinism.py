"""C3 — Serial-vs-parallel 1e-9 determinism pin.

Same ``master_seed`` + same ``grid`` must produce the same results whether
we run with ``n_jobs=1`` or ``n_jobs=-1`` (loky pool). loky is a process
backend, so the worker's RNG has no shared in-memory state — determinism
is purely driven by the derived seed.

Also pins the seed-derivation formula: ``master_seed * (index + 1)``. A
change there is a regression against every prior parallel run.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src.assembled_core.qa.parallel_grid import (
    GridPoint,
    _derive_seed,
    run_grid_parallel,
)

pytestmark = pytest.mark.phase_speed


def _work(params: dict[str, Any], seed: int) -> dict[str, float]:
    """Tiny pure-NumPy workload exercising rngs + the param payload.

    Picklable (top-level function) so loky can ship it to workers.
    """
    rng = np.random.default_rng(seed)
    mu = float(params.get("mu", 0.0))
    sigma = float(params.get("sigma", 1.0))
    draws = rng.normal(loc=mu, scale=sigma, size=1024)
    return {
        "seed": seed,
        "mean": float(draws.mean()),
        "std": float(draws.std()),
    }


def _extract(results: list[GridPoint]) -> list[dict[str, float]]:
    return [{"idx": g.index, "seed": g.seed, **g.result} for g in results]


def test_derive_seed_formula_stable() -> None:
    assert _derive_seed(42, 0) == 42
    assert _derive_seed(42, 1) == 84
    assert _derive_seed(42, 9) == 420
    # Zero master seed → all zeros; caller responsibility
    assert _derive_seed(0, 7) == 0


def test_serial_vs_parallel_bit_identical() -> None:
    grid = [{"mu": 0.1 * i, "sigma": 1.0} for i in range(6)]

    serial = _extract(run_grid_parallel(grid, _work, master_seed=42, n_jobs=1))
    parallel = _extract(run_grid_parallel(grid, _work, master_seed=42, n_jobs=-1))

    assert len(serial) == len(parallel) == 6
    for s, p in zip(serial, parallel, strict=True):
        assert s["idx"] == p["idx"]
        assert s["seed"] == p["seed"]
        # Pure NumPy RNG over a fixed seed → bit-identical floats.
        assert s["mean"] == p["mean"], f"mean diverged on idx {s['idx']}"
        assert s["std"] == p["std"], f"std diverged on idx {s['idx']}"


def test_order_preserved_across_backends() -> None:
    grid = [{"mu": float(i)} for i in range(4)]
    out = run_grid_parallel(grid, _work, master_seed=7, n_jobs=-1)
    assert [g.index for g in out] == [0, 1, 2, 3]
    assert [g.params["mu"] for g in out] == [0.0, 1.0, 2.0, 3.0]


def _broken_worker(params: dict[str, Any], seed: int) -> dict[str, float]:
    if params.get("fail", False):
        raise RuntimeError("injected grid failure")
    return _work(params, seed)


def test_error_point_does_not_abort_sweep() -> None:
    grid = [
        {"mu": 0.0, "sigma": 1.0, "fail": False},
        {"mu": 1.0, "sigma": 1.0, "fail": True},
        {"mu": 2.0, "sigma": 1.0, "fail": False},
    ]
    out = run_grid_parallel(grid, _broken_worker, master_seed=11, n_jobs=1)
    statuses = [g.status for g in out]
    assert statuses == ["ok", "error", "ok"]
    assert out[1].error is not None and "injected grid failure" in out[1].error
    # Good neighbours still produced real results.
    assert out[0].result is not None
    assert out[2].result is not None


def test_empty_grid_returns_empty() -> None:
    assert run_grid_parallel([], _work, master_seed=1, n_jobs=-1) == []
