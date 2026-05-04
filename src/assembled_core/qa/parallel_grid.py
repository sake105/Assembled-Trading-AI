"""C3 — Parallel parameter-grid runner for QA / research.

Runs a user-supplied ``run_one(params, seed)`` callable across a parameter
grid in parallel. Deliberately parallelises the **grid axis**, not timeline
bars (serial dependency) or symbols (cross-symbol state like correlation
guard, gross-cap). Each point gets a deterministic derived seed so
``seriell == parallel`` holds bit-identical with the same ``master_seed``.

Backend: ``joblib.Parallel`` with ``loky``. loky is process-based, so it
avoids the GIL and survives worker crashes. Workers are started fresh for
each call — no shared in-process state can leak.

Seed derivation::

    worker_seed = master_seed * (grid_index + 1)

This is trivial but stable. Any change to this formula is a regression for
determinism tests relying on it.

Usage::

    results = run_grid_parallel(
        grid=[{"ema_fast": f, "ema_slow": s} for f in (5, 10) for s in (20, 50)],
        run_one=my_backtest_point,
        master_seed=42,
        n_jobs=-1,
    )
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Iterable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GridPoint:
    """One evaluation point in the grid run."""

    index: int
    params: dict[str, Any]
    seed: int
    result: Any
    status: str  # "ok" | "error"
    error: str | None = None


def _derive_seed(master_seed: int, index: int) -> int:
    """Deterministic per-worker seed. Index is 0-based; we use (index+1) so
    point 0 doesn't collapse to 0 when master_seed is non-zero."""
    return int(master_seed) * (int(index) + 1)


def _default_n_jobs(n_jobs: int | None) -> int:
    if n_jobs is None or n_jobs == 0:
        return 1
    if n_jobs == -1:
        return max(1, (os.cpu_count() or 1) - 1)
    return max(1, int(n_jobs))


def _run_point(
    index: int,
    params: dict[str, Any],
    master_seed: int,
    run_one: Callable[[dict[str, Any], int], Any],
) -> GridPoint:
    seed = _derive_seed(master_seed, index)
    try:
        value = run_one(params, seed)
        return GridPoint(
            index=index,
            params=dict(params),
            seed=seed,
            result=value,
            status="ok",
        )
    except Exception as exc:  # noqa: BLE001 — grid-wide resilience is required
        logger.warning("[grid] point %d failed: %s", index, exc, exc_info=True)
        return GridPoint(
            index=index,
            params=dict(params),
            seed=seed,
            result=None,
            status="error",
            error=str(exc),
        )


def run_grid_parallel(
    grid: Iterable[dict[str, Any]],
    run_one: Callable[[dict[str, Any], int], Any],
    *,
    master_seed: int = 42,
    n_jobs: int | None = -1,
    backend: str = "loky",
) -> list[GridPoint]:
    """Run ``run_one(params, seed)`` over ``grid`` in parallel.

    Args:
        grid: Iterable of parameter dicts. Materialised to a list.
        run_one: Callable invoked per point. Must be picklable for loky backend.
            Called as ``run_one(params, seed)``.
        master_seed: Master seed used to derive per-point seeds.
        n_jobs: Worker count. ``-1`` means ``cpu_count()-1``. ``1`` means
            serial (executes in the current process). Missing / 0 → 1.
        backend: joblib backend. Default ``loky`` (process-based, GIL-free).
            For tests that need in-process execution and shared state,
            ``sequential`` is also honored.

    Returns:
        List of ``GridPoint`` in the *original* grid order. Errors are kept
        as points with ``status="error"`` rather than raised, so a bad point
        can't abort the whole sweep.
    """
    points = list(grid)
    if not points:
        return []

    effective_jobs = _default_n_jobs(n_jobs)

    # Fast path: serial — avoids joblib fork overhead for tiny grids
    if effective_jobs == 1 or backend == "sequential":
        return [_run_point(i, p, master_seed, run_one) for i, p in enumerate(points)]

    from joblib import Parallel, delayed  # imported lazily — heavy optional dep

    results = Parallel(n_jobs=effective_jobs, backend=backend)(
        delayed(_run_point)(i, p, master_seed, run_one) for i, p in enumerate(points)
    )
    # Joblib preserves order, but sort defensively on index for belt+braces.
    results_sorted: list[GridPoint] = sorted(results, key=lambda g: g.index)
    return results_sorted
