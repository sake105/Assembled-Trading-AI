"""B4 — Pin the fill simulator to the vectorised (no-iterrows) shape.

The plan's B4 fix replaced ``for _, order in orders.iterrows()`` in
``_simulate_fills`` and the cash-gate loop in ``_simulate_fills_with_cost``
with pre-extracted Python lists + integer indexing. ``iterrows()`` creates a
fresh pandas Series per row and is roughly 3-5× slower than the list form on
the per-order hot path; re-introducing it would silently regress every paper
bar and every backtest timestamp.

Heuristic: the two target methods must not call ``.iterrows()`` on their
inputs. Other ``iterrows()`` usages in the class (price-feed joins, lifecycle
bookkeeping, etc.) are out of scope for B4 and stay unchecked.
"""

from __future__ import annotations

import inspect

import pytest

from src.assembled_core.execution.unified_paper_engine import UnifiedPaperEngine

pytestmark = pytest.mark.phase_speed


def test_simulate_fills_has_no_iterrows() -> None:
    source = inspect.getsource(UnifiedPaperEngine._simulate_fills)
    assert ".iterrows(" not in source, (
        "B4 regression: UnifiedPaperEngine._simulate_fills calls .iterrows(). "
        "Use list/itertuples extraction + integer indexing instead."
    )


def test_simulate_fills_with_cost_has_no_iterrows_in_cash_gate() -> None:
    source = inspect.getsource(UnifiedPaperEngine._simulate_fills_with_cost)
    assert ".iterrows(" not in source, (
        "B4 regression: UnifiedPaperEngine._simulate_fills_with_cost cash-gate "
        "loop calls .iterrows(). Use pre-extracted lists + integer iteration."
    )


def test_hotloop_extracts_hot_columns_to_lists() -> None:
    source = inspect.getsource(UnifiedPaperEngine._simulate_fills)
    # Sentinel for the vectorised pattern introduced by B4.
    assert "_sym_arr" in source and "_qty_arr" in source and "_side_arr" in source, (
        "Expected B4 pre-extraction sentinels (_sym_arr/_qty_arr/_side_arr) "
        "missing — fill hot-loop may have been re-row-ified."
    )
