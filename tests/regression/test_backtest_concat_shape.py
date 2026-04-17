"""B3 — Pin backtest-engine concat topology.

The plan's B3 fix was to ensure ``pd.concat`` is only called **once** per
aggregation (terminal concat over a pre-accumulated list) rather than inside a
per-bar loop. The current ``backtest_engine.py`` already conforms; this test
is the regression pin so a future refactor can't silently re-introduce a hot
loop concat.

Heuristic: there should be exactly **one** ``pd.concat(all_<name>, ...)``
terminal call per aggregator (``all_orders``, ``all_signals_list``,
``all_targets``), and **no** ``pd.concat`` that mutates a list bucket inside
``for ... timestamp``/``for ... ts`` iteration.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from src.assembled_core.qa import backtest_engine as be

pytestmark = pytest.mark.phase_speed


def _source_tree() -> ast.Module:
    src = Path(inspect.getfile(be)).read_text(encoding="utf-8")
    return ast.parse(src)


def test_no_concat_mutates_aggregator_inside_for_loop() -> None:
    tree = _source_tree()

    class ConcatInLoopVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.offenders: list[tuple[int, str]] = []
            self._in_for_stack: list[ast.For] = []

        def visit_For(self, node: ast.For) -> None:  # noqa: N802
            self._in_for_stack.append(node)
            self.generic_visit(node)
            self._in_for_stack.pop()

        def visit_AsyncFor(self, node: ast.AsyncFor) -> None:  # noqa: N802
            self._in_for_stack.append(node)
            self.generic_visit(node)
            self._in_for_stack.pop()

        def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
            if not self._in_for_stack:
                self.generic_visit(node)
                return
            # Detect: all_X = pd.concat([all_X, ...]) or similar.
            if not (
                isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == "concat"
            ):
                self.generic_visit(node)
                return
            # Is the target re-assigning an aggregator named all_*?
            targets = node.targets
            for t in targets:
                if isinstance(t, ast.Name) and t.id.startswith("all_"):
                    self.offenders.append((node.lineno, ast.unparse(node)))
            self.generic_visit(node)

    visitor = ConcatInLoopVisitor()
    visitor.visit(tree)
    assert not visitor.offenders, (
        f"B3 regression: pd.concat found mutating an all_* aggregator inside a "
        f"for-loop. Convert to list-append + terminal concat:\n"
        + "\n".join(f"  line {ln}: {src}" for ln, src in visitor.offenders)
    )


def test_terminal_concat_pattern_is_preserved() -> None:
    """all_orders/all_signals_list/all_targets must still be gathered with
    ``.append`` and collapsed with a single ``pd.concat`` call each."""
    src = Path(inspect.getfile(be)).read_text(encoding="utf-8")
    for agg in ("all_orders", "all_signals_list", "all_targets"):
        append_hits = src.count(f"{agg}.append(")
        concat_hits = src.count(f"pd.concat({agg}")
        assert append_hits >= 1, f"{agg} is no longer accumulated via .append"
        assert concat_hits <= 1, (
            f"{agg} is collapsed by pd.concat {concat_hits} times — "
            "must be exactly one terminal concat"
        )
