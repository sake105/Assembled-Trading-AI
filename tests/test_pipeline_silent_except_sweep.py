"""FU-1/FU-2 regression: sibling silent-except sweep for trading_cycle_shared.py.

Static source-text checks that D5 (intermarket), D6 (candlestick), D9 (earnings)
follow the same inner/outer try/except convention as D10 (congress) after the
A1b + bc290fb fix.
"""

from __future__ import annotations
from pathlib import Path

TRADING_CYCLE_SHARED = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "assembled_core"
    / "pipeline"
    / "trading_cycle_shared.py"
)


def _source() -> str:
    return TRADING_CYCLE_SHARED.read_text(encoding="utf-8")


def test_d5_intermarket_has_narrowed_except():
    src = _source()
    assert (
        "Intermarket features SILENTLY DISABLED" in src
        or "intermarket features SILENTLY DISABLED" in src.lower()
    ), "D5 should have a narrowed-except WARNING after FU-1 sweep"


def test_d6_candlestick_has_narrowed_except():
    src = _source()
    assert (
        "Candlestick features SILENTLY DISABLED" in src
        or "candlestick features SILENTLY DISABLED" in src.lower()
    ), "D6 should have a narrowed-except WARNING after FU-1 sweep"


def test_d9_earnings_has_narrowed_except():
    src = _source()
    assert (
        "Earnings features SILENTLY DISABLED" in src
        or "earnings features SILENTLY DISABLED" in src.lower()
    ), "D9 should have a narrowed-except WARNING after FU-1 sweep"


def test_outer_safety_net_present_for_all_four():
    """D5/D6/D9/D10 all should have an outer logger.error catch-all for data-op errors."""
    src = _source()
    # Count outer "feature load/merge failed" or equivalent ERROR messages
    error_messages = src.count("feature load/merge failed")
    assert error_messages >= 4, (
        f"Expected ≥4 outer-error handlers (D5+D6+D9+D10), found {error_messages}"
    )
