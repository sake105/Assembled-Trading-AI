"""Tests for erweiterung.execution.

almgren_chriss removed in cleanup — mainline has it under
src/assembled_core/execution/almgren_chriss.py (361 LoC, production-grade).
Only adaptive_slippage retained as a lightweight model.
"""

from __future__ import annotations

import numpy as np

from erweiterung.execution import adaptive_slippage


def test_slippage_increases_with_size():
    s_small = adaptive_slippage.slippage_bps(100, avg_daily_volume=1_000_000)
    s_large = adaptive_slippage.slippage_bps(100_000, avg_daily_volume=1_000_000)
    assert s_large > s_small


def test_slippage_zero_adv_returns_default():
    out = adaptive_slippage.slippage_bps(100, avg_daily_volume=0)
    assert out > 0
    assert np.isfinite(out)


def test_execution_price_buy_higher():
    p_buy = adaptive_slippage.execution_price(100, side=+1, slippage_bps_value=10)
    p_sell = adaptive_slippage.execution_price(100, side=-1, slippage_bps_value=10)
    assert p_buy > 100
    assert p_sell < 100
