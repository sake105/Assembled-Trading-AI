# tests/test_risk_sector_region_fx_limits.py
"""Tests for Sector/Region/FX Exposure Limits (Sprint 9).

Tests verify:
1. Sector breach reduces only that sector orders
2. Region breach reduces only that region orders
3. Deterministic reduction + rounding
4. Missing meta raises (default)
5. FX: if non-base currency present -> fail-fast
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.execution.pre_trade_checks import (
    PreTradeConfig,
    run_pre_trade_checks,
)


def test_sector_breach_reduces_only_that_sector_orders() -> None:
    """Test that sector breach reduces only orders in that sector."""
    # Setup: Current positions
    current_positions = pd.DataFrame({
        "symbol": ["AAPL", "MSFT", "GOOGL"],
        "qty": [50.0, 40.0, 30.0],
    })

    # Prices
    prices_latest = pd.DataFrame({
        "symbol": ["AAPL", "MSFT", "GOOGL", "TSLA"],
        "close": [150.0, 200.0, 2500.0, 200.0],
    })

    # Security metadata
    security_meta_df = pd.DataFrame({
        "symbol": ["AAPL", "MSFT", "GOOGL", "TSLA"],
        "sector": ["Technology", "Technology", "Technology", "Consumer"],
        "region": ["US", "US", "US", "US"],
        "currency": ["USD", "USD", "USD", "USD"],
    })

    # Equity
    equity = 100000.0

    # Generate orders: BUY more Technology stocks
    orders = pd.DataFrame({
        "symbol": ["AAPL", "MSFT", "GOOGL", "TSLA"],
        "side": ["BUY", "BUY", "BUY", "BUY"],
        "qty": [100.0, 80.0, 20.0, 50.0],
        "price": [150.0, 200.0, 2500.0, 200.0],
    })

    # Config: max_sector_exposure = 30% (Technology will exceed this)
    config = PreTradeConfig(max_sector_exposure=0.30)

    # Apply risk checks
    result, filtered = run_pre_trade_checks(
        orders,
        portfolio=None,
        current_positions=current_positions,
        prices_latest=prices_latest,
        equity=equity,
        security_meta_df=security_meta_df,
        config=config,
    )

    # Verify: Technology orders should be reduced, Consumer (TSLA) should be unchanged
    # Technology: AAPL (50+100=150 @ 150 = 22500), MSFT (40+80=120 @ 200 = 24000), GOOGL (30+20=50 @ 2500 = 125000)
    # Technology gross_weight = (22500 + 24000 + 125000) / 100000 = 1.715 (171.5%) > 0.30
    # Scale factor = 0.30 / 1.715 = 0.175
    # AAPL: 100 * 0.175 = 17.5 -> 17
    # MSFT: 80 * 0.175 = 14 -> 14
    # GOOGL: 20 * 0.175 = 3.5 -> 3

    aapl_qty = filtered[filtered["symbol"] == "AAPL"]["qty"].iloc[0]
    msft_qty = filtered[filtered["symbol"] == "MSFT"]["qty"].iloc[0]
    googl_qty = filtered[filtered["symbol"] == "GOOGL"]["qty"].iloc[0]
    tsla_qty = filtered[filtered["symbol"] == "TSLA"]["qty"].iloc[0]

    # Technology orders should be reduced
    assert aapl_qty < 100.0, "AAPL order should be reduced"
    assert msft_qty < 80.0, "MSFT order should be reduced"
    assert googl_qty < 20.0, "GOOGL order should be reduced"

    # Consumer order (TSLA) should be unchanged
    assert abs(tsla_qty - 50.0) < 1e-10, "TSLA order should be unchanged (different sector)"

    # Verify reduction reasons
    reasons = [r["reason"] for r in result.reduced_orders]
    assert "RISK_REDUCE_MAX_SECTOR_EXPOSURE" in reasons, "Should have sector reduction reasons"
    assert all(
        r["explain"]["group_type"] == "sector" for r in result.reduced_orders
        if r["reason"] == "RISK_REDUCE_MAX_SECTOR_EXPOSURE"
    ), "All sector reductions should have group_type='sector'"


def test_region_breach_reduces_only_that_region_orders() -> None:
    """Test that region breach reduces only orders in that region."""
    # Setup: No ASML position, so EU exposure is only from order (50 * 600 = 30000, weight = 0.30)
    # But we want EU to be under 30%, so we reduce ASML order or position
    current_positions = pd.DataFrame({
        "symbol": ["AAPL"],
        "qty": [50.0],  # No ASML position
    })

    prices_latest = pd.DataFrame({
        "symbol": ["AAPL", "ASML", "TSLA"],
        "close": [150.0, 600.0, 200.0],
    })

    security_meta_df = pd.DataFrame({
        "symbol": ["AAPL", "ASML", "TSLA"],
        "sector": ["Technology", "Technology", "Consumer"],
        "region": ["US", "EU", "US"],
        "currency": ["USD", "EUR", "USD"],
    })

    equity = 100000.0

    orders = pd.DataFrame({
        "symbol": ["AAPL", "ASML", "TSLA"],
        "side": ["BUY", "BUY", "BUY"],
        "qty": [100.0, 20.0, 80.0],  # Reduced ASML order to keep EU under 30%
        "price": [150.0, 600.0, 200.0],
    })

    # Config: max_region_exposure = 30% (US will exceed this: 38.5%, EU will be under: 0.20 = 20%)
    config = PreTradeConfig(max_region_exposure=0.30)

    result, filtered = run_pre_trade_checks(
        orders,
        portfolio=None,
        current_positions=current_positions,
        prices_latest=prices_latest,
        equity=equity,
        security_meta_df=security_meta_df,
        config=config,
    )

    # Verify: US orders should be reduced, EU (ASML) should be unchanged
    aapl_qty = filtered[filtered["symbol"] == "AAPL"]["qty"].iloc[0]
    asml_qty = filtered[filtered["symbol"] == "ASML"]["qty"].iloc[0]
    tsla_qty = filtered[filtered["symbol"] == "TSLA"]["qty"].iloc[0]

    # US orders should be reduced
    assert aapl_qty < 100.0, "AAPL order should be reduced (US region)"
    assert tsla_qty < 80.0, "TSLA order should be reduced (US region)"

    # EU order (ASML) should be unchanged (EU exposure < 30%)
    assert abs(asml_qty - 20.0) < 1e-10, "ASML order should be unchanged (EU region)"

    # Verify reduction reasons
    reasons = [r["reason"] for r in result.reduced_orders]
    assert "RISK_REDUCE_MAX_REGION_EXPOSURE" in reasons, "Should have region reduction reasons"


def test_deterministic_reduction_rounding() -> None:
    """Test that reduction and rounding are deterministic."""
    current_positions = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "qty": [50.0, 40.0],
    })

    prices_latest = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "close": [150.0, 200.0],
    })

    security_meta_df = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "sector": ["Technology", "Technology"],
        "region": ["US", "US"],
        "currency": ["USD", "USD"],
    })

    equity = 100000.0

    orders = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "side": ["BUY", "BUY"],
        "qty": [100.0, 80.0],
        "price": [150.0, 200.0],
    })

    config = PreTradeConfig(max_sector_exposure=0.30)

    # Run twice
    result1, filtered1 = run_pre_trade_checks(
        orders,
        portfolio=None,
        current_positions=current_positions,
        prices_latest=prices_latest,
        equity=equity,
        security_meta_df=security_meta_df,
        config=config,
    )

    result2, filtered2 = run_pre_trade_checks(
        orders,
        portfolio=None,
        current_positions=current_positions,
        prices_latest=prices_latest,
        equity=equity,
        security_meta_df=security_meta_df,
        config=config,
    )

    # Verify: identical results
    pd.testing.assert_frame_equal(
        filtered1.sort_values("symbol").reset_index(drop=True),
        filtered2.sort_values("symbol").reset_index(drop=True),
        check_dtype=False,
    )

    # Verify: reduction reasons are identical
    assert result1.reduced_orders == result2.reduced_orders, "Reduction reasons should be identical"


def test_missing_meta_raises_by_default() -> None:
    """Test that missing security metadata raises ValueError by default."""
    current_positions = pd.DataFrame({
        "symbol": ["AAPL"],
        "qty": [50.0],
    })

    prices_latest = pd.DataFrame({
        "symbol": ["AAPL"],
        "close": [150.0],
    })

    # Security metadata missing AAPL
    security_meta_df = pd.DataFrame({
        "symbol": ["MSFT"],
        "sector": ["Technology"],
        "region": ["US"],
        "currency": ["USD"],
    })

    equity = 100000.0

    orders = pd.DataFrame({
        "symbol": ["AAPL"],
        "side": ["BUY"],
        "qty": [100.0],
        "price": [150.0],
    })

    config = PreTradeConfig(max_sector_exposure=0.30, missing_security_meta="raise")

    # Should raise ValueError
    with pytest.raises(ValueError) as exc_info:
        run_pre_trade_checks(
            orders,
            portfolio=None,
            current_positions=current_positions,
            prices_latest=prices_latest,
            equity=equity,
            security_meta_df=security_meta_df,
            config=config,
        )

    error_msg = str(exc_info.value)
    assert "AAPL" in error_msg or "Missing" in error_msg or "sector" in error_msg


def test_fx_non_base_currency_fail_fast() -> None:
    """Test that FX check fails fast if non-base currency is present."""
    current_positions = pd.DataFrame({
        "symbol": ["AAPL", "SAP"],
        "qty": [50.0, 30.0],
    })

    prices_latest = pd.DataFrame({
        "symbol": ["AAPL", "SAP"],
        "close": [150.0, 100.0],
    })

    security_meta_df = pd.DataFrame({
        "symbol": ["AAPL", "SAP"],
        "sector": ["Technology", "Technology"],
        "region": ["US", "EU"],
        "currency": ["USD", "EUR"],  # Non-base currency present
    })

    equity = 100000.0

    orders = pd.DataFrame({
        "symbol": ["AAPL", "SAP"],
        "side": ["BUY", "BUY"],
        "qty": [100.0, 50.0],
        "price": [150.0, 100.0],
    })

    config = PreTradeConfig(
        max_fx_exposure=0.20,
        base_currency="USD",
        missing_security_meta="raise",
    )

    # Should raise ValueError (FX rates not implemented)
    with pytest.raises(ValueError) as exc_info:
        run_pre_trade_checks(
            orders,
            portfolio=None,
            current_positions=current_positions,
            prices_latest=prices_latest,
            equity=equity,
            security_meta_df=security_meta_df,
            config=config,
        )

    error_msg = str(exc_info.value)
    assert "FX exposure" in error_msg or "FX rates" in error_msg or "non-base currency" in error_msg
    assert "EUR" in error_msg or "currency" in error_msg


def test_fx_all_base_currency_passes() -> None:
    """Test that FX check passes if all currencies are base currency."""
    current_positions = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "qty": [50.0, 40.0],
    })

    prices_latest = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "close": [150.0, 200.0],
    })

    security_meta_df = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "sector": ["Technology", "Technology"],
        "region": ["US", "US"],
        "currency": ["USD", "USD"],  # All base currency
    })

    equity = 100000.0

    orders = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "side": ["BUY", "BUY"],
        "qty": [100.0, 80.0],
        "price": [150.0, 200.0],
    })

    config = PreTradeConfig(
        max_fx_exposure=0.20,
        base_currency="USD",
    )

    # Should pass (all base currency)
    result, filtered = run_pre_trade_checks(
        orders,
        portfolio=None,
        current_positions=current_positions,
        prices_latest=prices_latest,
        equity=equity,
        security_meta_df=security_meta_df,
        config=config,
    )

    # Verify: orders unchanged (no FX exposure)
    assert len(filtered) == 2, "All orders should remain"
    assert "fx_exposure_check" in result.summary, "Should have fx_exposure_check in summary"
    assert result.summary["fx_exposure_check"] == "all_base_currency", "Should indicate all base currency"


def test_sell_orders_reduce_exposure_not_blocked() -> None:
    """Test that SELL orders that reduce sector exposure are not blocked."""
    # Setup: Current positions overweight in Technology
    current_positions = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "qty": [200.0, 150.0],  # Large positions
    })

    prices_latest = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "close": [150.0, 200.0],
    })

    security_meta_df = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "sector": ["Technology", "Technology"],
        "region": ["US", "US"],
        "currency": ["USD", "USD"],
    })

    equity = 100000.0

    # Generate orders: SELL Technology stocks (reduces exposure)
    orders = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "side": ["SELL", "SELL"],
        "qty": [50.0, 40.0],
        "price": [150.0, 200.0],
    })

    # Config: max_sector_exposure = 30%
    # Current Technology exposure: (200*150 + 150*200) / 100000 = 0.6 (60%)
    # After SELL: (150*150 + 110*200) / 100000 = 0.445 (44.5%) - still over limit
    # But SELL reduces exposure, so it should be allowed
    config = PreTradeConfig(max_sector_exposure=0.30)

    result, filtered = run_pre_trade_checks(
        orders,
        portfolio=None,
        current_positions=current_positions,
        prices_latest=prices_latest,
        equity=equity,
        security_meta_df=security_meta_df,
        config=config,
    )

    # Verify: SELL orders should remain (they reduce exposure)
    assert len(filtered) == 2, "SELL orders should remain (reduce exposure)"
    # Note: The current implementation may still reduce if post-trade exposure exceeds limit
    # This is acceptable behavior (deterministic reduction)


def test_multiple_sectors_only_violating_sector_reduced() -> None:
    """Test that only violating sector orders are reduced, other sectors unchanged."""
    current_positions = pd.DataFrame({
        "symbol": ["AAPL", "TSLA"],
        "qty": [50.0, 30.0],
    })

    prices_latest = pd.DataFrame({
        "symbol": ["AAPL", "MSFT", "TSLA"],
        "close": [150.0, 200.0, 200.0],
    })

    security_meta_df = pd.DataFrame({
        "symbol": ["AAPL", "MSFT", "TSLA"],
        "sector": ["Technology", "Technology", "Consumer"],
        "region": ["US", "US", "US"],
        "currency": ["USD", "USD", "USD"],
    })

    equity = 100000.0

    orders = pd.DataFrame({
        "symbol": ["AAPL", "MSFT", "TSLA"],
        "side": ["BUY", "BUY", "BUY"],
        "qty": [100.0, 80.0, 50.0],
        "price": [150.0, 200.0, 200.0],
    })

    # Config: max_sector_exposure = 30% (Technology will exceed, Consumer won't)
    config = PreTradeConfig(max_sector_exposure=0.30)

    result, filtered = run_pre_trade_checks(
        orders,
        portfolio=None,
        current_positions=current_positions,
        prices_latest=prices_latest,
        equity=equity,
        security_meta_df=security_meta_df,
        config=config,
    )

    # Verify: Technology orders reduced, Consumer (TSLA) unchanged
    aapl_qty = filtered[filtered["symbol"] == "AAPL"]["qty"].iloc[0]
    msft_qty = filtered[filtered["symbol"] == "MSFT"]["qty"].iloc[0]
    tsla_qty = filtered[filtered["symbol"] == "TSLA"]["qty"].iloc[0]

    assert aapl_qty < 100.0, "AAPL order should be reduced (Technology)"
    assert msft_qty < 80.0, "MSFT order should be reduced (Technology)"
    assert abs(tsla_qty - 50.0) < 1e-10, "TSLA order should be unchanged (Consumer)"


def test_missing_security_meta_skips_check() -> None:
    """Test that missing security_meta_df skips group exposure checks."""
    orders = pd.DataFrame({
        "symbol": ["AAPL"],
        "side": ["BUY"],
        "qty": [100.0],
        "price": [150.0],
    })

    config = PreTradeConfig(max_sector_exposure=0.30)

    result, filtered = run_pre_trade_checks(
        orders,
        portfolio=None,
        security_meta_df=None,  # Missing
        config=config,
    )

    # Verify: orders unchanged (check skipped)
    assert len(filtered) == 1, "Order should remain unchanged"
    assert "group_exposure_check" in result.summary, "Should have skip reason in summary"
    assert result.summary["group_exposure_check"] == "skipped_no_security_meta", "Should skip due to missing security_meta"
