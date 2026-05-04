"""A7: Corporate actions wired into backtest_engine + unified_paper_engine default flipped."""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.mark.fast
def test_unified_paper_engine_default_ca_true():
    """unified_paper_engine.PaperEngineConfig.enable_corporate_actions defaults to True."""
    from src.assembled_core.execution.unified_paper_engine import UnifiedPaperConfig

    cfg = UnifiedPaperConfig.__dataclass_fields__["enable_corporate_actions"]
    assert cfg.default is True, "enable_corporate_actions must default to True"


@pytest.mark.fast
def test_backtest_engine_has_corporate_actions_param():
    """run_portfolio_backtest must accept enable_corporate_actions parameter."""
    import inspect
    from src.assembled_core.qa.backtest_engine import run_portfolio_backtest

    sig = inspect.signature(run_portfolio_backtest)
    assert (
        "enable_corporate_actions" in sig.parameters
    ), "run_portfolio_backtest must have enable_corporate_actions parameter"


@pytest.mark.fast
def test_adjust_prices_for_splits_wired_in_source():
    """backtest_engine.py must reference adjust_prices_for_splits (A7 wiring check)."""
    import inspect
    import src.assembled_core.qa.backtest_engine as mod

    src_text = inspect.getsource(mod)
    assert (
        "adjust_prices_for_splits" in src_text
    ), "backtest_engine must wire adjust_prices_for_splits for A7 compliance"


@pytest.mark.fast
def test_split_adjustment_removes_price_drop(tmp_path):
    """With split data, adjust_prices_for_splits corrects the apparent price drop."""
    from src.assembled_core.data.corporate_actions import adjust_prices_for_splits

    # AAPL 2-for-1 split on 2024-02-01: price should halve, qty should double
    prices = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-31", "2024-02-01", "2024-02-02"]),
            "symbol": ["AAPL", "AAPL", "AAPL"],
            "close": [200.0, 100.0, 101.0],  # price drops 50% at split date
            "open": [198.0, 99.0, 100.0],
            "high": [202.0, 102.0, 103.0],
            "low": [197.0, 98.0, 99.0],
            "volume": [1e6, 2e6, 1.5e6],
        }
    )
    splits = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "action_type": ["SPLIT"],
            "effective_date": ["2024-02-01"],
            "split_ratio": [2.0],
        }
    )

    adjusted = adjust_prices_for_splits(prices, splits)
    # After adjustment, pre-split prices should be halved to match post-split level
    pre_split_close = adjusted.loc[
        adjusted["timestamp"] == pd.Timestamp("2024-01-31"), "close"
    ].iloc[0]
    post_split_close = adjusted.loc[
        adjusted["timestamp"] == pd.Timestamp("2024-02-01"), "close"
    ].iloc[0]
    # Pre-split adjusted price should be approximately equal to post-split price (~100)
    assert abs(pre_split_close - post_split_close) < 5.0, (
        f"After split adjustment, pre-split close ({pre_split_close}) should match "
        f"post-split close ({post_split_close})"
    )
