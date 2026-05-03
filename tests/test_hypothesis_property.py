"""
D3: Hypothesis property tests for qa/, risk/, and portfolio/ modules.

Invariants tested:
- Sharpe ratio sign matches mean-return sign (when std > 0)
- Max drawdown is always non-positive
- CAGR is positive when end > start (and enough periods)
- Vol-scaled weights are non-negative for long signals
- Kelly weights are non-negative for positive expected returns
- Gross exposure is non-negative for long-only positions
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# qa/metrics — Sharpe ratio
# ---------------------------------------------------------------------------

@given(
    returns=st.lists(
        st.floats(min_value=-0.3, max_value=0.3, allow_nan=False, allow_infinity=False),
        min_size=30,
        max_size=250,
    )
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_sharpe_sign_matches_mean_return(returns: list[float]) -> None:
    """Sharpe ratio sign equals sign of mean return when std > 0 (no risk-free rate)."""
    from src.assembled_core.qa.metrics import compute_sharpe_ratio

    arr = np.array(returns, dtype=float)
    std = float(arr.std())
    assume(std > 1e-6)

    returns_series = pd.Series(arr)
    sharpe = compute_sharpe_ratio(returns_series, freq="1d", risk_free_rate=0.0)
    assume(sharpe is not None)

    mean_r = float(arr.mean())
    if mean_r > 1e-8:
        assert sharpe >= 0, f"Positive mean return should give non-negative Sharpe; got {sharpe}"
    elif mean_r < -1e-8:
        assert sharpe <= 0, f"Negative mean return should give non-positive Sharpe; got {sharpe}"


# ---------------------------------------------------------------------------
# qa/metrics — drawdown
# ---------------------------------------------------------------------------

@given(
    values=st.lists(
        st.floats(min_value=0.1, max_value=200.0, allow_nan=False, allow_infinity=False),
        min_size=10,
        max_size=200,
    )
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_drawdown_is_non_positive(values: list[float]) -> None:
    """Max drawdown (absolute) is always <= 0 by definition."""
    from src.assembled_core.qa.metrics import compute_drawdown

    dates = pd.date_range("2020-01-01", periods=len(values), freq="D")
    equity = pd.Series(values, index=dates, dtype=float)
    _, max_dd, _, _ = compute_drawdown(equity)

    assert max_dd <= 1e-9, f"Max drawdown must be <= 0; got {max_dd}"


# ---------------------------------------------------------------------------
# qa/metrics — CAGR
# ---------------------------------------------------------------------------

@given(
    start=st.floats(min_value=1000.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
    growth=st.floats(min_value=0.01, max_value=5.0, allow_nan=False, allow_infinity=False),
    periods=st.integers(min_value=252, max_value=2520),
)
@settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
def test_cagr_positive_for_growing_equity(start: float, growth: float, periods: int) -> None:
    """CAGR is positive when end_value > start_value and periods >= 1 year."""
    from src.assembled_core.qa.metrics import compute_cagr

    end = start * (1.0 + growth)
    cagr = compute_cagr(start, end, periods, freq="1d")
    assume(cagr is not None)

    assert cagr > 0, f"CAGR should be positive for growing equity; got {cagr}"


# ---------------------------------------------------------------------------
# portfolio/position_sizing — vol-scaled weights
# ---------------------------------------------------------------------------

def _signals_df(symbols: list[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "symbol": symbols,
        "direction": ["LONG"] * len(symbols),
        "score": [1.0] * len(symbols),
    })


@given(
    vols=st.lists(
        st.floats(min_value=0.05, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=15,
    ),
    target_vol=st.floats(min_value=0.05, max_value=0.5, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
def test_vol_scaled_weights_non_negative(vols: list[float], target_vol: float) -> None:
    """Vol-scaled weights for long signals must all be >= 0."""
    from src.assembled_core.portfolio.position_sizing import compute_vol_scaled_weights

    symbols = [f"SYM{i}" for i in range(len(vols))]
    signals = _signals_df(symbols)
    volatilities = pd.Series(dict(zip(symbols, vols)))

    result = compute_vol_scaled_weights(signals, volatilities, target_vol=target_vol)

    if result.empty:
        return  # acceptable if all vols are invalid

    for _, row in result.iterrows():
        assert row["target_weight"] >= -1e-9, (
            f"Weight for {row['symbol']} must be >= 0; got {row['target_weight']}"
        )


# ---------------------------------------------------------------------------
# portfolio/position_sizing — Kelly weights
# ---------------------------------------------------------------------------

@given(
    win_rates=st.lists(
        st.floats(min_value=0.51, max_value=0.99, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=10,
    ),
    payoff_ratios=st.lists(
        st.floats(min_value=1.01, max_value=5.0, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=10,
    ),
)
@settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
def test_kelly_weights_non_negative_for_favourable_bets(
    win_rates: list[float], payoff_ratios: list[float]
) -> None:
    """Kelly weights for win_rate > 0.5 and payoff > 1 must be non-negative."""
    from src.assembled_core.portfolio.position_sizing import compute_kelly_weights

    n = min(len(win_rates), len(payoff_ratios))
    assume(n >= 2)

    symbols = [f"SYM{i}" for i in range(n)]
    signals = _signals_df(symbols)

    wr = pd.Series(dict(zip(symbols, win_rates[:n])))
    pr = pd.Series(dict(zip(symbols, payoff_ratios[:n])))

    result = compute_kelly_weights(signals, win_rates=wr, payoff_ratios=pr, fraction=0.5)

    if result.empty:
        return  # acceptable

    for _, row in result.iterrows():
        assert row["target_weight"] >= -1e-9, (
            f"Kelly weight for {row['symbol']} with win_rate>{0.5} must be >= 0; "
            f"got {row['target_weight']}"
        )


# ---------------------------------------------------------------------------
# risk/exposure_engine — gross exposure
# ---------------------------------------------------------------------------

@given(
    n_positions=st.integers(min_value=1, max_value=10),
    quantities=st.lists(
        st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=10,
    ),
    prices_raw=st.lists(
        st.floats(min_value=1.0, max_value=500.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=10,
    ),
)
@settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
def test_long_only_gross_exposure_non_negative(
    n_positions: int, quantities: list[float], prices_raw: list[float]
) -> None:
    """Gross exposure is >= 0 for all long positions."""
    from src.assembled_core.risk.exposure_engine import compute_exposures

    n = min(len(quantities), len(prices_raw), n_positions)
    assume(n >= 1)

    symbols = [f"SYM{i}" for i in range(n)]
    qtys = quantities[:n]
    prices = prices_raw[:n]

    target_positions = pd.DataFrame({"symbol": symbols, "target_qty": qtys})
    prices_df = pd.DataFrame({"symbol": symbols, "close": prices})

    notional = sum(q * p for q, p in zip(qtys, prices))
    equity = max(notional, 1.0)

    _, summary = compute_exposures(
        target_positions, prices_df, equity=equity, missing_price_handling="zero"
    )

    assert summary.gross_exposure >= -1e-9, (
        f"Gross exposure must be >= 0 for long positions; got {summary.gross_exposure}"
    )
