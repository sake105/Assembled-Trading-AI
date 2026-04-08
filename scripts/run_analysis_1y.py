"""1-Year Backtest Analysis using V1-V20 Improvements (Phase 6)."""
from __future__ import annotations

import sys
import os
import json

sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

from assembled_core.signals.rules_trend import generate_trend_signals_from_prices
from assembled_core.features.ta_features import add_all_features
from assembled_core.qa.benchmark_metrics import compute_benchmark_metrics
from assembled_core.ml.cpcv import generate_cpcv_splits, compute_cpcv_sharpe_distribution
from assembled_core.risk.liquidity_scoring import compute_liquidity_scores
from assembled_core.data.cost_model_policy import get_per_symbol_costs


def main():
    # Load data
    prices = pd.read_parquet("data/sample/backtest_1y.parquet")
    prices["timestamp"] = pd.to_datetime(prices["timestamp"])
    prices = prices.sort_values(["symbol", "timestamp"])

    # Add features
    prices_feat = add_all_features(prices)
    print(
        f"Data: {len(prices)} rows, {prices['symbol'].nunique()} symbols, "
        f"{prices['timestamp'].min().date()} to {prices['timestamp'].max().date()}"
    )

    # Generate signals
    signals = generate_trend_signals_from_prices(prices_feat, ma_fast=20, ma_slow=50)
    n_long = len(signals[signals["direction"] == "LONG"])
    n_short = len(signals[signals["direction"] == "SHORT"])
    n_flat = len(signals[signals["direction"] == "FLAT"])
    print(f"Signals: {len(signals)} rows, LONG={n_long}, SHORT={n_short}, FLAT={n_flat}")

    # Build SPY benchmark
    has_spy = False
    spy_returns = np.array([])
    try:
        spy = pd.read_parquet("data/raw/equities_eod/yfinance/SPY.parquet")
        spy["timestamp"] = pd.to_datetime(spy["timestamp"])
        spy = spy[
            (spy["timestamp"] >= prices["timestamp"].min())
            & (spy["timestamp"] <= prices["timestamp"].max())
        ]
        spy = spy.sort_values("timestamp")
        spy_returns = spy["close"].pct_change().dropna().values
        has_spy = True
        print(f"SPY benchmark: {len(spy_returns)} returns")
    except Exception as e:
        print(f"No SPY benchmark: {e}")

    # Simple portfolio simulation: equal-weight top-10 long signals, monthly rebalance
    dates = sorted(prices["timestamp"].unique())
    rebal_dates = dates[::20]  # ~Monthly

    portfolio_value = [100000.0]
    holdings: dict[str, float] = {}
    daily_returns = []

    for i, date in enumerate(dates[1:], 1):
        prev_date = dates[i - 1]

        # Get signals for this date
        day_signals = signals[signals["timestamp"] == date]

        # Rebalance on rebal dates
        long_syms = day_signals[day_signals["direction"] == "LONG"]
        if date in rebal_dates and not long_syms.empty:
            n = min(len(long_syms), 10)
            top_syms = long_syms.nlargest(n, "score")["symbol"].tolist()
            holdings = {s: 1.0 / n for s in top_syms}

        # Compute daily return
        port_ret = 0.0
        for sym, w in holdings.items():
            sym_prices = prices[
                (prices["symbol"] == sym)
                & (prices["timestamp"].isin([prev_date, date]))
            ]
            if len(sym_prices) >= 2:
                sym_prices = sym_prices.sort_values("timestamp")
                ret = (sym_prices["close"].iloc[-1] / sym_prices["close"].iloc[0]) - 1.0
                port_ret += w * ret

        daily_returns.append(port_ret)
        portfolio_value.append(portfolio_value[-1] * (1 + port_ret))

    returns = np.array(daily_returns)
    equity = np.array(portfolio_value)

    print(f"\n{'=' * 60}")
    print("1-YEAR BACKTEST RESULTS (Apr 2025 - Mar 2026)")
    print(f"{'=' * 60}")
    print("Starting Capital: $100,000")
    print(f"Final Value:      ${equity[-1]:,.2f}")
    print(f"Total Return:     {(equity[-1] / equity[0] - 1) * 100:.2f}%")

    # Key metrics
    sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0
    sortino_down = returns[returns < 0]
    sortino = (
        float(np.mean(returns) / np.std(sortino_down) * np.sqrt(252))
        if len(sortino_down) > 0 and np.std(sortino_down) > 0
        else 0
    )
    cum = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cum)
    dd = (cum - running_max) / running_max
    max_dd = float(dd.min())
    vol = float(np.std(returns) * np.sqrt(252))
    cagr = float((equity[-1] / equity[0]) ** (252 / len(returns)) - 1) if len(returns) > 0 else 0
    calmar = float(cagr / abs(max_dd)) if abs(max_dd) > 0 else 0

    print("\nRisk-Adjusted Metrics:")
    print(f"  Sharpe Ratio:   {sharpe:.3f}")
    print(f"  Sortino Ratio:  {sortino:.3f}")
    print(f"  Calmar Ratio:   {calmar:.3f}")
    print(f"  CAGR:           {cagr * 100:.2f}%")
    print(f"  Volatility:     {vol * 100:.2f}%")
    print(f"  Max Drawdown:   {max_dd * 100:.2f}%")

    print("\nTrading Stats:")
    print(f"  Rebalance Days: {len(rebal_dates)}")
    print("  Avg Positions:  ~10 (top scoring)")

    # Benchmark comparison
    bm_dict = {}
    if has_spy:
        spy_ret_aligned = spy_returns[: len(returns)]
        if len(spy_ret_aligned) > 10:
            bm = compute_benchmark_metrics(
                pd.Series(returns[: len(spy_ret_aligned)]),
                pd.Series(spy_ret_aligned),
            )
            bm_dict = {
                "alpha": bm.alpha,
                "beta": bm.beta,
                "information_ratio": bm.information_ratio,
                "tracking_error": bm.tracking_error,
                "active_return": bm.active_return,
                "up_capture": bm.up_capture,
                "down_capture": bm.down_capture,
            }
            spy_total = float(np.prod(1 + spy_ret_aligned) - 1)
            spy_sharpe = float(
                np.mean(spy_ret_aligned) / np.std(spy_ret_aligned) * np.sqrt(252)
            )
            print("\nBenchmark Comparison (vs SPY):")
            print(f"  Alpha (ann.):     {bm.alpha * 100:.2f}%")
            print(f"  Beta:             {bm.beta:.3f}")
            print(f"  Information Ratio:{bm.information_ratio:.3f}")
            print(f"  Tracking Error:   {bm.tracking_error * 100:.2f}%")
            print(f"  Active Return:    {bm.active_return * 100:.2f}%")
            print(f"  Up Capture:       {bm.up_capture * 100:.1f}%")
            print(f"  Down Capture:     {bm.down_capture * 100:.1f}%")
            print(f"\n  SPY Total Return: {spy_total * 100:.2f}%")
            print(f"  SPY Sharpe:       {spy_sharpe:.3f}")

    # CPCV Analysis
    print("\nCPCV Overfitting Analysis:")
    splits = generate_cpcv_splits(
        len(returns), n_groups=6, k_test_groups=2, purge_length=5, embargo_length=3
    )
    cpcv_dict = {}
    if splits:
        path_returns = []
        for train_idx, test_idx in splits:
            test_rets = returns[test_idx]
            path_returns.append(test_rets)
        cpcv = compute_cpcv_sharpe_distribution(path_returns)
        cpcv_dict = {
            "n_paths": cpcv.n_paths,
            "mean_sharpe": cpcv.mean_sharpe,
            "std_sharpe": cpcv.std_sharpe,
            "prob_positive_sharpe": cpcv.prob_positive_sharpe,
            "prob_sharpe_above_1": cpcv.prob_sharpe_above_1,
            "deflated_sharpe": cpcv.deflated_sharpe,
            "is_likely_overfit": cpcv.is_likely_overfit,
        }
        print(f"  Paths:            {cpcv.n_paths}")
        print(f"  Mean Sharpe:      {cpcv.mean_sharpe:.3f}")
        print(f"  Std Sharpe:       {cpcv.std_sharpe:.3f}")
        print(f"  P(Sharpe > 0):    {cpcv.prob_positive_sharpe:.2%}")
        print(f"  P(Sharpe > 1):    {cpcv.prob_sharpe_above_1:.2%}")
        if cpcv.deflated_sharpe is not None:
            print(f"  Deflated Sharpe:  {cpcv.deflated_sharpe:.3f}")
        print(f"  Likely Overfit:   {cpcv.is_likely_overfit}")

    # Liquidity scores
    print("\nLiquidity Analysis:")
    liq_scores = compute_liquidity_scores(prices)
    liq_dict = {}
    if liq_scores:
        tiers: dict[str, int] = {}
        for s in liq_scores:
            tiers[s.tier] = tiers.get(s.tier, 0) + 1
        avg_score = float(np.mean([s.score for s in liq_scores]))
        liq_dict = {"tier_distribution": tiers, "avg_score": avg_score}
        print(f"  Tier distribution: {tiers}")
        print(f"  Avg liquidity:    {avg_score:.3f}")

    # Cost analysis
    print("\nCost Model Analysis:")
    cost_dict = {}
    try:
        costs = get_per_symbol_costs(prices)
        if not costs.empty:
            cost_dict = {
                "n_symbols": len(costs),
                "avg_one_way_bps": float(costs["one_way_cost_bps"].mean()),
                "min_bps": float(costs["one_way_cost_bps"].min()),
                "max_bps": float(costs["one_way_cost_bps"].max()),
            }
            print(f"  Symbols with costs: {len(costs)}")
            print(f"  Avg one-way cost:  {costs['one_way_cost_bps'].mean():.1f} bps")
            print(f"  Cost range:        {costs['one_way_cost_bps'].min():.1f} - {costs['one_way_cost_bps'].max():.1f} bps")
    except Exception as e:
        print(f"  Cost model: {e}")

    print(f"\n{'=' * 60}")
    print("ANALYSIS COMPLETE")
    print(f"{'=' * 60}")

    # Save results
    results = {
        "total_return": float(equity[-1] / equity[0] - 1),
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "cagr": cagr,
        "volatility": vol,
        "max_drawdown": max_dd,
        "n_rebalances": len(rebal_dates),
        "n_symbols": int(prices["symbol"].nunique()),
        "n_trading_days": len(returns),
        "benchmark": bm_dict,
        "cpcv": cpcv_dict,
        "liquidity": liq_dict,
        "costs": cost_dict,
    }
    os.makedirs("output/backtest_1y", exist_ok=True)
    with open("output/backtest_1y/analysis_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("Results saved to output/backtest_1y/analysis_results.json")


if __name__ == "__main__":
    main()
