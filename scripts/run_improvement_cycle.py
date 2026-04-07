"""Phase 7: Improvement cycle after backtest analysis.

Applies V9 (cost-aware optimizer), V15 (liquidity sizing), V16 (trailing stops),
and V14 (less frequent rebalancing) to improve the baseline backtest.
"""
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
from assembled_core.risk.liquidity_scoring import compute_liquidity_scores, apply_liquidity_adjusted_sizing
from assembled_core.risk.trailing_stops import compute_trailing_stops, apply_stop_reductions_to_weights
from assembled_core.portfolio.cost_aware_optimizer import optimize_portfolio, OptimizerConfig
from assembled_core.data.cost_model_policy import get_per_symbol_costs


def main():
    print("=" * 60)
    print("PHASE 7: IMPROVEMENT CYCLE")
    print("=" * 60)

    # Load data
    prices = pd.read_parquet("data/sample/backtest_1y.parquet")
    prices["timestamp"] = pd.to_datetime(prices["timestamp"])
    prices = prices.sort_values(["symbol", "timestamp"])
    prices_feat = add_all_features(prices)

    # SPY benchmark
    spy = pd.read_parquet("data/raw/equities_eod/yfinance/SPY.parquet")
    spy["timestamp"] = pd.to_datetime(spy["timestamp"])
    spy = spy[
        (spy["timestamp"] >= prices["timestamp"].min())
        & (spy["timestamp"] <= prices["timestamp"].max())
    ].sort_values("timestamp")
    spy_returns = spy["close"].pct_change().dropna().values

    # Generate signals
    signals = generate_trend_signals_from_prices(prices_feat, ma_fast=20, ma_slow=50)

    # Cost model
    try:
        cost_df = get_per_symbol_costs(prices)
        cost_bps = dict(zip(cost_df["symbol"], cost_df["one_way_cost_bps"]))
    except Exception:
        cost_bps = {}

    # Liquidity scores
    liq_scores = compute_liquidity_scores(prices)

    # Compute covariance for optimizer
    symbols = sorted(prices["symbol"].unique())
    pivot = prices.pivot(index="timestamp", columns="symbol", values="close")
    rets_matrix = pivot.pct_change().dropna()
    cov_matrix = rets_matrix.cov() * 252  # Annualized

    dates = sorted(prices["timestamp"].unique())
    rebal_dates = dates[::20]  # Monthly

    # IMPROVEMENT SETTINGS
    optimizer_config = OptimizerConfig(
        risk_aversion=1.5,
        turnover_penalty=0.005,  # Higher penalty for turnover reduction
        max_weight=0.15,
        max_gross_exposure=1.0,
        long_only=True,
    )

    portfolio_value = [100000.0]
    holdings: dict[str, float] = {}
    daily_returns = []
    total_turnover = 0.0
    total_cost = 0.0
    n_rebalances = 0

    # Trailing stop state
    stop_states: dict = {}

    for i, date in enumerate(dates[1:], 1):
        prev_date = dates[i - 1]
        day_signals = signals[signals["timestamp"] == date]

        # Rebalance check
        should_rebal = date in rebal_dates
        long_syms = day_signals[day_signals["direction"] == "LONG"]

        if should_rebal and not long_syms.empty:
            n_rebalances += 1
            top_syms = long_syms.nlargest(min(len(long_syms), 15), "score")
            sym_list = top_syms["symbol"].tolist()

            # Build expected returns from signal scores
            exp_ret = pd.Series(
                {row["symbol"]: float(row["score"]) for _, row in top_syms.iterrows()}
            )

            # Get covariance subset
            available_syms = [s for s in sym_list if s in cov_matrix.columns]
            if len(available_syms) >= 2:
                sub_cov = cov_matrix.loc[available_syms, available_syms]
                exp_ret_sub = exp_ret.reindex(available_syms).fillna(0.0)

                # V9: Cost-aware optimization
                result = optimize_portfolio(
                    expected_returns=exp_ret_sub,
                    covariance=sub_cov,
                    current_weights=holdings,
                    per_symbol_cost_bps=cost_bps,
                    config=optimizer_config,
                )

                new_weights = result.weights

                # V15: Liquidity-adjusted sizing
                new_weights = apply_liquidity_adjusted_sizing(
                    new_weights, liq_scores, alpha=0.3
                )

                # V16: Trailing stop reductions
                positions_for_stops = {
                    s: {"entry_price": float(prices[(prices["symbol"] == s) & (prices["timestamp"] == date)]["close"].iloc[0]) if not prices[(prices["symbol"] == s) & (prices["timestamp"] == date)].empty else 0.0}
                    for s in new_weights if new_weights[s] > 0.01
                }
                if positions_for_stops:
                    stop_result = compute_trailing_stops(
                        positions_for_stops, prices[prices["timestamp"] <= date],
                        regime="unknown", prior_states=stop_states,
                    )
                    new_weights = apply_stop_reductions_to_weights(new_weights, stop_result)
                    stop_states = {s.symbol: s for s in stop_result.stops}

                # Track turnover
                all_syms = set(holdings.keys()) | set(new_weights.keys())
                turnover = sum(abs(holdings.get(s, 0) - new_weights.get(s, 0)) for s in all_syms)
                total_turnover += turnover

                # Track costs
                sym_cost = sum(
                    abs(holdings.get(s, 0) - new_weights.get(s, 0)) * cost_bps.get(s, 6.0) / 10000
                    for s in all_syms
                )
                total_cost += sym_cost

                holdings = {s: w for s, w in new_weights.items() if abs(w) > 0.001}

        # Compute daily return
        port_ret = 0.0
        for sym, w in holdings.items():
            sym_prices = prices[
                (prices["symbol"] == sym) & (prices["timestamp"].isin([prev_date, date]))
            ]
            if len(sym_prices) >= 2:
                sym_prices = sym_prices.sort_values("timestamp")
                ret = (sym_prices["close"].iloc[-1] / sym_prices["close"].iloc[0]) - 1.0
                port_ret += w * ret

        # Deduct daily cost fraction
        port_ret -= total_cost / max(len(dates), 1) * 0.01  # Amortize

        daily_returns.append(port_ret)
        portfolio_value.append(portfolio_value[-1] * (1 + port_ret))

    returns = np.array(daily_returns)
    equity = np.array(portfolio_value)

    # === RESULTS ===
    print(f"\n{'=' * 60}")
    print(f"IMPROVED BACKTEST RESULTS (Apr 2025 - Mar 2026)")
    print(f"{'=' * 60}")
    print(f"Starting Capital: $100,000")
    print(f"Final Value:      ${equity[-1]:,.2f}")
    print(f"Total Return:     {(equity[-1] / equity[0] - 1) * 100:.2f}%")

    sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0
    sortino_down = returns[returns < 0]
    sortino = float(np.mean(returns) / np.std(sortino_down) * np.sqrt(252)) if len(sortino_down) > 0 and np.std(sortino_down) > 0 else 0
    cum = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cum)
    dd = (cum - running_max) / running_max
    max_dd = float(dd.min())
    vol = float(np.std(returns) * np.sqrt(252))
    cagr = float((equity[-1] / equity[0]) ** (252 / len(returns)) - 1) if len(returns) > 0 else 0
    calmar = float(cagr / abs(max_dd)) if abs(max_dd) > 0 else 0

    print(f"\nRisk-Adjusted Metrics:")
    print(f"  Sharpe Ratio:   {sharpe:.3f}")
    print(f"  Sortino Ratio:  {sortino:.3f}")
    print(f"  Calmar Ratio:   {calmar:.3f}")
    print(f"  CAGR:           {cagr * 100:.2f}%")
    print(f"  Volatility:     {vol * 100:.2f}%")
    print(f"  Max Drawdown:   {max_dd * 100:.2f}%")

    print(f"\nTrading Efficiency:")
    print(f"  Rebalances:     {n_rebalances}")
    print(f"  Total Turnover: {total_turnover:.2f}x")
    print(f"  Annualized TO:  {total_turnover * 252 / len(returns):.1f}x")
    print(f"  Total Cost:     {total_cost * 10000:.1f} bps")

    # Benchmark
    spy_ret_aligned = spy_returns[:len(returns)]
    if len(spy_ret_aligned) > 10:
        bm = compute_benchmark_metrics(
            pd.Series(returns[:len(spy_ret_aligned)]),
            pd.Series(spy_ret_aligned),
        )
        print(f"\nBenchmark Comparison (vs SPY):")
        print(f"  Alpha (ann.):     {bm.alpha * 100:.2f}%")
        print(f"  Beta:             {bm.beta:.3f}")
        print(f"  Information Ratio:{bm.information_ratio:.3f}")
        print(f"  Tracking Error:   {bm.tracking_error * 100:.2f}%")

    # CPCV
    splits = generate_cpcv_splits(len(returns), n_groups=6, k_test_groups=2, purge_length=5, embargo_length=3)
    if splits:
        path_returns = [returns[test_idx] for _, test_idx in splits]
        cpcv = compute_cpcv_sharpe_distribution(path_returns)
        print(f"\nCPCV Analysis:")
        print(f"  Mean Sharpe:      {cpcv.mean_sharpe:.3f}")
        print(f"  P(Sharpe > 0):    {cpcv.prob_positive_sharpe:.2%}")
        print(f"  Deflated Sharpe:  {cpcv.deflated_sharpe}")
        print(f"  Likely Overfit:   {cpcv.is_likely_overfit}")

    # Load baseline for comparison
    print(f"\n{'=' * 60}")
    print(f"COMPARISON: BASELINE vs IMPROVED")
    print(f"{'=' * 60}")
    try:
        with open("output/backtest_1y/analysis_results.json") as f:
            baseline = json.load(f)
        print(f"{'Metric':<25} {'Baseline':>12} {'Improved':>12} {'Delta':>12}")
        print("-" * 61)
        comparisons = [
            ("Total Return", baseline["total_return"] * 100, (equity[-1] / equity[0] - 1) * 100, "%"),
            ("Sharpe", baseline["sharpe"], sharpe, ""),
            ("Sortino", baseline["sortino"], sortino, ""),
            ("Calmar", baseline["calmar"], calmar, ""),
            ("CAGR", baseline["cagr"] * 100, cagr * 100, "%"),
            ("Volatility", baseline["volatility"] * 100, vol * 100, "%"),
            ("Max Drawdown", baseline["max_drawdown"] * 100, max_dd * 100, "%"),
        ]
        for name, base, improved, unit in comparisons:
            delta = improved - base
            arrow = "+" if delta > 0 else ""
            print(f"{name:<25} {base:>10.2f}{unit:>2} {improved:>10.2f}{unit:>2} {arrow}{delta:>9.2f}{unit}")
    except Exception as e:
        print(f"Could not load baseline: {e}")

    # Save improved results
    improved_results = {
        "total_return": float(equity[-1] / equity[0] - 1),
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "cagr": cagr,
        "volatility": vol,
        "max_drawdown": max_dd,
        "n_rebalances": n_rebalances,
        "total_turnover": total_turnover,
        "total_cost_bps": total_cost * 10000,
        "optimizer": "cost_aware_v9",
        "liquidity_adjusted": True,
        "trailing_stops": True,
    }
    os.makedirs("output/backtest_1y", exist_ok=True)
    with open("output/backtest_1y/improved_results.json", "w") as f:
        json.dump(improved_results, f, indent=2)
    print(f"\nResults saved to output/backtest_1y/improved_results.json")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
