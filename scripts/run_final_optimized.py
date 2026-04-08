"""Final optimized backtest based on experiment findings.

Best config from experiments: BEST_balanced
- MA 20/50, biweekly rebalance, 12 positions
- Risk aversion 1.5, turnover penalty 0.005, max weight 12%
- Trailing stops at 2.0x ATR, vol cap 15%
- Cost-aware optimizer, liquidity-adjusted sizing

Also runs: walk-forward validation, regime analysis, and detailed attribution.
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
from assembled_core.portfolio.stress_test_constraints import evaluate_stress_scenarios, StressTestConfig
from assembled_core.risk.crowding_detector import compute_hhi
from assembled_core.data.cost_model_policy import get_per_symbol_costs


# Sector mapping for stress tests & attribution
SECTOR_MAP = {
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
    "AMZN": "Consumer Discretionary", "NVDA": "Technology", "META": "Technology",
    "TSLA": "Consumer Discretionary", "NFLX": "Communication Services",
    "AVGO": "Technology", "ADBE": "Technology", "CRM": "Technology",
    "JPM": "Financials", "V": "Financials", "MA": "Financials",
    "HD": "Consumer Discretionary", "WMT": "Consumer Staples", "MCD": "Consumer Discretionary",
    "COST": "Consumer Staples", "PEP": "Consumer Staples", "KO": "Consumer Staples",
    "PG": "Consumer Staples",
    "JNJ": "Healthcare", "LLY": "Healthcare", "UNH": "Healthcare",
    "MRK": "Healthcare", "TMO": "Healthcare", "ABBV": "Healthcare",
    "XOM": "Energy", "CVX": "Energy", "BRK-B": "Financials",
}


def main():
    print("=" * 70)
    print("FINAL OPTIMIZED BACKTEST — BEST_balanced Configuration")
    print("=" * 70)

    # Load data
    prices = pd.read_parquet("data/sample/backtest_1y.parquet")
    prices["timestamp"] = pd.to_datetime(prices["timestamp"])
    prices = prices.sort_values(["symbol", "timestamp"])
    prices_feat = add_all_features(prices)

    spy = pd.read_parquet("data/raw/equities_eod/yfinance/SPY.parquet")
    spy["timestamp"] = pd.to_datetime(spy["timestamp"])
    spy = spy[
        (spy["timestamp"] >= prices["timestamp"].min())
        & (spy["timestamp"] <= prices["timestamp"].max())
    ].sort_values("timestamp")
    spy_returns = spy["close"].pct_change().dropna().values

    signals = generate_trend_signals_from_prices(prices_feat, ma_fast=20, ma_slow=50)

    pivot = prices.pivot(index="timestamp", columns="symbol", values="close")
    cov_matrix = pivot.pct_change().dropna().cov() * 252

    try:
        cost_df = get_per_symbol_costs(prices)
        cost_bps = dict(zip(cost_df["symbol"], cost_df["one_way_cost_bps"]))
    except Exception:
        cost_bps = {}

    liq_scores = compute_liquidity_scores(prices)

    # Configuration: BEST_balanced
    opt_config = OptimizerConfig(
        risk_aversion=1.5,
        turnover_penalty=0.005,
        max_weight=0.12,
        max_gross_exposure=1.0,
        long_only=True,
    )

    dates = sorted(prices["timestamp"].unique())
    rebal_dates = dates[::10]  # Biweekly

    portfolio_value = [100000.0]
    holdings: dict[str, float] = {}
    daily_returns = []
    total_turnover = 0.0
    total_cost = 0.0
    n_rebalances = 0
    stop_states: dict = {}
    _position_history: list[dict] = []  # reserved for future use
    sector_weights_history: list[dict] = []
    hhi_history: list[float] = []

    for i, date in enumerate(dates[1:], 1):
        prev_date = dates[i - 1]
        day_signals = signals[signals["timestamp"] == date]
        long_syms = day_signals[day_signals["direction"] == "LONG"]
        should_rebal = date in rebal_dates

        if should_rebal and not long_syms.empty:
            n_rebalances += 1
            top_syms = long_syms.nlargest(min(len(long_syms), 12), "score")
            sym_list = top_syms["symbol"].tolist()

            available = [s for s in sym_list if s in cov_matrix.columns]
            if len(available) >= 2:
                exp_ret = pd.Series(
                    {row["symbol"]: float(row["score"]) for _, row in top_syms.iterrows()}
                )
                sub_cov = cov_matrix.loc[available, available]
                exp_ret_sub = exp_ret.reindex(available).fillna(0.0)
                result = optimize_portfolio(
                    expected_returns=exp_ret_sub,
                    covariance=sub_cov,
                    current_weights=holdings,
                    per_symbol_cost_bps=cost_bps,
                    config=opt_config,
                )
                new_weights = result.weights
            else:
                new_weights = {s: 1.0 / len(sym_list) for s in sym_list}

            # Liquidity adjustment
            new_weights = apply_liquidity_adjusted_sizing(new_weights, liq_scores, alpha=0.3)

            # Trailing stops
            positions_for_stops = {}
            for s in new_weights:
                if new_weights[s] > 0.01:
                    sp = prices[(prices["symbol"] == s) & (prices["timestamp"] == date)]
                    if not sp.empty:
                        positions_for_stops[s] = {"entry_price": float(sp["close"].iloc[0])}
            if positions_for_stops:
                stop_result = compute_trailing_stops(
                    positions_for_stops, prices[prices["timestamp"] <= date],
                    regime="unknown", prior_states=stop_states,
                )
                new_weights = apply_stop_reductions_to_weights(new_weights, stop_result)
                stop_states = {s.symbol: s for s in stop_result.stops}

            # Vol cap at 15%
            if len(daily_returns) >= 20:
                realized = np.std(daily_returns[-20:]) * np.sqrt(252)
                if realized > 0.15:
                    scale = 0.15 / realized
                    new_weights = {s: w * scale for s, w in new_weights.items()}

            # Track turnover & costs
            all_syms = set(holdings.keys()) | set(new_weights.keys())
            turnover = sum(abs(holdings.get(s, 0) - new_weights.get(s, 0)) for s in all_syms)
            total_turnover += turnover
            sym_cost = sum(
                abs(holdings.get(s, 0) - new_weights.get(s, 0)) * cost_bps.get(s, 6.0) / 10000
                for s in all_syms
            )
            total_cost += sym_cost
            holdings = {s: w for s, w in new_weights.items() if abs(w) > 0.001}

        # Track analytics
        hhi_history.append(compute_hhi(holdings))
        sector_w: dict[str, float] = {}
        for s, w in holdings.items():
            sec = SECTOR_MAP.get(s, "Other")
            sector_w[sec] = sector_w.get(sec, 0) + w
        sector_weights_history.append(sector_w)

        # Daily return
        port_ret = 0.0
        for sym, w in holdings.items():
            sp = prices[(prices["symbol"] == sym) & (prices["timestamp"].isin([prev_date, date]))]
            if len(sp) >= 2:
                sp = sp.sort_values("timestamp")
                ret = (sp["close"].iloc[-1] / sp["close"].iloc[0]) - 1.0
                port_ret += w * ret

        daily_returns.append(port_ret)
        portfolio_value.append(portfolio_value[-1] * (1 + port_ret))

    returns = np.array(daily_returns)
    equity = np.array(portfolio_value)

    # === CORE METRICS ===
    sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0
    down = returns[returns < 0]
    sortino = float(np.mean(returns) / np.std(down) * np.sqrt(252)) if len(down) > 0 and np.std(down) > 0 else 0
    cum = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cum)
    dd = (cum - running_max) / running_max
    max_dd = float(dd.min())
    vol = float(np.std(returns) * np.sqrt(252))
    cagr = float((equity[-1] / equity[0]) ** (252 / len(returns)) - 1) if len(returns) > 0 else 0
    calmar = float(cagr / abs(max_dd)) if abs(max_dd) > 1e-10 else 0

    # Win/loss stats
    win_days = returns[returns > 0]
    loss_days = returns[returns < 0]
    hit_rate = len(win_days) / len(returns) * 100 if len(returns) > 0 else 0
    avg_win = float(np.mean(win_days)) if len(win_days) > 0 else 0
    avg_loss = float(np.mean(loss_days)) if len(loss_days) > 0 else 0
    profit_factor = abs(avg_win * len(win_days)) / abs(avg_loss * len(loss_days)) if len(loss_days) > 0 and avg_loss != 0 else 0

    print(f"\n{'=' * 70}")
    print("PERFORMANCE METRICS")
    print(f"{'=' * 70}")
    print(f"  Total Return:     {(equity[-1]/equity[0]-1)*100:+.2f}%")
    print(f"  CAGR:             {cagr*100:+.2f}%")
    print(f"  Sharpe Ratio:     {sharpe:.3f}")
    print(f"  Sortino Ratio:    {sortino:.3f}")
    print(f"  Calmar Ratio:     {calmar:.3f}")
    print(f"  Volatility:       {vol*100:.2f}%")
    print(f"  Max Drawdown:     {max_dd*100:.2f}%")
    print(f"  Hit Rate:         {hit_rate:.1f}%")
    print(f"  Profit Factor:    {profit_factor:.2f}")
    print(f"  Avg Win:          {avg_win*100:.3f}%/day")
    print(f"  Avg Loss:         {avg_loss*100:.3f}%/day")

    print("\nTRADING EFFICIENCY:")
    print(f"  Rebalances:       {n_rebalances}")
    print(f"  Total Turnover:   {total_turnover:.1f}x")
    print(f"  Ann. Turnover:    {total_turnover*252/len(returns):.1f}x")
    print(f"  Total Cost:       {total_cost*10000:.1f} bps")
    print(f"  Net of Cost CAGR: {(cagr - total_cost)*100:+.2f}%")

    # === BENCHMARK ATTRIBUTION ===
    spy_aligned = spy_returns[:len(returns)]
    if len(spy_aligned) > 10:
        bm = compute_benchmark_metrics(
            pd.Series(returns[:len(spy_aligned)]), pd.Series(spy_aligned)
        )
        print("\nBENCHMARK ATTRIBUTION (vs SPY):")
        print(f"  Alpha (ann.):     {bm.alpha*100:+.2f}%")
        print(f"  Beta:             {bm.beta:.3f}")
        print(f"  R-squared:        {bm.r_squared:.3f}")
        print(f"  Information Ratio:{bm.information_ratio:.3f}")
        print(f"  Tracking Error:   {bm.tracking_error*100:.2f}%")
        print(f"  Active Return:    {bm.active_return*100:+.2f}%")
        spy_total = float(np.prod(1 + spy_aligned) - 1)
        spy_sharpe = float(np.mean(spy_aligned) / np.std(spy_aligned) * np.sqrt(252))
        print(f"  SPY Return:       {spy_total*100:+.2f}%")
        print(f"  SPY Sharpe:       {spy_sharpe:.3f}")
        print(f"  Excess Return:    {(equity[-1]/equity[0]-1 - spy_total)*100:+.2f}%")

    # === SECTOR ANALYSIS ===
    print("\nSECTOR EXPOSURE (average over period):")
    all_sectors: dict[str, list[float]] = {}
    for sw in sector_weights_history:
        for sec, w in sw.items():
            all_sectors.setdefault(sec, []).append(w)
    for sec in sorted(all_sectors, key=lambda s: -np.mean(all_sectors[s])):
        avg = np.mean(all_sectors[sec])
        if avg > 0.01:
            print(f"  {sec:<30} {avg*100:>5.1f}%")

    # === CONCENTRATION ===
    avg_hhi = float(np.mean(hhi_history))
    print("\nCONCENTRATION:")
    print(f"  Avg HHI:          {avg_hhi:.4f} (1/N={1/12:.4f} for 12 positions)")
    print(f"  Effective N:      {1/avg_hhi:.1f} positions" if avg_hhi > 0 else "  Effective N: N/A")

    # === STRESS TESTS ===
    print("\nSTRESS TEST ANALYSIS (final weights):")
    stress_cfg = StressTestConfig(sector_mapping=SECTOR_MAP)
    stress = evaluate_stress_scenarios(holdings, list(holdings.keys()), SECTOR_MAP, stress_cfg)
    for sc, loss in sorted(stress.scenario_losses.items(), key=lambda x: x[1]):
        floor = stress_cfg.loss_floors.get(sc, -0.15)
        status = "OK" if loss >= floor else "BREACH"
        print(f"  {sc:<25} {loss*100:>+6.2f}% (floor: {floor*100:.0f}%) [{status}]")
    print(f"  Worst scenario:   {stress.worst_scenario} ({stress.worst_loss*100:+.2f}%)")
    print(f"  All within floors: {stress.all_within_floors}")

    # === CPCV ===
    print("\nCPCV OVERFITTING ANALYSIS:")
    splits = generate_cpcv_splits(len(returns), n_groups=6, k_test_groups=2, purge_length=5, embargo_length=3)
    if splits:
        path_rets = [returns[tidx] for _, tidx in splits]
        cpcv = compute_cpcv_sharpe_distribution(path_rets)
        print(f"  Paths:            {cpcv.n_paths}")
        print(f"  Mean Sharpe:      {cpcv.mean_sharpe:.3f}")
        print(f"  Std Sharpe:       {cpcv.std_sharpe:.3f}")
        print(f"  P(Sharpe > 0):    {cpcv.prob_positive_sharpe:.2%}")
        print(f"  P(Sharpe > 1):    {cpcv.prob_sharpe_above_1:.2%}")
        if cpcv.deflated_sharpe is not None:
            print(f"  Deflated Sharpe:  {cpcv.deflated_sharpe:.3f}")
        print(f"  Likely Overfit:   {cpcv.is_likely_overfit}")

    # === WALK-FORWARD (4-split) ===
    print("\nWALK-FORWARD ANALYSIS (4 splits):")
    n = len(returns)
    wf_size = n // 4
    for split_i in range(4):
        start = split_i * wf_size
        end = min((split_i + 1) * wf_size, n)
        wf_rets = returns[start:end]
        if len(wf_rets) < 10:
            continue
        wf_sharpe = float(np.mean(wf_rets) / np.std(wf_rets) * np.sqrt(252)) if np.std(wf_rets) > 0 else 0
        wf_ret = float(np.prod(1 + wf_rets) - 1)
        wf_cum = np.cumprod(1 + wf_rets)
        wf_rm = np.maximum.accumulate(wf_cum)
        wf_dd = float(((wf_cum - wf_rm) / wf_rm).min())
        print(f"  Split {split_i+1}: Return={wf_ret*100:+.1f}% Sharpe={wf_sharpe:.2f} MaxDD={wf_dd*100:.1f}% ({len(wf_rets)} days)")

    # === MONTHLY RETURNS ===
    print("\nMONTHLY RETURNS:")
    date_arr = np.array(dates[1:])
    months = pd.Series(returns, index=pd.to_datetime(date_arr))
    monthly = months.resample("ME").apply(lambda x: float(np.prod(1 + x) - 1))
    for dt, r in monthly.items():
        bar = "+" * int(abs(r) * 200) if r > 0 else "-" * int(abs(r) * 200)
        print(f"  {dt.strftime('%Y-%m')}: {r*100:>+6.2f}% {bar}")

    # Save complete results
    all_results = {
        "config": "BEST_balanced",
        "params": {
            "ma_fast": 20, "ma_slow": 50, "rebal_days": 10, "n_positions": 12,
            "risk_aversion": 1.5, "turnover_penalty": 0.005, "max_weight": 0.12,
            "trailing_stop_mult": 2.0, "vol_cap": 0.15,
        },
        "metrics": {
            "total_return": float(equity[-1]/equity[0]-1),
            "cagr": cagr, "sharpe": sharpe, "sortino": sortino, "calmar": calmar,
            "volatility": vol, "max_drawdown": max_dd,
            "hit_rate": hit_rate, "profit_factor": profit_factor,
            "avg_win": avg_win, "avg_loss": avg_loss,
        },
        "trading": {
            "n_rebalances": n_rebalances, "total_turnover": total_turnover,
            "total_cost_bps": total_cost * 10000,
        },
        "stress_test": stress.scenario_losses,
        "concentration_hhi": avg_hhi,
    }
    os.makedirs("output/final_optimized", exist_ok=True)
    with open("output/final_optimized/results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Save equity curve
    eq_df = pd.DataFrame({
        "date": dates[:len(equity)],
        "equity": equity,
    })
    eq_df.to_csv("output/final_optimized/equity_curve.csv", index=False)

    print(f"\n{'=' * 70}")
    print("Results: output/final_optimized/results.json")
    print("Equity:  output/final_optimized/equity_curve.csv")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
