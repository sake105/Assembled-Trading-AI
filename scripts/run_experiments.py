"""Systematic parameter experimentation across strategy dimensions.

Tests variations of: rebalance frequency, position count, MA windows,
risk aversion, turnover penalty, max weight, trailing stop tightness,
and vol cap. Produces a comparison table of all configurations.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass

sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

from assembled_core.data.cost_model_policy import get_per_symbol_costs
from assembled_core.features.ta_features import add_all_features
from assembled_core.ml.cpcv import (
    compute_cpcv_sharpe_distribution,
    generate_cpcv_splits,
)
from assembled_core.portfolio.cost_aware_optimizer import (
    OptimizerConfig,
    optimize_portfolio,
)
from assembled_core.qa.benchmark_metrics import compute_benchmark_metrics
from assembled_core.risk.liquidity_scoring import (
    apply_liquidity_adjusted_sizing,
    compute_liquidity_scores,
)
from assembled_core.risk.trailing_stops import (
    apply_stop_reductions_to_weights,
    compute_trailing_stops,
)
from assembled_core.signals.rules_trend import generate_trend_signals_from_prices


@dataclass
class ExperimentConfig:
    name: str
    ma_fast: int = 20
    ma_slow: int = 50
    rebal_every_n_days: int = 20
    n_positions: int = 10
    risk_aversion: float = 1.5
    turnover_penalty: float = 0.005
    max_weight: float = 0.15
    use_optimizer: bool = True
    use_liquidity: bool = True
    use_trailing_stops: bool = True
    trailing_stop_mult: float = 2.0
    vol_cap: float | None = None  # If set, scale down when port vol > cap


@dataclass
class ExperimentResult:
    name: str
    total_return: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    cagr: float = 0.0
    volatility: float = 0.0
    max_drawdown: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    turnover: float = 0.0
    n_rebalances: int = 0
    cost_bps: float = 0.0
    cpcv_mean_sharpe: float = 0.0
    cpcv_overfit: bool = False
    runtime_s: float = 0.0


def run_experiment(
    config: ExperimentConfig,
    prices: pd.DataFrame,
    prices_feat: pd.DataFrame,
    spy_returns: np.ndarray,
    cov_matrix: pd.DataFrame,
    cost_bps: dict[str, float],
    liq_scores: list,
) -> ExperimentResult:
    """Run a single experiment configuration."""
    t0 = time.time()

    # Generate signals with configured MA windows
    signals = generate_trend_signals_from_prices(
        prices_feat, ma_fast=config.ma_fast, ma_slow=config.ma_slow
    )

    dates = sorted(prices["timestamp"].unique())
    rebal_dates = dates[:: config.rebal_every_n_days]

    opt_config = OptimizerConfig(
        risk_aversion=config.risk_aversion,
        turnover_penalty=config.turnover_penalty,
        max_weight=config.max_weight,
        max_gross_exposure=1.0,
        long_only=True,
    )

    portfolio_value = [100000.0]
    holdings: dict[str, float] = {}
    daily_returns = []
    total_turnover = 0.0
    total_cost = 0.0
    n_rebalances = 0
    stop_states: dict = {}
    recent_vols: list[float] = []

    for i, date in enumerate(dates[1:], 1):
        prev_date = dates[i - 1]
        day_signals = signals[signals["timestamp"] == date]
        long_syms = day_signals[day_signals["direction"] == "LONG"]
        should_rebal = date in rebal_dates

        if should_rebal and not long_syms.empty:
            n_rebalances += 1
            top_syms = long_syms.nlargest(
                min(len(long_syms), config.n_positions), "score"
            )
            sym_list = top_syms["symbol"].tolist()

            if config.use_optimizer and len(sym_list) >= 2:
                exp_ret = pd.Series(
                    dict(zip(top_syms["symbol"], top_syms["score"].astype(float)))
                )
                available = [s for s in sym_list if s in cov_matrix.columns]
                if len(available) >= 2:
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
            else:
                new_weights = {s: 1.0 / len(sym_list) for s in sym_list}

            # Liquidity adjustment
            if config.use_liquidity and liq_scores:
                new_weights = apply_liquidity_adjusted_sizing(
                    new_weights, liq_scores, alpha=0.3
                )

            # Trailing stops
            if config.use_trailing_stops:
                positions_for_stops = {}
                for s in new_weights:
                    if new_weights[s] > 0.01:
                        sp = prices[
                            (prices["symbol"] == s) & (prices["timestamp"] == date)
                        ]
                        if not sp.empty:
                            positions_for_stops[s] = {
                                "entry_price": float(sp["close"].iloc[0])
                            }
                if positions_for_stops:
                    from assembled_core.risk.trailing_stops import _REGIME_MULTIPLIERS

                    custom_mult = dict(_REGIME_MULTIPLIERS)
                    for k in custom_mult:
                        custom_mult[k] = config.trailing_stop_mult
                    stop_result = compute_trailing_stops(
                        positions_for_stops,
                        prices[prices["timestamp"] <= date],
                        regime="unknown",
                        custom_multipliers=custom_mult,
                        prior_states=stop_states,
                    )
                    new_weights = apply_stop_reductions_to_weights(
                        new_weights, stop_result
                    )
                    stop_states = {s.symbol: s for s in stop_result.stops}

            # Vol targeting
            if config.vol_cap and len(recent_vols) >= 20:
                realized = np.std(daily_returns[-20:]) * np.sqrt(252)
                if realized > config.vol_cap:
                    scale = config.vol_cap / realized
                    new_weights = {s: w * scale for s, w in new_weights.items()}

            # Turnover & cost tracking
            all_syms = set(holdings.keys()) | set(new_weights.keys())
            turnover = sum(
                abs(holdings.get(s, 0) - new_weights.get(s, 0)) for s in all_syms
            )
            total_turnover += turnover
            sym_cost = sum(
                abs(holdings.get(s, 0) - new_weights.get(s, 0))
                * cost_bps.get(s, 6.0)
                / 10000
                for s in all_syms
            )
            total_cost += sym_cost

            holdings = {s: w for s, w in new_weights.items() if abs(w) > 0.001}

        # Daily return
        port_ret = 0.0
        for sym, w in holdings.items():
            sp = prices[
                (prices["symbol"] == sym)
                & (prices["timestamp"].isin([prev_date, date]))
            ]
            if len(sp) >= 2:
                sp = sp.sort_values("timestamp")
                ret = (sp["close"].iloc[-1] / sp["close"].iloc[0]) - 1.0
                port_ret += w * ret

        daily_returns.append(port_ret)
        portfolio_value.append(portfolio_value[-1] * (1 + port_ret))

    returns = np.array(daily_returns)
    equity = np.array(portfolio_value)

    # Metrics
    sharpe = (
        float(np.mean(returns) / np.std(returns) * np.sqrt(252))
        if np.std(returns) > 0
        else 0
    )
    down = returns[returns < 0]
    sortino = (
        float(np.mean(returns) / np.std(down) * np.sqrt(252))
        if len(down) > 0 and np.std(down) > 0
        else 0
    )
    cum = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cum)
    dd = (cum - running_max) / running_max
    max_dd = float(dd.min())
    vol = float(np.std(returns) * np.sqrt(252))
    cagr = (
        float((equity[-1] / equity[0]) ** (252 / len(returns)) - 1)
        if len(returns) > 0
        else 0
    )
    calmar = float(cagr / abs(max_dd)) if abs(max_dd) > 1e-10 else 0

    # Benchmark
    alpha_val = 0.0
    beta_val = 0.0
    spy_aligned = spy_returns[: len(returns)]
    if len(spy_aligned) > 10:
        bm = compute_benchmark_metrics(
            pd.Series(returns[: len(spy_aligned)]), pd.Series(spy_aligned)
        )
        alpha_val = bm.alpha
        beta_val = bm.beta

    # Quick CPCV
    cpcv_sharpe = 0.0
    cpcv_overfit = False
    splits = generate_cpcv_splits(
        len(returns), n_groups=6, k_test_groups=2, purge_length=5, embargo_length=3
    )
    if splits:
        path_rets = [returns[tidx] for _, tidx in splits]
        cpcv = compute_cpcv_sharpe_distribution(path_rets)
        cpcv_sharpe = cpcv.mean_sharpe
        cpcv_overfit = cpcv.is_likely_overfit

    runtime = time.time() - t0

    return ExperimentResult(
        name=config.name,
        total_return=float(equity[-1] / equity[0] - 1),
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        cagr=cagr,
        volatility=vol,
        max_drawdown=max_dd,
        alpha=alpha_val,
        beta=beta_val,
        turnover=total_turnover,
        n_rebalances=n_rebalances,
        cost_bps=total_cost * 10000,
        cpcv_mean_sharpe=cpcv_sharpe,
        cpcv_overfit=cpcv_overfit,
        runtime_s=runtime,
    )


def main():
    print("=" * 80)
    print("PARAMETER EXPERIMENTATION LAB")
    print("=" * 80)

    # Load & prepare data
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
    spy_returns = spy["close"].pct_change(fill_method=None).dropna().values

    pivot = prices.pivot(index="timestamp", columns="symbol", values="close")
    cov_matrix = pivot.pct_change(fill_method=None).dropna().cov() * 252

    try:
        cost_df = get_per_symbol_costs(prices)
        cost_bps = dict(zip(cost_df["symbol"], cost_df["one_way_cost_bps"]))
    except Exception:
        cost_bps = {}

    liq_scores = compute_liquidity_scores(prices)

    # === DEFINE EXPERIMENTS ===
    experiments = [
        # Baseline: equal weight, no optimizer
        ExperimentConfig(
            name="EW_baseline",
            use_optimizer=False,
            use_liquidity=False,
            use_trailing_stops=False,
        ),
        # --- Rebalance frequency ---
        ExperimentConfig(name="rebal_weekly", rebal_every_n_days=5),
        ExperimentConfig(name="rebal_biweekly", rebal_every_n_days=10),
        ExperimentConfig(name="rebal_monthly", rebal_every_n_days=20),
        ExperimentConfig(name="rebal_quarterly", rebal_every_n_days=63),
        # --- Position count ---
        ExperimentConfig(name="top5", n_positions=5),
        ExperimentConfig(name="top10", n_positions=10),
        ExperimentConfig(name="top15", n_positions=15),
        ExperimentConfig(name="top20", n_positions=20),
        # --- MA windows ---
        ExperimentConfig(name="MA_10_30", ma_fast=10, ma_slow=30),
        ExperimentConfig(name="MA_20_50", ma_fast=20, ma_slow=50),
        ExperimentConfig(name="MA_20_200", ma_fast=20, ma_slow=200),
        ExperimentConfig(name="MA_50_200", ma_fast=50, ma_slow=200),
        # --- Risk aversion ---
        ExperimentConfig(name="risk_low", risk_aversion=0.5),
        ExperimentConfig(name="risk_mid", risk_aversion=1.5),
        ExperimentConfig(name="risk_high", risk_aversion=3.0),
        ExperimentConfig(name="risk_ultra", risk_aversion=5.0),
        # --- Turnover penalty ---
        ExperimentConfig(name="to_pen_0", turnover_penalty=0.0),
        ExperimentConfig(name="to_pen_low", turnover_penalty=0.002),
        ExperimentConfig(name="to_pen_mid", turnover_penalty=0.005),
        ExperimentConfig(name="to_pen_high", turnover_penalty=0.02),
        # --- Max weight (concentration) ---
        ExperimentConfig(name="maxw_5pct", max_weight=0.05),
        ExperimentConfig(name="maxw_10pct", max_weight=0.10),
        ExperimentConfig(name="maxw_15pct", max_weight=0.15),
        ExperimentConfig(name="maxw_25pct", max_weight=0.25),
        # --- Trailing stop multiplier ---
        ExperimentConfig(name="stop_tight", trailing_stop_mult=1.0),
        ExperimentConfig(name="stop_normal", trailing_stop_mult=2.0),
        ExperimentConfig(name="stop_wide", trailing_stop_mult=3.0),
        ExperimentConfig(name="stop_off", use_trailing_stops=False),
        # --- Vol cap ---
        ExperimentConfig(name="volcap_8pct", vol_cap=0.08),
        ExperimentConfig(name="volcap_12pct", vol_cap=0.12),
        ExperimentConfig(name="volcap_20pct", vol_cap=0.20),
        ExperimentConfig(name="volcap_none", vol_cap=None),
        # --- Best-of combinations ---
        ExperimentConfig(
            name="BEST_conservative",
            ma_fast=20,
            ma_slow=50,
            rebal_every_n_days=20,
            n_positions=10,
            risk_aversion=3.0,
            turnover_penalty=0.01,
            max_weight=0.10,
            trailing_stop_mult=1.5,
            vol_cap=0.12,
        ),
        ExperimentConfig(
            name="BEST_balanced",
            ma_fast=20,
            ma_slow=50,
            rebal_every_n_days=10,
            n_positions=12,
            risk_aversion=1.5,
            turnover_penalty=0.005,
            max_weight=0.12,
            trailing_stop_mult=2.0,
            vol_cap=0.15,
        ),
        ExperimentConfig(
            name="BEST_aggressive",
            ma_fast=10,
            ma_slow=30,
            rebal_every_n_days=5,
            n_positions=8,
            risk_aversion=0.5,
            turnover_penalty=0.001,
            max_weight=0.20,
            trailing_stop_mult=3.0,
            vol_cap=None,
        ),
        ExperimentConfig(
            name="BEST_minvol",
            ma_fast=50,
            ma_slow=200,
            rebal_every_n_days=63,
            n_positions=15,
            risk_aversion=5.0,
            turnover_penalty=0.02,
            max_weight=0.07,
            trailing_stop_mult=1.5,
            vol_cap=0.08,
        ),
    ]

    # === RUN ALL ===
    results: list[ExperimentResult] = []
    for i, exp in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] Running: {exp.name}...", end=" ", flush=True)
        try:
            r = run_experiment(
                exp, prices, prices_feat, spy_returns, cov_matrix, cost_bps, liq_scores
            )
            results.append(r)
            print(
                f"Return={r.total_return*100:+.1f}% Sharpe={r.sharpe:.2f} "
                f"DD={r.max_drawdown*100:.1f}% TO={r.turnover:.1f}x ({r.runtime_s:.1f}s)"
            )
        except Exception as e:
            print(f"FAILED: {e}")

    # === RESULTS TABLE ===
    print(f"\n\n{'=' * 120}")
    print(f"EXPERIMENT RESULTS SUMMARY ({len(results)} configurations)")
    print(f"{'=' * 120}")

    header = (
        f"{'Name':<22} {'Return':>8} {'Sharpe':>7} {'Sortino':>8} "
        f"{'MaxDD':>7} {'Vol':>7} {'Alpha':>7} {'Beta':>6} "
        f"{'TO':>6} {'Cost':>6} {'CPCV_S':>7} {'Ofit':>5}"
    )
    print(header)
    print("-" * 120)

    # Sort by Sharpe descending
    results.sort(key=lambda r: r.sharpe, reverse=True)

    for r in results:
        ofit = "YES" if r.cpcv_overfit else "no"
        line = (
            f"{r.name:<22} {r.total_return*100:>+7.1f}% {r.sharpe:>7.2f} {r.sortino:>8.2f} "
            f"{r.max_drawdown*100:>6.1f}% {r.volatility*100:>6.1f}% {r.alpha*100:>+6.1f}% {r.beta:>5.2f} "
            f"{r.turnover:>5.1f}x {r.cost_bps:>5.0f}bp {r.cpcv_mean_sharpe:>7.2f} {ofit:>5}"
        )
        print(line)

    # === CATEGORY WINNERS ===
    print(f"\n{'=' * 80}")
    print("CATEGORY WINNERS")
    print(f"{'=' * 80}")

    categories = [
        ("Highest Sharpe", max, lambda r: r.sharpe),
        ("Highest Return", max, lambda r: r.total_return),
        ("Lowest Drawdown", min, lambda r: abs(r.max_drawdown)),
        ("Lowest Volatility", min, lambda r: r.volatility),
        ("Highest Alpha", max, lambda r: r.alpha),
        ("Lowest Turnover", min, lambda r: r.turnover if r.turnover > 0 else 999),
        ("Best Calmar", max, lambda r: r.calmar),
        (
            "Best Risk-Adj (Sharpe/Vol)",
            max,
            lambda r: r.sharpe / max(r.volatility, 0.01),
        ),
    ]
    for cat_name, func, key_fn in categories:
        winner = func(results, key=key_fn)
        print(
            f"  {cat_name:<30} -> {winner.name:<22} "
            f"(Ret={winner.total_return*100:+.1f}%, Sharpe={winner.sharpe:.2f}, DD={winner.max_drawdown*100:.1f}%)"
        )

    # === INSIGHTS ===
    print(f"\n{'=' * 80}")
    print("KEY INSIGHTS")
    print(f"{'=' * 80}")

    # Rebalance frequency effect
    rebal_exps = [r for r in results if r.name.startswith("rebal_")]
    if rebal_exps:
        best_rebal = max(rebal_exps, key=lambda r: r.sharpe)
        print(
            f"  Rebalancing: Best frequency = {best_rebal.name} (Sharpe {best_rebal.sharpe:.2f})"
        )

    # Position count effect
    pos_exps = [r for r in results if r.name.startswith("top")]
    if pos_exps:
        best_pos = max(pos_exps, key=lambda r: r.sharpe)
        print(
            f"  Positions: Best count = {best_pos.name} (Sharpe {best_pos.sharpe:.2f})"
        )

    # MA window effect
    ma_exps = [r for r in results if r.name.startswith("MA_")]
    if ma_exps:
        best_ma = max(ma_exps, key=lambda r: r.sharpe)
        print(
            f"  MA Windows: Best combo = {best_ma.name} (Sharpe {best_ma.sharpe:.2f})"
        )

    # Risk aversion effect
    risk_exps = [r for r in results if r.name.startswith("risk_")]
    if risk_exps:
        best_risk = max(risk_exps, key=lambda r: r.sharpe)
        print(
            f"  Risk Aversion: Best = {best_risk.name} (Sharpe {best_risk.sharpe:.2f})"
        )

    # Trailing stops effect
    stop_exps = [r for r in results if r.name.startswith("stop_")]
    if stop_exps:
        best_stop = max(stop_exps, key=lambda r: r.sharpe)
        print(
            f"  Trailing Stops: Best = {best_stop.name} (Sharpe {best_stop.sharpe:.2f})"
        )

    # Overall best
    overall_best = max(results, key=lambda r: r.sharpe)
    print(f"\n  OVERALL BEST: {overall_best.name}")
    print(f"    Return: {overall_best.total_return*100:+.1f}%")
    print(f"    Sharpe: {overall_best.sharpe:.3f}")
    print(f"    MaxDD:  {overall_best.max_drawdown*100:.1f}%")
    print(f"    Alpha:  {overall_best.alpha*100:+.1f}%")
    print(
        f"    CPCV:   {overall_best.cpcv_mean_sharpe:.2f} (overfit={overall_best.cpcv_overfit})"
    )

    # Save all results
    os.makedirs("output/experiments", exist_ok=True)
    rows = []
    for r in results:
        rows.append(
            {
                "name": r.name,
                "total_return": r.total_return,
                "sharpe": r.sharpe,
                "sortino": r.sortino,
                "calmar": r.calmar,
                "cagr": r.cagr,
                "volatility": r.volatility,
                "max_drawdown": r.max_drawdown,
                "alpha": r.alpha,
                "beta": r.beta,
                "turnover": r.turnover,
                "n_rebalances": r.n_rebalances,
                "cost_bps": r.cost_bps,
                "cpcv_mean_sharpe": r.cpcv_mean_sharpe,
                "cpcv_overfit": r.cpcv_overfit,
                "runtime_s": r.runtime_s,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv("output/experiments/all_results.csv", index=False)
    print("\nFull results saved to output/experiments/all_results.csv")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
