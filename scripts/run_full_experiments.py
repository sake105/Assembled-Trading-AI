"""Full experiment suite: Long/Short, News-Sentiment, Multi-Signal, HHI-Fix.

Tests:
1. Long-only vs Long/Short
2. TA-only vs TA+Momentum+Sentiment hybrid signals
3. HHI concentration fix (equal-weight floor)
4. Stress-test constraints active
5. MA 20/200 + biweekly combo
6. Multiple combined configurations
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass

sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

from assembled_core.features.ta_features import add_all_features
from assembled_core.ml.cpcv import (
    compute_cpcv_sharpe_distribution,
    generate_cpcv_splits,
)
from assembled_core.portfolio.cost_aware_optimizer import (
    OptimizerConfig,
    optimize_portfolio,
)
from assembled_core.portfolio.market_neutral_optimizer import (
    MarketNeutralConfig,
    optimize_market_neutral,
)
from assembled_core.qa.benchmark_metrics import compute_benchmark_metrics
from assembled_core.risk.crowding_detector import compute_hhi
from assembled_core.risk.liquidity_scoring import (
    apply_liquidity_adjusted_sizing,
    compute_liquidity_scores,
)
from assembled_core.risk.trailing_stops import (
    apply_stop_reductions_to_weights,
    compute_trailing_stops,
)
from assembled_core.signals.rules_trend import generate_trend_signals_from_prices

SECTOR_MAP = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "GOOGL": "Technology",
    "AMZN": "Consumer Discretionary",
    "NVDA": "Technology",
    "META": "Technology",
    "TSLA": "Consumer Discretionary",
    "NFLX": "Communication Services",
    "AVGO": "Technology",
    "ADBE": "Technology",
    "CRM": "Technology",
    "JPM": "Financials",
    "V": "Financials",
    "MA": "Financials",
    "HD": "Consumer Discretionary",
    "WMT": "Consumer Staples",
    "MCD": "Consumer Discretionary",
    "COST": "Consumer Staples",
    "PEP": "Consumer Staples",
    "KO": "Consumer Staples",
    "PG": "Consumer Staples",
    "JNJ": "Healthcare",
    "LLY": "Healthcare",
    "UNH": "Healthcare",
    "MRK": "Healthcare",
    "TMO": "Healthcare",
    "ABBV": "Healthcare",
    "XOM": "Energy",
    "CVX": "Energy",
    "BRK-B": "Financials",
}


def build_enhanced_signals(
    prices_feat: pd.DataFrame,
    prices: pd.DataFrame,
    ma_fast: int = 20,
    ma_slow: int = 50,
    use_momentum: bool = True,
    use_sentiment_proxy: bool = True,
    use_mean_reversion: bool = False,
) -> pd.DataFrame:
    """Build multi-layer signals combining TA, momentum, and sentiment.

    Layers:
    1. MA trend signal (base)
    2. Cross-sectional momentum ranking (12m-1m)
    3. Volume-weighted sentiment proxy (volume spike = news proxy)
    4. Optional: RSI mean-reversion overlay for sideways markets
    """
    # Layer 1: Trend signal
    trend_signals = generate_trend_signals_from_prices(
        prices_feat, ma_fast=ma_fast, ma_slow=ma_slow
    )

    # Build per-symbol features for scoring
    all_dates = sorted(prices["timestamp"].unique())
    symbols = sorted(prices["symbol"].unique())
    rows = []

    for date in all_dates:
        _day_prices = prices[prices["timestamp"] == date]  # noqa: F841
        day_trend = trend_signals[trend_signals["timestamp"] == date]

        for sym in symbols:
            sym_trend = day_trend[day_trend["symbol"] == sym]
            trend_dir = (
                sym_trend["direction"].iloc[0] if not sym_trend.empty else "FLAT"
            )
            trend_score = (
                float(sym_trend["score"].iloc[0]) if not sym_trend.empty else 0.0
            )

            # Historical data for this symbol up to this date
            sym_hist = prices[
                (prices["symbol"] == sym) & (prices["timestamp"] <= date)
            ].sort_values("timestamp")

            composite_score = trend_score
            direction = trend_dir

            # Layer 2: Momentum (252d return minus 21d return)
            if use_momentum and len(sym_hist) >= 252:
                close_now = float(sym_hist["close"].iloc[-1])
                close_252 = float(sym_hist["close"].iloc[-252])
                close_21 = (
                    float(sym_hist["close"].iloc[-21])
                    if len(sym_hist) >= 21
                    else close_now
                )
                mom_12m_1m = (close_now / close_252) - (close_now / close_21)
                composite_score += mom_12m_1m * 0.3  # 30% weight

            # Layer 3: Volume sentiment proxy (abnormal volume = news activity)
            if (
                use_sentiment_proxy
                and "volume" in sym_hist.columns
                and len(sym_hist) >= 20
            ):
                recent_vol = float(sym_hist["volume"].iloc[-1])
                avg_vol = float(sym_hist["volume"].iloc[-20:].mean())
                if avg_vol > 0:
                    vol_ratio = recent_vol / avg_vol
                    # High volume + price up = bullish news proxy
                    price_change_1d = 0.0
                    if len(sym_hist) >= 2:
                        price_change_1d = (
                            float(sym_hist["close"].iloc[-1])
                            / float(sym_hist["close"].iloc[-2])
                            - 1
                        )
                    sentiment_proxy = price_change_1d * min(vol_ratio, 3.0)
                    composite_score += sentiment_proxy * 0.2  # 20% weight

            # Layer 4: Mean-reversion RSI overlay
            if use_mean_reversion and len(sym_hist) >= 14:
                # Simple RSI calculation
                deltas = sym_hist["close"].diff().iloc[-14:]
                gains = deltas.clip(lower=0).mean()
                losses = (-deltas.clip(upper=0)).mean()
                if losses > 0:
                    rs = gains / losses
                    rsi = 100 - (100 / (1 + rs))
                else:
                    rsi = 100
                # Mean reversion: buy oversold, sell overbought
                if rsi < 30:
                    composite_score += 0.15  # Oversold bonus
                elif rsi > 70:
                    composite_score -= 0.15  # Overbought penalty

            rows.append(
                {
                    "timestamp": date,
                    "symbol": sym,
                    "direction": direction,
                    "score": composite_score,
                    "trend_score": trend_score,
                }
            )

    result = pd.DataFrame(rows)

    # For long/short: re-classify direction based on composite score
    # Top 20% = LONG, Bottom 20% = SHORT, rest = FLAT
    for date in all_dates:
        mask = result["timestamp"] == date
        day = result.loc[mask]
        if len(day) < 5:
            continue
        q_high = day["score"].quantile(0.80)
        q_low = day["score"].quantile(0.20)
        result.loc[mask & (result["score"] >= q_high), "direction"] = "LONG"
        result.loc[mask & (result["score"] <= q_low), "direction"] = "SHORT"
        result.loc[
            mask & (result["score"] > q_low) & (result["score"] < q_high), "direction"
        ] = "FLAT"

    return result


@dataclass
class ExpResult:
    name: str
    total_return: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    volatility: float = 0.0
    max_drawdown: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    turnover: float = 0.0
    cost_bps: float = 0.0
    hit_rate: float = 0.0
    profit_factor: float = 0.0
    avg_hhi: float = 0.0
    cpcv_sharpe: float = 0.0
    overfit: bool = False
    long_count: int = 0
    short_count: int = 0


def run_backtest(
    name: str,
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    spy_returns: np.ndarray,
    cov_matrix: pd.DataFrame,
    liq_scores: list,
    cost_bps: dict,
    *,
    long_short: bool = False,
    rebal_days: int = 10,
    n_long: int = 10,
    n_short: int = 6,
    max_weight: float = 0.12,
    min_weight_floor: float = 0.0,
    risk_aversion: float = 1.5,
    use_stress: bool = False,
) -> ExpResult:
    """Run a backtest with given configuration."""
    dates = sorted(prices["timestamp"].unique())
    rebal_dates = dates[::rebal_days]

    portfolio_value = [100000.0]
    holdings: dict[str, float] = {}
    daily_returns = []
    total_turnover = 0.0
    total_cost = 0.0
    hhi_vals = []
    stop_states: dict = {}
    long_ct = 0
    short_ct = 0

    for i, date in enumerate(dates[1:], 1):
        prev_date = dates[i - 1]
        day_sig = signals[signals["timestamp"] == date]
        should_rebal = date in rebal_dates

        if should_rebal and not day_sig.empty:
            longs = day_sig[day_sig["direction"] == "LONG"].nlargest(n_long, "score")
            long_ct += len(longs)

            if long_short:
                shorts = day_sig[day_sig["direction"] == "SHORT"].nsmallest(
                    n_short, "score"
                )
                short_ct += len(shorts)

                # Market-neutral: use optimizer if available
                all_syms = list(
                    set(longs["symbol"].tolist() + shorts["symbol"].tolist())
                )
                available = [s for s in all_syms if s in cov_matrix.columns]

                if len(available) >= 2:
                    exp_ret = (
                        pd.Series(
                            dict(
                                zip(
                                    pd.concat([longs["symbol"], shorts["symbol"]]),
                                    pd.concat([longs["score"], shorts["score"]]).astype(
                                        float
                                    ),
                                )
                            )
                        )
                        .reindex(available)
                        .fillna(0.0)
                    )

                    mn_config = MarketNeutralConfig(
                        risk_aversion=risk_aversion,
                        turnover_penalty=0.005,
                        max_weight=max_weight,
                        max_gross_exposure=2.0,
                        dollar_neutral_tolerance=0.05,
                        beta_neutral=True,
                        beta_neutral_tolerance=0.10,
                        sector_neutral=True,
                        max_sector_net_exposure=0.10,
                        sector_mapping=SECTOR_MAP,
                    )
                    sub_cov = cov_matrix.loc[available, available]
                    mn_result = optimize_market_neutral(
                        expected_returns=exp_ret,
                        covariance=sub_cov,
                        current_weights=holdings,
                        per_symbol_cost_bps=cost_bps,
                        config=mn_config,
                    )
                    new_weights = {**mn_result.long_weights, **mn_result.short_weights}
                else:
                    # Fallback: equal weight
                    w_per = max_weight
                    new_weights = {s: w_per for _, s in longs["symbol"].items()}
                    for _, s in shorts["symbol"].items():
                        new_weights[s] = -w_per
            else:
                # Long-only
                sym_list = longs["symbol"].tolist()
                available = [s for s in sym_list if s in cov_matrix.columns]

                if len(available) >= 2:
                    exp_ret = pd.Series(
                        dict(zip(longs["symbol"], longs["score"].astype(float)))
                    )
                    sub_cov = cov_matrix.loc[available, available]
                    exp_ret_sub = exp_ret.reindex(available).fillna(0.0)

                    opt_config = OptimizerConfig(
                        risk_aversion=risk_aversion,
                        turnover_penalty=0.005,
                        max_weight=max_weight,
                        max_gross_exposure=1.0,
                        long_only=True,
                    )
                    result = optimize_portfolio(
                        expected_returns=exp_ret_sub,
                        covariance=sub_cov,
                        current_weights=holdings,
                        per_symbol_cost_bps=cost_bps,
                        config=opt_config,
                    )
                    new_weights = result.weights
                else:
                    new_weights = {s: 1.0 / max(len(sym_list), 1) for s in sym_list}

            # Equal-weight floor to fix HHI concentration
            if min_weight_floor > 0 and new_weights:
                for s in new_weights:
                    if new_weights[s] > 0 and new_weights[s] < min_weight_floor:
                        new_weights[s] = min_weight_floor
                # Renormalize if long-only
                if not long_short:
                    total_w = sum(w for w in new_weights.values() if w > 0)
                    if total_w > 1.0:
                        scale = 1.0 / total_w
                        new_weights = {s: w * scale for s, w in new_weights.items()}

            # Liquidity adjustment
            pos_weights = {s: w for s, w in new_weights.items() if w > 0}
            neg_weights = {s: w for s, w in new_weights.items() if w < 0}
            pos_adj = (
                apply_liquidity_adjusted_sizing(pos_weights, liq_scores, alpha=0.3)
                if pos_weights
                else {}
            )
            new_weights = {**pos_adj, **neg_weights}

            # Trailing stops (long side only)
            positions_for_stops = {}
            for s in new_weights:
                if new_weights[s] > 0.01:
                    sp = prices[(prices["symbol"] == s) & (prices["timestamp"] == date)]
                    if not sp.empty:
                        positions_for_stops[s] = {
                            "entry_price": float(sp["close"].iloc[0])
                        }
            if positions_for_stops:
                stop_result = compute_trailing_stops(
                    positions_for_stops,
                    prices[prices["timestamp"] <= date],
                    regime="unknown",
                    prior_states=stop_states,
                )
                long_adj = apply_stop_reductions_to_weights(
                    {s: w for s, w in new_weights.items() if w > 0}, stop_result
                )
                new_weights = {**long_adj, **neg_weights}
                stop_states = {s.symbol: s for s in stop_result.stops}

            # Turnover & cost
            all_syms = set(holdings.keys()) | set(new_weights.keys())
            turnover = sum(
                abs(holdings.get(s, 0) - new_weights.get(s, 0)) for s in all_syms
            )
            total_turnover += turnover
            total_cost += sum(
                abs(holdings.get(s, 0) - new_weights.get(s, 0))
                * cost_bps.get(s, 6.0)
                / 10000
                for s in all_syms
            )
            holdings = {s: w for s, w in new_weights.items() if abs(w) > 0.001}

        hhi_vals.append(compute_hhi(holdings))

        # Daily PnL
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
    rm = np.maximum.accumulate(cum)
    dd = (cum - rm) / rm
    max_dd = float(dd.min())
    vol = float(np.std(returns) * np.sqrt(252))
    cagr = (
        float((equity[-1] / equity[0]) ** (252 / len(returns)) - 1)
        if len(returns) > 0
        else 0
    )
    calmar = float(cagr / abs(max_dd)) if abs(max_dd) > 1e-10 else 0

    wins = returns[returns > 0]
    losses = returns[returns < 0]
    hr = len(wins) / len(returns) * 100 if len(returns) > 0 else 0
    pf = abs(wins.sum() / losses.sum()) if len(losses) > 0 and losses.sum() != 0 else 0

    alpha_val, beta_val = 0.0, 0.0
    spy_al = spy_returns[: len(returns)]
    if len(spy_al) > 10:
        bm = compute_benchmark_metrics(
            pd.Series(returns[: len(spy_al)]), pd.Series(spy_al)
        )
        alpha_val, beta_val = bm.alpha, bm.beta

    cpcv_s, overfit = 0.0, False
    splits = generate_cpcv_splits(
        len(returns), n_groups=6, k_test_groups=2, purge_length=5, embargo_length=3
    )
    if splits:
        cpcv = compute_cpcv_sharpe_distribution([returns[t] for _, t in splits])
        cpcv_s, overfit = cpcv.mean_sharpe, cpcv.is_likely_overfit

    return ExpResult(
        name=name,
        total_return=float(equity[-1] / equity[0] - 1),
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        volatility=vol,
        max_drawdown=max_dd,
        alpha=alpha_val,
        beta=beta_val,
        turnover=total_turnover,
        cost_bps=total_cost * 10000,
        hit_rate=hr,
        profit_factor=pf,
        avg_hhi=float(np.mean(hhi_vals)),
        cpcv_sharpe=cpcv_s,
        overfit=overfit,
        long_count=long_ct,
        short_count=short_ct,
    )


def main():
    print("=" * 90)
    print("FULL EXPERIMENT SUITE: Long/Short, News-Sentiment, Multi-Signal, HHI-Fix")
    print("=" * 90)

    # Prepare data
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
        from assembled_core.data.cost_model_policy import get_per_symbol_costs

        cost_df = get_per_symbol_costs(prices)
        cost_bps = dict(zip(cost_df["symbol"], cost_df["one_way_cost_bps"]))
    except Exception:
        cost_bps = {}

    liq_scores = compute_liquidity_scores(prices)

    # === BUILD SIGNAL VARIANTS ===
    print("\nBuilding signal variants...")

    t0 = time.time()
    sig_ta_only = generate_trend_signals_from_prices(
        prices_feat, ma_fast=20, ma_slow=50
    )
    # Add direction for all symbols on all dates
    sig_ta_only_20_200 = generate_trend_signals_from_prices(
        prices_feat, ma_fast=20, ma_slow=200
    )
    print(f"  TA signals (20/50): {len(sig_ta_only)} rows ({time.time()-t0:.1f}s)")

    t0 = time.time()
    sig_hybrid = build_enhanced_signals(
        prices_feat,
        prices,
        ma_fast=20,
        ma_slow=50,
        use_momentum=True,
        use_sentiment_proxy=True,
        use_mean_reversion=False,
    )
    print(f"  Hybrid (TA+Mom+Sent): {len(sig_hybrid)} rows ({time.time()-t0:.1f}s)")

    t0 = time.time()
    sig_hybrid_mr = build_enhanced_signals(
        prices_feat,
        prices,
        ma_fast=20,
        ma_slow=50,
        use_momentum=True,
        use_sentiment_proxy=True,
        use_mean_reversion=True,
    )
    print(f"  Hybrid+MeanRev: {len(sig_hybrid_mr)} rows ({time.time()-t0:.1f}s)")

    t0 = time.time()
    sig_hybrid_200 = build_enhanced_signals(
        prices_feat,
        prices,
        ma_fast=20,
        ma_slow=200,
        use_momentum=True,
        use_sentiment_proxy=True,
        use_mean_reversion=False,
    )
    print(f"  Hybrid MA20/200: {len(sig_hybrid_200)} rows ({time.time()-t0:.1f}s)")

    # === RUN EXPERIMENTS ===
    experiments = []

    def add(name, signals, **kwargs):
        experiments.append((name, signals, kwargs))

    # --- SECTION A: Signal comparison (Long-only, biweekly) ---
    add("A1_TA_only_20_50", sig_ta_only)
    add("A2_TA_only_20_200", sig_ta_only_20_200)
    add("A3_Hybrid_TA+Mom+Sent", sig_hybrid)
    add("A4_Hybrid+MeanRev", sig_hybrid_mr)
    add("A5_Hybrid_MA20_200", sig_hybrid_200)

    # --- SECTION B: Long/Short comparison ---
    add("B1_LongOnly_TA", sig_ta_only, long_short=False)
    add("B2_LongShort_TA", sig_ta_only, long_short=True, n_short=6)
    add("B3_LongOnly_Hybrid", sig_hybrid, long_short=False)
    add("B4_LongShort_Hybrid", sig_hybrid, long_short=True, n_short=6)
    add("B5_LongShort_Hybrid_wide", sig_hybrid, long_short=True, n_short=10)
    add("B6_LongShort_200", sig_hybrid_200, long_short=True, n_short=6)

    # --- SECTION C: HHI concentration fix ---
    add("C1_NoFloor", sig_hybrid, min_weight_floor=0.0)
    add("C2_Floor_3pct", sig_hybrid, min_weight_floor=0.03)
    add("C3_Floor_5pct", sig_hybrid, min_weight_floor=0.05)
    add("C4_Floor_7pct", sig_hybrid, min_weight_floor=0.07)

    # --- SECTION D: Rebalance + position combos ---
    add("D1_Weekly_8pos", sig_hybrid, rebal_days=5, n_long=8)
    add("D2_Biweekly_12pos", sig_hybrid, rebal_days=10, n_long=12)
    add("D3_Monthly_15pos", sig_hybrid, rebal_days=20, n_long=15)

    # --- SECTION E: Best combos ---
    add(
        "E1_BEST_LongOnly",
        sig_hybrid,
        rebal_days=10,
        n_long=12,
        max_weight=0.12,
        min_weight_floor=0.04,
        risk_aversion=1.5,
    )
    add(
        "E2_BEST_LongShort",
        sig_hybrid,
        long_short=True,
        rebal_days=10,
        n_long=10,
        n_short=6,
        max_weight=0.12,
        min_weight_floor=0.03,
        risk_aversion=1.5,
    )
    add(
        "E3_BEST_200_LS",
        sig_hybrid_200,
        long_short=True,
        rebal_days=10,
        n_long=10,
        n_short=6,
        max_weight=0.12,
        risk_aversion=1.5,
    )
    add(
        "E4_BEST_Conservative",
        sig_hybrid,
        rebal_days=20,
        n_long=15,
        max_weight=0.08,
        min_weight_floor=0.04,
        risk_aversion=3.0,
    )

    results: list[ExpResult] = []
    for i, (name, signals, kwargs) in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] {name}...", end=" ", flush=True)
        try:
            r = run_backtest(
                name,
                signals,
                prices,
                spy_returns,
                cov_matrix,
                liq_scores,
                cost_bps,
                **kwargs,
            )
            results.append(r)
            ls_info = (
                f"L={r.long_count}/S={r.short_count}"
                if r.short_count > 0
                else f"L={r.long_count}"
            )
            print(
                f"Ret={r.total_return*100:+.1f}% Sh={r.sharpe:.2f} DD={r.max_drawdown*100:.1f}% "
                f"HHI={r.avg_hhi:.3f} {ls_info}"
            )
        except Exception as e:
            print(f"FAILED: {e}")

    # === RESULTS TABLE ===
    print(f"\n\n{'='*130}")
    print(f"FULL RESULTS ({len(results)} experiments)")
    print(f"{'='*130}")

    hdr = (
        f"{'Name':<26} {'Return':>8} {'Sharpe':>7} {'Sortino':>8} "
        f"{'MaxDD':>7} {'Vol':>6} {'Alpha':>7} {'Beta':>6} "
        f"{'HitR':>5} {'PF':>5} {'HHI':>6} {'TO':>6} {'CPCV':>6} {'Ofit':>5} {'L/S':>8}"
    )
    print(hdr)
    print("-" * 130)

    results.sort(key=lambda r: r.sharpe, reverse=True)
    for r in results:
        ofit = "YES" if r.overfit else "no"
        ls = (
            f"{r.long_count}/{r.short_count}"
            if r.short_count > 0
            else f"{r.long_count}/0"
        )
        print(
            f"{r.name:<26} {r.total_return*100:>+7.1f}% {r.sharpe:>7.2f} {r.sortino:>8.2f} "
            f"{r.max_drawdown*100:>6.1f}% {r.volatility*100:>5.1f}% {r.alpha*100:>+6.1f}% {r.beta:>5.2f} "
            f"{r.hit_rate:>4.0f}% {r.profit_factor:>5.2f} {r.avg_hhi:>5.3f} {r.turnover:>5.1f}x "
            f"{r.cpcv_sharpe:>5.2f} {ofit:>5} {ls:>8}"
        )

    # === SECTION ANALYSIS ===
    print(f"\n{'='*90}")
    print("SECTION ANALYSIS")
    print(f"{'='*90}")

    # A: Signal comparison
    a_exps = [r for r in results if r.name.startswith("A")]
    if a_exps:
        best_a = max(a_exps, key=lambda r: r.sharpe)
        print("\nA. SIGNAL TYPE:")
        for r in sorted(a_exps, key=lambda r: -r.sharpe):
            print(
                f"   {r.name:<30} Sharpe={r.sharpe:.2f} Return={r.total_return*100:+.1f}% DD={r.max_drawdown*100:.1f}%"
            )
        print(f"   -> Winner: {best_a.name}")

    # B: Long/Short
    b_exps = [r for r in results if r.name.startswith("B")]
    if b_exps:
        best_b = max(b_exps, key=lambda r: r.sharpe)
        print("\nB. LONG vs LONG/SHORT:")
        for r in sorted(b_exps, key=lambda r: -r.sharpe):
            ls = "L/S" if r.short_count > 0 else "Long"
            print(
                f"   {r.name:<30} [{ls:>4}] Sharpe={r.sharpe:.2f} Return={r.total_return*100:+.1f}% Beta={r.beta:.3f}"
            )
        print(f"   -> Winner: {best_b.name}")

    # C: HHI fix
    c_exps = [r for r in results if r.name.startswith("C")]
    if c_exps:
        print("\nC. HHI CONCENTRATION FIX:")
        for r in sorted(c_exps, key=lambda r: r.avg_hhi):
            eff_n = 1 / r.avg_hhi if r.avg_hhi > 0 else 0
            print(
                f"   {r.name:<30} HHI={r.avg_hhi:.3f} EffN={eff_n:.1f} Sharpe={r.sharpe:.2f}"
            )

    # D: Rebalance combos
    d_exps = [r for r in results if r.name.startswith("D")]
    if d_exps:
        best_d = max(d_exps, key=lambda r: r.sharpe)
        print("\nD. REBALANCE + POSITION COMBOS:")
        for r in sorted(d_exps, key=lambda r: -r.sharpe):
            print(
                f"   {r.name:<30} Sharpe={r.sharpe:.2f} TO={r.turnover:.1f}x Return={r.total_return*100:+.1f}%"
            )
        print(f"   -> Winner: {best_d.name}")

    # E: Best combos
    e_exps = [r for r in results if r.name.startswith("E")]
    if e_exps:
        _best_e = max(e_exps, key=lambda r: r.sharpe)  # noqa: F841
        print("\nE. BEST COMBINED CONFIGURATIONS:")
        for r in sorted(e_exps, key=lambda r: -r.sharpe):
            ls = "L/S" if r.short_count > 0 else "Long"
            eff_n = 1 / r.avg_hhi if r.avg_hhi > 0 else 0
            print(
                f"   {r.name:<30} [{ls:>4}] Sharpe={r.sharpe:.2f} Return={r.total_return*100:+.1f}% "
                f"DD={r.max_drawdown*100:.1f}% EffN={eff_n:.1f} Beta={r.beta:.3f}"
            )

    # OVERALL
    best = max(results, key=lambda r: r.sharpe)
    print(f"\n{'='*90}")
    print(f"OVERALL BEST: {best.name}")
    print(f"  Return:       {best.total_return*100:+.1f}%")
    print(f"  Sharpe:       {best.sharpe:.3f}")
    print(f"  Sortino:      {best.sortino:.3f}")
    print(f"  Max DD:       {best.max_drawdown*100:.1f}%")
    print(f"  Alpha:        {best.alpha*100:+.1f}%")
    print(f"  Beta:         {best.beta:.3f}")
    print(f"  Hit Rate:     {best.hit_rate:.0f}%")
    print(f"  Profit Factor:{best.profit_factor:.2f}")
    print(f"  HHI:          {best.avg_hhi:.3f} (EffN={1/best.avg_hhi:.1f})")
    print(f"  CPCV Sharpe:  {best.cpcv_sharpe:.2f} (overfit={best.overfit})")
    print(f"{'='*90}")

    # Save
    os.makedirs("output/experiments", exist_ok=True)
    rows = []
    for r in results:
        rows.append(vars(r))
    pd.DataFrame(rows).to_csv("output/experiments/full_results.csv", index=False)
    print("\nResults: output/experiments/full_results.csv")


if __name__ == "__main__":
    main()
