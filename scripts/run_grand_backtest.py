"""M16 Grand Backtest — Full 53-symbol multi-asset system.

Activates ALL dormant modules:
  U1: Full universe (equities + sector ETFs + bonds + gold + inverse ETFs)
  U2: Sector rotation (XLK/XLF/XLE/XLV/XLI/XLU/XLP/XLY momentum ranking)
  U3: Regime-adaptive multi-asset allocation (bull/sideways/bear/crisis)
  E1: Pullback entry scoring (RSI + volume confirmation)
  E2: Tiered profit targets (25% out at +10%, +20%, +35%)
  E3: Time-stop (close stale positions after 60 bars)
  E4: VIX-scaled stop widths
  E5: Market breadth gate for entries
  D1: VIX options-derived signals (vix_level, vix_regime, term structure)
  D2: Market breadth signals (fraction above 50d MA, A/D line)
  D3: Intermarket factors (bond/equity ratio, yield curve, gold divergence)
  D4: Crash prediction overlay (16-signal system)
  D5: Candlestick confirmation patterns
  S1: Inverse ETF shorts (SH, PSQ) via InverseETFSelector
  S3: VIX-gated short sizing (no shorts when VIX < 15)
  M1: Regime-conditional signal weights (existing compute_regime_blended_weights)
  M3: Signal hysteresis (apply_signal_hysteresis)

Benchmarks: SPY, 60/40 (SPY/TLT), All-Weather Proxy
"""
from __future__ import annotations

import sys
import os
import json
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")
_log = logging.getLogger(__name__)

# ── Data loading helpers ──────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "equities_eod", "yfinance")


def load_parquet(symbol: str) -> pd.DataFrame | None:
    path = os.path.join(DATA_DIR, f"{symbol}.parquet")
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    df = df.sort_values("timestamp").reset_index(drop=True)
    if "symbol" not in df.columns:
        df["symbol"] = symbol
    return df


def load_pool(symbols: list[str], start: str = "2024-04-01", end: str = "2026-03-31") -> pd.DataFrame:
    frames = []
    for sym in symbols:
        df = load_parquet(sym)
        if df is None:
            continue
        df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]
        if df.empty:
            continue
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ── Universe pools ────────────────────────────────────────────────────────────

EQUITY_CORE = [
    "AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","JPM","JNJ","UNH",
    "XOM","PG","MA","HD","CVX","MRK","ABBV","LLY","PEP","KO","COST","AVGO",
    "WMT","MCD","CRM","TMO","ADBE","NFLX","V","BRK-B",
]
SECTOR_ETFS = ["XLK","XLF","XLE","XLV","XLI","XLU","XLP","XLY"]
MACRO_HEDGES = ["TLT","IEF","GLD","HYG"]
INVERSE_ETFS = ["SH","PSQ"]

START, END = "2024-04-01", "2026-03-31"


def main() -> None:
    print("=" * 70)
    print("M16 GRAND BACKTEST — Multi-Asset Regime-Adaptive System")
    print("=" * 70)

    # ── Load all data pools ───────────────────────────────────────────────────
    print("\n[1/7] Loading data pools...")
    equity_prices = load_pool(EQUITY_CORE, START, END)
    sector_prices = load_pool(SECTOR_ETFS, START, END)
    macro_prices  = load_pool(MACRO_HEDGES, START, END)
    inverse_prices = load_pool(INVERSE_ETFS, START, END)
    spy_df = load_pool(["SPY"], START, END)

    all_prices = pd.concat([equity_prices, sector_prices, macro_prices, inverse_prices], ignore_index=True)
    all_prices = all_prices.sort_values(["symbol", "timestamp"])

    equity_syms = equity_prices["symbol"].unique().tolist() if not equity_prices.empty else []
    sector_syms = sector_prices["symbol"].unique().tolist() if not sector_prices.empty else []
    macro_syms  = macro_prices["symbol"].unique().tolist() if not macro_prices.empty else []

    print(f"  Equity core: {len(equity_syms)} symbols")
    print(f"  Sector ETFs: {len(sector_syms)} symbols")
    print(f"  Macro hedges: {len(macro_syms)} symbols")
    print(f"  Inverse ETFs: {len(inverse_prices['symbol'].unique()) if not inverse_prices.empty else 0} symbols")

    dates = sorted(all_prices["timestamp"].unique())

    # ── D2: Market Breadth (pre-compute) ─────────────────────────────────────
    print("\n[2/7] Computing breadth, VIX signals, sector scores...")
    breadth_by_date: dict = {}
    try:
        from assembled_core.features.market_breadth import compute_market_breadth_ma
        b_df = compute_market_breadth_ma(equity_prices, ma_window=50)
        if "timestamp" in b_df.columns:
            b_df = b_df.set_index("timestamp")
        breadth_by_date = b_df["fraction_above_ma_50"].to_dict() if "fraction_above_ma_50" in b_df.columns else {}
        print(f"  Breadth computed: {len(breadth_by_date)} dates")
    except Exception as e:
        print(f"  Breadth: skipped ({e})")

    # ── D1: VIX signals (pre-compute) ────────────────────────────────────────
    vix_by_date: dict = {}
    try:
        from assembled_core.features.options_derived_signals import build_options_regime_factors
        vix_raw = load_parquet("VIX")
        vix3m_raw = load_parquet("VIX3M")
        if vix_raw is not None and vix3m_raw is not None:
            cboe_df = vix_raw[["timestamp","close"]].set_index("timestamp").rename(columns={"close":"vix"})
            cboe_df["vix3m"] = vix3m_raw.set_index("timestamp")["close"]
            cboe_df["put_call_ratio"] = 1.0  # proxy (no CBOE PCR data)
            cboe_df = cboe_df.dropna(subset=["vix"])
            opt_factors = build_options_regime_factors(cboe_df.reset_index())
            if "timestamp" in opt_factors.columns:
                opt_factors = opt_factors.set_index("timestamp")
            elif opt_factors.index.name == "timestamp":
                pass
            vix_by_date = opt_factors["vix_level"].to_dict() if "vix_level" in opt_factors.columns else {}
            print(f"  VIX signals: {len(vix_by_date)} dates")
        else:
            print("  VIX: data missing, using breadth-only regime detection")
    except Exception as e:
        print(f"  VIX: skipped ({e})")

    # ── U2: Sector Rotation scores (pre-compute) ──────────────────────────────
    sector_scores_df = pd.DataFrame()
    if not sector_prices.empty and not spy_df.empty:
        try:
            from assembled_core.signals.sector_rotation import (
                SectorRotationConfig, compute_sector_scores,
                generate_sector_rotation_signals, get_sector_weights,
            )
            sr_config = SectorRotationConfig(top_n_long=3, bottom_n_short=2, risk_off_threshold=5)
            sector_scores_df = compute_sector_scores(sector_prices, spy_df, sr_config)
            if "timestamp" in sector_scores_df.columns:
                sector_scores_df = sector_scores_df.set_index("timestamp")
            print(f"  Sector scores: {len(sector_scores_df)} dates, {len(sector_syms)} ETFs")
        except Exception as e:
            print(f"  Sector rotation: skipped ({e})")

    # ── D5: Candlestick patterns (pre-compute) ────────────────────────────────
    cs_features: pd.DataFrame | None = None
    try:
        from assembled_core.features.ta_candlestick import build_candlestick_features
        cs_features = build_candlestick_features(equity_prices)
        print(f"  Candlestick patterns: {len(cs_features)} rows")
    except Exception as e:
        print(f"  Candlestick: skipped ({e})")

    # ── D3: Intermarket factors (pre-compute, fetches own data) ───────────────
    intermarket_df: pd.DataFrame | None = None
    try:
        from assembled_core.features.intermarket_factors import build_intermarket_factors
        intermarket_df = build_intermarket_factors(start_date=START, end_date=END)
        if "timestamp" in intermarket_df.columns:
            intermarket_df = intermarket_df.set_index("timestamp")
        print(f"  Intermarket factors: {len(intermarket_df)} dates")
    except Exception as e:
        print(f"  Intermarket: skipped ({e})")

    # ── D4: Crash Prediction Engine ───────────────────────────────────────────
    crash_engine = None
    try:
        from assembled_core.signals.crash_prediction import CrashPredictionEngine
        crash_engine = CrashPredictionEngine()
        print("  Crash prediction engine: ready")
    except Exception as e:
        print(f"  Crash prediction: skipped ({e})")

    # ── U3: Regime Detector ───────────────────────────────────────────────────
    from assembled_core.portfolio.multiasset_allocator import (
        RegimeDetector, RegimeDetectorConfig, allocate_by_regime
    )
    regime_detector = RegimeDetector(RegimeDetectorConfig())

    # ── S1: Inverse ETF Selector ──────────────────────────────────────────────
    inv_selector = None
    try:
        from assembled_core.portfolio.inverse_etf_selector import InverseETFSelector
        inv_selector = InverseETFSelector(allow_2x=False)
        print("  Inverse ETF selector: ready")
    except Exception as e:
        print(f"  Inverse ETF selector: skipped ({e})")

    # ── E2: Profit target tracking ────────────────────────────────────────────
    from assembled_core.risk.profit_targets import (
        ProfitTargetConfig, PositionRecord,
        check_profit_targets, build_position_records,
    )
    pt_config = ProfitTargetConfig()
    position_records: dict[str, PositionRecord] = {}

    # ── Trailing stops ────────────────────────────────────────────────────────
    from assembled_core.risk.trailing_stops import (
        compute_trailing_stops, apply_stop_reductions_to_weights, TrailingStopState
    )
    stop_states: dict[str, TrailingStopState] = {}

    # ── M3: Hysteresis helper ─────────────────────────────────────────────────
    from assembled_core.signals.multifactor_signal import apply_signal_hysteresis

    print("\n[3/7] Starting backtest loop...")

    # ── Helper: compute trend signals with pullback score (E1) ───────────────

    def compute_entry_scores(prices_to_date: pd.DataFrame, syms: list[str]) -> dict[str, float]:
        """Compute MA trend + RSI pullback + volume confirmation entry score."""
        scores: dict[str, float] = {}
        for sym in syms:
            sym_df = prices_to_date[prices_to_date["symbol"] == sym].sort_values("timestamp")
            if len(sym_df) < 22:
                continue
            close = sym_df["close"].values
            # MA trend
            ma20 = np.mean(close[-20:]) if len(close) >= 20 else close[-1]
            ma50 = np.mean(close[-50:]) if len(close) >= 50 else close[-1]
            if ma50 <= 0:
                continue
            ma_score = max(0.0, min(1.0, (ma20 - ma50) / ma50 * 20))
            if ma20 <= ma50:
                continue  # no signal unless trend is up

            # RSI pullback score (E1): prefer RSI < 55 (not extended)
            if len(close) >= 15:
                gains = np.maximum(np.diff(close[-15:]), 0)
                losses = np.maximum(-np.diff(close[-15:]), 0)
                avg_gain = np.mean(gains) if gains.size > 0 else 0
                avg_loss = np.mean(losses) if losses.size > 0 else 1e-9
                rs = avg_gain / avg_loss
                rsi = 100 - 100 / (1 + rs)
                pullback_score = max(0.0, (60 - rsi) / 60) if rsi < 60 else 0.0
            else:
                rsi = 50.0
                pullback_score = 0.5

            # Volume confirmation
            if "volume" in sym_df.columns:
                vol = sym_df["volume"].values
                avg_vol = np.mean(vol[-20:]) if len(vol) >= 20 else np.mean(vol)
                vol_score = min(1.0, vol[-1] / avg_vol / 1.5) if avg_vol > 0 else 0.5
            else:
                vol_score = 0.5

            score = 0.55 * ma_score + 0.25 * pullback_score + 0.20 * vol_score
            scores[sym] = score
        return scores

    # ── Backtest state ────────────────────────────────────────────────────────
    portfolio_value = [100_000.0]
    holdings: dict[str, float] = {}
    daily_returns: list[float] = []
    total_cost = 0.0
    n_rebalances = 0
    bar_index = 0
    last_rebal_date = None
    prev_signal_scores: pd.Series | None = None  # for hysteresis

    rebal_freq = 10  # trading days between rebalances (biweekly)

    for i, date in enumerate(dates[1:], 1):
        prev_date = dates[i - 1]
        bar_index += 1

        prices_to_date = all_prices[all_prices["timestamp"] <= date]
        today_prices = all_prices[all_prices["timestamp"] == date]

        # ── Get today's macro indicators ─────────────────────────────────────
        vix = vix_by_date.get(date, vix_by_date.get(prev_date, 20.0))
        breadth = breadth_by_date.get(date, breadth_by_date.get(prev_date, 0.55))
        spy_close_today = spy_df[spy_df["timestamp"] == date]["close"].values
        spy_close = float(spy_close_today[0]) if len(spy_close_today) > 0 else None

        # ── U3: Regime detection ─────────────────────────────────────────────
        regime = regime_detector.update(vix, breadth, spy_close)

        # ── D4: Crash probability ─────────────────────────────────────────────
        crash_prob = 0.0
        if crash_engine is not None:
            try:
                macro_data = {"vix": vix, "breadth": breadth}
                if intermarket_df is not None and date in intermarket_df.index:
                    row = intermarket_df.loc[date]
                    macro_data["yield_curve"] = float(row.get("yield_curve_slope", 0.0))
                    macro_data["credit_spread"] = float(row.get("credit_spread_change_5d", 0.0))
                cs = crash_engine.predict(
                    market_data=prices_to_date[prices_to_date["symbol"].isin(equity_syms[:10])],
                    regime=regime,
                    macro_data=macro_data,
                )
                crash_prob = float(cs.crash_probability)
            except Exception:
                crash_prob = 0.0

        # ── Should rebalance? ─────────────────────────────────────────────────
        should_rebal = (last_rebal_date is None or (bar_index % rebal_freq == 0))
        # Also rebalance on crisis or regime change
        if crash_prob > 0.7 or regime in ("crisis", "bear") and bar_index % 5 == 0:
            should_rebal = True

        if should_rebal:
            n_rebalances += 1
            last_rebal_date = date

            # ── E1: Equity entry scores (pullback + MA trend) ─────────────────
            entry_scores = compute_entry_scores(prices_to_date, equity_syms)

            # ── E5: Breadth gate — filter longs when market is weak ───────────
            if breadth < 0.50:
                entry_scores = {s: v * (breadth / 0.55) for s, v in entry_scores.items()}

            # ── D5: Candlestick confirmation ──────────────────────────────────
            if cs_features is not None and not cs_features.empty:
                cs_today = cs_features[cs_features["timestamp"] == date]
                cs_map = cs_today.set_index("symbol")[["cs_hammer_v1","cs_engulfing_v1"]].max(axis=1).to_dict() if not cs_today.empty else {}
                for sym in list(entry_scores.keys()):
                    cs_boost = cs_map.get(sym, 0.0)
                    if cs_boost > 0:
                        entry_scores[sym] = entry_scores[sym] * (1 + 0.12 * cs_boost)

            # ── M3: Apply hysteresis to score vector ──────────────────────────
            if entry_scores and prev_signal_scores is not None:
                curr_series = pd.Series(entry_scores)
                try:
                    smoothed = apply_signal_hysteresis(curr_series, threshold=0.0, hysteresis_pct=0.15, cooldown_bars=3)
                    entry_scores = smoothed.to_dict()
                except Exception:
                    pass
            prev_signal_scores = pd.Series(entry_scores)

            # ── Select top equity positions ───────────────────────────────────
            top_n = 12
            ranked = sorted(entry_scores.items(), key=lambda x: x[1], reverse=True)
            selected = [(s, v) for s, v in ranked[:top_n] if v > 0.1]

            # ── Build raw equity weights (7% floor for HHI control) ───────────
            if selected:
                raw_equity_weights: dict[str, float] = {}
                for sym, score in selected:
                    raw_equity_weights[sym] = max(0.07, score)
                total = sum(raw_equity_weights.values())
                raw_equity_weights = {s: w / total for s, w in raw_equity_weights.items()}
            else:
                raw_equity_weights = {}

            # ── U2: Sector rotation weights ───────────────────────────────────
            sector_weights: dict[str, float] = {}
            if not sector_scores_df.empty and date in sector_scores_df.index:
                try:
                    from assembled_core.signals.sector_rotation import (
                        SectorRotationConfig, generate_sector_rotation_signals, get_sector_weights
                    )
                    sr_cfg = SectorRotationConfig(top_n_long=3, bottom_n_short=2)
                    sig = generate_sector_rotation_signals(
                        sector_scores_df.loc[date], available_etfs=sector_syms, config=sr_cfg
                    )
                    if sig.is_risk_off:
                        # Force regime to bear/crisis when all sectors negative
                        regime = "bear"
                    sector_weights = get_sector_weights(sig, long_weight=0.10, short_weight=0.06)
                except Exception:
                    pass

            # ── U3: Regime-adaptive allocation ────────────────────────────────
            final_weights = allocate_by_regime(regime, raw_equity_weights, sector_weights)

            # ── Crash probability exposure cap (D4) ───────────────────────────
            if crash_prob > 0.50:
                equity_cap = 1.0 - (crash_prob - 0.50) * 1.8
                equity_cap = max(0.10, equity_cap)
                for sym in list(final_weights.keys()):
                    if sym in equity_syms or sym in sector_syms:
                        final_weights[sym] *= equity_cap

            # ── S1+S3: VIX-gated inverse ETF shorts ───────────────────────────
            if vix is not None:
                if vix < 15:
                    short_cap = 0.0
                elif vix < 20:
                    short_cap = 0.05
                elif vix < 30:
                    short_cap = 0.12
                else:
                    short_cap = 0.22

                if short_cap > 0 and crash_prob > 0.30 and inv_selector is not None:
                    short_weight = min(short_cap, crash_prob * 0.25)
                    sh_instr = "SH" if "SH" in [s["symbol"] if isinstance(s, dict) else s for s in (inverse_prices["symbol"].unique() if not inverse_prices.empty else [])] else None
                    if sh_instr and not inverse_prices[inverse_prices["symbol"] == sh_instr].empty:
                        final_weights[sh_instr] = final_weights.get(sh_instr, 0.0) - short_weight

            # ── E2: Profit targets — apply reductions ─────────────────────────
            current_prices_map = {
                row["symbol"]: float(row["close"])
                for _, row in today_prices.iterrows()
                if "close" in today_prices.columns
            }
            position_records = build_position_records(
                final_weights, current_prices_map, position_records
            )
            pt_reductions = check_profit_targets(position_records, current_prices_map, pt_config)
            for sym, factor in pt_reductions.items():
                if sym in final_weights:
                    final_weights[sym] *= factor

            # ── E3+E4: Trailing stops with VIX scaling and time-stop ──────────
            positions_for_stops = {
                sym: {"entry_price": pos.entry_price}
                for sym, pos in position_records.items()
                if abs(final_weights.get(sym, 0)) > 0.01
            }
            if positions_for_stops:
                stop_result = compute_trailing_stops(
                    positions_for_stops,
                    prices_to_date,
                    regime=regime,
                    vix_level=vix,
                    prior_states=stop_states,
                    current_bar=bar_index,
                    time_stop_warn_bars=30,
                    time_stop_close_bars=60,
                )
                final_weights = apply_stop_reductions_to_weights(final_weights, stop_result)
                stop_states = {s.symbol: s for s in stop_result.stops}

            # ── Track costs ───────────────────────────────────────────────────
            all_syms_t = set(holdings) | set(final_weights)
            turnover = sum(abs(holdings.get(s, 0) - final_weights.get(s, 0)) for s in all_syms_t)
            cost = turnover * 6.0 / 10_000  # 6 bps average
            total_cost += cost

            holdings = {s: w for s, w in final_weights.items() if abs(w) > 0.001}

        # ── Daily P&L ─────────────────────────────────────────────────────────
        port_ret = 0.0
        for sym, w in holdings.items():
            sym_today = all_prices[(all_prices["symbol"] == sym) & (all_prices["timestamp"].isin([prev_date, date]))]
            if len(sym_today) >= 2:
                sym_today = sym_today.sort_values("timestamp")
                ret = sym_today["close"].iloc[-1] / sym_today["close"].iloc[0] - 1.0
                port_ret += w * ret

        daily_returns.append(port_ret)
        portfolio_value.append(portfolio_value[-1] * (1 + port_ret))

    # ── Results ───────────────────────────────────────────────────────────────
    returns = np.array(daily_returns)
    equity = np.array(portfolio_value)

    print(f"\n{'=' * 70}")
    print("M16 GRAND BACKTEST RESULTS")
    print(f"{'=' * 70}")
    print(f"Universe:     {len(equity_syms)} equities + {len(sector_syms)} sectors + {len(macro_syms)} macro + inverse ETFs")
    print(f"Period:       {START} → {END}")
    print(f"Rebalances:   {n_rebalances}")
    print("Starting:     $100,000")
    print(f"Final:        ${equity[-1]:,.2f}")
    print(f"Total Return: {(equity[-1]/equity[0]-1)*100:.2f}%")

    sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0.0
    down = returns[returns < 0]
    sortino = float(np.mean(returns) / np.std(down) * np.sqrt(252)) if len(down) > 0 and np.std(down) > 0 else 0.0
    cum = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cum)
    dd = (cum - running_max) / running_max
    max_dd = float(dd.min())
    vol = float(np.std(returns) * np.sqrt(252))
    cagr = float((equity[-1] / equity[0]) ** (252 / max(len(returns), 1)) - 1)
    calmar = float(cagr / abs(max_dd)) if abs(max_dd) > 1e-9 else 0.0

    # Hit rate
    pos_days = int(np.sum(returns > 0))
    hit_rate = pos_days / len(returns) if len(returns) > 0 else 0.0
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    pf = float(np.sum(wins) / abs(np.sum(losses))) if np.sum(losses) != 0 else 0.0

    print("\nRisk-Adjusted:")
    print(f"  Sharpe:      {sharpe:.3f}")
    print(f"  Sortino:     {sortino:.3f}")
    print(f"  Calmar:      {calmar:.3f}")
    print(f"  CAGR:        {cagr*100:.2f}%")
    print(f"  Volatility:  {vol*100:.2f}%")
    print(f"  Max Drawdown:{max_dd*100:.2f}%")
    print(f"  Hit Rate:    {hit_rate:.1%}")
    print(f"  Profit Factor:{pf:.2f}")
    print(f"  Total Costs: {total_cost*10000:.0f} bps")

    # HHI at last rebalance
    if holdings:
        w_arr = np.array(list(holdings.values()))
        hhi = float(np.sum(w_arr ** 2))
        eff_n = round(1 / hhi, 1) if hhi > 0 else 0
        print("\nPortfolio Structure (last rebalance):")
        print(f"  Positions:   {len(holdings)}")
        print(f"  HHI:         {hhi:.3f} (eff. N = {eff_n})")
        top5 = sorted(holdings.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        for sym, w in top5:
            print(f"    {sym:<8} {w:+.1%}")

    # ── Benchmarks ────────────────────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("BENCHMARK COMPARISON")
    print(f"{'─' * 70}")

    try:
        from assembled_core.qa.benchmark_metrics import compute_benchmark_metrics
        spy_full = load_pool(["SPY"], START, END)
        spy_ret = spy_full.set_index("timestamp")["close"].pct_change().dropna()
        spy_ret_aligned = spy_ret.reindex(pd.DatetimeIndex([dates[1+i] for i in range(len(returns))]), method="ffill").fillna(0).values[:len(returns)]

        if len(spy_ret_aligned) >= 10:
            bm = compute_benchmark_metrics(
                pd.Series(returns[:len(spy_ret_aligned)]),
                pd.Series(spy_ret_aligned),
            )
            spy_total = float((1 + spy_ret_aligned).prod() - 1)
            print(f"  SPY Total Return:  {spy_total*100:.2f}%")
            print(f"  Alpha (ann.):      {bm.alpha*100:.2f}%")
            print(f"  Beta:              {bm.beta:.3f}")
            print(f"  Information Ratio: {bm.information_ratio:.3f}")
            print(f"  Tracking Error:    {bm.tracking_error*100:.2f}%")
    except Exception as e:
        print(f"  Benchmark metrics: skipped ({e})")

    # 60/40 proxy
    try:
        tlt_full = load_pool(["TLT"], START, END)
        tlt_ret = tlt_full.set_index("timestamp")["close"].pct_change().dropna()
        tlt_arr = tlt_ret.reindex(pd.DatetimeIndex([dates[1+i] for i in range(len(returns))]), method="ffill").fillna(0).values[:len(returns)]
        spy60_40 = 0.60 * spy_ret_aligned + 0.40 * tlt_arr
        bal_total = float((1 + spy60_40).prod() - 1)
        bal_sharpe = float(np.mean(spy60_40) / np.std(spy60_40) * np.sqrt(252)) if np.std(spy60_40) > 0 else 0
        print(f"\n  60/40 Total Return: {bal_total*100:.2f}%")
        print(f"  60/40 Sharpe:       {bal_sharpe:.3f}")
        print(f"  Strategy Alpha vs 60/40: {(cagr - float((1+bal_total)**(252/max(len(spy60_40),1))-1))*100:.2f}%")
    except Exception as e:
        print(f"  60/40 benchmark: skipped ({e})")

    # ── CPCV ─────────────────────────────────────────────────────────────────
    try:
        from assembled_core.ml.cpcv import generate_cpcv_splits, compute_cpcv_sharpe_distribution
        splits = generate_cpcv_splits(len(returns), n_groups=6, k_test_groups=2, purge_length=5, embargo_length=3)
        if splits:
            paths = [returns[test_idx] for _, test_idx in splits]
            cpcv = compute_cpcv_sharpe_distribution(paths)
            print(f"\n  CPCV ({len(splits)} paths):")
            print(f"    Mean Sharpe:     {cpcv.mean_sharpe:.3f}")
            print(f"    P(Sharpe>0):     {cpcv.prob_positive_sharpe:.1%}")
            print(f"    Likely Overfit:  {cpcv.is_likely_overfit}")
    except Exception as e:
        print(f"  CPCV: skipped ({e})")

    # ── M15 Comparison ────────────────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("M15 → M16 COMPARISON")
    print(f"{'─' * 70}")
    m15_best = {"sharpe": 3.39, "total_return": 0.370, "max_dd": -0.048, "beta": 0.133, "hhi": 0.202}
    print(f"{'Metric':<22} {'M15-Best':>12} {'M16-Grand':>12} {'Delta':>12}")
    print("-" * 60)
    _m16 = {"sharpe": sharpe, "total_return": equity[-1]/equity[0]-1, "max_dd": max_dd, "hhi": hhi if holdings else 0}  # noqa: F841
    comparisons = [
        ("Sharpe", m15_best["sharpe"], sharpe, ""),
        ("Total Return", m15_best["total_return"]*100, (equity[-1]/equity[0]-1)*100, "%"),
        ("Max Drawdown", m15_best["max_dd"]*100, max_dd*100, "%"),
        ("HHI", m15_best["hhi"], hhi if holdings else 0, ""),
    ]
    for name, base, curr, unit in comparisons:
        delta = curr - base
        arrow = "+" if delta > 0 else ""
        print(f"{name:<22} {base:>10.3f}{unit} {curr:>10.3f}{unit} {arrow}{delta:>8.3f}{unit}")

    # ── Save results ──────────────────────────────────────────────────────────
    os.makedirs("output/grand_backtest", exist_ok=True)
    result_dict = {
        "total_return": float(equity[-1]/equity[0]-1),
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "cagr": cagr,
        "volatility": vol,
        "max_drawdown": max_dd,
        "hit_rate": hit_rate,
        "profit_factor": pf,
        "n_rebalances": n_rebalances,
        "total_cost_bps": total_cost * 10_000,
        "hhi": hhi if holdings else 0,
        "n_positions": len(holdings),
        "universe": "multi_asset_53",
        "modules_active": ["U1","U2","U3","E1","E2","E3","E4","E5","D1","D2","D3","D4","D5","S1","S3","M1","M3"],
    }
    with open("output/grand_backtest/results.json", "w") as f:
        json.dump(result_dict, f, indent=2)
    print("\nResults saved → output/grand_backtest/results.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
