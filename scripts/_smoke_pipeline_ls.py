"""SMOKE TEST (research, throwaway): does the LITERAL run_trading_cycle preserve
signed SHORT positions end-to-end when risk controls are ON?

Goal: before building the full pipeline-realistic OOS matrix, prove that a
market-neutral long/short concept expressed as (signal_fn, position_sizing_fn)
survives the real pipeline:
    signals -> size_positions -> generate_orders_from_targets
    -> _apply_risk_controls_default (gross/net caps, dd-damper, regime, georisk)
    -> simulate_with_costs

Decisive check: at the FIRST rebalance (flat book), market-neutral L/S should
produce BUY-notional ~= SELL-notional. If SELL-notional ~= 0 the shorts were
silently clipped (long-only degradation) and we must fall back to the
imported-overlay approach and label honestly.

NOT a production change. Reads only; writes nothing. Pure-price signal.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.assembled_core.data.prices_ingest import load_eod_prices  # noqa: E402
from src.assembled_core.pipeline.trading_cycle_shared import (  # noqa: E402
    TradingContext,
)
from src.assembled_core.qa.backtest_engine import (  # noqa: E402
    BacktestResult,
    make_cycle_fn,
    run_portfolio_backtest,
)
from src.assembled_core.strategies.multifactor_long_short import (  # noqa: E402
    MultiFactorStrategyConfig,
    compute_multifactor_long_short_positions,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
log = logging.getLogger("smoke_ls")
log.setLevel(logging.INFO)

CAPITAL = 100_000.0
FORM_BARS = 126
N_SIDE = 3  # 3 long / 3 short


def momentum_ls_signal_fn(prices_df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional 126-bar momentum, top N LONG / bottom N SHORT at last bar."""
    if prices_df.empty or "close" not in prices_df.columns:
        return pd.DataFrame(columns=["timestamp", "symbol", "direction", "score"])
    df = prices_df[["timestamp", "symbol", "close"]].dropna()
    last_ts = df["timestamp"].max()
    piv = df.pivot_table(index="timestamp", columns="symbol", values="close")
    piv = piv.sort_index()
    if len(piv) < FORM_BARS + 1:
        return pd.DataFrame(columns=["timestamp", "symbol", "direction", "score"])
    ret = piv.iloc[-1] / piv.iloc[-(FORM_BARS + 1)] - 1.0
    ret = ret.dropna()
    if len(ret) < 2 * N_SIDE:
        return pd.DataFrame(columns=["timestamp", "symbol", "direction", "score"])
    ranked = ret.sort_values(ascending=False)
    longs = ranked.index[:N_SIDE]
    shorts = ranked.index[-N_SIDE:]
    rows = []
    for sym in longs:
        rows.append((last_ts, sym, "LONG", float(ret[sym])))
    for sym in shorts:
        rows.append((last_ts, sym, "SHORT", float(ret[sym])))
    return pd.DataFrame(rows, columns=["timestamp", "symbol", "direction", "score"])


def make_ls_sizing_fn(capital_unused: float):
    cfg = MultiFactorStrategyConfig(
        bundle_path="",  # unused by compute_multifactor_long_short_positions
        max_gross_exposure=1.0,
        max_leverage=1.0,
        use_regime_overlay=False,
    )

    def sizing_fn(signals_df: pd.DataFrame, capital: float) -> pd.DataFrame:
        return compute_multifactor_long_short_positions(
            signals_df, capital=capital, config=cfg
        )

    return sizing_fn


def month_end_dates(timestamps: pd.Series) -> list[pd.Timestamp]:
    s = pd.to_datetime(pd.Series(timestamps.unique()), utc=True).sort_values()
    df = pd.DataFrame({"ts": s})
    df["ym"] = df["ts"].dt.year * 100 + df["ts"].dt.month
    out = df.groupby("ym")["ts"].max()
    return [pd.Timestamp(t) for t in out]


def main() -> int:
    log.info("Loading prices (offline cache)…")
    prices = load_eod_prices(None)
    prices["timestamp"] = pd.to_datetime(prices["timestamp"], utc=True)

    # small slice: ~12 best-covered symbols, 2018-06..2019-12 (warmup + test)
    lo = pd.Timestamp("2018-06-01", tz="UTC")
    hi = pd.Timestamp("2019-12-31", tz="UTC")
    win = prices[(prices["timestamp"] >= lo) & (prices["timestamp"] <= hi)]
    counts = win.groupby("symbol")["timestamp"].count().sort_values(ascending=False)
    syms = list(counts.index[:12])
    sl = win[win["symbol"].isin(syms)].copy()
    sl = sl.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    log.info("Slice: %d rows, %d symbols", len(sl), sl["symbol"].nunique())

    # rebalance only on 2019 month-ends (2018 = warmup for 126-bar formation)
    rebs = [
        t
        for t in month_end_dates(sl["timestamp"])
        if pd.Timestamp(t) >= pd.Timestamp("2019-01-01", tz="UTC")
    ]
    rebs = [pd.Timestamp(t) for t in rebs]
    log.info(
        "Rebalance dates: %d (first=%s)", len(rebs), rebs[0].date() if rebs else None
    )

    sizing_fn = make_ls_sizing_fn(CAPITAL)

    ctx_template = TradingContext(
        prices=sl,
        freq="1d",
        universe=syms,
        use_factor_store=False,
        # Pass raw prices as "precomputed" so the cycle skips heavy enrichment
        # (pure-price momentum signal needs no TA features); isolates short survival.
        precomputed_prices_with_features=sl,
        write_outputs=False,
        enable_risk_controls=True,  # <-- the realism layer we are testing
        backtest_use_snapshot=False,  # history-slice: signal needs price history
    )

    cycle_fn = make_cycle_fn(
        ctx_template,
        signal_fn=momentum_ls_signal_fn,
        position_sizing_fn=sizing_fn,
        capital=CAPITAL,
        enable_risk_controls=True,
    )

    log.info("Running literal pipeline backtest (risk controls ON, costs ON)…")
    result: BacktestResult = run_portfolio_backtest(
        prices=sl,
        signal_fn=momentum_ls_signal_fn,
        position_sizing_fn=sizing_fn,
        start_capital=CAPITAL,
        include_costs=True,
        include_trades=True,
        include_targets=True,
        compute_features=False,
        cycle_fn=cycle_fn,
        include_ledger=False,
        strict_session_gate=False,
        rebalance_schedule="monthly",
        rebalance_timestamps=rebs,
    )

    eq = result.equity
    trades = result.trades
    tgts = result.target_positions

    print("=" * 70)
    print("SMOKE RESULT")
    print("=" * 70)
    print(f"equity points : {len(eq)}")
    if eq is not None and not eq.empty and "equity" in eq.columns:
        print(f"equity start  : {eq['equity'].iloc[0]:.2f}")
        print(f"equity end    : {eq['equity'].iloc[-1]:.2f}")
    print(f"final_pf      : {result.metrics.get('final_pf')}")
    print(f"sharpe        : {result.metrics.get('sharpe')}")
    print(f"n trades      : {0 if trades is None else len(trades)}")

    if tgts is not None and not tgts.empty and "target_qty" in tgts.columns:
        neg = (tgts["target_qty"] < 0).sum()
        pos = (tgts["target_qty"] > 0).sum()
        print(f"targets: {pos} long (qty>0), {neg} short (qty<0)")

    verdict = "UNKNOWN"
    if trades is not None and not trades.empty:
        t = trades.copy()
        t["timestamp"] = pd.to_datetime(t["timestamp"], utc=True)
        first_ts = t["timestamp"].min()
        first = t[t["timestamp"] == first_ts]
        # notional per leg
        qcol = "fill_qty" if "fill_qty" in first.columns else "qty"
        pcol = "fill_price" if "fill_price" in first.columns else "price"
        first = first.copy()
        first["notional"] = first[qcol].abs() * first[pcol].abs()
        buy_n = first.loc[first["side"].str.upper() == "BUY", "notional"].sum()
        sell_n = first.loc[first["side"].str.upper() == "SELL", "notional"].sum()
        print(
            f"first rebalance {pd.Timestamp(first_ts).date()}: "
            f"BUY notional={buy_n:,.0f}  SELL notional={sell_n:,.0f}"
        )
        if buy_n > 0 and sell_n > 0:
            ratio = sell_n / buy_n
            print(f"SELL/BUY notional ratio = {ratio:.2f}")
            if 0.5 <= ratio <= 2.0:
                verdict = "SHORTS_SURVIVE"
            else:
                verdict = "SHORTS_PARTIAL"
        elif sell_n == 0:
            verdict = "SHORTS_CLIPPED"
    print(f"VERDICT       : {verdict}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
