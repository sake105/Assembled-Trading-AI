"""Intraday vs EOD entry comparison for the news_alpha module.

Question: "How much P&L do we leave on the table by entering at EOD close
instead of at the intraday open?"

Methodology:
- Same 20 historical crisis events as backtest_news_alpha.py
- Same routing table (no leverage, base_weight=0.08)
- For each event, simulate TWO entry points:
    open_entry:  Enter at daily Open on event_date  (intraday proxy)
    close_entry: Enter at daily Close on event_date (EOD baseline)
  Exit logic identical: same hold_days / stop_loss / take_profit
- Delta = open_entry P&L - close_entry P&L

Key insight: Energy/commodity events spike intraday.
    Open entry catches the pre-spike price.
    Close entry buys AFTER the spike — systematically worse for oil events.

Note: "Open" is an optimistic proxy for intraday entry — in practice the
runner would enter at the first quote AFTER news breaks, which could be
09:30 Open (pre-market release) or any point during the session.
This gives the best-case intraday advantage.

Run:
    python scripts/backtest_news_alpha_intraday.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import warnings
from dataclasses import dataclass

import pandas as pd
import yfinance as yf

from src.assembled_core.events.news_alpha.signal_generator import (
    generate_signals,
    signals_to_weights,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Events (identical to backtest_news_alpha.py)
# ---------------------------------------------------------------------------

EVENTS = [
    # OIL / ENERGY
    ("2013-08-21", "energy_crisis", 3, "Syria chemical attack -> oil spike"),
    ("2016-04-17", "energy_crisis", 2, "OPEC Doha deal collapse"),
    ("2019-07-18", "shipping_disruption", 2, "Iran seizes UK tanker, Hormuz tension"),
    ("2019-09-14", "energy_crisis", 3, "Saudi Aramco Abqaiq drone attack"),
    ("2022-03-07", "energy_crisis", 3, "Russia-Ukraine -> oil/gas crisis peak"),
    ("2023-12-19", "shipping_disruption", 3, "Red Sea Houthi attacks, Suez diversion"),
    # GEOPOLITICAL CONFLICT
    ("2022-02-24", "geopolitical_conflict", 3, "Russia invades Ukraine"),
    ("2022-08-02", "taiwan_strait", 2, "Pelosi Taiwan visit, PLA exercises"),
    ("2023-10-07", "geopolitical_conflict", 3, "Hamas attack -> Israel-Gaza war"),
    # NUCLEAR RISK
    ("2022-09-21", "nuclear_risk", 3, "Putin nuclear threat speech"),
    # CENTRAL BANK
    ("2018-12-19", "central_bank_hike", 3, "Fed surprise hike -> market drop"),
    ("2019-07-31", "central_bank_cut", 2, "Fed first cut since 2008"),
    ("2020-03-03", "central_bank_cut", 3, "Fed emergency cut 50bp (COVID)"),
    ("2022-03-16", "central_bank_hike", 3, "Fed first hike, signals aggressive path"),
    ("2022-06-15", "central_bank_hike", 3, "Fed 75bp hike, largest since 1994"),
    ("2023-07-26", "central_bank_hike", 2, "Fed last hike in cycle"),
    # MARKET CRASH / PANIC
    ("2018-10-10", "market_crash", 3, "Tech selloff, VIX spike to 28"),
    ("2020-02-24", "market_crash", 3, "COVID first major selloff"),
    ("2020-03-16", "market_crash", 3, "COVID Black Monday, circuit breakers"),
    ("2022-01-24", "market_crash", 2, "Rate fears, Nasdaq -15% YTD"),
]

NEEDED_SYMS = [
    "XLE",
    "XOM",
    "UCO",
    "LMT",
    "NOC",
    "RTX",
    "GLD",
    "SHY",
    "SH",
    "XLF",
    "TBT",
    "TLT",
    "QQQ",
    "UVXY",
]

CACHE_FILE = Path("output/news_alpha_ohlc_prices.parquet")


def load_ohlc() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load or download daily OHLC (Open + Close) for all needed symbols."""
    if CACHE_FILE.exists():
        raw = pd.read_parquet(CACHE_FILE)
        print(f"[prices] loaded OHLC from cache: {raw.shape}")
    else:
        print(f"[prices] downloading OHLC for {NEEDED_SYMS} via yfinance ...")
        raw = yf.download(
            NEEDED_SYMS,
            start="2012-01-01",
            end="2025-01-01",
            auto_adjust=True,
            progress=False,
        )
        raw.index = pd.to_datetime(raw.index, utc=True)
        raw.to_parquet(CACHE_FILE)
        print(f"[prices] downloaded & cached: {raw.shape}")

    # Extract Open and Close frames (MultiIndex: level-0 = field, level-1 = symbol)
    if isinstance(raw.columns, pd.MultiIndex):
        opens = raw["Open"]
        closes = raw["Close"]
    else:
        # Single symbol fallback (shouldn't happen with list)
        opens = raw[["Open"]].rename(columns={"Open": NEEDED_SYMS[0]})
        closes = raw[["Close"]].rename(columns={"Close": NEEDED_SYMS[0]})

    return opens, closes


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------


@dataclass
class SimResult:
    date: str
    topic: str
    severity: int
    label: str
    signals_count: int
    pnl_eod: float  # enter at daily Close (EOD baseline)
    pnl_open: float  # enter at daily Open (intraday proxy)
    details: list[tuple[str, float, float, float, float, str]]
    # (sym, weight, open_price, close_entry, exit_price, exit_reason)


def _sim_position(
    sym: str,
    weight: float,
    entry_price: float,
    hold_days: int,
    sl_pct: float,
    tp_pct: float,
    future_closes: pd.Series,
) -> tuple[float, str]:
    """Simulate one position given entry_price. Returns (pos_ret, exit_reason)."""
    exit_price = entry_price
    exit_reason = f"time_{hold_days}d"
    for i, d in enumerate(future_closes.index):
        if i >= hold_days:
            exit_price = future_closes.loc[d]
            exit_reason = f"time_{hold_days}d"
            break
        p = future_closes.loc[d]
        ret = p / entry_price - 1.0
        if ret >= tp_pct:
            exit_price = p
            exit_reason = f"take_profit({ret:.1%})"
            break
        if ret <= -sl_pct:
            exit_price = p
            exit_reason = f"stop_loss({ret:.1%})"
            break
    else:
        if len(future_closes) > 0:
            exit_price = future_closes.iloc[-1]
    pos_ret = (exit_price / entry_price - 1.0) * weight
    return pos_ret, exit_reason


def run_event(
    event_date: str,
    topic_id: str,
    severity: int,
    label: str,
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    policy: dict,
) -> SimResult:
    trigger = {
        "severity": severity,
        "topic": topic_id,
        "source": label,
        "event_id": f"{event_date}-{topic_id}",
    }
    signals = generate_signals([trigger], policy=policy)
    weights = signals_to_weights(signals, policy=policy)

    if not weights:
        return SimResult(event_date, topic_id, severity, label, 0, 0.0, 0.0, [])

    ts_event = pd.Timestamp(event_date, tz="UTC")
    avail_dates = closes.index[closes.index >= ts_event]
    if len(avail_dates) == 0:
        return SimResult(event_date, topic_id, severity, label, 0, 0.0, 0.0, [])
    entry_date = avail_dates[0]

    hold_days = signals[0].hold_days if signals else 5
    sl_pct = signals[0].stop_loss_pct if signals else 0.08
    tp_pct = signals[0].take_profit_pct if signals else 0.15

    pnl_eod = 0.0
    pnl_open = 0.0
    details = []

    for sym, weight in weights.items():
        if sym not in closes.columns or sym not in opens.columns:
            continue

        sym_closes = closes[sym].dropna()
        sym_opens = opens[sym].dropna()

        try:
            close_price = sym_closes.loc[entry_date]
            open_price = sym_opens.loc[entry_date]
        except KeyError:
            nearby_c = sym_closes.loc[sym_closes.index >= entry_date]
            nearby_o = sym_opens.loc[sym_opens.index >= entry_date]
            if len(nearby_c) == 0:
                continue
            close_price = nearby_c.iloc[0]
            open_price = nearby_o.iloc[0] if len(nearby_o) > 0 else close_price

        future = sym_closes.loc[sym_closes.index > entry_date].iloc[: hold_days + 5]

        pos_eod, exit_reason = _sim_position(
            sym, weight, float(close_price), hold_days, sl_pct, tp_pct, future
        )
        pos_open, _ = _sim_position(
            sym, weight, float(open_price), hold_days, sl_pct, tp_pct, future
        )

        pnl_eod += pos_eod
        pnl_open += pos_open
        details.append(
            (sym, weight, float(open_price), float(close_price), 0.0, exit_reason)
        )

    return SimResult(
        event_date, topic_id, severity, label, len(signals), pnl_eod, pnl_open, details
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Category labels for grouping in output
_ENERGY = {"energy_crisis", "shipping_disruption"}
_GEO = {"geopolitical_conflict", "taiwan_strait", "nuclear_risk"}
_CB = {"central_bank_hike", "central_bank_cut"}
_CRASH = {"market_crash"}


def _category(topic: str) -> str:
    if topic in _ENERGY:
        return "ENERGY"
    if topic in _GEO:
        return "GEO/DEFENSE"
    if topic in _CB:
        return "CENTRAL BANK"
    return "MARKET CRASH"


def main() -> None:
    policy = {
        "news_alpha": {
            "enabled": True,
            "base_weight": 0.08,
            "leverage_etfs_allowed": False,
            "max_gross_exposure": 0.40,
        }
    }

    opens, closes = load_ohlc()
    print(f"[prices] symbols: {sorted(closes.columns.tolist())}")
    print()

    results: list[SimResult] = []
    for date, topic, sev, label in EVENTS:
        r = run_event(date, topic, sev, label, opens, closes, policy)
        results.append(r)

    # ---------------------------------------------------------------------------
    # Print comparison table
    # ---------------------------------------------------------------------------
    w = 100
    print("=" * w)
    print(
        f"{'DATE':<12} {'TOPIC':<25} {'SEV'} {'CAT':<13} {'EOD P&L':>8}  {'OPEN P&L':>9}  {'DELTA':>8}"
    )
    print("=" * w)

    cat_stats: dict[str, dict] = {}
    total_eod = total_open = 0.0
    wins_eod = wins_open = fired = 0

    for r in results:
        if r.signals_count == 0:
            print(
                f"{r.date:<12} {r.topic:<25} {r.severity}  {'---':<13} {'(no signals)':>8}"
            )
            continue

        cat = _category(r.topic)
        if cat not in cat_stats:
            cat_stats[cat] = {"eod": 0.0, "open": 0.0, "n": 0}
        cat_stats[cat]["eod"] += r.pnl_eod
        cat_stats[cat]["open"] += r.pnl_open
        cat_stats[cat]["n"] += 1

        total_eod += r.pnl_eod
        total_open += r.pnl_open
        fired += 1
        if r.pnl_eod > 0:
            wins_eod += 1
        if r.pnl_open > 0:
            wins_open += 1

        delta = r.pnl_open - r.pnl_eod
        delta_str = f"{delta * 100:+.2f}%"
        print(
            f"{r.date:<12} {r.topic:<25} {r.severity}  {cat:<13} "
            f"{r.pnl_eod * 100:>+7.2f}%  {r.pnl_open * 100:>+8.2f}%  {delta_str:>8}"
        )

    print("=" * w)
    delta_total = total_open - total_eod
    print(f"\n{'TOTAL':}")
    print(f"  EOD  entry: {total_eod * 100:+.2f}% total | win rate {wins_eod}/{fired}")
    print(
        f"  Open entry: {total_open * 100:+.2f}% total | win rate {wins_open}/{fired}"
    )
    print(
        f"  Delta (Open - EOD): {delta_total * 100:+.2f}%  ({'open better' if delta_total > 0 else 'eod better'})"
    )

    print(f"\n{'BY CATEGORY':}")
    print(f"  {'Category':<14} {'EOD':>8}  {'Open':>9}  {'Delta':>8}  {'N':>3}")
    for cat, s in sorted(cat_stats.items()):
        d = (s["open"] - s["eod"]) * 100
        print(
            f"  {cat:<14} {s['eod'] * 100:>+7.2f}%  {s['open'] * 100:>+8.2f}%  {d:>+7.2f}%  {s['n']:>3}"
        )

    print("\nKey question: do ENERGY events show a large positive delta (open >> eod)?")
    energy = cat_stats.get("ENERGY", {})
    if energy:
        ed = (energy["open"] - energy["eod"]) * 100
        print(
            f"  Energy/Shipping: EOD={energy['eod'] * 100:+.2f}%  Open={energy['open'] * 100:+.2f}%  "
            f"Delta={ed:+.2f}%  n={energy['n']}"
        )
        if ed > 0.5:
            print(
                "  -> Open entry meaningfully better for energy events. Intraday runner justified."
            )
        elif ed < -0.5:
            print(
                "  -> EOD actually better (mean reversion? spike before open?). Reconsider."
            )
        else:
            print("  -> Delta near zero. Timing edge modest for energy in this sample.")


if __name__ == "__main__":
    main()
