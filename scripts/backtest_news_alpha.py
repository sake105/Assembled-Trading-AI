"""Standalone backtest for the news_alpha event-driven trading module.

Methodology:
- Curated timeline of 20 known historical crises (2013-2024)
- Each event fires through generate_signals() + signals_to_weights()
- Entry: close on event_date; Exit: close on event_date + hold_days
  (or early exit if stop_loss / take_profit hit)
- Portfolio P&L: sum of (weight_i * ret_i) per event
- No compounding — each event is independent (standalone alpha layer)

Run:
    python scripts/backtest_news_alpha.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Force UTF-8 output on Windows
if sys.stdout.encoding != "utf-8":
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
# Historical crisis events
# Each entry: (date_str, topic_id, severity, label)
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

# ---------------------------------------------------------------------------
# ETF symbols needed by the routing table
# ---------------------------------------------------------------------------
NEEDED_SYMS = [
    "XLE",
    "XOM",
    "UCO",  # energy
    "LMT",
    "NOC",
    "RTX",
    "GLD",  # defense / geo
    "SHY",
    "SH",  # safety / S&P inverse
    "XLF",
    "TBT",  # financials / bond inverse
    "TLT",
    "QQQ",  # duration / growth
    "UVXY",  # vol spike
]

CACHE_FILE = Path("output/news_alpha_etf_prices.parquet")


def load_prices() -> pd.DataFrame:
    """Load or download ETF price history (2012-2024)."""
    if CACHE_FILE.exists():
        df = pd.read_parquet(CACHE_FILE)
        print(f"[prices] loaded from cache: {df.shape}")
        return df

    print(f"[prices] downloading {NEEDED_SYMS} via yfinance …")
    raw = yf.download(
        NEEDED_SYMS,
        start="2012-01-01",
        end="2025-01-01",
        auto_adjust=True,
        progress=False,
    )
    close = raw["Close"] if "Close" in raw.columns else raw.xs("Close", axis=1, level=0)
    close.index = pd.to_datetime(close.index, utc=True)
    close.to_parquet(CACHE_FILE)
    print(f"[prices] downloaded & cached: {close.shape}")
    return close


@dataclass
class EventResult:
    date: str
    topic: str
    severity: int
    label: str
    signals_count: int
    portfolio_ret: float
    details: list[
        tuple[str, float, float, float, str]
    ]  # (sym, weight, entry, exit, exit_reason)


def run_event(
    event_date: str,
    topic_id: str,
    severity: int,
    label: str,
    prices: pd.DataFrame,
    policy: dict,
) -> EventResult:
    trigger = {
        "severity": severity,
        "topic": topic_id,
        "source": label,
        "event_id": f"{event_date}-{topic_id}",
    }
    signals = generate_signals([trigger], policy=policy)
    weights = signals_to_weights(signals, policy=policy)

    if not weights:
        return EventResult(event_date, topic_id, severity, label, 0, 0.0, [])

    # Find entry date (first trading day >= event_date)
    ts_event = pd.Timestamp(event_date, tz="UTC")
    avail_dates = prices.index[prices.index >= ts_event]
    if len(avail_dates) == 0:
        return EventResult(event_date, topic_id, severity, label, 0, 0.0, [])
    entry_date = avail_dates[0]

    # Determine hold_days from first signal
    hold_days = signals[0].hold_days if signals else 5
    sl_pct = signals[0].stop_loss_pct if signals else 0.08
    tp_pct = signals[0].take_profit_pct if signals else 0.15

    portfolio_ret = 0.0
    details = []

    for sym, weight in weights.items():
        if sym not in prices.columns:
            continue
        sym_prices = prices[sym].dropna()

        try:
            entry_price = sym_prices.loc[entry_date]
        except KeyError:
            nearby = sym_prices.loc[sym_prices.index >= entry_date]
            if len(nearby) == 0:
                continue
            entry_price = nearby.iloc[0]

        # Simulate day-by-day for stop/tp
        exit_price = entry_price
        exit_reason = f"time_{hold_days}d"
        future_dates = sym_prices.index[sym_prices.index > entry_date][: hold_days + 5]

        for i, d in enumerate(future_dates):
            if i >= hold_days:
                exit_price = sym_prices.loc[d]
                exit_reason = f"time_{hold_days}d"
                break
            p = sym_prices.loc[d]
            ret = p / entry_price - 1.0
            # For long positions (including inverse ETF longs)
            if ret >= tp_pct:
                exit_price = p
                exit_reason = f"take_profit({ret:.1%})"
                break
            if ret <= -sl_pct:
                exit_price = p
                exit_reason = f"stop_loss({ret:.1%})"
                break
        else:
            if len(future_dates) > 0:
                exit_price = sym_prices.loc[future_dates[-1]]

        pos_ret = (exit_price / entry_price - 1.0) * weight
        portfolio_ret += pos_ret
        details.append(
            (sym, weight, float(entry_price), float(exit_price), exit_reason)
        )

    return EventResult(
        event_date, topic_id, severity, label, len(signals), portfolio_ret, details
    )


def main() -> None:
    # Policy: no leverage for clean baseline
    policy = {
        "news_alpha": {
            "enabled": True,
            "base_weight": 0.08,
            "leverage_etfs_allowed": False,
            "max_gross_exposure": 0.40,
        }
    }

    prices = load_prices()
    print(f"[prices] available symbols: {sorted(prices.columns.tolist())}")
    print()

    results: list[EventResult] = []
    for date, topic, sev, label in EVENTS:
        r = run_event(date, topic, sev, label, prices, policy)
        results.append(r)

    # Print results
    print("=" * 90)
    print(f"{'DATE':<12} {'TOPIC':<25} {'SEV'} {'SIG':>4} {'P&L':>8}  LABEL")
    print("=" * 90)
    wins = losses = skips = 0
    total_ret = 0.0
    for r in results:
        if r.signals_count == 0:
            skips += 1
            pnl_str = "  (no signals)"
        else:
            total_ret += r.portfolio_ret
            pnl_pct = r.portfolio_ret * 100
            pnl_str = f"{pnl_pct:+.2f}%"
            if r.portfolio_ret > 0:
                wins += 1
            else:
                losses += 1
        print(
            f"{r.date:<12} {r.topic:<25} {r.severity}  {r.signals_count:>3}  {pnl_str:>8}  {r.label}"
        )

    print("=" * 90)
    fired = wins + losses
    print(f"\nEvents fired:  {fired}/{len(results)}  ({skips} no-route/no-data)")
    print(f"Win rate:      {wins}/{fired} = {wins / fired * 100:.0f}%" if fired else "")
    print(
        f"Total P&L:     {total_ret * 100:+.2f}% (sum of per-event portfolio returns)"
    )
    print(f"Avg per event: {total_ret / fired * 100:+.2f}%" if fired else "")
    print()

    # Detail for largest winners and losers
    by_ret = sorted(results, key=lambda r: r.portfolio_ret, reverse=True)
    print("── TOP 3 WINNERS ──")
    for r in by_ret[:3]:
        if r.signals_count > 0:
            print(f"  {r.date} {r.label}: {r.portfolio_ret * 100:+.2f}%")
            for sym, w, ep, xp, reason in r.details:
                ret = (xp / ep - 1) * 100
                print(
                    f"    {sym:6s} w={w:+.3f}  {ep:.2f}->{xp:.2f}  {ret:+.1f}%  [{reason}]"
                )

    print("\n── TOP 3 LOSERS ──")
    for r in by_ret[-3:]:
        if r.signals_count > 0:
            print(f"  {r.date} {r.label}: {r.portfolio_ret * 100:+.2f}%")
            for sym, w, ep, xp, reason in r.details:
                ret = (xp / ep - 1) * 100
                print(
                    f"    {sym:6s} w={w:+.3f}  {ep:.2f}->{xp:.2f}  {ret:+.1f}%  [{reason}]"
                )

    print("\n── LEVERAGE=TRUE COMPARISON ──")
    policy_lev = dict(policy)
    policy_lev["news_alpha"] = {**policy["news_alpha"], "leverage_etfs_allowed": True}
    lev_total = 0.0
    for date, topic, sev, label in EVENTS:
        r = run_event(date, topic, sev, label, prices, policy_lev)
        lev_total += r.portfolio_ret
    print(
        f"  leverage=True total P&L: {lev_total * 100:+.2f}% vs {total_ret * 100:+.2f}% no-leverage"
    )


if __name__ == "__main__":
    main()
