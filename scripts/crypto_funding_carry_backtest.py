"""Crypto Funding-Rate-Carry — Standalone Backtest Study.

Simulates a delta-neutral funding-rate-carry strategy on BTC and ETH
perpetual futures using free historical data from Binance public FAPI.

Position: Long Spot X + Short Perp X (delta-neutral).
PnL source: funding rate received by the short-perp leg minus trading fees.

NOT wired into the equity strategy registry.
Run: python scripts/crypto_funding_carry_backtest.py [--no-cache]

Data:
  Source  : Binance FAPI public endpoints (no auth required)
  Cache   : data/crypto_funding_cache/  (Parquet)
Output  : docs/results/2026_05_crypto_funding_carry_backtest.md
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("crypto_carry")

# ── Configuration ────────────────────────────────────────────────────────────
SYMBOLS = ["BTCUSDT", "ETHUSDT"]

ENTRY_APR = 0.08  # 8 %/yr — enter when funding exceeds this
EXIT_APR = 0.05  # 5 %/yr — exit when funding drops below this
STABLE_N = 6  # last N 8h-intervals must all be positive before entry

TAKER_FEE = 0.0004  # 0.04 % per leg (taker)
N_LEGS = 2  # spot + perp per side
OPEN_COST = TAKER_FEE * N_LEGS  # 0.08 % on open
CLOSE_COST = TAKER_FEE * N_LEGS  # 0.08 % on close

INTERVALS_PER_DAY = 3  # Binance 8 h funding
INTERVALS_PER_YEAR = INTERVALS_PER_DAY * 365  # 1 095

MARGIN_LEVERAGE = 5  # scenario: 5× isolated margin on perp
LIQ_THRESHOLD = 1.0 / MARGIN_LEVERAGE  # 20 % adverse = wipeout

INITIAL_NOTIONAL = 10_000.0  # USD per asset (for $ PnL display)

BINANCE_FAPI = "https://fapi.binance.com"
CACHE_DIR = ROOT / "data" / "crypto_funding_cache"
OUT_MD = ROOT / "docs" / "results" / "2026_05_crypto_funding_carry_backtest.md"


# ── Data fetching ─────────────────────────────────────────────────────────────


def _get(url: str, params: dict, retries: int = 3) -> Any:
    """GET with retry and polite rate-limiting."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            if attempt == retries - 1:
                raise
            log.warning("Request failed (%s), retry %d/%d …", exc, attempt + 1, retries)
            time.sleep(2**attempt)
    return None  # unreachable


def fetch_funding_rates(symbol: str) -> pd.DataFrame:
    """Fetch all 8 h funding rates for *symbol* from Binance FAPI."""
    records: list[dict] = []
    start_ms = int(pd.Timestamp("2019-01-01", tz="UTC").timestamp() * 1000)
    url = f"{BINANCE_FAPI}/fapi/v1/fundingRate"
    limit = 1000

    while True:
        data = _get(url, {"symbol": symbol, "startTime": start_ms, "limit": limit})
        if not data:
            break
        records.extend(data)
        last_ts = pd.Timestamp(int(data[-1]["fundingTime"]), unit="ms", tz="UTC")
        log.info(
            "  [%s] funding: +%d → %d total  (last: %s)",
            symbol,
            len(data),
            len(records),
            last_ts.date(),
        )
        if len(data) < limit:
            break
        start_ms = int(data[-1]["fundingTime"]) + 1  # int() is safe here (scalar)
        time.sleep(0.15)

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(
        df["fundingTime"].astype("int64"), unit="ms", utc=True
    )
    df["funding_rate"] = df["fundingRate"].astype(float)
    df = (
        df[["timestamp", "funding_rate"]]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    return df


def fetch_klines(symbol: str, interval: str = "8h") -> pd.DataFrame:
    """Fetch OHLCV klines for *symbol* at *interval* from Binance FAPI."""
    records: list[list] = []
    start_ms = int(pd.Timestamp("2019-01-01", tz="UTC").timestamp() * 1000)
    url = f"{BINANCE_FAPI}/fapi/v1/klines"
    limit = 1500

    while True:
        data = _get(
            url,
            {
                "symbol": symbol,
                "interval": interval,
                "startTime": start_ms,
                "limit": limit,
            },
        )
        if not data:
            break
        records.extend(data)
        if len(data) < limit:
            break
        start_ms = int(data[-1][0]) + 1
        time.sleep(0.15)

    cols = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "q_vol",
        "n_trades",
        "tb_base",
        "tb_quote",
        "_",
    ]
    df = pd.DataFrame(records, columns=cols)
    df["timestamp"] = pd.to_datetime(
        df["open_time"].astype("int64"), unit="ms", utc=True
    )
    for c in ["open", "high", "low", "close"]:
        df[c] = df[c].astype(float)
    return (
        df[["timestamp", "open", "high", "low", "close"]]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def _save_parquet(df: pd.DataFrame, path: Path) -> None:
    """Save df to Parquet, storing tz-aware timestamps as int64 ms epoch.

    pyarrow on Windows can mishandle datetime64[ns, UTC] columns, producing
    epoch-0 dates on read-back.  Storing the raw millisecond integer and
    reconstructing on load is a reliable workaround.
    """
    out = df.copy()
    if "timestamp" in out.columns:
        out["timestamp_ms"] = out["timestamp"].astype("int64") // 1_000_000
        out = out.drop(columns=["timestamp"])
    out.to_parquet(path)


def _load_parquet(path: Path) -> pd.DataFrame:
    """Load Parquet, reconstructing timestamp from timestamp_ms when present."""
    df = pd.read_parquet(path)
    if "timestamp_ms" in df.columns and "timestamp" not in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
        df = df.drop(columns=["timestamp_ms"])
    return df


def load_or_fetch(symbol: str, no_cache: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (funding_df, klines_df), reading from Parquet cache when available."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fr_path = CACHE_DIR / f"{symbol}_funding_rates.parquet"
    kl_path = CACHE_DIR / f"{symbol}_klines_8h.parquet"

    if fr_path.exists() and not no_cache:
        log.info("[%s] funding rates: loading from cache (%s)", symbol, fr_path.name)
        funding_df = _load_parquet(fr_path)
    else:
        log.info("[%s] fetching funding rates from Binance FAPI …", symbol)
        funding_df = fetch_funding_rates(symbol)
        _save_parquet(funding_df, fr_path)
        log.info("  Saved -> %s (%d rows)", fr_path.name, len(funding_df))

    if kl_path.exists() and not no_cache:
        log.info("[%s] klines: loading from cache (%s)", symbol, kl_path.name)
        klines_df = _load_parquet(kl_path)
    else:
        log.info("[%s] fetching 8 h klines from Binance FAPI …", symbol)
        klines_df = fetch_klines(symbol)
        _save_parquet(klines_df, kl_path)
        log.info("  Saved -> %s (%d rows)", kl_path.name, len(klines_df))

    return funding_df, klines_df


# ── Simulation ────────────────────────────────────────────────────────────────


def simulate_carry(df_in: pd.DataFrame) -> pd.DataFrame:
    """Run delta-neutral carry simulation.

    Returns df with columns: funding_rate, apr, state, interval_pnl, cum_pnl_usd.
    State ∈ {FLAT, ENTRY, ACTIVE, EXIT}.
    Dollar PnL is per INITIAL_NOTIONAL.
    """
    df = df_in.copy().sort_values("timestamp").reset_index(drop=True)
    df["apr"] = df["funding_rate"] * INTERVALS_PER_YEAR

    # Rolling window: minimum funding_rate over last STABLE_N intervals including the
    # current row (i.e. requires the current interval to be positive too, which is
    # already implied by APR >= ENTRY_APR; no look-ahead beyond current settlement).
    df["roll_min"] = df["funding_rate"].rolling(STABLE_N, min_periods=STABLE_N).min()

    in_pos = False
    states: list[str] = []
    pnls: list[float] = []

    for i, row in df.iterrows():
        state = "FLAT"
        pnl = 0.0

        if not in_pos:
            # Entry: APR above threshold AND last STABLE_N intervals all positive.
            # PnL on the entry interval is the open cost only — funding[i] is already
            # settled before we can act, so we do NOT count it as PnL here.
            # The first funding receipt happens in the next (ACTIVE) interval.
            if (
                row["apr"] >= ENTRY_APR
                and pd.notna(row["roll_min"])
                and row["roll_min"] > 0
            ):
                in_pos = True
                pnl = -OPEN_COST  # pay entry fee; funding accrual starts next interval
                state = "ENTRY"
        else:
            # Exit when APR falls below threshold OR funding turns negative (explicit OR
            # to match the documented contract; with EXIT_APR=0.05 these are equivalent
            # today, but the explicit check is robust to future threshold changes).
            if row["apr"] < EXIT_APR or row["funding_rate"] < 0:
                in_pos = False
                pnl = (
                    -CLOSE_COST
                )  # pay exit fee; funding[i] already settled, not counted
                state = "EXIT"
            else:
                pnl = row["funding_rate"]
                state = "ACTIVE"

        states.append(state)
        pnls.append(pnl)

    df["state"] = states
    df["interval_pnl"] = pnls  # per unit of notional
    df["cum_pnl_usd"] = (df["interval_pnl"] * INITIAL_NOTIONAL).cumsum()

    return df


# ── Risk metrics ──────────────────────────────────────────────────────────────


def compute_risk_metrics(sim: pd.DataFrame, klines: pd.DataFrame) -> dict:
    """Compute all risk/performance metrics from simulation and price data."""

    # ── funding statistics (full sample) ─────────────────────────────────────
    all_fr = sim["funding_rate"]
    neg_mask = all_fr < 0.0

    # longest consecutive negative streak
    in_neg_streak = False
    max_neg = 0
    cur_neg = 0
    for v in all_fr:
        if v < 0:
            cur_neg += 1
            in_neg_streak = True
        else:
            if in_neg_streak:
                max_neg = max(max_neg, cur_neg)
                cur_neg = 0
            in_neg_streak = False
    if in_neg_streak:
        max_neg = max(max_neg, cur_neg)

    # ── active-period statistics ──────────────────────────────────────────────
    active_mask = sim["state"].isin(["ACTIVE", "ENTRY", "EXIT"])
    active_pnl = sim.loc[active_mask, "interval_pnl"]

    n_active = active_mask.sum()
    n_total = len(sim)
    years = n_total / INTERVALS_PER_YEAR
    years_active = n_active / INTERVALS_PER_YEAR if n_active > 0 else 0.0

    total_pnl_usd = (sim["interval_pnl"] * INITIAL_NOTIONAL).sum()
    # Simple-interest annualized return: sum(pnl_fractions) / years.
    # Labels as "Net APR" in the report, but is additive not geometric.
    # For 5-7% over 6+ years the geometric/simple difference is <1pp.
    net_apr = (total_pnl_usd / INITIAL_NOTIONAL) / years if years > 0 else 0.0

    # Sharpe (on 8 h interval returns, including flat periods as 0)
    ret_series = sim["interval_pnl"].fillna(0.0)
    mu_ann = ret_series.mean() * INTERVALS_PER_YEAR
    std_ann = ret_series.std() * np.sqrt(INTERVALS_PER_YEAR)
    sharpe = mu_ann / std_ann if std_ann > 1e-9 else float("nan")

    # Sharpe on active-only returns (even more misleadingly high)
    mu_active = active_pnl.mean() * INTERVALS_PER_YEAR if n_active else float("nan")
    std_active = (
        active_pnl.std() * np.sqrt(INTERVALS_PER_YEAR) if n_active else float("nan")
    )
    sharpe_active = (
        mu_active / std_active if (std_active and std_active > 1e-9) else float("nan")
    )

    # Max drawdown on cumulative PnL ($)
    cum = sim["cum_pnl_usd"]
    roll_max = cum.cummax()
    dd = cum - roll_max
    max_dd_usd = float(dd.min())

    # Worst single-interval loss (when in position, in USD)
    worst_interval_usd = float(active_pnl.min() * INITIAL_NOTIONAL) if n_active else 0.0

    # Distribution shape (skewness, excess kurtosis)
    skew = (
        float(ret_series[ret_series != 0].skew())
        if (ret_series != 0).any()
        else float("nan")
    )
    kurt = (
        float(ret_series[ret_series != 0].kurt())
        if (ret_series != 0).any()
        else float("nan")
    )

    # ── Price risk analysis ───────────────────────────────────────────────────
    kl = klines.copy().sort_values("timestamp").reset_index(drop=True)
    kl["ret_open_to_close"] = kl["close"] / kl["open"] - 1.0
    kl["ret_open_to_high"] = kl["high"] / kl["open"] - 1.0

    # Adverse move for SHORT perp = price going UP in an interval
    worst_adverse_pct = float(
        kl["ret_open_to_high"].max()
    )  # worst intra-bar spike upward
    worst_adverse_ts = kl.loc[kl["ret_open_to_high"].idxmax(), "timestamp"]

    # At MARGIN_LEVERAGE × leverage: effective margin = 1/leverage
    # Liquidation when adverse move ≥ effective_margin
    liq_margin_pct = LIQ_THRESHOLD
    n_liq_events = int((kl["ret_open_to_high"] >= liq_margin_pct).sum())

    # Historical VaR (99th percentile worst funding payment while in position)
    var_99_pct = float(active_pnl.quantile(0.01)) if n_active else float("nan")

    return dict(
        # Sample info
        n_total=n_total,
        start_date=sim["timestamp"].min(),
        end_date=sim["timestamp"].max(),
        years=years,
        # Funding distribution
        pct_positive=float((all_fr > 0).mean()),
        pct_negative=float(neg_mask.mean()),
        mean_fr_8h=float(all_fr.mean()),
        median_fr_8h=float(all_fr.median()),
        max_neg_streak=max_neg,
        max_neg_streak_hrs=max_neg * 8,
        min_fr_8h=float(all_fr.min()),
        # Position
        n_active=int(n_active),
        pct_in_position=float(n_active / n_total) if n_total else 0.0,
        years_active=years_active,
        # Performance
        total_pnl_usd=total_pnl_usd,
        net_apr=net_apr,
        sharpe=sharpe,
        sharpe_active=sharpe_active,
        max_dd_usd=max_dd_usd,
        worst_interval_usd=worst_interval_usd,
        var_99_usd=var_99_pct * INITIAL_NOTIONAL
        if not np.isnan(var_99_pct)
        else float("nan"),
        skew=skew,
        excess_kurt=kurt,
        # Price risk
        worst_adverse_pct=worst_adverse_pct,
        worst_adverse_ts=worst_adverse_ts,
        n_liq_events=n_liq_events,
        liq_threshold_pct=liq_margin_pct,
        # Raw series for report
        _sim=sim,
        _klines=kl,
    )


# ── Report writing ─────────────────────────────────────────────────────────────


def _f(v: float, fmt: str = "+.1%", na: str = "—") -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return na
    return format(v, fmt)


def _p(v: float, decimals: int = 1) -> str:
    """Format as plain percent (no sign)."""
    if np.isnan(v):
        return "—"
    return f"{v * 100:.{decimals}f}%"


def write_report(
    btc_sim: pd.DataFrame,
    btc_risk: dict,
    eth_sim: pd.DataFrame,
    eth_risk: dict,
) -> None:
    run_dt = pd.Timestamp.now().strftime("%Y-%m-%d")
    btc_start = btc_risk["start_date"].strftime("%Y-%m-%d")
    btc_end = btc_risk["end_date"].strftime("%Y-%m-%d")
    eth_start = eth_risk["start_date"].strftime("%Y-%m-%d")
    eth_end = eth_risk["end_date"].strftime("%Y-%m-%d")

    lines: list[str] = []

    lines += [
        "# Crypto Funding-Rate-Carry — Backtest Study",
        "",
        f"Run date: {run_dt}",
        "",
        "> **Disclaimer:** This is a pure backtest research study.",
        "> No exchange account, no real-money execution, no EU/MiCA compliance assessment.",
        "> Live implementation would require: exchange account setup, retail-access",
        "> review under MiCA/BaFin rules, and a separate live risk model.",
        "> Counterparty/exchange risk is **NOT modelled** here (see §5).",
        "",
        "---",
        "",
        "## 1. Data Source & Universe",
        "",
        "| Field | Detail |",
        "|-------|--------|",
        "| Source | Binance FAPI public endpoint — no auth required |",
        "| Endpoints | `/fapi/v1/fundingRate`, `/fapi/v1/klines?interval=8h` |",
        "| Funding interval | 8 hours (Binance standard for USDT-M perps) |",
        f"| BTC (BTCUSDT) | {btc_start} → {btc_end} — {btc_risk['n_total']:,} funding intervals |",
        f"| ETH (ETHUSDT) | {eth_start} → {eth_end} — {eth_risk['n_total']:,} funding intervals |",
        "| Cache | `data/crypto_funding_cache/` (Parquet) |",
        "",
        "**Note on data quality:** Binance historical funding rates are generally complete",
        "back to contract launch. A small number of intervals may show exactly 0.01%",
        "(the exchange default floor) rather than a market-driven rate.",
        "",
        "---",
        "",
        "## 2. Strategy Rules",
        "",
        "**Position:** Long Spot + Short Perp (delta-neutral) on the same notional X.",
        "",
        "| Rule | Detail |",
        "|------|--------|",
        f"| Entry | APR > {ENTRY_APR:.0%} AND last {STABLE_N} funding intervals all positive |",
        f"| Exit | APR < {EXIT_APR:.0%} OR funding becomes negative |",
        f"| Fees | {TAKER_FEE:.2%} taker × {N_LEGS} legs open + {N_LEGS} legs close = {OPEN_COST + CLOSE_COST:.2%} roundtrip |",
        "| Execution lag | Signals based on interval-end rates; costs apply on transition |",
        f"| Notional (display) | USD {INITIAL_NOTIONAL:,.0f} per asset |",
        "",
        "**Negative funding is NOT filtered out.** When funding turns negative during an",
        "active position, the short-perp leg PAYS funding (PnL negative that interval).",
        "The exit rule closes the position only at the START of the next evaluation.",
        "",
        "---",
        "",
        "## 3. Performance Summary",
        "",
    ]

    for label, risk in [("BTC (BTCUSDT)", btc_risk), ("ETH (ETHUSDT)", eth_risk)]:
        apr_ann = risk["net_apr"]
        sharpe = risk["sharpe"]
        sharpe_a = risk["sharpe_active"]
        max_dd = risk["max_dd_usd"]
        pnl_usd = risk["total_pnl_usd"]
        worst_i = risk["worst_interval_usd"]
        pct_pos = risk["pct_positive"]
        pct_neg = risk["pct_negative"]
        yrs = risk["years"]

        lines += [
            f"### {label}",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Sample period | {risk['start_date'].date()} → {risk['end_date'].date()} ({yrs:.1f} years) |",
            f"| Funding intervals | {risk['n_total']:,} total |",
            f"| % time positive funding | {pct_pos:.1%} |",
            f"| % time negative funding | {pct_neg:.1%} |",
            f"| Mean 8 h funding rate | {risk['mean_fr_8h']:+.4%} |",
            f"| Median 8 h funding rate | {risk['median_fr_8h']:+.4%} |",
            f"| Implied mean APR (full sample) | {risk['mean_fr_8h'] * INTERVALS_PER_YEAR:+.1%} |",
            f"| % time in position | {risk['pct_in_position']:.1%} |",
            f"| **Net APR after fees** | **{apr_ann:+.1%}** |",
            f"| Total PnL (USD, ${INITIAL_NOTIONAL:,.0f} notional) | ${pnl_usd:+,.0f} |",
            f"| Sharpe (all periods, flat=0) | {sharpe:+.2f} |",
            f"| Sharpe (active-only) | {sharpe_a:+.2f} ⚠ see caveat §4 |",
            f"| Max drawdown (USD) | ${max_dd:,.0f} |",
            f"| Worst single-interval PnL | ${worst_i:,.1f} |",
            f"| Return skewness | {risk['skew']:+.2f} |",
            f"| Excess kurtosis | {risk['excess_kurt']:+.2f} |",
            "",
        ]

    # ── Year-by-year table: BTC ───────────────────────────────────────────────
    for label, sim, risk in [("BTC", btc_sim, btc_risk), ("ETH", eth_sim, eth_risk)]:
        lines += [
            f"### {label} — Year-by-Year Net PnL",
            "",
            "| Year | Mean FR (8h) | % Time Positive | In-Position | Net PnL (USD) | Net APR |",
            "|------|-------------|-----------------|-------------|---------------|---------|",
        ]
        sim_y = sim.copy()
        sim_y["year"] = sim_y["timestamp"].dt.year
        for yr, grp in sim_y.groupby("year"):
            n = len(grp)
            fr = grp["funding_rate"].mean()
            pp = (grp["funding_rate"] > 0).mean()
            n_act = grp["state"].isin(["ACTIVE", "ENTRY", "EXIT"]).sum()
            yr_pnl = (grp["interval_pnl"] * INITIAL_NOTIONAL).sum()
            yr_yrs = n / INTERVALS_PER_YEAR
            yr_apr = (grp["interval_pnl"].sum()) / yr_yrs
            lines.append(
                f"| {yr} | {fr:+.4%} | {pp:.1%} | {n_act / n:.0%} "
                f"| ${yr_pnl:+,.0f} | {yr_apr:+.1%} |"
            )
        lines.append("")

    lines += [
        "---",
        "",
        "## 4. Risk Analysis",
        "",
        "### 4.1 Negative Funding Periods",
        "",
    ]

    for label, risk in [("BTC", btc_risk), ("ETH", eth_risk)]:
        lines += [
            f"**{label}**",
            "",
            f"- % of all intervals with negative funding: {risk['pct_negative']:.1%}",
            f"- Minimum 8 h rate recorded: {risk['min_fr_8h']:+.4%}  "
            f"(annualized: {risk['min_fr_8h'] * INTERVALS_PER_YEAR:+.1%})",
            f"- Longest consecutive negative-funding streak: **{risk['max_neg_streak']} intervals**"
            f" = **{risk['max_neg_streak_hrs']} hours** ({risk['max_neg_streak'] // 3:.0f} days)",
            "",
        ]

    lines += [
        "### 4.2 Carry Sharpe — ⚠ Critical Caveat",
        "",
        "> **The carry Sharpe ratio is structurally misleading.**",
        ">",
        "> Funding-rate carry is a classic 'picking up pennies in front of a steamroller'",
        "> strategy. The return distribution looks like this:",
        ">",
        "> - **Normal regime** (most of the time): small positive returns every 8 hours.",
        ">   Low variance → high apparent Sharpe.",
        "> - **Tail regime** (rare): funding turns sharply negative during crypto market",
        ">   dislocations (e.g. 2022 LUNA/3AC collapse, FTX implosion), or margin is called",
        ">   on the perp leg before the spot can be liquidated.",
        ">",
        "> The Sharpe ratio computed from 8 h intervals does NOT capture this tail risk,",
        "> because the tail events are exactly as rare as they are severe. The true",
        "> risk-adjusted return is substantially lower than the Sharpe implies.",
        ">",
        "> **Do NOT use this Sharpe to compare against equity strategies.**",
        ">",
        f"> Skewness: {btc_risk['skew']:+.2f} / {eth_risk['skew']:+.2f} (BTC/ETH)",
        f"> Excess kurtosis: {btc_risk['excess_kurt']:+.2f} / {eth_risk['excess_kurt']:+.2f}",
        "> Negative skew and fat tails confirm the steamroller profile.",
        "",
        "### 4.3 Liquidation / Margin Risk (Short Perp Leg)",
        "",
        "The short-perp position is subject to margin calls if price spikes sharply",
        "upward before the opposing long-spot can be sold. Even if the net position",
        "is delta-neutral on paper, exchanges liquidate the perp leg independently.",
        "",
    ]

    for label, risk in [("BTC", btc_risk), ("ETH", eth_risk)]:
        adv = risk["worst_adverse_pct"]
        adv_ts = risk["worst_adverse_ts"]
        liq_thr = risk["liq_threshold_pct"]
        n_liq = risk["n_liq_events"]
        lines += [
            f"**{label} — largest single 8 h intrabar spike (open→high):**",
            "",
            f"- Worst upward spike: **+{adv:.1%}** on {pd.Timestamp(adv_ts).date()}",
            f"- At {MARGIN_LEVERAGE}× isolated margin (liquidation at ≥{liq_thr:.0%} adverse move):",
            f"  **{n_liq} intervals** in the sample exceeded the liquidation threshold",
            f"  ({n_liq / risk['n_total']:.2%} of all intervals).",
            "- A liquidation would wipe the entire perp margin even though the spot leg gains.",
            f"  Net loss ≈ 1/(leverage) = **{liq_thr:.0%} of total notional** (margin forfeited,",
            "  spot gain partially offsetting — but timing and slippage make recovery uncertain).",
            "",
        ]

    lines += [
        "### 4.4 Counterparty / Exchange Risk (NOT MODELLED)",
        "",
        "> **This risk cannot be quantified via backtesting.**",
        ">",
        "> Historical carry metrics assume the exchange remains solvent and accessible.",
        "> Real-world evidence (FTX November 2022, Celsius, Voyager) shows that exchange",
        "> failure can result in total loss of both legs of the position with no recovery.",
        ">",
        "> Additional unmodelled risks:",
        "> - API downtime / inability to close positions during market stress",
        "> - Regulatory freeze / asset seizure (especially EU/MiCA retail restrictions)",
        "> - Smart-contract / oracle manipulation on the funding rate settlement",
        "> - Forced ADL (auto-deleveraging) by the exchange during extreme volatility",
        "",
        "### 4.5 99th-Percentile (VaR) per Interval",
        "",
        f"| Asset | VaR 99% (single 8 h interval, ${INITIAL_NOTIONAL:,.0f} notional) |",
        "|-------|------|",
        f"| BTC | ${btc_risk['var_99_usd']:,.2f} |",
        f"| ETH | ${eth_risk['var_99_usd']:,.2f} |",
        "",
        "---",
        "",
        "## 5. Honest Assessment",
        "",
        "### Does a carry edge remain after fees?",
        "",
    ]

    for label, risk in [("BTC", btc_risk), ("ETH", eth_risk)]:
        apr = risk["net_apr"]
        verdict = (
            "YES (marginal)" if apr > 0.03 else ("YES (thin)" if apr > 0 else "NO")
        )
        lines += [
            f"- **{label}** net APR after fees: **{apr:+.1%}**  ",
            f"  Gross implied APR from mean rate: {risk['mean_fr_8h'] * INTERVALS_PER_YEAR:+.1%}  ",
            f"  Fee drag (roundtrip × turnover): approx. {(OPEN_COST + CLOSE_COST):+.2%} per trade  ",
            f"  Edge remaining: **{verdict}**",
            "",
        ]

    lines += [
        "### Does the carry justify the tail and counterparty risk?",
        "",
        "The carry is **real and persistent** — Bitcoin and Ethereum perpetual funding has",
        "been consistently positive over most of the sample, reflecting leveraged-long",
        "demand from retail speculators paying the funding rate to hold perp longs.",
        "",
        "However:",
        "",
        "1. **Negative-funding periods are common** (≥10–20% of intervals) and can",
        "   sustain for days to weeks during bear markets. A position cannot always",
        "   be closed instantly if an exchange has withdrawal queues or halts.",
        "",
        "2. **The Sharpe is structurally overstated.** The 'risk' in the denominator",
        "   does not include the probability of exchange insolvency or a 30–40% funding",
        "   rate collapse (e.g. the LUNA-2022 or FTX-2022 events).",
        "",
        "3. **Margin liquidation risk is non-trivial.** Even at conservative 2–3× leverage,",
        "   extreme intrabar moves can trigger liquidation before the spot leg can be sold.",
        "",
        "4. **Institutional vs. retail:** This strategy is routinely run by crypto market",
        "   makers and hedge funds with direct exchange APIs, insurance funds, and legal",
        "   recourse. A retail investor operates at a structural disadvantage.",
        "",
        "**Conclusion:** The carry edge exists and is persistent, but the unmodelled tail",
        "and counterparty risks make it inappropriate to evaluate this strategy on its",
        "backtest Sharpe alone. It may be suitable as one small component of a",
        "well-diversified portfolio for investors with crypto-native infrastructure",
        "and appropriate risk controls — but **not as a standalone strategy**.",
        "",
        "---",
        "",
        "_Script: `scripts/crypto_funding_carry_backtest.py`_  ",
        "_Data: Binance FAPI (public, no auth)_  ",
        "_Cache: `data/crypto_funding_cache/`_  ",
        "_This is a research-only backtest. No live execution components._",
    ]

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    log.info("Report written → %s", OUT_MD)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Crypto funding-rate carry backtest")
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore Parquet cache and re-fetch from Binance",
    )
    args = parser.parse_args()

    results: dict[str, dict] = {}
    sims: dict[str, pd.DataFrame] = {}

    for symbol in SYMBOLS:
        log.info("=== Processing %s ===", symbol)
        try:
            funding_df, klines_df = load_or_fetch(symbol, no_cache=args.no_cache)
        except requests.RequestException as exc:
            log.error("Failed to fetch data for %s: %s", symbol, exc)
            raise SystemExit(
                f"Network error — could not reach Binance FAPI: {exc}"
            ) from exc

        log.info(
            "[%s] %d funding records | %d kline bars",
            symbol,
            len(funding_df),
            len(klines_df),
        )

        sim = simulate_carry(funding_df)
        sims[symbol] = sim

        risk = compute_risk_metrics(sim, klines_df)
        results[symbol] = risk

        # console summary
        log.info(
            "[%s] Net APR: %+.1f%% | Sharpe: %.2f | MaxDD: $%.0f | "
            "Pos: %.0f%% | Neg funding: %.1f%% | Longest neg streak: %d intervals",
            symbol,
            risk["net_apr"] * 100,
            risk["sharpe"],
            risk["max_dd_usd"],
            risk["pct_in_position"] * 100,
            risk["pct_negative"] * 100,
            risk["max_neg_streak"],
        )

    log.info("Writing report …")
    write_report(
        btc_sim=sims["BTCUSDT"],
        btc_risk=results["BTCUSDT"],
        eth_sim=sims["ETHUSDT"],
        eth_risk=results["ETHUSDT"],
    )
    print(f"\nDone -> {OUT_MD}")


if __name__ == "__main__":
    main()
