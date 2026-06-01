"""Feed-divergence note: yfinance full-history cache vs the live Alpaca daily store.

Writes docs/results/2026_06_sector_fullhist_feed_divergence.md.

Usage:
    python scripts/_sector_fullhist_feed_divergence.py

WHY THIS EXISTS (auditor follow-up to the full-history robustness study)
------------------------------------------------------------------------
docs/results/2026_06_sector_rotation_oos_fullhist.md runs the sector-rotation
falsification on a yfinance free feed (the full SPDR history back to 1998). The
LIVE production factor instead reads output/aggregates/daily.parquet (the Alpaca
era, 2018+). For the yfinance robustness verdict to be informative about the
LIVE factor, the two feeds must AGREE on the overlapping window. This read-only
script QUANTIFIES that agreement per symbol.

WHAT IT FOUND (and self-diagnoses)
----------------------------------
It compares the live close to BOTH yfinance series — raw Close and Adj Close —
and reports which one the live store tracks. Empirically the live close matches
yfinance **Adj Close** essentially exactly (~0 bps), i.e. output/aggregates/
daily.parquet `close` is **total-return (split+dividend) adjusted**, NOT raw.
That is a useful correction: the live falsification doc
(docs/results/2026_06_sector_rotation_oos.md) and the fullhist robustness doc
both describe the live store as "raw close" — it is in fact adjusted. This does
NOT change any REJECTED verdict (an adjusted, i.e. total-return, book still fails
to beat SPY on a deflated, significant basis; the SPY benchmark is the same
basis), but the price-type wording in those docs is superseded by this note.

SCOPE / SAFETY
--------------
Purely diagnostic. Reads two existing parquet files, writes one short markdown
note. Touches NO production module, NO live state, NO network. Both inputs are
pre-existing parquet files.

CI: not run in CI; local one-shot.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# Reuse the cache path + symbol set from the committed full-history harness so
# there is a single source of truth (no Doppelstruktur — Rule 50). Importing it
# is side-effect-free (its main() is __main__-guarded).
import _oos_wf_sector_rotation_fullhist as fh  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("sector_fullhist_feed_divergence")

DAILY = ROOT / "output" / "aggregates" / "daily.parquet"
OUT_MD = ROOT / "docs" / "results" / "2026_06_sector_fullhist_feed_divergence.md"

# A matched-basis pooled median below this is treated as "feeds are the same
# series" (rounding/tick noise only). Adjusted-to-adjusted came in at ~0.00 bps.
MATCH_BPS = 5.0


def _midnight_utc(s: pd.Series) -> pd.Series:
    """Normalize any (tz-naive UTC-wallclock or tz-aware) timestamp series to
    midnight-UTC tz-aware, so the yfinance ET-derived 04:00/05:00 stamps align to
    the daily.parquet midnight grid on the same calendar date."""
    return pd.to_datetime(s, utc=True).dt.normalize()


def _load_yf() -> pd.DataFrame:
    if not fh.CACHE.exists():
        raise FileNotFoundError(
            f"yfinance cache not found: {fh.CACHE} — run "
            "scripts/_oos_wf_sector_rotation_fullhist.py first."
        )
    df = pd.read_parquet(fh.CACHE)[
        ["timestamp", "symbol", "close_raw", "close_adj"]
    ].copy()
    df["timestamp"] = _midnight_utc(df["timestamp"])
    return df


def _load_daily() -> pd.DataFrame:
    if not DAILY.exists():
        raise FileNotFoundError(f"live daily store not found: {DAILY}")
    df = pd.read_parquet(DAILY)
    df = df[df["symbol"].isin(fh.WANT)][["timestamp", "symbol", "close"]].copy()
    df["timestamp"] = _midnight_utc(df["timestamp"])
    df = df.rename(columns={"close": "live_close"})
    # daily.parquet may legitimately carry one partial in-progress bar per symbol;
    # keep the last value per (symbol, day) defensively.
    df = df.sort_values(["symbol", "timestamp"]).drop_duplicates(
        subset=["symbol", "timestamp"], keep="last"
    )
    return df


def _rel_bps(a: pd.Series, b: pd.Series) -> pd.Series:
    """Relative |a-b|/b in basis points, on rows where b>0 and both finite."""
    return (a - b).abs() / b * 1e4


def _per_symbol_divergence(yf: pd.DataFrame, live: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for sym in fh.WANT:
        a = yf[yf["symbol"] == sym][["timestamp", "close_raw", "close_adj"]]
        b = live[live["symbol"] == sym][["timestamp", "live_close"]]
        m = a.merge(b, on="timestamp", how="inner").dropna(
            subset=["close_raw", "close_adj", "live_close"]
        )
        m = m[m["live_close"] > 0]
        if m.empty:
            rows.append(
                dict(
                    symbol=sym,
                    n_common=0,
                    span_lo=None,
                    span_hi=None,
                    raw_med_bps=np.nan,
                    adj_med_bps=np.nan,
                    matched_med_bps=np.nan,
                    matched_max_bps=np.nan,
                    matched_max_date=None,
                )
            )
            continue
        raw_bps = _rel_bps(m["close_raw"], m["live_close"])
        adj_bps = _rel_bps(m["close_adj"], m["live_close"])
        # The live store tracks whichever yfinance series is closer in the median.
        use_adj = float(adj_bps.median()) <= float(raw_bps.median())
        matched = adj_bps if use_adj else raw_bps
        imax = int(matched.values.argmax())
        rows.append(
            dict(
                symbol=sym,
                n_common=int(len(m)),
                span_lo=m["timestamp"].min().date(),
                span_hi=m["timestamp"].max().date(),
                raw_med_bps=float(raw_bps.median()),
                adj_med_bps=float(adj_bps.median()),
                matched_med_bps=float(matched.median()),
                matched_max_bps=float(matched.max()),
                matched_max_date=m["timestamp"].iloc[imax].date(),
            )
        )
    return pd.DataFrame(rows)


def _fmt(x: float, nd: int = 1) -> str:
    return "—" if not np.isfinite(x) else f"{x:.{nd}f}"


def _write_report(div: pd.DataFrame) -> None:
    have = div[div["n_common"] > 0]
    if have.empty:
        raise RuntimeError("no overlapping (symbol, day) rows between the two feeds")

    overlap_lo = min(d for d in have["span_lo"])
    overlap_hi = max(d for d in have["span_hi"])
    raw_pooled = float(np.nanmedian(have["raw_med_bps"]))
    adj_pooled = float(np.nanmedian(have["adj_med_bps"]))
    matched_pooled = float(np.nanmedian(have["matched_med_bps"]))
    worst = have.loc[have["matched_max_bps"].idxmax()]

    # Which yfinance series does the live store track? (adj => total-return)
    live_is_adjusted = adj_pooled <= raw_pooled
    basis = "total-return (split+dividend adjusted)" if live_is_adjusted else "raw"
    feeds_match = matched_pooled < MATCH_BPS

    if feeds_match and live_is_adjusted:
        verdict = (
            f"On the MATCHING basis the two feeds are effectively identical: pooled median "
            f"divergence is **{matched_pooled:.2f} bps** (worst single symbol-day "
            f"**{worst['matched_max_bps']:.2f} bps**, {worst['symbol']} {worst['matched_max_date']}). "
            f"The live close matches yfinance **Adj Close**, not raw Close — i.e. "
            f"`output/aggregates/daily.parquet` `close` is **{basis}**. Feed-independence is "
            f"therefore established on the correct (total-return) basis: the yfinance full-history "
            f"robustness study sits on prices materially identical to the live Alpaca store over "
            f"the shared window, so its REJECTED verdict carries over as a fair cross-check.  \n\n"
            f"**Correction (supersedes the price-type wording in the prior docs):** both "
            f"`docs/results/2026_06_sector_rotation_oos.md` and "
            f"`docs/results/2026_06_sector_rotation_oos_fullhist.md` describe the live store as "
            f'"raw close". It is in fact total-return adjusted. This does NOT change any verdict '
            f"(an adjusted/total-return book still fails to beat SPY on a deflated, significant "
            f"basis, and the SPY benchmark uses the same basis), and it means the live falsification "
            f"was already on total-return prices — so the fullhist `adj` mode, not `raw`, is the "
            f"true live-methodology match."
        )
    elif feeds_match:
        verdict = (
            f"On the matching basis the feeds agree to **{matched_pooled:.2f} bps** pooled median, "
            f"and the live store tracks yfinance **raw Close** (`close` basis = {basis}). "
            f"Feed-independence holds: the robustness study's `raw` mode is the live-methodology "
            f"match and its REJECTED verdict carries over."
        )
    else:
        verdict = (
            f"The feeds DIVERGE non-trivially even on the closest basis ({basis}): pooled median "
            f"**{matched_pooled:.2f} bps**. The yfinance robustness study cannot be treated as a "
            f"drop-in proxy for the live Alpaca store without this caveat — read its verdict as "
            f"feed-specific."
        )

    parts: list[str] = [
        "# Feed divergence — yfinance full-history cache vs live Alpaca daily store",
        "",
        "Run date (UTC): 2026-06-01  ",
        "**Status: DIAGNOSTIC NOTE.** Bounds how far the free yfinance feed used by the "
        "full-history robustness study diverges from the live `output/aggregates/daily.parquet` "
        "(Alpaca) that the production `sector_rotation_bias` factor actually reads. It also "
        "self-diagnoses which adjustment basis the live store uses.  ",
        "",
        f"Symbols: {int((div['n_common'] > 0).sum())} of {len(fh.WANT)} "
        f"({', '.join(fh.WANT)}); compared on common trading days.  ",
        f"Overlap window: **{overlap_lo} → {overlap_hi}** (the live store's full sector-ETF "
        f"coverage; the yfinance cache spans 1998+, so the overlap = the entire live window).  ",
        f"Live-store basis (self-diagnosed): **{basis}** — live `close` tracks yfinance "
        f"**{'Adj Close' if live_is_adjusted else 'raw Close'}**.  ",
        f"Pooled median divergence — raw-vs-live: **{raw_pooled:.1f} bps** · adj-vs-live: "
        f"**{adj_pooled:.2f} bps** · matched basis: **{matched_pooled:.2f} bps**.  ",
        "",
        "## Verdict",
        "",
        verdict,
        "",
        "## Per-symbol divergence (basis points)",
        "",
        "_`raw bps` = yfinance raw Close vs live; `adj bps` = yfinance Adj Close vs live; the "
        "smaller identifies the live basis. `matched max` = worst single day on the matched basis._",
        "",
        "| Symbol | Common bars | Overlap | raw bps (med) | adj bps (med) | matched (med) | matched (max) | max date |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in div.iterrows():
        if r["n_common"] == 0:
            parts.append(f"| {r['symbol']} | 0 | — (no overlap) | — | — | — | — | — |")
            continue
        parts.append(
            f"| {r['symbol']} | {r['n_common']} | {r['span_lo']}→{r['span_hi']} | "
            f"{_fmt(r['raw_med_bps'])} | {_fmt(r['adj_med_bps'], 2)} | "
            f"{_fmt(r['matched_med_bps'], 2)} | {_fmt(r['matched_max_bps'], 2)} | "
            f"{r['matched_max_date']} |"
        )

    parts += [
        "",
        "## Caveats (binding)",
        "",
        "- **Self-diagnosed basis.** The script does not assume raw or adjusted; it compares the "
        "live close to BOTH yfinance series and reports whichever matches. The matched basis came "
        "in at ~0 bps, the other at hundreds of bps (the cumulative dividend adjustment), which is "
        "what reveals the live store is total-return adjusted.",
        "- **Agreement ≠ identical corporate-action handling on every day.** A ~0 bps median means "
        "the feeds agree on the level; per-symbol matched-max + date are listed so any outlier day "
        "is inspectable.",
        "- **Verdicts unaffected.** The price-type correction changes wording, not numbers: the "
        "live and fullhist REJECTED verdicts stand. Total-return (adjusted) is the more correct "
        'backtest basis anyway; only the prior docs\' "raw" label was wrong.',
        "- **Read-only.** No production module, live state or network touched.",
        "",
        "---",
        "_Script: `scripts/_sector_fullhist_feed_divergence.py` (read-only diagnostic; reuses the "
        "cache path + symbol set from `scripts/_oos_wf_sector_rotation_fullhist.py`)._  ",
        "_Inputs: `output/research/sector_fullhist_yf.parquet` (yfinance, gitignored) vs "
        "`output/aggregates/daily.parquet` (live Alpaca store)._  ",
        "_Companion to: `docs/results/2026_06_sector_rotation_oos_fullhist.md` and the live verdict "
        "`docs/results/2026_06_sector_rotation_oos.md`._  ",
    ]

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(parts), encoding="utf-8")
    log.info("Report written -> %s", OUT_MD)


def main() -> int:
    yf = _load_yf()
    live = _load_daily()
    log.info(
        "yfinance rows=%d (%d syms); live rows=%d (%d syms)",
        len(yf),
        yf["symbol"].nunique(),
        len(live),
        live["symbol"].nunique(),
    )
    div = _per_symbol_divergence(yf, live)
    _write_report(div)
    for _, r in div.iterrows():
        log.info(
            "[%s] common=%d raw_med=%s bps adj_med=%s bps matched_med=%s bps",
            r["symbol"],
            r["n_common"],
            _fmt(r["raw_med_bps"]),
            _fmt(r["adj_med_bps"], 2),
            _fmt(r["matched_med_bps"], 2),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
