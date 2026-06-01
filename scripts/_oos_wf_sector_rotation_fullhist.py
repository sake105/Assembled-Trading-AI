"""Full-history OOS robustness extension of the sector_rotation_bias falsification.

Writes docs/results/2026_06_sector_rotation_oos_fullhist.md.

Usage:
    python scripts/_oos_wf_sector_rotation_fullhist.py
    python scripts/_oos_wf_sector_rotation_fullhist.py --refetch   # ignore cache

WHY THIS EXISTS (the two auditor follow-ups)
--------------------------------------------
The live-store falsification (scripts/_oos_wf_sector_rotation.py ->
docs/results/2026_06_sector_rotation_oos.md) REJECTED all three sector-rotation
books, but it carried two explicit, auditor-flagged limitations:

  (1) History starts 2018 — only the live offline store's depth (the
      Alpaca/master-panel era), NOT the full SPDR history. ~7 OOS folds.
  (2) Prices are RAW close (dividend-omitted), so absolute CAGRs understate
      total return.

This harness addresses BOTH as a *robustness cross-check*: it sources a DEEPER
history (the full Select-Sector-SPDR record back to their 1998 inception) from a
DIFFERENT feed (yfinance) and runs the identical methodology in two price modes:
  - raw  : raw Close — PIT-clean and methodology-matched to the live verdict.
  - adj  : Adj Close (total-return) — removes the dividend-omission CAGR
           understatement (auditor follow-up #2).

It answers ONE question: does the 2018+ REJECTED verdict flip when the factor is
given ~3.5x more history and total-return prices? If the verdict holds, the
live-store rejection is not an artifact of the short window or raw close.

NOT THE PRODUCTION VERDICT
--------------------------
This is a HISTORICAL-ROBUSTNESS EXTENSION, deliberately separated from the live
verdict. The production factor reads output/aggregates/daily.parquet (the live
store), NOT yfinance. yfinance is a free feed of differing provenance; its
adjustments and back-history can diverge from the Alpaca store. The live verdict
remains the one in docs/results/2026_06_sector_rotation_oos.md.

METHODOLOGY REUSE (no Doppelstruktur — Rule 50)
-----------------------------------------------
The walk-forward, fold simulation, frictions, edge-metric suite, ranking and
weighting are IMPORTED unchanged from scripts/_oos_wf_sector_rotation.py
(_run_wf / _simulate_fold / _edge_metrics / _spy_pooled_edge / _rank_scores_at /
_weights_for_mode, plus the report cell-builders). Importing that module is
side-effect-free (its main() is guarded by __main__). The committed module's WF
window (252/252/252), month-end rebalance, 130-bar warm-up, 5 bps/leg + 30 bps/yr
frictions and n_trials=3 DSR deflation therefore apply IDENTICALLY here. Only the
data source (yfinance full history, raw + adjusted), the period, and the report
are new.

PIT DISCIPLINE — incl. the adjusted-close case (binding honesty)
----------------------------------------------------------------
Scores are computed ONCE over the full series; compute_sector_scores uses only
trailing shifts, so row t depends only on rows <= t. The monthly signal is read
at (rebal_date - 1 bar) and positions are lagged one more bar, a 2-bar gap.

Adjusted (total-return) close does NOT introduce look-ahead for THIS signal:
the score and the L/S returns are purely RATIO-based (momentum = price ratios,
RS = ratio differences; PnL = pct_change). yfinance anchors the adjustment so
the latest bar equals raw close; that single normalization constant CANCELS in
every ratio. A ratio adj(t)/adj(t-k) therefore depends only on the corporate
actions with ex-dates INSIDE (t-k, t] — all of which are in the PAST relative to
the evaluation date t. So total-return momentum at t uses only information known
at t. The ONLY residual is that yfinance may *retroactively revise* historical
adjustments (vendor data repair) — that is a data-QUALITY caveat, not a
structural look-ahead, and is disclosed as such in the report.

SURVIVORSHIP
------------
N/A. The 8 ranked ETFs (XLK/XLF/XLE/XLV/XLI/XLU/XLP/XLY) are the *original*
Select Sector SPDRs, all live since their Dec-1998 inception, plus SPY (1993).
None delisted. (The later XLRE/XLC are not in the ranked set, so no staggered
inception inside the tested universe — anti-pattern E-030 does not apply.)

CI: not run in CI; local one-shot. No production module touched (read-only price
data via yfinance; imports the live signal + the committed harness unchanged).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# Live signal (unchanged) + committed methodology (imported, not re-implemented).
from src.assembled_core.signals.sector_rotation import (  # noqa: E402
    SECTOR_ETFS,
    compute_sector_scores,
)

import _oos_wf_sector_rotation as oos  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("oos_wf_sector_rotation_fullhist")

# ---------------------------------------------------------------------------
# Config (only data/period/report differ from the committed harness)
# ---------------------------------------------------------------------------
CACHE = ROOT / "output" / "research" / "sector_fullhist_yf.parquet"
OUT_MD = ROOT / "docs" / "results" / "2026_06_sector_rotation_oos_fullhist.md"

FETCH_START = "1998-01-01"
FETCH_END = "2026-05-31"  # yfinance end-exclusive; captures through the last trading day before this

WANT = list(SECTOR_ETFS) + ["SPY"]
PRICE_MODES = ("adj", "raw")  # adj first = total-return primary; raw = PIT-clean match
MODE_LABEL = {
    "adj": "total-return (Adj Close)",
    "raw": "raw Close (matches live methodology)",
}


# ---------------------------------------------------------------------------
# Data — yfinance full history (raw Close + Adj Close), cached to gitignored output/
# ---------------------------------------------------------------------------
def _fetch_or_load(refetch: bool = False) -> pd.DataFrame:
    """Return long-format [timestamp, symbol, close_raw, close_adj] for WANT.

    Cached to output/research/ (gitignored). A per-symbol yfinance failure is a
    WARN, not fatal, as long as >=5 sectors + SPY survive.
    """
    if CACHE.exists() and not refetch:
        df = pd.read_parquet(CACHE)
        log.info("Loaded cached yfinance panel: %d rows -> %s", len(df), CACHE)
        return df

    import yfinance as yf

    def _hist_one(sym: str) -> pd.DataFrame | None:
        """yf.Ticker().history (chart endpoint — less throttled than download).

        auto_adjust=False keeps BOTH 'Close' (raw) and 'Adj Close' (total-return).
        Per-symbol backoff on 429; None means this symbol could not be fetched.
        """
        backoff = [15, 30, 60, 120]
        for attempt, wait in enumerate([0, *backoff], start=1):
            if wait:
                time.sleep(wait)
            try:
                d = yf.Ticker(sym).history(
                    start=FETCH_START,
                    end=FETCH_END,
                    interval="1d",
                    auto_adjust=False,  # need BOTH Close (raw) and Adj Close
                    actions=False,
                )
            except Exception as exc:  # noqa: BLE001 — free feed; classify 429 vs other
                es = str(exc).lower()
                if "429" in es or "too many requests" in es or "rate limit" in es:
                    log.warning(
                        "[fetch] %s 429 (attempt %d/%d) — backing off",
                        sym,
                        attempt,
                        len(backoff) + 1,
                    )
                    continue
                log.warning("[fetch] %s error: %s", sym, exc)
                return None
            if d is None or d.empty:
                log.warning("[fetch] %s returned no rows", sym)
                return None
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)
            d = d.rename_axis("Date").reset_index()
            if "Close" not in d.columns or "Adj Close" not in d.columns:
                log.warning(
                    "[fetch] %s missing Close/Adj Close: %s", sym, list(d.columns)
                )
                return None
            ts = pd.to_datetime(d["Date"])
            ts = (
                ts.dt.tz_localize("UTC")
                if ts.dt.tz is None
                else ts.dt.tz_convert("UTC")
            )
            return pd.DataFrame(
                {
                    "timestamp": ts.values,
                    "symbol": sym,
                    "close_raw": pd.to_numeric(d["Close"], errors="coerce")
                    .astype(float)
                    .values,
                    "close_adj": pd.to_numeric(d["Adj Close"], errors="coerce")
                    .astype(float)
                    .values,
                }
            ).dropna(subset=["close_raw", "close_adj"])
        log.warning("[fetch] %s: all retries exhausted (429)", sym)
        return None

    frames: list[pd.DataFrame] = []
    for sym in WANT:
        out = _hist_one(sym)
        if out is None or out.empty:
            continue
        frames.append(out)
        log.info(
            "[fetch] %s: %d bars %s..%s",
            sym,
            len(out),
            out["timestamp"].min().date(),
            out["timestamp"].max().date(),
        )
        time.sleep(2.0)  # gentle pacing between symbols

    if not frames:
        raise RuntimeError("yfinance returned no data for any symbol")
    long_all = pd.concat(frames, ignore_index=True)

    present = sorted(set(long_all["symbol"]) & set(SECTOR_ETFS))
    if len(present) < 5 or "SPY" not in set(long_all["symbol"]):
        raise RuntimeError(
            f"Insufficient coverage from yfinance: sectors={present}, "
            f"SPY={'SPY' in set(long_all['symbol'])}"
        )

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    long_all.to_parquet(CACHE, index=False)
    log.info("Cached yfinance panel: %d rows -> %s", len(long_all), CACHE)
    return long_all


def _build_panel(
    long_all: pd.DataFrame, price_col: str
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Build (sector_wide, spy_close, scores) for one price mode.

    Mirrors the committed _load_sector_panel transformation exactly, but on the
    yfinance long frame and a selectable price column (close_raw / close_adj).
    """
    df = long_all[long_all["symbol"].isin(WANT)].copy()
    # Normalize to midnight UTC so the date grid matches the committed harness's
    # daily.parquet grid (its _load_sector_panel relies on midnight timestamps so
    # the month-end pd.date_range snap picks the true last trading day, not the
    # day before). yfinance .history() returns tz-aware ET timestamps that convert
    # to 05:00 UTC; .normalize() drops the intraday offset (matches the production
    # yfinance_source loader, which does the same).
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.normalize()
    df = df.sort_values(["symbol", "timestamp"]).drop_duplicates(
        subset=["symbol", "timestamp"], keep="last"
    )

    wide = (
        df.pivot_table(index="timestamp", columns="symbol", values=price_col)
        .sort_index()
        .ffill()
    )
    sector_cols = [c for c in SECTOR_ETFS if c in wide.columns]
    sector_wide = wide[sector_cols]
    spy_close = wide["SPY"]

    # PIT-safe: scores computed ONCE over the full series (trailing shifts only).
    sector_long = (
        sector_wide.reset_index()
        .melt(id_vars="timestamp", var_name="symbol", value_name="close")
        .dropna(subset=["close"])
    )
    spy_long = spy_close.reset_index()
    spy_long.columns = ["timestamp", "close"]
    spy_long["symbol"] = "SPY"

    scores = (
        compute_sector_scores(sector_long, spy_long).set_index("timestamp").sort_index()
    )
    log.info(
        "[panel:%s] %d sectors + SPY, %d bars %s..%s; scores rows=%d",
        price_col,
        len(sector_cols),
        len(wide),
        wide.index.min().date(),
        wide.index.max().date(),
        len(scores),
    )
    return sector_wide, spy_close, scores


# ---------------------------------------------------------------------------
# Run one price mode (reuses committed WF + edge suite verbatim)
# ---------------------------------------------------------------------------
def _run_mode(long_all: pd.DataFrame, price_col: str) -> dict:
    sector_wide, spy_close, scores = _build_panel(long_all, price_col)
    all_results, all_edges = {}, {}
    pooled_spy_ref = None
    for mode in oos.MODES:
        results, pooled_s, pooled_b = oos._run_wf(sector_wide, spy_close, scores, mode)
        edge = oos._edge_metrics(pooled_s, pooled_b, results)
        all_results[mode] = results
        all_edges[mode] = edge
        if pooled_spy_ref is None and len(pooled_b.dropna()) > 0:
            pooled_spy_ref = pooled_b
        log.info(
            "[%s/%s] AnnSharpe %.2f | CAGR %.1f%% | IR %.2f (t=%.2f) | DSR %.2f (pass=%s) | beta %.2f",
            price_col,
            mode,
            edge["ann_sharpe"],
            edge["cagr"] * 100,
            edge["ir"],
            edge["ir_t"],
            edge["dsr_prob"],
            edge["dsr_pass"],
            edge["beta"],
        )
    spy_edge = (
        oos._spy_pooled_edge(pooled_spy_ref) if pooled_spy_ref is not None else {}
    )
    n_bars = len(sector_wide)
    span = (sector_wide.index.min(), sector_wide.index.max())
    return dict(
        results=all_results,
        edges=all_edges,
        spy_edge=spy_edge,
        n_bars=n_bars,
        span=span,
        sectors=list(
            sector_wide.columns
        ),  # ACTUAL ranked sectors present (may be < 8 on degraded fetch)
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def _prospects(edges: dict, spy_edge: dict) -> list[str]:
    spy_sharpe = spy_edge.get("ann_sharpe", float("nan"))
    return [
        m
        for m in oos.MODES
        if (
            np.isfinite(edges[m]["ann_sharpe"])
            and np.isfinite(spy_sharpe)
            and edges[m]["ann_sharpe"] > spy_sharpe
            and edges[m]["dsr_pass"]
            and np.isfinite(edges[m]["ir_t"])
            and edges[m]["ir_t"] > 1.96
        )
    ]


def _prospects_across_modes(by_mode: dict) -> list[str]:
    """Flatten per-price-mode PROSPECTs into 'pricemode:book' tags across all modes."""
    tags: list[str] = []
    for pm in PRICE_MODES:
        for m in _prospects(by_mode[pm]["edges"], by_mode[pm]["spy_edge"]):
            tags.append(f"{pm}:{m}")
    return tags


def _overall_verdict(prospects_any: list[str]) -> str:
    """Overall verdict paragraph. Pure function of the PROSPECT tag list so the
    (normally dead) PROSPECT branch is unit-testable without a full WF run."""
    if prospects_any:
        return (
            f"**{len(prospects_any)} book/price-mode combination(s) show a PROSPECT** "
            f"({', '.join(prospects_any)}). This would WEAKEN the live-store rejection and "
            f"warrant a deeper look — but it is NOT a production claim: it is on a different "
            f"feed (yfinance) and still needs CI validation and a live-store re-confirmation "
            f"before any non-zero `sector_rotation_bias` weight."
        )
    return (
        "**ALL books REJECTED in BOTH price modes** — neither deeper history (back to the "
        "1998 SPDR inception, ~3.5x the live window) NOR total-return (adjusted) prices "
        "reveal a multiple-testing-deflated, statistically-significant edge over SPY. The "
        "live-store 2018+ rejection (docs/results/2026_06_sector_rotation_oos.md) is "
        "therefore NOT an artifact of the short window or the raw-close dividend omission. "
        "The production regime weight for `sector_rotation_bias` stays ~0."
    )


def _write_report(by_mode: dict) -> None:
    adj = by_mode["adj"]
    raw = by_mode["raw"]
    span_lo, span_hi = adj["span"]

    # Honesty: report the ACTUAL ranked universe, not the constant. A degraded
    # yfinance fetch (>=5 sectors tolerated) must not silently masquerade as a
    # full-universe robustness claim — surface the shortfall in the header.
    present_sectors = adj.get("sectors", list(SECTOR_ETFS))
    universe_complete = len(present_sectors) == len(SECTOR_ETFS)
    universe_warning = (
        ""
        if universe_complete
        else (
            f"  \n> **DEGRADED UNIVERSE:** only {len(present_sectors)}/{len(SECTOR_ETFS)} ranked "
            f"sector ETFs were available from yfinance ({present_sectors}); "
            f"missing {sorted(set(SECTOR_ETFS) - set(present_sectors))}. The rotation ranking "
            f"is over a partial sector set — treat this robustness verdict with extra caution."
        )
    )

    prospects_any = _prospects_across_modes(by_mode)
    overall = _overall_verdict(prospects_any)

    parts: list[str] = [
        "# sector_rotation_bias — Full-History OOS Robustness (1998–2026, total-return + raw)",
        "",
        "Run date (UTC): 2026-06-01  ",
        "**Status: HISTORICAL-ROBUSTNESS EXTENSION — NOT the production verdict.** The live "
        "factor reads `output/aggregates/daily.parquet` (Alpaca era, 2018+); the binding "
        "falsification is `docs/results/2026_06_sector_rotation_oos.md`. This study sources a "
        "DIFFERENT feed (yfinance) and DEEPER history purely to test whether that REJECTED "
        "verdict is an artifact of the short 2018+ window or of raw (dividend-omitted) close.  ",
        "",
        "Data: yfinance — full Select-Sector-SPDR history, raw Close + Adj Close  ",
        f"Universe: {len(present_sectors)} SPDR sector ETFs {present_sectors} + SPY "
        f"(benchmark/RS factor)  " + universe_warning,
        f"History: {adj['n_bars']} bars {span_lo.date()} → {span_hi.date()} "
        f"(~{adj['n_bars'] / 252.0:.0f}y; vs the live store's ~8y 2018+)  ",
        "Signal: production `compute_sector_scores` (3m·0.50 + 6m·0.30 + 20d-RS·0.20), "
        "top-3 long / bottom-2 short — unchanged  ",
        f"WF: {oos.TRAIN_WINDOW}/{oos.TEST_WINDOW}/{oos.STEP_SIZE} (train/test/step), month-end "
        "rebalance, 2-bar data→fill gap — IMPORTED unchanged from the live harness  ",
        f"Frictions: {oos.COST_BPS:.0f} bps/leg turnover, {oos.BORROW_BPS_ANNUAL:.0f} bps/yr "
        "short borrow — IMPORTED unchanged  ",
        f"DSR deflation: n_trials = {oos.N_TRIALS_DSR} per price mode (one fixed signal config, "
        "3 portfolio constructions). The 2 price modes are a robustness cross-check, not "
        "independent parameter searches; read a marginal pass conservatively.  ",
        "",
        "## Honesty / caveats (binding)",
        "",
        "- **Two price modes.** `adj` = Adj Close (total-return) — fixes the absolute-CAGR "
        "understatement the live verdict flagged (auditor follow-up #2). `raw` = raw Close — "
        "PIT-clean and methodology-matched to the live verdict for apples-to-apples.",
        "- **Adjusted-close is PIT-clean for THIS signal.** Score and L/S returns are purely "
        "ratio-based; yfinance's today-anchored normalization constant cancels in every ratio, "
        "and a ratio adj(t)/adj(t−k) depends only on corporate actions with ex-dates inside "
        "(t−k, t] — all in the past relative to t. The only residual is possible *retroactive* "
        "yfinance adjustment revisions, a data-QUALITY caveat (free feed), not a structural "
        "look-ahead.",
        "- **Feed provenance.** yfinance ≠ the production Alpaca store; back-history splits/"
        "adjustments and exact closes can differ. This is why the study is kept SEPARATE from "
        "the live verdict rather than overriding it.",
        "- **Survivorship N/A.** All 8 ranked ETFs are original Select Sector SPDRs live since "
        "Dec-1998; SPY since 1993. None delisted; no staggered inception in the ranked set.",
        "- **CI:** not run; local one-shot. No production module touched. WF/friction/edge "
        "methodology imported unchanged from `scripts/_oos_wf_sector_rotation.py`.",
        "",
    ]

    for pm in PRICE_MODES:
        d = by_mode[pm]
        edges, spy_edge = d["edges"], d["spy_edge"]
        verdict_lines = [oos._verdict_line(m, edges[m], spy_edge) for m in oos.MODES]
        parts += [
            f"## Verdict — {MODE_LABEL[pm]}",
            "",
            *verdict_lines,
            "",
            f"### OOS-Edge table — {MODE_LABEL[pm]}",
            "",
            oos._edge_table(edges, spy_edge),
            "",
        ]

    parts += [overall, ""]

    # Per-fold tables: primary (total-return) mode for all 3 books; raw mode for
    # the dollar-neutral sector_ls (purest ranking test) for the apples-to-apples.
    parts += ["## Per-fold detail — total-return (Adj Close)", ""]
    for mode in oos.MODES:
        parts += [
            f"### {oos.MODE_TITLES[mode]}",
            "",
            oos._fold_table(adj["results"][mode]),
            "",
        ]
    parts += [
        "## Per-fold detail — raw Close (sector_ls, methodology-matched)",
        "",
        oos._fold_table(raw["results"]["sector_ls"]),
        "",
    ]

    parts += [
        "---",
        "_Script: `scripts/_oos_wf_sector_rotation_fullhist.py` (read-only research harness; "
        "imports the live `compute_sector_scores` AND the committed WF/edge methodology from "
        "`scripts/_oos_wf_sector_rotation.py` unchanged; no production file modified)._  ",
        "_Live verdict (binding): `docs/results/2026_06_sector_rotation_oos.md`._  ",
        "_Signal: `src/assembled_core/signals/sector_rotation.py`._  ",
        "_Data cache (gitignored): `output/research/sector_fullhist_yf.parquet` (yfinance)._  ",
    ]

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(parts), encoding="utf-8")
    log.info("Report written -> %s", OUT_MD)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refetch",
        action="store_true",
        help="Ignore the yfinance cache and re-download",
    )
    args = parser.parse_args()

    long_all = _fetch_or_load(refetch=args.refetch)

    by_mode = {}
    for pm in PRICE_MODES:
        price_col = "close_adj" if pm == "adj" else "close_raw"
        log.info("=== Running price mode: %s (%s) ===", pm, price_col)
        by_mode[pm] = _run_mode(long_all, price_col)

    _write_report(by_mode)

    for pm in PRICE_MODES:
        d = by_mode[pm]
        log.info(
            "DONE[%s]. SPY Sharpe %.2f / CAGR %.1f%%. Sector Sharpes: %s",
            pm,
            d["spy_edge"].get("ann_sharpe", float("nan")),
            d["spy_edge"].get("cagr", float("nan")) * 100,
            {m: round(d["edges"][m]["ann_sharpe"], 2) for m in oos.MODES},
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
