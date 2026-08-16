"""Point-in-time price panel adapter (survivorship-aware).

WHY THIS EXISTS
---------------
The project runs on two disjoint price universes, and the worse one carries the
money:

  - operational: ``output/aggregates/daily.parquet`` — 220 symbols, of which
    ZERO are delisted or merged names. Every ``data/universe/*.csv`` is 100%
    ``status: active``. By construction this panel has never seen a loser.
  - research:    ``research/mandat/data/prices_verdict.parquet`` — 1,167 symbols
    selected point-in-time from S&P 500 membership snapshots 1996-2026, and it
    does contain the failures (BSC, SIVB, FRC, CELG, RTN, ATVI, ...).

The measured cost of that gap, from ``research/mandat2/BEFUND_DATENQUALITAET.md``:
**2.36 to 2.90 percentage points p.a.** of pure composition bias, against a
campaign decision margin of 1.5 pp. The bias is larger than the effect being
measured.

This module makes the research panel loadable from the operational code path.
It is the survivorship analogue of a fix that was already made in architecture
but never in data (``KNOWN_ISSUES.md`` §0.1).

WHAT THIS MODULE DOES NOT CLAIM
-------------------------------
"Survivorship-free" would be too strong, and the research authors retracted
that wording themselves (``BEFUND_P12_INTRADAY.md``): the SELECTION is
point-in-time correct, the COVERAGE is not complete, and the gap is not
neutral. Known residuals:

  - the constituent history starts 1996; nothing before that
  - 36 of 1,202 requested symbols never returned data (``pull_eodhd_report.json``)
  - ticker recycling: 29 columns of the source panel carry TWO companies
    (E-114 / E-117), because a ticker is a time-series attribute, not a key
  - 48,380 symbol-days sit on a wrong price scale (upper bound not established)

Treat this panel as *materially better*, not as *clean*.

THE OHLC PROBLEM — READ BEFORE USING
------------------------------------
The source panel has ``close`` only. No open, high, low. This adapter
synthesises ``open = high = low = close`` so that the frame satisfies the
``load_eod_prices`` contract, and that synthesis is a LIE for every consumer
that reads a bar's range:

  - ATR (``features/ta_factors_core.py``) is understated — see the measurement
    below; it does NOT become zero, which is what makes it dangerous
  - candlestick patterns (``features/ta_candlestick.py``) become meaningless
  - intraday range, gap and spread models degenerate

Trading survivorship bias for an invisible feature bias would be a bad deal, so
this is made loud rather than convenient:

  - a ``logger.warning`` on every load
  - ``df.attrs["ohlc_synthetic"] = True`` on the returned frame
  - :func:`assert_no_synthetic_ohlc` for callers that must refuse such a frame

MEASURED EFFECT ON ATR (2026-08-15) — the earlier claim in this docstring that
"ATR degenerates to 0" was WRONG and is corrected here. True Range is
``max(high-low, |high-prev_close|, |low-prev_close|)``; with
``open=high=low=close`` the first term vanishes but the other two collapse to
the close-to-close move, which is not zero. Measured against real OHLC over the
last 300 bars of 8 liquid symbols, synthetic ATR comes out at a **median 61% of
true ATR** (range 0.40–0.64). ``trailing_stops.py:179`` additionally floors the
case ``atr == 0`` with a 10% fallback, so there is no zero-distance stop and no
runaway liquidation.

The real hazard is therefore quieter and harder to catch: ATR-scaled trailing
stops sit roughly 39% TIGHTER than intended and exit early, systematically and
without any error. That is exactly the class of bias this module must not trade
survivorship bias for.

FRESHNESS
---------
The source panel is frozen at **2026-07-06** (last pull 2026-07-07) and cannot
be extended: the EODHD access ended 2026-08-05 (see ``docs/DATENZUGANG_STATUS.md``).
This panel is therefore for BACKTEST AND RESEARCH ONLY. It must never feed the
live path, where a 40-day-old price is a real loss, not a statistic.
"""

from __future__ import annotations

import csv
import logging
from bisect import bisect_right
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Repo root: src/assembled_core/data/pit_prices.py -> up 4
REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_PIT_PANEL = (
    REPO_ROOT / "research" / "mandat" / "data" / "prices_verdict.parquet"
)
DEFAULT_CONSTITUENTS = (
    REPO_ROOT / "research" / "mandat" / "data" / "sp500_historical_constituents.csv"
)

#: Column order of output/aggregates/daily.parquet. The adapter emits exactly
#: this so that load_eod_prices and every downstream consumer accept the frame
#: without a single change to prices_ingest.py.
DAILY_SCHEMA = [
    "timestamp",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
]

#: Marker set on frames whose OHLC was synthesised from close.
SYNTHETIC_OHLC_ATTR = "ohlc_synthetic"

#: Last bar available in the frozen source panel.
PANEL_FROZEN_AT = "2026-07-06"


class SyntheticOHLCError(RuntimeError):
    """Raised when a range-dependent consumer is handed a synthetic-OHLC frame."""


def load_pit_prices(
    symbols: list[str] | None = None,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    panel_path: Path | None = None,
    warn_synthetic: bool = True,
) -> pd.DataFrame:
    """Load the point-in-time price panel in ``daily.parquet`` schema.

    Args:
        symbols: restrict to these symbols; ``None`` loads all 1,167.
        start / end: inclusive timestamp bounds.
        panel_path: override the source parquet.
        warn_synthetic: emit the synthetic-OHLC warning (set False only in
            tests that assert on log output).

    Returns:
        DataFrame with :data:`DAILY_SCHEMA` columns, ``timestamp`` tz-aware UTC,
        ``attrs[SYNTHETIC_OHLC_ATTR] = True``.

    The read is column- and row-filtered through pyarrow. The source has
    6,096,910 rows in 6 row groups; loading it whole to slice afterwards would
    be wasteful and, on a constrained runner, fatal.
    """
    path = Path(panel_path) if panel_path is not None else DEFAULT_PIT_PANEL
    if not path.exists():
        raise FileNotFoundError(
            f"PIT price panel not found: {path}. It is produced by "
            f"research/mandat/pull_eodhd_verdict.py, which needs EODHD access "
            f"(unavailable since 2026-08-05 — see docs/DATENZUGANG_STATUS.md)."
        )

    filters = []
    if symbols:
        filters.append(("symbol", "in", set(symbols)))
    if start is not None:
        filters.append(("timestamp", ">=", pd.Timestamp(start, tz="UTC")))
    if end is not None:
        filters.append(("timestamp", "<=", pd.Timestamp(end, tz="UTC")))

    df = pd.read_parquet(
        path,
        columns=["timestamp", "symbol", "close", "volume"],
        filters=filters or None,
    )

    if df.empty:
        logger.warning(
            "[pit_prices] no rows matched (symbols=%s, start=%s, end=%s) in %s",
            "all" if not symbols else len(symbols),
            start,
            end,
            path,
        )
        empty = pd.DataFrame(columns=DAILY_SCHEMA)
        empty.attrs[SYNTHETIC_OHLC_ATTR] = True
        return empty

    # as_unit("ns") is load-bearing, not cosmetic: the DAILY_SCHEMA contract
    # promises the same dtype as daily.parquet (datetime64[ns, UTC]). Newer
    # pandas keeps whatever resolution pyarrow hands back from parquet — often
    # microseconds — so without the cast the dtype depends on the installed
    # pandas version. That is exactly what broke CI: local pandas 2.2.3
    # (requirements.txt pin) yielded ns, CI's `pip install -e ".[dev]"`
    # resolves the pyproject RANGE to a newer pandas and yielded us, on both
    # runners (Rule-40 drift class: range vs. pin).
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.as_unit("ns")

    # --- synthetic OHLC ---------------------------------------------------
    # close is EODHD adjusted_close, i.e. total-return adjusted, which is the
    # SAME semantics as daily.parquet.close (pipeline/io.py:33-41). So no price
    # conversion is needed here; the only gap is the missing range.
    close = df["close"].astype("float64")
    df["open"] = close
    df["high"] = close
    df["low"] = close
    df["close"] = close
    df["adj_close"] = close
    df["volume"] = df["volume"].astype("float64")

    df = df[DAILY_SCHEMA].sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    df.attrs[SYNTHETIC_OHLC_ATTR] = True

    if warn_synthetic:
        logger.warning(
            "[pit_prices] loaded %d rows / %d symbols with SYNTHETIC OHLC "
            "(open=high=low=close). ATR-, candlestick- and range/spread-based "
            "features are BIASED on this frame: ATR measures ~61%% of its true "
            "value (median, 8 symbols), so ATR-scaled stops sit ~39%% too tight. "
            "Panel is frozen at %s and must not feed the live path.",
            len(df),
            df["symbol"].nunique(),
            PANEL_FROZEN_AT,
        )

    return df


def is_synthetic_ohlc(df: pd.DataFrame) -> bool:
    """True if ``df`` carries synthesised OHLC.

    Checks the ``attrs`` marker first, then falls back to inspecting the data.
    The fallback is not belt-and-braces, it is the load-bearing half:
    ``DataFrame.attrs`` survives ``assign`` / boolean masks / ``reset_index``
    but is DROPPED by ``merge``, ``concat`` with mixed attrs, ``pivot`` and
    ``groupby().apply()`` — and those sit squarely between this loader and any
    feature or sizing path. A marker that evaporates exactly where it matters
    would make this guard decorative (E-142).

    The data check is cheap and unambiguous: real bars essentially never have
    ``open == high == low == close`` on EVERY row, while synthesised ones have
    it on every row by construction.

    Narrow exception handling on purpose: a genuine bug must not be able to
    answer "not synthetic", which is the one direction this guard must never
    fail in.
    """
    try:
        if bool(df.attrs.get(SYNTHETIC_OHLC_ATTR, False)):
            return True
    except (AttributeError, TypeError):  # pragma: no cover - defensive
        return False

    try:
        if df.empty or not {"open", "high", "low", "close"} <= set(df.columns):
            return False
        return bool(
            (df["open"] == df["close"]).all()
            and (df["high"] == df["close"]).all()
            and (df["low"] == df["close"]).all()
        )
    except (AttributeError, TypeError, KeyError):  # pragma: no cover - defensive
        return False


def assert_no_synthetic_ohlc(df: pd.DataFrame, consumer: str) -> None:
    """Refuse a synthetic-OHLC frame on behalf of a range-dependent consumer.

    Call this from anything that reads high/low. Failing loudly here is the
    whole point: a silent ATR of 0 is the kind of bias that survives review
    because it never raises (the E-142/E-143 family).
    """
    if is_synthetic_ohlc(df):
        raise SyntheticOHLCError(
            f"{consumer} reads bar ranges (high/low), but this frame has "
            f"SYNTHETIC OHLC (open=high=low=close) from the PIT panel. "
            f"ATR/candlestick/spread results would be silently meaningless. "
            f"Use output/aggregates/daily.parquet for range-dependent features."
        )


def _load_membership_snapshots(
    constituents_path: Path | None = None,
) -> list[tuple[pd.Timestamp, frozenset[str]]]:
    """Read the S&P 500 membership snapshots, sorted ascending by date."""
    path = (
        Path(constituents_path)
        if constituents_path is not None
        else DEFAULT_CONSTITUENTS
    )
    if not path.exists():
        raise FileNotFoundError(
            f"PIT constituent history not found: {path}. Source is the "
            f"fja05680/sp500 GitHub repository; there is no pull script in "
            f"this repo, it must be obtained separately "
            f"(see tests/mandat2_daten_guard.py)."
        )

    snaps: list[tuple[pd.Timestamp, frozenset[str]]] = []
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            day = pd.Timestamp(row["date"], tz="UTC")
            members = frozenset(
                t.strip() for t in row["tickers"].split(",") if t.strip()
            )
            snaps.append((day, members))
    snaps.sort(key=lambda item: item[0])
    return snaps


def pit_members(
    as_of: str | pd.Timestamp,
    *,
    constituents_path: Path | None = None,
) -> list[str]:
    """Index members as of ``as_of``, from the most recent snapshot at or before it.

    Returns a SORTED list. Sorting is deliberate: the snapshots are stored as
    frozensets, and iterating a frozenset is order-unstable across runs — that
    was E-051 (frozenset determinism bug), and it produced results that could
    not be reproduced.

    Raises:
        ValueError: if ``as_of`` predates the first snapshot (1996-01-02).
            Returning an empty list there would silently look like "no members".
    """
    stamp = pd.Timestamp(as_of)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")

    snaps = _load_membership_snapshots(constituents_path)
    dates = [d for d, _ in snaps]
    idx = bisect_right(dates, stamp) - 1
    if idx < 0:
        raise ValueError(
            f"as_of={stamp.date()} predates the first membership snapshot "
            f"({dates[0].date()}). There is no point-in-time universe for that "
            f"date; do not fall back to today's members."
        )
    return sorted(snaps[idx][1])


def build_pit_universe_history(
    panel_path: Path | None = None,
    *,
    coverage_grace_days: int = 5,
) -> pd.DataFrame:
    """Derive a ``data/universe`` history table from the PIT panel.

    Reuses :func:`assembled_core.data.universe.build_universe_history_from_prices`
    so the resulting CSV is byte-compatible with everything that already reads
    ``data/universe/<name>.csv`` — no change to ``universe.py`` required.

    The point of doing this at all: ``get_universe_members_pit`` has existed for
    months with zero production callers, because no universe file carried real
    ``end_date`` values. Every one of the 13 existing files is 100% ``active``.
    Fed from this panel it finally has delisting windows to work with.
    """
    from src.assembled_core.data.universe import build_universe_history_from_prices

    prices = load_pit_prices(panel_path=panel_path, warn_synthetic=False)
    if prices.empty:
        raise ValueError("PIT panel produced no rows; cannot build universe history")

    history = build_universe_history_from_prices(
        prices, coverage_grace_days=coverage_grace_days
    )
    logger.info(
        "[pit_prices] built universe history: %d symbols, %d with an end_date "
        "(i.e. actually left the panel)",
        len(history),
        int(history["end_date"].notna().sum()) if "end_date" in history else 0,
    )
    return history


__all__ = [
    "DAILY_SCHEMA",
    "DEFAULT_CONSTITUENTS",
    "DEFAULT_PIT_PANEL",
    "PANEL_FROZEN_AT",
    "SYNTHETIC_OHLC_ATTR",
    "SyntheticOHLCError",
    "assert_no_synthetic_ohlc",
    "build_pit_universe_history",
    "is_synthetic_ohlc",
    "load_pit_prices",
    "pit_members",
]
