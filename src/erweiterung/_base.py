"""Shared base helpers for the erweiterung package.

Bewusst klein gehalten — alles, was sonst kreuz und quer dupliziert würde.
"""

from __future__ import annotations

import functools
import hashlib
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def get_cache_dir(subdir: str = "") -> Path:
    """Cache-Verzeichnis für altdata-Downloads.

    Standard: ``output/erweiterung_cache/<subdir>``.
    Per env ``ERWEITERUNG_CACHE_DIR`` überschreibbar.
    """
    base = os.environ.get("ERWEITERUNG_CACHE_DIR", "output/erweiterung_cache")
    p = Path(base)
    if subdir:
        p = p / subdir
    p.mkdir(parents=True, exist_ok=True)
    return p


def stable_hash(*parts: Any) -> str:
    """Deterministischer Kurzhash für Cache-Keys."""
    s = "|".join(str(p) for p in parts).encode("utf-8")
    return hashlib.sha1(s).hexdigest()[:16]  # noqa: S324 — non-security


def safe_div(a: float | pd.Series, b: float | pd.Series, default: float = 0.0):
    """Division mit NaN-/0-Schutz (vektorisiert)."""
    if isinstance(a, pd.Series) or isinstance(b, pd.Series):
        out = pd.Series(a) / pd.Series(b)
        return out.replace([np.inf, -np.inf], np.nan).fillna(default)
    if b is None or b == 0 or pd.isna(b):
        return default
    out = a / b
    if not np.isfinite(out):
        return default
    return out


def winsorize(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Vektorisierte Winsorisierung für robuste Z-Scores."""
    if s.empty or s.isna().all():
        return s
    lo = s.quantile(lower)
    hi = s.quantile(upper)
    return s.clip(lo, hi)


def zscore(s: pd.Series, robust: bool = False) -> pd.Series:
    """Z-Score (optional robust via Median + MAD)."""
    if s.empty:
        return s
    if robust:
        med = s.median()
        mad = (s - med).abs().median()
        if mad == 0 or pd.isna(mad):
            return s * 0
        return 0.6745 * (s - med) / mad
    mu = s.mean()
    sd = s.std(ddof=0)
    if sd == 0 or pd.isna(sd):
        return s * 0
    return (s - mu) / sd


def cross_sectional_zscore(
    df: pd.DataFrame, value_col: str, group_col: str, by_date: str = "date"
) -> pd.Series:
    """Cross-sectional z-score per (date, group). PIT-sicher: kein Look-ahead."""
    grp = df.groupby([by_date, group_col])[value_col]
    return grp.transform(lambda x: zscore(x, robust=True))


def rate_limited(min_interval_s: float = 1.0) -> Callable:
    """Einfacher Rate-Limit-Decorator pro Funktion (Single-Process)."""

    def deco(fn: Callable) -> Callable:
        last_call: dict[str, float] = {"t": 0.0}

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            elapsed = time.monotonic() - last_call["t"]
            if elapsed < min_interval_s:
                time.sleep(min_interval_s - elapsed)
            try:
                return fn(*args, **kwargs)
            finally:
                last_call["t"] = time.monotonic()

        return wrapper

    return deco


def retry_with_backoff(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> Callable:
    """Exponential backoff retry. Fehler werden geloggt und neu geworfen, wenn alle Versuche scheitern."""

    def deco(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            delay = base_delay
            last_exc: Optional[BaseException] = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as e:
                    last_exc = e
                    if attempt == max_attempts:
                        logger.warning(
                            "[%s] giving up after %d attempts: %s",
                            fn.__name__,
                            max_attempts,
                            e,
                        )
                        raise
                    logger.info(
                        "[%s] attempt %d/%d failed (%s) — retry in %.1fs",
                        fn.__name__,
                        attempt,
                        max_attempts,
                        e,
                        delay,
                    )
                    time.sleep(delay)
                    delay = min(delay * 2, max_delay)
            if last_exc is None:
                raise RuntimeError("retry_with_backoff: invariant violation")
            raise last_exc

        return wrapper

    return deco


def to_utc_date(x: Any) -> pd.Timestamp:
    """Robust zu UTC-Tagesgranularität konvertieren."""
    ts = pd.Timestamp(x)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.normalize()


@dataclass(frozen=True)
class FetchResult:
    """Standardisiertes Ergebnis von altdata-Fetchern."""

    df: pd.DataFrame
    source: str
    as_of: pd.Timestamp
    rows: int
    notes: str = ""
    is_synthetic: bool = False
