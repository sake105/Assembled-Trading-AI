"""yfinance-Cache-Loader: Liest pro-Symbol Parquets aus ``data/cache/yfinance/``.

Quelle
------
Das Mainline-Projekt unterhält einen Cache-Ordner mit einzelnen Parquet-Dateien
pro Ticker (Schema: ``date``-Index + ``open/high/low/close/volume/symbol``).
Dieser Loader kombiniert mehrere Symbol-Files in ein lang-Format-Panel
``[date, symbol, open, high, low, close, volume, return]`` — das genau jenem
Format entspricht, das ``scripts/erweiterung/run_real_backtest.py`` erwartet.

Hinweise
--------
- Keine Modifikation am Cache. Read-Only.
- Returns werden per Symbol gegrouped berechnet (pct_change).
- Volume bleibt als raw-Integer; weitere Normalisierungen sind Caller-Sache.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def list_cached_symbols(cache_dir: str | Path) -> list[str]:
    """Liste aller Ticker, für die ein Parquet im Cache existiert."""
    p = Path(cache_dir)
    if not p.exists():
        return []
    return sorted(f.stem for f in p.glob("*.parquet"))


def load_symbol_parquet(
    cache_dir: str | Path, symbol: str, start: str | None = None, end: str | None = None
) -> pd.DataFrame:
    """Lade einzelnes Symbol als DataFrame.

    Args:
        cache_dir: Pfad zum yfinance-Cache.
        symbol: Ticker.
        start/end: optionale Bereichs-Filter (YYYY-MM-DD).

    Returns:
        DataFrame [date, symbol, open, high, low, close, volume].
    """
    f = Path(cache_dir) / f"{symbol}.parquet"
    if not f.exists():
        raise FileNotFoundError(f"no cache entry for {symbol} at {f}")
    df = pd.read_parquet(f)
    df = df.reset_index()
    if "date" not in df.columns and "Date" in df.columns:
        df = df.rename(columns={"Date": "date"})
    df["date"] = pd.to_datetime(df["date"], utc=True)
    if "symbol" not in df.columns:
        df["symbol"] = symbol
    if start is not None:
        df = df[df["date"] >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        df = df[df["date"] <= pd.Timestamp(end, tz="UTC")]
    return df.reset_index(drop=True)


def load_universe_panel(
    cache_dir: str | Path,
    symbols: list[str],
    start: str | None = None,
    end: str | None = None,
    require_min_rows: int = 100,
    skip_missing: bool = True,
) -> pd.DataFrame:
    """Lade ein Multi-Symbol-Panel im lang-Format.

    Args:
        cache_dir: Cache-Ordner.
        symbols: Liste von Tickern.
        start/end: optionale Bereichs-Filter.
        require_min_rows: Symbol verwerfen, wenn weniger Zeilen.
        skip_missing: True -> fehlende Tickers überspringen, sonst Exception.

    Returns:
        Long-Panel sortiert nach (symbol, date), mit Spalten
        [date, symbol, open, high, low, close, volume, return].
    """
    cache = Path(cache_dir)
    frames: list[pd.DataFrame] = []
    skipped: list[str] = []
    for sym in symbols:
        f = cache / f"{sym}.parquet"
        if not f.exists():
            if skip_missing:
                skipped.append(sym)
                continue
            raise FileNotFoundError(f"missing cache for {sym}")
        try:
            sub = load_symbol_parquet(cache, sym, start, end)
        except Exception:
            if skip_missing:
                skipped.append(sym)
                continue
            raise
        if len(sub) < require_min_rows:
            skipped.append(sym)
            continue
        frames.append(sub)

    if not frames:
        raise RuntimeError("no usable symbols in cache for given universe/date-range")

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)
    panel["return"] = panel.groupby("symbol")["close"].pct_change()
    panel.attrs["skipped_symbols"] = skipped
    return panel


def panel_coverage_report(panel: pd.DataFrame) -> pd.DataFrame:
    """Coverage-Report: pro Symbol n_rows, date_min, date_max."""
    grp = panel.groupby("symbol")
    out = pd.DataFrame(
        {
            "n_rows": grp.size(),
            "date_min": grp["date"].min(),
            "date_max": grp["date"].max(),
            "nan_close_pct": grp["close"].apply(lambda s: s.isna().mean()),
        }
    ).reset_index()
    return out.sort_values("symbol").reset_index(drop=True)


__all__ = [
    "list_cached_symbols",
    "load_symbol_parquet",
    "load_universe_panel",
    "panel_coverage_report",
]
