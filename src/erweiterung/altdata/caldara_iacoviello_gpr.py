"""Caldara-Iacoviello Geopolitical Risk Index (GPR) — Daten-Loader.

Quelle
------
Caldara, D. & Iacoviello, M. (2018). "Measuring Geopolitical Risk".
Federal Reserve Board Working Paper.
URL: https://www.matteoiacoviello.com/gpr.htm

Daten
-----
- GPR: Recent country-aggregate index (Monthly, 1985+)
- GPRH: Historical index (Monthly, 1900+)
- GPRT: Threats sub-index
- GPRA: Acts sub-index
- GPRD: Daily index (~1985+, separate file)

Diese Loader-Klasse ist eine direkte **Ergänzung** zur Mainline-API
``src.assembled_core.features.geopolitical_features.compute_gpr_proxy``:
Mainline sagt "If Caldara-Iacoviello GPR data is available via FRED, those
are used directly" — implementiert aber keinen Loader dafür. Hier wird
das nachgereicht.

Output-Format ist kompatibel zu ``compute_gpr_proxy`` (gpr_level,
gpr_zscore, gpr_momentum, gpr_regime), sodass die Mainline-Pipeline
``risk/georisk_overlay.py`` ohne Code-Änderung profitieren kann.

PR-Hinweis: dieses Modul könnte in den Mainline-Pfad
``src/assembled_core/data/altdata/gpr_loader.py`` portiert werden,
als ergänzender Daten-Provider zu den existierenden GDELT/Finnhub-
Pipelines.

Cache
-----
data/cache/gpr/sheet1.parquet (befüllt via scripts/erweiterung/fetch_gpr_index.py)
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd


GPR_SOURCE_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls"
DEFAULT_CACHE = Path("data/cache/gpr/sheet1.parquet")


def fetch_gpr_excel(url: str = GPR_SOURCE_URL, timeout: int = 30) -> pd.DataFrame:
    """Lade Caldara-Iacoviello GPR-Excel direkt vom Autor-Server.

    Returns:
        DataFrame mit Spalten month, GPR, GPRT, GPRA, GPRH, GPRHT, GPRHA, ...
    """
    import requests

    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return pd.read_excel(BytesIO(r.content), sheet_name="Sheet1")


def load_gpr_cached(cache_path: Path | str = DEFAULT_CACHE) -> pd.DataFrame:
    """Lade GPR-Daten aus dem Cache.

    Returns:
        DataFrame mit DatetimeIndex (Monthly) und Spalten GPR, GPRT, GPRA,
        GPRH, GPRHT, GPRHA + Subscores.
    """
    cache_path = Path(cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"GPR cache not found: {cache_path}. "
            "Run scripts/erweiterung/fetch_gpr_index.py to populate."
        )
    df = pd.read_parquet(cache_path)
    # Normalize date-index
    if "month" in df.columns:
        df["date"] = pd.to_datetime(df["month"], errors="coerce")
    elif "date" not in df.columns:
        df["date"] = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    df = df[df["date"].notna()].set_index("date").sort_index()
    # Filter only numeric series
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    return df[numeric_cols]


def expand_to_daily(
    monthly_gpr: pd.DataFrame, daily_index: pd.DatetimeIndex
) -> pd.DataFrame:
    """Expandiere Monthly-GPR zu Daily-Index via forward-fill.

    Args:
        monthly_gpr: DataFrame mit DatetimeIndex (Monthly).
        daily_index: Ziel-Index (Daily, typisch trading-days).

    Returns:
        DataFrame mit gleichen Spalten, Daily-Index, ffilled.
    """
    if monthly_gpr.empty:
        return pd.DataFrame(index=daily_index)
    # Localize to UTC if naive
    if monthly_gpr.index.tz is None:
        monthly_gpr = monthly_gpr.copy()
        monthly_gpr.index = monthly_gpr.index.tz_localize("UTC")
    daily_target = (
        daily_index
        if daily_index.tz is not None
        else daily_index.tz_localize("UTC")
    )
    return monthly_gpr.reindex(daily_target, method="ffill")


def compute_gpr_features(
    daily_gpr: pd.DataFrame, rolling_window: int = 252, zscore_window: int = 63
) -> pd.DataFrame:
    """Compute Mainline-kompatible Feature-Outputs aus GPR-Daten.

    Output: gleiches Schema wie
    ``src.assembled_core.features.geopolitical_features.compute_gpr_proxy``:
    gpr_level (0-100 percentile), gpr_zscore, gpr_momentum, gpr_regime.

    Args:
        daily_gpr: DataFrame mit mindestens 'GPR' oder 'GPRH' Spalte.
        rolling_window: Window für Percentile-Normalisierung (default 252).
        zscore_window: Window für Z-Score (default 63 = ~3 Monate).

    Returns:
        DataFrame [gpr_level, gpr_zscore, gpr_momentum, gpr_regime].
    """
    if "GPR" in daily_gpr.columns:
        gpr_series = daily_gpr["GPR"]
    elif "GPRH" in daily_gpr.columns:
        gpr_series = daily_gpr["GPRH"]
    else:
        raise ValueError("Need 'GPR' or 'GPRH' column")

    out = pd.DataFrame(index=daily_gpr.index)
    # gpr_level: percentile-rank über rolling window
    out["gpr_level"] = gpr_series.rolling(rolling_window, min_periods=20).rank(
        pct=True
    ) * 100
    # gpr_zscore: 63-day z-score
    mean = gpr_series.rolling(zscore_window, min_periods=10).mean()
    std = gpr_series.rolling(zscore_window, min_periods=10).std()
    out["gpr_zscore"] = (gpr_series - mean) / std.replace(0, np.nan)
    # gpr_momentum: 5-day change in gpr_level
    out["gpr_momentum"] = out["gpr_level"].diff(5)
    # gpr_regime: quartile (1=calm, 4=elevated)
    out["gpr_regime"] = pd.qcut(
        out["gpr_level"].fillna(out["gpr_level"].median()),
        q=4,
        labels=False,
        duplicates="drop",
    ).fillna(0).astype(int) + 1

    return out


def gpr_state_hint(gpr_level: float, zscore: float) -> str:
    """Map GPR-Level + Z-Score → State-Hint kompatibel zu Mainline.

    Mainline ``risk/georisk_overlay.py`` nutzt state_hint ∈
    {WATCH, ACTIVE, COOLDOWN, PAUSE}. Diese Funktion mappt Daten-Werte
    auf dieselben Labels.

    Args:
        gpr_level: 0-100 Percentile.
        zscore: 63-day Z-Score.

    Returns:
        State-Label.
    """
    if not np.isfinite(gpr_level):
        return "WATCH"
    if zscore > 2.0 or gpr_level > 90:
        return "PAUSE"
    if zscore > 1.0 or gpr_level > 75:
        return "ACTIVE"
    if zscore < -1.0 or gpr_level < 25:
        return "COOLDOWN"
    return "WATCH"


__all__ = [
    "GPR_SOURCE_URL",
    "fetch_gpr_excel",
    "load_gpr_cached",
    "expand_to_daily",
    "compute_gpr_features",
    "gpr_state_hint",
]
