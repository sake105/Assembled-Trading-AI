"""FRED-MD — McCracken/Ng monatliches Macro-Panel (frei, ~130 Variablen).

Quelle
------
Federal Reserve Bank of St. Louis hosted FRED-MD:
    https://research.stlouisfed.org/wp/more/2015-012/

Direkter CSV-Download (kein Key):
    https://files.stlouisfed.org/files/htdocs/fred-md/monthly/current.csv

Inhalt
------
~130 monatliche Macro-Zeitreihen:
- Output & Income (Industrial Production, Real Personal Income, ...)
- Labor Market (Nonfarm Payrolls, Unemployment, ...)
- Consumption (Real Consumer Goods, Vehicle Sales, ...)
- Money & Credit (M1, M2, Reserves, ...)
- Interest & Exchange Rates (3M, 1Y, 10Y, EUR/USD, ...)
- Prices (CPI, PPI, PCE, ...)
- Stock Market (S&P 500, P/E, Dividend Yield, ...)

Anwendung
---------
1. Faktor-Modell: PCA über transformierte Reihen -> 8-10 Macro-Faktoren.
2. Nowcasting: Aggregat-Index für Konjunkturphase (Recession-Prob).
3. Macro-Beta-Faktoren in Cross-Section.

Transformations-Codes
---------------------
McCracken/Ng definieren Codes 1..7 in ``transformations.csv`` für jede Variable:
    1: Level (no transform)
    2: First difference
    3: Second difference
    4: Logarithm
    5: First difference of log
    6: Second difference of log
    7: Δ(x_t / x_{t-1} - 1) (rate of growth)
"""

from __future__ import annotations

import io
import logging

import numpy as np
import pandas as pd

from erweiterung._base import (
    FetchResult,
    get_cache_dir,
    rate_limited,
    retry_with_backoff,
)

logger = logging.getLogger(__name__)

_FRED_MD_URL = "https://files.stlouisfed.org/files/htdocs/fred-md/monthly/current.csv"


@rate_limited(min_interval_s=1.0)
@retry_with_backoff(max_attempts=3, base_delay=2.0)
def _download_fred_md() -> pd.DataFrame:
    import requests

    r = requests.get(_FRED_MD_URL, timeout=30)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))


def fetch_fred_md(use_cache: bool = True) -> FetchResult:
    """Hole das aktuelle FRED-MD-Panel.

    Returns:
        FetchResult mit DataFrame: Index = ``date`` (monthly), Columns = Variablen.
        Erste Zeile (transformation codes) ist als Attribut ``transform`` separiert.
    """
    cache_path = get_cache_dir("fred_md") / "current.parquet"
    transform_path = get_cache_dir("fred_md") / "transforms.parquet"
    if use_cache and cache_path.exists() and transform_path.exists():
        df = pd.read_parquet(cache_path)
        return FetchResult(df, "fred_md", pd.Timestamp.utcnow(), len(df), "cache")

    raw = _download_fred_md()
    transforms = raw.iloc[0].to_dict()  # erste Zeile = Codes
    data = raw.iloc[1:].copy()
    data.columns = [c.strip() for c in data.columns]
    date_col = "sasdate" if "sasdate" in data.columns else data.columns[0]
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data = data.dropna(subset=[date_col]).set_index(date_col)
    data.index.name = "date"
    for c in data.columns:
        data[c] = pd.to_numeric(data[c], errors="coerce")

    transforms_clean = {k: v for k, v in transforms.items() if k != date_col}
    transform_df = pd.DataFrame.from_dict(
        transforms_clean, orient="index", columns=["code"]
    )
    transform_df.index.name = "variable"

    if use_cache:
        data.to_parquet(cache_path)
        transform_df.to_parquet(transform_path)
    return FetchResult(data, "fred_md", pd.Timestamp.utcnow(), len(data), "")


def apply_mccracken_transforms(
    data: pd.DataFrame, transforms: pd.DataFrame
) -> pd.DataFrame:
    """Wende McCracken/Ng-Transforms an, um stationäre Series zu erhalten.

    Args:
        data: Raw FRED-MD-Datensatz.
        transforms: DataFrame mit Index=variable, Spalte 'code'.

    Codes
    -----
    1=level, 2=Δx, 3=Δ²x, 4=ln(x), 5=Δln(x), 6=Δ²ln(x), 7=Δ(x/x₋₁ − 1)
    """
    out = pd.DataFrame(index=data.index)
    for var in data.columns:
        x = data[var].astype(float)
        code = int(transforms.loc[var, "code"]) if var in transforms.index else 1
        try:
            out[var] = _apply_one(x, code)
        except Exception as e:  # noqa: BLE001
            logger.warning("[fred-md] transform %s code=%d failed: %s", var, code, e)
            out[var] = x
    return out


def _apply_one(x: pd.Series, code: int) -> pd.Series:
    if code == 1:
        return x
    if code == 2:
        return x.diff()
    if code == 3:
        return x.diff().diff()
    if code == 4:
        return np.log(x.replace({0: np.nan}).clip(lower=1e-12))
    if code == 5:
        return np.log(x.replace({0: np.nan}).clip(lower=1e-12)).diff()
    if code == 6:
        return np.log(x.replace({0: np.nan}).clip(lower=1e-12)).diff().diff()
    if code == 7:
        ratio = x / x.shift(1)
        return ratio.diff()
    return x


def macro_pca_factors(
    transformed: pd.DataFrame,
    n_components: int = 8,
    min_history_months: int = 60,
) -> pd.DataFrame:
    """Extrahiere Macro-PCA-Faktoren (rolling window, PIT-sicher).

    Args:
        transformed: Output von ``apply_mccracken_transforms``.
        n_components: Anzahl Faktoren.
        min_history_months: Mindesthistorie pro Berechnung.

    Returns:
        DataFrame ``[date, F1, F2, ..., F_n]`` mit den PCA-Faktoren je Monat.
        Jeder Faktor je Datum wird mit Daten ``<= datum`` berechnet (rolling).

    Achtung
    -------
    Vollständige rolling PCA ist O(T² × p × k) — kann teuer werden.
    Für eine Gesamtschätzung (Backtest-Time) verwende ``static_pca_factors``.
    """
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except ImportError as e:
        raise RuntimeError("scikit-learn required for PCA") from e

    if transformed.empty or len(transformed) < min_history_months:
        return pd.DataFrame()

    out_rows: list[dict] = []
    cols = [
        c
        for c in transformed.columns
        if transformed[c].notna().sum() > min_history_months
    ]
    if not cols:
        return pd.DataFrame()
    sub = transformed[cols].copy()

    for end_idx in range(min_history_months, len(sub)):
        end_date = sub.index[end_idx]
        train = sub.iloc[: end_idx + 1].dropna(axis=1, how="any")
        if train.shape[1] < n_components:
            continue
        if len(train) < n_components + 5:
            continue
        scaler = StandardScaler()
        X = scaler.fit_transform(train.values)
        pca = PCA(n_components=min(n_components, X.shape[1] - 1))
        F = pca.fit_transform(X)
        latest = F[-1]
        row = {"date": end_date}
        for i, val in enumerate(latest):
            row[f"F{i+1}"] = float(val)
        out_rows.append(row)

    return pd.DataFrame(out_rows)


def static_pca_factors(
    transformed: pd.DataFrame, n_components: int = 8
) -> pd.DataFrame:
    """Statische PCA über das gesamte Panel (für Forschung; nicht PIT-sicher!).

    Nur für Plotting / Faktorinterpretation.
    """
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except ImportError as e:
        raise RuntimeError("scikit-learn required") from e

    sub = transformed.dropna(axis=1, how="any").dropna(axis=0, how="any")
    if sub.empty:
        return pd.DataFrame()
    scaler = StandardScaler()
    X = scaler.fit_transform(sub.values)
    pca = PCA(n_components=min(n_components, X.shape[1] - 1))
    F = pca.fit_transform(X)
    out = pd.DataFrame(
        F, index=sub.index, columns=[f"F{i+1}" for i in range(F.shape[1])]
    )
    out.index.name = "date"
    return out.reset_index()


__all__ = [
    "fetch_fred_md",
    "apply_mccracken_transforms",
    "macro_pca_factors",
    "static_pca_factors",
]
