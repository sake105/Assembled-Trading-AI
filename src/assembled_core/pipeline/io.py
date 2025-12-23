# src/assembled_core/pipeline/io.py
"""Input/Output utilities for price and order data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.assembled_core.config import OUTPUT_DIR


def ensure_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Ensure required columns exist in DataFrame.

    Args:
        df: Input DataFrame
        cols: List of required column names

    Returns:
        DataFrame with validated columns

    Raises:
        KeyError: If any required column is missing
    """
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Fehlende Spalten: {missing} | vorhanden={df.columns.tolist()}")
    return df


def coerce_price_types(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce price DataFrame to correct types.

    Args:
        df: DataFrame with price data

    Returns:
        DataFrame with coerced types (timestamp UTC, close float64, symbol string)
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce").astype("float64")
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype("string")
    df = df.dropna(subset=["timestamp", "close"])
    return df


def get_default_price_path(freq: str, output_dir: Path | str | None = None) -> Path:
    """Get default price file path for a frequency.

    Args:
        freq: Frequency string ("1d" or "5min")
        output_dir: Base output directory (default: None, uses config.OUTPUT_DIR)

    Returns:
        Path to price parquet file

    Raises:
        ValueError: If freq is not supported
    """
    base = Path(output_dir) if output_dir else OUTPUT_DIR if output_dir else OUTPUT_DIR
    if freq == "1d":
        return base / "aggregates" / "daily.parquet"
    if freq == "5min":
        return base / "aggregates" / "5min.parquet"
    raise ValueError(f"Unbekannte freq: {freq}")


def load_prices(
    freq: str,
    price_file: Path | str | None = None,
    output_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Load price data from parquet file.

    Args:
        freq: Frequency string ("1d" or "5min")
        price_file: Optional explicit path to price file. If None, uses default path.
        output_dir: Base output directory (default: None, uses config.OUTPUT_DIR)

    Returns:
        DataFrame with columns: timestamp (UTC), symbol, close
        Sorted by symbol, then timestamp

    Raises:
        FileNotFoundError: If price file does not exist
        KeyError: If required columns are missing
    """
    p = Path(price_file) if price_file else get_default_price_path(freq, output_dir)
    if not p.exists():
        raise FileNotFoundError(f"Preis-File nicht gefunden: {p}")
    df = pd.read_parquet(p)
    df = ensure_cols(df, ["timestamp", "symbol", "close"])
    df = coerce_price_types(df)[["timestamp", "symbol", "close"]]
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    return df


def load_prices_with_fallback(
    freq: str, output_dir: Path | str | None = None
) -> pd.DataFrame:
    """Load price data with fallback paths.

    Args:
        freq: Frequency string ("1d" or "5min")
        output_dir: Base output directory (default: None, uses config.OUTPUT_DIR)

    Returns:
        DataFrame with columns: timestamp (UTC), symbol, close

    Raises:
        FileNotFoundError: If no price file found in any fallback location
    """
    base = Path(output_dir) if output_dir else OUTPUT_DIR
    if freq == "1d":
        candidates = [base / "aggregates" / "daily.parquet"]
    elif freq == "5min":
        candidates = [
            base / "aggregates" / "5min.parquet",  # bevorzugt
            base / "assembled_intraday" / "5min.parquet",  # falls vorhanden
            base / "features" / "base_5min.parquet",  # Fallback
        ]
    else:
        raise ValueError(f"Unbekannte freq '{freq}'")

    p = next((c for c in candidates if c.exists()), None)
    if p is None:
        raise FileNotFoundError(
            f"Kein Preis-File gefunden. Versucht: {', '.join(map(str, candidates))}"
        )

    df = pd.read_parquet(p)
    need = {"timestamp", "symbol", "close"}
    if not need.issubset(df.columns):
        raise ValueError(
            f"Preis-File {p} hat nicht alle Spalten (brauche {sorted(need)}), hat: {list(df.columns)}"
        )

    df = df[["timestamp", "symbol", "close"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce").astype("float64")
    df = (
        df.dropna(subset=["timestamp", "symbol", "close"])
        .sort_values(["timestamp", "symbol"])
        .reset_index(drop=True)
    )
    return df


def load_orders(
    freq: str, output_dir: Path | str | None = None, strict: bool = True
) -> pd.DataFrame:
    """Load orders from CSV file.

    Args:
        freq: Frequency string ("1d" or "5min")
        output_dir: Base output directory (default: None, uses config.OUTPUT_DIR)
        strict: If True, raise error if file missing. If False, return empty DataFrame.

    Returns:
        DataFrame with columns: timestamp, symbol, side, qty, price
        Sorted by timestamp

    Raises:
        FileNotFoundError: If strict=True and orders file does not exist
        ValueError: If required columns are missing (when strict=True)
    """
    base = Path(output_dir) if output_dir else OUTPUT_DIR
    p = base / f"orders_{freq}.csv"
    if not p.exists():
        if strict:
            raise FileNotFoundError(
                f"Orders nicht gefunden: {p} – erst sprint9_execute.py laufen lassen."
            )
        return pd.DataFrame(columns=["timestamp", "symbol", "side", "qty", "price"])

    df = pd.read_csv(p)

    if strict:
        for c in ["timestamp", "symbol", "side", "qty", "price"]:
            if c not in df.columns:
                raise ValueError(f"Spalte '{c}' fehlt in {p}")
    else:
        # Toleranter Modus (für backtest)
        if "timestamp" not in df.columns:
            raise ValueError(f"Orders-File {p} hat keine 'timestamp'-Spalte.")
        for c in ("symbol", "side"):
            if c not in df.columns:
                df[c] = ""
        for c in ("qty", "price"):
            if c not in df.columns:
                df[c] = 0.0

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0.0)
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
    df["side"] = df["side"].astype(str).str.upper().str.strip()
    df["symbol"] = df["symbol"].astype(str).str.strip()

    if strict:
        df = df.dropna(subset=["timestamp", "symbol", "side"])
    else:
        df = df.dropna(subset=["timestamp"])

    return df.sort_values(
        ["timestamp", "symbol"] if strict else ["timestamp"]
    ).reset_index(drop=True)
