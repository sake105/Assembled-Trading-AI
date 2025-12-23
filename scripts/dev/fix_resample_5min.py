# scripts/dev/fix_resample_5min.py
from __future__ import annotations
import pathlib as pl
import pandas as pd

RAW_DIR = pl.Path("data/raw/1min")
OUT_DIR = pl.Path("output/aggregates")
OUT_FILE = OUT_DIR / "5min.parquet"


def _read_one(p: pl.Path) -> pd.DataFrame:
    df = pd.read_parquet(p)
    # Symbol sicherstellen (aus Datei ableiten, falls Spalte fehlt/leer)
    sym = (p.stem or "").upper()
    if "symbol" not in df.columns or df["symbol"].isna().all():
        df["symbol"] = sym
    else:
        # vereinheitlichen
        df["symbol"] = df["symbol"].astype(str).str.upper().fillna(sym)

    # Timestamp säubern
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()

    # Nur benötigte Spalten
    if "close" not in df.columns:
        raise ValueError(f"{p}: Spalte 'close' fehlt.")
    return df[["timestamp", "symbol", "close"]]


def build_from_raw() -> pd.DataFrame:
    files = sorted(RAW_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"Keine Dateien in {RAW_DIR} gefunden.")

    parts = []
    for fp in files:
        d = _read_one(fp)
        # pro Symbol resamplen → keine MultiIndex-Magie
        for sym, g in d.groupby("symbol", sort=False):
            g = g.sort_values("timestamp")
            r = (
                g.set_index("timestamp")["close"]
                .resample("5min")
                .last()
                .to_frame("close")
                .reset_index()
            )
            r["symbol"] = sym
            parts.append(r)

    df5 = pd.concat(parts, ignore_index=True)
    # Aufräumen & sortieren
    df5 = (
        df5[["symbol", "timestamp", "close"]]
        .sort_values(["timestamp", "symbol"])
        .reset_index(drop=True)
    )
    return df5


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df5 = build_from_raw()
    df5.to_parquet(OUT_FILE, index=False)
    print(
        f"wrote {OUT_FILE} rows={len(df5)} "
        f"symbols={df5['symbol'].nunique()} "
        f"first={df5['timestamp'].min()} last={df5['timestamp'].max()}"
    )


if __name__ == "__main__":
    main()
