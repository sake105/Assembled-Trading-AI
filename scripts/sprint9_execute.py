# scripts/sprint9_execute.py
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

OUT_DIR = Path("output")


def _ensure_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Fehlende Spalten: {missing} | vorhanden={df.columns.tolist()}")
    return df


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce").astype("float64")
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype("string")
    df = df.dropna(subset=["timestamp", "close"])
    return df


def _default_price_path(freq: str) -> Path:
    if freq == "1d":
        return Path("output/aggregates/daily.parquet")
    if freq == "5min":
        return Path("output/aggregates/5min.parquet")
    raise ValueError(f"Unbekannte freq: {freq}")


def _read_prices(freq: str, price_file: str | None = None) -> pd.DataFrame:
    p = Path(price_file) if price_file else _default_price_path(freq)
    if not p.exists():
        raise FileNotFoundError(f"Preis-File nicht gefunden: {p}")
    df = pd.read_parquet(p)
    df = _ensure_cols(df, ["timestamp", "symbol", "close"])
    df = _coerce_types(df)[["timestamp", "symbol", "close"]]
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    return df


def _ema_signal_for_symbol(d: pd.DataFrame, fast: int, slow: int) -> pd.DataFrame:
    # Symbol aus Spalte oder Gruppenschlüssel ableiten
    sym = d["symbol"].iloc[0] if "symbol" in d.columns else d.name
    d = _ensure_cols(d, ["timestamp", "close"]).sort_values("timestamp")
    px = d["close"].astype("float64")

    ema_fast = px.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = px.ewm(span=slow, adjust=False, min_periods=slow).mean()
    sig = (ema_fast > ema_slow).astype(np.int8) - (ema_fast < ema_slow).astype(np.int8)

    return pd.DataFrame(
        {
            "timestamp": d["timestamp"].values,
            "symbol": np.full(len(d), sym, dtype=object),
            "sig": sig.values,
            "price": d["close"].values,
        }
    )


def _gen_orders_for_symbol(d: pd.DataFrame) -> pd.DataFrame:
    # Wenn include_groups=False, fehlt 'symbol' → aus d.name rekonstruieren
    if "symbol" not in d.columns:
        d = d.assign(symbol=d.name)

    d = d.sort_values("timestamp").reset_index(drop=True)
    d = _ensure_cols(d, ["timestamp", "symbol", "sig", "price"])

    sig = d["sig"].fillna(0).astype("int8")
    sig_prev = sig.shift(1).fillna(0).astype("int8")
    delta = sig - sig_prev

    chg = d.loc[delta != 0, ["timestamp", "symbol", "price"]].copy()
    chg["side"] = np.where(sig.loc[chg.index] > sig_prev.loc[chg.index], "BUY", "SELL")
    chg["qty"] = 1.0
    return chg[["timestamp", "symbol", "side", "qty", "price"]].reset_index(drop=True)


@dataclass
class ExecArgs:
    freq: str
    ema_fast: int
    ema_slow: int
    price_file: str | None
    out_dir: Path


def make_orders(freq: str, fast: int, slow: int, price_file: str | None = None) -> pd.DataFrame:
    prices = _read_prices(freq, price_file=price_file)

    sig = (
        prices.groupby("symbol", group_keys=False)
        .apply(lambda d: _ema_signal_for_symbol(d, fast, slow), include_groups=False)
        .reset_index(drop=True)
    )

    orders = (
        sig.groupby("symbol", group_keys=False)
        .apply(_gen_orders_for_symbol, include_groups=False)
        .reset_index(drop=True)
    )

    for c in ["timestamp", "symbol", "side", "qty", "price"]:
        if c not in orders.columns:
            raise KeyError(f"Orders-Spalte fehlt: {c}")

    orders["timestamp"] = pd.to_datetime(orders["timestamp"], utc=True)
    orders["qty"] = pd.to_numeric(orders["qty"], errors="coerce").astype("float64")
    orders["price"] = pd.to_numeric(orders["price"], errors="coerce").astype("float64")
    orders["side"] = orders["side"].astype("string")
    orders["symbol"] = orders["symbol"].astype("string")
    orders = orders.dropna(subset=["timestamp", "symbol", "side", "qty", "price"])
    orders = orders.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    return orders


def _write_orders(orders: pd.DataFrame, freq: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"orders_{freq}.csv"
    orders.to_csv(path, index=False)
    return path


def _print_symbol_counts(orders: pd.DataFrame) -> None:
    print("[EXEC] Orders by symbol:")
    if orders.empty:
        print("  - (keine Orders)")
        return
    for sym, grp in orders.groupby("symbol"):
        print(f"  - {sym}: {len(grp)}")


def run_execution(a: ExecArgs) -> Tuple[Path, pd.DataFrame]:
    print(f"[EXEC] START Execution | freq={a.freq}")
    orders = make_orders(a.freq, a.ema_fast, a.ema_slow, price_file=a.price_file)
    out_path = _write_orders(orders, a.freq, a.out_dir)
    print(f"[EXEC] [OK] written: {out_path} | rows={len(orders)}")
    _print_symbol_counts(orders)
    print("[EXEC] DONE Execution")
    return out_path, orders


def parse_args() -> ExecArgs:
    p = argparse.ArgumentParser(description="EMA-Crossover Execution (orders csv)")
    p.add_argument("--freq", choices=["1d", "5min"], required=True, help="Zeitebene")
    p.add_argument("--ema-fast", type=int, default=20)
    p.add_argument("--ema-slow", type=int, default=60)
    p.add_argument("--price-file", type=str, default=None,
                   help="Optional eigener Pfad zu Preisen (Parquet mit timestamp,symbol,close)")
    p.add_argument("--out", type=str, default=str(OUT_DIR), help="Output-Ordner (default: output)")
    args = p.parse_args()
    return ExecArgs(
        freq=args.freq,
        ema_fast=int(args.ema_fast),
        ema_slow=int(args.ema_slow),
        price_file=args.price_file,
        out_dir=Path(args.out),
    )


def main() -> None:
    a = parse_args()
    if a.ema_fast <= 0 or a.ema_slow <= 0:
        raise ValueError("EMA-Längen müssen > 0 sein.")
    if a.ema_fast >= a.ema_slow:
        print(f"[WARN] ema_fast ({a.ema_fast}) >= ema_slow ({a.ema_slow}) – üblich ist fast < slow.")
    run_execution(a)


if __name__ == "__main__":
    main()
