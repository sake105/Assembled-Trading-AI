#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sprint 10 – Portfolio Simulator (robust)
----------------------------------------
- Lädt Feature-Daten aus output/features/*_{freq}.parquet
- Erzeugt simple Zielgewichte auf Basis von 1-Step-Momentum (ret_1 = sign(Δclose))
- Robust gegen Duplikate (drop_duplicates) und nutzt pivot_table(aggfunc='last')
- Kostenmodell: total_bps = commission_bps + spread_w*1 + impact_w*1
- Schreibt:
    output/portfolio_trades.csv
    output/portfolio_equity_{freq}.csv
    output/portfolio_report.md
"""

from __future__ import annotations
import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd


# ---------- Logging ----------
def log(m: str) -> None:
    ts = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{ts}] [PF10] {m}")


# ---------- CLI ----------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--freq", default="5min", help="Bar-Frequenz (z.B. 5min)")
    p.add_argument("--start-capital", type=float, default=10_000.0)
    p.add_argument("--exposure", type=float, default=1.0)
    p.add_argument("--max-leverage", type=float, default=1.0)
    # Kosten/Slippage (bps)
    p.add_argument("--commission-bps", type=float, default=0.5)
    p.add_argument("--spread-w", type=float, default=1.0)
    p.add_argument("--impact-w", type=float, default=1.0)
    return p.parse_args()


# ---------- IO ----------
def _project_root() -> Path:
    return Path.cwd()


def _features_path(freq: str) -> Path:
    return _project_root() / "output" / "features" / f"base_{freq}.parquet"


def _ensure_cols(df: pd.DataFrame, cols: List[str], name: str) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"[{name}] Missing columns: {miss}")


# ---------- Feature Load & Signals ----------
def load_features(freq: str) -> pd.DataFrame:
    f = _features_path(freq)
    if not f.exists():
        raise FileNotFoundError(f"Features not found: {f}")
    df = pd.read_parquet(f)

    base_cols = ["timestamp", "symbol", "close"]
    _ensure_cols(df, base_cols, f.name)

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp", "symbol", "close"])

    # Robust: je (timestamp, symbol) nur die letzte Zeile behalten
    df = df.sort_values(["timestamp", "symbol"])
    df = df.drop_duplicates(subset=["timestamp", "symbol"], keep="last").reset_index(drop=True)

    # ret_1 bereitstellen, falls nicht vorhanden
    if "ret_1" not in df.columns:
        df["ret_1"] = (
            df.sort_values(["symbol", "timestamp"])
              .groupby("symbol", group_keys=False)["close"]
              .pct_change()
        )
    df["ret_1"] = df["ret_1"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


def _make_signals(df: pd.DataFrame) -> np.ndarray:
    s = np.sign(df["ret_1"].astype(float).to_numpy())
    return np.where(s > 0, 1.0, np.where(s < 0, -1.0, 0.0))


def _position_target_weights(df: pd.DataFrame) -> pd.DataFrame:
    """
    Zielgewichte pro Timestamp:
      - LONG bei ret_1>0, SHORT bei ret_1<0, sonst 0
      - Normierung je Timestamp auf Sum(|w|)=1
    Ohne groupby.apply → keine FutureWarnings, timestamp bleibt erhalten.
    """
    df = df.copy()
    df["sig"] = _make_signals(df)
    df["abs_sig"] = df["sig"].abs()

    denom = df.groupby("timestamp", group_keys=False)["abs_sig"].transform("sum")
    df["w"] = 0.0
    mask = denom > 0
    df.loc[mask, "w"] = df.loc[mask, "sig"] / denom[mask]

    return df[["timestamp", "symbol", "w", "close"]]


# ---------- Simulation ----------
@dataclass
class SimParams:
    start_capital: float
    exposure: float
    max_lev: float
    total_bps: float  # (Kommission + Spread + Impact) in bps


def _timeline_and_pivots(df: pd.DataFrame) -> Tuple[pd.DatetimeIndex, pd.Index, pd.DataFrame, pd.DataFrame]:
    # Pivot robust gegen Duplikate: aggfunc='last'
    px_close = pd.pivot_table(
        df, index="timestamp", columns="symbol", values="close", aggfunc="last"
    ).sort_index()

    w_target = pd.pivot_table(
        df, index="timestamp", columns="symbol", values="w", aggfunc="last"
    ).sort_index()

    timeline = px_close.index.union(w_target.index).unique().sort_values()
    symbols = px_close.columns.union(w_target.columns).unique()

    px_close = px_close.reindex(index=timeline, columns=symbols).ffill()
    w_target = w_target.reindex(index=timeline, columns=symbols).fillna(0.0)

    return timeline, symbols, px_close, w_target


def _position_dollars(equity: float, w: np.ndarray, expo: float, max_lev: float) -> np.ndarray:
    gross_target = min(equity * expo, equity * max_lev)
    return w * gross_target


def simulate(df_in: pd.DataFrame, params: SimParams) -> Tuple[pd.DataFrame, pd.DataFrame]:
    wdf = _position_target_weights(df_in)
    timeline, symbols, px_close, w_target = _timeline_and_pivots(wdf)

    equity = float(params.start_capital)
    prev_shares = np.zeros(len(symbols), dtype=float)
    trades_rows = []
    equity_rows = []

    cost_rate = params.total_bps / 10_000.0

    for i, ts in enumerate(timeline):
        w = w_target.loc[ts].to_numpy(dtype=float)
        px = px_close.loc[ts].to_numpy(dtype=float)

        tgt_dollars = _position_dollars(equity, w, params.exposure, params.max_lev)
        tgt_shares = np.divide(tgt_dollars, np.maximum(px, 1e-9), where=px > 0)

        trade_shares = tgt_shares - prev_shares
        trade_notional = float(np.sum(np.abs(trade_shares) * px))
        costs = trade_notional * cost_rate
        equity -= costs

        if i + 1 < len(timeline):
            px_next = px_close.iloc[i + 1].to_numpy(dtype=float)
            pnl = float(np.sum(tgt_shares * (px_next - px)))
        else:
            pnl = 0.0

        equity += pnl

        equity_rows.append({"timestamp": ts, "equity": equity})
        nz = np.where(np.abs(trade_shares) > 0)[0]
        for idx in nz:
            trades_rows.append(
                {
                    "timestamp": ts,
                    "symbol": symbols[idx],
                    "shares": float(trade_shares[idx]),
                    "price": float(px[idx]),
                    "trade_value": float(trade_shares[idx] * px[idx]),
                    "costs": float(np.abs(trade_shares[idx]) * px[idx] * cost_rate),
                }
            )

        prev_shares = tgt_shares

    trades_df = pd.DataFrame(trades_rows)
    equity_df = pd.DataFrame(equity_rows)
    return trades_df, equity_df


# ---------- Reporting ----------
@dataclass
class Metrics:
    start: float
    end: float
    ret: float
    days: int
    cagr: float
    max_dd: float
    n_bars: int


def _max_drawdown(series: pd.Series) -> float:
    s = pd.Series(series, dtype=float)
    cm = s.cummax()
    dd = (s - cm) / cm.replace(0, np.nan)
    dd = dd.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return float(dd.min())


def _metrics(equity_df: pd.DataFrame) -> Metrics:
    df = equity_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp", "equity"]).sort_values("timestamp")
    if df.empty:
        return Metrics(0, 0, 0, 0, 0, 0, 0)

    start, end = float(df["equity"].iloc[0]), float(df["equity"].iloc[-1])
    days = max(int((df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).days), 1)
    ret = (end / max(start, 1e-9)) - 1.0
    try:
        cagr = float((end / max(start, 1e-9)) ** (365.0 / days) - 1.0)
    except Exception:
        cagr = np.nan
    max_dd = _max_drawdown(df["equity"])
    return Metrics(start, end, ret, days, cagr, max_dd, len(df))


def write_outputs(trades: pd.DataFrame, equity: pd.DataFrame, freq: str) -> None:
    root = _project_root()
    outdir = root / "output"
    outdir.mkdir(parents=True, exist_ok=True)

    trades_file = outdir / "portfolio_trades.csv"
    equity_file = outdir / f"portfolio_equity_{freq}.csv"
    report_file = outdir / "portfolio_report.md"

    trades.to_csv(trades_file, index=False)
    equity.to_csv(equity_file, index=False)

    m = _metrics(equity)
    lines = [
        "# Portfolio Report",
        "",
        f"- Start equity: {m.start:,.2f}",
        f"- End equity:   {m.end:,.2f}",
        f"- Total return:  {m.ret*100:,.2f} %",
        f"- Bars:          {m.n_bars}",
        f"- Period (days): {m.days}",
        f"- CAGR (approx): {0.0 if pd.isna(m.cagr) else m.cagr*100:,.2f} %",
        f"- Max Drawdown:  {m.max_dd*100:,.2f} %",
        "",
    ]
    report_file.write_text("\n".join(lines), encoding="utf-8")

    log(f"[OK] written: {trades_file}")
    log(f"[OK] written: {equity_file}")
    log(f"[OK] written: {report_file}")


# ---------- Main ----------
def main() -> None:
    args = parse_args()
    log(f"START Portfolio | freq={args.freq} start_capital={args.start_capital} "
        f"exp={args.exposure} lev={args.max_leverage} "
        f"comm_bps={args.commission_bps} spread_w={args.spread_w} impact_w={args.impact_w}")

    total_bps = float(args.commission_bps) + float(args.spread_w) * 1.0 + float(args.impact_w) * 1.0

    df = load_features(args.freq)

    params = SimParams(
        start_capital=float(args.start_capital),
        exposure=float(args.exposure),
        max_lev=float(args.max_leverage),
        total_bps=float(total_bps),
    )

    trades_df, equity_df = simulate(df, params)
    write_outputs(trades_df, equity_df, args.freq)
    log("DONE Portfolio")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        sys.stderr.write(str(e) + "\n")
        sys.exit(1)
