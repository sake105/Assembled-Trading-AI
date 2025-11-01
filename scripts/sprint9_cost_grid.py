#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sprint 9 — Cost Grid
Liest Feature-Parquets, erzeugt ein simples Signal, simuliert Kosten
über ein Gitter (commission_bps, spread_w, impact_w) und schreibt
einen Markdown-Report nach output/cost_grid_report.md.

Kompatibel mit den PowerShell-Logs:
[GRID] START Grid | freq=5min | commission=[...] | spread_w=[...] | impact_w=[...]
[GRID] [OK] written: {out_path.resolve()}
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd


# ---------- Args ----------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--freq", default="5min", type=str)
    # kompatibel zu PS-Skripten:
    p.add_argument("--notional", default=10_000, type=float)
    p.add_argument(
        "--commission-bps",
        nargs="+",
        type=float,
        default=[0.0, 0.5, 1.0],
        dest="commission_bps",
    )
    p.add_argument(
        "--spread-w",
        nargs="+",
        type=float,
        default=[0.5, 1.0, 2.0],
        dest="spread_ws",
    )
    p.add_argument(
        "--impact-w",
        nargs="+",
        type=float,
        default=[0.5, 1.0, 2.0],
        dest="impact_ws",
    )
    return p.parse_args()


# ---------- I/O Helpers ----------

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "output"
FEAT = OUT / "features"


def _ensure_ts(df: pd.DataFrame) -> pd.DataFrame:
    # Timestamp vereinheitlichen → tz-naiv (pandas/numpy kompatibel)
    if "timestamp" not in df.columns:
        raise ValueError("Features müssen eine 'timestamp'-Spalte besitzen.")
    ts = pd.to_datetime(df["timestamp"], utc=True)
    df["timestamp"] = ts.dt.tz_convert(None)
    return df


def load_features(freq: str) -> pd.DataFrame:
    """
    Lädt base_{freq}.parquet, micro_{freq}.parquet, regime_{freq}.parquet
    und merged sie auf ['timestamp','symbol'].
    """
    base_p = FEAT / f"base_{freq}.parquet"
    micr_p = FEAT / f"micro_{freq}.parquet"
    reg_p = FEAT / f"regime_{freq}.parquet"

    if not base_p.exists() or not micr_p.exists() or not reg_p.exists():
        raise FileNotFoundError("Feature-Dateien fehlen. Bitte Sprint8 ausführen.")

    base = pd.read_parquet(base_p)
    micr = pd.read_parquet(micr_p)
    regi = pd.read_parquet(reg_p)

    base = _ensure_ts(base)
    micr = _ensure_ts(micr)
    regi = _ensure_ts(regi)

    on = ["timestamp", "symbol"]
    df = base.merge(micr, on=on, how="inner", validate="m:1").merge(regi, on=on, how="inner", validate="m:1")

    # Wichtige Spalten absichern
    if "close" not in df.columns:
        # ggf. 'price' oder ähnlich als close benutzen
        maybe = [c for c in ["price", "last", "adj_close"] if c in df.columns]
        if not maybe:
            raise KeyError("Spalte 'close' nicht gefunden.")
        df = df.rename(columns={maybe[0]: "close"})

    # Typen säubern
    df["symbol"] = df["symbol"].astype(str)
    df = df.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

    return df


# ---------- Signal ----------

def ema_regime(df: pd.DataFrame, span_fast: int = 8, span_slow: int = 21) -> pd.DataFrame:
    """
    Einfache Trend-Regime-Logik: fast EMA vs slow EMA.
    Gibt 'trend_regime' {+1, -1, 0} zurück.
    FutureWarning-frei: keine groupby.apply-Rückgabe mit gruppierenden Spalten.
    """
    def _per_symbol(d: pd.DataFrame) -> pd.DataFrame:
        s = d["close"].astype(float)
        ema_f = s.ewm(span=span_fast, adjust=False).mean()
        ema_s = s.ewm(span=span_slow, adjust=False).mean()
        reg = np.sign(ema_f - ema_s).astype(int)
        out = d[["timestamp", "symbol"]].copy()
        out["trend_regime"] = reg.values
        return out

    g = df.groupby("symbol", group_keys=False)
    parts: List[pd.DataFrame] = []
    for _, d in g:
        parts.append(_per_symbol(d))
    out = pd.concat(parts, ignore_index=True)
    return df.merge(out, on=["timestamp", "symbol"], how="left")


def build_signal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Finales Signal:
    - Nutzt vorhandenes 'trend_regime' oder berechnet EMA-Regime.
    - Konvertiert in Positionsgröße p ∈ {-1, 0, +1}.
    """
    if "trend_regime" not in df.columns:
        df = ema_regime(df)

    pos = df["trend_regime"].fillna(0).astype(int).clip(-1, 1)
    out = df[["timestamp", "symbol", "close"]].copy()
    out["pos"] = pos.values
    return out


# ---------- Costs & Backtest ----------

@dataclass(frozen=True)
class CostParams:
    commission_bps: float
    spread_w: float
    impact_w: float


def _turnover(pos: pd.Series) -> pd.Series:
    # |Δ Position|
    return pos.diff().abs().fillna(0.0)


def simulate_costs(signal: pd.DataFrame, costs: CostParams, notional: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Rechnet P&L (einfach) mit Kostenmodell über alle Symbole zusammen.
    Gibt:
      equity_df:  timestamp, equity
      rows_df:    eine Zeile mit Metriken für diesen Kostenpunkt
    """
    df = signal.sort_values(["symbol", "timestamp"]).copy()

    # Log-Returns je Symbol
    df["ret"] = (
        df.groupby("symbol", group_keys=False)["close"]
          .apply(lambda s: np.log(s).diff())
          .fillna(0.0)
          .astype(float)
    )

    # Brutto-PL = pos[t-1] * ret[t] * Notional
    df["pos_shift"] = df.groupby("symbol", group_keys=False)["pos"].shift().fillna(0.0)
    df["pnl_gross"] = df["pos_shift"] * df["ret"] * notional

    # Kosten
    # Commission (bps) pro gehandeltem Notional beim Umschalten
    turn = (
        df.groupby("symbol", group_keys=False)["pos"]
          .apply(_turnover)
          .reset_index(level=0, drop=True)
    )
    df["turnover"] = turn

    commission = costs.commission_bps / 10_000.0
    # Einfaches Spread-/Impact-Modell proportional zum Turnover
    spread_cost = costs.spread_w * 0.5 / 10_000.0  # 0.5 bps baseline * weight
    impact_cost = costs.impact_w * 0.5 / 10_000.0  # 0.5 bps baseline * weight

    df["costs"] = (commission + spread_cost + impact_cost) * df["turnover"] * notional
    df["pnl_net"] = df["pnl_gross"] - df["costs"]

    # Aggregation über Symbole → Zeitreihe
    pnl_t = df.groupby("timestamp", as_index=False)["pnl_net"].sum()
    pnl_t = pnl_t.sort_values("timestamp")
    equity = pnl_t["pnl_net"].cumsum() + notional

    # Kennzahlen
    trades = int(df["turnover"].astype(float).gt(0).sum())
    pnl_pos = df["pnl_net"].clip(lower=0).sum()
    pnl_neg = -df["pnl_net"].clip(upper=0).sum()
    profit_factor = (pnl_pos / pnl_neg) if pnl_neg > 0 else math.inf

    # Sharpe (einfach: mean/std der Tages-/Bar-Returns des Portfolios)
    port_ret = pnl_t["pnl_net"] / notional
    sharpe = float(port_ret.mean() / (port_ret.std(ddof=1) + 1e-12))

    equity_df = pd.DataFrame({"timestamp": pnl_t["timestamp"], "equity": equity})
    row = pd.DataFrame(
        [{
            "commission_bps": costs.commission_bps,
            "spread_w": costs.spread_w,
            "impact_w": costs.impact_w,
            "profit_factor": float(profit_factor),
            "sharpe": float(sharpe),
            "trades": trades,
            "final_equity": float(equity.iloc[-1]),
        }]
    )
    return equity_df, row


def run_grid(
    df: pd.DataFrame,
    commissions: Iterable[float],
    spread_ws: Iterable[float],
    impact_ws: Iterable[float],
    notional: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    signal = build_signal(df)

    curves: List[pd.DataFrame] = []
    rows: List[pd.DataFrame] = []

    for c in commissions:
        for sw in spread_ws:
            for iw in impact_ws:
                e, r = simulate_costs(signal, CostParams(c, sw, iw), notional)
                # Output für spätere CSV/Plot optional sammeln
                e = e.copy()
                e["commission_bps"] = c
                e["spread_w"] = sw
                e["impact_w"] = iw
                curves.append(e)
                rows.append(r)

    curves_df = pd.concat(curves, ignore_index=True)
    res_df = pd.concat(rows, ignore_index=True)
    return curves_df, res_df


# ---------- Report ----------

def write_report(res_df: pd.DataFrame, out_md: Path, freq: str) -> str:
    res = res_df.copy()
    res = res.sort_values(["profit_factor", "sharpe", "final_equity"], ascending=[False, False, False])
    best = res.iloc[0].to_dict()

    # Markdown
    lines = []
    lines.append(f"# Cost Grid Report ({freq})")
    lines.append("")
    lines.append("## Bestes Setting")
    lines.append("")
    lines.append(
        f"- **PF**={best['profit_factor']:.4f} | **Sharpe**={best['sharpe']:.4f} | "
        f"**Final Equity**={best['final_equity']:.2f} | **Trades**={int(best['trades'])}"
    )
    lines.append(
        f"- **Commission**={best['commission_bps']} bps | **SpreadW**={best['spread_w']} | **ImpactW**={best['impact_w']}"
    )
    lines.append("")
    lines.append("## Tabelle")
    lines.append("")
    tab = res.copy()
    tab["profit_factor"] = tab["profit_factor"].map(lambda x: f"{x:.4f}")
    tab["sharpe"] = tab["sharpe"].map(lambda x: f"{x:.4f}")
    tab["final_equity"] = tab["final_equity"].map(lambda x: f"{x:.2f}")
    lines.append(tab.to_markdown(index=False))

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")

    # Für RunAll-Log-Zeile zurückgeben:
    summary = (
        f"Bestes Grid → PF={best['profit_factor']} | comm={best['commission_bps']}bps | "
        f"spread={best['spread_w']} | impact={best['impact_w']} | trades={int(best['trades'])} | "
        f"equity={int(best['final_equity'])} | sharpe={best['sharpe']}"
    )
    return summary


# ---------- Main ----------

def main():
    args = parse_args()

    print(f"[GRID] START Grid | freq={args.freq} | "
          f"commission={list(args.commission_bps)} | spread_w={list(args.spread_ws)} | impact_w={list(args.impact_ws)}")

    df = load_features(args.freq)
    curves_df, res_df = run_grid(
        df,
        commissions=args.commission_bps,
        spread_ws=args.spread_ws,
        impact_ws=args.impact_ws,
        notional=args.notional,
    )

    # (Optional) CSVs ablegen
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cost_grid_curves.csv").write_text(
        curves_df.to_csv(index=False), encoding="utf-8"
    )
    (OUT / "cost_grid_results.csv").write_text(
        res_df.to_csv(index=False), encoding="utf-8"
    )

    md = OUT / "cost_grid_report.md"
    summary = write_report(res_df, md, args.freq)
    print(f"[GRID] [OK] written: {md}")
    print("[GRID] DONE Grid")
    # auch die kurze Zusammenfassung für nachgelagertes Parsing ausgeben:
    print(summary)


if __name__ == "__main__":
    main()


