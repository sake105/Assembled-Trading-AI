#!/usr/bin/env python
"""Anomaly-Audit aller verfügbaren Equity-Curves.

Wendet ``erweiterung.qa.equity_curve_audit`` auf:
- Original-Mainline-Equity-Curves: ``output/equity_curve_*.csv``
- Erweiterung-Curves: ``output/erweiterung_*_equity.csv``

Ziel: zeige objektiv messbare Inkonsistenzen, die der Original-Backtest
gegenüber den Erweiterungs-Backtests aufweist.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from erweiterung.qa.equity_curve_audit import audit_equity_curve  # noqa: E402


def _load_original_equity(path: Path) -> pd.Series | None:
    df = pd.read_csv(path)
    # Standard: date,timestamp,equity,daily_return,cash
    if "equity" not in df.columns:
        return None
    if "timestamp" in df.columns:
        df["date"] = pd.to_datetime(df["timestamp"], utc=True)
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], utc=True)
    else:
        df["date"] = pd.to_datetime(df.iloc[:, 0], utc=True)
    return df.set_index("date")["equity"].sort_index()


def _load_erweiterung_multi_equity(path: Path) -> dict[str, pd.Series]:
    """Erweiterung-CSVs haben mehrere Strategien als Spalten."""
    df = pd.read_csv(path)
    first_col = df.columns[0]
    if first_col in ("date", "Unnamed: 0", "Date"):
        df = df.rename(columns={first_col: "date"})
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.set_index("date").sort_index()
    return {c: df[c] for c in df.columns if df[c].notna().sum() > 50}


def _market_proxy() -> pd.Series | None:
    """Lade SPY als Markt-Referenz."""
    f = Path("data/cache/yfinance/SPY.parquet")
    if not f.exists():
        return None
    spy = pd.read_parquet(f).reset_index()
    if "date" not in spy.columns and "Date" in spy.columns:
        spy = spy.rename(columns={"Date": "date"})
    spy["date"] = pd.to_datetime(spy["date"], utc=True)
    return spy.set_index("date")["close"].sort_index()


def main():
    market = _market_proxy()

    results: list[dict] = []

    # Original-Mainline-Equity-Curves
    orig_files = [
        Path("output/equity_curve_baseline.csv"),
        Path("output/equity_curve_1d.csv"),
        Path("output/equity_curve_t2_nolev_2025_26.csv"),
        Path("output/equity_curve_t3_2023_24.csv"),
        Path("output/equity_curve_altdata.csv"),
        Path("output/equity_curve_test1_aitech_qagate.csv"),
    ]
    for p in orig_files:
        if not p.exists():
            continue
        eq = _load_original_equity(p)
        if eq is None or len(eq) < 60:
            continue
        # Auf gleichen Zeitraum wie SPY trimmen für Korrelation
        if market is not None:
            bench = market.loc[eq.index.min() : eq.index.max()]
        else:
            bench = None
        audit = audit_equity_curve(
            eq, name=f"ORIG::{p.name}", bootstrap_benchmark=bench
        )
        results.append(audit.to_dict())

    # Erweiterung
    erw_eq_csv = Path("output/erweiterung_expanded_universe_equity.csv")
    if erw_eq_csv.exists():
        for name, ser in _load_erweiterung_multi_equity(erw_eq_csv).items():
            bench = (
                market.loc[ser.index.min() : ser.index.max()]
                if market is not None
                else None
            )
            audit = audit_equity_curve(
                ser, name=f"ERW::{name}", bootstrap_benchmark=bench
            )
            results.append(audit.to_dict())

    erw_regime_csv = Path("output/erweiterung_regime_conditional_equity.csv")
    if erw_regime_csv.exists():
        df = pd.read_csv(erw_regime_csv)
        if "date" in df.columns and "equity" in df.columns:
            df["date"] = pd.to_datetime(df["date"], utc=True)
            ser = df.set_index("date")["equity"].sort_index()
            bench = (
                market.loc[ser.index.min() : ser.index.max()]
                if market is not None
                else None
            )
            audit = audit_equity_curve(
                ser, name="ERW::regime_conditional", bootstrap_benchmark=bench
            )
            results.append(audit.to_dict())

    # Tabular print
    print("\n" + "=" * 130)
    print("EQUITY-CURVE ANOMALY AUDIT")
    print("=" * 130)
    print(
        f"{'Name':<48} {'Sharpe':>7} {'AC1':>7} {'AC5':>7} {'Skew':>6} {'Kurt':>7} {'MDD':>7} {'WD/Vol':>7} {'Flags':<30}"
    )
    print("-" * 130)
    for r in results:
        sharpe = r.get("overall_sharpe")
        ac1 = r.get("return_autocorr_lag1")
        ac5 = r.get("return_autocorr_lag5")
        sk = r.get("skew")
        kt = r.get("kurtosis")
        mdd = r.get("max_drawdown")
        wdv = r.get("worst_day_vol_ratio")
        flags = ", ".join(r.get("flags", [])[:3])
        print(
            f"  {r['name']:<46} "
            f"{(sharpe or 0):>+6.2f} "
            f"{(ac1 or 0):>+6.2f} "
            f"{(ac5 or 0):>+6.2f} "
            f"{(sk or 0):>+5.2f} "
            f"{(kt or 0):>+6.2f} "
            f"{(mdd or 0):>+6.2%} "
            f"{(wdv or 0):>+6.2f} "
            f"{flags:<30}"
        )

    Path("output/equity_curve_audit.json").write_text(
        json.dumps(results, indent=2, default=str)
    )
    print(f"\nSaved -> output/equity_curve_audit.json ({len(results)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
