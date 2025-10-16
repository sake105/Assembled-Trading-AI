# scripts/sprint9_cost_grid.py
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Tuple, List

import numpy as np
import pandas as pd

# ----------------------------
# CLI & Paths
# ----------------------------
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
FEATURES_DIR = OUT / "features"

@dataclass(frozen=True)
class CostParams:
    commission_bps: float
    spread_w: float
    impact_w: float

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--freq", default="5min")
    p.add_argument("--notional", type=float, default=10_000.0)
    p.add_argument("--commission-bps", nargs="*", type=float, default=[0.0, 0.5, 1.0, 2.0, 5.0])
    p.add_argument("--spread-w",       nargs="*", type=float, default=[0.25, 0.5, 1.0, 2.0, 3.0])
    p.add_argument("--impact-w",       nargs="*", type=float, default=[0.0, 0.25, 0.5, 1.0, 2.0])
    p.add_argument("--signal", choices=["simple","ema","regime"], default="ema")
    p.add_argument("--plot", action="store_true", help="Optional: Heatmaps/Curves plotten, wenn matplotlib installiert ist")
    return p.parse_args()

# ----------------------------
# Loading & sanitizing
# ----------------------------
def _load_parquet_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Nicht gefunden: {path}")
    df = pd.read_parquet(path)
    # timestamp robust → tz-aware UTC
    if "timestamp" in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        elif df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
        else:
            df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")
    # Spaltennamen normalisieren
    df.columns = [str(c) for c in df.columns]
    return df

def load_features(freq: str) -> pd.DataFrame:
    # Wir brauchen mindestens base + micro; regime optional
    base = _load_parquet_safe(FEATURES_DIR / f"base_{freq}.parquet")
    micro = _load_parquet_safe(FEATURES_DIR / f"micro_{freq}.parquet")
    maybe_regime = (FEATURES_DIR / f"regime_{freq}.parquet")
    if maybe_regime.exists():
        regime = _load_parquet_safe(maybe_regime)
        df = base.merge(micro, on=["timestamp","symbol"], how="outer").merge(regime, on=["timestamp","symbol"], how="outer")
    else:
        df = base.merge(micro, on=["timestamp","symbol"], how="outer")

    # Duplikate bereinigen
    df = df.drop_duplicates(subset=["timestamp","symbol"]).sort_values(["timestamp","symbol"])
    # Basisfelder sichern
    needed = ["timestamp","symbol","close"]
    for c in needed:
        if c not in df.columns:
            raise KeyError(f"Fehlende Spalte '{c}' in Features.")
    # NaN/Inf cleanup
    for c in ["open","high","low","close","volume"]:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            df[c] = s.replace([np.inf,-np.inf], np.nan)
    df = df.sort_values(["timestamp","symbol"])
    return df

# ----------------------------
# Signals
# ----------------------------
def signal_simple(df: pd.DataFrame) -> pd.DataFrame:
    """Toy-Signal: +1 wenn close über 5-EMA, -1 wenn darunter."""
    def _per_symbol(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("timestamp").copy()
        ema = g["close"].ewm(span=5, adjust=False).mean()
        g["weight"] = np.where(g["close"] > ema, 1.0, -1.0)
        return g
    return df.groupby("symbol", group_keys=False).apply(_per_symbol)

def signal_ema(df: pd.DataFrame) -> pd.DataFrame:
    """Cross von kurzen/längeren EMAs → -1/0/+1."""
    def _per_symbol(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("timestamp").copy()
        ema_fast = g["close"].ewm(span=8, adjust=False).mean()
        ema_slow = g["close"].ewm(span=21, adjust=False).mean()
        sig = np.sign((ema_fast - ema_slow).fillna(0.0))
        # Glätten, um Churn zu reduzieren
        sig = sig.rolling(3, min_periods=1).mean().round().clip(-1,1)
        g["weight"] = sig.astype(float)
        return g
    return df.groupby("symbol", group_keys=False).apply(_per_symbol)

def signal_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Wenn 'trend_regime' vorhanden: >0 → long, <0 → short, sonst 0."""
    if "trend_regime" not in df.columns:
        # Fallback: ema
        return signal_ema(df)
    def _per_symbol(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("timestamp").copy()
        sig = pd.to_numeric(g["trend_regime"], errors="coerce").fillna(0.0)
        g["weight"] = np.sign(sig).astype(float)
        return g
    return df.groupby("symbol", group_keys=False).apply(_per_symbol)

# ----------------------------
# Cost model & simulation
# ----------------------------
def _portfolio_returns(df: pd.DataFrame, weight_col: str = "weight") -> pd.Series:
    # Log-Returns je Symbol
    ret = df.groupby("symbol")["close"].transform(lambda s: np.log(s).diff()).fillna(0.0)
    w = pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0)

    # Rebalance-Kosten: Kommission + Spread/Impact pro Gewichtswechsel
    dw = w.groupby(df["symbol"]).transform(lambda s: s.diff().abs()).fillna(0.0)
    return ret, w, dw

def simulate_costs(df: pd.DataFrame, costs: CostParams, notional: float) -> Tuple[pd.Series, dict]:
    df = df.sort_values(["timestamp","symbol"]).copy()

    # returns & weights
    ret, w, dw = _portfolio_returns(df)

    # Brutto-Portfolio-Return = Sum over symbols (w * ret) (gleichgewichtet über verfügbare Symbole je Timestamp)
    # Pivots für saubere Zeitachsen-Aggregation
    idx = df["timestamp"]
    sym = df["symbol"]

    # Werte in breite Matrizen
    ret_wide = pd.pivot_table(pd.DataFrame({"timestamp": idx, "symbol": sym, "ret": ret}),
                              index="timestamp", columns="symbol", values="ret", aggfunc="first").sort_index()
    w_wide   = pd.pivot_table(pd.DataFrame({"timestamp": idx, "symbol": sym, "w": w}),
                              index="timestamp", columns="symbol", values="w", aggfunc="first").sort_index()
    dw_wide  = pd.pivot_table(pd.DataFrame({"timestamp": idx, "symbol": sym, "dw": dw}),
                              index="timestamp", columns="symbol", values="dw", aggfunc="first").sort_index()

    # Vorwärts auffüllen (Timestamps konsistent)
    ret_wide = ret_wide.fillna(0.0)
    w_wide   = w_wide.ffill().fillna(0.0)
    dw_wide  = dw_wide.fillna(0.0)

    # Kosten in Brutto-Return (bps → Dezimal)
    commission = costs.commission_bps * 1e-4
    trading_cost = commission * dw_wide + costs.spread_w * 1e-4 * dw_wide + costs.impact_w * 1e-4 * dw_wide

    gross = (w_wide * ret_wide).mean(axis=1)  # gleichgewichtet über Symbole
    net = gross - trading_cost.mean(axis=1)

    # Equity-Kurve
    eq = (np.log1p(net).cumsum()).apply(np.expm1)
    eq = (1.0 + eq) * notional

    # Kennzahlen
    pnl = eq.diff().fillna(0.0)
    vol = pnl.std(ddof=0)
    mean = pnl.mean()
    sharpe = 0.0 if vol == 0 else (mean / vol) * np.sqrt(252 * (6.5*60/5))  # intraday 5min ~ 78 Bars/Tag
    pf = 0.0
    gains = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    if losses > 0:
        pf = gains / losses

    stats = dict(
        pf=float(pf),
        sharpe=float(sharpe),
        trades=int((dw_wide.values > 0).sum()),
        final_equity=float(eq.iloc[-1]),
    )
    return eq.astype(float), stats

# ----------------------------
# Grid driver
# ----------------------------
def run_grid(df: pd.DataFrame,
             commissions: Iterable[float],
             spread_ws: Iterable[float],
             impact_ws: Iterable[float],
             notional: float
             ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Signal wählen
    sig_df = {
        "simple": signal_simple,
        "ema":    signal_ema,
        "regime": signal_regime,
    }[args.signal](df)

    curves: List[pd.Series] = []
    rows: List[dict] = []

    for c in commissions:
        for sw in spread_ws:
            for iw in impact_ws:
                eq, stats = simulate_costs(sig_df, CostParams(c, sw, iw), notional)
                tag = f"c{c}_sw{sw}_iw{iw}"
                eq.name = tag
                curves.append(eq)
                rows.append(dict(commission_bps=c, spread_w=sw, impact_w=iw, **stats))

    curves_df = pd.concat(curves, axis=1)
    res_df = pd.DataFrame(rows).sort_values(["commission_bps","spread_w","impact_w"]).reset_index(drop=True)
    return curves_df, res_df

# ----------------------------
# Reporting
# ----------------------------
def write_report(freq: str, res: pd.DataFrame, curves: pd.DataFrame) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # Markdown
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    best = res.sort_values("pf", ascending=False).iloc[0]
    md = [
        f"# Cost Grid Report ({freq})",
        "",
        f"Generated: **{ts}**  |  Best PF: **{best['pf']:.2f}** | comm={best['commission_bps']}, spread={best['spread_w']}, impact={best['impact_w']}",
        "",
        res.to_markdown(index=False),
        ""
    ]
    (OUT / "cost_grid_report.md").write_text("\n".join(md), encoding="utf-8")
    # CSV/Parquet
    res.to_csv(OUT / "cost_grid_results.csv", index=False)
    curves.to_parquet(OUT / "cost_grid_curves.parquet")

def try_plot(res: pd.DataFrame, curves: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
        import itertools

        # Heatmap PF über (spread_w × impact_w) je commission
        commissions = sorted(res["commission_bps"].unique())
        for c in commissions:
            sub = res[res["commission_bps"] == c].pivot(index="spread_w", columns="impact_w", values="pf")
            plt.figure()
            plt.imshow(sub.values, aspect="auto")
            plt.xticks(range(len(sub.columns)), sub.columns)
            plt.yticks(range(len(sub.index)), sub.index)
            plt.title(f"PF heatmap (commission={c} bps)")
            plt.colorbar()
            plt.xlabel("impact_w")
            plt.ylabel("spread_w")
            plt.tight_layout()
            plt.savefig(OUT / f"cost_grid_heatmap_comm_{c}.png", dpi=140)
            plt.close()

        # Top-3 Equity
        top3 = res.sort_values("pf", ascending=False).head(3)
        plt.figure()
        for _, row in top3.iterrows():
            tag = f"c{row['commission_bps']}_sw{row['spread_w']}_iw{row['impact_w']}"
            if tag in curves.columns:
                curves[tag].plot()
        plt.title("Top-3 Equity Curves (by PF)")
        plt.xlabel("time")
        plt.ylabel("equity")
        plt.tight_layout()
        plt.savefig(OUT / "cost_grid_top3_equity.png", dpi=140)
        plt.close()
    except Exception as e:
        # plotting ist optional → nur still schlucken
        pass

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    args = parse_args()
    print(f"[GRID] START Grid | freq={args.freq} | commission={args.commission_bps} | spread_w={args.spread_w} | impact_w={args.impact_w}")

    df = load_features(args.freq)
    curves_df, res_df = run_grid(df, args.commission_bps, args.spread_w, args.impact_w, notional=args.notional)

    write_report(args.freq, res_df, curves_df)
    if args.plot:
        try_plot(res_df, curves_df)

    print(f"[GRID] [OK] written: {OUT / 'cost_grid_report.md'}")
    print("[GRID] DONE Grid")
