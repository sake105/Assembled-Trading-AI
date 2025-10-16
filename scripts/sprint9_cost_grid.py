# scripts/sprint9_cost_grid.py
# Sprint 9 – Cost Grid (robust: transform/shift statt apply; CLI kompatibel)

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
FEAT_DIR = OUT / "features"
AGG_DIR = OUT / "aggregates"

def info(msg: str) -> None:
    print(f"[GRID] {msg}")

def _to_float(s) -> float:
    # akzeptiert "0,5" und "0.5"
    if isinstance(s, (int, float)):
        return float(s)
    return float(str(s).replace(",", ".").strip())

def _to_float_list(vals, default):
    if not vals:
        return default
    return [_to_float(v) for v in vals]

def _read_assembled(freq: str) -> pd.DataFrame:
    fp = AGG_DIR / f"assembled_intraday_{freq}.parquet"
    if not fp.exists():
        raise FileNotFoundError(f"Not found: {fp}")
    df = pd.read_parquet(fp)

    lower = {c.lower(): c for c in df.columns}
    need = ["timestamp", "symbol", "close"]
    miss = [c for c in need if c not in lower]
    if miss:
        raise KeyError(f"Missing columns in assembled parquet: {miss}")

    df = df.rename(columns={
        lower["timestamp"]: "timestamp",
        lower["symbol"]: "symbol",
        lower["close"]: "close",
    })
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    return df[["timestamp", "symbol", "close"]]

def _read_features(freq: str) -> pd.DataFrame | None:
    base_fp = FEAT_DIR / f"base_{freq}.parquet"
    reg_fp  = FEAT_DIR / f"regime_{freq}.parquet"
    if not base_fp.exists():
        return None

    base = pd.read_parquet(base_fp)
    lower = {c.lower(): c for c in base.columns}
    need = ["timestamp", "symbol", "close"]
    miss = [c for c in need if c not in lower]
    if miss:
        return None

    base = base.rename(columns={
        lower["timestamp"]: "timestamp",
        lower["symbol"]: "symbol",
        lower["close"]: "close",
    })
    base["timestamp"] = pd.to_datetime(base["timestamp"], utc=True, errors="coerce")
    base = base.dropna(subset=["timestamp"]).sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    if reg_fp.exists():
        reg = pd.read_parquet(reg_fp)
        cand = [c for c in reg.columns if c.lower() in ("trend_regime", "regime_trend", "regime")]
        if cand:
            reg = reg.rename(columns={cand[0]: "trend_regime"})
            reg["timestamp"] = pd.to_datetime(reg["timestamp"], utc=True, errors="coerce")
            reg = reg.dropna(subset=["timestamp"])
            base = base.merge(reg[["timestamp", "symbol", "trend_regime"]], on=["timestamp", "symbol"], how="left")

    return base

def _ensure_trend_regime(df: pd.DataFrame) -> pd.DataFrame:
    if "trend_regime" in df.columns:
        return df
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    close = df["close"].astype(float)
    # EMA je Symbol mit transform → index-kompatibel
    ema = close.groupby(df["symbol"]).transform(lambda s: s.ewm(span=20, adjust=False, min_periods=5).mean())
    regime = np.sign(close - ema)
    regime = pd.Series(regime, index=df.index).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(int)
    df["trend_regime"] = regime
    return df

def simple_signal(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_trend_regime(df)
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    # Position = gestriges Regime (per Symbol)
    pos = df.groupby("symbol")["trend_regime"].shift(1).fillna(0).astype(int)
    # Trade = Positionsänderung je Symbol
    prev_pos = pos.groupby(df["symbol"]).shift(1).fillna(0).astype(int)
    trade = (pos - prev_pos).astype(int)

    out = df.copy()
    out["pos"] = pos
    out["trade"] = trade
    return out[["timestamp", "symbol", "pos", "trade", "close"]]

def simulate_costs(df_sig: pd.DataFrame,
                   commission_bps: float,
                   spread_w: float,
                   impact_w: float) -> dict:
    df = df_sig.sort_values(["symbol", "timestamp"]).reset_index(drop=True).copy()

    # Log-Returns je Symbol → transform für 1:1 Index-Ausrichtung
    logp = np.log(df["close"].astype(float))
    df["ret"] = logp.groupby(df["symbol"]).diff().fillna(0.0)

    # Brutto-P&L: Vortages-Position * Return (je Symbol shiften)
    pos_lag = df.groupby("symbol")["pos"].shift(1).fillna(0).astype(float)
    df["gross"] = pos_lag * df["ret"]

    # Transaktionskosten
    dpos = df["trade"].abs().astype(float)
    comm = commission_bps * 1e-4
    volproxy = df["ret"].abs().clip(upper=0.01)
    cost_unit = comm + 0.5 * spread_w * volproxy + 0.5 * impact_w * volproxy
    df["tcost"] = dpos * cost_unit

    df["net"] = df["gross"] - df["tcost"]

    # Aggregation über Zeit
    s = df.groupby("timestamp")["net"].sum().sort_index()
    mu = s.mean()
    sd = s.std(ddof=1)
    sharpe = float(mu / sd * np.sqrt(252 * 78)) if sd > 0 else 0.0
    pnl_pos = s[s > 0].sum()
    pnl_neg = -s[s < 0].sum()
    pf = float(pnl_pos / pnl_neg) if pnl_neg > 0 else np.inf

    return {
        "pf": pf,
        "sharpe": sharpe,
        "trades": int(dpos.sum()),
        "commission_bps": commission_bps,
        "spread_w": spread_w,
        "impact_w": impact_w,
    }

def run_grid(df: pd.DataFrame,
             commissions: list[float],
             spread_ws: list[float],
             impact_ws: list[float]) -> pd.DataFrame:
    sig = simple_signal(df)
    rows = []
    for c in commissions:
        for sw in spread_ws:
            for iw in impact_ws:
                rows.append(simulate_costs(sig, c, sw, iw))
    return pd.DataFrame(rows).sort_values(["commission_bps", "spread_w", "impact_w"])

def write_report(res: pd.DataFrame, fp: Path) -> None:
    fp.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Execution Cost Grid\n")
    lines.append("")
    lines.append("| commission (bps) | spread_w | impact_w | PF | Sharpe | Trades |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for _, r in res.iterrows():
        lines.append(
            f"| {r['commission_bps']:.2f} | {r['spread_w']:.2f} | {r['impact_w']:.2f} | "
            f"{r['pf']:.2f} | {r['sharpe']:.2f} | {int(r['trades'])} |"
        )
    fp.write_text("\n".join(lines), encoding="utf-8")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freq", default="5min")
    ap.add_argument("--notional", default=None)  # akzeptiert, aber unbenutzt (normierte Sim)
    ap.add_argument("--commission-bps", nargs="*", default=None)
    ap.add_argument("--spread-w", nargs="*", default=None)
    ap.add_argument("--impact-w", nargs="*", default=None)
    args = ap.parse_args()

    commissions = _to_float_list(args.commission_bps, default=[0.0, 0.5, 1.0])
    spread_ws   = _to_float_list(args.spread_w,      default=[0.5, 1.0, 2.0])
    impact_ws   = _to_float_list(args.impact_w,      default=[0.5, 1.0, 2.0])

    info(f"START Grid | freq={args.freq} "
         f"| commission={', '.join(str(x) for x in commissions)} "
         f"| spread_w={', '.join(str(x) for x in spread_ws)} "
         f"| impact_w={', '.join(str(x) for x in impact_ws)}")

    df_feat = _read_features(args.freq)
    df = df_feat if df_feat is not None else _read_assembled(args.freq)

    if not {"timestamp", "symbol", "close"}.issubset(df.columns):
        raise KeyError("DataFrame lacks required columns: timestamp, symbol, close")

    res = run_grid(df, commissions, spread_ws, impact_ws)
    rep = OUT / "cost_grid_report.md"
    write_report(res, rep)
    info(f"[OK] written: {rep}")
    info("DONE Grid")

if __name__ == "__main__":
    main()
