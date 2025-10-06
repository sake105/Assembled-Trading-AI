# sprint10_portfolio.py
# Portfolio-Layer (Sprint 10) mit Kostenmodell:
#   - Kommission: commission_bps * Turnover
#   - Spread:     spread_w   * spread_proxy_bps * Turnover
#   - Impact:     impact_w   * impact_proxy_bps * Turnover
#
# Erwartete Dateien:
#   Preise:  output/aggregates/assembled_intraday_<freq>.parquet
#   Features (optional): output/features/micro_<freq>.parquet  mit Spalten:
#                        'spread_proxy' (in bps), optional 'impact_proxy' (in bps)
#
# Robuste Punkte:
#   - De-Dupe vor Pivot (timestamp,symbol)
#   - Fallbacks, wenn Features fehlen
#   - Reports & CSV-Ausgabe wie bisher

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
AGG_DIR = OUT / "aggregates"
FEAT_DIR = OUT / "features"


@dataclass
class Args:
    freq: str
    start_capital: float
    exposure: float
    max_leverage: float
    commission_bps: float
    spread_w: float
    impact_w: float


def parse_args() -> Args:
    p = argparse.ArgumentParser(description="Sprint 10 Portfolio Simulator (mit Spread/Impact-Kosten)")
    p.add_argument("--freq", dest="freq", default="5min")
    p.add_argument("--start-capital", dest="start_capital", type=float, default=10_000.0)
    p.add_argument("--exposure", dest="exposure", type=float, default=1.0)
    p.add_argument("--max-leverage", dest="max_leverage", type=float, default=1.0)
    p.add_argument("--commission-bps", dest="commission_bps", type=float, default=0.0)
    p.add_argument("--spread-w", dest="spread_w", type=float, default=1.0)
    p.add_argument("--impact-w", dest="impact_w", type=float, default=1.0)
    a = p.parse_args()
    return Args(a.freq, a.start_capital, a.exposure, a.max_leverage, a.commission_bps, a.spread_w, a.impact_w)


def _log(msg: str) -> None:
    ts = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{ts}] [PF10] {msg}")


def load_prices(freq: str) -> pd.DataFrame:
    cand = [
        AGG_DIR / f"assembled_intraday_{freq}.parquet",
        AGG_DIR / f"{freq}.parquet",
    ]
    for f in cand:
        if f.exists():
            df = pd.read_parquet(f)
            break
    else:
        raise FileNotFoundError(f"Keine Aggregates gefunden (gesucht: {', '.join(str(x) for x in cand)})")

    # Spalten normalisieren
    cols = {c.lower(): c for c in df.columns}
    def has(name: str) -> bool: return name in cols or name in df.columns

    if has("timestamp"):
        ts_col = cols.get("timestamp", "timestamp")
    elif has("time"):
        ts_col = cols.get("time", "time"); df = df.rename(columns={ts_col: "timestamp"}); ts_col = "timestamp"
    else:
        raise ValueError("Spalte 'timestamp' fehlt in Aggregates")

    if has("symbol"):
        sym_col = cols.get("symbol", "symbol")
    elif has("ticker"):
        sym_col = cols.get("ticker", "ticker"); df = df.rename(columns={sym_col: "symbol"}); sym_col = "symbol"
    else:
        raise ValueError("Spalte 'symbol' fehlt in Aggregates")

    lc = {k.lower(): k for k in df.columns}
    close = lc.get("close")
    if close is None:
        px = lc.get("px") or lc.get("price")
        if px is not None:
            df = df.rename(columns={px: "close"})
        else:
            open_c = lc.get("open"); high_c = lc.get("high"); low_c = lc.get("low"); close_c = lc.get("close")
            if all(v is not None for v in [open_c, high_c, low_c, close_c]):
                df["close"] = (df[open_c] + df[high_c] + df[low_c] + df[close_c]) / 4.0
            else:
                raise ValueError("Konnte 'close' nicht herleiten (fehlende Preis-Spalten).")

    df = df.rename(columns={ts_col: "timestamp", sym_col: "symbol"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.sort_values(["timestamp", "symbol"]).dropna(subset=["timestamp", "symbol"])

    dups = int(df.duplicated(["timestamp", "symbol"]).sum())
    if dups:
        _log(f"WARN: found {dups} duplicate (timestamp,symbol) rows – collapsing by last")
        df = df.drop_duplicates(["timestamp", "symbol"], keep="last")

    return df[["timestamp", "symbol", "close"]].copy()


def load_signals(freq: str, symbols: list[str], timeline: pd.DatetimeIndex) -> pd.DataFrame:
    file = FEAT_DIR / f"regime_{freq}.parquet"
    if file.exists():
        sig = pd.read_parquet(file)
        for c in ("timestamp", "symbol"):
            if c not in sig.columns and c.upper() in sig.columns:
                sig = sig.rename(columns={c.upper(): c})
        if "risk_on" not in sig.columns:
            alt = [c for c in sig.columns if c.lower() in ("risk_on", "riskon", "regime", "li_regime")]
            if alt:
                sig["risk_on"] = (sig[alt[0]] > 0).astype(int)
            else:
                return pd.DataFrame(1.0, index=timeline, columns=symbols)
        sig["timestamp"] = pd.to_datetime(sig["timestamp"], utc=True, errors="coerce")
        sig = sig.dropna(subset=["timestamp", "symbol"]).sort_values(["timestamp", "symbol"])
        sig = sig.drop_duplicates(["timestamp", "symbol"], keep="last")
        mat = sig.pivot_table(index="timestamp", columns="symbol", values="risk_on", aggfunc="last")
        mat = mat.reindex(index=timeline, columns=symbols).fillna(method="ffill").fillna(0.0)
        return mat.clip(lower=0.0, upper=1.0)
    else:
        return pd.DataFrame(1.0, index=timeline, columns=symbols)


def load_cost_proxies(freq: str,
                      symbols: list[str],
                      timeline: pd.DatetimeIndex,
                      rets: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Liefert (spread_proxy_bps_series, impact_proxy_bps_series) als Zeitreihen.
    Fallbacks:
      - Wenn micro_<freq>.parquet fehlt: beide 0.
      - Wenn 'impact_proxy' fehlt: Approx über Rolling-Vol der Returns (in bps).
    """
    spread_bps = pd.Series(0.0, index=timeline)
    impact_bps = pd.Series(0.0, index=timeline)

    file = FEAT_DIR / f"micro_{freq}.parquet"
    if file.exists():
        micro = pd.read_parquet(file)
        # Normalisieren
        for c in ("timestamp", "symbol"):
            if c not in micro.columns and c.upper() in micro.columns:
                micro = micro.rename(columns={c.upper(): c})
        micro["timestamp"] = pd.to_datetime(micro["timestamp"], utc=True, errors="coerce")
        micro = micro.dropna(subset=["timestamp"]).sort_values(["timestamp", "symbol"])

        # Spread
        if "spread_proxy" in micro.columns:
            sp = micro.pivot_table(index="timestamp", columns="symbol", values="spread_proxy", aggfunc="last")
            # Mittel über Symbole (gleichgewichtet); fehlende Spalten werden gelassen
            spread_bps = sp.reindex(index=timeline, columns=symbols).ffill().mean(axis=1).fillna(0.0)

        # Impact
        if "impact_proxy" in micro.columns:
            ip = micro.pivot_table(index="timestamp", columns="symbol", values="impact_proxy", aggfunc="last")
            impact_bps = ip.reindex(index=timeline, columns=symbols).ffill().mean(axis=1).fillna(0.0)

    # Fallback Impact: nutze Rolling-Vol der Returns (z.B. 12 Bars) und wandle in bps um
    if (impact_bps == 0).all():
        vol = rets.abs().rolling(12, min_periods=1).mean().mean(axis=1)  # ~ durchschnittl. |Return|
        impact_bps = (vol * 1e4).fillna(0.0)  # returns -> bps

    # Negative/NaN schützen
    spread_bps = spread_bps.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    impact_bps = impact_bps.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    return spread_bps, impact_bps


def simulate(df_prices: pd.DataFrame,
             signals: pd.DataFrame,
             args: Args) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    timeline = pd.DatetimeIndex(df_prices["timestamp"].sort_values().unique())
    symbols = sorted(df_prices["symbol"].unique().tolist())

    px_close = (
        df_prices.pivot_table(index="timestamp", columns="symbol", values="close", aggfunc="last")
        .reindex(index=timeline, columns=symbols)
        .ffill()
    )
    rets = px_close.pct_change().fillna(0.0)

    sig = signals.reindex(index=timeline, columns=symbols).fillna(0.0).clip(0.0, 1.0)
    active = sig.sum(axis=1).replace(0, np.nan)
    weights = sig.div(active, axis=0).fillna(0.0) * float(args.exposure)

    tot_abs = weights.abs().sum(axis=1)
    scale = (tot_abs / max(args.max_leverage, 1e-9))
    over = scale > 1.0
    if over.any():
        weights.loc[over] = weights.loc[over].div(scale.loc[over], axis=0)

    gross_ret = (weights.shift(1).fillna(0.0) * rets).sum(axis=1)

    # Turnover-basiertes Kostenmodell (bps)
    w_prev = weights.shift(1).fillna(0.0)
    delta = (weights - w_prev)
    turnover = delta.abs().sum(axis=1)

    # Proxies laden (bps-Zeitreihen)
    spread_bps, impact_bps = load_cost_proxies(args.freq, symbols, timeline, rets)

    cost_comm   = (args.commission_bps / 1e4) * turnover
    cost_spread = (args.spread_w     * spread_bps / 1e4) * turnover
    cost_impact = (args.impact_w     * impact_bps / 1e4) * turnover

    net_ret = gross_ret - (cost_comm + cost_spread + cost_impact)

    equity = (1.0 + net_ret).cumprod() * float(args.start_capital)
    equity_df = pd.DataFrame({"timestamp": timeline, "equity": equity.values, "pf_ret": net_ret.values})

    trades_list = []
    for t, ts in enumerate(timeline):
        row_d = delta.iloc[t]
        row_prev = w_prev.iloc[t]
        row_now = weights.iloc[t]
        for s in symbols:
            dw = float(row_d[s])
            if abs(dw) > 1e-12:
                trades_list.append({
                    "timestamp": ts,
                    "symbol": s,
                    "weight_prev": float(row_prev[s]),
                    "weight": float(row_now[s]),
                    "delta_w": dw
                })
    trades_df = pd.DataFrame(trades_list)
    weights_df = weights.copy()
    weights_df.insert(0, "timestamp", timeline)

    return trades_df, equity_df, weights_df


def write_outputs(freq: str,
                  trades: pd.DataFrame,
                  equity: pd.DataFrame,
                  args: Args,
                  spread_impact_note: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    trades_file = OUT / "portfolio_trades.csv"
    equity_file = OUT / f"portfolio_equity_{freq}.csv"
    report_file = OUT / "portfolio_report.md"

    if not trades.empty:
        t = trades.copy()
        t["timestamp"] = pd.to_datetime(t["timestamp"]).dt.tz_convert("UTC")
        t.sort_values(["timestamp", "symbol"], inplace=True)
        t.to_csv(trades_file, index=False)
    else:
        pd.DataFrame(columns=["timestamp", "symbol", "weight_prev", "weight", "delta_w"]).to_csv(trades_file, index=False)

    e = equity.copy()
    e["timestamp"] = pd.to_datetime(e["timestamp"]).dt.tz_convert("UTC")
    e.to_csv(equity_file, index=False)

    lines = ["# Portfolio Report", ""]
    if not e.empty:
        eq = e.set_index("timestamp")["equity"].replace([np.inf, -np.inf], np.nan).dropna()
        if len(eq) >= 2:
            days = max((eq.index[-1] - eq.index[0]).days, 1)
            total_ret = float(eq.iloc[-1] / max(eq.iloc[0], 1e-9) - 1.0)
            cagr = float(((eq.iloc[-1] / max(eq.iloc[0], 1e-9)) ** (365.0 / days) - 1.0)) if days > 0 else np.nan
            dd = float((eq / eq.cummax() - 1.0).min())
            lines += [
                f"- Period days: {days}",
                f"- Total return (net): {total_ret:.4f}",
                f"- CAGR (net): {cagr if np.isfinite(cagr) else 'nan'}",
                f"- Max Drawdown: {dd:.4f}",
                "",
                "## Settings",
                f"- Exposure: {args.exposure}",
                f"- Max Leverage: {args.max_leverage}",
                f"- Commission (bps per turnover): {args.commission_bps}",
                f"- Spread weight: {args.spread_w}",
                f"- Impact weight: {args.impact_w}",
                "",
                "## Notes",
                spread_impact_note.strip()
            ]
    Path(report_file).write_text("\n".join(lines), encoding="utf-8")

    _log(f"[OK] written: {trades_file}")
    _log(f"[OK] written: {equity_file}")
    _log(f"[OK] written: {report_file}")


def main():
    args = parse_args()
    _log(
        f"START Portfolio | freq={args.freq} start_capital={args.start_capital} "
        f"exp={args.exposure} lev={args.max_leverage} "
        f"comm_bps={args.commission_bps} spread_w={args.spread_w} impact_w={args.impact_w}"
    )

    prices = load_prices(args.freq)
    timeline = pd.DatetimeIndex(prices["timestamp"].sort_values().unique())
    symbols = sorted(prices["symbol"].unique().tolist())
    signals = load_signals(args.freq, symbols, timeline)

    trades_df, equity_df, _weights = simulate(prices, signals, args)

    # Hinweistext zu verwendeten Proxies
    note = (
        "- Spread/Impact-Proxys werden aus `output/features/micro_<freq>.parquet` gelesen, "
        "Spalten `spread_proxy` (bps) und optional `impact_proxy` (bps). "
        "Fehlende Spalten werden robust auf 0 gesetzt. "
        "Fallback für Impact ist eine bps-Skala der Rolling-Vol der Returns (12 Bars)."
    )
    write_outputs(args.freq, trades_df, equity_df, args, note)
    _log("DONE Portfolio")


if __name__ == "__main__":
    main()
