"""H-051 — §23-Tax-Free-Asset-Sleeve vs 100 % Aktien-ETF (nach dt. Steuer).

Buy-and-Hold-to-terminal (§23-optimal: nie < 1 J verkaufen -> Gold/Krypto-Gewinne steuerfrei).
Aktien-ETF: 18,46 % Teilfreistellung am Ende. §23-Sleeve (Gold GLD, Krypto BTC): 0 % (> 1 J).
Krypto-Ergebnis Hindsight-verzerrt -> Haircut-Sensitivität (BTC-Renditen × 0,5).
Guardrail 4: nur Spot. KEIN Rebalance-Turnover (isoliert den reinen Tax-Wedge + Diversifikation).
"""

from __future__ import annotations

import json
import sys
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from h011_kandidat_a import OUTD  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"
START = 100_000.0
TAX = {"SPY": 0.1846, "GLD": 0.0, "GLD_hc": 0.0, "BTC": 0.0, "BTC_hc": 0.0}


def load() -> pd.DataFrame:
    pc = pd.read_parquet(DATA / "prices_crisis.parquet")
    px = pc.pivot(index="timestamp", columns="symbol", values="close").sort_index()
    df = px[["SPY", "GLD"]].copy()
    df.index = pd.DatetimeIndex(df.index)
    btc = pd.read_parquet(DATA / "crypto_BTCUSDCC.parquet")["close"]
    btc.index = pd.DatetimeIndex(btc.index)
    df["BTC"] = btc.reindex(df.index).ffill()
    # crypto return-haircut path (×0.5 of daily returns) — Hindsight-Stresstest
    r = df["BTC"].pct_change().fillna(0.0) * 0.5
    df["BTC_hc"] = (1 + r).cumprod()
    df.loc[df["BTC"].isna(), "BTC_hc"] = np.nan
    # gold return-haircut (×0.3 ≈ Rückkehr zur mageren Langfrist-Norm) — Robustheits-Stress
    rg = df["GLD"].pct_change().fillna(0.0) * 0.3
    df["GLD_hc"] = (1 + rg).cumprod()
    df.loc[df["GLD"].isna(), "GLD_hc"] = np.nan
    return df


def blend(df: pd.DataFrame, w: dict, start: str) -> dict:
    assets = [a for a in w if w[a] > 0]
    sub = df[df.index >= pd.Timestamp(start, tz="UTC")].dropna(subset=assets)
    norm = sub[assets] / sub[assets].iloc[0]
    port = sum(w[a] * norm[a] for a in assets)  # buy-hold value path, start=1
    ret = port.pct_change().dropna()
    years = (sub.index[-1] - sub.index[0]).days / 365.25
    net = START
    for a in assets:
        gain = w[a] * START * (float(norm[a].iloc[-1]) - 1.0)
        net += gain * (1 - TAX[a])
    return {
        "net_final": round(net),
        "cagr_net_pct": round(((net / START) ** (1 / years) - 1) * 100, 2),
        "sharpe": round(float(ret.mean() / ret.std() * sqrt(252)), 3),
        "maxdd": round(float((port / port.cummax() - 1).min()), 3),
        "years": round(years, 1),
    }


def main() -> int:
    df = load()
    out = {}

    # --- Window A: Equity + Gold (2005+, ehrlich, kein Krypto-Hindsight) ---
    out["A_equity_gold_2005"] = {
        "100% SPY": blend(df, {"SPY": 1.0}, "2005-01-01"),
        "90/10 SPY/Gold": blend(df, {"SPY": 0.9, "GLD": 0.1}, "2005-01-01"),
        "80/20 SPY/Gold": blend(df, {"SPY": 0.8, "GLD": 0.2}, "2005-01-01"),
        "70/30 SPY/Gold": blend(df, {"SPY": 0.7, "GLD": 0.3}, "2005-01-01"),
    }

    # --- Window A-Stress: Gold-Renditen ×0.3 (Norm-Rückkehr) ---
    out["A_gold_haircut_x0.3_2005"] = {
        "100% SPY": blend(df, {"SPY": 1.0}, "2005-01-01"),
        "80/20 SPY/Gold_hc": blend(df, {"SPY": 0.8, "GLD_hc": 0.2}, "2005-01-01"),
        "70/30 SPY/Gold_hc": blend(df, {"SPY": 0.7, "GLD_hc": 0.3}, "2005-01-01"),
    }

    # --- Window B: + Krypto (2016+, Hindsight-FLAG) ---
    s = "2016-01-01"
    out["B_with_crypto_2016"] = {
        "100% SPY": blend(df, {"SPY": 1.0}, s),
        "90/10 SPY/Gold": blend(df, {"SPY": 0.9, "GLD": 0.1}, s),
        "95/5 SPY/BTC": blend(df, {"SPY": 0.95, "BTC": 0.05}, s),
        "90/10 SPY/BTC": blend(df, {"SPY": 0.9, "BTC": 0.1}, s),
        "80/10/10 SPY/Gold/BTC": blend(df, {"SPY": 0.8, "GLD": 0.1, "BTC": 0.1}, s),
    }

    # --- Window B haircut: BTC-Renditen ×0,5 (Hindsight-Stresstest) ---
    out["B_crypto_haircut_x0.5"] = {
        "95/5 SPY/BTC_hc": blend(df, {"SPY": 0.95, "BTC_hc": 0.05}, s),
        "90/10 SPY/BTC_hc": blend(df, {"SPY": 0.9, "BTC_hc": 0.1}, s),
        "80/10/10 SPY/Gold/BTC_hc": blend(
            df, {"SPY": 0.8, "GLD": 0.1, "BTC_hc": 0.1}, s
        ),
    }

    (OUTD / "h051_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("[RESULT]", json.dumps(out, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
