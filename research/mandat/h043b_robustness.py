"""H-043b — adversariale Robustheit des Intraday-Crisis-Signals (VOR jeder PASS-Aussage).

Prüft, ob der marginale 60min-Excess (t~2,7) ein deployabler Edge ist oder ein
Artefakt weniger Extremtage / eines Regimes / zu billig gerechneter Kosten.

Checks:
  1) Drop-Top-K: entferne die K grössten |60m-Excess|-Spike-Tage -> t überlebt?
  2) Jahr-für-Jahr: ist es nur 2022 (Ukraine)?
  3) Realistische 4-Bein-Kosten (long 3 ETFs + short SPY): net bei 6/10/14 bps all-in.
  4) DSR der Spike-only-Strategie (Tagesserie: 60m-Excess an Spike-Tagen, sonst 0).
  5) Schwellen-Sensitivität z>0.5 / 1.0 / 1.5.
KEIN neuer Trial im Ledger-Sinn (Robustheit von H-043, Lauf 113/114-Fenster).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from h011_kandidat_a import OUTD  # noqa: E402
from h043_crisis_intraday import CRISIS, load_z  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"


def excess_60m() -> pd.Series:
    bars = pd.read_parquet(DATA / "intraday_crisis_5m.parquet")
    bars["date"] = bars["datetime"].dt.normalize()
    rec = {}
    for (sym, d), g in bars.groupby(["symbol", "date"], sort=False):
        g = g.sort_values("datetime")
        if len(g) < 13:
            continue
        entry = float(g.iloc[0]["open"])
        if entry > 0:
            rec[(sym, d)] = float(g.iloc[12]["close"]) / entry - 1.0  # 60 min
    s = pd.Series(rec)
    s.index = pd.MultiIndex.from_tuples(s.index, names=["symbol", "date"])
    w = s.unstack("symbol")
    return (w[CRISIS].mean(axis=1) - w["SPY"]).dropna()


def tstat(x: pd.Series) -> float:
    return float(x.mean() / (x.std() / np.sqrt(len(x)))) if len(x) > 2 else float("nan")


def main() -> int:
    from src.assembled_core.qa.deflated_sharpe import deflated_sharpe

    ex = excess_60m()
    z = load_z().reindex(ex.index)
    out: dict = {}

    for thr in (0.5, 1.0, 1.5):
        spk = ex[z > thr].dropna()
        out[f"z>{thr}"] = {
            "n": int(len(spk)),
            "mean_bps": round(float(spk.mean()) * 1e4, 2),
            "t": round(tstat(spk), 2),
        }
    spk = ex[z > 1.0].dropna()

    # 1) drop-top-K by |excess|
    order = spk.reindex(spk.abs().sort_values(ascending=False).index)
    drops = {}
    for k in (1, 3, 5, 10):
        kept = order.iloc[k:]
        drops[f"drop_top{k}"] = {
            "n": int(len(kept)),
            "mean_bps": round(float(kept.mean()) * 1e4, 2),
            "t": round(tstat(kept), 2),
        }
    out["drop_top_abs"] = drops

    # 2) year-by-year
    yb = {}
    for yr, g in spk.groupby(spk.index.year):
        yb[int(yr)] = {
            "n": int(len(g)),
            "mean_bps": round(float(g.mean()) * 1e4, 2),
            "t": round(tstat(g), 2),
        }
    out["by_year"] = yb

    # 3) realistic 4-leg costs (long 3 ETFs + short SPY); gross 60m mean
    gross_bps = float(spk.mean()) * 1e4
    out["gross_bps"] = round(gross_bps, 2)
    out["net_after_cost"] = {
        f"{c}bps_allin": round(gross_bps - c, 2) for c in (6, 10, 14)
    }
    # after German <1y tax on the residual (only if positive)
    out["net_after_cost_and_tax"] = {
        f"{c}bps_allin": round(max(gross_bps - c, 0) * (1 - 0.26375), 2)
        for c in (6, 10, 14)
    }

    # 4) DSR of spike-only daily strategy (10bps all-in cost applied per spike day)
    strat = pd.Series(0.0, index=sorted(set(ex.index)))
    strat.loc[spk.index] = spk.values - 10e-4  # net of 10 bps
    r = strat
    ann_sharpe = (
        float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else float("nan")
    )
    dsr = deflated_sharpe(r, n_trials=118)
    out["spike_strategy_net10bps"] = {
        "ann_sharpe": round(ann_sharpe, 3),
        "DSR_prob": round(float(dsr.deflated_sharpe_probability), 3),
        "DSR_passes_5pct": bool(dsr.passes_5pct),
        "trade_days_per_year": round(
            len(spk) / ((ex.index.max() - ex.index.min()).days / 365.25), 1
        ),
    }

    # verdict
    surv_drop = drops["drop_top5"]["t"] > 2.0
    net_pos = out["net_after_cost_and_tax"]["10bps_allin"] > 0
    out["ROBUST_PASS"] = bool(surv_drop and net_pos and dsr.passes_5pct)

    (OUTD / "h043b_robustness.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8"
    )
    print("[ROBUST]", json.dumps(out, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
