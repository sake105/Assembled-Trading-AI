"""H-043 — Crisis-Alpha „erste Minuten" INTRADAY (Lückenschluss zu H-039).

Test A (Event-Study, entscheidend): an Tagen NACH geopolitischem Intensitäts-Spike
(z>1, PIT wie H-039) — Crisis-Basket EW(XLE/GLD/ITA) minus SPY Intraday-Rendite von
Session-Open bis +5/15/30/60 min. Vergleich Spike- vs Nicht-Spike-Tage.
Test B: ist der Move netto capturable (Intraday-Kosten + 26,375 % <1J-Steuer)?

Fenster ~2020-2026 (Intraday-Tiefe) -> EXPLORATIV, 1 Regime. PIT: z nutzt Intensität
bis Vortag -> Reaktion am heutigen Open ist look-ahead-frei.
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

DATA = Path(__file__).resolve().parent / "data"
CRISIS = ["XLE", "GLD", "ITA"]
HORIZONS = {"5m": 1, "15m": 3, "30m": 6, "60m": 12}  # 5-min bars from open bar
ROUND_TRIP_BPS = 6.0  # optimistic intraday round-trip cost (liquid ETFs)


def load_z() -> pd.Series:
    inten = pd.read_parquet(DATA / "geopol_intensity.parquet").set_index("date")[
        "n_articles"
    ]
    inten.index = pd.DatetimeIndex(inten.index)
    z = (inten - inten.rolling(252, min_periods=120).mean()) / inten.rolling(
        252, min_periods=120
    ).std()
    return z.shift(1)  # PIT: yesterday-known at today's open


def intraday_open_to_h(bars: pd.DataFrame) -> pd.DataFrame:
    """Per (symbol, date): open-bar entry -> close at +k bars. Returns date x sym x horizon ret."""
    bars = bars.copy()
    bars["date"] = bars["datetime"].dt.normalize()
    recs = {}
    for (sym, d), g in bars.groupby(["symbol", "date"], sort=False):
        g = g.sort_values("datetime")
        if len(g) < max(HORIZONS.values()) + 1:
            continue
        entry = float(g.iloc[0]["open"])
        if not np.isfinite(entry) or entry <= 0:
            continue
        for hname, k in HORIZONS.items():
            exit_px = float(g.iloc[k]["close"])
            recs[(sym, d, hname)] = exit_px / entry - 1.0
    s = pd.Series(recs)
    s.index = pd.MultiIndex.from_tuples(s.index, names=["symbol", "date", "h"])
    return s.unstack("symbol")


def main() -> int:
    bars = pd.read_parquet(DATA / "intraday_crisis_5m.parquet")
    z = load_z()
    rr = intraday_open_to_h(bars)  # index (date,h) x columns symbol
    if not set(CRISIS + ["SPY"]).issubset(rr.columns):
        print(f"[ERR] missing symbols, have {list(rr.columns)}", flush=True)
        return 1
    rr["crisis"] = rr[CRISIS].mean(axis=1)
    rr["excess"] = rr["crisis"] - rr["SPY"]

    dates = rr.index.get_level_values("date").unique()
    zd = z.reindex(dates)
    spike_dates = set(zd[zd > 1.0].dropna().index)
    n_spk = len(spike_dates)
    print(
        f"[DATA] {len(dates)} trading days intraday, {n_spk} spike-days (z>1), "
        f"{bars['datetime'].min().date()} -> {bars['datetime'].max().date()}",
        flush=True,
    )

    out = {"n_days": int(len(dates)), "n_spike": int(n_spk), "horizons": {}}
    for hname in HORIZONS:
        sub = rr.xs(hname, level="h")["excess"].dropna()
        spk = sub[[d in spike_dates for d in sub.index]]
        base = sub[[d not in spike_dates for d in sub.index]]
        m_spk = float(spk.mean())
        t_spk = (
            float(m_spk / (spk.std() / np.sqrt(len(spk))))
            if len(spk) > 2
            else float("nan")
        )
        out["horizons"][hname] = {
            "spike_mean_excess_pct": round(m_spk * 100, 4),
            "spike_t": round(t_spk, 2),
            "spike_n": int(len(spk)),
            "baseline_mean_excess_pct": round(float(base.mean()) * 100, 4),
            "spike_minus_base_pct": round((m_spk - float(base.mean())) * 100, 4),
        }

    # Test A: >0 with t>2 in >=2 horizons (spike days)
    a_pass = (
        sum(
            1
            for h in out["horizons"].values()
            if h["spike_mean_excess_pct"] > 0 and h["spike_t"] > 2
        )
        >= 2
    )
    # Test B: best-horizon gross excess vs round-trip cost (before even tax)
    best = max(out["horizons"].values(), key=lambda h: h["spike_mean_excess_pct"])
    gross_bps = best["spike_mean_excess_pct"] * 100
    net_capturable = gross_bps > ROUND_TRIP_BPS  # tax would only worsen this
    out["test_A_pass"] = bool(a_pass)
    out["test_B_net_capturable_pregtax"] = bool(net_capturable)
    out["best_horizon_gross_bps"] = round(gross_bps, 2)
    out["round_trip_cost_bps"] = ROUND_TRIP_BPS
    out["PASS"] = bool(a_pass and net_capturable)

    (OUTD / "h043_results.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8"
    )
    print("[VERDICT]", json.dumps(out, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
