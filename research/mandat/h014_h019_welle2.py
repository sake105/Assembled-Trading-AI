"""Welle 2 — H-014 bis H-019 (Registry: research/registry.md, registriert VOR Lauf).

EXPLORATIV (Datenlage wie H-011/012). 13 Läufe:
  H-014 TLH-Overlay auf H-012-Familie (3) | H-015 ATR 3x/4x/5x auf out60 (3)
  H-016 GEM SPY/IEF (2, survivorship-immun)  | H-017 Low-Vol (1)
  H-018 52wk-High (1) | H-019 Gate-Familie sma/vol/both (3)
"""

from __future__ import annotations

import datetime as dt
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from h011_kandidat_a import (  # noqa: E402
    DATA,
    OUTD,
    START_CAPITAL,
    TaxedPortfolio,
    cscv_pbo,
    load_prices,
    run_variant,
)

ETF_TAX = 0.185
ETF_PATH_FINAL = 373260.78  # H-011 benchmark window
H012_BEST_FINAL = 614905.0
EW_SHARPE = 0.851


def ensure_ief() -> pd.Series:
    extra = DATA / "prices_extra.parquet"
    if extra.exists():
        df = pd.read_parquet(extra)
    else:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
        from alpaca.data import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        client = StockHistoricalDataClient(
            api_key=os.environ["ALPACA_API_KEY"],
            secret_key=os.environ["ALPACA_API_SECRET"],
        )
        req = StockBarsRequest(
            symbol_or_symbols=["IEF"],
            timeframe=TimeFrame.Day,
            start=dt.datetime(2016, 1, 1),
            end=dt.datetime(2026, 7, 4),
            adjustment="all",
        )
        df = client.get_stock_bars(req).df.reset_index()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.normalize()
        df.to_parquet(extra, index=False)
    s = df[df["symbol"] == "IEF"].set_index("timestamp")["close"].sort_index()
    return s


def run_gem(spy: pd.Series, ief: pd.Series, *, absolute_only: bool, label: str):
    """Monthly GEM: hold SPY if 12M SPY beats (IEF 12M and 0) [classic], or if
    12M > 0 [absolute-only]; else hold IEF. FIFO tax on every switch."""
    both = pd.concat([spy.rename("SPY"), ief.rename("IEF")], axis=1).dropna()
    r12 = both / both.shift(252) - 1.0
    month_ends = set(
        pd.Series(both.index, index=both.index).groupby(both.index.to_period("M")).max()
    )
    px = {k: both[k] for k in ("SPY", "IEF")}
    pf = TaxedPortfolio(START_CAPITAL)
    cur: str | None = None
    pending: str | None = None
    equity = []
    for t in both.index:
        if pending is not None and pending != cur:
            if cur is not None:
                q = pf.qty(cur)
                if q > 0:
                    pf.sell(cur, q, float(px[cur].at[t]))
            pf.buy(pending, pf.cash, float(px[pending].at[t]))
            cur = pending
        pending = None
        v = pf.cash + sum(
            pf.qty(k) * float(px[k].at[t]) for k in ("SPY", "IEF") if pf.qty(k) > 0
        )
        equity.append((t, v))
        if t in month_ends and np.isfinite(r12["SPY"].at[t]):
            if absolute_only:
                target = "SPY" if r12["SPY"].at[t] > 0 else "IEF"
            else:
                target = (
                    "SPY"
                    if (r12["SPY"].at[t] > r12["IEF"].at[t] and r12["SPY"].at[t] > 0)
                    else "IEF"
                )
            if target != cur:
                pending = target
    eq = pd.Series(dict(equity)).sort_index()
    eq = eq[r12["SPY"].notna()]
    r = eq.pct_change().dropna()
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    return (
        {
            "label": label,
            "final_value": float(eq.iloc[-1] / eq.iloc[0] * START_CAPITAL),
            "cagr_net": float((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1),
            "sharpe_net": float(r.mean() / r.std() * np.sqrt(252)),
            "maxdd_net": float((eq / eq.cummax() - 1).min()),
            "tax_paid": float(pf.tax_paid),
            "years": float(years),
        },
        r,
    )


def main() -> int:
    from src.assembled_core.qa.deflated_sharpe import deflated_sharpe

    close, high, low = load_prices()
    close = close.dropna(axis=1, thresh=int(len(close) * 0.5))
    high, low = high[close.columns], low[close.columns]
    print(f"[DATA] {close.shape[1] - 1} symbols", flush=True)

    results: dict = {}
    fam_rets: dict[str, dict] = {}

    def stock_run(name, fam, **kw):
        res, _eq, ret = run_variant(close, high, low, None, label=name, **kw)
        results[name] = res
        fam_rets.setdefault(fam, {})[name] = ret
        print(
            f"[RUN] {name}: final={res['final_value']:.0f} sharpe={res['sharpe_net']:.3f} tax={res['tax_paid']:.0f}",
            flush=True,
        )

    base = dict(
        use_quality=False, use_gate=False, use_atr_backstop=False, no_retrim=True
    )
    # H-014 TLH on H-012 family
    for out in (60, 80, 100):
        stock_run(f"H014_tlh_out{out}", "H014", top_out=out, tlh_pct=0.15, **base)
    # H-015 ATR family on out60
    for m in (3.0, 4.0, 5.0):
        stock_run(
            f"H015_atr{int(m)}x",
            "H015",
            top_out=60,
            atr_mult=m,
            use_quality=False,
            use_gate=False,
            use_atr_backstop=True,
            no_retrim=True,
        )
    # H-017 low-vol | H-018 52wk-high
    stock_run("H017_lowvol", "H017", top_out=60, signal="lowvol", **base)
    stock_run("H018_high52", "H018", top_out=60, signal="high52", **base)
    # H-019 gate family on out60
    for gm in ("sma", "vol", "both"):
        stock_run(
            f"H019_gate_{gm}",
            "H019",
            top_out=60,
            gate_mode=gm,
            use_quality=False,
            use_gate=True,
            use_atr_backstop=False,
            no_retrim=True,
        )
    # H-016 GEM (index level)
    spy = close["SPY"].dropna()
    ief = ensure_ief()
    for absolute, name in ((False, "H016_gem_classic"), (True, "H016_abs_only")):
        res, ret = run_gem(spy, ief, absolute_only=absolute, label=name)
        results[name] = res
        fam_rets.setdefault("H016", {})[name] = ret
        print(
            f"[RUN] {name}: final={res['final_value']:.0f} sharpe={res['sharpe_net']:.3f} tax={res['tax_paid']:.0f}",
            flush=True,
        )

    # family metrics (N per registry: H014->53, H015->56, H016->58, H017->59, H018->60, H019->63)
    fam_n = {"H014": 53, "H015": 56, "H016": 58, "H017": 59, "H018": 60, "H019": 63}
    for fam, rets in fam_rets.items():
        best = max(rets, key=lambda k: results[k]["final_value"])
        dsr = deflated_sharpe(rets[best], n_trials=fam_n[fam])
        entry = {
            "selected": best,
            "DSR_prob": float(dsr.deflated_sharpe_probability),
            "DSR_passes": bool(dsr.passes_5pct),
            "n_trials": fam_n[fam],
        }
        if len(rets) >= 2:
            entry["PBO_CSCV"] = float(cscv_pbo(pd.DataFrame(rets)))
        else:
            entry["PBO_CSCV"] = "n/a (Einzellauf)"
        results[f"_family_{fam}"] = entry

    out_path = OUTD / "welle2_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"[DONE] -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
