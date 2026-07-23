"""Small-Cap-Universum-Loader (survivorship-frei) fuer H-035/H-036.

Baut aus data/smallcap/part_*.parquet ein breites (Tag x Symbol) close- und
ADV60-Panel + gemergte SPY-Benchmark. Liefert eine Band-Membership-Funktion,
mit der run_verdict (verdict_engine) unveraendert auf einem Small-/Large-Band
laeuft (Membership = Titel im Band je Monatsende).

Daten-Ehrlichkeit (Mandat §2.5):
- Groessen-/Liquiditaetsachse = ADV60 (rollierender 60T-Mittelwert von
  close*volume), PIT um 1 Tag geshiftet. KEINE Shares-Outstanding fuer das
  volle Delisted-Universum verfuegbar -> Dollar-Volumen ist die handelbare
  Groessenachse (ehrlich eine liquiditaets-proxied size; die deployable
  Version der Size-Praemie ist ohnehin ~ Illiquiditaetspraemie).
- Handelbarkeits-Floor je Rebalance (PIT): close >= $5 UND ADV60 >= $1M/Tag.
  Schliesst untradeable Penny/Micro-Namen aus, wo Momentum nur auf dem Papier
  "funktioniert".
- Rechnerischer Vorfilter: Panel auf Namen reduziert, die JE (>=40 Tage)
  close>=$5 & Dollar-Vol>=$0.5M erfuellten. Das entfernt nur Namen, die den
  Rebalance-Floor NIE clearen koennten (nie selektierbar) -> Ergebnis identisch
  zum Vollpanel, nur Speicher gespart. Der PIT-Floor wird je Monatsende weiter
  hart angewandt.
- Hygiene: gleiche Impossible-Jump-Truncation wie verdict_engine
  (|ret|>100% & Vortagskurs<$1 -> ab da NaN; Engine erzwingt Delisting-Verkauf).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent / "data"
SC_DIR = DATA / "smallcap"
CACHE_CLOSE = DATA / "_sc_close.parquet"
CACHE_ADV = DATA / "_sc_adv.parquet"

FLOOR_PRICE = 5.0
FLOOR_ADV = 1_000_000.0
PRE_PRICE = 5.0
PRE_DV = 500_000.0
PRE_DAYS = 40


def _build_panels() -> tuple[pd.DataFrame, pd.DataFrame]:
    parts = sorted(SC_DIR.glob("part_*.parquet"))
    if not parts:
        raise FileNotFoundError("no smallcap parts pulled yet")
    frames = []
    for p in parts:
        df = pd.read_parquet(p)
        df["close"] = df["close"].astype("float32")
        df["volume"] = df["volume"].astype("float32")
        frames.append(df)
    long = pd.concat(frames, ignore_index=True)
    long = long.drop_duplicates(subset=["timestamp", "symbol"], keep="last")
    long["dv"] = long["close"].astype("float64") * long["volume"].astype("float64")

    # computational pre-filter (superset gate; documented above)
    ok = (long["close"] >= PRE_PRICE) & (long["dv"] >= PRE_DV)
    days_ok = ok.groupby(long["symbol"]).sum()
    keep = set(days_ok[days_ok >= PRE_DAYS].index)
    long = long[long["symbol"].isin(keep)]
    print(
        f"[SC] pre-filter: {len(keep)} of {days_ok.size} symbols ever tradable",
        flush=True,
    )

    close = long.pivot(index="timestamp", columns="symbol", values="close").sort_index()
    dv = long.pivot(index="timestamp", columns="symbol", values="dv").sort_index()

    # hygiene truncation (impossible micro-price jumps)
    r = close.pct_change(fill_method=None)
    bad = (r.abs() > 1.0) & (close.shift(1) < 1.0)
    n_trunc = 0
    for sym in close.columns[bad.any()]:
        fb = bad.index[bad[sym]][0]
        close.loc[fb:, sym] = np.nan
        dv.loc[fb:, sym] = np.nan
        n_trunc += 1
    print(f"[SC] hygiene: truncated {n_trunc} corrupt series", flush=True)

    adv = dv.rolling(60, min_periods=30).mean().shift(1)  # PIT: yesterday-known
    close = close.astype("float32")
    adv = adv.astype("float32")
    return close, adv


def load_smallcap(*, use_cache: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    if use_cache and CACHE_CLOSE.exists() and CACHE_ADV.exists():
        close = pd.read_parquet(CACHE_CLOSE)
        adv = pd.read_parquet(CACHE_ADV)
    else:
        close, adv = _build_panels()
        close.to_parquet(CACHE_CLOSE)
        adv.to_parquet(CACHE_ADV)

    # merge SPY benchmark (adjusted) from verdict panel
    vp = pd.read_parquet(DATA / "prices_verdict.parquet")
    spy = vp[vp["symbol"] == "SPY"].set_index("timestamp")["close"].sort_index()
    spy.index = pd.DatetimeIndex(spy.index)
    close = close.copy()
    close["SPY"] = spy.reindex(close.index).astype("float32")
    print(
        f"[SC] panel {close.shape[0]}d x {close.shape[1] - 1} names + SPY, "
        f"{close.index[0].date()} -> {close.index[-1].date()}",
        flush=True,
    )
    return close, adv


def band_membership(
    close: pd.DataFrame,
    adv: pd.DataFrame,
    *,
    side: str,  # "small" | "large"
    small_max_pct: float = 0.70,
    large_min_pct: float = 0.80,
) -> pd.Series:
    """Monatsende -> frozenset(Titel im Band). PIT-Floor pro Rebalance."""
    idx = close.index
    month_ends = idx.to_series().groupby(idx.to_period("M")).max()
    out: dict[pd.Timestamp, frozenset] = {}
    for me in month_ends:
        px = close.loc[me]
        av = adv.loc[me] if me in adv.index else pd.Series(dtype="float32")
        qual = av[(px.reindex(av.index) >= FLOOR_PRICE) & (av >= FLOOR_ADV)].dropna()
        qual = qual[qual.index != "SPY"]
        if len(qual) < 20:
            continue
        pct = qual.rank(pct=True)  # 0..1, higher = larger ADV
        if side == "small":
            sel = pct[pct <= small_max_pct].index
        else:
            sel = pct[pct >= large_min_pct].index
        out[me] = frozenset(sel)
    return pd.Series(out)
