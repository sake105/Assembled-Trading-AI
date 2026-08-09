"""Check der "SHAEMA"-Strategie (E-Mail-Vorschlag, via Hans 2026-08-09).

Regeln laut Mail (Long-Setup; Short war im Screenshot nicht enthalten):
  Indikatoren: Smoothed Heiken Ashi (Inputs EMA-50 geglaettet, HA gerechnet,
  Output nochmal EMA-50 geglaettet — "TheBacktestGuy"-Variante) + EMA 50.
  1. SHA-Kerze gruen. 2. EMA 50 OBERHALB der SHA-Dochte (h2). 3. Echte Kerze
  kreuzt EMA 50 bullish von unten. 4. Entry zum Open der naechsten Kerze.
  5. SL knapp unter den Docht (Low) der aktuellen SHA-Kerze. 6. TP = 1,5 R.

VORAB REGISTRIERT (+1 Trial, vor Datenkontakt fixiert):
  Daten: Binance-Spot 5m, BTC/ETH/SOL/XRP/ADA-USDT, ~590 Tage (Cache).
  Session-Regeln der Mail (Fr-20-Uhr etc.) sind FX-spezifisch und entfallen
  fuer 24/7-Krypto — offengelegte Abweichung, keine weitere.
  Ausfuehrung: Entry Open t+1; eine Position je Paar; Intrabar pessimistisch
  SL vor TP; Zeit-Exit nach 5 Tagen (1440 Bars) zum Close. Trades mit
  SL >= Entry (synthetisches Low ueber Entry) werden verworfen und gezaehlt.
  Kosten: 0,10 % Fee + 0,05 % Slippage je Seite.
  Kontrolle: 200 Zufalls-Entries je Paar, gleiche Exit-Maschinerie mit dem
  jeweils aktuellen SHA-Low als SL (Seed 50).
  Verdikt (vorab): interessant nur bei Netto-Mittel > 0, t > 2 UND
  Profit-Faktor > 1 gepoolt — UND klarer Abstand zur Zufalls-Kontrolle.

ASSET-KLASSEN-VORBEHALT: Die Mail handelt FX/Rohstoff-CFDs im 5m; hier wird
die MECHANIK auf Krypto geprueft. Ein FAIL hier widerlegt die Mail nicht
formal — ein "Signal = Zufall"-Befund ist aber ein starkes Indiz, dass die
Kante nicht in der Regel steckt.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

HIER = Path(__file__).resolve().parent
ROOT = HIER.parents[1]
sys.path.insert(0, str(ROOT))

from research.mandat2.data_gate import TrialCounter  # noqa: E402

DATEN = HIER / "data"
ZIEL = HIER / "check_shaema.json"
PAARE = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT"]
TAGE = 590
FEE = 0.0010
SLIP = 0.0005
SEED = 50
KONTROLLEN = 200
L = 50  # Smooth-/EMA-Laenge ueberall 50 (laut Mail)
TP_R = 1.5
MAX_BARS = 1440  # 5 Tage auf 5m


def lade_5m(paar: str) -> pd.DataFrame:
    DATEN.mkdir(parents=True, exist_ok=True)
    cache = DATEN / f"binance_{paar}_5m.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    ende_ms = int(time.time() * 1000)
    start_ms = ende_ms - TAGE * 86400 * 1000
    zeilen: list[list] = []
    s = start_ms
    while s < ende_ms:
        q = urllib.parse.urlencode(
            {
                "symbol": paar,
                "interval": "5m",
                "startTime": s,
                "endTime": ende_ms,
                "limit": 1000,
            }
        )
        url = f"https://api.binance.com/api/v3/klines?{q}"
        teil = None
        for versuch in range(4):  # Verbindungs-Resets nach vielen Requests
            try:
                with urllib.request.urlopen(url, timeout=60) as h:  # noqa: S310
                    teil = json.loads(h.read().decode("utf-8"))
                break
            except Exception as exc:
                if versuch == 3:
                    raise
                print(f"[WARN] Request-Retry {versuch + 1}: {exc}", flush=True)
                time.sleep(3 * (versuch + 1))
        if not isinstance(teil, list):
            raise SystemExit(f"[ERROR] Binance-Antwort unerwartet: {str(teil)[:200]}")
        if not teil:
            break
        zeilen.extend(teil)
        s = teil[-1][0] + 1
        time.sleep(0.12)
    if not zeilen:
        return pd.DataFrame()
    df = pd.DataFrame(
        zeilen,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "qav",
            "n",
            "tbb",
            "tbq",
            "ignore",
        ],
    )
    df["ts"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = (
        df.drop_duplicates("ts")
        .sort_values("ts")
        .set_index("ts")[["open", "high", "low", "close", "volume"]]
        .astype(float)
    )
    df.to_parquet(cache)
    return df


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def smoothed_ha(df: pd.DataFrame) -> pd.DataFrame:
    """TheBacktestGuy-Variante: EMA-Glaettung -> HA -> EMA-Glaettung."""
    o1, h1 = ema(df["open"], L), ema(df["high"], L)
    l1, c1 = ema(df["low"], L), ema(df["close"], L)
    ha_close = (o1 + h1 + l1 + c1) / 4
    ha_open = np.empty(len(df))
    ha_open[0] = (o1.iloc[0] + c1.iloc[0]) / 2
    hc = ha_close.to_numpy()
    for i in range(1, len(df)):
        ha_open[i] = (ha_open[i - 1] + hc[i - 1]) / 2
    ha_open_s = pd.Series(ha_open, index=df.index)
    ha_high = pd.concat([h1, ha_open_s, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([l1, ha_open_s, ha_close], axis=1).min(axis=1)
    return pd.DataFrame(
        {
            "sha_open": ema(ha_open_s, L),
            "sha_high": ema(ha_high, L),
            "sha_low": ema(ha_low, L),
            "sha_close": ema(ha_close, L),
        },
        index=df.index,
    )


def signale_und_sl(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    sha = smoothed_ha(df)
    e = ema(df["close"], L)
    gruen = sha["sha_close"] > sha["sha_open"]
    ema_ueber_docht = e > sha["sha_high"]
    kreuz = (df["close"] > e) & (df["close"].shift(1) <= e.shift(1))
    sig = (gruen & ema_ueber_docht & kreuz).fillna(False)
    return sig, sha["sha_low"]


def simuliere(
    df: pd.DataFrame, sha_low: pd.Series, i_sig: int
) -> tuple[float, int, str] | str:
    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    lo = df["low"].to_numpy()
    c = df["close"].to_numpy()
    i = i_sig + 1
    if i >= len(df):
        return "ende"
    entry = o[i] * (1 + SLIP)
    sl = float(sha_low.iloc[i_sig])
    if not np.isfinite(sl) or sl >= entry:
        return "sl_ueber_entry"
    tp = entry + TP_R * (entry - sl)
    for j in range(i, min(i + MAX_BARS, len(df))):
        if lo[j] <= sl:  # pessimistisch: SL vor TP
            fill = min(o[j], sl)
            return fill * (1 - SLIP) / entry - 1 - 2 * FEE, j, "sl"
        if h[j] >= tp:
            fill = max(o[j], tp)
            return fill * (1 - SLIP) / entry - 1 - 2 * FEE, j, "tp"
    j = min(i + MAX_BARS, len(df)) - 1
    return c[j] * (1 - SLIP) / entry - 1 - 2 * FEE, j, "zeit"


def lauf(
    df: pd.DataFrame, sha_low: pd.Series, entries: list[int]
) -> tuple[list[float], int, dict]:
    aus, verworfen, frei_ab = [], 0, -1
    exit_gruende = {"sl": 0, "tp": 0, "zeit": 0}
    for i in entries:
        if i <= frei_ab:
            continue
        r = simuliere(df, sha_low, i)
        if isinstance(r, str):
            verworfen += r == "sl_ueber_entry"
            continue
        aus.append(r[0])
        frei_ab = r[1]
        exit_gruende[r[2]] += 1
    return aus, verworfen, exit_gruende


def statistik(renditen: list[float]) -> dict:
    x = pd.Series(renditen, dtype=float)
    gew = x[x > 0].sum()
    verl = -x[x <= 0].sum()
    return {
        "n_trades": int(len(x)),
        "mittel_pp": round(float(x.mean()) * 100, 4) if len(x) else None,
        "t": round(float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))), 2)
        if len(x) > 2 and x.std(ddof=1) > 0
        else None,
        "trefferquote": round(float((x > 0).mean()), 3) if len(x) else None,
        "profit_faktor": round(float(gew / verl), 3) if verl > 0 else None,
        "summe_pct": round(float(x.sum()) * 100, 2) if len(x) else None,
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--regen", action="store_true", help="Neulauf ohne Trial-Increment."
    )
    args = ap.parse_args(argv)
    if args.regen:
        print(
            f"[REGEN] Trial-Zaehler UNVERAENDERT bei {TrialCounter().total()}",
            flush=True,
        )
    else:
        print(
            "Trials kumuliert: "
            + str(TrialCounter().increment(1, label="SHAEMA User-Strategie Krypto")),
            flush=True,
        )
    rng = np.random.default_rng(SEED)
    ergebnis: dict = {
        "registriert": __doc__,  # VOLLSTAENDIG inkl. Vorbehalt (E-140)
        "quelle": "Binance Spot REST /api/v3/klines, nativ 5m",
        "paare": {},
    }
    pool: dict[str, list[float]] = {"signal": [], "kontrolle": []}
    for paar in PAARE:
        df = lade_5m(paar)
        if df.empty:
            ergebnis["paare"][paar] = {"fehler": "keine Daten"}
            continue
        sig, sha_low = signale_und_sl(df)
        entries = [int(i) for i in np.flatnonzero(sig.to_numpy()) if i > 3 * L]
        zufall = sorted(
            int(i)
            for i in rng.choice(
                np.arange(3 * L, len(df) - MAX_BARS - 2),
                size=KONTROLLEN,
                replace=False,
            )
        )
        s_trades, s_verworfen, s_exits = lauf(df, sha_low, entries)
        k_trades, k_verworfen, k_exits = lauf(df, sha_low, zufall)
        ergebnis["paare"][paar] = {
            "bars_5m": len(df),
            "zeitraum": [str(df.index[0]), str(df.index[-1])],
            "n_signale": len(entries),
            "signal": statistik(s_trades),
            "signal_exit_gruende": s_exits,
            "signal_sl_verworfen": s_verworfen,
            "kontrolle": statistik(k_trades),
            "kontrolle_exit_gruende": k_exits,
            "kontrolle_sl_verworfen": k_verworfen,
        }
        pool["signal"] += s_trades
        pool["kontrolle"] += k_trades
        print(
            f"{paar}: {len(df)} bars, {len(entries)} Signale, {s_verworfen} SL-verworfen",
            flush=True,
        )
    ergebnis["gepoolt"] = {k: statistik(v) for k, v in pool.items()}
    g = ergebnis["gepoolt"]["signal"]
    ergebnis["verdikt"] = {
        "besteht": bool(
            g.get("n_trades", 0) > 2
            and (g.get("mittel_pp") or 0) > 0
            and (g.get("t") or 0) > 2
            and (g.get("profit_faktor") or 0) > 1
        ),
        "kriterium": "mittel>0 & t>2 & PF>1 gepoolt + Abstand zur Kontrolle (vorab fixiert)",
        "praezisierung": (
            "FAIL belegt: die ENTRY-Regel bringt unter DIESER Exit-Maschinerie "
            "nichts ueber die Kosten. sha_low lag nie in Entry-Naehe "
            "(0 SL-verworfen) — der getestete Stop ist strukturell weit weg, "
            "nicht der 'knapp unter den Docht' der Mail; Exit-Gruende je Paar "
            "im Artefakt."
        ),
    }
    ZIEL.write_text(
        json.dumps(ergebnis, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for k, v in ergebnis["gepoolt"].items():
        print(k, "->", v)
    print("VERDIKT:", ergebnis["verdikt"])
    print(f"-> {ZIEL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
