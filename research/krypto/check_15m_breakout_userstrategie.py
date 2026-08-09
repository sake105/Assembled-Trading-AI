"""Check der User-Strategie "FifteenMinuteBreakout_TrailingRSI" (Freqtrade, Krypto).

AUFTRAG (Hans, 2026-08-09): "check die mal durch und lass ein paar test laufen".
Freqtrade ist hier nicht installiert — die Logik wird nachgebaut und auf echten
NATIVEN 15m-Daten der oeffentlichen Binance-Spot-API simuliert (QUELLEN-WECHSEL
2026-08-09, preisblind: EODHD-Token tot, 401 auch fuer Aktien; urspruenglich
registriert war EODHD 5m->15m — Regeln unveraendert). SANITY-PRUEFUNG, keine
Freqtrade-Replikation (kein Orderbuch, aggregierter Preis-Feed).

VORAB REGISTRIERT (vor jedem Datenkontakt fixiert, +2 Trials):
  Variante A "wie beabsichtigt": Struktur-Stop auf prev_low (ratchet nur nach
    oben), Trailing (aktiv ab +2 % Peak, 1,5 % Abstand), TP +3 %.
  Variante B "wie der Code real laeuft": KEIN Struktur-Stop (der Custom-Stop
    faellt nachweislich in den except-Pfad -> return 1 = kein Stop),
    Trailing + TP aktiv (best case fuer den Code; ob Freqtrade Engine-Trailing
    bei use_custom_stoploss=True ueberhaupt anwendet, ist versionsabhaengig —
    B ist damit die WOHLWOLLENDSTE Lesart).
  Paare: BTC/ETH/SOL/XRP/ADA (USDT-Spot, Binance), nativ 15m, ~590 Tage.
  Ausfuehrung: Signal auf 15m-Kerze t -> Entry zum OPEN von t+1 (Freqtrade-
    Konvention). Eine Position je Paar. Intrabar-Prioritaet PESSIMISTISCH:
    Stop vor Trailing vor TP.
  Kosten: 0,10 % Taker-Fee + 0,05 % Slippage JE SEITE (0,30 % Roundtrip).
  Kontrolle: identische Maschinerie auf 200 Zufalls-Entries je Paar (Seed 49).
  Verdikt-Kriterium (vorab): interessant nur, wenn Netto-Erwartungswert je
    Trade > 0 mit t > 2 UND Profit-Faktor > 1 nach Kosten, gepoolt ueber
    alle Paare, in der jeweiligen Variante. Sonst FAIL.

Signal (PIT-sauber, alles shift(1)-basiert wie im Original):
  close[t] > high[t-1]  UND  RSI14[t] < 62  UND  close[t-1] <= high[t-2]
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
ZIEL = HIER / "check_15m_breakout.json"
PAARE = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT"]
TAGE = 590
FEE = 0.0010
SLIP = 0.0005
SEED = 49
KONTROLLEN = 200

TP = 0.03
TRAIL_AKTIV = 0.02
TRAIL_ABSTAND = 0.015
RSI_MAX = 62


def lade_15m(paar: str) -> pd.DataFrame:
    """Binance klines, natives 15m, 1000 Bars je Request, keine Auth."""
    DATEN.mkdir(parents=True, exist_ok=True)
    cache = DATEN / f"binance_{paar}_15m.parquet"
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
                "interval": "15m",
                "startTime": s,
                "endTime": ende_ms,
                "limit": 1000,
            }
        )
        url = f"https://api.binance.com/api/v3/klines?{q}"
        with urllib.request.urlopen(url, timeout=60) as h:  # noqa: S310
            teil = json.loads(h.read().decode("utf-8"))
        if not isinstance(teil, list):
            raise SystemExit(f"[ERROR] Binance-Antwort unerwartet: {str(teil)[:200]}")
        if not teil:
            break
        zeilen.extend(teil)
        s = teil[-1][0] + 1  # naechster Request ab letzter Open-Time + 1ms
        time.sleep(0.15)
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


def rsi14(close: pd.Series) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def signale(df: pd.DataFrame) -> pd.Series:
    prev_high = df["high"].shift(1)
    r = rsi14(df["close"])
    cond = (
        (df["close"] > prev_high)
        & (r < RSI_MAX)
        & (df["close"].shift(1) <= prev_high.shift(1))
    )
    return cond.fillna(False)


def simuliere_trade(
    df: pd.DataFrame, i_entry: int, mit_stop: bool
) -> tuple[float, int] | None:
    """Netto-Rendite eines Trades ab Entry-Kerze i_entry (Entry = deren open).

    Pessimistische Intrabar-Prioritaet: Stop -> Trailing -> TP. Exits fuellen
    zum jeweiligen Level (bzw. open, wenn die Kerze unter dem Level oeffnet).
    """
    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    lo = df["low"].to_numpy()
    entry = o[i_entry] * (1 + SLIP)
    stop = lo[i_entry - 1] if mit_stop and i_entry >= 1 else -np.inf
    peak = entry
    for j in range(i_entry, min(i_entry + 4 * 96, len(df))):  # max 4 Tage halten
        if mit_stop and j > i_entry:
            stop = max(stop, lo[j - 1])  # prev_low-Ratchet (nur aufwaerts)
        trail = (
            peak * (1 - TRAIL_ABSTAND) if peak >= entry * (1 + TRAIL_AKTIV) else -np.inf
        )
        eff_stop = max(stop, trail)
        # Kerze j: erst Stop (pessimistisch), dann TP
        if lo[j] <= eff_stop:
            fill = min(o[j], eff_stop)
            return fill * (1 - SLIP) / entry - 1 - 2 * FEE, j
        tp_level = entry / (1 + SLIP) * (1 + TP)
        if h[j] >= tp_level:
            fill = max(o[j], tp_level)
            return fill * (1 - SLIP) / entry - 1 - 2 * FEE, j
        peak = max(peak, h[j])
    j = min(i_entry + 4 * 96, len(df)) - 1
    return df["close"].to_numpy()[j] * (1 - SLIP) / entry - 1 - 2 * FEE, j


def lauf(df: pd.DataFrame, entries: list[int], mit_stop: bool) -> list[float]:
    aus, frei_ab = [], -1
    for i in entries:
        if i <= frei_ab or i + 1 >= len(df):
            continue
        r = simuliere_trade(df, i + 1, mit_stop)  # Entry NAECHSTE Kerze
        if r is not None:
            aus.append(r[0])
            frei_ab = r[1]
    return aus


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
        "--regen", action="store_true", help="Neulauf ohne Trial-Increment (E-090)."
    )
    args = ap.parse_args(argv)
    # BUCHUNGS-OFFENLEGUNG: Der Erstlauf buchte +2 und scheiterte am toten
    # EODHD-Token VOR jedem Datenkontakt. Der Re-Run auf Binance laeuft mit
    # --regen — gleiche registrierte Studie, nur andere Datenquelle.
    if args.regen:
        print(
            f"[REGEN] Trial-Zaehler UNVERAENDERT bei {TrialCounter().total()}",
            flush=True,
        )
    else:
        print(
            "Trials kumuliert: "
            + str(
                TrialCounter().increment(2, label="Krypto 15m-Breakout User-Strategie")
            ),
            flush=True,
        )
    rng = np.random.default_rng(SEED)
    ergebnis: dict = {
        "registriert": __doc__,  # VOLLSTAENDIG, kein split (E-140)
        "quelle": "Binance Spot REST /api/v3/klines, nativ 15m",
        "abruf_hinweis": (
            "Parquet-Cache research/krypto/data (gitignored); Zeitraum je Paar unten"
        ),
        "vergleichs_hinweis": (
            "A und B sind KEINE vergleichbaren Stichproben (A stoppt binnen "
            "Bars aus -> ~20x mehr Trades); jede Variante NUR gegen die "
            "eigene Kontrolle lesen"
        ),
        "paare": {},
    }
    pool: dict[str, list[float]] = {
        "A": [],
        "B": [],
        "kontrolle_A": [],
        "kontrolle_B": [],
    }
    for paar in PAARE:
        df = lade_15m(paar)
        if df.empty:
            ergebnis["paare"][paar] = {"fehler": "keine Daten"}
            print(f"[WARN] {paar}: keine Daten", flush=True)
            continue
        sig = signale(df)
        entries = list(np.flatnonzero(sig.to_numpy()))
        zufall = sorted(
            rng.choice(
                np.arange(100, len(df) - 400),
                size=min(KONTROLLEN, len(df) // 10),
                replace=False,
            )
        )
        r_a = lauf(df, entries, True)
        r_b = lauf(df, entries, False)
        k_a = lauf(df, list(zufall), True)
        k_b = lauf(df, list(zufall), False)
        p = {
            "bars_15m": len(df),
            "zeitraum": [str(df.index[0]), str(df.index[-1])],
            "n_signale": len(entries),
            "A_mit_stop": statistik(r_a),
            "B_ohne_stop": statistik(r_b),
            "kontrolle_A": statistik(k_a),
            "kontrolle_B": statistik(k_b),
        }
        pool["A"] += r_a
        pool["B"] += r_b
        pool["kontrolle_A"] += k_a
        pool["kontrolle_B"] += k_b
        ergebnis["paare"][paar] = p
        print(f"{paar}: {len(df)} bars, {len(entries)} Signale", flush=True)
    ergebnis["gepoolt"] = {k: statistik(v) for k, v in pool.items()}

    def besteht(s: dict) -> bool:
        return bool(
            s.get("n_trades", 0) > 2
            and (s.get("mittel_pp") or 0) > 0
            and (s.get("t") or 0) > 2
            and (s.get("profit_faktor") or 0) > 1
        )

    ergebnis["verdikt"] = {
        "A_besteht": besteht(ergebnis["gepoolt"]["A"]),
        "B_besteht": besteht(ergebnis["gepoolt"]["B"]),
        "kriterium": "mittel>0 & t>2 & PF>1 nach Kosten, gepoolt (vorab fixiert)",
    }
    ZIEL.write_text(
        json.dumps(ergebnis, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for k in ("A", "B", "kontrolle_A", "kontrolle_B"):
        print(k, "->", ergebnis["gepoolt"][k])
    print("VERDIKT:", ergebnis["verdikt"])
    print(f"-> {ZIEL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
