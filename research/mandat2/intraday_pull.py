"""EODHD-Intraday-Ingest fuer Mandat II — Korrektur einer falschen Annahme.

WAS ICH FALSCH GESAGT HABE
--------------------------
Im ABSCHLUSS stand: „Intraday nicht getestet — braeuchte das EODHD-Intraday-
Paket (ab ca. Okt 2020)". Beides war falsch:

* Das Paket IST freigeschaltet und wird bereits genutzt (es liegen
  ``research/mandat/data/intraday_crisis_5m.parquet`` mit 452k 5-Minuten-Bars
  fuer 4 ETFs 2020-2026 und ``research/fable_exploration/.../earnings_minute
  .parquet`` mit 246k 1-Minuten-Bars fuer 20 Titel 2024-2026 auf der Platte).
* „Ab Okt 2020" gilt nur fuer den **1h**-Endpunkt. Der **1m**-Endpunkt reicht
  bei Einzelaktien bis **2004** zurueck — empirisch geprueft fuer AAPL, MSFT,
  GE, XOM, KO.

Damit ist der Intraday-Strang NICHT datenblockiert. 2004-2026 sind 22 Jahre,
also genug fuer rollierende 10-Jahres-Fenster.

GEPRUEFTE RANDBEDINGUNGEN (2026-08-03, gegen die Live-API)
----------------------------------------------------------
* Max. **120 Tage** pro Call; 200 Tage -> HTTP 422.
* **SPY erst ab ca. 2014** — der ETF taugt auf dieser Aufloesung nicht als
  Benchmark ueber die volle Strecke. Loesung: der Benchmark bleibt der
  TAEGLICHE SPY (liegt ab 1995 vor); Intraday wird nur fuer die Strategie
  gebraucht. Die Zielfunktion vergleicht Endvermoegen ueber 10-Jahres-Fenster
  und braucht dafuer keine Intraday-Aufloesung auf der Benchmark-Seite.
* 1m-Rohdaten sind zu gross zum Aufheben (~2,2 Mio Bars je Symbol ueber 22 J).
  Deshalb wird **beim Ingest auf Stundenbars verdichtet** — ~39k Zeilen je
  Symbol. Wer spaeter feiner braucht, zieht gezielt nach.

Der Puller ist wiederaufnehmbar: fertige Symbole werden uebersprungen.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "raw" / "intraday_1h"
FENSTER_TAGE = 110  # unter dem 120-Tage-Limit, mit Sicherheitsabstand
PAUSE_S = 0.35
START = dt.datetime(2004, 1, 1)
ENDE = dt.datetime(2026, 7, 6)


def token() -> str:
    for line in (
        (ROOT / ".env").read_text(encoding="utf-8", errors="replace").splitlines()
    ):
        if line.startswith("EODHD_API_TOKEN="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("EODHD_API_TOKEN nicht in .env gefunden")


def hole_fenster(sym: str, von: dt.datetime, bis: dt.datetime, tok: str) -> list[dict]:
    u = (
        f"https://eodhd.com/api/intraday/{sym}.US?interval=1m"
        f"&from={int(von.timestamp())}&to={int(bis.timestamp())}"
        f"&api_token={tok}&fmt=json"
    )
    for versuch in range(3):
        try:
            return json.load(urllib.request.urlopen(u, timeout=90))
        except urllib.error.HTTPError as e:
            if e.code == 422:  # Fenster zu gross / Symbol unbekannt
                return []
            if versuch == 2:
                raise
            time.sleep(2 * (versuch + 1))
        except Exception:
            if versuch == 2:
                raise
            time.sleep(2 * (versuch + 1))
    return []


def auf_stunden(bars: list[dict]) -> pd.DataFrame:
    """1m -> 1h. Verdichtung beim Ingest, weil die Rohmenge nicht lagerbar ist."""
    if not bars:
        return pd.DataFrame()
    df = pd.DataFrame(bars)
    if "datetime" not in df.columns:
        return pd.DataFrame()
    df["ts"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).set_index("ts").sort_index()
    for c in ("open", "high", "low", "close", "volume"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    agg = {
        k: v
        for k, v in (
            ("open", "first"),
            ("high", "max"),
            ("low", "min"),
            ("close", "last"),
            ("volume", "sum"),
        )
        if k in df.columns
    }
    return df.resample("1h").agg(agg).dropna(subset=["close"])


def pull_symbol(sym: str, tok: str) -> tuple[int, str]:
    ziel = OUT / f"{sym}.parquet"
    if ziel.exists():
        return -1, "uebersprungen"
    teile: list[pd.DataFrame] = []
    von = START
    while von < ENDE:
        bis = min(von + dt.timedelta(days=FENSTER_TAGE), ENDE)
        bars = hole_fenster(sym, von, bis, tok)
        if bars:
            h = auf_stunden(bars)
            if not h.empty:
                teile.append(h)
        von = bis
        time.sleep(PAUSE_S)
    if not teile:
        return 0, "keine Daten"
    df = pd.concat(teile).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df["symbol"] = sym
    ziel.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(ziel)
    return len(df), f"{df.index.min().date()}..{df.index.max().date()}"


def main() -> int:
    symbole = [s.strip().upper() for s in sys.argv[1:] if s.strip()]
    if not symbole:
        print("Aufruf: intraday_pull.py SYM [SYM ...]")
        return 2
    tok = token()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Ziel: {OUT}  |  {len(symbole)} Symbole  |  {START.date()}..{ENDE.date()}")
    for i, sym in enumerate(symbole, 1):
        t0 = time.time()
        try:
            n, info = pull_symbol(sym, tok)
        except Exception as e:
            print(
                f"  [{i}/{len(symbole)}] {sym:<6} FEHLER {type(e).__name__}: {str(e)[:60]}",
                flush=True,
            )
            continue
        if n == -1:
            print(f"  [{i}/{len(symbole)}] {sym:<6} {info}", flush=True)
        else:
            print(
                f"  [{i}/{len(symbole)}] {sym:<6} {n:>7,} Stundenbars  {info}"
                f"  ({time.time() - t0:.0f}s)",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
