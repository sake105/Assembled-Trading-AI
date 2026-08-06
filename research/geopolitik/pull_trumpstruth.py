"""Schritt 1 — Vollarchiv der Truth-Social-Posts ziehen (Welle 48, kein Trial).

WARUM ALLES UND NICHT EINE SUCHE
--------------------------------
Schritt 0 hat gezeigt, dass die Erinnerung die Treffer selektiert: die
Anker-Beispiele existierten, aber ihre Groessen gehoerten ueberwiegend anderen
Ursachen. Eine Stichwort-Query beim Pull wuerde dieselbe Selektion nur in die
Query verschieben. Deshalb: ID-Walk ueber ALLE Posts, die Ereignis-Regel
(Welle 48, vorab fixiert) wird erst DANACH angewandt.

QUELLE
------
`trumpstruth.org` — oeffentliches Archiv, robots.txt erlaubt alles, versteht
sich selbst als "public archive". Fortlaufende Status-IDs, Volltext,
minutengenaue Zeitstempel. Hoeflicher Takt (1,2 s), deklarierter User-Agent.

ZEITZONE — OFFENE VERIFIKATION
------------------------------
Die Archiv-Zeitstempel tragen keine Zone (vermutlich ET). VOR jeder Analyse
gegen ein extern dokumentiertes Ereignis pruefen (Waffenruhe-Post 23.06.2025
~18 Uhr ET). Bis dahin werden die Stempel roh gespeichert, NICHT konvertiert —
eine falsche Konvertierung waere schlimmer als keine.

BETRIEB
-------
Resumefaehig ueber vorhandene Chunk-Dateien. Laufzeit fuer ~40.600 IDs bei
1,2 s: rund 14 Stunden. 404/410 (geloeschte Posts) sind normale Zustaende und
werden als Luecken protokolliert, nicht als Fehler.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

HIER = Path(__file__).resolve().parent
OUT = HIER / "data" / "trumpstruth"
BASIS = "https://trumpstruth.org/statuses/{sid}"
UA = "Assembled-Trading-AI Forschung (hans.oertel2@gmail.com)"

#: Hoeflicher Abstand. Die Seite ist ein Ein-Personen-Archiv, kein CDN.
ABSTAND_S = 1.2

#: Posts je Chunk-Datei. Klein genug fuer engmaschiges Resume, gross genug,
#: um nicht tausende Dateien zu erzeugen.
CHUNK = 500

RE_ZEIT = re.compile(r"([A-Z][a-z]{2,8}\s+\d{1,2},\s+\d{4},\s+\d{1,2}:\d{2}\s+[AP]M)")
RE_INHALT = re.compile(r'class="status__content[^"]*"[^>]*>(.*?)</div>', re.S)
RE_TAG = re.compile(r"<[^>]+>")


def hole(sid: int) -> dict | None:
    """Einen Post laden. None = existiert nicht (geloescht/uebersprungen)."""
    req = urllib.request.Request(BASIS.format(sid=sid), headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=45) as h:  # noqa: S310
            seite = h.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        if e.code in (404, 410):
            return None
        raise
    zeit = RE_ZEIT.search(seite)
    inhalt = RE_INHALT.search(seite)
    text = ""
    if inhalt:
        text = html.unescape(RE_TAG.sub(" ", inhalt.group(1)))
        text = re.sub(r"\s+", " ", text).strip()
    return {
        "status_id": sid,
        # Roh, ohne Zonenkonvertierung — siehe Docstring.
        "zeit_roh": zeit.group(1) if zeit else None,
        "text": text,
        "hat_medien": "status-attachment__image" in seite,
    }


def erledigte_ids() -> set[int]:
    done: set[int] = set()
    for p in OUT.glob("chunk_*.parquet"):
        done.update(
            int(x) for x in pd.read_parquet(p, columns=["status_id"])["status_id"]
        )
    for p in OUT.glob("chunk_*.luecken.json"):
        done.update(json.loads(p.read_text(encoding="utf-8")))
    return done


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--von", type=int, default=1)
    ap.add_argument("--bis", type=int, default=40650)
    ap.add_argument("--max-minuten", type=float, default=0, help="0 = bis fertig")
    args = ap.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    done = erledigte_ids()
    todo = [i for i in range(args.von, args.bis + 1) if i not in done]
    print(
        f"[START] {args.bis - args.von + 1} IDs im Fenster, {len(done)} erledigt, "
        f"{len(todo)} offen",
        flush=True,
    )

    puffer: list[dict] = []
    luecken: list[int] = []
    t0 = time.monotonic()
    geholt = 0
    fehler_folge = 0

    def flush() -> None:
        nonlocal puffer, luecken
        if not puffer and not luecken:
            return
        n = len(list(OUT.glob("chunk_*.parquet"))) + 1
        name = f"chunk_{n:04d}"
        if puffer:
            pd.DataFrame(puffer).to_parquet(OUT / f"{name}.parquet", index=False)
        if luecken:
            (OUT / f"{name}.luecken.json").write_text(
                json.dumps(luecken), encoding="utf-8"
            )
        print(
            f"[CHUNK] {name}: {len(puffer)} Posts, {len(luecken)} Luecken | "
            f"{geholt} geholt in {(time.monotonic() - t0) / 60:.0f} min",
            flush=True,
        )
        puffer, luecken = [], []

    for sid in todo:
        if args.max_minuten and (time.monotonic() - t0) / 60 > args.max_minuten:
            print("[STOP] Zeitlimit — Resume beim naechsten Aufruf.", flush=True)
            break
        try:
            zeile = hole(sid)
            fehler_folge = 0
        except Exception as e:
            fehler_folge += 1
            print(f"[WARN] id={sid}: {type(e).__name__}: {e}", flush=True)
            if fehler_folge >= 8:
                print(
                    "[ERROR] 8 Fehler in Folge — Abbruch, Resume moeglich.", flush=True
                )
                flush()
                return 1
            time.sleep(20)
            continue
        if zeile is None:
            luecken.append(sid)
        else:
            puffer.append(zeile)
        geholt += 1
        if len(puffer) + len(luecken) >= CHUNK:
            flush()
        time.sleep(ABSTAND_S)

    flush()
    print(f"[FERTIG] {geholt} IDs in diesem Lauf.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
