"""wikifolio-Erhebung — Hypothesen-Generator, NIE Evidenz (Mandat II, P6).

RECHTLICHER UND METHODISCHER RAHMEN
------------------------------------
* **robots.txt geprueft (2026-08-01):** wikifolio erlaubt ``User-agent: *``
  alles ausser ``/search`` und Tracking-Parameter-URLs. Die 34.647
  wikifolio-Seiten stehen im offiziellen Sitemap und sind zur Indexierung
  vorgesehen. Diese Erhebung liest ausschliesslich solche Seiten.
* **eToro dagegen wird NICHT erhoben.** Dessen robots.txt sperrt fuer alle
  Agents ausdruecklich ``/portfolio``, ``/portfolio/*``, ``*/api/*`` und
  ``*/sapi*`` — also genau die Trader-Daten. Das wird respektiert.
* **Keine personenbezogenen Daten.** Die Seiten liefern Klarnamen der Trader
  mit; die werden nicht gespeichert. Erhoben werden nur Strategie-Merkmale und
  das anonyme wikifolio-Symbol.
* **Rate-Limit** 1 Anfrage pro 1,5 s, sequenziell, mit Identifikation im
  User-Agent.

WARUM DAS KEINE EVIDENZ IST
---------------------------
Der Datensatz ist massiv survivorship-verzerrt: eingestellte und
gescheiterte wikifolios verschwinden aus dem Index. Wer hier „Strategien mit
hoher Rendite" zaehlt, zaehlt Ueberlebende. Die Erhebung kann deshalb genau
eine Frage beantworten: **welche Stil-Merkmale kommen unter den sichtbaren
wikifolios ueberhaupt vor** — als Ideenquelle fuer eigene, sauber getestete
Hypothesen. Jede Zahl daraus geht als HYPOTHESE in die Registry, nie als Beleg.
"""

from __future__ import annotations

import gzip
import json
import re
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

OUT = Path(__file__).resolve().parent / "results"
SITEMAP = "https://www.wikifolio.com/de-de-w-sitemap-001.xml"
UA = "Mozilla/5.0 (compatible; assembled-trading-research/1.0; nicht-kommerziell)"
PAUSE_S = 1.5


@dataclass
class WikifolioZeile:
    """Nur Strategie-Merkmale. Kein Name, kein Trader-Bezug."""

    symbol: str
    status: int | None
    daily_fee: float | None
    performance_fee: float | None
    hebelprodukte: bool | None
    waehrung: str | None
    beta_1j_spx: float | None
    beta_3j_spx: float | None
    beta_5j_spx: float | None
    beta_1j_dax: float | None
    #: Performance-Kennzahlen aus topKpis. Die Werte liegen unter
    #: item["ranking"]["value"], nicht auf oberster Ebene — der erste
    #: Parser-Anlauf las die falschen Schluessel und lieferte leere Dicts.
    perf_seit_start: float | None
    perf_pa: float | None
    perf_1j: float | None
    kurzbeschreibung_laenge: int | None
    #: Stil-Stichworte aus der oeffentlichen Kurzbeschreibung. Nur Zaehlwerte,
    #: kein Freitext — der koennte personenbezogene Angaben enthalten.
    stil_flags: dict


STIL_MUSTER = {
    "trend": r"\btrend|momentum\b",
    "value": r"\bvalue|unterbewert|substanz\b",
    "dividende": r"\bdividend|ausschütt\b",
    "wachstum": r"\bwachstum|growth\b",
    "technisch": r"\bchart|technische analyse|indikator\b",
    "hebel": r"\bhebel|leverage|knock.?out\b",
    "kurzfrist": r"\bdaytrad|swing|kurzfrist\b",
    "langfrist": r"\blangfrist|buy.and.hold\b",
    "quant": r"\bquantitativ|algorithm|systematisch\b",
    "krypto": r"\bkrypto|bitcoin|crypto\b",
}


def _hole(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    raw = urllib.request.urlopen(req, timeout=30).read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return raw.decode("utf-8", "replace")


def sitemap_urls(limit: int | None = None) -> list[str]:
    t = _hole(SITEMAP)
    urls = re.findall(r"<loc>(.*?)</loc>", t)
    return urls[:limit] if limit else urls


def _topkpi(kf: dict, praefix: str) -> float | None:
    """Wert aus topKpis, gematcht ueber den Label-Praefix."""
    for item in kf.get("topKpis") or []:
        r = (item or {}).get("ranking") or {}
        label = (r.get("label") or "").lower()
        if label.startswith(praefix.lower()):
            return r.get("value")
    return None


def _beta(kf: dict, jahre: str, index: str) -> float | None:
    for gruppe in (kf.get("beta") or {}).get("groups", []):
        if gruppe.get("label") != jahre:
            continue
        for r in gruppe.get("rankings", []):
            if (r.get("ranking") or {}).get("label") == index:
                return r["ranking"].get("value")
    return None


def parse(html: str) -> WikifolioZeile | None:
    m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.S)
    if not m:
        return None
    d = json.loads(m.group(1))
    data = ((d.get("props") or {}).get("pageProps") or {}).get("data") or {}
    w = data.get("wikifolio") or {}
    kf = data.get("keyFigures") or {}
    kurz = (w.get("shortDescription") or "").lower()
    return WikifolioZeile(
        symbol=w.get("symbol") or "",
        status=w.get("status"),
        daily_fee=w.get("dailyFee"),
        performance_fee=w.get("performanceFee"),
        hebelprodukte=w.get("containsLeverageProducts"),
        waehrung=w.get("currency"),
        beta_1j_spx=_beta(kf, "1 Jahr", "Beta S&P 500"),
        beta_3j_spx=_beta(kf, "3 Jahre", "Beta S&P 500"),
        beta_5j_spx=_beta(kf, "5 Jahre", "Beta S&P 500"),
        beta_1j_dax=_beta(kf, "1 Jahr", "Beta DAX"),
        perf_seit_start=_topkpi(kf, "seit "),
        perf_pa=_topkpi(kf, "Ø-Perf"),
        perf_1j=_topkpi(kf, "Performance (1 J"),
        kurzbeschreibung_laenge=len(kurz),
        stil_flags={k: bool(re.search(p, kurz)) for k, p in STIL_MUSTER.items()},
    )


def erheben(n: int = 300, pause: float = PAUSE_S) -> list[WikifolioZeile]:
    urls = sitemap_urls()
    print(f"Sitemap: {len(urls)} wikifolios, erhebe {n} (Pause {pause}s)", flush=True)
    # Gleichmaessig ueber den Index streuen statt die ersten n zu nehmen —
    # die Sitemap-Reihenfolge ist nicht zufaellig.
    schritt = max(1, len(urls) // n)
    ausgewaehlt = urls[::schritt][:n]
    zeilen: list[WikifolioZeile] = []
    fehler = 0
    for i, u in enumerate(ausgewaehlt, 1):
        try:
            z = parse(_hole(u))
            if z and z.symbol:
                zeilen.append(z)
        except Exception as e:  # Netzwerk/Parse — zaehlen, nicht verschlucken
            fehler += 1
            if fehler <= 3:
                print(f"  [FEHLER] {u}: {type(e).__name__}", flush=True)
        if i % 25 == 0:
            print(
                f"  {i}/{len(ausgewaehlt)} ({len(zeilen)} ok, {fehler} Fehler)",
                flush=True,
            )
        time.sleep(pause)
    print(f"Fertig: {len(zeilen)} erhoben, {fehler} Fehler", flush=True)
    return zeilen


def main() -> int:
    OUT.mkdir(exist_ok=True)
    zeilen = erheben(n=300)
    (OUT / "wikifolio_stichprobe.json").write_text(
        json.dumps([asdict(z) for z in zeilen], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    # Auswertung: welche Stile kommen vor, und wie sieht ihr Beta aus?
    n = len(zeilen)
    print(f"\n=== Stil-Haeufigkeit unter {n} sichtbaren wikifolios ===")
    for stil in STIL_MUSTER:
        treffer = [z for z in zeilen if z.stil_flags.get(stil)]
        betas = [z.beta_1j_spx for z in treffer if z.beta_1j_spx is not None]
        m = sum(betas) / len(betas) if betas else float("nan")
        print(
            f"  {stil:<12}: {len(treffer):>4} ({len(treffer) / max(n, 1):>5.1%})  Beta-1J-SPX Ø {m:>6.2f}"
        )
    mit_hebel = [z for z in zeilen if z.hebelprodukte]
    print(
        f"\n  mit Hebelprodukten: {len(mit_hebel)} ({len(mit_hebel) / max(n, 1):.1%})"
    )
    gebuehren = [z.performance_fee for z in zeilen if z.performance_fee is not None]
    if gebuehren:
        print(f"  Performance-Fee Ø {sum(gebuehren) / len(gebuehren):.1%}")
    print(f"\n-> {OUT / 'wikifolio_stichprobe.json'}")
    print("\nSTATUS DIESER ZAHLEN: Hypothesen-Material. Survivorship-verzerrt.")
    print("Kein Verdikt, kein Beleg, keine Registry-Eintragung als Ergebnis.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
