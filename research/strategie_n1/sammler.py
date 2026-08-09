"""Strategie-N1 PIT-Sammler — archiviert Multiquellen-Headlines mit Abrufzeit.

Zweck: Forward-Shadow-Datenbasis fuer Strategie N1 (SPEZIFIKATION.md).
Jeder Eintrag traegt fetched_utc = der PIT-Verfuegbarkeitszeitpunkt. Es wird
NICHTS interpretiert, gescort oder gehandelt — nur roh archiviert. Idempotent
via (quelle, link)-Dedupe gegen die bestehende Tagesdatei.

Ablage: research/strategie_n1/archiv/YYYY-MM-DD.jsonl (gitignored, ein File
je UTC-Tag, append-only) + stand.json (letzter Lauf, je Quelle Status).

Betrieb: manuell oder Task-Scheduler (Operator-Entscheidung), sinnvoll
1-2x/Stunde. Ein Lauf macht ~20 Requests mit 1s Abstand.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import time
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

HIER = Path(__file__).resolve().parent
ARCHIV = HIER / "archiv"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Forschung-N1"

#: (name, url, typ) — typ: rss | telegram. Reihenfolge = Abrufreihenfolge.
QUELLEN: list[tuple[str, str, str]] = [
    # v2 (2026-08-09): Bestand per feed_probe.py verifiziert (Status ok UND
    # robots.txt nicht verbietend; "?" = robots unlesbar -> zugelassen, im
    # CSV dokumentiert). ENTFERNT wegen robots-Verbot: Google-News-Reuters-
    # Proxy, CNBC, Reddit-RSS (legaler Reddit-Weg = OAuth-App, Operator-
    # Entscheidung). Reuters-Inhalte kommen jetzt via investing_com
    # (<author>Reuters</author>; pubDate TZ-NAIV -> beim Parsen festnageln).
    # Geopolitik International
    ("nyt_world", "https://rss.nytimes.com/services/xml/rss/nyt/World.xml", "rss"),
    ("nyt_me", "https://rss.nytimes.com/services/xml/rss/nyt/MiddleEast.xml", "rss"),
    ("bbc_world", "https://feeds.bbci.co.uk/news/world/rss.xml", "rss"),
    ("bbc_me", "https://feeds.bbci.co.uk/news/world/middle_east/rss.xml", "rss"),
    ("aljazeera", "https://www.aljazeera.com/xml/rss/all.xml", "rss"),
    ("guardian_world", "https://www.theguardian.com/world/rss", "rss"),
    ("guardian_israel", "https://www.theguardian.com/world/israel/rss", "rss"),
    ("dw_world", "https://rss.dw.com/rdf/rss-en-world", "rss"),
    ("france24_me", "https://www.france24.com/en/middle-east/rss", "rss"),
    ("cnn_me", "http://rss.cnn.com/rss/edition_meast.rss", "rss"),
    ("tagesschau", "https://www.tagesschau.de/index~rss2.xml", "rss"),
    ("anadolu", "https://www.aa.com.tr/en/rss/default?cat=guncel", "rss"),
    (
        "un_news_me",
        "https://news.un.org/feed/subscribe/en/news/region/middle-east/feed/rss.xml",
        "rss",
    ),
    ("foreignpolicy", "https://foreignpolicy.com/feed/", "rss"),
    # Geopolitik Think-Tank/OSINT (langsam, hohe Qualitaet)
    ("crisisgroup", "https://www.crisisgroup.org/rss/91", "rss"),
    ("bellingcat", "https://www.bellingcat.com/feed/", "rss"),
    ("ecfr", "https://www.ecfr.eu/feed/", "rss"),
    ("atlanticcouncil", "https://www.atlanticcouncil.org/feed/", "rss"),
    # Regionale Medien (Bias je Quelle im Tier-Modell dokumentieren)
    ("jpost", "https://rss.jpost.com/rss/rssfeedsfrontpage.aspx", "rss"),
    ("middleeasteye", "https://www.middleeasteye.net/rss", "rss"),
    ("al_monitor", "https://www.al-monitor.com/rss", "rss"),
    # Finanz EN
    (
        "wsj_markets",
        "https://feeds.content.dowjones.io/public/rss/RSSMarketsMain",
        "rss",
    ),
    ("wsj_usbusiness", "https://feeds.a.dj.com/rss/WSJcomUSBusiness.xml", "rss"),
    (
        "marketwatch",
        "https://feeds.content.dowjones.io/public/rss/mw_topstories",
        "rss",
    ),
    ("investing_com", "https://www.investing.com/rss/news.rss", "rss"),
    ("yahoo_finance", "https://finance.yahoo.com/news/rssindex", "rss"),
    ("seekingalpha", "https://seekingalpha.com/market_currents.xml", "rss"),
    ("fortune", "https://fortune.com/feed", "rss"),
    ("ft", "https://www.ft.com/?format=rss", "rss"),
    ("politico_playbook", "https://rss.politico.com/playbook.xml", "rss"),
    # Finanz DE
    ("faz_finanzen", "https://www.faz.net/rss/aktuell/finanzen/", "rss"),
    ("faz_wirtschaft", "https://www.faz.net/rss/aktuell/wirtschaft/", "rss"),
    (
        "handelsblatt_top",
        "https://www.handelsblatt.com/contentexport/feed/schlagzeilen",
        "rss",
    ),
    (
        "handelsblatt_fin",
        "https://www.handelsblatt.com/contentexport/feed/finanzen",
        "rss",
    ),
    ("ntv_wirtschaft", "https://www.n-tv.de/wirtschaft/rss", "rss"),
    (
        "wallstreet_online",
        "https://www.wallstreet-online.de/rss/nachrichten-alle.xml",
        "rss",
    ),
    ("zeit_wirtschaft", "https://newsfeed.zeit.de/wirtschaft/index", "rss"),
    ("manager_magazin", "https://www.manager-magazin.de/news/index.rss", "rss"),
    ("spiegel_wirtschaft", "https://www.spiegel.de/wirtschaft/index.rss", "rss"),
    (
        "tagesschau_wirtschaft",
        "https://www.tagesschau.de/wirtschaft/index~rss2.xml",
        "rss",
    ),
    # Primaerquellen Zentralbanken
    ("ezb_presse", "https://www.ecb.europa.eu/rss/press.html", "rss"),
    ("fed_presse", "https://www.federalreserve.gov/feeds/press_all.xml", "rss"),
    # Social: Telegram-Mirror (Clash Report; Tier C/D, nie alleinige Evidenz)
    ("tg_clashreport", "https://t.me/s/ClashReport", "telegram"),
]


def _hole(url: str) -> bytes:
    req = urllib.request.Request(
        url, headers={"User-Agent": UA, "Accept-Encoding": "gzip"}
    )
    with urllib.request.urlopen(req, timeout=30) as h:  # noqa: S310
        roh = h.read()
        if h.headers.get("Content-Encoding") == "gzip":
            roh = gzip.decompress(roh)
        return roh


def _rss_eintraege(roh: bytes) -> list[dict]:
    """Minimal-Parser fuer RSS 2.0 / Atom / RDF — nur Titel, Link, Datum."""
    aus = []
    root = ET.fromstring(roh)
    ns_atom = "{http://www.w3.org/2005/Atom}"
    for item in root.iter():
        tag = item.tag.split("}")[-1]
        if tag not in ("item", "entry"):
            continue
        titel = link = pub = ""
        for kind in item:
            k = kind.tag.split("}")[-1]
            if k == "title":
                titel = (kind.text or "").strip()
            elif k == "link":
                link = (kind.get("href") or kind.text or "").strip()
            elif k in ("pubDate", "published", "updated", "date"):
                pub = pub or (kind.text or "").strip()
        if titel:
            aus.append({"titel": titel, "link": link, "quelle_datum": pub})
    _ = ns_atom
    return aus


_TG_MSG = re.compile(r'class="tgme_widget_message_text[^"]*"[^>]*>(.*?)</div>', re.S)
_TG_ZEIT = re.compile(r'datetime="([^"]+)"')
_TAGS = re.compile(r"<[^>]+>")


def _telegram_eintraege(roh: bytes) -> list[dict]:
    html = roh.decode("utf-8", errors="replace")
    texte = [_TAGS.sub(" ", t).strip() for t in _TG_MSG.findall(html)]
    zeiten = _TG_ZEIT.findall(html)
    aus = []
    for i, t in enumerate(texte):
        if not t:
            continue
        aus.append(
            {
                "titel": t[:500],
                "link": hashlib.sha1(t.encode()).hexdigest()[:16],
                "quelle_datum": zeiten[i] if i < len(zeiten) else "",
            }
        )
    return aus


def main() -> int:
    ARCHIV.mkdir(parents=True, exist_ok=True)
    jetzt = datetime.now(timezone.utc)
    tagesdatei = ARCHIV / f"{jetzt.date().isoformat()}.jsonl"
    bekannt: set[tuple[str, str]] = set()
    if tagesdatei.exists():
        for zeile in tagesdatei.read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(zeile)
                bekannt.add((d["quelle"], d["link"]))
            except Exception:
                continue

    status: dict[str, str] = {}
    neu_gesamt = 0
    with tagesdatei.open("a", encoding="utf-8") as fh:
        for name, url, typ in QUELLEN:
            try:
                roh = _hole(url)
                eintraege = (
                    _telegram_eintraege(roh)
                    if typ == "telegram"
                    else _rss_eintraege(roh)
                )
                neu = 0
                fetched = datetime.now(timezone.utc).isoformat()
                for e in eintraege:
                    key = (name, e["link"])
                    if key in bekannt:
                        continue
                    bekannt.add(key)
                    fh.write(
                        json.dumps(
                            {"fetched_utc": fetched, "quelle": name, **e},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    neu += 1
                neu_gesamt += neu
                status[name] = f"ok:{len(eintraege)}/neu:{neu}"
                print(f"[OK] {name}: {len(eintraege)} Eintraege, {neu} neu", flush=True)
            except Exception as exc:
                # Einzelquelle darf ausfallen (429/DNS/Zertifikat) — LAUT
                # protokolliert, naechster Lauf versucht es wieder. Der
                # Sammler als Ganzes faellt nur, wenn ALLES ausfaellt.
                status[name] = f"FEHLER:{type(exc).__name__}"
                print(
                    f"[WARN] {name}: {type(exc).__name__} {str(exc)[:80]}", flush=True
                )
            time.sleep(1.0)

    ok = sum(1 for v in status.values() if v.startswith("ok"))
    (HIER / "stand.json").write_text(
        json.dumps(
            {"letzter_lauf_utc": jetzt.isoformat(), "quellen": status},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(
        f"[{'OK' if ok else 'ERROR'}] {ok}/{len(QUELLEN)} Quellen, {neu_gesamt} neue Eintraege -> {tagesdatei}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
