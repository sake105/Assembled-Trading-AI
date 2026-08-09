"""Feed-Probe fuer Strategie N1 — prueft Kandidaten-Feeds inkl. robots.txt.

Umsetzung von Schritt 1 der Quellen-Recherche (Hans, 2026-08-09): alle
Kandidaten einmal real abrufen, robots.txt-Erlaubnis pruefen, Ergebnis als
CSV. Falsifikationsanker aus der Recherche: < 25 ok-Feeds -> Quellenbasis
zu duenn, Strategie-Scope reduzieren statt Quellen erfinden.

Klassifikation je URL:
  ok             HTTP 200 + parsebares XML/HTML mit Items + robots erlaubt
  ok_robots_verboten  liefert Daten, aber robots.txt untersagt den Pfad ->
                      NICHT in den Sammler (ToS-Disziplin)
  bot_block      403/429/Cloudflare
  tot            404/410/DNS/Timeout/TLS
  leer           200, aber keine Items extrahierbar
"""

from __future__ import annotations

import csv
import time
import urllib.parse
import urllib.request
import urllib.robotparser
import xml.etree.ElementTree as ET
from pathlib import Path

HIER = Path(__file__).resolve().parent
ZIEL = HIER / "probe_results.csv"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Forschung-N1"

KANDIDATEN: list[tuple[str, str]] = [
    # --- bereits im Sammler (Re-Verifikation inkl. robots) ---
    ("bbc_world", "https://feeds.bbci.co.uk/news/world/rss.xml"),
    ("nyt_world", "https://rss.nytimes.com/services/xml/rss/nyt/World.xml"),
    ("aljazeera_all", "https://www.aljazeera.com/xml/rss/all.xml"),
    ("guardian_world", "https://www.theguardian.com/world/rss"),
    ("dw_world", "https://rss.dw.com/rdf/rss-en-world"),
    ("tagesschau", "https://www.tagesschau.de/index~rss2.xml"),
    ("anadolu", "https://www.aa.com.tr/en/rss/default?cat=guncel"),
    ("faz_finanzen", "https://www.faz.net/rss/aktuell/finanzen/"),
    (
        "cnbc_world",
        "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100727362",
    ),
    ("marketwatch", "https://feeds.content.dowjones.io/public/rss/mw_topstories"),
    ("wsj_markets_cdn", "https://feeds.content.dowjones.io/public/rss/RSSMarketsMain"),
    (
        "handelsblatt_top",
        "https://www.handelsblatt.com/contentexport/feed/schlagzeilen",
    ),
    ("ntv_wirtschaft", "https://www.n-tv.de/wirtschaft/rss"),
    ("wallstreet_online", "https://www.wallstreet-online.de/rss/nachrichten-alle.xml"),
    ("ezb_presse", "https://www.ecb.europa.eu/rss/press.html"),
    ("fed_presse", "https://www.federalreserve.gov/feeds/press_all.xml"),
    ("reddit_geopolitics", "https://www.reddit.com/r/geopolitics/new/.rss"),
    ("reddit_worldnews", "https://www.reddit.com/r/worldnews/new/.rss"),
    (
        "gnews_reuters",
        "https://news.google.com/rss/search?q=site:reuters.com%20geopolitics&hl=en",
    ),
    # --- Recherche-Dokument: verifiziert ok (Gegenprobe hier) ---
    ("investing_com", "https://www.investing.com/rss/news.rss"),
    ("yahoo_finance", "https://finance.yahoo.com/news/rssindex"),
    ("wsj_markets_dj", "https://feeds.a.dj.com/rss/RSSMarketsMain.xml"),
    ("handelsblatt_fin", "https://www.handelsblatt.com/contentexport/feed/finanzen"),
    # --- 3.1 offiziell/staatlich ---
    (
        "un_news_me",
        "https://news.un.org/feed/subscribe/en/news/region/middle-east/feed/rss.xml",
    ),
    ("unhcr", "https://www.unhcr.org/rss.xml"),
    ("mfa_israel", "https://mfa.gov.il/MFA/PressRoom/rss.xml"),
    # --- 3.2 Think Tanks ---
    ("crisisgroup", "https://www.crisisgroup.org/rss/91"),
    ("csis_me", "https://www.csis.org/regions/middle-east/rss.xml"),
    ("mei", "https://www.mei.edu/rss.xml"),
    ("washinginst", "https://www.washingtoninstitute.org/feed"),
    (
        "brookings_mena",
        "https://www.brookings.edu/topic/middle-east-north-africa/feed/",
    ),
    # --- 3.3 OSINT ---
    ("bellingcat", "https://www.bellingcat.com/feed/"),
    ("osintcurious", "https://osintcurio.us/feed/"),
    # --- 3.4 internationale Medien ---
    ("bbc_me", "https://feeds.bbci.co.uk/news/world/middle_east/rss.xml"),
    ("france24_me", "https://www.france24.com/en/middle-east/rss"),
    ("cnn_me", "http://rss.cnn.com/rss/edition_meast.rss"),
    ("ap_hub", "https://apnews.com/hub/israel-hamas-war/feed"),
    ("guardian_israel", "https://www.theguardian.com/world/israel/rss"),
    ("foreignpolicy", "https://foreignpolicy.com/feed/"),
    ("nyt_me", "https://rss.nytimes.com/services/xml/rss/nyt/MiddleEast.xml"),
    # --- 3.5 regionale Medien (Auswahl, Bias -> Tier-Feld spaeter) ---
    ("timesofisrael", "https://www.timesofisrael.com/feed/"),
    ("jpost", "https://rss.jpost.com/rss/rssfeedsfrontpage.aspx"),
    ("ynetnews", "https://www.ynetnews.com/RSS/5.xml"),
    ("middleeasteye", "https://www.middleeasteye.net/rss"),
    ("al_monitor", "https://www.al-monitor.com/rss"),
    # --- 3.9 geopolitische Breite ---
    ("isw", "https://www.understandingwar.org/feeds"),
    ("ecfr", "https://www.ecfr.eu/feed/"),
    ("chathamhouse", "https://www.chathamhouse.org/rss/all"),
    ("sipri", "https://www.sipri.org/rss.xml"),
    ("atlanticcouncil", "https://www.atlanticcouncil.org/feed/"),
    ("carnegie", "https://carnegieendowment.org/rss/"),
    # --- 4.1 Finanz DE ---
    ("finanzen_net", "https://www.finanzen.net/rss/news"),
    ("faz_wirtschaft", "https://www.faz.net/rss/aktuell/wirtschaft/"),
    ("zeit_wirtschaft", "https://newsfeed.zeit.de/wirtschaft/index"),
    ("manager_magazin", "https://www.manager-magazin.de/news/index.rss"),
    ("spiegel_wirtschaft", "https://www.spiegel.de/wirtschaft/index.rss"),
    ("tagesschau_wirtschaft", "https://www.tagesschau.de/wirtschaft/index~rss2.xml"),
    ("deraktionaer", "https://www.deraktionaer.de/rss/news.xml"),
    (
        "finanznachrichten",
        "https://www.finanznachrichten.de/rss-aktien-nachrichten/alle.htm",
    ),
    # --- 4.2 Finanz EN ---
    ("seekingalpha", "https://seekingalpha.com/market_currents.xml"),
    ("fortune", "https://fortune.com/feed"),
    ("ft", "https://www.ft.com/?format=rss"),
    ("wsj_usbusiness", "https://feeds.a.dj.com/rss/WSJcomUSBusiness.xml"),
    ("politico_playbook", "https://rss.politico.com/playbook.xml"),
    # --- 4.3 Zentralbanken/Regulatoren ---
    ("bundesbank", "https://www.bundesbank.de/service/rss"),
    ("bis_press", "https://www.bis.org/list/press_rlsscbs/rss.xml"),
]


_robots_cache: dict[str, urllib.robotparser.RobotFileParser | None] = {}


def robots_erlaubt(url: str) -> bool | None:
    """True/False laut robots.txt; None wenn robots.txt nicht lesbar."""
    host = urllib.parse.urlsplit(url)
    basis = f"{host.scheme}://{host.netloc}"
    if basis not in _robots_cache:
        rp = urllib.robotparser.RobotFileParser()
        try:
            req = urllib.request.Request(
                basis + "/robots.txt", headers={"User-Agent": UA}
            )
            with urllib.request.urlopen(req, timeout=10) as h:  # noqa: S310
                rp.parse(h.read().decode("utf-8", errors="replace").splitlines())
            _robots_cache[basis] = rp
        except Exception:
            _robots_cache[basis] = None
    rp = _robots_cache[basis]
    if rp is None:
        return None
    return rp.can_fetch("*", url)


def probe(url: str) -> tuple[str, int]:
    """(status, n_items)"""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=20) as h:  # noqa: S310
            roh = h.read()
    except urllib.error.HTTPError as e:
        return ("bot_block" if e.code in (401, 403, 429) else "tot"), 0
    except Exception:
        return "tot", 0
    try:
        root = ET.fromstring(roh)
        n = sum(1 for el in root.iter() if el.tag.split("}")[-1] in ("item", "entry"))
        return ("ok" if n else "leer"), n
    except ET.ParseError:
        return ("leer", 0)


def main() -> int:
    zeilen = []
    for name, url in KANDIDATEN:
        status, n = probe(url)
        erlaubt = robots_erlaubt(url) if status in ("ok", "leer") else None
        if status == "ok" and erlaubt is False:
            status = "ok_robots_verboten"
        zeilen.append(
            {
                "source_id": name,
                "url": url,
                "status": status,
                "n_items": n,
                "robots_erlaubt": {True: "ja", False: "NEIN", None: "?"}[erlaubt],
            }
        )
        print(
            f"{status:<20} items={n:<4} robots={zeilen[-1]['robots_erlaubt']:<4} {name}",
            flush=True,
        )
        time.sleep(0.7)
    with ZIEL.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(zeilen[0].keys()))
        w.writeheader()
        w.writerows(zeilen)
    ok = sum(1 for z in zeilen if z["status"] == "ok")
    print(
        f"\n{ok} von {len(zeilen)} Feeds ok (Falsifikationsanker: >= 25 noetig) -> {ZIEL}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
