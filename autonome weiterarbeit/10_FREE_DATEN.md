# 10 — Free Datenquellen (0 EUR/Monat)

**Zweck:** Alle Datenquellen, die du ohne einen Cent Ausgaben nutzen kannst. Dieses Dokument ist dein Ingest-Fundament.

**Regel:** Alles in diesem Dokument ist mit **kommerzieller Nutzung kompatibel oder klar als non-commercial gekennzeichnet**. Wenn du später Richtung B (B2B-SaaS) gehst, sind die markierten Quellen eine Lizenz-Entscheidung.

---

## Module in diesem Dokument

| # | Modul | Zweck | Kommerziell? |
|---|---|---|---|
| 10.1 | SEC EDGAR via `edgartools` | Form 4, 8-K, 10-K, 13F | ✅ Public Domain |
| 10.2 | FRED via `fredapi` | Macro-Backbone | ✅ |
| 10.3 | FINRA Short-Interest | Short-Sentiment | ✅ |
| 10.4 | CBOE Public CSVs | Put/Call, VIX-Termstruktur | ✅ |
| 10.5 | GDELT 2.1 | News-Event-Graph | ✅ |
| 10.6 | Alpaca Free-Tier (IEX) | US-Live-Quotes | ✅ |
| 10.7 | yfinance | Fallback, Prototyping | ⚠ **nicht für SaaS** |
| 10.8 | Stooq | EU-EOD-Fallback | ⚠ grauzone |
| 10.9 | Finnhub Free | News + Earnings + Ratings | ✅ |
| 10.10 | PRNewswire/BusinessWire/GlobeNewswire RSS | Press-Release-Wires | ✅ |
| 10.11 | CoinMetrics Community | Crypto-Macro | ✅ |
| 10.12 | USPTO via `patent_client` | Patent-Filings | ✅ Public Domain |
| 10.13 | OpenWeatherMap / NOAA / Open-Meteo | Weather für Agrar/Energie | ✅ (OWM: ODbL-Attribution) |
| 10.14 | Wikipedia Page Views via `mwviews` | Retail-Attention-Signal | ✅ CC-BY-SA |
| 10.15 | Reddit via PRAW | WSB-Sentiment | ⚠ Commercial-License nötig |
| 10.16 | Stocktwits Public-Endpoints | Self-labeled Bullish/Bearish | ⚠ Toleranzzugang |
| 10.17 | Bluesky Jetstream | Twitter-Ersatz post-2026 | ✅ |
| 10.18 | Alpha Vantage Free | News-Sentiment pre-computed | ✅ |

---

## 10.1 SEC EDGAR via `edgartools`

**Was:** Form 4 Insider-Trading, 8-K Material Events, 10-K/Q, 13F.
**Rate-Limit:** 10 req/s bei gesetztem User-Agent (Pflicht: `"Company Name email@example.com"`, sonst HTTP 403).
**Filing-Lag:** Form 4 erscheint 5 Minuten nach Submission, 8-K T+0 bis T+4-Business-Days je nach Item.
**Install:** `pip install edgartools`
**Zweck im System:** Real-time 8-K als News-Quelle, Form 4 als Insider-Signal (siehe `13_FREE_MODULE.md` §13.4).

**Wichtige Filter:**
- Form-4-Transaction-Codes: nur `P` (Purchase) und `S` (Sell) als Sentiment. Codes `A` (Award), `M` (Option-Exercise) haben **keinen** Informationsgehalt.
- 10b5-1-Plan-Dispositions (Footnote-Markierung) filtern — nicht-informativ.

**RSS-Polling:** `https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=4&output=atom` alle 60-120 Sekunden.

---

## 10.2 FRED via `fredapi`

**Was:** Federal Reserve Economic Data — Zinsen, Inflation, Spreads, Makro-Indikatoren.
**Rate-Limit:** 120 req/min, API-Key kostenlos, unbegrenzte Calls/Tag.
**Install:** `pip install fredapi==0.5.2`
**Setup:** Key kostenlos auf https://fred.stlouisfed.org/docs/api/api_key.html

**Kern-Serien für das System:**

| Serie | Bedeutung | Verwendung |
|---|---|---|
| `DGS10` | 10y-Treasury-Yield | Yield-Curve-Feature |
| `DFII10` | 10y-TIPS | Breakeven-Inflation = `DGS10 - DFII10` |
| `T10Y2Y` | 2s10s-Curve-Slope | Recession-Indicator, Regime-Feature |
| `EFFR` | Fed-Funds-Rate | Policy-Stance |
| `ECBDFR` | ECB-Deposit-Rate | Central-Bank-Divergence |
| `BAMLH0A0HYM2` | HY-OAS-Spread | Risk-Appetite-Proxy |
| `VIXCLS` | VIX-Close | Fear-Index |
| `DTWEXBGS` | Broad-Dollar-Index | USD-Strength |
| `NFCI` | National Financial Conditions | Systemisch-Risk |
| `T10Y3M` | 10y-3m | Yield-Curve-Inversion-Signal |

**FOMC-Minutes und Beige-Book:** nicht API-exposed, aber via Scraping von `federalreserve.gov/monetarypolicy/fomccalendars.htm` oder `fedtools`-Library zugänglich.

---

## 10.3 FINRA Short-Interest

**Was:** Short-Interest-Daten komplett free, kein Key.
**Endpoint:** `https://api.finra.org/data/group/otcMarket/name/EquityShortInterest`
**Methode:** POST mit JSON-Body.
**Update:** bi-monthly mit T+8-Business-Day-Lag.
**Daily-Variante (noch wertvoller):** `regShoDaily`-Endpoint — granular über das Universum, deutlich reaktiver.
**Abdeckung:** seit Juni 2021 inkl. Exchange-Listed Securities.

**Verwendung im System:** Short-Interest-Build-up als Signal (siehe `13_FREE_MODULE.md` §13.5).

---

## 10.4 CBOE Public CSVs

**Was:** Put/Call-Ratios, VX-Futures-Settlement, VIX-Termstruktur — alles free, keine Registrierung.
**Endpoints:**
```
https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/equitypcarchive.csv
https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/indexpcarchive.csv
https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/totalpcarchive.csv
```
**VIX-Termstruktur:** `^VIX9D, ^VIX, ^VIX3M, ^VIX6M, ^VVIX, ^SKEW` via yfinance kostenlos — **du brauchst kein Unusual-Whales-Abo** solange du nicht auf Optionen-Chain-Flow gehst.

**Verwendung:** Volatility-Surface-Dimension im Composite (siehe `31_COMPOSITE_SCORE.md` Dim-6).

---

## 10.5 GDELT 2.1 (tiefer nutzen als bisher)

**Was:** Global Database of Events, Language, and Tone. 2200+ GCAM-Emotions, Mentions-DB, Frontpage-Graph.
**Status:** **Es gibt keine 3.0.** Wenn jemand von GDELT 3.0 spricht, ist das Verwirrung mit einem anderen Produkt.
**Zugang:** direkt via `gdeltdoc`-Library oder BigQuery (GCP Free-Tier: 1 TB Queries/Monat).
**Install:** `pip install gdeltdoc`

**Unter-genutzte Features im typischen Setup:**

- **GCAM-Dimensionen:** 2200+ Emotions/Themes-Scores (z.B. `c6.8` = Negative economic outlook). In der aktuellen News-Engine vermutlich nicht berücksichtigt.
- **Mentions-Datenbank:** Tracking von Article-Amplification über Zeit.
- **Frontpage-Graph:** 50k News-Homepages stündlich, Editorial-Positioning als Meta-Signal.
- **NumMentions × AvgTone:** amplifiziertes Sentiment-Feature.

**Qualitäts-Caveats:** Accuracy der Key-Fields laut MDPI-Paper (Oct 2025) ~55 %, Redundanz ~20 %. **Dedup + Correction-Layer zwingend** vor Produktivnutzung.

---

## 10.6 Alpaca Free-Tier (IEX)

**Was:** US-Aktien + Options Paper-Trading + Live-Quotes (IEX-Feed).
**Limits:** 200 req/min, IEX-Only (~2 % Marktvolumen), 15min-delayed für manche Endpoints.
**Preis:** 0 USD für Paper-Trading.
**Wichtig:** IEX-Coverage ist für Microstructure-Features (OFI, VPIN) verzerrt — siehe `31_COMPOSITE_SCORE.md` Dim-3. Für Composite-Score TA/News OK.

**Alpaca Options:** kostenlos mit Paper-Trading, Snapshots inklusive Greeks + IV.
**Historische Bars:** für expired Options-Contracts verfügbar.

**Verwendung:** Primärer Broker für Paper-Trading + primäre US-Live-Quote-Quelle.

---

## 10.7 yfinance (eingeschränkt)

**Was:** Yahoo-Finance-Scraping-Library, deckt US + EU-Ticker (mit Suffix `.DE`, `.L`, `.PA`, `.MI` etc.), Options-Chains, Fundamentals.
**Install:** `pip install yfinance`
**Rechtliche Lage:** Yahoo-ToS verbietet explizit "automated means, scrapers". **Privat toleriert, kommerziell nicht.** Bei Richtung B (B2B-SaaS) zwingend durch Polygon/Alpha-Vantage-Paid ersetzen.

**Verwendung im System:**
- **Prototyping-Phase:** OK.
- **EU-Ticker-Fallback** solange keine paid Alternative.
- **VIX/VVIX/SKEW-Indizes** und Options-Chain-Snapshots (delayed).
- **NICHT** in einer Pipeline, die du weiterverkaufen willst.

---

## 10.8 Stooq

**Was:** EU-EOD-Daten, free, via `pandas-datareader`.
**Coverage:** Deutschland, UK, Polen, teils USA.
**Rechtliche Lage:** keine explizite Scraping-Klausel, inoffizieller Endpoint.
**Install:** `pip install pandas-datareader`

**Verwendung:** EU-EOD-Fallback. **Keine Intraday.** Keine Delisted.

---

## 10.9 Finnhub Free

**Was:** News + Earnings-Calendar + Analyst-Ratings + Price-Targets + basic Fundamentals.
**Rate-Limit:** 60 req/min, WebSocket-Streaming bis 50 Symbole.
**Install:** `pip install finnhub-python`
**Setup:** Free-Key auf finnhub.io

**Endpoints, die du brauchst:**

| Endpoint | Zweck |
|---|---|
| `/news-sentiment` | aggregiertes News-Sentiment pro Ticker |
| `/company-news` | Headlines pro Ticker |
| `/stock/earnings` | EPS-Actual + Estimate (für PEAD/SUE) |
| `/stock/recommendation` | Analyst-Ratings Aggregat |
| `/stock/insider-transactions` | Alternative zu SEC-Form-4 |

**Warum wertvoll:** Bestes Preis-Leistung für News+Earnings+Ratings im Free-Tier.

---

## 10.10 PRNewswire / BusinessWire / GlobeNewswire RSS

**Was:** Press-Release-Primärquellen — **schnellste Earnings-Headlines** (Sekunden vor Reuters).
**Zugang:** kostenlose RSS-Feeds.
**Install:** `pip install feedparser`

**Feeds:**
- PRNewswire: https://www.prnewswire.com/rss/news-releases-list.rss (category-spezifisch konfigurierbar)
- BusinessWire: https://www.businesswire.com/portal/site/home/news/ (RSS-Endpoints pro Kategorie)
- GlobeNewswire: https://www.globenewswire.com/rss/country/United%20States (country-spezifisch)

**Verwendung:** News-Pipeline-Primärquelle für Earnings-Releases und M&A-Announcements.

---

## 10.11 CoinMetrics Community API

**Was:** 550+ Asset Reference-Rates, Network-Data-Pro-Subset, Stablecoin-Flows.
**Rate-Limit:** 10 req/6s sliding, 1000 req/10min total.
**Install:** `pip install coinmetrics-api-client`

**Kern-Makro-Signale:**

| Metrik | Bedeutung |
|---|---|
| USDT+USDC Stablecoin-Supply | Liquidity-Proxy |
| Exchange Net Flows (`FlowInExNtv − FlowOutExNtv`) | Risk-Appetite |
| Active Addresses | Adoption-Momentum |

**Verwendung:** Crypto-Macro-Feature im Breadth/Intermarket-Dim (siehe `31_COMPOSITE_SCORE.md` Dim-7).

**Vermeiden:** Glassnode (799 USD/Monat), CryptoQuant (109/Monat) — sprengen Budget ohne klaren Mehrwert für EOD-Equity.

---

## 10.12 USPTO via `patent_client`

**Was:** Patent-Filings, Public Domain.
**Rate-Limit:** kein hartes Limit, ~45 req/min praxistauglich.
**Install:** `pip install patent_client`
**Alternativ:** Google Patents BigQuery Public Dataset (GCP Free-Tier 1 TB/Monat).

**Verwendung:** Innovations-Signal (nur für Tech-Namen sinnvoll, Filing-Lag mehrere Wochen). **Relevanz 4/10** für EOD-Equity — Long-Term-Conviction-Signal.

---

## 10.13 Weather: OpenWeatherMap / NOAA / Open-Meteo

**Drei Optionen:**

| Service | Free-Limit | Commercial? |
|---|---|---|
| **OpenWeatherMap** | 60 calls/min, 1M/Monat | ODbL-Attribution in SaaS-UI Pflicht |
| **NOAA NWS** (`api.weather.gov`) | kein Key, nur User-Agent | ✅ Public Domain |
| **Open-Meteo** | non-commercial 10k calls/day | **No-Key-Alternative** mit ECMWF/GFS-Daten |

**Kern-Use-Cases:**
- HDD-Berechnung aus 8 US-Region-Temps für Nat-Gas-Demand
- NCEI-Drought-Index kombiniert mit Sentinel-2-NDVI für Corn/Wheat

**Verwendung:** nur bei Energy/Agrar-Exposure im Portfolio relevant.

---

## 10.14 Wikipedia Page Views via `mwviews`

**Was:** Retail-Attention-Signal.
**Rate-Limit:** 100 req/s free, kein Key.
**Install:** `pip install mwviews==0.3`

**Literatur:** Moat et al. 2013 (Nature Sci Rep) — 1-Tag-Lag-Views prediziert Drawdowns, long-short Sharpe ~0.3.

**Verwendung:** Attention-Feature als 8. oder 9. Composite-Subfeature. **Hoch-empfohlen für Phase 2.**

---

## 10.15 Reddit via PRAW (eingeschränkt)

**Was:** WSB-Sentiment, strukturiertes Subreddit-Monitoring.
**Rate-Limit:** 100 QPM OAuth-authenticated, für persönliche/akademische Nutzung unbegrenzt.
**Install:** `pip install praw==7.8.1`
**Rechtliche Lage:** Seit Juli 2023 **kostenpflichtig für Commercial**. Für Phase A (Personal Quant) OK. Für Phase B (SaaS) Commercial-License nötig.

**DSGVO-Achtung:** Bei Persistierung muss Cascade-Delete bei User-Löschung implementiert sein. **On-Demand-Aggregation ohne Rohdaten-Speicherung** ist der saubere Pfad.

**Pushshift ist tot.** Nachfolger: `pullpush.io` (~2000 req/min, kein Key) für historische Backfills, oder `arctic-shift` von Arthur Heitmann für Bulk-Dumps.

**Verwendung:** WSB-Mention-Velocity als Kontra-Indikator (nicht als Alpha-Primary). Retail-Euphorie-Signal.

---

## 10.16 Stocktwits Public-Endpoints

**Was:** Trending-Tickers + Streams pro Symbol mit User-Self-Labeled Sentiment (Bullish/Bearish).
**Achtung:** Dev-Portal nimmt keine neuen Registrierungen mehr an (Stand 2026).
**Public-Endpoints (funktionieren ohne Key):**
```
https://api.stocktwits.com/api/2/trending/symbols.json
https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json
```
**Rate-Limit:** ~200/h pro IP.

**Der Gold-Punkt:** Der native `entities.sentiment.basic`-Tag (Bullish/Bearish) ist schwer zu ersetzen — User haben sich selbst gelabelt. Andere NLP-Systeme müssen Sentiment aus Text inferieren, hier ist es explizit.

**Exit-Plan vorbereiten:** Falls Public-Endpoints geschlossen werden.

---

## 10.17 Bluesky Jetstream (Twitter-Ersatz)

**Was:** AT-Protocol Firehose, Public-Timeline.
**Rate-Limit:** kein Key nötig, WebSocket-Stream.
**Install:** `pip install atproto`
**Finanz-Cashtag-Volumen:** ~30 % von Twitter, wachsend.

**Warum relevant:** Twitter/X hat am 06.02.2026 auf Pay-per-Use umgestellt (0.005 USD/Post-Read, Cap 2M/Monat). 100-EUR-Budget = ~22.000 Reads/Monat. **Twitter/X aus primärer Pipeline entfernen.**

**Alternativen, die NICHT gehen:**
- Nitter: seit 2024 tot
- `twitterapi.io`, `twscrape`, `snscrape`: ToS-Verstoß und/oder kaputt

---

## 10.18 Alpha Vantage Free-Tier

**Was:** News-Sentiment + Earnings + FX + Technical-Indicators, pre-computed.
**Rate-Limit:** 25 req/Tag free, 500 req/Tag mit Free-Key.
**Install:** HTTP direkt oder via `alpha_vantage`-Library.

**Highlights Free-Tier:**
- **News-Sentiment-Endpoint** mit Relevance-Scores und Topic-Filter (earnings/M&A/macro).
- Earnings-Call-Transcripts (25 calls/Tag free) — **legal vs. Seeking-Alpha-Scraping**.

**Verwendung:** Sekundäre News-Quelle, primär für Earnings-Call-Transcripts.

---

## Was NICHT in diesen Katalog gehört

Explizit **nicht** als Free-Quelle empfehlen (trotz kostenloser Zugänge):

- **pytrends / Google-Trends:** April 2025 offiziell archiviert. Maintainer-Statement: Google liefert bei Bot-Detection gefälschte Daten. Alternative `trendspy` hat gleiche Quality-Risiken.
- **Satellite-Data für Retail-Equity:** Sentinel/Copernicus kostenlos, aber TB-Storage + Vision-Models + Geo-Referencing sprengen Solo-Kapazität. Nur für spezielle Commodity-Cases.
- **LinkedIn/Indeed/Glassdoor Job-Scraping:** ToS verbietet. `python-jobspy` rechtlich riskant.
- **Credit-Card-Panel:** 10k+ USD/Monat unerreichbar.
- **Shipping/AIS:** MarineTraffic paid, AIS-Hub nur bei eigener Receiver-Hardware.
- **Seeking Alpha via RapidAPI:** Scraping-basiert, ToS-Verstoß — nicht für Cloud-Deployment.

---

## Umsetzungs-Checkliste für dieses Modul

- [ ] SEC EDGAR User-Agent konfiguriert und in Config
- [ ] FRED-API-Key generiert und in `.env`
- [ ] FINRA-Endpoint getestet mit POST + JSON
- [ ] CBOE-CSV-Daily-Download als Cron aufgesetzt
- [ ] GDELT-Deep-Features (GCAM, Mentions) in Pipeline eingebaut
- [ ] Alpaca Paper-Account existiert, IEX-Feed funktioniert
- [ ] Stooq + yfinance als EU-Fallback testen
- [ ] Finnhub-Free-Key, alle 5 Kern-Endpoints produktiv
- [ ] PRNewswire/BusinessWire/GlobeNewswire RSS eingebunden
- [ ] CoinMetrics Community-Connection funktioniert
- [ ] `mwviews` für Top-20 Ticker einmal pro Tag abrufen
- [ ] PRAW-OAuth + ein Subreddit-Monitor (WSB)
- [ ] Bluesky Jetstream für Cashtag-Monitoring
- [ ] Alpha Vantage Earnings-Call-Transcripts Integration

---

## Ehrliche Einschätzung

Dieser Free-Stack ist **90 % so gut wie ein 500-EUR/Monat-Paid-Stack** für ein Retail-Quant-System mit EOD-Fokus. Die 10 % Lücke sind:

1. **Delisted-Coverage** (Survivorship-Bias-Schutz) — kommt aus EODHD 19.99 USD, siehe `20_PAID_DATEN.md`.
2. **Real-time SIP-Feed statt IEX** (Microstructure-Präzision) — kommt aus Alpaca Algo Trader Plus 9 USD oder Polygon Developer 29 USD.
3. **LLM-Qualität der Sentiment-Extraktion** — kommt aus Claude Haiku 4.5 API (<10 EUR/Monat).

Alles andere — Alt-Data-Abos, Premium-News, Options-Flow — hat **geringen Grenznutzen gegenüber diesem Free-Stack**, solange deine Backtests nicht sauber laufen.

**Also: erst den Free-Stack stabil und cpcv-validiert, dann gezielt einkaufen.**
