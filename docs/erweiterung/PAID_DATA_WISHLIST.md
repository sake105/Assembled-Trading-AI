# Paid Data Wishlist

Diese Liste dokumentiert, welche kostenpflichtigen Datenquellen den größten Mehrwert über die kostenlosen Quellen hinaus bringen würden — und wie sie ins ERWEITERUNG-Framework integrierbar wären, sobald Budget verfügbar ist.

Geordnet nach **erwartetem Alpha-Beitrag pro Dollar/Monat**.

---

## 1. OptionMetrics IvyDB (Optionen, USA)

- **Was:** Historische End-of-Day Options-Chains, IV-Surfaces, Greeks, OI für US-Aktien, Indizes, ETFs ab 1996.
- **Warum:** Yahoo-Options liefern nur Snapshot. Für rigorose Backtests historischer Skew/Term-Structure-Strategien gibt es keine kostenlose Alternative.
- **Nutzen:**
  - Variance-Risk-Premium-Backtest (5-10 % p.a. dokumentiert)
  - Skew-Crash-Indikator (Cremers/Weinbaum 2010)
  - Implied-Vol-Term-Structure-Inversion
- **Modul:** `altdata/optionmetrics_loader.py` (geplant).
- **Kosten:** ~$10-30k/Jahr akademisch, höher kommerziell.

## 2. RavenPack News Analytics

- **Was:** Maschinenlesbares News-Sentiment auf Symbol-Ebene mit präzisem Sub-Sekunden-Timestamp. Keine Look-Ahead.
- **Warum:** Kostenlose RSS-Feeds haben Veröffentlichungs-Verzögerung von Minuten bis Stunden. Reddit-Mentions sind verzerrt.
- **Nutzen:**
  - Saubere Event-Studies
  - Day-of-News-Reaction-Modeling
- **Kosten:** ~$50-100k/Jahr.

## 3. Compustat / WRDS Fundamentaldaten

- **Was:** Bereinigte Fundamentaldaten (Bilanz, GuV, Cash-Flow) ab 1950er, Survivorship-bias-frei.
- **Warum:** yfinance/Polygon/Alpha-Vantage haben (a) keine PIT-Sicherheit (Restatements), (b) nur ~5 Jahre, (c) Survivorship-Bias.
- **Nutzen:** echte Faktor-Backtests Fama-French-Style, Earnings-Quality, Accruals-Anomalien.
- **Kosten:** WRDS akademisch ~$25k/Jahr; kommerziell $100k+.

## 4. Bloomberg / Refinitiv Datafeed (BPipe)

- **Was:** Real-Time Quotes, Tick-by-Tick, alle globalen Märkte.
- **Warum:** Yahoo ist ungenau und 15-Min-delayed; Polygon ist nur USA.
- **Nutzen:** Cross-Asset Strategien (FX, Bonds, Commodities), globale Equities.
- **Kosten:** ab $24k/Jahr (BPipe), Terminal $24k/Jahr.

## 5. CRSP Daily/Monthly Database

- **Was:** Survivorship-bias-freier US-Equity-Returns-Datensatz seit 1925, mit Delistings.
- **Warum:** yfinance-Backtests sind **systematisch zu optimistisch** wegen Survivorship-Bias.
- **Nutzen:** Echte Cross-Sectional-Backtests; Reproduzierbarkeit akademischer Studien.
- **Kosten:** WRDS-Subscription ~$20k/Jahr akademisch.

## 6. SEC EDGAR XBRL Premium-Parser

- **Was:** Strukturierte Bilanz-Daten aus SEC-Filings (Cap-IQ-Style).
- **Warum:** Unser kostenloser EDGAR-Loader liefert nur Filing-Index. Volle XBRL-Extraktion ist aufwendig.
- **Alternative kostenlos:** `edgartools` Python-Package — gut, aber langsam.
- **Kosten:** AlphaSense / Sentieo ab $10k/Jahr.

## 7. Glassnode / Coin Metrics On-Chain

- **Was:** Crypto On-Chain-Metriken (Realized-Cap, Active-Addresses, Whale-Holdings).
- **Warum:** CoinGecko liefert nur Preis-/Volumendaten. On-Chain ist die echte Alpha-Quelle für Crypto-Trading.
- **Kosten:** Glassnode ab $30/Monat (Lite) bis $1k/Monat (Pro).

## 8. T2 / DTN / Quandl Insider-Trading-Premium

- **Was:** Aggregierte Insider-Transaction-Daten mit Filtering (Cluster, Quality-Tags).
- **Warum:** SEC-Form-4-Parser ist roh; "echte" Insider-Signale brauchen Cluster-Detection und Officer-Quality-Tags.
- **Kosten:** ~$2-5k/Jahr.

## 9. CFTC Disaggregated COT — Premium Cleansed

- **Was:** Bereinigte COT-Reports mit Mapping zu liquiden Underlyings.
- **Warum:** Die kostenlose Socrata-API ist roh; viele Marktteilnehmer-Codes sind dupliziert oder veraltet.
- **Kosten:** $1-3k/Jahr (Hightower / Briese).

## 10. Borsa Italiana / Deutsche Börse Tick-Data

- **Was:** EU-Tick-Daten für Cross-Listed-Equities.
- **Warum:** Yahoo hat dünne EU-Coverage.
- **Kosten:** ~$5-15k/Jahr je Markt.

---

## Zusammenfassung Budget-Stufen

| Budget | Empfehlung |
|--------|------------|
| $0 / Monat | Aktuelle ERWEITERUNG (10 freie Quellen) |
| ~$50 / Monat | Glassnode-Standard für Crypto-Alpha |
| ~$2k / Jahr | + COT-Premium + Insider-Premium |
| ~$25k / Jahr | + WRDS akademisch (CRSP, Compustat) |
| ~$100k / Jahr | + RavenPack + OptionMetrics |
| $250k+ / Jahr | + Bloomberg-Terminal + BPipe |

Empfehlung für früheste Investition: **WRDS akademisch** — survivorship-bias-freie US-Equity-Backtests sind die Grundlage jeder seriösen Forschung und ohne keine Backtest-Aussage belastbar.

## Integration in ERWEITERUNG

Wenn ein Datenquelle dazukommt, bekommt sie ein neues Modul `altdata/<source>.py` mit dem gleichen `FetchResult`-Schema. Backtest- und Signal-Module bleiben unverändert — sie konsumieren nur den neuen Faktor.
