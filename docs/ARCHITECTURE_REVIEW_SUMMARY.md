# Architektur-Review & Systematische Überprüfung

**Erstellt:** 2025-12-09  
**Status:** Vollständige Überprüfung der API- & Daten-Architektur, Feature-Layer, Workflows und Hidden Traps

---

## 1. API- & Daten-Architektur ✅

### Preise

**✅ Konsistent:**
- Preise kommen **ausschließlich** aus lokalen Parquet-Snapshots
- `LocalParquetPriceDataSource` + `ASSEMBLED_LOCAL_DATA_ROOT`
- Backtests, Factor-Analysis, Event-Studies – alle Price-Flows laufen darüber
- **Keine** Finnhub-Candles, **keine** Twelve-Data-Candles im Plan für die Factor-Engine

**✅ Design-Ziel erreicht:**
- Alt-Daten-Snapshot als stabile Basis
- Live-Daten nur optional / später

### Alt-Daten (Finnhub)

**B1 – Earnings & Insider:**
- `finnhub_events.py`: `fetch_earnings_events()`, `fetch_insider_events()`
- `download_altdata_finnhub_events.py`: Schreibt nach `data/raw/altdata/finnhub/...` (raw) und `output/altdata/events_*.parquet` (clean)
- **✅ Nur Events, keine Preise** – konsistent mit dem Plan

**B2 – News & Macro:**
- `finnhub_news_macro.py`: `fetch_news()`, `fetch_news_sentiment()`, `fetch_macro_series()`
- `download_altdata_finnhub_news_macro.py`: Schreibt nach `data/raw/altdata/finnhub/...` (raw) und `output/altdata/news_*.parquet`, `macro_*.parquet` (clean)
- **✅ Alle Endpunkte sind Non-Price-Altdata** (News, Sentiment, Economic)

**Fazit:**
- ✅ Provider-Plan ist sauber: Prices = lokale Snapshots, Alt-Data = Finnhub
- ✅ Kein Vermischen von Preis-APIs und Alt-Data-APIs

---

## 2. Feature-Layer ✅

### B1 – Earnings & Insider Factors

**`altdata_earnings_insider_factors.py`:**
- `build_earnings_surprise_factors()`: EPS/Revenue Surprise, Flags, Post-Earnings-Drift
- `build_insider_activity_factors()`: Net Notional, Buy/Sell-Counts, Ratio, Normalisierung

**✅ Design passt:**
- Panel-Format (timestamp, symbol, Faktoren)
- Join über timestamp + symbol
- Nutzt Preise nur über übergebenen `prices`-DataFrame (kein versteckter API-Call)

### B2 – News & Macro Factors

**`altdata_news_macro_factors.py`:**
- `build_news_sentiment_factors()`: Rolling mean, Trend, Shock-Flag, Volume
- `build_macro_regime_factors()`: Growth/Inflation/Risk-Regime (gleicher Wert für alle Symbole pro Tag)

**✅ Design passt:**
- Panel-Output, gleiche Struktur wie B1
- Preise zur Zeit-Ausrichtung, nicht aus API gezogen
- Einfach mit Core-TA-Faktoren kombinierbar

---

## 3. Workflows & CLI ✅

### `analyze_factors` CLI

**Factor-Sets (aktuell):**
- **Phase A/C:** `core`, `vol_liquidity`, `core+vol_liquidity`, `all`
- **Phase B1:** `alt_earnings_insider`, `core+alt`
- **Phase B2:** `alt_news_macro`, `core+alt_news`, `core+alt_full`

**✅ Logisch, aber Naming könnte aufgeräumt werden:**
- Später möglich: `core+alt_b1`, `core+alt_b2`, `core+alt_all`

**✅ Integration in `run_factor_analysis.py`:**
- Lädt Preise über `PriceDataSource(data_source="local")`
- Lädt Alt-Data-Parquets aus `output/altdata`
- Falls Datei fehlt → Warnung, aber kein Crash
- Join't alles in ein Factor-Panel
- **Perfekt fürs Zusammenspiel mit C1/C2**

### Event-Study-Workflow

**`qa/event_study.py` + `research/events/event_study_template_core.py`:**
- B1 & B2 können: reale Earnings-Events (B1), optional News-Events (B2), Macro-Regime als Zusatz-Overlay (B2)
- **✅ Passt alles:**
  - Events/Alt-Data von Finnhub
  - Prices aus lokalen Parquets
  - Event-Engine (C3) sitzt "darüber"

---

## 4. Tests & Doku ✅

**B1:**
- ✅ Tests für `altdata_earnings_insider_factors`
- ✅ Tests für `finnhub_events`-Client

**B2:**
- ✅ Tests für `altdata_news_macro_factors`
- ✅ Tests für `finnhub_news_macro`-Client

**C1/C2/C3:**
- ✅ Tests für Factor-IC/IR, Portfolio-Engine + Event-Study
- ✅ Alles mit `@pytest.mark.advanced` markiert

**Offener Punkt:**
- 2 Fehler in `test_factor_analysis.py` bei älterer Funktion (`compute_rank_ic` alte Signatur)
- Unabhängig von B1/B2 – kann in eigenem Mini-Sprint gefixt werden

---

## 5. Hidden Traps & Empfehlungen

### (1) End-to-End-Smoketests

**Empfohlen:** Drei konkrete End-to-End-Läufe durchführen:

1. **Plain Core-Factors** (nur Alt-Prices)
   - Prüft Basis-Factor-Pipeline
   - Universe: `macro_world_etfs_tickers.txt`
   - Factor-Set: `core`

2. **Core + B1** (Earnings/Insider)
   - Prüft Alt-Data B1-Integration
   - Vorher: `download_altdata_finnhub_events.py` laufen lassen
   - Factor-Set: `core+alt`

3. **Core + B1 + B2** (Full Alt-Stack)
   - Prüft kompletten Alt-Data-Stack
   - Vorher: Events + News + Macro downloaden
   - Factor-Set: `core+alt_full`

**Wenn alle drei Reports erfolgreich generiert werden:** System ist faktisch "Research-ready"

**Dokumentiert in:** `docs/WORKFLOWS_FACTOR_ANALYSIS.md` → "End-to-End Smoketests"

### (2) Symbol-Mismatches

**Problem:**
- Alt-Data-Events (Earnings, Insider, News) können für Symbole existieren, die keine Price-Historie haben (z.B. europäische Ticker)
- Events für Symbole ohne Price-Historie werden beim Join in `build_*_factors()` stillschweigend verworfen

**Verhalten:**
- Funktional korrekt, kann aber zu "verlorenen" Events führen
- Factor-Builder filtern Events auf Symbole, die in `prices` DataFrame vorhanden sind

**Empfehlung:**
- Vor Factor-Analysis-Lauf: Prüfen, welche Symbole in Alt-Data-Dateien vorhanden sind
- Sicherstellen, dass für alle relevanten Symbole Price-Daten vorhanden sind
- Oder: Universen so definieren, dass nur Symbole mit vollständigen Daten enthalten sind

**Dokumentiert in:** `docs/WORKFLOWS_FACTOR_ANALYSIS.md` → "Wichtige Hinweise" → "Symbol-Mismatches"

**Beispiel-Check-Code:**
```python
# Prüfen, welche Symbole in Events vorhanden sind
events_earnings = pd.read_parquet("output/altdata/events_earnings.parquet")
event_symbols = set(events_earnings["symbol"].unique())

# Prüfen, welche Symbole in Price-Daten vorhanden sind
prices = pd.read_parquet("output/aggregates/1d.parquet")
price_symbols = set(prices["symbol"].unique())

# Symbole mit Events aber ohne Prices
missing_prices = event_symbols - price_symbols
if missing_prices:
    print(f"Warnung: {len(missing_prices)} Symbole haben Events aber keine Price-Daten: {missing_prices}")
```

### (3) Naming & Factor-Sets

**Aktueller Stand:**
- 9 verschiedene Factor-Sets (historisch gewachsen)
- Funktional korrekt, aber UX könnte verbessert werden

**Dokumentiert in:** `docs/WORKFLOWS_FACTOR_ANALYSIS.md` → "Factor-Set Übersicht"

**Vorschlag für spätere Umbenennung:**
- `core` → bleibt
- `vol_liquidity` → bleibt
- `core+vol_liquidity` → bleibt
- `all` → bleibt
- `alt_earnings_insider` → `alt_b1`
- `alt_news_macro` → `alt_b2`
- `core+alt` → `core+alt_b1`
- `core+alt_news` → `core+alt_b2`
- `core+alt_full` → `core+alt_all`

**Status:** Kein Bug, eher UX-Verbesserung für später

---

## 6. Nächster sinnvoller Schritt

**Empfohlener Standard-Workflow:**

**Universe:**
- `macro_world_etfs_tickers.txt` + `universe_ai_tech_tickers.txt`

**Zeitraum:**
- 2010–2025

**Factor-Set:**
- `core+alt_full`

**Darauf:**
1. IC/IR auswerten
2. Portfolio-Summaries anschauen
3. Erste "Top-Faktoren"-Liste erstellen (für A/B/C/D-Phasen)

**Workflow:**
```powershell
# 1. Alt-Data herunterladen (falls noch nicht geschehen)
python scripts/download_altdata_finnhub_events.py `
  --symbols-file config/universe_ai_tech_tickers.txt `
  --start-date 2010-01-01 `
  --end-date 2025-12-03 `
  --event-types earnings insider

python scripts/download_altdata_finnhub_news_macro.py `
  --symbols-file config/universe_ai_tech_tickers.txt `
  --start-date 2010-01-01 `
  --end-date 2025-12-03 `
  --download-news-sentiment `
  --download-macro

# 2. Factor-Analysis durchführen
$env:ASSEMBLED_LOCAL_DATA_ROOT = "F:\Python_Projekt\Aktiengerüst\datensammlungen\altdaten\stand 3-12-2025"

python scripts/cli.py analyze_factors `
  --freq 1d `
  --symbols-file config/universe_ai_tech_tickers.txt `
  --data-source local `
  --start-date 2010-01-01 `
  --end-date 2025-12-03 `
  --factor-set core+alt_full `
  --horizon-days 20 `
  --quantiles 5

# 3. Ergebnisse analysieren
# - output/factor_analysis/factor_ic_summary.csv
# - output/factor_analysis/factor_portfolio_summary.csv
```

---

## 7. Zusammenfassung

### ✅ Was gut funktioniert:

1. **API-Architektur:**
   - Saubere Trennung: Prices = lokale Snapshots, Alt-Data = Finnhub
   - Kein Vermischen von Preis-APIs und Alt-Data-APIs

2. **Feature-Layer:**
   - B1 & B2 hängen logisch zusammen
   - Panel-Format konsistent
   - Preise nur über übergebenen DataFrame (kein versteckter API-Call)

3. **Workflows:**
   - `analyze_factors` CLI integriert alle Komponenten sauber
   - Event-Study-Workflow nutzt B1/B2 korrekt

4. **Tests:**
   - Umfassende Test-Suites für B1/B2/C1/C2/C3
   - Alles mit `@pytest.mark.advanced` markiert

### ⚠️ Was beachtet werden sollte:

1. **Symbol-Mismatches:**
   - Events für fehlende Symbole werden verworfen
   - Dokumentiert in `WORKFLOWS_FACTOR_ANALYSIS.md`
   - Beispiel-Check-Code vorhanden

2. **Factor-Set-Naming:**
   - 9 verschiedene Sets (historisch gewachsen)
   - Funktional korrekt, aber UX könnte verbessert werden
   - Tabelle in Doku vorhanden

3. **End-to-End-Smoketests:**
   - Drei konkrete Testläufe empfohlen
   - Dokumentiert in `WORKFLOWS_FACTOR_ANALYSIS.md`

### 📋 Nächste Schritte:

1. **Sofort:** End-to-End-Smoketests durchführen (siehe Doku)
2. **Kurzfristig:** Top-Faktoren-Liste erstellen basierend auf IC/IR
3. **Mittelfristig:** Factor-Set-Naming aufräumen (optional)
4. **Langfristig:** Regime-Analyse (Phase D) mit Macro-Faktoren

---

## 8. Architektur-Status

**Gesamtbewertung:** ✅ **Architektur ist sauber und konsistent**

- API- & Daten-Architektur: ✅ Sauber getrennt
- Feature-Layer: ✅ Logisch zusammenhängend
- Workflows & CLI: ✅ Konsistent integriert
- Tests & Doku: ✅ Umfassend abgedeckt
- Hidden Traps: ✅ Identifiziert und dokumentiert

**System ist faktisch "Research-ready"** nach erfolgreichen End-to-End-Smoketests.

