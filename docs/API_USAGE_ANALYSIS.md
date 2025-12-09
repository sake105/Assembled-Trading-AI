# API-Nutzung & Performance-Analyse

**Erstellt:** 2025-12-09  
**Status:** Analyse der aktuellen API-Nutzung, Performance-Probleme und Baustellen

---

## 1. API-Provider-Übersicht

### 1.1 Preis-Daten (Historical)

| Provider | Verwendung | Status | Rate Limits | Performance |
|----------|------------|--------|-------------|-------------|
| **LocalParquetPriceDataSource** | Primär für alle Workflows | ✅ Empfohlen | Keine | ⚡ Sehr schnell |
| **YahooPriceDataSource** | Fallback/Live-Daten | ⚠️ Rate-Limits | Unbekannt, oft blockiert | 🐌 Langsam bei vielen Symbols |
| **FinnhubPriceDataSource** | Alternative für Preise | ⚠️ Nicht empfohlen | 60/min (Free) | 🐌 Langsam (Symbol-für-Symbol) |
| **TwelveDataPriceDataSource** | Alternative für Preise | ⚠️ Nicht empfohlen | 8/min (Free) | 🐌 Sehr langsam (Symbol-für-Symbol) |

**Empfehlung:** 
- **Immer `data_source="local"` verwenden** für alle Workflows (Backtest, Factor Analysis, etc.)
- Preis-Daten nur über Download-Skripte aktualisieren (z.B. `download_historical_snapshot.py`)
- **NICHT** in Backtests oder Factor-Analysis-Workflows direkt API-Calls für Preise machen

### 1.2 Alt-Data (Events, News, Macro)

| Provider | Verwendung | Status | Rate Limits | Performance |
|----------|------------|--------|-------------|-------------|
| **Finnhub Events API** | Earnings & Insider Events | ✅ OK | 60/min (Free) | ⚠️ Symbol-für-Symbol Loop |
| **Finnhub News/Macro API** | News, Sentiment, Macro | ✅ OK | 60/min (Free) | ⚠️ Symbol-für-Symbol Loop |

**Empfehlung:**
- Alt-Data nur über Download-Skripte holen (`download_altdata_finnhub_events.py`, `download_altdata_finnhub_news_macro.py`)
- **NICHT** in Factor-Analysis-Workflows direkt API-Calls für Alt-Data machen
- Alle Alt-Data aus lokalen Parquet-Dateien laden (`output/altdata/*.parquet`)

---

## 2. Aktuelle API-Nutzung im Code

### 2.1 Preis-Daten-Quellen

**`src/assembled_core/data/data_source.py`:**

#### LocalParquetPriceDataSource ✅ (Optimal)
- **Verwendung:** Primär für alle Workflows
- **Performance:** ⚡ Sehr schnell (lokale Dateien)
- **Keine API-Calls:** Lädt nur lokale Parquet-Dateien
- **Status:** ✅ Empfohlen

#### YahooPriceDataSource ⚠️ (Rate-Limits)
- **Verwendung:** Fallback für Live-Daten
- **Performance:** 🐌 Langsam bei vielen Symbols
- **Problem:** Loop über Symbole (Zeile 299: `for symbol in symbols:`)
- **Rate-Limits:** Unbekannt, oft blockiert
- **Status:** ⚠️ Nur für kleine Universen oder als Fallback

#### FinnhubPriceDataSource ⚠️ (Nicht empfohlen)
- **Verwendung:** Alternative für Preise (aber nicht empfohlen!)
- **Performance:** 🐌 Langsam (Loop über Symbole, Zeile 468: `for symbol in symbols:`)
- **Rate-Limits:** 60/min (Free Tier)
- **Problem:** 
  - **Symbol-für-Symbol Loop** (ineffizient)
  - **Keine Batch-API** verfügbar
  - **Keine Delays** zwischen Requests (kann Rate-Limits überschreiten)
- **Status:** ⚠️ Nicht für Produktions-Workflows verwenden

#### TwelveDataPriceDataSource ⚠️ (Sehr langsam)
- **Verwendung:** Alternative für Preise (aber nicht empfohlen!)
- **Performance:** 🐌 Sehr langsam (Loop über Symbole, Zeile 656: `for symbol in symbols:`)
- **Rate-Limits:** 8/min (Free Tier) - sehr restriktiv
- **Problem:**
  - **Symbol-für-Symbol Loop** (ineffizient)
  - **Keine Batch-API** verfügbar
  - **Sehr lange Delays** nötig (7.5s zwischen Requests)
- **Status:** ⚠️ Nicht für Produktions-Workflows verwenden

### 2.2 Alt-Data-Quellen

**`src/assembled_core/data/altdata/finnhub_events.py`:**

#### fetch_earnings_events() ⚠️ (Performance-Problem)
- **Verwendung:** Download-Skripte (`download_altdata_finnhub_events.py`)
- **Performance:** 🐌 Langsam bei vielen Symbols
- **Problem:** 
  - **Ein API-Call für alle Symbole** (Zeile 107-140), aber dann Filterung
  - **Besser:** Batch-API nutzen, wenn verfügbar
- **Rate-Limits:** 60/min (Free Tier)
- **Delays:** `RATE_LIMIT_DELAY_SECONDS = 1.0` (Zeile 33)
- **Status:** ⚠️ OK für Download-Skripte, aber könnte optimiert werden

#### fetch_insider_events() ⚠️ (Performance-Problem)
- **Verwendung:** Download-Skripte (`download_altdata_finnhub_events.py`)
- **Performance:** 🐌 Langsam bei vielen Symbols
- **Problem:** 
  - **Symbol-für-Symbol Loop** (Zeile 269: `for symbol in symbols:`)
  - **Keine Batch-API** verfügbar
  - **Delays:** `RATE_LIMIT_DELAY_SECONDS = 1.0` zwischen Symbols
- **Status:** ⚠️ OK für Download-Skripte, aber sehr langsam bei großen Universen

**`src/assembled_core/data/altdata/finnhub_news_macro.py`:**

#### fetch_news() ⚠️ (Performance-Problem)
- **Verwendung:** Download-Skripte (`download_altdata_finnhub_news_macro.py`)
- **Performance:** 🐌 Langsam bei vielen Symbols
- **Problem:** 
  - **Symbol-für-Symbol Loop** (Zeile 113: `for symbol in symbols:`)
  - **Keine Batch-API** verfügbar
  - **Delays:** `RATE_LIMIT_DELAY_SECONDS = 1.0` zwischen Symbols
- **Status:** ⚠️ OK für Download-Skripte, aber sehr langsam bei großen Universen

#### fetch_news_sentiment() ⚠️ (Performance-Problem)
- **Verwendung:** Download-Skripte (`download_altdata_finnhub_news_macro.py`)
- **Performance:** 🐌 Langsam (nutzt `fetch_news()` intern, also auch Symbol-Loop)
- **Status:** ⚠️ OK für Download-Skripte

#### fetch_macro_series() ✅ (OK)
- **Verwendung:** Download-Skripte (`download_altdata_finnhub_news_macro.py`)
- **Performance:** ⚡ OK (Loop über Macro-Codes, aber typischerweise wenige Codes)
- **Status:** ✅ OK

---

## 3. Performance-Probleme

### 3.1 Symbol-für-Symbol Loops (Kritisch)

**Problem:** Viele Funktionen iterieren über Symbole und machen einzelne API-Calls:

1. **`YahooPriceDataSource.get_history()`** (Zeile 299)
   - Loop: `for symbol in symbols:`
   - Ein API-Call pro Symbol
   - **Lösung:** yfinance unterstützt Batch-Downloads! → `yf.download(symbols)` verwenden

2. **`FinnhubPriceDataSource.get_history()`** (Zeile 468)
   - Loop: `for symbol in symbols:`
   - Ein API-Call pro Symbol
   - **Problem:** Keine Batch-API verfügbar, aber keine Delays zwischen Requests
   - **Lösung:** Delays zwischen Requests hinzufügen

3. **`TwelveDataPriceDataSource.get_history()`** (Zeile 656)
   - Loop: `for symbol in symbols:`
   - Ein API-Call pro Symbol
   - **Problem:** Sehr restriktive Rate-Limits (8/min)
   - **Lösung:** Delays zwischen Requests (7.5s minimum)

4. **`fetch_insider_events()`** (Zeile 269)
   - Loop: `for symbol in symbols:`
   - Ein API-Call pro Symbol
   - **Problem:** Sehr langsam bei großen Universen
   - **Lösung:** Batch-API prüfen oder parallele Requests (mit Rate-Limit-Respektierung)

5. **`fetch_news()`** (Zeile 113)
   - Loop: `for symbol in symbols:`
   - Ein API-Call pro Symbol
   - **Problem:** Sehr langsam bei großen Universen
   - **Lösung:** Batch-API prüfen oder parallele Requests (mit Rate-Limit-Respektierung)

### 3.2 Ineffiziente DataFrame-Operationen

**Problem:** Viele kleine Merges statt einem großen Merge:

1. **`build_news_sentiment_factors()`** (Zeile 241-307)
   - Loop über Symbole: `for symbol in result[group_col].unique():`
   - **Zwei `merge_asof`-Calls pro Symbol** (Zeile 258, 283)
   - **Viele `pd.concat()`-Calls** (Zeile 307)
   - **Lösung:** Alle Symbole auf einmal mergen (mit MultiIndex oder groupby)

2. **`build_macro_regime_factors()`** (Zeile 569-590)
   - Loop über Symbole: `for symbol in result[group_col].unique():`
   - **Ein `merge_asof`-Call pro Symbol** (Zeile 575)
   - **Viele `pd.concat()`-Calls** (Zeile 590)
   - **Lösung:** Da alle Symbole denselben Regime-Wert haben, kann man direkt mergen ohne Loop

3. **`build_earnings_surprise_factors()`** (Zeile 206-256)
   - Loop über Symbole: `for symbol in result[group_col].unique():`
   - **Ein `merge_asof`-Call pro Symbol** (Zeile 230)
   - **Viele `pd.concat()`-Calls** (Zeile 256)
   - **Lösung:** Alle Symbole auf einmal mergen

4. **`build_insider_activity_factors()`** (Zeile 482-577)
   - Loop über Symbole: `for symbol in result[group_col].unique():`
   - **Komplexe Aggregationen pro Symbol**
   - **Lösung:** `groupby().apply()` verwenden statt Loop

### 3.3 Rate-Limit-Handling

**Problem:** Inkonsistentes Rate-Limit-Handling:

1. **`FinnhubPriceDataSource`** (Zeile 468-525)
   - **KEINE Delays** zwischen Requests
   - **Problem:** Kann Rate-Limits überschreiten (60/min)
   - **Lösung:** `time.sleep(1.0)` zwischen Requests hinzufügen

2. **`TwelveDataPriceDataSource`** (Zeile 656-734)
   - **KEINE Delays** zwischen Requests
   - **Problem:** Kann Rate-Limits überschreiten (8/min = 7.5s zwischen Requests)
   - **Lösung:** `time.sleep(7.5)` zwischen Requests hinzufügen

3. **`fetch_insider_events()`** (Zeile 269-372)
   - **Delays vorhanden:** `time.sleep(RATE_LIMIT_DELAY_SECONDS)` (Zeile 368)
   - **Rate-Limit-Error-Handling:** 60s Wait bei Rate-Limit (Zeile 283-284)
   - **Status:** ✅ OK

4. **`fetch_news()`** (Zeile 113-188)
   - **Delays vorhanden:** `time.sleep(RATE_LIMIT_DELAY_SECONDS)` (Zeile 184)
   - **Rate-Limit-Error-Handling:** 60s Wait bei Rate-Limit (Zeile 128-129)
   - **Status:** ✅ OK

---

## 4. Dopplungen und Inkonsistenzen

### 4.1 Doppelte `_get_finnhub_session()` Funktionen

**Problem:** Zwei identische Funktionen in verschiedenen Modulen:

1. **`src/assembled_core/data/altdata/finnhub_events.py`** (Zeile 36-71)
2. **`src/assembled_core/data/altdata/finnhub_news_macro.py`** (Zeile 37-74)

**Lösung:** In gemeinsames Modul verschieben (z.B. `src/assembled_core/data/altdata/finnhub_common.py`)

### 4.2 Inkonsistente Error-Handling

**Problem:** Unterschiedliche Error-Handling-Strategien:

1. **YahooPriceDataSource:** Loggt Warnung, fährt fort mit nächstem Symbol
2. **FinnhubPriceDataSource:** Loggt Warnung, fährt fort mit nächstem Symbol
3. **TwelveDataPriceDataSource:** Loggt Warnung, fährt fort mit nächstem Symbol
4. **Alt-Data Clients:** Loggt Warnung, gibt leeres DataFrame zurück

**Status:** ✅ Konsistent (alle fahren fort, keine Crashes)

### 4.3 Inkonsistente Rate-Limit-Delays

**Problem:** Unterschiedliche Delay-Werte:

1. **Finnhub Events:** `RATE_LIMIT_DELAY_SECONDS = 1.0` (Zeile 33)
2. **Finnhub News/Macro:** `RATE_LIMIT_DELAY_SECONDS = 1.0` (Zeile 34)
3. **Alpha Vantage:** `base_sleep = 13.0` (in `pull_intraday_av.py`)
4. **Twelve Data:** Keine Delays (sollte 7.5s sein)

**Lösung:** Zentrale Konfiguration für Rate-Limits

---

## 5. Kritische Baustellen

### 5.1 🚨 KRITISCH: Symbol-für-Symbol Loops in Preis-DataSources

**Betroffen:**
- `YahooPriceDataSource.get_history()` (Zeile 299)
- `FinnhubPriceDataSource.get_history()` (Zeile 468)
- `TwelveDataPriceDataSource.get_history()` (Zeile 656)

**Problem:**
- Sehr langsam bei großen Universen (100+ Symbole)
- Kann Stunden dauern
- Rate-Limits werden oft überschreiten

**Lösung:**
1. **Yahoo:** `yf.download(symbols)` verwenden (Batch-Download)
2. **Finnhub/Twelve Data:** Delays zwischen Requests hinzufügen
3. **Empfehlung:** Immer `data_source="local"` verwenden, Preise nur über Download-Skripte aktualisieren

### 5.2 ⚠️ WICHTIG: Ineffiziente DataFrame-Merges

**Betroffen:**
- `build_news_sentiment_factors()` (Zeile 241-307)
- `build_macro_regime_factors()` (Zeile 569-590)
- `build_earnings_surprise_factors()` (Zeile 206-256)
- `build_insider_activity_factors()` (Zeile 482-577)

**Problem:**
- Loop über Symbole mit vielen kleinen Merges
- `pd.concat()` wird oft aufgerufen
- Langsam bei großen Universen

**Lösung:**
- `groupby().apply()` verwenden statt Loops
- Oder: Alle Symbole auf einmal mergen (mit MultiIndex oder groupby)

### 5.3 ⚠️ WICHTIG: Fehlende Rate-Limit-Delays

**Betroffen:**
- `FinnhubPriceDataSource.get_history()` (Zeile 468)
- `TwelveDataPriceDataSource.get_history()` (Zeile 656)

**Problem:**
- Keine Delays zwischen Requests
- Rate-Limits werden überschreiten

**Lösung:**
- `time.sleep()` zwischen Requests hinzufügen
- Finnhub: 1.0s Delay
- Twelve Data: 7.5s Delay (8/min = 7.5s)

### 5.4 ⚠️ MITTEL: Doppelte `_get_finnhub_session()` Funktionen

**Betroffen:**
- `finnhub_events.py` und `finnhub_news_macro.py`

**Problem:**
- Code-Duplikation
- Wartungsaufwand

**Lösung:**
- In gemeinsames Modul verschieben (`finnhub_common.py`)

### 5.5 ⚠️ MITTEL: Ineffiziente Trend-Berechnung

**Betroffen:**
- `build_news_sentiment_factors()` (Zeile 166-179, 221-234)

**Problem:**
- `compute_trend()` Funktion macht Loop über alle Indizes
- `np.polyfit()` wird für jedes Fenster aufgerufen
- Langsam bei langen Zeitreihen

**Lösung:**
- Vectorisierte Berechnung verwenden (z.B. `np.polyfit` auf gesamte Serie anwenden)

---

## 6. Empfehlungen

### 6.1 Sofortige Maßnahmen (Kritisch)

1. **Alle Preis-DataSources:** Immer `data_source="local"` verwenden
   - Preise nur über Download-Skripte aktualisieren
   - Keine API-Calls in Backtests oder Factor-Analysis-Workflows

2. **Rate-Limit-Delays hinzufügen:**
   - `FinnhubPriceDataSource`: `time.sleep(1.0)` zwischen Requests
   - `TwelveDataPriceDataSource`: `time.sleep(7.5)` zwischen Requests

3. **YahooPriceDataSource optimieren:**
   - `yf.download(symbols)` verwenden statt Loop

### 6.2 Mittelfristige Maßnahmen (Wichtig)

1. **DataFrame-Merges optimieren:**
   - `groupby().apply()` verwenden statt Loops
   - Alle Symbole auf einmal mergen

2. **Code-Duplikation reduzieren:**
   - `_get_finnhub_session()` in gemeinsames Modul verschieben

3. **Trend-Berechnung optimieren:**
   - Vectorisierte Berechnung verwenden

### 6.3 Langfristige Maßnahmen (Nice-to-Have)

1. **Batch-APIs nutzen:**
   - Prüfen, ob Finnhub Batch-APIs für Insider/News hat
   - Parallele Requests mit Rate-Limit-Respektierung

2. **Caching implementieren:**
   - API-Responses cachen (z.B. für Macro-Daten)
   - TTL-basiertes Caching

3. **Zentrale Rate-Limit-Konfiguration:**
   - Alle Rate-Limits in einer Konfigurationsdatei
   - Automatisches Delay-Management

---

## 7. Zusammenfassung

### ✅ Was gut funktioniert:
- **LocalParquetPriceDataSource:** Sehr schnell, keine API-Calls
- **Alt-Data Download-Skripte:** Rate-Limits werden respektiert
- **Error-Handling:** Robust, keine Crashes

### ⚠️ Was verbessert werden muss:
- **Symbol-für-Symbol Loops:** Sehr langsam bei großen Universen
- **Ineffiziente DataFrame-Merges:** Viele kleine Merges statt einem großen
- **Fehlende Rate-Limit-Delays:** In Preis-DataSources
- **Code-Duplikation:** `_get_finnhub_session()` doppelt vorhanden

### 🚨 Kritische Punkte:
1. **Preis-APIs in Workflows:** Sollten **NIE** verwendet werden, nur `data_source="local"`
2. **Rate-Limit-Überschreitungen:** Können zu API-Blockierungen führen
3. **Performance bei großen Universen:** Kann Stunden dauern

---

## 8. Nächste Schritte

1. **Sofort:** Rate-Limit-Delays zu Preis-DataSources hinzufügen
2. **Sofort:** YahooPriceDataSource optimieren (Batch-Download)
3. **Kurzfristig:** DataFrame-Merges optimieren
4. **Mittelfristig:** Code-Duplikation reduzieren
5. **Langfristig:** Batch-APIs und Caching implementieren

