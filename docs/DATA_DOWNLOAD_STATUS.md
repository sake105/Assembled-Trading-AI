# Alt-Daten Download Status

**Letzte Aktualisierung:** 2025-12-09

## Provider-Strategie

### Primär: Twelve Data
- **Status:** ✅ Implementiert und aktiv
- **API-Key:** Free-Tier (8 Calls/Min, 800 Calls/Tag)
- **Empfehlung:** Starter-Plan ($9.99/Monat) für größere Universen
- **Verwendung:** Primärer Provider für EOD-Preise (20-25 Jahre Historie)

### Sekundär: Finnhub
- **Status:** ✅ Implementiert (API-Key vorhanden, aber 403 bei candle-API)
- **Verwendung:** Geplant für Alt-Daten (Insider, Congress, News) - später
- **Hinweis:** Free-Tier hat möglicherweise keine candle-API-Zugriff

### Fallback: Yahoo Finance
- **Status:** ⚠️ Unzuverlässig (Rate-Limits)
- **Verwendung:** Nur noch als Notfall-Fallback für einzelne Symbole

---

## Download-Status nach Universen

### ✅ Macro World ETFs (10/10 - 100%)
- **Status:** Vollständig
- **Zeitraum:** 2010-01-04 bis 2025-12-02 (~4004 Zeilen pro Symbol)
- **Hinweis:** Startet bei 2010 (Twelve Data Free-Tier Limit: 5000 Zeilen max)

### ✅ AI Tech Universe (22/24 - 92%)
- **Status:** Fast vollständig
- **Fehlende Symbole:**
  - `IOS.DE` - Deutscher Ticker, möglicherweise nicht bei Twelve Data verfügbar
  - `SMHN.DE` - Deutscher Ticker, möglicherweise nicht bei Twelve Data verfügbar
- **Hinweis:** Beide sind deutsche Ticker (.DE), möglicherweise Format-Problem

### ⏳ Healthcare Biotech (2/4 - 50%)
- **Status:** In Bearbeitung
- **Vorhanden:** SRT3.DE, VRNA
- **Fehlend:** BAVA.CO, EUZ.DE
- **Hinweis:** BAVA.CO (Dänemark) und EUZ.DE (Deutschland) - möglicherweise nicht verfügbar

### ⏳ Energy Resources Cyclicals (0/7 - 0%)
- **Status:** Download läuft

### ⏳ Defense Security Aero (0/11 - 0%)
- **Status:** Download läuft

### ⏳ Consumer Financial Misc (0/3 - 0%)
- **Status:** Download läuft

---

## Problem-Symbole

### Bekannte Probleme

1. **Deutsche Ticker (.DE)**
   - IOS.DE, SMHN.DE, EUZ.DE, BAVA.CO
   - **Mögliche Ursachen:**
     - Twelve Data unterstützt möglicherweise nicht alle europäischen Ticker
     - Ticker-Format könnte anders sein (z.B. "IOS" statt "IOS.DE")
     - Ticker existiert nicht mehr oder wurde umbenannt

2. **Lösungsansätze:**
   - Einzelne Tests mit verschiedenen Ticker-Formaten
   - Alternative Provider für europäische Ticker prüfen
   - Ticker-Validierung im Finnhub-Dashboard

---

## Nächste Schritte

### 1. Problem-Symbole testen
```powershell
.\scripts\test_problem_symbols.ps1 -Symbols @("IOS.DE", "SMHN.DE", "BAVA.CO", "EUZ.DE")
```

### 2. Vollständigkeitsprüfung
```powershell
.\.venv\Scripts\python.exe scripts/check_data_completeness.py --all-universes --target-root "F:\Python_Projekt\Aktiengerüst\datensammlungen\altdaten\stand 3-12-2025" --interval 1d
```

### 3. Pipeline-Integration
- `ASSEMBLED_LOCAL_DATA_ROOT` auf Alt-Daten-Ordner setzen
- Backtest mit neuen Daten testen
- Factor-Report ausführen

### 4. Dokumentation
- Download-Workflow dokumentieren
- Problem-Symbole dokumentieren
- Provider-Strategie finalisieren

---

## Technische Details

### Download-Konfiguration
- **Provider:** Twelve Data
- **Zeitraum:** 2000-01-01 bis 2025-12-03
- **Interval:** 1d (Daily)
- **Rate-Limit:** 8 Sekunden zwischen Calls (8 Calls/Min)
- **Output:** `<target-root>/1d/<SYMBOL>.parquet`

### Datenformat
- **Spalten:** timestamp (UTC), symbol, open, high, low, close, volume
- **Format:** Parquet
- **Sortierung:** Nach timestamp, dann symbol

### Qualitätsprüfung
- Dateigröße: Mindestens 1KB
- Zeilenanzahl: ~4000-6500 (abhängig von Startdatum)
- Spalten: timestamp, symbol, close (minimal), open, high, low, volume (optional)

