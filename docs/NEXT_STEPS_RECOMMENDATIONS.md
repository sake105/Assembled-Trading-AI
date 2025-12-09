# Nächste Schritte - Empfehlungen

**Datum:** 2025-12-09  
**Status:** Downloads laufen, 22/24 AI-Tech Symbole erfolgreich

---

## ✅ Was bereits erledigt ist

1. **Twelve Data Integration**
   - ✅ `TwelveDataPriceDataSource` implementiert
   - ✅ Download-Skript erweitert (`--provider twelve_data`)
   - ✅ Settings erweitert (`twelve_data_api_key`)
   - ✅ Factory-Funktion erweitert

2. **Downloads**
   - ✅ Macro-ETFs: 10/10 vollständig
   - ✅ AI-Tech: 22/24 (92%)
   - ⏳ Andere Universen: Downloads laufen

3. **Tools erstellt**
   - ✅ `check_data_completeness.py` - Vollständigkeitsprüfung
   - ✅ `test_problem_symbols.ps1` - Problem-Symbole testen
   - ✅ `setup_pipeline_integration.ps1` - Pipeline-Integration vorbereiten

---

## 🔄 Aktuelle Aufgaben (laufend)

### 1. Downloads abschließen
**Status:** Läuft im Hintergrund

**Aktionen:**
- Warten auf Abschluss der laufenden Downloads
- Fehlende Symbole identifizieren
- Problem-Symbole dokumentieren

**Erwartete Zeit:** ~15-20 Minuten für alle Universen

---

## 📋 Nächste Schritte (Priorität)

### 1. Vollständigkeitsprüfung (Hoch)
**Ziel:** Alle heruntergeladenen Daten validieren

```powershell
# Alle Universen prüfen
.\.venv\Scripts\python.exe scripts/check_data_completeness.py `
  --all-universes `
  --target-root "F:\Python_Projekt\Aktiengerüst\datensammlungen\altdaten\stand 3-12-2025" `
  --interval 1d `
  --expected-start 2000-01-01 `
  --expected-end 2025-12-03
```

**Erwartetes Ergebnis:**
- Liste aller fehlenden Symbole
- Qualitätsbericht (Zeilen, Datumsbereiche, Spalten)
- Identifikation von Problem-Symbolen

---

### 2. Problem-Symbole dokumentieren (Mittel)
**Ziel:** Nicht verfügbare Symbole identifizieren und dokumentieren

**Bekannte Problem-Symbole:**
- `IOS.DE` - Twelve Data: "symbol invalid"
- `SMHN.DE` - Twelve Data: "symbol invalid"
- `BAVA.CO` - Noch zu testen
- `EUZ.DE` - Noch zu testen

**Aktionen:**
1. Alle fehlenden Symbole einzeln testen
2. Alternative Ticker-Formate prüfen (z.B. "IOS" statt "IOS.DE")
3. Dokumentieren, welche Symbole nicht verfügbar sind
4. Entscheidung: Aus Universe entfernen oder später mit anderem Provider nachladen

---

### 3. Pipeline-Integration (Hoch)
**Ziel:** Heruntergeladene Daten in die Pipeline integrieren

```powershell
# Environment-Variable setzen
$env:ASSEMBLED_LOCAL_DATA_ROOT = "F:\Python_Projekt\Aktiengerüst\datensammlungen\altdaten\stand 3-12-2025"

# Oder Setup-Skript nutzen
.\scripts\setup_pipeline_integration.ps1
```

**Tests:**
1. **Datenladen testen:**
   ```python
   from src.assembled_core.data.data_source import get_price_data_source
   from src.assembled_core.config.settings import Settings
   
   settings = Settings()
   settings.local_data_root = Path("F:/Python_Projekt/Aktiengerüst/datensammlungen/altdaten/stand 3-12-2025")
   ds = get_price_data_source(settings, "local")
   df = ds.get_history(["SPY"], "2010-01-01", "2025-12-03", "1d")
   print(f"Loaded {len(df)} rows for SPY")
   ```

2. **Backtest testen:**
   ```powershell
   python scripts/cli.py backtest `
     --freq 1d `
     --symbols-file config/macro_world_etfs_tickers.txt `
     --start-date 2010-01-01 `
     --end-date 2025-12-03
   ```

3. **Factor-Report testen:**
   ```powershell
   python scripts/cli.py factor_report `
     --freq 1d `
     --symbols-file config/macro_world_etfs_tickers.txt `
     --start-date 2010-01-01 `
     --end-date 2025-12-03 `
     --factor-set core `
     --fwd-horizon-days 5
   ```

---

### 4. Datenqualität optimieren (Mittel)
**Ziel:** Zeitraum auf 2000-2025 erweitern (aktuell: 2010-2025)

**Problem:** Twelve Data Free-Tier limitiert auf 5000 Zeilen pro Request

**Lösungen:**
1. **Mehrere Requests:** Zeitraum in Chunks aufteilen (z.B. 2000-2010, 2010-2020, 2020-2025)
2. **Starter-Plan:** $9.99/Monat für mehr Daten
3. **Aktueller Stand akzeptieren:** 2010-2025 ist für viele Analysen ausreichend

**Empfehlung:** Erstmal mit 2010-2025 arbeiten, später auf Starter-Plan upgraden wenn nötig

---

### 5. Dokumentation aktualisieren (Niedrig)
**Ziel:** Download-Workflow und Provider-Strategie dokumentieren

**Aktionen:**
1. `docs/DATA_DOWNLOAD_STATUS.md` aktualisieren (bereits erstellt)
2. `README.md` erweitern mit Download-Anleitung
3. Provider-Vergleich dokumentieren
4. Problem-Symbole-Liste pflegen

---

## 🎯 Empfohlene Reihenfolge

### Sofort (heute):
1. ✅ Downloads abschließen lassen
2. ✅ Vollständigkeitsprüfung durchführen
3. ✅ Problem-Symbole dokumentieren

### Diese Woche:
1. Pipeline-Integration testen
2. Backtest mit neuen Daten ausführen
3. Factor-Report mit Phase A Faktoren testen

### Nächste Woche:
1. Datenqualität optimieren (Zeitraum erweitern)
2. Finnhub für Alt-Daten vorbereiten
3. Dokumentation finalisieren

---

## 📊 Erwartete Ergebnisse

### Nach Vollständigkeitsprüfung:
- **~50-55/59 Symbole** erfolgreich (85-93%)
- **~4-9 Problem-Symbole** (meist europäische Ticker)
- **Qualitätsbericht** für alle Universen

### Nach Pipeline-Integration:
- Backtest läuft mit neuen Daten
- Factor-Report funktioniert
- Alle Phase A Faktoren können berechnet werden

---

## ⚠️ Bekannte Probleme

1. **Twelve Data Free-Tier Limits:**
   - 8 Calls/Minute → 8 Sekunden Pause zwischen Calls
   - 800 Calls/Tag → ~100 Symbole pro Tag möglich
   - 5000 Zeilen max → Zeitraum 2010-2025 statt 2000-2025

2. **Europäische Ticker:**
   - Viele .DE, .CO, .AX Ticker nicht bei Twelve Data verfügbar
   - Lösung: Später mit Finnhub oder anderem Provider nachladen

3. **Rate-Limits:**
   - Bei zu schnellen Downloads: "API credits exhausted"
   - Lösung: Längere Pausen (8+ Sekunden) zwischen Calls

---

## 💡 Tipps

1. **Downloads überwachen:**
   - Log-Dateien in `logs/` prüfen
   - Status regelmäßig mit `check_data_completeness.py` prüfen

2. **Problem-Symbole:**
   - Nicht verfügbare Symbole aus Universe entfernen oder markieren
   - Später mit alternativem Provider nachladen

3. **Pipeline-Tests:**
   - Zuerst mit kleinen Universen testen (z.B. Macro-ETFs)
   - Dann auf größere Universen erweitern

