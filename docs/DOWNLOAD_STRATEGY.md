# Download-Strategie für Alt-Daten-Snapshot

## Problem: Yahoo Finance Rate-Limits

Yahoo Finance hat aggressive Rate-Limits:
- **Nach zu vielen Requests**: Alle Downloads werden blockiert
- **Timeout**: Rate-Limits können mehrere Stunden dauern, bis sie zurückgesetzt werden
- **Empfehlung**: Maximal 1 Request pro Minute

## Lösung: Strategischer Download-Plan

### Option 1: Einzelne Downloads mit langen Pausen (EMPFOHLEN)

**1. Wartezeit nach Rate-Limit:**
```powershell
# Warte mindestens 2-3 Stunden, bevor du erneut versuchst
# Lass das System das Rate-Limit zurücksetzen
```

**2. Einzelne Downloads (1 Symbol pro 60 Sekunden):**

```powershell
# PowerShell-Skript mit sehr langen Pausen
.\.venv\Scripts\python.exe scripts\download_historical_snapshot.py `
  --symbol NVDA `
  --start 2000-01-01 `
  --end 2025-12-03 `
  --interval 1d `
  --target-root "F:\Python_Projekt\Aktiengerüst\datensammlungen\altdaten\stand 3-12-2025"

# Dann 60 Sekunden warten, bevor das nächste Symbol
Start-Sleep -Seconds 60

.\.venv\Scripts\python.exe scripts\download_historical_snapshot.py `
  --symbol AAPL `
  --start 2000-01-01 `
  --end 2025-12-03 `
  --interval 1d `
  --target-root "F:\Python_Projekt\Aktiengerüst\datensammlungen\altdaten\stand 3-12-2025"
```

**3. Automatisiertes Einzeldownload-Skript:**

```powershell
# Nutze das robuste Einzeldownload-Skript
# Es wartet automatisch 5 Minuten vor dem Start und 60 Sekunden zwischen Downloads
.\.venv\Scripts\pwsh.exe scripts\download_one_by_one_robust.ps1 `
  -DelayBetweenSymbols 60 `
  -InitialWait 300 `
  -SkipExisting $true
```

### Option 2: Über mehrere Tage verteilen

**Plan:**
- **Tag 1**: Download von 10-15 Symbolen (mit 60s Pause dazwischen)
- **Tag 2**: Weitere 10-15 Symbole
- **Tag 3**: Restliche Symbole

**Beispiel:**
```powershell
# Tag 1: Nur AI Tech Universe (24 Symbole, ca. 24 Minuten bei 60s Pause)
# Tag 2: Healthcare + Energy (11 Symbole)
# Tag 3: Defense + Consumer (14 Symbole)
```

### Option 3: Alternative Datenquellen

**Wenn Yahoo Finance zu restriktiv ist:**
1. **Alpha Vantage**: Kostenlose API, 5 Requests/Minute, 500 Requests/Tag
2. **Polygon.io**: Kostenlose API, 5 Requests/Minute
3. **Quandl/Nasdaq Data Link**: Teilweise kostenlos
4. **Kauf-Datenquellen**: Polygon, IEX Cloud, etc.

## Empfohlene Workflow

### Schritt 1: Warte auf Rate-Limit-Reset

Nach dem Rate-Limit-Fehler:
```powershell
# Warte mindestens 3 Stunden (am besten über Nacht)
# Dann starte den Download erneut
```

### Schritt 2: Starte mit dem robusten Skript

```powershell
# Das Skript:
# - Wartet 5 Minuten vor dem Start (damit Rate-Limit zurückgesetzt wird)
# - Lädt ein Symbol nach dem anderen
# - Wartet 60 Sekunden zwischen jedem Download
# - Überspringt bereits existierende Dateien
.\.venv\Scripts\pwsh.exe scripts\download_one_by_one_robust.ps1 `
  -DelayBetweenSymbols 60 `
  -InitialWait 300
```

**Erwartete Dauer:**
- 49 Symbole × 60 Sekunden = ca. 49 Minuten reine Download-Zeit
- Plus initiale 5 Minuten Wartezeit
- **Gesamt: ca. 1 Stunde**

### Schritt 3: Fortschritt überprüfen

```powershell
# Prüfe, welche Dateien bereits existieren
Get-ChildItem "F:\Python_Projekt\Aktiengerüst\datensammlungen\altdaten\stand 3-12-2025\1d\*.parquet" | Select-Object Name

# Validiere heruntergeladene Dateien
.\.venv\Scripts\python.exe scripts\validate_altdata_snapshot.py `
  --root "F:\Python_Projekt\Aktiengerüst\datensammlungen\altdaten\stand 3-12-2025" `
  --interval 1d
```

### Schritt 4: Fehlgeschlagene Downloads wiederholen

Wenn einige Downloads fehlgeschlagen sind:
```powershell
# Das Skript überspringt automatisch bereits existierende Dateien
# Führe es einfach erneut aus, um fehlgeschlagene Symbole zu retry
.\.venv\Scripts\pwsh.exe scripts\download_one_by_one_robust.ps1 `
  -DelayBetweenSymbols 60 `
  -InitialWait 0  # Keine initiale Wartezeit beim Retry
```

## Notfall-Strategie: Wenn Rate-Limits zu stark sind

### Alternative 1: Daten manuell kaufen/erhalten

- Historische Daten von kommerziellen Anbietern kaufen
- Daten von anderen Quellen beziehen (CSV, Excel, etc.)
- Datenbanken mit historischen Finanzdaten nutzen

### Alternative 2: Stückweise Download über Wochen

- Jeden Tag nur 5-10 Symbole downloaden
- Über 1-2 Wochen verteilen
- Langfristig nachhaltig

### Alternative 3: Proxy/VPN verwenden

**⚠️ Achtung:** Kann gegen Yahoo Finance Terms of Service verstoßen!
- Verschiedene IP-Adressen verwenden
- Nicht empfohlen, kann zu Account-Sperrung führen

## Aktuelle Situation

**Status:** Alle Downloads werden aktuell von Yahoo Finance blockiert (Rate-Limit).

**Empfohlene nächste Schritte:**
1. **Warte 3-4 Stunden** (am besten über Nacht)
2. **Starte das robuste Einzeldownload-Skript** mit 60 Sekunden Pause
3. **Lass es über Nacht laufen** - das Skript kann mehrere Stunden brauchen
4. **Prüfe am nächsten Morgen** die Ergebnisse

## Verfügbare Skripte

1. **`scripts/download_one_by_one_robust.ps1`** ⭐ EMPFOHLEN
   - Download eines Symbols nach dem anderen
   - Lange Pausen (60s+ zwischen Downloads)
   - Automatisches Überspringen bereits existierender Dateien
   - Robuste Fehlerbehandlung

2. **`scripts/download_all_universes_batch.ps1`**
   - Lädt komplette Universen
   - Für den Fall, dass Rate-Limits zurückgesetzt sind

3. **`scripts/download_historical_snapshot.py`**
   - Basis-Download-Skript
   - Unterstützt einzelne Symbole, mehrere Symbole, oder Dateien

## Checkliste vor dem Download

- [ ] Rate-Limit-Timeout abgewartet (min. 3 Stunden)
- [ ] Internetverbindung stabil
- [ ] Zielordner existiert und ist beschreibbar
- [ ] Python-Umgebung aktiviert (`.venv\Scripts\activate`)
- [ ] Alle Abhängigkeiten installiert (`pip install -r requirements.txt`)
- [ ] Test-Download mit einem Symbol erfolgreich

## Nach dem Download

1. **Validiere alle Dateien:**
   ```powershell
   .\.venv\Scripts\python.exe scripts\validate_altdata_snapshot.py `
     --root "F:\Python_Projekt\Aktiengerüst\datensammlungen\altdaten\stand 3-12-2025" `
     --interval 1d
   ```

2. **Prüfe fehlgeschlagene Symbole:**
   - Liste fehlgeschlagene Symbole in Logs
   - Retry fehlgeschlagene Downloads

3. **Backtest-Tests:**
   - Teste Backtests mit heruntergeladenen Daten
   - Stelle sicher, dass `LocalParquetPriceDataSource` die Dateien findet

