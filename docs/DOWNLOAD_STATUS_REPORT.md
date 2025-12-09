# Alt-Daten Download Status Report

**Datum:** 2025-12-04  
**Ziel:** Download historischer Daten für alle Universen (2000-01-01 bis 2025-12-03)

## Status

### Erfolgreich heruntergeladen: 3 von 59 Symbolen (5%)

- ✓ **NVDA** (311.8 KB) - AI Tech Universe
- ✓ **SRT3.DE** (277.9 KB) - Healthcare Biotech Universe  
- ✓ **VRNA** (92.3 KB) - Healthcare Biotech Universe

### Fehlgeschlagen/Fehlend: 56 von 59 Symbolen (95%)

**Problem:** Yahoo Finance Rate-Limits blockieren die meisten Downloads sehr aggressiv.

## Strategie

Wir verwenden einen **One-by-One Safe Mode** Ansatz:

1. **Einzelne Downloads:** Jeder Ticker wird einzeln geladen (nicht batch)
2. **Lange Delays:** 10 Sekunden Pause zwischen Downloads
3. **Rate-Limit-Handling:** Bei Rate-Limit → 10 Minuten Wartezeit
4. **Skip Existing:** Bereits vorhandene Dateien werden übersprungen

## Verfügbare Skripte

### 1. Einzelnes Universe
```powershell
.\scripts\download_one_by_one_safe.ps1 `
  -UniverseFile "config\healthcare_biotech_tickers.txt" `
  -DelayBetweenSymbols 10 `
  -InitialWait 60
```

### 2. Alle Universen
```powershell
.\scripts\download_all_universes_safe.ps1 `
  -DelayBetweenSymbols 10 `
  -InitialWait 60
```

## Empfehlungen

### Kurzfristig (heute)
1. **Warten:** 1-2 Stunden Pause, dann erneut versuchen
2. **Manuell:** Einzelne problematische Ticker manuell testen
3. **Kleineres Datum:** Falls möglich, kürzeren Zeitraum testen (z.B. 2020-2025)

### Mittelfristig (diese Woche)
1. **Stückweise:** Täglich 5-10 Symbole herunterladen
2. **Zeitversetzt:** Downloads über mehrere Tage verteilen
3. **Alternative Quellen:** Prüfen, ob andere Datenquellen verfügbar sind

### Langfristig
1. **Alternative APIs:** Alpha Vantage, Polygon.io, etc.
2. **Kaufdaten:** Professionelle Datenanbieter (Bloomberg, Refinitiv)
3. **Caching:** Lokale Datenbank für bereits geladene Daten

## Nächste Schritte

1. **Warten:** 1-2 Stunden, dann erneut versuchen
2. **Retry:** Fehlgeschlagene Symbole erneut versuchen
3. **Monitoring:** Logs prüfen (`logs/download_one_by_one_*.log`)

## Logs

Alle Downloads werden in `logs/download_one_by_one_*.log` protokolliert.

