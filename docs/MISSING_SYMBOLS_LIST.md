# Fehlende Symbole - To-Do Liste

**Datum:** 2025-12-09  
**Status:** Markiert in Universen, müssen später hinzugefügt werden  
**Provider:** Twelve Data Free-Tier unterstützt diese nicht

---

## Übersicht

**Total fehlend:** 14 Symbole (alle europäische Ticker)

**Grund:** Twelve Data Free-Tier unterstützt nicht alle europäischen Ticker (.DE, .L, .CO, .AX)

**Lösung:** Später mit alternativem Provider hinzufügen (z.B. EODHD, Finnhub Paid-Tier, oder Alpha Vantage)

---

## Fehlende Symbole nach Universen

### AI Tech Universe (2 fehlend)
- `IOS.DE` - Deutsche Ticker
- `SMHN.DE` - Deutsche Ticker

### Healthcare Biotech (2 fehlend)
- `BAVA.CO` - Dänischer Ticker
- `EUZ.DE` - Deutsche Ticker

### Defense Security Aero (7 fehlend)
- `BA.L` - UK Ticker (London)
- `DRO.AX` - Australischer Ticker
- `HAG.DE` - Deutsche Ticker
- `R3NK.DE` - Deutsche Ticker
- `RHM.DE` - Deutsche Ticker
- `RR.L` - UK Ticker (London)
- `TKA.DE` - Deutsche Ticker

### Energy Resources Cyclicals (3 fehlend)
- `PAH3.DE` - Deutsche Ticker
- `VOW3.DE` - Deutsche Ticker
- `XDW0.DE` - Deutsche Ticker

---

## Markierung in Universen

Alle fehlenden Symbole sind in den Universe-Dateien markiert mit:
```
# MISSING: Not available in Twelve Data Free-Tier - TODO: Add later with alternative provider
SYMBOL
```

**Dateien:**
- `config/universe_ai_tech_tickers.txt`
- `config/healthcare_biotech_tickers.txt`
- `config/defense_security_aero_tickers.txt`
- `config/energy_resources_cyclicals_tickers.txt`

---

## Nächste Schritte

### Kurzfristig (aktuell):
- ✅ Symbole in Universen markiert
- ✅ Pipeline-Tests mit verfügbaren Symbolen durchführen
- ✅ Backtests und Factor-Reports ausführen

### Mittelfristig (später):
1. **Alternative Provider evaluieren:**
   - EODHD (spezialisiert auf europäische Märkte)
   - Finnhub Paid-Tier (falls candle-API verfügbar)
   - Alpha Vantage (unterstützt einige europäische Ticker)

2. **Download-Skript erweitern:**
   - Multi-Provider-Support
   - Automatischer Fallback für fehlende Symbole

3. **Symbole nachladen:**
   - Mit alternativem Provider
   - In gleiches Format konvertieren
   - Zu bestehenden Daten hinzufügen

---

## Priorität

**Niedrig:** Diese Symbole sind für die meisten Analysen nicht kritisch, da:
- 45/59 Symbole (76%) bereits verfügbar sind
- Alle wichtigen US-Ticker vorhanden sind
- ETFs vollständig sind

**Aber:** Für vollständige Universen-Analysen sollten sie später hinzugefügt werden.

---

## Technische Details

### Warum fehlen sie?
- Twelve Data Free-Tier hat limitierte Abdeckung für europäische Märkte
- Fehler: "symbol invalid" oder "symbol parameter is missing or invalid"

### Alternative Ticker-Formate getestet?
- Nein, aber könnte helfen (z.B. "IOS" statt "IOS.DE")
- **TODO:** Verschiedene Formate testen

### Können sie später hinzugefügt werden?
- Ja, ohne Probleme
- Einfach Parquet-Dateien in `<target-root>/1d/` hinzufügen
- Pipeline erkennt sie automatisch

---

**Status:** ✅ Dokumentiert und markiert - Bereit für spätere Integration

