# Integrationsplan: Alternative Datenanbieter

## Ziel

Yahoo Finance durch zuverlässigere APIs ersetzen oder ergänzen, um:
- Rate-Limit-Probleme zu vermeiden
- Reproduzierbare Downloads zu ermöglichen
- Langfristig stabile Datenpipeline aufzubauen

## Architektur-Plan

### 1. Erweiterung der PriceDataSource-Abstraktion

**Aktueller Stand:**
- `PriceDataSource` (Protocol)
- `LocalParquetPriceDataSource` (lokale Dateien)
- `YahooPriceDataSource` (Yahoo Finance)

**Geplante Erweiterung:**
- `AlphaVantagePriceDataSource`
- `TwelveDataPriceDataSource`
- `TiingoPriceDataSource`
- (Optional: weitere Provider)

### 2. Settings-Erweiterung

**Neue Settings in `config/settings.py`:**

```python
class Settings(BaseSettings):
    # ... existing settings ...
    
    # Data provider configuration
    data_provider: str = "yahoo"  # "yahoo" | "alpha_vantage" | "twelve_data" | "tiingo"
    
    # API Keys (optional, nur wenn Provider benötigt)
    alpha_vantage_api_key: str | None = None
    twelve_data_api_key: str | None = None
    tiingo_api_key: str | None = None
    
    class Config:
        env_prefix = "ASSEMBLED_"
```

**Environment Variables:**
```bash
ASSEMBLED_DATA_PROVIDER=twelve_data
ASSEMBLED_TWELVE_DATA_API_KEY=your_api_key_here
```

### 3. Factory-Funktion erweitern

**In `src/assembled_core/data/data_source.py`:**

```python
def get_price_data_source(settings: Settings) -> PriceDataSource:
    """Factory function to create appropriate PriceDataSource."""
    if settings.data_source == "local":
        return LocalParquetPriceDataSource(settings)
    
    # Online providers
    provider = settings.data_provider or "yahoo"
    
    if provider == "yahoo":
        return YahooPriceDataSource(settings)
    elif provider == "alpha_vantage":
        if not settings.alpha_vantage_api_key:
            raise ValueError("alpha_vantage_api_key required for Alpha Vantage provider")
        return AlphaVantagePriceDataSource(settings)
    elif provider == "twelve_data":
        if not settings.twelve_data_api_key:
            raise ValueError("twelve_data_api_key required for Twelve Data provider")
        return TwelveDataPriceDataSource(settings)
    elif provider == "tiingo":
        if not settings.tiingo_api_key:
            raise ValueError("tiingo_api_key required for Tiingo provider")
        return TiingoPriceDataSource(settings)
    else:
        raise ValueError(f"Unknown data provider: {provider}")
```

### 4. Neue Provider-Module

**Struktur:**
```
src/assembled_core/data/
  ├── data_source.py (Protocol, Factory)
  ├── local_data_source.py
  ├── yahoo_data_source.py
  ├── alpha_vantage_data_source.py (NEU)
  ├── twelve_data_data_source.py (NEU)
  └── tiingo_data_source.py (NEU)
```

**Gemeinsame Struktur für alle Provider:**

```python
class TwelveDataPriceDataSource(PriceDataSource):
    """Twelve Data API implementation of PriceDataSource."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.api_key = settings.twelve_data_api_key
        if not self.api_key:
            raise ValueError("twelve_data_api_key required")
    
    def get_history(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
        freq: str = "1d",
    ) -> pd.DataFrame:
        """
        Download historical data from Twelve Data API.
        
        Returns DataFrame in standard format:
        - timestamp (UTC)
        - symbol
        - open, high, low, close, adj_close, volume
        """
        # API-Call zu Twelve Data
        # Normalisierung zu unserem Format
        # ...
```

### 5. Download-Skript erweitern

**In `scripts/download_historical_snapshot.py`:**

```python
parser.add_argument(
    "--provider",
    choices=["yahoo", "alpha_vantage", "twelve_data", "tiingo"],
    default="yahoo",
    help="Data provider to use (default: yahoo)"
)

# In main():
if args.provider != "yahoo":
    # Load API key from settings or env
    settings = Settings()
    provider_key = {
        "alpha_vantage": settings.alpha_vantage_api_key,
        "twelve_data": settings.twelve_data_api_key,
        "tiingo": settings.tiingo_api_key,
    }[args.provider]
    
    if not provider_key:
        raise ValueError(f"API key required for provider: {args.provider}")
```

## Implementierungsreihenfolge

### Phase 1: Twelve Data (Empfehlung für Start)

**Warum Twelve Data:**
- ✅ Kostenloser API-Key
- ✅ Gute Free-Tier-Limits (800 Calls/Tag, 8 Calls/Min)
- ✅ Sehr günstige bezahlte Pläne ($9.99/Monat)
- ✅ Stabile API

**Schritte:**
1. Registrierung bei Twelve Data → API-Key holen
2. `TwelveDataPriceDataSource` implementieren
3. `download_historical_snapshot.py` erweitern
4. Test mit Macro-ETFs (10 Symbole)
5. Vollständiger Download aller Universen

**Geschätzter Aufwand:** 2-3 Stunden

### Phase 2: Tiingo (Optional, wenn Budget vorhanden)

**Warum Tiingo:**
- ✅ Sehr umfangreiche Daten
- ✅ Sehr günstig ($10/Monat)
- ✅ Sehr stabil

**Schritte:**
1. Registrierung bei Tiingo → API-Key holen
2. `TiingoPriceDataSource` implementieren
3. Test und Vergleich mit Twelve Data

**Geschätzter Aufwand:** 1-2 Stunden

### Phase 3: Multi-Provider-Support (Optional)

**Ziel:** Automatischer Fallback (z.B. Twelve Data → Yahoo)

**Schritte:**
1. Fallback-Logik in Factory-Funktion
2. Retry mit alternativem Provider bei Fehlern
3. Dokumentation

**Geschätzter Aufwand:** 1-2 Stunden

## Nächste Schritte

### Sofort (heute)
1. ✅ Macro-ETFs mit verkürztem Zeitraum (2010-2025) testen
2. ✅ Vergleichsliste der APIs erstellen

### Diese Woche
1. **Twelve Data API-Key registrieren** (kostenlos)
2. **TwelveDataPriceDataSource implementieren**
3. **Download-Skript erweitern**
4. **Test mit Macro-ETFs**

### Nächste Woche
1. Vollständiger Download aller Universen mit Twelve Data
2. Vergleich: Twelve Data vs. Yahoo (Qualität, Geschwindigkeit)
3. Entscheidung: Free-Tier vs. Paid-Tier

## Beispiel-Commands (nach Integration)

### Mit Twelve Data (Free-Tier)
```powershell
$env:ASSEMBLED_DATA_PROVIDER = "twelve_data"
$env:ASSEMBLED_TWELVE_DATA_API_KEY = "your_api_key_here"

python scripts/download_historical_snapshot.py `
  --symbols-file config/macro_world_etfs_tickers.txt `
  --start 2000-01-01 `
  --end 2025-12-03 `
  --interval 1d `
  --provider twelve_data `
  --target-root "F:\Python_Projekt\Aktiengerüst\datensammlungen\altdaten\stand 3-12-2025" `
  --sleep-seconds 8  # 8 Calls/Min Limit beachten
```

### Mit Tiingo (Paid)
```powershell
$env:ASSEMBLED_DATA_PROVIDER = "tiingo"
$env:ASSEMBLED_TIINGO_API_KEY = "your_api_key_here"

python scripts/download_historical_snapshot.py `
  --symbols-file config/universe_ai_tech_tickers.txt `
  --start 2000-01-01 `
  --end 2025-12-03 `
  --interval 1d `
  --provider tiingo `
  --target-root "F:\Python_Projekt\Aktiengerüst\datensammlungen\altdaten\stand 3-12-2025" `
  --sleep-seconds 6  # 10 Calls/Min Limit beachten
```

## Dependencies

**Neue Python-Packages (optional):**
- `requests` (bereits vorhanden)
- `httpx` (optional, für async)

**Keine neuen Dependencies nötig** - alle APIs nutzen REST, `requests` reicht aus.

