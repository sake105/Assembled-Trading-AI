# Datenanbieter-Vergleich für Alt-Daten-Downloads

**Ziel:** Historische EOD-Daten (2000-2025) für ~60 Symbole zuverlässig herunterladen

## Vergleich: Kostenlose vs. Bezahlte APIs

### 1. Alpha Vantage ⭐ (Empfehlung für Start)

**Was bietet es:**
- EOD-Historie (Daily Adjusted)
- Intraday (1min, 5min, 15min, 30min, 60min)
- Fundamentals (optional)
- **Kostenloser API-Key** (nach Registrierung)

**Limits (Free Tier):**
- 5 API-Calls pro Minute
- 500 Calls pro Tag
- **→ Für 60 Symbole: ~1 Download pro Tag möglich (sehr langsam, aber stabil)**

**Kosten (Paid):**
- Premium: $49.99/Monat (75 Calls/Min, 1200 Calls/Tag)
- **→ Für 60 Symbole: ~1-2 Downloads pro Tag möglich**

**Technische Integration:**
- REST API, JSON Response
- Einfache Authentifizierung (API-Key als Query-Parameter)
- Gute Dokumentation

**Pro:**
- ✅ Kostenloser API-Key
- ✅ Stabile API
- ✅ Keine Rate-Limit-Probleme wie Yahoo
- ✅ Offizielle API (legal, keine Scraping-Probleme)

**Contra:**
- ❌ Sehr langsam im Free-Tier (5 Calls/Min)
- ❌ Für 60 Symbole braucht man ~12 Minuten (mit Pausen)

**Empfehlung:** Gut für kleine Universen oder als Backup-Provider

---

### 2. Twelve Data ⭐⭐ (Beste Balance)

**Was bietet es:**
- EOD-Historie
- Intraday (1min, 5min, etc.)
- Real-time Quotes
- **Kostenloser API-Key** (nach Registrierung)

**Limits (Free Tier):**
- 800 Calls/Tag
- 8 Calls/Minute
- **→ Für 60 Symbole: ~1 Download pro Tag möglich**

**Kosten (Paid):**
- Starter: $9.99/Monat (8000 Calls/Tag, 8 Calls/Min)
- Professional: $29.99/Monat (Unlimited Calls/Tag, 60 Calls/Min)
- **→ Starter reicht für 60 Symbole locker aus**

**Technische Integration:**
- REST API, JSON/CSV Response
- API-Key als Header oder Query-Parameter
- Sehr gute Dokumentation

**Pro:**
- ✅ Kostenloser API-Key
- ✅ Stabiler als Yahoo
- ✅ Gute Free-Tier-Limits
- ✅ Bezahlte Pläne sehr günstig

**Contra:**
- ❌ Im Free-Tier immer noch langsam (8 Calls/Min)

**Empfehlung:** **Beste Option für Start** - Free-Tier testen, dann ggf. auf Starter upgraden

---

### 3. Tiingo ⭐⭐⭐ (Professionell, günstig)

**Was bietet es:**
- EOD-Historie (sehr umfangreich)
- Intraday (1min, 5min, etc.)
- Fundamentals, News
- **Kostenloser API-Key** (nach Registrierung)

**Limits (Free Tier):**
- 1000 Calls/Tag
- 10 Calls/Minute
- **→ Für 60 Symbole: ~1 Download pro Tag möglich**

**Kosten (Paid):**
- Starter: $10/Monat (Unlimited Calls/Tag, 10 Calls/Min)
- Professional: $30/Monat (Unlimited Calls/Tag, 60 Calls/Min)
- **→ Starter reicht für 60 Symbole aus**

**Technische Integration:**
- REST API, JSON/CSV Response
- API-Key als Header
- Sehr gute Dokumentation

**Pro:**
- ✅ Sehr umfangreiche Daten
- ✅ Sehr günstige bezahlte Pläne
- ✅ Sehr stabil

**Contra:**
- ❌ Im Free-Tier immer noch langsam

**Empfehlung:** Sehr gut, wenn man bereit ist, $10/Monat zu investieren

---

### 4. Polygon.io ⭐⭐⭐⭐ (Professionell, teurer)

**Was bietet es:**
- EOD-Historie
- Intraday (1min, 5min, etc.)
- Real-time Data
- Options, Fundamentals
- **Kostenloser API-Key** (nach Registrierung)

**Limits (Free Tier):**
- 5 Calls/Minute
- **→ Für 60 Symbole: sehr langsam**

**Kosten (Paid):**
- Starter: $29/Monat (Unlimited Calls/Tag, 5 Calls/Min)
- Developer: $99/Monat (Unlimited Calls/Tag, 15 Calls/Min)
- **→ Teurer, aber sehr professionell**

**Technische Integration:**
- REST API, JSON Response
- API-Key als Header
- Sehr gute Dokumentation

**Pro:**
- ✅ Sehr professionell
- ✅ Sehr umfangreiche Daten

**Contra:**
- ❌ Teurer als Alternativen
- ❌ Free-Tier sehr limitiert

**Empfehlung:** Nur wenn man professionelles Trading/Research macht

---

### 5. Finnhub ⭐⭐ (Gut für Fundamentals)

**Was bietet es:**
- EOD-Historie
- Real-time Quotes
- Fundamentals, News
- **Kostenloser API-Key** (nach Registrierung)

**Limits (Free Tier):**
- 60 Calls/Minute
- **→ Für 60 Symbole: sehr schnell möglich!**

**Kosten (Paid):**
- Basic: $9.99/Monat (Unlimited Calls/Tag, 60 Calls/Min)
- Professional: $39.99/Monat (Unlimited Calls/Tag, 120 Calls/Min)

**Technische Integration:**
- REST API, JSON Response
- API-Key als Query-Parameter
- Gute Dokumentation

**Pro:**
- ✅ Sehr gute Free-Tier-Limits (60 Calls/Min!)
- ✅ Günstige bezahlte Pläne

**Contra:**
- ❌ Weniger fokussiert auf historische Daten

**Empfehlung:** Gut, wenn man auch Fundamentals braucht

---

## Empfehlung nach Budget

### Budget: €0 (nur Free-Tier)
1. **Twelve Data** (800 Calls/Tag, 8 Calls/Min) - Beste Balance
2. **Tiingo** (1000 Calls/Tag, 10 Calls/Min) - Sehr gut
3. **Finnhub** (60 Calls/Min) - Sehr schnell, aber weniger fokussiert

### Budget: €10-20/Monat
1. **Tiingo Starter** ($10/Monat) - Beste Option
2. **Twelve Data Starter** ($9.99/Monat) - Sehr gut
3. **Finnhub Basic** ($9.99/Monat) - Gut

### Budget: €30+/Monat
1. **Tiingo Professional** ($30/Monat) - Sehr professionell
2. **Twelve Data Professional** ($29.99/Monat) - Sehr gut
3. **Polygon.io Starter** ($29/Monat) - Professionell

---

## Technische Integration

### Gemeinsame Struktur

Alle APIs können in die bestehende `PriceDataSource`-Abstraktion integriert werden:

```python
# src/assembled_core/data/data_source.py

class AlphaVantagePriceDataSource(PriceDataSource):
    def __init__(self, settings: Settings):
        self.api_key = settings.alpha_vantage_api_key
        # ...
    
    def get_history(self, symbols, start_date, end_date, freq):
        # API-Call zu Alpha Vantage
        # Normalisierung zu unserem Format
        # ...
```

### Download-Skript-Erweiterung

```python
# scripts/download_historical_snapshot.py

parser.add_argument(
    "--provider",
    choices=["yahoo", "alpha_vantage", "twelve_data", "tiingo"],
    default="yahoo",
    help="Data provider to use"
)
```

---

## Nächste Schritte

1. **Kurzfristig:** Macro-ETFs mit Yahoo (verkürzter Zeitraum)
2. **Mittelfristig:** Einen Provider auswählen (z.B. Twelve Data Free-Tier)
3. **Langfristig:** Multi-Provider-Support (Yahoo als Fallback, Hauptprovider für Bulk-Downloads)

