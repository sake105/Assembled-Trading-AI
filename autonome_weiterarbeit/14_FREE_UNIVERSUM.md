# 14 — Free Universum (0 EUR/Monat)

**Zweck:** Ticker-Listen und Segmentierungs-Strategie, die du **ohne kostenpflichtige Datenquellen** aufbauen kannst. Alpaca IEX + yfinance + Stooq reichen für Phase 1.

**Limitation:** Ohne kostenpflichtige Delisted-History hast du einen Survivorship-Bias. Dein Backtest wird 0.1-0.3 Sharpe zu optimistisch sein. Das ist die Hauptbegrenzung des Free-Universums. Mit dem Paid-Upgrade (siehe `23_PAID_UNIVERSUM.md`) löst du das für 19.99 USD/Monat.

---

## Free-Tier-Strategie

**Maximale Abdeckung unter Null-Budget:**

| Tier | Ticker-Anzahl | Datenquelle | Updates |
|---|---|---|---|
| Tier-1 Core | ~585 | Alpaca IEX (US) + yfinance (EU) | Min/Hour |
| Tier-2 Expansion (eingeschränkt) | +800 | Stooq + yfinance Fallback | EOD |
| Tier-3 On-Demand | unlimitiert | yfinance Batch-Pull | Nur bei Trigger |

**Kompromiss:** Ohne EODHD fehlt dir sauberes Europa und Delisted-Coverage. Du kannst Phase 1 komplett free laufen lassen, aber Backtests sind eingeschränkt valide.

---

## 14.1 Tier-1 Core (~585 Ticker)

**US-Large-Cap: S&P 500 (503 Tickers)**

Quelle: Wikipedia-Scraping oder iShares IVV Holdings-CSV.

```python
import pandas as pd

def get_sp500_tickers():
    # Wikipedia ist kostenlos + tagesaktuell
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    sp500 = tables[0]['Symbol'].tolist()
    # BRK.B, BF.B etc. mit Punkt, manche APIs wollen "-"
    return [t.replace('.', '-') for t in sp500]
```

**Europa-Top: EURO STOXX 50 (50 Tickers)**

Quelle: Wikipedia oder iShares EUE Holdings.

**ETF-Core (35 Tickers):**

```python
ETF_CORE = [
    # US-Broad
    'SPY', 'VOO', 'IVV', 'QQQ', 'IWM', 'VTI', 'DIA',
    # Int-Developed
    'VGK', 'VEA', 'EFA', 'IEFA', 'VXUS',
    # Emerging
    'EEM', 'VWO', 'IEMG',
    # Sektoren
    'XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLP', 
    'XLI', 'XLB', 'XLRE', 'XLU', 'XLC',
    # Commodities
    'GLD', 'SLV', 'USO', 'UNG', 'DBC',
    # Bonds
    'TLT', 'HYG', 'LQD',
    # Vol
    'VXX',
    # Crypto (post 2024 Spot-ETFs)
    'IBIT',
]
```

**Total Tier-1: ~588 Ticker**

---

## 14.2 Tier-2 Expansion (eingeschränkt im Free-Tier)

**Was du ohne EODHD bekommst:**

| Quelle | Coverage | Qualität |
|---|---|---|
| yfinance | US-Small-Cap (Russell 2000) mit Suffix | mäßig, nicht-survivorship-korrigiert |
| yfinance | EU-Ticker (`.DE`, `.L`, `.PA`) | OK für EOD |
| Stooq | EU + teils US + Polen | Fallback |

**Realistische Tier-2-Liste ohne Paid:**

```python
# S&P MidCap 400 via iShares IJH Holdings (Wikipedia via IJH)
# S&P SmallCap 600 via iShares IJR
# STOXX Europe 600 partiell via yfinance mit Suffixen

def get_russell2000_sample():
    # Via iShares IWM Holdings - CSV-Download
    import requests
    url = "https://www.ishares.com/us/products/239710/ishares-russell-2000-etf/1467271812596.ajax?fileType=csv&fileName=IWM_holdings&dataType=fund"
    df = pd.read_csv(url, skiprows=9)
    return df['Ticker'].tolist()
```

**Problem im Free-Tier:**
- yfinance gibt keine **historischen Delisted-Tickers**. Wenn du Russell 2000 2020 hattest und 2024 sind 200 Firmen delisted, siehst du nur die überlebenden 1800. **→ Survivorship-Bias.**
- Stooq hat partielle Coverage, inkonsistent.

**Workaround:** Nur aktuelle Ticker im Universum, **keine historischen Backtests älter als 3 Jahre**. Oder: akzeptiere den Bias und dokumentiere ihn.

---

## 14.3 Tier-3 On-Demand

**Zweck:** Russell 2000 (oder Rest-Russell) bei News-Event-Trigger analysieren, nicht permanent speichern.

**Trigger:**
- News-Velocity-Spike
- Earnings-Announcement
- Unusual-Volume (>3× 20d-Average)
- Gap-Open (>3%)

**Pattern:**
```python
async def on_demand_analysis(ticker: str):
    # 1. yfinance Batch-Pull
    data = yf.Ticker(ticker).history(period="60d")
    # 2. Minimale Feature-Extraction
    features = compute_basic_features(data)
    # 3. Composite-Score ohne teuere News/NLP
    score = lightweight_composite(features)
    # 4. 7d-TTL-Cache, kein permanentes Storage
    await redis.setex(f"ondemand:{ticker}", 86400*7, score)
```

---

## 14.4 Liquiditäts-Filter (harte Empfehlung)

**Regel:** Unter diesen Grenzen wird nicht gehandelt, egal wie gut das Signal ist.

```python
def liquidity_filter(ticker_data):
    return (
        ticker_data['avg_dollar_volume_30d'] > 1_000_000 and  # 1M USD/day
        ticker_data['market_cap'] > 300_000_000 and  # 300M USD
        ticker_data['avg_bid_ask_spread_bps'] < 20 and  # <20 bps
        ticker_data['price'] > 5 and  # >$5
        ticker_data['trading_days_ytd_pct'] > 0.9  # >90% Handelstage
    )
```

**Permissive-Variante für Small-Caps:**
- `avg_dollar_volume_30d` > 500k USD
- `market_cap` > 100M USD
- Rest identisch

**Nach Filter typischerweise:**
- S&P 500: ~495 verbleiben
- Russell 2000: ~1.200 statt 1.960

---

## 14.5 Segmentierung für ML-Training

**Warum nicht ein Modell über alle 585 Ticker?** Heterogenität:
- Biotech-σ ~60% vs Utilities ~15%
- Unterschiedliche Corporate-Action-Frequenzen
- Tick-Size-Regime
- USD-Sensitivität bei ADRs

**Empfohlene Hybrid-Segmentierung:**

1. **Globales Cross-Sectional-Ranking-Modell** auf S&P 500 + EURO STOXX 50 (~550 Ticker)
   - LightGBM mit Sektor als `categorical_feature`
   - Cross-Sectional-Z-Scores pro Datum (Skalen-frei)
   - Haupt-Alpha-Source

2. **11 Sektor-spezifische LightGBM-Modelle** als Confirmation-Overlay
   - Ein Modell pro GICS-Sektor
   - Nur Features, die Sektor-spezifisch sinnvoll sind
   - Overlay-Multiplier auf globales Ranking

3. **Separates EU-Modell** (EURO STOXX 50 + später STOXX 600)
   - Andere Calendar-Effekte (deutsche Feiertage)
   - ECB statt Fed im Macro-Feature-Set

4. **ETF-Modell** separat
   - Macro-driven, niedrigere Vola
   - Implied-Correlation-Features
   - Keine Fundamentals-Features

**Train/Val/Test:**
- **Niemals Random-Shuffle** auf Panel (Cross-Sectional-Leakage!).
- Zeitliche Splits Pflicht.
- Embargo ≥ `max(label_horizon, prediction_horizon) + 1d`.
- Purged-K-Fold oder CPCV.

---

## 14.6 Integration in FastAPI

**Concurrency-Pattern:**
```python
import asyncio
from httpx import AsyncClient, Limits

class TierProcessor:
    def __init__(self):
        self.tier1_semaphore = asyncio.Semaphore(50)
        self.tier2_semaphore = asyncio.Semaphore(20)
        self.tier3_semaphore = asyncio.Semaphore(5)
        self.client = AsyncClient(
            limits=Limits(max_connections=100, max_keepalive=20)
        )
    
    async def process_tier1(self, tickers):
        async with self.tier1_semaphore:
            tasks = [self.analyze(t) for t in tickers]
            return await asyncio.gather(*tasks)
```

**Alpaca-Multi-Symbol-Endpoint** (günstig):
```python
# Bis zu 200 Symbole pro Request
url = "https://data.alpaca.markets/v2/stocks/bars/latest"
params = {
    'symbols': ','.join(tier1_tickers[:200]),
    'feed': 'iex',
}
```

Bei 585 Tier-1-Ticker sind das 3 Requests/Zyklus. Alpaca Free-Limit (200 req/min) wird nur zu 5% ausgelastet.

**Polling-Frequenzen:**

| Tier | Frequenz | Daily-Calls |
|---|---|---|
| Tier-1 | 1 min (nur Session-Zeit 390 min) | ~1.200 Alpaca |
| Tier-2 | 5 min (Session) oder EOD | ~500-1.500 |
| Tier-3 | On-Demand | 100-500 |

---

## 14.7 Storage (lokal, Free)

**Parquet + DuckDB:**

```
features/
  freq=1min/
    year=2025/
      month=01/
        ticker=AAPL.parquet
  freq=1hour/
    ...
  freq=1day/
    ...
```

**Kalkulation für Free-Tier:**

| Tier | Ticker | Bars | Storage |
|---|---|---|---|
| Tier-1 | 585 | 1-Min × 390 Session-Minuten × 252 Tage × 5 Jahre | ~1.2 GB |
| Tier-2 | 800 | 1-Day × 252 × 5 Jahre | ~80 MB |
| Tier-3 | 500 (7d-Cache) | 1-Day × 60 | ~5 MB |
| **Total** | **~1.9 GB** | | |

Passt problemlos auf Standard-Windows-SSD oder Oracle Always-Free (200 GB).

---

## 14.8 Priority-Score für Batch-Processing

**Problem:** Pro Zyklus alle 585 Tier-1-Ticker analysieren kostet Compute.

**Lösung:** Priority-Score, nur Top-200 analysieren:

```python
def priority_score(ticker, context):
    score = (
        0.4 * news_velocity(ticker) +
        0.3 * abs(last_ta_score.get(ticker, 0)) +
        0.3 * np.log1p(avg_dollar_volume(ticker))
    )
    # Events force priority
    if has_earnings_today(ticker) or has_fomc_impact(ticker):
        score += 10  # massive boost
    return score

def get_top_n(n=200):
    scores = {t: priority_score(t, context) for t in tier1_tickers}
    return sorted(scores, key=scores.get, reverse=True)[:n]
```

**Reduziert Compute um 65% ohne Alpha-Verlust.**

---

## 14.9 Ehrliche Einschätzung Free vs. Paid

**Was du im Free-Tier WIRKLICH verlierst:**

1. **Delisted-History** — Backtest-Sharpe um 0.1-0.3 zu hoch
2. **Saubere Europa-Intraday** — nur EOD via Stooq/yfinance
3. **Splits/Dividends-Adjustment für EU** — yfinance macht das nicht sauber
4. **Small-Cap-Long-History** — Russell 2000 nur aktuell, nicht historisch

**Was das praktisch bedeutet:**
- Für **Phase 1 MVP** (erste 3 Monate) ist Free-Tier **ausreichend**.
- Für **seriöse Backtests** brauchst du EODHD 19.99 USD/Monat.
- Für **B2B-SaaS** brauchst du Commercial-Licences (ab 399 USD).

**Der Free-to-Paid-Trigger:** Wenn du anfängst, echte Eigenkapital-Entscheidungen basierend auf Backtest-Sharpe zu treffen. Vorher Free reicht.

---

## Umsetzungs-Checkliste

- [ ] S&P 500 Ticker-Liste aus Wikipedia oder IVV
- [ ] EURO STOXX 50 aus iShares EUE
- [ ] 35 ETF-Core hart-kodiert
- [ ] Liquiditäts-Filter aktiv
- [ ] Tier-1-Polling via Alpaca Multi-Symbol-Endpoint
- [ ] Tier-2 ETF-Holdings-CSV-Download wöchentlich
- [ ] Tier-3 On-Demand-Trigger-Logik
- [ ] Priority-Score für Batch-Reduktion auf Top-200
- [ ] Parquet-Hive-Partitioning konfiguriert
- [ ] Sektor-Mapping (GICS) per Ticker
- [ ] Segmentiertes ML-Training: Global + 11 Sektor + EU + ETF

---

## Verweise

- Datenquellen-Details: siehe `10_FREE_DATEN.md`
- Paid-Universum (1.800 Ticker): siehe `23_PAID_UNIVERSUM.md`
- Storage-Architektur: siehe `12_FREE_INFRASTRUKTUR.md` §12.6
