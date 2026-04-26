# 23 — Paid Universum (mit EODHD-Upgrade)

**Zweck:** Das volle Universum, das du mit EODHD All-World EOD (19.99 USD/Monat) bauen kannst. ~1 800 Ticker aktiv, mit Delisted-Coverage und sauberer Europa-Abdeckung.

**Kosten:** 19.99 USD/Monat EODHD allein. Optional +9 USD Alpaca Algo Trader Plus für SIP-Feed.

---

## 23.1 Der Drei-Tier-Hybrid

Dieses Universum ist die Kern-Strategie-Entscheidung: **US + Europa + internationale ETF-Exposure**, segmentiert in drei Tiers mit unterschiedlichen Update-Frequenzen.

```
╔══════════════════════════════════════════════════════════╗
║  Tier-1 Core  (~585 Ticker, vollständige Features)       ║
║  • S&P 500 (503)                                          ║
║  • EURO STOXX 50 (50)                                     ║
║  • 35 Core-ETFs                                           ║
║  → Alpaca IEX (US Live) + EODHD (EU Live + Backfill)      ║
║  → Update: 1 min intraday, EOD-Bars 1-Tages-Bars          ║
╠══════════════════════════════════════════════════════════╣
║  Tier-2 Expansion (~1200 Ticker, reduzierte Features)    ║
║  • S&P MidCap 400 (400)                                   ║
║  • S&P SmallCap 600 nach Filter (~450)                    ║
║  • STOXX Europe 600 Rest (~400)                           ║
║  • 25 ADRs (Asien-Exposure)                              ║
║  → EODHD EOD + Fundamentals                               ║
║  → Update: EOD (täglich 16:30 ET)                        ║
╠══════════════════════════════════════════════════════════╣
║  Tier-3 Opportunistic (~800 Ticker, on-demand)            ║
║  • Russell 2000 minus S&P 600                             ║
║  → EODHD Delisted + yfinance Batch                        ║
║  → Update: nur bei Trigger (News, Vol, Gap)              ║
║  → 7-Tage-TTL-Cache                                       ║
╚══════════════════════════════════════════════════════════╝

Total aktiv: ~1.800 Ticker (Tier 1 + 2)
Total on-demand: +800 (Tier 3)
```

---

## 23.2 Ticker-Listen (detailliert)

### Tier-1-US-Core (503 Ticker)

**S&P 500 Composition** aus **iShares IVV Holdings-CSV** (Goldstandard, tagesaktuell, lizensiert für Informational).

```python
import requests
import pandas as pd

def get_sp500_from_ivv():
    url = ("https://www.ishares.com/us/products/239726/"
           "ishares-core-sp-500-etf/1467271812596.ajax"
           "?fileType=csv&fileName=IVV_holdings&dataType=fund")
    df = pd.read_csv(url, skiprows=9)
    df = df[df['Ticker'] != '-']
    return df['Ticker'].unique().tolist()
```

**Achtung BRK.A/B, GOOG/GOOGL:** 503 Symbole, nicht 500, weil Dual-Share-Classes.

### Tier-1-EU-Core (50 Ticker)

**EURO STOXX 50** aus iShares EUE Holdings. Update alle drei Monate (Index-Rebalancing).

### Tier-1-ETF-Core (35 Ticker)

```python
ETF_CORE_TIER1 = [
    # US-Broad (7)
    'SPY', 'VOO', 'IVV', 'QQQ', 'IWM', 'VTI', 'DIA',
    # Int-Developed (5)
    'VGK', 'VEA', 'EFA', 'IEFA', 'VXUS',
    # Emerging (3)
    'EEM', 'VWO', 'IEMG',
    # US-Sektoren (11 SPDR)
    'XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLP',
    'XLI', 'XLB', 'XLRE', 'XLU', 'XLC',
    # Commodities (3)
    'GLD', 'SLV', 'DBC',
    # Bonds (3)
    'TLT', 'HYG', 'LQD',
    # Vol + Crypto (3)
    'VXX', 'IBIT', 'ETHA',
]
# = 35 Ticker
```

### Tier-2-US-MidCap (400 Ticker)

**S&P MidCap 400** via iShares IJH Holdings.

### Tier-2-US-SmallCap (~450 nach Filter)

**S&P SmallCap 600** via iShares IJR. **Vorteil gegenüber Russell 2000:** Quality-Filter (4 GAAP-Gewinn-Quartale) — robustere Features.

**Filter nach Anwendung liquidity_filter() (siehe `14_FREE_UNIVERSUM.md` §14.4):** ca. 450 aktive Ticker.

### Tier-2-Europa-Rest (~400 Ticker)

**STOXX Europe 600** via iShares EXSA Holdings (Wikipedia als Backup).

Länder-Breakdown:
- UK: ~100 Ticker
- Schweiz: ~40
- Frankreich: ~70
- Deutschland: ~60
- Niederlande: ~30
- Italien, Spanien, Skandinavien: Rest

### Tier-2-ADRs (25 Ticker)

```python
ADR_LIST = [
    # Asien-Tech/Consumer
    'TSM',   # Taiwan Semiconductor
    'BABA', 'JD', 'PDD', 'BIDU', 'NTES', 'TCEHY',
    # Chinese EVs
    'LI', 'XPEV', 'NIO',
    # Services
    'BEKE', 'TCOM',
    # Japan
    'TM', 'SONY', 'HMC', 'MUFG', 'SMFG', 'MFG', 'NMR',
    # India
    'INFY', 'WIT', 'HDB', 'IBN', 'RDY',
    # SE-Asia
    'SE', 'GRAB',
    # Latam
    'YMM', 'CPNG',
]
```

**Warnung:** TCEHY und NTES sind OTC-Pink-Sheets mit breiteren Spreads. **Im Liquiditäts-Filter mit eigener Grenze:** `max_bid_ask_spread_bps=50` (statt 20).

### Tier-3-Event-Driven (~800 Ticker)

**Russell 2000 minus S&P 600** = ca. 1.400 Ticker.

**Nach Liquiditäts-Filter (`permissive-Variante`):** ca. 800.

Nur aktiv bei:
- News-Velocity > 2σ
- Unusual Volume > 3× 20d-AvgVol
- Gap > 3%
- Earnings innerhalb 24h
- Analyst-Upgrade zu "Buy"

---

## 23.3 Datenquellen-Stack

| Ticker-Gruppe | Primary | Sekundär | EOD | Intraday |
|---|---|---|---|---|
| Tier-1 US-Core | Alpaca IEX | EODHD | ✅ | ✅ |
| Tier-1 EU | EODHD | yfinance | ✅ | ⚠ delayed |
| Tier-1 ETFs | Alpaca | EODHD | ✅ | ✅ |
| Tier-2 US | EODHD | Alpaca | ✅ | ❌ |
| Tier-2 EU | EODHD | Stooq | ✅ | ❌ |
| Tier-2 ADRs | Alpaca IEX | EODHD | ✅ | ✅ |
| Tier-3 | EODHD Delisted | yfinance | ✅ | ❌ |

**EODHD-Call-Kalkulation:**

```
Tier-1 EU + ETFs (85 Ticker × 1 min × 390 Session):
  ~33.000 Calls/Tag (falls EODHD Intraday genutzt)
Tier-2 EOD (1 200 Ticker × 1 Call):
  1.200 Calls/Tag
Tier-3 On-Demand:
  100-500/Tag
Fundamentals-Refresh (wöchentlich 1.800 × 1 Call/10 Min):
  180/Tag amortisiert
Total: ~15.000-35.000 Calls/Tag
```

**EODHD-Limit:** 100.000 Calls/Tag → **15-35% Auslastung**, komfortabler Puffer.

---

## 23.4 Historische S&P-/Russell-Memberships

**Problem ohne historische Memberships:** Wenn ein Ticker in 2022 im S&P 500 war und 2024 rausfällt, zeigen moderne Listen nur die heutigen 500. Ein Backtest auf "S&P 500 2020-2024" mit aktueller Liste ist verzerrt.

**Lösung mit EODHD:**
```python
# EODHD /fundamental/ endpoint gibt historische Components
sp500_components_2022 = eodhd_client.get_fundamental_data(
    ticker="GSPC.INDX",
    as_of="2022-06-30"
)
```

**Alternative für Deep-Historie:** Norgate Data Platinum (52 USD/Monat) hat historische S&P/Russell-Memberships seit 1950. Nur wenn Multi-Decade-Backtests essentiell.

---

## 23.5 Segmentierung für ML-Training (Paid-Variante)

Mit 1.800 aktiven Tickern lohnt sich die Hybrid-Segmentierung richtig.

**Strategie:**

1. **Globales Cross-Sectional-Modell** auf Tier-1+2 (1.800 Ticker)
   - LightGBM
   - Sektor + Region als `categorical_feature`
   - Cross-Sectional-Rank-Transform pro Datum
   - **Haupt-Alpha-Source**

2. **11 Sektor-Modelle** (GICS-Level-1)
   - Jedes Modell trainiert auf 100-300 Ticker
   - Overlay-Multiplier auf globales Ranking

3. **3 Regional-Modelle:** US, EU, ADRs
   - Andere Calendar-Effekte, Macro-Sensitivitäten

4. **ETF-Modell** separat

**Ein Modell für alle 1.800 funktioniert nicht** — Heterogenität ist zu groß:
- Biotech-σ ~60 % vs Utilities ~15 %
- EU handelt mit ECB-Zinsdaten, US mit Fed
- ADRs haben USD-Sensitivität

**Panel-Data-Korrekturen:**
- Entity-Fixed-Effects via LightGBM-categorical oder Within-Transform
- Time-Fixed-Effects via Cross-Sectional-Demean pro Datum
- Cluster-Robust-SEs für Finale-Regression (mit `linearmodels`)

---

## 23.6 Liquiditäts-Filter mit Paid-Daten

EODHD liefert saubere **Bid-Ask-Spread-Historie** und **ADV** für Filter:

```python
def liquidity_filter_tier_aware(ticker_data, tier):
    if tier == 1:
        return (
            ticker_data['avg_dollar_volume_30d'] > 5_000_000 and  # strikt
            ticker_data['market_cap'] > 1_000_000_000 and
            ticker_data['avg_bid_ask_spread_bps'] < 10 and
            ticker_data['price'] > 5
        )
    elif tier == 2:
        return (
            ticker_data['avg_dollar_volume_30d'] > 1_000_000 and
            ticker_data['market_cap'] > 300_000_000 and
            ticker_data['avg_bid_ask_spread_bps'] < 25 and
            ticker_data['price'] > 5
        )
    else:  # tier 3
        return (
            ticker_data['avg_dollar_volume_30d'] > 500_000 and
            ticker_data['market_cap'] > 100_000_000 and
            ticker_data['price'] > 3
        )
```

---

## 23.7 Storage-Dimensionierung

**Erwartete Größe Parquet Feature-Store:**

| Tier | Ticker | Frequenz | 5 Jahre Bars | Storage |
|---|---|---|---|---|
| Tier-1 US (503) | 503 | 1-min × 390 × 252 × 5 | ~1.0 GB |
| Tier-1 EU (50) | 50 | 5-min × 510 × 252 × 5 | ~150 MB |
| Tier-1 ETFs (35) | 35 | 1-min × 390 × 252 × 5 | ~70 MB |
| Tier-2 (1.200) | 1.200 | 1-day × 252 × 5 | ~90 MB |
| Tier-3 cache | 800 | 1-day × 60 (rolling) | ~10 MB |
| News-Text-Cache | ~5 GB | 2 Jahre Headlines | ~5 GB |
| **Total** | | | **~7 GB** |

Passt auf Hetzner CX22 40-GB-SSD mit Puffer. Hetzner CX32 80-GB-SSD für Phase 2 bequem.

---

## 23.8 Integration in FastAPI

**Tier-aware Scheduling:**
```python
from apscheduler.schedulers.asyncio import AsyncIOScheduler

scheduler = AsyncIOScheduler()

# Tier-1 Intraday (nur Session)
scheduler.add_job(tier1_poll, 'cron', minute='*',
                  hour='9-16', day_of_week='mon-fri',
                  timezone='America/New_York')

# Tier-2 EOD nach US-Close
scheduler.add_job(tier2_eod_pull, 'cron',
                  hour=16, minute=30,
                  day_of_week='mon-fri',
                  timezone='America/New_York')

# Tier-3 Event-triggered (nicht scheduled)
# Wird durch News/Vol/Gap-Triggers aus Redis Stream ausgelöst

# Fundamentals-Refresh wöchentlich
scheduler.add_job(fundamentals_refresh, 'cron',
                  day_of_week='sat', hour=3)
```

**Priority-Score** (siehe `14_FREE_UNIVERSUM.md` §14.8) reduziert Tier-1-Compute auf Top-200/Zyklus.

---

## 23.9 Kosten- und Call-Budget-Kalkulation

**Monatlich:**
- EODHD All-World EOD: 19.99 USD (~18 EUR)
- Alpaca IEX Paper: 0 EUR
- **Total Daten: ~18 EUR**

**EODHD-Calls:**
- Daily-EOD-Bulk: ~3.000/Tag
- Intraday-EU (wenn genutzt): ~30.000/Tag
- Fundamentals wöchentlich: amortisiert ~200/Tag
- Tier-3-Event: bis 500/Tag
- **Total: 33.000-35.000/Tag von 100.000 Limit → 33-35% Auslastung**

**Bei Budget-Überschreitung** auf Intraday-EU: EODHD gibt 5-min-delayed kostenlos, nur Paid-Feed Real-time — für EU-EOD-Strategien nicht kritisch.

---

## 23.10 Rechtliche Lage

**EODHD All-World EOD** (19.99 USD):
- Personal-Use + internes Trading **OK**.
- **B2B-SaaS-Weitergabe NICHT OK** — braucht Commercial-Plan ab 399 USD/Monat.
- Für Phase 1-3 (Personal Quant) ausreichend.

**Bei Richtung B (B2B-SaaS):**
```
EODHD Commercial:   399-999 USD/Monat
Finnhub Commercial: ab 99 USD/Monat (+ je nach Symbol-Universum)
Alpaca:             Pro-Account + SIP-Data (39 USD/Monat)
```

Das ist ein eigener Budget-Sprung bei SaaS-Launch, nicht in Phase 1-3.

---

## Umsetzungs-Checkliste

**Phase 2 (mit Paid-Stack):**
- [ ] EODHD-Account + API-Key
- [ ] iShares IVV + IJH + IJR Holdings-CSV-Pipeline
- [ ] iShares EUE + iShares EXSA für EU
- [ ] 35 ETF-Core-Liste fest im Config
- [ ] 25 ADR-Liste mit Spread-Anpassung
- [ ] Liquiditäts-Filter tier-aware
- [ ] EODHD Delisted-Pull für Survivorship-Korrektur
- [ ] Historische S&P-Composition für Backtest-Windows
- [ ] Storage-Partitionierung nach Tier/Freq/Ticker
- [ ] Priority-Score für Top-200/Zyklus
- [ ] 11 Sektor-Modelle + 3 Regional + 1 ETF + 1 Global
- [ ] Tier-3-Trigger-Logik (News, Vol, Gap, Earnings)

---

## Der Unterschied Free vs Paid-Universum

| Dimension | Free-Tier | Paid-Tier (EODHD) |
|---|---|---|
| Aktive Ticker | ~585 | ~1.800 |
| Delisted-Coverage | nein | **ja (US + EU)** |
| Europa-Abdeckung | STOXX 50 + Rest via Stooq brüchig | **STOXX 600 sauber** |
| Historische Indices | nur aktuelle Compositions | **historische S&P/Stoxx-Mitgliedschaften** |
| Split-/Dividend-Adj (EU) | unvollständig | **sauber** |
| Backtest-Sharpe-Korrektur | 0.1-0.3 zu optimistisch | **sauber** |

Die 18 EUR/Monat für EODHD sind **der eine Paid-Posten, der sich fast immer rechtfertigt**, sobald du seriöse Backtests fährst. Alle anderen Paid-Optionen (Polygon, FMP, Finnhub Premium) sind Nice-to-Have.
