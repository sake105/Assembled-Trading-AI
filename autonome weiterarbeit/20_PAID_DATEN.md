# 20 — Paid Datenquellen (mit Kosten, unter 100 EUR/Monat)

**Zweck:** Alle Datenquellen, die Geld kosten, **aber echten Mehrwert bringen**. Jede Position einzeln gerechtfertigt, nichts als "nice-to-have" drin.

**Budget-Philosophie:** 100 EUR/Monat Obergrenze. Der Plan liegt bei 40-65 EUR — Puffer für Upgrades.

---

## Module in diesem Dokument

| # | Service | Monatlich | Empfehlung |
|---|---|---|---|
| 20.1 | **EODHD All-World EOD** | **19.99 USD** | **MUST für ernsthafte Backtests** |
| 20.2 | **Claude Haiku 4.5 API** | **<10 EUR** | MUST für News-LLM-Zweitrunde |
| 20.3 | Alpaca Algo Trader Plus | 9 USD | Optional — SIP-Feed statt IEX |
| 20.4 | Finnhub Premium | 9.99-49.99 USD | Optional — höhere Limits + WS |
| 20.5 | FMP Starter | 19 USD | Optional — Fundamentals |
| 20.6 | Polygon Developer | 29 USD | Optional — besserer Tick-Data |
| 20.7 | Alpha Vantage Premium | 49.99 USD | **Skip** — nur wenn News-Sentiment-150 gebraucht |
| 20.8 | Norgate Data Platinum | 52 USD | Phase 3 Upgrade wenn Backtest-Disziplin ernst wird |
| 20.9 | Tradier Developer | 0 USD | FREE, aber Paid-Sandbox-API-Key |
| 20.10 | Databento Credit | 125 USD einmalig | Optional — 1-2 Sample-Tage für Microstructure-Research |

**Realistische Gesamt-Stacks:**

| Config | Ausgaben | Was du bekommst |
|---|---|---|
| **Minimal (MUSS)** | ~22 EUR | EODHD All-World |
| **Typisch** | ~45-55 EUR | + Claude Haiku + optional Finnhub Premium |
| **Maximal** | ~95 EUR | + Polygon Developer + FMP |

---

## 20.1 EODHD All-World EOD — MUST

**Was:** End-of-Day-Daten mit **voller Delisted-Coverage** für 70+ Börsen weltweit.

**Preis:** 19.99 USD/Monat (~18 EUR).

**Limits:** 100.000 API-Calls/Tag (vs. 20/Tag free).

**Warum MUST:**
1. **Survivorship-Bias-Schutz.** Ohne Delisted werden Backtest-Sharpe um 0.1-0.3 überschätzt. Bei ernsthaften Eigenkapital-Entscheidungen ist das der Unterschied zwischen "funktioniert" und "verliert Geld live".
2. **Saubere EU-Coverage.** yfinance mit Suffixen ist brüchig. EODHD hat direktes Feed.
3. **Historische Point-in-Time-Indices** (S&P 500 Composition über Zeit).

**Install:** `pip install eodhd`

**Pattern:**
```python
from eodhd import APIClient

api = APIClient(api_key=settings.eodhd_api_key)

# Delisted historisch
delisted_sample = api.get_eod_historical_stock_market_data(
    symbol="DELISTED_TICKER.US",
    period="d",
    from_date="2015-01-01",
    to_date="2024-12-31"
)

# Index Historical Components
sp500_components = api.get_fundamental_data("GSPC.INDX")
```

**Alternative: Norgate Data Platinum (52 USD/Monat)** — besser für tiefe US-Historie (25k+ delisted seit 1950, historische S&P/Russell-Memberships, Python-Plugin `norgatedata`). **Nur Upgrade wenn Backtesting-Disziplin echte Priorität wird.**

---

## 20.2 Claude Haiku 4.5 API — MUST

**Was:** LLM-Zweitrunde für News-Understanding jenseits von FinBERT-Tone.

**Preis:** ~0.25 USD pro 1M Input-Tokens, 1.25 USD pro 1M Output-Tokens.

**Budget-Kalkulation:** 500 Nachrichten/Tag × 400 Input-Tokens × Haiku 4.5 ≈ **6 USD/Monat Input + 3 USD Output = <10 EUR/Monat**.

**Install:** `pip install anthropic`

**Use-Cases:**

1. **Target-Sentiment-Extraction:** FinBERT gibt binär, Haiku kann Multi-Target + Nuance.

```python
import anthropic

client = anthropic.Anthropic(api_key=settings.anthropic_key)

def extract_news_context(headline, body):
    resp = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": f"""Analysiere diese Finanz-Nachricht. Extrahiere als JSON:
- primary_tickers: Liste der Haupt-Betroffenen
- mentioned_tickers: sekundäre Erwähnungen
- sentiment_primary: -1 bis +1
- event_type: one of [earnings, m&a, management, regulatory, product, legal, analyst, macro]
- rationale: 1 Satz

Headline: {headline}
Body: {body[:1500]}

Nur JSON, kein Kommentar."""
        }]
    )
    return json.loads(resp.content[0].text)
```

2. **Mentioned-vs-Primary-Klassifikation:** 
   - spaCy-NER findet Entities
   - Haiku entscheidet Primary vs Mentioned
   - Nur für Top-20 Nachrichten/Tag (Budget-Grenze)

3. **Veto-Call bei Composite-Score-Grenzfällen:**
   - Nur wenn `|score| > 0.8` → LLM-Veto abfragen
   - "Würde ein Portfolio-Manager diesem Trade zustimmen?"

**Alternatives:**

| Modell | Preis | Qualität | Wann nutzen |
|---|---|---|---|
| **Claude Haiku 4.5** | ~10 EUR/Monat | Sehr gut JSON | **Empfehlung** |
| GPT-5 mini | ähnlich | Sehr gut | Alternative |
| Gemini 2.5 Flash-Lite | billiger | Akzeptabel | Historische Backfills |
| DeepSeek V3.2 | sehr billig | OK | Bulk-Processing |
| Lokales Llama 3.1 8B Q4 (Ollama) | 0 EUR | Mäßig | Fallback ohne API-Key |

---

## 20.3 Alpaca Algo Trader Plus

**Was:** SIP-Feed (100% US-Marktvolumen) statt IEX (2%).

**Preis:** 9 USD/Monat.

**Warum erwägen:** Wenn Microstructure-Features (VPIN, OFI, Lee-Ready-Trade-Signing) genau sein sollen. Mit IEX sind die Zahlen verzerrt, aber **für EOD-Composite-Score ist das OK**.

**Verdict:** Skip in Phase 1-2, **upgrade bei Intraday-Expansion in Phase 3**.

---

## 20.4 Finnhub Premium

**Preise:**
- Starter: 9.99 USD
- Standard: 19.99 USD
- Professional: 49.99 USD

**Warum erwägen:**
- Höhere Rate-Limits (300 req/min statt 60)
- WebSocket für mehr Symbole (statt 50)
- Historical News Data (tiefere Backfills)
- Insider Transactions Real-time

**Verdict:** **9.99 USD Starter-Tier wenn dein News-Volumen wächst**, sonst Free reicht.

---

## 20.5 FMP (Financial Modeling Prep) Starter

**Was:** Fundamentals-Fokus. 10-K/Q Income-Statement, Balance-Sheet, Cash-Flow historisch.

**Preis:** 19 USD/Monat (Starter-Tier, unlimited API-Calls).

**Warum erwägen:** Free-Tier nur 250 req/Tag, für 600+ Ticker schnell ausgeschöpft.

**Alternative:** SEC EDGAR via `edgartools` (free) deckt ~90% der FMP-Daten ab. **FMP lohnt nur wenn du Fundamental-Features systematisch extrahierst und EDGAR-Parsing zu viel Arbeit ist.**

**Verdict:** Skip bis Phase 3 Fundamentals-Deep-Dive.

---

## 20.6 Polygon Developer

**Was:** Bester Tick-Data-Provider für US. 15-Minuten-Real-time, unlimited Historie.

**Preis:** 29 USD/Monat.

**Warum erwägen:** Wenn du **Microstructure-Signale** ernsthaft bauen willst (OFI auf Tick-Level, sauberes Lee-Ready). Alpaca IEX reicht dafür nicht.

**Alternative:** Databento Credit einmalig 125 USD für 1-2 Sample-Tage zum Experimentieren (siehe 20.10).

**Verdict:** Erst in Phase 3, wenn Intraday-Strategie entwickelt wird.

---

## 20.7 Alpha Vantage Premium Plan 150

**Was:** News-Sentiment pre-computed + Technical Indicators + Earnings-Transcripts.

**Preis:** 99.99 USD/Monat — **kritisch an der Budget-Grenze**.

**Wann lohnt es:** Wenn du News-Volumen von 2000+ Nachrichten/Tag brauchst und Finnhub-Free nicht reicht.

**Verdict:** **Skip** — Claude Haiku 4.5 API ist die bessere Wahl. Gleicher Preis-Bereich, aber du kontrollierst die Logik.

---

## 20.8 Norgate Data Platinum

**Was:** Gold-Standard für US-Backtest-Data.

**Preis:** ~52 USD/Monat.

**Was du bekommst:**
- 25.222+ delisted US-Tickers seit **1950**
- Historische S&P/Russell-Memberships
- Corporate-Action-korrigiert
- Python-Plugin `norgatedata`

**Wann upgrade:** Phase 3, wenn EODHD für US-Historie zu flach wird und ernsthafte Multi-Decade-Backtests gefahren werden.

**Verdict:** **Phase 3 Option.** Für Phase 1-2 reicht EODHD.

---

## 20.9 Tradier Developer Sandbox

**Was:** Options-Chain + Greeks + IV, 15-min-delayed, Paper-Trading.

**Preis:** **0 USD** (Sandbox) — aber separate Registrierung mit API-Key nötig.

**Warum erwähnen:** Als **zweite Options-Quelle neben Alpaca** sinnvoll. Wenn Alpaca-Options-API nicht passt oder Alpaca-Limits greifen.

**Verdict:** Free, einfach einrichten, ergänzend zu Alpaca nutzen.

---

## 20.10 Databento One-Time Credit

**Was:** 125 USD Sign-up-Credit, dann Pay-as-you-go.

**Warum erwähnen:** Perfekt für **1-2 Sample-Tage MBO-Data** (Market-By-Order) zum Experimentieren mit Level-3-Microstructure. Danach Replay in `nautilus_trader` oder `hftbacktest`.

**Verdict:** Einmaliger Test, nicht dauerhaft.

---

## Die empfohlenen Stacks

### Stack 1: "Paid-Minimum" (~22 EUR)

Für ernsthafte Phase-2-Umsetzung, wenn du echte Backtests fährst.

| Posten | Monatlich | Zweck |
|---|---|---|
| **EODHD All-World** | 19.99 USD | Delisted + EU |
| Everything else | 0 | Alpaca Paper, GDELT, SEC, FRED, Finnhub Free |

**Summe:** ~18 EUR.

### Stack 2: "Typisch" (~45 EUR)

Für aktive Nutzung mit LLM-Enhanced-News.

| Posten | Monatlich | Zweck |
|---|---|---|
| EODHD All-World | 18 EUR | Delisted + EU |
| Claude Haiku 4.5 API | ~10 EUR | LLM-Zweitrunde |
| Finnhub Starter | 9 EUR | Höhere Limits |
| Optional FMP | 17 EUR | Fundamentals |

**Summe:** ~35-55 EUR.

### Stack 3: "Maximal" (~95 EUR)

Für Phase 3, wenn Intraday & SaaS-Vorbereitung läuft.

| Posten | Monatlich | Zweck |
|---|---|---|
| EODHD All-World | 18 EUR | Delisted + EU |
| Claude Haiku 4.5 API | ~10 EUR | LLM-Zweitrunde |
| Polygon Developer | 27 EUR | US-Tick-Data |
| Finnhub Standard | 19 EUR | WS + History |
| FMP Starter | 17 EUR | Fundamentals |

**Summe:** ~91 EUR.

---

## Was NICHT im Paid-Katalog ist (explizit ausgeschlossen)

- **Bloomberg Terminal** (~24k USD/Jahr) — Enterprise.
- **Refinitiv Eikon** (~22k USD/Jahr) — Enterprise.
- **FactSet** — Enterprise.
- **CRSP Academic** — privat nicht zugänglich.
- **Wallmine / Simplywall.st** — Konsumenten-Fokus, kein Alpha.
- **Stocktwits Premium** — kein Mehrwert über Free-API.
- **Seeking Alpha Premium** — Scraping-Problem, kein API.
- **Unusual Whales** (48-95 USD) — teuer für Retail-Options-Flow.
- **SqueezeMetrics GEX-Daten** — ähnlich teuer.

Diese sprengen Budget **und** haben Ersatz über Free-Tier + LLM-Analyse.

---

## Umsetzungs-Checkliste

**Monat 3-4 (Paid-Minimum):**
- [ ] EODHD-Account + API-Key in `.env`
- [ ] Delisted-Pull-Pipeline für historische Backfills
- [ ] EU-Ticker via EODHD statt yfinance
- [ ] Historische S&P-Compositions als Universe-Basis

**Monat 4-6 (Typisch):**
- [ ] Anthropic-Account + API-Key
- [ ] Claude Haiku 4.5 für Top-20 News/Tag
- [ ] Finnhub Starter-Upgrade wenn Rate-Limits greifen
- [ ] Budget-Alerts auf Anthropic-Usage

**Monat 6-12 (Maximal bei Bedarf):**
- [ ] Polygon Developer wenn Intraday ernst
- [ ] FMP Starter wenn Fundamentals-Features
- [ ] Norgate wenn Multi-Decade-Backtests
- [ ] Databento 125-USD-Credit für Microstructure-Sample

---

## Ehrliche Einschätzung

**Die zwei Paid-Posten mit klarem ROI sind:**
1. **EODHD (18 EUR)** — ohne Survivorship-Bias-Schutz sind deine Backtests Fake.
2. **Claude Haiku 4.5 API (10 EUR)** — die News-Pipeline bekommt Context-Verständnis, das kein FinBERT hat.

**Alles andere ist Nice-to-Have.** Wenn du bei 28 EUR/Monat bleibst, hast du schon 95% des möglichen Paid-Edge. Die weiteren 5% kosten 60+ EUR zusätzlich — schlechtes Preis-Leistung.

**Die einzige Ausnahme:** Wenn du **B2B-SaaS (Richtung B) startest**, brauchst du Commercial-Licences:
- EODHD Commercial: ab 399 USD/Monat
- Finnhub Commercial: ab 99 USD/Monat  
- Anthropic ist eh Commercial-ready

Das ist dann ein eigener Budget-Sprung bei SaaS-Launch, nicht in Phase 1-3.
