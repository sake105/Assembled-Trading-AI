# Wettbewerbsanalyse v4: Tiefer Tauchgang

**Datum:** 2026-04-27
**Vorgänger:** v1 (909 Zeilen), v2 (1459 Zeilen), v3 (~1015 Zeilen)
**Charakter v4:** Selbst v3 hat noch Module übersehen und Themen nicht angesprochen. Bei einem 120k-LOC, 25-Modul-Repo wie deinem ist die Arbeit hier nie wirklich fertig. v4 erweitert v3 in zwei Richtungen:

**Teil A — Repo-Module, die in v1-v3 nie systematisch betrachtet wurden:** Bereiche wie `api/`, `data/sources/`, `events/crisis_alpha/`, `events/disclosures/`, `signals/regime/`, `events/news/` — du hast hier teils gigantische Codebasen, die wir nie aufgegriffen haben.

**Teil B — Echt neue Themen:** Causal Inference, GNN, VVIX, Time-Series-DBs, Event-Sourcing/CQRS, Auto-Feature-Engineering, Anomalie-Detection, Reproducibility-Stack, satellite/weather alt-data, FastAPI 2026, prediction markets, RAG für Finance.

**Wichtige Erkenntnis nach erneutem Repo-Scan:**

Ich habe in v3 schon die Module gelistet, die du bereits hast und die wir nie hätten "einbauen" empfehlen sollen. Bei diesem Pass kamen NOCH MEHR dazu:

- **`api/` (3463 LOC FastAPI-Server)** — komplett übersehen. 9 Routers (monitoring, oms, orders, paper_trading, performance, portfolio, qa, risk, signals). Vermutlich dein Frontend-Backend.
- **`data/feature_store.py` (259 LOC)** — du hast einen ECHTEN DuckDB+Parquet Feature-Store mit ASOF-JOIN PIT-Safety. v2 §18.3 hat empfohlen einen zu bauen — **Fehler von mir, du hast bereits einen.**
- **`data/factor_store.py` (495 LOC)** — eigener Factor-Store mit Hive-Partitionierung
- **`data/synthetic_generator.py` (98 LOC)** — Crisis-Template-Synthetic-Generator (2008_gfc, 2020_covid, 2000_dotcom, 1987_crash, 2022_rate_shock). v3 §B2 hat empfohlen Synthetic-Data-Generierung einzubauen — **Fehler von mir, du hast bereits eine schlanke Variante.**
- **`data/sources/` mit 17 Sources** — alphavantage, bls, bluesky, cboe, coinmetrics, earnings_calendar, edgar, finra, fred, gdelt_gcam, newsapi, polygon, stooq, weather, wikipedia_views, worldbank, yfinance. Du hast ein riesiges Daten-Universum.
- **`events/crisis_alpha/` (~1500 LOC)** — eigene Crisis-Alpha-Pipeline mit baskets, context, entry, exit_rules, gates, pipeline, risk_budget, state_machine
- **`events/disclosures/` (~1700 LOC)** — EDGAR + House-PTR (Insider-Trading von US-Kongressabgeordneten!) Pipeline
- **`events/news/pipeline.py` (720 LOC)** — vollständige News-Engine mit GDELT, RSS, ACLED, NER, Burst-Detection, Clustering, Trigger-Scoring
- **`signals/regime/hmm_posterior.py` (108 LOC)** — eigenes Regime-Posterior

**Der Punkt ist:** Bevor man dir IRGENDWAS empfiehlt, muss man wissen, was du bereits hast. v4 versucht das umfassender als v1-v3.

---

## Inhaltsverzeichnis

### Teil A — Repo-Module nie systematisch betrachtet

A11. [`api/` (3463 LOC FastAPI-Server) — was 2026-Best-Practices wären](#a11-api-fastapi-server)
A12. [`data/sources/` (17 Datenquellen) — was du sammelst und was fehlt](#a12-data-sources-17-quellen)
A13. [`events/crisis_alpha/` — Vergleich mit Alpha-Architect-CAOS-Methodik](#a13-events-crisis-alpha)
A14. [`events/disclosures/fetch_house_ptr.py` (523 LOC) — Insider-Trading-Daten](#a14-events-disclosures-house-ptr)
A15. [`events/news/pipeline.py` (720 LOC) — News-Pipeline-Vergleich](#a15-events-news-pipeline)
A16. [`signals/regime/hmm_posterior.py` — Posterior-Inference](#a16-signals-regime-posterior)
A17. [`features/triple_barrier.py` (307 LOC) — größer als v3 angenommen](#a17-triple-barrier-revisited)
A18. [`features/market_breadth.py` (665 LOC) — was es da draußen gibt](#a18-features-market-breadth)
A19. [`signals/crash_prediction.py` (577 LOC) — was Industrie macht](#a19-signals-crash-prediction)
A20. [`signals/multifactor_signal.py` (908 LOC) — was Bridgewater/AQR machen](#a20-signals-multifactor)

### Teil B — Wirklich neue Bereiche

B11. [VVIX statt VIX als Tail-Risk-Indikator](#b11-vvix-tail-risk)
B12. [Causal Inference mit DoWhy/EconML/CausalPy](#b12-causal-inference)
B13. [Graph Neural Networks für Stock-Prediction](#b13-graph-neural-networks)
B14. [ClickHouse/TimescaleDB für Tick-Data](#b14-clickhouse-timescaledb)
B15. [Event Sourcing & CQRS Architektur](#b15-event-sourcing-cqrs)
B16. [Automated Feature Engineering: tsfresh, featuretools](#b16-tsfresh-featuretools)
B17. [Anomalie-Detection-Stack: PyOD, DeepOD, TSB-AD](#b17-anomaly-detection)
B18. [DVC + MLflow Reproducibility Stack](#b18-reproducibility-stack)
B19. [Open-Meteo + NOAA Weather/Satellite Alternative Data](#b19-weather-satellite-altdata)
B20. [FastAPI Best-Practices 2026](#b20-fastapi-best-practices)
B21. [Prediction Markets als Alpha-Quelle (Polymarket, Kalshi)](#b21-prediction-markets)
B22. [Knowledge-Graph-RAG für Finance-Research](#b22-knowledge-graph-rag)

### [Konsolidierter Adoption-Plan v4](#konsolidierter-adoption-plan-v4)

---

# Teil A — Repo-Module nie systematisch betrachtet

## A11. `api/` FastAPI-Server

### Was du hast

**3463 LOC FastAPI-Server** mit:
- `api/app.py` (110 LOC) — App-Setup
- `api/models.py` (876 LOC) — Pydantic-Models
- 9 Routers in `api/routers/`:
  - `monitoring.py` (620 LOC) — größter Router, Health/Metrics/Status
  - `paper_trading.py` (355 LOC)
  - `portfolio.py` (317 LOC)
  - `qa.py` (401 LOC)
  - `oms.py` (196 LOC) — Order Management System
  - `signals.py` (195 LOC)
  - `performance.py` (162 LOC)
  - `risk.py` (121 LOC)
  - `orders.py` (91 LOC)

### Was wir in v1-v3 nie diskutiert haben

Wir haben uns auf Trading-Logik konzentriert und API-Architektur völlig ignoriert. Das ist ein Fehler — wenn du einen 3463-LOC FastAPI-Server hast, hast du eine reale Web-Architektur, die Best-Practices verdient.

### Was die Industry 2026 macht

**`zhanymkanov/fastapi-best-practices`** ist DIE Referenz. Tested gegen Python 3.11+, FastAPI 0.115+, Pydantic 2.7+, SQLAlchemy 2.0+. Hauptpunkte:

1. **Decouple & Reuse Dependencies** — Dependency-Injection für DB-Sessions, Auth, etc. Cached innerhalb eines Requests.
2. **SQL-first, Pydantic-second** — bei Performance-kritischen Pfaden direktes SQL, nicht ORM.
3. **Async vs Sync Routes** — `async def` nur wenn echtes async I/O. CPU-bound → sync Route (FastAPI runs sie in Thread Pool).
4. **Feature-based Project Structure** — gruppiere by Feature nicht by Type:
   ```
   api/routers/
   ├── auth/
   │   ├── auth.py        # endpoints
   │   ├── models.py      # pydantic models
   │   ├── services.py    # business logic
   │   └── tasks.py       # celery tasks
   ├── payments/
   ├── core/
   ```
   Du hast aktuell nur einen flachen `api/routers/`-Ordner. Refactoring zu feature-based macht es deutlich wartbarer.
5. **Pydantic Settings** — `pydantic-settings` validiert Env-Vars beim Startup. Verhindert Production-Crashes wegen vergessener Variable.
6. **Health Checks: Liveness vs Readiness** — Liveness = "App läuft", Readiness = "App + Dependencies (DB, Redis, Broker) bereit". Load-Balancer routen nur an Readiness-OK-Instanzen.
7. **Request IDs für Distributed Tracing** — Middleware setzt `X-Request-ID`, jeder Log-Eintrag enthält die ID. Bei Bug-Reports kannst du den kompletten Request-Path tracen.

### Konkrete Empfehlung

Ich vermute (ohne deinen Code im Detail zu kennen), dass dein FastAPI-Setup von 2024/2025 stammt. Ein paar Quick-Wins:

```python
# api/middleware.py (neu)
import time
import uuid
from fastapi import Request
from contextvars import ContextVar

request_id_var: ContextVar[str] = ContextVar("request_id", default="")

async def request_id_middleware(request: Request, call_next):
    """Add X-Request-ID for distributed tracing."""
    rid = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request_id_var.set(rid)
    start = time.time()
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    response.headers["X-Process-Time-Ms"] = f"{(time.time() - start) * 1000:.1f}"
    return response


# api/routers/monitoring.py erweitern
@router.get("/health/live")
async def liveness():
    """Process is running. No dependency checks."""
    return {"status": "alive"}

@router.get("/health/ready")
async def readiness():
    """All critical dependencies reachable."""
    checks = {}
    try:
        # DB ping
        await db.execute("SELECT 1")
        checks["db"] = "ok"
    except Exception as e:
        checks["db"] = f"fail: {e}"
    
    try:
        # Broker connection
        broker_status = await broker.health_check()
        checks["broker"] = broker_status
    except Exception as e:
        checks["broker"] = f"fail: {e}"
    
    all_ok = all(v == "ok" for v in checks.values())
    return {"status": "ready" if all_ok else "not_ready", "checks": checks}
```

**Aufwand:** 8-15h für ein systematisches Audit deiner FastAPI-Setup gegen die Best-Practices. Lohnt sich vor allem wenn du eine UI auf dem Server laufen lässt oder Webhooks empfängst.

**Lizenz:** zhanymkanov/fastapi-best-practices ist MIT — direkt adoptierbar.

---

## A12. `data/sources/` 17 Quellen

### Was du hast

```
data/sources/
├── alphavantage_source.py     (164 LOC)
├── bls_source.py              (158 LOC)  # Bureau of Labor Statistics!
├── bluesky_source.py          (121 LOC)  # AT-Protokoll Social
├── cboe_source.py             (200 LOC)  # CBOE direkt!
├── coinmetrics_source.py      (189 LOC)
├── earnings_calendar_source.py (287 LOC)
├── edgar_source.py            (142 LOC)
├── finra_source.py            (132 LOC)  # FINRA TRACE
├── fred_source.py             (137 LOC)
├── gdelt_gcam_source.py       (178 LOC)  # GDELT Global Conflict
├── newsapi_source.py          (150 LOC)
├── polygon_source.py          (161 LOC)
├── stooq_source.py            (115 LOC)
├── weather_source.py          (245 LOC)  # WETTER!
├── wikipedia_views_source.py  (186 LOC)  # WikiPedia-Suchvolumen!
├── worldbank_source.py        (138 LOC)
└── yfinance_source.py         (158 LOC)
```

**Insgesamt 2897 LOC für Datenquellen.** Das ist ein breiteres Datenuniversum als die meisten Hedge-Funds.

### Bewertung der einzelnen Quellen

| Quelle | Was es liefert | Wert für Trading |
|---|---|---|
| **alphavantage** | OHLCV, Fundamentals, FX | Standard, etwas inkonsistent |
| **bls** | US-Arbeitsmarktdaten (NFP, CPI) | Sehr gut für Macro-Regime |
| **bluesky** | Social-Media-Posts | Niedrig — wenig Volumen vs Twitter |
| **cboe** | Options-Chains, VIX, VVIX, SKEW | **EXZELLENT** für Options-Trading |
| **coinmetrics** | Crypto on-chain Metrics | Niedrig wenn du keine Crypto handelst |
| **earnings_calendar** | Earnings-Termine | Standard, essentiell für PEAD |
| **edgar** | SEC-Filings (10-K, 10-Q, 8-K) | Sehr hoch für US-Equities |
| **finra** | TRACE Bond-Trade-Reports | Niedrig wenn keine Bonds |
| **fred** | Macro-Indikatoren | **EXZELLENT** für Recession-Probability |
| **gdelt_gcam** | Global Conflict Events | Sehr gut für Geopolitik |
| **newsapi** | News | Standard |
| **polygon** | Real-time Trades, Quotes | **EXZELLENT** für Equity-Day-Trading |
| **stooq** | Free Equity-Daten | Gut als yfinance-Fallback |
| **weather** | Weather-Daten | Mittel für Commodities |
| **wikipedia_views** | Suchvolumen je Symbol | **Underrated** — Retail-Sentiment-Proxy |
| **worldbank** | Country-Macro | Mittel für EM-Trading |
| **yfinance** | OHLCV, Fundamentals | Standard |

### Was fehlt

Schauen wir nochmal das Universum durch:

**Was du haben solltest, aber wahrscheinlich nicht hast:**

1. **Open-Meteo statt klassisches Weather-API** — Open-Meteo ist gratis (CC-BY-4.0), 80 Jahre Historie, 10km Auflösung, AGPLv3 für Self-Hosting. Wenn dein `weather_source.py` (245 LOC) eine kostenpflichtige Quelle nutzt, ist Migration zu Open-Meteo Aufwand-niedrig. Quote: "With over 80 years of hourly weather data available at a 10 kilometre resolution".

2. **NOAA Space Weather Prediction Center** — Solar-Flares können Satelliten-Funktion und FX-Märkte (Algorithmen!) beeinflussen. Marginalien für die meisten, aber für Tail-Risk-Hedging interessant.

3. **Polymarket / Kalshi Prediction Markets** — siehe B21. Markt-implizierte Wahrscheinlichkeiten für politische/wirtschaftliche Events.

4. **Quiver Quantitative API** — Aggregierter Insider-Trading + Politiker-Trades + Government-Contracts + Lobbying. Du hast `house_ptr` aber Quiver hat noch mehr (Senate, Pelosi-Tracker, etc.).

5. **Bundesbank/Eurostat Data Portal** — Du wohnst in Deutschland. EU-Macro-Daten sind oft besser für deine Trading-Universum als US-Daten allein. Bundesbank hat eine REST-API.

### Konkrete Empfehlung

Erstelle ein `data/sources/MISSING.md` mit einer Wishlist und prioritätsweise:

**Tier 1 (~5h pro Source):**
- Open-Meteo (Migration deines weather_source falls nötig)
- Bundesbank-Daten

**Tier 2 (~10h pro Source):**
- NOAA Space Weather
- Quiver (kostenpflichtig, aber starke Coverage)

**Tier 3 (Spekulativ):**
- Polymarket/Kalshi (kostenpflichtig, aber einzigartige Daten)

**Lizenz:** Open-Meteo ist AGPLv3 für Code, **CC-BY-4.0 für Daten** — also du musst Open-Meteo als Daten-Quelle attributieren in deinen Reports. Kommerzielle Self-Hosting möglich.

---

## A13. `events/crisis_alpha/`

### Was du hast

8 Module mit ca. 1500 LOC:
- `baskets.py` — vermutlich Asset-Baskets für Crisis-Long und Crisis-Short
- `context.py` — Markt-Kontext-Detection
- `entry.py` — Entry-Signale
- `exit_rules.py` — Exit-Regeln
- `gates.py` — Gates für Aktivierung
- `pipeline.py` (175 LOC) — Orchestrator
- `risk_budget.py` (191 LOC) — Risiko-Budget für Crisis-Trades
- `state_machine.py` (326 LOC) — größtes Modul, vermutlich State-Tracking ("normal", "stress", "crisis", "recovery")

### Was die Industry macht

Zwei dominante Crisis-Alpha-ETFs in 2026:

**TAIL (Cambria Tail Risk ETF)** — klassischer "buy puts on SPX, accept negative carry, hope for crash". Funktioniert in scharfen Crashes, blutet in normalen Markten.

**CAOS (Alpha Architect Tail Risk ETF)** — moderne Variante. Drei-Komponenten-Strategie:
1. **Protective Puts** auf SPX, OTM, für tatsächliche Crash-Konvexität
2. **Put-Spread-Overlays** statt nackte Puts — reduziert Carry
3. **Box-Spreads für Yield-Generation** — Box-Spread ist quasi-risk-free (T-Bill-Equivalent), generiert Carry, der die Put-Kosten teilweise deckt

Aus dem Investing.com-Review: Über 2.94 Jahre (Mar 2023 - Feb 2026) hat ein 80/20-CAOS-Portfolio ähnliche kumulative Returns wie 80/20-Bonds, aber **bessere Sharpe** (1.08 vs 1.05). Das ist der Heilige Gral — Crisis-Hedge ohne dauerhaften Carry-Drag.

### Konkrete Empfehlung

**Schritt 1: Audit deine `entry.py` und `exit_rules.py`.** Welche Indikatoren triggern Crisis-Mode? Wenn nur VIX > X, ist das wackelig. Industry-Standard 2026:
- VIX-Term-Structure-Inversion (siehe v3 §B9)
- VVIX > 110-120 (siehe B11)
- SKEW-Index > 135-140
- Credit-Spread-Widening (HY-IG-Spread)
- USD-Strength (DXY) bei Risk-Off

**Schritt 2: Refactoring `baskets.py` zu CAOS-style 3-Component:**
1. Long-Puts (deine Konvexität)
2. Put-Spreads (Carry-Reduktion)
3. Box-Spreads (Yield-Quelle)

**Aufwand:** 15-25h. Setzt voraus, dass du Optionen-Daten hast (siehe `cboe_source.py` — sollte es ermöglichen).

**Lizenz:** Strategien sind nicht patentiert, eigene Implementation OK.

---

## A14. `events/disclosures/fetch_house_ptr.py`

### Was du hast

**523 LOC** für House-PTR-Fetching. PTR = Periodic Transaction Report, die US-Kongressabgeordnete pflichtweise einreichen wenn sie Aktien kaufen/verkaufen. Berühmt durch Nancy Pelosi, deren Trading-Performance angeblich SPY schlägt.

Du hast also eine Pipeline, die:
- Die offizielle House-Disclosure-Website pollt (clerk.house.gov)
- PDFs runterlädt und parst (häufig OCR nötig)
- Trades extrahiert und in normalisierte Form bringt

Das ist **nicht trivial** — die Daten sind schmutzig, oft handschriftlich, manchmal Wochen verspätet eingereicht.

### Was es draußen gibt

**Quiver Quantitative** — kommerzielle API mit aggregierten Politiker-Trades. Hat höhere Coverage als nur House (auch Senate, Lobbying, Pelosi-Tracker, Government-Contracts).

**unusual-whales-api** — Aggregator mit Insider, Politiker, Options-Flow. Auch kostenpflichtig.

**capitol-trades.com** — kostenfreie Webseite, scrapebar.

### Konkrete Empfehlung

Du hast es selbst gebaut, das ist beachtlich. Aber:

1. **Senate-PTRs auch fetchen** — die Senatoren reichen separat ein. Die Performance-Statistiken (Pelosi etc.) bzw die "Politicians beat the market"-These bezieht sich auf BEIDE Kammern. Wenn du nur House hast, missst du die Hälfte.

2. **Aggregations-Signal-Konstruktion:**
   ```python
   # signals/political_consensus.py (neu, oder ergänze insider_cluster.py)
   
   def political_consensus_signal(symbol: str, lookback_days: int = 30) -> float:
       """
       Aggregate political trades for symbol.
       +1 = strong buy consensus, -1 = strong sell, 0 = mixed/none
       """
       trades = fetch_political_trades(symbol, lookback_days)
       if len(trades) == 0:
           return 0.0
       
       buys = sum(1 for t in trades if t.action == "BUY")
       sells = sum(1 for t in trades if t.action == "SELL")
       total = buys + sells
       if total == 0:
           return 0.0
       
       # weight by trader's historical alpha
       weighted_buys = sum(t.trader_alpha_score for t in trades if t.action == "BUY")
       weighted_sells = sum(t.trader_alpha_score for t in trades if t.action == "SELL")
       
       return (weighted_buys - weighted_sells) / max(weighted_buys + weighted_sells, 1.0)
   ```

3. **Backtest-Studie:** Schau nach, ob deine Pelosi-trades-Pipeline in den letzten 3 Jahren positives Alpha generiert hätte. Akademische Studien sind gemischt — manche zeigen +5% p.a. Alpha, andere finden es nach Transaktionskosten weg.

**Aufwand:** Senate-Add-on: 15-25h. Aggregations-Signal: 8-12h. Backtest-Studie: 8-12h.

---

## A15. `events/news/pipeline.py`

### Was du hast

**720 LOC** Hauptpipeline plus weitere Module:
- `clustering.py` (429 LOC) — News-Clustering
- `ner_extractor.py` (319 LOC) — Named-Entity-Recognition
- `dedupe_store.py` (214 LOC) — SQLite-basiertes Dedupe
- `trigger_scoring.py` (210 LOC) — Trigger-Scores für Trading
- `entities.py` (203 LOC) — Entity-Mapping
- `normalize.py` (220 LOC) — Normalisierung
- `baseline.py` (174 LOC) — Baseline-Tracking
- `fingerprint.py` (45 LOC) — vermutlich SimHash für Near-Dup
- `tfidf.py` (97 LOC) — TF-IDF
- `burst.py` (118 LOC) — Burst-Detection
- `fetch_gdelt.py` (250 LOC) — GDELT-Integration
- `fetch_acled.py` (65 LOC) — ACLED (Conflict-Data!)
- `fetch_rss.py` (137 LOC)

Insgesamt ein ausgereiftes News-System mit **Clustering, NER, Dedupe, Burst-Detection, Trigger-Scoring**. Die meisten Solo-Trader und sogar Hedge-Funds haben so etwas nicht.

### Was die Industry 2026 macht

State-of-the-Art ist:
1. **LLM-basierte Entity-Linking** statt klassische NER-Modelle. GPT-5/Claude-Opus erkennt Entitäten mit höherer Accuracy als spaCy oder finetuned BERT, kostet aber pro Artikel ~$0.001-0.01.

2. **Embedding-basiertes Clustering** statt TF-IDF — Sentence-Transformers oder OpenAI-Embeddings + HDBSCAN. Robuster gegen Synonyme und Paraphrasen.

3. **Knowledge-Graph-Persistence in Neo4j** — siehe v3 §A3. Du hast vermutlich nur In-Memory-NER-Output, kein persistenter Graph.

4. **Cross-Article-Coherence-Scoring** — wenn 5 Artikel zur gleichen Story leicht widersprüchliche Fakten haben, ist die Story unsicher.

5. **Multi-language Sentiment** — du hast `news_language.py`, vermutlich lange-detection. Dann musst du multilingual sentimenten — FinBERT-DE, FinBERT-EN, FinBERT-zh getrennt.

### Konkrete Empfehlung

Du hast schon SEHR viel. Ich würde nicht alles umbauen. Aber zwei targeted Erweiterungen:

**Erweiterung 1: LLM-Entity-Linking als Augmentation deines NER (~15-25h)**

```python
# events/news/llm_entity_linker.py (neu)

class LLMEntityLinker:
    """Augment classical NER with LLM-based entity linking."""
    
    def __init__(self, model: str = "claude-opus-4-7"):
        self.client = Anthropic()
        self.model = model
        self.cache = {}  # SHA256(text) -> result
    
    def link(self, text: str, classical_entities: list[Entity]) -> list[Entity]:
        """Use LLM to (a) verify classical entities, (b) find missed ones, (c) link to tickers."""
        cache_key = hashlib.sha256(text.encode()).hexdigest()
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        prompt = f"""Given this news article and the entities extracted by a classical NER:

Article: {text[:2000]}
Classical NER entities: {classical_entities}

Output a JSON list of entities. For each:
- name: canonical entity name
- type: ORG/PERSON/LOCATION/EVENT/PRODUCT
- ticker: stock ticker if applicable, else null
- confidence: 0.0-1.0
- linked_to_article_topic: bool (is this a primary subject vs incidental mention?)

Only output valid JSON, no preamble."""
        
        response = self.client.messages.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
        )
        
        entities = json.loads(response.content[0].text)
        self.cache[cache_key] = entities
        return entities
```

**Erweiterung 2: Embedding-based Clustering als Drop-in für TF-IDF (~10-15h)**

```python
# events/news/embedding_clustering.py (neu)

from sentence_transformers import SentenceTransformer
import hdbscan

class EmbeddingNewsClustering:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")  # MIT, free
    
    def cluster(self, articles: list[NewsArticle]) -> list[Cluster]:
        texts = [a.title + " " + a.snippet for a in articles]
        embeddings = self.model.encode(texts)
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=2,
            metric="cosine",
            cluster_selection_method="eom",
        )
        labels = clusterer.fit_predict(embeddings)
        
        clusters = {}
        for article, label in zip(articles, labels):
            if label == -1:
                continue  # noise
            clusters.setdefault(label, []).append(article)
        return [Cluster(id=k, articles=v) for k, v in clusters.items()]
```

**Lizenz:** sentence-transformers Apache 2.0, hdbscan BSD-3, Anthropic API kommerziell.

---

## A16. `signals/regime/hmm_posterior.py`

### Was du hast

**108 LOC**. Das ist klein. Vermutlich ein Wrapper über deine `ml/regime_hmm.py` (416 LOC) und `risk/regime_models.py` (719 LOC), der die Posterior-State-Probabilities als trainable Signal exposiert.

### Was wir nie diskutiert haben

Posterior-Probabilities als Inputs für andere Signale ist **regime-conditional modeling** — du machst eine Strategie, die auf Bull-Markt geht, eine andere auf Crisis. Klassisch:

```python
# signals/composite_regime_aware.py (neu, ergänzt composite_score.py 386 LOC)

def regime_aware_signal(
    features: pd.DataFrame,
    posterior: pd.DataFrame,  # cols: P(bull), P(neutral), P(bear)
) -> pd.Series:
    """Weighted average of regime-specific signals by regime posterior."""
    bull_signal = momentum_strategy(features)       # works in bull
    neutral_signal = mean_reversion_strategy(features)  # works in chop
    bear_signal = -1 * momentum_strategy(features)  # invert in bear
    
    weighted = (
        posterior["P(bull)"] * bull_signal +
        posterior["P(neutral)"] * neutral_signal +
        posterior["P(bear)"] * bear_signal
    )
    return weighted
```

**Was du checken solltest:** Wird `hmm_posterior.py` aktuell von deinen Composite-Signalen wirklich genutzt, oder hängt er als isolierter Output rum?

**Aufwand für Integration:** 5-10h.

---

## A17. `features/triple_barrier.py` (Revisited)

In v3 hatte ich angenommen, das Modul sei klein. **Ist es nicht — 307 LOC.** Damit ist es vermutlich schon eine ordentliche Implementierung.

Trotzdem prüfe:
1. Nutzt es **vol-targeted barriers** (siehe v3 §A7)?
2. Nutzt es **CUSUM-Filter** für Event-Sampling?
3. Hat es **Meta-Labeling-Hooks** (Verbindung zu `signals/meta_model.py` 453 LOC)?

Wenn alle 3 Ja: super, lass es. Wenn 2 oder weniger: das v3-§A7-Refactoring lohnt sich.

---

## A18. `features/market_breadth.py`

### Was du hast

**665 LOC**. Das ist das **drittgrößte Feature-Modul** nach `ta_features.py` (831) und `altdata_earnings_insider_factors.py` (714). Beim diesen Volumen vermute ich:

- Advance/Decline Line
- McClellan Oscillator
- Percentage of stocks above MA50/MA200
- New Highs / New Lows
- Volatility-of-individual-stocks vs Index
- TICK index
- TRIN (Arms Index)
- Volume-Ratio (Up Volume / Down Volume)

### Was wir hätten anschauen sollen

**Hindenburg Omen** — Compositionssignal aus Market-Breadth-Indikatoren, das angeblich Marktcrashs signalisiert (umstritten). Die Definition:
1. New 52w highs UND new 52w lows BEIDE > 2.8% des NYSE-Volumens
2. NYSE-Index ist über 50d-Moving-Average
3. McClellan Oscillator ist negativ
4. New 52w highs ≤ 2x new 52w lows

Hindenburg ist umstritten (viele False Positives), aber als ZUSÄTZLICHER Crisis-Indikator interessant.

**Composite Bull-Bear-Indicator (CBBI)** — Investmentmanagement-Standard von State-Street und Renaissance. Aggregiert ~10 Breadth-Indikatoren zu einem 0-100-Score.

**Konkrete Empfehlung:** Wahrscheinlich hast du das Meiste schon. Quick-Check: ist Hindenburg als Composite drin? Ist McClellan-Summation-Index drin?

---

## A19. `signals/crash_prediction.py`

### Was du hast

**577 LOC**. Das deutet auf eine Multi-Faktor-Crash-Prediction hin. Vermutlich basierend auf:
- VIX-Term-Structure
- Yield-Curve
- Credit-Spreads
- Tech-Indicators (RSI-Divergenz)
- Vielleicht ML-Modell mit historischen Crashes als positive Class

### Was die Industry macht

**Robert Shiller's "CAPE-Ratio"** — Cyclically Adjusted P/E. Hoch korreliert mit 10-Jahres-Returns, schlecht für Timing.

**Buffett-Indikator** — Total-Market-Cap / GDP. Über 200% = "Crash-Wahrscheinlichkeit erhöht". Aktuell historisch hoch.

**Hussman Margin-Adjusted CAPE** — bessere Variante des CAPE.

**Probabilistic Crash-Models** akademisch: Sornette's LPPLS-Modell (Log-Periodic Power Law). Detektiert Bubbles vor dem Crash, hat 2008 und 2000 ex-post gut funktioniert. Code: `Boulder-Investment-Technologies/lppls` (MIT-Lizenz).

### Konkrete Empfehlung

Wenn dein crash_prediction.py noch nicht LPPLS hat, ist das ein interessanter Add-on:

```python
# signals/lppls_crash.py (neu, ~100 LOC)

from lppls import lppls
import numpy as np

class LPPLSCrashDetector:
    """Log-Periodic Power Law Singularity bubble/crash detector."""
    
    def __init__(self, fit_window: int = 252, max_searches: int = 50):
        self.fit_window = fit_window
        self.max_searches = max_searches
    
    def fit_and_score(self, prices: np.ndarray) -> dict:
        """Returns crash probability + estimated tc (critical time)."""
        log_prices = np.log(prices[-self.fit_window:])
        time_index = np.arange(len(log_prices))
        
        model = lppls.LPPLS(observations=np.array([time_index, log_prices]))
        results = model.fit(self.max_searches)
        
        # Confidence indicator from Sornette et al
        ci = model.compute_indicators(results)
        
        return {
            "tc_estimate": results["tc"],  # critical time (crash time)
            "crash_confidence": ci["pos_conf"],  # 0-1
            "time_to_crash_days": results["tc"] - len(log_prices),
        }
```

**Aufwand:** 8-12h.
**Lizenz:** lppls ist MIT.

---

## A20. `signals/multifactor_signal.py`

### Was du hast

**908 LOC**. Das ist dein zweitgrößtes Signal-Modul. Vermutlich kombiniert es Value, Momentum, Quality, Low-Vol, etc. zu einem Composite-Score.

### Was Bridgewater/AQR/2Sigma machen

State-of-the-Art Multi-Factor:

1. **Risk-Premia-Adjusted Combination** — nicht einfach Sum-of-Z-Scores, sondern jedes Faktor-Premium gewichtet nach erwartetem Sharpe.

2. **Time-Varying Factor Weights** — Momentum funktioniert in Trends, Value in Mean-Reversion-Märkten. Regime-bedingt umgewichten.

3. **Factor Crowding Detection** — wenn alle Hedge-Funds in Long-Momentum sind, ist Momentum gefährdet ("Quant-Quake 2007"). Du hast `risk/crowding_detector.py` — das ist die richtige Ebene.

4. **Cost-Aware Factor Construction** — beim Standard-Approach (long top quintile, short bottom quintile) ignorieren viele die Liquidität. Cost-aware: gewichte Stocks nach (Signal-Strength) / (Bid-Ask-Spread × Daily-Volume-Constraint).

### Konkrete Empfehlung

Audit deines `multifactor_signal.py` gegen diese 4 Kriterien:

1. Risk-Premia-adjusted? — vermutlich nicht ohne Forschung
2. Time-Varying via Regime? — checke ob Verbindung zu `signals/regime/`
3. Crowding-aware? — checke ob Verbindung zu `risk/crowding_detector.py`
4. Cost-aware? — du hast `portfolio/cost_aware_optimizer.py`. Wird das wirklich angeschlossen?

**Aufwand für ein Audit + Anpassungen:** 15-25h.

---

# Teil B — Wirklich neue Bereiche

## B11. VVIX statt VIX als Tail-Risk

### Was du hast

`signals/tail_risk_hedge.py` (173 LOC). Klein. Vermutlich VIX-basiert.

### Warum VVIX viel besser ist

**VVIX** ist die "Volatility of Volatility" — IV der VIX-Optionen. Während VIX die erwartete 30-Tage-SPX-Volatilität misst, misst VVIX die erwartete Bewegung der VIX selbst.

Konkrete Industry-Erkenntnisse 2026:

1. "an increase in the volatility-of-volatility as measured by the VVIX index raises current prices of tail risk hedging options, such as S&P 500 puts and VIX calls, and lowers their subsequent returns over the next three to four weeks" — heißt: wenn VVIX hoch ist, ist Tail-Hedge teuer und renditeschwach. **VVIX timing > VIX timing.**

2. "Historically, the VVIX has maintained a long-term mean near 90 to 100 points. When the index breaches the 110 to 120 threshold, it typically indicates an environment of extreme uncertainty".

3. "The CBOE SKEW Index measures the perceived tail risk in the S&P 500. It tracks the price of out-of-the-money put options relative to at-the-money options. A high SKEW reading (typically above 135-140) indicates that investors are paying a high premium for 'crash protection'".

4. **Volmageddon Februar 2018:** "the VVIX began ascending well before the VIX itself experienced its historic doubling, providing a lead time of several trading sessions for observant risk managers".

### Konkrete Empfehlung

Erweitere `signals/tail_risk_hedge.py` um VVIX-Reading. Implementation-skizze:

```python
# signals/tail_risk_vvix.py (neu, ergänzt tail_risk_hedge.py)

import yfinance as yf
import pandas as pd

class VVIXTailRiskSignal:
    """
    Tail-risk warning system based on VVIX, VIX-Term-Structure, and SKEW.
    Returns: 0 (calm) / 1 (elevated) / 2 (high) / 3 (extreme).
    """
    
    THRESHOLDS = {
        "vvix": {"calm": 90, "elevated": 100, "high": 110, "extreme": 130},
        "skew": {"calm": 130, "elevated": 135, "high": 140, "extreme": 150},
    }
    
    def fetch_data(self) -> pd.DataFrame:
        """Pull VIX, VVIX, VIX9D, VIX3M, SKEW from CBOE/Yahoo."""
        symbols = {
            "VIX": "^VIX",
            "VVIX": "^VVIX",
            "VIX9D": "^VIX9D",
            "VIX3M": "^VIX3M",
            "SKEW": "^SKEW",
        }
        df = pd.DataFrame()
        for name, sym in symbols.items():
            df[name] = yf.Ticker(sym).history(period="2y")["Close"]
        df["term_structure"] = df["VIX"] - df["VIX3M"]  # >0 = backwardation
        df["short_inversion"] = df["VIX9D"] - df["VIX"]  # >0 = panic
        return df
    
    def regime(self, latest: pd.Series) -> dict:
        """Classify current tail-risk regime."""
        vvix_level = self._classify(latest["VVIX"], "vvix")
        skew_level = self._classify(latest["SKEW"], "skew")
        backwardation = latest["term_structure"] > 0
        
        # Aggregate score
        levels = ["calm", "elevated", "high", "extreme"]
        scores = {"calm": 0, "elevated": 1, "high": 2, "extreme": 3}
        max_score = max(scores[vvix_level], scores[skew_level])
        if backwardation:
            max_score += 1
        max_score = min(max_score, 3)
        
        return {
            "regime": levels[max_score],
            "vvix": latest["VVIX"],
            "vvix_level": vvix_level,
            "skew": latest["SKEW"],
            "skew_level": skew_level,
            "backwardation": backwardation,
            "term_structure": latest["term_structure"],
            "score_0_3": max_score,
            # Action recommendation
            "hedge_attractive": (max_score == 0),  # cheap hedge in calm
            "hedge_too_expensive": (max_score >= 2),  # ride out, don't add hedges
        }
    
    def _classify(self, value: float, indicator: str) -> str:
        thresh = self.THRESHOLDS[indicator]
        if value < thresh["elevated"]:
            return "calm"
        elif value < thresh["high"]:
            return "elevated"
        elif value < thresh["extreme"]:
            return "high"
        return "extreme"
```

**Aufwand:** 5-8h.
**Was du gewinnst:** Hedge-Timing wird sehr viel besser. **Buy hedges when VVIX is calm, NOT when VVIX has already spiked.**

---

## B12. Causal Inference

### Wirklich neuer Bereich

In v1-v3 nie angesprochen. **DoWhy / EconML / CausalPy / CausalML** sind die führenden Causal-Inference-Libraries.

### Wofür du das brauchen würdest

Du machst News-zu-Trade-Inferenz: "X Nachricht erschien, Aktie Y bewegt sich nach Z%". Das ist **korrelativ**, nicht kausal.

Causal-Inference-Frage wäre: "**Wenn** ich diese News mit X% Sentiment **gesehen** hätte, **was wäre meine Action gewesen** und **wie viel Alpha** hätte das gebracht?"

Konkrete Use-Cases für dich:

1. **Treatment-Effekt von News-Triggers messen.** Du hast `news_trade_attribution.py`. Aber: korreliert News-Trigger mit Trade-Performance, weil die News kausal war, oder weil andere konfundierende Variablen (Marktregime, Sektor-Rotation) sowohl News als auch Performance beeinflussen?

2. **Heterogene Treatment-Effekte.** Vielleicht funktionieren News-Trigger nur in bestimmten Sektoren oder Marktregimen. EconML's Causal Forest schätzt das automatisch.

3. **Counterfactual Reasoning für Strategien.** "Wäre meine Strategie auch profitabel gewesen, wenn ich diese eine Whipsaw-Episode anders gehandelt hätte?" Synthetic Control bzw. Counterfactual-Methoden.

### Was es draußen gibt

**`py-why/dowhy`** — Microsoft, Apache 2.0. Pearl-style Causal-DAG-Framework. Workflow:
1. Modell aufstellen mit Causal-DAG
2. Identifikation des kausalen Effekts (Backdoor/Front-door/Instrument-Variable)
3. Schätzung mit verschiedenen Methoden (Propensity Score, DML, IPW)
4. Refutation-Tests (random common cause, placebo treatment)

**`py-why/EconML`** — Microsoft ALICE, Apache 2.0. Speziell auf **Heterogene Treatment-Effekte** mit ML-basierten Schätzern: Double Machine Learning, Causal Forest, Deep IV, Meta-Learners.

**`uber/causalml`** — Uber, Apache 2.0. Fokus auf **Uplift-Modeling** — wer profitiert am meisten von einer Intervention?

**`pymc-labs/CausalPy`** — PyMC Labs, Apache 2.0. Quasi-experimentelle Designs: Difference-in-Differences, Regression-Discontinuity, Synthetic Control.

### Konkrete Empfehlung

**Phase 1 (10-15h): DoWhy-Tutorial durcharbeiten und ein Pilot-Setup in `qa/causal_validation.py`** schreiben:

```python
# qa/causal_validation.py (neu)

import dowhy
from dowhy import CausalModel
import pandas as pd

def estimate_news_trigger_effect(
    trades_df: pd.DataFrame,  # cols: trade_id, has_news_trigger, return, sector, vol_regime, market_cap
) -> dict:
    """
    Estimate the causal effect of news triggers on trade returns.
    Treatment = has_news_trigger (binary)
    Outcome = return
    Confounders = sector, vol_regime, market_cap
    """
    model = CausalModel(
        data=trades_df,
        treatment="has_news_trigger",
        outcome="return",
        common_causes=["sector", "vol_regime", "market_cap"],
    )
    
    # Identify causal effect
    identified = model.identify_effect(proceed_when_unidentifiable=True)
    
    # Estimate (multiple methods for robustness)
    estimates = {}
    for method in ["backdoor.propensity_score_matching", 
                   "backdoor.linear_regression",
                   "backdoor.econml.dml.LinearDML"]:
        try:
            est = model.estimate_effect(identified, method_name=method)
            estimates[method] = est.value
        except Exception as e:
            estimates[method] = f"failed: {e}"
    
    # Refutation
    refutations = {}
    if "backdoor.linear_regression" in estimates:
        ref = model.refute_estimate(
            identified,
            model.estimate_effect(identified, method_name="backdoor.linear_regression"),
            method_name="random_common_cause",
        )
        refutations["random_common_cause"] = {
            "original_effect": ref.estimated_effect,
            "new_effect": ref.new_effect,
            "p_value": ref.refutation_result["p_value"] if ref.refutation_result else None,
        }
    
    return {
        "estimates": estimates,
        "refutations": refutations,
        "interpretation": "If estimates are similar across methods AND refutation p-value > 0.05, causal effect is robust.",
    }
```

**Phase 2 (15-25h): EconML's Causal Forest für Heterogene Effekte** — du findest raus, wo (welcher Sektor, welches Regime) die News-Trigger am stärksten wirken.

**Lizenz:** DoWhy/EconML/CausalML/CausalPy alle Apache 2.0 — frei nutzbar.

---

## B13. Graph Neural Networks

### Wirklich neuer Bereich

Du hast Stocks als unabhängige Time-Series modelliert. GNN modellieren sie als **Knoten in einem Graph** — mit Kanten für Sektor, Lieferketten, Customer-Supplier-Beziehungen, Korrelations-Cluster.

### Was es draußen gibt

**`timothewt/SP100AnalysisWithGNNs`** — S&P100-Analyse mit:
- Sektor- und Fundamentals-basierter Graph-Konstruktion
- PyTorch Geometric für Custom-Datasets
- T-GCN und A3T-GCN (Temporal Graph Convolutional Networks)
- Spatio-Temporal GNN für Forecasting
- Deep Graph Clustering für Stock-Cluster

**`ZihanChen1995/ChatGPT-GNN-StockPredict`** — Innovativer Ansatz:
1. ChatGPT extrahiert Stock-Beziehungen aus News
2. Daraus täglich aktualisierte Graph-Struktur
3. GNN + LSTM auf der dynamischen Graph
4. **Schlägt klassische Deep-Learning-Benchmarks** auf DOW30

**`kyawlin/GNN-finance`** und **`jwwthu/GNN4Fintech`** — Curated Lists der wichtigsten Papers/Repos.

### Wofür GNN für dich relevant wäre

Du hast `features/supply_chain_features.py` (358 LOC) und `features/correlation_features.py` (305 LOC). Das sind eigentlich **Graph-Inputs** in Tabellen-Form. GNN würden direkt auf der Graph-Struktur operieren und sind oft Performance-überlegen.

### Konkrete Empfehlung

**Niedrige Priorität, weil viel ML-Engineering.** Aber als langfristiger Research-Path:

```python
# signals/gnn_stock_relations.py (neu, ~300 LOC)

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader
import pandas as pd

class StockGCN(torch.nn.Module):
    def __init__(self, num_features: int, hidden: int = 64):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.predictor = torch.nn.Linear(hidden, 1)
    
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return self.predictor(x)


def build_stock_graph(
    symbols: list[str],
    correlation_window: int = 60,
    correlation_threshold: float = 0.5,
    sector_map: dict[str, str] = None,
) -> Data:
    """Build stock graph with edges for high-correlation pairs + same-sector."""
    returns = fetch_returns(symbols, window=correlation_window)
    corr = returns.corr()
    
    edges = []
    for i, s1 in enumerate(symbols):
        for j, s2 in enumerate(symbols[i+1:], i+1):
            # Edge if high correlation OR same sector
            if abs(corr.loc[s1, s2]) > correlation_threshold:
                edges.append([i, j])
                edges.append([j, i])  # undirected
            elif sector_map and sector_map.get(s1) == sector_map.get(s2):
                edges.append([i, j])
                edges.append([j, i])
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    # Node features: recent returns, vol, vol-of-vol, momentum
    node_features = build_features(symbols)  # shape: (n_stocks, n_features)
    
    return Data(
        x=torch.tensor(node_features, dtype=torch.float),
        edge_index=edge_index,
    )
```

**Aufwand:** 40-80h für ein erstes GNN-System (Forschung + Engineering).
**Lizenz:** PyTorch Geometric MIT.

**Vorsicht:** GNN-Forschungs-Papers haben oft **Schwierigkeiten zu replizieren** (Trading-Daten sind sehr noisy). Erwarte realistisch +0.1-0.3 Sharpe nach Transaktionskosten, nicht +1.0.

---

## B14. ClickHouse / TimescaleDB

### Wirklich neuer Bereich

Du nutzt **DuckDB + Parquet** (`data/feature_store.py`). Das ist exzellent für Batch-Processing. Aber für **Real-Time Tick-Data** ist es nicht optimal.

### Was die Industry 2026 macht

"Hot Data (Trading): Use InfluxDB or TimescaleDB for real-time market data capture and sub-second queries. Low write latency and time-series optimized queries are critical here. Warm Data (Analytics): Use ClickHouse for analytical workloads, backtesting, and reporting. According to benchmarks, ClickHouse processes complex analytical queries 3-10x faster than TimescaleDB."

Konkrete Zahlen aus 2026-Benchmarks:

- ClickHouse: 2-3 Mio Datenpunkte/Sek bei batched Inserts, 10-30x bessere Kompression als TimescaleDB
- ClickHouse-Queries bei 280ms vs Sekunden bei TimescaleDB für komplexe Aggregationen
- TimescaleDB: 100K-500K points/Sek, aber besser für High-Frequency Point-Lookups (10ms vs 15ms)

### Wofür du das brauchen würdest

Falls du irgendwann von Daily auf Intraday umsteigst (1-Minute-Bars oder Tick-Data), wird DuckDB+Parquet zu langsam. ClickHouse ist dann der nächste Schritt.

Kurz: **Niedrige Priorität wenn du Daily-Trader bist.**

### Konkrete Empfehlung

Wenn nicht jetzt, dann als Backup-Plan dokumentieren:

```yaml
# docker-compose.clickhouse.yml (für später)
services:
  clickhouse:
    image: clickhouse/clickhouse-server:25.3
    container_name: assembled-clickhouse
    ports: ["8123:8123", "9000:9000"]
    volumes:
      - ./data/clickhouse:/var/lib/clickhouse
    ulimits:
      nofile:
        soft: 262144
        hard: 262144
```

**Aufwand für Migration falls nötig:** 30-50h.
**Lizenz:** ClickHouse Apache 2.0, TimescaleDB Apache 2.0 (Community), kommerzielle Features unter TSL.

---

## B15. Event Sourcing & CQRS

### Was du teilweise hast

`events/replayer.py`, `events/store.py`, `events/schema.py` — das ist ein **Event-Store-Skelett**. Du hast also Anfänge von Event-Sourcing gemacht.

Du hast aber nicht (vermutlich):
- Vollständiges CQRS (Command-Query Separation)
- Append-Only-Event-Log mit Replay-Capability für Debugging
- Materialized Views für schnelle Queries

### Was die Industry macht

**Event Sourcing-Prinzipien:**
1. Statt aktueller State zu speichern, speichere die **Geschichte aller Events**
2. State ist eine Funktion `f(events) -> state`
3. Audit-Trail ist gratis (alle Änderungen sind im Event-Log)
4. Time-Travel-Debugging: "Was war der State am 2026-03-15 14:23:00?"

**CQRS:**
- **Commands** ändern State (Order placen, Trade ausführen)
- **Queries** lesen State (aktuelle Position, Performance)
- **Separate Modelle** für Read und Write

### Was es für Python gibt

**`pyeventsourcing/eventsourcing`** — pure-Python Library für Event Sourcing. BSD-3.

**Beispiele:**
- `ifnesi/python-kafka-microservices` — CQRS+Event Sourcing mit Kafka
- `marcosvs98/cqrs-architecture-with-python` — FastAPI + MongoDB + Redis + Kafka

### Wofür das relevant wäre

Du hast `accounting/ledger.py` (610 LOC) und `accounting/evidence_pack.py` (1147 LOC). Wenn du diese Module zu einem **echten Event-Sourcing-Pattern** machst, hast du:

1. **Tamper-evident Audit-Trail** — jeder Trade ist ein unveränderbares Event
2. **Replay-Capability** — du kannst dein Portfolio zu jedem Zeitpunkt rekonstruieren
3. **Compliance-Ready** — BaFin/§147 AO-Aufbewahrung wird trivial

### Konkrete Empfehlung

```python
# events/trading_events.py (neu, ergänzt accounting/ledger.py)

from pyeventsourcing import Aggregate, event
from datetime import datetime
from decimal import Decimal

class Trade(Aggregate):
    @event("OrderPlaced")
    def __init__(self, symbol: str, qty: Decimal, side: str, intent: str):
        self.symbol = symbol
        self.qty = qty
        self.side = side
        self.intent = intent
        self.fills = []
        self.status = "PENDING"
    
    @event("OrderFilled")
    def fill(self, qty: Decimal, price: Decimal, ts: datetime, broker_order_id: str):
        self.fills.append({"qty": qty, "price": price, "ts": ts, "broker_order_id": broker_order_id})
        if sum(f["qty"] for f in self.fills) >= self.qty:
            self.status = "FILLED"
    
    @event("OrderCancelled")
    def cancel(self, reason: str):
        self.status = "CANCELLED"
        self.cancel_reason = reason


# usage
trade = Trade("AAPL", Decimal("100"), "BUY", "momentum_signal_v3")
trade.fill(Decimal("50"), Decimal("180.25"), datetime.now(), "alpaca-12345")
trade.fill(Decimal("50"), Decimal("180.50"), datetime.now(), "alpaca-12346")

# All events are stored, can be replayed
events = trade._collect_()
# [OrderPlaced, OrderFilled, OrderFilled]
```

**Aufwand:** 25-40h für ein vollständiges Event-Sourcing-Refactoring deiner Trading-Pipeline.
**Lizenz:** pyeventsourcing BSD-3.

---

## B16. Auto-Feature-Engineering tsfresh / featuretools

### Was du hast

Du hast 9441 LOC manuell gebaute Features in `features/`. Beeindruckend.

### Was tsfresh / featuretools machen

**`blue-yonder/tsfresh`** — extrahiert **automatisch ~750 Features** aus Time-Series:
- Statistische Momente
- Frequenz-Domain-Features (FFT)
- Entropy-Maße
- Symmetry-Tests
- Quantile
- Number of peaks
- Time reversal symmetry statistic
- ...

Plus **automatisches Feature-Selection** via Hypothesis-Testing — wirft irrelevante Features raus.

**`alteryx/featuretools`** — Deep Feature Synthesis. Generiert automatisch Features durch Aggregation über relationale Datasets.

### Wofür das relevant wäre

**Zwei mögliche Wege:**

**Weg A: Augmentation deiner manuellen Features.** Lass tsfresh über deine Returns/Volume-Time-Series laufen, generiere 750 Features, dann nutze tsfresh's Filter um die signifikanten zu selektieren. Du bekommst potentiell 50-100 NEUE alpha-positive Features, die du selbst nicht gebaut hast.

**Weg B: Replacement.** Manuelle Features sind kuratierter, aber langsam. tsfresh ist generic, aber komplett. Hybrid: dein Feature-Stack als Kuration, plus tsfresh-augmented "exploration set".

### Konkrete Empfehlung

```python
# features/tsfresh_augmentation.py (neu)

from tsfresh import extract_features, select_features
from tsfresh.feature_extraction import EfficientFCParameters
import pandas as pd

class TsfreshFeatureAugmenter:
    """Auto-generate features using tsfresh, then filter by significance."""
    
    def augment(
        self,
        prices: pd.DataFrame,  # cols: ts, symbol, close
        targets: pd.Series,     # forward returns
    ) -> pd.DataFrame:
        # tsfresh expects long format with id column
        long_format = prices.melt(id_vars=["ts", "symbol"], value_vars=["close"])
        
        # Extract ~750 features (use EfficientFCParameters for speed)
        features = extract_features(
            long_format,
            column_id="symbol",
            column_sort="ts",
            default_fc_parameters=EfficientFCParameters(),
            n_jobs=4,
        )
        
        # Filter by relevance to target
        filtered = select_features(features, targets, fdr_level=0.05)
        return filtered
```

**Aufwand:** 10-15h für eine Pilot-Integration.
**Lizenz:** tsfresh ist MIT, featuretools ist BSD-3.

---

## B17. Anomalie-Detection

### Wirklich neuer Bereich

Du hast `features/change_point_detection.py` (162 LOC) — das ist Strukturbruch-Detection. Aber **Anomalie-Detection im Trading-Kontext** ist anderes:

- Detect ungewöhnliche Trade-Größen (Fat-Finger, Front-Running)
- Detect Market-Manipulation (Pump-and-Dump-Patterns)
- Detect Datenqualitäts-Probleme (Outlier-Quotes von Broker)
- Detect Strategie-Drift (deine eigene Strategie läuft anders als historisch)

### Was es draußen gibt

**`yzhao062/pyod`** — der Standard. 60+ Algorithmen:
- Isolation Forest
- Local Outlier Factor (LOF)
- One-Class SVM
- AutoEncoder-basierte
- HBOS (Histogram-based)
- Angle-based Outlier Detection
- Copula-based COPOD (sehr gut für Tabular!)

In v3 hatte ich PyOD/alibi-detect als Future-Topic erwähnt. Hier die konkrete Anwendung.

**`xuhongzuo/DeepOD`** — Deep-Learning-spezifisch. Anomaly Transformer, TimesNet, DCdetector.

**`thedatumorg/TSB-AD`** — Time-Series-Anomaly-Detection-Benchmark. 30 Algorithmen, NeurIPS 2024 paper.

### Konkrete Empfehlung

```python
# qa/anomaly_detection.py (neu)

from pyod.models.iforest import IForest
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
import pandas as pd

class TradeAnomalyDetector:
    """
    Detect anomalies in trades / signals / market data.
    Uses ensemble of three PyOD detectors.
    """
    
    def __init__(self):
        self.detectors = {
            "iforest": IForest(contamination=0.01),
            "copod": COPOD(contamination=0.01),
            "ecod": ECOD(contamination=0.01),
        }
        self.fitted = False
    
    def fit(self, baseline_df: pd.DataFrame):
        """Fit on historical normal data."""
        for det in self.detectors.values():
            det.fit(baseline_df)
        self.fitted = True
    
    def score(self, current_df: pd.DataFrame) -> pd.DataFrame:
        """Score new data; flag if majority of detectors say anomaly."""
        scores = {}
        flags = {}
        for name, det in self.detectors.items():
            scores[name] = det.decision_function(current_df)
            flags[name] = det.predict(current_df)
        
        scores_df = pd.DataFrame(scores)
        flags_df = pd.DataFrame(flags)
        scores_df["consensus_anomaly"] = (flags_df.sum(axis=1) >= 2).astype(int)
        return scores_df
```

**Aufwand:** 12-20h.
**Lizenz:** PyOD BSD-2.

**Was du gewinnst:** Operational Risk-Reduktion. Wenn deine Strategie plötzlich 10x mehr Trades als üblich generiert, weißt du es **bevor** echte Schäden entstehen.

---

## B18. Reproducibility-Stack

### Was du hast

`certify/` (324 LOC), `accounting/evidence_pack.py` (1147 LOC). Du baust schon eigene Reproducibility/Audit-Logs. Sehr gut.

### Was die Industry zusätzlich nutzt

**DVC (Data Version Control)** — `iterative/dvc`, Apache 2.0. Git-style Versionierung für Daten und ML-Modelle. Konkret: deine Parquet-Files in `data/` werden nicht in Git committed, aber DVC trackt Version + Hash + Storage-Location.

**MLflow** — `mlflow/mlflow`, Apache 2.0. Experiment-Tracking, Model-Registry, Reproducible Pipelines. Du hast `experiments/batch_config.py` (401 LOC) — das könnte zu MLflow migriert werden.

**OpenLineage** — Metadata-Standard für Daten-Lineage. Definiert: welcher Job hat welche Inputs gelesen, welche Outputs erzeugt. Tooling-Integration mit Airflow, dbt, Spark.

**W3C PROV** — älterer Standard für Provenance. Mehr im Audit-Bereich verbreitet.

### Konkrete Empfehlung

**Phase 1 (4-6h): DVC für deine `data/` und `output/` Verzeichnisse.**

```bash
# init DVC
cd /home/claude/Assembled-Trading-AI-fresh
dvc init
dvc remote add -d s3-storage s3://my-bucket/dvc/

# track data
dvc add data/parquet/
git add data/parquet.dvc .gitignore
git commit -m "Track data with DVC"

# checkpoint after each trading day
dvc add output/positions/2026-04-27.parquet
```

**Phase 2 (8-15h): MLflow für deine Experiments.**

```python
# experiments/mlflow_integration.py (neu, ergänzt batch_config.py)

import mlflow
import mlflow.sklearn

with mlflow.start_run(run_name="multifactor_v2_2026Q2"):
    mlflow.log_params({
        "lookback_days": 252,
        "rebalance_frequency": "weekly",
        "factor_weights": {"value": 0.3, "momentum": 0.4, "quality": 0.3},
    })
    
    # ... train, backtest ...
    
    mlflow.log_metrics({
        "sharpe_oos": 1.42,
        "max_dd_oos": -0.12,
        "calmar": 1.8,
    })
    mlflow.log_artifact("backtest_report.html")
    mlflow.sklearn.log_model(meta_model, "meta_model")
```

**Aufwand:** 12-21h gesamt.
**Lizenz:** DVC Apache 2.0, MLflow Apache 2.0.

---

## B19. Open-Meteo + NOAA Alternative Data

### Was du hast

`data/sources/weather_source.py` (245 LOC). Vermutlich eine API.

### Was es 2026 gibt

**`open-meteo/open-meteo`** — kostenlos für non-commercial, 80 Jahre Historie, 10km-Auflösung, AGPLv3-Code, CC-BY-4.0-Daten. Self-Hosted für unbegrenzte API-Calls.

**Spire Global** — kommerzielle satellite-based Daten:
- 300.000+ Schiffe getrackt (Maritime)
- Atmosphärische Daten weltweit
- Aviation-Tracking

**Polymarket Weather Markets** — 463+ aktive Wetter-Markets mit $1.3M Volumen. Markt-implizite Wetter-Wahrscheinlichkeiten für Trading-Strategien.

### Wofür das für dich relevant wäre

Du hast `features/shipping_features.py` (104 LOC) und `features/supply_chain_features.py` (358 LOC). Wetter beeinflusst:

1. **Commodity-Preise** — Mais, Weizen, Kakao, Kaffee bei Dürre/Frost
2. **Energiepreise** — Heizöl, Gas im Winter
3. **Versicherungs-Aktien** — Hurrikan-Schäden
4. **Reise-Aktien** — Wetter beeinflusst Buchungen

Wenn dein `weather_source.py` aktuell nur einfache Daten liefert, ist **Open-Meteo** ein massiver Upgrade ohne Lizenz-Kosten.

### Konkrete Empfehlung

**Phase 1 (4-8h): Open-Meteo-Integration für historische Daten.**

```python
# data/sources/openmeteo_source.py (neu, oder ergänze weather_source.py)

import requests

def fetch_weather(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    variables: list[str] = ["temperature_2m_max", "precipitation_sum"],
) -> pd.DataFrame:
    """Fetch historical weather from Open-Meteo Archive API."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(variables),
        "timezone": "UTC",
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    df = pd.DataFrame(data["daily"])
    df["date"] = pd.to_datetime(df["time"])
    return df.drop("time", axis=1)


# Beispiel: Brazilian coffee regions, signal "drought stress"
brazilian_coffee_lats_lons = [(-21.18, -41.95), (-19.03, -42.93)]
weather_dfs = [fetch_weather(lat, lon, "2020-01-01", "2026-04-27") 
               for lat, lon in brazilian_coffee_lats_lons]
```

**Aufwand:** 4-8h für Open-Meteo + Refactoring von weather_source.

**Lizenz:** Open-Meteo CC-BY-4.0 — du musst attributieren in Reports.

---

## B20. FastAPI Best-Practices 2026

(Bereits in A11 abgehandelt — siehe oben.)

---

## B21. Prediction Markets als Alpha-Quelle

### Wirklich neuer Bereich

**Polymarket** und **Kalshi** sind 2024-2026 explodiert. Sie bieten markt-basierte Wahrscheinlichkeiten für:
- Wahlen (US-Präsident, Bundestag)
- Geopolitische Events (Krieg, Sanktionen)
- Wirtschaftsdaten (CPI, NFP)
- Wetter und Sport
- Crypto-Preise

### Was es draußen gibt

**`pmxt`** — "CCXT für Prediction Markets". Unified API für Polymarket, Kalshi.

**`SimpleFunctions`** — JS-CLI für Kalshi/Polymarket-Intelligence. Causal-Thesis-Models, Edge-Detection, 24/7 Orderbook-Monitoring.

**`PolyMind`** — Python, Real-Time Polymarket-Trading-Alerts mit Multi-AI-Analyse (Groq/Claude/Gemini). Whale-Bets-Tracking, Volume-Spikes, 12 Signal-Typen.

### Wofür das für dich relevant wäre

**Use-Cases:**

1. **Macro-Probability als Feature.** Polymarket-Preis für "FED-Rate-Cut Q3" gibt dir markt-implizite Wahrscheinlichkeit. Wenn das stark divergiert von deinem ML-Modell, ist das ein interessantes Signal.

2. **Election-Trading.** US-Präsidentschaftswahlen 2024 → Polymarket war oft schneller und genauer als klassische Polls. Du kannst auf Election-Outcome-Korrelationen mit Equities handeln (Tech vs Energy bei verschiedenen Outcomes).

3. **Event-Hedging.** "P(großer Krieg im Q3)" auf Polymarket kann als billiger Hedge funktionieren.

### Konkrete Empfehlung

**Niedrige Priorität, aber spannend für 2026-2027.**

**Phase 1 (5-8h): Read-Only Polymarket-API als zusätzliche Datenquelle.** Einfach Preise und Volumen pollen, in `data/sources/polymarket_source.py`.

**Phase 2 (15-25h): Cross-Asset-Korrelations-Studie.** Wie korreliert Polymarket-Preis für "Recession 2026" mit Yield-Curve, mit deinem `signals/recession_probability.py`?

**Lizenz:** APIs sind kostenfrei für Read, kostenpflichtig für Trading.

---

## B22. Knowledge-Graph-RAG für Finance

### Wirklich neuer Bereich

Du hast einen News-Knowledge-Graph (`intel/news_entity_graph.py`, siehe v3 §A3). Aber: Du hast vermutlich keinen **Retrieval-Augmented-Generation**-Stack on top.

**RAG für Finance** = LLM kann auf deinen Knowledge-Graph + News-Archiv + EDGAR-Filings zugreifen. Du fragst: "Was passiert wahrscheinlich mit AAPL, wenn China-Sanktionen kommen?" → LLM ruft historische ähnliche Episoden ab, synthesiziert.

### Was es draußen gibt

**`Open-Finance-Lab/AgenticTrading`** — bereits in v3 §B5 erwähnt, aber spezifisch für RAG-Pattern: Neo4j-basiertes shared Memory zwischen Agents.

**`langchain-neo4j`** — LangChain-Integration für Neo4j-basierte Graph-RAG.

**Microsoft GraphRAG** — Hybrid-Approach: Vector-Search + Graph-Traversal. Bessere als reine Vector-RAG bei komplexen Queries.

### Konkrete Empfehlung

**Sehr ambitioniert. Niedrige Priorität, hoher Lerneffekt.**

```python
# intel/graph_rag.py (Skizze, ~500 LOC bei voller Implementation)

from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_anthropic import ChatAnthropic

class FinanceGraphRAG:
    def __init__(self, neo4j_url: str, api_key: str):
        self.graph = Neo4jGraph(url=neo4j_url)
        self.llm = ChatAnthropic(model="claude-opus-4-7")
        self.qa = GraphCypherQAChain.from_llm(
            llm=self.llm,
            graph=self.graph,
            verbose=True,
        )
    
    def query(self, question: str) -> str:
        """Natural-language query over the financial knowledge graph."""
        return self.qa.run(question)
    
    def find_similar_episodes(self, current_event: dict, top_k: int = 5) -> list:
        """
        Find historically similar episodes in the news graph.
        E.g., "Russia sanctions in Q1 2026" → similar to "Iran sanctions 2018".
        """
        # Use LLM to construct Cypher query
        cypher_prompt = f"""
        Find historical events similar to: {current_event}
        Return the {top_k} most similar with their downstream market impact.
        """
        return self.qa.run(cypher_prompt)
```

**Aufwand:** 50-80h für ein erstes funktionierendes RAG-System.
**Lizenz:** LangChain MIT, Neo4j Community Edition GPL (als externer Service OK).

---

# Konsolidierter Adoption-Plan v4

Inkluziv aller Empfehlungen aus v1-v4. **Tier 1-4 sortiert nach (Impact / Aufwand)**:

### Tier 1: Sofort einbauen (~10-25h Stunden)

| # | Empfehlung | Aufwand | Quelle |
|---|---|---|---|
| 1 | gitleaks pre-commit | 1h | v2 |
| 2 | empyrical-reloaded | 3-4h | v1 |
| 3 | Healthcheck-Pattern | 30min | v1 |
| 4 | Parkinson/Garman-Klass Vol | 2-3h | v2 |
| 5 | Pre-commit-Stack vollständig | 2-3h | v2 |
| 6 | uv als pip-Ersatz | 2-3h | v2 |
| 7 | Term-Structure-Features VIX/Yield | 8-15h | v3 |
| 8 | Hash-Chain in evidence_pack | 2-3h | v3 |
| 9 | **VVIX-Tail-Risk-Signal** | 5-8h | **v4 §B11** |
| 10 | **Open-Meteo-Migration** | 4-8h | **v4 §B19** |
| 11 | **Senate-PTRs zusätzlich zu House** | 15-25h | **v4 §A14** |

### Tier 2: Sprint einplanen (~30-70h)

| # | Empfehlung | Aufwand | Quelle |
|---|---|---|---|
| 12 | Anti-Leakage erweitern | 8-12h | v1 |
| 13 | Bootstrap-CIs | 3-4h | v1 |
| 14 | quantstats Tearsheets | 2-3h | v1 |
| 15 | TWAP Order-Slicing | 4-6h | v2 |
| 16 | SOPS+age | 2-4h | v2 |
| 17 | Brinson-Hood-Beebower Attribution | 8-12h | v3 |
| 18 | Vol-Targeted Triple-Barrier | 6-10h | v3 |
| 19 | Almgren-Chriss empirisch | 8-12h | v3 |
| 20 | MAPIE-Backend für Conformal | 6-10h | v3 |
| 21 | Online-HMM-Updates | 6-10h | v3 |
| 22 | ELSTER/AWV-Reports | 12-20h | v3 |
| 23 | **DVC für Daten-Versionierung** | 4-6h | **v4 §B18** |
| 24 | **PyOD Anomalie-Detection** | 12-20h | **v4 §B17** |
| 25 | **tsfresh-Augmentation** | 10-15h | **v4 §B16** |
| 26 | **FastAPI Health Checks + Request-IDs** | 8-15h | **v4 §A11** |
| 27 | **LPPLS Crash-Detector** | 8-12h | **v4 §A19** |
| 28 | **Causal Inference DoWhy Pilot** | 10-15h | **v4 §B12** |

### Tier 3: Strategische Investments (~70-180h)

| # | Empfehlung | Aufwand | Quelle |
|---|---|---|---|
| 29 | DIY Feature-Store [SKIP — du hast einen!] | — | v2 (REVIDIERT) |
| 30 | Broker-Adapter | 8-16h | v1 |
| 31 | DataBlob-Pattern | 8-12h | v1 |
| 32 | qrun-Pattern | 16-30h | v2 |
| 33 | VWAP+AC | 10-15h | v2 |
| 34 | Toraniko Risk-Model | 16-25h | v3 |
| 35 | News-KG Neo4j | 20-35h | v3 |
| 36 | Vine-Copulas | 12-18h | v3 |
| 37 | PyMC Bayesian | 12-20h | v3 |
| 38 | YAML-DSL | 25-40h | v3 |
| 39 | Compliance-Modul DE | 6-10h | v3 |
| 40 | **Event-Sourcing-Refactoring** | 25-40h | **v4 §B15** |
| 41 | **MLflow Experiment-Tracking** | 8-15h | **v4 §B18** |
| 42 | **EconML Heterogeneous Effects** | 15-25h | **v4 §B12** |
| 43 | **CAOS-style Crisis-Alpha-Refactoring** | 15-25h | **v4 §A13** |
| 44 | **LLM-Entity-Linker für news/** | 15-25h | **v4 §A15** |
| 45 | **Embedding-Clustering für news/** | 10-15h | **v4 §A15** |

### Tier 4: Spekulativ/Zukunft (~weitere 80-200h)

| # | Empfehlung | Aufwand | Quelle |
|---|---|---|---|
| 46 | FinBERT | 4-6h | v1 |
| 47 | darts | 4-6h | v2 |
| 48 | NannyML [SKIP — du hast Drift schon] | — | v2 (REVIDIERT) |
| 49 | FinRL | 40-80h | v2 |
| 50 | BERTopic | 8-12h | v1 |
| 51 | Riskfolio-Migration [SKIP — du hast es!] | — | v1 (REVIDIERT) |
| 52 | Volatility-Surface | 30-50h | v3 |
| 53 | Synthetic-Data via TimeGAN [eingeschränkt — du hast Crisis-Templates] | 40-60h | v3 (REVIDIERT) |
| 54 | TradingAgents Multi-Agent | 20-40h | v3 |
| 55 | AlphaGen RL | 30-50h | v3 |
| 56 | HSMM | 12-20h | v3 |
| 57 | Block-Bootstrap | 5h | v3 |
| 58 | **Graph Neural Networks** | 40-80h | **v4 §B13** |
| 59 | **ClickHouse für Tick-Data** | 30-50h | **v4 §B14** |
| 60 | **Knowledge-Graph-RAG** | 50-80h | **v4 §B22** |
| 61 | **Polymarket als Datenquelle** | 5-25h | **v4 §B21** |

---

## Schluss-Bemerkung

Mit v4 sind wir bei **61 distinkten Empfehlungen** angekommen. Das ist absurd viel — unmöglich alles in 6 Monaten umzusetzen.

**Realistische Roadmap für Hans für die nächsten 6 Monate:**

**Monate 1-2: Foundation (~30-40h)**
- Tier 1 #7 (Term-Structure), #8 (Hash-Chain), #9 (VVIX-Signal)
- Tier 2 #17 (Brinson-Attribution), #18 (Vol-Targeted-TB), #23 (DVC)

**Monate 3-4: Strategische Investitionen (~40-60h)**
- Wahlweise: Tier 3 #34 (Toraniko Risk-Model) ODER #37 (PyMC)
- Tier 2 #28 (Causal Inference Pilot)
- Tier 2 #24 (PyOD Anomalie-Detection)

**Monate 5-6: Eigenes Spezialinteresse (~30-50h)**
- **Wenn Compliance/Audit-Fokus:** #40 (Event Sourcing) + #22 (ELSTER/AWV)
- **Wenn Research-Fokus:** #38 (YAML-DSL) + #54 (TradingAgents)
- **Wenn ML-Fokus:** #43 (EconML) + #58 (GNN-Pilot)

---

## Was wir jetzt **NICHT** mehr aufnehmen (REVIDIERTE Liste)

Nach v4 ist die "du hast das schon"-Liste massiv gewachsen:
- ❌ CPCV (`qa/cpcv_validation.py`)
- ❌ Deflated Sharpe (`qa/deflated_sharpe.py`)
- ❌ Meta-Labeling (`signals/meta_model.py` 453 LOC)
- ❌ Almgren-Chriss (`execution/almgren_chriss.py` 347 LOC)
- ❌ Riskfolio (`portfolio/riskfolio_optimizer.py`)
- ❌ Drift-Detection mit Evidently+NannyML (`ops/drift_monitor.py` + `qa/drift_detection.py`)
- ❌ HMM (`ml/regime_hmm.py` + `risk/regime_hmm.py` + `risk/regime_models.py`)
- ❌ Conformal-Prediction-Sizing (`portfolio/conformal_position.py`)
- ❌ SHAP (`ops/shap_explainer.py`)
- ❌ Trade-Journal/TCA (`qa/tca.py` + `qa/trade_tca.py`)
- ❌ Kelly-Sizing (`portfolio/kelly_uncertainty.py` + `portfolio/position_sizing.py`)
- ❌ Black-Litterman (`portfolio/black_litterman.py`)
- ❌ HRP (`portfolio/hierarchical_risk_parity.py`)
- ❌ Walk-Forward (`qa/walk_forward.py` 1277 LOC)
- ❌ GARCH (`risk/garch_vol_forecast.py`)
- ❌ **Feature-Store (`data/feature_store.py`)** — REVIDIERT, du hast einen!
- ❌ **Synthetic-Data (`data/synthetic_generator.py`)** — REVIDIERT, du hast Crisis-Templates
- ❌ **FastAPI-Server-Setup (`api/`)** — du hast 3463 LOC
- ❌ **17 Datenquellen (`data/sources/`)** — du hast schon ein riesiges Universum
- ❌ **House-PTR-Insider-Trading (`events/disclosures/`)** — du hast eine eigene 523-LOC-Pipeline
- ❌ **News-Pipeline (`events/news/`)** — du hast 720 LOC Pipeline + 30+ News-Module

---

**Soll ich jetzt was Konkretes anfangen?** Mein Vorschlag:

1. **Wenn du eine schnelle Win willst:** `signals/tail_risk_vvix.py` (Tier 1 #9, 5-8h). VVIX schlägt VIX als Tail-Risk-Indikator deutlich.

2. **Wenn du ein größeres Projekt willst:** `qa/causal_validation.py` (Tier 2 #28) + EconML-Erweiterung — gibt dir ein neues quantitatives Werkzeug, dass die meisten Solo-Trader nicht haben.

3. **Wenn du am Repo aufräumen willst:** Tier 1 #1-#6 (alle Hygiene-Tools) — keine Strategie-Veränderung, aber ordentliches Foundation.

Lass mich wissen womit du anfangen willst, und ich liefere konkreten Code mit präzisen Repo-Stellen, Tests und Effort-Estimates.
