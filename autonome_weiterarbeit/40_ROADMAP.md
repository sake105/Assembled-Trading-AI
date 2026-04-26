# 40 — 12-Monats-Roadmap

**Zweck:** Konkreter Phasenplan mit Commits und Entscheidungspunkten. Jeder Monat hat klare Deliverables und Exit-Kriterien.

---

## Übersicht

```
Monat 1-3   │ Phase 1: Foundation                    │ 0-22 EUR/Monat
Monat 4-6   │ Phase 2: Expansion + Meta-Labeling     │ 22-45 EUR/Monat
Monat 7-9   │ Phase 3: 2D-Matrix + Conformal         │ 45-55 EUR/Monat
Monat 10-12 │ Phase 4: Fine-Tune + Experimente       │ 55-65 EUR/Monat
```

---

## Monat 1 — Hausaufgaben-Sprint

**Ziel:** Repo ist sauber, Secrets sind nicht in Git, die Basis-Architektur steht.

### Deliverables

- [ ] Repo auf **private** gesetzt (kritisch!)
- [ ] Git-History nach Email-Leaks durchsucht (`git log --all --grep='@'`), bei Funden `git filter-repo` oder BFG
- [ ] SOPS + age-Keys eingerichtet (`12_FREE_INFRASTRUKTUR.md` §12.14)
- [ ] `.env.sops.yaml` committed, plain `.env` gitignored
- [ ] pre-commit-hooks mit gitleaks + ruff + black + mypy
- [ ] Python 3.12 + `uv` als Package-Manager
- [ ] Repo-Pfad ohne Umlaut/Leerzeichen

### Was du NICHT machst
- Keine neuen Features
- Keine Module-Refactors

### Exit-Kriterium
`pre-commit run --all-files` → grün.
`gitleaks detect --log-opts='--all'` → keine Findings.

---

## Monat 2 — Daten-Pipeline sauber

**Ziel:** Alle Free-Datenquellen integriert, Single-Source-of-Truth pro Datentyp.

### Deliverables

- [ ] SEC EDGAR via `edgartools` (Form 4 + 8-K real-time)
- [ ] FRED via `fredapi` — Top-10-Makro-Series in Feature-Store
- [ ] FINRA Short-Interest täglicher Pull
- [ ] CBOE Put/Call + VIX-Term via yfinance
- [ ] Finnhub Free — News + Earnings + Ratings integriert
- [ ] PRNewswire/BusinessWire/GlobeNewswire RSS
- [ ] CoinMetrics Community für Crypto-Macro
- [ ] Alpaca Paper + IEX-Feed produktiv

### Datenqualität
- [ ] Jedes Dataset hat `available_at`-Timestamp
- [ ] Parquet mit Hive-Partitioning konfiguriert
- [ ] DuckDB-ASOF-JOIN für Feature-Fetch

### Exit-Kriterium
Alle 10 Datenquellen laufen via APScheduler, jede mit Health-Check in Uptime Kuma.

---

## Monat 3 — Composite-Score MVP

**Ziel:** 5 von 9 Signal-Dimensionen live, Composite-Score pro Ticker pro Tag.

### Deliverables

- [ ] Dim 1: Multi-Timeframe-Alignment
- [ ] Dim 2: Klassische TA mit Regime-Params
- [ ] Dim 7: Breadth/Intermarket
- [ ] Dim 8: Seasonality (einfach)
- [ ] Dim 9: News als 9. BaseSignal (siehe `30_NEWS_TA_FUSION.md` Schicht 1)
- [ ] Regime-Classifier (Rule-Based: VIX + Term-Slope + HY-Spread)
- [ ] Composite-Score-Funktion mit Regime-Gewichtung

### Neue Module (free, siehe `13_FREE_MODULE.md`)
- [ ] §13.1 Liquidity-Condition-Index (30 LOC, Quick-Win)
- [ ] §13.2 Regime-Switching HMM (Phase 1 noch Rule-Based, HMM-Upgrade in Phase 2)
- [ ] §13.3 GARCH Vol-Forecast für Position-Sizing
- [ ] §13.7 Residual-Momentum FF5

### Validation
- [ ] skfolio CombinatorialPurgedCV als Standard
- [ ] Deflated Sharpe Ratio-Funktion
- [ ] Rolling IC pro Signal in Dashboard

### Exit-Kriterium
Composite-Score wird täglich für S&P 500 + EURO STOXX 50 + 35 ETFs berechnet. Shadow-Mode aktiv.

---

## Monat 4 — Paid-Minimum + Meta-Labeling-Vorbereitung

**Ziel:** EODHD integriert, Survivorship-Bias-Schutz, Meta-Labeling im Shadow.

### Deliverables

- [ ] EODHD All-World EOD abonniert (19.99 USD, siehe `20_PAID_DATEN.md` §20.1)
- [ ] Historische Delisted-US-Ticker im Feature-Store
- [ ] EURO STOXX 50 via EODHD statt yfinance
- [ ] Historische S&P-Composition für Backtest-Windows
- [ ] 10-Jahre-Backfill für Tier-1-Universum

### Meta-Labeling-Skeleton (siehe `30_NEWS_TA_FUSION.md` Schicht 2)
- [ ] mlfinpy installiert und getestet
- [ ] CUSUM-Events via `mlfinpy.filters.cusum_filter`
- [ ] Triple-Barrier mit dynamic PT/SL (News-aware)
- [ ] 12-15 Meta-Features definiert (alle verfügbar im Feature-Store)
- [ ] LightGBM Meta-Model mit Isotonic-Calibration
- [ ] **Meta-Model läuft im Shadow-Mode**, keine Live-Gates

### Validation
- [ ] PBO-Test implementiert
- [ ] CPCV-Pipeline für Meta-Labeling mit purged_size ≥ vertical_barrier

### Exit-Kriterium
Meta-Model produziert täglich Predictions. Backtest zeigt DSR > 0.85. **Noch keine Live-Gates.**

---

## Monat 5 — Claude Haiku + Universum-Expansion

**Ziel:** LLM-Zweitrunde für News aktiv, Tier-2 Universum live.

### Deliverables

- [ ] Anthropic-Account + API-Key in SOPS
- [ ] Budget-Guard für max 10 EUR/Monat
- [ ] Target-Sentiment-Extraction via Haiku 4.5 (siehe `21_PAID_MODELLE.md` §21.1)
- [ ] Primary-vs-Mentioned-Disambiguation für Top-20 News/Tag
- [ ] Prometheus-Metrik für Anthropic-Spending

### Universum Tier-2 (siehe `23_PAID_UNIVERSUM.md`)
- [ ] S&P 400 + S&P 600 (Quality-Filter) über iShares IJH/IJR
- [ ] STOXX Europe 600 Rest via EODHD
- [ ] 25 ADRs (Asien-Exposure)
- [ ] Total ~1 800 Ticker aktiv

### Weitere Module (free)
- [ ] §13.4 Insider Form-4 Cluster-Buy-Score
- [ ] §13.5 Analyst-Revisions via Finnhub
- [ ] §13.6 PEAD/SUE für Earnings-Event-Trading

### Exit-Kriterium
1 800 Ticker werden täglich gescored. LLM-Enhanced-Features sichtbar in Composite-Score-Verbesserung (Shadow-Mode-IC).

---

## Monat 6 — Meta-Labeling live

**Ziel:** Shadow-Mode verlassen, Meta-Labeling als Binary-Gate produktiv.

### Deliverables

- [ ] Meta-Model 60+ Tage im Shadow → Promote-Entscheidung
- [ ] Meta-Gate live mit θ_meta=0.55 (permissiv)
- [ ] **Kein τ_veto noch** — erst 4 Wochen θ-only beobachten
- [ ] A/B-Vergleich: System mit Gate vs System ohne Gate
- [ ] Canary-Deployment-Pipeline (siehe `32_VALIDIERUNG.md` §32.7)

### Cloud-Migration (optional, siehe `22_PAID_INFRASTRUKTUR.md`)
- [ ] Hetzner CX22 aufgesetzt (4.25 EUR/Monat)
- [ ] Docker-Compose-Stack deployed
- [ ] Caddy mit Auto-HTTPS
- [ ] Grafana Cloud Free-Tier Dashboards

### Sektor-Modelle
- [ ] 11 Sektor-spezifische LightGBM-Modelle trainiert
- [ ] Cross-Sectional-Ranking mit Sektor als `categorical_feature`
- [ ] Separates EU-Modell

### Exit-Kriterium
Meta-Gate ist live. Canary-Size-Schedule aktiv (10% → 33% → 100%). Drift-Monitoring läuft.

---

## Monat 7 — 2D-Decision-Matrix

**Ziel:** Schicht 3 der News-TA-Fusion live.

### Deliverables

- [ ] 5×5-Matrix via Optuna GPSampler kalibriert (siehe `30_NEWS_TA_FUSION.md`)
- [ ] CPCV-aggregiertes Objective (kein reines Sharpe-Maximize)
- [ ] Agreement-Multiplier als Size-Faktor
- [ ] Bayesian-Beta-Update im Live-Pfad
- [ ] τ_veto=1.5 aktiviert (News-Veto bei hohem Sign-Conflict)

### Weitere Module Phase 2
- [ ] §13.8 Macro-4-Quadrant
- [ ] §13.9 Recession-Probability
- [ ] §13.10 Sentiment-Panel (Fear&Greed-Replikation)
- [ ] §13.11 FINRA Short-Interest-Features

### Exit-Kriterium
Alle 3 Schichten der News-TA-Fusion laufen. Composite-Score berücksichtigt News trifach.

---

## Monat 8 — Conformal Prediction

**Ziel:** Phase-3-Einleitung: verteilungsfreie Uncertainty-Intervalle.

### Deliverables

- [ ] MAPIE 0.9.x installiert
- [ ] EnbPI für Return-Forecast (verteilungsfrei, für Zeitreihen)
- [ ] Prediction-Interval-Width als Konfidenz-Proxy
- [ ] Position-Sizing: enge Intervalle → Upsize, breite → Downsize
- [ ] Validation: Coverage-Rate-Check (sollte bei 90%-PI nahe 90% sein)

### Regime-HMM produktiv
- [ ] hmmlearn mit 3-4 States auf VIX + Term-Slope + HY-Spread
- [ ] Wöchentliches Retraining via APScheduler
- [ ] Label-Flipping-Handling
- [ ] Replace Rule-Based-Regime durch HMM

### Weitere Module Phase 2
- [ ] §13.12 Buyback-Drift via 8-K-Parser
- [ ] §13.13 ETF-Flow self-computed
- [ ] §13.14 Wikipedia Page-Views Top-100

### Exit-Kriterium
Conformal-Intervalle in Production. Sizing beachtet Uncertainty. Regime-HMM trained und deployed.

---

## Monat 9 — Cross-Impact-Graph

**Ziel:** Schicht 4 — News-Propagation über Ticker.

### Deliverables

- [ ] Pearson-Correlation-Graph mit Ledoit-Wolf-Shrinkage
- [ ] OpenFIGI v3 Integration (vor V2-Sunset 01.07.2026!)
- [ ] Supply-Chain-JSON für Top-60 Ticker (manuell)
- [ ] Operator-Graph (CEOs via Wikidata P169)
- [ ] News-Propagation-Algorithmus: Weight × Correlation

### Entity-Linking-Stack
- [ ] spaCy `en_core_web_lg`
- [ ] Cashtag-Regex
- [ ] GLiNER 0.2+ als Zweitrunde (Zero-Shot)
- [ ] Company-to-Ticker-Cache

### Exit-Kriterium
News über AAPL propagieren mit gerichtetem Sentiment auf TSM, QCOM, AVGO. Cross-Impact-Feature im Composite.

---

## Monat 10 — FinBERT Fine-Tuning

**Ziel:** Eigenes FinBERT-Modell auf 5-10k selbst-gelabelten Headlines.

### Deliverables

- [ ] Distant-Supervision-Labeling-Pipeline (Label = Preisreaktion ±t Minuten)
- [ ] 5k-10k Training-Samples generiert
- [ ] FinBERT-Tone als Base-Model, Fine-Tuning via HuggingFace-Trainer
- [ ] ONNX-Export für schnellere Inferenz
- [ ] A/B-Test vs FinBERT-Tone-Baseline

### Compute
- [ ] Einmalig ~20 EUR auf Lambda Labs oder RunPod für H100 × 4h
- [ ] Alternativ: Oracle Always-Free (länger, aber gratis)

### Weitere Module Phase 3
- [ ] §13.15 Cross-Asset-Carry-Overlay
- [ ] §13.16 Tail-Risk-Hedge (erst wenn Alpaca-Options live)

### Exit-Kriterium
Eigenes FinBERT zeigt im Shadow-Mode höhere IC als Baseline. Promote-Entscheidung nach 30 Tagen.

---

## Monat 11 — Tier-3 Event-Driven

**Ziel:** Russell 2000 Small-Cap-Scan bei News/Vol/Gap-Triggern.

### Deliverables

- [ ] Tier-3-Trigger-Logik in Worker
- [ ] On-Demand-Analysis mit 7d-TTL-Cache
- [ ] LLM-Screening via Haiku für Relevanz-Check
- [ ] Liquiditäts-Filter permissiv (500k USD ADV, 100M USD MCap)

### Portfolio-Construction
- [ ] Riskfolio-Lib + cvxpy für CVaR-Portfolio
- [ ] HRP via skfolio für Diversifikations-Baseline
- [ ] Vol-Targeting 15% annualisiert

### Exit-Kriterium
Event-driven kann binnen 1 Minute 50+ Small-Cap-Tickers scannen und Top-5-Kandidaten generieren.

---

## Monat 12 — Konsolidierung + Phase-A/B-Entscheidung

**Ziel:** Review, Stabilisierung, Dokumentation.

### Deliverables

- [ ] Vollständiger 12-Monats-Backtest aller aktiven Signale
- [ ] Deflated-Sharpe-Report pro Signal
- [ ] PBO-Report
- [ ] IC-Decay-Analyse
- [ ] Drift-Historie (PSI pro Feature über 12 Monate)
- [ ] Post-Mortem für alle verworfenen Signale

### Entscheidungspunkt: Richtung A oder B?

**Richtung A — Personal Quant:**
- Live-Capital langsam erhöhen (<5k EUR initial)
- Paper-System bleibt parallel aktiv
- Fokus: Sharpe-Stabilisierung, Drawdown-Control

**Richtung B — B2B-SaaS:**
- UG (haftungsbeschränkt) gründen (400-700 EUR Startkosten)
- Kunden-Akquise: 3-5 Proprietary-Trading-Firmen oder Fonds ansprechen
- Commercial-Daten-Licences einkalkulieren (EODHD 399 USD, Finnhub 99 USD)
- API-Vorbereitung für Signal-Subscription-Service

**Parallel-Pfad:** Beides möglich, aber dann Budget auf 400-500 EUR/Monat.

### Optional: Experimente

- [ ] GNN-Experiment via PyTorch Geometric (strikt Point-in-Time, CPCV)
- [ ] Voyage-Finance-2 Embeddings A/B vs BGE-small
- [ ] Alternative LLM-Provider-A/B (Gemini 2.5 Flash-Lite, DeepSeek V3.2)

### Exit-Kriterium
Das System hat 12 Monate stabile Pipeline gezeigt. Entscheidung für Richtung A oder B getroffen.

---

## Budget-Verlauf über 12 Monate

| Monat | Ausgaben/Monat | Neu hinzu |
|---|---|---|
| 1-3 | 0 EUR | — |
| 4 | ~18 EUR | + EODHD |
| 5 | ~25 EUR | + Claude Haiku |
| 6 | ~30 EUR | + Hetzner CX22 |
| 7-9 | ~35 EUR | stabil |
| 10 | ~55 EUR | + einmalig 20 EUR Compute |
| 11-12 | ~45-55 EUR | evtl. + Finnhub Starter |

**12-Monats-Summe:** ~450 EUR über das ganze Jahr — unter 40 EUR/Monat Durchschnitt.

---

## Was zu vermeiden ist (jeden Monat relevant)

1. **Kein Big-Bang-Deployment.** Jedes neue Signal ≥60 Tage Shadow.
2. **Kein Feature-Addition-Rausch.** Erst alte Signale stabilisieren, dann neue.
3. **Kein Skip von Validation.** CPCV + DSR + PBO vor jeder Live-Entscheidung.
4. **Kein yfinance in SaaS-Richtung.** Commercial-Licences Pflicht bei Richtung B.
5. **Kein GNN in Phase 1/2.** Phase 3 optional, wenn alles andere stabil.
6. **Keine Microservices.** Monolith + Workers reicht.
7. **Keine Paid-Data-Käufe ohne klaren Edge.** EODHD + Haiku = 28 EUR reichen für 95% des möglichen Paid-Edge.

---

## Wichtige Meilensteine

- **Monat 3:** Erstes Signal live (ohne Meta-Gate)
- **Monat 6:** Meta-Gate produktiv
- **Monat 9:** Alle 3 News-TA-Fusion-Schichten live
- **Monat 12:** Richtung-Entscheidung A/B

**Bei Verzögerung:** Schiebe Monate nach hinten, aber **überspringe keine Validation-Schritte**. Ein 15-Monats-Plan mit sauberem System ist besser als ein 12-Monats-Plan mit Overfitting.
