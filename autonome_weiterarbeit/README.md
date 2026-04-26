# Assembled Trading AI — Modularer Umsetzungsplan

**Stand:** 2026-04-24
**Scope:** Richtung A (Personal Quant, eigenes Geld). B2B-SaaS-Themen sind bewusst ausgeklammert — kommen später wenn das System fertig ist.

Dieses Paket ist der vollständige Umsetzungsplan, der aus den drei Deep-Research-Runs destilliert wurde, plus drei ergänzende Dokumente für Execution, Compliance und Incident-Response.

## Die Dateien

### Master-Plan

- **`00_MASTER_PLAN.md`** — der Überblick. Liest du zuerst. Gesamtarchitektur, 4 Layer, Entscheidungen.

### Free-Stack (0 EUR/Monat)

- **`10_FREE_DATEN.md`** — 18 Datenquellen ohne Kosten (SEC EDGAR, FRED, FINRA, CBOE, GDELT-Deep, Wikipedia, Reddit, Stocktwits, Alpaca Free etc.)
- **`11_FREE_MODELLE.md`** — 18 Modelle & Libraries (FinBERT-Tone, Embeddings, Conformal Prediction, Regime-HMM, skfolio-CPCV etc.)
- **`12_FREE_INFRASTRUKTUR.md`** — 15 Infrastruktur-Module (Oracle Always-Free, lokaler Windows-Start, Hetzner unter 5 EUR, MLflow, DuckDB etc.)
- **`13_FREE_MODULE.md`** — 16 Trading-Module priorisiert (Liquidity-Index, Regime-HMM, Residual-Momentum, Form-4, Analyst-Revisions, PEAD etc.)
- **`14_FREE_UNIVERSUM.md`** — Ticker-Listen & Segmentierung ohne Kosten (S&P 500, Tier-Struktur)

### Paid-Stack (unter 100 EUR/Monat)

- **`20_PAID_DATEN.md`** — 10 Paid-Datenquellen, jede einzeln gerechtfertigt (EODHD, Finnhub Premium, FMP, Polygon etc.)
- **`21_PAID_MODELLE.md`** — Claude Haiku 4.5 als einziger Paid-Modell-Must, Ollama-Backup
- **`22_PAID_INFRASTRUKTUR.md`** — Hosting (Hetzner CX22/CX32, Backups, Domain)
- **`23_PAID_UNIVERSUM.md`** — 1 800-Ticker-Universum mit EODHD-Upgrade

### Integration & Architektur

- **`30_NEWS_TA_FUSION.md`** — News-TA-Fusion-Architektur (3 Schichten: Feature, Meta-Gate, 2D-Matrix)
- **`31_COMPOSITE_SCORE.md`** — 9 Signal-Dimensionen mit Formeln und Regime-Gewichtung
- **`32_VALIDIERUNG.md`** — CPCV, Deflated Sharpe, Meta-Labeling, Walk-Forward
- **`33_EXECUTION_ORDERMANAGEMENT.md`** — Layer 4: Idempotenz, Partial-Fill, Reconciliation, Kill-Switch, Position-Sizing, BaseSignal-Plugin-Architektur

### Roadmap & Lockfile

- **`40_ROADMAP.md`** — 12-Monats-Fahrplan mit Wochen-Granularität, Go/No-Go-Gates
- **`99_STACK_LOCKFILE.md`** — vollständiger pinned Python-Stack

### Betrieb

- **`50_COMPLIANCE_RECHT.md`** — Steuer (Tax-Lots, FIFO, USD/EUR), DSGVO, Daten-Lizenzen, PDT-Regel, Paper-zu-Live-Übergang, Notfall-Dokumentation — alles für Personal-Use
- **`51_INCIDENT_PLAYBOOK.md`** — 13 Runbooks für die häufigsten Ausfälle, Chaos-Test-Schedule, Post-Mortem-Template

---

## So arbeitest du mit den Dateien

1. **Lies zuerst** `00_MASTER_PLAN.md` — du verstehst die Gesamtarchitektur.
2. **Dann** `40_ROADMAP.md` — du siehst, wann was dran ist.
3. **Dann** `30_NEWS_TA_FUSION.md` — die Kern-Innovation.
4. **Arbeite modular:** Jede Datei ist in sich geschlossen. Du kannst z.B. nur `13_FREE_MODULE.md` anfangen und den Rest ignorieren.
5. **Checkliste am Ende jeder Datei:** Hake ab, was fertig ist.

## Die strategischen Leitsätze

1. **Free-First:** Nichts kostet, was nicht echten Alpha-Lift bringt. Der Free-Stack liefert 90% des Werts.
2. **Edge-nach-Kosten statt Hype:** Regime-Filterung + Meta-Labeling + CPCV-Validierung schlagen jeden GNN und jedes Satellite-Abo.
3. **Execution ist kritischer als Signale:** Ein perfekter Signal-Stack verliert Geld, wenn Orders doppelt oder halb oder gar nicht rauskommen.
4. **Runbooks vor Stolpern:** Jeder wahrscheinliche Ausfall hat eine dokumentierte Antwort.

---

## Budget-Zusammenfassung

| Phase | Monatlich | Kritische Posten |
|---|---|---|
| Phase 1 (M1-3) | 0 EUR | Free-Stack, Windows lokal |
| Phase 2 (M4-6) | ~22 EUR | + EODHD (19.99 USD) |
| Phase 3 (M7-9) | ~45 EUR | + Claude Haiku, Hetzner CX22 |
| Phase 4 (M10-12) | ~55-65 EUR | + optional Finnhub Starter |

Klar unter 100 EUR/Monat mit Puffer für Upgrades.

---

## Reihenfolge zum Durchlesen

**Erste Stunde:** `README → 00 → 40`

**Zweite Stunde:** `30 → 31 → 32`

**Umsetzung Phase 1:** `13 → 14 → 12 → 33 → 50`

**Rest:** nach Bedarf als Nachschlagewerk.
