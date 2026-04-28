# Wettbewerbsanalyse v3: Vertiefung bestehender Module + komplett neue Themen

**Datum:** 2026-04-27
**Vorgänger:** v1 (`COMPETITIVE_ANALYSIS_2026-04-27.md`, 909 Zeilen), DEEPDIVE (`COMPETITIVE_ANALYSIS_2026-04-27_DEEPDIVE.md`, 1062 Zeilen) und v2 (`COMPETITIVE_ANALYSIS_v2_2026-04-27.md`, 1459 Zeilen)
**Was hier neu ist:** v3 hat einen **anderen Charakter** als v1/v2/DEEPDIVE. Es ist eine **gezielte Recherche basierend auf einer Repo-Inspektion**.

## Vorbemerkung — wichtige Korrektur zu v1+v2

Bei der Inspektion deines Repos (`/home/claude/Assembled-Trading-AI-fresh/`) habe ich festgestellt: **v1 und v2 haben eine Reihe von Empfehlungen gemacht für Bereiche, die du längst implementiert hast.** Konkret:

| v1/v2 hat empfohlen | Wo es bei dir tatsächlich schon liegt |
|---|---|
| CPCV (Combinatorial Purged CV) | `qa/cpcv_validation.py` |
| Deflated Sharpe Ratio | `qa/deflated_sharpe.py` |
| Anti-Leakage-Tool | `qa/leakage_tests/` (eigenes Modul) |
| Drift Detection (Evidently+NannyML) | `ops/drift_monitor.py` + `qa/drift_detection.py` (Kommentar verweist auf "From 11_FREE_MODELLE.md §11.17") |
| Meta-Labeling | `signals/meta_model.py` |
| Almgren-Chriss Execution | `execution/almgren_chriss.py` |
| Smart Order Routing | `execution/smart_order_router.py` |
| Riskfolio-Lib Integration | `portfolio/riskfolio_optimizer.py` |
| Conformal Prediction (Position-Sizing) | `portfolio/conformal_position.py` |
| HMM Regime-Detection | `ml/regime_hmm.py` (416 LOC), `risk/regime_hmm.py` |
| Copula Models | `ml/copula_models.py` (Clayton/Gumbel/Gaussian, 281 LOC) |
| SHAP Explainer | `ops/shap_explainer.py` |
| Trade Journal | `ops/trade_journal.py` |
| Prometheus Metrics Exporter | `ops/metrics_exporter.py` |

**Entschuldigung dafür.** Dein Repo ist deutlich reifer als ich in v1/v2 angenommen habe.

v3 macht es richtig: erst Repo-Stand checken, dann recherchieren.

---

## Inhaltsverzeichnis

**Teil A — Vertiefung bestehender Module** (Wo du schon was hast, aber noch verbessern könntest)
1. [accounting/ — deutsche Steuer-Reports & ELSTER](#a1-accounting--deutsche-steuer-reports)
2. [intel/news_entity_graph.py — Neo4j-Persistierung](#a2-news_entity_graph--neo4j)
3. [ml/copula_models.py — Vine-Copulas für höher-dim](#a3-copula_models--vine-copulas)
4. [features/triple_barrier.py — mlfinlab-Vergleich](#a4-triple_barrier--mlfinlab-vergleich)
5. [risk/factor_exposures.py — toraniko Barra-Style Risk Model](#a5-factor_exposures--toraniko)
6. [signals/meta_model.py — Meta-Labeling-Validierung](#a6-meta_model--validierung)
7. [portfolio/conformal_position.py — Puncc/MAPIE-Vergleich](#a7-conformal_position--puncc)
8. [attribution/ — Brinson-Hood-Beebower & Karnosky-Singer](#a8-attribution--brinson)
9. [certify/ — Audit-Trail-Standards](#a9-certify--audit-trail)
10. [ops/metrics_exporter.py — Custom-Metrics für Trading](#a10-metrics_exporter--custom-metrics)

**Teil B — Komplett neue Themen** (Bisher 0% angesprochen)
11. [Synthetic Financial Time Series (TimeGAN, TC-VAE)](#b1-synthetic-data)
12. [Volatility Surface Modeling (SVI, SABR, vanna-volga)](#b2-volatility-surface)
13. [Causal Inference für Trading (EconML, CausalML, DoWhy)](#b3-causal-inference)
14. [MLflow + DVC für Experiment-Tracking](#b4-mlflow--dvc)
15. [DuckDB als Backtest-Datastore](#b5-duckdb)
16. [Cross-Asset Carry Strategies](#b6-cross-asset-carry)
17. [Term Structure Modelling (VIX-Futures, Yield-Curve)](#b7-term-structure)
18. [Liquidity-aware Position Sizing (real-world)](#b8-liquidity-aware-sizing)
19. [Volume-Synchronized PIN (VPIN) für Toxic Flow](#b9-vpin)
20. [Bayesian Optimization für Execution Schedules](#b10-bayesian-execution)

[Adoption-Plan v3](#adoption-plan-v3)

---

# Teil A — Vertiefung bestehender Module

## A1. accounting/ — deutsche Steuer-Reports

### Was du hast

`accounting/` ist mit **6.564 LOC** dein zweitgrößtes Modul. Die Hauptdateien:
- `tax_lots.py` (395 LOC) — FIFO/LIFO/HIFO Cost-Basis
- `ledger.py` (610 LOC) + `ledger_integration.py` (489 LOC)
- `position_engine.py` (512 LOC)
- `reconciliation.py` (395 LOC) + `reconciliation_report.py` (491 LOC)
- `evidence_pack.py` (1.147 LOC) — Audit-Trail-Generator
- `broker_snapshot_*` — IBKR/Alpaca-Imports
- `currency.py` (77 LOC)

Sehr ausgereiftes System. Bereiche, die wahrscheinlich noch fehlen:

### Was die Konkurrenz macht

**`stadlmax/pyFIFOtax`** ist genau für deinen Use-Case relevant. Es generiert:

1. **ELSTER-konforme Excel-Sheets** — die Behörde-CSVs in dem Format, das dein Finanzamt will
2. **Z4 / Z10 AWV-Reports** für die Bundesbank — bei Transaktionen über 12.500 EUR meldepflichtig
3. **EUR-Konvertierung mit korrekten Stichtagskursen** (Bundesbank-Referenzkurse)
4. **IBKR Custom Statement Import** — das ist der genauere Datenpfad als die Alpaca-Snapshots

**`eprbell/rp2`** ist das größere Projekt: Apache 2.0 Lizenz, programmable Plugin-Architektur für country-spezifische Reports.

### Konkrete Empfehlung

Du hast `accounting/`, aber vermutlich keinen ELSTER-Export-Workflow. Das wäre eine kleine, aber sehr nützliche Erweiterung:

```python
# accounting/elster_exporter.py (Vorschlag)

import pandas as pd
from pathlib import Path

class ElsterExporter:
    """Generates ELSTER-compatible reports from your existing ledger."""
    
    def __init__(self, ledger: Ledger, currency_converter: CurrencyConverter):
        self.ledger = ledger
        self.fx = currency_converter  # uses Bundesbank reference rates
    
    def generate_anlage_kap(self, tax_year: int, output_path: Path) -> pd.DataFrame:
        """Anlage KAP (Kapitalerträge) for the German tax return."""
        sells = self.ledger.get_realized_pnl(year=tax_year)
        
        # Convert each transaction to EUR using settlement-day reference rate
        sells["eur_proceeds"] = sells.apply(
            lambda row: self.fx.convert(
                amount=row["proceeds"],
                from_ccy=row["ccy"],
                to_ccy="EUR",
                date=row["settle_date"],
            ),
            axis=1,
        )
        
        # Categorize by Anlage-KAP-Zeilen
        result = pd.DataFrame({
            "Zeile 7": [...],   # Capital gains short-term
            "Zeile 19": [...],  # Capital gains long-term
            "Zeile 24": [...],  # Foreign withholding tax
        })
        result.to_excel(output_path / f"anlage_kap_{tax_year}.xlsx")
        return result
    
    def generate_z4_z10_awv(self, tax_year: int, threshold_eur: float = 12500.0):
        """Z4/Z10 reports for Bundesbank AWV (foreign exchange reporting)."""
        # Z4: Outgoing payments > 12.5k EUR
        # Z10: Securities transactions
        ...
```

**Aufwand:** 12-20h. Sehr hoher Nutzen wenn du in DE steuerpflichtig bist (was du als Hans wahrscheinlich bist) und nicht jedes Jahr manuell eine Excel-Datei ausfüllen willst.

**Lizenz:** rp2 (Apache 2.0) und pyFIFOtax (MIT) erlauben Code-Studium und freie Adaption. Logik direkt nachimplementieren ist OK.

---

## A2. news_entity_graph — Neo4j-Persistierung

### Was du hast

`intel/news_entity_graph.py` (206 LOC) — vermutlich eine in-memory Graph-Repräsentation von News-Entity-Relationen. Plus 30+ andere `news_*` Module mit hochentwickelter Pipeline:
- `news_classifier.py` (637 LOC)
- `news_dedupe.py` (373 LOC) + `news_semantic_dedup.py` (162 LOC)
- `news_enricher.py`, `news_entity_mapper.py`, `news_event_store.py`
- `news_impact_estimator.py`, `news_corroboration.py`, `news_contradiction.py`

Das ist eine sehr ausgereifte News-Intelligence-Pipeline.

### Was state-of-the-art aussieht

**`neo4j-labs/llm-graph-builder`** — Apache 2.0, von Neo4j selbst. Kombiniert LLM-Extraction mit Neo4j-Persistierung:
- LLM (OpenAI/Gemini/Claude) extrahiert Entities und Relations aus Text
- Custom-Schema für Domain (Person, Company, Country, Event...)
- Speicherung in Neo4j als persistenter Knowledge-Graph
- Cypher-Query-Interface für komplexe Abfragen

**`lighteternal/Automated-Knowledge-Graph-Construction`** — älter, aber zeigt das Pattern: spaCy NER → Coreference Resolution → Neo4j-Persistierung.

### Was bei dir fehlen könnte

Dein `news_entity_graph.py` ist mit 206 LOC vermutlich:
- In-Memory (kein Persistent-Layer)
- Keine Cypher-Query-Schnittstelle
- Schwierig, multi-hop-Beziehungen abzufragen ("welche Companies mit Tier-1-Suppliern in TW sind heute in News?")
- Kein einfacher Visualisierungs-Layer

### Konkrete Empfehlung

Erweiterung um Neo4j-Persistierung:

```python
# intel/news_entity_graph_neo4j.py (Vorschlag)

from neo4j import GraphDatabase

class NewsEntityGraphNeo4j:
    """Persistent knowledge graph for news entities via Neo4j."""
    
    def __init__(self, uri: str, auth: tuple[str, str]):
        self.driver = GraphDatabase.driver(uri, auth=auth)
    
    def upsert_entity(self, entity_type: str, name: str, properties: dict):
        """Add or update an entity (Person, Company, Country, etc.)."""
        with self.driver.session() as session:
            session.run(
                f"""
                MERGE (e:{entity_type} {{name: $name}})
                SET e += $props
                """,
                name=name, props=properties,
            )
    
    def find_supply_chain_risk(self, target_country: str, hops: int = 2):
        """Find companies exposed to a country via N-hop supplier relationships."""
        with self.driver.session() as session:
            result = session.run(
                f"""
                MATCH path = (c:Company)-[:SUPPLIER*1..{hops}]->(supplier:Company)
                              -[:LOCATED_IN]->(country:Country {{name: $country}})
                RETURN c.name AS company,
                       length(path) AS hops,
                       supplier.name AS connected_via
                ORDER BY hops ASC
                """,
                country=target_country,
            )
            return [dict(r) for r in result]
```

**Was du gewinnst:**
- Persistierung über mehrere Sessions
- Multi-hop-Abfragen (kritisch für Supply-Chain-Risk, georisk_overlay etc.)
- Visualisierung über Neo4j Bloom (kostenlos für Single-User)
- Bessere Integration mit deinem `risk/georisk_overlay.py` (Geo-Risk könnte direkt Cypher-Queries nutzen)

**Aufwand:** 8-12h für eine saubere Migrations-Schicht. Neo4j Community-Edition ist GPL-3 (nur als externes Tool nutzen, nicht Code-Linkage). Der Python-Driver `neo4j` ist Apache 2.0.

**Alternative:** Wenn dir Neo4j zu schwer ist, ist **NetworkX in einer DuckDB-Tabelle** auch eine Option (siehe B5 unten).

---

## A3. copula_models — Vine-Copulas

### Was du hast

`ml/copula_models.py` (281 LOC) — Clayton/Gumbel/Gaussian-Copulas für Tail-Dependence. Sehr gut für **bivariate** Tail-Risk-Analyse.

### Was state-of-the-art ist

**Vine-Copulas** (R-Vine, C-Vine, D-Vine) erlauben **höher-dimensionale** Copula-Modelle. Wenn du 10 Symbole im Portfolio hast, brauchst du ein 10-dimensionales Joint-Distribution-Modell — bivariate Copulas reichen nicht.

**Library:** **`pyvinecopulib`** (BSD-3, Cambridge-University) — die Standard-Library:
- Implementiert R-, C- und D-Vine-Copulas
- Mehr als 40 Bivariate-Copula-Familien als Bausteine (auch Joe, BB1, BB7, ...)
- Schnell (C++ Backend, Python-Bindings via pybind11)
- Truncation für sparse high-dim-Vines

**Empirical Copulas:**
**`scipy.stats.gaussian_kde`** als Marginal + **Empirical Copula** ist ein non-parametrisches Alternative zu deinen parametrischen Modellen. Bei dir ist das ein Fallback, wenn Clayton/Gumbel/Gaussian alle schlecht fitten.

### Konkrete Empfehlung

```python
# ml/vine_copula.py (Erweiterung)

import pyvinecopulib as pv
import numpy as np

class VineCopulaPortfolio:
    """High-dimensional dependence model via R-Vine."""
    
    def __init__(self, returns: pd.DataFrame):
        # Convert each margin to uniform via empirical CDF
        self.marginals = {col: returns[col].rank() / (len(returns) + 1) 
                          for col in returns.columns}
        u_data = np.column_stack(list(self.marginals.values()))
        
        # Fit R-Vine
        controls = pv.FitControlsVinecop(
            family_set=[pv.BicopFamily.gaussian, pv.BicopFamily.clayton,
                        pv.BicopFamily.gumbel, pv.BicopFamily.frank,
                        pv.BicopFamily.joe, pv.BicopFamily.bb1, pv.BicopFamily.bb7],
            num_threads=4,
            truncation_level=3,  # truncate beyond 3rd tree -> sparse
        )
        self.model = pv.Vinecop(u_data, controls=controls)
    
    def joint_tail_probability(self, threshold: float = 0.05) -> float:
        """P(all assets < threshold simultaneously)."""
        u_threshold = np.full(self.model.dim, threshold)
        return self.model.cdf(u_threshold)
```

**Use-Case:** Stress-Tests, die nicht nur "alle Returns korreliert" sondern "alle gleichzeitig im Tail" simulieren. Dein `risk/correlation_guard.py` und `risk/crowding_detector.py` könnten massiv von einem Vine-Copula profitieren.

**Aufwand:** 8-15h. **Lizenz:** pyvinecopulib BSD-3.

---

## A4. triple_barrier — mlfinlab-Vergleich

### Was du hast

`features/triple_barrier.py` (existiert)

### Was die Standard-Implementierung macht

**`hudson-and-thames/mlfinlab`** (kommerziell seit ~2022, aber Konzept ist López de Prado public). Die kanonische Implementation hat:

1. **Volatility-adaptive Barriers** statt fester Prozent-Werte
2. **Vertical-Barrier mit Bar-Count** statt fixer Tage (für Information-driven Bars)
3. **Sample-Weights via `bins`** — Labels werden mit Concurrent-Events-Korrektur gewichtet (wichtig für überlappende Labels!)
4. **Multiprocessing-Implementation** für große Datenmengen

### Was bei dir wahrscheinlich fehlt

Schau dir konkret an:
- Hast du **Sample-Weights für Concurrent Events** in deinem ML-Training? (López de Prado Chapter 4)
- Wie groß ist der Performance-Hit, wenn du `triple_barrier_labels` für eine Million Events ausführst?

Das ist eines der subtilen, aber wichtigen Konzepte in AFML — wenn du Triple-Barrier-Labels generierst, sind viele davon **zeitlich überlappend**. Wenn du Modell darauf naiv trainierst, kriegst du implizite Sample-Bias.

```python
# Pseudocode für Concurrent-Events-Korrektur (aus López de Prado Ch. 4)

def get_concurrent_events(close_idx, t1):
    """Count how many other labels overlap each label."""
    iloc = close_idx.searchsorted(np.array([t1.index[0], t1.max()]))
    iloc[1] += 1
    count = pd.Series(0, index=close_idx[iloc[0]:iloc[1]])
    for tIn, tOut in t1.items():
        count.loc[tIn:tOut] += 1
    return count

def get_avg_uniqueness(t1, num_co_events):
    """Compute average uniqueness of each label over its lifetime."""
    weight = pd.Series(index=t1.index)
    for tIn, tOut in t1.items():
        weight.loc[tIn] = (1.0 / num_co_events.loc[tIn:tOut]).mean()
    return weight

# Use as sample weights when training:
sklearn_clf.fit(X, y, sample_weight=weight)
```

### Empfehlung

Vergleich deine `triple_barrier.py` Zeile-für-Zeile mit der mlfinlab-Implementation. Achte besonders auf:
1. Sample-Weights über Concurrent-Events — falls fehlt, ist das eine niedrig-hängende Frucht
2. Sequentially Bootstrapped Bagging — wenn du Tree-Ensembles nutzt

**Aufwand:** 4-6h Vergleich, 6-8h für Concurrent-Events-Korrektur falls nicht vorhanden.

---

## A5. factor_exposures — toraniko Barra-Style Risk Model

### Was du hast

`risk/factor_exposures.py` + `risk/factor_models.py`. Wahrscheinlich Fama-French-Style oder ähnlich. Aber: **kein vollständiges Barra-Style Risk Model** mit Market/Sector/Style-Decomposition.

### Was toraniko anders macht

**`0xfdf/toraniko`** ist ein **vollständiges institutional-grade Risk Model** mit nur numpy + polars als Dependencies. Das ist ein sehr guter Fund:

- **Market Factor**: globaler Market-Return
- **Sector Factors**: jeden Sektor als Dummy-Variable
- **Style Factors**: Value, Size, Momentum eingebaut + erweiterbar
- **Custom Factors**: einfach custom Style-Factors hinzufügen
- **Cross-sectional Regression** zur Schätzung der Factor-Returns
- **Factor Covariance Matrix** für Portfolio-Optimierung mit Factor-Exposure-Constraints

Das Repo dokumentiert dass es "Barra-Faktor-Returns approximativ reproduziert".

### Warum das wichtig ist

Wenn du eine Strategie hast, die hauptsächlich Momentum-Exposure ist, ist sie **riskanter** als sie aussieht — Momentum-Crashs (1999, 2009, 2020) treffen sie alle gleichzeitig. Ein Barra-Style-Risk-Modell zerlegt deine Portfolio-Returns in:
- X% Market-Beta-Risk
- Y% Sector-Concentration-Risk
- Z% Style-Factor-Risk (Momentum/Value/Size...)
- W% Idiosyncratic-Risk (das eigentliche Alpha)

### Konkrete Empfehlung

```python
# risk/barra_risk_model.py (Vorschlag)

import polars as pl
from toraniko.model import estimate_factor_returns
from toraniko.styles import factor_mom, factor_sze, factor_val

class BarraRiskModel:
    def __init__(self, returns: pl.DataFrame, fundamentals: pl.DataFrame):
        mom = factor_mom(returns, lookback=252, lag=21)
        sze = factor_sze(fundamentals)  # market cap
        val = factor_val(fundamentals)  # book/price
        self.style_scores = pl.concat([mom, sze, val], how="horizontal")
    
    def decompose_portfolio_risk(
        self,
        portfolio_weights: pl.DataFrame,
    ) -> dict:
        """How much of portfolio variance comes from market/sector/style/idio?"""
        return {
            "market_var_pct": ...,
            "sector_var_pct": ...,
            "style_var_pct": ...,
            "idio_var_pct": ...,
        }
```

**Was du gewinnst:**
- "Mein Sharpe 2.0 — aber ist das Alpha oder Momentum-Style-Exposure?"
- Anti-Crowding-Detektor: wenn alle deine Strategien gleiche Style-Faktoren laden
- Risk-Parity über Style-Faktoren statt nur über Symbole

**Aufwand:** 12-20h für vollständige Integration. **Lizenz:** toraniko ist MIT.

---

## A6. meta_model — Meta-Labeling-Validierung

### Was du hast

`signals/meta_model.py` — Meta-Labeling existiert. Aber: **läuft es richtig?**

### Häufige Fehler in Meta-Labeling-Implementations

1. **Information-Leakage zwischen Primary und Meta**: Wenn das Primary-Modell und das Meta-Modell auf den **gleichen Daten** trainiert werden, lernt das Meta-Modell, das Primary perfekt vorherzusagen — auf den **Test-Daten** versagt es dann.

2. **Sample-Weight Mismatch**: Wenn das Primary-Modell nur 10% der Zeit "Trade" sagt, hat das Meta-Modell ein extrem unausgewogenes Trainings-Set.

3. **Threshold-Tuning auf Test-Set**: Der Confidence-Threshold (0.5? 0.6? 0.7?) MUSS auf einem separaten Validation-Set gewählt werden, nicht auf dem Test-Set.

4. **Concurrent-Events-Korrektur** (siehe A4): wenn Primary über mehrere Bars Position hält, sind die Labels überlappend.

### Validation-Tests für dein meta_model.py

```python
# qa/leakage_tests/meta_model_leakage_test.py (Vorschlag)

def test_meta_model_no_leakage():
    """Critical: meta-model must not see primary's training data."""
    train_period = ("2020-01-01", "2022-12-31")
    val_period   = ("2023-01-01", "2023-06-30")
    test_period  = ("2023-07-01", "2024-12-31")
    
    # 1. Train PRIMARY on training period
    primary = train_primary_strategy(returns.loc[train_period])
    
    # 2. Generate primary signals on VALIDATION period (out-of-sample!)
    primary_val_signals = primary.predict(returns.loc[val_period])
    
    # 3. Triple-barrier labels for val period
    val_labels = triple_barrier(
        prices=returns.loc[val_period],
        events=primary_val_signals.index[primary_val_signals != 0],
    )
    
    # 4. Train META only on validation period
    meta = train_meta_model(
        features=val_features,
        labels=val_labels,
    )
    
    # 5. Evaluate combined system on TEST period (clean OOS)
    train_test_correlation = correlation(
        primary_train_predictions, meta_predictions_on_train
    )
    assert train_test_correlation < 0.1, (
        f"Meta-model is leaking from primary training data!"
    )
```

### Konkrete Empfehlung

Schau dir an:
1. Wo in deinem Code der Primary trainiert wird
2. Wo der Meta trainiert wird
3. Sind die Trainings-Daten **wirklich disjoint**?

Wenn nicht — das ist ein subtiler Bug, der dein Backtest-Sharpe um 0.5-1.0 zu hoch macht.

**Aufwand:** 6-10h für ein striktes Leakage-Test-Setup. **Du hast schon `qa/leakage_tests/`** — also passt es perfekt rein.

---

## A7. conformal_position — Puncc/MAPIE-Vergleich

### Was du hast

`portfolio/conformal_position.py` — Conformal Prediction für Position-Sizing. Vermutlich basierend auf einem split-conformal-Approach.

### Was state-of-the-art ist

**`deel-ai/puncc`** (Apache 2.0) — sehr ausgereifte Conformal-Prediction-Library:
- Split CP, Cross-CP, Jackknife+, CV+
- **Adaptive Conformal Inference (ACI)** für Streaming-Settings
- Conformal Quantile Regression
- Time-Series-spezifische Methoden (z.B. EnbPI für Time-Series)

**`scikit-learn-contrib/MAPIE`** (BSD-3) — die scikit-learn-kompatible Standard-Library

**`darts.models.forecasting.conformal_models`** — Conformal Prediction direkt für Time-Series-Forecasts.

### Was bei dir wahrscheinlich fehlt

**Adaptive Conformal Inference (ACI)** — passt den Konfidenz-Level dynamisch an:
- Bei steigender Volatilität: weitere Intervalle
- Bei stabilen Märkten: engere Intervalle
- Self-correcting: wenn Coverage schlechter ist als geplant, adjustiert es sich

Genau das, was Trading braucht — feste Konfidenz-Levels sind in nicht-stationären Märkten schlecht.

### Konkrete Empfehlung

```python
# portfolio/adaptive_conformal_position.py (Erweiterung)

from puncc.regression.split_cp_aci import AdaptiveConformalInference

class AdaptiveConformalSizer:
    def __init__(self, base_predictor, alpha: float = 0.1, gamma: float = 0.005):
        self.aci = AdaptiveConformalInference(
            predictor=base_predictor,
            alpha=alpha,
            gamma=gamma,
        )
    
    def update_and_size(
        self,
        new_features: pd.DataFrame,
        last_realized_outcome: float | None,
    ) -> dict:
        if last_realized_outcome is not None:
            self.aci.update(last_realized_outcome)
        
        y_lower, y_upper = self.aci.predict(new_features, alpha=self.alpha)
        interval_width = y_upper - y_lower
        confidence = 1.0 / (1.0 + interval_width)
        position_size = confidence * self.max_position
        
        return {
            "position_size": position_size,
            "current_alpha": self.aci._current_alpha,
        }
```

**Aufwand:** 6-10h. **Lizenz:** puncc Apache 2.0.

---

## A8. attribution — Brinson-Hood-Beebower & Karnosky-Singer

### Was du hast

`attribution/composite.py` (66 LOC), `schemas.py` (49 LOC), `storage.py` (97 LOC). Total: ~210 LOC.

Sehr klein. Vermutlich basic Performance-Attribution.

### Was state-of-the-art ist

**Brinson-Hood-Beebower (1986/1988)** — der klassische Attribution-Standard:

Decomposiert Active Return in:
- **Allocation Effect**: durch Sektor-Tilts (Übergewichtung gewinnender Sektoren)
- **Selection Effect**: durch Stock-Picking innerhalb Sektoren
- **Interaction Effect**: Kreuz-Effekt zwischen Allocation und Selection

Mathematisch:
```
Active Return = Σᵢ (wᵢ_p - wᵢ_b) · rᵢ_b   [Allocation]
              + Σᵢ wᵢ_b · (rᵢ_p - rᵢ_b)   [Selection]
              + Σᵢ (wᵢ_p - wᵢ_b)·(rᵢ_p - rᵢ_b)  [Interaction]
```

**Karnosky-Singer (1994)** — Erweiterung für Multi-Currency-Portfolios:
- Trennt Currency-Hedging-Effect von Asset-Allocation
- Wichtig wenn du USD-Symbole hältst und EUR der Base-Currency ist

### Konkrete Empfehlung

```python
# attribution/brinson.py (Vorschlag)

class BrinsonAttribution:
    """Brinson-Hood-Beebower (1986) attribution."""
    
    def __init__(self, portfolio_weights: pd.DataFrame, benchmark_weights: pd.DataFrame):
        self.w_p = portfolio_weights
        self.w_b = benchmark_weights
    
    def attribute(
        self,
        sector_returns_portfolio: pd.DataFrame,
        sector_returns_benchmark: pd.DataFrame,
    ) -> pd.DataFrame:
        weight_diff = self.w_p - self.w_b
        
        allocation = (weight_diff * sector_returns_benchmark).sum(axis=1)
        selection = (self.w_b * (sector_returns_portfolio - sector_returns_benchmark)).sum(axis=1)
        interaction = (weight_diff * (sector_returns_portfolio - sector_returns_benchmark)).sum(axis=1)
        
        return pd.DataFrame({
            "allocation": allocation,
            "selection": selection,
            "interaction": interaction,
            "active_total": allocation + selection + interaction,
        })
```

**Aufwand:** 6-10h für solide Brinson-Hood-Beebower. Karnosky-Singer +4-6h.

**Wo das eingehängt wird:** dein `qa/performance_attribution.py` (existiert!) wäre der Place-to-be.

---

## A9. certify — Audit-Trail-Standards

### Was du hast

`certify/generator.py` (245 LOC), `certify/schema.py` (79 LOC). Vermutlich generiert es Audit-Reports / Schema-Validation für Backtest-Outputs.

### Was state-of-the-art ist (Reproducibility-Standards)

**Drei Säulen für reproduzierbare ML-Experimente:**

1. **Code-Versioning**: Git-Commit-Hash zu jedem Run
2. **Data-Versioning**: DVC-Hashes für Inputs (siehe B4 unten)
3. **Run-Tracking**: MLflow oder W&B-Run-IDs

**`MLflow`** (Apache 2.0) — Standard für Run-Tracking:
- `mlflow.start_run()` als Wrapper um jeden Backtest
- Automatic Logging von Params, Metrics, Artifacts
- Web-UI für Run-Vergleich

**`DVC`** (Apache 2.0) — Standard für Data-Versioning

### Konkrete Empfehlung

Erweitere `certify/` um automatisches Run-Tracking:

```python
# certify/mlflow_integration.py (Vorschlag)

import mlflow
import git

class CertifiedBacktestRunner:
    """Wrapper that ensures every backtest is fully reproducible."""
    
    def __init__(self, mlflow_uri: str = "file:./mlruns"):
        mlflow.set_tracking_uri(mlflow_uri)
        self.repo = git.Repo(Path.cwd())
    
    def run(self, backtest_fn, params: dict, experiment_name: str):
        # Ensure clean working tree
        if self.repo.is_dirty(untracked_files=False):
            raise RuntimeError(
                "Working tree has uncommitted changes! "
                "Commit before running for reproducibility."
            )
        
        commit_hash = self.repo.head.commit.hexsha
        
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run() as run:
            mlflow.log_params(params)
            mlflow.log_param("git_commit", commit_hash)
            mlflow.log_artifact("requirements.lock")
            
            result = backtest_fn(**params)
            
            for metric_name, value in result.metrics.items():
                mlflow.log_metric(metric_name, value)
            
            # Generate certificate via your existing certify/generator.py
            cert = generate_certificate(
                run_id=run.info.run_id,
                git_commit=commit_hash,
                params=params,
                metrics=result.metrics,
            )
            mlflow.log_artifact(cert.path)
        
        return result, run.info.run_id
```

**Aufwand:** 8-12h für saubere MLflow-Integration. DVC-Setup +4-6h.

---

## A10. metrics_exporter — Custom-Metrics für Trading

### Was du hast

`ops/metrics_exporter.py` — vermutlich Prometheus-Metrics-Exporter (basierend auf Modulnamen). Verbunden mit deinem `ops/health_check.py`, `ops/heartbeat.py`, `ops/alert_manager.py`.

### Was Standard ist (vs. was Trading-spezifisch ist)

Standard-Web-App-Metrics (HTTP-Latency, Error-Rate) hast du wahrscheinlich. Trading-spezifische Metrics sind:

```python
# ops/trading_specific_metrics.py (Erweiterung-Vorschlag)

from prometheus_client import Counter, Histogram, Gauge

# Order-related
orders_submitted = Counter(
    "trading_orders_submitted_total",
    "Total orders submitted",
    labelnames=["strategy", "symbol", "side", "order_type"],
)

orders_rejected = Counter(
    "trading_orders_rejected_total",
    "Orders rejected by broker or pre-trade-check",
    labelnames=["strategy", "reason"],  # 'pdt', 'fat_finger', 'kill_switch', etc.
)

# Slippage analysis
slippage_bps_signed = Histogram(
    "trading_slippage_bps_signed",
    "Slippage relative to decision-time mid (signed)",
    labelnames=["strategy", "symbol", "side"],
    buckets=[-200, -50, -20, -10, -5, -1, 0, 1, 5, 10, 20, 50, 200],
)

# Risk metrics
drawdown_current_pct = Gauge(
    "trading_drawdown_current_pct",
    "Current drawdown from equity peak (positive number)",
    labelnames=["strategy"],
)

gross_exposure = Gauge(
    "trading_gross_exposure_usd",
    "Gross exposure (long + short) in USD",
    labelnames=["strategy"],
)

# Drift integration (you have ops/drift_monitor.py!)
feature_psi = Gauge(
    "trading_feature_psi",
    "Population Stability Index per feature",
    labelnames=["feature_name"],
)

# Kill-switch state
kill_switch_active = Gauge(
    "trading_kill_switch_active",
    "1 if kill switch is active, 0 otherwise",
    labelnames=["scope"],  # global, strategy, symbol
)
```

### Konkrete Empfehlung

Du hast `metrics_exporter.py`. Erweitere es um diese Trading-spezifischen Metrics. Dann baue 3-4 **Standard-Grafana-Dashboards**:

1. **Slippage-Dashboard** — pro Strategie/Symbol, mit Histogram + Time-Series
2. **Drawdown-Dashboard** — Drawdown-Pfad pro Strategie, gegen Equity-Peak
3. **Order-Flow-Dashboard** — Submitted vs. Filled vs. Rejected, mit Reasons
4. **Drift-Dashboard** — PSI pro Feature über Zeit, Trigger-Events markiert

**Aufwand:** Custom-Metrics 4-6h, Grafana-Dashboards 6-10h.

---

# Teil B — Komplett neue Themen

## B1. Synthetic Financial Time Series

### Worum es geht

In Backtests hast du **eine** historische Realisierung. Aber: war 2020-2024 repräsentativ? Wahrscheinlich nicht. Synthetic Data generiert **viele plausible Alternativ-Verläufe**, die du gegen deine Strategien laufen lassen kannst.

### Drei Ansätze

**1. TimeGAN (NeurIPS 2019, Yoon et al.)**
- GAN, das Time-Series mit korrekten zeitlichen Abhängigkeiten generiert
- **Repo:** `stefan-jansen/synthetic-data-for-finance` (MIT) — exakte Referenz-Implementation
- Pros: Realistisch wirkende Stylized Facts
- Cons: Schwer zu trainieren, kann Mode-Collapse erleiden

**2. TC-VAE (Time-Causal Variational Autoencoder, 2024)**
- Variational Autoencoder mit Causality-Constraints
- **Repo:** `justinhou95/TimeCausalVAE`
- Pros: Stabiler als TimeGAN, theoretisch sauber begründet
- Cons: Neu, weniger Tooling

**3. Sig-Wasserstein GAN**
- Nutzt **Path Signatures** (Rough Path Theory) zur Time-Series-Charakterisierung
- **Repo:** `SigCGANs/Sig-Wasserstein-GANs`

### Konkrete Empfehlung für dich

Wenn du Trading-Strategien rigoros validieren willst:

```python
# qa/synthetic_validation.py (Vorschlag)

class SyntheticBacktestValidator:
    """Validate strategy on synthetic price paths."""
    
    def validate_strategy(
        self,
        strategy,
        n_synthetic_paths: int = 1000,
        path_length_days: int = 252,
    ) -> dict:
        """Run strategy on N synthetic paths, compute distribution of metrics."""
        sharpes = []
        max_dds = []
        
        for _ in range(n_synthetic_paths):
            synthetic_returns = self.generator.generate(length=path_length_days)
            result = strategy.backtest(synthetic_returns)
            sharpes.append(result.sharpe)
            max_dds.append(result.max_drawdown)
        
        # Compare real-data result to distribution
        real_result = strategy.backtest(self.real_returns)
        
        return {
            "real_sharpe": real_result.sharpe,
            "synthetic_sharpe_5pct": np.percentile(sharpes, 5),
            "synthetic_sharpe_95pct": np.percentile(sharpes, 95),
            "real_in_synthetic_distribution": (
                np.percentile(sharpes, 5) <= real_result.sharpe <= np.percentile(sharpes, 95)
            ),
        }
```

**Was du gewinnst:** Wenn dein Real-Sharpe 2.0 ist, aber 95% der Synthetic-Sharpes zwischen -0.5 und +1.0 liegen, ist dein Real-Sharpe **außerhalb des Plausibilitätsbereichs** → Hinweis auf Overfit oder Glück.

**Aufwand:** Hoch (40-80h für Generator-Training). **Niedrige Priorität**, aber für strenge Validierung Goldstandard.

**Lizenz:** TimeGAN (MIT), TC-VAE (Apache 2.0).

---

## B2. Volatility Surface Modeling (SVI, SABR)

### Was du hast

`signals/options_iv.py`, `features/options_derived_signals.py`. Du nutzt Implied-Volatility-Daten als Features. Aber: vermutlich keine **kalibrierte Volatility-Surface**.

### Was state-of-the-art ist

**Stochastic Volatility Inspired (SVI) Parameterization** (Gatheral 2004):
```
σ²(k) = a + b · {ρ(k - m) + √[(k - m)² + σ²]}
```
mit 5 Parametern (a, b, ρ, m, σ) pro Maturity. Vorteile:
- Closed-form, schnell zu fitten
- Arbitrage-Freiheit testbar via Bedingungen auf die Parameter
- Standard in der Industrie

**SABR-Modell** (Hagan et al. 2002):
- Stochastic-Volatility-Modell, das Smile + Term-Structure modelliert
- Closed-form Approximation für IV als Funktion von Strike und Maturity
- Standard für FX und Rates

### Open-Source-Implementations

**`XanderRobbins/Arbitrage-Free-Volatility-Surface`** — komplette SVI + Heston Pipeline mit:
- IV-Solver (Newton-Raphson + Brent fallback)
- Arbitrage-Checks (Butterfly, Calendar, Put-Call-Parity)
- SVI-Calibration
- Heston-Modell (COS-Methode)

**`vilkovgr/qmoms`** — Option-Implied Moments aus IV-Surface:
- Risk-Neutral Variance, Skewness, Kurtosis
- CVIX (Andersen-Bondarenko-Gonzalez-Perez)
- Tail-Loss-Measure (TLM)

**`pysabr`** — Python-SABR-Implementation.

### Konkrete Empfehlung

Wenn du Options-Daten als Signal-Features nutzt, lohnt sich der Schritt zur kalibrierten Surface:

```python
# features/volatility_surface.py (Vorschlag)

class VolatilitySurfaceFeatures:
    """Extract features from a calibrated SVI volatility surface."""
    
    def __init__(self, spot_price: float, risk_free: float = 0.05):
        self.surface = VolatilitySurface(S=spot_price, r=risk_free)
    
    def get_features(self) -> pd.Series:
        """Extract trading-relevant features from the surface."""
        features = {}
        
        # ATM-IV per maturity
        for dte in [7, 30, 60, 90]:
            features[f"atm_iv_{dte}d"] = self.surface.atm_iv(dte)
        
        # Term structure slope
        features["ts_slope_short"] = (
            self.surface.atm_iv(30) - self.surface.atm_iv(7)
        )
        
        # 25-delta skew (put-call IV gap)
        features["skew_25d_30d"] = self.surface.skew_25d(30)
        
        # Risk-reversal & Butterfly
        features["risk_reversal_25d_30d"] = self.surface.risk_reversal(0.25, 30)
        features["butterfly_25d_30d"] = self.surface.butterfly(0.25, 30)
        
        # Gamma exposure (GEX) signal
        features["gex_normalized"] = self.surface.gex() / self.surface.spot_price
        
        # Variance Risk Premium (IV - RV)
        rv_30d = self.realized_vol_30d
        features["vrp_30d"] = features["atm_iv_30d"] - rv_30d
        
        return pd.Series(features)
```

**Was du gewinnst (was deine `options_iv.py` wahrscheinlich nicht hat):**
- Konsistente IV-Werte (kein Single-Strike-IV, sondern smile-aware)
- Arbitrage-Free-Garantie
- 25-Delta-Skew als sehr starkes Sentiment-Signal
- VRP (IV - RV) als klassisches Vol-Premium-Signal
- GEX (Gamma Exposure) als Dealer-Positioning-Indikator

**Aufwand:** 16-30h. Anspruchsvoll, aber sehr wertvoll wenn du Optionsdaten nutzt.

**Lizenz:** XanderRobbins (Repo nicht explizit lizenziert — Code-Studium OK, aber nicht Code-Übernahme). pysabr (MIT).

---

## B3. Causal Inference für Trading

### Worum es geht

Klassisches ML: "Wenn Feature X = 5, dann Return = +0.5%."
Causal Inference: "Wenn ich **interveniere** und X auf 5 setze, was wäre der Effekt auf Return?"

Beispiele wo Causal Inference relevant ist:
- "Würde meine Strategie ohne News-Feature genauso performen?" (Counterfactual)
- "Welcher meiner Features hat **kausalen** Einfluss vs. nur korrelative?" (Backdoor-Adjustment)
- "Welche meiner Trades waren **wirklich** durch das Signal getrieben vs. durch Glück?" (Treatment-Effect-Schätzung)

### Drei Standard-Libraries

**1. `microsoft/EconML`** — die akademisch saubere Library:
- DoubleML, Causal Forest, X-Learner, R-Learner
- Heterogeneous Treatment Effects (HTE)
- MIT-Lizenz

**2. `uber/causalml`** — die praktische Library:
- Uplift Modeling Focus
- Tree-based + Neural Net-based Estimators
- Apache 2.0

**3. `py-why/dowhy`** — die end-to-end Library:
- Identifikation + Estimation + Refutation in einem Workflow
- Causal-Graph-basiert (DAGs als Input)
- MIT-Lizenz

### Konkreter Use-Case für Trading

**Frage:** "Welcher meiner Features hat den größten **kausalen** Einfluss auf zukünftige Returns, kontrolliert für andere Features?"

```python
# qa/feature_causal_importance.py (Vorschlag)

from econml.dml import LinearDML

class CausalFeatureImportance:
    """Estimate causal importance of features using Double ML."""
    
    def estimate(
        self,
        features: pd.DataFrame,
        target_returns: pd.Series,
        feature_of_interest: str,
        controls: list[str] | None = None,
    ) -> dict:
        T = features[feature_of_interest].values  # treatment
        Y = target_returns.values                  # outcome
        X = features[controls or []].values        # controls
        
        est = LinearDML(model_y="auto", model_t="auto", cv=5)
        est.fit(Y=Y, T=T, X=X if len(controls) > 0 else None, W=X)
        
        ate = est.ate()
        ate_lower, ate_upper = est.ate_interval(alpha=0.05)
        
        return {
            "feature": feature_of_interest,
            "causal_effect": ate,
            "ci_95_lower": ate_lower,
            "ci_95_upper": ate_upper,
            "is_significant": (ate_lower > 0 or ate_upper < 0),
        }
```

**Was du gewinnst:**
- Statt SHAP (zeigt Korrelations-Importance) bekommst du Causal-Importance
- Features mit hoher SHAP-Importance, aber Causal-Effect ≈ 0 sind **Spurious** — sie korrelieren nur mit anderen kausalen Features
- Hilft dir, den "echten" Alpha-Driver vom "Confounder" zu trennen

**Anwendung:** Vor jedem ML-Modell-Retraining: laufe Causal-Feature-Importance-Test → behalte nur Features mit |ATE / SE| > 2 (signifikanter Causal-Effect).

**Aufwand:** 12-20h. **Lizenz:** Alle MIT/Apache. Kein Lizenz-Problem.

---

## B4. MLflow + DVC für Experiment-Tracking

### Was du hast

`qa/experiment_tracking.py`, `strategy/experiment_tracker.py`, `experiments/batch_config.py`. Solides eigenes System.

### Was MLflow + DVC zusätzlich bringen

| Feature | Dein eigenes System | MLflow + DVC |
|---|---|---|
| Run-IDs | wahrscheinlich ja | ja (UUID) |
| Code-Hash-Tracking | wahrscheinlich ja | ja, automatisch |
| Param-Logging | ja | ja, mit Auto-Logging-Hooks |
| Metric-Logging | ja | ja |
| Artifact-Storage | wahrscheinlich Filesystem | strukturiert, mit Backend-Optionen |
| Web-UI für Run-Vergleich | wahrscheinlich nicht | ja, in 3 Klicks |
| Data-Versioning | unwahrscheinlich | DVC: Git-like für Daten |
| Pipelines | wahrscheinlich Custom-Skripte | DVC: deklarative `dvc.yaml` |
| Model Registry | wahrscheinlich nicht | MLflow: stage promotions (staging → prod) |

### Konkrete Empfehlung

Du musst MLflow nicht statt deines Systems nutzen — du kannst beides parallel laufen lassen:

```python
# qa/experiment_tracking.py (Erweiterung)

import mlflow

class ExperimentTracker:
    """Wrapper around your existing tracker + MLflow."""
    
    def __init__(self, mlflow_enabled: bool = True):
        self.mlflow_enabled = mlflow_enabled
    
    def start_run(self, experiment_name: str, params: dict):
        run_id = self._create_run(experiment_name, params)
        
        if self.mlflow_enabled:
            mlflow.set_experiment(experiment_name)
            mlflow_run = mlflow.start_run(run_name=run_id)
            mlflow.log_params(params)
            self.mlflow_run_id = mlflow_run.info.run_id
        
        return run_id
    
    def log_metric(self, name: str, value: float, step: int | None = None):
        self._log_metric_local(name, value, step)
        if self.mlflow_enabled:
            mlflow.log_metric(name, value, step=step)
```

**DVC-Integration für Daten:**

```yaml
# dvc.yaml — define your data pipeline
stages:
  prepare_features:
    cmd: python scripts/prepare_features.py
    deps:
      - scripts/prepare_features.py
      - data/raw/prices.parquet
    outs:
      - data/features/features.parquet
  
  train_model:
    cmd: python scripts/train.py --config configs/model_v1.yaml
    deps:
      - scripts/train.py
      - data/features/features.parquet
    outs:
      - models/model_v1.pkl
    metrics:
      - results/model_v1_metrics.json
```

Dann: `dvc repro` führt alle Stages aus, die durch geänderte Inputs neu gemacht werden müssen.

**Aufwand:** 8-12h. **Niedriger Risiko-Pfad**, weil parallel zu deinem System.

---

## B5. DuckDB als Backtest-Datastore

### Worum es geht

Du hast wahrscheinlich Parquet-Files in `data/`. Bei großen Backtests wird das Lesen langsam, weil du IMMER alles laden musst.

**DuckDB** ist eine **embedded analytical database**:
- Eine einzelne `.duckdb`-Datei pro Repo
- Sehr schneller analytischer SQL-Layer
- Native Parquet-Reads (kann direkt Parquet-Files queryen ohne Import)
- Python-API auf Pandas/Polars-Niveau

### Use-Cases für dich

**1. Time-Travel-Queries** (PIT-Universe-Korrektheit):
```python
import duckdb

con = duckdb.connect("data/feature_store.duckdb")

features_as_of_2023_06_15 = con.sql("""
    SELECT *
    FROM features
    WHERE timestamp <= '2023-06-15'
      AND snapshot_ts <= '2023-06-15'
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY symbol, feature_name
        ORDER BY snapshot_ts DESC
    ) = 1
""").pl()
```

**2. Schnelle Universe-Membership** (PIT-Universe):
```sql
-- Was im S&P 500 am 2023-06-15?
SELECT symbol
FROM index_membership
WHERE index_name = 'SPX'
  AND from_date <= '2023-06-15'
  AND (to_date IS NULL OR to_date > '2023-06-15')
```

**3. Joins über mehrere Datenquellen**:
```sql
SELECT 
    p.symbol, p.timestamp, p.close,
    f.feature_value,
    n.news_score
FROM prices p
LEFT JOIN features f USING (symbol, timestamp)
LEFT JOIN news_signals n USING (symbol, timestamp)
WHERE p.timestamp BETWEEN '2024-01-01' AND '2024-06-30'
```

### Konkrete Empfehlung

```python
# data/duckdb_store.py (Vorschlag)

import duckdb

class DuckDBDataStore:
    """Read-only analytical layer over your Parquet files."""
    
    def __init__(self, parquet_root: Path):
        self.con = duckdb.connect(":memory:")
        self.parquet_root = parquet_root
        self._register_tables()
    
    def _register_tables(self):
        self.con.sql(f"""
            CREATE VIEW prices AS 
            SELECT * FROM read_parquet('{self.parquet_root}/prices/*.parquet')
        """)
        self.con.sql(f"""
            CREATE VIEW features AS
            SELECT * FROM read_parquet('{self.parquet_root}/features/*.parquet')
        """)
```

**Was du gewinnst:**
- 10-100x schnellere analytische Queries vs. naive Pandas-Iteration
- SQL für komplexe Joins ohne Performance-Probleme
- Memory-efficient: kein "lade 50GB ins RAM"
- Native Polars-Compatibility

**Aufwand:** 8-12h. **Lizenz:** DuckDB MIT, Python-API MIT.

---

## B6. Cross-Asset Carry Strategies

### Worum es geht

**Carry** ist eine der robustesten Anomalien im Trading. Hat in jeder Asset-Klasse eine Variation:
- **FX-Carry**: höhere Zinsen → Long, niedrigere → Short
- **Bond-Carry**: längere Maturity vs. kürzere → Roll-Down
- **Equity-Carry**: höhere Dividend-Yield → Long
- **Commodity-Carry**: Backwardation (Future < Spot) → Long, Contango → Short
- **Crypto-Carry**: Funding-Rates auf Perp-Futures → harvest-able

### Du hast: `signals/cross_asset_carry.py` 

Existiert in deinem Repo. Aber: vermutlich nur **eine** Asset-Klasse oder nur Equity-Dividend-Carry.

### State-of-the-Art

**Asness, Moskowitz, Pedersen (2013) — "Value and Momentum Everywhere"**: Pioneer-Paper, das zeigt wie Carry über Asset-Klassen hinweg funktioniert.

```python
# signals/cross_asset_carry_v2.py (Erweiterung)

class UniversalCarrySignal:
    """Cross-asset carry signal (FX, Bonds, Commodities, Crypto)."""
    
    def fx_carry(self, rate_differentials: pd.DataFrame) -> pd.DataFrame:
        """Long high-yield currencies, short low-yield."""
        ranked = rate_differentials.rank(axis=1, pct=True)
        return (ranked - 0.5) * 2  # -1 to +1
    
    def commodity_carry(self, futures_curves: pd.DataFrame) -> pd.DataFrame:
        """Long backwardation, short contango (1-month roll)."""
        front = futures_curves.xs("M1", level="contract")
        second = futures_curves.xs("M2", level="contract")
        carry = (front - second) / front
        ranked = carry.rank(axis=1, pct=True)
        return (ranked - 0.5) * 2
    
    def crypto_carry(self, funding_rates: pd.DataFrame) -> pd.DataFrame:
        """Long-spot/short-perp when funding is positive."""
        signal_strength = funding_rates.rolling(8).mean() * 365
        return signal_strength.clip(-1, 1)
```

**Aufwand:** 12-18h. **Lizenz:** Akademisch frei.

---

## B7. Term Structure Modelling

### Worum es geht

Term-Structure-Daten sind eine der reichsten Informationsquellen, die viele Retail-Trader ignorieren:
- **VIX-Futures-Term-Structure** (VIX, VIX1M, VIX3M, VIX6M)
- **Yield-Curve** (3M, 2Y, 10Y, 30Y Treasury)
- **Credit-Spreads** (IG vs. HY)
- **Commodity-Futures-Curves** (CL1, CL2, CL3, ...)

### Was bei dir wahrscheinlich fehlt

`features/intermarket_factors.py` und `features/macro_features.py` existieren — aber wahrscheinlich nur statische Features ("aktueller VIX-Wert"), nicht **Curve-Shape-Features**.

### Standard-Curve-Features

```python
# features/term_structure.py (Vorschlag)

class TermStructureFeatures:
    """Extract shape features from a futures/yield curve."""
    
    def vix_term_structure(self, vix_quotes: pd.DataFrame) -> pd.DataFrame:
        """vix_quotes columns: ['VIX', 'VIX1M', 'VIX3M', 'VIX6M']"""
        features = pd.DataFrame(index=vix_quotes.index)
        
        # Slope (carry signal)
        features["vix_slope_short"] = vix_quotes["VIX1M"] - vix_quotes["VIX"]
        features["vix_slope_long"] = vix_quotes["VIX3M"] - vix_quotes["VIX1M"]
        
        # Contango/Backwardation flag
        features["vix_contango"] = (vix_quotes["VIX1M"] > vix_quotes["VIX"]).astype(int)
        
        # Curvature (butterfly)
        features["vix_curvature"] = (
            vix_quotes["VIX"] - 2 * vix_quotes["VIX1M"] + vix_quotes["VIX3M"]
        )
        
        # PCA on the curve (load to first principal component)
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3)
        curve_changes = vix_quotes.diff().dropna()
        pca_components = pca.fit_transform(curve_changes)
        features.loc[curve_changes.index, "vix_pc1"] = pca_components[:, 0]  # level
        features.loc[curve_changes.index, "vix_pc2"] = pca_components[:, 1]  # slope
        features.loc[curve_changes.index, "vix_pc3"] = pca_components[:, 2]  # curvature
        
        return features
    
    def yield_curve_features(self, treasury_yields: pd.DataFrame) -> pd.DataFrame:
        """treasury_yields cols: ['3M', '2Y', '5Y', '10Y', '30Y']"""
        features = pd.DataFrame(index=treasury_yields.index)
        
        features["yc_2y10y"] = treasury_yields["10Y"] - treasury_yields["2Y"]
        features["yc_3m10y"] = treasury_yields["10Y"] - treasury_yields["3M"]
        features["yc_inverted"] = (treasury_yields["2Y"] > treasury_yields["10Y"]).astype(int)
        
        return features
```

**Was VIX-Term-Structure als Signal kann:**
- VIX1M/VIX > 1.05 (Contango) → "Short VIX" (verkaufe VXX, kaufe SVXY)
- VIX1M/VIX < 0.95 (Backwardation) → "Long VIX" (Crash-Hedge an, Risk-Off-Regime)

**Was Yield-Curve-Inversion als Signal kann:**
- 2y10y < 0 → bisher 8 von 9 Recessions vorhergesagt
- Trigger für Defensive-Strategie-Allokation

**Aufwand:** 8-12h. **Lizenz:** Akademisch frei.

---

## B8. Liquidity-Aware Position Sizing

### Was du hast

`risk/liquidity_scoring.py` — Liquidity-Score per Symbol. Aber: vermutlich nicht **Position-Sizing-Constraint**.

### Was state-of-the-art ist

**Beraldi-Lehalle-Almgren Liquidity-Sizing-Rule:**

Position-Größe wird begrenzt durch:
```
max_qty = min(
    risk_target_qty,                              # vom Signal/Kelly
    liquidity_pct * avg_daily_volume,             # nicht mehr als X% ADV
    max_dollar_loss / (volatility * price * 5),   # Stop-Loss-konform
)
```

Plus: **"Time-to-liquidate"-Constraint** — kannst du die Position in einem Tag schließen?

```python
# portfolio/liquidity_aware_sizer.py (Vorschlag)

class LiquidityAwareSizer:
    """Sizing that respects market-impact constraints."""
    
    def __init__(
        self,
        max_pct_adv: float = 0.05,
        max_pct_market_cap: float = 0.001,
        max_days_to_liquidate: float = 1.0,
        target_pov_pct: float = 0.10,
    ):
        self.max_pct_adv = max_pct_adv
        self.max_pct_mcap = max_pct_market_cap
        self.max_days_liq = max_days_to_liquidate
        self.target_pov = target_pov_pct
    
    def size_position(
        self,
        signal_target_qty: int,
        symbol_data: dict,
    ) -> dict:
        adv = symbol_data["avg_daily_volume"]
        mcap = symbol_data["market_cap"]
        price = symbol_data["price"]
        
        # Cap 1: % of ADV
        adv_cap = int(adv * self.max_pct_adv)
        
        # Cap 2: % of market cap
        mcap_cap = int(mcap / price * self.max_pct_mcap)
        
        # Cap 3: Time-to-liquidate
        liq_cap = int(self.max_days_liq * self.target_pov * adv)
        
        applicable_cap = min(adv_cap, mcap_cap, liq_cap)
        final_qty = min(signal_target_qty, applicable_cap)
        
        return {
            "signal_qty": signal_target_qty,
            "final_qty": final_qty,
            "binding_constraint": (
                "adv" if adv_cap == applicable_cap
                else "mcap" if mcap_cap == applicable_cap
                else "liquidity_time"
            ),
            "expected_days_to_liquidate": final_qty / (self.target_pov * adv),
        }
```

**Was du gewinnst:** Du gehst nicht versehentlich 30% ADV in einem Microcap, der 3 Tage zum Liquidieren braucht.

**Aufwand:** 6-10h.

---

## B9. Volume-Synchronized PIN (VPIN)

### Worum es geht

**VPIN** (Volume-Synchronized Probability of Informed Trading) — Easley, López de Prado, O'Hara (2012) — misst **toxicity** im Order-Flow.

Idee: Wenn Buy-Volume und Sell-Volume sehr unbalanced ist innerhalb eines Volume-Buckets, ist das ein Indiz für **Informierte Trader** (Insider, Algos mit Edge). Toxic Flow → Market-Maker ziehen sich zurück → Liquidity-Crash.

VPIN ist als Frühwarn-Signal für **Flash-Crashes** validiert. Beim Mai-2010-Flash-Crash spike VPIN ~30 Minuten **vor** dem Crash.

### Konkrete Implementierung

```python
# qa/vpin.py (Vorschlag)

class VPINCalculator:
    """Volume-Synchronized Probability of Informed Trading."""
    
    def __init__(self, n_buckets: int = 50, bucket_size_pct_adv: float = 0.01):
        self.n_buckets = n_buckets
        self.bucket_size_pct_adv = bucket_size_pct_adv
    
    def compute(self, trades: pd.DataFrame, avg_daily_volume: float) -> pd.Series:
        """
        trades: DataFrame with columns ['timestamp', 'price', 'size', 'side']
        side: +1 for buy, -1 for sell (use Lee-Ready tick rule if not given)
        """
        bucket_volume = avg_daily_volume * self.bucket_size_pct_adv
        
        buckets = []
        current_buy_vol = 0.0
        current_sell_vol = 0.0
        current_total = 0.0
        
        for _, trade in trades.iterrows():
            sz = trade["size"]
            if trade["side"] == 1:
                current_buy_vol += sz
            else:
                current_sell_vol += sz
            current_total += sz
            
            if current_total >= bucket_volume:
                buckets.append({
                    "timestamp": trade["timestamp"],
                    "imbalance": abs(current_buy_vol - current_sell_vol) / current_total,
                })
                current_buy_vol = 0.0
                current_sell_vol = 0.0
                current_total = 0.0
        
        bucket_df = pd.DataFrame(buckets).set_index("timestamp")
        vpin = bucket_df["imbalance"].rolling(self.n_buckets).mean()
        return vpin
```

**Anwendung als Trading-Signal:**

```python
vpin = compute_vpin(trades, adv)
if vpin.iloc[-1] > vpin.quantile(0.95):
    # Toxic flow detected -> de-risk
    apply_size_multiplier(0.5)
    if vpin.iloc[-1] > vpin.quantile(0.99):
        trigger_kill_switch("VPIN extreme: " + str(vpin.iloc[-1]))
```

**Was du hast vs. das hier:**
- Du hast `risk/crowding_detector.py` und `risk/market_stress.py` — VPIN würde **die** kanonische Mikrostruktur-Toxicity-Metrik hinzufügen
- Du hast `execution/transaction_costs.py` (1008 LOC) — VPIN könnte als Input dienen ("toxic flow → wider spread expected")

**Aufwand:** 8-12h. **Lizenz:** Akademisch frei (López de Prado Buch).

---

## B10. Bayesian Optimization für Execution Schedules

### Worum es geht

Du hast `execution/almgren_chriss.py`. Almgren-Chriss hat **Parameter** (η = temporary impact, γ = permanent impact, λ = risk aversion). Diese werden typisch **statisch** gesetzt.

**Bayesian-Optimization** kann diese Parameter **online** lernen:
- Beobachte tatsächliche Slippage über deine Trades
- Aktualisiere Posterior über η, γ
- Bei nächstem Trade: nutze neue Parameter

Damit lernt dein System **wie der Markt reagiert**, statt feste Lehrbuch-Parameter zu nutzen.

### Konkrete Implementation

```python
# execution/adaptive_execution.py (Vorschlag)

from skopt import gp_minimize
from skopt.space import Real

class AdaptiveAlmgrenChriss:
    """Almgren-Chriss with Bayesian-optimized parameters."""
    
    def __init__(self, prior_eta: float = 1e-4, prior_gamma: float = 1e-5):
        self.observations = []
        self.eta = prior_eta
        self.gamma = prior_gamma
    
    def record_execution(self, qty: int, decision_price: float, executed_price: float):
        """Record the result of an execution to update parameters."""
        self.observations.append({
            "qty": qty,
            "slippage_bps": (executed_price - decision_price) / decision_price * 10000,
        })
        
        # Refit eta, gamma every N observations
        if len(self.observations) % 50 == 0:
            self._refit_parameters()
    
    def _refit_parameters(self):
        """Bayesian Optimization to find best eta, gamma given observations."""
        def loss(params):
            eta_test, gamma_test = params
            total_error = 0.0
            for obs in self.observations:
                predicted = self._predict_slippage(obs["qty"], eta_test, gamma_test)
                actual = obs["slippage_bps"]
                total_error += (predicted - actual) ** 2
            return total_error
        
        result = gp_minimize(
            loss,
            [Real(1e-6, 1e-2, prior="log-uniform"),
             Real(1e-7, 1e-3, prior="log-uniform")],
            n_calls=30,
            random_state=42,
        )
        self.eta, self.gamma = result.x
```

**Was du gewinnst:** Statt feste Slippage-Schätzung in `cost_model_calibrator.py` (du hast das!) hast du ein **online-lernendes** Modell.

**Aufwand:** 10-15h. **Lizenz:** Akademisch frei.

---

# Adoption-Plan v3

Sortiert nach **Score = (Nutzen × Anwendbarkeit) / Aufwand**, unter Berücksichtigung dass du viele Sachen schon hast:

### Tier 1: Sofort empfehlenswert (~25-40h, sehr fokussierter Nutzen)

| # | Empfehlung | Aufwand | Section |
|---|---|---|---|
| 1 | **Custom Trading-Metrics in Prometheus** + Grafana-Dashboards | 10-16h | A10 |
| 2 | **Brinson-Hood-Beebower Attribution** in dein `attribution/` | 6-10h | A8 |
| 3 | **MLflow** parallel zu deinem Experiment-Tracker | 8-12h | A9, B4 |

### Tier 2: Hoher Nutzen (~30-50h)

| # | Empfehlung | Aufwand | Section |
|---|---|---|---|
| 4 | **toraniko Barra-Style Risk-Model** (zerlege deine Returns) | 12-20h | A5 |
| 5 | **VPIN** (Toxic-Flow-Detector) | 8-12h | B9 |
| 6 | **Term-Structure-Features** (VIX-Curve, Yield-Curve) | 8-12h | B7 |
| 7 | **Liquidity-Aware Sizer** | 6-10h | B8 |
| 8 | **Meta-Model Leakage-Test** (verifizier dein bestehendes meta_model) | 6-10h | A6 |
| 9 | **DuckDB als Feature-Store-Layer** | 8-12h | B5 |

### Tier 3: Strategische Investments (~50-100h)

| # | Empfehlung | Aufwand | Section |
|---|---|---|---|
| 10 | **ELSTER-Exporter** für deine accounting/ | 12-20h | A1 |
| 11 | **Volatility-Surface (SVI)** Features | 16-30h | B2 |
| 12 | **Causal-Inference-Feature-Importance** | 12-20h | B3 |
| 13 | **Vine-Copulas** für höher-dim Tail-Risk | 8-15h | A3 |
| 14 | **Cross-Asset Carry erweitern** | 12-18h | B6 |
| 15 | **Adaptive Conformal Prediction** in conformal_position.py | 6-10h | A7 |
| 16 | **Adaptive Almgren-Chriss** mit BayesOpt | 10-15h | B10 |

### Tier 4: Spekulativ / Research-Pfad (~40-80h pro Item)

| # | Empfehlung | Aufwand | Section |
|---|---|---|---|
| 17 | **Neo4j-Persistierung** für news_entity_graph | 8-12h | A2 |
| 18 | **Synthetic-Data-Validierung** für Backtests | 40-80h | B1 |
| 19 | **mlfinlab Concurrent-Events-Korrektur** in triple_barrier | 6-8h | A4 |

---

## Schluss

Mein konkreter Rat:

1. **Fang mit Tier 1 #1 an** (Custom-Metrics + Grafana). Das ist der schnellste Weg zu "ich sehe live, was meine Strategien tun". Visibility ist der Multiplikator für alles andere.

2. **Direkt danach Tier 2 #4** (toraniko Barra-Style Risk Model). Das löst eine fundamentale Frage: "Ist das Alpha oder Style-Exposure?" — eine Antwort, die deine Strategie-Auswahl drastisch verändern kann.

3. **Parallel** Tier 1 #3 (MLflow). Sehr niedriger Aufwand, du hast das halbe System sowieso. Reproducibility-Boost.

4. **Tier 2 #8** (Meta-Model-Leakage-Test) ist der **Bug-Hunter** unter den Empfehlungen. Wenn du eine subtile Leakage in deinem `meta_model.py` hast, ist das in Backtest-Returns versteckt und der größte Risiko-Posten in deinem Repo.

5. **Tier 3 #10** (ELSTER-Exporter) ist der **persönliche Hebel** — wenn du jedes Jahr 4-8h in Steuer-Excel investierst, amortisiert sich das schnell.

Lass mich wissen, wo du anfangen willst — ich kann zu jedem Item den konkreten Code, die genaue Repo-Stelle, und Tests schreiben.
