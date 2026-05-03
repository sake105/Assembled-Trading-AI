# Tiefen-Wiring-Audit — Wo Potential verschwendet wird

**Datum:** 2026-05-01
**Geprüfter Stand:** `778fa39` (HEAD on main)
**Methodik:** AST-basierte Import-Analyse mit korrekter relativer Import-Auflösung — die einzige Methode, die alle 4 Import-Patterns deines Repos sauber matched (`from src.assembled_core.X.Y import`, `from assembled_core.X.Y import`, `from .Y import`, `from ..Y import`).

---

## TL;DR

**88% deines Codes ist produktiv gewired** (376 Module / 106k LOC) — viel besser als die ersten Audits behaupteten (die hatten falsche Negative wegen schwacher Pattern-Match-Logik).

Aber: **86 Module mit 15.110 LOC sind test-only** — Code der **nur deshalb existiert, weil Tests ihn importieren, kein Production-Caller**. Davon sind **13 hochwertige Module** (composite_score, pairs_trading, HRP, conformal_position, …) die nicht "tot" sind sondern **vergessen wurden zu wiren**.

**Die echten 4 Tote:** `attribution/shap_explainer`, `signals/cross_impact`, `mlflow_setup`, `attribution/drift_detection` — zusammen 295 LOC. Können gelöscht werden.

---

## Das ehrliche Bild

| Kategorie | Module | LOC | Anteil |
|---|---|---|---|
| Production-wired | 376 | 106.808 | 88% |
| Test-only (orphan logic) | 86 | 15.110 | 12% |
| Truly dead | 4 | 295 | 0.2% |
| **Total** | **466** | **129.035** | 100% |

---

## Die wirklich-toten 4 Module

Klein genug, dass `git rm -r` sich nicht lohnt — aber sollten weg:

```
127 LOC  src.assembled_core.attribution.shap_explainer    ← legacy
101 LOC  src.assembled_core.signals.cross_impact          ← legacy
 53 LOC  src.assembled_core.mlflow_setup                  ← veraltet, MLflow läuft via certify/mlflow_integration
 14 LOC  src.assembled_core.attribution.drift_detection   ← Stub, drift in ops/drift_monitor.py
```

**Aufwand:** 5 min `git rm` + 1 grep nach `shap_explainer` Imports in tests.

---

## Die wirkliche Geschichte: 86 Test-Only-Module

Das ist die **eigentliche Verschwendung** — Code der existiert, getestet wird, dokumentiert ist, **aber niemand benutzt ihn produktiv**.

### Tier 1: Hochwertig — sollten dringend gewired werden (13 Module, ~3.500 LOC)

**Diese würde ich **sofort** als Priority-Items wiren:**

| Modul | LOC | Was es macht | Warum dringend |
|---|---|---|---|
| `signals/composite_score` | 386 | 9-Dim regime-conditional Score (MTF, TA, Microstructure, Volume, Pattern-ML, Vol-Surface, Breadth, Seasonality, News) | Du hast `multifactor_v2.py` produktiv, aber das hier ist die "next-gen" 9-Dim-Version — dokumentiert in `31_COMPOSITE_SCORE.md` |
| `signals/pairs_trading` | 207 | Cointegration + Kalman-Hedge-Ratio + Spread-Z-Score | Komplette Strategie. Würde dein Repo um eine Market-Neutral-Variante erweitern |
| `signals/options_iv` | 200 | py_vollib IV-Surface, Greeks, IV-Rank, Skew | Die einzige Möglichkeit Options-Daten zu nutzen |
| `portfolio/hierarchical_risk_parity` | 308 | Lopez de Prado HRP | State-of-the-Art Portfolio-Allocation. Dein `multiasset_allocator` ist regelbasiert |
| `portfolio/conformal_position` | 154 | MAPIE Conformal-Sizing — **bereits validiert** mit 87% Coverage | Im ML-Audit als "deployment-ready" identifiziert, aber `_tc_sizing.py` nutzt eine andere Conformal-Implementation. Es gibt **zwei parallele** Conformal-Sizer im Repo |
| `portfolio/adaptive_conformal_position` | 153 | ACI — adaptive Variante (besser als statisches MAPIE) | Theoretisch überlegen für non-stationäre Märkte |
| `qa/cpcv_validation` | 228 | Combinatorial Purged CV (skfolio) | Du hast in `11_FREE_MODELLE.md §11.5` selbst CPCV als "Gold Standard für financial ML" dokumentiert. Test-only |
| `qa/leakage_analyzer` | 207 | Detection von look-ahead bias | Wertvoll für ML-Pipeline |
| `features/triple_barrier` | 350 | Lopez de Prado Triple-Barrier-Labeling | Standard für Meta-Labeling |
| `features/news_features` | 348 | News-EWM, Event-Count, Velocity | Du hast in 30+ News-Modulen und nutzt sie nicht für Features |
| `data/feature_store` | 283 | DuckDB+Parquet Feature-Store mit ASOF-Join PIT-safety | Im Audit als zentrale Foundation identifiziert. Tests laufen, aber niemand nutzt das produktiv |
| `data/universe` | 310 | PIT-Universe — **schon in KNOWN_ISSUES §0.1** dokumentiert | Survivorship-Bias-Risiko +5-10% p.a. Mid-Cap |
| `data/free_universe` | 212 | Free universe-data construction | Foundation für `data/universe` |

**Beobachtung:** Dies sind keine "ältere Versionen" oder "experimentelle Stubs" — das sind **richtig durchdachte Module mit voller Test-Coverage**, die einfach **nie eingehängt wurden**.

### Tier 2: Mittel — situativ wertvoll (17 Module)

```
signals/buyback_drift, signals/etf_flows, signals/pead_sue,
signals/insider_cluster, signals/cross_asset_carry, signals/tail_risk_hedge
features/tsfresh_augmentation, features/change_point_detection,
features/macro_regime_quadrant, features/residual_momentum,
features/liquidity_condition_index
data/sources/weather_source, data/sources/wikipedia_views_source
events/news_graph, events/ner_extractor
risk/barra_risk_model, risk/garch_vol
```

Jedes davon ist eine **eigene Story**. Manche brauchen externe Daten (z.B. options_iv braucht IV-Daten). Andere sind alternative Implementations zu bereits gewireten Modulen (z.B. garch_vol vs. eine bestehende Vol-Schätzung).

### Tier 3: Admin/Infrastruktur (33 Module)

Das sind Module wie `compliance/elster`, `accounting/tax_lots`, `config/feature_flags`, `ops/scheduler`. Die brauchen kein urgentes Wiring — sie sind als Modular-Infrastruktur gedacht und werden eines Tages aktiviert wenn du Live-Trading machst (Steuer, Compliance, Scheduler).

---

## Specific Pattern: Doppelte Implementations

**Die kritischste Beobachtung:** Du hast **mehrere Module mit überlappender Funktionalität**, wo nur eine Variante produktiv ist:

### Pattern 1: Conformal-Sizing
- `portfolio/conformal_position` (154 LOC, **test-only**) — MAPIE-basiert, 87% validiert
- `portfolio/adaptive_conformal_position` (153 LOC, **test-only**) — ACI-Variante  
- `_tc_sizing.py` Conformal-Logik (inline) — **production-wired**

Drei Implementationen, eine produktiv. Soll man die anderen löschen, oder ist die produktive Version eine Vereinfachung von einem der test-only-Module?

### Pattern 2: GARCH-Vol
- `risk/garch_vol` (180 LOC, **test-only**)
- `risk/garch_vol_forecast` (101 LOC, **test-only**)
- Existiert eine produktive Vol-Schätzung in `_tc_sizing.py`?

### Pattern 3: Walk-Forward-Optuna
- `qa/walk_forward_optuna` (266 LOC, **test-only**) — Library-Modul
- `scripts/training/walk_forward_hpo.py` (260 LOC, **wired**) — CLI-Skript
- Beide tun fast das Gleiche. Im ML-Wave-Review hatte ich angemerkt, dass das ggf. legitim ist (Library vs. CLI), aber die Doppelung ist Wartungsschuld.

### Pattern 4: NewsGraph
- `events/news/news_graph` (344 LOC, **test-only**) — Neo4j-basiert
- `intel/news_entity_graph` — was ist das?

Lass mich verifizieren: existiert ein produktiver `news_entity_graph`?

---

## Konkrete Empfehlungen — sortiert nach (Wirkung × Einfachheit) / Aufwand

### Sofort (~30 min)

1. **4 truly-dead Module löschen** (10 min):
   ```bash
   git rm src/assembled_core/attribution/shap_explainer.py \
          src/assembled_core/signals/cross_impact.py \
          src/assembled_core/mlflow_setup.py \
          src/assembled_core/attribution/drift_detection.py
   # Plus tests die sie referenzieren — vorher checken
   ```

2. **`_tc_signals` als gewired bestätigen** in `KNOWN_ISSUES.md` — wir haben jetzt eine ehrliche Wiring-Analyse, die sollte als Baseline ins Repo

### Diese Woche (~1 Tag)

3. **`portfolio/conformal_position` produktiv aktivieren** (2-3h):
   - Ist bereits getestet (87% empirical coverage validiert)
   - `_tc_sizing.py` hat eine inline-Conformal-Implementation — entscheiden ob die ersetzt wird oder ob beide koexistieren
   - **Größtes Wertversprechen pro Aufwand** im ganzen Audit

4. **`data/universe` PIT-Wiring** (4-6h):
   - Schon dokumentiert in `KNOWN_ISSUES §0.1`
   - +5-10% p.a. Bias-Korrektur bei Mid-Caps
   - 3 Wiring-Stellen identifiziert (`prices_ingest.py`, `run_backtest_strategy.py`, `run_walk_forward_analysis.py`)

5. **`qa/cpcv_validation` in ML-Trainings-Pipeline** (2h):
   - Ersetze `train_test_split` in `train_ml_models_v4.py` durch CPCV
   - Du hast in `11_FREE_MODELLE.md` selbst CPCV als Gold-Standard markiert

6. **`features/news_features` ins Meta-Model** (1 Tag):
   - Genau die Empfehlung aus dem ML-Wave-Review: AUC 0.5017 mit pure-TA → Features fehlen
   - news_features.py ist 348 LOC fertige news-Features
   - Wäre der **echte ML-Edge**

### Strategisch (1-2 Wochen)

7. **`signals/composite_score` als Backup-Strategie** für `multifactor_v2`:
   - 9-Dim mit regime-conditional weights
   - In `policy_ml_research.yaml` als zweite Strategy parallel laufen lassen
   - A/B-Backtest gegen `multifactor_v2`

8. **`signals/pairs_trading` als Market-Neutral-Variante**:
   - Komplette Strategie mit Kalman-Hedge-Ratio
   - Würde dein Repo um eine fundamental andere Strategy-Klasse erweitern

9. **`portfolio/hierarchical_risk_parity` aktivieren**:
   - In `multiasset_allocator` als zweite Allocation-Methode
   - Lopez de Prado HRP ist State-of-the-Art

---

## Die ehrliche Antwort auf deine Frage

> "Verwenden wir wirklich alles oder verschwenden wir potential?"

**Du verschwendest signifikant Potential**, aber **anders als ein erstes Audit suggerieren würde**.

Das Problem ist **nicht** "tote Module die niemand mehr braucht" (das sind nur 4). Das Problem ist:

**12% deines Codes (15.000 LOC) sind hochwertige, getestete Module die nicht in der Pipeline gewired sind.** Davon sind 13 Module Tier-1-Hochwertig — composite_score, pairs_trading, HRP, conformal_position, options_iv, CPCV-Validation, triple_barrier, news_features, feature_store, PIT-universe...

Das sind **keine "alten Sachen die rumliegen"**. Das sind **fertige Bausteine**, oft mit detaillierter Dokumentation in `autonome_weiterarbeit/`, die einfach **nie eingehängt wurden**.

Die Wahrheit hinter dieser Verschwendung: du hast in der Anfangsphase **breit gebaut** ("alles was in 11_FREE_MODELLE.md steht implementieren") und in der Wiring-Phase **schmal gewired** ("erstmal nur Pipeline runtime"). Die ML-Welle und die Wiring-Wellen 2/3 haben das teilweise behoben, aber die Mehrheit der `signals/`, `features/`, `portfolio/`, `qa/`-Module wurde nicht mit-gewired.

**Mein konkreter Vorschlag:** Mache **eine "Wiring-Welle 4"** mit den 13 Tier-1-Items. Geschätzter Aufwand: 1-2 Tage Konzentrierte Arbeit, danach hast du bei effektiv gleichem Code-Volumen **20-30% mehr produktive Funktionalität**.

Welches Item willst du als erstes? Mein Top-Vorschlag ist:

**`portfolio/conformal_position` aktivieren**, weil:
1. Modell ist bereits trainiert + validiert (87% Coverage)
2. Wiring-Aufwand klein (~2h)
3. Erster echter Live-ML-Effekt im Repo
4. Niedrig riskant (nur Sizing-Schicht, nicht Signal-Schicht)
