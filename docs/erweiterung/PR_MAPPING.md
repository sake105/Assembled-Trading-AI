# PR-Mapping: Erweiterung → assembled_core

**Branch:** ERWEITERUNG → main (später per PR)
**Stand:** 2026-05-11

Dieses Dokument plant die finale Migration von ERWEITERUNG-Modulen ins
Mainline-`src/assembled_core/` für den späteren PR. Jedes Modul wird klassifiziert:

- **PORT**: Modul wird ins Mainline-Repo verschoben/integriert
- **KEEP**: Modul bleibt im Erweiterungs-Branch (Research-only)
- **DEPRECATE**: Modul wird gestrichen (negativ-validated / Duplikat)

---

## 1. Strategien (`src/erweiterung/strategies/`)

| Modul | Status | Mainline-Ziel | Begründung |
|-------|--------|---------------|------------|
| `master_allocator.py` | **PORT** | `assembled_core/portfolio/master_allocator.py` | p=0.997 vs 60/40 validiert, Production-API |
| `volatility_targeting.py` | **PORT** | `assembled_core/risk/vol_target_overlay.py` | Self-Hedge-Edge bestätigt, Master nutzt es |
| `cross_section_helpers.py` | **PORT** | `assembled_core/portfolio/cs_helpers.py` | Vektorisierte CS-Ops, 50× schneller als pandas-groupby |
| `ema_trend_cross_section.py` | **PORT** | `assembled_core/strategies/ema_trend_v1.py` | Komplementär zu existierendem `ema_trend_v0.py` |
| `intermarket_macro_factors.py` | **PORT** | `assembled_core/features/intermarket_macro_factors.py` | Komplementär zu `intermarket_factors.py` (bessere Signature) |
| `profit_lock_overlay.py` | **PORT** | bereits in `assembled_core/risk/profit_lock.py` | Mainline hat Original — Erweiterung als Diff/Improvement nur falls API-Verbesserung |
| `macro_regime_quadrant.py` | **PORT** | bereits in `assembled_core/features/macro_regime_quadrant.py` | Mainline hat Original — nur API-Wrapper-PR sinnvoll |
| `multi_signal_regime.py` | **KEEP** | (Research-Modul) | Walk-Forward-widerlegt, nicht für Production |
| `macro_stress_signals.py` | **KEEP** | (Research-Modul) | VIX-Z + YC-Stress sind im multifactor_v2 schon |
| `ensemble_regime.py` | **KEEP** | (Research-Modul) | Voting-Demo, p=0.99 wenig sig. Edge |
| `regime_conditional_allocator.py` | **DEPRECATE** | — | Mainline hat eigenes `portfolio/regime_conditional_allocator.py` |
| `tail_risk_hedge.py` | **KEEP** mit Caveat-Doku | — | Negativ-validated (p=0.02 schlechter mit Master) — bewahren als Negativ-Beispiel |
| `multi_factor_vol_target.py` | **KEEP** | — | Overfit-validiert, Single-Factor-VolTarget besser |
| `news_sentiment_strategy.py` | **PORT** | `assembled_core/signals/news_sentiment_cross_section.py` | Komplementär zu `news_features.py` (CS-Ranking statt feature-only) |
| `fomc_macro_signal.py` | **KEEP** | — | Bug-fix dokumentiert, aber kein echter Edge auf Demo-Statements. Bei echtem FOMC-Archive portierbar. |

---

## 2. Backtest (`src/erweiterung/backtest/`)

| Modul | Status | Mainline-Ziel |
|-------|--------|---------------|
| `calmar_bootstrap.py` | **PORT** | `assembled_core/backtest/calmar_bootstrap.py` |
| `performance_metrics.py` | **DEPRECATE** | bereits Mainline `qa/performance_metrics.py` |
| `white_reality_check.py` | **KEEP** | Mainline hat eigene Variante |
| `deflated_sharpe.py` | **KEEP** | Mainline hat eigene Variante |
| `cpcv.py` | **KEEP** | Mainline hat `qa/cpcv.py` |

**Begründung:** Calmar-Bootstrap ist methodische Innovation der Erweiterung
(stationary bootstrap, Politis-Romano) und fehlt im Mainline — klarer
PR-Kandidat.

---

## 3. Live + Performance (`src/erweiterung/live/`)

| Modul | Status | Mainline-Ziel |
|-------|--------|---------------|
| `live_decision_engine.py` | **PORT** | `assembled_core/execution/live_decision_engine.py` |
| `order_router.py` | **PORT** | `assembled_core/execution/order_router_lite.py` |
| `data_cache.py` | **PORT** | `assembled_core/data/cache/data_cache.py` |

**Begründung:** Vollständig neue Live-Trading-Layer mit 1.65ms-Latenz.
Mainline hat aktuell keine vergleichbare Inkrementelle Decision-Engine.

---

## 4. QA (`src/erweiterung/qa/`)

| Modul | Status | Mainline-Ziel |
|-------|--------|---------------|
| `equity_curve_audit.py` | **PORT** | `assembled_core/qa/equity_curve_audit.py` |

**Begründung:** Fand 3 identische Files + Sharpe-4.6-Anomalien in Mainline-
Equity-Curves. Sollte als CI-Smoke-Test integriert werden.

---

## 5. Altdata (`src/erweiterung/altdata/`)

| Modul | Status | Mainline-Ziel |
|-------|--------|---------------|
| `caldara_iacoviello_gpr.py` | **PORT** | `assembled_core/data/altdata/caldara_iacoviello_gpr.py` |
| `yfinance_cache_loader.py` | **PORT** | `assembled_core/data/altdata/yfinance_cache_loader.py` |
| `gdelt_extended.py` | **KEEP** | Mainline hat `events/news/fetch_gdelt.py` |
| `worldbank_macro.py` | **KEEP** | Komplementär |
| `coingecko_crypto.py` | **KEEP** | Komplementär |
| `yahoo_options.py` | **KEEP** | Komplementär |

**GPR-Loader füllt eine konkret dokumentierte Mainline-Lücke** (Doc-Comment in
`features/geopolitical_features.py` erwähnt direkte FRED-Daten, aber Loader
fehlt).

---

## 6. Risk (`src/erweiterung/risk/`)

| Modul | Status | Mainline-Ziel |
|-------|--------|---------------|
| `gpr_overlay.py` | **PORT (mit Caveat-Doku)** | `assembled_core/risk/gpr_overlay.py` |

**Caveat:** Trading-Edge nicht signifikant (p=0.21 auf Master, p=0.11 auf Pure-Mom).
Sollte als Sub-Komponente in `multifactor_v2`'s `geo_risk_composite` mit kleinem
Gewicht (5 %) integriert werden, NICHT als Standalone-Master-Overlay.

---

## 7. Robustness (`src/erweiterung/robustness/`)

| Modul | Status | Mainline-Ziel |
|-------|--------|---------------|
| `walk_forward.py` | **PORT** | `assembled_core/qa/walk_forward.py` |
| `sub_period.py` | **PORT** | `assembled_core/qa/sub_period_analysis.py` |

---

## 8. Andere

Folgende Subpackages bleiben in Erweiterung (Research-only):

- `news_impact/` — Skeleton-Module für News-Surprise/Reactivity/Spillover.
  Bei verfügbarem Multi-Jahr-News-Feed (GDELT-Backfill) später portierbar.
- `transcripts/` — FOMC/Earnings-Call-Tone-Module. Bei echtem Transcript-
  Archive portierbar.
- `live_pipeline/` — EDGAR-Filing-Classifier. Module ready, braucht
  SEC-EDGAR-API-Polling.
- `factors/` — Fama-French + IC-Diagnostik. Mainline hat eigene Faktor-Welt.
- `meta/`, `ml/`, `dl/`, `rl/`, `discovery/`, `bayesian/`, `state_space/`,
  `survival/`, `microstructure/`, `volatility/`, `intraday/`, `orderbook/`,
  `dl_advanced/`, `crossasset/`, `info_theory/`, `causal_inference/`,
  `economic_data/`, `signals/`, `online_learning/`, `cost_analytics/`,
  `options_pricing/`, `attribution/`, `report/`, `risk_metrics/`,
  `stress_test/`, `graph_methods/`, `factors/` (Cross-Section-Variante),
  `classical_ml/`, `timeseries_tools/` — **alle Research-only**, keine
  klare PR-Kandidaten ohne Datenfundierung.

---

## 9. PR-Reihenfolge-Vorschlag (Modular)

Statt einem Mega-PR sollten mehrere kleinere PRs erstellt werden:

**PR 1: Calmar-Bootstrap + Equity-Curve-Audit (methodische Infrastruktur)**
- `qa/calmar_bootstrap.py`
- `qa/equity_curve_audit.py`
- `qa/walk_forward.py`
- `qa/sub_period_analysis.py`
- Tests dazu
- **Effekt:** Bessere Validation-Methodik im Mainline, fängt zukünftige
  Sharpe-4.6-Anomalien ab.

**PR 2: Caldara-Iacoviello GPR-Loader**
- `data/altdata/caldara_iacoviello_gpr.py`
- `data/altdata/yfinance_cache_loader.py`
- Erweitert `features/geopolitical_features.py` um echte GPR-Daten
- Tests dazu
- **Effekt:** Erfüllt Mainline-Doc-Versprechen "If GPR available, use directly".

**PR 3: Cross-Section-Helpers + Vol-Target-Overlay**
- `portfolio/cs_helpers.py`
- `risk/vol_target_overlay.py`
- Tests dazu
- **Effekt:** 50× schnellere Cross-Section-Ops + saubere Vol-Targeting-API.

**PR 4: Master-Allocator + Live-Decision-Engine**
- `portfolio/master_allocator.py`
- `execution/live_decision_engine.py`
- `execution/order_router_lite.py`
- `data/cache/data_cache.py`
- Tests dazu
- **Effekt:** Vollständig neue Live-Trading-Layer mit 19y-validierter
  Statistik (p=0.997 vs 60/40).

**PR 5: Optional GPR-Overlay**
- `risk/gpr_overlay.py` mit Caveat-Doc dass standalone p=0.21 (nicht sig)
- Empfehlung: nur als Sub-Komponente in `multifactor_v2` integrieren.

---

## 10. Konflikte/Probleme bei PR

| Konflikt | Lösung |
|----------|--------|
| Mainline-`profit_lock.py` existiert schon | Erweiterung-Variante NICHT portieren (Mainline besser) |
| Mainline-`macro_regime_quadrant.py` existiert schon | Erweiterung-Variante NICHT portieren |
| `regime_conditional_allocator.py` exists in Mainline | Erweiterung-Variante DEPRECATE |
| `performance_metrics` exists in Mainline qa | Erweiterung-Variante DEPRECATE |
| `cpcv.py` exists in Mainline | Erweiterung-Variante KEEP (separate Implementation) |

---

## 11. Zusammenfassung

- **17 Module + 7 Tests** PR-Kandidaten in 5 modularen PRs
- **23 Subpackages** bleiben Research-only in Erweiterung
- **6 Module** explizit DEPRECATE (Duplikate ohne Mainline-Vorteil)
- **Keine zerstörerischen Mainline-Änderungen** — alle PRs sind additiv

Branch ERWEITERUNG bleibt als Research-Lab erhalten. PRs werden modular
gegen `main` erstellt.
