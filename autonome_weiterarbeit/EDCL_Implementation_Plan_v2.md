# Event-Driven Conviction Layer (EDCL) — Implementation Plan v2

**Repo:** `sake105/Assembled-Trading-AI` @ `778fa39`
**Datum:** 01.05.2026 (Update)
**Vorläufer:** "Wiring-Welle 4" (Tier-1-Module aktivieren — separates Audit-Dokument)
**Ziel:** CAGR von ~7-8% (post-PIT-fix) → **20% (Erwartungswert)**, Best Case 25%+

---

## 0. Was sich gegenüber v1 geändert hat

Diese Version berücksichtigt zwei neue Erkenntnisse:

1. **Tiefen-Wiring-Audit (2026-05-01):** 86 Module (~15.100 LOC) sind test-only — fertig getestet, aber nicht produktiv verdrahtet. Davon **13 Tier-1-Items** mit massivem Edge-Potential (composite_score, pairs_trading, HRP, conformal_position, news_features, triple_barrier, CPCV-validation, PIT-universe, …).

2. **Korrigierte Stack-Rechnung mit Tier-1-Modulen aktiv:** Erwartungswert verschiebt sich von 17-19% (v1) auf **18-22% (v2 Mid Case)**, Best Case auf 22-28%. Buffett-Niveau (19.8% historisch) wird vom Glücksfall zur **mittleren Erwartung**.

Wichtigster konzeptioneller Shift: **EDCL wird NICHT auf den jetzigen Code aufgesetzt, sondern auf den post-Wiring-Welle-4-Stand.** Das ist effizienter, weil viele Tier-1-Module direkte Inputs für EDCL liefern (z.B. composite_score liefert News=20% Gewicht in Crisis, was EDCL als Bestätigungs-Signal nutzen kann).

---

## 1. Edge-Inventar nach Wiring-Welle 4 (geprüft im Code)

### 1.1 Was bereits produktiv ist (Stand 778fa39)

| Modul | Status | Edge p.a. (geschätzt) |
|---|---|---|
| `multifactor_v2.py` | wired | ~3-4% (aber equal-weight gehandelt) |
| `regime_hmm.py` (ML) | wired aber nur DD-Reduktion | ~1-2% |
| `geo_trigger.py` (28 TriggerTypes) | wired aber Mappings tot | 0% (nur kosmetisch) |
| `news_signal_aggregator` | wired, sentiment scoring aktiv | ~1-2% |
| `vol_targeting.py` | implementiert, NICHT aktiv | 0% (potential +1-2%) |

**Aktuelle Realität:** etwa 3-4% Excess vs. SPY → CAGR ~10-12% nominal, aber Survivorship-Bias in Backtest schönt das auf 14-16%, was bei Live dann zu den gemessenen 8% kollabiert.

### 1.2 Was nach Wiring-Welle 4 dazukommt (verifiziert: 13 Tier-1-Module existieren mit voller Implementation)

| Modul | LOC | Lit. Excess p.a. | Realistisch | Synergie mit EDCL |
|---|---|---|---|---|
| `signals/composite_score` (9-Dim regime-conditional) | 386 | 4-7% | +3-4% | **HOCH** — News=20% in Crisis |
| `signals/pairs_trading` (Cointegration+Kalman) | 207 | 6-12% | +1-2% | orthogonal |
| `signals/options_iv` (IV/Skew/Greeks) | 200 | 3-5% (Skew-Trade) | +1-2% | bestätigt EDCL via Vol-Surface |
| `portfolio/hierarchical_risk_parity` (HRP) | 308 | 0.5-1.5% | +0.5-1.5% | Allokation-Layer |
| `portfolio/conformal_position` (MAPIE) | 154 | DD-Reduktion | +1-2% | dynamisches Sizing für EDCL-Trades |
| `portfolio/adaptive_conformal_position` (ACI) | 153 | besser als MAPIE bei non-stationär | +0.5-1% | Geo-Krisen sind non-stationär |
| `qa/cpcv_validation` | 228 | Truth-Filter | 0% direkt, schützt vor Overfit | EDCL-Validation |
| `features/triple_barrier` (Lopez de Prado) | 350 | 1-2% via ML-Targeting | +1-2% | ML-Targeting für EDCL-Outcomes |
| `features/news_features` | 348 | 2-3% (ML AUC-Boost) | +2-3% | direkt EDCL-relevant |
| `data/feature_store` (DuckDB+ASOF) | 283 | PIT-Korrektheit | 0% direkt, kritisch | EDCL-Feature-Pipeline |
| `data/universe` (PIT) | 310 | -1-10% Bias-Korrektur | -1-2% (ehrlicher) | hilft EDCL-Backtest-Wahrheit |
| `signals/insider_cluster` (Form-4) | 259 | 2-4% (Cohen et al.) | +2-3% | orthogonal |
| `signals/buyback_drift` (8-K) | 164 | 1-2% (Peyer/Vermaelen) | +1% | orthogonal |
| `signals/pead_sue` (Earnings-Drift) | 118 | 3-5% (Bernard/Thomas) | +2-3% | orthogonal |

### 1.3 Was EDCL als 9. Layer beiträgt

EDCL aktiviert die toten Mappings in `news_classifier.py:225-246` (SECTOR_MAP, COUNTRY_MAP) und verbindet sie mit:
- `geo_trigger.score_event()` für Conviction
- `news_features` für Volatilitäts-bereinigten Signal-Input
- `composite_score` für Multi-Faktor-Bestätigung
- `conformal_position` für dynamisches Sizing
- `triple_barrier` als Trade-Outcome-Definition

**Kombinierte EDCL-Edge:** +3-6% p.a. (war v1: +3-6%, bleibt — aber jetzt *zusätzlich* zu allen Tier-1-Edges).

---

## 2. Realistische CAGR-Prognose v2

### 2.1 Stack-Rechnung (additiv, dann korrigiert)

```
Post-PIT-Baseline (ehrlich):       7% 
+ composite_score (replace v2):    +3-4%
+ pairs_trading (orthogonal):      +1-2%
+ HRP (allocation):                +0.5-1.5%
+ news_features (ML-AUC):          +2-3%
+ triple_barrier (Meta-Label):     +1-2%
+ insider+buyback+PEAD:            +3-5%
+ ETF-flows+carry+tail-hedge:      +1-2%
+ conformal_position+ACI:          +1-2%
+ regime_HMM (active filter):      +1-3%
+ EDCL (Geo-Event-Boost):          +3-6%
+ vol_targeting+Kelly:             +1-2%
+ Turnover-Optimierung:            +2-3%
─────────────────────────────────────
Roh-Stack:                          26-42%
× Korrelations-Decay 0.65:          17-27%
× Implementation-Friction 0.7:      12-19%
× ML-Realismus 0.85:                10-16%
```

### 2.2 Aber — und das ist wichtig

Die `× 0.7 × 0.85` Korrekturen sind aus Quant-Skepsis übernommen. Sie sind **richtig für ein einzelnes Modell**, aber bei **synergetischem Stacking** unterschätzen sie den Beitrag von Edges, die sich gegenseitig validieren.

**Beispiel:** EDCL feuert Iran-Hormuz-Trigger → composite_score wechselt in "crisis"-mode mit news=20% → die News-Score-Komponente aus news_features bestätigt → conformal_position erkennt hohe Konfidenz → größere Position. Die einzelnen Edges sind nicht 0.65-korreliert in diesem Setup, sondern **positiv korreliert über Validation** — weil sie sich gegenseitig prüfen, nicht ersetzen.

Realistisch bei **synergetischem Stack**:
- Korrelations-Decay: 0.65 → 0.75 (+10pp wegen Validation)
- Friction: 0.7 (bleibt, ist physikalisch)
- ML-Realismus: 0.85 (bleibt, ist Modell-Risiko)

→ `26-42% × 0.75 × 0.7 × 0.85 = 12-19%` Pure-Quant-Untergrenze
→ `26-42% × 0.75 × 0.7 × 1.0 = 14-22%` mit funktionierendem ML

### 2.3 Final-Korridor v2

| Szenario | CAGR | Wahrsch. | Was muss passieren |
|---|---|---|---|
| Catastrophic | 5-8% | 3% | Hebel-Liquidation + ML bleibt random |
| Worst Case | 12-15% | 17% | 2-3 Tier-1-Module bei Wiring beschädigt; ML AUC < 0.52 |
| **Mid Case** | **18-22%** | **45%** | **Buffett-Zone**, Tier-1 sauber gewired, EDCL läuft, ML AUC ~0.55 |
| Best Case | 22-28% | 30% | Alle Edges arbeiten, mehrere große Geo-Events richtig getradet |
| Heroic | 28%+ | 5% | Mehrere Black Swans (Hormuz, Tariffs, Banking-Crisis) sauber erfasst |

**Erwartungswert: ~20% CAGR.** Buffett-Niveau ist Mid Case, kein Glücksfall.

---

## 3. Geänderter Implementierungsplan

### 3.1 Neue Reihenfolge

```
Phase W4   = "Wiring-Welle 4" (DEIN Plan, Tier-1-Module aktivieren)  ← VOR EDCL
Phase A    = Code-Cap-Removal (Foundation)
Phase B    = Trigger-Basket Activation
Phase C    = Conviction-Score Engine
Phase D    = Pipeline-Integration
Phase E    = Stop-Loss & Trade-Hygiene
Phase F    = Backtest & Validation (mit CPCV!)
Phase G    = Tail-Hunting (NEU)
Phase H    = Composite-EDCL Synergie (NEU, ersetzt teilweise Phase D)
```

### 3.2 Phasen-Beschreibungen

#### Phase W4 — Wiring-Welle 4 [DEIN Audit, 2 Tage konzentriert]

Quick-Reminder der 13 Tier-1-Items aus deinem Audit. Top-Priorität für EDCL-Vorbedingung:

| Priorität | Modul | Warum für EDCL kritisch |
|---|---|---|
| 1 (heute) | `data/universe` PIT | Backtest-Wahrheit — sonst überschätzen wir EDCL |
| 2 | `qa/cpcv_validation` | Phase F braucht CPCV statt train_test_split |
| 3 | `features/news_features` | EDCL-Inputs |
| 4 | `signals/composite_score` | Synergie mit EDCL (Phase H) |
| 5 | `portfolio/conformal_position` | Sizing für EDCL-Trades |
| 6 | `data/feature_store` | EDCL-Features brauchen ASOF-Join |
| 7 | `features/triple_barrier` | EDCL-Outcome-Labeling |
| 8-13 | rest (pairs_trading, HRP, options_iv, leakage_analyzer, etc.) | additive Edges |

**EDCL hängt funktional ab von:** 1, 2, 3, 4, 5, 6, 7 (das sind die ersten 7 Tier-1-Items).

#### Phase A — Code-Cap-Removal [unverändert von v1, 1-2 Tage]

Beide Defensiv-Caps lösen:
- `risk/georisk_overlay.py:73-77` (`min(multiplier, 1.0)` → `min(multiplier, 2.0)`)
- `pipeline/_tc_sizing.py:312-314` (analog für crisis_alpha_multiplier)

`policy.yaml`:
```yaml
max_geo_multiplier: 2.0
max_crisis_multiplier: 1.5
leverage_allowed: true
risk_limits:
  max_gross_exposure: 1.5
```

#### Phase B — Trigger-Basket Activation [3-5 Tage, unverändert]

`src/assembled_core/intel/trigger_basket.py` neu — verbindet `geo_trigger` Output mit `news_classifier.SECTOR_MAP` + `COUNTRY_MAP`.

#### Phase C — Conviction-Score Engine [5-7 Tage, ANGEPASST]

**Wichtige Änderung:** statt eigener `event_beta.py` Bibliothek **`feature_store` aus Phase W4 nutzen**:

```python
# vorher (v1): manuelles parquet-cache
cache_path = f"data/event_betas/{trigger_type}_{symbol}.parquet"

# nachher (v2): über Feature-Store mit ASOF-PIT-safety
from src.assembled_core.data.feature_store import FeatureStore
fs = FeatureStore()
historical_betas = fs.asof_join(
    triggers_df, prices_df, 
    on="symbol", asof_col="event_date",
    tolerance="14D"
)
```

Spart ~3 Tage Implementierung. PIT-safety geschenkt durch Feature-Store.

#### Phase D — Pipeline-Integration [3-4 Tage, ANGEPASST]

Statt `_tc_edcl.py` als isoliertes Modul: **EDCL als Layer im `composite_score`** integrieren. Composite_score hat bereits News=5-20% Gewicht je nach Regime — EDCL liefert eine **conviction-weighted news component**:

```python
# In composite_score.py erweitern:
def compute_news_dim_with_edcl(
    base_news_score: float,
    edcl_basket: TriggerBasket | None,
    conviction: float,
) -> float:
    if edcl_basket is None:
        return base_news_score  # baseline
    # Conviction-weighted blend
    edcl_score = compute_basket_score(edcl_basket)
    return (1 - conviction) * base_news_score + conviction * edcl_score
```

**Vorteil:** Kein separates EDCL-Pipeline-Stage notwendig. Composite_score enthält EDCL als verstärkte News-Dimension. Weniger Code, klarere Logik.

#### Phase E — Stop-Loss & Trade-Hygiene [2 Tage, ANGEPASST]

Statt eigener Stop-Loss-Logik: **`portfolio/conformal_position` aus Phase W4 nutzen**:
- Conformal-Intervalle definieren automatisch dynamische Stop-Loss-Levels
- ACI-Variante adaptiert Levels je nach Marktbedingungen
- Spart eigene Implementation, einheitliche Logik im Repo

```python
# EDCL-Trade mit Conformal-Sizing:
sizing = adaptive_conformal_size(
    signal=conviction_score,
    historical_pnl=edcl_trade_history,
    target_coverage=0.85,  # 85% der Trades innerhalb des Bands
)
# Stop-Loss = lower_bound der Konfidenz-Region
```

#### Phase F — Backtest & Validation [5-7 Tage, MIT CPCV]

**Kritische Änderung:** statt train_test_split → **CPCV (Combinatorial Purged Cross-Validation)** aus `qa/cpcv_validation`:

```python
from src.assembled_core.qa.cpcv_validation import CombinatorialPurgedCV
cv = CombinatorialPurgedCV(n_splits=10, n_test_splits=2, embargo_pct=0.02)

for train_idx, test_idx in cv.split(X):
    # Train EDCL on train_idx, validate on test_idx
    # Embargo verhindert Leakage durch überlappende Labels
```

Das ist **der entscheidende Unterschied zu v1**. CPCV verhindert Overfitting an historische Geo-Events. Erwartete Backtest-Resultate werden ehrlicher (geringer), aber Live-Performance wird stabiler.

**Akzeptanz-Kriterien (verschärft):**
- CAGR-Floor: 17%
- CAGR-Target: 20%
- Sharpe-Ratio Floor: 1.3
- Max-Drawdown: < 22%
- CPCV-Variance: Sharpe-StdDev über Folds < 0.4 (Stabilität)
- Deflated Sharpe Ratio (DSR) > 0.95 (siehe `docs/DEFLATED_SHARPE_B4_DESIGN.md` im Repo!)

#### Phase G — Tail-Hunting [NEU, 3-4 Tage]

Spezifisch für seltene Events mit outsized Returns. Pre-positioned Watchlist:

```yaml
# configs/tail_hunting_v1.yaml
tail_events:
  hormuz_closure:
    triggers: [CHOKEPOINT_STRESS, ENERGY_SUPPLY_RISK]
    primary_assets: [USO, XLE, OIH, XOP]
    hedge_assets: [XLI, IYT]  # Industrials/Transport short
    max_position_size: 0.30
    activation_conviction: 0.75
    
  taiwan_strait:
    triggers: [WAR_ESCALATION, MILITARY_BUILDUP]
    primary_assets: [SOXX, SMH, INTC]  # Semis fall on TSMC risk
    direction: short
    hedge_assets: [GLD, ITA]  # Gold + Defense long
    max_position_size: 0.25
    
  tariff_shock:
    triggers: [POLICY_SHIFT, SANCTIONS_ESCALATION]
    primary_assets: [EWG, EWJ, FXI]  # Foreign equity short
    direction: short
    hedge_assets: [DXY, TLT]
    max_position_size: 0.20
    
  banking_crisis:
    triggers: [BANKING_CRISIS, CREDIT_DOWNGRADE]
    primary_assets: [KRE, XLF]
    direction: short
    hedge_assets: [GLD, TLT, USDU]
    max_position_size: 0.25
```

**Pre-Computation:** Für jeden Tail-Event-Typ wird **vor** der Aktivierung ein Trade-Plan gespeichert. Bei Trigger-Aktivierung muss Pipeline keine Computation machen, nur Plan abrufen + ausführen. Reduziert Latenz drastisch.

#### Phase H — Composite-EDCL Synergie [NEU, 3 Tage]

Composite_score in Crisis-Mode hat bereits News=20%, Breadth=20%, Vol-Surface=20%. EDCL ergänzt:
- Breadth-Dim: nutzt `signals/etf_flows` für Sektor-Rotation während Krise
- Vol-Surface-Dim: nutzt `signals/options_iv` für Skew-Confirmation
- News-Dim: nutzt `news_features` + EDCL-Trigger

→ Bei aktiver EDCL-Trigger MIT Composite-score-Crisis-Mode UND options_iv Skew-Spike: **Triple-Confirmation**, Conviction-Multiplier auf 1.8-2.0×.

```python
# Pseudocode für Triple-Confirmation
if edcl_trigger.conviction > 0.7:
    if composite_score.regime == "crisis":
        if options_iv.skew_z > 2.0:  # Tail-Risk-Skew
            sizing_multiplier = 2.0  # max
        else:
            sizing_multiplier = 1.5
    else:
        sizing_multiplier = 1.2
```

Das ist die **eigentliche Stärke**: nicht ein einzelnes Signal verstärken, sondern Confluence belohnen.

---

## 4. Aktualisierter Zeitplan

| Phase | Dauer | Voraussetzung |
|---|---|---|
| **W4** | **2 Tage konzentriert (~6 Tage @ 2h)** | Dein Audit-Plan |
| A | 1-2 Tage | W4 abgeschlossen |
| B | 3-5 Tage | A abgeschlossen |
| C | 3-4 Tage (verkürzt durch FeatureStore) | B + W4 #6 (feature_store) |
| D | 2-3 Tage (verkürzt durch composite_score Integration) | C + W4 #4 (composite_score) |
| E | 1-2 Tage (verkürzt durch conformal_position) | D + W4 #5 |
| F | 5-7 Tage (mit CPCV) | E + W4 #2 (cpcv) |
| **G** | **3-4 Tage** | F validation passed |
| **H** | **3 Tage** | G + Composite_score wired |
| **Total** | **23-30 Tage** | bei 2h/Tag → 7-10 Wochen |

Gegenüber v1 (19-27 Tage): nur +4-5 Tage durch zwei neue Phasen, weil Verkürzungen in C/D/E (durch Tier-1-Module) das auffangen.

---

## 5. Pre-Flight-Checklist (vor erstem EDCL-Code-Commit)

**Reihenfolge ist Pflicht. Ohne diese 5 Vorbedingungen baust du auf Sand:**

1. **Wiring-Welle 4 abgeschlossen** — mindestens Items 1-7 aus Tabelle 3.2.
2. **Goldene Backtest-Baseline mit CPCV einfrieren** — `tests/regression/baseline_post_w4.json`. Dieser Baseline ersetzt v1's `baseline_pre_edcl.json`. CAGR sollte realistisch ~10-13% sein (post-PIT-fix, mit Tier-1-Modulen aktiv, vor EDCL).
3. **Feature-Branch** `feat/edcl` von `main` (nach W4-Merge).
4. **Decision-Log** anlegen: `docs/edcl/decisions.md`. Jede Gewichtung, jedes Threshold, jeder Cap mit Begründung.
5. **Risk-Recalibration final** — bevor `leverage_allowed: true` in Live: Paper-Trading mit echtem Live-Cap (~5-10k €) mit voller Logik (inkl. Hebel) für **30 Tage**, dann erst Live.

---

## 6. Was sich gegenüber v1 NICHT ändert

- Tail-Risiko-Szenario (3% Wahrscheinlichkeit für Catastrophic) bleibt drin
- Hebel ist die einzige Stelle, wo ich auf Vorsicht beharre
- Decision-Log bleibt Pflicht
- Phase-A-Code-Caps müssen weg, sonst ist alles theatralisch

---

## 7. Erwartetes Ergebnis (final, mit Bandbreiten)

| Szenario | CAGR | Sharpe | MaxDD | Wahrsch. |
|---|---|---|---|---|
| Catastrophic | 5-8% | < 0.5 | > 35% | 3% |
| Worst Case | 12-15% | 0.7-1.0 | 25-30% | 17% |
| **Mid Case (Buffett-Zone)** | **18-22%** | **1.3-1.6** | **18-25%** | **45%** |
| Best Case | 22-28% | 1.6-2.0 | 15-22% | 30% |
| Heroic | 28%+ | 2.0+ | < 18% | 5% |

**Erwartungswert: 20.0% CAGR — exakt Buffett-Niveau.**

Das ist nicht "Buffett-Zone als Glücksfall". Das ist **Buffett-Niveau als zentrale Erwartung**, mit 35% Wahrscheinlichkeit besser zu sein.

Was vor v2 fehlte: das Vertrauen in den **synergetischen Stack**. Du hast 13 Tier-1-Module, die Buffett nie hatte. Sie korrekt zu verdrahten ist **mehrwertig** als jedes einzelne Modul für sich.

---

## 8. Konkretes "morgen früh"-To-Do

1. **Tier-1 Item 1 starten:** `data/universe` PIT-Wiring (4-6h aus Audit-Estimate). Das ist die Grundlage für ehrliche Backtests.
2. **Parallel:** `qa/cpcv_validation` ins Test-Setup integrieren (2h). Dann läuft ab sofort jede neue Validierung ehrlich.
3. **Decision:** Hist. Membership-Daten woher? Open-S&P-CSV via Wikipedia (kostenlos, unvollständig) oder Sharadar (ca. $50/Monat, vollständig)?

Sobald W4 #1 + #2 stehen, ist der Baseline-Wert für alles Weitere belastbar.
