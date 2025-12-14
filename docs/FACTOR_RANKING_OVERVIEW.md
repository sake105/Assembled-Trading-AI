# Factor Ranking Overview

## 1. Kurzüberblick

**Ziel:** Dieser Dokument gibt einen Überblick über die aktuell wichtigsten Faktoren basierend auf systematischer Evaluation über verschiedene Universes und Factor-Sets.

**Datenbasis:**
- Alt-Daten-Snapshot: Lokale Parquet-Dateien (via `ASSEMBLED_LOCAL_DATA_ROOT`)
- Factor-Analysis-Outputs: Generiert durch `analyze_factors` CLI-Kommando
- **Wichtig:** Alle Analysen basieren ausschließlich auf lokalen Daten. Keine Live-API-Calls werden verwendet.

**Erfasste Factor-Kategorien:**
- **Core TA/Price Factors (Phase A1):** Multi-Horizon Returns, Momentum, Trend Strength, Short-Term Reversal
- **Volatility & Liquidity Factors (Phase A2):** Realized Volatility, Vol-of-Vol, Turnover, Liquidity Proxies
- **Alt-Data Earnings & Insider (Phase B1):** Earnings Surprise, Post-Earnings Drift, Insider Activity
- **Alt-Data News & Macro (Phase B2):** News Sentiment, Macro Regime Indicators

**Ranking-Methode:**
- Kombinierter Score aus IC-IR (Information Ratio) und Deflated Sharpe Ratio
- Konsolidierung über mehrere Universes und Zeiträume
- Details siehe [Factor Ranking Workflows](#methodik)

---

## 2. Methodik

### 2.1 IC/IR-Betrachtung (Phase C1)

Die Information Coefficient (IC) misst die Korrelation zwischen Faktorwerten und Forward-Returns auf Querschnittsbasis (pro Timestamp über alle Symbole).

**Metriken:**
- **Mean IC:** Durchschnittliche IC über den Evaluationszeitraum
- **IC-IR (Information Ratio):** `mean_ic / std_ic` – Maß für die Konsistenz der Faktor-Performance
- **Hit Ratio:** Prozentualer Anteil der Perioden mit positivem IC

**Interpretation:**
- IC-IR > 0.5: Starker, konsistenter Faktor
- IC-IR 0.1–0.5: Moderater, aber stabiler Faktor
- IC-IR < 0.1: Schwacher oder instabiler Faktor

### 2.2 Portfolio-Returns & Deflated Sharpe (Phase C2)

Für jeden Faktor wird ein Long/Short-Portfolio konstruiert:
- **Long:** Top Quantile (höchste Faktorwerte)
- **Short:** Bottom Quantile (niedrigste Faktorwerte)

**Metriken:**
- **Sharpe Ratio:** Risiko-adjustierte Rendite des L/S-Portfolios
- **Deflated Sharpe Ratio (DSR):** Sharpe Ratio korrigiert um Multiple-Testing-Bias
- **Annualized Return:** Annualisierte Rendite des L/S-Portfolios
- **Max Drawdown:** Maximaler Peak-to-Trough-Verlust

**Interpretation:**
- DSR > 1.0: Robustes Signal trotz Multiple Testing
- DSR 0.5–1.0: Moderates Signal
- DSR < 0.5: Möglicherweise zufällig (überprüfen)

### 2.3 Combined Score

Der `combined_score` kombiniert IC-IR und DSR zu einer einzigen Ranking-Metrik:

```
combined_score = 0.6 * normalized(ic_ir) + 0.4 * normalized(dsr)
```

Beide Metriken werden auf einen 0–1 Bereich normalisiert, bevor sie kombiniert werden.

### 2.4 Einschränkungen

**Universes:**
- Die Analysen wurden über folgende Universes durchgeführt:
  - `macro_world_etfs`: Breite Markt-ETFs (Major Indices, Sektoren, Länder)
  - `universe_ai_tech`: AI/Tech-Unternehmen (US-listed)
  - (Weitere Universes können hinzugefügt werden)

**Zeiträume:**
- Evaluationszeitraum variiert je Universe und Factor-Set
- Typisch: 2010–2025 für Core-Faktoren, 2015–2025 für Alt-Data-Faktoren (wegen Verfügbarkeit)

**Data Quality:**
- Alle Analysen basieren auf lokalen Alt-Daten-Snapshots
- Fehlende Symbole oder Datenlücken können die Rankings beeinflussen
- Europäische Ticker können unterrepräsentiert sein

**Risk-Attribution (Phase D2):**
- Faktoren können zukünftig auch mit Risk-Attribution kombiniert werden
- Beispiel: Faktoren mit hohem Combined Score, aber schlechtem Risk-Profil in bestimmten Regimes
- Siehe [Risk Metrics & Attribution Workflows](WORKFLOWS_RISK_METRICS_AND_ATTRIBUTION.md) für Details zur Regime- und Faktor-Gruppen-Attribution

**ML-Validierung & Feature Importance (Phase E1/E2):**
- Faktor-Rankings können gemeinsam mit ML-Validierung betrachtet werden
- Identifiziere Faktoren, die sowohl im klassischen Ranking (IC-IR, Deflated Sharpe) als auch in ML-Modellen hohe Beiträge liefern
- Beispiel: Faktoren mit hohem Combined Score UND hoher Feature-Importance in Random Forest-Modellen
- ML-Modelle können auch neue Faktor-Kombinationen entdecken, die im klassischen Ranking nicht sichtbar sind
- Siehe [ML Validation & Model Comparison Workflows](WORKFLOWS_ML_VALIDATION_AND_MODEL_COMPARISON.md) für Details zur ML-Validierung auf Factor-Panels

#### Feature Importance vs. Factor Ranking

**Kombinierte Interpretation:**

Factor Rankings (Phase C1/C2) und Feature Importance (Phase E2) ergänzen sich:

1. **Factor Rankings (Klassisch):**
   - IC-IR misst die prädiktive Kraft eines einzelnen Faktors
   - Deflated Sharpe Ratio quantifiziert die Performance von Single-Factor-Portfolios
   - Robust gegenüber Modell-Annahmen

2. **Feature Importance (ML-basiert):**
   - Misst den Beitrag eines Faktors zu Multi-Faktor-Vorhersagen
   - Erfasst Faktor-Interaktionen und nicht-lineare Effekte
   - Modell-spezifisch (verschiedene Modelle können Faktoren unterschiedlich gewichten)

**Interpretations-Hinweise:**

- **Hoher IC-IR + Hohe Feature Importance:**
  - Starker Faktor, der sowohl einzeln als auch in Kombination mit anderen Faktoren funktioniert
  - Beispiel: `factor_mom` mit IC-IR > 0.5 und Feature Importance > 0.7 im Ridge-Modell

- **Hoher IC-IR + Niedrige Feature Importance:**
  - Faktor ist einzeln stark, aber möglicherweise redundant in Multi-Faktor-Kontext
  - Andere Faktoren enthalten bereits ähnliche Informationen
  - Kann für Feature-Selection wichtig sein

- **Niedriger IC-IR + Hohe Feature Importance:**
  - Faktor trägt durch Interaktionen mit anderen Faktoren bei
  - Nicht-lineare Effekte, die in klassischen Rankings nicht sichtbar sind
  - Random Forest-Modelle können solche Effekte besser erfassen als lineare Modelle

**Praxis-Beispiel:**

Nach einem ML-Validation-Run können Sie die Outputs vergleichen:

1. **Factor Ranking CSV:** `output/factor_analysis/.../factor_rankings.csv`
   - Sortiert nach `combined_score` (IC-IR + Deflated Sharpe)
   - Top-Faktoren: `factor_mom` (combined_score = 0.85), `factor_value` (0.72)

2. **Feature Importance CSV:** `output/ml_validation/ml_feature_importance_ridge_20d.csv`
   - Sortiert nach `importance`
   - Top-Faktoren: `factor_mom` (importance = 0.82), `factor_vol` (0.65)

3. **Interpretation:**
   - `factor_mom`: Hoher IC-IR UND hohe Feature Importance → Konsens zwischen beiden Methoden
   - `factor_value`: Hoher IC-IR, aber möglicherweise niedrigere Feature Importance → Prüfen, ob andere Faktoren ähnliche Signale liefern
   - `factor_vol`: Niedrigere IC-IR, aber hohe Feature Importance → Trägt durch Interaktionen bei

**Wichtig:** Alle Feature-Importance-Berechnungen basieren auf lokalen Factor-Panels. Es werden keine Live-APIs im Explainability-Workflow verwendet.

---

## 3. Top-Faktoren Global

Die folgende Tabelle zeigt die **Top 10 Faktoren** nach `combined_score` aus der konsolidierten Ranking-Tabelle (`output/factor_analysis/factor_ranking_overview.csv`).

*Hinweis: Diese Tabelle wird automatisch aktualisiert, wenn `scripts/summarize_factor_rankings.py` ausgeführt wird.*

| Rank | Factor Name | Combined Score | IC-IR | Deflated Sharpe | Beschreibung |
|------|-------------|----------------|-------|-----------------|--------------|
| 1 | `return_12m_excl_1m` | *[Wird automatisch gefüllt]* | *[Wird automatisch gefüllt]* | *[Wird automatisch gefüllt]* | 12-Monats-Momentum ohne den letzten Monat (vermeidet kurzfristige Reversal-Effekte) |
| 2 | `trend_strength_50` | *[Wird automatisch gefüllt]* | *[Wird automatisch gefüllt]* | *[Wird automatisch gefüllt]* | Trend-Stärke basierend auf (Price - MA_50) / ATR_50 |
| 3 | `earnings_eps_surprise_last` | *[Wird automatisch gefüllt]* | *[Wird automatisch gefüllt]* | *[Wird automatisch gefüllt]* | Letzter Earnings Surprise (EPS Actual vs. Estimate) |
| 4 | `insider_net_notional_60d` | *[Wird automatisch gefüllt]* | *[Wird automatisch gefüllt]* | *[Wird automatisch gefüllt]* | Net Insider Transaction Value über 60 Tage (Buys minus Sells) |
| 5 | `news_sentiment_trend_20d` | *[Wird automatisch gefüllt]* | *[Wird automatisch gefüllt]* | *[Wird automatisch gefüllt]* | Trend in News Sentiment (Slope über 20 Tage) |
| 6 | `realized_volatility_20` | *[Wird automatisch gefüllt]* | *[Wird automatisch gefüllt]* | *[Wird automatisch gefüllt]* | Realized Volatility (annualisiert, 20-Tage Fenster) |
| 7 | `short_term_reversal_1d` | *[Wird automatisch gefüllt]* | *[Wird automatisch gefüllt]* | *[Wird automatisch gefüllt]* | Short-Term Reversal (Z-Score der 1-Tages-Returns) |
| 8 | `turnover` | *[Wird automatisch gefüllt]* | *[Wird automatisch gefüllt]* | *[Wird automatisch gefüllt]* | Turnover (Volume / Free Float, falls verfügbar) |
| 9 | `post_earnings_drift_return_20d` | *[Wird automatisch gefüllt]* | *[Wird automatisch gefüllt]* | *[Wird automatisch gefüllt]* | Post-Earnings-Drift (Forward Return nach Earnings-Event) |
| 10 | `macro_growth_regime` | *[Wird automatisch gefüllt]* | *[Wird automatisch gefüllt]* | *[Wird automatisch gefüllt]* | Makro-Wachstums-Regime-Indikator (+1 = Expansion, -1 = Rezession) |

**Hinweis:** Die tatsächlichen Rankings können variieren je nach Universe und Evaluationszeitraum. Siehe Abschnitt 4 für Universe-spezifische Rankings.

### 3.1 Faktor-Beschreibungen (Kurzreferenz)

#### Core TA/Price Factors (Phase A1)
- **Multi-Horizon Returns:** `return_1m`, `return_3m`, `return_6m`, `return_12m` – Log-Returns über verschiedene Horizonte
- **Momentum ex 1M:** `return_12m_excl_1m` – 12-Monats-Return ohne letzten Monat
- **Trend Strength:** `trend_strength_XX` – Normalisierte Distanz zwischen Price und Moving Average (MA_XX / ATR_XX)
- **Short-Term Reversal:** `short_term_reversal_Xd` – Z-Score der X-Tages-Returns (negative Korrelation erwartet)

#### Volatility & Liquidity (Phase A2)
- **Realized Volatility:** `realized_volatility_XX` – Annualisierte Volatilität aus log-Returns (XX-Tage Fenster)
- **Vol-of-Vol:** `vol_of_vol_XX` – Volatilität der Volatilität (zweite Ableitung)
- **Turnover:** `turnover` – Volume / Free Float (Liquiditäts-Proxy)
- **Spread Proxy:** `spread_proxy` – (High - Low) / Close (Intraday-Spread-Proxy)

#### Alt-Data Earnings & Insider (Phase B1)
- **Earnings Surprise:** `earnings_eps_surprise_last`, `earnings_revenue_surprise_last` – Prozentuale Abweichung von Schätzungen
- **Post-Earnings Drift:** `post_earnings_drift_return_XXd` – Forward Return nach Earnings-Event
- **Insider Activity:** `insider_net_notional_XXd`, `insider_buy_sell_ratio_XXd` – Aggregierte Insider-Transaktionen

#### Alt-Data News & Macro (Phase B2)
- **News Sentiment:** `news_sentiment_mean_XXd`, `news_sentiment_trend_XXd` – Aggregiertes News-Sentiment (Rolling Mean, Trend)
- **News Volume:** `news_sentiment_volume_XXd` – Anzahl News-Artikel über Zeitfenster
- **Macro Regime:** `macro_growth_regime`, `macro_inflation_regime`, `macro_risk_aversion_proxy` – Makroökonomische Regime-Indikatoren

---

## 4. Top-Faktoren je Universe (Kurzfassung)

### 4.1 Macro World ETFs (`macro_world_etfs`)

**Top 3–5 Faktoren:**

1. **[Wird automatisch aus `FACTOR_RANKING_BY_UNIVERSE.md` gefüllt]**
2. **[Wird automatisch gefüllt]**
3. **[Wird automatisch gefüllt]**

**Charakteristika:**
- Macro-ETFs zeigen typischerweise stärkere Reaktion auf makroökonomische Faktoren
- Trend-Stärke und Momentum-Faktoren sind oft dominant
- Volatilitäts-Faktoren können invers sein (hohe Vol → niedrige Returns bei breiten Indices)

**Factor-Set Breakdown:**
- **Core:** [Top 3 Core-Faktoren]
- **Core+Alt:** [Top 3 mit Alt-Data]
- **Core+Alt Full:** [Top 3 mit allen Faktoren]

### 4.2 AI/Tech Universe (`universe_ai_tech`)

**Top 3–5 Faktoren:**

1. **[Wird automatisch aus `FACTOR_RANKING_BY_UNIVERSE.md` gefüllt]**
2. **[Wird automatisch gefüllt]**
3. **[Wird automatisch gefüllt]**

**Charakteristika:**
- AI/Tech-Unternehmen zeigen oft stärkere Reaktion auf Earnings-Surprises
- News-Sentiment kann besonders wichtig sein (Technologie-News sind häufiger und einflussreicher)
- Insider-Activity kann frühe Signale für Technologie-Trends liefern

**Factor-Set Breakdown:**
- **Core:** [Top 3 Core-Faktoren]
- **Core+Alt:** [Top 3 mit Alt-Data]
- **Core+Alt Full:** [Top 3 mit allen Faktoren]

### 4.3 Unterschiede zwischen Universes

**Gemeinsamkeiten:**
- Momentum und Trend-Stärke sind typischerweise in beiden Universes stark
- Short-Term Reversal zeigt konsistent negative IC (wie erwartet)

**Unterschiede:**
- **Macro ETFs:** Stärker auf Makro-Regime-Faktoren angewiesen
- **AI/Tech:** Stärker auf Earnings/Insider/News-Faktoren angewiesen

**Interpretation:**
- Unterschiedliche Universes erfordern unterschiedliche Faktor-Strategien
- Ein universeller "One-Size-Fits-All" Faktor-Mix ist suboptimal
- Universe-spezifische Modelle (siehe Abschnitt 5) sind empfehlenswert

---

## 5. Nächste Schritte

### 5.0 Regime-spezifische Faktor-Analyse (Phase D1)

Die Faktor-Performance variiert je nach Marktregime. Mit dem Regime-Modell (Phase D1) können Rankings zukünftig auch **pro Regime** betrachtet werden:

- **IC/IR nach Regime**: Welche Faktoren funktionieren in Bull-Märkten vs. Bear-Märkten?
- **Regime-spezifische Bundles**: Anpassung von Factor-Bundles je nach identifiziertem Regime
- **Adaptive Strategien**: Regime-basierte Exposure-Steuerung und Faktor-Selektion

**Siehe:** [Workflows – Regime Models & Risk Overlay](WORKFLOWS_REGIME_MODELS_AND_RISK.md) für detaillierte Workflows und Beispiele.

### 5.1 Integration in Bestehende Strategien

**Trend-basierte Strategien:**
- **Aktuell:** Nutzt EMA-Crossover und Trend-Stärke (aus `sprint9_execute.py`)
- **Erweiterung:** Integriere Top-Ranked Trend-Stärke-Faktoren (`trend_strength_50`, `trend_strength_200`)
- **Vorgehen:**
  1. Multi-Faktor-Scoring: Kombiniere mehrere Top-Trend-Faktoren zu einem zusammengesetzten Score
  2. Signal-Filterung: Nutze IC-IR > 0.3 als Mindest-Schwelle für Faktor-Inklusion
  3. Backtesting: Validiere Performance-Impact im bestehenden Backtest-Framework

**Event-basierte Strategien:**
- **Aktuell:** Event-Study-Framework (Phase C3) für Earnings/Insider-Events
- **Erweiterung:** Nutze Top-Ranked Alt-Data-Faktoren für Event-Selektion
- **Vorgehen:**
  1. Filter Events: Nur Earnings-Events mit `earnings_eps_surprise_last > threshold`
  2. Insider-Filter: Nutze `insider_net_notional_60d` als zusätzliches Signal
  3. News-Overlay: Kombiniere mit `news_sentiment_trend_20d` für Kontext

### 5.2 Neue Strategien

**Multi-Faktor-Momentum:**
- Kombiniere Top-Ranked Momentum-Faktoren (`return_12m_excl_1m`, `trend_strength_50`)
- Long Top Decile, Short Bottom Decile basierend auf kombinierter Faktor-Score
- Rebalance monatlich, forward-looking IC als Gewichtung

**Alt-Data-Driven Strategies:**
- **Earnings Surprise Strategy:** Long positive Surprises, Short negative Surprises
- **Insider Activity Strategy:** Long hohe Insider-Buy-Ratios, Short hohe Sell-Ratios
- **News Sentiment Momentum:** Long positive Sentiment-Trends, Short negative Trends

**Macro-Regime-Aware Strategies:**
- Nutze `macro_growth_regime` und `macro_inflation_regime` für Regime-Filter
- In Expansion: Long Momentum-Faktoren
- In Rezession: Long Defensive/Reversal-Faktoren

### 5.3 Integration in ML-Modelle (Phase E)

**Feature Engineering:**
- Die Top-Ranked Faktoren können als **Features** in ML-Modellen verwendet werden
- Priorisierung: Beginne mit Top 10–20 Faktoren nach `combined_score`
- Feature Selection: Nutze IC-IR als Feature-Importance-Proxy (spart Rechenzeit)

**Ensemble-Ansätze:**
- **Level 1:** Einzelne Faktoren als Features
- **Level 2:** Ensemble mehrerer Faktoren (z.B. PCA oder Feature Aggregation)
- **Level 3:** Meta-Model über Faktor-Portfolios (siehe bestehende Meta-Model-Infrastruktur)

**Regime-Adaptive Modelle:**
- Train separate Modelle für verschiedene Makro-Regimes
- Switching-Logic basierend auf `macro_growth_regime` und `macro_inflation_regime`

**Validation:**
- Out-of-Sample IC/IR als Hauptmetrik für ML-Modell-Evaluation
- Deflated Sharpe Ratio für Portfolio-Performance
- Vermeide Overfitting durch Cross-Validation über Universes

### 5.4 Workflow-Integration

**Automatisierte Ranking-Updates:**
1. Nach jedem `analyze_factors` Run: Führe `summarize_factor_rankings.py` aus
2. Nach Universe-spezifischen Runs: Führe `factor_ranking_by_universe.py` aus
3. Aktualisiere dieses Dokument mit neuen Rankings (manuell oder per Script)

**Monitoring:**
- Track IC/IR über Zeit (Rolling Windows)
- Alerte bei signifikanten Änderungen in Top-Rankings
- Dokumentiere Regime-Wechsel und deren Impact auf Rankings

---

## 6. Configured Factor Bundles

Factor Bundles sind vorkonfigurierte Kombinationen von Faktoren mit Gewichtungen und Verarbeitungsoptionen. Sie ermöglichen die einfache Verwendung von bewährten Faktor-Kombinationen in Strategien ohne manuelle Gewichtung.

### Verfügbare Bundles

Die folgenden Factor Bundles sind konfiguriert:

| Bundle-Name | Universe | Factor-Set | Horizon (Tage) | Top-Faktoren | Gewichte |
|-------------|----------|------------|----------------|--------------|----------|
| `macro_world_etfs_core_bundle.yaml` | `macro_world_etfs` | `core+vol_liquidity` | 20 | `momentum_12m_excl_1m` (30%), `trend_strength_200` (25%), `trend_strength_50` (20%), `rv_20` (15%), `returns_12m` (10%) | Summe: 100% |
| `ai_tech_core_alt_bundle.yaml` | `universe_ai_tech` | `core+alt_full` | 20 | `momentum_12m_excl_1m` (25%), `trend_strength_50` (20%), `earnings_eps_surprise_last` (20%), `insider_net_notional_60d` (15%), `news_sentiment_trend_20d` (10%), `rv_20` (10%) | Summe: 100% |

**Bundle-Details:**

**1. Macro World ETFs Core Bundle:**
- **Zweck:** Breite Markt-ETF-Analyse mit Core TA/Price- und Volatilitäts-Faktoren
- **Fokus:** Momentum und Trend-Stärke für ETF-Universes
- **Faktoren:**
  - `momentum_12m_excl_1m` (30%): Langfristiges Momentum ohne Reversal-Effekte
  - `trend_strength_200` (25%): Langfristiger Trend-Indikator
  - `trend_strength_50` (20%): Mittelfristiger Trend-Indikator
  - `rv_20` (15%, negativ): Realized Volatility (invers: niedrige Vol = besser)
  - `returns_12m` (10%): 12-Monats-Returns als Momentum-Proxy

**2. AI/Tech Core + Alt Bundle:**
- **Zweck:** Sektor-spezifische Analyse für Tech-Unternehmen mit Alt-Data-Faktoren
- **Fokus:** Kombination aus Momentum/Trend und Alt-Data-Signalen (Earnings, Insider, News)
- **Faktoren:**
  - `momentum_12m_excl_1m` (25%): Momentum ohne Reversal
  - `trend_strength_50` (20%): Mittelfristiger Trend
  - `earnings_eps_surprise_last` (20%): Earnings Surprise (Alt-Data B1)
  - `insider_net_notional_60d` (15%): Insider Activity (Alt-Data B1)
  - `news_sentiment_trend_20d` (10%): News Sentiment Trend (Alt-Data B2)
  - `rv_20` (10%, negativ): Volatility als Risiko-Indikator

### Verwendung

**Laden eines Bundles:**

```python
from src.assembled_core.config.factor_bundles import load_factor_bundle

# Bundle laden
bundle = load_factor_bundle("config/factor_bundles/macro_world_etfs_core_bundle.yaml")

# Zugriff auf Konfiguration
print(f"Universe: {bundle.universe}")
print(f"Factor-Set: {bundle.factor_set}")
print(f"Horizon: {bundle.horizon_days} days")

# Faktoren und Gewichte
for factor in bundle.factors:
    print(f"{factor.name}: weight={factor.weight}, direction={factor.direction}")

# Verarbeitungsoptionen
print(f"Winsorize: {bundle.options.winsorize}")
print(f"Z-Score: {bundle.options.zscore}")
```

**Alle verfügbaren Bundles auflisten:**

```python
from src.assembled_core.config.factor_bundles import list_available_factor_bundles

bundles = list_available_factor_bundles()
for bundle_path in bundles:
    print(bundle_path.name)
```

### Verarbeitungsoptionen

Alle Bundles unterstützen folgende Verarbeitungsoptionen:

- **Winsorize:** Extreme Werte werden auf Quantile-Limits gekappt (standardmäßig 1% und 99%)
- **Z-Score:** Faktoren werden z-standardisiert (Mittelwert=0, Std=1)
- **Neutralize:** Optional können Faktoren gegen ein Feld neutralisiert werden (z.B. "sector")

Diese Optionen werden in zukünftigen Implementierungen der Factor-Bundle-Engine verwendet.

### Erweiterte Bundles

Weitere Bundles können durch Erstellen neuer YAML-Dateien in `config/factor_bundles/` hinzugefügt werden. Die Struktur ist dokumentiert in den bestehenden Bundle-Dateien.

**Wichtig:** Die Faktornamen müssen exakt mit den Spaltennamen in den Factor-Analysis-Outputs übereinstimmen (siehe `output/factor_analysis/*_ic_summary.csv`).

---

## 7. Referenzen & Weitere Dokumentation

### Workflows
- **[Factor Analysis Workflows](WORKFLOWS_FACTOR_ANALYSIS.md)**: Detaillierte Anleitung für `analyze_factors` CLI, Interpretation von IC/IR und Portfolio-Metriken, Smoketests

### Architektur & Design
- **[Advanced Analytics & Factor Labs](ADVANCED_ANALYTICS_FACTOR_LABS.md)**: Gesamtüberblick über Phase A (Factor Engineering), Phase B (Alt-Data), Phase C (Analysis Engine), Roadmap

- **[Architecture Review Summary](ARCHITECTURE_REVIEW_SUMMARY.md)**: Systematische Architektur-Überprüfung, API- & Daten-Architektur, Feature-Layer-Integration, Hidden Traps

### Scripts & Tools
- **`scripts/summarize_factor_rankings.py`**: Konsolidiert alle Factor-Analysis-Outputs zu einer globalen Ranking-Tabelle
- **`scripts/run_factor_analysis_smoketests.py`**: Automatisierte End-to-End-Tests für Factor-Analysis-Pipeline
- **`research/factors/factor_ranking_by_universe.py`**: Universe-spezifische Ranking-Analyse mit Plots

### Output-Dateien
- **`output/factor_analysis/factor_ranking_overview.csv`**: Konsolidierte Ranking-Tabelle (alle Universes/Factor-Sets)
- **`output/factor_analysis/factor_ranking_overview.md`**: Markdown-Report mit Top 20 Faktoren
- **`output/factor_analysis/FACTOR_RANKING_BY_UNIVERSE.md`**: Universe-spezifische Rankings
- **`output/factor_analysis/plots/*.png`**: Visualisierungen der Top-Faktoren (optional)

---

**Letzte Aktualisierung:** *[Wird automatisch beim Ausführen der Scripts aktualisiert]*

**Datenbasis:** Lokale Alt-Daten-Snapshots (keine Live-API-Calls)

