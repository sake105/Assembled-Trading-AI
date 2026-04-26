# 32 — Validierung (CPCV, Deflated Sharpe, Walk-Forward)

**Zweck:** Saubere Validierung ist der **entscheidende Alpha-Hebel**. Die meisten Retail-Quant-Strategien verlieren Geld, weil sie gegen falsche Baselines trainieren oder Leakage in der Validation haben.

Dieses Dokument ist der wichtigste Schutz vor den Standard-Fallen.

---

## Die vier Säulen

1. **CPCV (Combinatorial Purged Cross-Validation)** statt K-Fold
2. **Deflated Sharpe Ratio** statt nackter Sharpe
3. **Meta-Labeling mit Event-Triple-Barrier** statt einfachem Classification
4. **Walk-Forward + Shadow-Mode + Canary-Deployment** statt Big-Bang

---

## 32.1 CPCV — Combinatorial Purged Cross-Validation

### Warum nicht K-Fold?

Standard-`sklearn.KFold` bei Zeitreihen ist **Leakage-generierend**:

```
K-Fold auf Tagesdaten 2020-2024:
  Fold 1: Test 2020-Q1, Train Q2-Q4 + 2021-24  ← FUTURE LEAKAGE
  Fold 2: Test 2020-Q2, Train Q1 + Q3-24       ← FUTURE LEAKAGE
  ...
```

Training-Daten enthalten **Zukunft** relativ zum Test-Fold. Bei Finanz-Features mit Auto-Korrelation führt das zu **fiktivem Sharpe** (historisch 30-60% zu hoch).

### Was CPCV anders macht

López de Prado (AFML Kap. 7) definiert:
- **Purging:** Train-Samples, die zeitlich mit Test-Label überlappen, werden entfernt.
- **Embargo:** Nach jedem Test-Fold eine Lücke, damit Feature-Propagation nicht leakt.
- **Combinatorial:** Statt K-Fold (1 Test, K-1 Train) → `C(K, k)` Paths.

### Library

```python
from skfolio.model_selection import CombinatorialPurgedCV

cpcv = CombinatorialPurgedCV(
    n_folds=6,
    n_test_folds=2,
    purged_size=vertical_barrier_max,  # CRITICAL: ≥ Triple-Barrier-Vertical
    embargo_size=2  # Business-Days
)

# Anzahl Paths: C(6,2) = 15 Backtest-Pfade
for i, (train_idx, test_idx) in enumerate(cpcv.split(X, y)):
    model.fit(X[train_idx], y[train_idx])
    preds = model.predict(X[test_idx])
    # Sammle OOS-Returns pro Pfad
```

### Der häufigste Fehler

**`purged_size < vertical_barrier`** = Leakage trotz CPCV. Standard-Bug in Online-Tutorials.

**Regel:** Wenn deine Triple-Barrier vertical=10 Bars hat, ist `purged_size=10` das absolute Minimum. Lieber 15.

### Embargo-Wahl

Embargo schützt vor **Feature-Propagation**:
- Features mit `lookback_window_n` Bars → Embargo ≥ `n + 1`
- Rolling-Z-Scores 30d → Embargo ≥ 30 (überkonservativ) oder `prediction_horizon + 1`

**Default:** `embargo_size=5` Business-Days bei Daily-Bars. Überkonservativ.

---

## 32.2 Deflated Sharpe Ratio

### Warum nackter Sharpe lügt

Wenn du 20 Strategien testest, findest du **per Zufall** eine mit hohem Sharpe. Das ist Multiple-Testing-Bias.

Bailey/López de Prado (2014) entwickelten **Deflated Sharpe Ratio (DSR)** als formellen Test.

### Die Formel

```
DSR = E[( Ŝ − Ŝ_0 ) / σ(Ŝ)]
```

Wobei `Ŝ_0` = Expected-Maximum Sharpe aus `N` Zufalls-Trials mit Länge `T`:

```
Ŝ_0 = √((1 − γ) × Z^{-1}(1 − 1/N) + γ × Z^{-1}(1 − 1/(N × e)))
```

mit `γ ≈ 0.5772` (Euler-Mascheroni).

### Implementierung

```python
import numpy as np
from scipy import stats

def deflated_sharpe_ratio(sharpe_observed, n_trials, n_observations,
                          skew=0.0, kurtosis=3.0):
    """
    Bailey/López de Prado 2014, JFDS 2(3).
    """
    # Expected maximum Sharpe from N trials
    gamma = 0.5772156649
    e_max_sr = ((1 - gamma) * stats.norm.ppf(1 - 1/n_trials) +
                gamma * stats.norm.ppf(1 - 1/(n_trials * np.e)))
    
    # Sharpe-Varianz unter Nicht-Normalverteilung
    sigma_sr = np.sqrt(
        (1 - skew * sharpe_observed +
         ((kurtosis - 1) / 4) * sharpe_observed**2) / (n_observations - 1)
    )
    
    # DSR
    dsr_prob = stats.norm.cdf(
        ((sharpe_observed - e_max_sr) * np.sqrt(n_observations - 1)) /
        np.sqrt(1 - skew * sharpe_observed + ((kurtosis - 1) / 4) * sharpe_observed**2)
    )
    return dsr_prob  # P(Strategy is not due to chance)
```

### Interpretation

- `DSR > 0.95`: Sehr wahrscheinlich echte Alpha
- `DSR > 0.90`: Wahrscheinlich, aber vorsichtig
- `DSR < 0.80`: Wahrscheinlich Zufall, nicht live gehen

**Regel:** DSR vor jeder Live-Entscheidung berechnen. Anzahl "Trials" = alle Hyperparameter-Konfigurationen, die du je getestet hast — ja, auch die manuell verworfenen zählen.

---

## 32.3 PBO — Probability of Backtest Overfitting

**López de Prado Bailey 2016.** PBO schätzt, wie wahrscheinlich der beste In-Sample-Strategie-Kandidat im Out-of-Sample schlecht abschneiden wird.

### Implementierung

```python
from mlfinpy.cross_validation import combinatorial_purged_cv
# oder eigene Implementierung

def probability_backtest_overfitting(is_returns_matrix, oos_returns_matrix):
    """
    is_returns_matrix: (n_strategies, n_periods_is)
    oos_returns_matrix: (n_strategies, n_periods_oos)
    
    Returns: PBO in [0, 1], lower is better
    """
    ranks_is = np.argsort(is_returns_matrix.mean(axis=1))[::-1]
    best_strategy_is = ranks_is[0]
    
    # wie oft rankt sie im OOS unter Median?
    ranks_oos = np.argsort(oos_returns_matrix.mean(axis=1))[::-1]
    oos_rank = np.where(ranks_oos == best_strategy_is)[0][0]
    median_rank = len(ranks_oos) // 2
    
    pbo = oos_rank / len(ranks_oos)  # 0 = best, 1 = worst
    return pbo
```

### Target

**PBO < 0.5** für Paper-Reports. **PBO < 0.3** für Live-Deployment.

---

## 32.4 Meta-Labeling mit Event-Triple-Barrier

### Die Wahrheit über Standard-Backtests

Standard-Pattern: "Buy when signal >0, sell next day" ist ein **Stiefbroder der Realität**:
- Keine Stop-Losses
- Kein Profit-Targeting
- Keine Event-basierten Exits

**Triple-Barrier** fixt das:

```python
from mlfinpy.labeling import get_events, get_bins, add_vertical_barrier

def triple_barrier_labels(close, t_events, pt_sl=[2, 2], target_vol=None,
                          min_ret=0.005, num_threads=4,
                          vertical_barrier_days=5):
    # Volatility für dynamic PT/SL
    if target_vol is None:
        target_vol = close.pct_change().rolling(20).std()
    
    # Vertical Barrier
    t1 = add_vertical_barrier(t_events, close,
                              num_days=vertical_barrier_days)
    
    # Events: Touch of Upper/Lower/Vertical
    events = get_events(close=close, t_events=t_events,
                        pt_sl=pt_sl, target=target_vol,
                        min_ret=min_ret, num_threads=num_threads, t1=t1)
    
    # Labels: {-1, 0, +1}
    bins = get_bins(events, close)
    return bins
```

**PT/SL=2,2 × daily_vol** ist Default. Pro Event-Context anpassen (siehe `30_NEWS_TA_FUSION.md` §Dynamic Triple-Barrier).

---

## 32.5 Walk-Forward-Analysis

### Struktur

```
Time:   Jan2020 ──────── Jan2022 ──── Jul2022 ──── Jan2023 ...
Fold 1:   [───── Train ─────][Test]
Fold 2:              [───── Train ─────][Test]
Fold 3:                         [───── Train ─────][Test]
...
```

**Expanding Window** vs **Rolling Window:**
- Expanding: Train-Set wächst (mehr Daten → stabilere Modelle)
- Rolling: Fixed Train-Length (reagiert schneller auf Regime-Shifts)

**Für Solo-Quant:** Rolling mit `train=504 Bars (2 Jahre)`, `test=63 Bars (3 Monate)`, `step=63 Bars`.

### Implementierung mit skfolio

```python
from skfolio.model_selection import WalkForward

wf = WalkForward(
    train_size=504,
    test_size=63,
    step_size=63,
    expanding_window=False,
)

for i, (train_idx, test_idx) in enumerate(wf.split(X)):
    model.fit(X[train_idx], y[train_idx])
    preds[i] = model.predict(X[test_idx])
```

### Performance-Drift-Erkennung

```python
def detect_wf_drift(oos_sharpes_by_fold, window=5, alarm_threshold=-1.0):
    recent_mean = np.mean(oos_sharpes_by_fold[-window:])
    historical_mean = np.mean(oos_sharpes_by_fold[:-window])
    historical_std = np.std(oos_sharpes_by_fold[:-window])
    
    z = (recent_mean - historical_mean) / historical_std
    if z < alarm_threshold:
        return "DRIFT ALARM: recent performance degrading"
    return "OK"
```

---

## 32.6 Shadow-Mode

Jedes neue Signal läuft **≥60 Tage im Shadow-Mode**, bevor es Live-Size bekommt.

```python
class ShadowSignal:
    def __init__(self, name, handler):
        self.name = name
        self.handler = handler
        self.live = False
        self.shadow_trades = []
    
    def emit(self, context):
        result = self.handler(context)
        result['shadow'] = not self.live
        result['signal_name'] = self.name
        
        if not self.live:
            self.shadow_trades.append(result)
            # Kein Order-Placement
        
        # Log für Analyse
        log_signal_emission(result)
        return result
```

**Metrics während Shadow:**
- Rolling IC pro Signal (Spearman-Rank 60d)
- Hit-Rate
- Average-Win / Average-Loss
- Simulated-Sharpe gegen reales Portfolio

**Promote-Regeln nach ≥60 Tagen:**
- IC_60d > 0.03 **und** Sharpe_60d > 0.5 → Canary mit 10% Size
- IC_60d < 0 für 20 Tage in Folge → Discard
- IC_60d zwischen → Weiter Shadow-Mode

---

## 32.7 Canary-Deployment

Nach Promote aus Shadow → **Canary-Deployment** mit wachsender Size:

```python
CANARY_SCHEDULE = [
    (0, 5, 0.0),      # Tag 1-5: Shadow (0%)
    (6, 20, 0.10),    # Tag 6-20: 10% Size, wenn Sharpe_15d>0.5
    (21, 45, 0.33),   # Tag 21-45: 33%, wenn Sharpe>0.5 + DD<1.5× Training
    (46, None, 1.0),  # Tag 46+: Full Size
]

def canary_size(days_since_live, sharpe_15d, drawdown_ratio):
    for start, end, target_size in CANARY_SCHEDULE:
        if end is None or days_since_live <= end:
            # Conditions
            if target_size > 0:
                if sharpe_15d < 0.5 or drawdown_ratio > 1.5:
                    return 0  # pause
            return target_size
    return 1.0
```

**Auto-Rollback-Regel:**
```python
def auto_rollback(drawdown_observed, drawdown_95q_simulated):
    if drawdown_observed > 2 * drawdown_95q_simulated:
        alert_slack("Auto-Rollback: DD exceeds 2× simulated 95th percentile")
        pause_signal()
        return True
    return False
```

---

## 32.8 Drift-Detection (Production)

**PSI (Population Stability Index)** pro Feature vs. Training-Baseline:

```python
def psi(reference, current, bins=10):
    """Population Stability Index."""
    ref_hist, _ = np.histogram(reference, bins=bins)
    cur_hist, _ = np.histogram(current, bins=bins)
    
    ref_pct = ref_hist / ref_hist.sum()
    cur_pct = cur_hist / cur_hist.sum()
    
    # Avoid log(0)
    ref_pct = np.where(ref_pct == 0, 1e-6, ref_pct)
    cur_pct = np.where(cur_pct == 0, 1e-6, cur_pct)
    
    return np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
```

**Interpretation:**
- PSI < 0.1: Stabil
- PSI 0.1-0.2: Moderate Drift
- PSI > 0.2: Signifikante Drift → Review
- PSI > 0.3: Auto-Pause

**Cronjob:** Täglich PSI berechnen, bei Threshold alert.

---

## 32.9 Der komplette Validation-Flow

```
┌─────────────────────────────────────────────┐
│ 1. Training mit CPCV                         │
│    - n_folds=6, n_test_folds=2              │
│    - purged_size ≥ vertical_barrier          │
│    - embargo_size=5                          │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│ 2. Deflated Sharpe berechnen                │
│    DSR > 0.90 → weiter                      │
│    DSR < 0.80 → verwerfen                   │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│ 3. PBO-Test                                  │
│    PBO < 0.3 → weiter                       │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│ 4. Walk-Forward-Analysis                     │
│    12+ OOS-Folds, stabil?                   │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│ 5. Shadow-Mode ≥60 Tage                     │
│    Rolling IC_60d > 0.03, Sharpe > 0.5      │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│ 6. Canary-Deployment (10% → 33% → 100%)     │
│    Auto-Rollback bei DD > 2× simulated      │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│ 7. Production-Monitoring                     │
│    PSI-Drift täglich, IC-Decay-Alert        │
└─────────────────────────────────────────────┘
```

---

## Umsetzungs-Checkliste

- [ ] skfolio CombinatorialPurgedCV als Standard-CV
- [ ] Deflated-Sharpe-Funktion im Validation-Utils
- [ ] PBO-Implementierung oder via mlfinpy
- [ ] Triple-Barrier via mlfinpy mit dynamic PT/SL
- [ ] Walk-Forward-Pipeline mit ≥12 OOS-Folds
- [ ] Shadow-Signal-Klasse mit Rolling-IC-Tracking
- [ ] Canary-Deployment-Schedule mit Auto-Rollback
- [ ] PSI-Drift-Daily-Cronjob mit Slack-Alerts
- [ ] MLflow-Run pro Validation-Lauf gespeichert
- [ ] Dashboard: Pro-Signal-IC + DSR + PSI

---

## Ehrliche Einschätzung

**Der Alpha-Hebel ist nicht die Indikator-Wahl oder das coolste ML-Modell.** Es ist Validierungs-Disziplin.

**Was die meisten Solo-Quants falsch machen:**
- K-Fold statt CPCV → fiktiver Sharpe
- Kein Deflated Sharpe → Multiple-Testing-Blindheit
- Kein Shadow-Mode → Big-Bang-Deployment mit Überraschungen
- Kein Drift-Monitoring → alte Signale, die längst Geld verlieren

**Die Validation-Disziplin ist unsexy, aber es ist der Unterschied zwischen "System funktioniert im Backtest" und "System funktioniert live".**

**Die Abkürzung gibt es nicht.** Nur geprüftes Alpha ist echtes Alpha.
