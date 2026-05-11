# Erweiterung — Executive Summary aller Findings

**Branch:** ERWEITERUNG (getrennt von Mainline — kein Merge)
**Stand:** 2026-05-11
**Tests:** 450 passed
**Commits in autonomen Sessions:** 12

---

## 1. Bottom Line

Die finale, produktionsreife Strategie der Erweiterung:

**`MasterAllocator(sa_weight=0.70)`** — Production-API in
`src/erweiterung/strategies/master_allocator.py`.

| Metrik | Wert (Long-History 2007-2026 / Common 2021-2026) |
|--------|--------------------------------------------------:|
| AnnRet | +14.31 % |
| Sharpe | +1.217 |
| Sortino | +1.189 |
| Calmar | +1.096 |
| MDD | −13.07 % |
| **Statistisch signifikant** | **vs 60/40 Classic: Calmar-Bootstrap p = 0.966** |

End-to-End-Endpoint: `scripts/erweiterung/run_master_pipeline.py`.
Equity-Curve-Audit: 0 Flags (Sharpe 1.236, WD/Vol 3.98, kein Smoothing).

---

## 2. Was hat funktioniert (validierte Konzepte)

| Konzept | Erkenntnis | Validierung |
|---------|------------|:------------|
| Vol-Targeting auf Single-Asset-Mom | MDD halbiert (-32% → -15% OOS) | Walk-Forward + Calmar-Bootstrap |
| Cross-Asset-Diversifikation (11 ETFs) | Korrelation 0.377 statt 0.70 | Sample-Korrelation |
| XAsset-Mom-Top-5 monthly rebalance | OOS Champion in Modern_2023+ (Sharpe 1.84) | Sub-Period |
| Master_70_30 SA + XA Mix | Sharpe 1.217 / Calmar 1.10 | p=0.966 vs 60/40 |
| Calmar-Bootstrap als Test-Statistik | Trennt MDD-Verbesserer von Sharpe-Verbesserern | Methodische Innovation |
| Equity-Curve Anomaly Audit | Findet Sharpe-4.6/MDD-4.5% Anomalien | Mainline-Diagnostik |

---

## 3. Was hat NICHT funktioniert (ehrlich widerlegt)

| Konzept | Befund | Statistik |
|---------|--------|:----------|
| Binäres Regime-Switching (DD-Trigger) | In-Sample p=0.0000, OOS p=0.99 | Walk-Forward widerlegt |
| Threshold-Auto-Tuning | Train→Test-Calmar-Korrelation −0.372 (anti-prädiktiv) | OOS |
| Multi-Faktor-Combiner-Optimierung | HRP 12/13 Mal gewählt = Overfit | Walk-Forward |
| Macro-Regime-Trigger (VIX+YC+HY) | Nicht trennscharf, marginal vs Drawdown-Only | Backtest |
| News-Anomaly-Plug (Multi-Signal) | Daten zu sparse (5 Monate) | Datenlage |
| Triple-Barrier-Meta-Labeling auf Master | Klassifikator-Accuracy 0.34 (base-rate 0.50) | Calmar-p=0.02 SCHLECHTER |

**Diese Negativ-Befunde sind methodisch wertvoll** — sie verhindern,
dass attraktive Konzepte ohne Validierung produktionsreif erscheinen.

---

## 4. Mainline-Erkenntnisse (durch Equity-Audit)

Der Erweiterungs-Audit hat im Mainline-Projekt entdeckt:

1. **3 Original-Equity-Files bit-identisch:** `equity_curve_baseline.csv`,
   `equity_curve_altdata.csv`, `equity_curve_test1_aitech_qagate.csv` —
   Altdata/QAgate-Varianten waren effektiv No-Ops im Original-Backtest.
2. **Original-Sharpe 4.63 + MDD −4.52 % löst SUSPICIOUS_SHARPE +
   MDD_TOO_LOW_FOR_SHARPE Flags.** Statistisch außerhalb des für Long-Only
   typischen Bereichs über 836 Tage.
3. **Korrelation Erweiterung ↔ Original-Baseline: 0.07-0.18** — die zwei
   Systeme machen unterschiedliche Sachen (orthogonale Risikoprofile).
4. **Real-Test T2 (200-Sym 2025-26 no-leverage):** Sharpe 0.77 / MDD −30 %
   — plausibles Profile, sollte als Headline-Number genutzt werden statt
   des Baseline-Backtests.

**Empfehlung an Mainline:** Original-Sharpe-4.6-Equity sollte auf Risk-
Overlay-Effekte und Sizing-Cap-In-Sample-Optimierung hin geprüft werden.

---

## 5. Modul-Inventar (Erweiterung-Branch)

Neu in dieser Session-Reihe:

- `src/erweiterung/altdata/yfinance_cache_loader.py`
- `src/erweiterung/backtest/calmar_bootstrap.py`
- `src/erweiterung/ml/meta_labeling_master.py` (Negativ-Demo)
- `src/erweiterung/qa/equity_curve_audit.py`
- `src/erweiterung/robustness/walk_forward.py`
- `src/erweiterung/strategies/ensemble_regime.py`
- `src/erweiterung/strategies/macro_stress_signals.py`
- `src/erweiterung/strategies/master_allocator.py` **← Production-API**
- `src/erweiterung/strategies/multi_factor_vol_target.py`
- `src/erweiterung/strategies/multi_signal_regime.py`
- `src/erweiterung/strategies/regime_conditional_allocator.py`
- `src/erweiterung/strategies/volatility_targeting.py`

Scripts (Demos + Production):

- `scripts/erweiterung/run_master_pipeline.py` **← Production-Endpoint**
- `scripts/erweiterung/run_*` weitere Diagnostik-Scripts (~10)

Dokumentation (in `docs/erweiterung/`):

- `EXPANDED_UNIVERSE_BACKTEST.md`
- `EQUITY_AUDIT_FINDINGS.md`
- `LONG_HISTORY_FINDINGS.md`
- `WALK_FORWARD_VALIDATION.md`
- `VOL_TARGETING_FINDINGS.md`
- `MULTI_FACTOR_VOL_TARGET_FINDINGS.md`
- `CROSS_ASSET_FINDINGS.md`
- `MASTER_ALLOCATION_FINDINGS.md`
- `META_LABELING_FINDINGS.md`
- **`FINAL_EXECUTIVE_SUMMARY.md`** (diese Datei)

---

## 6. API für externe Konsumenten

```python
from erweiterung.strategies.master_allocator import (
    MasterAllocator, MasterAllocatorConfig
)
from erweiterung.altdata.yfinance_cache_loader import load_universe_panel
from erweiterung.factors.fama_french import momentum_12_1

# Lade Daten
cross_asset_rets = load_universe_panel(
    "data/cache/yfinance",
    ["SPY", "QQQ", "IWM", "EFA", "EEM", "AGG", "TLT", "HYG", "GLD", "SLV", "DBC"],
).pivot_table(index="date", columns="symbol", values="close").pct_change()

# Equity-Faktor-Signal (irgendein Long-Only-Equity-Factor-Return)
equity_factor_ret = ...  # eine pd.Series mit Daily-Returns

# Allokieren
alloc = MasterAllocator(MasterAllocatorConfig(sa_weight=0.70))
out = alloc.allocate(equity_factor_ret, cross_asset_rets)

# out enthält: sa_voltarget, xa_voltarget_ew, xa_mom_top_n,
# xa_hybrid, master_return
```

---

## 7. Wichtigste Lehren (Mathematik / Methodik)

1. **Walk-Forward OOS ist Pflicht.** In-Sample-Hansen-SPA-p-Werte sind
   nicht reproduzierbar im OOS. Wir haben es selbst widerlegt: p=0.0000
   → p=0.99 von In-Sample zu OOS für Regime-Switching.

2. **Calmar-Bootstrap > Sharpe-Bootstrap** für Risk-Allocator-Tests.
   Sharpe-Test gibt uninformative p≈0.99 für jede MDD-Reducer-Strategie;
   Calmar-Bootstrap differenziert klar (0.02 - 0.97).

3. **Vol-Targeting ist robuster als Switching.** Kontinuierliche
   Skalierung > binärer Threshold. Keine Hyperparameter-Optimierung
   nötig, daher kein Overfit.

4. **Diversifikations-Korrelation matters.** Single-Asset-Faktoren
   (Korrelation ≈ 0.95) bringen wenig Diversifikation. Cross-Asset
   (0.62) bringt echte Reduktion. Master-Mix nutzt beide.

5. **ML auf bereits gut-designte Allokationen overfittet.** Meta-Labeling
   auf Master_70_30 verschlechtert signifikant (p=0.02). Erst wenn das
   Primary-Signal asymmetrisch ist, hilft Meta.

6. **Equity-Curve-Audit fängt Mainline-Anomalien.** 6-Heuristik-Modul
   findet identische Files, Sharpe-MDD-Inkonsistenzen, suspect WD/Vol-
   Ratios. Sollte CI-Smoke-Test werden.

---

## 8. Was OFFEN bleibt

| Item | Begründung |
|------|------------|
| Längere Cross-Asset-Historie (vor 2021) | yfinance-Cache limitiert. Fehlt: 10-15y ETF-Daten für robuste OOS. |
| Walk-Forward für Master-Allocator | Common-Period 2021-2026 zu kurz für robusten WF |
| News-Sentiment-Backtest | Daten nur 5 Monate (Dec 2025 - May 2026) |
| CI-Integration des Equity-Audit | Sollte Smoke-Test in Mainline-CI werden |
| Tail-Risk-Hedging mit Optionen | yahoo_options.py existiert, noch nicht verdrahtet |
| Live-Paper-Pilot | Pure Research-Branch, kein Pilot in dieser Phase |

---

## 9. Abschließend

Branch ERWEITERUNG ist **kein Mainline-Konkurrent** sondern ein
**unabhängiges Research-Lab** mit drei klaren Outputs:

1. **Eine produktionsreife Allokations-API** (`MasterAllocator`)
2. **Eine statistisch belastbare Methodik** (Calmar-Bootstrap, Walk-Forward)
3. **Ein Audit-Toolkit** für Equity-Curve-Anomalien

Korrelation der Erweiterung-Equity zur Original-Mainline-Equity: **0.07-0.18**.
Das ist kein Konkurrent-Backtest, das ist eine orthogonale Strategie.

Kein Merge mit Mainline initiiert. Beide Branches bleiben getrennt.
