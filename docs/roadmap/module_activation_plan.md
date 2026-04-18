# Module-Activation Plan (Post v3-Ultra-Plan)

**Erstellt:** 2026-04-17
**Kontext:** Plan-v3 ist abgeschlossen. System-Audit hat **60 Orphan-Module** (0 Importer) und **116 tests-only-Module** gefunden, die getestet aber nicht in den Produktionspfad verdrahtet sind. Dieser Plan priorisiert Aktivierung nach **Risk-Wertschöpfung × Aufwand × Rückrollbarkeit**.

## Leitprinzipien

1. **Kein Big-Bang.** Sprints ≤ 5 Module pro Batch.
2. **Shadow vor Enable.** D-Sequenz-Methodik (5d Shadow + 10d Enabled).
3. **Reversibel.** Jede Aktivierung muss via Policy-Flag in 1 Commit revertibel sein.
4. **Keine sensitive Kern-Umbauten ohne expliziten Auftrag** (CLAUDE.md §6).

---

## Tier 1 — Risk-kritisch, minimaler Aufwand (JETZT)

Module, die bereits eine klare Call-Site im Pipeline-Pfad haben könnten und risiko-erhöhend sind, wenn unverdrahtet. Aktivierung in diesem Commit.

| Modul | Status | Wiring-Ziel | Rollback-Flag |
|---|---|---|---|
| `risk/var_methods.py` | tests-only | `pipeline/trading_cycle` — pre-trade Exposure-Check mit parametrischer VaR | `policy.risk.var_gate.enabled=False` |
| `risk/circuit_breaker.py` | tests-only | `pipeline/trading_cycle` — nach Reconcile; engage bei N% MDD intraday | `policy.risk.circuit_breaker.enabled=False` |
| `execution/symbol_kill_switch.py` | tests-only | `execution/unified_paper_engine` — per-symbol Halt | `policy.execution.symbol_kill.enabled=False` |
| `accounting/currency.py` | tests-only | `accounting/position_engine` — FX-Aware Mark-to-Market (Schutz gg. Multi-Currency-Drift) | n/a (passive, nur addiert Spalten) |
| `qa/signal_decay.py` | **DONE** | CI-Script wires `strategies/signal_decay_gate.py` | `policy.signal_decay.enabled=False` |

---

## Tier 2 — Institutional-Härtung (nächster Sprint)

Module, die den institutionellen Anspruch erhöhen, aber größere Integrationsarbeit erfordern.

| Modul | Einsatzgebiet | Schätzung | Risiko |
|---|---|---|---|
| `portfolio/hrp_sizing.py` | Hierarchical Risk Parity als alt. Sizer | 2-3d | MED |
| `portfolio/bl_sizing.py` | Black-Litterman mit Views | 2-3d | MED |
| `portfolio/risk_budgeting.py` | Equal-Risk-Contribution | 1-2d | LOW |
| `portfolio/mvo_optimizer.py` | Markowitz-MVO | 2d | LOW |
| `execution/almgren_chriss.py` | optimale Execution-Trajectories | 3-4d | MED-HIGH (Memory behauptet gewired — widerlegt) |
| `execution/cost_model_calibrator.py` | Real-vs-Sim Kalibrierung | Wire in E5-Loop | LOW |
| `execution/portfolio_execution.py` | Batch-Order-Manager | 2-3d | MED |
| `risk/tail_hedging.py` | Put-Protection-Overlay | 3d + D-Shadow | HIGH |
| `risk/attribution.py` | Risk-Attribution-Report | 1-2d | LOW |

---

## Tier 3 — Alt-Data & Intel (selektiv, viele sind experimentell)

116 Intel/Alt-Data/ML-Module sind tests-only oder orphan. Die meisten sind **experimentelle/early-stage Module**, deren Aktivierung nur nach einer strategischen Entscheidung sinnvoll ist (CLAUDE.md §13.4 "spätere Ausbaurichtung").

Beispielklassen:
- `intel/*` — 11 Orphans, 10 tests-only: nur via `run_intel_cycle.py` erreichbar
- `events/news/*` — komplette Pipeline außer `pipeline.py` ist tests-only
- `ml/gnn_stocks.py`, `ml/maml.py`, `ml/rl_portfolio.py` — ML-Experimente
- `features/*_features.py` (10 tests-only) — spezielle Factor-Module

**Empfehlung:** Erst nach einer bewussten Strategie-Entscheidung wiedereingelesen, ob die Domain überhaupt aktiviert werden soll. Alternative: dokumentierter Deprecation-Lauf, wenn die Module nicht mehr Teil der Roadmap sind.

---

## Tier 4 — Explizit aufzuräumen (Deprecation / Entfernen)

Module, die keinem aktuellen Strategie-Bild entsprechen und wegen fehlenden Maintainer-Fokus besser entfernt werden sollten. **Nur nach Auftrag** (CLAUDE.md §10.4: keine großen Doku-Umbauten ohne Auftrag; analog für Code).

Kandidaten:
- `strategies/stat_arb/*` — 3 Orphan-Module ohne Stat-Arb-Roadmap
- `data/streaming/*` — 2 Orphan-Module ohne Streaming-Roadmap
- `events/evidence_engine/*` — 4 Orphan-Module ohne Evidence-Engine-Roadmap

---

## Post-Tier-1-Exit-Gate

Nach Abschluss Tier 1:

- [ ] `risk/var_methods` produziert Artefakt pro Bar
- [ ] `risk/circuit_breaker` Shadow-Log mit simulierten Trips
- [ ] `execution/symbol_kill_switch` engage-testbar
- [ ] `accounting/currency` liefert FX-ausgewiesene Equity-Zeile
- [ ] phase12 weiterhin 1243+ passed
- [ ] Neue regression tests für alle Tier-1-Aktivierungen
