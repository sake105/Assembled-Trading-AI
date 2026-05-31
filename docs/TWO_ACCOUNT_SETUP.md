# Two-Account-Setup (Audit C2-074)

**Status:** Operations-Doc Template (2026-05-18).
**Pflicht-Lesen:** vor jeder Live-/Real-Money Trading-Aktivität.

## Zweck

Trennung zwischen **Research-Account** (R&D, Experimente, Risiko-Toleranz hoch)
und **Trading-Account** (echtes Geld, harte Risk-Grenzen, nur "promoted"
Strategien).

Verhindert das klassische Anti-Pattern: eine Strategie wird im Live-Trading
aktiviert, bevor sie ausreichend out-of-sample und unter realistischer
Risk-Disziplin validiert ist. Ohne Account-Trennung sind die Anreize, eine
schwächelnde Forschungs-Idee "noch mal eine Woche zu geben", strukturell
zu stark.

---

## Account-Struktur

### Account R (Research)

| Attribut | Wert |
|---|---|
| Zweck | Strategie-Entwicklung, Hypothesen-Tests, Out-of-Sample-Validierung |
| Kapital | Klein (1-5% des Trading-Kapitals oder ein fixed amount, z. B. €1k-5k) |
| Position-Sizing | Aggressiv (volle Sharpe-Maximierung erlaubt) |
| Risk-Grenzen | Lockerer (z. B. MaxDD 20-30%) |
| Hebel | Nicht erlaubt — auch hier |
| Audit-Disziplin | Forensic-Audits laufen, aber Verdicts NICHT-blocking |
| Reports | Wöchentlich Sharpe / MDD / Bias-Audit (`scripts/forensic/`) |

### Account T (Trading)

| Attribut | Wert |
|---|---|
| Zweck | Echtgeld-Ausführung promoted Strategien |
| Kapital | Hauptkapital (z. B. €10k-100k) |
| Position-Sizing | Konservativ (half-Kelly oder weniger, vol-target) |
| Risk-Grenzen | Hart (MaxDD 10-15%, daily-loss-cap 2-3%) |
| Hebel | **Kein Leverage** (PROJEKT_STATUS.md, „Strategische Ausrichtung & Constraints") |
| Audit-Disziplin | Promotion-Gate (siehe unten); Forensic-Verdicts BLOCKING |
| Reports | Daily QA Report + Daily Reconcile |

---

## Promotion-Gate: Wann darf Account R → Account T?

Eine Strategie darf erst aus Account R nach Account T promoted werden, wenn
ALLE folgenden Kriterien erfüllt sind. Das ist eine **harte Checkliste** —
not satisfying ANY of them blocks promotion.

### Pflicht-Kriterien (alle müssen erfüllt sein)

- [ ] **Mindest-Live-Track-Record auf Account R:** ≥ 90 Tage Echtgeld-Paper
      oder echtem Account-R-Geld, NICHT nur Backtest.
- [ ] **Sharpe-Konsistenz Account R:** durchschnittlicher Sharpe der letzten
      30/60/90 Tage alle ≥ 1.0 (nicht durch einen einzelnen Glücks-Monat
      verzerrt).
- [ ] **Maximum Drawdown auf Account R:** < 20% über den gesamten R-Live-Run.
- [ ] **Out-of-Sample-Holdout statistisch signifikant:** `hold_out_leakage_test.py`
      Verdict ≠ `hold_out_negative_sharpe` UND ≠ `undefined`.
- [ ] **Bias-Audits laufen ohne kritische Befunde:**
      - `survivorship_bias_check.py` Verdict ≤ `medium` (oder Universe ist
        echte CRSP-Daten + Verdict `low`)
      - `out_of_regime_test.py` Verdict = `robust` ODER mindestens 1 Bear-
        Sample im Run (nicht 100% Bull)
      - `fill_model_audit.py` Verdict ≠ `high`
- [ ] **PIT-Property-Tests grün:** `tests/test_pit_strategy_features.py`
      und `tests/test_property_fsm_pit.py` alle pass.
- [ ] **Replay-Determinismus:** `tests/test_replay_determinism.py` alle pass.
- [ ] **Equity-Curve-Audit-DSR > Schwelle:** `equity_curve_audit.py`
      DSR-Schätzung > 1.0 (Bonferroni-conservativ mit n_strategies_tried = 10).
- [ ] **Kill-Switch + Pre-Trade-Gates aktiv und getestet:** mindestens 1
      simulierter Kill-Switch-Trigger im R-Run dokumentiert.
- [ ] **Position-Sizing auf Account T < Position-Sizing auf Account R:**
      mindestens halbierte Kelly-Fraktion ODER vol-target × 0.5.
- [ ] **Cost-Model gegen Account R Broker-Statements abgeglichen:** reale
      commission/spread/slippage liegt innerhalb der `cost_tiers.yaml`-Tiers
      (oder Tiers werden auf reale Werte angepasst — siehe
      `fill_model_audit.py`).

### Empfohlene Kriterien (sollten erfüllt sein, sind aber nicht hard-blocking)

- Operator-Tilt-Index (Tilt-Detection C2-073) zeigt im R-Run keine 3-Loss-Tag-
  Pause-Events
- Strategie hat ein eigenes `review_*.ipynb` (Adversarial Reviewer Pattern
  C2-051, in `research/`)
- Mindestens 1 Monthly QA Report ohne Discord-Failure-Notification
- Total Trade Count auf Account R ≥ 100 (Sample-Größe für statistische
  Aussagen)

---

## Demotion-Trigger: Wann muss Account T → zurück auf Account R?

Account T-Strategien werden **sofort** zurück auf Account R demoted, wenn:

- Daily Loss > 3% des Trading-Kapitals
- MaxDD über 30 Tage > 15%
- Kill-Switch ausgelöst (egal aus welchem Grund) — Demotion bis zur
  vollständigen Post-Mortem-Analyse
- Reconcile-Differenz zwischen Sim und Real > 50bps an einem Tag
- 5 aufeinanderfolgende Verlust-Tage

Demotion = Position-Sizing auf Account T sofort auf 0, neue Trades nur noch
auf Account R, Strategy-Lifecycle zurück auf Promotion-Gate-Checkliste.

---

## Workflow

1. **Strategie-Idee:** Forschungs-Notebook in `research/research_*.ipynb`,
   Adversarial Review in `research/review_*.ipynb`.
2. **Backtest:** `scripts/run_backtest_strategy.py` mit OOS-Split.
3. **Forensic-Audits laufen:**
   ```
   python scripts/forensic/equity_curve_audit.py
   python scripts/forensic/out_of_regime_test.py
   python scripts/forensic/survivorship_bias_check.py
   python scripts/forensic/hold_out_leakage_test.py
   python scripts/forensic/fill_model_audit.py
   ```
4. **Account-R-Live-Run:** ≥ 90 Tage Paper oder R-Geld.
5. **Promotion-Gate-Check:** `scripts/ops/check_promotion_gate.py` (siehe
   unten — Skript-Skelett).
6. **Wenn Promotion-Gate PASS:** Strategy in Account-T-Config kopieren mit
   halbiertem Position-Sizing.
7. **Daily-Monitoring auf Account T:** Daily QA Report + Daily Reconcile +
   Demotion-Trigger.

---

## Promotion-Gate-Skript

Ein automatisierter Check, der die Pflicht-Kriterien gegen den aktuellen
R-Run-Stand verifiziert: `scripts/ops/check_promotion_gate.py`.

Output: JSON + Markdown mit per-Kriterium `pass / fail / pending` Status
plus aggregiertem `promotion_verdict`: `ready` / `blocked / criteria
missing` / `demotion_required`.

Heute (2026-05-18) ist der Skript-Skeleton noch nicht produktiv-gewired —
das ist der nächste Sprint nach diesem Operations-Doc. Tracking als
KNOWN_ISSUES §8.x bzw. eigene Aufgabe.

---

## Audit-Bezug

| Audit-Item | Bezug |
|---|---|
| C2-074 | Two-Account-Setup (dieses Doc) |
| C2-065 | Robust-Kelly-Sizing (Account-T halve) |
| C2-066 | Vol-Targeting (Account-T conservative) |
| C2-073 | Tilt-Detection (Demotion-Trigger) |
| §8.7 | Bias-Audits (Promotion-Gate-Pflicht) |
| C2-050 | Replay-Test CI (Promotion-Gate-Pflicht) |
| C2-051 | Adversarial Reviewer (Promotion-Gate-recommended) |
| C3-030 | Equity-Curve-Audit (Promotion-Gate-Pflicht) |
| PROJEKT_STATUS.md (Constraints) | "kein Leverage" — gilt für beide Accounts |

---

## Risiko-Notizen

Dieses Doc beschreibt eine **Konvention**, kein automatisierter Enforcer.
Die Account-Trennung wirkt nur, wenn der Operator sich diszipliniert
daran hält:

- Trade-Setup auf Account T NIEMALS ohne vorherigen Promotion-Gate-Check.
- Bei Zweifel: zurück auf Account R, nicht promoten.
- Position-Sizing-Skript NICHT manuell von R-Settings auf T-Settings
  kopieren ohne den halbierungs-Schritt explizit zu machen.

Diese Disziplin ist die einzige Sicherung gegen "ich glaube an die Strategie,
ich erhöhe das Sizing" — der häufigste Grund für signifikante Drawdowns
in Quant-Privatkonten.
