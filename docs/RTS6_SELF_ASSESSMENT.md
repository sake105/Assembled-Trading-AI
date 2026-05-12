# RTS-6 Self-Assessment Skeleton

> Audit C2-077 + C4-086 — when this trading backend ever crosses the
> "Eigengeschäft" threshold into a commercial / professional setup, the
> regulator (BaFin / ESMA) expects an annual self-assessment per
> Regulatory Technical Standards 6 ("Organisational requirements for
> investment firms engaged in algorithmic trading").
>
> **Status today: not gewerblich. This document is the skeleton you'd
> fill in BEFORE that transition, kept ready so the answer is "yes, here
> it is" — not "give me four weeks to write it from scratch".**

---

## 0. Scope

- Firm: `<Solo / UG / GmbH>` — currently: solo private trader, not regulated.
- Algorithmic trading systems in use: `<list each one in §1>`.
- Trading venues: `<list>`.
- Date of assessment: `<YYYY-MM-DD>`.
- Reviewer: `<name / role>`.

## 1. Algorithm inventory (RTS-6 Art. 7)

Each algorithm carries: owner, version, sign-off date, last validation date.

| Algo name | Module path | Owner | Version | Sign-off date | Last validation | Status |
|---|---|---|---|---|---|---|
| trend_baseline | `src/assembled_core/strategies/trend_baseline.py` | `<name>` | `<git_sha-prefix>` | `<YYYY-MM-DD>` | `<YYYY-MM-DD>` | `paper / live / archived` |
| volatility_targeting | `src/erweiterung/strategies/volatility_targeting.py` (ERWEITERUNG branch) | `<name>` | n/a | n/a | n/a | research-only |
| multifactor_v2 | `<path>` | `<name>` | `<sha>` | `<date>` | `<date>` | `<status>` |
| ... | | | | | | |

Mark **status** as one of: `live`, `paper`, `research-only`, `archived`.
Only `live` and `paper` are in scope for the rest of this document.

## 2. Pre-trade controls (RTS-6 Art. 13)

| Control | Implementation | Test coverage | Owner |
|---|---|---|---|
| Maximum order size per symbol | `pre_trade_checks.PreTradeConfig.max_notional_per_symbol` | unit tests `tests/test_pre_trade_checks.py` | `<name>` |
| Maximum weight per symbol | `PreTradeConfig.max_weight_per_symbol` | dito | `<name>` |
| Sector exposure cap | `PreTradeConfig.max_sector_exposure` | dito | `<name>` |
| Region exposure cap | `PreTradeConfig.max_region_exposure` | dito | `<name>` |
| Gross exposure cap | `PreTradeConfig.max_gross_exposure` | dito | `<name>` |
| Turnover cap | `PreTradeConfig.turnover_cap` | dito | `<name>` |
| Drawdown auto-derisk | `PreTradeConfig.drawdown_threshold` + `de_risk_scale` | dito | `<name>` |
| Fat-finger check | `fat_finger_guard` | `tests/test_fat_finger_guard.py` | `<name>` |
| Kill-switch (single guard) | `execution/kill_switch.py` | `tests/test_property_fsm_pit.py` + audit-chain test | `<name>` |
| Gawande pre-trade gate | `pre_trade_checks.pre_trade_gate` (raises on first failure) | `tests/test_audit_additions.py` | `<name>` |

## 3. Post-trade / monitoring (RTS-6 Art. 17)

| Capability | Implementation | SLO |
|---|---|---|
| Real-time order/position monitor | FastAPI `/api/v1/monitoring/*` endpoints | < 5 s alert latency target (planned via Prometheus) |
| Reconciliation (ledger vs broker) | `accounting/reconciliation.evaluate_reconcile_slo` + JSONL audit log | warn @ 5 bps, fail @ 25 bps cash diff |
| Alert dispatch | `ops/alerting.AlertManager` — slack/telegram/email channels | cooldown configurable per rule |
| Audit log integrity | hash-chained JSONL (`kill_switch`), append-only fsync'd, `verify_audit_chain` API | tamper-evident |

## 4. Stress testing (RTS-6 Art. 9)

- Backtest gates: `quant_gates` in `configs/policy.yaml` block promotion
  when DSR ≤ 0, PSR < 0.95, PBO > 0.50, or MinTRL > observed track.
- Permutation test (`qa/metrics.permutation_p_value`, audit C2-016)
  with required p < 0.01 before promotion.
- Adversarial / OOS battery: TODO — wire crisis-injection (audit C2-017)
  + out-of-universe test (C2-018) before commercial.

## 5. Annual validation report

- Date of last full review: `<YYYY-MM-DD>`.
- Material changes since previous review: `<list>`.
- Findings & remediation:
  - `<finding 1>` — closed via `<commit / ticket>`.
  - `<finding 2>` — open, ETA `<date>`.
- Sign-off: `<name>`, `<role>`, `<date>`.

## 6. Conformance testing

- Broker(s) used: `<list>`.
- Conformance test plan: `<reference to broker's CTP / cert procedure>`.
- Last passed: `<YYYY-MM-DD>`.

## 7. Disaster recovery

- RTO / RPO targets: paper 30 min / 1 h; live (if applicable) 15 min / 5 min.
- Last DR drill: `<YYYY-MM-DD>`.
- Findings: `<see docs/FMEA.md row update>`.

## 8. Document control

- This skeleton lives at `docs/RTS6_SELF_ASSESSMENT.md`.
- Once activated (gewerblich), version it with semver: `vYYYY.Q-N.md`
  copies under `docs/regulatory/` and link the latest from here.

---

## What this skeleton is NOT

- Not legal advice — confirm with a regulated counsel before submitting
  anything to BaFin / ESMA.
- Not complete — sections 4 (adversarial OOS battery), 6 (broker
  conformance), and 7 (DR drill cadence) need real artefacts before this
  becomes a defensible document.
- Not a one-time deliverable — RTS-6 expects an **annual** refresh.
