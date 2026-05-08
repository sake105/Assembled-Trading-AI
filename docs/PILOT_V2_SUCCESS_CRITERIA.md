# Pilot v2 — Success Criteria (Item 148)

**Pre-committed before pilot start. Do not change during the pilot run.**  
Source of truth: `output/pilot/pilot_manifest.json` → `success_criteria`

---

## Evaluation Timeline

- **After day 14:** Check rolling Sharpe. If < 0.5 for 3 consecutive days → hard stop.
- **After day 30:** Full verdict using the criteria below.
- **Unclear zone:** Extend pilot by 14 days, then final verdict.

---

## Success (GO — proceed to live trading)

All three criteria must be met:

| Metric | Threshold | Notes |
|--------|-----------|-------|
| CAGR (annualised) | **> 20%** | From pilot equity curve |
| Sharpe ratio | **> 1.5** | Annualised, daily returns |
| Max drawdown | **> -10%** | Less than 10% from peak |
| Unexpected crashes | **0** | rc != 0 days count as crash |

Action: Activate live trading with $5k initial capital. Document go/no-go rationale in `docs/learning/`.

---

## Failure (NO-GO — halt and review)

Any one of these triggers failure:

| Metric | Threshold | Notes |
|--------|-----------|-------|
| CAGR | **< 5%** | Insufficient edge |
| Sharpe ratio | **< 0.5** | Noise-dominated returns |
| Max drawdown | **> -15%** | Unacceptable risk |

Action: **14 days offline** + strategy review before next pilot.
Document failure mode in `docs/learning/incidents/`.
Do not restart pilot without root-cause analysis and documented fix.

---

## Unclear (extend)

If all metrics fall between success and failure thresholds:

- Extend pilot by **14 additional days** (to day 44).
- After day 44: apply same criteria. No further extensions.
- If still unclear at day 44 → treat as NO-GO.

---

## Pre-Commitment Statement

These criteria were set before pilot start on 2026-05-07.
They reflect the backtest baseline:
- OOS 2025-2026 CAGR 58.4% / Sharpe 3.78 / MDD -5.19% (survivorship-biased, treat as upper bound)
- Realistic expectation: CAGR 20-30%, Sharpe 1.5-3.0, MDD -5% to -12%

The success bar (CAGR > 20%, Sharpe > 1.5) is deliberately set well below the backtest
to account for:
- Survivorship bias in backtest universe
- Live execution friction
- Regime shifts
- News/macro factors not in backtest

**If you are considering lowering the bar mid-pilot: don't. That is p-hacking.**
