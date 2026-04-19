# Archive: Intel Research 2026-Q2

Archived **2026-04-19** as part of the goofy-questing-crystal remediation plan (Phase C, T3.1–T3.8).

All files here were confirmed to have **0 productive call-sites** in `src/` and `scripts/` at time of archival.

## Decision per module

| File | Finding | Decision |
|------|---------|----------|
| intel/shock_correlation.py | V1 — 0 call-sites | ARCHIVE |
| intel/multichannel_propagation.py | V1 — 0 call-sites | ARCHIVE |
| intel/feedback_loops.py | V1 — 0 call-sites | ARCHIVE |
| intel/sensitivity_analysis.py | V3 — 0 call-sites | ARCHIVE |
| intel/escalation_tracker.py | V2 — 0 external call-sites | ARCHIVE |
| intel/escalation_model.py | V2 — only used by escalation_tracker | ARCHIVE |
| intel/scenario_trees.py | V2 — 0 call-sites | ARCHIVE |
| intel/wargaming.py | V2 — 0 call-sites | ARCHIVE |
| intel/hegemonic_dynamics.py | V6 — 0 imports | ARCHIVE |
| intel/structural_cycles.py | V6 — 0 imports | ARCHIVE |
| features/earnings_call_nlp.py | V7 — 0 call-sites | ARCHIVE |
| features/analyst_features.py | V7 — 0 call-sites | ARCHIVE |
| events/disclosures/evidence.py | N6 dead code | ARCHIVE |
| events/news/fetch_acled.py | N5 pure stub | ARCHIVE |

## Files NOT archived (confirmed active)

- `intel/shock_propagation.py` — used in `scripts/run_intel_cycle.py`
- `intel/market_confirmation.py` — used in `scripts/run_intel_cycle.py`
- `intel/bayesian_confidence.py` — wired in shadow mode (T2.8)
- `intel/crisis_alpha_worker.py` — used by `scripts/run_crisis_alpha_worker.py` (deprecated, see T3.6)
- `events/news/entities.py` — imported in `events/news/normalize.py`

## To restore

```bash
git mv archive/intel_research_2026q2/<path> src/assembled_core/<path>
```
