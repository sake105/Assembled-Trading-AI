# Crisis Alpha Specification (v1)

**Status:** Active  
**Version:** 1.0 (2026-04-19, closes ROADMAP_STATE:486, M5-T13)

---

## Purpose

Crisis Alpha detects elevated geopolitical risk, assesses evidence quality,
and provides a risk posture recommendation to the trading loop.

It does NOT directly generate orders. It produces a structured result that
the trading cycle consumes (shadow-only by default, T4.1).

---

## Components

```
GeoTrigger scoring (intel/geo_trigger.py)
        ↓
News Clustering (intel/news_cluster.py)
        ↓
Evidence Engine (events/evidence_engine.py)
        ↓ grade: A/B/C/D
Evidence Grade Gate (events/crisis_alpha/gates.py)
        ↓ default-deny (T2.5)
Crisis Alpha Pipeline (events/crisis_alpha/pipeline.py)
        ↓
CrisisAlphaResult
```

---

## Evidence Grades

| Grade | Criteria | Gate |
|-------|---------|------|
| **A** | ≥2 Tier-A sources OR ≥3 independent Tier-B, low misinfo | ACTIVE allowed |
| **B** | ≥1 Tier-A OR ≥2 independent Tier-B, acceptable misinfo | ACTIVE allowed |
| **C** | ≥1 source but below B threshold | ACTIVE blocked |
| **D** | No qualifying evidence or misinfo > 0.70 | ACTIVE blocked |

Policy: `evidence_engine.require_grade_for_active: "B"` (minimum B to allow activation).

Missing or unknown grade → **default-deny** (T2.5, fixed 2026-04-19).

---

## Crisis States (v1)

| State | Geo Score | Description |
|-------|-----------|-------------|
| NORMAL | 0 | No elevated risk |
| WATCH | 1 | Elevated, monitoring |
| ACTIVE | 2 | Confirmed risk; hedges may activate |
| COOLDOWN | 1→0 | Recovering; reduced exposure |
| PAUSE | ≥3 | Full stop; all new orders blocked (T4.3) |

---

## Pre-Trade Kill Switch (T4.3)

`execution/risk_controls.py::check_crisis_alpha_kill_switch(ctx)`:
- PAUSE state → all new orders BLOCKED
- All other states → ALLOWED
- Gate: `policy.intel.crisis_alpha.enabled`; disabled → no check

---

## Trading Loop Wiring (T4.1)

In `trading_cycle.py`, after signal generation, before order execution:

1. Load `policy.intel.crisis_alpha.enabled` (default: `false`)
2. If enabled: call `run_crisis_alpha_pipeline(ctx)`
3. If `shadow_only: true`: log result, no order impact
4. If `shadow_only: false`: result influences basket orders (Step 3, not yet implemented)

---

## Policy Configuration

```yaml
intel:
  crisis_alpha:
    enabled: false         # flip to true to activate
    shadow_only: true      # flip to false only after Step 3 review
```

---

## Tests

63 tests in `tests/events/crisis_alpha/` covering:
- Context construction
- Evidence gate (default-deny)
- Grade hierarchy (A > B > C > D)
- Pipeline execution
- State transitions

---

## Open Work

- T4.1 Step 3: wire `CrisisAlphaResult.target_weights` to order generation
- T4.3 shadow_only=false: user review gate required
- X2 Entity-Linker: enables symbol-level crisis impact (not portfolio-wide)
