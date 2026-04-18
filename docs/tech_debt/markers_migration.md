# Tech Debt — Pytest Marker Consolidation

**Opened:** 2026-04-18
**Sunset target:** 2026-07-01
**Tracked P0:** A9 (Deep Run v2, 2026-04-18)
**Owner:** unassigned
**Status:** Open (legacy markers aliased, not yet removed)

---

## What this is

Over M4–M13 the test suite accumulated **13 overlapping phase markers**
(`phase4`, `phase6`, `phase7`, `phase8`, `phase9`, `phase10`, `phase11`,
`phase12`, `phase13`, `phase_zero`, `phase_speed`, `phase_realism`,
`phase_depth`) plus the unrelated pytest defaults (`slow`, `unit`,
`integration`, `smoke`, `regression`, …).

Selecting a meaningful subset from CI has become painful — `-m phase12`
picks up a mix of unit and integration tests, and `-m phase_zero` has no
relationship to `-m phase12` even though both are "the canonical suite we
care about today". Newcomers pick a marker by mimicry, not meaning.

A9 collapses this onto **four canonical markers** that describe the test's
operational role, not the sprint it was authored in:

| Canonical marker | Intent |
|---|---|
| `fast`         | Unit-level tests. Must run in ≪ 1s. No external I/O, no heavy fixtures. |
| `integration`  | Multi-module flows. May hit disk, may compose several subsystems. |
| `regression`   | Parity / invariant / gate tests. Exist to catch behaviour drift. |
| `smoke`        | End-to-end smoke runs. Longest; closest to real pipeline. |

## Current state (2026-04-18)

- `pytest.ini` declares both the canonical four and the 13 legacy phase
  markers (so `--strict-markers` does not reject either).
- `tests/conftest.py::pytest_collection_modifyitems` **aliases** each
  legacy marker onto one canonical marker at collection time:

  ```
  phase4|6|7|8|9|10|11|12|13                  → fast
  phase_zero | phase_speed | phase_realism | phase_depth → regression
  ```

  Every test retains its old marker **and** gains the canonical one, so
  `-m phase12` and `-m fast` both work today. This is deliberate — a
  one-shot rename of 1000+ test files would be a CI-risk-for-no-benefit
  change. Aliasing lets us migrate selections module-by-module while the
  suite stays green.

## Definition of done (sunset criteria)

To close this tech-debt entry on or before **2026-07-01**:

1. All CI workflow selectors (`.github/workflows/*.yml`) reference only
   the canonical four (`-m fast`, `-m "fast or regression"`, …). No
   workflow still says `-m phase12`.
2. All local developer docs (`docs/runbooks/*`, `docs/cursor/*`,
   `CLAUDE.md`) use the canonical four.
3. Remove the `phaseN` and `phase_*` entries from `pytest.ini::markers`.
4. Delete `_LEGACY_MARKER_ALIASES` and `pytest_collection_modifyitems` in
   `tests/conftest.py`.
5. Run `rg -n "pytest.mark.phase" tests/` — expected: 0 hits. Replace
   each `@pytest.mark.phaseN` in test files with the canonical marker.

Step 5 is the only step that touches test files. Doing it *last* means
the alias keeps CI green until the final sweep, and that sweep is a
mechanical rename, not a behavioural change.

## What to do if 2026-07-01 arrives without closure

Two options, pick one before the sunset:

- **(a) Finish the sweep** per the DoD above. Preferred — the alias
  layer is dead weight the day after migration.
- **(b) Re-open this entry** with a new sunset date and a concrete
  reason. The canonical-four scheme is not load-bearing in any way
  except documentation clarity, so a scope decision to keep the phase
  markers is acceptable — but not accidentally, only deliberately.

Not acceptable: silently extending. The whole point of A9 was that
"we'll clean up the markers later" had already stretched across four
milestones.

## Related

- P0 A8 `parity_gap.md` — parity regression uses `regression` (via the
  `phase_zero` alias). Any future unification there should land on the
  canonical marker, not a new phase-style one.
- P1 A13 alert-drill — not test-suite-adjacent; listed for cross-ref.
