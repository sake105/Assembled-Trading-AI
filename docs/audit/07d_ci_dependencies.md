# Audit 07d — CI/CD Workflows & Dependency / Supply-Chain Surface

Round 3, read-only audit. Scope: `.github/workflows/*`, `pyproject.toml`, `requirements.txt`,
and the CI linchpin scripts (`scripts/dev/run_checks.py`, `scripts/dev/release_sprint13.py`,
`scripts/release_gate_walk_forward.py`). **Nothing changed.** Findings prefix `CI-`.

Method: every finding cites file:line + evidence. "Verified at source" = read in the YAML/script.
"Inferred" = reasoned from code without execution. Quick read-only commands run are listed at the
bottom under "What I actually ran".

Severity legend: KRITISCH / HOCH / MITTEL / NIEDRIG.

---

## Executive verdict

"7 workflows green" is **partly protection, partly theater.** Two genuine blocking test gates exist
and they collect a real fraction of the suite (`ci.yml` 7511 tests, `backend-ci.yml` fast+regression
2836 tests). But the three Windows gates named after governance domains
(`accounting-ci`, `evidence-pack-ci`, `release-gate-ci`) and the release-gate quant checks are
**much weaker than their names imply**: they run hand-picked smoke/doc test lists (≈8–12 files each),
the statistical release gate (E3/E4 Walk-Forward + Deflated Sharpe) **never blocks** (runs without
`--enforce` on synthetic random-walk data), and the "accounting" deep check is `continue-on-error`.
No workflow uses `needs:`, so the workflows are independent — a green check on one says nothing about
another. There is real CVE-ignore drift (16 ignores, several open-ended) and concrete
pyproject-vs-requirements pin drift beyond the two documented in rule 40.

---

## POSITIVE confirmations (gates that genuinely block)

- **CI-POS-1** (`ci.yml:67-71`): `pytest -m "not advanced and not slow" --maxfail=3` is a real
  blocking gate. Verified collect-only: **7511 tests** of 8059 default-collected. Runs on
  ubuntu + windows matrix. This is the strongest single gate in the repo.
- **CI-POS-2** (`backend-ci.yml:61-69`): `fast and not slow` (2723 tests) + `regression and not slow`
  (113 tests) both `--maxfail=1`, blocking, on py3.10 + py3.11. Real coverage, not a stub.
- **CI-POS-3** (`backend-ci.yml:71-118`): `pip-audit` runs as a **blocking** scan; any NEW
  (non-ignored) CVE fails CI. `bandit -r src/assembled_core --severity-level medium` (line 120-126)
  also blocking. Both genuinely gate.
- **CI-POS-4** (`secrets-scan.yml`): two independent jobs — `gitleaks detect ... --exit-code 1`
  (line 32-38) and `detect-secrets` baseline diff (line 63-72 → `scripts/detect_secrets_baseline_diff.py`
  returns 1 on any new finding). Both block. Solid.
- **CI-POS-5** (`evidence-pack-ci.yml:42-49`, `ops-evidence-ci.yml:43-52`): the primary preset step
  propagates `$LASTEXITCODE` via `exit $exitCode` — these DO block on the preset they name.
- **CI-POS-6** (collection health): `pytest --collect-only -q` completes with **8059 tests, 0
  collection errors** (the single "error" grep hit is the filename `test_errors_log.py`). Memory
  baseline confirmed/raised. No stub-collection rot.
- **CI-POS-7** (`pyproject.toml:228-235`): `filterwarnings` promotes `FutureWarning`/`DeprecationWarning`
  from `src.assembled_core.*` to errors (targeted), so first-party deprecations DO fail tests even
  though the blanket `-W error` was removed.

---

## Findings

### CI-001 — Release-gate quant checks (E3/E4) never block — HOCH

**Verified at source.** `release-gate-ci.yml:67-103` runs `walk-forward-gate` calling
`scripts/release_gate_walk_forward.py --verbose` (line 95). That script
(`release_gate_walk_forward.py:296-300`) returns `0` on gate miss unless `--enforce` is passed, and
the workflow **never passes `--enforce`**. Worse, the gate runs on **synthetic random-walk prices**
(`_synthetic_prices`, line 47-77, `seed=42`) — by construction there is no alpha, so E3
(`oos_sharpe>=0.3`) / E4 (`DSR>=0.5`) are structurally un-passable and explicitly non-blocking
(self-documented grace period through 2026-07-01, line 67-76). Net: a workflow named "Release Gate"
ships a statistical-validity gate that can never turn the job red. Visibility only. The grace-period
expiry (2026-07-01) is a future date — until then this is a no-op gate by design, but the **naming
implies enforcement that does not exist.**

### CI-002 — `release_sprint13` "release gate" runs only doc/smoke tests, not the trading suite — HOCH

**Verified at source.** The blocking job `windows-release-gate` (`release-gate-ci.yml:40-49`) runs
`scripts/dev/release_sprint13.py`, which (`release_sprint13.py:51-54`) calls two presets:
`release_sprint13` and `evidence_pack`. The `release_sprint13` preset
(`run_checks.py:416-432`) is a list of **12 test files** that are almost entirely
docs/CLI-smoke/inventory checks: `test_ci_workflows_inventory_smoke.py`, `test_docs_sanity_sprint13.py`,
`test_docs_links_smoke.py`, `test_release_notes_header_smoke.py`, evidence-pack CLI smokes, etc. None
of risk / execution / portfolio / accounting numeric correctness is in this list. So the
on-push-to-main "release gate" verifies that **docs links and evidence-pack CLI schemas are stable**,
not that the trading system is correct. The real test coverage lives in `ci.yml` / `backend-ci.yml`,
which is fine — but the gate's *name* oversells it.

### CI-003 — `test_ci_workflows_inventory_smoke.py` only checks file existence — MITTEL

**Verified at source.** `tests/test_ci_workflows_inventory_smoke.py:26-39`: the "inventory" test that
the release gate (CI-002) relies on asserts only `path.exists()` and `path.is_file()` for 4 workflow
files. It does **no YAML parse, no job/step validation, no selector check**. A workflow could be
gutted to a no-op and this test still passes. It is the only structural workflow check in the suite
and it verifies nothing about workflow *behavior*.

### CI-004 — `accounting-ci.yml` deep accounting check is non-blocking — MITTEL

**Verified at source.** `accounting-ci.yml:42-49` blocks on the `broker_snapshot` preset (good), but
the step named "Run accounting preset checks" (line 61-64) is `continue-on-error: true` with
`|| echo "...skipped or failed (non-blocking)"`. So in a workflow titled **Accounting CI**, the
`accounting` preset itself (which compiles `src/assembled_core/accounting/` + orchestrator + reconcile
tests, `run_checks.py:286-294 / 358-366`) **cannot fail the job**. The job is green as long as
`broker_snapshot` passes. The domain headline overstates what blocks.

### CI-005 — `evidence-pack-ci.yml` optional presets masked, same pattern — NIEDRIG

**Verified at source.** `evidence-pack-ci.yml:61-69`: `broker_snapshot` and `accounting` presets both
`continue-on-error: true` with `|| echo`. Acceptable (the named `evidence_pack` preset does block at
line 42-49), but it means two of three steps in this Windows job are decorative.

### CI-006 — `mypy` is fully neutered (double-masked) — MITTEL

**Verified at source.** `backend-ci.yml:138-142`: the mypy step has BOTH `|| true` at the end of the
command AND `continue-on-error: true`. Type checking can never affect CI status. The comment says
"optional for now" — honest, but it means the strict-typing overrides declared in
`pyproject.toml:197-207` (kill_switch, order_lifecycle, api.auth, retry, clock_drift, reproducibility
must stay typed) are **aspirational only** and not enforced anywhere. A regression that breaks typing
on the safety-critical surfaces ships silently.

### CI-007 — Open-ended / unbounded CVE ignores in pip-audit — HOCH

**Verified at source.** `backend-ci.yml:101-117` ignores **16 vulnerabilities**. Several are bounded
with a plausible rationale (toolchain-only pip CVEs, docs tools). But the ignores are **open-ended**:
there is no expiry date, no tracking issue link, no "re-check by" marker on any of them. Two are
flagged in-comment as deferred to "Sprint 4 dep refresh" (`CVE-2026-25645`, `CVE-2024-47081` on
`requests`, line 82-84) but `requests` is still pinned `==2.32.3` in `requirements.txt:16` — so the
deferral has not happened and there is no mechanism to surface that it is overdue. `--ignore-vuln`
without a date is a permanent suppression; "16 ignores" will only grow. Recommend (not applied):
move ignores to a dated `pip-audit` config with review cadence. Note also `--skip-editable` (line 101)
means the local package itself is not audited — acceptable (it has no PyPI provenance) but worth naming.

### CI-008 — Dependency drift pyproject (ranges) vs requirements (pins) beyond the documented two — HOCH

**Verified at source** (`pyproject.toml:27-57` vs `requirements.txt:11-83`). Rule 40 documents
`pandas`/`numpy` drift. The full picture is broader. Every runtime dep has a floor in pyproject and a
hard pin in requirements; CI installs from requirements (pins), local `pip install -e .` resolves from
ranges. Concrete divergences where the local-resolved version can differ from the CI-pinned version:

| Package | pyproject range | requirements pin | Drift risk |
|---|---|---|---|
| pandas | `>=2.0.0` | `==2.2.3` | documented (rule 40 said 2.3.3 — **pin has since moved to 2.2.3**, rule 40 is now stale) |
| numpy | `>=1.24.0` | `==2.2.6` | documented (rule 40 said 2.3.3 — **pin moved to 2.2.6**, rule 40 stale; numpy 2.x is an API break vs the 1.24 floor — local install could pull numpy 1.x and behave differently) |
| pyarrow | `>=10.0.0` | `==21.0.0` | wide gap (10 → 21); local could resolve a far older pyarrow |
| matplotlib | `>=3.7.0` | `==3.10.6` | moderate |
| fastapi | `>=0.100.0` | `==0.122.0` | wide; Starlette behavior changes between these |
| pydantic | `>=2.0.0` | `==2.12.5` | v2 API stable but validation edge cases differ |
| arch | `>=6.0.0` | `==8.0.0` | **major-version gap** (6 → 8); GARCH API changed between arch 6 and 8 — local install on the floor would behave differently from CI |
| exchange-calendars | `>=4.5.0` | `==4.12` | calendar data differs by version |
| pandera | `>=0.21.0` | `==0.31.1` | schema-validation behavior drift |

Additionally **scipy/sklearn are NOT pinned in requirements** (`requirements.txt:47-48`:
`scipy>=1.10.0`, `scikit-learn>=1.3.0`) — these are ranges *inside the pin file*, intentionally
(comment line 45-46: py3.10 needs older builds than py3.11). Consequence: backend-ci's py3.10 and
py3.11 matrix legs **install different scipy/sklearn versions**, so a numeric test that passes on one
leg can differ on the other. This is a real (documented-intent) Python-version matrix gap, classified
HOCH because scipy/sklearn feed quant gates.

`statsmodels==0.14.6` (requirements:83) and `pandas-market-calendars==4.6.1` (requirements:78) are
pinned but have **no entry in pyproject `dependencies`** — they are pinned-only-in-requirements
(arch/statsmodels were added per the line 80-82 comment). So a local `pip install -e .` does **not**
install statsmodels at all unless requirements is used; modules importing statsmodels
(signals/pairs_*, recession_probability, residual_momentum) would `ImportError` locally. Drift in the
other direction.

### CI-009 — Windows-vs-Ubuntu install path divergence (ad-hoc pip lists vs requirements.txt) — MITTEL

**Verified at source.** The Windows governance workflows do **not** install from `requirements.txt`.
They `pip install <hand-typed list>`:
- `accounting-ci.yml:40`: `pip install pandas pyarrow fastparquet pytest ruff pydantic pydantic-settings pyyaml exchange_calendars scipy statsmodels scikit-learn` — **no version pins at all**, and **no numpy/arch/pandera/matplotlib**.
- `evidence-pack-ci.yml:40`, `ops-evidence-ci.yml:41`, `release-gate-ci.yml:31`: similar unpinned hand lists, each slightly different (release-gate omits fastparquet/pyyaml).

So these Windows jobs run on **whatever latest versions PyPI serves that day**, fully decoupled from
both the pins (requirements.txt) and the ranges (pyproject). A dependency that breaks the
accounting/evidence presets would be caught here only by luck of timing, and the same code is tested
against *different* dependency versions than backend-ci/ci.yml. This is exactly the local-vs-CI drift
rule 40 warns about, institutionalized into the workflow matrix.

### CI-010 — No `needs:` anywhere — a green check never implies an upstream passed — MITTEL

**Inferred from full read of all 21 workflows.** Not a single workflow declares `needs:` between
jobs, and there is no orchestrating "required checks" gate visible in-repo (branch protection is a
GitHub-side setting not in the tree, so cannot be verified here). Within multi-job workflows
(`release-gate-ci.yml` windows-release-gate + walk-forward-gate; `secrets-scan.yml` gitleaks +
detect-secrets; `weekly-drills.yml` two drills) the jobs run independently and in parallel. A green
"Release Gate CI" badge means *one of its two jobs* passed and the other (walk-forward) can never
fail anyway (CI-001). Whether the 7 workflows are *actually required* for merge is a branch-protection
question that must be checked in GitHub settings — **not verifiable from the repo**.

### CI-011 — Scheduled "producer" workflows can silently no-op and still commit nothing — NIEDRIG/MITTEL

**Verified at source.** The intel-refresh crons (`news-worker-ci.yml`, `disclosures-worker-ci.yml`,
`earnings-calendar-refresh.yml`, `signal-decay-update.yml`) are the freshness guarantee for live
gates (each header documents that a stale file makes a downstream gate "go dark"). But their failure
handling is soft:
- `news-worker-ci.yml:90-114` / `disclosures-worker-ci.yml:88-112`: if no artifact is produced the
  commit step prints "nothing to commit" and **exits 0**. A persistent producer failure (dead API
  key, upstream 403) leaves the consumed file aging out while the workflow stays green. The "Inspect
  output" step (line 55-76) only `print('[WARN] ...')`, never exits non-zero.
- `signal-decay-update.yml:64-71` is better — `if-no-files-found: error` on upload makes a missing
  artifact fail. `news`/`disclosures`/`earnings` use `warn`, so a vanished artifact is silent.
- `daily-paper-reconcile.yml:41`: `run_reconcile_worker.py --dry-run || true` — reconciliation
  failures are swallowed. `daily-diagnostics.yml:19-21` and `weekly-research.yml:19-23`: every step
  ends `|| echo "...skipped/failed"`, so drift/leakage/walk-forward/GARCH research checks **cannot
  fail the job**. These are visibility-only by design, but a reader trusting a green badge would be
  misled about whether leakage analysis actually ran clean.

### CI-012 — `repo-health.yml` adversarial-reviewer gate is a structural no-op today — NIEDRIG

**Verified at source.** `repo-health.yml:48-53`: the "Adversarial Reviewer Notebook Pattern" gate runs
`check_adversarial_reviewer_pattern.py`, and the in-file comment admits "Today: zero research_*.ipynb
exist, so script is a no-op gate." Honest, but it is a gate that passes vacuously and would only
acquire teeth if notebooks are ever added. Low risk; documented.

### CI-013 — `nightly-sync.yml` is fully disabled — informational — NIEDRIG

**Verified at source.** `nightly-sync.yml:33-36`: the only step is `echo "Heartbeat commits
disabled."`. The workflow is intentionally a stub (C6 audit, ~200 noise commits removed). Not a
defect — but it is a scheduled workflow that does literally nothing and still appears in the workflow
list, inflating the apparent CI surface.

### CI-014 — `pip` caching keyed on pyproject+requirements hash can mask range-resolved drift — NIEDRIG

**Inferred.** `backend-ci.yml:33-39`, `accounting-ci.yml:29-35`, etc. cache `~/.cache/pip` keyed on
`hashFiles('**/pyproject.toml','**/requirements.txt')`. For the requirements-installed jobs this is
fine (pins are deterministic). For `ci.yml:45-48` which installs `pip install -e ".[dev]"` (ranges),
a cached wheel set means a new upstream release that *would* be range-resolved on a cold cache is not
picked up until the cache key changes — so "it passed in CI" can reflect a stale resolved set. Minor,
but it is a vector by which local (fresh resolve) and CI (cached resolve) diverge.

### CI-015 — Secrets surface in workflows — informational, no leak found — NIEDRIG

**Verified at source.** Secrets used: `ALPACA_API_KEY`/`ALPACA_API_SECRET` (paper-trading-ci,
daily-paper-reconcile), `DISCORD_WEBHOOK` (paper-trading-ci, fail-drill, repo-health). All consumed
via `${{ secrets.* }}` env injection — none echoed, none written to artifacts. `paper-trading-ci.yml`
hits the **paper** Alpaca endpoint only (`ALPACA_BASE_URL: https://paper-api.alpaca.markets`,
hardcoded line 91 etc.). `permissions: contents: write` is granted to the producer crons and
paper-trading-ci (they push refreshed artifacts) — broad but justified by the commit-back pattern.
No secret-handling defect found.

---

## Summary table

| ID | Severity | One-line |
|---|---|---|
| CI-001 | HOCH | Release-gate Walk-Forward E3/E4 runs without `--enforce` on synthetic data → never blocks |
| CI-002 | HOCH | "Release gate" preset = 12 doc/smoke test files, no risk/exec/accounting correctness |
| CI-003 | MITTEL | Workflow "inventory" test only checks file existence, no behavior |
| CI-004 | MITTEL | accounting-ci's `accounting` preset is `continue-on-error` → cannot fail the job |
| CI-005 | NIEDRIG | evidence-pack-ci has 2 of 3 steps `continue-on-error` |
| CI-006 | MITTEL | mypy double-masked (`|| true` + `continue-on-error`); strict-typed safety surfaces unenforced |
| CI-007 | HOCH | 16 pip-audit CVE ignores, open-ended, no expiry; `requests` "Sprint 4" deferral overdue |
| CI-008 | HOCH | Pin-vs-range drift beyond rule 40: arch 6→8 major gap, numpy 1.24-floor vs 2.2.6 pin, scipy/sklearn unpinned across py3.10/3.11 matrix; rule 40 pandas/numpy numbers now stale |
| CI-009 | MITTEL | Windows governance jobs install ad-hoc UNPINNED pip lists, not requirements.txt → test vs different deps daily |
| CI-010 | MITTEL | No `needs:` anywhere; green check ≠ upstream passed; required-checks is branch-protection (unverifiable in-repo) |
| CI-011 | MITTEL | Producer crons (news/disclosures/earnings/reconcile/diagnostics) soft-fail → stale-file freshness gates can go dark while green |
| CI-012 | NIEDRIG | repo-health adversarial-reviewer gate vacuously passes (zero notebooks) |
| CI-013 | NIEDRIG | nightly-sync is a disabled stub |
| CI-014 | NIEDRIG | pip cache on `-e ".[dev]"` job can mask range-resolved drift |
| CI-015 | NIEDRIG | Secrets surface reviewed — paper endpoint only, no leak; `contents: write` broad but justified |

---

## What I actually ran (read-only)

- `pytest -m "fast and not slow" --collect-only -q` → summed per-file = **2723** tests collected.
- `pytest -m "regression and not slow" --collect-only -q` → **113** tests.
- `pytest -m "not advanced and not slow" --collect-only -q` (ci.yml selector) → **7511** tests.
- `pytest --collect-only -q` (default) → **8059** tests, **0 collection errors** (lone "error" grep
  hit = filename `test_errors_log.py`).
- Static reads of all 21 workflow YAMLs, `pyproject.toml`, `requirements.txt`, `run_checks.py`,
  `release_sprint13.py`, `release_gate_walk_forward.py`, `detect_secrets_baseline_diff.py`,
  `tests/conftest.py` (marker auto-alias), `tests/test_ci_workflows_inventory_smoke.py`.

No file was modified. No workflow was triggered.

## Caveats / unconfirmed

- **Branch-protection / required-checks status is GitHub-side and not in the repo** — whether the
  7 workflows are *enforced* as merge-blocking (CI-010) cannot be verified from the tree. The whole
  "green = protected" claim ultimately hinges on this off-repo setting.
- Marker counts (2723/113/7511/8059) are from the local environment's installed deps; CI's
  pinned/range-resolved environment could collect a slightly different count (esp. scipy/sklearn
  skip behavior, CI-008). Order of magnitude is reliable; exact numbers are local.
- `run_checks.py` preset *pytest* lists were read at source; I did not execute the presets, so
  whether each named test file currently passes is not asserted here — only what the gate *selects*.
