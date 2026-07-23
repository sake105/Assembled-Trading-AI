export const meta = {
  name: 'wf-deps-urllib3-mypy-prereqs',
  description: 'Non-protected dependency/CI hygiene: (A) urllib3 2.5.0 -> 2.6.3 security bump clearing 3 unwaived CVEs that will block CIs pip-audit gate; (B) mypy prereqs — pin mypy + add type stubs + ignore_missing_imports overrides for optional research libs (clears ~50-60 noise errors so the real surface is honest, prerequisite for the later staged blocking gate). requirements.txt / requirements.lock / pyproject.toml — none protected. Do NOT touch .github/workflows. Do NOT pip-freeze-regen the lock.',
  phases: [
    { title: 'Implement' },
    { title: 'Review' },
    { title: 'Audit' },
  ],
}

phase('Implement')

const IMPL_SPEC = [
  'You are implementing two non-protected dependency/CI-hygiene fixes in Assembled-Trading-AI (repo F:\\Python_Projekt\\Aktiengeruest, Windows, .venv\\Scripts\\python.exe). Files you MAY edit: requirements.txt, requirements.lock, pyproject.toml. You may NOT touch .github/workflows (protected) or any src/ path. CI is hochsensibel — be precise. Confirm each finding against current files first.',
  '',
  '=== PART A — urllib3 security bump (deterministic) ===',
  'Verified by a prior read-only pip-audit (PYTHONUTF8=1, pip-audit 2.10.0): urllib3 2.5.0 (current, requirements.lock) has THREE CVEs NOT in backend-ci.yml --ignore-vuln list — CVE-2025-66418 + CVE-2025-66471 (fixed in 2.6.0) and CVE-2026-21441 (fixed in 2.6.3). All three are cleared by urllib3 2.6.3. backend-ci.yml only waives the unrelated PYSEC-2026-141/142. CI installs deps from requirements.txt (pip install -r requirements.txt) + pip-audit scans that env, so the fixed urllib3 must be REACHABLE from requirements.txt.',
  'CHANGES:',
  '1. requirements.lock: bump `urllib3==2.5.0` -> `urllib3==2.6.3` (surgical single line). Update the lock header date/note lines if they carry a date. Do NOT change any other pin in the lock.',
  '2. requirements.txt: urllib3 is currently TRANSITIVE (not directly pinned). Add an explicit `urllib3==2.6.3` pin (near the other HTTP/network deps, e.g. by requests) so the CI install + pip-audit deterministically get the fixed version. (requirements.txt already carries hand-maintained pins incl. the FU-4b requests==2.32.4, so an explicit pin here is consistent with current practice.)',
  '3. pyproject.toml: add `urllib3>=2.6.3` to the [project].dependencies list (source-of-truth security floor) if urllib3 is not already constrained there.',
  'VERIFY A (you MAY install into the venv for evidence — a urllib3 patch bump is low-risk and aligns local with the new pin): `.venv\\Scripts\\python.exe -m pip install "urllib3==2.6.3"`; confirm no resolver conflict with the pinned requests==2.32.4 / alpaca-py / yfinance / polygon-api-client (requests 2.32.4 allows urllib3 <3,>=1.21.1 — should be fine; REPORT any cap conflict). Then under PYTHONUTF8=1 run `.venv\\Scripts\\python.exe -m pip_audit --desc --skip-editable` and confirm the THREE urllib3 CVEs no longer appear. Record the verbatim urllib3 audit lines (or their absence). Do NOT do a pip freeze regen of the lock — it would smuggle numpy/pandas/pyarrow drift (lock already shows numpy 2.3.3 vs txt 2.2.6 etc.) into the numeric-stack pins. Surgical single-line edit ONLY.',
  '',
  '=== PART B — mypy prereqs (deterministic; prerequisite for the later staged blocking gate) ===',
  'backend-ci.yml runs mypy NON-BLOCKING (continue-on-error: true) over `src/assembled_core/{data,features,signals,execution,portfolio}`. Local mypy (1.19.0) reports 190 errors / 86 files; ~36 are import-untyped (stub-installable: requests/yaml/pytz dominate) and ~33 are import-not-found (uninstalled optional research libs). Clearing this noise + pinning mypy makes the real surface honest and the future gate deterministic. This batch does NOT flip the gate (that is a separate protected .github batch) — it only reduces noise + pins mypy.',
  'CHANGES:',
  '1. requirements.txt: mypy is currently UNPINNED in the CI install path (rides the dev-extra range). Add an explicit `mypy==1.19.0` pin (the locally-measured version) so the gate is deterministic and will not break on upstream mypy releases.',
  '2. requirements.txt: add the missing stub packages that clear import-untyped — at minimum `types-requests`, `types-PyYAML`, `types-pytz` (confirm the exact stub package names + a compatible version by checking what mypy reports as import-untyped).',
  '3. pyproject.toml [tool.mypy] overrides: add `ignore_missing_imports = true` override block(s) for the OPTIONAL research libraries that mypy reports as import-not-found. DERIVE the exact module list from the actual mypy output (run mypy, collect the [import-not-found] module names), do NOT hardcode my guess. ONLY add optional/3rd-party research libs (e.g. cvxpy, riskfolio, pykalman, econml, ruptures, stumpy, etc.) — NEVER wildcard, and NEVER silence a first-party `assembled_core.*` module (a missing first-party import is a real bug, not noise).',
  'VERIFY B: install the stubs into the venv (`.venv\\Scripts\\python.exe -m pip install types-requests types-PyYAML types-pytz`), then re-run mypy EXACTLY as CI does (`.venv\\Scripts\\python.exe -m mypy src/assembled_core/data src/assembled_core/features src/assembled_core/signals src/assembled_core/execution src/assembled_core/portfolio`) and report the BEFORE (190) vs AFTER error count + the remaining error-code histogram. Confirm the remaining errors are GENUINE (no import-not-found for optional libs, no import-untyped for the stubbed pkgs). The gate stays non-blocking (no backend-ci.yml edit) — mypy must still EXIT NONZERO is fine since it is continue-on-error. Do NOT silence genuine errors.',
  '',
  '=== AFTER EDITING ===',
  '1. Confirm requirements.txt + requirements.lock + pyproject.toml are internally consistent (urllib3 floor, mypy pin, stubs).',
  '2. Do NOT git add / git commit. List ALL files modified.',
  '',
  'OUTPUT (markdown): PART A — exact pin edits + verbatim pip-audit evidence (3 urllib3 CVEs cleared, no resolver conflict); PART B — exact requirements/pyproject edits + mypy before/after error counts + remaining histogram + confirmation no genuine error was silenced + the exact optional-lib override list derived from mypy output; ALL files modified. Be explicit that the pip-audit + mypy GATE results are only authoritative in CI (local is supporting evidence).',
].join('\n')

const impl = await agent(IMPL_SPEC, { label: 'implement:deps', phase: 'Implement' })

const REVIEW_CONTEXT = [
  'Non-protected dependency/CI-hygiene batch (urllib3 security bump + mypy prereqs) just implemented in the MAIN working tree (uncommitted). Review ONLY the git diff. CI-relevant (security-scan + type-check gates).',
  '',
  '--- IMPLEMENTATION REPORT ---',
  impl,
  '--- END REPORT ---',
  '',
  'PART A: urllib3 2.5.0->2.6.3 (clears 3 unwaived CVEs) in requirements.lock + explicit pin in requirements.txt + pyproject floor; NO pip-freeze regen (would drift numpy/pandas). PART B: mypy==1.19.0 pin + types-requests/PyYAML/pytz + ignore_missing_imports overrides for optional research libs (noise-clearing only; gate NOT flipped). No .github/workflows edit, no src/ edit.',
].join('\n')

phase('Review')

const ci = await agent(
  REVIEW_CONTEXT + '\nYou are the ci-debugger. Verify: (1) urllib3 2.6.3 is pinned in requirements.lock AND reachable from requirements.txt (explicit pin) AND floored in pyproject; the pip-audit evidence genuinely shows all 3 urllib3 CVEs cleared; no resolver conflict with requests==2.32.4 / alpaca / yfinance / polygon pins (no cap below 2.6.3). (2) The lock was NOT pip-freeze-regenerated — confirm ONLY the urllib3 line changed in requirements.lock (numpy/pandas/pyarrow/alpaca/polygon pins byte-identical). (3) mypy==1.19.0 is pinned in requirements.txt; the stub pkgs + ignore_missing_imports overrides are scoped to optional 3rd-party libs only (NO first-party assembled_core.* silenced, no wildcard); the mypy error count dropped only by clearing genuine NOISE (import-untyped/import-not-found), not by silencing real errors. (4) No backend-ci.yml / no src/ change. (5) State that the gate results are CI-authoritative. Output YAML: stage, findings (BLOCKER/MAJOR/MINOR + file:line), urllib3_cves_cleared: yes|no, lock_no_freeze_drift: yes|no, mypy_no_genuine_silenced: yes|no, VERDICT.',
  { label: 'review:ci', phase: 'Review', agentType: 'ci-debugger' }
)

const senior = await agent(
  REVIEW_CONTEXT + '\nYou are the senior-code-reviewer. Confirm: the dependency edits are minimal + internally consistent (urllib3 floor vs pin vs lock; mypy pin); the pyproject mypy overrides do not over-broaden (no first-party silence, no blanket ignore_missing_imports=true global); the change is reversible; no protected path touched; the two concerns (security + typing) are honestly disclosed as one manifest-hygiene change. Output YAML: stage, findings (BLOCKER/MAJOR/MINOR + file:line + fix), VERDICT.',
  { label: 'review:senior', phase: 'Review', agentType: 'senior-code-reviewer' }
)

phase('Audit')

const audit = await agent(
  REVIEW_CONTEXT +
  '\n--- STAGE-2 REVIEWS ---\nCI-DEBUGGER:\n' + ci + '\n\nSENIOR:\n' + senior + '\n--- END REVIEWS ---\n\n' +
  'You are the task-completion-auditor. Decide if the deps batch is complete + safe: urllib3 bumped to 2.6.3 + reachable from the CI install + 3 CVEs cleared (evidence); lock NOT freeze-drifted (only urllib3 line changed); mypy pinned + only NOISE cleared (no genuine error silenced, no first-party import silenced, no global ignore); no protected path. If ci-debugger reports urllib3_cves_cleared:no OR lock_no_freeze_drift:no OR mypy_no_genuine_silenced:no, cannot be PASS. CI is authoritative for the gates (local = supporting evidence). Output YAML: stage: task-completion-auditor, verdict: PASS|CONDITIONAL|FAIL, verdict_reason, findings[], follow_ups[].',
  { label: 'audit:deps', phase: 'Audit', agentType: 'task-completion-auditor' }
)

return { impl, ci, senior, audit }
