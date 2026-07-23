export const meta = {
  name: 'wf-mypy-gateflip',
  description: 'Item D batch 10 (FINAL): flip the mypy CI gate from advisory to BLOCKING by removing continue-on-error: true from the backend-ci.yml mypy step. HARD GATE: re-run the EXACT CI mypy command first and ABORT if not 0 errors. Rule-20 CI disclosure (Ubuntu py3.10/3.11, blocking job). DISCLOSE the local-vs-CI dep-drift risk (CI numpy/pandas pins may surface residual errors). .github/workflows deny lifted; ci-debugger review.',
  phases: [
    { title: 'Implement' },
    { title: 'Review' },
    { title: 'Audit' },
  ],
}

phase('Implement')

const IMPL_SPEC = [
  'You are flipping the mypy CI gate to BLOCKING in Assembled-Trading-AI (repo F:\\Python_Projekt\\Aktiengeruest, Windows, .venv\\Scripts\\python.exe). .github/workflows/** deny is TEMPORARILY lifted. You may edit ONLY .github/workflows/backend-ci.yml (the mypy step). Do NOT touch any src/ path. CI is HOCHSENSIBEL (Rule 20). This is the final mypy batch — all 134 errors were fixed in prior commits (mypy 1.19.0 pinned, stubs + ignore_missing_imports overrides landed).',
  '',
  '=== HARD GATE (do this FIRST — ABORT if it fails) ===',
  'Run the EXACT CI mypy command: `.venv\\Scripts\\python.exe -m mypy src/assembled_core/data src/assembled_core/features src/assembled_core/signals src/assembled_core/execution src/assembled_core/portfolio`. It MUST report `Success: no issues found` (0 errors). If it reports ANY error, ABORT — do NOT edit backend-ci.yml — and report the residual errors (the gate cannot flip with residual errors, as that would turn CI red). Report the exact mypy output.',
  '',
  '=== THE FLIP (only if 0 errors) ===',
  'READ .github/workflows/backend-ci.yml around the mypy step (the "Run mypy (type checking)" step, ~:150-154). It currently has `continue-on-error: true` (the line that makes mypy NON-blocking). REMOVE exactly that `continue-on-error: true` line (and update the adjacent comment that says mypy is "optional/non-blocking" to reflect that it is now a BLOCKING gate). Change NOTHING else in the workflow — same mypy invocation, same targets, same matrix. The ONLY behavioural change is: a mypy error now FAILS the job instead of showing a red-but-passing step.',
  '',
  '=== VERIFY ===',
  '1. YAML validity: parse backend-ci.yml (python -c yaml.safe_load) — must parse clean. Confirm the diff is EXACTLY the continue-on-error removal + the comment update (no other line changed). Show the before/after of the mypy step.',
  '2. Re-confirm the mypy command is still 0 (so the now-blocking gate would pass IF CI matches local).',
  '',
  '=== DISCLOSURE (Rule 20 + Rule 40 — state explicitly in the output) ===',
  '- WHY: the 134 mypy errors are fixed; per the in-file ignore-list policy, prefer enforcing over indefinite advisory.',
  '- WHAT CHANGES: the mypy step (over data/features/signals/execution/portfolio) now BLOCKS the job on any type error (was advisory red-but-passing).',
  '- WHICH JOBS/PLATFORMS: the backend-ci type-check step; state the matrix (Ubuntu, Python 3.10/3.11 — confirm from the file).',
  '- CI RISK (the honest caveat): local mypy is 0, but CI installs numpy/pandas/pyarrow from requirements.txt (pinned) while the local .venv has drifted (lock warns numpy 2.3.3 local vs 2.2.6 pinned). Different stub versions COULD surface a few residual mypy errors in CI that local does not show (Rule 40 dependency-drift). So the gate-flip is NOT CI-confirmed — the first CI run after this commit is the real test. If CI mypy goes red, the fix is either the residual errors or a one-line revert of this change. State this plainly.',
  '',
  '=== AFTER EDITING ===',
  '- Do NOT git add / git commit.',
  '- If the HARD GATE failed (residual errors), report ABORTED + the errors and make NO edit.',
  '',
  'OUTPUT (markdown): the HARD-GATE mypy result (0 or the residual errors); IF flipped: the exact before/after of the mypy step (continue-on-error removed + comment updated), YAML-valid confirmation, and the full Rule-20 + dep-drift disclosure; IF aborted: the residual errors + no edit. Files modified.',
].join('\n')

const impl = await agent(IMPL_SPEC, { label: 'implement:gateflip', phase: 'Implement' })

const REVIEW_CONTEXT = [
  'The mypy CI gate-flip (.github/workflows/backend-ci.yml, deny lifted) just implemented in the MAIN working tree (uncommitted). Review ONLY the git diff. HOCHSENSIBLE CI CHANGE — makes the mypy step blocking.',
  '',
  '--- IMPLEMENTATION REPORT ---',
  impl,
  '--- END REPORT ---',
  '',
  'Removed continue-on-error: true from the mypy step so mypy now BLOCKS the job. Hard-gated on a local mypy 0 re-run. Only backend-ci.yml touched.',
].join('\n')

phase('Review')

const ci = await agent(
  REVIEW_CONTEXT + '\nYou are the ci-debugger. Verify: (1) the diff is EXACTLY the removal of continue-on-error: true on the mypy step (+ the comment update) — no other workflow line/step/job/matrix changed; (2) backend-ci.yml still parses as valid YAML; (3) the mypy invocation/targets are unchanged; (4) the HARD GATE was honored (local mypy reported 0 before the flip — if the impl reported residual errors it must have ABORTED with no edit); (5) the Rule-20 disclosure is present + accurate (which job/matrix blocks now) AND the local-vs-CI dep-drift caveat is stated (CI numpy/pandas pins may surface residual errors -> the flip is NOT CI-confirmed, first CI run is the test, one-line revertible). (6) no src/ path touched. Output YAML: stage, findings (BLOCKER/MAJOR/MINOR + file:line), only_continue_on_error_removed: yes|no, yaml_valid: yes|no, local_mypy_zero: yes|no, ci_confirmed: no, VERDICT.',
  { label: 'review:ci', phase: 'Review', agentType: 'ci-debugger' }
)

const senior = await agent(
  REVIEW_CONTEXT + '\nYou are the senior-code-reviewer. Confirm the change is minimal (one line removed + comment), reversible (re-adding continue-on-error restores advisory), honestly disclosed (CI-unverified, dep-drift caveat), no src touched, no other CI behaviour changed. Output YAML: stage, findings (BLOCKER/MAJOR/MINOR + file:line + fix), VERDICT.',
  { label: 'review:senior', phase: 'Review', agentType: 'senior-code-reviewer' }
)

phase('Audit')

const audit = await agent(
  REVIEW_CONTEXT +
  '\n--- STAGE-2 REVIEWS ---\nCI-DEBUGGER:\n' + ci + '\n\nSENIOR:\n' + senior + '\n--- END REVIEWS ---\n\n' +
  'You are the task-completion-auditor. Decide if the gate-flip is complete + safe: local mypy was 0 (hard gate honored), the diff is exactly the continue-on-error removal + comment, YAML valid, Rule-20 + dep-drift disclosure present, no src touched, reversible. CRITICAL: the gate result is NOT CI-confirmed (local-vs-CI dep drift) — this MUST be disclosed; the completion claim must say "mypy gate flipped to blocking; local 0; CI verification pending on next push (one-line revertible if CI surfaces residual errors)". If ci-debugger reports only_continue_on_error_removed:no OR yaml_valid:no OR local_mypy_zero:no, cannot be PASS. settings.json deny-restore (.github/workflows) orchestrator-handled. Output YAML: stage: task-completion-auditor, verdict: PASS|CONDITIONAL|FAIL, verdict_reason, findings[], follow_ups[].',
  { label: 'audit:gateflip', phase: 'Audit', agentType: 'task-completion-auditor' }
)

return { impl, ci, senior, audit }
