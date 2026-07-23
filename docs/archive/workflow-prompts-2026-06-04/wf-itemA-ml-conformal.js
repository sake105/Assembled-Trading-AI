export const meta = {
  name: 'wf-itemA-ml-conformal',
  description: 'Item A: fix the broken first-party import in signals/meta_model.py:188 (from src.assembled_core.ml.conformal import ConformalResult — both ml/conformal.py and ml/conformal_prediction.py were ARCHIVED, so the import always raises ModuleNotFoundError -> the except-branch returns degenerate zero-width intervals permanently). Option 2: drop the archived dependency, keep a minimal LOCAL result container + the inline q-residual-quantile interval logic that meta_model already computes, so predict_with_intervals returns REAL intervals again. signals/ NOT hard-deny.',
  phases: [
    { title: 'Implement' },
    { title: 'Review' },
    { title: 'Audit' },
  ],
}

phase('Implement')

const IMPL_SPEC = [
  'You are fixing a broken first-party import in Assembled-Trading-AI (repo F:\\Python_Projekt\\Aktiengeruest, Windows, .venv\\Scripts\\python.exe). Files you MAY edit: src/assembled_core/signals/meta_model.py and a NEW test under tests/. signals/ is NOT a hard-deny zone. Do NOT touch execution/risk/accounting/pipeline/paper/.github. Smallest safe change; confirm against current source.',
  '',
  '=== THE BUG (verified by scoping; re-confirm) ===',
  'src/assembled_core/signals/meta_model.py:188 has `from src.assembled_core.ml.conformal import ConformalResult` inside predict_with_intervals (~:185-225), wrapped in try/except Exception. BOTH ml/conformal.py AND ml/conformal_prediction.py were MOVED to archive/observability_graveyard_2026q2/ml/ (only stale .pyc remain live), so the import ALWAYS raises ModuleNotFoundError -> the except-branch logs "[MetaModel] Conformal failed ... returning point predictions" and returns DEGENERATE intervals (lower==upper, half_width=0.0, confidence=1.0). The feature is silently, permanently dead. NOTE: do NOT redirect the import to the archived conformal_prediction.py — its ConformalResult API is INCOMPATIBLE (no point_predictions, no .confidence()).',
  '',
  '=== FIX (Option 2 — inline, no archived dependency, restores REAL intervals) ===',
  'READ predict_with_intervals fully (~:185-225). The conformal interval logic it needs is already mostly local: the residual quantile `q` and the point predictions are computed in the function (scoping cited ~:192-196). The archived ConformalResult was only a CONTAINER constructed at ~:199-207 with point_predictions / lower_bounds / upper_bounds / half_width / alpha and a .confidence() method.',
  'Implement: (1) REMOVE the broken `from src.assembled_core.ml.conformal import ConformalResult` import. (2) Define a MINIMAL local result container with the EXACT API meta_model consumes — a small frozen @dataclass (e.g. `_ConformalResult`) at module level in meta_model.py (or a tiny local class), with fields point_predictions / lower_bounds / upper_bounds / half_width / alpha and a `confidence()` method returning 1.0 - alpha (or whatever the archived .confidence() returned — verify by reading how the caller uses .confidence()). (3) Compute the interval INLINE using the already-present q-residual-quantile logic: bounds = point_predictions -/+ q, half_width = q, so predict_with_intervals returns REAL (non-degenerate) intervals when a calib set is provided. (4) Keep a narrow try/except ONLY around genuinely fallible numeric steps (NOT around a doomed import); the degraded point-prediction fallback should remain for the no-calib / insufficient-data case, but the normal path must now produce real intervals. Preserve the public return shape/type that callers expect.',
  'CONFIRM the exact q computation + the .confidence() semantics by reading the current code + how predict_with_intervals output is consumed (grep callers). Do NOT invent a different interval method — reproduce what the conformal construction at :199-207 intended (preds +/- q). If q is NOT actually computed in the current function (only inside the dead path), compute it inline from the calib residuals the standard split-conformal way (|y_calib - pred_calib| empirical (1-alpha) quantile).',
  '',
  '=== TESTS ===',
  'New test (tests/, e.g. tests/test_meta_model_conformal_intervals.py): call predict_with_intervals with a real calibration set + test points; assert half_width > 0, lower < upper element-wise, confidence in [0,1], and point_predictions match the underlying model output. Add a no-calib / insufficient-data case asserting the graceful degraded fallback still works. The test must DISCRIMINATE (fail if intervals are degenerate/zero-width on the normal path). Note: existing tests/test_conformal_prediction.py targets the OTHER archived module via importorskip and stays skipped — it does NOT cover this path.',
  '',
  '=== AFTER EDITING ===',
  '1. ruff format + ruff check --fix; final ruff check must pass.',
  '2. Run the new test + `pytest tests/ -k "meta_model or conformal" -o addopts="" -p no:cacheprovider`. Report EXACT pass/fail. Confirm predict_with_intervals now returns real intervals on the normal path + the broken import is gone (grep meta_model.py for ml.conformal -> no match).',
  '3. Optionally note the stale ml/__pycache__/conformal*.pyc as a cosmetic cleanup follow-up (do NOT delete via a protected path; .pyc without a .py is never imported so it is harmless).',
  '4. Do NOT git add / git commit. LIST the new untracked test file.',
  '',
  'OUTPUT (markdown): the fix (removed import + local container + inline q-interval); confirmation real intervals are produced on the normal path + the degraded fallback preserved; whether predict_with_intervals is on the live order path (grep callers — scoping said NO live caller, confirm); new test file; exact ruff + pytest; files modified.',
].join('\n')

const impl = await agent(IMPL_SPEC, { label: 'implement:ml-conformal', phase: 'Implement' })

const REVIEW_CONTEXT = [
  'A fix for the broken ml.conformal import in signals/meta_model.py just implemented in the MAIN working tree (uncommitted). Review ONLY the git diff (+ new untracked test). NOT a protected/risk path (signals/), NOT on the live order path.',
  '',
  '--- IMPLEMENTATION REPORT ---',
  impl,
  '--- END REPORT ---',
  '',
  'Option 2: removed the broken archived-module import, added a minimal local result container + inline q-residual-quantile interval logic so predict_with_intervals returns REAL intervals (was permanently degenerate). Only signals/ + tests/ touched.',
].join('\n')

phase('Review')

const senior = await agent(
  REVIEW_CONTEXT + '\nYou are the senior-code-reviewer. Confirm: the broken `from ...ml.conformal import` is removed; the local result container reproduces the EXACT API the callers use (point_predictions/lower_bounds/upper_bounds/half_width/.confidence()); the inline q-interval reproduces the intended split-conformal bounds (preds +/- q, real half_width), NOT a degenerate or a different method; the degraded fallback for no-calib/insufficient-data is preserved; predict_with_intervals is NOT on the live order path (so no live behaviour risk); the test discriminates (fails on degenerate intervals); new test LISTED; no protected path touched; no archived module resurrected. Output YAML: stage, findings (BLOCKER/MAJOR/MINOR + file:line + fix), real_intervals_restored: yes|no, untracked_files: [..], VERDICT.',
  { label: 'review:senior', phase: 'Review', agentType: 'senior-code-reviewer' }
)

const tester = await agent(
  REVIEW_CONTEXT + '\nYou are the test-runner. Re-run the new test + `tests/ -k "meta_model or conformal"` (-o addopts="" -p no:cacheprovider) and report EXACT pass/fail. Independently verify: predict_with_intervals returns half_width>0 / lower<upper on a real calib set (not degenerate), the degraded fallback still works on no-calib, and the broken import is gone (grep). Confirm test_conformal_prediction.py stays skipped (importorskip, other module). Output YAML: stage, tests_pass: N passed/M failed, real_intervals_confirmed: yes|no, findings, VERDICT.',
  { label: 'review:tester', phase: 'Review', agentType: 'test-runner' }
)

phase('Audit')

const audit = await agent(
  REVIEW_CONTEXT +
  '\n--- STAGE-2 REVIEWS ---\nSENIOR:\n' + senior + '\n\nTEST-RUNNER:\n' + tester + '\n--- END REVIEWS ---\n\n' +
  'You are the task-completion-auditor. Decide if Item A is complete: the broken archived-module import is removed, predict_with_intervals returns REAL intervals on the normal path (not degenerate) via the inline q-logic + local container, the degraded fallback is preserved, no live order path affected, discriminating tests pass, no protected path, new test listed, no archived module resurrected. If senior reports real_intervals_restored:no OR tester real_intervals_confirmed:no, cannot be PASS. Output YAML: stage: task-completion-auditor, verdict: PASS|CONDITIONAL|FAIL, verdict_reason, findings[], follow_ups[].',
  { label: 'audit:ml-conformal', phase: 'Audit', agentType: 'task-completion-auditor' }
)

return { impl, senior, tester, audit }
