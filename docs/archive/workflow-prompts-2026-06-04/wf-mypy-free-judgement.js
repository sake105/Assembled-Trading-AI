export const meta = {
  name: 'wf-mypy-free-judgement',
  description: 'Item D batch 3-5 (mypy free-zone JUDGEMENT + REAL-BUGS): clear the remaining own-file mypy errors in features/signals/portfolio — no-any-return/arg-type/union-attr/assignment/return-value/index/dict-item (type-only annotations/guards, verify no value change) + the REAL bugs meta_model meta_labeler(:169)/LimeExplanation(:261-262) attr-defined + conformal_position(:89/92) Optional. Each behaviour-touching fix gets a test; NO signal-output/weight change. Non-protected.',
  phases: [
    { title: 'Implement' },
    { title: 'Review' },
    { title: 'Audit' },
  ],
}

phase('Implement')

const IMPL_SPEC = [
  'You are clearing the remaining mypy errors in the NON-protected free dirs in Assembled-Trading-AI (repo F:\\Python_Projekt\\Aktiengeruest, Windows, .venv\\Scripts\\python.exe). You may ONLY edit files under src/assembled_core/features/, src/assembled_core/signals/, src/assembled_core/portfolio/ + tests/. Do NOT touch data/ or execution/ or any protected zone (execution/risk/accounting/pipeline/paper/.github). The mechanical batch already landed (commit 24fc4baa). GOAL: drive the OWN-FILE mypy errors in features/signals/portfolio to ~0 (errors mypy surfaces from imported data/execution files are NOT yours — leave them; they resolve in those batches).',
  '',
  '=== RUN FIRST ===',
  '.venv\\Scripts\\python.exe -m mypy src/assembled_core/features src/assembled_core/signals src/assembled_core/portfolio 2>&1 — and work ONLY the errors whose file path is under features/signals/portfolio.',
  '',
  '=== TWO KINDS OF FIX ===',
  'A) TYPE-ONLY (most): no-any-return (add the precise return-type annotation OR an explicit cast that does NOT change the value — e.g. float(x) only if x is already numeric and the cast is value-preserving), arg-type (fix the annotation or correct a genuinely wrong arg), union-attr (add a None/empty guard so the attribute access is type-safe — must preserve runtime behaviour: if the value can really be None at runtime, the guard is a real fix; if it never is, narrow via assert/annotation), assignment/return-value/index/dict-item/var-annotated. CRITICAL: a cast or guard MUST NOT change any produced value, weight, or signal. For the multifactor narrowing (signals/multifactor_signal.py ~:630/656) verify NO weight renormalization / signal-output change (run the multifactor signal tests + compare output).',
  'B) REAL-BUGS (investigate + fix correctly, NOT silence; add a test each):',
  '  - signals/meta_model.py:169 meta_labeler.predict_confidence — the [union-attr] ignore is unused because the real error is [attr-defined]. Determine meta_labeler real type/contract: does it actually have predict_confidence? If yes, type meta_labeler properly so it resolves (annotate the attribute/param). If the method does NOT exist on the real object, that is a genuine bug — report it and fix the call or guard it. Do NOT just re-add a broad ignore.',
  '  - signals/meta_model.py:261-262 LimeExplanation has no attribute top_features/predicted_value — verify the real lime wrapper API (grep the LimeExplanation class / its source). If those attributes do not exist, this code path is a real bug (wrong attribute names) — fix to the correct attributes or guard. Add a test exercising the lime-explanation path (or assert the correct attribute names) if feasible; if lime is an optional dep that is not installed, use importorskip and at least fix the attribute reference to the correct name verified from the wrapper.',
  '  - portfolio/conformal_position.py:89/92 self._mapie None has no attribute fit/predict — _mapie is declared None then reassigned before use; annotate it Optional (self._mapie: <Type> | None = None) and/or add a None-guard so mypy is satisfied without changing runtime (it is already reassigned before fit/predict). Confirm byte-identical.',
  '',
  '=== DISCIPLINE ===',
  'For EVERY change ask: does this change any runtime value/weight/signal/control-flow? If yes, it is NOT a type-only fix — it must be a deliberate REAL-BUG fix with a test proving the corrected behaviour, and you must report it explicitly. If a fix is ambiguous/risky, prefer a precise annotation or a narrowing assert over a value-changing edit, and report it for review. Do NOT silence a real error with a broad ignore.',
  '',
  '=== AFTER EDITING ===',
  '1. Re-run mypy on the 3 free dirs; report BEFORE/AFTER own-file error count (target ~0 own-file errors in features/signals/portfolio). Confirm no NEW error introduced.',
  '2. ruff format + ruff check --fix; final ruff check must pass.',
  '3. Run the touched-module tests + the multifactor/meta_model/conformal_position/portfolio-optimizer suites (-o addopts="" -p no:cacheprovider). Report EXACT pass/fail. Confirm NO signal-output/weight regression (the multifactor narrowing especially). Note any PRE-EXISTING failures (verify via git stash) vs newly introduced.',
  '4. Do NOT git add / git commit. LIST new test files.',
  '',
  'OUTPUT (markdown): split the fixes into TYPE-ONLY vs REAL-BUG (with evidence the real-bugs are genuinely fixed, not silenced); BEFORE/AFTER own-file mypy counts for the 3 dirs; confirmation type-only changes are byte-identical + real-bug fixes have tests + no signal/weight regression; new tests; exact ruff + pytest; files modified; any error left deferred (with reason).',
].join('\n')

const impl = await agent(IMPL_SPEC, { label: 'implement:mypy-free-judge', phase: 'Implement' })

const REVIEW_CONTEXT = [
  'A mypy free-zone JUDGEMENT + REAL-BUG batch (features/signals/portfolio) just implemented in the MAIN working tree (uncommitted). Review ONLY the git diff (+ new tests). Type-only changes must be byte-identical; real-bug fixes must be correct (not silenced) + tested; NO signal-output/weight change.',
  '',
  '--- IMPLEMENTATION REPORT ---',
  impl,
  '--- END REPORT ---',
  '',
  'Mixed: type-only annotations/guards (no value change) + 3 real-bugs (meta_labeler/LimeExplanation attr-defined, conformal_position Optional). Only features/signals/portfolio + tests/.',
].join('\n')

phase('Review')

const senior = await agent(
  REVIEW_CONTEXT + '\nYou are the senior-code-reviewer. For EACH changed line classify type-only vs behaviour-change: confirm the type-only fixes (annotations, value-preserving casts, None-guards that match real runtime nullability) change NO value/weight/signal/control-flow; confirm the multifactor narrowing did NOT alter weight/signal output; confirm the REAL-BUG fixes (meta_labeler typing, LimeExplanation attribute names, conformal_position Optional) are genuinely correct (verified against the real class API), NOT silenced with a broad ignore, and each has a test; flag any cast that could change a value (e.g. int() truncation, float rounding) or any guard that changes a branch outcome. No protected path. New tests LISTED. Output YAML: stage, findings (BLOCKER/MAJOR/MINOR + file:line + fix), type_only_byte_identical: yes|no, real_bugs_correctly_fixed: yes|no, untracked_files: [..], VERDICT.',
  { label: 'review:senior', phase: 'Review', agentType: 'senior-code-reviewer' }
)

const tester = await agent(
  REVIEW_CONTEXT + '\nYou are the test-runner. Re-run mypy on the 3 free dirs + the touched-module/multifactor/meta_model/conformal tests (-o addopts="" -p no:cacheprovider). Confirm: own-file mypy errors in features/signals/portfolio dropped to ~0 with no new error; the new real-bug tests pass; NO signal-output/weight regression; separate any pre-existing failures (git stash check) from newly-introduced. Output YAML: stage, own_file_errors_cleared: yes|no, no_new_errors: yes|no, no_signal_regression: yes|no, tests_pass: N passed/M failed, newly_introduced_failures: [..], VERDICT.',
  { label: 'review:tester', phase: 'Review', agentType: 'test-runner' }
)

phase('Audit')

const audit = await agent(
  REVIEW_CONTEXT +
  '\n--- STAGE-2 REVIEWS ---\nSENIOR:\n' + senior + '\n\nTEST-RUNNER:\n' + tester + '\n--- END REVIEWS ---\n\n' +
  'You are the task-completion-auditor. Decide if the free-zone judgement+real-bug batch is complete + safe: own-file mypy errors in features/signals/portfolio cleared, type-only changes byte-identical, the 3 real-bugs genuinely fixed (not silenced) + tested, NO signal/weight regression, no new mypy error, no protected path. If senior reports type_only_byte_identical:no OR real_bugs_correctly_fixed:no, OR tester no_signal_regression:no OR newly_introduced_failures non-empty, cannot be PASS. Output YAML: stage: task-completion-auditor, verdict: PASS|CONDITIONAL|FAIL, verdict_reason, findings[], follow_ups[].',
  { label: 'audit:mypy-free-judge', phase: 'Audit', agentType: 'task-completion-auditor' }
)

return { impl, senior, tester, audit }
