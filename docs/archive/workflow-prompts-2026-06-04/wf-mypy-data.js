export const meta = {
  name: 'wf-mypy-data',
  description: 'Item D batch 6 (mypy data/ cluster): clear own-file mypy errors in src/assembled_core/data/ (calendar.py, cost_model_policy.py, factor_store.py + any others) — unused-ignore/no-any-return/var-annotated, TYPE-ONLY. data/ is advisory-sensitive (PIT): NO change to as_of slicing / availability-lag / PIT-window logic. risk-execution-reviewer verifies PIT-safety. Not hard-deny, no deny-lift.',
  phases: [
    { title: 'Implement' },
    { title: 'Review' },
    { title: 'Audit' },
  ],
}

phase('Implement')

const IMPL_SPEC = [
  'You are clearing mypy errors in src/assembled_core/data/ in Assembled-Trading-AI (repo F:\\Python_Projekt\\Aktiengeruest, Windows, .venv\\Scripts\\python.exe). You may edit files under src/assembled_core/data/ + tests/. data/ is NOT a hard-deny zone (no deny-lift needed) but is ADVISORY-SENSITIVE (PIT / timing / backtest realism per Rule 30). Do NOT touch execution/risk/accounting/pipeline/paper/.github. CRITICAL: every change must be TYPE-ONLY / byte-identical — and MUST NOT alter any as_of slicing, availability-lag, release-lag, PIT-window, or timestamp-comparison logic (look-ahead safety). The free-zone + mechanical mypy batches already landed (commits 24fc4baa, 9344e7cf).',
  '',
  '=== RUN FIRST ===',
  '.venv\\Scripts\\python.exe -m mypy src/assembled_core/data 2>&1 — and also confirm against the full CI command later. Work the errors whose file path is under data/ (e.g. calendar.py, cost_model_policy.py, factor_store.py + any others). Categories expected: unused-ignore (stale # type: ignore), no-any-return (typed local / value-preserving cast), var-annotated (factor_store.py:161 — add the minimal correct annotation), possibly arg-type/assignment.',
  '',
  '=== FIXES (type-only) ===',
  '- unused-ignore: remove stale ignores mypy flags as unused (restore + defer if removal surfaces a real error).',
  '- no-any-return: add the precise return-type annotation OR a value-preserving cast (float()/int() ONLY if the value is already that type and the cast cannot truncate/round-change it — be EXTRA careful in data/ that no timestamp/epoch/price value is altered).',
  '- var-annotated: add the minimal correct variable annotation inferred from usage.',
  '- arg-type/assignment: precise annotation; if a real type mismatch is a genuine bug in a PIT/timing path, REPORT it (do not silence) and prefer a narrowing assert-with-message or annotation.',
  'For ANY line that touches a timestamp, as_of, cutoff, lag, .loc/.iloc date slice, or merge_asof: do NOT change its behaviour — annotation only. If a fix would require changing such logic, STOP and report it for a dedicated PIT review instead.',
  '',
  '=== AFTER EDITING ===',
  '1. Re-run mypy over data/ + the full CI command (.venv\\Scripts\\python.exe -m mypy src/assembled_core/data src/assembled_core/features src/assembled_core/signals src/assembled_core/execution src/assembled_core/portfolio); report data/ own-file BEFORE/AFTER + the full-command total (should be down to just execution/ errors after this batch). Confirm NO new error.',
  '2. ruff format + ruff check --fix; final ruff check must pass.',
  '3. Run PIT-AWARE data tests + the touched-module tests (e.g. pytest tests/ -k "calendar or factor_store or cost_model or pit or latency or availability or as_of" -o addopts="" -p no:cacheprovider). Report EXACT pass/fail. Separate pre-existing failures (git stash) from newly-introduced.',
  '4. Do NOT git add / git commit. List files modified + any new test.',
  '',
  'OUTPUT (markdown): the fixes (file:line + category + why type-only); explicit confirmation NO as_of/PIT/timestamp logic changed (every data/ change is annotation/value-preserving); BEFORE/AFTER data/ own-file mypy + full-command total; ruff + pytest (PIT tests); any error deferred for a dedicated PIT review (with reason); files modified.',
].join('\n')

const impl = await agent(IMPL_SPEC, { label: 'implement:mypy-data', phase: 'Implement' })

const REVIEW_CONTEXT = [
  'A mypy data/ cleanup batch (advisory-sensitive, PIT) just implemented in the MAIN working tree (uncommitted). Review ONLY the git diff (+ new tests). Must be TYPE-ONLY and must NOT alter any as_of/PIT/timestamp/lag logic.',
  '',
  '--- IMPLEMENTATION REPORT ---',
  impl,
  '--- END REPORT ---',
  '',
  'Type-only mypy fixes in data/ (unused-ignore/no-any-return/var-annotated). No as_of slicing / availability-lag / PIT-window / timestamp-comparison change. Only data/ + tests/.',
].join('\n')

phase('Review')

const risk = await agent(
  REVIEW_CONTEXT + '\nYou are the risk-execution-reviewer (PIT focus). Verify NO look-ahead / PIT regression: every changed line in data/ is annotation-only or a strictly value-preserving cast; NO as_of cutoff, availability-lag, release-lag, PIT-window, timestamp comparison, .loc/.iloc date slice, or merge_asof behaviour changed; no float()/int() cast that could truncate or shift a timestamp/epoch/price; the data the loaders return is byte-identical. Confirm no execution/risk/accounting/pipeline/paper/.github path touched. Output YAML: stage, findings (BLOCKER/MAJOR/MINOR + file:line), pit_logic_unchanged: yes|no, type_only_byte_identical: yes|no, VERDICT.',
  { label: 'review:risk', phase: 'Review', agentType: 'risk-execution-reviewer' }
)

const senior = await agent(
  REVIEW_CONTEXT + '\nYou are the senior-code-reviewer. Confirm each fix is type-only (annotation / value-preserving), removed ignores were genuinely unused, no cast alters a value, no logic/control-flow change; new tests (if any) LISTED; no non-allowed path. Output YAML: stage, findings (BLOCKER/MAJOR/MINOR + file:line + fix), byte_identical: yes|no, untracked_files: [..], VERDICT.',
  { label: 'review:senior', phase: 'Review', agentType: 'senior-code-reviewer' }
)

phase('Audit')

const audit = await agent(
  REVIEW_CONTEXT +
  '\n--- STAGE-2 REVIEWS ---\nRISK:\n' + risk + '\n\nSENIOR:\n' + senior + '\n--- END REVIEWS ---\n\n' +
  'You are the task-completion-auditor. Decide if the data/ mypy batch is complete + safe: data/ own-file mypy errors cleared, every change type-only byte-identical with NO PIT/as_of/timestamp logic change, no new error, PIT tests pass, no protected path. If risk reports pit_logic_unchanged:no OR type_only_byte_identical:no, cannot be PASS. Output YAML: stage: task-completion-auditor, verdict: PASS|CONDITIONAL|FAIL, verdict_reason, findings[], follow_ups[].',
  { label: 'audit:mypy-data', phase: 'Audit', agentType: 'task-completion-auditor' }
)

return { impl, risk, senior, audit }
