export const meta = {
  name: 'wf-mypy-free-mechanical',
  description: 'Item D Batch 1-2 (mypy free-zone, MECHANICAL, byte-identical): remove unused-ignore stale # type: ignore + add var-annotated annotations (signal_diagnostics:107, factor_store:161) + fix name-defined (pead_sue:91 datetime forward-ref) in the NON-protected dirs features/signals/portfolio. Type-comment/annotation only, zero runtime change. data/ + execution/ are separate batches; do NOT touch them.',
  phases: [
    { title: 'Implement' },
    { title: 'Review' },
    { title: 'Audit' },
  ],
}

phase('Implement')

const IMPL_SPEC = [
  'You are doing MECHANICAL mypy cleanup in Assembled-Trading-AI (repo F:\\Python_Projekt\\Aktiengeruest, Windows, .venv\\Scripts\\python.exe). You may ONLY edit files under src/assembled_core/features/, src/assembled_core/signals/, src/assembled_core/portfolio/ (none are hard-deny) + tests/ if needed. Do NOT touch src/assembled_core/data/ or src/assembled_core/execution/ (separate batches) or any other protected zone (execution/risk/accounting/pipeline/paper/.github). Every change must be TYPE-ONLY / byte-identical at runtime — NO logic, value, cast-that-changes-value, or control-flow change.',
  '',
  '=== CONTEXT ===',
  'mypy==1.19.0 is pinned (commit d5e15f87) and runs NON-blocking in CI over src/assembled_core/{data,features,signals,execution,portfolio}. This batch fixes ONLY the MECHANICAL error classes in the free dirs (features/signals/portfolio): unused-ignore (stale `# type: ignore`), var-annotated (missing variable annotation), name-defined (forward-ref/TYPE_CHECKING import). Run mypy first to get the live list:',
  '  .venv\\Scripts\\python.exe -m mypy src/assembled_core/features src/assembled_core/signals src/assembled_core/portfolio 2>&1',
  '',
  '=== FIXES ===',
  '1. unused-ignore: for every `# type: ignore[...]` (or bare) that mypy reports as [unused-ignore] in features/signals/portfolio, REMOVE just the ignore comment (keep the code line). Only remove the ones mypy flags as unused — do NOT remove ignores that are still load-bearing. After removal, re-run mypy to confirm no NEW error appeared on that line (if removing an ignore surfaces a real error, RESTORE the ignore and report that line as needing a real fix in the judgement batch, do NOT leave it broken).',
  '2. var-annotated: e.g. signal_diagnostics.py:107, factor_store.py:161 (verify current lines) — add the minimal correct variable annotation (e.g. `x: dict[str, float] = {}`), inferred from how the variable is used. No value change.',
  '3. name-defined: pead_sue.py:91 — `datetime` is imported under `if TYPE_CHECKING:` and used in a forward-ref string annotation. Fix by adding `from __future__ import annotations` at the top (if not present) OR making the import mypy-visible, whichever is the smaller correct change. The runtime behaviour must not change (it already has a noqa).',
  '',
  '=== AFTER EDITING ===',
  '1. Re-run `.venv\\Scripts\\python.exe -m mypy src/assembled_core/features src/assembled_core/signals src/assembled_core/portfolio` and report BEFORE/AFTER counts for unused-ignore / var-annotated / name-defined specifically (they should drop to ~0 in these dirs; other error classes in these dirs are the NEXT batch, leave them). Confirm NO new error was introduced.',
  '2. ruff format + ruff check --fix; final ruff check must pass.',
  '3. Run the touched modules tests (e.g. pytest tests/ -k "signal_diagnostics or factor_store or pead or multifactor" -o addopts="" -p no:cacheprovider) to confirm byte-identical behaviour. Report exact pass/fail.',
  '4. Do NOT git add / git commit. List files modified + any new test.',
  '',
  'OUTPUT (markdown): the exact mechanical fixes (which ignores removed, which annotations added, the name-defined fix); BEFORE/AFTER mypy counts for the 3 categories in the 3 free dirs; confirmation every change is type-only/byte-identical (no value/logic change) + no new mypy error introduced; ruff + pytest; files modified. Explicitly note any unused-ignore whose removal surfaced a real error (deferred to the judgement batch).',
].join('\n')

const impl = await agent(IMPL_SPEC, { label: 'implement:mypy-free-mech', phase: 'Implement' })

const REVIEW_CONTEXT = [
  'A MECHANICAL mypy free-zone cleanup (features/signals/portfolio: unused-ignore removal + var-annotated + name-defined) just implemented in the MAIN working tree (uncommitted). Review ONLY the git diff. Must be byte-identical at runtime (type-comment/annotation only).',
  '',
  '--- IMPLEMENTATION REPORT ---',
  impl,
  '--- END REPORT ---',
  '',
  'Type-only: removed stale # type: ignore, added missing var annotations, fixed a forward-ref import. No logic/value/cast change. Only features/signals/portfolio + tests/.',
].join('\n')

phase('Review')

const senior = await agent(
  REVIEW_CONTEXT + '\nYou are the senior-code-reviewer. Confirm EVERY changed line is type-only / byte-identical at runtime: removed ignores were genuinely unused (no real error now surfaces on that line); added annotations match the actual runtime type and change no value; the name-defined fix (from __future__ import annotations or visible import) changes no runtime behaviour; NO cast that alters a value, NO logic/control-flow change. No protected path (data/execution/etc.) touched. Tests pass. Output YAML: stage, findings (BLOCKER/MAJOR/MINOR + file:line + fix), byte_identical: yes|no, VERDICT.',
  { label: 'review:senior', phase: 'Review', agentType: 'senior-code-reviewer' }
)

const tester = await agent(
  REVIEW_CONTEXT + '\nYou are the test-runner. Re-run mypy on the 3 free dirs + the touched-module tests (-o addopts="" -p no:cacheprovider). Confirm: the unused-ignore/var-annotated/name-defined categories dropped in features/signals/portfolio with NO new error introduced; the touched-module tests pass (byte-identical behaviour). Output YAML: stage, mypy_categories_cleared: yes|no, no_new_errors: yes|no, tests_pass: N passed/M failed, VERDICT.',
  { label: 'review:tester', phase: 'Review', agentType: 'test-runner' }
)

phase('Audit')

const audit = await agent(
  REVIEW_CONTEXT +
  '\n--- STAGE-2 REVIEWS ---\nSENIOR:\n' + senior + '\n\nTEST-RUNNER:\n' + tester + '\n--- END REVIEWS ---\n\n' +
  'You are the task-completion-auditor. Decide if the mechanical free-zone batch is complete + safe: unused-ignore/var-annotated/name-defined cleared in features/signals/portfolio, every change type-only byte-identical (no value/logic change), no new mypy error, tests pass, no protected path. If senior reports byte_identical:no OR tester no_new_errors:no, cannot be PASS. Output YAML: stage: task-completion-auditor, verdict: PASS|CONDITIONAL|FAIL, verdict_reason, findings[], follow_ups[].',
  { label: 'audit:mypy-free-mech', phase: 'Audit', agentType: 'task-completion-auditor' }
)

return { impl, senior, tester, audit }
