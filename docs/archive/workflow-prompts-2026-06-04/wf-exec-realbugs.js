export const meta = {
  name: 'wf-exec-realbugs',
  description: 'Item D batch 9 (execution/ REAL-BUGs, the last 7 mypy errors): per-bug investigate + fix or DEFER. kill_switch:80 no-untyped-def + :126 None->SoftFileLock (type-only, safety-critical); unified_paper_engine:142 log_experience_entry undefined (dead -> remove, byte-identical); unified_paper_engine:106 wrong ledger import (re-activates ledger-parquet writes -> behaviour change, confirm target exists + disclose + test); paper_trading_engine:368/370 schedule() call-arg TypeError (dead-crashing method -> fix signature or confirm dead). execution/ deny lifted; risk review per fix. Anything uncertain -> DEFER, do not rush a risk path.',
  phases: [
    { title: 'Implement' },
    { title: 'Review' },
    { title: 'Audit' },
  ],
}

phase('Implement')

const IMPL_SPEC = [
  'You are fixing the LAST execution/ mypy real-bugs in Assembled-Trading-AI (repo F:\\Python_Projekt\\Aktiengeruest, Windows, .venv\\Scripts\\python.exe). src/assembled_core/execution/** deny is TEMPORARILY lifted. You may edit execution/ + tests/. Do NOT touch risk/accounting/pipeline/paper/.github (read for tracing). execution/ is the MOST SENSITIVE zone (kill-switch, paper engine, ledger). These are GENUINE BUGS, not type-noise — each needs the CORRECT fix, never a broad silencing ignore. For any fix that is uncertain or risky, DEFER it (report it clearly for a user decision) rather than rush a risk path. Investigate each before editing. Goal: resolve all 7 mypy errors so the full CI mypy command reaches 0 (unless a bug must be deferred — then explain).',
  '',
  'RUN: .venv\\Scripts\\python.exe -m mypy src/assembled_core/execution to see the 7 errors. Handle each:',
  '',
  '=== BUG 1+2 — kill_switch.py:80 (no-untyped-def) + :126 (assignment None->SoftFileLock) ===',
  'READ both. :80 — add the missing type signature to the untyped function (annotation-only, NO logic change). :126 — a variable declared None is assigned a SoftFileLock (filelock); annotate it Optional (e.g. `_lock: SoftFileLock | None = None`) or restructure the annotation so mypy is satisfied WITHOUT changing the kill-switch lifecycle/behaviour. SAFETY-CRITICAL: verify the kill-switch engage/disengage/persist logic is byte-identical. Type-only.',
  '',
  '=== BUG 3 — unified_paper_engine.py:142 (log_experience_entry undefined) ===',
  'READ ~:138-146. `from ops.experience_log import log_experience_entry` — grep the WHOLE repo: does log_experience_entry exist ANYWHERE? Scoping says NO. It is wrapped in try/except -> _HAS_EXPERIENCE_LOG=False -> the call site is a masked no-op. CORRECT fix: REMOVE the dead import + its now-dead call site / guard (the feature was never implemented). This is BYTE-IDENTICAL (the path was always a no-op). Confirm no other behaviour depends on _HAS_EXPERIENCE_LOG. If removing is non-trivial, at minimum remove the import and keep the flag False explicitly. Report it as byte-identical dead-code removal.',
  '',
  '=== BUG 4 — unified_paper_engine.py:106 (wrong ledger import — BEHAVIOUR CHANGE, confirm + disclose) ===',
  'READ ~:100-112. `from accounting.ledger import store_ledger_events_parquet` raises (wrong module) -> try/except -> _HAS_LEDGER=False -> the ledger-parquet write path is SILENTLY DISABLED. First CONFIRM: does store_ledger_events_parquet ACTUALLY exist in accounting.ledger_store (grep its def + signature)? IF YES and the signature matches how unified_paper_engine calls it: fix the import to accounting.ledger_store -> this RE-ACTIVATES the ledger-parquet write path = a LIVE PAPER BEHAVIOUR CHANGE (the paper engine starts persisting ledger events to parquet, as originally intended). This is an additive persistence/output path (NOT order/fill/position logic) but it IS a behaviour change — DISCLOSE prominently, add a test that the re-activated write path runs without error and writes the expected artifact, and confirm it does not raise / corrupt state. IF store_ledger_events_parquet does NOT exist in ledger_store (or signature mismatch): treat like BUG 3 — remove the dead import (byte-identical), and report that the intended feature has no implementation. Do NOT guess a signature. If re-activation looks risky (could error in the live paper cycle), DEFER with a clear report rather than auto-enabling.',
  '',
  '=== BUG 5+6 — paper_trading_engine.py:368 (TWAP->VWAP scheduler type) + :370 (schedule() call-arg x2) ===',
  'READ ~:355-378. The schedule() call uses kwargs total_quantity=/reference_price= but the real scheduler.schedule signature is schedule(symbol, total_qty, side, start_time, end_time, ...). This raises TypeError on ANY call -> the method is dead/crashing (never worked). First determine: is this method ever CALLED in a live/paper path (grep callers)? IF the method is never called in live (dead): fixing the call signature is a LATENT correctness fix (byte-identical to live, since it never runs) — fix it to the correct signature so it WOULD work + add a unit test calling it. IF it IS called in a live path: it has been crashing -> fixing it is a real behaviour change (the method starts working) -> fix to the correct signature + DISCLOSE + test. Also fix :368 (the scheduler-type annotation/assignment) consistently. If the correct call mapping is ambiguous (which arg is total_qty vs reference_price), investigate the scheduler.schedule def + the TWAP/VWAP scheduler classes to map correctly; if genuinely ambiguous, DEFER.',
  '',
  '=== DISCIPLINE ===',
  'Per bug, classify: type-only | byte-identical-dead-removal | behaviour-change-fix | DEFERRED. For every behaviour-change-fix: disclose the exact change + add a test. NEVER silence a real bug with a broad ignore. If you cannot resolve a mypy error without a risky change, DEFER it (leave the error, report it) — a deferred error means the gate-flip waits or uses a tracked narrow ignore later; that is the orchestrator decision, not yours. asserts in src/ need a message.',
  '',
  '=== AFTER EDITING ===',
  '1. Re-run mypy over execution/ + full CI command; report which of the 7 are now resolved vs deferred (target 0, but deferral is acceptable if disclosed). Confirm NO new error, NO real bug silenced.',
  '2. ruff format + ruff check --fix; final ruff check must pass.',
  '3. Run kill_switch + unified/paper engine + paper_trading_engine + broker + ledger suites (-o addopts="" -p no:cacheprovider). Report EXACT pass/fail; separate pre-existing (git stash) from new. New tests for each behaviour-change/correctness fix.',
  '4. Do NOT git add / git commit. LIST new tests + any DEFERRED bug.',
  '',
  'OUTPUT (markdown): per BUG 1-6: classification + the fix (or DEFER reason) + behaviour-change disclosure + test; the resolved-vs-deferred mypy tally; ruff + pytest (no new failure); files modified; new tests.',
].join('\n')

const impl = await agent(IMPL_SPEC, { label: 'implement:exec-realbugs', phase: 'Implement' })

const REVIEW_CONTEXT = [
  'The execution/ REAL-BUG batch (deny lifted) just implemented in the MAIN working tree (uncommitted). Review ONLY the git diff (+ new tests). execution/ is the MOST sensitive zone. Some fixes are behaviour-changing (re-activating dormant paper paths / fixing a crashing method) — verify each is correct + disclosed + tested; type-only ones byte-identical; dead-removals byte-identical; deferred ones genuinely deferred (not silenced).',
  '',
  '--- IMPLEMENTATION REPORT ---',
  impl,
  '--- END REPORT ---',
  '',
  'Per-bug: kill_switch annotations (type-only); unified_paper_engine:142 dead-import removal (byte-identical); :106 ledger_store import re-activation (behaviour change, disclosed+tested) OR dead-removal; paper_trading_engine:368/370 schedule() call-arg fix (latent or behaviour change). Only execution/ + tests/.',
].join('\n')

phase('Review')

const risk = await agent(
  REVIEW_CONTEXT + '\nYou are the risk-execution-reviewer. Per fix verify: (1) kill_switch changes are type-only — engage/disengage/persist/file-lock LIFECYCLE byte-identical, no safety-logic change; (2) unified_paper_engine:142 removal is byte-identical (the function never existed, path was a masked no-op); (3) unified_paper_engine:106 — IF re-activated, store_ledger_events_parquet genuinely exists in ledger_store with a matching signature, the re-activated write is additive (no order/fill/position/state change), runs without error (tested), and the behaviour change is DISCLOSED; IF removed, byte-identical; (4) paper_trading_engine:368/370 — the corrected schedule() call maps args correctly to the real signature, and the live-vs-dead classification is correct (a re-activated previously-crashing method is disclosed); (5) NO real bug silenced with a broad ignore; deferred items still error in mypy; (6) no risk/accounting/pipeline/paper/.github path touched; tests green. Output YAML: stage, findings (BLOCKER/MAJOR/MINOR + file:line), per_bug_classification: {..}, behaviour_changes_disclosed_and_tested: yes|no, nothing_silenced: yes|no, VERDICT.',
  { label: 'review:risk', phase: 'Review', agentType: 'risk-execution-reviewer' }
)

const senior = await agent(
  REVIEW_CONTEXT + '\nYou are the senior-code-reviewer. Confirm each fix is correct + minimal for its class (type-only annotations change no logic; dead removals are truly dead; re-activation/correctness fixes target the real API with the right signature + a test); no broad silencing ignore; asserts have messages; deferred bugs clearly reported; new tests LISTED; no non-allowed path. Output YAML: stage, findings (BLOCKER/MAJOR/MINOR + file:line + fix), deferred: [..], untracked_files: [..], VERDICT.',
  { label: 'review:senior', phase: 'Review', agentType: 'senior-code-reviewer' }
)

phase('Audit')

const audit = await agent(
  REVIEW_CONTEXT +
  '\n--- STAGE-2 REVIEWS ---\nRISK:\n' + risk + '\n\nSENIOR:\n' + senior + '\n--- END REVIEWS ---\n\n' +
  'You are the task-completion-auditor. Decide if the execution/ real-bug batch is complete + safe: each of the 6 bugs is correctly fixed (type-only / byte-identical-dead-removal / disclosed-tested behaviour-change) OR honestly deferred (not silenced); kill-switch lifecycle byte-identical; any re-activation is additive + tested + disclosed; the schedule() fix maps args correctly; no real bug silenced; tests green; no non-allowed path. If risk reports behaviour_changes_disclosed_and_tested:no OR nothing_silenced:no, cannot be PASS. Note the resolved-vs-deferred mypy tally for the gate-flip prerequisite. settings.json deny-restore (execution) orchestrator-handled. Output YAML: stage: task-completion-auditor, verdict: PASS|CONDITIONAL|FAIL, verdict_reason, findings[], follow_ups[], mypy_remaining: N.',
  { label: 'audit:exec-realbugs', phase: 'Audit', agentType: 'task-completion-auditor' }
)

return { impl, risk, senior, audit }
