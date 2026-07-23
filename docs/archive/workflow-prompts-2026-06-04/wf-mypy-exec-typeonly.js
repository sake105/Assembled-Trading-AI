export const meta = {
  name: 'wf-mypy-exec-typeonly',
  description: 'Item D batch 7-8 (mypy execution/ TYPE-ONLY): clear the byte-identical execution/ mypy errors — unused-ignore (mechanical) + no-any-return/arg-type/assignment/union-attr (judgement, value-preserving annotations/casts/guards). DEFER the REAL-BUGs (paper_trading_engine:370 call-arg; unified_paper_engine:106/142 masked imports; kill_switch:80/126; any attr-defined that is a genuine bug) — leave their mypy error, do NOT silence. execution/ deny lifted; full risk-execution-reviewer chain. MOST SENSITIVE ZONE: zero order/fill/ledger/cost/risk behaviour change.',
  phases: [
    { title: 'Implement' },
    { title: 'Review' },
    { title: 'Audit' },
  ],
}

phase('Implement')

const IMPL_SPEC = [
  'You are clearing the TYPE-ONLY mypy errors in src/assembled_core/execution/ in Assembled-Trading-AI (repo F:\\Python_Projekt\\Aktiengeruest, Windows, .venv\\Scripts\\python.exe). src/assembled_core/execution/** deny is TEMPORARILY lifted. You may edit execution/ + tests/. Do NOT touch risk/accounting/pipeline/paper/.github. SAFETY-CRITICAL — execution/ is the MOST sensitive zone (orders, fills, ledger, cost, kill-switch). EVERY change must be TYPE-ONLY / byte-identical: NO change to any value, order qty/price, fill, ledger write, cost calc, kill-switch logic, control-flow, or signature behaviour. If a fix would change ANY of those, it is NOT type-only — DEFER it (it is the next REAL-BUG batch).',
  '',
  '=== RUN FIRST ===',
  '.venv\\Scripts\\python.exe -m mypy src/assembled_core/execution 2>&1 (the full CI command currently shows ~42 errors, all in execution/). Work ONLY the TYPE-ONLY classes here.',
  '',
  '=== DO (type-only) ===',
  '- unused-ignore: remove/narrow stale # type: ignore mypy flags as unused (restore + defer if removal surfaces a real error).',
  '- no-any-return: add the precise return-type annotation OR a value-preserving cast (typing.cast has ZERO runtime effect — PREFER cast over float()/int() in execution/ so no value can be truncated/rounded). NEVER use a cast that changes a number.',
  '- arg-type / assignment / union-attr / var-annotated / index / dict-item: precise annotation, or a None/empty guard that matches REAL runtime nullability (must not change a branch outcome). Use typing.cast or annotations, not value-changing edits.',
  '- If an assert is genuinely needed for narrowing in src/, it MUST have a message (repo policy: no bare assert in src). Prefer cast/annotation over assert.',
  '',
  '=== DEFER (do NOT fix here, do NOT silence) ===',
  'Leave the mypy error in place for these REAL-BUGs (they change behaviour / are genuine defects, handled in a dedicated risk-reviewed batch):',
  '- execution/paper_trading_engine.py:370 (call-arg ×2 — scheduler.schedule wrong signature, a real TypeError).',
  '- execution/unified_paper_engine.py:106 (attr-defined — wrong import accounting.ledger vs accounting.ledger_store, masked) and :142 (log_experience_entry undefined, masked).',
  '- execution/kill_switch.py:80/126 (whatever they are — kill-switch is safety-critical, do NOT touch in a type-only batch).',
  '- ANY other attr-defined / call-arg that reflects a genuine bug rather than a missing stub. Report each deferred item with its category + why it is a real bug not type-noise.',
  '(attr-defined that is purely a missing optional-dep stub — e.g. object-typed lazy clients — MAY be fixed type-only via an Any/ModuleType annotation if it changes NO runtime behaviour; but if unsure, DEFER.)',
  '',
  '=== AFTER EDITING ===',
  '1. Re-run mypy over execution/ + the full CI command; report execution/ BEFORE/AFTER own-file count and which errors REMAIN (should be exactly the deferred REAL-BUGs). Confirm NO new error, NO real bug silenced.',
  '2. ruff format + ruff check --fix; final ruff check must pass.',
  '3. Run execution/kill-switch/broker/idempotency/order/cost/paper-engine test suites (-o addopts="" -p no:cacheprovider). Report EXACT pass/fail. Separate pre-existing failures (git stash) from newly-introduced. ZERO new failures (type-only must not break anything).',
  '4. Do NOT git add / git commit. List files modified + any deferred real-bug.',
  '',
  'OUTPUT (markdown): the type-only fixes (file:line + category + why byte-identical, esp. that no order/fill/ledger/cost/kill-switch value changed); the LIST of DEFERRED real-bugs (with category + why real); execution/ BEFORE/AFTER mypy + remaining = deferred; ruff + pytest (no new failure); files modified.',
].join('\n')

const impl = await agent(IMPL_SPEC, { label: 'implement:mypy-exec-typeonly', phase: 'Implement' })

const REVIEW_CONTEXT = [
  'A TYPE-ONLY mypy execution/ batch (deny lifted) just implemented in the MAIN working tree (uncommitted). Review ONLY the git diff. execution/ is the MOST sensitive zone — every change must be byte-identical (no order/fill/ledger/cost/kill-switch/risk value or control-flow change). Real-bugs were DEFERRED (not fixed, not silenced).',
  '',
  '--- IMPLEMENTATION REPORT ---',
  impl,
  '--- END REPORT ---',
  '',
  'Type-only: unused-ignore + no-any-return (cast)/arg-type/assignment/union-attr annotations+guards in execution/. Real-bugs (paper_trading_engine:370, unified_paper_engine:106/142, kill_switch) deferred. Only execution/ + tests/.',
].join('\n')

phase('Review')

const risk = await agent(
  REVIEW_CONTEXT + '\nYou are the risk-execution-reviewer. Verify EVERY changed line in execution/ is byte-identical at runtime: NO change to any order qty/price, fill, ledger write/read, cost/slippage calc, kill-switch decision, idempotency/dup logic, or control-flow; casts are typing.cast (zero runtime) or value-preserving; guards match real nullability without changing a branch; NO real bug was silenced with a broad ignore (the deferred real-bugs still show their mypy error). Confirm the deferred list is genuinely the real-bugs (paper_trading_engine:370 etc.) and they are UNTOUCHED. No risk/accounting/pipeline/paper/.github path touched. execution tests green, no new failure. Output YAML: stage, findings (BLOCKER/MAJOR/MINOR + file:line), byte_identical: yes|no, real_bugs_deferred_not_silenced: yes|no, VERDICT.',
  { label: 'review:risk', phase: 'Review', agentType: 'risk-execution-reviewer' }
)

const senior = await agent(
  REVIEW_CONTEXT + '\nYou are the senior-code-reviewer. Confirm each fix is type-only (cast/annotation/guard, no value or control-flow change), removed ignores genuinely unused, no value-changing cast, asserts (if any) have messages; the deferred real-bugs are listed + untouched + not silenced; no non-allowed path; tests green. Output YAML: stage, findings (BLOCKER/MAJOR/MINOR + file:line + fix), byte_identical: yes|no, deferred_real_bugs: [..], VERDICT.',
  { label: 'review:senior', phase: 'Review', agentType: 'senior-code-reviewer' }
)

phase('Audit')

const audit = await agent(
  REVIEW_CONTEXT +
  '\n--- STAGE-2 REVIEWS ---\nRISK:\n' + risk + '\n\nSENIOR:\n' + senior + '\n--- END REVIEWS ---\n\n' +
  'You are the task-completion-auditor. Decide if the execution/ type-only batch is complete + safe: the type-only execution/ errors cleared byte-identically (no order/fill/ledger/cost/kill-switch value/control-flow change), the real-bugs DEFERRED (not fixed, not silenced — their mypy error remains), no new mypy error, execution tests green with no new failure, no non-allowed path. If risk reports byte_identical:no OR real_bugs_deferred_not_silenced:no, cannot be PASS. settings.json deny-restore (execution) orchestrator-handled. Output YAML: stage: task-completion-auditor, verdict: PASS|CONDITIONAL|FAIL, verdict_reason, findings[], follow_ups[].',
  { label: 'audit:mypy-exec-typeonly', phase: 'Audit', agentType: 'task-completion-auditor' }
)

return { impl, risk, senior, audit }
