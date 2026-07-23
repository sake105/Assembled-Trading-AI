export const meta = {
  name: 'fu4a-execution-followups',
  description: 'Follow-up 4a (execution): is_duplicate_error false-positive token-tightening (loose "order" substring -> word-boundary), get_order_status stub triage (Protocol/ABC vs reachable live stub), partial-throttle pass-through test (guard_orders 0<throttle<1 scaling), test_audit_additions ambient-state isolation. execution/ deny lifted; tests/ unprotected.',
  phases: [
    { title: 'Implement' },
    { title: 'Review' },
    { title: 'Audit' },
  ],
}

phase('Implement')

const IMPL_SPEC = [
  'You are implementing EXECUTION-area follow-ups in Assembled-Trading-AI (repo F:\\Python_Projekt\\Aktiengeruest, Windows, python at .venv\\Scripts\\python.exe). src/assembled_core/execution/** deny is TEMPORARILY lifted — you MAY edit execution/. tests/** is NOT protected. You may NOT touch risk/accounting/pipeline/paper/.github. SAFETY-CRITICAL (order idempotency + kill-switch). Smallest safe change; CONFIRM each finding against the CURRENT code first and report ALREADY-OK (with trace evidence) when the concern no longer holds — do NOT manufacture a change. Live behaviour for real broker signals must stay byte-identical except the over-broad false-positive being narrowed.',
  '',
  '=== FIX 1 (MINOR, false-positive tightening) — is_duplicate_error loose "order" substring ===',
  'src/assembled_core/execution/idempotency.py is_duplicate_error (~:67-125). The MAJOR false-NEGATIVE (real Alpaca "already exists"/"potential wash trade" not matching) is ALREADY FIXED — current code has has_order_ref = ("client_order_id" in msg or "order" in msg) and matches signatures (1)-(4). The RESIDUAL is the false-POSITIVE risk: `"order" in msg` is a raw SUBSTRING that also matches "border", "reorder", "recorder", "ordering", "disorder", "order book", etc. Combined with "duplicate"/"already exists" a benign/transient error could be misread as a duplicate. The docstring already notes a false-positive degrades to fail-safe re-raise (no fabricated order), so this is a HYGIENE tightening, not a safety hole — keep it minimal. Fix: change the bare `"order"` substring test to a WORD-BOUNDARY match (e.g. re.search(r"\\border\\b", msg)) so only the standalone word "order" (or "client_order_id", kept as the exact token) qualifies as an order/id reference. Preserve ALL FOUR accepted signatures for REAL broker strings: "duplicate order", "duplicate ... client_order_id", "order already exists", "client_order_id ... already exists", "422 ... already exists", "potential wash trade". Confirm with the existing tests that the real signatures STILL match; add cases proving "reorder"/"border"/"recording error" no longer false-trigger. Do NOT change the fail-safe re-raise composition in broker_adapter.',
  '',
  '=== FIX 2 (triage) — get_order_status stub ===',
  'src/assembled_core/execution/broker_adapter.py has get_order_status at ~:231 (body `...`) and a concrete Alpaca impl at ~:992 (real api.get_order_by_id). TRIAGE: determine the CLASS KIND of the :231 def — is it a typing.Protocol / abc.ABC abstractmethod (a contract, `...` is correct and never executed), or a CONCRETE base class that can be instantiated and whose `...` would silently return None to a LIVE caller? Grep ALL callers of get_order_status across the repo (src/ + scripts/ + ops/) and ALL classes that define/inherit it. If every concrete adapter implements it and no live/paper path can reach an unimplemented `...` stub -> report ALREADY-OK with the class-kind + caller evidence (NO code change). If a reachable live path CAN hit an unimplemented stub (silent None) -> smallest safe fix: raise NotImplementedError with a clear message (fail-loud) rather than returning None, OR implement if trivially derivable. Do NOT broaden scope to other adapter methods.',
  '',
  '=== FIX 3 (test coverage) — partial-throttle pass-through ===',
  'src/assembled_core/execution/kill_switch.py guard_orders (~:613-650): throttle_pct == 0.0 blocks ALL orders (empty frame); 0 < throttle_pct < 1 SCALES all order quantities by throttle_pct (partial pass-through); throttle_pct == 1.0 passes through unchanged. The prior B-exec test coverage exercised the block-all (0.0) path but the PARTIAL-throttle pass-through (e.g. 0.25 -> quantities scaled to 25%, orders still present, not dropped) was untested. ADD a test (tests/, NOT a protected edit) asserting: with the kill switch engaged at 0 < throttle_pct < 1, guard_orders RETURNS the orders with quantities scaled by throttle_pct (count preserved, qty * throttle_pct), and the audit record reflects throttled-not-blocked. If such a partial-throttle test ALREADY exists -> ALREADY-OK with the test path/name. Read guard_orders first to assert the EXACT scaling semantics (which qty column, rounding, sign handling). Do NOT edit kill_switch.py for this item unless the test reveals a real defect (if so, report it and stop for review rather than silently changing kill-switch behaviour).',
  '',
  '=== FIX 4 (test fragility) — test_audit_additions ambient-state ===',
  'tests/test_audit_additions.py (NOT protected). Some test(s) there depend on AMBIENT process state (env vars, cwd, a global singleton, module-level cache, or output/ files on disk) so they pass/fail depending on test ORDER or leftover state from other tests. READ the file; identify the ambient-state dependence; isolate it with monkeypatch / tmp_path / fixtures (set+restore env, chdir to tmp, reset the global) so each test is hermetic and order-independent. If the tests are ALREADY hermetic (no ambient dependence) -> ALREADY-OK with evidence (which fixtures already isolate state). Do NOT weaken any assertion to make a flaky test pass — isolate the STATE, keep the assertion.',
  '',
  '=== TESTS ===',
  '- FIX1: real broker duplicate strings ("order already exists", "potential wash trade", "duplicate order", "client_order_id ... already exists") still return True; "reorder cancelled", "border crossing", "recording error", a bare transient ("503 service unavailable", "timeout") return False.',
  '- FIX2: (if fixed) a live caller hitting an unimplemented adapter raises NotImplementedError not silent None; (if ALREADY-OK) no test needed, just evidence.',
  '- FIX3: guard_orders at throttle_pct in (0,1) scales qty + preserves orders (vs 0.0 block-all vs 1.0 pass-through).',
  '- FIX4: the previously-fragile test(s) now pass in isolation AND under reordering (e.g. run the file alone and after touching env).',
  'Put NEW tests in tests/test_fu4a_execution.py (+ extend tests/test_audit_additions.py in place for FIX4). Mirror existing style.',
  '',
  '=== AFTER EDITING ===',
  '1. ruff format + ruff check --fix; final ruff check must pass on every edited file.',
  '2. Run: new tests + existing idempotency + broker_adapter + kill_switch + test_audit_additions suites. Clear addopts if needed (-o addopts="" -p no:cacheprovider). Report EXACT pass/fail. Confirm no regression to B-exec idempotency/kill-switch behaviour.',
  '3. Do NOT git add / git commit. EXPLICITLY LIST any NEW (untracked) test files you create so they are not lost from the commit.',
  '',
  'OUTPUT (markdown): per FIX 1-4: CONFIRMED-and-fixed | ALREADY-OK | TRIAGE-RESULT, with trace evidence + exact behaviour-change scope (live byte-identical for real signals where claimed); whether subagent edits to execution/ were permitted; the LIST of new untracked test files; exact ruff + pytest; ALL files modified.',
].join('\n')

const impl = await agent(IMPL_SPEC, { label: 'implement:fu4a', phase: 'Implement' })

const REVIEW_CONTEXT = [
  'EXECUTION follow-up batch (user-authorized; execution/ deny temporarily lifted; tests/ unprotected) just implemented in the MAIN working tree (uncommitted). Review ONLY the git diff (AND check for new untracked test files via git ls-files --others). SAFETY-CRITICAL (order idempotency + kill switch).',
  '',
  '--- IMPLEMENTATION REPORT ---',
  impl,
  '--- END REPORT ---',
  '',
  'FIX1 is_duplicate_error: tighten loose "order" substring -> word-boundary (false-positive hygiene; real broker signals still match; fail-safe re-raise composition unchanged). FIX2 get_order_status: triage Protocol/ABC vs reachable live stub (fix only if a live path hits silent-None). FIX3 partial-throttle pass-through TEST only (no kill_switch.py behaviour change). FIX4 test_audit_additions ambient-state isolation (state isolated, assertions intact). Only execution/ (+ tests/) touched.',
].join('\n')

phase('Review')

const risk = await agent(
  REVIEW_CONTEXT + '\nYou are the risk-execution-reviewer. Verify: (1) FIX1 word-boundary tightening does NOT drop any REAL broker duplicate/idempotency signature (all four accepted forms still match) and only removes substring false-positives ("border"/"reorder"/"recorder"); the broker_adapter fail-safe re-raise composition is unchanged; false-positive still degrades to re-raise (no fabricated order). (2) FIX2 triage is correct: if ALREADY-OK, the :231 def is genuinely a Protocol/ABC contract with no reachable live silent-None; if fixed, it is fail-LOUD (NotImplementedError) not a behaviour change to a working path. (3) FIX3 is TEST-ONLY — kill_switch.py guard_orders is NOT modified (confirm git diff shows no kill_switch.py change); the test asserts the real scaling semantics. (4) FIX4 isolates STATE only — no assertion weakened. (5) no risk/accounting/pipeline/paper/.github path touched; no regression to B-exec idempotency/kill-switch. Output YAML: stage, findings (BLOCKER/MAJOR/MINOR + file:line), real_signals_preserved: yes|no, kill_switch_unmodified: yes|no, VERDICT.',
  { label: 'review:risk', phase: 'Review', agentType: 'risk-execution-reviewer' }
)

const senior = await agent(
  REVIEW_CONTEXT + '\nYou are the senior-code-reviewer. Confirm each fix is correct + minimal: the word-boundary regex is anchored correctly (does not accidentally exclude "client_order_id"); the triage conclusion is evidence-backed (class kind + callers); the partial-throttle test discriminates (would fail if scaling were broken); the audit-test isolation actually removes the ambient dependence (env/cwd/global restored); new untracked test files are LISTED (must be git add-ed); no non-allowed path; no regression. Output YAML: stage, findings (BLOCKER/MAJOR/MINOR + file:line + fix), untracked_test_files: [..], VERDICT.',
  { label: 'review:senior', phase: 'Review', agentType: 'senior-code-reviewer' }
)

phase('Audit')

const audit = await agent(
  REVIEW_CONTEXT +
  '\n--- STAGE-2 REVIEWS ---\nRISK:\n' + risk + '\n\nSENIOR:\n' + senior + '\n--- END REVIEWS ---\n\n' +
  'You are the task-completion-auditor. Decide if FU-4a is complete + safe: is_duplicate_error false-positive tightened without losing real signals; get_order_status triaged (ALREADY-OK with evidence OR fail-loud fix); partial-throttle pass-through TEST added (kill_switch.py unmodified); test_audit_additions made hermetic (assertions intact); discriminating tests pass; no non-allowed path; no regression. If risk reports real_signals_preserved:no OR kill_switch_unmodified:no, cannot be PASS. CONFIRM any new test file is listed so it gets git-added. settings.json deny-restore (execution) orchestrator-handled. Output YAML: stage: task-completion-auditor, verdict: PASS|CONDITIONAL|FAIL, verdict_reason, findings[], follow_ups[].',
  { label: 'audit:fu4a', phase: 'Audit', agentType: 'task-completion-auditor' }
)

return { impl, risk, senior, audit }
