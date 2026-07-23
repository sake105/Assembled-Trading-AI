export const meta = {
  name: 'wf-ops-reconcile-gate',
  description: 'Item 3 (B-acct-3): default-OFF, policy-gated next-cycle reconcile-blocking seam in the NON-protected ops/ layer (paper_runner). apply_reconcile_block_gate() reads the PRIOR cycle reconcile_latest.json; when armed: FAIL/unverified/unreadable -> block the next cycle. Default disabled -> byte-identical to live. ops/ + configs/ + tests/ — none in the 6 hard-deny zones, no deny-lift. MUST NOT be wired into backtest.',
  phases: [
    { title: 'Implement' },
    { title: 'Review' },
    { title: 'Audit' },
  ],
}

phase('Implement')

const IMPL_SPEC = [
  'You are implementing a SAFETY-policy seam in Assembled-Trading-AI (repo F:\\Python_Projekt\\Aktiengeruest, Windows, .venv\\Scripts\\python.exe). Files you MAY edit: src/assembled_core/ops/_paper_runner_gates.py, src/assembled_core/ops/paper_runner.py, configs/policy.yaml, and a NEW test under tests/. NONE of these are in the 6 hard-deny zones (execution/risk/accounting/pipeline/paper/.github) — confirm against .claude/settings.json before starting. You may NOT touch any hard-deny path. SAFETY-CRITICAL (pre-trade gate). Smallest safe change; model EXACTLY on the existing apply_tilt_gate. DEFAULT-OFF so live is byte-identical until opted in.',
  '',
  '=== CONTEXT (verified by scoping; re-confirm against source) ===',
  'FU-1 made a failed/unverified reconcile VISIBLE but nothing BLOCKS the next cycle. orchestrator.py:864-868 explicitly scopes next-cycle blocking out as a deliberate operational decision. The live paper driver ops/paper_runner.py computes reconcile AFTER trading (reconcile_status = report.get("status") ~:672-682) and never feeds it forward; its pre-cycle gate block (~:1244-1272) runs apply_halt_cache_gate / apply_tilt_gate but has NO reconcile gate. Durable artifact to gate on: output/reconcile_latest.json (schema run.reconcile.v1, status in {"OK","FAIL"}, written atomically by ops/reconcile.py). apply_tilt_gate in ops/_paper_runner_gates.py is the copy-shaped template (returns a blocked sentinel; paper_runner short-circuits with `return 0, "tilt_blocked"`).',
  '',
  '=== FIX 1 — apply_reconcile_block_gate() in ops/_paper_runner_gates.py ===',
  'READ apply_tilt_gate fully (its signature, the decision dataclass/sentinel it returns, how it reads config + root, how it logs). Add apply_reconcile_block_gate with the SAME shape. Behaviour (the POLICY, implement exactly):',
  '- Reads the gate config from paper_cfg["reconcile_block"] (or the policy path the other gates use). Keys: enabled (bool, DEFAULT False), block_on (list, default ["fail"]), artifact (str, default "output/reconcile_latest.json"). Optionally block_if_stale_hours (number|null, default null = disabled) for a freshness guard.',
  '- enabled False -> return a NOT-blocked decision immediately (pure pass-through, byte-identical). This is the only default.',
  '- enabled True (ARMED): load root/<artifact> JSON. Then:',
  '   * status == "OK" -> not blocked (unless block_if_stale_hours is set AND generated_utc is older than that -> blocked reason="reconcile_stale").',
  '   * status == "FAIL" -> blocked, reason="reconcile_fail" (ALWAYS when armed).',
  '   * artifact missing / unreadable / malformed JSON / no status field -> FAIL-CLOSED when armed: blocked, reason="reconcile_unverified" (an armed safety gate that cannot prove the last reconcile passed must not let the cycle trade).',
  '   * status is some unverified/other value -> blocked only if "unverified" in block_on, else not blocked.',
  '- Stamp the decision onto ctx (e.g. ctx.reconcile_gate_state) for observability, mirroring how the tilt/halt gates stamp state. Use a clear [RECONCILE-GATE] log prefix.',
  '- Do NOT raise on a normal blocked decision (return the sentinel like the tilt gate); only genuinely unexpected errors should surface — and even then fail-closed-when-armed (block), never fail-open.',
  '',
  '=== FIX 2 — call-site in ops/paper_runner.py ===',
  'READ the pre-cycle gate try-block (~:1244-1272) where apply_tilt_gate is called and how its blocked result short-circuits (the `return 0, "tilt_blocked"` shape ~:1270). After the tilt gate (and before run_trading_cycle), call apply_reconcile_block_gate(...); if decision.blocked -> log.warning("[RECONCILE-GATE] next cycle blocked: %s", reason) + `return 0, "reconcile_blocked"` (same 2-tuple shape as the tilt early-return). Pass the same root/output_dir + paper_cfg the other gates use. Confirm output_dir/root is resolvable at that point.',
  'CRITICAL: confirm paper_runner is the LIVE/PAPER driver only and is NOT used for mode=="backtest" (backtest uses qa/backtest_engine.py). The gate must NEVER run in backtest/replay (reading the current reconcile_latest.json in a historical replay would be a wrong-context read / determinism break — the same hazard guarded in _tc_risk.py). Since the gate lives in paper_runner, backtest is structurally excluded — VERIFY and state this; do NOT add the gate to qa/backtest_engine.py.',
  '',
  '=== FIX 3 — configs/policy.yaml ===',
  'Add under the existing paper_runner: section a reconcile_block: block — enabled: false, block_on: ["fail"], artifact: "output/reconcile_latest.json" (+ block_if_stale_hours: null if you implemented the freshness guard). Default-off. Confirm the YAML parses and matches how apply_reconcile_block_gate reads it. Do NOT change any other policy value.',
  '',
  '=== FIX 4 — tests (new tests/test_paper_runner_reconcile_block_gate.py) ===',
  'Unit tests for apply_reconcile_block_gate (construct a tmp root + write a reconcile_latest.json):',
  '(a) enabled false -> not blocked (pass-through), regardless of artifact contents.',
  '(b) enabled true + status "FAIL" -> blocked, reason reconcile_fail.',
  '(c) enabled true + status "OK" -> not blocked.',
  '(d) enabled true + artifact missing -> fail-closed blocked, reason reconcile_unverified.',
  '(e) enabled true + malformed JSON -> fail-closed blocked.',
  '(f) block_on ["fail","unverified"] honored vs default ["fail"].',
  '(g) (if implemented) block_if_stale_hours: a stale OK -> blocked reconcile_stale; a fresh OK -> not blocked.',
  'Tests must DISCRIMINATE (a wrong default, a fail-open on missing artifact, or a not-blocked on FAIL would fail them). Mirror existing tilt/halt gate test style if one exists.',
  '',
  '=== AFTER EDITING ===',
  '1. ruff format + ruff check --fix; final ruff check must pass on every edited/new file.',
  '2. Run the new test file + any existing _paper_runner_gates / paper_runner gate tests (-o addopts="" -p no:cacheprovider). Report EXACT pass/fail. Confirm the default-disabled path is a no-op (byte-identical).',
  '3. Do NOT git add / git commit. LIST the new untracked test file.',
  '',
  'OUTPUT (markdown): per FIX 1-4: what was implemented + trace evidence; the EXACT policy semantics (default off, armed=fail-closed on FAIL/unverified/unreadable, reads reconcile_latest.json only); CONFIRMATION the gate is never wired into backtest (paper_runner live/paper only) + byte-identical when disabled; whether any hard-deny path was touched (must be none); the new untracked test file; exact ruff + pytest; ALL files modified.',
].join('\n')

const impl = await agent(IMPL_SPEC, { label: 'implement:ops-gate', phase: 'Implement' })

const REVIEW_CONTEXT = [
  'A default-OFF next-cycle reconcile-blocking gate in the NON-protected ops/ layer just implemented in the MAIN working tree (uncommitted). Review ONLY the git diff (+ new untracked test via git ls-files --others). SAFETY-CRITICAL pre-trade gate.',
  '',
  '--- IMPLEMENTATION REPORT ---',
  impl,
  '--- END REPORT ---',
  '',
  'apply_reconcile_block_gate (ops/_paper_runner_gates.py) reads the PRIOR cycle reconcile_latest.json; armed -> FAIL/unverified/unreadable blocks the next cycle (fail-closed-when-armed), disabled -> pure pass-through (byte-identical, the only default). Call-site in ops/paper_runner.py pre-cycle block. policy.yaml reconcile_block default enabled:false. NEVER wired into backtest (paper_runner live/paper only). Only ops/ + configs/ + tests/ touched.',
].join('\n')

phase('Review')

const risk = await agent(
  REVIEW_CONTEXT + '\nYou are the risk-execution-reviewer. Verify: (1) DEFAULT enabled:false -> the gate is a pure no-op pass-through, live/paper byte-identical (no order/sizing/exec change) until an operator opts in; (2) ARMED semantics are FAIL-CLOSED: status FAIL blocks; missing/unreadable/malformed artifact blocks (reason reconcile_unverified) — NEVER fail-open when armed; status OK passes; block_on honored; (3) the gate reads the PRIOR persisted reconcile_latest.json (strictly past = PIT-correct) and is NEVER wired into backtest/replay (paper_runner is live/paper only; confirm no qa/backtest_engine.py edit, no current-artifact read in a historical context); (4) it does NOT cross-wire the EOD manifest reconciliation_ok=None (FU-1) truth source — reads reconcile_latest.json only; (5) NO hard-deny path touched (execution/risk/accounting/pipeline/paper/.github); the short-circuit returns the correct 2-tuple shape. Output YAML: stage, findings (BLOCKER/MAJOR/MINOR + file:line), default_off_byte_identical: yes|no, armed_fail_closed: yes|no, backtest_excluded: yes|no, VERDICT.',
  { label: 'review:risk', phase: 'Review', agentType: 'risk-execution-reviewer' }
)

const senior = await agent(
  REVIEW_CONTEXT + '\nYou are the senior-code-reviewer. Confirm: apply_reconcile_block_gate mirrors apply_tilt_gate shape correctly; the config read matches the policy.yaml keys exactly (no typo divergence); the call-site short-circuit is in the right place (pre-cycle, after tilt, before run_trading_cycle) with the correct 2-tuple; tests discriminate (cover disabled/FAIL/OK/missing/malformed/block_on); the new test file is LISTED; no hard-deny path; reversible. Output YAML: stage, findings (BLOCKER/MAJOR/MINOR + file:line + fix), untracked_files: [..], VERDICT.',
  { label: 'review:senior', phase: 'Review', agentType: 'senior-code-reviewer' }
)

phase('Audit')

const audit = await agent(
  REVIEW_CONTEXT +
  '\n--- STAGE-2 REVIEWS ---\nRISK:\n' + risk + '\n\nSENIOR:\n' + senior + '\n--- END REVIEWS ---\n\n' +
  'You are the task-completion-auditor. Decide if the ops reconcile gate is complete + safe: default-OFF byte-identical, armed=fail-closed (FAIL/unverified/unreadable all block, never fail-open), reads reconcile_latest.json only, NEVER wired into backtest, no hard-deny path touched, discriminating tests pass, new test listed. If risk reports default_off_byte_identical:no OR armed_fail_closed:no OR backtest_excluded:no, cannot be PASS. Output YAML: stage: task-completion-auditor, verdict: PASS|CONDITIONAL|FAIL, verdict_reason, findings[], follow_ups[].',
  { label: 'audit:ops-gate', phase: 'Audit', agentType: 'task-completion-auditor' }
)

return { impl, risk, senior, audit }
