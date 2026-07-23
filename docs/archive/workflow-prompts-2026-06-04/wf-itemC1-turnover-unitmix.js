export const meta = {
  name: 'wf-itemC1-turnover-unitmix',
  description: 'Item C1: fix the turnover_budget unit-mix latent bug (risk/turnover_budget.py:195-201 blends cq=current_positions[qty] SHARES with tq=incoming target_qty NOTIONAL via cq+scale*(tq-cq) — arithmetically wrong-by-units). FIRST confirm whether the mixed target_qty actually reaches order_generation (live cap-firing-day impact) or is re-derived from target_weight (latent). Add the equivalence test (price!=1.0, capital=100k) BEFORE editing. risk/ deny lifted; full risk-execution-reviewer chain. May change live order size on cap-firing days — disclose.',
  phases: [
    { title: 'Implement' },
    { title: 'Review' },
    { title: 'Audit' },
  ],
}

phase('Implement')

const IMPL_SPEC = [
  'You are fixing a unit-mix correctness bug in a RISK path in Assembled-Trading-AI (repo F:\\Python_Projekt\\Aktiengeruest, Windows, .venv\\Scripts\\python.exe). src/assembled_core/risk/** deny is TEMPORARILY lifted. You may edit src/assembled_core/risk/turnover_budget.py + tests/. Do NOT touch execution/accounting/pipeline/paper/.github (read them to trace data-flow, never edit). SAFETY-CRITICAL (order sizing). Confirm the live data-flow FIRST; add the equivalence test BEFORE the fix.',
  '',
  '=== THE BUG (verified by scoping; re-confirm) ===',
  'risk/turnover_budget.py ~:195-201 (apply_turnover_gate or similar): on a turnover-cap-firing path it computes the ramped position as `cq + scale*(tq - cq)` where cq = current_positions["qty"] (SHARES) and tq = the incoming target_qty. In the LIVE _tc_sizing flow target_qty is NOTIONAL dollars (= weight*capital). Blending shares (cq) with notional (tq) is arithmetically wrong-by-units. The same function ALSO correctly scales target_weight at ~:188-194.',
  '',
  '=== STEP 1 — CONFIRM LIVE IMPACT (decide before fixing) ===',
  'READ turnover_budget.py fully + trace how its OUTPUT frame is consumed: does execution/order_generation.py (the notional->shares boundary, ~:349-351 target_shares = target_qty / price) read the target_qty column that turnover_budget WRITES (-> the mixed value reaches live orders on cap-firing days = BEHAVIOUR-CHANGE), OR does order_generation / downstream re-derive target_qty from the correctly-scaled target_weight (-> the mixed target_qty is dead = the fix is BYTE-IDENTICAL/latent)? Also check the _tc_sizing turnover-gate wrapper (pipeline/_tc_sizing.py ~:1002-1067 forwards ctx.capital as portfolio_value) and whether current_positions carries shares or notional in the live call. REPORT the conclusion explicitly: is the mixed target_qty live-consumed (cap-firing days) or latent? This determines the behaviour-change scope.',
  '',
  '=== STEP 2 — EQUIVALENCE TEST FIRST (before the fix) ===',
  'Add a guard/equivalence test in tests/ that pins the CORRECT notional contract of the gated output: after apply_turnover_gate, the scaled target_qty must equal target_weight * portfolio_value element-wise (notional). USE price != 1.0 (e.g. 137.0) AND capital = 100_000 so a spurious factor-of-price (the exact bug: missing *price on cq) is EXPOSED. This test should FAIL against the current buggy mixed-unit code on a cap-firing case (prove it discriminates) and PASS after the fix. Also keep tests/test_turnover_budget.py, tests/test_risk_turnover_cap.py, tests/properties/test_turnover_gate_properties.py green.',
  '',
  '=== STEP 3 — FIX (smallest correct) ===',
  'Make the units consistent. PREFERRED: convert cq to NOTIONAL before blending (cq_notional = cq * price) so `cq_notional + scale*(tq - cq_notional)` is all-notional and matches the target_weight*portfolio_value contract — preserving the existing target_qty column meaning (notional). ALTERNATIVE (if price is not available at that point or the column is genuinely re-derived downstream): drop the target_qty mutation branch and let the correctly-scaled target_weight drive order-gen. Choose based on STEP 1. Do NOT change the target_weight scaling (~:188-194), which is already correct. Keep the change minimal + within turnover_budget.py.',
  '',
  '=== BEHAVIOUR DISCLOSURE ===',
  'State EXPLICITLY: is this byte-identical (latent fix) or does it change live order size on cap-firing days? If it changes live sizing, that is a deliberate CORRECTNESS fix (the prior value was wrong-by-a-factor-of-price on cap-firing days) — quantify the change direction. Non-cap-firing days (estimated <= effective_cap) must be byte-identical (the gate does not fire).',
  '',
  '=== AFTER EDITING ===',
  '1. ruff format + ruff check --fix; final ruff check must pass (NB: asserts in src/ need a message — but put assertions in TESTS, not src).',
  '2. Run tests/test_turnover_budget.py tests/test_risk_turnover_cap.py tests/properties/test_turnover_gate_properties.py + the new equivalence test (-o addopts="" -p no:cacheprovider). Report EXACT pass/fail. Confirm the new test FAILED pre-fix (discrimination) and PASSES post-fix. Confirm non-cap-firing path byte-identical.',
  '3. Do NOT git add / git commit. LIST new test files.',
  '',
  'OUTPUT (markdown): STEP-1 live-impact conclusion (live cap-firing-day change vs latent); the equivalence test (what it pins + discrimination proof); the fix (which option + why); the EXACT behaviour-change scope (byte-identical vs cap-firing-day sizing change + direction); exact ruff + pytest; files modified; new tests.',
].join('\n')

const impl = await agent(IMPL_SPEC, { label: 'implement:turnover-unitmix', phase: 'Implement' })

const REVIEW_CONTEXT = [
  'A turnover_budget unit-mix correctness fix (risk/ deny lifted) just implemented in the MAIN working tree (uncommitted). Review ONLY the git diff (+ new tests). SAFETY-CRITICAL (order sizing); may change live order size on cap-firing days.',
  '',
  '--- IMPLEMENTATION REPORT ---',
  impl,
  '--- END REPORT ---',
  '',
  'Fix: make the turnover-ramp units consistent (cq*price notional, or drop the dead target_qty branch) so the gated target_qty == target_weight*portfolio_value (notional). Equivalence test added with price!=1.0 + capital=100k to expose factor-of-price. Only risk/turnover_budget.py + tests/.',
].join('\n')

phase('Review')

const risk = await agent(
  REVIEW_CONTEXT + '\nYou are the risk-execution-reviewer. Verify: (1) the unit-mix is genuinely fixed — the gated/scaled target_qty is now consistently NOTIONAL (== target_weight*portfolio_value), no shares/notional blend remains; (2) the equivalence test uses price!=1.0 + capital!=1 and genuinely DISCRIMINATES (fails pre-fix on a cap-firing case); (3) the live-impact conclusion is correct (is the mixed target_qty actually consumed by order_generation on cap-firing days -> behaviour change, or latent -> byte-identical?) and is honestly disclosed; (4) NON-cap-firing path is byte-identical (the gate does not fire when estimated<=cap); (5) the target_weight scaling is unchanged; (6) no execution/accounting/pipeline/paper/.github path touched; existing turnover tests green. Output YAML: stage, findings (BLOCKER/MAJOR/MINOR + file:line), unit_mix_fixed: yes|no, behaviour_change_scope: byte-identical|cap-firing-days-only, noncap_byte_identical: yes|no, VERDICT.',
  { label: 'review:risk', phase: 'Review', agentType: 'risk-execution-reviewer' }
)

const senior = await agent(
  REVIEW_CONTEXT + '\nYou are the senior-code-reviewer. Confirm the fix is minimal + correct (units consistent; no off-by-price; target_weight branch untouched), the equivalence test discriminates + is well-formed, the behaviour-change scope is honestly reported, new tests LISTED, no non-allowed path. Output YAML: stage, findings (BLOCKER/MAJOR/MINOR + file:line + fix), untracked_files: [..], VERDICT.',
  { label: 'review:senior', phase: 'Review', agentType: 'senior-code-reviewer' }
)

phase('Audit')

const audit = await agent(
  REVIEW_CONTEXT +
  '\n--- STAGE-2 REVIEWS ---\nRISK:\n' + risk + '\n\nSENIOR:\n' + senior + '\n--- END REVIEWS ---\n\n' +
  'You are the task-completion-auditor. Decide if Item C1 is complete + safe: the unit-mix is fixed (gated target_qty consistently notional), the equivalence test discriminates + passes, the live-impact + behaviour-change scope is honestly disclosed (byte-identical vs cap-firing-day correctness change), non-cap-firing byte-identical, target_weight scaling unchanged, no non-allowed path. If risk reports unit_mix_fixed:no, cannot be PASS. A cap-firing-day sizing change is acceptable as a DELIBERATE correctness fix IF disclosed + tested. settings.json deny-restore (risk) orchestrator-handled. Output YAML: stage: task-completion-auditor, verdict: PASS|CONDITIONAL|FAIL, verdict_reason, findings[], follow_ups[].',
  { label: 'audit:turnover-unitmix', phase: 'Audit', agentType: 'task-completion-auditor' }
)

return { impl, risk, senior, audit }
