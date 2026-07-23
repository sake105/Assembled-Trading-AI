export const meta = {
  name: 'wf-pipeline-sibling-flags-fix',
  description: 'Fix-iteration for the sibling-flag reset: the code change (clear-at-top of _load_intel) is correct + safe + eliminates the whole-run latch, but the COMMENT/CLAIM overstated same-cycle de-latch for the state-machine consumer (compute_next_state runs in ingest_data L170/L188 BEFORE _load_intel L195, so it reads the prior completed bar by existing cycle design), and the end-to-end test REVERSED production ordering. Correct the claims to match reality + replace the misleading test with an honest two-bar production-ordering test. Do NOT silently re-order the risk-state cycle (that is a separate scoped follow-up). pipeline/ deny lifted.',
  phases: [
    { title: 'Implement' },
    { title: 'Review' },
    { title: 'Audit' },
  ],
}

phase('Implement')

const IMPL_SPEC = [
  'You are applying an HONESTY/ACCURACY fix-iteration in Assembled-Trading-AI (repo F:\\Python_Projekt\\Aktiengeruest, Windows, .venv\\Scripts\\python.exe). The sibling-flag reset from the prior batch is in the MAIN working tree UNCOMMITTED. src/assembled_core/pipeline/** deny is TEMPORARILY lifted. Files you MAY edit: src/assembled_core/pipeline/trading_cycle_v2.py (comment only — see below), tests/test_fu2_sibling_intel_health_flags.py. You may NOT touch execution/risk/accounting/paper/.github. The CODE LOGIC of the clear is CORRECT and must NOT change — this iteration corrects the COMMENT and the TEST to match reality.',
  '',
  '=== WHAT IS CORRECT (do not change) ===',
  'The clear-at-top of _load_intel (pop of intel_disclosures_triggers / intel_crisis_alpha / intel_market_stress) is correct: it ELIMINATES the whole-run latch (pre-fix, a one-bar transient DEGRADED persisted for the WHOLE run on the dataclasses.replace shared-by-reference intel_health_flags dict). It also FULLY de-latches the apply_disclosures_confirm consumer (disclosures_confirm.py:39) SAME-CYCLE, because that consumer runs LATER inside _load_intel (~L461), after the clear+producer. Live byte-identical (pop is a no-op on the single-cycle happy path). KEEP the clear code, the pop convention, and the FU-2 untouched. KEEP the producer failure branches untouched.',
  '',
  '=== THE MAJOR TO FIX (accuracy, confirmed by risk + senior + auditor) ===',
  'The clear-at-top of _load_intel does NOT make the STATE-MACHINE consumer see THIS bar disclosures health. compute_next_state (which reads intel_health_flags["intel_disclosures_triggers"]=="DEGRADED" at risk/state_machine.py:302 to force disclosures_confirmed=False and hold WATCH/COOLDOWN, suppressing ACTIVE crisis escalation) is called in ingest_data at ~L170/L188 — BEFORE _load_intel at ~L195 — so it reads the PRIOR completed bar disclosures health (the existing cycle design: intel from the last completed _load_intel). The fix removes the pathological WHOLE-RUN persistence (each bar re-derives, so the value the state machine reads is at most one bar old, never latched forever) but does NOT change the existing prior-bar-intel-availability design. The prior batch comment + test OVERSTATED this as a same-cycle state-machine de-latch.',
  '',
  '=== FIX 1 — correct the comment in trading_cycle_v2.py (comment text ONLY) ===',
  'READ the clear-at-top comment block. Rewrite it to state ACCURATELY: (1) the clear eliminates the whole-run latch on the shared-by-reference intel_health_flags dict so each bar re-derives the flags from its own load outcome; (2) the apply_disclosures_confirm consumer (later in _load_intel) is de-latched THIS cycle; (3) the state-machine consumer compute_next_state runs in ingest_data (~L170/L188) BEFORE _load_intel (~L195), so by existing cycle design it reads the most-recent COMPLETED bar disclosures health — this fix removes the whole-run latch there too (no longer permanent), but does NOT make the state machine use same-bar disclosures health; that same-bar re-ordering is a SEPARATE scoped follow-up on the risk-state path. Do NOT alter the clear code itself or the FU-2 comment correction (which is accurate). Keep it concise.',
  '',
  '=== FIX 2 — fix the misleading test ===',
  'The test test_state_machine_disclosures_gate_not_latched_after_clear drives _load_intel FIRST then compute_next_state — the REVERSE of production ingest_data ordering — giving false assurance of same-cycle state-machine de-latch. Replace/rescope it:',
  '- RENAME it to reflect it tests the clear MECHANISM in isolation (e.g. test_clear_mechanism_resets_disclosures_flag_in_isolation) and add a comment that it does NOT reflect production ingest_data->_load_intel ordering, OR delete it if redundant with the other clear tests.',
  '- ADD an HONEST production-ordering regression test: exercise the REAL two/three-bar behaviour on a shared/replace-built intel_health_flags ctx that respects compute_next_state-before-_load_intel ordering (drive it at the ingest_data or run_trading_cycle level if feasible; if a full cycle is too heavy, simulate the exact ordering: bar N _load_intel sets DEGRADED; bar N+1 compute_next_state reads the shared dict (still sees DEGRADED = prior bar) THEN _load_intel(N+1) clears+re-derives healthy; bar N+2 compute_next_state reads (now absent/healthy)). ASSERT THE REAL BEHAVIOUR: the whole-run latch is GONE (the bar-N transient DEGRADED does NOT persist to bar N+2 state machine), while honestly showing the state machine reads prior-completed-bar intel. The test must DISCRIMINATE: pre-fix (no clear) the DEGRADED persists to every later bar; post-fix it clears. Do NOT assert a false same-cycle state-machine de-latch.',
  '- Correct the test-count: the file has 5 tests (prior report said 6). Ensure any count reference is accurate.',
  '',
  '=== AFTER EDITING ===',
  '1. ruff format + ruff check --fix; final ruff check must pass.',
  '2. Run the test file + test_fu2_pipeline_risk + test_risk_state_machine + test_disclosures (-o addopts="" -p no:cacheprovider). Report EXACT pass/fail + the corrected test count. Confirm the new production-ordering test discriminates (fails pre-fix).',
  '3. Do NOT git add / git commit.',
  '',
  'OUTPUT (markdown): the corrected comment; the test rescope + the new honest production-ordering test (what it asserts + how it discriminates + that it does NOT claim same-cycle state-machine de-latch); confirmation the CLEAR CODE + FU-2 + producers are unchanged; exact ruff + pytest with accurate count; files modified. Explicitly DISCLOSE the residual: the state machine reads prior-completed-bar disclosures health by existing design (one-bar availability), and the same-bar re-order is a separate follow-up.',
].join('\n')

const impl = await agent(IMPL_SPEC, { label: 'implement:sibling-flags-fix', phase: 'Implement' })

const REVIEW_CONTEXT = [
  'Accuracy fix-iteration for the sibling-flag reset, applied in the MAIN working tree (uncommitted; pipeline/ deny lifted). The prior batch had two MAJORs: an overstated same-cycle state-machine de-latch claim, and a test that reversed production ordering. This iteration corrects the comment + replaces the misleading test; the clear CODE is unchanged. Review the FULL current pipeline diff + the test.',
  '',
  '--- FIX-ITERATION REPORT ---',
  impl,
  '--- END REPORT ---',
  '',
  'The clear-at-top code is unchanged (correct: eliminates the whole-run latch, de-latches disclosures_confirm same-cycle, live byte-identical). The comment now accurately states the state-machine consumer reads prior-completed-bar intel (compute_next_state at ingest_data L170/L188 before _load_intel L195) and the same-bar re-order is a separate follow-up. The misleading test is rescoped + an honest production-ordering test added.',
].join('\n')

phase('Review')

const risk = await agent(
  REVIEW_CONTEXT + '\nYou are the risk-execution-reviewer. Verify: (1) the clear CODE + FU-2 + producer branches are UNCHANGED (comment-only edit in trading_cycle_v2.py); (2) the corrected comment now ACCURATELY describes the state-machine consumer reading prior-completed-bar disclosures health (compute_next_state before _load_intel) and does NOT claim a same-cycle state-machine de-latch; (3) the whole-run latch is genuinely eliminated and the disclosures_confirm consumer is de-latched same-cycle (both still true); (4) the new test exercises REAL production ordering and asserts the true behaviour (whole-run latch gone; state machine reads prior bar) and discriminates against pre-fix; the misleading reverse-ordering assurance is removed; (5) live byte-identical preserved; no execution/risk/accounting/paper/.github path touched. Output YAML: stage, findings (BLOCKER/MAJOR/MINOR + file:line), claims_match_reality: yes|no, clear_code_unchanged: yes|no, VERDICT.',
  { label: 'review:risk', phase: 'Review', agentType: 'risk-execution-reviewer' }
)

const senior = await agent(
  REVIEW_CONTEXT + '\nYou are the senior-code-reviewer. Confirm the two prior MAJORs are resolved: (F-senior-1) the comment no longer overstates the state-machine de-latch and correctly scopes the same-bar reorder as a follow-up; (F-senior-2) the test no longer reverses production ordering to give false assurance — it either honestly exercises ingest_data ordering or is clearly rescoped to isolation-only, and a real production-ordering discriminating test exists. Clear code unchanged, count accurate, new test LISTED, no non-allowed path. Output YAML: stage, findings (BLOCKER/MAJOR/MINOR + file:line + fix), untracked_files: [..], VERDICT.',
  { label: 'review:senior', phase: 'Review', agentType: 'senior-code-reviewer' }
)

phase('Audit')

const audit = await agent(
  REVIEW_CONTEXT +
  '\n--- STAGE-2 REVIEWS ---\nRISK:\n' + risk + '\n\nSENIOR:\n' + senior + '\n--- END REVIEWS ---\n\n' +
  'You are the task-completion-auditor. The prior verdict was CONDITIONAL on two MAJORs (overstated state-machine de-latch claim + reverse-ordering test). Decide if they are NOW resolved: claims match reality (whole-run latch eliminated + disclosures_confirm same-cycle de-latched, but state machine reads prior-completed-bar by existing design = honestly disclosed, same-bar reorder scoped as follow-up); the test exercises real ordering / is honestly rescoped + discriminates; clear code + FU-2 + producers unchanged; live byte-identical; no non-allowed path. If risk reports claims_match_reality:no OR clear_code_unchanged:no, cannot be PASS. Output YAML: stage: task-completion-auditor, verdict: PASS|CONDITIONAL|FAIL, verdict_reason, findings[], follow_ups[].',
  { label: 'audit:sibling-flags-fix', phase: 'Audit', agentType: 'task-completion-auditor' }
)

return { impl, risk, senior, audit }
