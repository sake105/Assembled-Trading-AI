export const meta = {
  name: 'wf-itemB-comment-pitfirewall',
  description: 'Item B (DOCUMENT-AS-INTENDED, byte-identical): correct the misleading comment at trading_cycle_v2.py:248-257 — the state machine does NOT read prior-bar disclosures health (the backtest driver carries nothing across bars; it reads template/None), and _load_intel is DELIBERATELY ordered AFTER compute_next_state because news_geo/disclosures_triggers are non-as_of live snapshots — reordering would inject a backtest look-ahead into the risk-state path (E-002). Mark the same-bar follow-up CLOSED/WONT-FIX-by-design. Comment-only. pipeline/ deny lifted.',
  phases: [
    { title: 'Implement' },
    { title: 'Review' },
    { title: 'Audit' },
  ],
}

phase('Implement')

const IMPL_SPEC = [
  'You are correcting a misleading SAFETY comment (comment text ONLY — no logic change) in Assembled-Trading-AI (repo F:\\Python_Projekt\\Aktiengeruest, Windows, .venv\\Scripts\\python.exe). src/assembled_core/pipeline/** deny is TEMPORARILY lifted. You may edit ONLY src/assembled_core/pipeline/trading_cycle_v2.py (comment text), and optionally add a NON-protected structural test under tests/. Do NOT change any code/logic, do NOT touch execution/risk/accounting/paper/.github. Byte-identical to live.',
  '',
  '=== THE FINDING (verified PIT analysis; re-confirm against source) ===',
  'In trading_cycle_v2.py, ingest_data calls compute_next_state (state machine) at ~:170 (ephemeral) and ~:188 (persisted) BEFORE _load_intel at ~:195. The comment block at ~:248-257 (which I wrote in commit 498c9216) claims the state machine reads "the most-recent COMPLETED bar disclosures health (intel is one bar old by availability design)". THAT IS IMPRECISE/WRONG. Actual behaviour:',
  '- The canonical backtest driver (qa/backtest_engine.py make_cycle_fn ~:344) builds each bar via dataclasses.replace WITHOUT passing news_geo / disclosures_triggers / crisis_state_intel. So at compute_next_state time on bar N those ctx fields hold the TEMPLATE value (normally None), NOT bar N-1 loaded intel. There is NO per-bar carry of loaded intel into the next bar state-machine read. The state machine runs one STAGE behind the rest of the bar, not one BAR behind.',
  '- More importantly, the geo/disclosures data _load_intel reads is NOT as_of-indexed: data/intel/crisis_state.json -> ctx.news_geo is a single live "latest" snapshot (no as_of slice); load_disclosures_triggers(path) (-> ctx.disclosures_triggers) takes only a path, NO as_of, does NO PIT filtering (single snapshot keyed by one generated_utc). Only ctx.market_stress is genuinely PIT-guarded.',
  '- Therefore reordering _load_intel BEFORE compute_next_state would feed a NON-as_of live snapshot directly into the risk state transitions (WATCH->ACTIVE/PAUSE) -> converting a latent downstream look-ahead into a look-ahead ON THE RISK-STATE PATH (anti-pattern E-002 class), in the most sensitive component. The current ordering is a DELIBERATE PIT FIREWALL.',
  '',
  '=== FIX (comment text only) ===',
  'READ the comment block at ~:248-257 (and the surrounding clear-at-top + the FU-2 comment). Rewrite the state-machine-consumer part to state ACCURATELY:',
  '1. compute_next_state runs in ingest_data (~:170/:188) BEFORE _load_intel (~:195). The canonical backtest driver does NOT carry loaded intel across bars (replace omits news_geo/disclosures_triggers/crisis_state_intel), so the state machine reads the TEMPLATE/None value, not a prior bar.',
  '2. _load_intel is DELIBERATELY ordered AFTER compute_next_state: news_geo (crisis_state.json) and disclosures_triggers (triggers_latest.json) are NON-as_of live snapshots (no PIT filtering); reordering _load_intel ahead of the state machine would inject today snapshot into historical bars = a backtest look-ahead on the risk-state path (E-002). Only market_stress is PIT-guarded. This ordering is a PIT FIREWALL, not an accident.',
  '3. The per-bar clear (this batch / the FU-2 sibling fix) still correctly eliminates the WHOLE-RUN DEGRADED latch for the downstream apply_disclosures_confirm consumer (same-cycle) — keep that part accurate.',
  '4. Mark the same-bar-state-machine-disclosures follow-up as CLOSED / WONT-FIX-by-design (NOT "separate scoped follow-up"). Note that genuine same-bar wiring would FIRST require as_of-indexed PIT-safe disclosures/crisis panels (a separate large feature), else it is a look-ahead.',
  'Keep it concise. Do NOT alter the clear-at-top code, the producers, or the FU-2 daily_circuit_breaker logic. Optionally correct the stale "~L461" line ref to the actual line if you touch that line.',
  '',
  '=== OPTIONAL structural guard test (tests/, non-protected) ===',
  'Optionally add a source-order/AST test asserting compute_next_state precedes _load_intel in ingest_data, so a future silent reorder (which would introduce the look-ahead) fails loudly. Keep it simple + robust (source-line-order or AST). Skip if it would be brittle.',
  '',
  '=== AFTER EDITING ===',
  '1. ruff format + ruff check (comment change should be format-neutral; fix if needed).',
  '2. Confirm via git diff that ONLY comment lines changed in trading_cycle_v2.py (no code/logic). If you added a test, run it.',
  '3. Do NOT git add / git commit. LIST any new test file.',
  '',
  'OUTPUT (markdown): the corrected comment (before/after gist); confirmation it is comment-only (git diff shows no code/logic change); the PIT-firewall rationale is accurately stated + the follow-up marked WONT-FIX-by-design; any new test; files modified.',
].join('\n')

const impl = await agent(IMPL_SPEC, { label: 'implement:itemB-comment', phase: 'Implement' })

const REVIEW_CONTEXT = [
  'A comment-only precision fix in trading_cycle_v2.py (pipeline/ deny lifted) just implemented in the MAIN working tree (uncommitted). Review ONLY the git diff. The change documents WHY _load_intel is deliberately ordered after compute_next_state (PIT firewall) and marks the same-bar reorder WONT-FIX-by-design. Byte-identical to live.',
  '',
  '--- IMPLEMENTATION REPORT ---',
  impl,
  '--- END REPORT ---',
  '',
  'Comment-only: corrects the false "prior-bar" framing (the backtest driver carries nothing across bars -> state machine reads template/None) and documents that reordering would inject a non-as_of live snapshot into the risk-state path (E-002 look-ahead). Only pipeline/ comment (+ optional tests/).',
].join('\n')

phase('Review')

const risk = await agent(
  REVIEW_CONTEXT + '\nYou are the risk-execution-reviewer. Verify the PIT reasoning is CORRECT and the comment now matches reality: (1) compute_next_state (~:170/:188) genuinely runs before _load_intel (~:195); (2) the backtest driver replace() does NOT carry news_geo/disclosures_triggers/crisis_state_intel across bars (so the state machine reads template/None, not prior-bar); (3) load_disclosures_triggers takes no as_of + crisis_state.json is a non-as_of snapshot, so reordering WOULD introduce a backtest look-ahead on the risk-state path (E-002) — the current order is a correct PIT firewall; (4) market_stress IS PIT-guarded (so the comment correctly excludes it); (5) the change is COMMENT-ONLY (git diff shows no code/logic change), byte-identical; (6) no execution/risk/accounting/paper/.github path touched. Output YAML: stage, findings (BLOCKER/MAJOR/MINOR + file:line), pit_reasoning_correct: yes|no, comment_only_byte_identical: yes|no, VERDICT.',
  { label: 'review:risk', phase: 'Review', agentType: 'risk-execution-reviewer' }
)

const senior = await agent(
  REVIEW_CONTEXT + '\nYou are the senior-code-reviewer. Confirm: comment-only (no logic change in the diff); the corrected comment is accurate + concise + does not overstate; the follow-up is marked WONT-FIX-by-design with the as_of-panel prerequisite noted; any new structural test is robust not brittle + LISTED; no non-allowed path. Output YAML: stage, findings (BLOCKER/MAJOR/MINOR + file:line + fix), untracked_files: [..], VERDICT.',
  { label: 'review:senior', phase: 'Review', agentType: 'senior-code-reviewer' }
)

phase('Audit')

const audit = await agent(
  REVIEW_CONTEXT +
  '\n--- STAGE-2 REVIEWS ---\nRISK:\n' + risk + '\n\nSENIOR:\n' + senior + '\n--- END REVIEWS ---\n\n' +
  'You are the task-completion-auditor. Decide if Item B is complete: the comment accurately describes the PIT firewall (state machine reads template/None not prior-bar; reordering would inject a non-as_of snapshot = E-002 look-ahead; market_stress PIT-guarded), the same-bar follow-up is marked WONT-FIX-by-design, the change is comment-only byte-identical, no non-allowed path. If risk reports pit_reasoning_correct:no OR comment_only_byte_identical:no, cannot be PASS. settings.json deny-restore (pipeline) orchestrator-handled. Output YAML: stage: task-completion-auditor, verdict: PASS|CONDITIONAL|FAIL, verdict_reason, findings[], follow_ups[].',
  { label: 'audit:itemB-comment', phase: 'Audit', agentType: 'task-completion-auditor' }
)

return { impl, risk, senior, audit }
