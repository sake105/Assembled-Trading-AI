export const meta = {
  name: 'wf-pipeline-sibling-flags',
  description: 'Item 5: symmetric non-trip reset for the sibling intel health flags (intel_disclosures_triggers — REAL latent stale bug; intel_crisis_alpha + intel_market_stress — latent-inert), mirroring FU-2 daily_circuit_breaker. dataclasses.replace shares intel_health_flags by reference so a one-bar DEGRADED latches for the rest of a run; clear-at-top of _load_intel so each bar re-derives. Also correct the misleading FU-2 fresh-ctx comment. pipeline/ deny lifted.',
  phases: [
    { title: 'Implement' },
    { title: 'Review' },
    { title: 'Audit' },
  ],
}

phase('Implement')

const IMPL_SPEC = [
  'You are implementing a pipeline safety fix in Assembled-Trading-AI (repo F:\\Python_Projekt\\Aktiengeruest, Windows, .venv\\Scripts\\python.exe). src/assembled_core/pipeline/** deny is TEMPORARILY lifted — you MAY edit pipeline/. tests/** is NOT protected. You may NOT touch execution/risk/accounting/paper/.github. SAFETY-CRITICAL (intel health flags gate crisis-state escalation). Smallest safe change; confirm against current source first; live byte-identical except the stale-flag carry-over being closed.',
  '',
  '=== THE BUG (verified by scoping; re-confirm against source) ===',
  'FU-2 FIX2 (commit bdb8d0d1) added a non-trip-bar reset for ctx.intel_health_flags["daily_circuit_breaker"]["tripped"] in src/assembled_core/pipeline/trading_cycle_v2.py _load_intel. The THREE sibling intel health flags have the SAME structural asymmetry: each is set to "DEGRADED" on its FAILURE path and NEVER reset to healthy on the SUCCESS path. Because dataclasses.replace (qa/backtest_engine.py make_cycle_fn) does NOT deep-copy mutable fields, intel_health_flags (a field(default_factory=dict) in trading_cycle_shared.py) is SHARED BY REFERENCE across all bars of a run — so a "DEGRADED" set on ANY bar persists for the remainder of the run. Producer lines (verify current numbers):',
  '- intel_disclosures_triggers — set ~L249 (if not snap.generated_utc) and ~L254 (except path). REAL latent bug: consumers gate on it — risk/state_machine.py:302 (inside if require_confirm_now: forces disclosures_confirmed=False, blocking WATCH/COOLDOWN->ACTIVE) and risk/disclosures_confirm.py:39 (if _degraded == "DEGRADED": return — skips the confirm-boost). So one transient disclosures-load failure LATCHES the gate degraded for the whole run, suppressing crisis escalation.',
  '- intel_crisis_alpha — set ~L289 (except path only). Latent-INERT today: grep finds NO consumer reading intel_health_flags["intel_crisis_alpha"] (the generic _tc_sizing.py:1618 reader keys on "ERROR", not "DEGRADED").',
  '- intel_market_stress — set ~L316 (PIT-filter-failure path only). Latent-INERT today (same: no "DEGRADED" consumer; ctx.market_stress itself is recomputed fresh each bar).',
  '',
  '=== FIX ===',
  'READ _load_intel fully + confirm the FU-2 daily_circuit_breaker reset shape. At the START of _load_intel (before any of the three producers run), clear the three sibling keys so each bar re-derives them from scratch — mirroring the FU-2 non-trip reset. Recommended shape (use pop, NOT assign "OK": the codebase convention is healthy == key absent; getattr(...,{}).get(key) is None => not degraded; _tc_sizing only special-cases "ERROR"):',
  '  for _k in ("intel_disclosures_triggers", "intel_crisis_alpha", "intel_market_stress"):',
  '      ctx.intel_health_flags.pop(_k, None)',
  'Place it where the FU-2 daily_circuit_breaker handling is (or at the top of the flag-setting region), so on a reused-flags ctx each bar reflects ONLY that bar\'s load outcome. Leave the existing FU-2 daily_circuit_breaker reset AS-IS (do not refold it — keep this change minimal and separate from the accepted FU-2 logic). Do NOT change the producer except/failure branches (a failed load on THIS bar still sets "DEGRADED" after the clear — degraded handling preserved). Confirm the clear runs BEFORE the disclosures/crisis/market-stress producers and before _apply_risk_controls_default / state_machine / disclosures_confirm consume the flags within the same cycle.',
  '',
  '=== COMMENT CORRECTION ===',
  'The FU-2 comment (~trading_cycle_v2.py:391-393, and any mirror in trading_cycle_shared.py) asserting the canonical driver builds a FRESH TradingContext per as_of so a stale value "could never carry over" is INACCURATE for the flags dict — dataclasses.replace shares intel_health_flags BY REFERENCE, so carry-over IS possible even in the canonical driver. Correct the comment to state the accurate mechanism (shared-dict reference under replace; the per-bar clear is what guarantees freshness), without altering the FU-2 code behaviour.',
  '',
  '=== LIVE BYTE-IDENTICAL ARGUMENT (state it) ===',
  'On the single-cycle live/EOD/paper path, the loaders succeed and the flag was never set this cycle, so pop(...) is a no-op => live output byte-identical. On a live failure path the producer still sets "DEGRADED" after the clear, so degraded handling is preserved. The fix only removes CROSS-BAR leakage on a reused-flags ctx (backtest/replay) — strictly safer, PIT-correct (each bar reflects only its own load).',
  '',
  '=== TESTS (non-protected, tests/) ===',
  'New regression test: construct ONE TradingContext (or a replace-shared intel_health_flags dict), run _load_intel (or the relevant path) twice — bar 1 forces "DEGRADED" for intel_disclosures_triggers (e.g. a disclosures snap with no generated_utc, or monkeypatch the loader), bar 2 with healthy inputs; assert the flag is ABSENT/healthy on bar 2 AND that the state_machine disclosures-confirm gate is NOT latched (disclosures_confirmed not forced False on bar 2 by a bar-1 failure). Add a parallel assertion mirroring the existing FU-2 daily_circuit_breaker non-trip test for symmetry. Cover the inert siblings too (crisis_alpha/market_stress cleared on the healthy bar) so a future "DEGRADED"-sensitive consumer is protected. Tests must DISCRIMINATE (fail against the pre-fix sticky behaviour).',
  '',
  '=== AFTER EDITING ===',
  '1. ruff format + ruff check --fix; final ruff check must pass.',
  '2. Run the new test + existing trading_cycle / _load_intel / state_machine / disclosures_confirm / crisis suites (-o addopts="" -p no:cacheprovider). Report EXACT pass/fail. Confirm no regression to FU-2 daily_circuit_breaker or the crisis state machine.',
  '3. Do NOT git add / git commit. LIST any new untracked test file.',
  '',
  'OUTPUT (markdown): the clear-at-top diff + comment correction; confirmation the three producers + FU-2 daily_circuit_breaker reset are otherwise unchanged; the live-byte-identical argument; whether the disclosures-confirm latch is demonstrably broken pre-fix and fixed post-fix; new untracked test file; exact ruff + pytest; ALL files modified.',
].join('\n')

const impl = await agent(IMPL_SPEC, { label: 'implement:sibling-flags', phase: 'Implement' })

const REVIEW_CONTEXT = [
  'A pipeline sibling intel-health-flag reset (mirroring FU-2) just implemented in the MAIN working tree (uncommitted; pipeline/ deny temporarily lifted). Review ONLY the git diff (+ new untracked test). SAFETY-CRITICAL (flags gate crisis escalation).',
  '',
  '--- IMPLEMENTATION REPORT ---',
  impl,
  '--- END REPORT ---',
  '',
  'Clear-at-top of _load_intel pops intel_disclosures_triggers / intel_crisis_alpha / intel_market_stress so each bar re-derives them (the dataclasses.replace shared-dict carry-over that latches a one-bar DEGRADED for the whole run). disclosures_triggers is the real bug (latches the disclosures-confirm gate); the other two are latent-inert. Live byte-identical (pop is a no-op on the single-cycle happy path). FU-2 daily_circuit_breaker reset + producer failure branches unchanged. Only pipeline/ + tests/ touched.',
].join('\n')

phase('Review')

const risk = await agent(
  REVIEW_CONTEXT + '\nYou are the risk-execution-reviewer. Verify: (1) the per-bar clear removes the stale-DEGRADED carry-over on a reused/replace-shared intel_health_flags dict so a one-bar disclosures-load failure no longer latches the disclosures-confirm gate (state_machine.py:302 / disclosures_confirm.py:39) for the rest of a run; (2) LIVE byte-identical — on the single-cycle happy path the pop is a no-op (flag never set), and a live failure path still sets DEGRADED after the clear (degraded handling preserved); (3) the clear runs BEFORE the producers and before the flag consumers in the same cycle; (4) FU-2 daily_circuit_breaker reset + the three producer failure branches are UNCHANGED; (5) no execution/risk/accounting/paper/.github path touched; the comment correction is accurate (shared-dict-by-reference). Output YAML: stage, findings (BLOCKER/MAJOR/MINOR + file:line), stale_carryover_closed: yes|no, live_byte_identical: yes|no, VERDICT.',
  { label: 'review:risk', phase: 'Review', agentType: 'risk-execution-reviewer' }
)

const senior = await agent(
  REVIEW_CONTEXT + '\nYou are the senior-code-reviewer. Confirm: the clear-at-top is minimal + placed correctly (before producers/consumers), uses pop (healthy==absent convention) not a new "OK" string; FU-2 logic untouched; the test discriminates (fails against pre-fix sticky behaviour) + covers the real disclosures latch + the inert siblings; new untracked test LISTED; no non-allowed path; the corrected comment is accurate. Output YAML: stage, findings (BLOCKER/MAJOR/MINOR + file:line + fix), untracked_files: [..], VERDICT.',
  { label: 'review:senior', phase: 'Review', agentType: 'senior-code-reviewer' }
)

phase('Audit')

const audit = await agent(
  REVIEW_CONTEXT +
  '\n--- STAGE-2 REVIEWS ---\nRISK:\n' + risk + '\n\nSENIOR:\n' + senior + '\n--- END REVIEWS ---\n\n' +
  'You are the task-completion-auditor. Decide if Item 5 is complete + safe: the per-bar clear closes the stale-DEGRADED carry-over (disclosures latch demonstrably broken pre-fix, fixed post-fix), live byte-identical (no-op on happy single-cycle path), FU-2 + producer branches unchanged, comment corrected to the accurate shared-dict mechanism, discriminating tests pass, no non-allowed path. If risk reports stale_carryover_closed:no OR live_byte_identical:no, cannot be PASS. settings.json deny-restore (pipeline) orchestrator-handled. Output YAML: stage: task-completion-auditor, verdict: PASS|CONDITIONAL|FAIL, verdict_reason, findings[], follow_ups[].',
  { label: 'audit:sibling-flags', phase: 'Audit', agentType: 'task-completion-auditor' }
)

return { impl, risk, senior, audit }
