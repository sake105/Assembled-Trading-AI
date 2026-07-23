export const meta = {
  name: 'wf-ops-reconcile-gate-fix',
  description: 'Fix-iteration for the ops reconcile-block gate: close the MAJOR fail-OPEN hole (F-RECON-1) where an armed gate with a present-but-empty/null/whitespace status field returns NOT blocked under default block_on, then re-review (risk confirms armed_fail_closed now fully holds). ops/ + tests/ only, no deny zone.',
  phases: [
    { title: 'Implement' },
    { title: 'Review' },
    { title: 'Audit' },
  ],
}

phase('Implement')

const IMPL_SPEC = [
  'You are applying a SAFETY fix-iteration to the (uncommitted) reconcile-block gate in Assembled-Trading-AI (repo F:\\Python_Projekt\\Aktiengeruest, Windows, .venv\\Scripts\\python.exe). The gate from the prior batch is in the MAIN working tree UNCOMMITTED. Files you MAY edit: src/assembled_core/ops/_paper_runner_gates.py and tests/test_paper_runner_reconcile_block_gate.py. NOTHING else (no hard-deny path; do NOT touch paper_runner.py call-site or configs/app.yaml — they are correct). SAFETY-CRITICAL.',
  '',
  '=== THE MAJOR TO FIX (F-RECON-1, confirmed by risk + auditor) ===',
  'apply_reconcile_block_gate fails OPEN when ARMED if reconcile_latest.json has a present-but-EMPTY/null/whitespace status field. The guard (~:352) only checks `"status" not in report`; a structurally-valid artifact with status:"" / status:null / status:"  " survives that guard, then ~:355 `str(report.get("status") or "").strip().upper()` normalizes to "" which is neither FAIL nor OK, so control falls to the "other" branch (~:375-378) and returns blocked=False under the default block_on=["fail"]. This is the exact corrupt / partial-write / schema-drift case the gate exists to defend against — an armed safety gate MUST fail CLOSED there.',
  '',
  '=== FIX ===',
  'READ apply_reconcile_block_gate fully first. Fold the empty/blank/null-status case into the unverified (fail-closed-when-armed) branch BEFORE the OK/FAIL/other classification. The reviewer-suggested shape (adapt to the actual code structure + helper names like _decide):',
  '  if not isinstance(report, dict) or not str(report.get("status") or "").strip():',
  '      return _decide(True, "reconcile_unverified", report, report)   # armed: cannot prove last reconcile passed',
  'Place this guard so it covers: missing artifact (already handled), unreadable/malformed JSON (already handled), AND now present-but-empty/null/whitespace status. After the guard, the existing OK/FAIL/other logic is unchanged. Confirm: armed + status FAIL -> reconcile_fail; armed + status OK -> pass; armed + status ""/null/"  " -> reconcile_unverified (NOW blocked); armed + a genuine other value like "WARN" -> still reconcile_other (blocked only if "unverified" in block_on, unchanged). Default-OFF path unchanged (still byte-identical no-op). Do NOT change the disabled path, the OK/FAIL branches, the stale-guard, or the call-site.',
  '',
  '=== TESTS ===',
  'In tests/test_paper_runner_reconcile_block_gate.py add DISCRIMINATING tests: armed + status:"" -> blocked reason reconcile_unverified; armed + status:null (JSON null) -> blocked reconcile_unverified; armed + status:"   " (whitespace) -> blocked reconcile_unverified. Each must FAIL against the pre-fix code (fail-open) and PASS after. Keep all existing tests passing. (Also note: the prior report miscounted tests as 13; the file had 12 — just ensure the final count is reported accurately.)',
  '',
  '=== AFTER EDITING ===',
  '1. ruff format + ruff check --fix; final ruff check must pass.',
  '2. Run the full test file (-o addopts="" -p no:cacheprovider). Report EXACT pass/fail + the new total count. Confirm the 3 new empty/null/whitespace tests would FAIL pre-fix (state why they discriminate).',
  '3. Do NOT git add / git commit.',
  '',
  'OUTPUT (markdown): the exact fix diff hunk; confirmation the empty/null/whitespace armed path now returns reconcile_unverified (blocked); confirmation OK/FAIL/disabled/stale paths are UNCHANGED; exact ruff + pytest with the corrected total count; files modified.',
].join('\n')

const impl = await agent(IMPL_SPEC, { label: 'implement:ops-gate-fix', phase: 'Implement' })

const REVIEW_CONTEXT = [
  'Fix-iteration for the reconcile-block gate just applied in the MAIN working tree (uncommitted, on top of the prior gate batch). The prior batch had a MAJOR (F-RECON-1): armed gate fails OPEN on empty/null/whitespace status. Review the FULL current gate diff (git diff + the new test file). SAFETY-CRITICAL.',
  '',
  '--- FIX-ITERATION REPORT ---',
  impl,
  '--- END REPORT ---',
  '',
  'The fix folds empty/blank/null status into the reconcile_unverified fail-closed-when-armed branch. Confirm the fail-open is fully closed and nothing else changed (OK/FAIL/disabled/stale/call-site all unchanged).',
].join('\n')

phase('Review')

const risk = await agent(
  REVIEW_CONTEXT + '\nYou are the risk-execution-reviewer. Verify the F-RECON-1 fail-open is CLOSED: armed + status ""/null/whitespace/non-dict now returns blocked=True reason=reconcile_unverified (fail-closed); armed + FAIL still blocks; armed + OK still passes; armed + genuine "WARN"/other still gated by block_on (unchanged); the DISABLED default path is still a byte-identical no-op (no ctx touch, no file I/O); the stale-guard + call-site + app.yaml are unchanged; no hard-deny path touched. Confirm the new tests genuinely discriminate (fail pre-fix). Output YAML: stage, findings (BLOCKER/MAJOR/MINOR + file:line), armed_fail_closed: yes|no (must be yes now), default_off_byte_identical: yes|no, VERDICT.',
  { label: 'review:risk', phase: 'Review', agentType: 'risk-execution-reviewer' }
)

const senior = await agent(
  REVIEW_CONTEXT + '\nYou are the senior-code-reviewer. Confirm the fix is minimal + correct (the empty/null guard is placed before OK/FAIL classification, uses the existing _decide helper, does not regress any other branch), tests discriminate + pass, no scope creep, no hard-deny path. Output YAML: stage, findings (BLOCKER/MAJOR/MINOR + file:line + fix), VERDICT.',
  { label: 'review:senior', phase: 'Review', agentType: 'senior-code-reviewer' }
)

phase('Audit')

const audit = await agent(
  REVIEW_CONTEXT +
  '\n--- STAGE-2 REVIEWS ---\nRISK:\n' + risk + '\n\nSENIOR:\n' + senior + '\n--- END REVIEWS ---\n\n' +
  'You are the task-completion-auditor. The prior verdict was CONDITIONAL on F-RECON-1 (armed fail-open on empty/null status). Decide if it is NOW resolved: armed_fail_closed fully holds (empty/null/whitespace -> reconcile_unverified blocked), default-OFF byte-identical preserved, no other branch regressed, discriminating tests pass, no hard-deny path. If risk reports armed_fail_closed:no, cannot be PASS. Output YAML: stage: task-completion-auditor, verdict: PASS|CONDITIONAL|FAIL, verdict_reason, findings[], follow_ups[].',
  { label: 'audit:ops-gate-fix', phase: 'Audit', agentType: 'task-completion-auditor' }
)

return { impl, risk, senior, audit }
