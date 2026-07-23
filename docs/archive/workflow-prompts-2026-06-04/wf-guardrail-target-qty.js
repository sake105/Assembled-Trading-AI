export const meta = {
  name: 'wf-guardrail-target-qty',
  description: 'Item 7 minimal guardrail (NON-protected, byte-identical to live): pin the emit-time target_qty==target_notional parity invariant + the documented contract that _tc_sizing overlays mutate ONLY target_qty (never target_notional), and document the REAL hazard = the dual notional/shares semantics of target_qty (notional pre-order_generation, shares after). tests/ + docs/ only — no src/ edit. This is the parity oracle that must exist before any future target_shares disambiguation.',
  phases: [
    { title: 'Implement' },
    { title: 'Review' },
    { title: 'Audit' },
  ],
}

phase('Implement')

const IMPL_SPEC = [
  'You are adding GUARDRAIL tests + a contract note in Assembled-Trading-AI (repo F:\\Python_Projekt\\Aktiengeruest, Windows, .venv\\Scripts\\python.exe). You may ONLY create files under tests/ and edit a docs/ contract file. Do NOT edit any src/ file (position_sizing.py / _tc_sizing.py / order_generation.py / exposure_engine.py / turnover_budget.py are all READ-ONLY here — read them to write accurate tests, never edit). This is byte-identical to live (tests + docs only). Confirm every asserted value against the CURRENT source.',
  '',
  '=== BACKGROUND (verified by scoping; re-confirm against source) ===',
  'The Diagnostik "~22 target_qty emitters / value-identical alias drift" framing is MISLEADING. Two facts:',
  '- FACT A: target_notional is WRITE-ONLY — emitted in src/assembled_core/portfolio/position_sizing.py and NEVER read as a DataFrame column anywhere in src/ (the only other literal occurrences are an unrelated local var in execution/pre_trade_checks.py and a log-string label in execution/order_generation.py). position_sizing.py header (~:24-26) documents this: target_notional is an emit-time honest-name marker, NOT maintained through overlays; do NOT read it post-overlay. So drift between the two columns post-overlay is BY DESIGN and harmless.',
  '- FACT B (the REAL latent hazard): target_qty carries TWO incompatible semantics by pipeline stage — NOTIONAL dollars pre-conversion (position_sizing + strategy emitters + every _tc_sizing overlay mutation = round(w*capital,2)) and SHARES post-conversion (risk/exposure_engine.py:112 target_qty = qty+order_delta; risk/turnover_budget.py:195-201 ramps vs share counts; execution/order_generation.py fast-path treats target_qty as shares). order_generation.generate_orders bridges them: reads target_qty as notional then divides by price -> shares (~order_generation.py:351). Same column name = notional in, shares out.',
  '',
  '=== FIX 1 — emit-time parity test (NEW tests/portfolio/test_position_sizing_parity.py) ===',
  'READ src/assembled_core/portfolio/position_sizing.py. Find every emit path that writes BOTH target_qty AND target_notional (the equal-weight / score / kelly / risk-parity / vol-scaled emit functions ~:144-145/306-307/421-422/534-535, the cap-renorm path ~:825-835, and the base-fn path ~:910-915 — verify the exact line numbers + function names in the current file). Write a parametrized test that, for each emit path, builds a minimal realistic input and asserts `(result["target_qty"] == result["target_notional"]).all()` immediately after emit (the ONLY place both columns coexist and are equal-by-construction). The test must DISCRIMINATE: it would fail if an emit path set the two columns to different values or omitted one. Assert the equality is exact (both are round(w*capital,2) notional dollars at emit). Do NOT assert parity post-overlay (that is intentionally false per FACT A).',
  '',
  '=== FIX 2 — overlay-only-mutates-target_qty test (NEW tests/pipeline/test_tc_sizing_target_qty_notional_drift.py) ===',
  'READ src/assembled_core/pipeline/_tc_sizing.py overlay mutation sites (the scale loops + hedge/news_alpha inserts ~:1108-1121/1170-1171/1262/1311-1312/2004-2048/2435-2509 — verify current lines). Pin FACT A: after a representative overlay path runs over an emitted frame, target_qty is mutated but target_notional is NOT silently re-read as an authoritative post-overlay value. The smallest robust assertion: pick a representative _tc_sizing overlay function (or a thin integration over one) and assert that it mutates target_qty (changes it) while target_notional is either absent from its working contract or left at the stale pre-overlay value (i.e. NOT re-synced) — so a FUTURE edit that starts reading/maintaining target_notional post-overlay (resurrecting the drift hazard) fails this test loudly. If a clean unit seam is hard to construct, assert the invariant at the smallest honest level you can (e.g. grep-style structural assertion is NOT acceptable — it must be a behavioral test). Document precisely what the test pins.',
  '',
  '=== FIX 3 — contract note (docs/) ===',
  'Document the REAL hazard = dual notional/shares semantics of target_qty. Prefer appending to an EXISTING contract doc — look for docs/CONTRACTS.md (or a similarly-named data-contract/schema doc); if none fits, create docs/target_qty_semantics.md. State clearly: target_qty is NOTIONAL dollars from position_sizing emit through all _tc_sizing overlays, and becomes SHARES at/after order_generation (which divides notional by price); target_notional is an emit-time honest-name marker only, write-only, must NOT be read post-overlay; the future clean fix is a distinct target_shares column at the order_generation boundary (separate larger initiative, not done here). Do NOT edit docs/Diagnostik.md (read-only). Keep it concise + accurate.',
  '',
  '=== AFTER EDITING ===',
  '1. ruff format + ruff check --fix on the new test files; final ruff check must pass.',
  '2. Run the 2 new test files (clear addopts: -o addopts="" -p no:cacheprovider). Report EXACT pass/fail. Confirm they pass against CURRENT source (they pin existing behavior).',
  '3. Do NOT git add / git commit. EXPLICITLY LIST the new untracked test files + any new doc file.',
  '',
  'OUTPUT (markdown): per FIX 1-3: what was asserted/documented + the exact source lines it pins (verified against current code); confirmation NO src/ file was edited; the LIST of new files; exact ruff + pytest; whether the tests discriminate (would fail if the invariant broke).',
].join('\n')

const impl = await agent(IMPL_SPEC, { label: 'implement:guardrail', phase: 'Implement' })

const REVIEW_CONTEXT = [
  'Guardrail batch (tests + docs only, byte-identical to live) just implemented in the MAIN working tree (uncommitted). Review ONLY the git diff (AND new untracked files via git ls-files --others). The tests pin existing target_qty/target_notional behavior.',
  '',
  '--- IMPLEMENTATION REPORT ---',
  impl,
  '--- END REPORT ---',
  '',
  'FIX1 emit-time target_qty==target_notional parity test; FIX2 _tc_sizing overlays mutate only target_qty (target_notional not re-synced post-overlay); FIX3 docs contract note on the dual notional/shares semantics of target_qty. No src/ edit.',
].join('\n')

phase('Review')

const risk = await agent(
  REVIEW_CONTEXT + '\nYou are the risk-execution-reviewer. Verify the tests pin the CORRECT contract and do not encode a WRONG invariant: (1) FIX1 asserts parity ONLY at emit (where both columns are equal-by-construction notional), NOT post-overlay; the asserted values match position_sizing source (round(w*capital,2)); (2) FIX2 correctly pins that _tc_sizing overlays mutate target_qty and do NOT maintain target_notional post-overlay (FACT A), so it would catch a future edit that resurrects the drift hazard; (3) FIX3 doc accurately states target_qty = notional pre-order_generation, shares after, and target_notional write-only; (4) NO src/ file changed (confirm git diff has no src/ edit) — this is byte-identical to live; (5) the tests are behavioral + discriminating, not trivially-true. Output YAML: stage, findings (BLOCKER/MAJOR/MINOR + file:line), correct_invariant_pinned: yes|no, no_src_change: yes|no, VERDICT.',
  { label: 'review:risk', phase: 'Review', agentType: 'risk-execution-reviewer' }
)

const senior = await agent(
  REVIEW_CONTEXT + '\nYou are the senior-code-reviewer. Confirm: the tests discriminate (would fail if parity or the overlay-only invariant broke), assert exact values traceable to source, mirror existing test style; the doc note is accurate + concise; new untracked files are LISTED (must be git add-ed); no src/ or other protected path touched; no Diagnostik.md edit. Output YAML: stage, findings (BLOCKER/MAJOR/MINOR + file:line + fix), untracked_files: [..], VERDICT.',
  { label: 'review:senior', phase: 'Review', agentType: 'senior-code-reviewer' }
)

phase('Audit')

const audit = await agent(
  REVIEW_CONTEXT +
  '\n--- STAGE-2 REVIEWS ---\nRISK:\n' + risk + '\n\nSENIOR:\n' + senior + '\n--- END REVIEWS ---\n\n' +
  'You are the task-completion-auditor. Decide if the guardrail is complete + safe: emit-time parity test + overlay-only test pin the CORRECT contract (not a wrong invariant), the doc states the real dual-semantics hazard, byte-identical to live (no src/ change), tests discriminate + pass, new files listed for git-add. If risk reports correct_invariant_pinned:no OR no_src_change:no, cannot be PASS. Output YAML: stage: task-completion-auditor, verdict: PASS|CONDITIONAL|FAIL, verdict_reason, findings[], follow_ups[].',
  { label: 'audit:guardrail', phase: 'Audit', agentType: 'task-completion-auditor' }
)

return { impl, risk, senior, audit }
