export const meta = {
  name: 'fu4b-requests-cve-upgrade',
  description: 'Follow-up 4b (CI/security): upgrade requests 2.32.3 -> 2.32.4 (fixes CVE-2024-47081 .netrc credential leak), drop the now-obsolete --ignore-vuln CVE-2024-47081 waiver from backend-ci.yml pip-audit, update rationale comments. KEEP CVE-2026-25645 unless local pip-audit proves 2.32.4 also clears it. .github/workflows deny lifted; requirements.txt unprotected.',
  phases: [
    { title: 'Implement' },
    { title: 'Review' },
    { title: 'Audit' },
  ],
}

phase('Implement')

const IMPL_SPEC = [
  'You are implementing a CI/security dependency follow-up in Assembled-Trading-AI (repo F:\\Python_Projekt\\Aktiengeruest, Windows, python at .venv\\Scripts\\python.exe). The .github/workflows/** deny is TEMPORARILY lifted — you MAY edit .github/workflows/backend-ci.yml. requirements.txt is NOT protected. You may NOT touch any src/ path. This is a HOCHSENSIBLE CI change (security-scan gate) — be conservative, evidence-driven, and disclose CI-behaviour changes explicitly.',
  '',
  '=== CONTEXT (verified) ===',
  '- requirements.txt:16 pins `requests==2.32.3`. pyproject.toml:33 declares `requests>=2.32.0` (range ALREADY allows 2.32.4 — no pyproject change needed, no drift).',
  '- CVE-2024-47081 = requests .netrc credential leak, FIXED upstream in requests 2.32.4 (released 2025-06). This is the documented remediation.',
  '- backend-ci.yml runs a BLOCKING `pip-audit` security scan (Ubuntu job) with a long --ignore-vuln waiver list (~lines 112-129). Line ~120 is `--ignore-vuln CVE-2024-47081 \\`. Line ~119 is `--ignore-vuln CVE-2026-25645 \\` — a SEPARATE requests CVE ("upstream redirect issue") grouped with 47081 in the comment at ~lines 93-95. The in-file policy (lines 78-86) says every waiver is temporary, prefer upgrades, and remove the 47081 waiver once the pin is no longer affected.',
  '',
  '=== FIX A — bump the pin (requirements.txt, NOT protected) ===',
  'Change requirements.txt `requests==2.32.3` -> `requests==2.32.4`. Confirm this is the ONLY requests pin in the file and that no other requirement caps requests below 2.32.4 (grep). pyproject `>=2.32.0` already permits it.',
  '',
  '=== FIX B — local evidence (does 2.32.4 actually clear the CVEs?) ===',
  'Install the bumped pin into the venv and run pip-audit LOCALLY to get real evidence (this is the only way to verify the gate before CI):',
  '  .venv\\Scripts\\python.exe -m pip install "requests==2.32.4"',
  '  .venv\\Scripts\\python.exe -m pip_audit --desc 2>&1   (or: .venv\\Scripts\\python.exe -m pip install pip-audit then run it; if pip-audit is unavailable locally, report that and fall back to documented-fix evidence)',
  'From the pip-audit output determine: (1) is CVE-2024-47081 NO LONGER reported against requests 2.32.4? (expected: yes, cleared). (2) Is CVE-2026-25645 still reported against requests 2.32.4, or also cleared? Record the EXACT pip-audit lines for requests.',
  '',
  '=== FIX C — edit the waiver list + comments (backend-ci.yml, deny lifted) ===',
  'READ backend-ci.yml around the pip-audit block first (line numbers may have shifted). Then:',
  '1. REMOVE the entire `--ignore-vuln CVE-2024-47081 \\` line. CRITICAL YAML: it is part of a backslash-continued shell command — remove the WHOLE line including its trailing backslash, leaving the preceding continuation backslash intact so the command still chains correctly to the next --ignore-vuln. Do not leave a dangling or broken continuation.',
  '2. ONLY IF FIX B local pip-audit PROVES requests 2.32.4 ALSO clears CVE-2026-25645 (no longer reported): also remove the `--ignore-vuln CVE-2026-25645 \\` line (same YAML-continuation care). If pip-audit still reports 25645 (or pip-audit was unavailable / inconclusive), KEEP the CVE-2026-25645 waiver — do NOT remove a waiver you cannot prove obsolete (removing it would re-block CI).',
  '3. Update the rationale comments: change the ACTION ITEM block (~lines 83-86) to record that requests was upgraded to 2.32.4 and the CVE-2024-47081 waiver is REMOVED (resolved by upgrade). Update the requests bullet (~lines 93-95): if 25645 remains waived, keep its rationale and state 47081 is now resolved-by-upgrade; if 25645 was also removed, drop the whole requests bullet. Keep all OTHER waivers + comments byte-identical.',
  '',
  '=== VERIFY ===',
  '1. YAML validity: parse backend-ci.yml (python -c with yaml.safe_load, or yamllint if available) — must parse clean. Confirm the pip-audit command still has a well-formed backslash-continuation chain (no orphaned/missing `\\`).',
  '2. requests still imports + a quick smoke that the bump did not break runtime: .venv\\Scripts\\python.exe -c "import requests; print(requests.__version__)" must print 2.32.4. Run a SMALL targeted suite that exercises requests-using code if cheaply identifiable (e.g. API/broker adapter unit tests that do not hit the network) — report pass/fail; do NOT run the whole suite.',
  '3. Confirm NO other requests pin exists anywhere (requirements*.txt, pyproject, setup).',
  '',
  '=== AFTER EDITING ===',
  '- Do NOT git add / git commit.',
  '- State EXPLICITLY: which lines were removed from backend-ci.yml, the EXACT pip-audit evidence for requests (47081 cleared? 25645 cleared or kept?), the YAML-parse result, and the requests.__version__ smoke.',
  '',
  'OUTPUT (markdown): FIX A pin change; FIX B pip-audit evidence (verbatim requests lines + whether local pip-audit was available); FIX C exact removed line(s) + comment edits + decision on 25645 (removed only if proven clear, else kept with reason); YAML-validity result; requests version smoke; whether the backend-ci.yml edit was permitted (deny lift); ALL files modified. Be explicit that the pip-audit GATE result is only fully confirmed when CI runs (Ubuntu) — local pip-audit is strong but the authoritative gate is CI.',
].join('\n')

const impl = await agent(IMPL_SPEC, { label: 'implement:fu4b', phase: 'Implement' })

const REVIEW_CONTEXT = [
  'CI/security dependency follow-up (user-authorized; .github/workflows deny temporarily lifted; requirements.txt unprotected) just implemented in the MAIN working tree (uncommitted). Review ONLY the git diff. HOCHSENSIBLE CI CHANGE (blocking security-scan gate).',
  '',
  '--- IMPLEMENTATION REPORT ---',
  impl,
  '--- END REPORT ---',
  '',
  'Change: requests 2.32.3 -> 2.32.4 (fixes CVE-2024-47081); drop the --ignore-vuln CVE-2024-47081 waiver from backend-ci.yml pip-audit; KEEP CVE-2026-25645 unless local pip-audit proved 2.32.4 also clears it; update rationale comments. Only requirements.txt + backend-ci.yml touched.',
].join('\n')

phase('Review')

const ci = await agent(
  REVIEW_CONTEXT + '\nYou are the ci-debugger. Verify the CI impact rigorously: (1) backend-ci.yml still parses as valid YAML and the pip-audit shell command retains a well-formed backslash-continuation chain after the line removal (no orphaned `\\`, no two args merged onto one line, no missing continuation) — show the resulting pip-audit invocation. (2) ONLY CVE-2024-47081 was removed from the --ignore-vuln list (plus CVE-2026-25645 ONLY IF the report proves 2.32.4 clears it); every other waiver is byte-identical. (3) requirements.txt requests==2.32.4 is the sole requests pin and pyproject `>=2.32.0` permits it (no drift, no conflicting cap). (4) The pip-audit evidence: is CVE-2024-47081 genuinely cleared by 2.32.4? Is keeping/removing CVE-2026-25645 justified by the evidence (conservative = keep if unproven)? (5) State clearly that the authoritative gate result is the CI Ubuntu run, and what would make CI newly FAIL (e.g. 2.32.4 not clearing 47081 in pip-audit DB, or a broken continuation). Output YAML: stage, findings (BLOCKER/MAJOR/MINOR + file:line), yaml_valid: yes|no, only_intended_waivers_removed: yes|no, ci_break_risk: low|medium|high + why, VERDICT.',
  { label: 'review:ci', phase: 'Review', agentType: 'ci-debugger' }
)

const senior = await agent(
  REVIEW_CONTEXT + '\nYou are the senior-code-reviewer. Confirm: the pin bump is minimal + correct; the waiver removal is exactly scoped (47081 always; 25645 only if proven clear, else kept); the comment edits are accurate and do not leave stale/contradictory rationale; no src path touched; no other requests pin anywhere; the change is reversible. Output YAML: stage, findings (BLOCKER/MAJOR/MINOR + file:line + fix), VERDICT.',
  { label: 'review:senior', phase: 'Review', agentType: 'senior-code-reviewer' }
)

phase('Audit')

const audit = await agent(
  REVIEW_CONTEXT +
  '\n--- STAGE-2 REVIEWS ---\nCI-DEBUGGER:\n' + ci + '\n\nSENIOR:\n' + senior + '\n--- END REVIEWS ---\n\n' +
  'You are the task-completion-auditor. Decide if FU-4b is complete + safe: requests bumped to the CVE-2024-47081 fix version (2.32.4), the 47081 waiver removed, 25645 handled conservatively (kept unless proven clear), backend-ci.yml valid YAML with intact continuation, no drift, no other requests pin, comments accurate. If ci-debugger reports yaml_valid:no OR only_intended_waivers_removed:no OR ci_break_risk:high, cannot be PASS. Be explicit that the pip-audit gate is CI-authoritative (local pip-audit is supporting evidence, not a CI-green claim). settings.json deny-restore (.github/workflows) orchestrator-handled. Output YAML: stage: task-completion-auditor, verdict: PASS|CONDITIONAL|FAIL, verdict_reason, findings[], follow_ups[].',
  { label: 'audit:fu4b', phase: 'Audit', agentType: 'task-completion-auditor' }
)

return { impl, ci, senior, audit }
