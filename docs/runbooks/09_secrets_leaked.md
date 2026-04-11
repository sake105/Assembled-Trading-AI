# Runbook 09: Secrets Leaked

**Severity:** critical
**ETA to resolution:** 1–4 hours (rotation) + days (history rewrite, if chosen)
**On-call contact:** security + trading-ops
**Component:** `.env`, `.env.*`, CI secrets, any file in `configs/` that may have absorbed credentials

**Read `.claude/rules/20-security-and-secrets.md` before doing anything on this runbook.** It defines the mandatory behaviour when secrets may be exposed.

## Symptoms

- Credentials visible in a `git log -p` or `git show` over a current or historical commit.
- A secret scanner (`gitleaks`, `detect-secrets`, `trufflehog`) flags a file under version control.
- A partner or broker notifies that a key of ours appeared in a public place.
- `.env` or similar is staged or committed by accident.
- API calls begin failing with `unauthorized` after unexpected activity on the account — potential third-party use of a leaked key.

## Immediate Actions (first 5 min)

1. **Do not quote or reproduce the secret anywhere** — not in chat, not in a commit message, not in a log line. This includes copying it into a command for rotation; use the provider's UI.
2. **Stop the scheduler and any cron job that uses the leaked credential.** An attacker holding the key can act faster than a rotation.
3. Record the facts in `output/runs/_incident/secrets_leak_$(date +%s).md`:
   - which file
   - which provider
   - when the file first appeared in the repo (via `git log -- <file>`)
   - whether it is still in HEAD or only in history
   - without copying the secret value itself
4. Activate the global kill switch if the leaked key can submit orders:
   - `python -c "from src.assembled_core.execution.kill_switch import activate_kill_switch; activate_kill_switch(throttle_pct=100.0, reason='secret_rotation')"`

## Diagnosis

1. Which provider issued the key?
2. Is the key in the current working tree, in HEAD, or only in history?
3. Has any clone, fork, CI artifact, or public mirror ever had the affected commit?
4. Was the key used by services other than this repo (backtests, notebooks, personal scripts)?
5. Was it a read-only key or one with trading / withdrawal rights?

## Resolution

### Step 1: Rotate, always

Rotation is mandatory the moment a secret is on disk in a repo, regardless of whether the repo is public or private. Rotate via the provider's UI or admin API:

- Alpaca paper key + secret
- Polygon.io
- NewsAPI
- FRED
- Finnhub
- AlphaVantage
- any other provider listed in `.env` or the incident note

Document in the incident file: provider, time of rotation, who did it, confirmation that the old key is rejected. Do not paste the new key.

### Step 2: Remove the secret from the working tree

1. Add the path to `.gitignore` if it is not already.
2. Commit the removal from HEAD as a normal commit.
3. Understand that this does **not** remove it from history.

### Step 3: Decide about history rewrite

History rewrite is a destructive, force-push-heavy operation. It is a **project decision**, not an automatic step. Options:

- **Rewrite**: `git filter-repo` to purge the offending file from history, then force-push. Requires every clone to re-clone. Appropriate when the repo is public, or when compliance requires the history to be clean.
- **Accept and rotate**: leave the history alone, rely on the rotation. Appropriate when the repo is private, the team is small, and the key is dead.

Whichever option is chosen, record the decision and its justification in the incident file.

### Step 4: Tighten the gate

1. Ensure `.pre-commit-config.yaml` has a secret-scanner hook (`gitleaks` or `detect-secrets`).
2. Ensure the secret-scan CI job is **blocking** — advisory is not enough.
3. Review `settings.py` / config loader for a warning when `.env` sits at the repo root in a CI run.
4. Consider moving secrets to a real secret store (vault, 1Password CLI, doppler, GitHub Actions secrets) rather than a file.

## Post-Incident

- Write a post-mortem in `docs/post_mortems/YYYY-MM-DD_secrets_leak.md` covering:
  - what leaked
  - which providers
  - rotation timestamps
  - whether history was rewritten
  - how the scanner gate was tightened
- If the leaked key was used by an attacker, attach evidence and open a ticket with the affected provider.
- Audit `KNOWN_ISSUES.md` for any entry that suggested secrets were "ok to leave for now" — those entries are invalidated by this incident.
