# Incident: `.env` Tracked in Git, Exposed in History, Allowlisted in Scanner

**Status:** P0, partially mitigated — **keys still need rotation**
**Severity:** Institutional-Due-Diligence Killer
**Authoring finding:** System-Check Deep Run v2, A1 (2026-04-18)
**Related:** CLAUDE.md §20 Incident; `.claude/rules/20-security-and-secrets.md` §Incident

---

## 1. Triple-bypass summary

At the time this incident was authored, the project had three independent barriers to
secret exposure all simultaneously disabled:

| Barrier | Intended behavior | Actual behavior |
|---------|-------------------|-----------------|
| `.gitignore` | Prevent `.env` from being committed | **Cosmetic** — `.env` was committed before the ignore rule; `git ls-files .env` matches |
| Git history | Should contain no secrets | **`.env` has been present since at least Sprint-10 init** |
| `.gitleaks.toml` | Secret scanner flags exposed keys in PRs | **`\.env$` was allowlisted** with comment "scheduled for rotation" |

Each barrier individually would catch the leak. All three together fail open. Any
reviewer (auditor, investor, open-source contributor) running `git log -p -- .env` on a
clone or fork can read every key that was ever present in `.env`.

## 2. What actually changed today

- `.gitleaks.toml`: removed the `'''\.env$'''` allowlist path. The scanner now flags
  `.env` in every PR. This closes the **detection** barrier only. It does not rotate
  exposed keys and does not erase them from history.
- `docs/incidents/2026-04-18_env_exposure.md` (this file) — evidence record + decision
  log + rotation checklist.

## 3. What has **not** changed

- The tracked `.env` still exists in the working tree and in the index. Git history for
  `.env` still contains every past key value.
- No keys have been rotated. Every key that was ever committed is still valid at the
  provider until rotated.
- The history has **not** been rewritten. That is a separate project decision (see §5).

## 4. Rotation checklist (user action — cannot be done by the assistant)

The assistant cannot rotate third-party API keys. The user must:

- [ ] **Alpaca (paper)** — rotate API key + secret at
      https://app.alpaca.markets/paper/dashboard/overview → API Keys → Regenerate.
- [ ] **Alpaca (live)** if used — same flow on the live dashboard.
- [ ] **Alpha Vantage** — https://www.alphavantage.co/support/#api-key → request new key;
      old key invalidates automatically on request.
- [ ] **Finnhub** — https://finnhub.io/dashboard → regenerate token.
- [ ] **NewsAPI** — https://newsapi.org/account → regenerate key.
- [ ] **Polygon.io** (if used) — https://polygon.io/dashboard/keys → regenerate.
- [ ] **FRED** — https://fred.stlouisfed.org/docs/api/api_key.html → request new key.
- [ ] **Anthropic** — only if key was ever in `.env`. Verify via
      `git log -p -- .env | grep -i "anthropic\|sk-ant"`. If present: rotate at
      https://console.anthropic.com/settings/keys.

After rotation:

- [ ] Update local `.env` with new values. **Do not commit.**
- [ ] Confirm `.env` is untracked: `git status --porcelain -- .env` should output nothing.
      If it is still tracked, run `git rm --cached .env` and commit the removal.
- [ ] Verify the running scheduler / CI workflows pick up the new keys (e.g., trigger a
      paper-trading-ci dispatch and confirm Alpaca preflight succeeds with the new key).

## 5. History-rewrite decision (open)

Removing `.env` from the working tree + rotating keys is the **minimum** required to
close the immediate risk. It does **not** remove the keys from git history. Two mutually
exclusive paths:

**Path A — leave history intact (current default):**
- Pros: no disruption to clones, forks, hashes; no coordination cost.
- Cons: any future auditor sees the exposure on `git log -p`. Invasive for open-source
  publication or investor review.
- Acceptable only if: rotation is complete **and** the repo never becomes public /
  never goes through external review.

**Path B — rewrite history with `git filter-repo`:**
- Pros: `git log -p -- .env` shows empty; clean history for external review.
- Cons: destructive. All existing clones and forks must re-clone. All existing commit
  SHAs change. All open PR branches must be re-based.
- Only run with explicit user authorization (CLAUDE.md §9.2).
- Command sketch (for reference — do **not** execute without explicit go):
  ```bash
  git filter-repo --path .env --invert-paths
  git push --force-with-lease origin main
  # Followed by: inform every collaborator, invalidate CI caches, etc.
  ```

The assistant **will not execute Path B unilaterally**. This is a project decision.

## 6. Preventive controls (already in place after today)

- `.gitleaks.toml` — `.env$` allowlist removed. Future `.env` commits fail the scanner.
- `.github/workflows/secrets-scan.yml` — runs gitleaks + detect-secrets on every push
  and PR. Now effective for `.env`.

## 7. Preventive controls still missing

- **Pre-commit hook:** `pre-commit` with `gitleaks` stage to catch the leak before it
  ever reaches the remote. Recommended next step, not yet implemented (P1).
- **CI block on scanner failure:** verify the secrets-scan workflow is marked as a
  required check in branch protection. If it is advisory-only, the allowlist fix is
  cosmetic. (Check GitHub repo settings → Branches → Branch protection rules.)

## 8. Verification

After the user completes rotation, run locally:

```bash
gitleaks detect --config .gitleaks.toml --source . --verbose
```

Expected: either no leaks reported, OR a leak reported for `.env` if `.env` is still
tracked. If `.env` is untracked (the target end-state), gitleaks should scan the file
system and **report nothing from the tree**, but **still report historical leaks from
the git log** until history is rewritten.

A tracked-but-allowlist-removed `.env` will surface as a leak — that is correct
behavior and confirms the scanner is no longer neutered.
