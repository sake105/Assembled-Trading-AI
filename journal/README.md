# Trading Journal

> Audit C2-072 — one short markdown file per trading day under
> `journal/YYYY-MM-DD-<slug>.md`. The journal lives in git so it
> survives. Use it to capture the **process** (what you decided and why)
> separately from the result.

## When to write

- At the end of any day on which **you made a discretionary call** —
  a manual override, a kill-switch activation, a config change you chose
  to deploy, a strategy you decided to promote / archive.
- After any **drill or incident** (see also `docs/FMEA.md`).
- After any **closed paper-pilot position** worth reflecting on
  (use `docs/POST_TRADE_REVIEW_TEMPLATE.md` for those).

You do not need to journal pure no-op days.

## Template

```markdown
# YYYY-MM-DD — <slug>

**State at start of day**: <kill_switch state, paper portfolio nav, drift_status>

**Decisions made today**:
- <decision 1> — rationale: <one sentence>
- <decision 2> — rationale: <one sentence>

**Surprises**:
- <what did the system do that you didn't expect>

**Open follow-ups**:
- [ ] <action item>

**Tomorrow's first task**: <one sentence>
```

## What this directory is NOT

- Not a place for trade post-mortems — those live alongside the journal
  but follow `docs/POST_TRADE_REVIEW_TEMPLATE.md`.
- Not a place for strategy notes — those go in `docs/STRATEGY_<slug>.md`.
- Not a place for ephemeral todo lists — use task tracking.
