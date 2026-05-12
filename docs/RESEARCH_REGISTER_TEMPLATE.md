# Research Register Template

> Audit C2-047 — every research/backtest run that ends up on the roadmap
> or in a paper-track candidate MUST register itself here BEFORE the
> result is shared. This is the cheap version of the audit's
> `research_register.md` discipline: a flat append-only list, one block
> per registered run.

Goal: when someone asks "where did the Sharpe 2.4 number come from",
the answer is a path, a commit, a seed, and a data hash — not a Slack
screenshot. The block below is the minimum fingerprint that lets a
third party redo the run.

---

## How to register a run

1. Copy the **Entry template** at the bottom of this file.
2. Append it to `docs/research_register.md` (NOT this file — this is the
   template).
3. Fill EVERY field. `n/a` is allowed; empty strings are not.
4. Commit the entry **before** showing the result to anyone. If the
   result is later promoted to a candidate, the CI gate
   (`scripts/qa/check_research_register.py`, future addition) refuses any
   promotion PR whose backtest run is not registered.

---

## Entry template

```markdown
### YYYY-MM-DD — <slug>

**Run-ID:** `<uuid or backtest-id from script output>`
**Author:** `<name or handle>`
**Hypothesis:** <one sentence — what you expected to find>

**Code:**
- Repo: `Assembled-Trading-AI`
- Commit: `<git_sha>`
- Branch: `<branch>`
- Entrypoint: `<scripts/... command-line invocation>`

**Data:**
- Universe: `<watchlist file path or list>`
- Date range: `<from> → <to>`
- Data source(s): `<yahoo|polygon|alpaca|...>` (note Polygon/IBKR if paid)
- Data hash / DVC tag: `<dvc tag or sha256 of input parquet — n/a if not yet wired>`

**Environment:**
- Python: `<version>`
- numpy: `<version>`
- pandas: `<version>`
- scipy: `<version>` (if used)
- sklearn / lightgbm: `<version>` (if used)
- Determinism: seed=`<int>`; `set_deterministic` called? `<yes|no>`

**Result:**
- Sharpe: `<float>` (or NaN)
- CAGR: `<float>`
- MaxDD: `<float>`
- DSR / PSR / PBO: `<floats — required if claim is promotion-quality>`
- Permutation p-value: `<float — required for promotion>`
- Conclusion: `<one sentence>`

**Artefacts:**
- Equity curve: `<output/...csv>`
- Metrics JSON: `<output/...json>`
- Logs: `<logs/...log>`

**Status:** `exploratory | promoted | discarded | superseded-by-<run-id>`

**Notes:**
<freeform — caveats, known issues, follow-ups>
```

---

## Field rationale (audit references)

- `Data hash / DVC tag`: anchor against yfinance / vendor revisions
  (audit C2-045 — without it, "same backtest" can give different numbers
  six months later because the upstream price series changed).
- `seed` + `set_deterministic`: reproducibility (audit C2-049 / E-006).
- `Permutation p-value`: required by audit C2-016 before promotion.
- `DSR / PSR / PBO`: required by audit D-002/003/004 — promotion-gate
  values, not optional decoration.

## What this file is NOT

- Not a place to keep results that are still being iterated on — use a
  notebook for that and register the *final* run.
- Not a substitute for the actual code/data — it's an index pointing at
  them.
- Not where you summarise the strategy — that goes in
  `docs/STRATEGY_<slug>.md` (linked from the entry's `Notes`).
