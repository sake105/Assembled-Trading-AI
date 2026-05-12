# Research Dead-Ends

> Audit C2-048 — every research idea that **didn't work** gets a short
> note here. Purpose: stop re-running the same failed experiment six
> months later because nobody remembers it failed the first time.
>
> The list is checked by future research-PRs (CI hook planned —
> `scripts/qa/check_research_register.py`) — if a new idea matches an
> archived dead-end, the PR is asked to either reference the dead-end
> or explain what's different this time.

## When to file a dead-end

- A backtest beat the gates (Sharpe / DSR / PSR / PBO) and then **failed**
  the permutation test (audit C2-016) → dead-end.
- An OOS test (different universe or different regime) collapsed the
  edge → dead-end.
- A signal had look-ahead bias once leakage tests (audit C4-058) caught
  it → dead-end (unless the leak is fixable).
- A data source proved too unreliable or too costly to use → dead-end.

## Template

Create `research/dead_ends/YYYY-MM-DD-<slug>.md`:

```markdown
# <one-line hypothesis>

**Filed by**: <name> on YYYY-MM-DD
**Effort spent**: ~<hours>
**Original research-register entry**: <docs/research_register.md#run-id>

## What we tried

<two or three sentences — the strategy / signal / data source>

## Why it didn't work

<two or three sentences — root cause, NOT just "Sharpe was low">

## Concrete artefacts

- Code (now archived / removed): <git commit referencing the experiment>
- Failed backtest run-id: <id>
- Plots: <output/research/dead_ends/<slug>/...>

## When NOT to retry

- <condition 1: e.g. "as long as we are limited to yfinance EOD data">
- <condition 2: e.g. "if data is < 5 years">
- <condition 3: e.g. "without an intraday model of borrow costs">

## When TO retry (legitimate triggers)

- <condition: e.g. "if we get intraday tick data">
- <condition: e.g. "if borrow-cost data becomes free and reliable">
```

## What this directory is NOT

- Not a junk drawer — every entry must include "when NOT to retry" so
  it's actionable.
- Not a permanent rejection — every entry has retry conditions.
- Not where strategies that *worked but were retired* go — those belong
  in `docs/STRATEGY_<slug>.md` with status `archived`.
