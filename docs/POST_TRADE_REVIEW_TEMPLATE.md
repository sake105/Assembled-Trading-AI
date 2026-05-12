# Post-Trade Review Template

> Audit C2-071 — one block per closed position. Separates **decision
> quality** (was the call defensible with what we knew?) from **outcome
> quality** (did it make money?) per Annie Duke's "Thinking in Bets".
>
> Goal: stop confusing skill with luck, both ways. A bad call that
> happened to print is still a bad call.

---

## How to write a review

1. Wait until the position is **fully closed** (final P&L locked in).
2. Copy the **Entry template** below into a new file
   `journal/YYYY-MM-DD-<symbol>-<slug>.md` (per audit C2-072 the journal
   lives under git so it survives).
3. Fill BOTH columns. The decision side is filled **using only the
   information available at entry time** — no hindsight allowed.
4. Score the outcome AFTER the position is closed.
5. Commit. No "I'll write it tomorrow" — write it the day the position
   closes or the lesson is gone.

---

## Entry template

```markdown
# Post-Trade Review — <symbol> — <slug> — <date opened>

## Position summary

- Symbol: `<TICKER>`
- Side: `<LONG|SHORT>`
- Entry: `<date>` @ `<price>` × `<qty>` (notional `<$>`)
- Exit:  `<date>` @ `<price>` × `<qty>` (notional `<$>`)
- Holding period: `<n>` days
- Realised P&L: `<$ amount>` (`<%>` of entry notional)

## Decision quality (filled at the time the position was opened)

| Question | Answer |
|---|---|
| Hypothesis / thesis | `<one sentence — why did we expect this to work?>` |
| Signal source | `<strategy name + run-id from RESEARCH_REGISTER>` |
| Confidence at entry | `<low / medium / high — what did the signal score say?>` |
| Sizing rationale | `<position size relative to portfolio + why this size>` |
| Stop / exit plan | `<predefined stop, time stop, signal flip>` |
| Risk if wrong | `<expected loss if stop hits>` |
| Pre-trade gate result | `<passed / blocked-reasons from pre_trade_gate()>` |
| Kill-switch state | `<engaged / disengaged / throttle_pct>` |

### Decision-quality score: `<good / mediocre / poor>`

`<one sentence — was the decision defensible given what we knew?>`

## Outcome quality (filled at close)

| Question | Answer |
|---|---|
| Did the thesis play out? | `<yes / partially / no — separate from P&L>` |
| What did we learn that we didn't know at entry? | `<one sentence — only NEW information>` |
| Was the stop hit? Triggered correctly? | `<yes/no — if yes, did the system honour it without intervention?>` |
| Was the exit signal honoured? | `<yes / no / overridden — and why>` |
| Slippage vs. expected (bps) | `<x bps>` |

### Outcome-quality score: `<lucky / well-executed / unlucky / poorly-executed>`

The 2×2:
- **good decision + good outcome** = well-executed → repeat.
- **good decision + bad outcome** = unlucky → repeat the process.
- **bad decision + good outcome** = lucky → fix the process anyway.
- **bad decision + bad outcome** = poorly-executed → fix the process AND
  understand which guardrail should have caught this.

## Process follow-ups

- [ ] `<concrete action: e.g. tighten sector cap, add data quality check, etc.>`
- [ ] `<concrete action>`

## Links

- Strategy: `<docs/STRATEGY_*.md>`
- Research registration: `<docs/research_register.md#run-id>`
- Order audit: `<output/ops/orders_audit.jsonl line range>`
```

---

## Anti-patterns this template prevents

1. **Hindsight bias**: the decision side is locked at entry; you cannot
   silently retcon the rationale after seeing the outcome.
2. **Outcome-only retrospectives**: a winning trade with a bad process
   still gets flagged for cleanup.
3. **Vague action items**: every follow-up is a checkbox tied to a
   concrete change, not "do better next time".

## What this is NOT

- Not a trading journal in the live-thoughts sense — that goes in
  `journal/<date>.md` with whatever structure you prefer.
- Not a performance attribution report — that's `output/qa/...` artefacts.
- Not a strategy specification — that's `docs/STRATEGY_<slug>.md`.
