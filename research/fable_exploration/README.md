# research/fable_exploration — Fable free edge-exploration (2026-06-13)

**Playground** for a fresh edge search against SPY / 60-40, using the new PIT data
(`data/raw/insider_congress/`, `data/raw/fundamentals/`). No form constraints HERE,
but the falsification discipline (Phase 3 "the mill") is non-negotiable.

This folder writes ONLY under itself. Nothing here touches protected paths
(execution/risk/accounting/pipeline/paper/workflows) — any production change a
surviving candidate needs goes through the Phase-4 integration gate WITH operator
sign-off, never from here.

## Method discipline (the constitution for this search)
- **Negative is a full result.** The closure verdict (`docs/PROJEKT_ABSCHLUSS_2026_05.md`)
  already stands: on this survivorship-biased, large/mid-cap, long-only universe,
  classical cross-sectional factors are dead. New data does not change the universe.
- **All known biases here flatter the strategy** (survivorship, same-bar potential,
  optimistic MaxDD historically). So a NULL result is robust; a POSITIVE one is
  suspect until PIT-universe + realistic costs + DSR are all in place.
- **PIT first.** Every signal uses `available_at` (EDGAR acceptance / disclosure
  date), never the economic event date. SUE σ is strictly past-only expanding.
- **Multiple testing is real.** Many candidates → Deflated Sharpe with an honest
  trial count. No threshold-tuning until it passes.
- **Cheap probe before expensive mill.** If a raw signal is absent even WITH leaks
  that flatter it, the candidate is dead and skips the mill.

## Binding data realities (verified 2026-06-13)
- Prices `output/aggregates/daily.parquet`: total-return-adjusted `close`,
  1984→2026, but only ~80–95 symbols have full ~2018–2026 history; 125/195 start
  2024. **Mill universe = the deep-history survivors, 2018–2026.**
- Fundamentals XBRL: 351,939 rows / 178 symbols, available_at 100%; 23,431
  quarterly diluted-EPS rows / 171 symbols (median 162 q/symbol). Real PEAD feasible.
- Insider Form 4: 838,277 rows / 260 symbols; **only 9,744 open-market BUYS (P)**,
  311k sells, 517k "unknown" (grants/exercises). transaction_date has junk future
  values (max 2050) → PIT via `available_at` only; raw needs cleaning.
- Congress: 25,735 rows, House-dominant, no-SLA GitHub mirrors. Buys 14,593.

## Layout
- `_scratch/` — throwaway inspection + probe scripts (peek_*, probe_*).
- `HYPOTHESES.md` — pre-registered, falsifiable hypotheses (Phase 2).
- `phase1_raw_signal_probe.md` — cheap GO/NO-GO probe results (Phase 1).
- `mill/` — the falsification harness (Phase 3): PIT portfolio backtest,
  walk-forward OOS, costs, DSR, vs SPY + 60/40.
- `ERGEBNIS.md` — final verdict + gate (Phase 4).
