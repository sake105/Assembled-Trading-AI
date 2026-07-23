# ERGEBNIS — Fable free edge-exploration (2026-06-13)

**Bottom line: no candidate cleanly survived the falsification mill. The closure
verdict (passive core + `vol_target_overlay`) stands.** One hypothesis (insider
open-market buys) is real in-sample and is the *first* candidate to even partially
survive — but its apparent edge is survivorship-, concentration-, and early-period-
driven, and is ABSENT in the investable slice. It does not meet the live bar (§4.4
of `docs/PROJEKT_ABSCHLUSS_2026_05.md`). **No production integration is proposed.**

Scope honesty: local one-shot research, NOT CI-confirmed. Survivor-only universe
(53 deep-history names). All biases flatter the strategy → these NULLs are robust;
the one partial-positive is suspect by construction. Protected paths untouched.

---

## What was tested

| ID | Hypothesis | Phase-1 raw | Mill | Verdict |
|----|-----------|-------------|------|---------|
| H1 | Insider open-market BUYS (Form 4 'P'), long basket | +4.28%/60d excess, t=3.73 | full battery | **REAL in-sample, FAILS deployable bar** |
| H2 | Large congressional buys (>$50k), 60d | +2.19%/60d, t=2.11 | not advanced | marginal; trial-cost not justified |
| H3 | PEAD / SUE drift | +5d IC +0.04, 20/60d wrong-sign | not advanced | **NULL** (drift absent/inverted on large-cap survivors) |
| H4 | Earnings-announcement premium (pre-window basket) | pre +0.50%, t=3.88 | full mill | **DEAD as basket** (below SPY @10bps, fails DSR@33) |
| H5 | Aggregate insider net-buy → SPY market timing | IC +0.137 | full mill | **DEAD** (worse than constant-exposure control) |
| H6 | Short-flow level (FINRA RegSHO) long-low tilt | IC −0.044@20d (t=−7.4) | full mill | **DEAD** (worse than EW-universe; L/S Sharpe −0.07; fails DSR@34) |

## H1 in detail — the mill (universe 52 names, 2018-01→2026-06, net of costs)

Benchmarks: SPY Sharpe 0.775 (CAGR +13.9%, MaxDD −33.7%); 60/40 Sharpe 0.667.
Base variant (lookback 63d, weekly rebal, all P-buys), net of 20bps round-trip:
- **Sharpe 1.134, CAGR +30.5%, MaxDD −30.1%**, avg breadth ~9 names, 8/9 folds beat SPY.
- 2×-cost Sharpe 1.097; DSR (n_trials=23, var-across-grid) prob 0.974, passes 5%.
- CSCV-PBO across the 12-variant grid = **0.393** (selection not overfit).

These headline numbers are genuine and would, taken alone, clear several §4.4 lines.
The robustness battery is what kills them as a *deployable* edge:

| Falsification cut | Result | Interpretation |
|---|---|---|
| Drop top-3 PnL names (LLY,ENPH,SMCI) | Sharpe **0.788 ≈ SPY** | edge is 3-of-52 names deep, not broad |
| High-liquidity third @20bps | **0.563 < SPY** | investable/least-biased slice has NO edge |
| Low-liquidity third @80bps | 0.855 | edge lives where survivorship+capacity bias are worst |
| Recent half 2022H2–2026 | **0.968 vs SPY 0.967** | decayed to a dead heat |
| Cost stress 100bps | 0.988 | cost-robust (NOT the binding constraint) |

## Why it fails the live bar (§4.4)
- **MaxDD −30%** > the −20% limit. ✗
- **Survivorship-clean / external data** required — unmet, and the edge concentrates
  exactly in the illiquid names where survivorship inflation is largest. ✗ (binding)
- **Investable Sharpe**: the high-liquidity slice (0.56) is below SPY. ✗
- Independent replication on a separate dataset — none. ✗
The DSR/PBO passes are *necessary not sufficient*: they bound variant-selection
overfitting, not survivorship or name-concentration, which are the actual wounds.

## Honest forward pointer (not a recommendation to act now)
H1 is the **only** hypothesis across all 11 strategies tried (10 closure + this) that
is DSR-significant, cost-robust, and PBO-acceptable at the full-universe level. That
distinguishes it from the 10 cleanly-dead strategies. IF the project ever obtains
survivorship-clean, delisting-inclusive data (CRSP / Sharadar), the insider open-
market-buy effect is the single pre-registered hypothesis worth ONE clean retest —
specifically asking whether the edge survives in the high-liquidity, capacity-
realistic slice once bankrupt names are present (the expectation, per the liquidity
split, is that it weakens further). On the data we have, it does not survive.

## Round 2 (continuation "dann weiter") — timing/market-level ideas that DODGE the
## concentration/illiquidity/survivorship critique

Pre-registered (HYPOTHESES.md H4/H5) BEFORE testing. Both died under proper controls:

**H4 earnings-announcement premium** (Frazzini-Lamont). Real average effect — pre-window
[-5,-1] SPY-excess +0.50%, t=3.88 (n=2538) — but DECAYING (t=2.6/3.3 in 2018/2020 →
t<1.4 in 2024-26). As a tradeable ~5-name daily-rebalanced basket: Sharpe 0.74@10bps /
0.57@20bps / 0.21@40bps — **below SPY 0.775 at every realistic cost**, MaxDD −42%
(worse than SPY), 3/9 folds, DSR fails at n_trials=33 (prob 0.31). Turnover eats the
premium; the basket adds vol without risk-adjusted return.

**H5 aggregate insider net-buy ratio → SPY market timing** (Seyhun). The ONE idea that
dodges every killer bias (trades SPY only). Raw daily IC looked strong (+0.137) BUT:
- the raw signal is NON-STATIONARY (insider selling grew structurally); the strong
  result was an expanding-median artifact.
- DETRENDED (rolling z, past-only): insider-timed Sharpe **0.572 < SPY 0.792**.
- **CONTROL is decisive:** a dumb CONSTANT 41% SPY gives Sharpe 0.792 / MaxDD −15%,
  while the insider-TIMED 41%-average gives 0.572 / −24% — the signal has NEGATIVE
  timing value (worse than not timing).
- adds negative value over a plain vol-target overlay (0.78 vs 0.88 standalone);
  survives neither exclude-2020 (0.574) nor monthly significance (t=1.65) nor DSR.
- NB: SPY in `daily.parquet` starts 2018 (not 2004) — the longer-history hope was wrong.

## The cleanest cross-cutting summary
Across BOTH rounds (5 hypotheses, ~33 trials), **NOTHING passes the Deflated Sharpe at
the honest cumulative trial count.** Every signal with raw life dissolves under exactly
one of: realistic costs (H4), name-concentration + survivorship (H1), a constant-exposure
control (H5), or wrong-sign/no-stability (H3). This is the closure pattern reproduced with
NEW data AND the structurally-different families (event-driven, market-timing, defensive
overlay) that the Überprüfung named as the only remaining "maybe". They died too — which
makes the closure verdict STRONGER, not just unchanged.

## Round 3 (continuation "hole was sinn macht") — fetch new free data, test it

Free-data reachability was probed empirically (not assumed):

| source | status | verdict |
|--------|--------|---------|
| yfinance | reachable, **survivorship-biased** | delisted tickers return EMPTY (confirms the hole) |
| Stooq | **anti-bot JS-wall** | free delisted prices NOT obtainable here |
| Wikipedia | **403** | — |
| EDGAR EFTS (13D) | reachable but flaky (500s) + small-cap-thin | poor fit for our large-cap universe |
| **FINRA RegSHO short-volume** | **fully reachable, free, 195/195 breadth, daily** | pulled 2018-08→2026-06 |

**The binding constraint — survivorship-clean PRICES — is NOT obtainable for free from
here.** Delisted-name prices are paywalled (CRSP/Sharadar) or anti-bot-blocked.

**H6 short-flow (the one genuinely-new, breadth-rich dataset we could get):** raw
cross-sectional IC is real and correctly-signed (−0.044@20d, t=−7.4, full breadth) —
the strongest-looking cross-sectional signal of the search — but it is **economically
worthless**: the long-low-short-flow tilt (Sharpe 1.069) is WORSE than naive
equal-weight (1.133), the long-short spread is **−0.07**, and it fails DSR at n_trials=34
(0.834 < 0.95). A statistically-significant tiny IC ≠ a tradeable edge.

## THE decisive finding — the survivorship baseline explains everything
> **Equal-weighting the 69 survivor names with NO signal = Sharpe 1.133 vs SPY 0.775.**
> Insider-buy base = 1.134. Short-flow-low = 1.069. **All the same number.** The ~1.13
> Sharpe that made every long-only basket "beat SPY" is the SURVIVORSHIP BASELINE — these
> are the names that didn't go bankrupt/delist, so equal-weighting them mechanically
> crushes the index. No signal (insider, PEAD, short-flow, congress) added anything over
> doing nothing. Every in-sample "win" across all 6 hypotheses was this baseline in disguise.

This is the closure's central thesis proven in a single number, and it sets a hard ceiling:
**no long-only test on this survivor universe can separate signal from the +0.35
Sharpe survivorship gift.** Until delisting-inclusive prices exist, the answer is settled.

## Round 4 (continuation — broad experimentation: signals on/off × risk on/off, IS vs OOS)

Built a composable experiment engine (`experiment/engine.py`) and swept 252 configs
(63 signal subsets × 4 risk overlays), scored In-Sample AND OOS. See
`experiment/FINDINGS_overlay.md`.
- **Signals confirmed dead a 3rd way:** the best signal basket (shortflow) UNDERperforms
  the no-signal EW-universe; configs separate by OVERLAY, not signal.
- **ONE real candidate emerged — a trend/regime RISK overlay** (not alpha). On SPY
  (survivorship-immune): Sharpe 0.78→0.94, MaxDD −34%→−22%; beats the constant-exposure
  control (genuine timing); robust across MA 100–250 & all mappings; both IS and OOS.
  Same family as the incumbent `vol_target_overlay`, but trend-gate > vol-target on
  Sharpe, and **combined trend×vol dominates vol-target-alone**. Documented effect
  (time-series momentum), one pre-specified control — not a 252-trial pick.

## Decision (updated)
Per Phase-4 gate: **no signal-alpha survived** — confirmed across 4 rounds / ~34 mill
trials / 252 experiment configs. The closure's signal verdict (passive core) stands,
with a quantified mechanism (the +0.35 Sharpe survivor baseline that faked every
in-sample "win").

**One candidate looked like it cleared the bar on the RISK side** (a trend+vol overlay:
Sharpe lift via drawdown reduction, survivorship-immune, both periods). The user approved
integration ("A"). **On executing it, the honest finding: it ALREADY EXISTS in production.**
The incumbent `vol_target_overlay` already implements vol-target + SMA200 trend-halving +
IEF rotation (lines 105–113). Running the REAL function vs my variant: 0.93 vs 0.95 Sharpe
— statistically identical; my tiny DD edge is a 2022 bond-crash artifact, not robust
(`experiment/test_incumbent_overlay.py`). **No integration performed — building it would be
a duplicate parameter-variant in the risk path for a noise-level benefit** (Rule 50/Rule 30
violation). Protected paths never touched. NET: the one robust finding of the whole search
is already deployed in the existing architecture — a validation, not a new edge.

**Final state: passive core + the existing `vol_target_overlay` IS the answer.** Four rounds,
~34 mill trials, 252×2 experiment configs, full-breadth re-test, complete free data: no
tradeable signal-alpha, and the best risk overlay is the one the project already runs.

## Round 5 — Intraday / Event problem space (the last structurally-different direction)
See `FINDINGS_event.md`. Two-part honest result:
1. **True intraday (hours-timescale) is not testable** — intraday store = 2–3 symbols / a
   few days; multi-year intraday history is paid; the news event stream is ~5 months.
2. **The directional geopolitical thesis IS testable at daily resolution** (fetched the FREE
   daily GPR index 1985-2026 + the `asset_router` mapping) and is **NULL** — GPR spikes do
   NOT predict energy/defense/gold outperformance; at monthly resolution they MEAN-REVERT
   (wrong sign, energy −1.6%/defense −2.2% next month); daily forward effects all |t|<1.1.
The §5.5 "Hormuz" move is priced-in by daily resolution; any residual lives in the first
minutes — paid-data, highest-cost, most-arbitraged slice. **No free edge here either.**

### First-minutes test EXECUTED (paid Polygon intraday) — `FINDINGS_event.md`
Built a working Polygon minute-bar ingester and pulled 107 earnings events (2024-26,
mega-cap tech, extended hours, precise XBRL timestamps). Result: **no intraday
continuation** (gap REVERSES: open→close signed-by-gap −0.70%, t=−2.56). A real
earnings-gap-overreaction FADE exists (+0.70%/trade gross, t=2.56) BUT it is short-side
(needs shorting gap-ups; project is long-only), its long-only slice is insignificant
(t=0.62 net of costs), and it FAILS DSR@40 (0.41). Statistically real, not deployable.
**The intraday/event space joins the others: a detectable pattern, no deployable edge.** Risk-parameter exploration was pursued via H5 (insider-flow defensive overlay)
and was strictly dominated by the incumbent `vol_target_overlay`.

**Methodological recommendation (Bailey-LdP / Überprüfung §8):** the broad search should
STOP here. At ~33 trials the DSR threshold is now high enough that any further candidate
would need an extraordinary raw Sharpe to survive deflation — continuing to mine raises
P(false positive) faster than P(true find). The only honest way to reopen the question is
a DIFFERENT data regime (survivorship-clean / delisting-inclusive, e.g. CRSP / Sharadar),
where exactly ONE pre-registered hypothesis (H1 insider buys, in the high-liquidity slice)
deserves a single clean retest. On the data we have, the answer is settled: no edge.

Artifacts: `_scratch/probe_raw_signals.py`, `mill/insider_buy_mill.py`,
`mill/insider_robustness.py`, `mill/insider_mill_results.json`. Reproducible, read-only
on the data; the `daily.parquet` weight-leverage bug found mid-run is documented in
`phase1_raw_signal_probe.md`'s sibling note and was fixed before any verdict.
