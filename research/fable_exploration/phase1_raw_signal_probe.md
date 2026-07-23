# Phase 1 — Raw-signal probe (cheap GO/NO-GO)

Script: `_scratch/probe_raw_signals.py`. Run 2026-06-13. Read-only, no backtest yet.
Event excess returns = symbol forward return − SPY forward return, total-return-adj
close, entry = close of first trading day strictly AFTER `available_at`.

**These numbers are deliberately OPTIMISTIC** (survivorship universe; PEAD uses
latest-restated EPS; overlapping event windows inflate t-stats). They are a filter,
not evidence. A signal that's absent here is dead; one that shows here earns the mill.

## PEAD / SUE — NULL (does not advance)
benchmark=SPY, n_events=1999, 170 symbols, 2018–2026.
| horizon | Spearman IC | top tercile | bottom tercile | spread |
|--------:|------------:|------------:|---------------:|-------:|
| +5d  | +0.0403 | +0.215% | −0.139% | +0.354% |
| +20d | +0.0100 | +0.268% | +0.658% | **−0.390%** |
| +60d | −0.0037 | +0.680% | +2.549% | **−1.870%** |
By-year r60 IC: −0.08/+0.05/−0.18/+0.11/+0.02/+0.05/−0.02/−0.03/+0.14 — no stability.
**Read:** only a tiny +5d effect; drift reverses by 20–60d; tercile spread wrong sign.
Classic PEAD drift absent/inverted on large-cap survivors. Not advanced.

## Insider open-market BUYS (Form 4 'P') — LEAD (advances to mill)
n_buy_events=540, 109 symbols, 2018–2026.
| horizon | set | mean excess | median | t |
|--------:|-----|------------:|-------:|--:|
| +20d | all   | +1.316% | +0.840% | +2.85 |
| +20d | clust | +1.469% | +2.403% | +1.74 |
| +60d | all   | +4.283% | +1.441% | **+3.73** |
| +60d | clust | +3.587% | +1.546% | +1.63 |
**Read:** economically large, t clears the Harvey-Liu single-factor t>3 hurdle at 60d
(BEFORE multiple-testing deflation). Clustering does NOT beat the full P-buy set on t.
Caveats: overlapping 60d windows inflate t; survivorship inflates the level (insiders
buy falling names; bankruptcies absent). Portfolio-level mill + DSR required.

## Congress BUYS — marginal (secondary)
n_buy_events=2584, 173 symbols, 2018–2026.
| horizon | set | mean excess | median | t |
|--------:|-----|------------:|-------:|--:|
| +20d | all   | +0.234% | −0.005% | +1.47 |
| +60d | all   | +0.796% | −0.096% | +2.69 |
| +60d | >$50k | +2.193% | +1.048% | +2.11 |
**Read:** only the large-buy 60d subset is interesting; thin, House-dominant, no-SLA.
Secondary at best; expected to die on DSR.

## Phase-1 verdict
One genuine lead: **insider open-market purchases (H1)**. PEAD null, congress marginal.
Next: build the falsification mill for H1 (PIT portfolio, walk-forward OOS, realistic
costs, DSR with honest trial count, vs SPY AND 60/40, + survivorship stress test).
