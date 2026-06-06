# OOS Edge Re-Run After the Bug-Fix Sweep — Did Any Edge Change?

Run date (UTC): 2026-06-06
Code state: `faf1eef8` (post deferred-decisions + open-items + bucket-A/B sweeps; 116 `src/` files changed since the newest OOS baseline `329a3240`).
Question answered: **after the multi-day bug-fix sweep, did any out-of-sample edge verdict change — better, worse, or the same?**

## Bottom line

**No OOS edge verdict changed.** Every strategy that was REJECTED vs SPY stays REJECTED; the one
"gemischt" book (mfv2 full-stack) stays gemischt. The codebase-wide finding **"no strategy shows a
robust, DSR-deflated, statistically-significant OOS edge over SPY"** holds in full post-sweep.

The sweep produced exactly **one material code-effect on a backtest number**: in the *literal-pipeline*
harness, the beta-targeted `bab_ls` book moved pooled-OOS Sharpe **−0.06 → −0.36** (an already-rejected
research book got *more* rejected). It is isolated to the single leverage/beta-scaled strategy and does
not create or destroy any edge. Details + attribution below.

## Method — why this is a clean code-vs-data separation

Re-running blindly would confound two different "did it change?" questions:
1. did our **code** changes move the edges? (the question asked)
2. did the underlying **data** drift since the baselines? (a different question)

The 17 OOS harnesses were first classified by run-path. **13 of 17 are self-contained vectorized
research scripts** whose execution path imports only an *unchanged* data loader — the 116 changed
`src/` files are provably **off their path**. (Notably the edge-metric machinery `qa/metrics.py`,
`qa/deflated_sharpe.py`, `qa/bootstrap_metrics.py` changed only in docstrings + deprecation warnings on
*uncalled* functions; the actually-called `deflated_sharpe()` / `probabilistic_sharpe_ratio()` are
numerically identical, so no verdict can move from metric drift.)

Only **4 harnesses route changed code onto their decision path**. They were re-run, plus a set of
**controls** and **data-drift probes** to isolate code from data:

| Role | Harness | Data | What a delta means |
|---|---|---|---|
| Clean code-effect | `etf_pairs_literal` | frozen cache (05-31 = baseline) | pure **CODE** |
| Code + data | `pipeline_realistic` | daily.parquet (06-03) | CODE + DATA (disentangled via probes) |
| Network re-baseline | `mfv2_full` | fresh Alpaca | CODE + fresh-DATA |
| Control | `dual_momentum_literal`, `sector_rotation_fullhist` | frozen same-vintage | must reproduce bit-for-bit |
| Data-drift probe | `leverage_short`, `new_factors_sweep`, `low_max_lottery`, `lowvol_momentum`, `residual_momentum`, `etf_pairs_meanrev`, `sector_rotation` | daily.parquet (06-03) | pure **DATA** (code off-path) |

**Not re-run (deliberately):** `trend_baseline`, `dual_momentum`, `vol_target_overlay`,
`mfv_long_short` — zero changed code on their path **and** Alpaca-network data. Re-running them only
re-fetches a fresher data vintage; it cannot test whether the sweep moved their edge. Their prior
verdicts stand on code grounds (all REJECTED / negative vs SPY).

## Controls — determinism intact

- `dual_momentum_literal`: regenerated doc was **byte-identical** to HEAD (not even git-modified).
- `sector_rotation_fullhist`: every metric bit-identical in **both** price modes — adj `sector_lo`
  +0.65 / `eq_sector` +0.58, raw `sector_lo` +0.52 / `eq_sector` +0.47, all DSR/IR-t/fold-win identical.

Same data + no changed code → bit-for-bit reproduction. The harness machinery is deterministic.

## Data-drift probes — the 06-03 parquet refresh was inert on historical folds

`daily.parquet` was refreshed 2026-06-03 (after the 05-29/05-31 baselines). Effect on the
change-independent harnesses reading it:

| Harness | Headline old → new | Verdict |
|---|---|---|
| `leverage_short` | all 5 L/S **bit-identical** (`bab_ls` +0.47, `mom_ls` +0.78, `resmom_ls` −0.35, `reversal_ls` −0.55, `lowvol_ls` −0.72) | REJECTED (unch.) |
| `sector_rotation` | metric table **bit-identical** (SPY 0.91/17.3%, `sector_lo` 0.92, `eq_sector` 0.85) | REJECTED (unch.) |
| `new_factors_sweep` | **bit-identical** (`low_beta` Calmar +1.90→+1.89 only) | ALL 3 REJECTED (unch.) |
| `low_max_lottery` | headline **bit-identical** (Ø CAGR +9.8% / Sharpe +1.06) | no edge (unch.) |
| `lowvol_momentum` | combo **bit-identical** (Calmar +2.55→+2.57 only) | REJECTED (unch.) |
| `residual_momentum` | only Fold-6 cells moved (CAGR +9.0%→+8.9%); **all Ø rows identical** | REJECTED (unch.) |
| `etf_pairs_meanrev` | Full Ø Sharpe −0.49→**−0.55**, Long-only +0.71→**+0.78**, **CAGR bit-identical** | REJECTED (unch.) |

The refresh touched only a few recent-fold bars (2024-2025) and left CAGR / MaxDD / every aggregate
essentially unchanged. `etf_pairs_meanrev` (ETF pairs, cointegration-sensitive) is the only one to move
the headline, and it moved *away* from an edge — still REJECTED. **This is pure data drift, not our
fixes.** Crucially it also confirms the 75-symbol equity universe over 2019-2024 is stable, which is
what makes the `pipeline_realistic` attribution below airtight.

## Code-path harnesses

### `etf_pairs_literal` — clean code-effect (data frozen) → negligible

REJECTED → REJECTED. AnnSharpe −0.06 → **−0.07**, CAGR −0.6% → −0.7%, IR-t −2.18 → −2.19,
DSR-prob 0.02 (fail) unchanged. With the input cache frozen at the baseline vintage, this is the **pure
code effect** of the sweep on a literal-pipeline book: **~0.01 Sharpe — immaterial.**

### `pipeline_realistic` — the one material code-effect: `bab_ls`

All 11 strategies still REJECTED. **10 of 11 books are bit-identical** old→new. The exception:

| Book | OLD | NEW | Δ |
|---|---|---|---|
| `bab_ls` (Betting-Against-Beta L/S, beta-targeted) | Sharpe −0.06 / IR-t −1.86 / DSR 0.03 / vol-match −1.4% | Sharpe **−0.36** / IR-t −2.26 / DSR 0.00 / vol-match **−7.2%** | worse |
| all 10 others (`mom_ls`, `resmom_ls`*, `reversal_ls`, `lowvol_ls`, `mom_lo`, `high52w_lo`*, `reversal_lo`, `lowbeta_lo`, `resmom_lo`, `eq_weight`) | — | bit-identical (≤0.01) | none |

\* `resmom_ls` −0.10→−0.09 and `high52w_lo` +1.16→+1.15 are sub-0.01 rounding.

**Attribution = CODE (not data).** The control `leverage_short` runs the *same* `bab_ls` signal on the
*same* universe/window/data via *unchanged* vectorized code and is **bit-identical** old→new — so the
prices `bab_ls` consumes did not change. With data held fixed, the −0.30 Sharpe move is a code effect of
the sweep, and it is isolated to the **only leverage/beta-scaled book** (the other 10 are byte-stable).
The likeliest source is the turnover/sizing notional-unit corrections in the sweep (e.g. the
turnover-budget unit-mix fix `b3bde616`), which would bite only on a book that applies leverage/beta
scaling. **Direction-of-correctness (is −0.36 the corrected number and −0.06 the previously-inflated one,
or a regression?) is not yet determined** — see follow-up. Either way `bab_ls` was and remains REJECTED;
no edge is created or destroyed.

### `mfv2_full` — macro-PIT-fix re-baseline (the SUPERSEDED doc)

The baseline doc was marked SUPERSEDED because it predated the macro look-ahead fix (E-038): the macro
factors genuinely changed. Re-baselined on fresh Alpaca data:

| Metric | OLD (superseded) | NEW (post-E-038) |
|---|---|---|
| Ø CAGR | 10.7% | **10.7%** (identical) |
| Ø Sharpe | 0.36 | **0.37** |
| Sharpe Δ vs TA-only | +0.00 | **+0.01** |
| Verdict | gemischt (below SPY 0.95) | **gemischt** (unch.) |

**Removing the macro look-ahead changed the full-stack edge by +0.01 Sharpe — immaterial.** The
SUPERSEDED doubt is resolved: the look-ahead did *not* inflate the result; mfv2 full-stack has no
risk-adjusted edge with or without it (production weight stays ~0). (Caveat: fresh Alpaca vintage, so
code+data combined — but both are clearly negligible given CAGR is identical.)

## Disposition

The 12 regenerated `docs/results/*.md` were **restored to HEAD** rather than committed: the re-run
*confirmed* their numbers, and the committed docs carry hand-authored analysis + correction banners the
generating scripts do not reproduce (e.g. `pipeline_realistic` regenerates 161 fewer lines). This
consolidated report is the durable artifact of the exercise.

## Follow-ups

1. **`bab_ls` literal-pipeline code-effect (−0.06→−0.36).** One-commit bisect over the sweep's
   turnover/sizing changes to confirm whether `b3bde616` (turnover-budget notional-unit fix) is the
   cause and whether the new number is the *corrected* one. Non-urgent (research-harness book, rejected
   either way; no production strategy uses this path).
2. **mfv2_full SUPERSEDED banner** can be lifted/annotated: the re-baseline shows the look-ahead was
   immaterial, so the doc's conclusion was never unreliable.

_Harnesses re-run on the repo venv (Py 3.11.9, numpy 2.2.6, pandas 2.2.3 — the CI vintage). Local
one-shot; not a CI run. No production module edited; research harnesses set `ASSEMBLED_NO_CRISIS_OVERLAY=1`
+ `write_outputs=False` and write no live ops/risk state (E-035-clean)._
