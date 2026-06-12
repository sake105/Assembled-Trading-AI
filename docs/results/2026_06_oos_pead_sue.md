# OOS PEAD/SUE backtest — NEGATIVE (2026-06-12)

**Verdict: NO out-of-sample edge → `pead_sue_score` stays at weight 0 (shadow).**

After wiring the free SEC-XBRL fundamentals path into `_compute_pead_sue_factor`
(commit `4ac4b01a`, weight held at 0 pending evidence), a PIT-clean OOS backtest
was run to decide the live weight. It fails the project bar decisively.

## Method (PIT-clean)
- **Signal:** SUE = (actual_EPS − expected_EPS) / σ.
  - `expected` = TRUE `(fp, fy-1)` year-ago same-quarter EPS (fiscal-label join,
    `features.pead_sue.quarterly_seasonal_expected`) — NOT a positional shift(4).
  - **σ = EXPANDING std of forecast errors STRICTLY PRIOR to each event**, ordered
    by `available_at` — no full-sample / in-sample standardiser leak (closes the
    senior-review F-senior-4 concern; `compute_sue_from_expected`'s full-sample σ
    is fine for shadow logging but NOT for a live/OOS standardiser).
- **Event time** = `available_at` (EDGAR acceptance instant); forward return =
  close[t0 .. t0+60 trading days] (classic PEAD drift window) from the
  total-return-adjusted `output/aggregates/daily.parquet`.
- **Sample:** `data/raw/fundamentals/fundamentals_xbrl_full.parquet`, 171 symbols,
  6,012 events, 2011-10 → 2026-03.

## Results
| metric | value | bar |
|---|---|---|
| pooled Spearman IC(SUE, fwd60d) | **+0.009** | — |
| monthly IC mean → IR-t | **−0.13** | > 1.96 |
| L/S top-vs-bottom-quintile, ann. Sharpe | **~0.04** (t +0.23) | — |

## Why the earlier sanity cut looked positive
An initial sanity check reported IC ≈ **+0.062**. That was an ARTIFACT of
(a) a **full-sample σ** standardiser (in-sample look-ahead) and (b) a narrow
**29-name** unconditional cut. With the PIT-clean expanding σ and the full
171-name universe + monthly IR-t, the signal is null.

## Caveats (the null is, if anything, generous)
Latest-restatement EPS values (mild residual value look-ahead); single horizon;
no costs / sector-neutralisation; large-cap universe; overlapping 60d windows
inflate the t-stat (no Newey-West) — i.e. the true IR-t is ≤ −0.13.

## Conclusion
The free SEC-XBRL fundamentals **ingester + PIT loader** stand as durable
infrastructure (replacing paid fundamentals). The **PEAD/SUE factor stays
SHADOW (weight 0)** — wired and correct, but with no demonstrated edge, exactly
as the evidence-gate intended. Consistent with the project's repeated OOS
rejections of single-factor edges.
