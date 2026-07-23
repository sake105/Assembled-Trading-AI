# Experiment findings — signals vs risk overlays (round 4, 2026-06-13)

Built a composable experiment engine (`engine.py`): any subset of 6 signals
(shortflow, insider, pead, quality, congress, news) × 4 risk overlays (none /
voltgt10 / voltgt15 / regime-MA200-gate), each scored In-Sample (≤2022-06) AND OOS
(>2022-06). Swept all 63 signal subsets × 4 overlays = 252 configs (`engine.py`),
then validated the lead honestly (`validate_regime.py`, `sensitivity_regime.py`).

## Result 1 — the alt-data SIGNALS are dead (confirmed a 3rd way)
The shortflow long-basket (Sharpe 1.01) UNDERPERFORMS the no-signal EW-survivor
universe (1.10). Across the 252-config sweep the signal choice barely moves OOS
Sharpe; configs are separated almost entirely by their RISK OVERLAY. No subset of
insider/pead/quality/congress/news/shortflow adds robust risk-adjusted value.

## Result 2 — a trend/regime RISK overlay is real, robust, survivorship-immune
Cleanest test = apply the overlay to **SPY itself** (no universe, no signal, no
selection → survivorship-immune):

| on SPY | FULL Sharpe | MaxDD | IS | OOS |
|--------|-------------|-------|----|----|
| buy-hold | 0.78 | −33.7% | 0.53 | 1.14 |
| regime MA100/150/200/250 | 0.97/1.02/0.94/0.87 | −18…−22% | 0.67–0.84 | 1.11–1.23 |
| vol-target 10/15/20% | 0.84–0.87 | −12.8…−21.8% | 0.54–0.60 | 1.11–1.17 |
| **regime × vol-target** | **0.92** | **−16.8%** | 0.73 | 1.11 |
| CONST-exposure control (no timing) | 0.78 | −28.0% | 0.53 | 1.14 |

Why this is NOT the usual overfit:
- **Survivorship-immune** — it's just SPY.
- **Both periods** — improves IS (0.53→~0.8) and OOS (1.14→~1.2).
- **Beats the constant-exposure control** (0.94 vs 0.78) → genuine TIMING value, not
  just lower average exposure. (This is the exact control that killed insider-timing H5.)
- **Parameter-insensitive** — every MA window 100–250 and every mapping beats buy-hold.
- It is the documented time-series-momentum / trend-filter effect (Moskowitz-Ooi-
  Pedersen) — a single pre-specified risk control, not a 252-trial pick. (The *specific*
  shortflow+regime config fails DSR@252 — that's config-selection overfit; the OVERLAY
  itself is one robust effect that helps universally.)

## Honest framing
This is NOT new signal-alpha. It is a **risk-overlay improvement** in the same family
as the incumbent `vol_target_overlay` (which the closure already keeps). On this data:
trend-gate > vol-target on Sharpe; vol-target > trend-gate on MaxDD; **the combination
dominates vol-target-alone** (similar/better Sharpe, similar DD). Known tradeoff: trend
gates underperform in sharp V-recoveries (slow re-entry) — accepted cost for the DD cut.

## Re-run on ENRICHED universe (193 names, full 2018-history via yfinance backfill)
Cross-section doubled (94→192 deep-history). Findings HOLD: SPY×regime 0.80→0.97
(MaxDD −34→−22%, beats const-control 0.80, both periods); EW-universe 0.99→1.17. The
252-config sweep again separates by OVERLAY, not signal. Insider shows a period-
INCONSISTENT tilt (OOS 1.42 > baseline 1.33 but IS 0.60 < 0.75 → FULL 0.96 ≈ baseline
1.00; not robust). Congress below baseline. News has only ~5 months coverage (2026 only)
→ untestable over the full period. Data coverage proves signals are LIVE (insider 8,820
events/165 names) — they simply carry no consistent marginal edge over the survivors.

## Per-year walk-forward (the honest tempering — `wf_overlay.py`)
Per calendar fold the overlay wins on DRAWDOWN in 7–8/9 years but on SHARPE in only
**2/9**. The full-period Sharpe lift (0.80→0.97) is therefore the COMPOUNDING benefit of
dodging the 2020/2022 crashes, NOT consistent per-year outperformance (in choppy 2022 it
even hurts Sharpe via whipsaw while still cutting DD). So this is a **drawdown-protection
tool, not a return edge** — the SAME category as the incumbent `vol_target_overlay`.
Combined trend×vol = best DD-consistency (8/9), a MODEST refinement over vol-alone, not
transformative. No overselling: it manages risk, it does not beat the market.

## INTEGRATION CHECK → ALREADY EXISTS (no change made; `test_incumbent_overlay.py`)
Before any risk-path edit, ran the REAL production `vol_target_overlay` (SPY+IEF) vs my
variant: FULL Sharpe **0.93 (prod) vs 0.95 (mine)** — statistically identical. The
production overlay ALREADY implements trend×vol (vol-target + SMA200 halving, lines
105–113 of `strategies/vol_target_overlay.py`) and is arguably better (rotates de-risked
capital into IEF treasuries vs my cash). My tiny DD edge (−17% vs −22%) is a 2022
bond-crash artifact (cash beat bonds that year), not robust. Per-year neither dominates.
**Verdict: NOTHING to integrate — the one robust finding of the whole search is already
deployed.** Building it would be a duplicate parameter-variant in the risk path for a
noise-level benefit (violates Rule 50 no-duplication + Rule 30 smallest-safe-step). The
protected risk/pipeline paths were NOT touched.

## (superseded) Candidate for the Phase-4 gate (risk overlay, NOT alpha)
A combined trend+vol overlay is the one thing from the whole search worth proposing for
integration — as an enhancement/alternative to `vol_target_overlay`. That touches the
protected `risk/`+`pipeline/` paths → requires explicit operator go + a scoped, reviewed
edit (NOT blanket hook-disabling). Pre-integration: a rolling walk-forward head-to-head
vs the incumbent overlay + parameter-stability confirmation (largely done above).
