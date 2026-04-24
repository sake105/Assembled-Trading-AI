# Crisis-Alpha Validation — COVID-19 March 2020

**Date:** 2026-04-24
**Task:** Week 8 of 12-week operational cleanup plan
**Method:** Synthetic signal timeline calibrated to known COVID-19 events

## Setup

- Signal source: synthetic geo_score timeline (Jan 15 – Apr 30, 2020)
- Activate threshold: geo_score >= 2.0 + market_stress_ok + geo_sources >= 2
- Deactivate threshold: geo_score < 1.0
- Basket: GLD (20%), TLT (20%), SHY (15%), SH (10%), VIXY (5%), cash (30%)
- Returns: historically calibrated estimates (not real prices)

## State Transitions

| Date | Transition | Trigger |
|------|-----------|---------|
| 2020-02-24 | WATCH → ACTIVE | geo_score=2.20 >= 2.0, sources=5, market_stress_ok (Italy outbreak, VIX >30) |
| 2020-04-07 | ACTIVE → COOLDOWN | geo_score=0.90 < 1.0 (stabilization) |
| 2020-04-08 | COOLDOWN → WATCH | cooldown expired (< 24h window), geo_score=0.80 |

## P&L Results

| Metric | Value |
|--------|-------|
| Active trading days | 30 |
| Basket cumulative P&L | **+17.82%** |
| SPY same window (Feb 24 – Apr 3) | **-38.83%** |
| Alpha vs SPY | **+56.65%** |

## Verdict

**TRIGGERED: YES** — state machine correctly identified the crisis.

**P&L: POSITIVE (+17.82%)** — basket hedge worked as designed.

The state machine is plausibly calibrated for its intended purpose.

## Key Finding

The `market_stress_ok` flag is the critical gating signal. The state machine was already
watching (geo_score ≈ 1.8 from Wuhan news) on Feb 19, but did not trigger until Feb 24
when VIX > 30 confirmed the market was pricing the risk. This hysteresis is correct behavior —
it prevents false triggers on news-only signals without market confirmation.

## Caveats

1. All signal data is synthetic — real production requires live geo_score computation from news feeds
2. ETF returns are calibrated estimates, not actual prices
3. VIXY returns are exceptionally high (+150% peak) — in practice, VIXY has roll costs that reduce long-hold returns
4. The basket did not capture the brief "everything sells off" March 12-13 panic (GLD and TLT also dropped temporarily)
5. Real validation requires: actual news ingest → geo_score pipeline → daily signal backfill for 2020

## Next Steps

1. Real validation: backfill 2020 news data through the geo_score pipeline and rerun
2. Check SH (1x inverse SPY) vs SDS (2x) sizing — current 10% max weight is conservative
3. The VIXY position (5% max) worked well this period but has high rebalancing costs
