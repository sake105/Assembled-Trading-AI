# ADR-006: multifactor_v2 with Regime-Conditional Factor Weights

**Status:** proposed
**Date:** 2026-04-11
**Deciders:** research

## Context

`multifactor_v1` uses a fixed weight per factor across all market
regimes. This is simple and easy to test but has a known weakness:
different factors work in different regimes. Trend factors tend to
pay off in trending bull markets; mean-reversion factors tend to
pay off in sideways regimes; crash-protection factors only matter
in crises. A fixed blend is a compromise that under-weights whichever
factor is best in the current regime.

Two options for adapting:

1. Let a ML model learn the weights end-to-end. Powerful but prone
   to overfitting, especially with limited regime-labeled history.
2. Estimate per-regime information coefficients (IC) from history
   and assign weights proportionally, normalized per regime.

Option 2 is simpler, has a natural regularisation (negative IC →
zero weight), and the resulting weights are auditable by a human
reviewer.

## Decision

`multifactor_v2` will compute per-regime factor weights from a
historical IC estimate:

1. For each regime `r` (bull / sideways / bear / crisis) and each
   factor `f`, compute `IC(r,f) = spearman(score_f, fwd_return_5d)`
   over the observations that fell in regime `r`.
2. Normalise: `w(r,f) = max(0, IC(r,f)) / sum_f max(0, IC(r,f))`.
3. Require at least 100 observations per regime; fall back to the
   default static weights otherwise.
4. Persist the resulting weights as `configs/factor_weights_by_regime.json`
   and reload them at startup.

At inference time `compute_signals` detects the current regime and
picks the corresponding weight vector.

The meta-model (ADR placeholder, Sprint 2) is a **multiplicative**
confidence filter applied to the composite — not a 30th additive
factor.

## Consequences

### Positive

- Factor weights adapt to regime without per-bar parameter
  estimation (the weights are refreshed monthly, not daily).
- Negative-IC factors in a regime are zeroed automatically — a
  factor cannot drag the composite in a regime where it is a
  known loser.
- The weight file is inspectable; a human reviewer can sanity-check
  that, e.g., trend factors dominate in bull and mean-reversion in
  sideways.
- The fallback to static weights when sample size is low protects
  against overfitting in new regimes.

### Negative

- Requires a regime label generator that is itself trustworthy.
  Feeding v2 with noisy regime labels turns the per-regime weights
  into noise amplifiers.
- Monthly retraining needs a process (script + audit), not just an
  ad-hoc command.
- The 100-observation floor means a brand-new regime (e.g. a novel
  crisis pattern) falls back to static weights rather than learning
  on the fly.

## Alternatives Considered

- **End-to-end ML**: rejected. See context.
- **Exponentially-weighted IC**: deferred — potentially worth
  revisiting once the static-window approach is running and we
  have 1+ year of production data.
- **Bayesian updating**: overkill at current scale.
