# Spec: `risk/correlation_guard.py`

## Purpose

Detect clusters of correlated symbols in the target book and cap
per-cluster weight. Prevents a small number of highly-correlated
names from dominating the portfolio under the guise of
diversification.

## Public API

- `compute_correlation_matrix(returns_df) -> pd.DataFrame` — pairwise
  correlation over a rolling window.
- `detect_correlated_clusters(corr_matrix, threshold) -> list[list[str]]`
  — connected components above the threshold.
- `apply_correlation_guard(targets, returns_df, policy) -> tuple[pd.DataFrame, dict]`
  — scale cluster members so cluster gross weight respects the cap.
- `compute_avg_correlation(corr_matrix) -> float` — diagnostic.
- `detect_correlation_regime_shift(history) -> bool` — rolling-window
  regime shift flag.

## Inputs

- `returns_df`: long-format returns with `timestamp`, `symbol`,
  `return`.
- `targets`: target weights per symbol.
- `policy`: correlation threshold, cluster gross cap, lookback window.

## Invariants

- If no cluster is above the threshold, the function is a no-op.
- Cluster scaling is proportional within the cluster: the relative
  weights of cluster members are preserved.
- Symbols not in any cluster are untouched.

## Error handling

- Too few observations for a reliable correlation estimate → skip
  the guard, return targets unchanged, log a warning.
- NaN correlation values → treated as zero (no edge) with a warning.

## Test strategy

- Two-symbol cluster at correlation 0.99 → scaling kicks in.
- Three-symbol cluster with one dominant name → dominant name is
  scaled down more in absolute terms but relative ratios hold.
- Zero-correlation universe → no-op.
- Property: post-guard cluster gross ≤ cluster cap.

## Known limits

- Pearson correlation only. Tail-dependence is a separate concern
  handled by the copula-based path (dormant, Sprint 3 wiring).
- Single-window; no regime-aware thresholds yet.
