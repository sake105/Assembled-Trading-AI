"""HRP-based sizing wrapper (Sprint 3 / Plan W10).

A thin wrapper around ``compute_hrp_weights`` that turns a price panel plus
score-based preliminary weights into HRP-blended position weights. Designed
as the ``method="hrp"`` branch for strategy sizing callers.

Key properties:
  - pure function, never mutates inputs
  - scipy is optional; when missing, falls back to the input ``score_weights``
  - blends HRP weights with score weights at ``blend`` (default 0.7 HRP,
    0.3 score) per plan W10
  - scales the final weights to ``target_invested_pct``
  - keeps the symbol set equal to ``score_weights.keys()`` — any symbol
    missing from the HRP solution falls back to its score share of the blend
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.assembled_core.portfolio.hierarchical_risk_parity import compute_hrp_weights


def _pivot_returns(
    prices: pd.DataFrame,
    symbols: list[str],
    lookback_days: int,
) -> pd.DataFrame:
    """Build a (date × symbol) returns DataFrame. Empty on missing columns."""
    if prices is None or prices.empty:
        return pd.DataFrame()
    required = {"timestamp", "symbol", "close"}
    if not required.issubset(prices.columns):
        return pd.DataFrame()

    rows = prices[prices["symbol"].isin(symbols)].copy()
    if rows.empty:
        return pd.DataFrame()

    rows = rows.sort_values(["symbol", "timestamp"])
    pivot = rows.pivot_table(
        index="timestamp",
        columns="symbol",
        values="close",
        aggfunc="last",
    )
    if len(pivot) > lookback_days:
        pivot = pivot.iloc[-lookback_days:]
    returns = pivot.pct_change().dropna(how="all")
    return returns


def apply_hrp_sizing(
    score_weights: dict[str, float],
    prices: pd.DataFrame,
    *,
    lookback_days: int = 60,
    blend: float = 0.7,
    target_invested_pct: float = 1.0,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> tuple[dict[str, float], list[str]]:
    """Blend HRP weights with a score-based starting point.

    Args:
        score_weights: Preliminary weights from score-based sizing.
        prices: Price DataFrame with ``timestamp``, ``symbol``, ``close``.
        lookback_days: Rolling window for return panel.
        blend: Weight on the HRP component in the blend. ``0.0`` means score
            only, ``1.0`` means pure HRP.
        target_invested_pct: Final scaling target. The returned weights sum
            to (at most) this value.
        min_weight: Forwarded to ``compute_hrp_weights``.
        max_weight: Forwarded to ``compute_hrp_weights``.

    Returns:
        ``(adjusted_weights, reasons)`` tuple.
    """
    if not score_weights:
        return {}, []

    symbols = list(score_weights.keys())
    reasons: list[str] = []

    # Normalise score weights to sum to 1 for the blend math (guard zero-sum)
    score_total = sum(abs(v) for v in score_weights.values())
    if score_total <= 1e-12:
        return dict(score_weights), []
    score_norm = {s: score_weights[s] / score_total for s in symbols}

    returns = _pivot_returns(prices, symbols, lookback_days)
    usable = [c for c in symbols if c in returns.columns]

    hrp_norm: dict[str, float] = {}
    if len(usable) >= 2 and len(returns) >= 30:
        try:
            hrp_raw = compute_hrp_weights(
                returns[usable],
                min_weight=min_weight,
                max_weight=max_weight,
            )
        except Exception as exc:  # noqa: BLE001 - optional scipy path
            reasons.append(f"hrp_sizing: HRP compute failed ({exc}); falling back to score")
            hrp_raw = {}
        if hrp_raw:
            hrp_norm = dict(hrp_raw)
        else:
            reasons.append("hrp_sizing: HRP returned empty; falling back to score")
    else:
        reasons.append(
            f"hrp_sizing: insufficient data (symbols={len(usable)}, rows={len(returns)}); "
            "falling back to score"
        )

    # Blend. For symbols missing from HRP, use the score weight directly.
    blended: dict[str, float] = {}
    effective_blend = blend if hrp_norm else 0.0
    for sym in symbols:
        s_w = score_norm.get(sym, 0.0)
        h_w = hrp_norm.get(sym, s_w)  # fallback to score when HRP skipped symbol
        blended[sym] = effective_blend * h_w + (1.0 - effective_blend) * s_w

    # Final scale to target_invested_pct
    blended_sum = sum(abs(v) for v in blended.values())
    if blended_sum > 1e-12:
        scale = float(target_invested_pct) / blended_sum
        blended = {s: w * scale for s, w in blended.items()}

    if hrp_norm:
        reasons.append(
            f"hrp_sizing: blended HRP ({len(hrp_norm)} symbols) with score at "
            f"blend={blend:.2f}, scaled to target_invested_pct={target_invested_pct:.3f}"
        )

    return blended, reasons


def apply_hrp_sizing_from_policy(
    score_weights: dict[str, float],
    prices: pd.DataFrame,
    policy: dict[str, Any],
) -> tuple[dict[str, float], list[str]]:
    """Read HRP settings from ``policy['hrp_sizing']`` and apply.

    Returns ``(copy(score_weights), [])`` when disabled or missing config.
    """
    cfg = (policy or {}).get("hrp_sizing") or {}
    if not cfg.get("enabled", False):
        return dict(score_weights), []

    return apply_hrp_sizing(
        score_weights,
        prices,
        lookback_days=int(cfg.get("lookback_days", 60) or 60),
        blend=float(cfg.get("blend", 0.7) or 0.7),
        target_invested_pct=float(cfg.get("target_invested_pct", 1.0) or 1.0),
        min_weight=float(cfg.get("min_weight", 0.0) or 0.0),
        max_weight=float(cfg.get("max_weight", 1.0) or 1.0),
    )


def compute_hrp_target_weights(
    returns_panel: pd.DataFrame,
    min_history: int = 30,
    target_gross: float = 0.80,
) -> pd.Series:
    """Compute HRP target weights scaled to a target gross exposure.

    Thin sidecar wrapper around ``compute_hrp_weights`` that validates the
    input panel, drops columns with insufficient history, delegates to the
    dormant HRP module, and rescales weights so the sum equals
    ``target_gross``.

    Args:
        returns_panel: Wide-format DataFrame (index=dates, columns=symbols)
            of daily returns.
        min_history: Minimum rows required and minimum non-NaN observations
            per symbol column.
        target_gross: Target sum of weights (HRP produces weights summing to
            1.0; this rescales).

    Returns:
        ``pd.Series`` indexed by symbol, name ``"hrp_weight"``, summing to
        ``target_gross``.

    Raises:
        ValueError: On invalid input (too few rows, too few symbols, empty
            HRP result, or non-positive ``target_gross``).
    """
    if target_gross <= 0:
        raise ValueError(f"target_gross must be > 0, got {target_gross}")

    if not isinstance(returns_panel, pd.DataFrame):
        raise ValueError("returns_panel must be a pandas DataFrame")

    if len(returns_panel) < min_history:
        raise ValueError(
            f"insufficient history: {len(returns_panel)} rows < min_history={min_history}"
        )

    if returns_panel.shape[1] < 2:
        raise ValueError(
            f"need at least 2 symbols, got {returns_panel.shape[1]}"
        )

    valid_cols = [
        c for c in returns_panel.columns
        if returns_panel[c].notna().sum() >= min_history
    ]
    if len(valid_cols) < 2:
        raise ValueError(
            f"need at least 2 symbols with >= {min_history} observations, "
            f"got {len(valid_cols)}"
        )

    filtered = returns_panel[valid_cols]

    raw = compute_hrp_weights(filtered)
    if not raw:
        raise ValueError(
            "HRP module returned empty weights — scipy may be missing or "
            "data insufficient after dropna"
        )

    weights = pd.Series(raw, name="hrp_weight", dtype=float)
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("HRP produced non-positive weight sum")

    weights = weights * (target_gross / total)
    weights.name = "hrp_weight"
    return weights


def blend_hrp_with_score(
    hrp_weights: pd.Series,
    score_weights: pd.Series,
    hrp_alpha: float = 0.7,
) -> pd.Series:
    """Convex blend of HRP and score-based weights.

    ``out = hrp_alpha * hrp + (1 - hrp_alpha) * score``

    Inputs are aligned on the union of their symbol indices; missing symbols
    are treated as zero on the side where they are absent. The final result
    is renormalized to the maximum of the two input gross sums so the blend
    cannot inflate exposure.

    Args:
        hrp_weights: HRP weights indexed by symbol.
        score_weights: Score-based weights indexed by symbol.
        hrp_alpha: Blend coefficient in ``[0, 1]``. ``1.0`` returns HRP,
            ``0.0`` returns score (both up to renormalization).

    Returns:
        ``pd.Series`` indexed by the union of both input indices.

    Raises:
        ValueError: If ``hrp_alpha`` is outside ``[0, 1]``.
    """
    if not 0.0 <= hrp_alpha <= 1.0:
        raise ValueError(f"hrp_alpha must be in [0, 1], got {hrp_alpha}")

    all_symbols = hrp_weights.index.union(score_weights.index)
    hrp_aligned = hrp_weights.reindex(all_symbols, fill_value=0.0).astype(float)
    score_aligned = score_weights.reindex(all_symbols, fill_value=0.0).astype(float)

    blended = hrp_alpha * hrp_aligned + (1.0 - hrp_alpha) * score_aligned

    hrp_sum = float(hrp_weights.sum())
    score_sum = float(score_weights.sum())
    target = max(hrp_sum, score_sum)

    blended_sum = float(blended.sum())
    if blended_sum > 0 and target > 0:
        blended = blended * (target / blended_sum)

    blended.name = "blended_weight"
    return blended


__all__ = [
    "apply_hrp_sizing",
    "apply_hrp_sizing_from_policy",
    "compute_hrp_target_weights",
    "blend_hrp_with_score",
]
