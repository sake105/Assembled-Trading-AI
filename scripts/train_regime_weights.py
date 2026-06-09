"""Train Regime-Conditional Factor Weights (Plan B3.3 / Sprint 3).

Learns per-regime (bull/sideways/bear/crisis) factor weights from
historical IC (Information Coefficient = Spearman rank correlation
between factor scores and 5-day forward returns).

Usage:
    python scripts/train_regime_weights.py --data-dir output/runs
    python scripts/train_regime_weights.py --synthetic --out configs/factor_weights_by_regime.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)

DEFAULT_OUT = ROOT / "configs" / "factor_weights_by_regime.json"
REGIMES = ["bull", "sideways", "bear", "crisis"]
MIN_SAMPLES_PER_REGIME = 100

FACTOR_COLUMNS = [
    # v1 factors (1-15)
    "trend_ema_spread",
    "trend_ma200_position",
    "trend_adx_strength",
    "trend_macd_hist",
    "mom_rsi_centered",
    "mom_volume_weighted",
    "mom_obv_trend",
    "mr_bollinger_pctb",
    "mr_stoch_oversold",
    "vol_abnormal",
    "vol_tick_imbalance",
    "vola_regime_score",
    "vola_vov_penalty",
    "breadth_above_ma",
    "breadth_ad_line",
    # v2 factors (16-18)
    "mr_zscore_reversal_3d",
    "mr_rsi_extreme_uptrend",
    "sector_rotation_bias",
    # Earnings/Insider (19-20)
    "earnings_surprise_z",
    "insider_activity_score",
    # News/Macro (21-24)
    "news_sentiment_7d",
    "news_volume_spike",
    "macro_growth_momentum",
    "macro_inflation_surprise",
    # Intermarket (25-27)
    "intermarket_bond_equity",
    "intermarket_credit_spread",
    "intermarket_yield_curve",
    # Options (28-29)
    "options_put_call_extreme",
    "vix_regime_score",
    # Congress (30)
    "congress_activity",
]

# Factors that are intentionally pinned to 0.0 in EVERY regime, regardless of
# what the IC fit produces. These are dead on free feeds and re-fitting them is a
# reactivation landmine that silently un-zeros configs/factor_weights_by_regime.json
# (see that file's _note). Keep this list in sync with that _note:
#   - earnings_surprise_z: EPS estimates cached for only ~44 mega-caps -> degenerate
#     cross-section (free-feed ceiling, 2026-06-01) + a loader/wrapper schema bug.
#   - insider_activity_score: now fed by real EDGAR Form 4 (insider_form4.parquet,
#     classified P/S) since 2026-06-09 — kept at 0 PENDING an OOS re-baseline of edge
#     (prior full-stack OOS Sharpe-Delta was +0.00; data availability != activation).
#   - congress_activity: now fed by free House/Senate STOCK-Act (congress_trades.parquet)
#     since 2026-06-09 — kept at 0 PENDING an OOS re-baseline of edge.
# The trainer force-zeros these AFTER the fit so the canonical retrain path stays
# consistent with the hand-maintained policy. Sub-1.0 regime sums are safe (runtime
# renormalises by the live-factor sum at scoring time).
INTENTIONALLY_ZEROED_FACTORS = (
    "earnings_surprise_z",
    "insider_activity_score",
    "congress_activity",
)


def _generate_synthetic_data(
    n_per_regime: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """Synthetic dataset for smoke testing."""
    rng = np.random.default_rng(seed)
    rows = []
    for regime in REGIMES:
        for _ in range(n_per_regime):
            row: dict = {"regime": regime}
            for f in FACTOR_COLUMNS:
                row[f] = rng.standard_normal()
            # Forward return with regime-dependent factor relevance
            if regime == "bull":
                row["fwd_return_5d"] = (
                    0.20 * row["trend_ema_spread"]
                    + 0.10 * row["earnings_surprise_z"]
                    + 0.05 * row["insider_activity_score"]
                    + rng.normal(0, 0.02)
                )
            elif regime == "bear":
                row["fwd_return_5d"] = (
                    0.20 * row["mr_bollinger_pctb"]
                    + 0.10 * row["intermarket_bond_equity"]
                    + 0.08 * row["options_put_call_extreme"]
                    + rng.normal(0, 0.02)
                )
            elif regime == "crisis":
                row["fwd_return_5d"] = (
                    0.20 * row["vola_regime_score"]
                    + 0.15 * row["vix_regime_score"]
                    + 0.10 * row["intermarket_credit_spread"]
                    + rng.normal(0, 0.02)
                )
            else:  # sideways
                row["fwd_return_5d"] = (
                    0.08 * row["mom_rsi_centered"]
                    + 0.08 * row["news_sentiment_7d"]
                    + 0.05 * row["macro_growth_momentum"]
                    + 0.05 * row["congress_activity"]
                    + rng.normal(0, 0.02)
                )
            rows.append(row)
    return pd.DataFrame(rows)


def compute_regime_weights(
    df: pd.DataFrame,
    factors: list[str],
    min_samples: int = MIN_SAMPLES_PER_REGIME,
) -> dict[str, dict[str, float]]:
    """Compute IC-based weights per regime.

    Returns dict[regime -> dict[factor -> weight]].
    """
    from scipy.stats import spearmanr

    weights: dict[str, dict[str, float]] = {}
    for regime in REGIMES:
        mask = df["regime"] == regime
        subset = df[mask]
        if len(subset) < min_samples:
            logger.warning(
                "Regime '%s' has only %d samples (need %d), using equal weights",
                regime,
                len(subset),
                min_samples,
            )
            weights[regime] = {f: 1.0 / len(factors) for f in factors}
            continue

        ics: dict[str, float] = {}
        for f in factors:
            corr, _ = spearmanr(subset[f], subset["fwd_return_5d"])
            ics[f] = max(0.0, corr)  # no negative weights

        total = sum(ics.values())
        if total < 1e-9:
            weights[regime] = {f: 1.0 / len(factors) for f in factors}
        else:
            weights[regime] = {f: ic / total for f, ic in ics.items()}

        # Durably re-zero the policy-zeroed factors so a retrain cannot reactivate
        # them (sub-1.0 sums are renorm-safe at scoring time).
        for f in INTENTIONALLY_ZEROED_FACTORS:
            if f in weights[regime]:
                weights[regime][f] = 0.0

    return weights


def save_weights(weights: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(weights, indent=2))
    logger.info("Weights saved to %s", path)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Train regime-conditional factor weights"
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "output" / "runs")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args(argv)

    if args.synthetic:
        logger.info("Generating synthetic regime dataset")
        df = _generate_synthetic_data()
    else:
        logger.warning("Real backtest loader not yet wired — falling back to synthetic")
        df = _generate_synthetic_data()

    weights = compute_regime_weights(df, FACTOR_COLUMNS)
    save_weights(weights, args.out)

    for regime, w in weights.items():
        top3 = sorted(w.items(), key=lambda x: x[1], reverse=True)[:3]
        top_str = ", ".join(f"{f}={v:.3f}" for f, v in top3)
        print(f"  {regime}: {top_str}")

    print(f"[OK] Regime weights saved to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
