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
]


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
                row["fwd_return_5d"] = 0.3 * row["trend_ema_spread"] + rng.normal(0, 0.02)
            elif regime == "bear":
                row["fwd_return_5d"] = 0.3 * row["mr_bollinger_pctb"] + rng.normal(0, 0.02)
            elif regime == "crisis":
                row["fwd_return_5d"] = 0.3 * row["vola_regime_score"] + rng.normal(0, 0.02)
            else:
                row["fwd_return_5d"] = 0.1 * row["mom_rsi_centered"] + rng.normal(0, 0.02)
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
                regime, len(subset), min_samples,
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

    return weights


def save_weights(weights: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(weights, indent=2))
    logger.info("Weights saved to %s", path)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Train regime-conditional factor weights")
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
