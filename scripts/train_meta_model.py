"""Train Meta-Model (Plan W7 / Sprint 2).

Binary GradientBoosting classifier that predicts whether a signal's
5-day forward return exceeds a threshold. Used as a confidence filter
in multifactor_v2.

Training data comes from backtest signal files + price data. If no
real backtest history exists, the script generates a synthetic dataset
for smoke-testing the pipeline.

Usage:
    python scripts/train_meta_model.py --data-dir output/runs
    python scripts/train_meta_model.py --synthetic --out models/meta_model_v1.joblib
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = ROOT / "models" / "meta_model_v1.joblib"
DEFAULT_THRESHOLD = 0.03  # 3% forward return threshold
DEFAULT_N_FOLDS = 5
DEFAULT_EMBARGO_DAYS = 5

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


def _generate_synthetic_dataset(
    n_samples: int = 2000,
    n_factors: int = 15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Generate synthetic training data for smoke testing."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.standard_normal((n_samples, n_factors)),
        columns=FACTOR_COLUMNS[:n_factors],
    )
    # Add regime_label and vol_regime
    X["regime_label"] = rng.choice([0, 1, 2, 3], size=n_samples)
    X["vol_regime"] = rng.choice([0, 1, 2], size=n_samples)

    # Label: weak signal from first 3 factors + noise
    signal = 0.3 * X.iloc[:, 0] + 0.2 * X.iloc[:, 1] + 0.1 * X.iloc[:, 2]
    noise = rng.standard_normal(n_samples) * 0.5
    y = (signal + noise > 0.3).astype(int)
    return X, pd.Series(y, name="label")


def _load_backtest_dataset(data_dir: Path) -> tuple[pd.DataFrame, pd.Series] | None:
    """Load real training data from backtest runs. Returns None if insufficient."""
    signals_files = sorted(data_dir.glob("*/signals_latest.json"))
    if len(signals_files) < 100:
        logger.warning(
            "Only %d signal files found (need >= 100). Use --synthetic for smoke test.",
            len(signals_files),
        )
        return None
    # Placeholder: real implementation would parse signal files + compute
    # forward returns from prices. For now, return None to trigger synthetic.
    logger.info("Real backtest loader not yet wired — use --synthetic")
    return None


def train(
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int = DEFAULT_N_FOLDS,
    embargo_days: int = DEFAULT_EMBARGO_DAYS,
) -> tuple[object, dict]:
    """Train GradientBoosting with purged cross-validation.

    Returns (model, metrics_dict).
    """
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import KFold
    except ImportError:
        logger.error("scikit-learn is required: pip install scikit-learn")
        raise

    kf = KFold(n_splits=n_folds, shuffle=False)
    aucs = []

    for train_idx, test_idx in kf.split(X):
        # Simple embargo: drop test samples too close to train boundary
        if embargo_days > 0 and len(train_idx) > 0:
            boundary = train_idx[-1]
            test_idx = test_idx[test_idx > boundary + embargo_days]
            if len(test_idx) == 0:
                continue

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        clf = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
        )
        clf.fit(X_train, y_train)

        if len(y_test.unique()) > 1:
            proba = clf.predict_proba(X_test)[:, 1]
            aucs.append(roc_auc_score(y_test, proba))

    # Final model on all data
    final_model = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
    )
    final_model.fit(X, y)

    metrics = {
        "mean_auc": float(np.mean(aucs)) if aucs else 0.0,
        "std_auc": float(np.std(aucs)) if aucs else 0.0,
        "n_folds_evaluated": len(aucs),
        "n_samples": len(X),
        "n_features": X.shape[1],
        "feature_names": list(X.columns),
        "training_date": datetime.now(timezone.utc).isoformat(),
        "threshold": DEFAULT_THRESHOLD,
    }
    return final_model, metrics


def save_model(model: object, metrics: dict, path: Path) -> None:
    """Save model and metadata."""
    try:
        import joblib
    except ImportError:
        logger.error("joblib is required: pip install joblib")
        raise

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)

    meta_path = path.with_suffix(".json")
    meta_path.write_text(json.dumps(metrics, indent=2))
    logger.info("Model saved to %s (AUC=%.3f)", path, metrics["mean_auc"])


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Train meta-model for signal filtering"
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "output" / "runs")
    parser.add_argument("--out", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--synthetic", action="store_true", help="Use synthetic data for smoke testing"
    )
    parser.add_argument("--n-samples", type=int, default=2000)
    args = parser.parse_args(argv)

    if args.synthetic:
        logger.info("Generating synthetic dataset (n=%d)", args.n_samples)
        X, y = _generate_synthetic_dataset(n_samples=args.n_samples)
    else:
        result = _load_backtest_dataset(args.data_dir)
        if result is None:
            logger.warning("Falling back to synthetic data")
            X, y = _generate_synthetic_dataset(n_samples=args.n_samples)
        else:
            X, y = result

    model, metrics = train(X, y)
    save_model(model, metrics, args.out)
    print(
        f"[OK] Meta-model trained: AUC={metrics['mean_auc']:.3f} "
        f"({metrics['n_folds_evaluated']} folds, {metrics['n_samples']} samples)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
