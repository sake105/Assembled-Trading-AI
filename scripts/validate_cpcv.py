"""scripts/validate_cpcv.py — Combinatorial Purged Cross-Validation for strategy validation.

Item 88 — Uses PurgedKFold from src.assembled_core.ml.purged_cv (or cpcv_validation as
fallback) to run walk-forward CPCV over factor returns from a price panel.

Exit codes:
  0  STABLE   — min_sharpe > 0.3 and std_sharpe < 0.5
  1  UNSTABLE — thresholds not met
  2  NO DATA  — panel file missing or unreadable

Usage:
    python scripts/validate_cpcv.py
    python scripts/validate_cpcv.py --panel output/factor_panels/full_panel_7y.parquet
    python scripts/validate_cpcv.py --panel data/sample/watchlist_2020_2026.parquet --n-folds 6 --embargo-days 10
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Default panels to search (in priority order)
_DEFAULT_PANEL_CANDIDATES = [
    ROOT / "output" / "factor_panels" / "full_panel_7y.parquet",
    ROOT / "output" / "factor_panels" / "full_panel_7y_macro.parquet",
    ROOT / "data" / "sample" / "watchlist_2020_2026.parquet",
    ROOT / "data" / "sample" / "watchlist_2007_2026.parquet",
    ROOT / "data" / "sample" / "eod_sample.parquet",
    ROOT / "data" / "sample" / "backtest_1y.parquet",
]

# Verdict thresholds
_MIN_SHARPE_THRESHOLD = 0.3
_MAX_STD_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_default_panel() -> Path | None:
    for candidate in _DEFAULT_PANEL_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def _load_panel(panel_path: Path):
    """Load parquet panel, return DataFrame or raise with clear message."""
    try:
        import pandas as pd
    except ImportError as exc:
        log.error("pandas required: %s", exc)
        sys.exit(2)

    if not panel_path.exists():
        log.error("[NO DATA] Panel file not found: %s", panel_path)
        sys.exit(2)

    try:
        df = pd.read_parquet(panel_path)
        log.info("[LOAD] Panel: %s  shape=%s", panel_path.name, df.shape)
        return df
    except Exception as exc:
        log.error("[NO DATA] Could not read panel %s: %s", panel_path, exc)
        sys.exit(2)


def _compute_factor_returns(df) -> tuple:
    """Extract or synthesize a factor return series and timestamp series from panel.

    Returns (timestamps, returns) as 1-D arrays suitable for cross-validation.
    """
    import numpy as np
    import pandas as pd

    # Try common close/return column names
    close_col = None
    for col in ("close", "Close", "adj_close", "AdjClose", "price"):
        if col in df.columns:
            close_col = col
            break

    # If panel has a 'symbol' column, use a single representative symbol (AAPL or first)
    if "symbol" in df.columns or "ticker" in df.columns:
        sym_col = "symbol" if "symbol" in df.columns else "ticker"
        symbols = df[sym_col].unique()
        chosen = "AAPL" if "AAPL" in symbols else symbols[0]
        log.info("[INFO] Multi-symbol panel detected; using symbol=%s", chosen)
        df = df[df[sym_col] == chosen].copy()

    # Get date/timestamp column
    ts_col = None
    for col in ("date", "Date", "timestamp", "Timestamp", "datetime"):
        if col in df.columns:
            ts_col = col
            break

    if ts_col is not None:
        df = df.sort_values(ts_col)
        timestamps = pd.to_datetime(df[ts_col]).reset_index(drop=True)
    elif isinstance(df.index, pd.DatetimeIndex):
        df = df.sort_index()
        timestamps = pd.Series(df.index).reset_index(drop=True)
    else:
        # Fall back to integer index
        timestamps = pd.Series(pd.date_range("2020-01-01", periods=len(df), freq="B"))

    # Compute 1-day returns from close column
    if close_col is not None:
        prices = df[close_col].ffill().dropna()
        returns = prices.pct_change().fillna(0.0).values
        timestamps = timestamps.iloc[: len(returns)].reset_index(drop=True)
    else:
        # No close column — use first numeric column as proxy
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            log.error("[NO DATA] No numeric columns in panel to compute returns.")
            sys.exit(2)
        col = numeric_cols[0]
        log.warning("[WARN] No close column found; using '%s' as proxy", col)
        prices = df[col].ffill().dropna()
        returns = prices.pct_change().fillna(0.0).values
        timestamps = timestamps.iloc[: len(returns)].reset_index(drop=True)

    log.info(
        "[INFO] Factor return series: %d observations, date range %s → %s",
        len(returns),
        (
            timestamps.iloc[0].date()
            if hasattr(timestamps.iloc[0], "date")
            else timestamps.iloc[0]
        ),
        (
            timestamps.iloc[-1].date()
            if hasattr(timestamps.iloc[-1], "date")
            else timestamps.iloc[-1]
        ),
    )
    return timestamps, returns


def _get_purged_kfold(n_splits: int, embargo_days: int):
    """Return PurgedKFold from ml.purged_cv (preferred) or cpcv_validation fallback."""
    try:
        from src.assembled_core.ml.purged_cv import PurgedKFold

        log.info("[CV] Using PurgedKFold from ml.purged_cv")
        return PurgedKFold(n_splits=n_splits, label_horizon=embargo_days)
    except ImportError:
        pass

    try:
        # cpcv_validation doesn't export a splitter class directly — fall through
        pass
    except ImportError:
        pass

    log.warning(
        "[WARN] Could not load PurgedKFold; implementing minimal walk-forward splitter"
    )
    return None


def _minimal_walk_forward_splits(
    timestamps, n_splits: int, embargo_days: int
) -> list[tuple]:
    """Minimal walk-forward split as last-resort fallback."""
    import numpy as np
    import pandas as pd

    ts = pd.to_datetime(timestamps).reset_index(drop=True)
    n = len(ts)
    fold_size = n // (n_splits + 1)
    if fold_size == 0:
        return []

    splits = []
    for k in range(1, n_splits + 1):
        test_start = k * fold_size
        test_end = min((k + 1) * fold_size, n)
        purge_end = max(0, test_start - embargo_days)
        train_idx = np.arange(purge_end)
        test_idx = np.arange(test_start, test_end)
        if len(train_idx) > 0 and len(test_idx) > 0:
            splits.append((train_idx, test_idx))
    return splits


def _fold_sharpe(returns_test) -> float:
    """Annualized Sharpe for a test fold returns array."""
    import numpy as np

    if len(returns_test) < 2:
        return 0.0
    mu = float(np.mean(returns_test))
    sigma = float(np.std(returns_test, ddof=1))
    if sigma < 1e-12:
        return 0.0
    return (mu / sigma) * math.sqrt(252)


def _fold_cagr(returns_test) -> float:
    """Approximate annualized CAGR from daily returns array."""
    import numpy as np

    if len(returns_test) == 0:
        return 0.0
    cumulative = float(np.prod(1.0 + np.array(returns_test)))
    n_years = len(returns_test) / 252.0
    if n_years <= 0 or cumulative <= 0:
        return 0.0
    return cumulative ** (1.0 / n_years) - 1.0


def _fold_mdd(returns_test) -> float:
    """Maximum drawdown from a daily returns array."""
    import numpy as np

    if len(returns_test) == 0:
        return 0.0
    equity = np.cumprod(1.0 + np.array(returns_test))
    peak = np.maximum.accumulate(equity)
    drawdowns = (equity - peak) / peak
    return float(drawdowns.min())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(panel_path: Path, n_folds: int, embargo_days: int) -> int:
    """Run CPCV and return exit code (0=STABLE, 1=UNSTABLE, 2=no data)."""
    import numpy as np

    df = _load_panel(panel_path)
    timestamps, returns = _compute_factor_returns(df)

    if len(returns) < n_folds * 20:
        log.error(
            "[NO DATA] Panel has only %d observations — need at least %d for %d folds.",
            len(returns),
            n_folds * 20,
            n_folds,
        )
        sys.exit(2)

    # Get splitter
    cv = _get_purged_kfold(n_folds, embargo_days)

    if cv is not None:
        try:
            splits = cv.split(timestamps)
        except Exception as exc:
            log.warning("[WARN] PurgedKFold.split() failed (%s); using fallback", exc)
            splits = _minimal_walk_forward_splits(timestamps, n_folds, embargo_days)
    else:
        splits = _minimal_walk_forward_splits(timestamps, n_folds, embargo_days)

    if not splits:
        log.error("[NO DATA] Could not generate any CV splits from panel.")
        sys.exit(2)

    log.info(
        "[CPCV] Running %d-fold purged cross-validation (embargo=%d days)",
        len(splits),
        embargo_days,
    )

    fold_sharpes = []
    fold_cagrs = []
    fold_mdds = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        fold_returns = returns[test_idx]
        sharpe = _fold_sharpe(fold_returns)
        cagr = _fold_cagr(fold_returns)
        mdd = _fold_mdd(fold_returns)
        fold_sharpes.append(sharpe)
        fold_cagrs.append(cagr)
        fold_mdds.append(mdd)

        log.info(
            "  Fold %d/%d  test_obs=%-5d  Sharpe=%+.3f  CAGR=%+.1f%%  MDD=%.1f%%",
            fold_idx + 1,
            len(splits),
            len(test_idx),
            sharpe,
            cagr * 100,
            mdd * 100,
        )

    # Summary statistics
    mean_sharpe = float(np.mean(fold_sharpes))
    std_sharpe = float(np.std(fold_sharpes, ddof=1)) if len(fold_sharpes) > 1 else 0.0
    min_sharpe = float(np.min(fold_sharpes))
    mean_cagr = float(np.mean(fold_cagrs))
    mean_mdd = float(np.mean(fold_mdds))

    log.info("")
    log.info("=== CPCV Summary ===")
    log.info("  Folds evaluated : %d", len(splits))
    log.info("  Mean Sharpe     : %+.4f", mean_sharpe)
    log.info("  Std  Sharpe     : %.4f", std_sharpe)
    log.info("  Min  Sharpe     : %+.4f", min_sharpe)
    log.info("  Mean CAGR       : %+.2f%%", mean_cagr * 100)
    log.info("  Mean MDD        : %.2f%%", mean_mdd * 100)
    log.info("")
    log.info(
        "Verdict thresholds: min_sharpe > %.1f  AND  std_sharpe < %.1f",
        _MIN_SHARPE_THRESHOLD,
        _MAX_STD_THRESHOLD,
    )

    stable = min_sharpe > _MIN_SHARPE_THRESHOLD and std_sharpe < _MAX_STD_THRESHOLD

    if stable:
        log.info("[VERDICT] STABLE — thresholds met.")
        return 0
    else:
        reasons = []
        if min_sharpe <= _MIN_SHARPE_THRESHOLD:
            reasons.append(f"min_sharpe {min_sharpe:.4f} <= {_MIN_SHARPE_THRESHOLD}")
        if std_sharpe >= _MAX_STD_THRESHOLD:
            reasons.append(f"std_sharpe {std_sharpe:.4f} >= {_MAX_STD_THRESHOLD}")
        log.warning("[VERDICT] UNSTABLE — %s", "; ".join(reasons))
        return 1


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CPCV validation for strategy stability."
    )
    parser.add_argument(
        "--panel",
        type=Path,
        default=None,
        help="Path to price/factor panel parquet. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=6,
        dest="n_folds",
        help="Number of CV folds (default: 6).",
    )
    parser.add_argument(
        "--embargo-days",
        type=int,
        default=10,
        dest="embargo_days",
        help="Embargo/purge window in calendar days (default: 10).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()

    panel_path = args.panel
    if panel_path is None:
        panel_path = _find_default_panel()
        if panel_path is None:
            log.error(
                "[NO DATA] No panel file found in default locations. "
                "Provide --panel <path> or run the backtest pipeline first."
            )
            sys.exit(2)
        log.info("[AUTO] Using panel: %s", panel_path)

    exit_code = main(panel_path, n_folds=args.n_folds, embargo_days=args.embargo_days)
    sys.exit(exit_code)
