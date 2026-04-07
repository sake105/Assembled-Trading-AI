"""Cross-asset correlation and dispersion features.

Provides features that capture the correlation regime of the market:

- ``avg_pairwise_corr_20d`` / ``_60d``: Rolling average pairwise correlation
  across all symbols.  Low → diversification intact.  High → herding/crisis.
- ``return_dispersion``: Cross-sectional return dispersion (std of returns
  across symbols per day).  Low dispersion → herd behaviour, high → stock
  picking opportunity.
- ``intra_sector_dispersion`` / ``inter_sector_dispersion``: Sector-level
  decomposition — collapse of intra-sector dispersion signals sector crash risk.
- ``corr_to_vix``: Rolling correlation of each asset to VIX — assets with high
  VIX correlation are fragile in stress.
- ``corr_regime_zscore``: z-score of current avg correlation vs. 252d history —
  how unusual is the current correlation regime.

These features are critical for portfolio construction (diversification
monitoring) and regime detection (correlation breakdown signals crises).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_avg_pairwise_correlation(
    returns_wide: pd.DataFrame,
    windows: tuple[int, ...] = (20, 60),
) -> pd.DataFrame:
    """Compute rolling average pairwise correlation across assets.

    Args:
        returns_wide: Wide-format DataFrame (index=dates, columns=symbols)
            containing daily returns.
        windows: Rolling windows in trading days.

    Returns:
        DataFrame indexed by date with ``avg_pairwise_corr_{w}d`` columns.
    """
    if returns_wide.empty or returns_wide.shape[1] < 2:
        return pd.DataFrame(index=returns_wide.index)

    results = pd.DataFrame(index=returns_wide.index)

    for w in windows:
        col = f"avg_pairwise_corr_{w}d"
        avg_corrs = []
        for i in range(len(returns_wide)):
            if i < w - 1:
                avg_corrs.append(np.nan)
                continue
            window_data = returns_wide.iloc[i - w + 1:i + 1]
            corr_mat = window_data.corr()
            # Extract upper triangle (exclude diagonal)
            n = len(corr_mat)
            if n < 2:
                avg_corrs.append(np.nan)
                continue
            mask = np.triu(np.ones((n, n), dtype=bool), k=1)
            upper_vals = corr_mat.values[mask]
            upper_vals = upper_vals[np.isfinite(upper_vals)]
            avg_corrs.append(float(np.mean(upper_vals)) if len(upper_vals) > 0 else np.nan)

        results[col] = avg_corrs

    return results


def compute_return_dispersion(
    returns_wide: pd.DataFrame,
) -> pd.Series:
    """Cross-sectional return dispersion per day.

    Dispersion = std of returns across all symbols on each date.
    Low dispersion → herd behaviour (all stocks move together).
    High dispersion → idiosyncratic moves (stock-picking opportunity).

    Args:
        returns_wide: Wide-format returns (dates × symbols).

    Returns:
        Series of cross-sectional standard deviation per date.
    """
    return returns_wide.std(axis=1, skipna=True).rename("return_dispersion")


def compute_sector_dispersion(
    returns_wide: pd.DataFrame,
    sector_map: dict[str, str],
) -> pd.DataFrame:
    """Compute intra-sector and inter-sector dispersion.

    Args:
        returns_wide: Wide-format returns (dates × symbols).
        sector_map: Dict mapping symbol → sector name.

    Returns:
        DataFrame with ``intra_sector_dispersion`` and
        ``inter_sector_dispersion`` columns.
    """
    # Map symbols to sectors
    available = [s for s in returns_wide.columns if s in sector_map]
    if len(available) < 2:
        return pd.DataFrame(
            {"intra_sector_dispersion": np.nan, "inter_sector_dispersion": np.nan},
            index=returns_wide.index,
        )

    sectors = pd.Series({s: sector_map[s] for s in available})
    unique_sectors = sectors.unique()

    intra_disps = []
    inter_disps = []

    for idx in returns_wide.index:
        row = returns_wide.loc[idx, available]

        # Sector means
        sector_means = {}
        sector_stds = []
        for sec in unique_sectors:
            sec_symbols = sectors[sectors == sec].index.tolist()
            sec_returns = row[sec_symbols].dropna()
            if len(sec_returns) >= 2:
                sector_stds.append(float(sec_returns.std()))
            sector_means[sec] = float(sec_returns.mean()) if len(sec_returns) > 0 else np.nan

        # Intra-sector: average within-sector dispersion
        intra = float(np.mean(sector_stds)) if sector_stds else np.nan
        intra_disps.append(intra)

        # Inter-sector: dispersion of sector means
        mean_vals = [v for v in sector_means.values() if np.isfinite(v)]
        inter = float(np.std(mean_vals)) if len(mean_vals) >= 2 else np.nan
        inter_disps.append(inter)

    return pd.DataFrame(
        {"intra_sector_dispersion": intra_disps, "inter_sector_dispersion": inter_disps},
        index=returns_wide.index,
    )


def compute_correlation_to_benchmark(
    returns_wide: pd.DataFrame,
    benchmark_returns: pd.Series,
    window: int = 60,
) -> pd.DataFrame:
    """Rolling correlation of each asset to a benchmark (e.g., VIX or SPY).

    Args:
        returns_wide: Wide-format returns (dates × symbols).
        benchmark_returns: Series of benchmark returns, same index as
            ``returns_wide``.
        window: Rolling correlation window in days.

    Returns:
        DataFrame with ``corr_to_benchmark_{symbol}`` for each symbol.
    """
    results = pd.DataFrame(index=returns_wide.index)

    for sym in returns_wide.columns:
        col_name = f"corr_to_benchmark_{sym}"
        results[col_name] = returns_wide[sym].rolling(window, min_periods=20).corr(benchmark_returns)

    return results


def compute_correlation_regime_features(
    returns_wide: pd.DataFrame,
    short_window: int = 20,
    long_window: int = 60,
    history_window: int = 252,
) -> pd.DataFrame:
    """Compute correlation regime features.

    Produces a compact set of market-wide correlation features:

    - ``avg_corr_short``: Short-window average pairwise correlation
    - ``avg_corr_long``: Long-window average pairwise correlation
    - ``corr_regime_zscore``: z-score of short-window corr vs 252d history
    - ``corr_momentum``: 5d change in average correlation (rising = stress)

    Args:
        returns_wide: Wide-format returns (dates × symbols).
        short_window: Short rolling window.
        long_window: Long rolling window.
        history_window: History for z-scoring.

    Returns:
        DataFrame with correlation regime features.
    """
    corr_df = compute_avg_pairwise_correlation(
        returns_wide, windows=(short_window, long_window),
    )

    short_col = f"avg_pairwise_corr_{short_window}d"
    long_col = f"avg_pairwise_corr_{long_window}d"

    result = pd.DataFrame(index=returns_wide.index)
    result["avg_corr_short"] = corr_df[short_col]
    result["avg_corr_long"] = corr_df[long_col]

    # Z-score of short-window correlation vs. rolling history
    rolling_mean = result["avg_corr_short"].rolling(history_window, min_periods=60).mean()
    rolling_std = result["avg_corr_short"].rolling(history_window, min_periods=60).std()
    result["corr_regime_zscore"] = (result["avg_corr_short"] - rolling_mean) / rolling_std.replace(0, np.nan)

    # Correlation momentum (5d change)
    result["corr_momentum"] = result["avg_corr_short"].diff(5)

    return result


def build_correlation_features_panel(
    prices_df: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    symbol_col: str = "symbol",
    close_col: str = "close",
    sector_map: dict[str, str] | None = None,
    benchmark_col: str | None = None,
) -> pd.DataFrame:
    """Build correlation features from panel-format price data.

    Convenience function that converts panel prices to wide returns,
    computes all correlation features, and returns them in panel format.

    Args:
        prices_df: Panel DataFrame with timestamp, symbol, close.
        timestamp_col: Timestamp column.
        symbol_col: Symbol column.
        close_col: Close price column.
        sector_map: Optional symbol → sector mapping.
        benchmark_col: Optional symbol to use as benchmark for
            per-asset correlation (e.g., "^VIX").

    Returns:
        Panel DataFrame (timestamp, symbol + feature columns).
    """
    if prices_df is None or prices_df.empty:
        return pd.DataFrame()

    # Pivot to wide format
    wide = prices_df.pivot_table(
        index=timestamp_col, columns=symbol_col, values=close_col,
    )
    # Compute returns
    returns_wide = wide.pct_change().dropna(how="all")

    if returns_wide.empty or returns_wide.shape[1] < 2:
        return pd.DataFrame()

    # Market-wide features (same for all symbols on a given date)
    regime_features = compute_correlation_regime_features(returns_wide)
    dispersion = compute_return_dispersion(returns_wide)
    regime_features["return_dispersion"] = dispersion

    # Sector dispersion (if available)
    if sector_map:
        sector_disp = compute_sector_dispersion(returns_wide, sector_map)
        regime_features = regime_features.join(sector_disp)

    # Per-asset benchmark correlation
    if benchmark_col and benchmark_col in returns_wide.columns:
        bench_ret = returns_wide[benchmark_col]
        other_cols = [c for c in returns_wide.columns if c != benchmark_col]
        bench_corr = compute_correlation_to_benchmark(
            returns_wide[other_cols], bench_ret, window=60,
        )

    # Convert market-wide features back to panel format
    rows = []
    for sym in returns_wide.columns:
        sym_regime = regime_features.copy()
        sym_regime[symbol_col] = sym
        sym_regime[timestamp_col] = sym_regime.index

        # Add per-asset benchmark correlation if computed
        if benchmark_col and benchmark_col in returns_wide.columns and sym != benchmark_col:
            corr_col = f"corr_to_benchmark_{sym}"
            if corr_col in bench_corr.columns:
                sym_regime["corr_to_benchmark"] = bench_corr[corr_col].values

        rows.append(sym_regime)

    if not rows:
        return pd.DataFrame()

    result = pd.concat(rows, ignore_index=True)
    return result


__all__ = [
    "build_correlation_features_panel",
    "compute_avg_pairwise_correlation",
    "compute_correlation_regime_features",
    "compute_correlation_to_benchmark",
    "compute_return_dispersion",
    "compute_sector_dispersion",
]
