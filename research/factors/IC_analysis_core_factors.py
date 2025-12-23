"""IC Analysis Workflow for Core Factors.

This script demonstrates a simple workflow for factor IC analysis:
1. Load price data from local Parquet files
2. Compute core TA/Price factors and volatility/liquidity factors
3. Add forward returns
4. Compute IC and Rank-IC
5. Summarize IC statistics
6. Compute rolling IC statistics

Usage:
    # Set environment variable for local data root
    $env:ASSEMBLED_LOCAL_DATA_ROOT = "F:\Python_Projekt\Aktiengerüst\datensammlungen\altdaten\stand 3-12-2025"

    # Run the script
    python research/factors/IC_analysis_core_factors.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


from src.assembled_core.config.settings import get_settings
from src.assembled_core.data.data_source import get_price_data_source
from src.assembled_core.features.ta_factors_core import build_core_ta_factors
from src.assembled_core.features.ta_liquidity_vol_factors import (
    add_realized_volatility,
    add_turnover_and_liquidity_proxies,
)
from src.assembled_core.qa.factor_analysis import (
    add_forward_returns,
    compute_ic,
    compute_rank_ic,
    summarize_ic_series,
    compute_rolling_ic,
)


def load_symbols_from_file(symbols_file: Path) -> list[str]:
    """Load symbols from a text file (one per line, skip comments)."""
    symbols = []
    with symbols_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                symbols.append(line.upper())
    return symbols


def main():
    """Main workflow for IC analysis."""
    print("=" * 60)
    print("Factor IC Analysis Workflow")
    print("=" * 60)
    print()

    # Configuration
    symbols_file = ROOT / "config" / "macro_world_etfs_tickers.txt"
    start_date = "2010-01-01"
    end_date = "2025-12-03"
    horizons = [20, 60]  # 1 month and 3 months forward returns
    fwd_horizon_for_ic = 20  # Use 20-day forward returns for IC computation

    # 1. Load price data
    print("Step 1: Loading price data...")
    settings = get_settings()

    # Check if local data root is set
    if not settings.local_data_root:
        print("ERROR: ASSEMBLED_LOCAL_DATA_ROOT environment variable not set.")
        print("Please set it to your local data directory, e.g.:")
        print(
            '  $env:ASSEMBLED_LOCAL_DATA_ROOT = "F:\\Python_Projekt\\Aktiengerüst\\datensammlungen\\altdaten\\stand 3-12-2025"'
        )
        return 1

    price_source = get_price_data_source(settings, data_source="local")

    # Load symbols
    if symbols_file.exists():
        symbols = load_symbols_from_file(symbols_file)
        print(f"  Loaded {len(symbols)} symbols from {symbols_file.name}")
        # Limit to first 10 symbols for faster execution
        symbols = symbols[:10]
        print(f"  Using first {len(symbols)} symbols: {symbols}")
    else:
        # Fallback: use a small set of symbols
        symbols = ["SPY", "ACWI", "VT"]
        print(f"  Symbols file not found, using default symbols: {symbols}")

    # Load prices
    prices = price_source.get_history(
        symbols=symbols, start_date=start_date, end_date=end_date, freq="1d"
    )

    if prices.empty:
        print("ERROR: No price data loaded. Check data source and symbols.")
        return 1

    print(f"  Loaded {len(prices)} rows for {prices['symbol'].nunique()} symbols")
    print(f"  Date range: {prices['timestamp'].min()} to {prices['timestamp'].max()}")
    print()

    # 2. Compute factors
    print("Step 2: Computing factors...")

    # Core TA/Price factors
    print("  Computing core TA/Price factors...")
    factors_df = build_core_ta_factors(prices)

    # Volatility factors
    print("  Computing volatility factors...")
    factors_df = add_realized_volatility(factors_df, windows=[20, 60])

    # Liquidity factors (if volume available)
    if "volume" in factors_df.columns:
        print("  Computing liquidity factors...")
        factors_df = add_turnover_and_liquidity_proxies(factors_df)
    else:
        print("  Skipping liquidity factors (no volume column)")

    print(f"  Total columns: {len(factors_df.columns)}")
    print(
        f"  Factor columns: {[c for c in factors_df.columns if c not in ['timestamp', 'symbol', 'close', 'open', 'high', 'low', 'volume']]}"
    )
    print()

    # 3. Add forward returns
    print(f"Step 3: Adding forward returns (horizons: {horizons})...")
    factors_df = add_forward_returns(
        factors_df, horizon_days=horizons, return_type="log"
    )

    # Determine forward return column name
    if len(horizons) == 1:
        fwd_return_col = f"fwd_return_{fwd_horizon_for_ic}d"
    else:
        fwd_return_col = f"fwd_ret_{fwd_horizon_for_ic}"

    print(f"  Using forward return column: {fwd_return_col}")
    print()

    # 4. Compute IC and Rank-IC
    print("Step 4: Computing IC and Rank-IC...")
    ic_df = compute_ic(
        factors_df,
        forward_returns_col=fwd_return_col,
        group_col="symbol",
        method="pearson",
    )

    rank_ic_df = compute_rank_ic(
        factors_df, forward_returns_col=fwd_return_col, group_col="symbol"
    )

    print(
        f"  Computed IC for {len(ic_df.columns)} factors across {len(ic_df)} timestamps"
    )
    print()

    # 5. Summarize IC statistics
    print("Step 5: Summarizing IC statistics...")
    summary_ic = summarize_ic_series(ic_df)
    summary_rank_ic = summarize_ic_series(rank_ic_df)

    print()
    print("=" * 60)
    print("IC Summary (Pearson Correlation)")
    print("=" * 60)
    display_cols = ["factor", "mean_ic", "std_ic", "ic_ir", "hit_ratio", "count"]
    print(summary_ic[display_cols].to_string(index=False))
    print()

    print("=" * 60)
    print("Rank-IC Summary (Spearman Rank Correlation)")
    print("=" * 60)
    print(summary_rank_ic[display_cols].to_string(index=False))
    print()

    # 6. Rolling IC analysis
    print("Step 6: Computing rolling IC statistics...")
    rolling_ic = compute_rolling_ic(ic_df, window=60)
    print(f"  Computed rolling IC for {len(rolling_ic.columns) // 2} factors")
    print()

    # 7. Simple visualization (optional)
    try:
        import matplotlib.pyplot as plt

        print("Step 7: Creating simple plots...")

        # Plot IC time-series for top 3 factors
        top_factors = summary_ic.head(3)["factor"].values

        fig, axes = plt.subplots(
            len(top_factors), 1, figsize=(12, 3 * len(top_factors))
        )
        if len(top_factors) == 1:
            axes = [axes]

        for i, factor in enumerate(top_factors):
            ic_col = f"ic_{factor}"
            if ic_col in ic_df.columns:
                axes[i].plot(
                    ic_df.index, ic_df[ic_col], label=f"IC: {factor}", alpha=0.7
                )
                axes[i].axhline(y=0, color="r", linestyle="--", alpha=0.5)
                axes[i].set_title(f"IC Time-Series: {factor}")
                axes[i].set_xlabel("Date")
                axes[i].set_ylabel("IC")
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = ROOT / "output" / "factor_ic_timeseries.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Saved IC time-series plot to {output_path}")
        plt.close()

    except ImportError:
        print("  Skipping plots (matplotlib not available)")

    print()
    print("=" * 60)
    print("IC Analysis Workflow Completed Successfully")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
