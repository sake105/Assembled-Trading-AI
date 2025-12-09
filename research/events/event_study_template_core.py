"""Event Study Research Workflow Template.

This script demonstrates a simple workflow for event study analysis:
1. Load price data from local Parquet files (Alt-Daten)
2. Generate synthetic events (e.g., pseudo-earnings every 60 days)
3. Build event windows around events
4. Compute normal and abnormal returns
5. Aggregate results across events
6. Visualize Average/Cumulative Abnormal Returns
7. Optionally track experiment

Usage:
    # Set environment variable for local data root
    $env:ASSEMBLED_LOCAL_DATA_ROOT = "F:\Python_Projekt\Aktiengerüst\datensammlungen\altdaten\stand 3-12-2025"
    
    # Run the script
    python research/events/event_study_template_core.py

TODO: Integration with Real Event Data Sources
===============================================
This workflow is currently provider-agnostic and uses synthetic events for testing.
To integrate real events, replace the `generate_synthetic_events()` function with:

1. **Finnhub Earnings Calendar API:**
   ```python
   import requests
   
   def load_earnings_events(symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
       events = []
       for symbol in symbols:
           url = f"https://finnhub.io/api/v1/calendar/earnings"
           params = {
               "symbol": symbol,
               "from": start_date,
               "to": end_date,
               "token": settings.finnhub_api_key
           }
           response = requests.get(url, params=params)
           data = response.json()
           for earnings in data.get("earningsCalendar", []):
               events.append({
                   "timestamp": pd.Timestamp(earnings["date"], tz="UTC"),
                   "symbol": symbol,
                   "event_type": "earnings",
                   "event_id": f"earnings_{earnings['date']}_{symbol}",
                   "payload": {"eps_estimate": earnings.get("epsEstimate"), ...}
               })
       return pd.DataFrame(events)
   ```

2. **Finnhub Insider Transactions:**
   ```python
   def load_insider_events(symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
       # Similar structure, using /stock/insider-transactions endpoint
       ...
   ```

3. **CSV/Parquet Files:**
   ```python
   def load_events_from_file(file_path: Path) -> pd.DataFrame:
       df = pd.read_csv(file_path)  # or pd.read_parquet()
       # Ensure columns: timestamp, symbol, event_type, event_id (optional), payload (optional)
       df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
       return df
   ```

The event DataFrame format is standardized:
- Required: timestamp (UTC), symbol, event_type
- Optional: event_id (auto-generated if missing), payload (dict or additional columns)
- Compatible with build_event_window_prices() function
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd

from src.assembled_core.config.settings import get_settings
from src.assembled_core.data.data_source import get_price_data_source
from src.assembled_core.qa.event_study import (
    aggregate_event_study,
    build_event_window_prices,
    compute_event_returns,
)

# Optional: Experiment tracking
try:
    from src.assembled_core.qa.experiment_tracking import ExperimentTracker
    EXPERIMENT_TRACKING_AVAILABLE = True
except ImportError:
    EXPERIMENT_TRACKING_AVAILABLE = False
    print("Warning: Experiment tracking not available")


def load_symbols_from_file(symbols_file: Path) -> list[str]:
    """Load symbols from a text file (one per line, skip comments)."""
    symbols = []
    with symbols_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                symbols.append(line)
    return symbols


def generate_synthetic_events(
    prices_df: pd.DataFrame,
    event_type: str = "earnings",
    interval_days: int = 60,
    group_col: str = "symbol",
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Generate synthetic events for testing.
    
    Creates pseudo-events every `interval_days` days per symbol.
    This is a placeholder for real event data sources (Finnhub, CSV, etc.).
    
    Args:
        prices_df: Panel DataFrame with timestamp, symbol
        event_type: Type of event to generate (default: "earnings")
        interval_days: Interval between events in days (default: 60)
        group_col: Column name for grouping (default: "symbol")
        timestamp_col: Column name for timestamp (default: "timestamp")
    
    Returns:
        DataFrame with columns: timestamp, symbol, event_type
    """
    events = []
    
    for symbol in prices_df[group_col].unique():
        symbol_prices = prices_df[prices_df[group_col] == symbol].sort_values(timestamp_col)
        
        if symbol_prices.empty:
            continue
        
        # Get date range for this symbol
        first_date = symbol_prices[timestamp_col].min()
        last_date = symbol_prices[timestamp_col].max()
        
        # Generate events every interval_days
        current_date = first_date
        event_count = 0
        
        while current_date <= last_date:
            # Check if this date exists in prices
            matching_prices = symbol_prices[symbol_prices[timestamp_col] == current_date]
            
            if not matching_prices.empty:
                events.append({
                    timestamp_col: current_date,
                    group_col: symbol,
                    "event_type": event_type,
                })
                event_count += 1
            
            # Move to next event date
            current_date = current_date + pd.Timedelta(days=interval_days)
    
    events_df = pd.DataFrame(events)
    
    if not events_df.empty:
        # Ensure UTC-aware timestamps
        events_df[timestamp_col] = pd.to_datetime(events_df[timestamp_col], utc=True)
    
    return events_df


def visualize_event_study_results(
    aggregated_df: pd.DataFrame,
    output_path: Path | None = None,
    title: str = "Event Study: Average and Cumulative Abnormal Returns",
) -> None:
    """Visualize event study results.
    
    Creates a plot showing:
    - Average Abnormal Return (AAR) over relative days
    - Cumulative Abnormal Return (CAAR) over relative days
    - Confidence intervals (if available)
    
    Args:
        aggregated_df: Output from aggregate_event_study()
        output_path: Optional path to save plot (default: None, shows plot)
        title: Plot title (default: "Event Study: Average and Cumulative Abnormal Returns")
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    rel_days = aggregated_df["rel_day"].values
    avg_ret = aggregated_df["avg_ret"].values
    cum_ret = aggregated_df["cum_ret"].values
    
    # Plot 1: Average Abnormal Return
    ax1.plot(rel_days, avg_ret, marker="o", label="Average Abnormal Return", linewidth=2)
    
    # Add confidence intervals if available
    if "ci_lower" in aggregated_df.columns and "ci_upper" in aggregated_df.columns:
        ci_lower = aggregated_df["ci_lower"].values
        ci_upper = aggregated_df["ci_upper"].values
        ax1.fill_between(rel_days, ci_lower, ci_upper, alpha=0.3, label="95% CI")
    
    ax1.axhline(y=0, color="r", linestyle="--", linewidth=1, label="Zero line")
    ax1.axvline(x=0, color="gray", linestyle="--", linewidth=1, label="Event day")
    ax1.set_ylabel("Average Abnormal Return", fontsize=12)
    ax1.set_title(title, fontsize=14, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Cumulative Abnormal Return
    ax2.plot(rel_days, cum_ret, marker="s", label="Cumulative Abnormal Return (CAAR)", 
             linewidth=2, color="green")
    ax2.axhline(y=0, color="r", linestyle="--", linewidth=1, label="Zero line")
    ax2.axvline(x=0, color="gray", linestyle="--", linewidth=1, label="Event day")
    ax2.set_xlabel("Relative Day (0 = Event Day)", fontsize=12)
    ax2.set_ylabel("Cumulative Abnormal Return", fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def load_real_earnings_events(
    output_dir: Path | None = None,
    symbols: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Load real earnings events from Parquet file.
    
    Args:
        output_dir: Optional output directory (default: output/altdata)
        symbols: Optional list of symbols to filter (if None, load all)
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)
    
    Returns:
        DataFrame with earnings events (columns: timestamp, symbol, event_type, ...)
    """
    if output_dir is None:
        output_dir = ROOT / "output" / "altdata"
    else:
        output_dir = Path(output_dir)
    
    earnings_path = output_dir / "events_earnings.parquet"
    
    if not earnings_path.exists():
        print(f"Warning: Earnings events file not found: {earnings_path}")
        return pd.DataFrame()
    
    try:
        events = pd.read_parquet(earnings_path)
        
        # Ensure UTC-aware timestamps
        if "timestamp" in events.columns:
            events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True)
        
        # Filter by symbols if provided
        if symbols is not None:
            events = events[events["symbol"].isin(symbols)]
        
        # Filter by date range if provided
        if start_date:
            start_dt = pd.to_datetime(start_date, utc=True)
            events = events[events["timestamp"] >= start_dt]
        
        if end_date:
            end_dt = pd.to_datetime(end_date, utc=True)
            events = events[events["timestamp"] <= end_dt]
        
        print(f"Loaded {len(events)} earnings events from {earnings_path}")
        return events
    
    except Exception as e:
        print(f"Error loading earnings events: {e}")
        return pd.DataFrame()


def main():
    """Main workflow for event study analysis."""
    print("=" * 60)
    print("Event Study Research Workflow")
    print("=" * 60)
    print()
    
    # Load settings
    settings = get_settings()
    
    # Configuration
    symbols_file = ROOT / "config" / "macro_world_etfs_tickers.txt"
    freq = "1d"
    start_date = "2010-01-01"
    end_date = "2025-12-03"
    window_before = 20
    window_after = 40
    event_interval_days = 60  # Generate pseudo-earnings every 60 days
    
    # Flag: Use real events instead of synthetic
    USE_REAL_EVENTS = False  # Set to True to load real earnings events from output/altdata/events_earnings.parquet
    
    # Load symbols
    if symbols_file.exists():
        symbols = load_symbols_from_file(symbols_file)
        print(f"Loaded {len(symbols)} symbols from {symbols_file}")
    else:
        # Fallback: use a small test universe
        symbols = ["SPY", "ACWI", "VT"]
        print(f"Using default symbols: {symbols}")
    
    # Load price data
    print(f"\nLoading price data: {len(symbols)} symbols, freq={freq}, "
          f"date range: {start_date} to {end_date}")
    
    price_source = get_price_data_source(
        settings,
        data_source="local",
    )
    
    prices = price_source.get_history(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        freq=freq,
    )
    
    print(f"Loaded {len(prices)} rows for {prices['symbol'].nunique()} symbols")
    print(f"Date range: {prices['timestamp'].min()} to {prices['timestamp'].max()}")
    
    # Load events (real or synthetic)
    if USE_REAL_EVENTS:
        print(f"\nLoading real earnings events from output/altdata/events_earnings.parquet...")
        events = load_real_earnings_events(
            output_dir=ROOT / "output" / "altdata",
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
        )
        
        if events.empty:
            print("Warning: No real events found. Falling back to synthetic events.")
            events = generate_synthetic_events(
                prices,
                event_type="earnings",
                interval_days=event_interval_days,
            )
            print(f"Generated {len(events)} synthetic events (fallback)")
        else:
            print(f"Loaded {len(events)} real earnings events")
            print(f"Events per symbol: {events.groupby('symbol').size().to_dict()}")
    else:
        # Generate synthetic events
        print(f"\nGenerating synthetic events (pseudo-earnings every {event_interval_days} days)...")
        events = generate_synthetic_events(
            prices,
            event_type="earnings",
            interval_days=event_interval_days,
        )
        
        print(f"Generated {len(events)} synthetic events")
        print(f"Events per symbol: {events.groupby('symbol').size().to_dict()}")
    
    if events.empty:
        print("Warning: No events generated. Exiting.")
        return
    
    # Build event windows
    print(f"\nBuilding event windows (window: -{window_before} to +{window_after} days)...")
    windows = build_event_window_prices(
        prices,
        events,
        window_before=window_before,
        window_after=window_after,
    )
    
    print(f"Built windows for {windows['event_id'].nunique()} events")
    print(f"Total window rows: {len(windows)}")
    
    if windows.empty:
        print("Warning: No event windows generated. Exiting.")
        return
    
    # Compute returns
    print("\nComputing event returns...")
    returns = compute_event_returns(
        windows,
        price_col="close",
        return_type="log",
    )
    
    print(f"Computed returns for {returns['event_id'].nunique()} events")
    
    # Aggregate results
    print("\nAggregating event study results...")
    aggregated = aggregate_event_study(
        returns,
        use_abnormal=False,  # Use normal returns (no benchmark for now)
        confidence_level=0.95,
    )
    
    print(f"Aggregated results for {len(aggregated)} relative days")
    print(f"Number of events: {aggregated['n_events'].max()}")
    
    # Display summary statistics
    print("\n" + "=" * 60)
    print("Event Study Summary Statistics")
    print("=" * 60)
    print(f"\nEvent Day (rel_day = 0):")
    event_day = aggregated[aggregated["rel_day"] == 0]
    if not event_day.empty:
        row = event_day.iloc[0]
        print(f"  Average Return: {row['avg_ret']:.4f}")
        print(f"  n_events: {row['n_events']}")
        if "ci_lower" in row and "ci_upper" in row:
            print(f"  95% CI: [{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]")
    
    print(f"\nPost-Event Window (rel_day 1-5):")
    post_window = aggregated[aggregated["rel_day"].between(1, 5)]
    if not post_window.empty:
        avg_post = post_window["avg_ret"].mean()
        cum_post = post_window["cum_ret"].iloc[-1]
        print(f"  Average Return: {avg_post:.4f}")
        print(f"  Cumulative Return: {cum_post:.4f}")
    
    print(f"\nFull Window (rel_day -{window_before} to +{window_after}):")
    full_window = aggregated
    avg_full = full_window["avg_ret"].mean()
    cum_full = full_window["cum_ret"].iloc[-1]
    print(f"  Average Return: {avg_full:.4f}")
    print(f"  Cumulative Return: {cum_full:.4f}")
    
    # Visualize results
    print("\nGenerating visualization...")
    output_dir = ROOT / "output" / "event_studies"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_path = output_dir / "event_study_synthetic_earnings.png"
    visualize_event_study_results(
        aggregated,
        output_path=plot_path,
        title=f"Event Study: Synthetic Earnings Events (n={aggregated['n_events'].max()})",
    )
    
    # Save aggregated results to CSV
    csv_path = output_dir / "event_study_synthetic_earnings.csv"
    aggregated.to_csv(csv_path, index=False)
    print(f"Saved aggregated results to: {csv_path}")
    
    # Optional: Track experiment
    if EXPERIMENT_TRACKING_AVAILABLE:
        try:
            tracker = ExperimentTracker(settings.experiments_dir)
            run = tracker.start_run(
                name="event_study_synthetic_earnings",
                tags=["events", "synthetic", "earnings", "research"],
                config={
                    "symbols": symbols,
                    "freq": freq,
                    "start_date": start_date,
                    "end_date": end_date,
                    "window_before": window_before,
                    "window_after": window_after,
                    "event_interval_days": event_interval_days,
                    "n_events": len(events),
                },
            )
            
            # Save artifacts
            tracker.log_artifact(run, str(csv_path), "aggregated_results.csv")
            tracker.log_artifact(run, str(plot_path), "plot.png")
            
            # Save metrics
            metrics = {
                "event_day_avg_return": event_day.iloc[0]["avg_ret"] if not event_day.empty else None,
                "post_window_avg_return": avg_post if not post_window.empty else None,
                "full_window_cum_return": cum_full,
                "n_events": len(events),
            }
            tracker.log_metrics(run, metrics)
            tracker.finish_run(run, status="finished")
            
            print(f"\nExperiment tracked: Run ID = {run.run_id}")
        except Exception as e:
            print(f"\nWarning: Experiment tracking failed: {e}")
    
    print("\n" + "=" * 60)
    print("Event Study Workflow Completed")
    print("=" * 60)


if __name__ == "__main__":
    main()

