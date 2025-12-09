"""Download historical price snapshot for one or multiple symbols (Alt-Daten).

This script downloads historical OHLCV data from Yahoo Finance for one or multiple symbols
and saves them as Parquet files in a format compatible with the backend pipeline.

Usage (single symbol):
    python scripts/download_historical_snapshot.py \\
      --symbol NVDA \\
      --start 2000-01-01 \\
      --end 2025-12-03 \\
      --interval 1d \\
      --target-root "F:/Python_Projekt/Aktiengerüst/datensammlungen/altdaten/stand 3-12-2025"

Usage (multiple symbols):
    python scripts/download_historical_snapshot.py \\
      --symbols NVDA AAPL MSFT \\
      --start 2000-01-01 \\
      --end 2025-12-03 \\
      --interval 1d \\
      --target-root "F:/.../altdaten/stand 3-12-2025" \\
      --sleep-seconds 2.0

Usage (from file):
    python scripts/download_historical_snapshot.py \\
      --symbols-file watchlist.txt \\
      --start 2000-01-01 \\
      --end 2025-12-03 \\
      --target-root "F:/.../altdaten/stand 3-12-2025"

Validation:
    python scripts/download_historical_snapshot.py \\
      --validate-only \\
      --target-root "F:/.../altdaten/stand 3-12-2025" \\
      --interval 1d

Output:
    <target-root>/<interval>/<SYMBOL>.parquet
    Example: F:/.../stand 3-12-2025/1d/NVDA.parquet

Integration:
    The downloaded Parquet files can be used as input for LocalParquetPriceDataSource
    by configuring ASSEMBLED_LOCAL_DATA_ROOT to point to the target-root directory.
    See docs/WORKFLOWS_EOD_AND_QA.md for details on using local data sources.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def download_symbol_history_finnhub(
    symbol: str,
    start_date: str,
    end_date: str | None,
    interval: str,
    api_key: str,
    max_retries: int = 3,
    retry_delay: int = 5,
) -> pd.DataFrame:
    """Download historical OHLCV data for a single symbol from Finnhub API.
    
    This function downloads the raw data and returns it as a DataFrame.
    It does NOT save the data - use store_symbol_df() for that.
    
    Args:
        symbol: Stock ticker symbol (e.g., "NVDA")
        start_date: Start date in format "YYYY-MM-DD" (e.g., "2000-01-01")
        end_date: End date in format "YYYY-MM-DD" (e.g., "2025-12-03") or None for today
        interval: Data interval ("1d" for daily, "5min" for 5-minute bars)
        api_key: Finnhub API key
        max_retries: Maximum number of retry attempts (default: 3)
        retry_delay: Initial delay in seconds before retrying (default: 5)
    
    Returns:
        DataFrame with OHLCV data (columns: timestamp, open, high, low, close, volume)
        Empty DataFrame if no data available
    
    Raises:
        RuntimeError: If download fails after all retries
        ValueError: If dates are invalid or symbol is empty
        ImportError: If requests is not installed
    """
    if not symbol or not symbol.strip():
        raise ValueError("Symbol cannot be empty")
    
    symbol = symbol.strip().upper()
    
    # Validate dates
    try:
        start_dt = pd.to_datetime(start_date)
        if end_date:
            end_dt = pd.to_datetime(end_date)
            if end_dt < start_dt:
                raise ValueError(f"End date {end_date} is before start date {start_date}")
    except Exception as e:
        raise ValueError(f"Invalid date format: {e}") from e
    
    # Import requests
    try:
        import requests
    except ImportError as e:
        raise ImportError(
            "requests is not installed. Install it with: pip install requests"
        ) from e
    
    # Map interval to Finnhub resolution
    resolution_map = {
        "1d": "D",
        "5min": "5",
    }
    if interval not in resolution_map:
        raise ValueError(f"Unsupported interval for Finnhub: {interval}. Supported: 1d, 5min")
    
    resolution = resolution_map[interval]
    
    # Convert dates to Unix timestamps
    start_dt = pd.to_datetime(start_date, utc=True)
    end_dt = pd.to_datetime(end_date, utc=True) if end_date else pd.Timestamp.now(tz="UTC")
    from_timestamp = int(start_dt.timestamp())
    to_timestamp = int(end_dt.timestamp())
    
    # Download data from Finnhub API with retry logic
    base_url = "https://finnhub.io/api/v1/stock/candle"
    current_delay = retry_delay
    
    for attempt in range(1, max_retries + 1):
        try:
            params = {
                "symbol": symbol,
                "resolution": resolution,
                "from": from_timestamp,
                "to": to_timestamp,
                "token": api_key,
            }
            
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for errors in response
            if data.get("s") == "no_data":
                logger.warning(
                    f"No data returned for {symbol} from {start_date} to {end_date}. "
                    f"Symbol may be invalid or date range contains no trading days."
                )
                return pd.DataFrame()
            
            if data.get("s") != "ok":
                error_msg = data.get("s", "unknown error")
                raise RuntimeError(f"Finnhub API error: {error_msg}")
            
            # Extract data arrays
            timestamps = data.get("t", [])
            opens = data.get("o", [])
            highs = data.get("h", [])
            lows = data.get("l", [])
            closes = data.get("c", [])
            volumes = data.get("v", [])
            
            if not timestamps:
                logger.warning(f"No timestamps in response for {symbol}")
                return pd.DataFrame()
            
            # Create DataFrame
            df = pd.DataFrame({
                "Date": pd.to_datetime(timestamps, unit="s", utc=True),
                "Open": opens,
                "High": highs,
                "Low": lows,
                "Close": closes,
                "Volume": volumes,
            })
            
            # Set Date as index (to match yfinance format)
            df = df.set_index("Date")
            
            logger.info(f"Downloaded {len(df)} rows for {symbol} from Finnhub (attempt {attempt}/{max_retries})")
            return df
            
        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                logger.warning(
                    f"Request failed for {symbol} (attempt {attempt}/{max_retries}): {e}. "
                    f"Waiting {current_delay} seconds before retry..."
                )
                time.sleep(current_delay)
                current_delay *= 2  # Exponential backoff
                continue
            else:
                raise RuntimeError(f"Failed to download {symbol} from Finnhub after {max_retries} attempts: {e}") from e
        except Exception as e:
            if attempt < max_retries:
                logger.warning(
                    f"Error downloading {symbol} (attempt {attempt}/{max_retries}): {e}. "
                    f"Waiting {current_delay} seconds before retry..."
                )
                time.sleep(current_delay)
                current_delay *= 2
                continue
            else:
                raise RuntimeError(f"Failed to download {symbol} from Finnhub after {max_retries} attempts: {e}") from e
    
    return pd.DataFrame()


def download_symbol_history_twelve_data(
    symbol: str,
    start_date: str,
    end_date: str | None,
    interval: str,
    api_key: str,
    max_retries: int = 3,
    retry_delay: int = 5,
) -> pd.DataFrame:
    """Download historical OHLCV data for a single symbol from Twelve Data API.
    
    This function downloads the raw data and returns it as a DataFrame.
    It does NOT save the data - use store_symbol_df() for that.
    
    Args:
        symbol: Stock ticker symbol (e.g., "NVDA")
        start_date: Start date in format "YYYY-MM-DD" (e.g., "2000-01-01")
        end_date: End date in format "YYYY-MM-DD" (e.g., "2025-12-03") or None for today
        interval: Data interval ("1d" for daily, "5min" for 5-minute bars)
        api_key: Twelve Data API key
        max_retries: Maximum number of retry attempts (default: 3)
        retry_delay: Initial delay in seconds before retrying (default: 5)
    
    Returns:
        DataFrame with OHLCV data (columns: Date index, Open, High, Low, Close, Volume)
        Empty DataFrame if no data available
    
    Raises:
        RuntimeError: If download fails after all retries
        ValueError: If dates are invalid or symbol is empty
        ImportError: If requests is not installed
    """
    if not symbol or not symbol.strip():
        raise ValueError("Symbol cannot be empty")
    
    symbol = symbol.strip().upper()
    
    # Validate dates
    try:
        start_dt = pd.to_datetime(start_date)
        if end_date:
            end_dt = pd.to_datetime(end_date)
            if end_dt < start_dt:
                raise ValueError(f"End date {end_date} is before start date {start_date}")
    except Exception as e:
        raise ValueError(f"Invalid date format: {e}") from e
    
    # Import requests
    try:
        import requests
    except ImportError as e:
        raise ImportError(
            "requests is not installed. Install it with: pip install requests"
        ) from e
    
    # Map interval to Twelve Data interval
    interval_map = {
        "1d": "1day",
        "5min": "5min",
    }
    if interval not in interval_map:
        raise ValueError(f"Unsupported interval for Twelve Data: {interval}. Supported: 1d, 5min")
    
    twelve_interval = interval_map[interval]
    
    # Convert end_date
    if end_date is None:
        end_date = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    
    # Download data from Twelve Data API with retry logic
    base_url = "https://api.twelvedata.com/time_series"
    current_delay = retry_delay
    
    for attempt in range(1, max_retries + 1):
        try:
            params = {
                "symbol": symbol,
                "interval": twelve_interval,
                "start_date": start_date,
                "end_date": end_date,
                "apikey": api_key,
                "outputsize": 5000,  # Maximum for free tier
            }
            
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for errors in response
            if "status" in data and data["status"] == "error":
                error_msg = data.get("message", "unknown error")
                logger.warning(f"Twelve Data API error for {symbol}: {error_msg}")
                if attempt < max_retries:
                    time.sleep(current_delay)
                    current_delay *= 2
                    continue
                else:
                    raise RuntimeError(f"Twelve Data API error: {error_msg}")
            
            # Extract values array
            values = data.get("values", [])
            if not values:
                logger.warning(
                    f"No data returned for {symbol} from {start_date} to {end_date}. "
                    f"Symbol may be invalid or date range contains no trading days."
                )
                return pd.DataFrame()
            
            # Create DataFrame from values
            # Twelve Data returns: datetime, open, high, low, close, volume
            df = pd.DataFrame(values)
            
            if df.empty:
                logger.warning(f"Empty DataFrame for {symbol}")
                return pd.DataFrame()
            
            # Rename datetime to Date and set as index (to match yfinance format)
            if "datetime" in df.columns:
                df = df.rename(columns={"datetime": "Date"})
                df["Date"] = pd.to_datetime(df["Date"], utc=True)
                df = df.set_index("Date")
            else:
                # Fallback: if no datetime column, create one from index
                logger.warning(f"No datetime column in response for {symbol}")
                return pd.DataFrame()
            
            # Ensure column names match yfinance format (capitalized)
            column_mapping = {
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
            df = df.rename(columns=column_mapping)
            
            logger.info(f"Downloaded {len(df)} rows for {symbol} from Twelve Data (attempt {attempt}/{max_retries})")
            return df
            
        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                logger.warning(
                    f"Request failed for {symbol} (attempt {attempt}/{max_retries}): {e}. "
                    f"Waiting {current_delay} seconds before retry..."
                )
                time.sleep(current_delay)
                current_delay *= 2  # Exponential backoff
                continue
            else:
                raise RuntimeError(f"Failed to download {symbol} from Twelve Data after {max_retries} attempts: {e}") from e
        except Exception as e:
            if attempt < max_retries:
                logger.warning(
                    f"Error downloading {symbol} (attempt {attempt}/{max_retries}): {e}. "
                    f"Waiting {current_delay} seconds before retry..."
                )
                time.sleep(current_delay)
                current_delay *= 2
                continue
            else:
                raise RuntimeError(f"Failed to download {symbol} from Twelve Data after {max_retries} attempts: {e}") from e
    
    return pd.DataFrame()


def download_symbol_history(
    symbol: str,
    start_date: str,
    end_date: str | None,
    interval: str,
    max_retries: int = 3,
    retry_delay: int = 5,
) -> pd.DataFrame:
    """Download historical OHLCV data for a single symbol from Yahoo Finance.
    
    This function downloads the raw data and returns it as a DataFrame.
    It does NOT save the data - use store_symbol_df() for that.
    
    Args:
        symbol: Stock ticker symbol (e.g., "NVDA")
        start_date: Start date in format "YYYY-MM-DD" (e.g., "2000-01-01")
        end_date: End date in format "YYYY-MM-DD" (e.g., "2025-12-03") or None for today
        interval: Data interval for yfinance (e.g., "1d" for daily, "1h" for hourly)
        max_retries: Maximum number of retry attempts for rate limits (default: 3)
        retry_delay: Initial delay in seconds before retrying (default: 5, exponential backoff)
    
    Returns:
        DataFrame with raw OHLCV data from yfinance (Index = Datetime, columns = Open, High, Low, Close, Volume, etc.)
        Empty DataFrame if no data available
    
    Raises:
        RuntimeError: If download fails after all retries
        ValueError: If dates are invalid or symbol is empty
        ImportError: If yfinance is not installed
    """
    if not symbol or not symbol.strip():
        raise ValueError("Symbol cannot be empty")
    
    symbol = symbol.strip().upper()
    
    # Validate dates
    try:
        start_dt = pd.to_datetime(start_date)
        if end_date:
            end_dt = pd.to_datetime(end_date)
            if end_dt < start_dt:
                raise ValueError(f"End date {end_date} is before start date {start_date}")
    except Exception as e:
        raise ValueError(f"Invalid date format: {e}") from e
    
    # Import yfinance
    try:
        import yfinance as yf
    except ImportError as e:
        raise ImportError(
            "yfinance is not installed. Install it with: pip install yfinance"
        ) from e
    
    # Download data from Yahoo Finance with retry logic for rate limits
    hist = None
    current_delay = retry_delay
    
    for attempt in range(1, max_retries + 1):
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=False,  # Keep original prices, we'll handle adj_close separately
            )
            
            if hist.empty:
                logger.warning(
                    f"No data returned for {symbol} from {start_date} to {end_date}. "
                    f"Symbol may be invalid or date range contains no trading days."
                )
                return pd.DataFrame()  # Return empty DataFrame instead of raising
            
            logger.info(f"Downloaded {len(hist)} rows for {symbol} (attempt {attempt}/{max_retries})")
            break  # Success, exit retry loop
            
        except Exception as e:
            error_str = str(e).lower()
            
            # Check for rate limit errors
            if 'rate limit' in error_str or 'too many requests' in error_str or '429' in error_str:
                if attempt < max_retries:
                    logger.warning(
                        f"Rate limited (attempt {attempt}/{max_retries}): {e}. "
                        f"Waiting {current_delay} seconds before retry..."
                    )
                    time.sleep(current_delay)
                    current_delay *= 2  # Exponential backoff
                    continue
                else:
                    raise RuntimeError(
                        f"Rate limited after {max_retries} attempts for {symbol}. "
                        f"Yahoo Finance has rate limits. Please wait a few minutes and try again, "
                        f"or use a smaller date range. Error: {e}"
                    ) from e
            
            # For other errors, wrap and raise
            raise RuntimeError(
                f"Failed to download data for {symbol}: {e}. "
                f"Check your internet connection and that the symbol is valid."
            ) from e
    
    if hist is None:
        raise RuntimeError(
            f"Failed to download data for {symbol} after {max_retries} attempts."
        )
    
    return hist


def normalize_dataframe(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Normalize yfinance DataFrame to backend format.
    
    Converts the raw yfinance DataFrame (with Date index and Open/High/Low/Close columns)
    to the backend format (timestamp, symbol, open, high, low, close, adj_close, volume).
    
    Args:
        df: Raw DataFrame from yfinance (with Date index)
        symbol: Stock ticker symbol (will be added as column)
    
    Returns:
        Normalized DataFrame with columns: timestamp, symbol, open, high, low, close, adj_close (if available), volume
        Sorted by timestamp, duplicates removed
    """
    if df.empty:
        return df
    
    # Reset index to convert Date index to column
    df = df.reset_index()
    
    # Rename Date column to timestamp
    if "Date" in df.columns:
        df = df.rename(columns={"Date": "timestamp"})
    elif "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": "timestamp"})
    elif df.index.name in ["Date", "Datetime"]:
        # If index was Date/Datetime, it's already in the index after reset_index
        # Check if timestamp column exists
        if "timestamp" not in df.columns:
            # Create timestamp from index if it's datetime
            if isinstance(df.index, pd.DatetimeIndex):
                df["timestamp"] = df.index
            else:
                raise RuntimeError("Could not determine timestamp column from yfinance data")
    
    # Ensure timestamp is datetime and convert to UTC
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    
    # Add symbol column
    df["symbol"] = symbol
    
    # Rename columns to match backend format (lowercase, underscore)
    column_mapping = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            df = df.rename(columns={old_col: new_col})
    
    # Select and order required columns
    required_cols = ["timestamp", "symbol"]
    optional_cols = ["open", "high", "low", "close", "adj_close", "volume"]
    
    # Check which optional columns are available
    available_cols = [col for col in optional_cols if col in df.columns]
    
    # Ensure at least 'close' is present (minimal requirement)
    if "close" not in df.columns:
        raise RuntimeError(
            f"Required column 'close' not found in downloaded data for {symbol}. "
            f"Available columns: {list(df.columns)}"
        )
    
    # Select final columns
    final_cols = required_cols + available_cols
    df = df[final_cols].copy()
    
    # Ensure correct data types
    df["symbol"] = df["symbol"].astype("string")
    for col in ["open", "high", "low", "close", "adj_close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    
    # Sort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Remove duplicates (if any)
    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    
    return df


def store_symbol_df(
    symbol: str,
    df: pd.DataFrame,
    interval: str,
    target_root: Path,
) -> Path:
    """Store normalized DataFrame as Parquet file.
    
    Saves the DataFrame to <target-root>/<interval>/<SYMBOL>.parquet
    in the format compatible with LocalParquetPriceDataSource.
    
    Args:
        symbol: Stock ticker symbol (e.g., "NVDA")
        df: Normalized DataFrame (from normalize_dataframe())
        interval: Data interval (e.g., "1d")
        target_root: Root directory where the Parquet file will be saved
    
    Returns:
        Path to the written Parquet file
    
    Raises:
        RuntimeError: If DataFrame is empty or write fails
    """
    if df.empty:
        raise RuntimeError(f"Cannot store empty DataFrame for {symbol}")
    
    # Determine output path: <target-root>/<interval>/<SYMBOL>.parquet
    output_dir = target_root / interval
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"{symbol.upper()}.parquet"
    
    # Write to Parquet
    try:
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved {len(df)} rows to {output_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to write Parquet file to {output_path}: {e}") from e
    
    return output_path


def download_single_symbol_snapshot(
    symbol: str,
    start_date: str,
    end_date: Optional[str],
    interval: str,
    target_root: Path,
    provider: str = "yahoo",
    finnhub_api_key: str | None = None,
    twelve_data_api_key: str | None = None,
) -> Path:
    """Download and save historical OHLCV data for a single symbol.
    
    Convenience function that combines download_symbol_history(), normalize_dataframe(),
    and store_symbol_df() for backward compatibility.
    
    Args:
        symbol: Stock ticker symbol (e.g., "NVDA")
        start_date: Start date in format "YYYY-MM-DD"
        end_date: End date in format "YYYY-MM-DD" or None for today
        interval: Data interval for yfinance (e.g., "1d")
        target_root: Root directory where the Parquet file will be saved
    
    Returns:
        Path to the written Parquet file
    
    Raises:
        RuntimeError: If download fails or returns empty data
        ValueError: If dates are invalid or symbol is empty
    """
    logger.info(f"Downloading {symbol} from {start_date} to {end_date or 'today'}")
    logger.info(f"Interval: {interval}")
    logger.info(f"Provider: {provider}")
    
    # Download raw data
    if provider == "finnhub":
        if not finnhub_api_key:
            raise ValueError("finnhub_api_key is required when using provider='finnhub'")
        hist = download_symbol_history_finnhub(
            symbol, 
            start_date, 
            end_date, 
            interval,
            finnhub_api_key
        )
    elif provider == "twelve_data":
        if not twelve_data_api_key:
            raise ValueError("twelve_data_api_key is required when using provider='twelve_data'")
        hist = download_symbol_history_twelve_data(
            symbol, 
            start_date, 
            end_date, 
            interval,
            twelve_data_api_key
        )
    else:  # yahoo (default)
        hist = download_symbol_history(symbol, start_date, end_date, interval)
    
    if hist.empty:
        raise RuntimeError(
            f"No data available for {symbol} from {start_date} to {end_date}. "
            f"Check if the symbol is valid and the date range contains trading days."
        )
    
    # Normalize to backend format
    df = normalize_dataframe(hist, symbol)
    
    logger.info(f"Processed {len(df)} rows with columns: {list(df.columns)}")
    
    # Store as Parquet
    output_path = store_symbol_df(symbol, df, interval, target_root)
    
    return output_path


def load_symbols_from_file(symbols_file: Path) -> list[str]:
    """Load symbols from a text file (one symbol per line).
    
    Args:
        symbols_file: Path to text file with one symbol per line
            Lines starting with # are treated as comments and ignored
            Empty lines are ignored
    
    Returns:
        List of symbol strings (uppercase, stripped)
    
    Raises:
        FileNotFoundError: If symbols_file does not exist
    """
    if not symbols_file.exists():
        raise FileNotFoundError(f"Symbols file not found: {symbols_file}")
    
    symbols = []
    with open(symbols_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                symbols.append(line.upper())
    
    if not symbols:
        raise ValueError(f"No symbols found in file: {symbols_file}")
    
    return symbols


def download_universe_snapshot(
    symbols: list[str],
    start_date: str,
    end_date: str | None,
    interval: str,
    target_root: Path,
    sleep_seconds: float = 2.0,
    max_retries_per_symbol: int = 5,
    retry_delay_base: float = 60.0,
    provider: str = "yahoo",
    finnhub_api_key: str | None = None,
    twelve_data_api_key: str | None = None,
) -> dict[str, str]:
    """Download historical data for multiple symbols.
    
    Downloads data for each symbol in the list, with rate-limit protection
    (sleep between downloads) and error handling (continues on failure).
    
    Args:
        symbols: List of stock ticker symbols (e.g., ["NVDA", "AAPL", "MSFT"])
        start_date: Start date in format "YYYY-MM-DD"
        end_date: End date in format "YYYY-MM-DD" or None for today
        interval: Data interval for yfinance (e.g., "1d")
        target_root: Root directory where Parquet files will be saved
        sleep_seconds: Seconds to sleep between downloads (rate-limit protection, default: 2.0)
    
    Returns:
        Dictionary mapping symbol -> status ("success", "empty", "error")
    """
    results = {}
    total = len(symbols)
    
    logger.info("=" * 60)
    logger.info(f"Downloading universe snapshot")
    logger.info(f"  Symbols: {total}")
    logger.info(f"  Date range: {start_date} to {end_date or 'today'}")
    logger.info(f"  Interval: {interval}")
    logger.info(f"  Target root: {target_root}")
    logger.info(f"  Provider: {provider}")
    logger.info(f"  Sleep between downloads: {sleep_seconds}s")
    logger.info("=" * 60)
    
    for idx, symbol in enumerate(symbols, 1):
        symbol = symbol.strip().upper()
        logger.info("")
        logger.info(f"[{idx}/{total}] Processing {symbol}...")
        
        # Check if file already exists
        existing_file = target_root / interval / f"{symbol}.parquet"
        if existing_file.exists():
            try:
                size = existing_file.stat().st_size
                if size > 1024:  # At least 1KB
                    logger.info(f"[{idx}/{total}] File already exists for {symbol}, skipping download.")
                    results[symbol] = "success"  # Treat as success
                    continue
            except Exception:
                pass  # If check fails, proceed with download
        
        # Retry loop for this symbol
        symbol_success = False
        for retry_attempt in range(1, max_retries_per_symbol + 1):
            try:
                # Download raw data with retry logic
                if provider == "finnhub":
                    if not finnhub_api_key:
                        raise ValueError("finnhub_api_key is required when using provider='finnhub'")
                    hist = download_symbol_history_finnhub(
                        symbol, 
                        start_date, 
                        end_date, 
                        interval,
                        finnhub_api_key,
                        max_retries=3,  # Internal retries
                        retry_delay=5
                    )
                elif provider == "twelve_data":
                    if not twelve_data_api_key:
                        raise ValueError("twelve_data_api_key is required when using provider='twelve_data'")
                    hist = download_symbol_history_twelve_data(
                        symbol, 
                        start_date, 
                        end_date, 
                        interval,
                        twelve_data_api_key,
                        max_retries=3,  # Internal retries
                        retry_delay=5
                    )
                else:  # yahoo (default)
                    hist = download_symbol_history(
                        symbol, 
                        start_date, 
                        end_date, 
                        interval,
                        max_retries=3,  # Internal retries
                        retry_delay=5
                    )
                
                if hist.empty:
                    logger.warning(f"[{idx}/{total}] No data for {symbol}, skipping.")
                    results[symbol] = "empty"
                    symbol_success = True  # Not an error, just no data
                    break
                
                # Normalize to backend format
                df = normalize_dataframe(hist, symbol)
                
                if df.empty:
                    logger.warning(f"[{idx}/{total}] Normalized DataFrame is empty for {symbol}, skipping.")
                    results[symbol] = "empty"
                    symbol_success = True
                    break
                
                # Store as Parquet
                output_path = store_symbol_df(symbol, df, interval, target_root)
                
                # Log summary
                min_date = df["timestamp"].min().date()
                max_date = df["timestamp"].max().date()
                logger.info(
                    f"[{idx}/{total}] ✓ {symbol}: {len(df)} rows, "
                    f"date range {min_date} to {max_date}, saved to {output_path.name}"
                )
                
                results[symbol] = "success"
                symbol_success = True
                break  # Success, exit retry loop
                
            except Exception as exc:
                error_str = str(exc).lower()
                is_rate_limit = any(
                    keyword in error_str
                    for keyword in ["rate limit", "too many requests", "429", "rate limited"]
                )
                
                if is_rate_limit:
                    if retry_attempt < max_retries_per_symbol:
                        # Exponential backoff for rate limits
                        delay = retry_delay_base * (2 ** (retry_attempt - 1))
                        logger.warning(
                            f"[{idx}/{total}] Rate limited for {symbol} (attempt {retry_attempt}/{max_retries_per_symbol}). "
                            f"Waiting {delay:.0f} seconds before retry..."
                        )
                        time.sleep(delay)
                        continue
                    else:
                        logger.error(
                            f"[{idx}/{total}] ✗ Rate limited for {symbol} after {max_retries_per_symbol} attempts. "
                            f"Will skip and continue with next symbol."
                        )
                        results[symbol] = "error"
                        break
                else:
                    # Other errors
                    if retry_attempt < max_retries_per_symbol:
                        delay = retry_delay_base * 0.5  # Shorter delay for non-rate-limit errors
                        logger.warning(
                            f"[{idx}/{total}] Error for {symbol} (attempt {retry_attempt}/{max_retries_per_symbol}): {exc}. "
                            f"Waiting {delay:.0f} seconds before retry..."
                        )
                        time.sleep(delay)
                        continue
                    else:
                        logger.error(f"[{idx}/{total}] ✗ Failed to download {symbol} after {max_retries_per_symbol} attempts: {exc}")
                        results[symbol] = "error"
                        break
        
        if not symbol_success and results.get(symbol) != "error":
            results[symbol] = "error"
        
        # Rate-limit protection: sleep between downloads (except for last symbol)
        # Use longer delay if we just hit a rate limit
        if idx < total:
            # Check if last error was rate limit
            last_error_was_rate_limit = False
            if results.get(symbol) == "error":
                # Try to determine if it was rate limit (we can't access exc here, so check recent logs)
                # For now, use longer delay if error occurred
                last_error_was_rate_limit = True  # Conservative: assume rate limit if error
            
            if last_error_was_rate_limit:
                # Extra long delay after rate limit
                extra_delay = sleep_seconds * 10
                logger.info(f"  Waiting {extra_delay:.0f} seconds before next symbol (rate limit protection)...")
                time.sleep(extra_delay)
            else:
                time.sleep(sleep_seconds)
    
    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Download Summary")
    logger.info("=" * 60)
    success_count = sum(1 for v in results.values() if v == "success")
    empty_count = sum(1 for v in results.values() if v == "empty")
    error_count = sum(1 for v in results.values() if v == "error")
    logger.info(f"  Success: {success_count}/{total}")
    logger.info(f"  Empty: {empty_count}/{total}")
    logger.info(f"  Errors: {error_count}/{total}")
    logger.info("=" * 60)
    
    return results


def validate_snapshot_directory(
    target_root: Path,
    interval: str,
) -> None:
    """Validate existing Parquet files in the snapshot directory.
    
    Reads all Parquet files in <target-root>/<interval>/ and reports
    basic statistics (rows, date range, columns) for each symbol.
    
    Args:
        target_root: Root directory containing the snapshot files
        interval: Data interval (e.g., "1d")
    
    Raises:
        FileNotFoundError: If target_root/interval directory does not exist
    """
    snapshot_dir = target_root / interval
    
    if not snapshot_dir.exists():
        raise FileNotFoundError(
            f"Snapshot directory not found: {snapshot_dir}. "
            f"Run download first or check the path."
        )
    
    parquet_files = list(snapshot_dir.glob("*.parquet"))
    
    if not parquet_files:
        logger.warning(f"No Parquet files found in {snapshot_dir}")
        return
    
    logger.info("=" * 60)
    logger.info(f"Validating snapshot directory: {snapshot_dir}")
    logger.info(f"  Found {len(parquet_files)} Parquet file(s)")
    logger.info("=" * 60)
    
    for parquet_file in sorted(parquet_files):
        symbol = parquet_file.stem.upper()
        
        try:
            df = pd.read_parquet(parquet_file)
            
            if df.empty:
                logger.warning(f"  {symbol}: EMPTY FILE")
                continue
            
            # Check required columns
            required_cols = ["timestamp", "symbol", "close"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                logger.error(f"  {symbol}: MISSING COLUMNS: {missing_cols}")
                continue
            
            # Get date range
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            min_date = df["timestamp"].min().date()
            max_date = df["timestamp"].max().date()
            
            # Report
            logger.info(
                f"  {symbol}: {len(df):,} rows, "
                f"date range {min_date} to {max_date}, "
                f"columns: {list(df.columns)}"
            )
            
        except Exception as e:
            logger.error(f"  {symbol}: ERROR reading file: {e}")
    
    logger.info("=" * 60)


def main() -> None:
    """Main entry point for historical snapshot download script."""
    parser = argparse.ArgumentParser(
        description="Download historical price snapshots for one or multiple symbols (Alt-Daten).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download single symbol (backward compatible)
    python scripts/download_historical_snapshot.py \\
      --symbol NVDA \\
      --target-root "F:/Python_Projekt/Aktiengerüst/datensammlungen/altdaten/stand 3-12-2025"
    
    # Download multiple symbols
    python scripts/download_historical_snapshot.py \\
      --symbols NVDA AAPL MSFT \\
      --start 2000-01-01 \\
      --end 2025-12-03 \\
      --target-root "F:/.../altdaten/stand 3-12-2025" \\
      --sleep-seconds 2.0
    
    # Download from file
    python scripts/download_historical_snapshot.py \\
      --symbols-file watchlist.txt \\
      --start 2000-01-01 \\
      --target-root "F:/.../altdaten/stand 3-12-2025"
    
    # Validate existing snapshot
    python scripts/download_historical_snapshot.py \\
      --validate-only \\
      --target-root "F:/.../altdaten/stand 3-12-2025" \\
      --interval 1d
        """,
    )
    
    # Symbol input (mutually exclusive)
    symbol_group = parser.add_mutually_exclusive_group()
    symbol_group.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Single ticker symbol to download (e.g., NVDA). Mutually exclusive with --symbols and --symbols-file.",
    )
    symbol_group.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        default=None,
        metavar="SYMBOL",
        help="Multiple ticker symbols to download (e.g., --symbols NVDA AAPL MSFT). Mutually exclusive with --symbol and --symbols-file.",
    )
    symbol_group.add_argument(
        "--symbols-file",
        type=Path,
        default=None,
        metavar="FILE",
        help="Path to text file with one symbol per line. Mutually exclusive with --symbol and --symbols.",
    )
    
    # Date range
    parser.add_argument(
        "--start",
        type=str,
        default="2000-01-01",
        help="Start date in format YYYY-MM-DD (default: 2000-01-01).",
    )
    
    parser.add_argument(
        "--end",
        type=str,
        default="2025-12-03",
        help="End date in format YYYY-MM-DD (default: 2025-12-03). Use 'today' for current date.",
    )
    
    parser.add_argument(
        "--interval",
        type=str,
        default="1d",
        help="Data interval: '1d' for daily, '5min' for 5-minute bars, etc. (default: 1d).",
    )
    
    # Data provider
    parser.add_argument(
        "--provider",
        type=str,
        choices=["yahoo", "finnhub", "twelve_data"],
        default="yahoo",
        help="Data provider to use: 'yahoo' (Yahoo Finance, default), 'finnhub' (Finnhub API), "
             "or 'twelve_data' (Twelve Data API, recommended). "
             "For Finnhub, set ASSEMBLED_FINNHUB_API_KEY. "
             "For Twelve Data, set ASSEMBLED_TWELVE_DATA_API_KEY.",
    )
    
    # Target directory
    parser.add_argument(
        "--target-root",
        type=str,
        required=True,
        help="Target root directory to store the Parquet files. "
             "Files will be saved as <target-root>/<interval>/<SYMBOL>.parquet",
    )
    
    # Rate-limit protection
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=2.0,
        help="Seconds to sleep between downloads when processing multiple symbols (rate-limit protection, default: 2.0).",
    )
    
    # Validation mode
    parser.add_argument(
        "--validate-only",
        action="store_true",
        default=False,
        help="Only validate existing Parquet files in target-root, do not download new data.",
    )
    
    args = parser.parse_args()
    
    # Convert target_root to Path
    target_root = Path(args.target_root)
    
    # Validation mode
    if args.validate_only:
        try:
            validate_snapshot_directory(target_root, args.interval)
            exit(0)
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            exit(1)
    
    # Determine symbols
    symbols = []
    if args.symbol:
        symbols = [args.symbol]
    elif args.symbols:
        symbols = args.symbols
    elif args.symbols_file:
        symbols = load_symbols_from_file(args.symbols_file)
    else:
        parser.error("Must specify one of: --symbol, --symbols, or --symbols-file")
    
    # Convert end_date: handle "today" special case
    end_date = args.end
    if end_date.lower() == "today":
        end_date = None
    
    # Load API keys if needed
    finnhub_api_key = None
    twelve_data_api_key = None
    
    if args.provider == "finnhub":
        import os
        finnhub_api_key = os.environ.get("ASSEMBLED_FINNHUB_API_KEY")
        if not finnhub_api_key:
            logger.error(
                "Finnhub API key not found. "
                "Set ASSEMBLED_FINNHUB_API_KEY environment variable."
            )
            exit(1)
        logger.info(f"Using Finnhub provider with API key: {finnhub_api_key[:8]}...")
    elif args.provider == "twelve_data":
        import os
        twelve_data_api_key = os.environ.get("ASSEMBLED_TWELVE_DATA_API_KEY")
        if not twelve_data_api_key:
            logger.error(
                "Twelve Data API key not found. "
                "Set ASSEMBLED_TWELVE_DATA_API_KEY environment variable."
            )
            exit(1)
        logger.info(f"Using Twelve Data provider with API key: {twelve_data_api_key[:8]}...")
    
    try:
        # Single symbol (backward compatible)
        if len(symbols) == 1:
            symbol = symbols[0].strip().upper()
            out_path = download_single_symbol_snapshot(
                symbol=symbol,
                start_date=args.start,
                end_date=end_date,
                interval=args.interval,
                target_root=target_root,
                provider=args.provider,
                finnhub_api_key=finnhub_api_key,
                twelve_data_api_key=twelve_data_api_key,
            )
            
            # Read back and show summary
            df = pd.read_parquet(out_path)
            min_date = df["timestamp"].min().date()
            max_date = df["timestamp"].max().date()
            
            print("")
            print("=" * 60)
            print(f"[DONE] Wrote snapshot for {symbol}")
            print(f"  Path: {out_path}")
            print(f"  Rows: {len(df):,}")
            print(f"  Date range: {min_date} to {max_date}")
            print(f"  Columns: {list(df.columns)}")
            print("=" * 60)
        
        # Multiple symbols
        else:
            results = download_universe_snapshot(
                symbols=symbols,
                start_date=args.start,
                end_date=end_date,
                interval=args.interval,
                target_root=target_root,
                sleep_seconds=args.sleep_seconds,
                max_retries_per_symbol=5,
                retry_delay_base=60.0,  # 60 seconds base delay for rate limits
                provider=args.provider,
                finnhub_api_key=finnhub_api_key,
                twelve_data_api_key=twelve_data_api_key,
            )
            
            # Exit with error code if all failed
            if all(v != "success" for v in results.values()):
                logger.error("All downloads failed!")
                exit(1)
            elif any(v == "error" for v in results.values()):
                logger.warning("Some downloads failed, but at least one succeeded.")
                exit(0)
            else:
                exit(0)
        
    except (ValueError, RuntimeError, FileNotFoundError) as e:
        logger.error(f"Error: {e}")
        exit(1)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        exit(1)


if __name__ == "__main__":
    main()
