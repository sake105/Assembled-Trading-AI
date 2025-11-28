# src/assembled_core/api/routers/signals.py
"""Signals endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.assembled_core.api.models import Frequency, Signal, SignalType, SignalsResponse
from src.assembled_core.config import OUTPUT_DIR, SUPPORTED_FREQS
from src.assembled_core.ema_config import get_default_ema_config
from src.assembled_core.pipeline.io import load_prices
from src.assembled_core.pipeline.signals import compute_ema_signals

router = APIRouter()


def _map_sig_to_signal_type(sig: int) -> SignalType:
    """Map signal integer to SignalType enum.
    
    Args:
        sig: Signal value (-1, 0, or +1)
    
    Returns:
        SignalType enum value
    """
    if sig > 0:
        return SignalType.BUY
    elif sig < 0:
        return SignalType.SELL
    else:
        return SignalType.NEUTRAL


@router.get("/signals/{freq}", response_model=SignalsResponse)
def get_signals(freq: Frequency) -> SignalsResponse:
    """Get all EMA crossover signals for a given frequency.
    
    Args:
        freq: Trading frequency ("1d" or "5min")
    
    Returns:
        SignalsResponse with list of all signals
    
    Raises:
        HTTPException: 404 if price data not found, 500 if data is malformed
    """
    # Validate frequency
    if freq.value not in SUPPORTED_FREQS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported frequency: {freq.value}. Supported: {SUPPORTED_FREQS}"
        )
    
    try:
        # Load prices
        prices = load_prices(freq.value, output_dir=OUTPUT_DIR)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=f"Price data for freq {freq.value} not found. {str(e)}"
        )
    except ValueError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Malformed price data: {e}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error loading prices: {e}"
        )
    
    if prices.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Price data for freq {freq.value} is empty."
        )
    
    try:
        # Get EMA defaults for this frequency
        ema_config = get_default_ema_config(freq.value)
        
        # Compute signals
        signals_df = compute_ema_signals(prices, ema_config.fast, ema_config.slow)
        
        if signals_df.empty:
            # Return empty response
            return SignalsResponse(
                frequency=freq,
                signals=[],
                count=0,
                first_timestamp=None,
                last_timestamp=None
            )
        
        # Map DataFrame rows to Signal models
        signals_list = []
        for _, row in signals_df.iterrows():
            signals_list.append(
                Signal(
                    timestamp=row["timestamp"],
                    symbol=str(row["symbol"]),
                    signal_type=_map_sig_to_signal_type(int(row["sig"])),
                    price=float(row["price"]),
                    ema_fast=None,  # Not computed in current implementation
                    ema_slow=None  # Not computed in current implementation
                )
            )
        
        # Sort by timestamp descending (most recent first)
        signals_list.sort(key=lambda s: s.timestamp, reverse=True)
        
        # Get first and last timestamps
        first_ts = signals_df["timestamp"].min()
        last_ts = signals_df["timestamp"].max()
        
        return SignalsResponse(
            frequency=freq,
            signals=signals_list,
            count=len(signals_list),
            first_timestamp=first_ts,
            last_timestamp=last_ts
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error computing signals: {e}"
        )


@router.get("/signals/{freq}/latest", response_model=list[Signal])
def get_latest_signals(freq: Frequency) -> list[Signal]:
    """Get latest signal per symbol for a given frequency.
    
    Args:
        freq: Trading frequency ("1d" or "5min")
    
    Returns:
        List of Signal objects (one per symbol, latest timestamp)
    
    Raises:
        HTTPException: 404 if price data not found, 500 if data is malformed
    """
    # Validate frequency
    if freq.value not in SUPPORTED_FREQS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported frequency: {freq.value}. Supported: {SUPPORTED_FREQS}"
        )
    
    try:
        # Load prices
        prices = load_prices(freq.value, output_dir=OUTPUT_DIR)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=f"Price data for freq {freq.value} not found. {str(e)}"
        )
    except ValueError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Malformed price data: {e}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error loading prices: {e}"
        )
    
    if prices.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Price data for freq {freq.value} is empty."
        )
    
    try:
        # Get EMA defaults for this frequency
        ema_config = get_default_ema_config(freq.value)
        
        # Compute signals
        signals_df = compute_ema_signals(prices, ema_config.fast, ema_config.slow)
        
        if signals_df.empty:
            return []
        
        # Group by symbol and get latest timestamp per symbol
        latest_signals = []
        for symbol, group in signals_df.groupby("symbol"):
            # Get row with latest timestamp
            latest_row = group.loc[group["timestamp"].idxmax()]
            
            latest_signals.append(
                Signal(
                    timestamp=latest_row["timestamp"],
                    symbol=str(symbol),
                    signal_type=_map_sig_to_signal_type(int(latest_row["sig"])),
                    price=float(latest_row["price"]),
                    ema_fast=None,  # Not computed in current implementation
                    ema_slow=None  # Not computed in current implementation
                )
            )
        
        # Sort by symbol for consistent ordering
        latest_signals.sort(key=lambda s: s.symbol)
        
        return latest_signals
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error computing latest signals: {e}"
        )
