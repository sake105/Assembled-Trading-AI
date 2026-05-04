"""Shared EOD signal generation logic.

This module provides the canonical ``compute_signals_by_mode()`` function used
by both pipeline paths:

- ``orchestrator.py`` → called from ``scripts/run_eod_pipeline.py`` / ``assembled-run-daily``
- ``trading_cycle_v2.py`` → called from ``scripts/run_daily.py`` / paper runner

**Two-pipeline architecture (B5)**

The two pipelines share signal *generation* logic (this module) but differ in
downstream processing:

| Dimension | orchestrator | trading_cycle_v2 |
|-----------|-------------|-----------------|
| Entry | ``run_eod_pipeline`` | ``run_trading_cycle`` |
| Context | stateless (prices only) | ``TradingContext`` |
| Order gen | ``signals_to_orders`` | ``size_positions`` + ``route_orders`` |
| Risk | none | full risk overlay |
| Paper/Live | no | yes |

This divergence is intentional for now. Full convergence (Option A from the B5
audit) requires migrating ``assembled-run-daily`` to use ``run_trading_cycle``
and is deferred (estimated 12-20h).  Track in: ``autonome_weiterarbeit/AUDIT_2026-04-26_FINDINGS_AND_REMEDIATION_v2.md`` B5.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    pass

logger = logging.getLogger("assembled_core.pipeline")


def compute_signals_by_mode(
    prices: pd.DataFrame,
    policy: dict,
    freq: str = "1d",
) -> pd.DataFrame:
    """Compute signals from prices using the mode configured in policy.

    Dispatches to the right signal function based on
    ``policy["signal_generation"]["mode"]``.  Supported modes:
    ``multifactor``, ``ml_enhanced``, ``ema`` (default/legacy).

    Each mode falls back to EMA on error so that the pipeline can always
    continue; errors are logged at WARNING or ERROR level (not silenced).

    Args:
        prices: Price DataFrame (columns: timestamp, symbol, close, …).
        policy: Parsed policy.yaml dict (may be empty).
        freq: Trading frequency for EMA config lookup (default: "1d").

    Returns:
        Signals DataFrame.
    """
    from src.assembled_core.ema_config import get_default_ema_config
    from src.assembled_core.pipeline.signals import compute_ema_signals

    signal_mode = (policy.get("signal_generation") or {}).get("mode", "ema")

    if signal_mode == "multifactor":
        try:
            from src.assembled_core.strategies.multifactor_v2 import (
                compute_signals as mf_compute_signals,
            )

            signals = mf_compute_signals(prices)
            logger.info("[EOD] Signal mode: multifactor_v2 (%d signals)", len(signals))
        except Exception as exc:
            logger.error(
                "[EOD] multifactor signal generation failed, falling back to EMA: %s",
                exc,
                exc_info=True,
            )
            ema_config = get_default_ema_config(freq)
            signals = compute_ema_signals(prices, ema_config.fast, ema_config.slow)
    elif signal_mode == "ml_enhanced":
        logger.info(
            "[EOD] Signal mode: ml_enhanced (not yet trained, using multifactor)"
        )
        try:
            from src.assembled_core.strategies.multifactor_v2 import (
                compute_signals as mf_compute_signals,
            )

            signals = mf_compute_signals(prices)
        except Exception as exc:
            logger.error(
                "[EOD] ml_enhanced → multifactor fallback failed: %s",
                exc,
                exc_info=True,
            )
            ema_config = get_default_ema_config(freq)
            signals = compute_ema_signals(prices, ema_config.fast, ema_config.slow)
    else:
        ema_config = get_default_ema_config(freq)
        signals = compute_ema_signals(prices, ema_config.fast, ema_config.slow)
        logger.info(
            "[EOD] Signal mode: ema (fast=%d, slow=%d)",
            ema_config.fast,
            ema_config.slow,
        )

    return signals
