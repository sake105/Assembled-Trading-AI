"""Wild-Card Event Detection (Plan 4.7).

Detects events that don't match known trigger types:
- GDELT volume anomalies (>3 sigma over 30d baseline)
- Unclassified clusters
- Cross-domain spikes (>3 domains spiking simultaneously)
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def detect_volume_anomaly(
    event_counts: pd.Series,
    window: int = 30,
    sigma_threshold: float = 3.0,
) -> dict:
    """Detect GDELT volume anomalies.

    Args:
        event_counts: Daily event counts.
        window: Baseline window.
        sigma_threshold: Standard deviations for anomaly.

    Returns:
        Dict with is_anomaly, zscore, baseline_mean, current_count.
    """
    if len(event_counts) < window:
        return {"is_anomaly": False, "zscore": 0.0}

    baseline = event_counts.iloc[-(window + 1):-1]
    current = float(event_counts.iloc[-1])
    mean = float(baseline.mean())
    std = float(baseline.std())

    if pd.isna(std) or std < 1e-6:
        return {"is_anomaly": False, "zscore": 0.0, "baseline_mean": mean, "current_count": current}

    zscore = (current - mean) / std
    is_anomaly = zscore > sigma_threshold

    if is_anomaly:
        logger.warning("[WildCard] Volume anomaly: z=%.1f (current=%d, baseline_mean=%.0f)", zscore, current, mean)

    return {
        "is_anomaly": is_anomaly,
        "zscore": round(zscore, 2),
        "baseline_mean": round(mean, 1),
        "current_count": current,
    }


def detect_cross_domain_spike(
    domain_counts: dict[str, int],
    domain_baselines: dict[str, float],
    spike_threshold: float = 2.0,
    min_domains: int = 3,
) -> dict:
    """Detect when multiple GDELT domains spike simultaneously.

    Args:
        domain_counts: Domain → current event count.
        domain_baselines: Domain → 30d average count.
        spike_threshold: Multiplier for spike detection.
        min_domains: Minimum spiking domains for alert.

    Returns:
        Dict with alert, spiking_domains, n_spiking.
    """
    spiking = []
    for domain, count in domain_counts.items():
        baseline = domain_baselines.get(domain, count)
        if baseline > 0 and count > baseline * spike_threshold:
            spiking.append(domain)

    alert = len(spiking) >= min_domains

    if alert:
        logger.warning("[WildCard] Cross-domain spike: %d domains spiking: %s", len(spiking), spiking)

    return {
        "alert": alert,
        "spiking_domains": spiking,
        "n_spiking": len(spiking),
    }


__all__ = ["detect_volume_anomaly", "detect_cross_domain_spike"]
