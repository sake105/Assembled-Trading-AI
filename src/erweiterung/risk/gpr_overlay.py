"""GPR-Overlay — eigenständiges Risk-Reduction-Modul basierend auf Caldara-Iacoviello GPR.

Komplementiert das Mainline-``src/assembled_core/risk/georisk_overlay.py``:
Mainline-Overlay arbeitet mit ``ctx.news_geo`` und ``intel_geo_score`` aus
der intel-Pipeline. Diese Erweiterung arbeitet direkt mit Caldara-Iacoviello-
GPR-Daten (echte Multi-Decade-Werte 1900+) und ist daher ohne intel-Pipeline-
Dependencies nutzbar.

API ist absichtlich kompatibel zur Mainline-Signatur:
``compute_exposure_multiplier(ctx, policy) -> float``

PR-Pfad: Dieses Modul könnte in
``src/assembled_core/risk/gpr_overlay.py`` portiert werden, als
ergänzendes Overlay neben dem existierenden GeoRisk-Overlay.

Datenfluss
----------
Caldara-Iacoviello GPR (monthly) → ffilled Daily → state_hint via percentile
+ zscore → exposure_multiplier laut Policy.

State Mapping
-------------
- PAUSE (multiplier 0.40-0.60): GPR-Spike (z > 2.0 oder level > 90 %ile)
- ACTIVE (0.70-0.85): elevated (z > 1.0 oder level > 75 %ile)
- WATCH (1.00): normal
- COOLDOWN (1.05-1.10): post-spike-relief (z < -1.0 oder level < 25 %ile)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from erweiterung.altdata.caldara_iacoviello_gpr import (
    compute_gpr_features,
    expand_to_daily,
    gpr_state_hint,
    load_gpr_cached,
)


# Default mapping consistent with Mainline ``risk/georisk_overlay.py``
DEFAULT_STATE_MULTIPLIERS: dict[str, float] = {
    "PAUSE": 0.50,
    "ACTIVE": 0.75,
    "WATCH": 1.00,
    "COOLDOWN": 1.05,
}


@dataclass
class GPROverlayPolicy:
    """Policy-Config für GPR-Overlay.

    Felder kompatibel zur Mainline-Policy:
    ``policy["georisk_overlay"]["enabled"]``, ``mapping[state_hint]``,
    ``max_geo_multiplier``, ``confidence_floor``.
    """

    enabled: bool = True
    state_multipliers: dict[str, float] = None
    max_geo_multiplier: float = 1.20
    min_geo_multiplier: float = 0.30
    confidence_floor: float = 0.50
    smoothing_days: int = 3

    def __post_init__(self):
        if self.state_multipliers is None:
            self.state_multipliers = DEFAULT_STATE_MULTIPLIERS.copy()


def build_daily_gpr_overlay_series(
    daily_index: pd.DatetimeIndex,
    policy: GPROverlayPolicy | None = None,
    cache_path: str | None = None,
) -> pd.DataFrame:
    """Berechne Daily-Exposure-Multiplier basierend auf Caldara-Iacoviello GPR.

    Args:
        daily_index: Ziel-Index (Daily, typisch trading days).
        policy: GPROverlayPolicy.
        cache_path: optionaler Override für GPR-Parquet-Cache.

    Returns:
        DataFrame mit Spalten:
        - gpr_level (0-100 percentile)
        - gpr_zscore
        - state_hint
        - exposure_multiplier (clamped to [min, max])
    """
    cfg = policy or GPROverlayPolicy()

    # Load + expand
    try:
        if cache_path:
            monthly = load_gpr_cached(cache_path)
        else:
            monthly = load_gpr_cached()
    except FileNotFoundError:
        # No cache → no overlay (return all-1.0)
        return pd.DataFrame(
            {
                "gpr_level": np.nan,
                "gpr_zscore": np.nan,
                "state_hint": "WATCH",
                "exposure_multiplier": 1.0,
            },
            index=daily_index,
        )

    if not cfg.enabled:
        return pd.DataFrame(
            {
                "exposure_multiplier": 1.0,
                "state_hint": "DISABLED",
            },
            index=daily_index,
        )

    daily_gpr = expand_to_daily(monthly, daily_index)
    features = compute_gpr_features(daily_gpr)

    # State-Hint per row
    state_hints = []
    multipliers = []
    for level, z in zip(features["gpr_level"], features["gpr_zscore"]):
        hint = gpr_state_hint(level if pd.notna(level) else float("nan"),
                              z if pd.notna(z) else 0.0)
        state_hints.append(hint)
        mult = cfg.state_multipliers.get(hint, 1.0)
        # Clamp
        mult = max(cfg.min_geo_multiplier, min(cfg.max_geo_multiplier, mult))
        multipliers.append(mult)

    out = pd.DataFrame(
        {
            "gpr_level": features["gpr_level"],
            "gpr_zscore": features["gpr_zscore"],
            "state_hint": state_hints,
            "exposure_multiplier": multipliers,
        },
        index=daily_index,
    )

    # Smoothing der Multiplier
    if cfg.smoothing_days > 1:
        out["exposure_multiplier"] = (
            out["exposure_multiplier"]
            .rolling(cfg.smoothing_days, min_periods=1)
            .mean()
        )

    return out


def apply_gpr_overlay(
    portfolio_returns: pd.Series,
    policy: GPROverlayPolicy | None = None,
    cache_path: str | None = None,
) -> pd.DataFrame:
    """Wende GPR-Overlay auf Portfolio-Returns an.

    Args:
        portfolio_returns: Daily returns.
        policy: GPROverlayPolicy.
        cache_path: optional GPR-Cache-Override.

    Returns:
        DataFrame [raw_return, exposure_multiplier, hedged_return, state_hint].
    """
    overlay = build_daily_gpr_overlay_series(
        portfolio_returns.index, policy=policy, cache_path=cache_path
    )
    # t-1 lag: heute's exposure_multiplier kommt von gestern's state
    mult_lag = overlay["exposure_multiplier"].shift(1).fillna(1.0)
    hedged = portfolio_returns * mult_lag
    return pd.DataFrame(
        {
            "raw_return": portfolio_returns,
            "exposure_multiplier": mult_lag,
            "hedged_return": hedged,
            "state_hint": overlay["state_hint"],
        }
    )


def compute_exposure_multiplier(
    ctx: Any, policy: dict[str, Any] | None = None
) -> float:
    """Mainline-kompatible API für aktuelle Exposure-Multiplier.

    Identische Signatur zu
    ``src.assembled_core.risk.georisk_overlay.compute_exposure_multiplier``.
    Erlaubt drop-in-replacement in ``trading_cycle_v2``.

    Args:
        ctx: TradingContext-artiges Objekt mit `.timestamp` oder
            kompatible Attribute. Wenn None, nutzt heutigen Datum.
        policy: dict mit `"gpr_overlay"` Subsection.

    Returns:
        Exposure-Multiplier (float in [min, max]).
    """
    pol_dict = (policy or {}).get("gpr_overlay") or {}
    if not pol_dict.get("enabled", True):
        return 1.0

    overlay_cfg = GPROverlayPolicy(
        enabled=pol_dict.get("enabled", True),
        max_geo_multiplier=float(pol_dict.get("max_geo_multiplier", 1.20)),
        min_geo_multiplier=float(pol_dict.get("min_geo_multiplier", 0.30)),
    )

    # Determine "today"
    ts = getattr(ctx, "timestamp", None) if ctx else None
    if ts is None:
        ts = pd.Timestamp(datetime.utcnow(), tz="UTC")
    elif not isinstance(ts, pd.Timestamp):
        ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")

    # Build 1-day overlay
    idx = pd.DatetimeIndex([ts])
    df = build_daily_gpr_overlay_series(idx, policy=overlay_cfg)
    return float(df["exposure_multiplier"].iloc[0])


__all__ = [
    "GPROverlayPolicy",
    "DEFAULT_STATE_MULTIPLIERS",
    "build_daily_gpr_overlay_series",
    "apply_gpr_overlay",
    "compute_exposure_multiplier",
]
