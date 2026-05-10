"""Macro-Nowcasting + Recession-Probability auf Basis FRED-MD.

Theorie
-------
Recession Probabilities (Estrella/Mishkin 1998, Wright 2006) basieren typischerweise
auf:
- Yield Curve Slope (10Y − 3M)
- Credit Spreads (BAA − AAA, oder HY-OAS)
- Unemployment Trend (Sahm Rule)
- Consumer Confidence
- ISM PMI

Wir kombinieren diese in einen Composite-Score.

Sahm-Rule
---------
Recession startet wenn 3M-MA der UR um >0.5pp gegen 12M-Min steigt
(Sahm 2019, NBER).

Anwendung
---------
- ``recession_prob`` als macro-overlay-Faktor.
- Hohe Recession-Prob -> Reduce Equity-Beta-Exposure / Switch zu Defensives.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def yield_curve_slope(
    fred_md_data: pd.DataFrame,
    long_col: str = "GS10",
    short_col: str = "TB3MS",
) -> pd.Series:
    """10Y − 3M Yield Curve Slope. Werte < 0 = inverted yield curve."""
    if long_col not in fred_md_data.columns or short_col not in fred_md_data.columns:
        return pd.Series(dtype=float)
    return fred_md_data[long_col] - fred_md_data[short_col]


def sahm_rule(unemployment_rate: pd.Series, threshold: float = 0.5) -> pd.Series:
    """Sahm-Rule: 1 wenn 3M-MA der UR > 12M-Min + threshold (pp).

    Args:
        unemployment_rate: monatliche UR-Series.
        threshold: 0.5 (=Standard-Sahm).

    Returns:
        Series mit ``1.0`` für Recession-Phase, ``0.0`` sonst.
    """
    if unemployment_rate.empty:
        return unemployment_rate
    ma3 = unemployment_rate.rolling(3, min_periods=2).mean()
    min12 = unemployment_rate.rolling(12, min_periods=6).min()
    diff = ma3 - min12
    sig = (diff >= threshold).astype(float)
    return sig


def credit_spread_signal(
    fred_md_data: pd.DataFrame,
    baa_col: str = "BAAFFM",
    aaa_col: str = "AAAFFM",
) -> pd.Series:
    """Credit-Spread BAA-AAA in pp. Hohe Werte = Stress."""
    if baa_col not in fred_md_data.columns or aaa_col not in fred_md_data.columns:
        return pd.Series(dtype=float)
    return fred_md_data[baa_col] - fred_md_data[aaa_col]


def composite_recession_score(
    fred_md_data: pd.DataFrame,
    weights: Optional[dict[str, float]] = None,
) -> pd.DataFrame:
    """Composite-Recession-Wahrscheinlichkeit aus mehreren Signalen.

    Args:
        fred_md_data: Output von ``apply_mccracken_transforms`` (oder raw).
        weights: dict mit Komponenten-Gewichten.

    Returns:
        DataFrame [date, yield_slope, credit_spread, sahm, recession_score].

    Score-Skala
    -----------
    [0, 1] interpretierbar als Wahrscheinlichkeit. Aggregation via logit-mix.
    """
    weights = weights or {"yield_slope": 0.4, "credit_spread": 0.3, "sahm": 0.3}

    out = pd.DataFrame(index=fred_md_data.index)
    out["yield_slope"] = yield_curve_slope(fred_md_data)
    out["credit_spread"] = credit_spread_signal(fred_md_data)
    if "UNRATE" in fred_md_data.columns:
        out["sahm"] = sahm_rule(fred_md_data["UNRATE"])
    else:
        out["sahm"] = np.nan

    # Z-Skalieren mit historischer Verteilung (full-sample für Score-Calibration)
    def _norm(s: pd.Series, invert: bool = False) -> pd.Series:
        if s.empty or s.notna().sum() < 24:
            return pd.Series(0.0, index=s.index)
        z = (s - s.mean()) / s.std()
        if invert:
            z = -z
        # Convert to [0,1] via sigmoid
        return 1 / (1 + np.exp(-z))

    yc_score = _norm(out["yield_slope"], invert=True)  # negativer slope = höhere prob
    cs_score = _norm(out["credit_spread"], invert=False)
    sahm_score = out["sahm"].fillna(0)

    out["recession_score"] = (
        weights["yield_slope"] * yc_score.fillna(0)
        + weights["credit_spread"] * cs_score.fillna(0)
        + weights["sahm"] * sahm_score
    )
    return out.reset_index()


__all__ = [
    "yield_curve_slope",
    "sahm_rule",
    "credit_spread_signal",
    "composite_recession_score",
]
