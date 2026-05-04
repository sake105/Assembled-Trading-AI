"""VVIX/SKEW tail-risk signal — calm/elevated/high/extreme regime detection."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

logger = logging.getLogger(__name__)

Regime = Literal["calm", "elevated", "high", "extreme"]


@dataclass
class TailRiskState:
    regime: Regime
    vvix: float | None
    skew: float | None
    backwardation: bool
    score: int  # 0-3 (or 4 with backwardation modifier)
    raw: dict = field(default_factory=dict)


class VVIXTailRiskSignal:
    """Classify market tail risk using VVIX, CBOE SKEW, and VIX term structure.

    Sources: CBOE (^VVIX, ^SKEW, ^VIX, ^VIX9D, ^VIX3M) via yfinance / provider adapter.
    """

    THRESHOLDS: dict[str, dict[str, float]] = {
        "vvix": {"calm": 90.0, "elevated": 100.0, "high": 110.0, "extreme": 130.0},
        "skew": {"calm": 130.0, "elevated": 135.0, "high": 140.0, "extreme": 150.0},
    }

    def regime(self, latest: pd.Series | dict) -> TailRiskState:
        """Classify current tail risk from latest VVIX/SKEW/VIX data.

        Parameters
        ----------
        latest:
            Must contain at least ``vvix``. Optionally ``skew``, ``vix``, ``vix9d``, ``vix3m``.

        Returns
        -------
        TailRiskState with regime, score (0-3), and backwardation flag.
        """
        if isinstance(latest, pd.Series):
            latest = latest.to_dict()

        vvix = latest.get("vvix")
        skew = latest.get("skew")
        vix = latest.get("vix")
        vix3m = latest.get("vix3m")
        vix9d = latest.get("vix9d")

        vvix_score = self._score("vvix", vvix) if vvix is not None else 0
        skew_score = self._score("skew", skew) if skew is not None else 0
        score = max(vvix_score, skew_score)

        backwardation = False
        if vix is not None and vix3m is not None:
            backwardation = float(vix) > float(vix3m)
        if backwardation:
            score = min(score + 1, 3)

        regime: Regime = self._score_to_regime(score)

        return TailRiskState(
            regime=regime,
            vvix=float(vvix) if vvix is not None else None,
            skew=float(skew) if skew is not None else None,
            backwardation=backwardation,
            score=score,
            raw={
                "vix": vix,
                "vix3m": vix3m,
                "vix9d": vix9d,
                "term_structure": (
                    (float(vix) - float(vix3m))
                    if (vix is not None and vix3m is not None)
                    else None
                ),
                "short_inversion": (
                    (float(vix9d) - float(vix))
                    if (vix9d is not None and vix is not None)
                    else None
                ),
            },
        )

    def fetch_data(self) -> pd.DataFrame:
        """Fetch VVIX, SKEW, VIX, VIX9D, VIX3M from Yahoo Finance.

        Returns empty DataFrame if yfinance is unavailable.
        """
        try:
            import yfinance as yf  # noqa: PLC0415
        except ImportError:
            return pd.DataFrame()

        tickers = {
            "^VVIX": "vvix",
            "^SKEW": "skew",
            "^VIX": "vix",
            "^VIX9D": "vix9d",
            "^VIX3M": "vix3m",
        }
        frames: dict[str, pd.Series] = {}
        for ticker, col in tickers.items():
            try:
                hist = yf.Ticker(ticker).history(period="3mo")
                if not hist.empty:
                    frames[col] = hist["Close"].rename(col)
            except Exception as _exc:
                logger.debug("[tail_risk_vvix] failed to fetch %s: %s", ticker, _exc)

        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames.values(), axis=1)
        if "vix" in df.columns and "vix3m" in df.columns:
            df["term_structure"] = df["vix"] - df["vix3m"]
        if "vix9d" in df.columns and "vix" in df.columns:
            df["short_inversion"] = df["vix9d"] - df["vix"]
        return df

    def _score(self, metric: str, value: float) -> int:
        t = self.THRESHOLDS[metric]
        v = float(value)
        if v >= t["extreme"]:
            return 3
        if v >= t["high"]:
            return 2
        if v >= t["calm"]:  # calm threshold is the lower bound of elevated regime
            return 1
        return 0

    @staticmethod
    def _score_to_regime(score: int) -> Regime:
        mapping: dict[int, Regime] = {0: "calm", 1: "elevated", 2: "high", 3: "extreme"}
        return mapping.get(min(score, 3), "extreme")
