"""Online HMM Regime Detection auf Returns/Vol.

Ergänzt `NewsRegimeClassifier` (Round 4 — News-Sentiment-basiert) um eine
returns-basierte Sicht. Oft stimmen beide überein (z.B. CRISIS in beiden),
aber manchmal divergieren sie — dann gibt es echten Informationsgewinn.

Implementation:
- Primary: `hmmlearn.GaussianHMM` (offline/batch trained)
- Online-Update via expanding window re-fit pro Periode (nicht wirklich online,
  aber praktikabel ohne river-HMM)
- Fallback: einfache Vol-Regime-Schwellen wenn hmmlearn fehlt

3 States:
- Low-Vol
- Normal
- High-Vol / Crisis

PIT-Invariante: HMM nur auf historischen Returns fitten.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_REGIME_LABELS = ["LOW_VOL", "NORMAL", "HIGH_VOL"]


@dataclass
class RegimeState:
    regime_id: int
    regime_label: str
    probability: float = 1.0
    mean_return: float = 0.0
    volatility: float = 0.0


class OnlineHMMRegimeDetector:
    """3-State HMM auf Rolling-Return/Vol-Features.

    Graceful degradation: Wenn hmmlearn fehlt, fallback auf Vol-Quantil-Rules.
    """

    def __init__(
        self,
        n_states: int = 3,
        lookback: int = 252,
        retrain_freq: int = 60,
        random_state: int = 42,
    ) -> None:
        self.n_states = n_states
        self.lookback = lookback
        self.retrain_freq = retrain_freq
        self.random_state = random_state
        self._model: object | None = None
        self._state_to_label: dict[int, str] = {}
        self._available = False
        self._last_fit_size: int = 0
        self._try_init()

    def _try_init(self) -> None:
        try:
            from hmmlearn.hmm import GaussianHMM  # type: ignore
            self._GaussianHMM = GaussianHMM
            self._available = True
            logger.debug("[OnlineHMM] hmmlearn verfügbar")
        except ImportError:
            logger.info("[OnlineHMM] hmmlearn fehlt — Fallback auf Vol-Quantile")
            self._available = False

    def _make_features(self, returns: pd.Series) -> np.ndarray:
        """Rolling-Mean + Rolling-Vol als 2-dim Features."""
        rolling_mean = returns.rolling(20, min_periods=5).mean().bfill().fillna(0.0).values
        rolling_vol = returns.rolling(20, min_periods=5).std().bfill().fillna(returns.std() or 0.01).values
        return np.column_stack([rolling_mean, rolling_vol])

    def fit(self, returns: pd.Series) -> "OnlineHMMRegimeDetector":
        """Fit HMM auf historical returns."""
        clean = returns.dropna()
        if len(clean) < self.lookback // 2:
            logger.warning("[OnlineHMM] Nur %d Returns — zu wenig für Fit", len(clean))
            return self

        if not self._available:
            self._last_fit_size = len(clean)
            return self

        X = self._make_features(clean)
        try:
            self._model = self._GaussianHMM(
                n_components=self.n_states,
                covariance_type="full",
                random_state=self.random_state,
                n_iter=100,
            )
            self._model.fit(X)

            # Label mapping: sort states by volatility component
            means = self._model.means_  # type: ignore[attr-defined]
            vol_order = np.argsort(means[:, 1])
            self._state_to_label = {}
            for rank, state_id in enumerate(vol_order):
                self._state_to_label[int(state_id)] = _REGIME_LABELS[min(rank, len(_REGIME_LABELS) - 1)]

            self._last_fit_size = len(clean)
            logger.info(
                "[OnlineHMM] Fit auf %d Returns. State→Label: %s",
                len(clean), self._state_to_label,
            )
        except Exception as exc:
            logger.warning("[OnlineHMM] Fit failed: %s — fallback", exc)
            self._available = False

        return self

    def predict_current_regime(self, returns: pd.Series) -> RegimeState:
        """Aktuelles Regime aus letzten Returns."""
        clean = returns.dropna()
        if len(clean) < 20:
            return RegimeState(regime_id=1, regime_label="NORMAL")

        if not self._available or self._model is None:
            return self._fallback_predict(clean)

        # Re-fit if we've moved past retrain_freq
        if len(clean) - self._last_fit_size >= self.retrain_freq:
            self.fit(clean)

        X = self._make_features(clean)
        try:
            states = self._model.predict(X)  # type: ignore[attr-defined]
            current_state = int(states[-1])
            label = self._state_to_label.get(current_state, "NORMAL")

            # Posterior probabilities
            try:
                post = self._model.predict_proba(X[-1:])  # type: ignore[attr-defined]
                prob = float(post[0, current_state])
            except Exception:
                prob = 1.0

            return RegimeState(
                regime_id=current_state,
                regime_label=label,
                probability=prob,
                mean_return=float(clean.iloc[-20:].mean()),
                volatility=float(clean.iloc[-20:].std()),
            )
        except Exception as exc:
            logger.debug("[OnlineHMM] predict failed: %s", exc)
            return self._fallback_predict(clean)

    def _fallback_predict(self, returns: pd.Series) -> RegimeState:
        """Vol-Quantil-basiert."""
        recent_vol = returns.iloc[-20:].std() if len(returns) >= 20 else returns.std()
        long_vol = returns.std()
        if long_vol <= 1e-9:
            return RegimeState(0, "NORMAL")

        ratio = recent_vol / long_vol
        if ratio < 0.7:
            label = "LOW_VOL"
            rid = 0
        elif ratio > 1.5:
            label = "HIGH_VOL"
            rid = 2
        else:
            label = "NORMAL"
            rid = 1
        return RegimeState(
            regime_id=rid, regime_label=label, probability=1.0,
            mean_return=float(returns.iloc[-20:].mean()),
            volatility=float(recent_vol),
        )


__all__ = [
    "RegimeState",
    "OnlineHMMRegimeDetector",
]
