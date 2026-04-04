"""Hidden Markov Model for Regime Forecasting.

Wraps hmmlearn.hmm.GaussianHMM to learn latent market regime states
(e.g., bull / bear / sideways) from return data. Output schema is
compatible with the existing risk/regime_models.py so it can substitute
or supplement rule-based regime detection.

Usage:
    from src.assembled_core.ml.regime_hmm import RegimeHMM

    model = RegimeHMM(n_regimes=3)
    model.fit(returns_series)
    regimes = model.predict_regime(returns_series)
    proba = model.predict_regime_proba(returns_series)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from hmmlearn.hmm import GaussianHMM  # type: ignore

    HMMLEARN_AVAILABLE = True
except ImportError:
    HMMLEARN_AVAILABLE = False
    GaussianHMM = None  # type: ignore

# Default regime labels sorted by mean return (set during fit)
_DEFAULT_LABEL_MAP = {0: "sideways", 1: "bull", 2: "bear"}


class RegimeHMM:
    """Gaussian Hidden Markov Model for market regime classification.

    Learns n_regimes latent states from log return sequences. States are
    automatically labelled bull/bear/sideways by their estimated mean return
    after fitting.

    Attributes:
        n_regimes: Number of latent states (default: 3)
        covariance_type: HMM covariance type (default: "full")
        n_iter: EM iterations (default: 100)
        random_state: Reproducibility seed (default: 42)
    """

    def __init__(
        self,
        n_regimes: int = 3,
        covariance_type: str = "full",
        n_iter: int = 100,
        random_state: int = 42,
    ) -> None:
        if not HMMLEARN_AVAILABLE:
            raise ImportError(
                "hmmlearn is required. Install with: pip install 'hmmlearn>=0.3.0'"
            )
        self.n_regimes = n_regimes
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state
        self._model: GaussianHMM | None = None
        self._label_map: dict[int, str] = {}
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, returns: pd.Series) -> "RegimeHMM":
        """Train the HMM on a return series.

        Args:
            returns: Log or simple return series with DatetimeIndex or RangeIndex.
                     Must not contain NaN.

        Returns:
            self (for chaining)
        """
        arr = self._prepare(returns)
        model = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        model.fit(arr)
        self._model = model
        self._label_map = self._build_label_map(model)
        self._is_fitted = True
        logger.info(
            "[RegimeHMM] Fitted %d-state HMM. Regime means: %s",
            self.n_regimes,
            {self._label_map[i]: float(model.means_[i, 0]) for i in range(self.n_regimes)},
        )
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_regime(self, returns: pd.Series) -> pd.Series:
        """Predict the most likely regime state at each time step (Viterbi).

        Args:
            returns: Return series (same format as fit input)

        Returns:
            Series of regime labels ("bull", "bear", "sideways") with same index
            as input.
        """
        self._check_fitted()
        arr = self._prepare(returns)
        raw_states = self._model.predict(arr)  # type: ignore[union-attr]
        labels = pd.Series(
            [self._label_map.get(s, f"state_{s}") for s in raw_states],
            index=returns.index,
            name="regime",
        )
        return labels

    def predict_regime_proba(self, returns: pd.Series) -> pd.DataFrame:
        """Compute forward-probability (posterior) for each regime at each step.

        This is the result of the forward algorithm — gives a probabilistic
        view of the current regime rather than a hard assignment.

        Args:
            returns: Return series (same format as fit input)

        Returns:
            DataFrame with columns = regime label strings, index = returns.index
        """
        self._check_fitted()
        arr = self._prepare(returns)
        # posteriors: shape (T, n_regimes)
        _, posteriors = self._model.score_samples(arr)  # type: ignore[union-attr]
        cols = [self._label_map.get(i, f"state_{i}") for i in range(self.n_regimes)]
        df = pd.DataFrame(posteriors, index=returns.index, columns=cols)
        return df

    def predict_next_regime_proba(self, returns: pd.Series) -> dict[str, float]:
        """Predict the regime probability distribution for the *next* period.

        Uses the transition matrix to propagate the last posterior state forward
        one step.

        Args:
            returns: Return series ending at time T (the current bar)

        Returns:
            Dict mapping regime label -> probability for T+1
        """
        self._check_fitted()
        last_proba = self.predict_regime_proba(returns).iloc[-1].values
        next_proba = last_proba @ self._model.transmat_  # type: ignore[union-attr]
        cols = [self._label_map.get(i, f"state_{i}") for i in range(self.n_regimes)]
        return dict(zip(cols, next_proba.tolist()))

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save model state to a numpy .npz file."""
        import joblib  # type: ignore

        joblib.dump(
            {
                "model": self._model,
                "label_map": self._label_map,
                "n_regimes": self.n_regimes,
            },
            path,
        )
        logger.info("[RegimeHMM] Saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "RegimeHMM":
        """Load model from a joblib file."""
        import joblib  # type: ignore

        if not HMMLEARN_AVAILABLE:
            raise ImportError("hmmlearn is required for loading RegimeHMM")
        data = joblib.load(path)
        obj = cls(n_regimes=data["n_regimes"])
        obj._model = data["model"]
        obj._label_map = data["label_map"]
        obj._is_fitted = True
        return obj

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare(returns: pd.Series) -> np.ndarray:
        """Convert return Series to 2D float64 array required by hmmlearn."""
        arr = returns.dropna().values.astype(np.float64).reshape(-1, 1)
        if len(arr) < 10:
            raise ValueError("Return series is too short (< 10 observations)")
        return arr

    def _check_fitted(self) -> None:
        if not self._is_fitted or self._model is None:
            raise RuntimeError("RegimeHMM must be fitted before calling predict*")

    @staticmethod
    def _build_label_map(model: Any) -> dict[int, str]:
        """Map HMM states to bull/bear/sideways by estimated mean return."""
        means = model.means_[:, 0]  # shape: (n_states,)
        n = len(means)
        ranked = np.argsort(means)  # indices sorted low → high return
        if n == 1:
            return {int(ranked[0]): "bull"}
        if n == 2:
            return {int(ranked[0]): "bear", int(ranked[1]): "bull"}
        # n >= 3: bottom = bear, top = bull, middle = sideways
        label_map: dict[int, str] = {}
        label_map[int(ranked[0])] = "bear"
        label_map[int(ranked[-1])] = "bull"
        for idx in ranked[1:-1]:
            label_map[int(idx)] = "sideways"
        return label_map
