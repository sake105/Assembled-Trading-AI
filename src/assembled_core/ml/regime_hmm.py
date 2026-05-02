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
        if not self._is_fitted or self._model is None:
            if not HMMLEARN_AVAILABLE:
                # Graceful fallback: vol-threshold labels when hmmlearn is not installed
                vol = returns.rolling(20, min_periods=5).std()
                vol_pct = vol.rank(pct=True).fillna(0.5)
                labels = vol_pct.map(lambda v: "bear" if v > 0.7 else ("bull" if v < 0.3 else "sideways"))
                return labels.rename("regime")
            raise RuntimeError("RegimeHMM must be fitted before calling predict_regime")
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
        # _prepare drops NaN — track the cleaned index to avoid shape mismatch
        clean_idx = returns.dropna().index
        arr = self._prepare(returns)
        # posteriors: shape (T_clean, n_regimes)
        _, posteriors = self._model.score_samples(arr)  # type: ignore[union-attr]
        cols = [self._label_map.get(i, f"state_{i}") for i in range(self.n_regimes)]
        df = pd.DataFrame(posteriors, index=clean_idx, columns=cols)
        return df.reindex(returns.index)  # restore NaN rows for original index

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
    # Online / incremental update
    # ------------------------------------------------------------------

    def partial_update(
        self,
        new_returns: pd.Series,
        n_iter: int = 5,
        min_samples: int = 20,
    ) -> "RegimeHMM":
        """Warm-start re-estimation with new observations.

        Keeps the current model parameters as starting point and runs a few
        EM iterations on the new data only — faster than a full refit and
        prevents catastrophic forgetting when the window is small.

        Requires ``hmmlearn >= 0.3.0`` (``init_params=""`` support).

        Parameters
        ----------
        new_returns:
            New observations to incorporate (at least ``min_samples`` long).
        n_iter:
            Number of EM iterations (default 5 — enough for minor updates).
        min_samples:
            Minimum new observations required; skips update if fewer.

        Returns
        -------
        self
        """
        self._check_fitted()
        if len(new_returns.dropna()) < min_samples:
            logger.debug(
                "[RegimeHMM] partial_update skipped — only %d new samples (min %d)",
                len(new_returns.dropna()), min_samples,
            )
            return self
        arr = self._prepare(new_returns)
        if len(arr) < min_samples:
            logger.debug(
                "[RegimeHMM] partial_update skipped — only %d new samples (min %d)",
                len(arr), min_samples,
            )
            return self

        # Warm-start: reuse existing model parameters, run a few EM steps
        warm_model = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type=self.covariance_type,
            n_iter=n_iter,
            random_state=self.random_state,
            init_params="",  # do not reinitialise — warm start
        )
        warm_model.startprob_ = self._model.startprob_.copy()  # type: ignore[union-attr]
        warm_model.transmat_ = self._model.transmat_.copy()  # type: ignore[union-attr]
        warm_model.means_ = self._model.means_.copy()  # type: ignore[union-attr]
        warm_model.covars_ = self._model.covars_.copy()  # type: ignore[union-attr]

        try:
            warm_model.fit(arr)
            self._model = warm_model
            self._label_map = self._build_label_map(warm_model)
            logger.info(
                "[RegimeHMM] partial_update applied (%d new obs, %d EM iters)", len(arr), n_iter
            )
        except Exception as exc:
            logger.warning("[RegimeHMM] partial_update failed, keeping old model: %s", exc)

        return self

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
        try:
            data = joblib.load(path)
        except (FileNotFoundError, EOFError, Exception) as exc:
            raise RuntimeError(f"[RegimeHMM] Failed to load model from {path}: {exc}") from exc
        try:
            obj = cls(n_regimes=data["n_regimes"])
            obj._model = data["model"]
            obj._label_map = data["label_map"]
        except KeyError as exc:
            raise RuntimeError(f"[RegimeHMM] Model file {path} is missing required key: {exc}") from exc
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


# ---------------------------------------------------------------------------
# Multi-Feature HMM (Plan 2.3)
# ---------------------------------------------------------------------------


class MultiFeatureRegimeHMM:
    """Multi-observable HMM for earlier regime detection.

    Uses 2+ observables (daily_return, realized_vol_20d, and optionally
    vix_change, hy_spread_change). Features are standardised internally
    via a fitted StandardScaler so callers always pass raw feature values.

    When ``hmmlearn`` is not installed, falls back to a simple
    volatility-threshold classifier.
    """

    def __init__(
        self,
        n_regimes: int = 3,
        n_iter: int = 200,
        random_state: int = 42,
        covariance_type: str = "diag",
        n_seeds: int = 5,
    ):
        self.n_regimes = n_regimes
        self.n_iter = n_iter
        self.random_state = random_state
        self.covariance_type = covariance_type
        self.n_seeds = n_seeds
        self._model: Any = None
        self._scaler: Any = None
        self._label_map: dict[int, str] = {}
        self._fitted = False

    def _scale(self, X: np.ndarray) -> np.ndarray:
        """Apply fitted scaler; fit+transform if scaler not yet fit."""
        if self._scaler is None:
            return X
        return self._scaler.transform(X)

    def fit(self, features_df: pd.DataFrame) -> bool:
        """Fit multi-feature HMM with StandardScaler + multi-seed search.

        Args:
            features_df: DataFrame with numeric feature columns.
                Each row is one observation (time step). NaN rows are dropped.

        Returns:
            True if fitting succeeded.
        """
        if not HMMLEARN_AVAILABLE:
            logger.debug("[MultiHMM] hmmlearn not installed — using fallback")
            self._fitted = False
            return False

        clean = features_df.dropna()
        if len(clean) < 60:
            logger.debug("[MultiHMM] insufficient data (%d < 60)", len(clean))
            return False

        try:
            from sklearn.preprocessing import StandardScaler  # type: ignore
        except ImportError:
            logger.warning("[MultiHMM] scikit-learn not installed; skipping scaling")
            StandardScaler = None  # type: ignore

        raw_X = clean.values.astype(np.float64)
        if StandardScaler is not None:
            scaler = StandardScaler()
            X = scaler.fit_transform(raw_X)
            self._scaler = scaler
        else:
            X = raw_X
            self._scaler = None

        n_features = X.shape[1]
        best_score = -np.inf
        best_model = None
        seeds = [self.random_state + i * 7 for i in range(self.n_seeds)]

        for seed in seeds:
            try:
                m = GaussianHMM(
                    n_components=self.n_regimes,
                    covariance_type=self.covariance_type,
                    n_iter=self.n_iter,
                    random_state=seed,
                )
                m.fit(X)
                s = m.score(X)
                if s > best_score:
                    best_score = s
                    best_model = m
            except Exception:
                continue

        if best_model is None:
            logger.warning("[MultiHMM] all seeds failed to converge")
            return False

        self._model = best_model
        self._label_map = RegimeHMM._build_label_map(best_model)
        self._fitted = True
        logger.info(
            "[MultiHMM] Fitted %d-regime model with %d features on %d obs "
            "(score=%.2f, covariance=%s)",
            self.n_regimes, n_features, len(clean), best_score, self.covariance_type,
        )
        return True

    def predict_proba(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Predict regime probabilities for each row.

        Returns:
            DataFrame with regime probability columns (p_bull, p_bear, p_sideways).
        """
        if not self._fitted or self._model is None:
            return self._fallback_proba(features_df)

        clean = features_df.dropna()
        if clean.empty:
            return pd.DataFrame()

        try:
            X = self._scale(clean.values.astype(np.float64))
            _, posteriors = self._model.score_samples(X)
        except Exception:
            return self._fallback_proba(features_df)

        result = pd.DataFrame(index=clean.index)
        for state_idx, label in self._label_map.items():
            if state_idx < posteriors.shape[1]:
                result[f"p_{label}"] = posteriors[:, state_idx]

        return result

    def predict_regime(self, features_df: pd.DataFrame) -> pd.Series:
        """Predict most likely regime for each row (Viterbi hard assignment)."""
        if not self._fitted or self._model is None:
            return pd.Series(dtype=str)

        clean = features_df.dropna()
        if clean.empty:
            return pd.Series(dtype=str)

        try:
            X = self._scale(clean.values.astype(np.float64))
            states = self._model.predict(X)
            return pd.Series(
                [self._label_map.get(int(s), f"state_{s}") for s in states],
                index=clean.index,
                name="regime",
            )
        except Exception:
            proba = self.predict_proba(features_df)
            if proba.empty:
                return pd.Series(dtype=str)
            regime_cols = [c for c in proba.columns if c.startswith("p_")]
            return proba[regime_cols].idxmax(axis=1).str.replace("p_", "", regex=False)

    def save(self, path: str | "Path") -> None:
        """Persist the fitted model to disk."""
        import joblib
        from pathlib import Path as _Path
        _path = _Path(path)
        _path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self._model,
                "scaler": self._scaler,
                "label_map": self._label_map,
                "n_regimes": self.n_regimes,
                "n_iter": self.n_iter,
                "random_state": self.random_state,
                "covariance_type": self.covariance_type,
                "class": "MultiFeatureRegimeHMM",
            },
            _path,
        )
        logger.info("[MultiHMM] Saved to %s", _path)

    @classmethod
    def load(cls, path: str | "Path") -> "MultiFeatureRegimeHMM":
        """Load a previously saved MultiFeatureRegimeHMM."""
        import joblib
        from pathlib import Path as _Path
        try:
            data = joblib.load(_Path(path))
        except (FileNotFoundError, EOFError, Exception) as exc:
            raise RuntimeError(f"[MultiHMM] Failed to load model from {path}: {exc}") from exc
        try:
            obj = cls(
                n_regimes=data.get("n_regimes", 3),
                n_iter=data.get("n_iter", 200),
                random_state=data.get("random_state", 42),
                covariance_type=data.get("covariance_type", "diag"),
            )
            obj._model = data["model"]
            obj._scaler = data.get("scaler")
            obj._label_map = data["label_map"]
        except KeyError as exc:
            raise RuntimeError(f"[MultiHMM] Model file {path} is missing required key: {exc}") from exc
        obj._fitted = True
        return obj

    def crisis_alert(self, features_df: pd.DataFrame, threshold: float = 0.3) -> dict:
        """Check if crisis probability is rising above threshold.

        Returns:
            Dict with ``crisis_prob``, ``alert``, ``trend``.
        """
        proba = self.predict_proba(features_df)
        if proba.empty or "p_bear" not in proba.columns:
            return {"crisis_prob": 0.0, "alert": False, "trend": "unknown"}

        latest = float(proba["p_bear"].iloc[-1])
        if len(proba) >= 5:
            prev = float(proba["p_bear"].iloc[-5])
            trend = "rising" if latest > prev else "falling"
        else:
            trend = "unknown"

        return {
            "crisis_prob": round(latest, 4),
            "alert": latest > threshold and trend == "rising",
            "trend": trend,
        }

    @staticmethod
    def _fallback_proba(features_df: pd.DataFrame) -> pd.DataFrame:
        """Simple volatility-based fallback when hmmlearn unavailable."""
        if features_df.empty:
            return pd.DataFrame()

        # Use first column as proxy for returns
        col = features_df.columns[0]
        vol = features_df[col].rolling(20, min_periods=5).std()

        result = pd.DataFrame(index=features_df.index)
        # High vol → bear, low vol → bull
        vol_pct = vol.rank(pct=True)
        result["p_bull"] = (1 - vol_pct).clip(0, 1).fillna(0.5)
        result["p_bear"] = vol_pct.clip(0, 1).fillna(0.3)
        result["p_sideways"] = 1.0 - result["p_bull"] - result["p_bear"]
        result["p_sideways"] = result["p_sideways"].clip(0, 1)

        return result


def build_multifeature_observables(
    returns: pd.Series,
    vix_changes: pd.Series | None = None,
    hy_spread_changes: pd.Series | None = None,
    vol_window: int = 20,
) -> pd.DataFrame:
    """Build the standard 4-observable feature matrix for MultiFeatureRegimeHMM.

    Args:
        returns: Daily returns.
        vix_changes: Daily VIX changes (optional).
        hy_spread_changes: Daily HY spread changes (optional).
        vol_window: Window for realized volatility.

    Returns:
        DataFrame with aligned feature columns.
    """
    features = pd.DataFrame(index=returns.index)
    features["daily_return"] = returns
    features["realized_vol"] = returns.rolling(vol_window, min_periods=10).std()

    if vix_changes is not None:
        features["vix_change"] = vix_changes.reindex(returns.index)
    if hy_spread_changes is not None:
        features["hy_spread_change"] = hy_spread_changes.reindex(returns.index)

    return features.dropna()
