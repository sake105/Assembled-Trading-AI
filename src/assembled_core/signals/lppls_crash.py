"""Log-Periodic Power Law Singularity (LPPLS) bubble/crash detector.

Implements Johansen-Ledoit-Sornette (JLS) model in pure numpy.
Uses `lppls` pip package if installed; otherwise pure numpy fallback.

Reference: Sornette, D. (2003) Why Stock Markets Crash. Princeton University Press.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class LPPLSCrashDetector:
    """Fit LPPLS model to log-price series and estimate crash probability.

    The LPPLS formula:
        ln(p(t)) = A + B(tc - t)^m + C(tc - t)^m * cos(w*ln(tc - t) + phi)

    Parameters
    ----------
    fit_window:
        Number of past trading days to use for fitting (default 252 = 1 year).
    max_searches:
        Random restarts for the nonlinear optimisation.
    """

    def __init__(self, fit_window: int = 252, max_searches: int = 50) -> None:
        self.fit_window = fit_window
        self.max_searches = max_searches

    def fit_and_score(self, prices: np.ndarray) -> dict[str, Any]:
        """Fit LPPLS and return crash confidence metrics.

        Returns
        -------
        Dict with keys:
            - ``tc_estimate``: estimated critical time (days from last price)
            - ``crash_confidence``: 0-1 probability of imminent crash
            - ``time_to_crash_days``: estimated days until tc
            - ``params``: fitted (m, omega, phi, A, B, C)
            - ``method``: "lppls_lib" or "numpy_fallback"
        """
        try:
            return self._fit_lppls_lib(prices)
        except ImportError:
            pass
        except Exception as exc:
            logger.debug("lppls lib failed: %s — falling back to numpy", exc)
        return self._fit_numpy(prices)

    # ------------------------------------------------------------------
    # lppls library path
    # ------------------------------------------------------------------

    def _fit_lppls_lib(self, prices: np.ndarray) -> dict[str, Any]:
        from lppls import lppls as lppls_model  # noqa: PLC0415

        window = prices[-self.fit_window :]
        log_p = np.log(np.clip(window, 1e-9, None))
        t = np.arange(len(log_p))

        model = lppls_model.LPPLS(observations=np.array([t, log_p]))
        results = model.fit(self.max_searches)
        ci = model.compute_indicators(results)

        tc = float(results.get("tc", len(log_p)))
        return {
            "tc_estimate": tc,
            "crash_confidence": float(ci.get("pos_conf", 0.0)),
            "time_to_crash_days": tc - len(log_p),
            "params": results,
            "method": "lppls_lib",
        }

    # ------------------------------------------------------------------
    # Pure numpy fallback
    # ------------------------------------------------------------------

    def _fit_numpy(self, prices: np.ndarray) -> dict[str, Any]:
        """Simplified LPPLS fit via random-restart Levenberg-Marquardt.

        Linearises A, B, C given (tc, m, omega) and optimises nonlinear params.
        """
        window = prices[-self.fit_window :]
        log_p = np.log(np.clip(window, 1e-9, None))
        n = len(log_p)
        t = np.arange(n, dtype=float)

        best_result: dict[str, Any] | None = None
        best_err = np.inf

        rng = np.random.default_rng(0)
        for _ in range(self.max_searches):
            # Random initialisation of nonlinear parameters
            tc_init = float(rng.uniform(n * 1.05, n * 1.5))
            m_init = float(rng.uniform(0.1, 0.9))
            omega_init = float(rng.uniform(4.0, 15.0))

            result = self._lppls_gradient_step(t, log_p, tc_init, m_init, omega_init)
            if result is not None and result["residual"] < best_err:
                best_err = result["residual"]
                best_result = result

        if best_result is None:
            return {
                "tc_estimate": float(n),
                "crash_confidence": 0.0,
                "time_to_crash_days": 0.0,
                "params": {},
                "method": "numpy_fallback_failed",
            }

        tc = best_result["tc"]
        confidence = self._compute_confidence(best_result, n)
        return {
            "tc_estimate": float(tc),
            "crash_confidence": float(confidence),
            "time_to_crash_days": float(tc - n),
            "params": {k: float(v) for k, v in best_result.items() if k != "residual"},
            "method": "numpy_fallback",
        }

    def _lppls_gradient_step(
        self,
        t: np.ndarray,
        log_p: np.ndarray,
        tc: float,
        m: float,
        omega: float,
    ) -> dict[str, Any] | None:
        """One linearised solution: given (tc, m, omega) solve for A, B, C."""
        dt = tc - t
        if np.any(dt <= 0):
            return None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            power = dt**m
            phase = omega * np.log(dt)

        f1 = np.ones(len(t))
        f2 = power
        f3 = power * np.cos(phase)
        f4 = power * np.sin(phase)

        X = np.column_stack([f1, f2, f3, f4])
        try:
            coef, resid, *_ = np.linalg.lstsq(X, log_p, rcond=None)
        except np.linalg.LinAlgError:
            return None

        A, B, C1, C2 = coef
        C = np.sqrt(C1**2 + C2**2)
        phi = np.arctan2(C2, C1)
        residual = float(np.sum((log_p - X @ coef) ** 2))

        return {
            "tc": tc,
            "m": m,
            "omega": omega,
            "A": A,
            "B": B,
            "C": C,
            "phi": phi,
            "residual": residual,
        }

    @staticmethod
    def _compute_confidence(result: dict, n: int) -> float:
        """Heuristic crash confidence from parameter validity checks.

        Sornette conditions: 0.1 < m < 0.9, 6 < omega < 13, B < 0, C/B < 1.
        """
        score = 0
        m = result.get("m", 0.5)
        omega = result.get("omega", 8.0)
        B = result.get("B", -1.0)
        C = result.get("C", 0.5)
        tc = result.get("tc", n + 10)

        if 0.1 < m < 0.9:
            score += 1
        if 6 < omega < 13:
            score += 1
        if B < 0:
            score += 1
        if abs(C) < abs(B):
            score += 1
        if n < tc < n * 2:
            score += 1

        return score / 5.0
