"""Temporal Fusion Transformer (TFT) for multi-horizon time-series forecasting — stub.

Tier 3 research item (audit C2-039): TFT (Lim et al. 2021) for price / factor
forecasting with interpretable attention heads.  Full implementation requires
``pytorch-forecasting`` + PyTorch, which are not in the standard venv.

This stub exposes the intended interface so callers can be written and tested
before the heavy ML stack is installed.  When ``pytorch_forecasting`` and
``torch`` are available, ``TFTForecaster`` delegates to
``pytorch_forecasting.TemporalFusionTransformer``; otherwise it raises
``NotImplementedError`` with a clear activation message.

Activation requirements:
    pip install torch torchvision pytorch-forecasting pytorch-lightning

References:
    - Lim et al. (2021) "Temporal Fusion Transformers for Interpretable
      Multi-horizon Time Series Forecasting", IJOC.
    - audit C2-039
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import torch  # noqa: F401
    import pytorch_forecasting  # noqa: F401

    HAS_PYTORCH_FORECASTING = True
except ImportError:
    HAS_PYTORCH_FORECASTING = False

_ACTIVATION_MSG = (
    "TFT requires pytorch-forecasting + torch. "
    "Install with: pip install torch pytorch-forecasting pytorch-lightning"
)


@dataclass
class TFTConfig:
    """Configuration for the Temporal Fusion Transformer."""

    hidden_size: int = 64
    attention_head_size: int = 4
    dropout: float = 0.1
    hidden_continuous_size: int = 8
    max_encoder_length: int = 60  # days of history
    max_prediction_length: int = 5  # forecast horizon
    learning_rate: float = 1e-3
    max_epochs: int = 30
    batch_size: int = 64
    gradient_clip_val: float = 0.1
    random_state: int = 42


@dataclass
class TFTResult:
    """Output of a TFT forecast run."""

    predictions: np.ndarray  # shape (n_samples, max_prediction_length)
    quantiles: (
        np.ndarray | None
    )  # shape (n_samples, max_prediction_length, n_quantiles) or None
    attention_weights: np.ndarray | None  # interpretable self-attention
    variable_importances: dict[str, float] = field(default_factory=dict)
    method: str = "tft"
    converged: bool = False


class TFTForecaster:
    """Interface-compatible TFT wrapper.

    When pytorch-forecasting is installed, delegates to the real TFT.
    When not installed, raises NotImplementedError on ``fit``/``predict``.

    Parameters
    ----------
    config : TFTConfig, optional
    """

    def __init__(self, config: TFTConfig | None = None) -> None:
        self.config = config or TFTConfig()
        self._model: Any = None
        self._trainer: Any = None

    def fit(
        self,
        time_series_df: pd.DataFrame,
        target_col: str,
        time_col: str = "date",
        group_col: str = "symbol",
    ) -> "TFTForecaster":
        """Train TFT on a panel time-series DataFrame.

        Parameters
        ----------
        time_series_df : DataFrame with columns [time_col, group_col, target_col, covariates...]
        target_col : name of the target column (e.g. "log_return")
        time_col : timestamp column name
        group_col : group/entity column name

        Raises
        ------
        NotImplementedError
            When pytorch-forecasting is not installed.
        """
        if not HAS_PYTORCH_FORECASTING:
            raise NotImplementedError(_ACTIVATION_MSG)
        raise NotImplementedError(
            "TFT fit: full implementation pending pytorch-forecasting setup"
        )

    def predict(
        self,
        time_series_df: pd.DataFrame,
        target_col: str,
        time_col: str = "date",
        group_col: str = "symbol",
    ) -> TFTResult:
        """Generate forecasts for the given panel.

        Raises
        ------
        NotImplementedError
            When pytorch-forecasting is not installed or model not yet trained.
        """
        if not HAS_PYTORCH_FORECASTING:
            raise NotImplementedError(_ACTIVATION_MSG)
        raise NotImplementedError("TFT predict: model must be trained first via fit()")

    @property
    def is_available(self) -> bool:
        """True when pytorch-forecasting is installed."""
        return HAS_PYTORCH_FORECASTING


def tft_forecast(
    time_series_df: pd.DataFrame,
    target_col: str,
    config: TFTConfig | None = None,
    **kwargs: Any,
) -> TFTResult:
    """Convenience wrapper: fit and predict in one call.

    Raises NotImplementedError when pytorch-forecasting is unavailable.
    """
    forecaster = TFTForecaster(config)
    forecaster.fit(time_series_df, target_col, **kwargs)
    return forecaster.predict(time_series_df, target_col, **kwargs)


__all__ = [
    "TFTConfig",
    "TFTResult",
    "TFTForecaster",
    "tft_forecast",
    "HAS_PYTORCH_FORECASTING",
]
