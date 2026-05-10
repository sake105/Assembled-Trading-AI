"""Shared fixtures for erweiterung tests.

Alle Tests laufen offline-only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path, monkeypatch):
    """Isoliere Disk-Cache pro Test."""
    monkeypatch.setenv("ERWEITERUNG_CACHE_DIR", str(tmp_path / "cache"))


@pytest.fixture
def synthetic_returns():
    """Reproduzierbare Returns für 5 Symbole, 500 Tage."""
    rng = np.random.default_rng(42)
    n_days = 500
    n_sym = 5
    dates = pd.date_range("2022-01-01", periods=n_days, freq="B", tz="UTC")
    sym_names = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    # AR(1)-Struktur mit Cross-Korrelation
    factor = rng.normal(0.0005, 0.012, n_days)
    idio = rng.normal(0, 0.015, (n_days, n_sym))
    returns = factor[:, None] * np.array([1.0, 0.9, 0.7, -0.3, 0.5]) + idio
    df = pd.DataFrame(returns, index=dates, columns=sym_names)
    return df


@pytest.fixture
def synthetic_prices(synthetic_returns):
    """Cumulative price levels."""
    return (1 + synthetic_returns).cumprod() * 100


@pytest.fixture
def synthetic_panel(synthetic_returns):
    """Long-format panel."""
    rows = []
    for sym in synthetic_returns.columns:
        for d, r in synthetic_returns[sym].items():
            rows.append({"date": d, "symbol": sym, "return": float(r)})
    return pd.DataFrame(rows)
