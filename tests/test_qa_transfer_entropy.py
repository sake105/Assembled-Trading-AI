"""Tests for transfer_entropy_binned / transfer_entropy_ksg — C4-080 closure.

Reference:
- Schreiber, T. (2000). PRL 85(2): 461-464
- Kraskov, Stögbauer, Grassberger (2004). Phys. Rev. E 69, 066138
- Wibral, Vicente, Lindner (2014). Ch. 1 in Directed Information Measures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.qa.transfer_entropy import (
    transfer_entropy_binned,
    transfer_entropy_ksg,
)


def _causal_pair(
    n: int = 1000, coupling: float = 0.7, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Generate X causally driving Y: Y_t = coupling * X_{t-1} + noise."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    y = np.zeros(n)
    y[0] = rng.normal(0, 1)
    for t in range(1, n):
        y[t] = coupling * x[t - 1] + 0.3 * rng.normal()
    return x, y


def _independent_pair(n: int = 1000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Two independent series."""
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, n), rng.normal(0, 1, n)


# ---------------------------------------------------------------------------
# transfer_entropy_binned
# ---------------------------------------------------------------------------


def test_binned_te_positive_for_causal_direction():
    """When X causes Y → TE(X→Y) > 0 in nats."""
    x, y = _causal_pair(n=1500, coupling=0.7, seed=1)
    te = transfer_entropy_binned(x, y, lag=1, n_bins=8)
    assert te > 0.05, f"Expected substantial TE for causal pair, got {te:.4f}"


def test_binned_te_near_zero_for_independent():
    """Independent series → TE small (relative to causal case).

    Histogram-based TE has a finite-sample bias floor of order
    O(n_bins^2 / N) — for n_bins=8, N=3000 this is ~0.05-0.15 nats. We can't
    expect 0 even for true independence; what we CAN expect is that TE on
    independent series is much smaller than TE on a known-causal pair
    measured the same way (asserted in test_binned_te_much_smaller_than_causal).
    """
    x, y = _independent_pair(n=3000, seed=2)
    te = transfer_entropy_binned(x, y, lag=1, n_bins=6)
    # With n_bins=6 (lower) and N=3000 (higher) the bias floor drops to ~0.05.
    assert te < 0.15, f"Expected TE<<0.5 for independent, got {te:.4f}"


def test_binned_te_much_smaller_than_causal():
    """The strongest test for independence: TE(independent) << TE(causal) at
    the SAME (n_bins, N) settings — bias affects both equally."""
    n, n_bins = 3000, 6
    x_c, y_c = _causal_pair(n=n, coupling=0.7, seed=20)
    te_causal = transfer_entropy_binned(x_c, y_c, lag=1, n_bins=n_bins)
    x_i, y_i = _independent_pair(n=n, seed=21)
    te_indep = transfer_entropy_binned(x_i, y_i, lag=1, n_bins=n_bins)
    # Signal should be at least 3× the bias floor — much stronger than indep.
    assert te_causal > 3 * te_indep, (
        f"Causal TE {te_causal:.4f} should be >> independent TE {te_indep:.4f}"
    )


def test_binned_te_asymmetric_under_one_way_causation():
    """X causes Y (not vice versa) → TE(X→Y) > TE(Y→X)."""
    x, y = _causal_pair(n=1500, coupling=0.7, seed=3)
    te_forward = transfer_entropy_binned(x, y, lag=1, n_bins=8)
    te_backward = transfer_entropy_binned(y, x, lag=1, n_bins=8)
    assert te_forward > te_backward, (
        f"Expected TE(X→Y)>{te_forward:.4f} > TE(Y→X)={te_backward:.4f} for one-way causation"
    )


def test_binned_te_rejects_length_mismatch():
    with pytest.raises(ValueError, match="length mismatch"):
        transfer_entropy_binned([1.0] * 100, [2.0] * 50)


def test_binned_te_rejects_invalid_lag():
    with pytest.raises(ValueError, match="lag"):
        transfer_entropy_binned([1.0] * 100, [2.0] * 100, lag=0)


def test_binned_te_rejects_short_input():
    with pytest.raises(ValueError, match="obs"):
        transfer_entropy_binned([1.0] * 20, [2.0] * 20, lag=1)


def test_binned_te_returns_non_negative():
    """TE must always be ≥0 by mathematical construction (Schreiber 2000)."""
    x, y = _independent_pair(n=500, seed=99)
    te = transfer_entropy_binned(x, y, lag=1, n_bins=8)
    assert te >= 0


def test_binned_te_handles_nan_via_dropna():
    """NaN rows in source/target should be dropped jointly."""
    x, y = _causal_pair(n=500, coupling=0.6, seed=5)
    x_with_nan = x.copy()
    x_with_nan[100] = np.nan
    y_with_nan = y.copy()
    y_with_nan[200] = np.nan
    te = transfer_entropy_binned(x_with_nan, y_with_nan, lag=1, n_bins=8)
    assert te > 0.02  # Causal signal still detectable after dropping 2 NaN rows


# ---------------------------------------------------------------------------
# transfer_entropy_ksg
# ---------------------------------------------------------------------------


def test_ksg_te_returns_none_without_sklearn(monkeypatch):
    """Graceful degradation: returns None if sklearn unavailable.

    F-senior-c4080-2: must clear sklearn from sys.modules BEFORE the
    monkeypatched __import__ runs, otherwise the cached module is returned
    without invoking the import machinery and the test would silently pass
    via the cache rather than the absent-sklearn code path.
    """
    import builtins
    import sys

    # Evict any cached sklearn submodules so the import inside
    # transfer_entropy_ksg has to go through __import__ (and hit our stub).
    for mod_name in list(sys.modules):
        if mod_name == "sklearn" or mod_name.startswith("sklearn."):
            monkeypatch.delitem(sys.modules, mod_name, raising=False)

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sklearn" or name.startswith("sklearn."):
            raise ImportError("simulated sklearn absent")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    x, y = _causal_pair(n=200, seed=10)
    result = transfer_entropy_ksg(x, y, lag=1)
    assert result is None


def test_ksg_te_positive_for_causal_direction():
    """sklearn KSG-based TE positive for X→Y causation."""
    pytest.importorskip("sklearn")
    x, y = _causal_pair(n=1500, coupling=0.7, seed=11)
    te = transfer_entropy_ksg(x, y, lag=1, k=3)
    assert te is not None
    assert te > 0, f"Expected positive TE, got {te}"


def test_ksg_te_near_zero_for_independent():
    """sklearn KSG-based TE near zero for independent series."""
    pytest.importorskip("sklearn")
    x, y = _independent_pair(n=1500, seed=12)
    te = transfer_entropy_ksg(x, y, lag=1, k=3)
    assert te is not None
    # KSG with k=3 on finite samples has ~0.01-0.05 bias floor
    assert te < 0.1, f"Expected TE≈0 for independent, got {te}"


def test_ksg_te_rejects_length_mismatch():
    pytest.importorskip("sklearn")
    with pytest.raises(ValueError, match="length mismatch"):
        transfer_entropy_ksg([1.0] * 100, [2.0] * 50)


def test_ksg_te_returns_non_negative():
    pytest.importorskip("sklearn")
    x, y = _independent_pair(n=500, seed=13)
    te = transfer_entropy_ksg(x, y, lag=1, k=3)
    if te is not None:
        assert te >= 0


def test_ksg_te_accepts_pandas_series():
    """Should accept pd.Series in addition to np.ndarray."""
    pytest.importorskip("sklearn")
    x, y = _causal_pair(n=500, seed=14)
    te = transfer_entropy_ksg(pd.Series(x), pd.Series(y), lag=1, k=3)
    assert te is not None
    assert te >= 0
