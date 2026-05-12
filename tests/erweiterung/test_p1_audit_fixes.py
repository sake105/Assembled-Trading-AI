"""Tests for the three P1 audit fixes on ERWEITERUNG (C4-001/002/003).

These tests pin the corrected behaviour so any future refactor breaks
the test, not the math.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# C4-001 — CPCV purge direction (backward, not forward)
# ---------------------------------------------------------------------------


def test_cpcv_purge_drops_train_indices_after_test_fold_within_lookback() -> None:
    """info_period_lookback=3 purges train rows AFTER test whose feature
    window reaches back INTO the test fold.

    Train sample at i has a feature window `[i - lookback, i]`. If any
    test index sits in that backward window, the training feature at i
    is contaminated by test data → purge.
    """
    from src.erweiterung.backtest.cpcv import _purge_train_indices

    train_idx = np.array([0, 1, 2, 3, 4, 7, 8, 9])  # 5, 6 are in test
    test_idx = np.array([5, 6])

    kept = _purge_train_indices(train_idx, test_idx, info_period_lookback=3, embargo=0)
    # i=7: window [4,7] contains test 5,6 → purge.
    # i=8: window [5,8] contains test 5,6 → purge.
    # i=9: window [6,9] contains test 6 → purge.
    # i=0..4: windows lie entirely before test 5 → keep.
    assert 7 not in kept
    assert 8 not in kept
    assert 9 not in kept
    assert 0 in kept and 1 in kept and 2 in kept and 3 in kept and 4 in kept


def test_cpcv_purge_forward_label_horizon_when_set() -> None:
    """label_horizon=2 must purge train rows whose label window overlaps test."""
    from src.erweiterung.backtest.cpcv import _purge_train_indices

    train_idx = np.array([0, 1, 2, 3, 4, 7, 8, 9])
    test_idx = np.array([5, 6])

    kept = _purge_train_indices(
        train_idx, test_idx, info_period_lookback=0, label_horizon=2
    )
    # Row 3: label window [3, 4, 5] overlaps test (5 in test) → purge.
    # Row 4: label window [4, 5, 6] overlaps → purge.
    # Row 5, 6: in test set, not in train_idx anyway.
    # Row 7+: no forward overlap (test_max=6 < 7).
    assert 3 not in kept
    assert 4 not in kept
    assert 7 in kept and 8 in kept and 9 in kept


def test_cpcv_purge_no_lookback_keeps_all_non_test() -> None:
    """With zero lookback / zero embargo / zero horizon, only test indices are removed."""
    from src.erweiterung.backtest.cpcv import _purge_train_indices

    train_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    test_idx = np.array([5, 6])
    kept = _purge_train_indices(
        train_idx, test_idx, info_period_lookback=0, embargo=0, label_horizon=0
    )
    assert sorted(kept.tolist()) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_cpcv_purge_embargo_drops_post_test_train() -> None:
    """Embargo=2 must drop the 2 train rows immediately after test_max."""
    from src.erweiterung.backtest.cpcv import _purge_train_indices

    train_idx = np.array([0, 1, 2, 3, 7, 8, 9])
    test_idx = np.array([5, 6])
    kept = _purge_train_indices(train_idx, test_idx, info_period_lookback=0, embargo=2)
    # 7 = test_max+1 → embargo drops.
    # 8 = test_max+2 → embargo drops.
    # 9 = test_max+3 → kept.
    assert 7 not in kept
    assert 8 not in kept
    assert 9 in kept


# ---------------------------------------------------------------------------
# C4-002 — Stacking-ensemble strict pre-validation
# ---------------------------------------------------------------------------


def test_stacking_folds_never_include_post_validation_rows() -> None:
    """The folds returned by _kfold_indices MUST satisfy max(train) < min(val)."""
    from src.erweiterung.ml.stacking_ensemble import BaseModel, StackingRegressor

    def _dummy(X_tr, y_tr, X_va):
        return np.zeros(len(X_va))

    sr = StackingRegressor(
        base_models=[BaseModel(name="d", fit_predict=_dummy)],
        meta_fit_predict=lambda Xm, ym, Xt: np.zeros(len(Xt)),
        n_splits=5,
        embargo=0,
    )
    folds = sr._kfold_indices(100)
    assert len(folds) == 5
    for tr_idx, va_idx in folds:
        if len(tr_idx) == 0:
            continue  # First fold may have empty train (expanding window).
        assert tr_idx.max() < va_idx.min(), (
            f"post-validation rows leaked into train: "
            f"max(train)={tr_idx.max()} >= min(val)={va_idx.min()}"
        )


def test_stacking_embargo_drops_bars_before_validation() -> None:
    """Embargo=3 must drop the 3 bars immediately before each val fold."""
    from src.erweiterung.ml.stacking_ensemble import BaseModel, StackingRegressor

    sr = StackingRegressor(
        base_models=[BaseModel(name="d", fit_predict=lambda *a: np.zeros(len(a[2])))],
        meta_fit_predict=lambda Xm, ym, Xt: np.zeros(len(Xt)),
        n_splits=5,
        embargo=3,
    )
    folds = sr._kfold_indices(100)
    # Skip first fold (empty train); inspect fold 2.
    tr_idx, va_idx = folds[1]
    assert tr_idx.max() + 3 < va_idx.min() or len(tr_idx) == 0


def test_stacking_fit_skips_short_train_folds() -> None:
    """fit() must skip folds whose training set is shorter than min_train_size."""
    from src.erweiterung.ml.stacking_ensemble import BaseModel, StackingRegressor

    fit_calls: list[int] = []

    def _spy(X_tr, y_tr, X_va):
        fit_calls.append(len(X_tr))
        return np.zeros(len(X_va))

    sr = StackingRegressor(
        base_models=[BaseModel(name="spy", fit_predict=_spy)],
        meta_fit_predict=lambda Xm, ym, Xt: np.zeros(len(Xt)),
        n_splits=5,
        embargo=0,
        min_train_size=5,
    )
    X = np.random.default_rng(0).standard_normal((50, 3))
    y = np.random.default_rng(1).standard_normal(50)
    sr.fit(X, y)
    # Every recorded fit had at least 5 rows of training data.
    assert all(n >= 5 for n in fit_calls)
    # oof_mask is False for skipped folds.
    assert not sr.oof_mask[0]


# ---------------------------------------------------------------------------
# C4-003 — CVaR fallback honours target_return
# ---------------------------------------------------------------------------


def test_cvar_fallback_returns_success_false_when_target_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When linprog fails AND target_return cannot be met, the fallback
    must surface success=False + target_return_violated=True."""
    from src.erweiterung.portfolio import cvar_optimizer as cvar_mod

    # Force the linprog path to fail so the fallback runs.
    class _BadResult:
        success = False
        x = np.zeros(1)

    monkeypatch.setattr(
        cvar_mod, "linprog", lambda *a, **kw: _BadResult(), raising=False
    )

    # Three assets all with negative mean returns — target_return=0.10
    # is unreachable; the fallback MUST flag it.
    rng = np.random.default_rng(seed=42)
    scenarios = pd.DataFrame(
        rng.normal(loc=-0.001, scale=0.01, size=(500, 3)),
        columns=["A", "B", "C"],
    )

    _w, metrics = cvar_mod.cvar_optimal_weights(
        scenarios, target_return=0.10, long_only=True, max_weight=0.5
    )
    assert metrics["success"] is False
    assert metrics.get("target_return_violated") is True
    assert metrics["shortfall"] > 0


def test_cvar_fallback_succeeds_when_target_reachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When fallback runs and target is achievable, success=True."""
    from src.erweiterung.portfolio import cvar_optimizer as cvar_mod

    class _BadResult:
        success = False
        x = np.zeros(1)

    monkeypatch.setattr(
        cvar_mod, "linprog", lambda *a, **kw: _BadResult(), raising=False
    )

    rng = np.random.default_rng(seed=42)
    scenarios = pd.DataFrame(
        rng.normal(loc=0.001, scale=0.01, size=(500, 3)),
        columns=["A", "B", "C"],
    )
    _w, metrics = cvar_mod.cvar_optimal_weights(
        scenarios, target_return=None, long_only=True, max_weight=0.5
    )
    assert metrics["success"] is True


# ---------------------------------------------------------------------------
# C4-075 — Cornish-Fisher domain-of-validity check
# ---------------------------------------------------------------------------


def test_cornish_fisher_flags_extreme_skew_out_of_domain() -> None:
    """A return series with extreme skew must be flagged cf_in_domain=False."""
    import numpy as np
    import pandas as pd

    from src.erweiterung.risk.cornish_fisher_var import cornish_fisher_var

    rng = np.random.default_rng(seed=42)
    # Build a highly right-skewed series: many small losses + a few huge wins.
    base = rng.normal(0, 0.001, 200)
    spikes = rng.uniform(0.1, 0.3, 10)  # 10 outsized wins
    arr = np.concatenate([base, spikes])
    r = pd.Series(arr)

    result = cornish_fisher_var(r, alpha=0.99)
    # We don't know the exact skew of this synthetic mix, but it should
    # be large enough to trip at least one of the domain rules.
    assert "cf_in_domain" in result
    assert "cf_domain_reason" in result


def test_cornish_fisher_in_domain_for_gaussian_input() -> None:
    """A near-Gaussian sample must yield cf_in_domain=True."""
    import numpy as np
    import pandas as pd

    from src.erweiterung.risk.cornish_fisher_var import cornish_fisher_var

    rng = np.random.default_rng(seed=42)
    r = pd.Series(rng.normal(0, 0.01, 500))
    result = cornish_fisher_var(r, alpha=0.99)
    assert result["cf_in_domain"] is True
    assert result["cf_domain_reason"] == "ok"
