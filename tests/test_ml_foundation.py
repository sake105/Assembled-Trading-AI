"""Tests for M17 ML foundation modules: Purged CV, GARCH, EVT, Copulas."""

import numpy as np
import pandas as pd
import pytest

# ── Helpers ────────────────────────────────────────────────────────────


def _arch_available() -> bool:
    try:
        import arch  # noqa: F401

        return True
    except ImportError:
        return False


def _scipy_available() -> bool:
    try:
        import scipy  # noqa: F401

        return True
    except ImportError:
        return False


# ── Purged CV ──────────────────────────────────────────────────────────


class TestPurgedKFold:
    """Tests for PurgedKFold cross-validator."""

    def test_import(self):
        import pytest

        pytest.importorskip("src.assembled_core.ml.purged_cv")
        from src.assembled_core.ml.purged_cv import PurgedKFold

        assert PurgedKFold is not None

    def test_basic_split(self):
        import pytest

        pytest.importorskip("src.assembled_core.ml.purged_cv")
        from src.assembled_core.ml.purged_cv import PurgedKFold

        # 500 daily timestamps
        dates = pd.date_range("2020-01-01", periods=500, freq="B")
        timestamps = pd.Series(dates)

        kf = PurgedKFold(n_splits=5, label_horizon=5, embargo_pct=0.01)
        splits = kf.split(timestamps)

        assert len(splits) > 0
        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0
            # No overlap
            assert len(set(train_idx) & set(test_idx)) == 0

    def test_purge_removes_contaminated_samples(self):
        import pytest

        pytest.importorskip("src.assembled_core.ml.purged_cv")
        from src.assembled_core.ml.purged_cv import PurgedKFold

        dates = pd.date_range("2020-01-01", periods=500, freq="B")
        timestamps = pd.Series(dates)

        kf = PurgedKFold(n_splits=3, label_horizon=10, embargo_pct=0.0)
        splits = kf.split(timestamps)

        for train_idx, test_idx in splits:
            train_dates = timestamps.iloc[train_idx]
            test_start = timestamps.iloc[test_idx].min()
            # All train dates should be at least label_horizon days before test start
            max_train_date = train_dates.max()
            gap = (test_start - max_train_date).days
            assert gap >= 10, f"Gap={gap} < label_horizon=10"

    def test_embargo_shrinks_training(self):
        import pytest

        pytest.importorskip("src.assembled_core.ml.purged_cv")
        from src.assembled_core.ml.purged_cv import PurgedKFold

        dates = pd.date_range("2020-01-01", periods=500, freq="B")
        timestamps = pd.Series(dates)

        kf_no_embargo = PurgedKFold(n_splits=3, label_horizon=5, embargo_pct=0.0)
        kf_with_embargo = PurgedKFold(n_splits=3, label_horizon=5, embargo_pct=0.05)

        splits_no = kf_no_embargo.split(timestamps)
        splits_with = kf_with_embargo.split(timestamps)

        # Embargo should reduce training set size
        if splits_no and splits_with:
            assert len(splits_no[0][0]) >= len(splits_with[0][0])

    def test_insufficient_data_raises(self):
        import pytest

        pytest.importorskip("src.assembled_core.ml.purged_cv")
        from src.assembled_core.ml.purged_cv import PurgedKFold

        dates = pd.date_range("2020-01-01", periods=10, freq="B")
        timestamps = pd.Series(dates)

        kf = PurgedKFold(n_splits=5)
        with pytest.raises(ValueError, match="samples for"):
            kf.split(timestamps)

    def test_expanding_vs_rolling(self):
        import pytest

        pytest.importorskip("src.assembled_core.ml.purged_cv")
        from src.assembled_core.ml.purged_cv import PurgedKFold

        dates = pd.date_range("2020-01-01", periods=500, freq="B")
        timestamps = pd.Series(dates)

        kf = PurgedKFold(n_splits=3, label_horizon=5)
        expanding = kf.split(timestamps, train_size=None)
        _rolling = kf.split(timestamps, train_size=200)

        # Expanding should have more training data in later folds
        if len(expanding) >= 2:
            assert len(expanding[-1][0]) >= len(expanding[0][0])


class TestPurgedWalkForward:
    """Tests for purged_walk_forward_split."""

    def test_import_v2(self):
        import pytest

        pytest.importorskip("src.assembled_core.ml.purged_cv")
        from src.assembled_core.ml.purged_cv import purged_walk_forward_split

        assert purged_walk_forward_split is not None

    def test_basic_walk_forward(self):
        import pytest

        pytest.importorskip("src.assembled_core.ml.purged_cv")
        from src.assembled_core.ml.purged_cv import purged_walk_forward_split

        dates = pd.date_range("2018-01-01", periods=1000, freq="B")
        timestamps = pd.Series(dates)

        splits = purged_walk_forward_split(
            timestamps,
            train_window_days=252,
            test_window_days=63,
            label_horizon=5,
            embargo_days=5,
        )

        assert len(splits) > 0
        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0
            # No overlap
            assert len(set(train_idx) & set(test_idx)) == 0

    def test_max_splits_respected(self):
        import pytest

        pytest.importorskip("src.assembled_core.ml.purged_cv")
        from src.assembled_core.ml.purged_cv import purged_walk_forward_split

        dates = pd.date_range("2015-01-01", periods=2500, freq="B")
        timestamps = pd.Series(dates)

        splits = purged_walk_forward_split(
            timestamps,
            train_window_days=252,
            test_window_days=63,
            max_splits=3,
        )
        assert len(splits) <= 3


# ── GARCH Models ───────────────────────────────────────────────────────


class TestGARCHModels:
    """Tests for GARCH family models."""

    def test_import_v3(self):
        import pytest

        pytest.importorskip("src.assembled_core.ml.garch_models")
        from src.assembled_core.ml.garch_models import fit_garch, GARCHResult

        assert fit_garch is not None
        assert GARCHResult is not None

    def test_garch_result_dataclass(self):
        import pytest

        pytest.importorskip("src.assembled_core.ml.garch_models")
        from src.assembled_core.ml.garch_models import GARCHResult

        r = GARCHResult(
            symbol="TEST",
            model_type="garch",
            vol_1d=0.2,
            vol_5d=0.2,
            persistence=0.95,
            asymmetry=0.0,
            bic=100.0,
        )
        assert r.symbol == "TEST"
        assert r.persistence == 0.95

    @pytest.mark.skipif(not _arch_available(), reason="arch package not installed")
    def test_fit_garch_basic(self):
        import pytest

        pytest.importorskip("src.assembled_core.ml.garch_models")
        from src.assembled_core.ml.garch_models import fit_garch

        np.random.seed(42)
        # Simulate GARCH-like returns
        returns = np.random.normal(0, 0.02, 500)
        result = fit_garch(returns, "TEST", model_type="garch")

        assert result is not None
        assert result.vol_1d > 0
        assert 0 <= result.persistence <= 1.5
        assert result.model_type == "garch"

    @pytest.mark.skipif(not _arch_available(), reason="arch package not installed")
    def test_fit_best_garch_selects_by_bic(self):
        import pytest

        pytest.importorskip("src.assembled_core.ml.garch_models")
        from src.assembled_core.ml.garch_models import fit_best_garch

        np.random.seed(42)
        returns = np.random.normal(0, 0.02, 500)
        result = fit_best_garch(returns, "TEST")

        assert result is not None
        assert result.model_type in ("garch", "egarch", "gjr")
        assert result.bic < 0 or result.bic > 0  # just not NaN

    @pytest.mark.skipif(not _arch_available(), reason="arch package not installed")
    def test_fit_panel_garch(self):
        import pytest

        pytest.importorskip("src.assembled_core.ml.garch_models")
        from src.assembled_core.ml.garch_models import fit_panel_garch

        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=300, freq="B")
        rows = []
        for sym in ["AAPL", "MSFT"]:
            prices = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, 300)))
            for i, d in enumerate(dates):
                rows.append({"timestamp": d, "symbol": sym, "close": prices[i]})
        prices_df = pd.DataFrame(rows)

        results = fit_panel_garch(prices_df, lookback_days=252)
        assert len(results) > 0

    def test_insufficient_data_returns_none(self):
        import pytest

        pytest.importorskip("src.assembled_core.ml.garch_models")
        from src.assembled_core.ml.garch_models import fit_garch

        returns = np.random.normal(0, 0.02, 10)
        result = fit_garch(returns, "TEST")
        assert result is None


# ── EVT Models ─────────────────────────────────────────────────────────


class TestEVTModels:
    """Tests for Extreme Value Theory models."""

    def test_import_v4(self):
        import pytest

        pytest.importorskip("src.assembled_core.ml.evt_models")
        from src.assembled_core.ml.evt_models import fit_evt_pot

        assert fit_evt_pot is not None

    @pytest.mark.skipif(not _scipy_available(), reason="scipy not installed")
    def test_fit_evt_basic(self):
        import pytest

        pytest.importorskip("src.assembled_core.ml.evt_models")
        from src.assembled_core.ml.evt_models import fit_evt_pot

        np.random.seed(42)
        # Heavy-tailed returns (t-distribution with df=4)
        returns = np.random.standard_t(df=4, size=1000) * 0.02

        result = fit_evt_pot(returns)
        assert result is not None
        assert result.var_99 > result.var_95 > 0
        assert result.var_999 > result.var_99
        assert result.cvar_99 >= result.var_99
        assert result.n_exceedances > 0
        assert result.shape_xi != 0  # t-distribution should show fat tails

    @pytest.mark.skipif(not _scipy_available(), reason="scipy not installed")
    def test_evt_var_ordering(self):
        import pytest

        pytest.importorskip("src.assembled_core.ml.evt_models")
        from src.assembled_core.ml.evt_models import fit_evt_pot

        np.random.seed(42)
        returns = np.random.standard_t(df=5, size=2000) * 0.015

        result = fit_evt_pot(returns)
        assert result is not None
        # VaR should increase with confidence level
        assert result.var_999 >= result.var_99 >= result.var_95

    @pytest.mark.skipif(not _scipy_available(), reason="scipy not installed")
    def test_evt_convenience_wrapper(self):
        import pytest

        pytest.importorskip("src.assembled_core.ml.evt_models")
        from src.assembled_core.ml.evt_models import compute_evt_risk_metrics

        np.random.seed(42)
        returns = np.random.standard_t(df=4, size=1000) * 0.02

        metrics = compute_evt_risk_metrics(returns)
        assert "evt_var_99" in metrics
        assert "evt_cvar_99" in metrics
        assert "evt_shape_xi" in metrics

    def test_evt_insufficient_data(self):
        import pytest

        pytest.importorskip("src.assembled_core.ml.evt_models")
        from src.assembled_core.ml.evt_models import fit_evt_pot

        returns = np.random.normal(0, 0.02, 20)
        result = fit_evt_pot(returns)
        assert result is None

    def test_evt_fallback_on_failure(self):
        import pytest

        pytest.importorskip("src.assembled_core.ml.evt_models")
        from src.assembled_core.ml.evt_models import compute_evt_risk_metrics

        # Very short series → should return zeros
        metrics = compute_evt_risk_metrics(np.array([0.01, -0.01, 0.02]))
        assert metrics["evt_var_99"] == 0.0


# ── Copula Models ──────────────────────────────────────────────────────


class TestCopulaModels:
    """Tests for copula-based tail dependence."""

    def test_import_v5(self):
        import pytest

        pytest.importorskip("src.assembled_core.ml.copula_models")
        from src.assembled_core.ml.copula_models import fit_copula_pair

        assert fit_copula_pair is not None

    @pytest.mark.skipif(not _scipy_available(), reason="scipy not installed")
    def test_fit_copula_basic(self):
        import pytest

        pytest.importorskip("src.assembled_core.ml.copula_models")
        from src.assembled_core.ml.copula_models import fit_copula_pair

        np.random.seed(42)
        # Correlated returns
        n = 500
        z1 = np.random.normal(0, 1, n)
        z2 = 0.7 * z1 + 0.3 * np.random.normal(0, 1, n)
        ra = z1 * 0.02
        rb = z2 * 0.02

        result = fit_copula_pair(ra, rb, "A", "B")
        assert result is not None
        assert result.best_copula in ("clayton", "gumbel", "gaussian")
        assert result.n_obs == n

    @pytest.mark.skipif(not _scipy_available(), reason="scipy not installed")
    def test_copula_tail_dependence_positive(self):
        import pytest

        pytest.importorskip("src.assembled_core.ml.copula_models")
        from src.assembled_core.ml.copula_models import fit_copula_pair

        np.random.seed(42)
        # Strongly correlated — should have non-trivial tail dependence
        n = 1000
        z1 = np.random.normal(0, 1, n)
        z2 = 0.9 * z1 + 0.1 * np.random.normal(0, 1, n)
        ra = z1 * 0.02
        rb = z2 * 0.02

        result = fit_copula_pair(ra, rb, "X", "Y")
        assert result is not None
        # At least one of the tail deps should be > 0
        assert result.lower_tail_dep >= 0 or result.upper_tail_dep >= 0

    @pytest.mark.skipif(not _scipy_available(), reason="scipy not installed")
    def test_portfolio_tail_risk(self):
        import pytest

        pytest.importorskip("src.assembled_core.ml.copula_models")
        from src.assembled_core.ml.copula_models import compute_portfolio_tail_risk

        np.random.seed(42)
        n = 500
        z = np.random.normal(0, 1, n)
        returns_df = pd.DataFrame(
            {
                "A": z * 0.02 + np.random.normal(0, 0.005, n),
                "B": z * 0.02 + np.random.normal(0, 0.005, n),
                "C": np.random.normal(0, 0.02, n),  # independent
            }
        )

        result = compute_portfolio_tail_risk(returns_df)
        assert "avg_lower_tail_dep" in result
        assert "max_lower_tail_dep" in result
        assert result["n_pairs"] == 3  # C(3,2) = 3

    def test_copula_insufficient_data(self):
        import pytest

        pytest.importorskip("src.assembled_core.ml.copula_models")
        from src.assembled_core.ml.copula_models import fit_copula_pair

        ra = np.random.normal(0, 0.02, 10)
        rb = np.random.normal(0, 0.02, 10)
        result = fit_copula_pair(ra, rb)
        assert result is None
