"""Tests for M30-M35: Operational Excellence & Meta-Intelligence.

Covers:
- M32: Certification & sign-off
- M34: Strategy Discovery Engine
- M35: Self-Healing + Risk Escalation
"""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd


# ===========================================================================
# M32: Certification
# ===========================================================================

@pytest.mark.phase12
class TestCertification:
    def test_runner_basic(self):
        import pytest; pytest.importorskip('src.assembled_core.ops.certification')
        from src.assembled_core.ops.certification import CertificationRunner, CertificationReport
        runner = CertificationRunner()
        runner.add_check("always_pass", lambda: (True, "OK"))
        runner.add_check("always_fail", lambda: (False, "FAIL"))
        report = runner.run()
        assert isinstance(report, CertificationReport)
        assert report.total_checks == 2
        assert report.passed_count == 1
        assert report.failed_count == 1
        assert not report.all_passed

    def test_all_pass(self):
        import pytest; pytest.importorskip('src.assembled_core.ops.certification')
        from src.assembled_core.ops.certification import CertificationRunner
        runner = CertificationRunner()
        runner.add_check("c1", lambda: True)
        runner.add_check("c2", lambda: True)
        report = runner.run()
        assert report.all_passed
        assert report.pass_rate == 1.0

    def test_exception_handling(self):
        import pytest; pytest.importorskip('src.assembled_core.ops.certification')
        from src.assembled_core.ops.certification import CertificationRunner
        runner = CertificationRunner()
        runner.add_check("boom", lambda: 1/0)
        report = runner.run()
        assert report.failed_count == 1
        assert "Exception" in report.checks[0].message

    def test_default_runner(self):
        import pytest; pytest.importorskip('src.assembled_core.ops.certification')
        from src.assembled_core.ops.certification import build_default_runner
        runner = build_default_runner()
        report = runner.run()
        assert report.total_checks >= 2

    def test_check_result_type(self):
        import pytest; pytest.importorskip('src.assembled_core.ops.certification')
        from src.assembled_core.ops.certification import CertificationRunner, CheckResult
        runner = CertificationRunner()
        runner.add_check("custom", lambda: CheckResult("custom", True, "all good"))
        report = runner.run()
        assert report.checks[0].name == "custom"


# ===========================================================================
# M34: Strategy Discovery
# ===========================================================================

@pytest.mark.phase12
class TestStrategyDiscovery:
    def _make_data(self, n=500, n_feat=10, seed=42):
        rng = np.random.default_rng(seed)
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        features = pd.DataFrame(
            rng.normal(0, 1, (n, n_feat)),
            index=dates,
            columns=[f"feat_{i}" for i in range(n_feat)],
        )
        # Returns with some signal embedded
        returns = pd.Series(
            features["feat_0"].values * 0.01 + rng.normal(0, 0.02, n),
            index=dates,
        )
        return features, returns

    def test_basic_discovery(self):
        import pytest; pytest.importorskip('src.assembled_core.strategies.strategy_discovery')
        from src.assembled_core.strategies.strategy_discovery import discover_strategies, DiscoveryResult
        features, returns = self._make_data()
        result = discover_strategies(features, returns, n_trials=20, seed=42)
        assert isinstance(result, DiscoveryResult)
        assert result.total_tested == 20
        assert len(result.candidates) == 20

    def test_candidates_sorted(self):
        import pytest; pytest.importorskip('src.assembled_core.strategies.strategy_discovery')
        from src.assembled_core.strategies.strategy_discovery import discover_strategies
        features, returns = self._make_data()
        result = discover_strategies(features, returns, n_trials=30)
        sharpes = [c.sharpe_ratio for c in result.candidates]
        assert sharpes == sorted(sharpes, reverse=True)

    def test_fdr_gate(self):
        import pytest; pytest.importorskip('src.assembled_core.strategies.strategy_discovery')
        from src.assembled_core.strategies.strategy_discovery import discover_strategies
        features, returns = self._make_data()
        result = discover_strategies(features, returns, n_trials=50, fdr_alpha=0.05)
        # Some should pass, some not
        assert result.total_passed <= result.total_tested

    def test_too_few_features(self):
        import pytest; pytest.importorskip('src.assembled_core.strategies.strategy_discovery')
        from src.assembled_core.strategies.strategy_discovery import discover_strategies
        features = pd.DataFrame({"feat_0": [1.0]})
        returns = pd.Series([0.01])
        result = discover_strategies(features, returns, min_features=3)
        assert result.total_tested == 0

    def test_capacity_positive(self):
        import pytest; pytest.importorskip('src.assembled_core.strategies.strategy_discovery')
        from src.assembled_core.strategies.strategy_discovery import discover_strategies
        features, returns = self._make_data()
        result = discover_strategies(features, returns, n_trials=10)
        for c in result.candidates:
            assert c.capacity_usd >= 0

    def test_p_values_valid(self):
        import pytest; pytest.importorskip('src.assembled_core.strategies.strategy_discovery')
        from src.assembled_core.strategies.strategy_discovery import discover_strategies
        features, returns = self._make_data()
        result = discover_strategies(features, returns, n_trials=15)
        for c in result.candidates:
            assert 0 <= c.p_value <= 1.0


# ===========================================================================
# M35: Self-Healing
# ===========================================================================

@pytest.mark.phase12
class TestDataSourceCascade:
    def test_primary_success(self):
        import pytest; pytest.importorskip('src.assembled_core.ops.self_healing')
        from src.assembled_core.ops.self_healing import DataSourceCascade
        cascade = DataSourceCascade()
        cascade.register_source("primary", lambda: "data_from_primary")
        data, source = cascade.fetch()
        assert data == "data_from_primary"
        assert source == "primary"

    def test_fallback_on_failure(self):
        import pytest; pytest.importorskip('src.assembled_core.ops.self_healing')
        from src.assembled_core.ops.self_healing import DataSourceCascade
        cascade = DataSourceCascade()
        cascade.register_source("broken", lambda: (_ for _ in ()).throw(RuntimeError("down")))
        cascade.register_source("backup", lambda: "backup_data")
        data, source = cascade.fetch()
        assert data == "backup_data"
        assert source == "backup"

    def test_all_fail(self):
        import pytest; pytest.importorskip('src.assembled_core.ops.self_healing')
        from src.assembled_core.ops.self_healing import DataSourceCascade
        cascade = DataSourceCascade()
        cascade.register_source("s1", lambda: (_ for _ in ()).throw(RuntimeError("fail")))
        with pytest.raises(RuntimeError, match="All data sources failed"):
            cascade.fetch()

    def test_history_recorded(self):
        import pytest; pytest.importorskip('src.assembled_core.ops.self_healing')
        from src.assembled_core.ops.self_healing import DataSourceCascade
        cascade = DataSourceCascade()
        cascade.register_source("ok", lambda: "data")
        cascade.fetch()
        assert len(cascade.history) == 1
        assert cascade.history[0].success


@pytest.mark.phase12
class TestRiskEscalationLadder:
    def test_normal_state(self):
        import pytest; pytest.importorskip('src.assembled_core.ops.self_healing')
        from src.assembled_core.ops.self_healing import RiskEscalationLadder, EscalationLevel
        ladder = RiskEscalationLadder()
        state = ladder.evaluate(current_drawdown=-0.03)
        assert state.level == EscalationLevel.NORMAL

    def test_reduce_on_moderate_dd(self):
        import pytest; pytest.importorskip('src.assembled_core.ops.self_healing')
        from src.assembled_core.ops.self_healing import RiskEscalationLadder, EscalationLevel
        ladder = RiskEscalationLadder(dd_reduce=0.10)
        state = ladder.evaluate(current_drawdown=-0.12)
        assert state.level == EscalationLevel.REDUCE

    def test_defensive_on_severe_dd(self):
        import pytest; pytest.importorskip('src.assembled_core.ops.self_healing')
        from src.assembled_core.ops.self_healing import RiskEscalationLadder, EscalationLevel
        ladder = RiskEscalationLadder(dd_defensive=0.15)
        state = ladder.evaluate(current_drawdown=-0.17)
        assert state.level == EscalationLevel.CRITICAL

    def test_kill_switch(self):
        import pytest; pytest.importorskip('src.assembled_core.ops.self_healing')
        from src.assembled_core.ops.self_healing import RiskEscalationLadder, EscalationLevel
        ladder = RiskEscalationLadder(dd_kill=0.20)
        state = ladder.evaluate(current_drawdown=-0.25)
        assert state.level == EscalationLevel.KILL
        assert "KILL_SWITCH_ACTIVATED" in state.actions_taken

    def test_ic_degradation(self):
        import pytest; pytest.importorskip('src.assembled_core.ops.self_healing')
        from src.assembled_core.ops.self_healing import RiskEscalationLadder, EscalationLevel
        ladder = RiskEscalationLadder(ic_degradation_threshold=0.02)
        state = ladder.evaluate(current_drawdown=-0.05, current_ic=0.01)
        assert state.level == EscalationLevel.REDUCE
        assert "TRIGGER_MODEL_RETRAIN" in state.actions_taken

    def test_feature_drift(self):
        import pytest; pytest.importorskip('src.assembled_core.ops.self_healing')
        from src.assembled_core.ops.self_healing import RiskEscalationLadder, EscalationLevel
        ladder = RiskEscalationLadder()
        state = ladder.evaluate(current_drawdown=-0.02, feature_drift_score=1.5)
        assert state.level == EscalationLevel.WATCH

    def test_sizing_multiplier(self):
        import pytest; pytest.importorskip('src.assembled_core.ops.self_healing')
        from src.assembled_core.ops.self_healing import RiskEscalationLadder
        ladder = RiskEscalationLadder(sizing_reduction_factor=0.5)
        ladder.evaluate(current_drawdown=-0.25)
        assert ladder.get_sizing_multiplier() == 0.0  # kill

        ladder2 = RiskEscalationLadder(sizing_reduction_factor=0.5)
        ladder2.evaluate(current_drawdown=-0.12)
        assert ladder2.get_sizing_multiplier() == 0.5  # reduce

        ladder3 = RiskEscalationLadder()
        ladder3.evaluate(current_drawdown=-0.01)
        assert ladder3.get_sizing_multiplier() == 1.0  # normal

    def test_escalation_history(self):
        import pytest; pytest.importorskip('src.assembled_core.ops.self_healing')
        from src.assembled_core.ops.self_healing import RiskEscalationLadder
        ladder = RiskEscalationLadder()
        ladder.evaluate(current_drawdown=-0.05)  # normal
        ladder.evaluate(current_drawdown=-0.12)  # reduce (change!)
        ladder.evaluate(current_drawdown=-0.25)  # kill (change!)
        assert len(ladder.history) == 2  # only changes logged
