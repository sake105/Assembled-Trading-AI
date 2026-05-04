"""Tests for M17 Wave 1 features: correlation, supply chain, GPR, WI, scenarios, HRP, barbell, MC VaR, Brinson."""

import numpy as np
import pandas as pd
import pytest

# ── Correlation Features ──────────────────────────────────────────────


class TestCorrelationFeatures:

    def test_import(self):
        from src.assembled_core.features.correlation_features import (
            compute_avg_pairwise_correlation,
        )

        assert compute_avg_pairwise_correlation is not None

    def test_avg_pairwise_correlation(self):
        from src.assembled_core.features.correlation_features import (
            compute_avg_pairwise_correlation,
        )

        np.random.seed(42)
        n = 100
        z = np.random.normal(0, 1, n)
        returns = pd.DataFrame(
            {
                "A": z * 0.02 + np.random.normal(0, 0.005, n),
                "B": z * 0.02 + np.random.normal(0, 0.005, n),
                "C": np.random.normal(0, 0.02, n),
            },
            index=pd.date_range("2020-01-01", periods=n, freq="B"),
        )

        result = compute_avg_pairwise_correlation(returns, windows=(20,))
        assert "avg_pairwise_corr_20d" in result.columns
        # Correlated pair A/B should push average above 0
        last_val = result["avg_pairwise_corr_20d"].dropna().iloc[-1]
        assert last_val > 0

    def test_return_dispersion(self):
        from src.assembled_core.features.correlation_features import (
            compute_return_dispersion,
        )

        returns = pd.DataFrame(
            {
                "A": [0.01, -0.02, 0.03],
                "B": [0.01, -0.02, 0.03],  # same as A
                "C": [-0.01, 0.02, -0.03],  # opposite
            }
        )

        disp = compute_return_dispersion(returns)
        assert len(disp) == 3
        # Row 0: A=0.01, B=0.01, C=-0.01 → some dispersion
        assert disp.iloc[0] > 0

    def test_correlation_regime_features(self):
        from src.assembled_core.features.correlation_features import (
            compute_correlation_regime_features,
        )

        np.random.seed(42)
        n = 300
        returns = pd.DataFrame(
            {
                "A": np.random.normal(0, 0.02, n),
                "B": np.random.normal(0, 0.02, n),
                "C": np.random.normal(0, 0.02, n),
            },
            index=pd.date_range("2020-01-01", periods=n, freq="B"),
        )

        result = compute_correlation_regime_features(returns)
        assert "avg_corr_short" in result.columns
        assert "corr_regime_zscore" in result.columns
        assert "corr_momentum" in result.columns

    def test_sector_dispersion(self):
        from src.assembled_core.features.correlation_features import (
            compute_sector_dispersion,
        )

        np.random.seed(42)
        returns = pd.DataFrame(
            {
                "AAPL": np.random.normal(0, 0.02, 50),
                "MSFT": np.random.normal(0, 0.02, 50),
                "XOM": np.random.normal(0, 0.03, 50),
                "CVX": np.random.normal(0, 0.03, 50),
            },
            index=pd.date_range("2020-01-01", periods=50, freq="B"),
        )

        sector_map = {"AAPL": "Tech", "MSFT": "Tech", "XOM": "Energy", "CVX": "Energy"}
        result = compute_sector_dispersion(returns, sector_map)
        assert "intra_sector_dispersion" in result.columns
        assert "inter_sector_dispersion" in result.columns


# ── Supply Chain Features ─────────────────────────────────────────────


class TestSupplyChainFeatures:

    def test_import_v2(self):
        from src.assembled_core.features.supply_chain_features import (
            build_supply_chain_features,
        )

        assert build_supply_chain_features is not None

    def test_supply_chain_depth(self):
        from src.assembled_core.features.supply_chain_features import (
            compute_supply_chain_depth,
        )

        edges = [("A", "B", 1.0), ("B", "C", 1.0), ("C", "D", 1.0)]
        result = compute_supply_chain_depth(edges, ["A", "B", "C", "D"])
        assert result["A"] == 3  # A→B→C→D
        assert result["D"] == 0  # leaf

    def test_single_source_dependency(self):
        from src.assembled_core.features.supply_chain_features import (
            compute_single_source_dependency,
        )

        edges = [("S1", "TARGET", 0.9), ("S2", "TARGET", 0.1)]
        result = compute_single_source_dependency(edges, ["TARGET"])
        assert result["TARGET"] == pytest.approx(0.9, abs=0.01)

    def test_network_centrality(self):
        from src.assembled_core.features.supply_chain_features import (
            compute_network_centrality,
        )

        edges = [("A", "B", 1.0), ("B", "C", 1.0), ("A", "C", 1.0)]
        result = compute_network_centrality(edges, ["A", "B", "C"])
        # All connected — centralities should be > 0
        assert all(v > 0 for v in result.values())

    def test_build_supply_chain_features(self):
        from src.assembled_core.features.supply_chain_features import (
            build_supply_chain_features,
        )

        edges = [("A", "B", 1.0), ("B", "C", 1.0)]
        result = build_supply_chain_features(
            ["A", "B", "C"],
            dependency_edges=edges,
        )
        assert "supply_chain_depth" in result.columns
        assert "single_source_dep" in result.columns
        assert len(result) == 3


# ── GPR Features ──────────────────────────────────────────────────────


class TestGPRFeatures:

    def test_import_v3(self):
        from src.assembled_core.features.geopolitical_features import compute_gpr_proxy

        assert compute_gpr_proxy is not None

    def test_gpr_proxy_basic(self):
        from src.assembled_core.features.geopolitical_features import compute_gpr_proxy

        np.random.seed(42)
        dates = pd.date_range("2018-01-01", periods=500, freq="B")
        events = pd.Series(np.random.poisson(10, 500), index=dates, dtype=float)
        vix = pd.Series(20 + np.random.normal(0, 3, 500), index=dates)

        result = compute_gpr_proxy(gdelt_event_counts=events, vix_series=vix)
        assert "gpr_level" in result.columns
        assert "gpr_zscore" in result.columns
        assert "gpr_momentum" in result.columns

    def test_gpr_from_fred(self):
        from src.assembled_core.features.geopolitical_features import (
            compute_gpr_from_fred,
        )

        dates = pd.date_range("2010-01-01", periods=500, freq="ME")
        gpr = pd.Series(100 + np.random.normal(0, 20, 500), index=dates)

        result = compute_gpr_from_fred(gpr)
        assert "gpr_level" in result.columns

    def test_empty_returns_empty(self):
        from src.assembled_core.features.geopolitical_features import compute_gpr_proxy

        result = compute_gpr_proxy()
        assert result.empty


# ── Weaponized Interdependence ────────────────────────────────────────


class TestWeaponizedInterdependence:

    def test_import_v4(self):
        import pytest

        pytest.importorskip("src.assembled_core.intel.weaponized_interdependence")
        from src.assembled_core.intel.weaponized_interdependence import (
            compute_wi_score,
        )

        assert compute_wi_score is not None

    def test_wi_score_basic(self):
        import pytest

        pytest.importorskip("src.assembled_core.intel.weaponized_interdependence")
        from src.assembled_core.intel.weaponized_interdependence import compute_wi_score

        result = compute_wi_score(
            trade_volume_a_to_b=100,
            total_imports_a=1000,
            market_share_b_in_a=0.9,
            substitutability=0.1,
            centrality_b=0.8,
        )
        assert result.vulnerability > result.sensitivity
        assert result.wi_score > 1.0  # asymmetric dependence
        assert result.is_chokepoint

    def test_known_wi_pairs(self):
        import pytest

        pytest.importorskip("src.assembled_core.intel.weaponized_interdependence")
        from src.assembled_core.intel.weaponized_interdependence import (
            get_known_wi_pairs,
        )

        pairs = get_known_wi_pairs()
        assert len(pairs) >= 5
        domains = [p["domain"] for p in pairs]
        assert "semiconductors" in domains
        assert "dollar_system" in domains

    def test_panoptikon_scores(self):
        import pytest

        pytest.importorskip("src.assembled_core.intel.weaponized_interdependence")
        from src.assembled_core.intel.weaponized_interdependence import (
            compute_panoptikon_scores,
        )

        adjacency = {
            "US": {"CN": 5.0, "EU": 8.0, "JP": 3.0},
            "CN": {"US": 4.0, "EU": 3.0},
            "EU": {"US": 6.0, "CN": 2.0},
            "JP": {"US": 2.0},
        }
        result = compute_panoptikon_scores(adjacency)
        assert len(result) >= 3
        # US should have highest centrality (most connections)
        assert result[0].node == "US" or result[0].betweenness_centrality > 0

    def test_symbol_wi_exposure(self):
        import pytest

        pytest.importorskip("src.assembled_core.intel.weaponized_interdependence")
        from src.assembled_core.intel.weaponized_interdependence import (
            score_symbol_wi_exposure,
        )

        result = score_symbol_wi_exposure(
            "NVDA",
            {"NVDA": "Semiconductors"},
            {"NVDA": ["US", "TW", "CN"]},
        )
        assert result["wi_semiconductor_risk"] > 0
        assert result["wi_exposure"] > 0


# ── Scenario Trees ────────────────────────────────────────────────────


class TestScenarioTrees:

    def test_import_v5(self):
        import pytest

        pytest.importorskip("src.assembled_core.intel.scenario_trees")
        from src.assembled_core.intel.scenario_trees import build_scenario_tree

        assert build_scenario_tree is not None

    def test_build_basic_tree(self):
        import pytest

        pytest.importorskip("src.assembled_core.intel.scenario_trees")
        from src.assembled_core.intel.scenario_trees import build_scenario_tree

        tree = build_scenario_tree("SANCTIONS_NEW", "New sanctions on country X")
        assert len(tree.scenarios) == 4
        assert tree.is_valid  # probabilities sum to ~1
        assert tree.expected_impact < 0  # negative = loss
        assert tree.tail_impact < tree.expected_impact  # tail is worse

    def test_impact_skew(self):
        import pytest

        pytest.importorskip("src.assembled_core.intel.scenario_trees")
        from src.assembled_core.intel.scenario_trees import build_scenario_tree

        tree = build_scenario_tree("NUCLEAR_THREAT")
        assert tree.impact_skew > 1.0  # tail is much worse than expected

    def test_custom_escalation_probability(self):
        import pytest

        pytest.importorskip("src.assembled_core.intel.scenario_trees")
        from src.assembled_core.intel.scenario_trees import build_scenario_tree

        tree_low = build_scenario_tree("TRADE_WAR", escalation_probability=0.1)
        tree_high = build_scenario_tree("TRADE_WAR", escalation_probability=0.7)

        # Higher escalation probability → worse expected impact
        assert tree_high.expected_impact < tree_low.expected_impact

    def test_portfolio_scenario_risk(self):
        import pytest

        pytest.importorskip("src.assembled_core.intel.scenario_trees")
        from src.assembled_core.intel.scenario_trees import (
            build_scenario_tree,
            evaluate_portfolio_scenario_risk,
        )

        trees = [
            build_scenario_tree("SANCTIONS_NEW"),
            build_scenario_tree("TRADE_WAR"),
        ]

        result = evaluate_portfolio_scenario_risk(trees, portfolio_exposure=0.8)
        assert "total_expected_impact" in result
        assert "within_budget" in result
        assert result["total_expected_impact"] < 0


# ── HRP ───────────────────────────────────────────────────────────────


def _scipy_available():
    try:
        import scipy  # noqa: F401

        return True
    except ImportError:
        return False


class TestHRP:

    def test_import_v6(self):
        from src.assembled_core.portfolio.hierarchical_risk_parity import (
            compute_hrp_weights,
        )

        assert compute_hrp_weights is not None

    @pytest.mark.skipif(not _scipy_available(), reason="scipy not installed")
    def test_hrp_basic(self):
        from src.assembled_core.portfolio.hierarchical_risk_parity import (
            compute_hrp_weights,
        )

        np.random.seed(42)
        returns = pd.DataFrame(
            {
                "A": np.random.normal(0, 0.02, 200),
                "B": np.random.normal(0, 0.03, 200),
                "C": np.random.normal(0, 0.01, 200),
            },
            index=pd.date_range("2020-01-01", periods=200, freq="B"),
        )

        weights = compute_hrp_weights(returns)
        assert len(weights) == 3
        assert abs(sum(weights.values()) - 1.0) < 0.01
        # Lower vol asset C should get higher weight (inverse variance)
        assert weights["C"] > weights["B"]

    @pytest.mark.skipif(not _scipy_available(), reason="scipy not installed")
    def test_hrp_weight_constraints(self):
        from src.assembled_core.portfolio.hierarchical_risk_parity import (
            compute_hrp_weights,
        )

        np.random.seed(42)
        returns = pd.DataFrame(
            {
                "A": np.random.normal(0, 0.02, 200),
                "B": np.random.normal(0, 0.03, 200),
            },
            index=pd.date_range("2020-01-01", periods=200, freq="B"),
        )

        weights = compute_hrp_weights(returns, max_weight=0.8)
        assert all(w <= 0.8 + 0.01 for w in weights.values())

    @pytest.mark.skipif(not _scipy_available(), reason="scipy not installed")
    def test_hrp_single_asset(self):
        from src.assembled_core.portfolio.hierarchical_risk_parity import (
            compute_hrp_weights,
        )

        returns = pd.DataFrame({"A": np.random.normal(0, 0.02, 100)})
        weights = compute_hrp_weights(returns)
        assert weights == {"A": 1.0}


# ── Barbell Strategy ──────────────────────────────────────────────────


class TestBarbellStrategy:

    def test_import_v7(self):
        import pytest

        pytest.importorskip("src.assembled_core.portfolio.barbell_strategy")
        from src.assembled_core.portfolio.barbell_strategy import (
            compute_tail_risk_score,
        )

        assert compute_tail_risk_score is not None

    def test_tail_risk_score_low(self):
        import pytest

        pytest.importorskip("src.assembled_core.portfolio.barbell_strategy")
        from src.assembled_core.portfolio.barbell_strategy import (
            compute_tail_risk_score,
        )

        score, reasons = compute_tail_risk_score(
            vix_current=15.0,
            hmm_crisis_prob=0.05,
        )
        assert score < 0.3
        assert len(reasons) == 0

    def test_tail_risk_score_high(self):
        import pytest

        pytest.importorskip("src.assembled_core.portfolio.barbell_strategy")
        from src.assembled_core.portfolio.barbell_strategy import (
            compute_tail_risk_score,
        )

        score, reasons = compute_tail_risk_score(
            evt_var_99=0.08,
            evt_var_99_historical_avg=0.03,
            hmm_crisis_prob=0.6,
            vix_current=40.0,
            vix_5d_change=10.0,
            avg_copula_tail_dep=0.7,
        )
        assert score > 0.5
        assert len(reasons) >= 3

    def test_barbell_activation(self):
        import pytest

        pytest.importorskip("src.assembled_core.portfolio.barbell_strategy")
        from src.assembled_core.portfolio.barbell_strategy import (
            build_barbell_allocation,
        )

        result = build_barbell_allocation(
            tail_risk_score=0.7,
            trigger_reasons=["HMM crisis", "VIX spike"],
            alpha_scores={"AAPL": 0.8, "NVDA": 0.6, "MSFT": 0.4},
        )
        assert result.active
        assert result.safe_weight > 0.7
        assert len(result.speculative_symbols) > 0
        assert len(result.safe_symbols) > 0

    def test_barbell_not_triggered(self):
        import pytest

        pytest.importorskip("src.assembled_core.portfolio.barbell_strategy")
        from src.assembled_core.portfolio.barbell_strategy import (
            build_barbell_allocation,
        )

        result = build_barbell_allocation(
            tail_risk_score=0.1,
            trigger_reasons=[],
            alpha_scores={"AAPL": 0.8},
        )
        assert not result.active


# ── Monte Carlo VaR ───────────────────────────────────────────────────


class TestMonteCarloVaR:

    def test_import_v8(self):
        from src.assembled_core.risk.risk_metrics import compute_monte_carlo_var

        assert compute_monte_carlo_var is not None

    def test_mc_var_basic(self):
        from src.assembled_core.risk.risk_metrics import compute_monte_carlo_var

        np.random.seed(42)
        returns = pd.DataFrame(
            {
                "A": np.random.normal(0, 0.02, 500),
                "B": np.random.normal(0, 0.03, 500),
            }
        )

        result = compute_monte_carlo_var(returns)
        assert "mc_var_99" in result
        assert "mc_cvar_99" in result
        assert result["mc_var_99"] > 0
        assert result["mc_cvar_99"] >= result["mc_var_99"]

    def test_mc_var_with_weights(self):
        from src.assembled_core.risk.risk_metrics import compute_monte_carlo_var

        np.random.seed(42)
        returns = pd.DataFrame(
            {
                "A": np.random.normal(0, 0.01, 500),
                "B": np.random.normal(0, 0.04, 500),
            }
        )

        # All weight in low-vol asset → lower VaR
        var_safe = compute_monte_carlo_var(returns, weights=np.array([1.0, 0.0]))
        var_risky = compute_monte_carlo_var(returns, weights=np.array([0.0, 1.0]))
        assert var_safe["mc_var_99"] < var_risky["mc_var_99"]

    def test_mc_var_deterministic(self):
        from src.assembled_core.risk.risk_metrics import compute_monte_carlo_var

        np.random.seed(42)
        returns = pd.DataFrame({"A": np.random.normal(0, 0.02, 200)})

        r1 = compute_monte_carlo_var(returns, seed=42)
        r2 = compute_monte_carlo_var(returns, seed=42)
        assert r1["mc_var_99"] == r2["mc_var_99"]


# ── Brinson-Fachler Attribution ───────────────────────────────────────


class TestBrinsonFachler:

    def test_import_v9(self):
        from src.assembled_core.risk.risk_metrics import (
            compute_brinson_fachler_attribution,
        )

        assert compute_brinson_fachler_attribution is not None

    def test_basic_attribution(self):
        from src.assembled_core.risk.risk_metrics import (
            compute_brinson_fachler_attribution,
        )

        port_w = {"AAPL": 0.3, "MSFT": 0.2, "XOM": 0.3, "CVX": 0.2}
        bench_w = {"AAPL": 0.25, "MSFT": 0.25, "XOM": 0.25, "CVX": 0.25}
        port_r = {"AAPL": 0.05, "MSFT": 0.03, "XOM": -0.02, "CVX": -0.01}
        bench_r = {"AAPL": 0.04, "MSFT": 0.02, "XOM": -0.03, "CVX": -0.02}
        sectors = {"AAPL": "Tech", "MSFT": "Tech", "XOM": "Energy", "CVX": "Energy"}

        result = compute_brinson_fachler_attribution(
            port_w,
            bench_w,
            port_r,
            bench_r,
            sectors,
        )
        assert "sector" in result.columns
        assert "allocation_effect" in result.columns
        assert "selection_effect" in result.columns
        assert "TOTAL" in result["sector"].values

    def test_attribution_sums(self):
        from src.assembled_core.risk.risk_metrics import (
            compute_brinson_fachler_attribution,
        )

        port_w = {"A": 0.6, "B": 0.4}
        bench_w = {"A": 0.5, "B": 0.5}
        port_r = {"A": 0.10, "B": -0.05}
        bench_r = {"A": 0.08, "B": -0.03}
        sectors = {"A": "S1", "B": "S2"}

        result = compute_brinson_fachler_attribution(
            port_w,
            bench_w,
            port_r,
            bench_r,
            sectors,
        )
        total = result[result["sector"] == "TOTAL"].iloc[0]
        # Total effect should be close to active return
        port_total = sum(port_w[s] * port_r[s] for s in port_w)
        bench_total = sum(bench_w[s] * bench_r[s] for s in bench_w)
        active = port_total - bench_total
        assert abs(total["total_effect"] - active) < 0.001


# ── Volatility Features (GARCH) ──────────────────────────────────────


class TestVolatilityFeatures:

    def test_import_v10(self):
        import pytest

        pytest.importorskip("src.assembled_core.features.volatility_features")
        from src.assembled_core.features.volatility_features import (
            compute_garch_features,
        )

        assert compute_garch_features is not None

    def test_snapshot_no_arch(self):
        """Without arch package, should return empty dict."""
        import pytest

        pytest.importorskip("src.assembled_core.features.volatility_features")
        from src.assembled_core.features.volatility_features import ARCH_AVAILABLE

        if ARCH_AVAILABLE:
            pytest.skip("arch is installed")
        from src.assembled_core.features.volatility_features import (
            compute_garch_features_snapshot,
        )

        result = compute_garch_features_snapshot(pd.DataFrame())
        assert result == {}


# ── IC Weights and Neutralization ─────────────────────────────────────


class TestICWeightsAndNeutralization:

    def test_neutralize_by_group(self):
        from src.assembled_core.signals.multifactor_signal import neutralize_by_group

        df = pd.DataFrame(
            {
                "timestamp": ["2020-01-01"] * 4,
                "symbol": ["A", "B", "C", "D"],
                "sector": ["Tech", "Tech", "Energy", "Energy"],
                "factor_x": [10, 20, 100, 200],
            }
        )

        result = neutralize_by_group(df, "factor_x", "sector")
        # Within Tech: A=10, B=20 → z-scores should be symmetric
        assert abs(result.iloc[0] + result.iloc[1]) < 0.01
        # Within Energy: C=100, D=200 → same
        assert abs(result.iloc[2] + result.iloc[3]) < 0.01

    def test_ic_weights_import(self):
        from src.assembled_core.signals.multifactor_signal import compute_ic_weights

        assert compute_ic_weights is not None
