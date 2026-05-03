"""Tests for the 6 previously-deferred items now implemented.

A1  ELSTER Anlage-KAP XML export
A2  Neo4j News Graph (memory-fallback)
A3  D-vine Copula (3-asset)
B1  Enhanced Synthetic Data Generator (GARCH / Jump-Diffusion / Regime-Switch)
B2  SVI Volatility Surface
B10 Adaptive Almgren-Chriss
"""
from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# B2: SVI Volatility Surface
# ---------------------------------------------------------------------------

class TestSVIVolSurface:
    def _params(self):
        from assembled_core.risk.vol_surface_svi import SVIParams
        return SVIParams(a=0.02, b=0.15, rho=-0.4, m=0.0, sigma=0.1, expiry_T=0.25)

    def test_total_variance_positive(self):
        from assembled_core.risk.vol_surface_svi import svi_total_variance
        k = np.linspace(-0.5, 0.5, 50)
        w = svi_total_variance(k, self._params())
        assert np.all(w > 0)

    def test_atm_implied_vol_reasonable(self):
        from assembled_core.risk.vol_surface_svi import svi_implied_vol
        iv = svi_implied_vol(np.array([0.0]), self._params())
        assert 0.05 < float(iv[0]) < 0.60

    def test_fit_recovers_params(self):
        from assembled_core.risk.vol_surface_svi import fit_svi, svi_total_variance
        pytest.importorskip("scipy")
        p = self._params()
        k = np.linspace(-0.3, 0.3, 30)
        w_obs = svi_total_variance(k, p)
        fitted = fit_svi(k, w_obs, p.expiry_T)
        assert fitted is not None
        assert fitted.fit_rmse < 1e-3

    def test_butterfly_arbitrage_free(self):
        from assembled_core.risk.vol_surface_svi import butterfly_arbitrage_free
        result = butterfly_arbitrage_free(self._params())
        assert result["arbitrage_free"] is True
        assert result["min_g"] > -1e-6

    def test_params_validity_check(self):
        from assembled_core.risk.vol_surface_svi import SVIParams
        assert self._params().is_valid()
        bad = SVIParams(a=-999.0, b=0.15, rho=-0.4, m=0.0, sigma=0.1, expiry_T=0.25)
        assert not bad.is_valid()

    def test_surface_summary_keys(self):
        from assembled_core.risk.vol_surface_svi import surface_summary
        s = surface_summary(self._params())
        for key in ("atm_iv", "skew_dw_dk", "put_wing_iv", "call_wing_iv"):
            assert key in s


# ---------------------------------------------------------------------------
# A1: ELSTER Anlage-KAP export
# ---------------------------------------------------------------------------

class TestElsterExport:
    def _summary(self):
        import datetime
        from assembled_core.compliance.tax_report import summarize_closed_lots
        lots = [
            {"realized_pnl_eur": 8000.0, "trade_date": datetime.date(2025, 4, 1)},
            {"realized_pnl_eur": -1500.0, "trade_date": datetime.date(2025, 9, 1)},
        ]
        return summarize_closed_lots(lots, year=2025)

    def test_xml_contains_required_tags(self):
        from assembled_core.compliance.elster import build_anlage_kap_xml, ElsterExportConfig
        cfg = ElsterExportConfig(tax_year=2025, steuerpflichtiger_id="12345678901")
        xml = build_anlage_kap_xml(self._summary(), cfg)
        for tag in ("AnlageKAP", "Kap_Z7_Veraeusserungsgewinne", "Kap_Z36_Abgeltungsteuer"):
            assert tag in xml

    def test_xml_is_well_formed(self):
        import xml.etree.ElementTree as ET
        from assembled_core.compliance.elster import build_anlage_kap_xml, ElsterExportConfig
        cfg = ElsterExportConfig(tax_year=2025, steuerpflichtiger_id="12345678901")
        xml = build_anlage_kap_xml(self._summary(), cfg)
        body = xml.replace('<?xml version="1.0" encoding="UTF-8"?>\n', "")
        root = ET.fromstring(body)
        assert root is not None

    def test_loss_year_generates_carry_forward_tag(self):
        import datetime
        from assembled_core.compliance.tax_report import summarize_closed_lots
        from assembled_core.compliance.elster import build_anlage_kap_xml, ElsterExportConfig
        lots = [{"realized_pnl_eur": -500.0, "trade_date": datetime.date(2025, 3, 1)}]
        summary = summarize_closed_lots(lots, year=2025)
        cfg = ElsterExportConfig(tax_year=2025, steuerpflichtiger_id="98765432100")
        xml = build_anlage_kap_xml(summary, cfg)
        assert "Kap_Z18_VerlustuebertragAktien" in xml

    def test_steuerpflichtiger_id_embedded(self):
        from assembled_core.compliance.elster import build_anlage_kap_xml, ElsterExportConfig
        cfg = ElsterExportConfig(tax_year=2025, steuerpflichtiger_id="11122233344")
        xml = build_anlage_kap_xml(self._summary(), cfg)
        assert "11122233344" in xml


# ---------------------------------------------------------------------------
# A2: Neo4j News Graph (always uses memory fallback in CI)
# ---------------------------------------------------------------------------

class TestNewsGraph:
    def _graph(self):
        from assembled_core.events.news.news_graph import NewsGraph
        return NewsGraph(bolt_uri=None)

    def _event(self, eid, ticker, entities):
        import datetime
        from assembled_core.events.news.news_graph import NewsNode
        return NewsNode(eid, f"Headline {eid}", "TestSrc",
                        datetime.datetime(2025, 1, 1), 0.5, ticker, entities)

    def test_add_and_query_entity_neighbors(self):
        g = self._graph()
        g.add_event(self._event("e1", "AAPL", ["Apple"]))
        g.add_event(self._event("e2", "MSFT", ["Apple", "Microsoft"]))
        assert set(g.entity_neighbors("Apple")) == {"e1", "e2"}

    def test_related_symbols_via_edge(self):
        g = self._graph()
        g.add_event(self._event("e1", "AAPL", []))
        g.add_event(self._event("e2", "TSMC", []))
        g.add_related("e1", "e2", 0.9)
        assert "TSMC" in g.find_related_symbols("AAPL")

    def test_stats_counts_correct(self):
        from assembled_core.events.news.news_graph import GraphStats
        g = self._graph()
        g.add_event(self._event("e1", "AAPL", ["Apple", "Tim Cook"]))
        s = g.stats()
        assert isinstance(s, GraphStats)
        assert s.n_events == 1
        assert s.n_entities == 2
        assert s.backend == "memory"

    def test_unknown_entity_returns_empty(self):
        g = self._graph()
        g.add_event(self._event("e1", "AAPL", ["Apple"]))
        assert g.entity_neighbors("Unknown Corp") == []

    def test_neo4j_available_is_bool(self):
        from assembled_core.events.news.news_graph import NEO4J_AVAILABLE
        assert isinstance(NEO4J_AVAILABLE, bool)


# ---------------------------------------------------------------------------
# B1: Enhanced Synthetic Data Generator
# ---------------------------------------------------------------------------

class TestSyntheticGenerator:
    def test_garch_shape(self):
        from assembled_core.data.synthetic_generator import generate_garch_returns
        df = generate_garch_returns(n_days=252, n_assets=4, seed=7)
        assert df.shape == (252, 4)

    def test_garch_vol_varies(self):
        from assembled_core.data.synthetic_generator import generate_garch_returns
        df = generate_garch_returns(n_days=252, n_assets=2, seed=7)
        rolling_std = df.rolling(20).std().dropna()
        assert rolling_std.std().mean() > 0

    def test_jump_diffusion_shape(self):
        from assembled_core.data.synthetic_generator import generate_jump_diffusion_returns
        df = generate_jump_diffusion_returns(n_days=252, n_assets=3, seed=9)
        assert df.shape == (252, 3)

    def test_jump_diffusion_fat_tails(self):
        pytest.importorskip("scipy")
        from scipy.stats import kurtosis
        from assembled_core.data.synthetic_generator import generate_jump_diffusion_returns
        df = generate_jump_diffusion_returns(
            n_days=1000, n_assets=1, jump_intensity=15.0, jump_std=0.05, seed=99
        )
        assert kurtosis(df.values.ravel()) > 1.5

    def test_regime_switching_shape(self):
        from assembled_core.data.synthetic_generator import generate_regime_switching_returns
        df, regime = generate_regime_switching_returns(n_days=504, n_assets=3, seed=5)
        assert df.shape == (504, 3)
        assert regime.shape == (504,)
        assert set(np.unique(regime)).issubset({0, 1})
        assert 0 in regime and 1 in regime

    def test_regime_bear_higher_vol(self):
        from assembled_core.data.synthetic_generator import generate_regime_switching_returns
        df, regime = generate_regime_switching_returns(
            n_days=2000, n_assets=1, seed=42,
            bull_vol_annual=0.10, bear_vol_annual=0.40,
        )
        bull_vol = df["ASSET_0"][regime == 0].std()
        bear_vol = df["ASSET_0"][regime == 1].std()
        assert bear_vol > bull_vol


# ---------------------------------------------------------------------------
# A3: D-vine Copula (3-asset)
# ---------------------------------------------------------------------------

class TestDVineCopula:
    def _trio(self, seed=42, n=300):
        rng = np.random.default_rng(seed)
        z = rng.normal(0, 0.01, (n, 3))
        z[:, 1] += 0.3 * z[:, 0]
        z[:, 2] += 0.3 * z[:, 1]
        return z[:, 0], z[:, 1], z[:, 2]

    def test_returns_dvine_result(self):
        pytest.importorskip("scipy")
        from assembled_core.ml.copula_models import fit_dvine_trio, DVineResult
        result = fit_dvine_trio(*self._trio(), "A", "B", "C")
        assert isinstance(result, DVineResult)
        assert result.n_obs == 300
        assert result.symbols == ("A", "B", "C")

    def test_copula_names_valid(self):
        pytest.importorskip("scipy")
        from assembled_core.ml.copula_models import fit_dvine_trio
        result = fit_dvine_trio(*self._trio())
        valid = {"clayton", "gumbel", "gaussian"}
        assert result.copula_12 in valid
        assert result.copula_23 in valid
        assert result.copula_13_2 in valid

    def test_loglik_finite(self):
        pytest.importorskip("scipy")
        from assembled_core.ml.copula_models import fit_dvine_trio
        result = fit_dvine_trio(*self._trio())
        assert np.isfinite(result.log_likelihood)

    def test_too_few_obs_returns_none(self):
        pytest.importorskip("scipy")
        from assembled_core.ml.copula_models import fit_dvine_trio
        tiny = np.random.default_rng(0).normal(0, 0.01, 10)
        assert fit_dvine_trio(tiny, tiny, tiny) is None


# ---------------------------------------------------------------------------
# B10: Adaptive Almgren-Chriss
# ---------------------------------------------------------------------------

class TestAdaptiveAC:
    def _setup(self):
        from assembled_core.execution.execution_router import ExecutionConfig, AdaptiveACState
        cfg = ExecutionConfig(twap_slices=5, almgren_eta=0.1,
                              almgren_gamma=0.05, almgren_lambda=1e-6)
        state = AdaptiveACState.from_config(cfg, ewma_alpha=0.15)
        return cfg, state

    def test_initial_eta_from_config(self):
        cfg, state = self._setup()
        assert state.eta_hat == pytest.approx(cfg.almgren_eta)

    def test_slices_sum_to_order_qty(self):
        from assembled_core.execution.execution_router import adaptive_ac_split, Order
        cfg, state = self._setup()
        state.obs_count = 10
        slices = adaptive_ac_split(Order("AAPL", "BUY", 10000, 150.0, "o1"), cfg, state)
        assert sum(s.quantity for s in slices) == 10000

    def test_update_raises_eta_on_high_slippage(self):
        _, state = self._setup()
        eta_before = state.eta_hat
        state.obs_count = 10
        state.update(qty_filled=5000, expected_price=100.0,
                     actual_price=101.0, side="BUY", sigma_daily=0.015)
        assert state.eta_hat > eta_before

    def test_update_ignored_below_min_obs(self):
        _, state = self._setup()
        eta_before = state.eta_hat
        state.update(5000, 100.0, 101.0, "BUY", 0.015)
        assert state.eta_hat == eta_before

    def test_partial_order_residual(self):
        from assembled_core.execution.execution_router import adaptive_ac_split, Order
        cfg, state = self._setup()
        state.obs_count = 10
        order = Order("MSFT", "SELL", 3000, 200.0, "o2")
        slices = adaptive_ac_split(order, cfg, state,
                                   remaining_qty=3000, remaining_slices=2)
        assert sum(s.quantity for s in slices) == 3000
        assert len(slices) <= 2

    def test_algo_label(self):
        from assembled_core.execution.execution_router import adaptive_ac_split, Order
        cfg, state = self._setup()
        state.obs_count = 10
        slices = adaptive_ac_split(Order("X", "BUY", 1000, 50.0), cfg, state)
        assert all(s.algo == "almgren_chriss" for s in slices)
