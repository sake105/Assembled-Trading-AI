"""Tests for M10: ETF Universe Upgrade."""

from __future__ import annotations
import pytest
from pathlib import Path
from src.assembled_core.data.universe_etf import (
    load_etf_universe,
    get_all_symbols,
    get_symbols_by_asset_class,
    get_symbols_by_group,
    get_defensive_symbols,
    build_symbol_metadata,
)

_UNIVERSE_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "universe_etf_v1.yaml"
)


@pytest.mark.phase12
@pytest.mark.phase13
class TestLoadETFUniverse:
    def test_loads_successfully(self):
        u = load_etf_universe(_UNIVERSE_PATH)
        assert "etfs" in u

    def test_universe_version_set(self):
        u = load_etf_universe(_UNIVERSE_PATH)
        assert u.get("universe_version") is not None

    def test_raises_for_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_etf_universe(tmp_path / "nonexistent.yaml")


@pytest.mark.phase12
@pytest.mark.phase13
class TestGetAllSymbols:
    def test_returns_list(self):
        u = load_etf_universe(_UNIVERSE_PATH)
        syms = get_all_symbols(u)
        assert isinstance(syms, list)

    def test_contains_spy(self):
        u = load_etf_universe(_UNIVERSE_PATH)
        assert "SPY" in get_all_symbols(u)

    def test_contains_tlt(self):
        u = load_etf_universe(_UNIVERSE_PATH)
        assert "TLT" in get_all_symbols(u)

    def test_contains_gld(self):
        u = load_etf_universe(_UNIVERSE_PATH)
        assert "GLD" in get_all_symbols(u)

    def test_at_least_20_symbols(self):
        u = load_etf_universe(_UNIVERSE_PATH)
        assert len(get_all_symbols(u)) >= 20

    def test_sorted(self):
        u = load_etf_universe(_UNIVERSE_PATH)
        syms = get_all_symbols(u)
        assert syms == sorted(syms)

    def test_no_duplicates(self):
        u = load_etf_universe(_UNIVERSE_PATH)
        syms = get_all_symbols(u)
        assert len(syms) == len(set(syms))


@pytest.mark.phase12
@pytest.mark.phase13
class TestGetSymbolsByAssetClass:
    def test_equity_symbols_include_spy(self):
        u = load_etf_universe(_UNIVERSE_PATH)
        syms = get_symbols_by_asset_class(u, "equity")
        assert "SPY" in syms

    def test_fixed_income_includes_tlt(self):
        u = load_etf_universe(_UNIVERSE_PATH)
        syms = get_symbols_by_asset_class(u, "fixed_income")
        assert "TLT" in syms

    def test_unknown_class_returns_empty(self):
        u = load_etf_universe(_UNIVERSE_PATH)
        syms = get_symbols_by_asset_class(u, "nonexistent")
        assert syms == []


@pytest.mark.phase12
@pytest.mark.phase13
class TestGetSymbolsByGroup:
    def test_equity_broad_returns_spy(self):
        u = load_etf_universe(_UNIVERSE_PATH)
        syms = get_symbols_by_group(u, "equity_broad")
        assert "SPY" in syms

    def test_fixed_income_group(self):
        u = load_etf_universe(_UNIVERSE_PATH)
        syms = get_symbols_by_group(u, "fixed_income")
        assert len(syms) >= 3

    def test_unknown_group_returns_empty(self):
        u = load_etf_universe(_UNIVERSE_PATH)
        assert get_symbols_by_group(u, "nonexistent_group") == []


@pytest.mark.phase12
@pytest.mark.phase13
class TestGetDefensiveSymbols:
    def test_includes_gld(self):
        u = load_etf_universe(_UNIVERSE_PATH)
        assert "GLD" in get_defensive_symbols(u)

    def test_includes_tlt(self):
        u = load_etf_universe(_UNIVERSE_PATH)
        assert "TLT" in get_defensive_symbols(u)

    def test_includes_shy(self):
        u = load_etf_universe(_UNIVERSE_PATH)
        assert "SHY" in get_defensive_symbols(u)

    def test_not_empty(self):
        u = load_etf_universe(_UNIVERSE_PATH)
        assert len(get_defensive_symbols(u)) >= 4


@pytest.mark.phase12
@pytest.mark.phase13
class TestBuildSymbolMetadata:
    def test_spy_has_metadata(self):
        u = load_etf_universe(_UNIVERSE_PATH)
        meta = build_symbol_metadata(u)
        assert "SPY" in meta
        assert meta["SPY"]["asset_class"] == "equity"

    def test_all_entries_have_required_keys(self):
        u = load_etf_universe(_UNIVERSE_PATH)
        meta = build_symbol_metadata(u)
        for sym, m in meta.items():
            assert "name" in m
            assert "asset_class" in m
            assert "group" in m

    def test_gld_is_commodity(self):
        u = load_etf_universe(_UNIVERSE_PATH)
        meta = build_symbol_metadata(u)
        assert meta["GLD"]["asset_class"] == "commodity"
