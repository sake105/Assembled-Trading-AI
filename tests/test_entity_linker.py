"""Unit tests for EntityLinker (X2)."""

from __future__ import annotations

import pytest

pytest.importorskip("src.assembled_core.intel.entity_linker")

import csv
from pathlib import Path

import pytest

from src.assembled_core.intel.entity_linker import EntityLinker


@pytest.mark.phase12
class TestEntityLinkerBasic:
    def test_exact_ticker_match(self):
        linker = EntityLinker(symbols=["AAPL", "MSFT"])
        assert linker.link("AAPL") == ["AAPL"]
        assert linker.link("aapl") == ["AAPL"]

    def test_builtin_alias(self):
        linker = EntityLinker()
        result = linker.link("Apple")
        assert "AAPL" in result

    def test_builtin_alias_google(self):
        linker = EntityLinker()
        result = linker.link("Google")
        assert "GOOGL" in result

    def test_case_insensitive(self):
        linker = EntityLinker()
        assert linker.link("APPLE") == linker.link("apple")

    def test_unknown_entity_returns_empty(self):
        linker = EntityLinker()
        result = linker.link("XYZ_UNKNOWN_COMPANY_12345")
        assert result == []

    def test_empty_input_returns_empty(self):
        linker = EntityLinker()
        assert linker.link("") == []
        assert linker.link(None) == []

    def test_geo_to_etf_us(self):
        linker = EntityLinker()
        result = linker.geo_to_etf("US")
        assert "SPY" in result

    def test_geo_to_etf_unknown(self):
        linker = EntityLinker()
        assert linker.geo_to_etf("ZZ") == []

    def test_extra_aliases(self):
        linker = EntityLinker(extra_aliases={"my_company": ["MYCO"]})
        assert linker.link("my_company") == ["MYCO"]

    def test_add_alias_runtime(self):
        linker = EntityLinker()
        linker.add_alias("Assembled Inc", ["ASSM"])
        assert linker.link("assembled inc") == ["ASSM"]

    def test_link_many(self):
        linker = EntityLinker()
        result = linker.link_many(["Apple", "Tesla"])
        assert "AAPL" in result.get("Apple", [])
        assert "TSLA" in result.get("Tesla", [])

    def test_partial_match(self):
        linker = EntityLinker()
        result = linker.link("Berkshire Hathaway")
        assert "BRK.B" in result

    def test_sector_alias(self):
        linker = EntityLinker()
        result = linker.link("technology")
        assert len(result) > 0

    def test_max_results_respected(self):
        linker = EntityLinker()
        result = linker.link("technology", max_results=1)
        assert len(result) <= 1


@pytest.mark.phase12
class TestEntityLinkerFromCSV:
    def test_load_from_csv(self, tmp_path):
        csv_path = tmp_path / "master.csv"
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["symbol", "sector"])
            writer.writeheader()
            writer.writerows(
                [
                    {"symbol": "MYCO", "sector": "Technology"},
                    {"symbol": "OTHR", "sector": "Energy"},
                ]
            )
        linker = EntityLinker.from_security_master(csv_path)
        assert linker.symbol_count == 2
        assert linker.link("MYCO") == ["MYCO"]

    def test_missing_csv_graceful(self, tmp_path):
        linker = EntityLinker.from_security_master(tmp_path / "nonexistent.csv")
        assert linker.symbol_count == 0
        # Builtin aliases still work
        assert len(linker.link("Apple")) > 0

    def test_from_real_security_master(self):
        """Integration: load from actual configs/security_master.csv."""
        path = Path("configs/security_master.csv")
        if not path.exists():
            pytest.skip("configs/security_master.csv not found")
        linker = EntityLinker.from_security_master(path)
        assert linker.symbol_count > 0
        assert linker.link("AAPL") == ["AAPL"]
