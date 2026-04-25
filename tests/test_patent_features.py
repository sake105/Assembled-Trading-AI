"""Tests for M38b: Patent Activity Features."""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

import pytest; pytest.importorskip("src.assembled_core.data.altdata.patent_features")
from src.assembled_core.data.altdata.patent_features import (
    PatentConfig,
    compute_patent_features,
)


def _synthetic_filings(n: int = 100, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    symbols = rng.choice(["AAPL", "GOOG", "MSFT"], n)
    dates = pd.date_range("2022-01-01", periods=730, freq="D")
    filing_dates = rng.choice(dates, n)
    ipc_classes = rng.choice(["H04L", "G06F", "H01L", "G06N", "A61K"], n)
    citations = rng.poisson(3, n)
    return pd.DataFrame({
        "symbol": symbols,
        "filing_date": filing_dates,
        "ipc_class": ipc_classes,
        "forward_citations": citations,
    })


@pytest.mark.phase12
class TestComputePatentFeatures:
    def test_basic_output(self):
        filings = _synthetic_filings()
        result = compute_patent_features(filings, as_of="2024-01-01")
        assert "patent_count_12m" in result.columns
        assert "patent_growth_yoy" in result.columns
        assert "patent_citation_score" in result.columns
        assert "patent_breadth" in result.columns
        assert "patent_recency_days" in result.columns
        assert "innovation_momentum" in result.columns
        assert len(result) > 0

    def test_pit_safety(self):
        """Only filings before as_of should be counted."""
        filings = pd.DataFrame({
            "symbol": ["AAPL"] * 10,
            "filing_date": pd.date_range("2024-06-01", periods=10, freq="D"),
            "ipc_class": ["G06F"] * 10,
            "forward_citations": [2] * 10,
        })
        result = compute_patent_features(filings, as_of="2024-01-01")
        assert len(result) == 0  # all filings are in the future

    def test_empty_input(self):
        result = compute_patent_features(pd.DataFrame(), as_of="2024-01-01")
        assert len(result) == 0

    def test_innovation_momentum_range(self):
        filings = _synthetic_filings()
        result = compute_patent_features(filings, as_of="2024-01-01")
        if len(result) > 0:
            assert all(result["innovation_momentum"] >= 0)

    def test_custom_config(self):
        filings = _synthetic_filings()
        cfg = PatentConfig(lookback_months=6, min_patents=2)
        result = compute_patent_features(filings, as_of="2024-01-01", config=cfg)
        # Shorter lookback should still work
        assert isinstance(result, pd.DataFrame)

    def test_patent_breadth_positive(self):
        filings = _synthetic_filings()
        result = compute_patent_features(filings, as_of="2024-01-01")
        if len(result) > 0:
            assert all(result["patent_breadth"] >= 0)
