"""Sub-Project A / Task A1 — insider_ingest fail-loud contract.

Schema reference (verified at src/assembled_core/data/insider_ingest.py:73-80):
columns = timestamp (UTC), symbol, trades_count, net_shares, role.
"""

from __future__ import annotations

import pytest

from src.assembled_core.data.insider_ingest import load_insider_sample


def test_load_insider_sample_without_allow_sample_raises():
    """No real path + no explicit opt-in → ValueError (no silent dummy)."""
    with pytest.raises(ValueError, match="allow_sample=True"):
        load_insider_sample(path=None)


def test_load_insider_sample_with_allow_sample_returns_dummy():
    """Explicit opt-in → dummy data returned with the documented sample schema."""
    df = load_insider_sample(path=None, allow_sample=True)
    assert not df.empty
    # Schema per insider_ingest.py:73-80 (NOT ticker/date)
    assert "symbol" in df.columns
    assert "timestamp" in df.columns
    assert "trades_count" in df.columns
    assert "net_shares" in df.columns
    assert "role" in df.columns


def test_load_insider_sample_with_real_path_loads_file(tmp_path):
    """Real path overrides allow_sample (real data always wins)."""
    import pandas as pd

    real_path = tmp_path / "real_insider.parquet"
    pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01"], utc=True),
            "symbol": ["AAPL"],
            "trades_count": [3],
            "net_shares": [1000],
            "role": ["CEO"],
        }
    ).to_parquet(real_path)
    df = load_insider_sample(path=str(real_path), allow_sample=False)
    assert df.iloc[0]["symbol"] == "AAPL"
