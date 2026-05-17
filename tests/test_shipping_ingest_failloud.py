"""Sub-Project A / Task A2 — shipping_routes_ingest fail-loud contract.

Schema reference (verified at src/assembled_core/data/shipping_routes_ingest.py:98-108):
columns = timestamp (UTC), route_id, port_from, port_to, symbol, ships, congestion_score.
"""

from __future__ import annotations

import pytest

from src.assembled_core.data.shipping_routes_ingest import load_shipping_sample


def test_load_shipping_sample_without_allow_sample_raises():
    with pytest.raises(ValueError, match="allow_sample=True"):
        load_shipping_sample(path=None)


def test_load_shipping_sample_with_allow_sample_returns_dummy():
    df = load_shipping_sample(path=None, allow_sample=True)
    assert not df.empty
    for col in (
        "timestamp",
        "route_id",
        "port_from",
        "port_to",
        "symbol",
        "ships",
        "congestion_score",
    ):
        assert col in df.columns, f"Missing column {col} in sample schema"


def test_load_shipping_sample_with_real_path_loads_file(tmp_path):
    import pandas as pd

    real_path = tmp_path / "real_shipping.parquet"
    pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01"], utc=True),
            "route_id": ["US-EU-001"],
            "port_from": ["NYC"],
            "port_to": ["HAM"],
            "symbol": ["MSFT"],
            "ships": [12],
            "congestion_score": [45],
        }
    ).to_parquet(real_path)
    df = load_shipping_sample(path=str(real_path), allow_sample=False)
    assert df.iloc[0]["route_id"] == "US-EU-001"
