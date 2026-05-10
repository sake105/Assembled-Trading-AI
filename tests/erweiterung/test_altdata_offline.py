"""Offline-Tests für altdata-Module — keine Netzwerk-Calls.

Nach Duplikat-Audit (siehe docs/erweiterung/DUPLICATE_AUDIT.md) wurden
sec_edgar, finra_short_interest, wikipedia_pageviews aus erweiterung gelöscht
— die mainline-Versionen unter src/assembled_core/data/sources/ sind funktional
besser und produktiver.

Hier verbleiben Tests für die echten Add-On-Quellen:
- cftc_cot (CFTC Commitments of Traders)
- fred_md (McCracken/Ng FRED-MD-Panel mit PCA-Faktoren)
- google_trends (pytrends-Wrapper)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from erweiterung.altdata import cftc_cot, fred_md, google_trends


def test_trends_zscore_basic():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=100, tz="UTC"),
            "keyword": ["TSLA"] * 100,
            "svi": np.random.default_rng(0).integers(20, 100, 100),
        }
    )
    out = google_trends.trends_zscore(df, lookback=30)
    assert "svi_z" in out.columns


def test_cot_net_zscore_basic():
    n = 60
    cols_long = "m_money_positions_long_all"
    cols_short = "m_money_positions_short_all"
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-05", periods=n, freq="W-FRI", tz="UTC"),
            "market": ["E-MINI S&P 500"] * n,
            cols_long: np.random.default_rng(0).integers(100_000, 200_000, n),
            cols_short: np.random.default_rng(1).integers(80_000, 180_000, n),
        }
    )
    out = cftc_cot.cot_net_position_zscore(df, category="m_money", lookback_weeks=20)
    assert "net_pos_z" in out.columns


def test_fred_md_apply_transforms():
    n = 50
    data = pd.DataFrame(
        {
            "GDP": np.linspace(15000, 17000, n),
            "INFL": np.linspace(1.5, 2.5, n),
            "UNRATE": np.linspace(4.0, 3.5, n),
        },
        index=pd.date_range("2020-01-01", periods=n, freq="MS"),
    )
    transforms = pd.DataFrame(
        {"code": [5, 1, 1]},  # Δlog, level, level
        index=["GDP", "INFL", "UNRATE"],
    )
    transforms.index.name = "variable"
    out = fred_md.apply_mccracken_transforms(data, transforms)
    assert "GDP" in out.columns
    # GDP transformed to Δlog -> first value NaN
    assert pd.isna(out["GDP"].iloc[0])


def test_attention_to_signal_with_mock_yahoo():
    """yahoo_options-Wrapper darf ohne yfinance importiert werden, ohne zu crashen."""
    from erweiterung.altdata import yahoo_options

    out = yahoo_options.put_call_ratio(pd.DataFrame())
    assert pd.isna(out["pc_volume"])
