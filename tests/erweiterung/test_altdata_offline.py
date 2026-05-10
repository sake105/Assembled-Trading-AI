"""Offline-Tests für altdata-Module — keine Netzwerk-Calls.

Wir testen die Logic von Parsing, Aggregation, Z-Scoring etc., ohne externe APIs.
"""

from __future__ import annotations


import numpy as np
import pandas as pd

from erweiterung.altdata import (
    cftc_cot,
    finra_short_interest,
    fred_md,
    google_trends,
    sec_edgar,
    wikipedia_pageviews,
)


def test_wiki_attention_score_basic():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=60, tz="UTC"),
            "symbol": ["AAPL"] * 60,
            "article": ["Apple_Inc."] * 60,
            "views": np.random.default_rng(0).integers(1000, 50000, 60),
        }
    )
    df["log_views"] = np.log(df["views"].astype(float) + 1.0)
    out = wikipedia_pageviews.attention_score(df, lookback=20, shift_days=1)
    assert "attention_score" in out.columns
    assert out["attention_score"].notna().sum() > 0


def test_wiki_default_map_contains_majors():
    assert "AAPL" in wikipedia_pageviews.DEFAULT_MAP.mapping
    assert "MSFT" in wikipedia_pageviews.DEFAULT_MAP.mapping
    assert wikipedia_pageviews.DEFAULT_MAP.article_for("aapl") == "Apple_Inc."


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


def test_short_pressure_signal():
    n = 100
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, tz="UTC"),
            "symbol": ["AMC"] * n,
            "short_volume": np.random.default_rng(0).integers(100, 1000, n),
            "total_volume": np.random.default_rng(0).integers(1000, 10000, n),
            "short_ratio": np.random.default_rng(0).uniform(0.1, 0.6, n),
        }
    )
    out = finra_short_interest.short_pressure_signal(df, lookback=20)
    assert "short_pressure" in out.columns


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


def test_form4_xml_parser():
    xml = """
    <ownershipDocument>
        <rptOwnerName>John Smith</rptOwnerName>
        <transactionCode>P</transactionCode>
        <transactionShares><value>1000</value></transactionShares>
        <transactionPricePerShare><value>150.50</value></transactionPricePerShare>
        <isDirector>1</isDirector>
        <isOfficer>0</isOfficer>
    </ownershipDocument>
    """
    out = sec_edgar.parse_form4_xml(xml)
    assert out["owner"] == "John Smith"
    assert out["transaction_code"] == "P"
    assert out["shares"] == 1000.0
    assert out["price"] == 150.50
    assert out["is_director"] is True
    assert out["is_officer"] is False


def test_form4_empty_xml():
    out = sec_edgar.parse_form4_xml("")
    assert out == {}


def test_filings_to_event_features():
    df = pd.DataFrame(
        {
            "ticker": ["AAPL"] * 4,
            "form": ["4", "4", "8-K", "10-Q"],
            "filing_date": pd.to_datetime(
                [
                    "2024-01-05",
                    "2024-01-15",
                    "2024-01-20",
                    "2024-02-01",
                ],
                utc=True,
            ),
        }
    )
    out = sec_edgar.filings_to_event_features(df, lookback_days=30)
    assert "ticker" in out.columns
    assert "has_8k_recent" in out.columns


def test_attention_to_signal_with_mock_yahoo():
    """Stelle sicher, dass yahoo_options-Wrapper ohne yfinance abstürzt nicht."""
    from erweiterung.altdata import yahoo_options

    # leere Options-Daten
    out = yahoo_options.put_call_ratio(pd.DataFrame())
    assert pd.isna(out["pc_volume"])
