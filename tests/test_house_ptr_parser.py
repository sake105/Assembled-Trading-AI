"""Tests for HousePTR parser (T5.3)."""

from __future__ import annotations

import pytest
import pandas as pd

from src.assembled_core.data.altdata.house_ptr_parser import (
    HousePTRTransaction,
    parse_house_ptr_csv,
    filter_stock_transactions,
    to_altdata_events,
)


@pytest.mark.phase12
class TestHousePTRTransaction:
    def test_purchase_type(self):
        t = HousePTRTransaction(
            filer_name="Smith, Jane",
            symbol="AAPL",
            asset_description="Apple Inc.",
            transaction_type="Purchase",
            amount_range="$15,001 - $50,000",
            event_date="2024-03-15",
            disclosure_date="2024-04-20",
        )
        assert t.event_type == "house_ptr_purchase"
        assert t.value_usd_low == 15001
        assert t.value_usd_high == 50000

    def test_sale_type(self):
        t = HousePTRTransaction(
            filer_name="Jones, Bob",
            symbol="MSFT",
            asset_description="Microsoft",
            transaction_type="Sale (Full)",
            amount_range="$1,001 - $15,000",
            event_date="2024-03-10",
            disclosure_date="2024-04-10",
        )
        assert t.event_type == "house_ptr_sale"

    def test_unknown_amount_range(self):
        t = HousePTRTransaction(
            filer_name="X",
            symbol="X",
            asset_description="X",
            transaction_type="Purchase",
            amount_range="Unknown Range",
            event_date="2024-01-01",
            disclosure_date="2024-02-01",
        )
        assert t.value_usd_low == 0.0
        assert t.value_usd_high == 0.0

    def test_other_type(self):
        t = HousePTRTransaction(
            filer_name="X",
            symbol="X",
            asset_description="X",
            transaction_type="Exchange",
            amount_range="$1,001 - $15,000",
            event_date="2024-01-01",
            disclosure_date="2024-02-01",
        )
        assert t.event_type == "house_ptr_other"


@pytest.mark.phase12
class TestParsePTRCsv:
    def _make_csv(self, tmp_path, rows: list[dict]) -> str:
        df = pd.DataFrame(rows)
        path = tmp_path / "ptr.csv"
        df.to_csv(path, index=False)
        return str(path)

    def test_basic_parse(self, tmp_path):
        rows = [{
            "MemberName": "Smith, Jane",
            "Ticker": "AAPL",
            "AssetName": "Apple",
            "Type": "Purchase",
            "Amount": "$1,001 - $15,000",
            "TransactionDate": "03/15/2024",
            "Filed": "04/20/2024",
        }]
        path = self._make_csv(tmp_path, rows)
        df = parse_house_ptr_csv(path)
        assert not df.empty
        assert "symbol" in df.columns

    def test_missing_file(self):
        df = parse_house_ptr_csv("/nonexistent/file.csv")
        assert df.empty

    def test_empty_csv(self, tmp_path):
        path = tmp_path / "empty.csv"
        path.write_text("MemberName,Ticker,Type,Amount,TransactionDate,Filed\n")
        df = parse_house_ptr_csv(str(path))
        assert df.empty or len(df) == 0


@pytest.mark.phase12
class TestFilterAndConvert:
    def _make_df(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"symbol": "AAPL", "value_usd_low": 1001, "event_type": "house_ptr_purchase",
             "value_usd_high": 15000, "filer_name": "A", "event_date": "2024-01-10",
             "disclosure_date": "2024-02-20", "source_tier": "T2",
             "transaction_type": "Purchase", "asset_description": "Apple", "amount_range": "$1,001 - $15,000"},
            {"symbol": "", "value_usd_low": 500, "event_type": "house_ptr_other",
             "value_usd_high": 1000, "filer_name": "B", "event_date": "2024-01-11",
             "disclosure_date": "2024-02-21", "source_tier": "T2",
             "transaction_type": "Exchange", "asset_description": "Bond", "amount_range": "Unknown"},
        ])

    def test_filter_removes_no_ticker(self):
        df = self._make_df()
        filtered = filter_stock_transactions(df)
        assert len(filtered) == 1
        assert filtered.iloc[0]["symbol"] == "AAPL"

    def test_filter_min_value(self):
        df = self._make_df()
        filtered = filter_stock_transactions(df, min_value_usd=2000.0)
        assert filtered.empty

    def test_to_altdata_events_schema(self):
        df = self._make_df().iloc[:1]
        events = to_altdata_events(df)
        assert not events.empty
        expected_cols = {"event_id", "symbol", "event_date", "disclosure_date",
                         "event_type", "source_tier", "value_usd", "filer_name"}
        assert expected_cols.issubset(set(events.columns))

    def test_event_id_unique(self):
        df = pd.concat([self._make_df().iloc[:1], self._make_df().iloc[:1]], ignore_index=True)
        events = to_altdata_events(df)
        # Same row duplicated → same hash
        assert events["event_id"].nunique() == 1
