"""House Periodic Transaction Report (PTR) parser (T5.3).

Parses House of Representatives member financial disclosures from:
  - efts.house.gov XML/PDF filings (public, no API key required)
  - house.gov/disclosure raw transaction data (when available as CSV/XML)

Produces a normalized DataFrame matching the altdata disclosure schema
(event_date, disclosure_date, symbol, event_type, value_usd, filer_name).

PIT: disclosure_date = filing_date (when the PTR was submitted to the clerk).
MNPI note: All data is PUBLIC. The House PTR requirement mandates disclosure
within 45 days of the transaction. This creates a known T+45d latency bound.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_HOUSE_PTR_SCHEMA = {
    "filer_name": str,
    "symbol": str,
    "asset_description": str,
    "transaction_type": str,  # Purchase, Sale, Exchange
    "amount_range": str,      # e.g. "$1,001 - $15,000"
    "value_usd_low": float,
    "value_usd_high": float,
    "event_date": str,        # transaction date
    "disclosure_date": str,   # PTR filing date (publicly available date)
    "event_type": str,        # "house_ptr_purchase", "house_ptr_sale"
    "source_tier": str,       # "T2" (public disclosure, 45d latency)
}

_AMOUNT_RANGES = {
    "$1,001 - $15,000": (1001, 15000),
    "$15,001 - $50,000": (15001, 50000),
    "$50,001 - $100,000": (50001, 100000),
    "$100,001 - $250,000": (100001, 250000),
    "$250,001 - $500,000": (250001, 500000),
    "$500,001 - $1,000,000": (500001, 1000000),
    "$1,000,001 - $5,000,000": (1000001, 5000000),
    "Over $5,000,000": (5000001, 10000000),
}


@dataclass
class HousePTRTransaction:
    filer_name: str
    symbol: str
    asset_description: str
    transaction_type: str
    amount_range: str
    event_date: str
    disclosure_date: str
    value_usd_low: float = 0.0
    value_usd_high: float = 0.0
    event_type: str = ""

    def __post_init__(self) -> None:
        lo, hi = _AMOUNT_RANGES.get(self.amount_range, (0, 0))
        self.value_usd_low = float(lo)
        self.value_usd_high = float(hi)
        tx_lower = self.transaction_type.lower()
        if "purchase" in tx_lower or "buy" in tx_lower:
            self.event_type = "house_ptr_purchase"
        elif "sale" in tx_lower or "sell" in tx_lower:
            self.event_type = "house_ptr_sale"
        else:
            self.event_type = "house_ptr_other"


def parse_house_ptr_csv(path: str | Path) -> pd.DataFrame:
    """Parse a House PTR CSV export into the normalized disclosure schema.

    Expected CSV columns (House efts.house.gov export format):
        Last, First, StateDist, Transactions, FilingDate, DocID

    or the detailed transaction CSV:
        MemberName, TransactionDate, Owner, Ticker, AssetName, Type, Amount, Filed

    Returns an empty DataFrame on parse failure.
    """
    path = Path(path)
    if not path.exists():
        logger.warning("[WARN] HousePTR: file not found: %s", path)
        return pd.DataFrame()

    try:
        raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception as exc:
        logger.warning("[WARN] HousePTR: CSV parse error %s: %s", path, exc)
        return pd.DataFrame()

    # Normalize column names to lowercase
    raw.columns = [c.strip().lower().replace(" ", "_") for c in raw.columns]

    transactions: list[dict[str, Any]] = []
    for _, row in raw.iterrows():
        # Try to extract ticker / asset name
        symbol = str(row.get("ticker", "") or row.get("asset", "")).strip().upper()
        if not symbol or symbol == "N/A":
            symbol = ""

        asset_desc = str(row.get("assetname", row.get("asset_name", row.get("description", "")))).strip()
        tx_type = str(row.get("type", row.get("transaction_type", ""))).strip()
        amount_str = str(row.get("amount", "")).strip()
        event_date = str(row.get("transactiondate", row.get("transaction_date", row.get("date", "")))).strip()
        disclosure_date = str(row.get("filed", row.get("filingdate", row.get("filing_date", "")))).strip()
        filer = str(row.get("membername", row.get("member_name", row.get("last", "")))).strip()

        try:
            txn = HousePTRTransaction(
                filer_name=filer,
                symbol=symbol,
                asset_description=asset_desc,
                transaction_type=tx_type,
                amount_range=amount_str,
                event_date=event_date,
                disclosure_date=disclosure_date,
            )
            transactions.append({
                "filer_name": txn.filer_name,
                "symbol": txn.symbol,
                "asset_description": txn.asset_description,
                "transaction_type": txn.transaction_type,
                "amount_range": txn.amount_range,
                "event_date": txn.event_date,
                "disclosure_date": txn.disclosure_date,
                "value_usd_low": txn.value_usd_low,
                "value_usd_high": txn.value_usd_high,
                "event_type": txn.event_type,
                "source_tier": "T2",
            })
        except Exception as exc:
            logger.debug("[SKIP] HousePTR row parse error: %s", exc)
            continue

    if not transactions:
        logger.warning("[WARN] HousePTR: no valid transactions parsed from %s", path)
        return pd.DataFrame(columns=list(_HOUSE_PTR_SCHEMA.keys()))

    df = pd.DataFrame(transactions)
    logger.info("[OK] HousePTR: parsed %d transactions from %s", len(df), path)
    return df


def filter_stock_transactions(
    df: pd.DataFrame,
    min_value_usd: float = 1001.0,
) -> pd.DataFrame:
    """Keep only transactions with a known ticker and above minimum value.

    Filters out non-equity assets (bonds, mutual funds without tickers)
    and transactions below the minimum threshold.
    """
    if df.empty:
        return df
    has_ticker = df["symbol"].astype(str).str.strip().str.len() > 0
    above_min = df["value_usd_low"] >= min_value_usd
    result = df[has_ticker & above_min].copy()
    logger.debug("[OK] HousePTR.filter_stock_transactions: %d → %d rows", len(df), len(result))
    return result


def to_altdata_events(df: pd.DataFrame) -> pd.DataFrame:
    """Convert parsed PTR DataFrame to standard altdata event schema.

    Output columns: event_id, symbol, event_date, disclosure_date,
    event_type, source_tier, value_usd (midpoint), filer_name.
    """
    if df.empty:
        return pd.DataFrame()

    result = df.copy()
    result["value_usd"] = (result["value_usd_low"] + result["value_usd_high"]) / 2.0

    import hashlib
    result["event_id"] = result.apply(
        lambda r: "hptr_" + hashlib.sha256(
            f"{r['filer_name']}_{r['symbol']}_{r['event_date']}_{r['event_type']}".encode()
        ).hexdigest()[:12],
        axis=1,
    )

    return result[
        ["event_id", "symbol", "event_date", "disclosure_date",
         "event_type", "source_tier", "value_usd", "filer_name"]
    ].reset_index(drop=True)


__all__ = [
    "HousePTRTransaction",
    "parse_house_ptr_csv",
    "filter_stock_transactions",
    "to_altdata_events",
]
