"""German tax lot tracking for US-broker trades (Alpaca).

From 50_COMPLIANCE_RECHT.md §50.1.

Tracks all buy/sell lots, applies FIFO matching on close, and computes
realized P&L in EUR using the ECB reference rate.  The EUR conversion is
required for the German Anlage-KAP tax filing.

Requires no external services; ECB rate can be provided manually or via
the optional async `get_ecb_usd_eur_rate` helper (needs httpx + asyncio).
"""
from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class TaxLot:
    """Single FIFO lot — either an open buy or a closed position."""
    id: str
    symbol: str
    side: str                    # "buy" | "sell"
    qty: float
    price_usd: float
    price_eur: float
    usd_eur_rate: float
    trade_date: date
    trade_timestamp: datetime
    fees_usd: float = 0.0
    fees_eur: float = 0.0
    matched_against: str | None = None   # lot id of the opening trade
    realized_pnl_eur: float | None = None
    holding_days: int | None = None
    status: str = "open"         # "open" | "closed"

    @classmethod
    def open_lot(
        cls,
        symbol: str,
        qty: float,
        price_usd: float,
        usd_eur_rate: float,
        trade_date: date,
        trade_timestamp: datetime,
        fees_usd: float = 0.0,
    ) -> "TaxLot":
        fees_eur = fees_usd * usd_eur_rate
        return cls(
            id=str(uuid.uuid4()),
            symbol=symbol,
            side="buy",
            qty=qty,
            price_usd=price_usd,
            price_eur=price_usd * usd_eur_rate,
            usd_eur_rate=usd_eur_rate,
            trade_date=trade_date,
            trade_timestamp=trade_timestamp,
            fees_usd=fees_usd,
            fees_eur=fees_eur,
            status="open",
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["trade_date"] = self.trade_date.isoformat()
        d["trade_timestamp"] = self.trade_timestamp.isoformat()
        return d


@dataclass
class FIFOCloseResult:
    """Result of matching a closing trade against open lots."""
    lots_closed: list[dict]   # list of {lot_id, qty, pnl_eur, holding_days}
    total_pnl_eur: float
    qty_remaining: float      # > 0 if insufficient open lots


# ---------------------------------------------------------------------------
# FIFO matching logic (sync, pure Python)
# ---------------------------------------------------------------------------


def match_fifo(
    open_lots: list[TaxLot],
    qty_to_close: float,
    exit_price_usd: float,
    usd_eur_rate: float,
    exit_date: date,
) -> FIFOCloseResult:
    """Apply FIFO matching: close oldest lots first.

    Args:
        open_lots: List of open TaxLot objects, sorted oldest-first.
        qty_to_close: Total shares to close.
        exit_price_usd: Fill price in USD.
        usd_eur_rate: ECB rate on exit date (1 USD = x EUR).
        exit_date: Trade date of the close.

    Returns:
        FIFOCloseResult with per-lot breakdown and total P&L in EUR.
    """
    remaining = qty_to_close
    lots_closed = []

    for lot in sorted(open_lots, key=lambda l: (l.trade_date, l.trade_timestamp)):
        if remaining <= 0:
            break
        if lot.status != "open":
            continue

        match_qty = min(lot.qty, remaining)
        entry_eur = match_qty * lot.price_eur
        exit_eur = match_qty * exit_price_usd * usd_eur_rate
        pnl_eur = exit_eur - entry_eur - lot.fees_eur * (match_qty / lot.qty)
        holding_days = (exit_date - lot.trade_date).days

        lots_closed.append({
            "lot_id": lot.id,
            "qty": match_qty,
            "pnl_eur": round(pnl_eur, 4),
            "holding_days": holding_days,
        })
        remaining -= match_qty

    total_pnl = sum(l["pnl_eur"] for l in lots_closed)
    return FIFOCloseResult(
        lots_closed=lots_closed,
        total_pnl_eur=round(total_pnl, 4),
        qty_remaining=max(0.0, remaining),
    )


# ---------------------------------------------------------------------------
# ECB FX rate (sync fallback)
# ---------------------------------------------------------------------------


def get_ecb_usd_eur_rate_sync(
    trade_date: date,
    fallback_rate: float = 0.93,
) -> float:
    """Fetch ECB USD/EUR reference rate for a given date.

    Returns `fallback_rate` if the ECB API is unreachable or the date has
    no rate (weekends/holidays → uses previous business day rate).

    For production use, cache the result to avoid repeated calls.
    """
    try:
        import urllib.request
        url = (
            "https://data-api.ecb.europa.eu/service/data/EXR/D.USD.EUR.SP00.A"
            f"?startPeriod={trade_date.isoformat()}&endPeriod={trade_date.isoformat()}"
            "&format=jsondata"
        )
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())
        series = data["dataSets"][0]["series"]
        obs = next(iter(series.values()))["observations"]
        usd_per_eur = float(next(iter(obs.values()))[0])
        return round(1.0 / usd_per_eur, 6)
    except Exception:
        return fallback_rate


# ---------------------------------------------------------------------------
# SQLite-backed store
# ---------------------------------------------------------------------------


class TaxLotStore:
    """Persistent FIFO tax lot store backed by SQLite.

    Schema mirrors 50_COMPLIANCE_RECHT.md §50.1.
    """

    def __init__(self, db_path: str = "data/tax_lots.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tax_lots (
                    id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    qty REAL NOT NULL,
                    price_usd REAL NOT NULL,
                    price_eur REAL NOT NULL,
                    usd_eur_rate REAL NOT NULL,
                    trade_date TEXT NOT NULL,
                    trade_timestamp TEXT NOT NULL,
                    fees_usd REAL DEFAULT 0,
                    fees_eur REAL DEFAULT 0,
                    matched_against TEXT,
                    realized_pnl_eur REAL,
                    holding_days INTEGER,
                    status TEXT NOT NULL DEFAULT 'open'
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_lots_symbol_status "
                "ON tax_lots(symbol, status)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_lots_year "
                "ON tax_lots(trade_date)"
            )

    def add_lot(self, lot: TaxLot) -> None:
        d = lot.to_dict()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO tax_lots
                   (id, symbol, side, qty, price_usd, price_eur, usd_eur_rate,
                    trade_date, trade_timestamp, fees_usd, fees_eur,
                    matched_against, realized_pnl_eur, holding_days, status)
                   VALUES
                   (:id,:symbol,:side,:qty,:price_usd,:price_eur,:usd_eur_rate,
                    :trade_date,:trade_timestamp,:fees_usd,:fees_eur,
                    :matched_against,:realized_pnl_eur,:holding_days,:status)""",
                d,
            )

    def open_lots_for(self, symbol: str) -> list[TaxLot]:
        """Return all open lots for *symbol*, oldest first."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM tax_lots WHERE symbol=? AND status='open' "
                "ORDER BY trade_date ASC, trade_timestamp ASC",
                (symbol,),
            ).fetchall()
        return [self._row_to_lot(r) for r in rows]

    def close_lots(
        self,
        symbol: str,
        qty_to_close: float,
        exit_price_usd: float,
        usd_eur_rate: float,
        exit_date: date,
    ) -> FIFOCloseResult:
        """Execute FIFO close and persist the updated lot status."""
        open_lots = self.open_lots_for(symbol)
        result = match_fifo(open_lots, qty_to_close, exit_price_usd, usd_eur_rate, exit_date)

        lot_map = {l.id: l for l in open_lots}
        with sqlite3.connect(self.db_path) as conn:
            for close in result.lots_closed:
                lot = lot_map[close["lot_id"]]
                if abs(close["qty"] - lot.qty) < 1e-9:
                    conn.execute(
                        "UPDATE tax_lots SET status='closed', realized_pnl_eur=?, "
                        "holding_days=? WHERE id=?",
                        (close["pnl_eur"], close["holding_days"], lot.id),
                    )
                else:
                    # Partial close: reduce open lot, insert closed partial
                    new_qty = lot.qty - close["qty"]
                    conn.execute(
                        "UPDATE tax_lots SET qty=? WHERE id=?",
                        (new_qty, lot.id),
                    )
                    partial = TaxLot(
                        id=str(uuid.uuid4()),
                        symbol=lot.symbol,
                        side="buy",
                        qty=close["qty"],
                        price_usd=lot.price_usd,
                        price_eur=lot.price_eur,
                        usd_eur_rate=lot.usd_eur_rate,
                        trade_date=lot.trade_date,
                        trade_timestamp=lot.trade_timestamp,
                        fees_usd=0.0,
                        fees_eur=0.0,
                        matched_against=lot.id,
                        realized_pnl_eur=close["pnl_eur"],
                        holding_days=close["holding_days"],
                        status="closed",
                    )
                    conn.execute(
                        """INSERT INTO tax_lots
                           (id,symbol,side,qty,price_usd,price_eur,usd_eur_rate,
                            trade_date,trade_timestamp,fees_usd,fees_eur,
                            matched_against,realized_pnl_eur,holding_days,status)
                           VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            partial.id, partial.symbol, partial.side, partial.qty,
                            partial.price_usd, partial.price_eur, partial.usd_eur_rate,
                            partial.trade_date.isoformat(),
                            partial.trade_timestamp.isoformat(),
                            partial.fees_usd, partial.fees_eur,
                            partial.matched_against, partial.realized_pnl_eur,
                            partial.holding_days, partial.status,
                        ),
                    )
        logger.debug(
            "FIFO close %s qty=%.4f pnl_eur=%.2f",
            symbol, qty_to_close, result.total_pnl_eur,
        )
        return result

    def realized_pnl_for_year(self, year: int) -> float:
        """Total realized P&L in EUR for a given tax year."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(realized_pnl_eur),0) FROM tax_lots "
                "WHERE status='closed' AND trade_date LIKE ?",
                (f"{year}-%",),
            ).fetchone()
        return float(row[0]) if row else 0.0

    def _row_to_lot(self, row: sqlite3.Row) -> TaxLot:
        return TaxLot(
            id=row["id"],
            symbol=row["symbol"],
            side=row["side"],
            qty=row["qty"],
            price_usd=row["price_usd"],
            price_eur=row["price_eur"],
            usd_eur_rate=row["usd_eur_rate"],
            trade_date=date.fromisoformat(row["trade_date"]),
            trade_timestamp=datetime.fromisoformat(row["trade_timestamp"]),
            fees_usd=row["fees_usd"],
            fees_eur=row["fees_eur"],
            matched_against=row["matched_against"],
            realized_pnl_eur=row["realized_pnl_eur"],
            holding_days=row["holding_days"],
            status=row["status"],
        )


# ---------------------------------------------------------------------------
# Simple in-memory tracker (no DB, no EUR conversion)
# ---------------------------------------------------------------------------


class TaxLotTracker:
    """Lightweight in-memory FIFO tracker for USD-only P&L.

    Useful for tests and simple simulations that don't need the full
    TaxLotStore (no SQLite, no EUR conversion).
    """

    def __init__(self) -> None:
        self._lots: dict[str, list[tuple[float, float]]] = {}  # symbol → [(qty, price)]

    def buy(self, symbol: str, qty: float, price: float, trade_date: date | None = None) -> None:
        self._lots.setdefault(symbol, []).append((qty, price))

    def sell(self, symbol: str, qty_to_sell: float, exit_price: float, trade_date: date | None = None) -> float:
        """FIFO sell; returns realized P&L in USD."""
        remaining = qty_to_sell
        pnl = 0.0
        lots = self._lots.get(symbol, [])
        new_lots: list[tuple[float, float]] = []

        for lot_qty, lot_price in lots:
            if remaining <= 0:
                new_lots.append((lot_qty, lot_price))
                continue
            match = min(lot_qty, remaining)
            pnl += match * (exit_price - lot_price)
            remaining -= match
            if lot_qty - match > 1e-9:
                new_lots.append((lot_qty - match, lot_price))

        self._lots[symbol] = new_lots
        return round(pnl, 4)

    def open_qty(self, symbol: str) -> float:
        return sum(q for q, _ in self._lots.get(symbol, []))


__all__ = [
    "TaxLot", "TaxLotStore", "TaxLotTracker",
    "FIFOCloseResult", "match_fifo", "get_ecb_usd_eur_rate_sync",
]
