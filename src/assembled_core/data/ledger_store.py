"""SQLite-backed Paper Trading Ledger.

Replaces the in-memory/JSON paper ledger with an ACID-compliant SQLite store.
Provides the same interface as ops/paper_ledger.py so existing code can
migrate by swapping the import.

Database schema:
    fills        — executed fill records
    positions    — current position snapshot
    equity_curve — daily equity mark-to-market
    orders       — all orders (open, filled, cancelled)

Usage:
    from src.assembled_core.data.ledger_store import LedgerStore

    ledger = LedgerStore("data/paper_ledger.db")
    ledger.apply_fill(fill_dict)
    ledger.mark_to_market(prices, as_of=today)
    equity = ledger.load_equity_curve()
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS fills (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    fill_id     TEXT UNIQUE,
    symbol      TEXT NOT NULL,
    side        TEXT NOT NULL,
    quantity    REAL NOT NULL,
    price       REAL NOT NULL,
    commission  REAL DEFAULT 0.0,
    filled_at   TEXT NOT NULL,
    order_id    TEXT,
    strategy    TEXT,
    extra_json  TEXT
);

CREATE TABLE IF NOT EXISTS positions (
    symbol          TEXT PRIMARY KEY,
    quantity        REAL NOT NULL DEFAULT 0.0,
    avg_cost        REAL NOT NULL DEFAULT 0.0,
    unrealized_pnl  REAL DEFAULT 0.0,
    realized_pnl    REAL DEFAULT 0.0,
    last_updated    TEXT
);

CREATE TABLE IF NOT EXISTS equity_curve (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    as_of       TEXT NOT NULL,
    equity      REAL NOT NULL,
    cash        REAL NOT NULL,
    positions_value REAL DEFAULT 0.0,
    UNIQUE(as_of)
);

CREATE TABLE IF NOT EXISTS orders (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id    TEXT UNIQUE,
    symbol      TEXT NOT NULL,
    side        TEXT NOT NULL,
    quantity    REAL NOT NULL,
    order_type  TEXT DEFAULT 'market',
    limit_price REAL,
    status      TEXT DEFAULT 'open',
    created_at  TEXT NOT NULL,
    filled_at   TEXT,
    fill_price  REAL,
    strategy    TEXT,
    extra_json  TEXT
);

CREATE TABLE IF NOT EXISTS ledger_meta (
    key     TEXT PRIMARY KEY,
    value   TEXT
);
"""

_INDEXES_SQL = """
CREATE INDEX IF NOT EXISTS idx_fills_symbol ON fills(symbol);
CREATE INDEX IF NOT EXISTS idx_fills_filled_at ON fills(filled_at);
CREATE INDEX IF NOT EXISTS idx_equity_curve_as_of ON equity_curve(as_of);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
"""


class LedgerStore:
    """SQLite-backed paper trading ledger.

    Thread-safe for single-process use. Uses WAL mode for better concurrent
    read performance.

    Attributes:
        db_path: Path to the SQLite database file.
        initial_cash: Starting cash balance (used on first init).
    """

    def __init__(
        self,
        db_path: str | Path = "data/paper_ledger.db",
        initial_cash: float = 100_000.0,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initial_cash = initial_cash
        self._init_db()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        con = sqlite3.connect(str(self.db_path), timeout=30)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA foreign_keys=ON")
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    def _init_db(self) -> None:
        """Create tables and indexes on first open."""
        with self._conn() as con:
            con.executescript(_SCHEMA_SQL)
            con.executescript(_INDEXES_SQL)
            # Initialize cash if not set
            row = con.execute("SELECT value FROM ledger_meta WHERE key='cash'").fetchone()
            if row is None:
                con.execute(
                    "INSERT INTO ledger_meta(key, value) VALUES('cash', ?)",
                    (str(self._initial_cash),),
                )
        logger.debug("[LedgerStore] Initialized at %s", self.db_path)

    # ------------------------------------------------------------------
    # State access
    # ------------------------------------------------------------------

    def get_cash(self) -> float:
        with self._conn() as con:
            row = con.execute("SELECT value FROM ledger_meta WHERE key='cash'").fetchone()
            return float(row["value"]) if row else self._initial_cash

    def set_cash(self, cash: float) -> None:
        with self._conn() as con:
            con.execute(
                "INSERT OR REPLACE INTO ledger_meta(key, value) VALUES('cash', ?)",
                (str(cash),),
            )

    def get_positions(self) -> pd.DataFrame:
        """Return current positions as DataFrame."""
        with self._conn() as con:
            rows = con.execute("SELECT * FROM positions WHERE quantity != 0").fetchall()
        if not rows:
            return pd.DataFrame(columns=["symbol", "quantity", "avg_cost", "unrealized_pnl", "realized_pnl"])
        return pd.DataFrame([dict(r) for r in rows])

    def get_position(self, symbol: str) -> dict:
        """Return position dict for a single symbol."""
        with self._conn() as con:
            row = con.execute("SELECT * FROM positions WHERE symbol=?", (symbol,)).fetchone()
        if row is None:
            return {"symbol": symbol, "quantity": 0.0, "avg_cost": 0.0,
                    "unrealized_pnl": 0.0, "realized_pnl": 0.0}
        return dict(row)

    # ------------------------------------------------------------------
    # Fill application
    # ------------------------------------------------------------------

    def apply_fill(self, fill: dict) -> None:
        """Apply a fill to the ledger, updating positions and cash.

        Args:
            fill: Dict with keys: symbol, side, quantity, price, fill_id (optional),
                  commission (optional), filled_at (optional), order_id (optional),
                  strategy (optional).
        """
        symbol = fill["symbol"]
        side = fill["side"].upper()
        quantity = float(fill["quantity"])
        price = float(fill["price"])
        commission = float(fill.get("commission", 0.0))
        filled_at = fill.get("filled_at", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f"))
        fill_id = fill.get("fill_id", f"{symbol}_{filled_at}")
        order_id = fill.get("order_id")
        strategy = fill.get("strategy")
        extra = fill.get("extra")

        with self._conn() as con:
            # Insert fill record
            con.execute(
                """INSERT OR IGNORE INTO fills
                   (fill_id, symbol, side, quantity, price, commission, filled_at, order_id, strategy, extra_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (fill_id, symbol, side, quantity, price, commission, filled_at,
                 order_id, strategy, json.dumps(extra) if extra else None),
            )

            # Update position
            pos = dict(con.execute(
                "SELECT * FROM positions WHERE symbol=?", (symbol,)
            ).fetchone() or {})

            current_qty = float(pos.get("quantity", 0.0))
            current_avg = float(pos.get("avg_cost", 0.0))
            current_realized = float(pos.get("realized_pnl", 0.0))

            signed_qty = quantity if side == "BUY" else -quantity
            new_qty = current_qty + signed_qty

            # Update avg cost and realized PnL
            if side == "BUY":
                if current_qty >= 0:
                    # Adding to long position
                    total_cost = current_qty * current_avg + quantity * price
                    new_avg = total_cost / new_qty if new_qty != 0 else price
                else:
                    # Covering short
                    realized = -current_qty * (current_avg - price) if current_qty < 0 else 0.0
                    current_realized += realized
                    new_avg = price if new_qty > 0 else 0.0
            else:  # SELL
                if current_qty > 0:
                    # Reducing long
                    closed_qty = min(quantity, current_qty)
                    realized = closed_qty * (price - current_avg)
                    current_realized += realized
                    new_avg = current_avg if new_qty > 0 else 0.0
                else:
                    # Opening/adding to short
                    total_cost = abs(current_qty) * current_avg + quantity * price
                    denom = abs(new_qty) if new_qty != 0 else 1.0
                    new_avg = total_cost / denom
            new_avg = new_avg if new_qty != 0 else 0.0

            con.execute(
                """INSERT OR REPLACE INTO positions
                   (symbol, quantity, avg_cost, realized_pnl, last_updated)
                   VALUES (?, ?, ?, ?, ?)""",
                (symbol, new_qty, new_avg, current_realized, filled_at),
            )

            # Update cash: BUY reduces cash, SELL increases cash
            cash_row = con.execute("SELECT value FROM ledger_meta WHERE key='cash'").fetchone()
            cash = float(cash_row["value"]) if cash_row else self._initial_cash
            if side == "BUY":
                cash -= quantity * price + commission
            else:
                cash += quantity * price - commission
            con.execute(
                "INSERT OR REPLACE INTO ledger_meta(key, value) VALUES('cash', ?)", (str(cash),)
            )

        logger.debug("[LedgerStore] Fill applied: %s %s %s qty=%.2f price=%.4f",
                     side, symbol, fill_id, quantity, price)

    # ------------------------------------------------------------------
    # Mark to market
    # ------------------------------------------------------------------

    def mark_to_market(
        self,
        prices: dict[str, float] | pd.DataFrame,
        as_of: Optional[str] = None,
        symbol_col: str = "symbol",
        close_col: str = "close",
    ) -> float:
        """Update unrealized PnL and record equity curve point.

        Args:
            prices: Dict of symbol → price, or DataFrame with symbol/close columns.
            as_of: Date string for equity curve record (default: today UTC).
            symbol_col: Symbol column if prices is DataFrame.
            close_col: Close price column if prices is DataFrame.

        Returns:
            Total portfolio equity (cash + positions value).
        """
        as_of = as_of or datetime.now(timezone.utc).strftime("%Y-%m-%d")

        if isinstance(prices, pd.DataFrame):
            price_map = dict(zip(prices[symbol_col], prices[close_col]))
        else:
            price_map = dict(prices)

        with self._conn() as con:
            positions = con.execute("SELECT * FROM positions WHERE quantity != 0").fetchall()
            positions_value = 0.0

            for pos in positions:
                sym = pos["symbol"]
                qty = float(pos["quantity"])
                avg = float(pos["avg_cost"])
                cur_price = price_map.get(sym, avg)  # fall back to avg cost if no price
                unrealized = qty * (cur_price - avg)
                positions_value += qty * cur_price

                con.execute(
                    "UPDATE positions SET unrealized_pnl=?, last_updated=? WHERE symbol=?",
                    (unrealized, as_of, sym),
                )

            cash_row = con.execute("SELECT value FROM ledger_meta WHERE key='cash'").fetchone()
            cash = float(cash_row["value"]) if cash_row else self._initial_cash
            equity = cash + positions_value

            con.execute(
                """INSERT OR REPLACE INTO equity_curve(as_of, equity, cash, positions_value)
                   VALUES (?, ?, ?, ?)""",
                (as_of, equity, cash, positions_value),
            )

        return equity

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def load_equity_curve(self) -> pd.DataFrame:
        """Return full equity curve as DataFrame."""
        with self._conn() as con:
            rows = con.execute(
                "SELECT as_of, equity, cash, positions_value FROM equity_curve ORDER BY as_of"
            ).fetchall()
        if not rows:
            return pd.DataFrame(columns=["as_of", "equity", "cash", "positions_value"])
        df = pd.DataFrame([dict(r) for r in rows])
        df["as_of"] = pd.to_datetime(df["as_of"])
        return df

    def query_fills(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """Query fill history with optional filters."""
        conditions = []
        params: list[Any] = []
        if symbol:
            conditions.append("symbol = ?")
            params.append(symbol)
        if start_date:
            conditions.append("filled_at >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("filled_at <= ?")
            params.append(end_date)
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        # Safe: `where` is built from a fixed set of hardcoded clauses with
        # parameterized `?` placeholders; `limit` is cast to int. No user string
        # is interpolated into the SQL.
        sql = f"SELECT * FROM fills {where} ORDER BY filled_at DESC LIMIT {int(limit)}"  # nosec B608
        with self._conn() as con:
            rows = con.execute(sql, params).fetchall()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([dict(r) for r in rows])

    def import_from_json(self, json_path: str | Path) -> int:
        """Import fills from an existing JSON ledger file (migration helper).

        Args:
            json_path: Path to JSON file containing a list of fill dicts or
                       a dict with a 'fills' key.

        Returns:
            Number of fills imported.
        """
        path = Path(json_path)
        if not path.exists():
            logger.warning("[LedgerStore] JSON path not found: %s", path)
            return 0

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.error("[LedgerStore] JSON parse error in %s: %s", path, exc)
            return 0
        fills = data if isinstance(data, list) else data.get("fills", [])
        count = 0
        for fill in fills:
            try:
                self.apply_fill(fill)
                count += 1
            except Exception as exc:
                logger.warning("[LedgerStore] Import failed for fill %s: %s", fill, exc)

        logger.info("[LedgerStore] Imported %d fills from %s", count, path)
        return count
