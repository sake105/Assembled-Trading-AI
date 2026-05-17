"""SQLite-backed attribution storage.

From 38_FEATURE_ATTRIBUTION_DASHBOARD.md §2.3.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path

_logger = logging.getLogger(__name__)

from src.assembled_core.attribution.schemas import CompositeAttribution


class AttributionStore:
    """Append-write attribution store backed by SQLite.

    B4-AT-01 R6 fix: enables WAL journal mode + connection timeout for
    concurrent-safety. Previously fresh sqlite3.connect() per save() with
    no WAL, no timeout — concurrent writes from paper_runner + research
    process collided with "database is locked".
    """

    _CONN_TIMEOUT_S = 5.0

    def __init__(self, db_path: str = "data/attributions.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        """Open connection with WAL mode + timeout (B4-AT-01)."""
        conn = sqlite3.connect(self.db_path, timeout=self._CONN_TIMEOUT_S)
        # WAL allows concurrent readers while writer holds lock; reduces
        # "database is locked" errors significantly. PRAGMA is per-connection
        # for synchronous, but WAL is database-file-level once set.
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")  # WAL-recommended
        except sqlite3.Error:
            # If WAL setup fails (e.g. read-only filesystem), continue with
            # default journal — connection still usable.
            pass
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS attributions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    composite_score REAL NOT NULL,
                    dimension_contributions_json TEXT NOT NULL,
                    dimension_raw_scores_json TEXT NOT NULL,
                    dimension_weights_json TEXT NOT NULL,
                    strategy_id TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    regime TEXT NOT NULL
                )
            """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ts_ticker ON attributions(timestamp, ticker)"
            )

    def save(self, attr: CompositeAttribution) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO attributions
                   (timestamp, ticker, composite_score,
                    dimension_contributions_json, dimension_raw_scores_json,
                    dimension_weights_json, strategy_id, model_version, regime)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    attr.timestamp.isoformat(),
                    attr.ticker,
                    attr.composite_score,
                    json.dumps(
                        attr.dimension_contributions,
                        default=lambda v: v.item() if hasattr(v, "item") else str(v),
                    ),
                    json.dumps(
                        attr.dimension_raw_scores,
                        default=lambda v: v.item() if hasattr(v, "item") else str(v),
                    ),
                    json.dumps(
                        attr.dimension_weights,
                        default=lambda v: v.item() if hasattr(v, "item") else str(v),
                    ),
                    attr.strategy_id,
                    attr.model_version,
                    attr.regime,
                ),
            )

    def load_for_ticker(
        self,
        ticker: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[CompositeAttribution]:
        with self._connect() as conn:
            query = "SELECT * FROM attributions WHERE ticker = ?"
            params: list = [ticker]
            if start:
                query += " AND timestamp >= ?"
                params.append(start.isoformat())
            if end:
                query += " AND timestamp <= ?"
                params.append(end.isoformat())
            query += " ORDER BY timestamp"
            cursor = conn.execute(query, params)
            return [self._row_to_attribution(row) for row in cursor]

    def _row_to_attribution(self, row: tuple) -> CompositeAttribution:
        try:
            dim_contributions = json.loads(row[4])
            dim_raw_scores = json.loads(row[5])
            dim_weights = json.loads(row[6])
        except json.JSONDecodeError as exc:
            _logger.warning(
                "[AttributionStore] corrupted JSON in row for %s: %s", row[2], exc
            )
            dim_contributions = {}
            dim_raw_scores = {}
            dim_weights = {}
        return CompositeAttribution(
            timestamp=datetime.fromisoformat(row[1]),
            ticker=row[2],
            composite_score=row[3],
            dimension_contributions=dim_contributions,
            dimension_raw_scores=dim_raw_scores,
            dimension_weights=dim_weights,
            strategy_id=row[7],
            model_version=row[8],
            regime=row[9],
        )


__all__ = ["AttributionStore"]
