"""SQLite-backed attribution storage.

From 38_FEATURE_ATTRIBUTION_DASHBOARD.md §2.3.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from src.assembled_core.attribution.schemas import CompositeAttribution


class AttributionStore:
    """Append-write attribution store backed by SQLite."""

    def __init__(self, db_path: str = "data/attributions.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
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
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ts_ticker ON attributions(timestamp, ticker)"
            )

    def save(self, attr: CompositeAttribution) -> None:
        with sqlite3.connect(self.db_path) as conn:
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
                    json.dumps(attr.dimension_contributions),
                    json.dumps(attr.dimension_raw_scores),
                    json.dumps(attr.dimension_weights),
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
        with sqlite3.connect(self.db_path) as conn:
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
        return CompositeAttribution(
            timestamp=datetime.fromisoformat(row[1]),
            ticker=row[2],
            composite_score=row[3],
            dimension_contributions=json.loads(row[4]),
            dimension_raw_scores=json.loads(row[5]),
            dimension_weights=json.loads(row[6]),
            strategy_id=row[7],
            model_version=row[8],
            regime=row[9],
        )


__all__ = ["AttributionStore"]
