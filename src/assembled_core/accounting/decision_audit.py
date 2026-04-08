"""Decision Audit Trail (Plan 8.10).

Records the full decision context per order:
signal score, regime, risk state, factor weights.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class DecisionRecord:
    """Full context for a single trading decision."""
    timestamp: str
    symbol: str
    direction: str
    signal_score: float
    regime: str = ""
    risk_state: str = ""
    factor_weights: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    notes: str = ""


class DecisionAuditTrail:
    """Accumulates and persists decision records."""

    def __init__(self) -> None:
        self.records: list[DecisionRecord] = []

    def record(self, decision: DecisionRecord) -> None:
        self.records.append(decision)

    def save(self, path: str = "output/audit/decisions.jsonl") -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a") as f:
            for r in self.records:
                f.write(json.dumps(asdict(r), default=str) + "\n")
        logger.info("[Audit] Saved %d decision records to %s", len(self.records), path)
        self.records.clear()

    def summary(self) -> dict:
        return {
            "n_records": len(self.records),
            "symbols": list({r.symbol for r in self.records}),
            "regimes": list({r.regime for r in self.records if r.regime}),
        }


__all__ = ["DecisionRecord", "DecisionAuditTrail"]
