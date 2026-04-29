"""Trade-Level Transaction Cost Analysis (TCA).

Pro Trade Analyse von:
- Implementation Shortfall (Arrival Price vs Execution Price) — implementiert
- VWAP-Slippage (Execution vs VWAP) — implementiert

Aggregation pro Symbol/Broker/Strategy liefert Cost-Patterns.

PIT-Invariante: Benchmarks aus historischen Prices zur Execution-Zeit.

Architecture note (2026-04-22)
------------------------------
Related modules with overlapping responsibilities:

- ``qa/tca.py``: cost_bps breakdown from ``trades_df`` (aggregate, not per-fill).
- ``qa/tca_arrival.py``: Sprint C11 arrival-IS sidecar. Same IS formula as
  the ``compute_trade_tca()`` function below (`(fill - arrival)/arrival * 10000
  * sign`).

Consolidation direction is pending Ownership/Call-Site-Analyse (see
`docs/roadmap/SYSTEM_CHECK_REMEDIATION_2026-04-22.md`, P2.1).
Until then: this module stays additive (per-trade, per-symbol aggregation),
``tca_arrival`` stays the canonical per-fill IS sidecar, ``tca.py`` stays
the cost_bps aggregator.

Previous docstring listed "Effective Spread" and "Timing-Cost"; neither is
implemented here — removed from the description to match reality.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TradeTCA:
    """TCA für einen einzelnen Trade."""

    trade_id: str
    symbol: str
    side: str  # "buy" / "sell"
    quantity: float
    arrival_price: float
    execution_price: float
    vwap_price: float = 0.0
    implementation_shortfall_bps: float = 0.0
    vwap_slippage_bps: float = 0.0
    total_cost_bps: float = 0.0


@dataclass
class TCAAggregateReport:
    """Aggregierte TCA-Statistik."""

    n_trades: int = 0
    mean_impact_bps: float = 0.0
    median_impact_bps: float = 0.0
    mean_vwap_slippage_bps: float = 0.0
    total_cost_bps: float = 0.0
    per_symbol: dict = field(default_factory=dict)
    per_strategy: dict = field(default_factory=dict)


def compute_trade_tca(
    trade_id: str,
    symbol: str,
    side: str,
    quantity: float,
    execution_price: float,
    arrival_price: float,
    vwap_price: float | None = None,
) -> TradeTCA:
    """Berechnet TCA für einen einzelnen Trade.

    Args:
        trade_id: Eindeutiger Trade-Identifier
        symbol: Ticker
        side: "buy" oder "sell"
        quantity: Menge
        execution_price: Tatsächlicher Exec-Price
        arrival_price: Mid-Price zum Zeitpunkt der Order
        vwap_price: Optional VWAP der Session; None = no VWAP-slip computation
    """
    if arrival_price <= 0:
        return TradeTCA(
            trade_id=trade_id, symbol=symbol, side=side, quantity=quantity,
            arrival_price=arrival_price, execution_price=execution_price,
        )

    side_multiplier = 1.0 if side.lower() == "buy" else -1.0

    # IS: (exec - arrival) × sign; buy paid more than arrival → positive cost
    is_bps = float(side_multiplier * (execution_price - arrival_price) / arrival_price * 10000.0)

    vwap_slip_bps = 0.0
    if vwap_price is not None and vwap_price > 0:
        vwap_slip_bps = float(side_multiplier * (execution_price - vwap_price) / vwap_price * 10000.0)

    total_cost = is_bps  # einfach, könnte erweitert werden

    return TradeTCA(
        trade_id=trade_id,
        symbol=symbol,
        side=side,
        quantity=quantity,
        arrival_price=arrival_price,
        execution_price=execution_price,
        vwap_price=float(vwap_price) if vwap_price else 0.0,
        implementation_shortfall_bps=round(is_bps, 2),
        vwap_slippage_bps=round(vwap_slip_bps, 2),
        total_cost_bps=round(total_cost, 2),
    )


def aggregate_tca(tcas: list[TradeTCA]) -> TCAAggregateReport:
    """Erstellt Aggregate-Report pro Symbol/Strategy."""
    if not tcas:
        return TCAAggregateReport()

    is_arr = np.array([t.implementation_shortfall_bps for t in tcas])
    vwap_arr = np.array([t.vwap_slippage_bps for t in tcas])

    per_symbol: dict[str, dict] = {}
    for t in tcas:
        per_symbol.setdefault(t.symbol, {"n": 0, "mean_is_bps": 0.0, "total_cost_bps": 0.0})
        per_symbol[t.symbol]["n"] += 1
        per_symbol[t.symbol]["mean_is_bps"] += t.implementation_shortfall_bps
        per_symbol[t.symbol]["total_cost_bps"] += t.total_cost_bps
    for sym, stats in per_symbol.items():
        stats["mean_is_bps"] = round(stats["mean_is_bps"] / stats["n"], 2)
        stats["total_cost_bps"] = round(stats["total_cost_bps"], 2)

    return TCAAggregateReport(
        n_trades=len(tcas),
        mean_impact_bps=round(float(np.mean(is_arr)), 2),
        median_impact_bps=round(float(np.median(is_arr)), 2),
        mean_vwap_slippage_bps=round(float(np.mean(vwap_arr)), 2),
        total_cost_bps=round(float(np.sum(is_arr)), 2),
        per_symbol=per_symbol,
    )


def run_tca_from_learning_store(
    learning_store_path: Path,
    output_path: Path | None = None,
) -> dict:
    """Liest learning_store.jsonl und erstellt TCA-Report.

    Erwartet Records mit: trade_id, symbol, side, quantity, execution_price,
    arrival_price (oder open_price), optional vwap_price.

    Args:
        learning_store_path: JSONL-File
        output_path: Optional — wo der Report geschrieben wird

    Returns:
        dict mit aggregiertem Report, oder {} wenn keine Daten.
    """
    if not learning_store_path.exists():
        return {}

    tcas: list[TradeTCA] = []
    with learning_store_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                exec_price = rec.get("execution_price") or rec.get("exec_price")
                arrival = rec.get("arrival_price") or rec.get("open_price") or rec.get("decision_price")
                if not exec_price or not arrival:
                    continue
                tcas.append(compute_trade_tca(
                    trade_id=str(rec.get("trade_id", rec.get("id", ""))),
                    symbol=str(rec.get("symbol", "")),
                    side=str(rec.get("side", "buy")),
                    quantity=float(rec.get("quantity", 0.0)),
                    execution_price=float(exec_price),
                    arrival_price=float(arrival),
                    vwap_price=float(rec.get("vwap_price")) if rec.get("vwap_price") else None,
                ))
            except Exception:
                continue

    if not tcas:
        return {}

    report = aggregate_tca(tcas)
    report_dict = {
        "n_trades": report.n_trades,
        "mean_impact_bps": report.mean_impact_bps,
        "median_impact_bps": report.median_impact_bps,
        "mean_vwap_slippage_bps": report.mean_vwap_slippage_bps,
        "total_cost_bps": report.total_cost_bps,
        "per_symbol": report.per_symbol,
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report_dict, indent=2, default=str), encoding="utf-8"
        )
        logger.info("[TCA] Report geschrieben: %s", output_path)

    return report_dict


__all__ = [
    "TradeTCA",
    "TCAAggregateReport",
    "compute_trade_tca",
    "aggregate_tca",
    "run_tca_from_learning_store",
]
