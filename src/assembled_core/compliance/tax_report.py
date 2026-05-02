"""Annual tax-report generator stubs.

From 50_COMPLIANCE_RECHT.md §50.1.

The full DB-backed implementation (with async PostgreSQL and FIFO lot matching)
lives in accounting/tax_lots.py.  This module provides:

- A pure-Python ``TaxReportSummary`` dataclass for aggregation.
- ``summarize_closed_lots()`` to produce a summary from a list of lot dicts.
- Constants for German Abgeltungsteuer (25 % + 5.5 % Soli = 26.375 %).

The annual tax report (Anlage KAP) requires:
  - Total realized P&L in EUR for the tax year
  - Split between gains and losses (Verlustverrechnungstopf)
  - Estimated tax after Sparer-Pauschbetrag
  - Count of trades and holding days
"""
from __future__ import annotations

from dataclasses import dataclass, field

ABGELTUNGSTEUER_RATE: float = 0.25
SOLIDARITAETSZUSCHLAG_RATE: float = 0.055
EFFECTIVE_TAX_RATE: float = ABGELTUNGSTEUER_RATE * (1 + SOLIDARITAETSZUSCHLAG_RATE)  # 0.26375
SPARER_PAUSCHBETRAG_EUR: float = 1_000.0  # 2026, single person


@dataclass
class TaxReportSummary:
    """Summary suitable for Anlage KAP."""

    year: int
    trade_count: int = 0
    wins_count: int = 0
    losses_count: int = 0
    total_realized_pnl_eur: float = 0.0
    total_wins_eur: float = 0.0
    total_losses_eur: float = 0.0
    taxable_pnl_eur: float = 0.0          # after Sparer-Pauschbetrag
    estimated_tax_eur: float = 0.0
    effective_tax_rate: float = EFFECTIVE_TAX_RATE
    notes: list[str] = field(default_factory=list)


def summarize_closed_lots(
    closed_lots: list[dict],
    year: int,
    sparer_pauschbetrag: float = SPARER_PAUSCHBETRAG_EUR,
) -> TaxReportSummary:
    """Build a tax report summary from a list of closed lot records.

    Each dict in *closed_lots* must have at least:
    - ``realized_pnl_eur``: float — realized P&L in EUR for this lot
    - ``trade_date``: :class:`datetime.date` — closing date

    Only lots whose ``trade_date.year`` equals *year* are included.
    """
    relevant = [
        lot for lot in closed_lots
        if hasattr(lot.get("trade_date"), "year")
        and lot["trade_date"].year == year
    ]

    total = sum(float(lot.get("realized_pnl_eur") or 0.0) for lot in relevant)
    wins = [lot for lot in relevant if float(lot.get("realized_pnl_eur") or 0.0) > 0]
    losses = [lot for lot in relevant if float(lot.get("realized_pnl_eur") or 0.0) < 0]

    taxable = max(0.0, total - sparer_pauschbetrag)
    estimated_tax = taxable * EFFECTIVE_TAX_RATE

    summary = TaxReportSummary(
        year=year,
        trade_count=len(relevant),
        wins_count=len(wins),
        losses_count=len(losses),
        total_realized_pnl_eur=total,
        total_wins_eur=sum(float(l.get("realized_pnl_eur") or 0.0) for l in wins),
        total_losses_eur=sum(float(l.get("realized_pnl_eur") or 0.0) for l in losses),
        taxable_pnl_eur=taxable,
        estimated_tax_eur=estimated_tax,
    )

    if total <= 0:
        summary.notes.append("Verlustjahr — Verlustverrechnungstopf prüfen.")
    if total > sparer_pauschbetrag:
        summary.notes.append(
            f"Sparer-Pauschbetrag ({sparer_pauschbetrag:.0f} EUR) verbraucht."
        )

    return summary
