# src/assembled_core/api/routers/ledger.py
"""Paper ledger endpoint (GO_LIVE F2).

Exposes the running paper-pilot's JSON ledger as a structured API response.
Read-only.  Never raises 500 or 404 when the ledger file is simply absent —
returns status='no_ledger' with zeroed fields instead (frontend can handle
cold-start gracefully).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Query

from src.assembled_core.api.models import LedgerPosition, LedgerResponse
from src.assembled_core.config import OUTPUT_DIR
from src.assembled_core.logging_utils import get_logger

router = APIRouter()
logger = get_logger(__name__)

_DEFAULT_LEDGER = str(OUTPUT_DIR / "runs/_paper_ledger/ledger_state.json")


@router.get("/ledger", response_model=LedgerResponse)
def get_ledger(
    date: Optional[str] = Query(
        default=None,
        description="Historical date lookup (YYYY-MM-DD). Returns equity at that date.",
    ),
    ledger_path: str = Query(
        default=_DEFAULT_LEDGER,
        description="Path to the paper ledger JSON state file.",
    ),
) -> LedgerResponse:
    """Return tagesaktueller Ledgerstand from the paper pilot.

    When the ledger file is absent (pilot not yet started) the response has
    ``status='no_ledger'`` with all numeric fields set to 0 — never 404/500.

    Optional ``?date=YYYY-MM-DD`` filters the equity value to that day's
    equity_curve entry.  Positions are always the most recent state (no
    position history stored in the JSON ledger).

    The ``unrealized_pnl_approx`` field is:
        equity − cash − sum(abs(qty) × avg_price)
    This approximation uses the last mark-to-market equity; it is NOT a live
    market quote.
    """
    jpath = Path(ledger_path)
    if not jpath.is_absolute():
        # Resolve relative to repo root (same convention as paper_runner)
        jpath = OUTPUT_DIR.parent / ledger_path

    if not jpath.exists():
        return LedgerResponse(
            status="no_ledger",
            as_of=None,
            cash=0.0,
            equity=0.0,
            n_positions=0,
            positions=[],
            unrealized_pnl_approx=None,
            date_requested=date,
        )

    try:
        from src.assembled_core.ops.paper_ledger import load_ledger_state

        # Use sentinel start_capital so a full-fallback state is detectable:
        # _fresh_state(-1.0) → cash<0 AND updated_utc=None → all candidates failed.
        state = load_ledger_state(jpath, start_capital=-1.0)
    except Exception as exc:
        logger.warning("[Ledger] failed to load ledger state from %s: %s", jpath, exc)
        return LedgerResponse(
            status="no_ledger",
            as_of=None,
            cash=0.0,
            equity=0.0,
            n_positions=0,
            positions=[],
            unrealized_pnl_approx=None,
            date_requested=date,
        )

    # Detect silent loader fallback: _fresh_state(-1.0) ↔ all candidates unreadable.
    # A real ledger always has updated_utc set by save_ledger_state.
    if state.get("updated_utc") is None and (state.get("cash") or 0) < 0:
        logger.warning(
            "[Ledger] possible corrupt ledger at %s — all candidates unreadable", jpath
        )
        return LedgerResponse(
            status="no_ledger",
            as_of=None,
            cash=0.0,
            equity=0.0,
            n_positions=0,
            positions=[],
            unrealized_pnl_approx=None,
            date_requested=date,
        )

    equity_curve: list[dict] = state.get("equity_curve") or []
    cash = round(float(state.get("cash") or 0.0), 2)

    # Resolve equity (optionally filtered to requested date)
    if date:
        matched = [e for e in equity_curve if str(e.get("utc", ""))[:10] == date]
        if not matched:
            # No equity_curve entry for the requested date — return zeroed fields
            # so the caller receives a self-consistent response (equity < cash violation avoided).
            return LedgerResponse(
                status="ok",
                as_of=None,
                cash=0.0,
                equity=0.0,
                n_positions=0,
                positions=[],
                unrealized_pnl_approx=None,
                date_requested=date,
            )
        equity_val = float(matched[-1]["equity"])
        as_of = matched[-1].get("utc")
    else:
        as_of = state.get("updated_utc")
        equity_val = float(equity_curve[-1]["equity"]) if equity_curve else cash

    equity_val = round(equity_val, 2)

    # Build positions list
    raw_positions: dict = state.get("positions") or {}
    positions: list[LedgerPosition] = []
    total_cost_basis = 0.0

    for sym, pos in raw_positions.items():
        qty = float(pos.get("qty") or 0.0)
        if qty == 0.0:
            continue
        avg_price = float(pos.get("avg_price") or 0.0)
        cb = round(abs(qty) * avg_price, 2)
        total_cost_basis += cb
        positions.append(
            LedgerPosition(
                symbol=sym,
                qty=qty,
                avg_price=avg_price,
                cost_basis=cb,
            )
        )

    positions.sort(key=lambda p: p.symbol)

    # Approx unrealized PnL: equity - cash - cost_basis
    unrealized_pnl_approx: Optional[float] = None
    if equity_val and positions:
        unrealized_pnl_approx = round(equity_val - cash - total_cost_basis, 2)

    return LedgerResponse(
        status="ok",
        as_of=as_of,
        cash=cash,
        equity=equity_val,
        n_positions=len(positions),
        positions=positions,
        unrealized_pnl_approx=unrealized_pnl_approx,
        date_requested=date,
    )
