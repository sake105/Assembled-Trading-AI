# src/assembled_core/api/routers/trades.py
"""Trade journal and explanation endpoints (V2 mockup)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from src.assembled_core.api.models import TradeExplanationResponse, TradeJournalEntry, TradeJournalResponse
from src.assembled_core.logging_utils import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.get("/trades/journal", response_model=TradeJournalResponse)
def get_trade_journal(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    symbol: str | None = Query(default=None),
    days: int | None = Query(default=None, description="Filter to last N days"),
) -> TradeJournalResponse:
    """Load trade journal entries with optional pagination and filtering."""
    try:
        from src.assembled_core.ops.trade_journal import load_trade_journal
        entries_raw = load_trade_journal(days=days)

        if symbol:
            sym_upper = symbol.upper()
            entries_raw = [e for e in entries_raw if e.get("symbol", "").upper() == sym_upper]

        total = len(entries_raw)
        page = entries_raw[offset : offset + limit]

        entries = []
        for raw in page:
            known = {
                "trade_id", "timestamp_utc", "symbol", "side", "qty",
                "fill_price", "notional", "signal_score", "signal_reason", "run_id",
            }
            extra = {k: v for k, v in raw.items() if k not in known}
            entries.append(TradeJournalEntry(
                trade_id=raw.get("trade_id"),
                timestamp_utc=raw.get("timestamp_utc"),
                symbol=raw.get("symbol"),
                side=raw.get("side"),
                qty=raw.get("qty"),
                fill_price=raw.get("fill_price"),
                notional=raw.get("notional"),
                signal_score=raw.get("signal_score"),
                signal_reason=raw.get("signal_reason"),
                run_id=raw.get("run_id"),
                extra=extra,
            ))

        return TradeJournalResponse(total=total, limit=limit, offset=offset, entries=entries)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("trade journal error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/trades/{trade_id}/explanation", response_model=TradeExplanationResponse)
def get_trade_explanation(trade_id: str) -> TradeExplanationResponse:
    """Return factor attribution and reasoning for a specific trade."""
    try:
        from src.assembled_core.ops.trade_journal import load_trade_journal
        from pathlib import Path
        import json

        # Search in trade_journal (JSONL) first
        entries = load_trade_journal()
        match = next((e for e in entries if e.get("trade_id") == trade_id), None)

        if match is None:
            # Also search journal JSONL files in output/
            from src.assembled_core.config import OUTPUT_DIR
            for jf in sorted((OUTPUT_DIR / "trade_journal").rglob("*.jsonl"), reverse=True):
                for line in jf.read_text(encoding="utf-8").splitlines():
                    try:
                        e = json.loads(line)
                        if e.get("trade_id") == trade_id or e.get("order_id") == trade_id:
                            match = e
                            break
                    except Exception:
                        pass
                if match:
                    break

        if match is None:
            return TradeExplanationResponse(
                trade_id=trade_id,
                found=False,
                reasoning_text=f"Trade {trade_id!r} not found in journal.",
            )

        # Build reasoning text from available fields
        lines = []
        sym = match.get("symbol", "?")
        side = match.get("side", "?").upper() if match.get("side") else "?"
        qty = match.get("qty", "?")
        price = match.get("fill_price") or match.get("target_price", "?")
        ts = match.get("timestamp_utc") or match.get("timestamp", "?")
        lines.append(f"{ts}  {side} {qty} {sym} @ {price}")

        if match.get("signal_reason"):
            lines.append(f"Signal reason: {match['signal_reason']}")
        if match.get("signal_score") is not None:
            lines.append(f"Signal score: {match['signal_score']:.4f}")

        factor_keys = [k for k in match if k.startswith("factor_") or k in (
            "regime", "risk_state", "exposure_mult", "conviction")]
        factors = {k: match[k] for k in factor_keys}

        return TradeExplanationResponse(
            trade_id=trade_id,
            found=True,
            symbol=match.get("symbol"),
            timestamp=match.get("timestamp_utc") or match.get("timestamp"),
            side=match.get("side"),
            reasoning_text="\n".join(lines),
            factors=factors,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("trade explanation error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
