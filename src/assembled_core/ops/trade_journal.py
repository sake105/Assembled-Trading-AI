"""Trade Journal — Per-trade logging and daily summary generation.

Append-only JSONL journal at output/journal/trade_journal.jsonl.
Each fill gets a journal entry with signal context, P&L tracking,
and exit reason (for sells).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_JOURNAL_PATH = Path("output/journal/trade_journal.jsonl")


def _next_trade_id(date_str: str, journal_path: Path) -> str:
    """Generate sequential trade ID for the day: TJ-YYYYMMDD-NNN."""
    count = 0
    prefix = f"TJ-{date_str.replace('-', '')}-"
    if journal_path.exists():
        try:
            for line in journal_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get("trade_id", "").startswith(prefix):
                        count += 1
                except json.JSONDecodeError:
                    continue
        except Exception as exc:
            import logging as _logging
            _logging.getLogger(__name__).warning("[TradeJournal] Could not read journal for ID sequencing: %s", exc)
    return f"{prefix}{count + 1:03d}"


def append_trade_journal_entry(
    fill: dict[str, Any],
    signal_context: dict[str, Any] | None = None,
    ledger_state: dict[str, Any] | None = None,
    run_id: str = "",
    journal_path: Path | str | None = None,
) -> dict[str, Any]:
    """Append a single trade entry to the journal.

    Args:
        fill: Dict with symbol, side, qty, price.
        signal_context: Optional dict with score, reason, regime, etc.
        ledger_state: Ledger state (for P&L calc on sells).
        run_id: Current run identifier.
        journal_path: Path to journal file.

    Returns:
        The journal entry dict.
    """
    jpath = Path(journal_path) if journal_path else DEFAULT_JOURNAL_PATH
    jpath.parent.mkdir(parents=True, exist_ok=True)

    now_utc = datetime.now(timezone.utc).isoformat()
    date_str = now_utc[:10]
    symbol = fill.get("symbol", "")
    side = str(fill.get("side", "BUY")).upper()
    qty = float(fill.get("qty", 0))
    price = float(fill.get("price", 0))
    notional = qty * price

    ctx = signal_context or {}
    entry: dict[str, Any] = {
        "trade_id": _next_trade_id(date_str, jpath),
        "timestamp_utc": now_utc,
        "symbol": symbol,
        "side": side,
        "qty": qty,
        "fill_price": price,
        "notional": round(notional, 2),
        "signal_score": ctx.get("score"),
        "signal_reason": ctx.get("reason"),
        "run_id": run_id,
    }

    # For sells: compute realized P&L from ledger
    if side == "SELL" and ledger_state:
        positions = ledger_state.get("positions") or {}
        pos = positions.get(symbol, {})
        avg_price = float(pos.get("avg_price", 0))
        if avg_price > 0:
            entry["entry_price"] = avg_price
            entry["pnl_dollar"] = round((price - avg_price) * qty, 2)
            entry["pnl_pct"] = round((price - avg_price) / avg_price * 100, 2)
            entry["exit_reason"] = ctx.get("exit_reason", "signal")

    try:
        with open(jpath, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except Exception as exc:
        logger.warning("[TradeJournal] failed to write entry: %s", exc)

    return entry


def append_trade_journal_entries(
    fills: list[dict[str, Any]],
    signal_context: dict[str, Any] | None = None,
    ledger_state: dict[str, Any] | None = None,
    run_id: str = "",
    journal_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Append multiple trade entries to the journal."""
    entries = []
    for fill in fills:
        entry = append_trade_journal_entry(
            fill,
            signal_context=signal_context,
            ledger_state=ledger_state,
            run_id=run_id,
            journal_path=journal_path,
        )
        entries.append(entry)
    return entries


def load_trade_journal(
    journal_path: Path | str | None = None,
    days: int | None = None,
) -> list[dict[str, Any]]:
    """Load trade journal entries."""
    jpath = Path(journal_path) if journal_path else DEFAULT_JOURNAL_PATH
    if not jpath.exists():
        return []

    entries = []
    for line in jpath.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    if days and entries:
        from datetime import timedelta

        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        entries = [e for e in entries if e.get("timestamp_utc", "") >= cutoff]

    return entries


def write_daily_summary(
    date_str: str,
    ledger_state: dict[str, Any],
    equity: float,
    start_capital: float,
    fills: list[dict[str, Any]] | None = None,
    journal_path: Path | str | None = None,
    output_dir: Path | str | None = None,
) -> Path | None:
    """Write a human-readable daily trading summary.

    Args:
        date_str: Date string YYYY-MM-DD.
        ledger_state: Current ledger state.
        equity: Current portfolio equity.
        start_capital: Starting capital.
        fills: Today's fills.
        journal_path: Path to trade journal (for P&L lookups).
        output_dir: Directory for summary file.

    Returns:
        Path to summary file, or None on error.
    """
    out_dir = Path(output_dir) if output_dir else Path("output/journal")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"summary_{date_str}.txt"

    positions = ledger_state.get("positions") or {}
    cash = float(ledger_state.get("cash", 0))
    invested = sum(
        float(p.get("qty", 0)) * float(p.get("avg_price", 0))
        for p in positions.values()
    )
    invested_pct = invested / equity * 100 if equity > 0 else 0
    total_return = (equity - start_capital) / start_capital * 100 if start_capital > 0 else 0

    lines = [
        f"{'=' * 55}",
        f"  Paper Trading Summary — {date_str}",
        f"{'=' * 55}",
        "",
        f"  Portfolio:  ${equity:>12,.2f}  ({total_return:+.2f}%)",
        f"  Invested:   ${invested:>12,.2f}  ({invested_pct:.1f}%)",
        f"  Cash:       ${cash:>12,.2f}",
        f"  Positions:  {len(positions)}",
        "",
    ]

    # Today's fills
    if fills:
        buys = [f for f in fills if str(f.get("side", "")).upper() == "BUY"]
        sells = [f for f in fills if str(f.get("side", "")).upper() == "SELL"]

        if buys:
            lines.append("  NEW / INCREASED POSITIONS:")
            for f in buys:
                sym = f.get("symbol", "?")
                qty = float(f.get("qty", 0))
                px = float(f.get("price", 0))
                notional = qty * px
                lines.append(f"    {sym:<6} BUY  {qty:>8.2f} @ ${px:>8.2f}  (${notional:>10,.2f})")
            lines.append("")

        if sells:
            lines.append("  CLOSED / REDUCED POSITIONS:")
            for f in sells:
                sym = f.get("symbol", "?")
                qty = float(f.get("qty", 0))
                px = float(f.get("price", 0))
                notional = qty * px
                lines.append(f"    {sym:<6} SELL {qty:>8.2f} @ ${px:>8.2f}  (${notional:>10,.2f})")
            lines.append("")

    # Open positions
    if positions:
        lines.append("  OPEN POSITIONS:")
        lines.append(f"    {'Symbol':<6} {'Qty':>8} {'AvgPx':>10} {'Notional':>12}")
        lines.append(f"    {'-' * 40}")
        for sym, pos in sorted(positions.items()):
            qty = float(pos.get("qty", 0))
            avg = float(pos.get("avg_price", 0))
            notional = qty * avg
            lines.append(f"    {sym:<6} {qty:>8.2f} ${avg:>9.2f} ${notional:>11,.2f}")
        lines.append("")

    lines.append(f"{'=' * 55}")

    try:
        summary_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("[TradeJournal] summary written: %s", summary_path)
        return summary_path
    except Exception as exc:
        logger.warning("[TradeJournal] failed to write summary: %s", exc)
        return None
