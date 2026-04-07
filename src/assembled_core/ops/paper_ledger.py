"""OPS-4: Paper execution ledger — load/save state, simulate fills, apply to ledger, mark-to-market equity.

Safety features:
- File locking via filelock (prevents concurrent writes)
- Backup rotation (3 generations: .1, .2, .3)
- JSON validation on load (falls back to backup if corrupted)
- Equity curve deduplication (prevents duplicate entries on re-run)
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "paper.ledger_state.v1"
_BACKUP_GENERATIONS = 3


def load_ledger_state(
    path: str | Path,
    start_capital: float = 10000.0,
) -> dict[str, Any]:
    """Load ledger state from JSON. If missing or corrupted, try backups, then return fresh state."""
    p = Path(path)

    # Try main file first, then backups (.1, .2, .3)
    candidates = [p] + [p.with_suffix(p.suffix + f".{i}") for i in range(1, _BACKUP_GENERATIONS + 1)]
    data = None
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            raw = candidate.read_text(encoding="utf-8")
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                data = parsed
                if candidate != p:
                    logger.warning(
                        "[paper_ledger] main file corrupted, loaded from backup: %s",
                        candidate,
                    )
                break
        except Exception as exc:
            logger.warning("[paper_ledger] failed to load %s: %s", candidate, exc)
            continue

    if data is None:
        return _fresh_state(start_capital)
    # Normalize schema
    cash = data.get("cash")
    try:
        cash = float(cash) if cash is not None else start_capital
    except (TypeError, ValueError):
        cash = start_capital
    positions = data.get("positions")
    if not isinstance(positions, dict):
        positions = {}
    equity_curve = data.get("equity_curve")
    if not isinstance(equity_curve, list):
        equity_curve = []
    return {
        "schema_version": data.get("schema_version") or SCHEMA_VERSION,
        "updated_utc": data.get("updated_utc"),
        "cash": cash,
        "positions": {str(k): _norm_position(v) for k, v in positions.items()},
        "equity_curve": list(equity_curve),
    }


def _norm_position(v: Any) -> dict[str, float]:
    if isinstance(v, dict):
        qty = v.get("qty")
        avg_price = v.get("avg_price")
        try:
            qty_f = float(qty) if qty is not None else 0.0
        except (TypeError, ValueError):
            qty_f = 0.0
        try:
            avg_f = float(avg_price) if avg_price is not None else 0.0
        except (TypeError, ValueError):
            avg_f = 0.0
        return {"qty": qty_f, "avg_price": avg_f}
    return {"qty": 0.0, "avg_price": 0.0}


def _fresh_state(start_capital: float) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "updated_utc": None,
        "cash": float(start_capital),
        "positions": {},
        "equity_curve": [],
    }


def _rotate_backups(p: Path) -> None:
    """Rotate backup files: .3 deleted, .2 → .3, .1 → .2, current → .1."""
    for i in range(_BACKUP_GENERATIONS, 1, -1):
        src = p.with_suffix(p.suffix + f".{i - 1}")
        dst = p.with_suffix(p.suffix + f".{i}")
        if src.exists():
            try:
                shutil.copy2(str(src), str(dst))
            except OSError:
                pass
    # Current → .1
    if p.exists():
        try:
            shutil.copy2(str(p), str(p.with_suffix(p.suffix + ".1")))
        except OSError:
            pass


def save_ledger_state(state: dict[str, Any], path: str | Path) -> Path:
    """Persist ledger state atomically with file locking and backup rotation.

    Safety measures:
    - Rotates previous state into .1, .2, .3 backup generations
    - Uses file lock to prevent concurrent writes
    - Writes to temp file then renames (atomic on most filesystems)
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    state = dict(state)
    state["updated_utc"] = datetime.now(timezone.utc).isoformat()
    state["schema_version"] = state.get("schema_version") or SCHEMA_VERSION

    lock_path = p.with_suffix(p.suffix + ".lock")
    try:
        from filelock import FileLock

        lock = FileLock(str(lock_path), timeout=10)
    except ImportError:
        lock = None
        logger.warning("[paper_ledger] filelock not installed, skipping file lock")

    def _do_save() -> Path:
        _rotate_backups(p)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(
            json.dumps(state, indent=2, ensure_ascii=True), encoding="utf-8"
        )
        tmp.replace(p)
        return p

    if lock is not None:
        with lock:
            return _do_save()
    else:
        return _do_save()


def simulate_fills(
    orders: pd.DataFrame,
    prices_latest: pd.DataFrame,
    cost_model_cfg: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Simulate fills: v1 fill at close (or next best), qty as requested; optional commission/slippage bps."""
    fills: list[dict[str, Any]] = []
    if orders.empty:
        return fills
    cost = cost_model_cfg or {}
    commission_bps = float(cost.get("commission_bps", 0) or 0)
    slippage_bps = float(cost.get("slippage_bps", 0) or 0)

    price_col = "close" if "close" in prices_latest.columns else "price"
    if "symbol" not in prices_latest.columns or price_col not in prices_latest.columns:
        return fills
    price_map = prices_latest.set_index("symbol")[price_col].to_dict()

    for _, row in orders.iterrows():
        symbol = row.get("symbol")
        if symbol is None or pd.isna(symbol):
            continue
        symbol = str(symbol).strip()
        side = row.get("side")
        qty = row.get("qty")
        try:
            qty_f = float(qty) if qty is not None else 0.0
        except (TypeError, ValueError):
            continue
        if qty_f <= 0:
            continue
        base_price = float(price_map.get(symbol, 0.0) or 0.0)
        if base_price <= 0:
            continue
        # Slippage: buy => pay more, sell => receive less
        slippage_mult = 1.0 + (slippage_bps / 10000.0)
        if str(side).upper() == "SELL":
            fill_price = base_price / slippage_mult
        else:
            fill_price = base_price * slippage_mult
        # Commission (per-share approximation)
        commission_per_share = base_price * (commission_bps / 10000.0)
        fill_price = fill_price + (
            commission_per_share
            if str(side).upper() == "BUY"
            else -commission_per_share
        )
        fills.append(
            {
                "symbol": symbol,
                "side": str(side).upper() if side else "BUY",
                "qty": qty_f,
                "price": fill_price,
            }
        )
    return fills


def apply_fills_to_ledger(
    state: dict[str, Any], fills: list[dict[str, Any]]
) -> dict[str, Any]:
    """Update cash and positions from fills. Positions: qty and avg_price (weighted). Returns new state (copy)."""
    out = {
        "schema_version": state.get("schema_version") or SCHEMA_VERSION,
        "updated_utc": state.get("updated_utc"),
        "cash": float(state.get("cash", 0)),
        "positions": {k: dict(v) for k, v in (state.get("positions") or {}).items()},
        "equity_curve": list(state.get("equity_curve") or []),
    }
    for f in fills:
        symbol = f.get("symbol")
        if not symbol:
            continue
        symbol = str(symbol).strip()
        side = str(f.get("side", "BUY")).upper()
        qty = float(f.get("qty", 0))
        price = float(f.get("price", 0))
        if qty <= 0 or price <= 0:
            continue
        pos = out["positions"].setdefault(symbol, {"qty": 0.0, "avg_price": 0.0})
        pos_qty = pos["qty"]
        pos_avg = pos["avg_price"]
        if side == "BUY":
            new_qty = pos_qty + qty
            new_avg = (pos_avg * pos_qty + price * qty) / new_qty if new_qty else 0.0
            out["positions"][symbol] = {"qty": new_qty, "avg_price": new_avg}
            out["cash"] -= qty * price
        else:
            new_qty = pos_qty - qty
            if new_qty <= 0:
                out["positions"].pop(symbol, None)
                out["cash"] += pos_qty * pos_avg if pos_qty > 0 else 0.0
                out["cash"] += (qty - pos_qty) * price if qty > pos_qty else qty * price
            else:
                out["positions"][symbol] = {"qty": new_qty, "avg_price": pos_avg}
                out["cash"] += qty * price
    return out


def append_equity_curve_deduped(
    state: dict[str, Any], utc_iso: str, equity: float
) -> None:
    """Append equity curve entry, deduplicating by date (not full timestamp).

    Prevents duplicate entries if the same day is run twice.
    """
    curve = state.setdefault("equity_curve", [])
    date_str = utc_iso[:10]  # Extract YYYY-MM-DD
    # Check if this date already exists
    for i, entry in enumerate(curve):
        existing_date = str(entry.get("utc", ""))[:10]
        if existing_date == date_str:
            # Replace existing entry for this date
            curve[i] = {"utc": utc_iso, "equity": equity}
            return
    # New date — append
    curve.append({"utc": utc_iso, "equity": equity})


def mark_to_market_equity(state: dict[str, Any], prices_latest: pd.DataFrame) -> float:
    """Compute equity = cash + sum(position qty * latest price)."""
    cash = float(state.get("cash", 0))
    positions = state.get("positions") or {}
    if not positions:
        return cash
    price_col = "close" if "close" in prices_latest.columns else "price"
    if "symbol" not in prices_latest.columns or price_col not in prices_latest.columns:
        return cash
    price_map = prices_latest.set_index("symbol")[price_col].to_dict()
    mtm = cash
    for sym, pos in positions.items():
        qty = float(pos.get("qty", 0))
        if qty == 0:
            continue
        p = price_map.get(sym)
        try:
            px = float(p) if p is not None else 0.0
        except (TypeError, ValueError):
            px = 0.0
        mtm += qty * px
    return mtm


def write_ledger_snapshot(
    output_dir: str | Path,
    state: dict[str, Any],
    equity: float,
) -> Path:
    """Write output_dir/ledger_snapshot.json (schema paper.ledger_snapshot.v1)."""
    out_dir = Path(output_dir)
    path = out_dir / "ledger_snapshot.json"
    payload: dict[str, Any] = {
        "schema_version": "paper.ledger_snapshot.v1",
        "cash": state.get("cash"),
        "positions": state.get("positions"),
        "equity": equity,
        "updated_utc": state.get("updated_utc")
        or datetime.now(timezone.utc).isoformat(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    tmp.replace(path)
    return path
