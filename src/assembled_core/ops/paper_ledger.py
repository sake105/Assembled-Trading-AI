"""OPS-4: Paper execution ledger — load/save state, simulate fills, apply to ledger, mark-to-market equity."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

SCHEMA_VERSION = "paper.ledger_state.v1"


def load_ledger_state(
    path: str | Path,
    start_capital: float = 10000.0,
) -> dict[str, Any]:
    """Load ledger state from JSON. If missing or invalid, return fresh state with cash=start_capital, positions empty."""
    p = Path(path)
    if not p.exists():
        return _fresh_state(start_capital)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return _fresh_state(start_capital)
    if not isinstance(data, dict):
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


def save_ledger_state(state: dict[str, Any], path: str | Path) -> Path:
    """Persist ledger state atomically (tmp file + rename)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    state = dict(state)
    state["updated_utc"] = datetime.now(timezone.utc).isoformat()
    state["schema_version"] = state.get("schema_version") or SCHEMA_VERSION
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=True), encoding="utf-8")
    tmp.replace(p)
    return p


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
        fill_price = fill_price + (commission_per_share if str(side).upper() == "BUY" else -commission_per_share)
        fills.append({
            "symbol": symbol,
            "side": str(side).upper() if side else "BUY",
            "qty": qty_f,
            "price": fill_price,
        })
    return fills


def apply_fills_to_ledger(state: dict[str, Any], fills: list[dict[str, Any]]) -> dict[str, Any]:
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
        "updated_utc": state.get("updated_utc") or datetime.now(timezone.utc).isoformat(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    tmp.replace(path)
    return path
