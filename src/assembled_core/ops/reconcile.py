"""OPS-5: Reconcile report and invariants for paper ledger runs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from src.assembled_core.ops.paper_ledger import mark_to_market_equity

SCHEMA_VERSION = "run.reconcile.v1"


def _to_fills_list(fills: list[dict[str, Any]] | pd.DataFrame) -> list[dict[str, Any]]:
    if isinstance(fills, pd.DataFrame):
        if fills.empty:
            return []
        return fills.to_dict("records")
    return list(fills) if fills else []


def _to_orders_list(
    orders: list[dict[str, Any]] | pd.DataFrame,
) -> list[dict[str, Any]]:
    if isinstance(orders, pd.DataFrame):
        if orders.empty:
            return []
        return orders.to_dict("records")
    return list(orders) if orders else []


def _isfinite(x: float) -> bool:
    return math.isfinite(x) if isinstance(x, (int, float)) else False


def build_reconcile_report(
    as_of_utc: str,
    ledger_before: dict[str, Any],
    ledger_after: dict[str, Any],
    orders: list[dict[str, Any]] | pd.DataFrame,
    fills: list[dict[str, Any]] | pd.DataFrame,
    prices_latest: pd.DataFrame,
    cost_model_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build reconcile report (schema run.reconcile.v1) with cash/equity deltas, trading stats, invariants."""
    cost_model_cfg = cost_model_cfg or {}
    orders_list = _to_orders_list(orders)
    fills_list = _to_fills_list(fills)

    cash_before = float(ledger_before.get("cash", 0))
    cash_after = float(ledger_after.get("cash", 0))
    equity_before = mark_to_market_equity(ledger_before, prices_latest)
    equity_after = mark_to_market_equity(ledger_after, prices_latest)

    pos_before = ledger_before.get("positions") or {}
    pos_after = ledger_after.get("positions") or {}
    n_before = len([p for p in pos_before.values() if float(p.get("qty", 0)) != 0])
    n_after = len([p for p in pos_after.values() if float(p.get("qty", 0)) != 0])

    notional_traded = sum(
        float(f.get("qty", 0)) * float(f.get("price", 0)) for f in fills_list
    )
    commission_bps = float(cost_model_cfg.get("commission_bps", 0) or 0)
    estimated_costs = (
        (notional_traded * commission_bps / 10000.0) if notional_traded else 0.0
    )
    slippage_bps = float(cost_model_cfg.get("slippage_bps", 0) or 0)

    order_symbols = set()
    for o in orders_list:
        s = o.get("symbol")
        if s is not None and not (isinstance(s, float) and math.isnan(s)):
            order_symbols.add(str(s).strip())
    fill_symbols = set(f.get("symbol") for f in fills_list if f.get("symbol"))
    fills_match_orders = (
        len(fills_list) <= len(orders_list) and fill_symbols <= order_symbols
    )

    invariants: list[dict[str, Any]] = []
    cash_ok = cash_after >= -1e-6
    invariants.append({"name": "cash_non_negative", "ok": cash_ok, "value": cash_after})
    eq_before_ok = _isfinite(equity_before)
    invariants.append(
        {
            "name": "equity_finite",
            "ok": eq_before_ok and _isfinite(equity_after),
            "value": {"before": equity_before, "after": equity_after},
        }
    )
    positions_finite_ok = True
    for sym, p in list(pos_after.items()) + list(pos_before.items()):
        q = p.get("qty")
        try:
            qf = float(q) if q is not None else 0.0
        except (TypeError, ValueError):
            positions_finite_ok = False
            break
        if not _isfinite(qf):
            positions_finite_ok = False
            break
    invariants.append(
        {"name": "positions_finite", "ok": positions_finite_ok, "value": n_after}
    )
    invariants.append(
        {
            "name": "fills_match_orders",
            "ok": fills_match_orders,
            "value": {"n_orders": len(orders_list), "n_fills": len(fills_list)},
        }
    )

    any_fail = not (
        cash_ok
        and eq_before_ok
        and _isfinite(equity_after)
        and positions_finite_ok
        and fills_match_orders
    )
    status = "FAIL" if any_fail else "OK"
    notes: list[str] = []
    if any_fail:
        for inv in invariants:
            if not inv.get("ok"):
                notes.append(f"invariant_{inv['name']}_failed")

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": as_of_utc,
        "status": status,
        "cash": {
            "before": cash_before,
            "after": cash_after,
            "delta": cash_after - cash_before,
        },
        "equity": {
            "before": equity_before,
            "after": equity_after,
            "delta": equity_after - equity_before,
        },
        "positions": {"n_before": n_before, "n_after": n_after},
        "trading": {
            "n_orders": len(orders_list),
            "n_fills": len(fills_list),
            "notional_traded": notional_traded,
            "estimated_costs": estimated_costs,
            "avg_slippage_bps": slippage_bps,
        },
        "invariants": invariants,
        "notes": notes,
    }
    return report


def write_reconcile_artifact(output_dir: str | Path, report: dict[str, Any]) -> Path:
    """Write output_dir/reconcile_latest.json atomically (schema run.reconcile.v1)."""
    out_dir = Path(output_dir)
    path = out_dir / "reconcile_latest.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    tmp.replace(path)
    return path
