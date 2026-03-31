"""KPI artifact writer for paper/shadow runs (OPS-1).

This module computes and writes per-run KPI JSON artifacts that explain
why the system is (under-)invested, which overlays/gates were active,
and which triggers drove the current risk state.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import json
from datetime import datetime, timezone

import pandas as pd

from src.assembled_core.risk.georisk_overlay import compute_exposure_multiplier


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> Path:
    """Write JSON atomically via tmp file + rename."""
    out_dir = path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    data = json.dumps(payload, indent=2, ensure_ascii=True)
    tmp_path.write_text(data, encoding="utf-8")
    tmp_path.replace(path)
    return path


def write_run_kpis(
    output_dir: str | Path,
    ctx: Any,
    result: Any,
    policy: Dict[str, Any] | None,
    mode: str,
) -> Path:
    """Write per-run KPI artifact (run_kpis.json) to output_dir.

    The writer is defensive: missing fields are represented as null/{}
    rather than raising, so that operator artifacts are always written.
    """
    out_dir = Path(output_dir)
    path = out_dir / "run_kpis.json"

    policy = policy or {}

    # Multipliers
    try:
        georisk_mult = compute_exposure_multiplier(ctx, policy)
    except Exception:
        georisk_mult = 1.0

    profit_lock_meta = (getattr(result, "meta", {}) or {}).get("profit_lock") or {}
    profit_lock_mult = _safe_float(profit_lock_meta.get("multiplier")) or 1.0

    turnover_meta = (getattr(result, "meta", {}) or {}).get("turnover_budget") or {}
    turnover_scale = _safe_float(turnover_meta.get("scale_factor")) or 1.0

    final_exposure_mult = float(georisk_mult) * float(profit_lock_mult)

    # Risk-state related
    risk_state = getattr(ctx, "risk_state", None)
    news_geo = getattr(ctx, "news_geo", None)
    market_stress = getattr(ctx, "market_stress", None)

    news_triggers = getattr(ctx, "news_triggers", None)
    if news_triggers is not None and hasattr(news_triggers, "summary"):
        triggers_summary = news_triggers.summary  # type: ignore[assignment]
    else:
        triggers_summary = {}

    if news_geo is not None and hasattr(news_geo, "top_triggers"):
        top_triggers = list(getattr(news_geo, "top_triggers"))  # type: ignore[arg-type]
    else:
        top_triggers = []

    # Targets summary
    target_positions = getattr(result, "target_positions", None)
    if isinstance(target_positions, pd.DataFrame) and not target_positions.empty:
        n_targets = int(len(target_positions))
        if "target_weight" in target_positions.columns:
            sum_target_weight = float(
                pd.to_numeric(target_positions["target_weight"], errors="coerce")
                .fillna(0.0)
                .sum()
            )
        else:
            sum_target_weight = None
    else:
        n_targets = 0
        sum_target_weight = None

    payload: Dict[str, Any] = {
        "schema_version": "run.kpis.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "mode": str(mode),
        "risk_state": risk_state,
        "news_geo": news_geo,
        "market_stress": market_stress,
        "multipliers": {
            "georisk": float(georisk_mult),
            "profit_lock": float(profit_lock_mult),
            "final_exposure_multiplier": float(final_exposure_mult),
            "turnover_scale_factor": float(turnover_scale),
        },
        "turnover_budget": turnover_meta or None,
        "profit_lock": profit_lock_meta or None,
        "triggers_summary": triggers_summary,
        "top_triggers": top_triggers,
        "targets_summary": {
            "n_targets": n_targets,
            "sum_target_weight": sum_target_weight,
        },
        "intel_orchestration": (getattr(result, "meta", None) or {}).get(
            "intel_orchestration"
        ),
        "news_triggers_summary": (getattr(result, "meta", None) or {}).get(
            "news_triggers_summary"
        ),
        "disclosures_triggers_summary": (getattr(result, "meta", None) or {}).get(
            "disclosures_triggers_summary"
        ),
        "news_debug_funnel": (getattr(result, "meta", None) or {}).get(
            "news_debug_funnel"
        ),
    }

    return _atomic_write_json(path, payload)


def write_targets_artifact(
    output_dir: str | Path,
    target_positions: Any,
) -> Path:
    """Write final target positions for the run (run.targets.v1)."""
    out_dir = Path(output_dir)
    path = out_dir / "targets_latest.json"

    items: list[Dict[str, Any]] = []
    if isinstance(target_positions, pd.DataFrame) and not target_positions.empty:
        has_weight = "target_weight" in target_positions.columns
        has_qty = "target_qty" in target_positions.columns
        for _, row in target_positions.iterrows():
            item: Dict[str, Any] = {"symbol": row.get("symbol")}
            if has_weight:
                item["target_weight"] = _safe_float(row.get("target_weight"))
            if has_qty:
                item["target_qty"] = _safe_float(row.get("target_qty"))
            items.append(item)

    payload: Dict[str, Any] = {
        "schema_version": "run.targets.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "items": items,
    }
    return _atomic_write_json(path, payload)


def write_orders_artifact(
    output_dir: str | Path,
    orders: Any,
) -> Path:
    """Write orders generated by the run (run.orders.v1)."""
    out_dir = Path(output_dir)
    path = out_dir / "orders_latest.json"

    items: list[Dict[str, Any]] = []
    if isinstance(orders, pd.DataFrame) and not orders.empty:
        for _, row in orders.iterrows():
            item: Dict[str, Any] = {}
            for key in ("timestamp", "symbol", "side", "qty", "price"):
                if key in row.index:
                    val = row.get(key)
                    if key == "timestamp" and pd.notna(val):
                        try:
                            item[key] = pd.to_datetime(val).isoformat()
                        except Exception:  # pragma: no cover - defensive
                            item[key] = str(val)
                    elif key in ("qty", "price"):
                        item[key] = _safe_float(val)
                    else:
                        item[key] = val
            items.append(item)

    payload: Dict[str, Any] = {
        "schema_version": "run.orders.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "items": items,
    }
    return _atomic_write_json(path, payload)


def write_reasons_artifact(
    output_dir: str | Path,
    ctx: Any,
    result: Any,
    policy: Dict[str, Any] | None,
    mode: str,
) -> Path:
    """Write human-readable reason codes / gate explanations (run.reasons.v1)."""
    out_dir = Path(output_dir)
    path = out_dir / "reasons_latest.json"

    policy = policy or {}

    # Multipliers (reuse from KPI logic)
    try:
        georisk_mult = compute_exposure_multiplier(ctx, policy)
    except Exception:
        georisk_mult = 1.0

    meta = getattr(result, "meta", {}) or {}
    profit_lock_meta = meta.get("profit_lock") or {}
    turnover_meta = meta.get("turnover_budget") or {}

    profit_lock_mult = _safe_float(profit_lock_meta.get("multiplier")) or 1.0
    turnover_scale = _safe_float(turnover_meta.get("scale_factor")) or 1.0
    final_mult = float(georisk_mult) * float(profit_lock_mult)

    # Risk state + reasons
    risk_state_obj = getattr(ctx, "risk_state", None)
    if isinstance(risk_state_obj, dict):
        risk_state_state = risk_state_obj.get("state")
        risk_state_reason = risk_state_obj.get("reason") or risk_state_obj.get(
            "reason_code"
        )
    else:
        risk_state_state = getattr(risk_state_obj, "state", None)
        risk_state_reason = getattr(risk_state_obj, "reason", None)

    # Geo intel
    news_geo = getattr(ctx, "news_geo", None) or {}
    if isinstance(news_geo, dict):
        geo_score = news_geo.get("geo_score")
        geo_conf = news_geo.get("geo_confidence")
        geo_state_hint = news_geo.get("state_hint")
    else:
        geo_score = getattr(news_geo, "geo_score", None)
        geo_conf = getattr(news_geo, "geo_confidence", None)
        geo_state_hint = getattr(news_geo, "state_hint", None)

    # Include full news_geo (e.g. boost block from disclosures_confirm) for transparency
    news_geo_raw = (
        news_geo
        if isinstance(news_geo, dict)
        else (getattr(news_geo, "__dict__", None) or {})
    )

    # Market stress
    market_stress = getattr(ctx, "market_stress", None) or {}
    stress_ok = None
    if isinstance(market_stress, dict):
        stress_ok = market_stress.get("stress_ok")

    # Turnover gate from policy + meta
    tb_policy = (
        (policy.get("turnover_budget") or {}) if isinstance(policy, dict) else {}
    )
    turnover_cap = _safe_float(tb_policy.get("cap"))
    turnover_behavior = tb_policy.get("behavior")

    # QC / intel flags
    intel_flags = getattr(ctx, "intel_health_flags", None)

    payload: Dict[str, Any] = {
        "schema_version": "run.reasons.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "mode": str(mode),
        "risk_state": {
            "state": risk_state_state,
            "reason": risk_state_reason,
            "raw": risk_state_obj,
        },
        "geo": {
            "geo_score": geo_score,
            "geo_confidence": geo_conf,
            "state_hint": geo_state_hint,
            "raw": news_geo_raw,
        },
        "market_stress": {
            "stress_ok": stress_ok,
            "raw": market_stress,
        },
        "multipliers": {
            "georisk": float(georisk_mult),
            "profit_lock": float(profit_lock_mult),
            "final_exposure_multiplier": float(final_mult),
        },
        "turnover_gate": {
            "estimated_turnover": _safe_float(turnover_meta.get("estimated_turnover")),
            "cap": turnover_cap,
            "scale_factor": float(turnover_scale),
            "behavior": turnover_behavior,
            "meta": turnover_meta or None,
        },
        "qc_flags": intel_flags,
    }

    return _atomic_write_json(path, payload)


def maybe_execute_orders(mode: str, orders: pd.DataFrame) -> pd.DataFrame:
    """Placeholder hook for future paper-mode fill/ledger simulation.

    In \"shadow\" mode this is intentionally a no-op. In future versions,
    \"paper\" mode may call simulate_with_costs / ledger integration here.
    """
    if mode == "shadow":
        return orders
    # For v1, paper behaves the same; hook point only.
    return orders


def write_diff_vs_prev(
    output_dir: str | Path,
    prev_dir: str | Path,
    current_targets: Any,
    current_kpis: Dict[str, Any],
) -> Path:
    """Write diff_vs_prev.json comparing current run vs previous run (if present)."""
    out_dir = Path(output_dir)
    path = out_dir / "diff_vs_prev.json"

    prev_dir_path = Path(prev_dir)
    notes: list[str] = []
    prev_kpis: Dict[str, Any] | None = None
    prev_targets_items: list[Dict[str, Any]] = []

    if (
        prev_dir_path.exists()
        and (prev_dir_path / "run_kpis.json").exists()
        and (prev_dir_path / "targets_latest.json").exists()
    ):
        try:
            prev_kpis = json.loads(
                (prev_dir_path / "run_kpis.json").read_text(encoding="utf-8")
            )
            prev_targets_json = json.loads(
                (prev_dir_path / "targets_latest.json").read_text(encoding="utf-8")
            )
            prev_targets_items = list(prev_targets_json.get("items", []))
        except Exception:  # pragma: no cover - defensive
            notes.append("prev_run_read_error")
            prev_kpis = None
            prev_targets_items = []
    else:
        notes.append("no_prev_run_found")

    # Dates
    prev_date = None
    if prev_kpis and "generated_utc" in prev_kpis:
        try:
            prev_date = (
                datetime.fromisoformat(prev_kpis["generated_utc"]).date().isoformat()
            )
        except Exception:  # pragma: no cover
            prev_date = None

    curr_date = None
    if "generated_utc" in current_kpis:
        try:
            curr_date = (
                datetime.fromisoformat(current_kpis["generated_utc"]).date().isoformat()
            )
        except Exception:  # pragma: no cover
            curr_date = None

    # Multiplier deltas
    delta_multipliers: Dict[str, Any] = {}
    if prev_kpis and "multipliers" in prev_kpis:
        prev_m = prev_kpis.get("multipliers") or {}
        curr_m = current_kpis.get("multipliers") or {}
        for key in (
            "georisk",
            "profit_lock",
            "final_exposure_multiplier",
            "turnover_scale_factor",
        ):
            pv = _safe_float(prev_m.get(key))
            cv = _safe_float(curr_m.get(key))
            if pv is not None or cv is not None:
                delta_multipliers[key] = {
                    "prev": pv,
                    "curr": cv,
                    "delta": None if pv is None or cv is None else cv - pv,
                }

    # Risk state delta
    delta_risk_state: Dict[str, Any] | None = None
    if prev_kpis is not None:
        delta_risk_state = {
            "prev": prev_kpis.get("risk_state"),
            "curr": current_kpis.get("risk_state"),
        }

    # Targets delta
    prev_weights: Dict[str, float] = {}
    for it in prev_targets_items:
        sym = it.get("symbol")
        if sym is None:
            continue
        prev_weights[str(sym)] = _safe_float(it.get("target_weight")) or 0.0

    curr_weights: Dict[str, float] = {}
    if (
        isinstance(current_targets, pd.DataFrame)
        and not current_targets.empty
        and "symbol" in current_targets.columns
    ):
        has_weight = "target_weight" in current_targets.columns
        for _, row in current_targets.iterrows():
            sym = row.get("symbol")
            if sym is None:
                continue
            if has_weight:
                curr_weights[str(sym)] = _safe_float(row.get("target_weight")) or 0.0
            else:
                curr_weights[str(sym)] = 0.0

    symbols = sorted(set(prev_weights) | set(curr_weights))
    delta_targets: list[Dict[str, Any]] = []
    abs_delta_sum = 0.0
    n_changed = 0
    for sym in symbols:
        pw = prev_weights.get(sym, 0.0)
        cw = curr_weights.get(sym, 0.0)
        dw = cw - pw
        if abs(dw) > 0:
            abs_delta_sum += abs(dw)
            n_changed += 1
        delta_targets.append(
            {
                "symbol": sym,
                "prev_weight": pw,
                "curr_weight": cw,
                "delta_weight": dw,
            }
        )

    payload: Dict[str, Any] = {
        "schema_version": "run.diff.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "prev_date": prev_date,
        "current_date": curr_date,
        "delta_multipliers": delta_multipliers,
        "delta_risk_state": delta_risk_state,
        "delta_targets": delta_targets,
        "summary": {
            "abs_delta_weight_sum": abs_delta_sum,
            "n_symbols_changed": n_changed,
        },
        "notes": notes,
    }

    return _atomic_write_json(path, payload)
