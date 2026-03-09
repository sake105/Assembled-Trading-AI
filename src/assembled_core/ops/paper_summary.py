"""OPS-6: Paper range summary — aggregate metrics from daily run artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "paper.summary.v1"


def _safe_float(x: Any) -> float | None:
    try:
        return float(x) if x is not None else None
    except (TypeError, ValueError):
        return None


def _collect_equity_curve(output_root: Path, dates: list[str]) -> list[tuple[str, float]]:
    """Collect (date, equity) one point per day: last point from ledger_state or ledger_snapshot."""
    out: list[tuple[str, float]] = []
    for d in dates:
        day_dir = output_root / d
        eq_val: float | None = None
        ledger_path = day_dir / "ledger_state.json"
        if ledger_path.exists():
            try:
                data = json.loads(ledger_path.read_text(encoding="utf-8"))
                curve = data.get("equity_curve") or []
                if curve:
                    last_pt = curve[-1]
                    eq_val = _safe_float(last_pt.get("equity"))
            except Exception:
                pass
        if eq_val is None:
            snapshot_path = day_dir / "ledger_snapshot.json"
            if snapshot_path.exists():
                try:
                    data = json.loads(snapshot_path.read_text(encoding="utf-8"))
                    eq_val = _safe_float(data.get("equity"))
                except Exception:
                    pass
        if eq_val is not None:
            out.append((d, eq_val))
    return out


def _total_return_and_drawdown(equity_values: list[float]) -> tuple[float | None, float | None]:
    if not equity_values or len(equity_values) < 2:
        return None, None
    start_val = equity_values[0]
    end_val = equity_values[-1]
    if start_val <= 0:
        return None, None
    total_ret = (end_val / start_val) - 1.0
    peak = start_val
    max_dd = 0.0
    for v in equity_values:
        if v > peak:
            peak = v
        dd = (peak - v) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return total_ret, max_dd


def _daily_returns(equity_values: list[float]) -> list[float]:
    if len(equity_values) < 2:
        return []
    rets = []
    for i in range(1, len(equity_values)):
        prev = equity_values[i - 1]
        curr = equity_values[i]
        if prev and prev > 0:
            rets.append((curr / prev) - 1.0)
    return rets


def build_paper_summary(output_root: str | Path, dates: list[str]) -> dict[str, Any]:
    """Build paper.summary.v1 from artifacts under output_root for the given date list."""
    output_root = Path(output_root)
    dates = sorted(dates)

    equity_points = _collect_equity_curve(output_root, dates)
    equity_by_date: dict[str, float] = dict(equity_points)
    ordered_equity = [equity_by_date[d] for d in dates if d in equity_by_date]

    total_return, max_drawdown = _total_return_and_drawdown(ordered_equity)
    daily_rets = _daily_returns(ordered_equity)
    daily_return_mean = (sum(daily_rets) / len(daily_rets)) if daily_rets else None
    daily_return_std = (
        (sum((r - daily_return_mean) ** 2 for r in daily_rets) / len(daily_rets)) ** 0.5
        if daily_rets and daily_return_mean is not None
        else None
    )

    final_multipliers: list[float] = []
    turnover_scale_factors: list[float] = []
    risk_states: list[Any] = []
    state_distribution: dict[str, int] = {"WATCH": 0, "ACTIVE": 0, "COOLDOWN": 0, "PAUSE": 0}
    reason_counts: dict[str, int] = {}
    alerts_by_level: dict[str, int] = {}
    alerts_by_kind: dict[str, int] = {}

    for d in dates:
        day_dir = output_root / d
        kpis_path = day_dir / "run_kpis.json"
        if kpis_path.exists():
            try:
                kpis = json.loads(kpis_path.read_text(encoding="utf-8"))
                mult = kpis.get("multipliers") or {}
                fm = _safe_float(mult.get("final_exposure_multiplier"))
                if fm is not None:
                    final_multipliers.append(fm)
                ts = _safe_float(mult.get("turnover_scale_factor"))
                if ts is not None:
                    turnover_scale_factors.append(ts)
                rs = kpis.get("risk_state")
                if rs is not None:
                    risk_states.append(rs)
                    state = (rs.get("state") if isinstance(rs, dict) else getattr(rs, "state", None)) or ""
                    state_distribution[state] = state_distribution.get(state, 0) + 1
                    reason = (rs.get("reason") if isinstance(rs, dict) else getattr(rs, "reason", None)) or ""
                    if reason:
                        reason_counts[reason] = reason_counts.get(reason, 0) + 1
            except Exception:
                pass
        alerts_path = day_dir / "alerts_latest.json"
        if alerts_path.exists():
            try:
                alerts_data = json.loads(alerts_path.read_text(encoding="utf-8"))
                for item in alerts_data.get("items") or []:
                    level = item.get("level") or "info"
                    kind = item.get("kind") or "unknown"
                    alerts_by_level[level] = alerts_by_level.get(level, 0) + 1
                    alerts_by_kind[kind] = alerts_by_kind.get(kind, 0) + 1
            except Exception:
                pass

    risk_state_transitions = 0
    prev_state = None
    for rs in risk_states:
        s = rs.get("state") if isinstance(rs, dict) else getattr(rs, "state", None)
        if prev_state is not None and s != prev_state:
            risk_state_transitions += 1
        prev_state = s

    n_dates = len(dates)
    risk_state_pct: dict[str, float] = {}
    if n_dates > 0:
        for st, count in state_distribution.items():
            risk_state_pct[st] = round(count / n_dates, 4)

    avg_final_multiplier = (sum(final_multipliers) / len(final_multipliers)) if final_multipliers else None
    avg_turnover_scale_factor = (sum(turnover_scale_factors) / len(turnover_scale_factors)) if turnover_scale_factors else None

    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "start_date": dates[0] if dates else None,
        "end_date": dates[-1] if dates else None,
        "n_dates": len(dates),
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "daily_return_mean": daily_return_mean,
        "daily_return_std": daily_return_std,
        "avg_final_multiplier": avg_final_multiplier,
        "avg_turnover_scale_factor": avg_turnover_scale_factor,
        "risk_state_transitions": risk_state_transitions,
        "risk_state_distribution": dict(state_distribution),
        "risk_state_reason_counts": dict(reason_counts),
        "risk_state_pct": risk_state_pct,
        "alerts_count_by_level": dict(alerts_by_level),
        "alerts_count_by_kind": dict(alerts_by_kind),
        "equity_curve_dates": list(equity_by_date.keys()),
    }
    return summary


def write_paper_summary(
    output_root: str | Path,
    start_date: str,
    end_date: str,
    summary: dict[str, Any],
) -> Path:
    """Write output_root/_summaries/paper_summary_<start>_<end>.json atomically."""
    output_root = Path(output_root)
    summaries_dir = output_root / "_summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    path = summaries_dir / f"paper_summary_{start_date}_{end_date}.json"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    tmp.replace(path)
    return path
