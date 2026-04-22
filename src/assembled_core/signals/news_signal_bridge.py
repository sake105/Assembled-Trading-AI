"""Part B deeper wiring: news → signal score bridge.

Reads news triggers (``output/intel/news/triggers_latest.json``) and derives
per-ticker score adjustments. This is the minimal F1-style bridge that
connects the news engine output into signal generation.

Scoring model (simple, conservative):

    per_ticker_score = severity * urgency * sentiment_sign

where:
- severity is 1 (WATCH) or 2+ (ACTIVE)
- urgency is a float in [0, 1] (default 0.5 if absent)
- sentiment_sign is +1 / 0 / -1 from trigger "sentiment" or derived from
  topic (default 0, neutral)

The bridge produces a weight-scaled delta that is added onto existing
signal scores. Shorts are blocked when ``allow_short=false``.

Gated in policy:

```yaml
intel:
  news_signal_bridge:
    enabled: true
    weight: 0.10           # max score delta contribution
    min_severity: 1
    allow_short: false
```

All paths are defensive — a missing artifact or empty items is a silent
no-op.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

_SENTIMENT_TO_SIGN = {
    "positive": 1,
    "bullish": 1,
    "negative": -1,
    "bearish": -1,
    "neutral": 0,
}


def _sentiment_sign(trig: dict[str, Any]) -> int:
    raw = trig.get("sentiment")
    if raw is None:
        return 0
    if isinstance(raw, (int, float)):
        if raw > 0.1:
            return 1
        if raw < -0.1:
            return -1
        return 0
    key = str(raw).strip().lower()
    return int(_SENTIMENT_TO_SIGN.get(key, 0))


def _urgency(trig: dict[str, Any], default: float = 0.5) -> float:
    raw = trig.get("urgency")
    if raw is None:
        return default
    try:
        u = float(raw)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, u))


def _extract_tickers(trig: dict[str, Any]) -> list[str]:
    for k in ("tickers", "symbols", "affected_assets"):
        v = trig.get(k)
        if isinstance(v, (list, tuple)):
            return [str(t).strip().upper() for t in v if str(t).strip()]
    return []


def compute_news_deltas(
    triggers: list[dict[str, Any]],
    *,
    min_severity: int = 1,
    allow_short: bool = False,
) -> dict[str, float]:
    """Aggregate per-ticker raw score deltas from the triggers list.

    Returns dict {symbol: delta} where delta ∈ [-severity_max, +severity_max].
    """
    deltas: dict[str, float] = {}
    if not triggers:
        return deltas
    for t in triggers:
        try:
            sev = int(t.get("severity", 0))
        except (TypeError, ValueError):
            continue
        if sev < min_severity:
            continue
        tickers = _extract_tickers(t)
        if not tickers:
            continue
        urg = _urgency(t)
        sign = _sentiment_sign(t)
        if sign < 0 and not allow_short:
            continue
        delta = float(sev) * urg * float(sign)
        if delta == 0.0:
            # Even neutral sentiment: give a small positive nudge when
            # severity/urgency is high (news attention weighted as long bias)
            delta = 0.1 * float(sev) * urg
        for sym in tickers:
            deltas[sym] = deltas.get(sym, 0.0) + delta
    return deltas


def load_and_apply_news_signals(
    signals: pd.DataFrame,
    *,
    root: Path,
    policy: dict[str, Any],
    news_triggers_path: str | None = None,
    as_of: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load news triggers from disk and apply per-ticker deltas to signal scores.

    Returns ``(signals_out, meta)``. ``signals_out`` may contain new symbols
    if a high-severity trigger targets a ticker not yet in signals.
    """
    bridge_cfg = (policy.get("intel") or {}).get("news_signal_bridge") or {}
    meta: dict[str, Any] = {
        "enabled": bool(bridge_cfg.get("enabled", False)),
        "applied": 0,
        "added": 0,
        "total_delta_abs": 0.0,
    }
    if not meta["enabled"]:
        return signals, meta

    triggers_path = (
        Path(news_triggers_path)
        if news_triggers_path
        else root / "output" / "intel" / "news" / "triggers_latest.json"
    )
    if not triggers_path.exists():
        log.debug("[NEWS-SIGNAL] triggers artifact missing: %s", triggers_path)
        return signals, meta

    try:
        data = json.loads(triggers_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("[NEWS-SIGNAL] failed to load triggers: %s", exc)
        return signals, meta

    items = data.get("triggers") or data.get("items") or []
    if not isinstance(items, list) or not items:
        return signals, meta

    weight = float(bridge_cfg.get("weight", 0.10))
    min_sev = int(bridge_cfg.get("min_severity", 1))
    allow_short = bool(bridge_cfg.get("allow_short", False))

    deltas = compute_news_deltas(items, min_severity=min_sev, allow_short=allow_short)
    if not deltas:
        return signals, meta

    out = signals.copy() if signals is not None else pd.DataFrame()
    if out.empty:
        out = pd.DataFrame(columns=["timestamp", "symbol", "direction", "score"])

    existing_syms: set[str] = (
        set(out["symbol"].astype(str).values) if "symbol" in out.columns else set()
    )

    ts_now = as_of if as_of is not None else pd.Timestamp.now("UTC")

    applied = 0
    added = 0
    abs_sum = 0.0
    new_rows: list[dict[str, Any]] = []

    for sym, delta in deltas.items():
        scaled = delta * weight
        abs_sum += abs(scaled)
        if sym in existing_syms and "score" in out.columns:
            mask = out["symbol"].astype(str) == sym
            if mask.any():
                out.loc[mask, "score"] = out.loc[mask, "score"].astype(float) + scaled
                applied += 1
        else:
            direction = "LONG" if scaled >= 0 else "SHORT"
            if direction == "SHORT" and not allow_short:
                continue
            new_rows.append({
                "timestamp": ts_now,
                "symbol": sym,
                "direction": direction,
                "score": round(scaled, 4),
            })
            added += 1

    if new_rows:
        out = pd.concat([out, pd.DataFrame(new_rows)], ignore_index=True)

    meta.update({
        "applied": applied,
        "added": added,
        "total_delta_abs": round(abs_sum, 4),
    })

    log.info(
        "[NEWS-SIGNAL] bridge applied: %d existing boosted, %d new symbols added, |Δ|=%.2f",
        applied, added, abs_sum,
    )

    return out, meta


__all__ = [
    "compute_news_deltas",
    "load_and_apply_news_signals",
]
