"""D5 — Signal-decay read-path for the multi-factor combiner.

The plan flags D5 as ``unwired (0 hits)``: ``qa/signal_decay.py`` ships a
full decay-profile API but no strategy reads it. Stale factors keep getting
traded at full weight.

Contract with the offline weekly job
------------------------------------

* Job writes ``output/qa/signal_decay/latest.json`` on a weekly cadence.
* Shape::

      {
          "generated_at": "<ISO timestamp>",
          "universe": "<name>",
          "factors": {
              "<factor_name>": {
                  "ic_mean": float,
                  "ic_half_life_days": float | null,
                  "is_stale": bool
              },
              ...
          }
      }

* Stale factors (``is_stale=True``) are downweighted by a policy-controlled
  multiplier (default 0.0 → fully muted). Missing factors fall back to 1.0
  so we don't silently zero out a working signal just because the decay job
  hasn't produced a profile yet.

Policy gate
-----------

* ``policy.signal_decay.enabled=False`` (default): compute multipliers but
  **do not** modify returned weights. Shadow-mode caller should snapshot the
  hypothetical modification to ``output/shadow/signal_decay_<date>.json`` and
  diff against the applied weights at D5 go/no-go time.
* ``policy.signal_decay.enabled=True``: applied. Returned weights have
  ``weights[name] * multipliers[name]`` in place.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


DEFAULT_REPORT_PATH = Path("output/qa/signal_decay/latest.json")
DEFAULT_STALE_MULTIPLIER = 0.0  # stale factor → muted
DEFAULT_HEALTHY_MULTIPLIER = 1.0


def _load_report(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        logger.info(
            "[signal_decay] no report at %s — falling back to 1.0 multipliers", path
        )
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[signal_decay] failed to parse %s (%s) — falling back to 1.0", path, exc
        )
        return None


def compute_multipliers(
    factor_names: list[str],
    *,
    report_path: Path | None = None,
    stale_multiplier: float = DEFAULT_STALE_MULTIPLIER,
) -> dict[str, float]:
    """Return a ``{factor_name: multiplier}`` map for the given factor list.

    A factor is reported as stale → gets ``stale_multiplier``. Otherwise it
    gets 1.0. Missing / unknown factors also get 1.0 (no silent muting).
    """
    path = report_path or DEFAULT_REPORT_PATH
    report = _load_report(path)
    if report is None:
        return {name: DEFAULT_HEALTHY_MULTIPLIER for name in factor_names}

    factors_section = report.get("factors", {})
    if not isinstance(factors_section, dict):
        logger.warning("[signal_decay] 'factors' section is not a dict — falling back")
        return {name: DEFAULT_HEALTHY_MULTIPLIER for name in factor_names}

    out: dict[str, float] = {}
    for name in factor_names:
        entry = factors_section.get(name)
        if isinstance(entry, dict) and bool(entry.get("is_stale", False)):
            out[name] = float(stale_multiplier)
        else:
            out[name] = DEFAULT_HEALTHY_MULTIPLIER
    return out


def apply_multipliers(
    weights: dict[str, float],
    *,
    report_path: Path | None = None,
    enabled: bool = False,
    stale_multiplier: float = DEFAULT_STALE_MULTIPLIER,
) -> tuple[dict[str, float], dict[str, float]]:
    """Return ``(effective_weights, multipliers_used)``.

    If ``enabled=False`` → returns the original weights unchanged but the
    multiplier map describes what *would* have been applied. Callers should
    snapshot the multiplier map to the shadow dir.
    """
    multipliers = compute_multipliers(
        list(weights), report_path=report_path, stale_multiplier=stale_multiplier
    )
    if not enabled:
        return dict(weights), multipliers
    effective = {
        name: float(w) * multipliers.get(name, 1.0) for name, w in weights.items()
    }
    return effective, multipliers
