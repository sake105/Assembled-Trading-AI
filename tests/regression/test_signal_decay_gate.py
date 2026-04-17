"""D5 — Signal-decay gate regression pins.

Enforces:

* Default is disabled → weights pass through unchanged.
* Enabled + stale report → stale factors muted, healthy factors preserved.
* Missing report → all multipliers fall back to 1.0 (no silent muting).
* multifactor_v1 imports and calls the gate in ``compute_signals``.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from src.assembled_core.strategies import multifactor_v1
from src.assembled_core.strategies.signal_decay_gate import (
    DEFAULT_STALE_MULTIPLIER,
    apply_multipliers,
    compute_multipliers,
)

pytestmark = pytest.mark.phase_depth


def _write_report(path: Path, stale: dict[str, bool]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "generated_at": "2026-04-17T00:00:00+00:00",
        "universe": "test",
        "factors": {
            name: {"ic_mean": 0.02, "ic_half_life_days": 5.0, "is_stale": flag}
            for name, flag in stale.items()
        },
    }
    path.write_text(json.dumps(body), encoding="utf-8")


def test_missing_report_all_healthy(tmp_path: Path) -> None:
    mult = compute_multipliers(
        ["trend_ema_spread", "mom_rsi_centered"],
        report_path=tmp_path / "does_not_exist.json",
    )
    assert mult == {"trend_ema_spread": 1.0, "mom_rsi_centered": 1.0}


def test_stale_flag_mutes_factor(tmp_path: Path) -> None:
    path = tmp_path / "decay.json"
    _write_report(path, {"trend_ema_spread": True, "mom_rsi_centered": False})

    mult = compute_multipliers(
        ["trend_ema_spread", "mom_rsi_centered", "unknown"],
        report_path=path,
    )
    assert mult["trend_ema_spread"] == DEFAULT_STALE_MULTIPLIER == 0.0
    assert mult["mom_rsi_centered"] == 1.0
    # Unknown factors fall back to healthy — never silently mute.
    assert mult["unknown"] == 1.0


def test_apply_multipliers_disabled_passthrough(tmp_path: Path) -> None:
    path = tmp_path / "decay.json"
    _write_report(path, {"trend_ema_spread": True})
    weights = {"trend_ema_spread": 0.15, "mom_rsi_centered": 0.08}

    effective, mult = apply_multipliers(weights, report_path=path, enabled=False)
    # Disabled → returned weights equal input, multipliers show what would apply.
    assert effective == weights
    assert mult["trend_ema_spread"] == 0.0
    assert mult["mom_rsi_centered"] == 1.0


def test_apply_multipliers_enabled_mutates_weights(tmp_path: Path) -> None:
    path = tmp_path / "decay.json"
    _write_report(path, {"trend_ema_spread": True, "mom_rsi_centered": False})
    weights = {"trend_ema_spread": 0.15, "mom_rsi_centered": 0.08}

    effective, _ = apply_multipliers(weights, report_path=path, enabled=True)
    assert effective["trend_ema_spread"] == 0.0  # muted
    assert effective["mom_rsi_centered"] == 0.08  # preserved


def test_multifactor_v1_wires_decay_gate() -> None:
    src = inspect.getsource(multifactor_v1.compute_signals)
    assert "apply_multipliers" in src, (
        "D5 regression: multifactor_v1 no longer reads the signal-decay gate. "
        "Stale factors are being traded at full weight again."
    )
    assert "signal_decay" in src, "D5 cfg key 'signal_decay' was removed"


def test_custom_stale_multiplier_respected(tmp_path: Path) -> None:
    path = tmp_path / "decay.json"
    _write_report(path, {"trend_ema_spread": True})
    weights = {"trend_ema_spread": 0.2}
    effective, _ = apply_multipliers(
        weights, report_path=path, enabled=True, stale_multiplier=0.25
    )
    assert effective["trend_ema_spread"] == pytest.approx(0.05)
