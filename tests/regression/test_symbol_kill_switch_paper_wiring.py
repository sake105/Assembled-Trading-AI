"""Tier-1 wiring — verify execution.symbol_kill_switch is consumed by the
unified paper engine ``_apply_risk_controls`` path.

Guards against silent drift where the module is imported but no order ever
touches it. A blocked symbol must be filtered out once the feature flag is
on; with the flag off the filter is a no-op.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytestmark = [pytest.mark.phase12]

from src.assembled_core.execution.symbol_kill_switch import block_symbol  # noqa: E402
from src.assembled_core.execution.unified_paper_engine import (  # noqa: E402
    UnifiedPaperConfig,
    UnifiedPaperEngine,
)


def _make_orders() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"symbol": "AAPL", "side": "buy", "qty": 10, "limit_price": 150.0},
            {"symbol": "XYZ", "side": "buy", "qty": 5, "limit_price": 30.0},
        ]
    )


def test_symbol_kill_switch_filters_only_when_enabled(tmp_path: Path) -> None:
    state_path = tmp_path / "symbol_kill.json"
    block_symbol("XYZ", reason="unit-test-halt", state_path=state_path)

    # Flag OFF → engine must NOT filter; both orders survive.
    cfg_off = UnifiedPaperConfig(
        state_dir=tmp_path / "state_off",
        ledger_dir=tmp_path / "ledger_off",
        lifecycle_dir=tmp_path / "lifecycle_off",
        enable_symbol_kill_switch=False,
        symbol_kill_state_path=state_path,
        enable_ledger=False,
        enable_reconciliation=False,
        enable_lifecycle_tracking=False,
        enable_fat_finger=False,
        enable_kill_switch=False,
    )
    eng_off = UnifiedPaperEngine(config=cfg_off)
    kept_off = eng_off._apply_risk_controls(_make_orders())
    assert set(kept_off["symbol"]) == {"AAPL", "XYZ"}

    # Flag ON → the blocked symbol is dropped, the allowed one survives.
    cfg_on = UnifiedPaperConfig(
        state_dir=tmp_path / "state_on",
        ledger_dir=tmp_path / "ledger_on",
        lifecycle_dir=tmp_path / "lifecycle_on",
        enable_symbol_kill_switch=True,
        symbol_kill_state_path=state_path,
        enable_ledger=False,
        enable_reconciliation=False,
        enable_lifecycle_tracking=False,
        enable_fat_finger=False,
        enable_kill_switch=False,
    )
    eng_on = UnifiedPaperEngine(config=cfg_on)
    kept_on = eng_on._apply_risk_controls(_make_orders())
    assert set(kept_on["symbol"]) == {"AAPL"}
