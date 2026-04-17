"""E2 — HTB rate table loader regression pins."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.phase_realism]

yaml = pytest.importorskip("yaml")

from src.assembled_core.execution.borrow_costs import (  # noqa: E402
    BorrowRateTable,
    compute_borrow_cost,
    load_rate_table_from_yaml,
)

_ROOT = Path(__file__).resolve().parents[2]
_HTB_CONFIG = _ROOT / "config" / "htb_symbols.yaml"


def test_seed_list_loads_and_applies_overrides() -> None:
    assert _HTB_CONFIG.exists(), "HTB seed YAML missing"
    table = load_rate_table_from_yaml(_HTB_CONFIG)
    assert isinstance(table, BorrowRateTable)
    # GME is on the special tier with explicit 500 bps override.
    assert table.rate_bps("GME") == 500.0
    # TQQQ is easy — should fall back to easy default.
    assert table.rate_bps("TQQQ") == 50.0
    # Unknown symbol → default.
    assert table.rate_bps("AAPL") == table.default_rate_bps


def test_htb_cost_accrues_daily() -> None:
    table = load_rate_table_from_yaml(_HTB_CONFIG)
    rate = table.rate_bps("GME")  # 500 bps
    cost = compute_borrow_cost(qty=-100.0, price=20.0, rate_bps_annual=rate)
    # 100 * 20 * 0.05 / 365 ≈ 0.274
    assert cost == pytest.approx(2_000.0 * 0.05 / 365.0, rel=1e-9)


def test_long_positions_pay_zero() -> None:
    assert compute_borrow_cost(qty=100.0, price=20.0, rate_bps_annual=500.0) == 0.0


def test_yaml_missing_symbols_block_falls_back_cleanly(tmp_path: Path) -> None:
    cfg = tmp_path / "empty.yaml"
    cfg.write_text("version: 1\n", encoding="utf-8")
    table = load_rate_table_from_yaml(cfg)
    assert table.overrides == {}
    assert table.htb_symbols == set()
    assert table.rate_bps("XYZ") == table.default_rate_bps
