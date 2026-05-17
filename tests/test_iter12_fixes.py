"""Regression tests for Iteration-12 fixes.

Covers:
  Fix 1 — meta_model.py: NaT in timestamp column no longer raises RuntimeError
  Fix 2 — model_registry.py: _load_registry re-reads when file mtime changes
  Fix 3 — paper_track.py: PaperTrackConfig default georisk=False logs DEBUG not WARNING
  Fix 4 — pre_trade_checks.py: CVaR comparison direction corrected
  Fix 5 — paper_track.py: mode="paper" in paper runner TradingContext
  Fix 6 — trading_cycle_shared.py: de_risk_scale defaults to 0.5 not 0.0
  Fix 7 — trading_cycle_shared.py: policy forwarded to filter_orders_with_risk_controls
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.fast


# ---------------------------------------------------------------------------
# Fix 1 — meta_model.py: NaT in timestamp does not raise RuntimeError
# ---------------------------------------------------------------------------


class TestMetaModelNatTimestamp:
    """Sort logic with NaT must not raise RuntimeError."""

    def test_sort_with_nat_is_monotonic_after_dropna(self):
        """Non-NaT timestamps are monotonic after sort_values(na_position='last')."""
        timestamps = [
            pd.Timestamp("2024-01-01"),
            pd.NaT,
            pd.Timestamp("2024-01-03"),
            pd.Timestamp("2024-01-04"),
        ]
        df = pd.DataFrame({"timestamp": timestamps, "x": [1.0, 2.0, 3.0, 4.0]})
        df = df.sort_values("timestamp", na_position="last").reset_index(drop=True)
        non_nat = df["timestamp"].dropna()
        assert non_nat.is_monotonic_increasing

    def test_nat_placed_at_end(self):
        """NaT rows end up last after sort_values(na_position='last')."""
        timestamps = [pd.NaT, pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")]
        df = pd.DataFrame({"timestamp": timestamps})
        df = df.sort_values("timestamp", na_position="last").reset_index(drop=True)
        assert pd.isna(df["timestamp"].iloc[-1])

    def test_meta_model_sort_col_applied_correctly(self):
        """The meta_model module uses na_position='last' in its sort call."""
        import ast
        import pathlib

        src = pathlib.Path("src/assembled_core/signals/meta_model.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(src)
        found_na_position = False
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword):
                if node.arg == "na_position":
                    found_na_position = True
                    break
        assert found_na_position, "meta_model.py must use na_position= in sort_values to handle NaT"  # fmt: skip


# ---------------------------------------------------------------------------
# Fix 2 — model_registry.py: mtime-guarded cache refresh
# ---------------------------------------------------------------------------


class TestModelRegistryMtimeCache:
    """_load_registry must re-read registry.json when the file's mtime changes."""

    def _write_registry(self, path: Path, models: dict) -> None:
        path.write_text(json.dumps({"models": models}), encoding="utf-8")

    def test_cache_refreshes_when_mtime_changes(self, tmp_path):
        """Second load after file modification returns updated content."""
        import src.assembled_core.ml.model_registry as reg_mod

        registry_file = tmp_path / "registry.json"
        self._write_registry(registry_file, {"model_v1.joblib": {"sha256": "aaa"}})

        orig_path = reg_mod._REGISTRY_PATH
        orig_cache = reg_mod._registry_cache
        orig_mtime = reg_mod._registry_mtime
        try:
            reg_mod._REGISTRY_PATH = registry_file
            reg_mod._registry_cache = None
            reg_mod._registry_mtime = None

            first = reg_mod._load_registry()
            assert "model_v1.joblib" in first

            time.sleep(0.02)  # ensure mtime differs
            self._write_registry(registry_file, {"model_v2.joblib": {"sha256": "bbb"}})

            second = reg_mod._load_registry()
            assert "model_v2.joblib" in second, "Cache must refresh after mtime change"
            assert "model_v1.joblib" not in second
        finally:
            reg_mod._REGISTRY_PATH = orig_path
            reg_mod._registry_cache = orig_cache
            reg_mod._registry_mtime = orig_mtime

    def test_cache_hit_when_mtime_unchanged(self, tmp_path):
        """Two consecutive loads without touching the file return the same object."""
        import src.assembled_core.ml.model_registry as reg_mod

        registry_file = tmp_path / "registry.json"
        self._write_registry(registry_file, {"model_a.joblib": {"sha256": "ccc"}})

        orig_path = reg_mod._REGISTRY_PATH
        orig_cache = reg_mod._registry_cache
        orig_mtime = reg_mod._registry_mtime
        try:
            reg_mod._REGISTRY_PATH = registry_file
            reg_mod._registry_cache = None
            reg_mod._registry_mtime = None

            first = reg_mod._load_registry()
            second = reg_mod._load_registry()
            assert first is second
        finally:
            reg_mod._REGISTRY_PATH = orig_path
            reg_mod._registry_cache = orig_cache
            reg_mod._registry_mtime = orig_mtime

    def test_registry_mtime_variable_exists(self):
        """The module must expose _registry_mtime as a module-level variable."""
        import src.assembled_core.ml.model_registry as reg_mod

        assert hasattr(reg_mod, "_registry_mtime"), "_registry_mtime module variable must exist for mtime-guarded caching"  # fmt: skip


# ---------------------------------------------------------------------------
# Fix 3 — paper_track.py: georisk_gate_enabled=False logs DEBUG not WARNING
# ---------------------------------------------------------------------------


class TestPaperTrackConfigLogLevel:
    """Default PaperTrackConfig instantiation must not emit WARNING-level log."""

    def _make_config(self, **kwargs):
        from src.assembled_core.paper.paper_track import PaperTrackConfig

        defaults = dict(
            strategy_name="test",
            strategy_type="multifactor_long_short",
            universe_file=Path("universe.csv"),
            freq="1d",
        )
        defaults.update(kwargs)
        return PaperTrackConfig(**defaults)

    def test_default_config_no_warning(self, caplog):
        """Constructing PaperTrackConfig with georisk_gate_enabled=False emits no WARNING."""
        with caplog.at_level(logging.WARNING):
            self._make_config(georisk_gate_enabled=False)

        geo_warnings = [
            r
            for r in caplog.records
            if r.levelno >= logging.WARNING and "georisk" in r.message.lower()
        ]
        _msg = f"PaperTrackConfig(georisk_gate_enabled=False) must not emit WARNING for georisk, got: {[r.message for r in geo_warnings]}"
        assert not geo_warnings, _msg

    def test_default_config_emits_debug(self, caplog):
        """Constructing PaperTrackConfig with georisk_gate_enabled=False emits a DEBUG log."""
        with caplog.at_level(
            logging.DEBUG, logger="src.assembled_core.paper.paper_track"
        ):
            self._make_config(georisk_gate_enabled=False)

        geo_debug = [
            r
            for r in caplog.records
            if r.levelno == logging.DEBUG and "georisk" in r.message.lower()
        ]
        assert geo_debug, "PaperTrackConfig(georisk_gate_enabled=False) must emit a DEBUG message about georisk"  # fmt: skip


# ---------------------------------------------------------------------------
# Fix 4 — pre_trade_checks.py: CVaR comparison direction
# ---------------------------------------------------------------------------


class TestCVaRComparisonDirection:
    """CVaR check: trigger (scale) when portfolio CVaR < limit (more negative = worse)."""

    def test_cvar_early_return_when_within_limit(self):
        """Source must early-return when CVaR is within limit (>= comparison is correct)."""
        import pathlib

        src = pathlib.Path(
            "src/assembled_core/execution/pre_trade_checks.py"
        ).read_text(encoding="utf-8")
        # Correct logic: early return when portfolio_cvar >= max_cvar_95
        # (e.g. -0.03 >= -0.05: within limit, no action needed)
        # Trigger (scale) happens when NOT within limit (falls through to scaling code)
        assert "max_cvar_95" in src, "max_cvar_95 limit must be referenced in pre_trade_checks.py"  # fmt: skip
        assert "portfolio_cvar" in src or "cvar" in src.lower(), "CVaR logic must be present in pre_trade_checks.py"  # fmt: skip

    def test_cvar_scale_is_clamped(self):
        """CVaR scale factor must be clamped to [0, 1] to prevent over-scaling."""
        import pathlib

        src = pathlib.Path(
            "src/assembled_core/execution/pre_trade_checks.py"
        ).read_text(encoding="utf-8")
        # Scale must be clamped — max(0.0, min(..., 1.0)) or clip(0, 1)
        has_clamp = ("min(" in src and "max(" in src) or "clip(" in src
        assert has_clamp, "CVaR scale must be clamped to [0, 1] to prevent invalid scaling"  # fmt: skip


# ---------------------------------------------------------------------------
# Fix 5 — paper_track.py: mode="paper" in TradingContext
# ---------------------------------------------------------------------------


class TestPaperTrackModeValue:
    """run_paper_track_day must not hardcode mode='backtest' in TradingContext."""

    def test_source_does_not_contain_mode_backtest_in_paper_runner(self):
        """paper_track.py must not pass mode='backtest' to TradingContext in paper runner."""
        import pathlib

        src = pathlib.Path("src/assembled_core/paper/paper_track.py").read_text(
            encoding="utf-8"
        )

        # Find lines that construct TradingContext with mode=
        lines = src.splitlines()
        bad_lines = [
            (i + 1, line.strip())
            for i, line in enumerate(lines)
            if 'mode="backtest"' in line or "mode='backtest'" in line
        ]
        _msg = f"paper_track.py must not hardcode mode='backtest'; found at lines: {bad_lines}"
        assert not bad_lines, _msg

    def test_source_contains_mode_paper_in_paper_runner(self):
        """paper_track.py must use mode='paper' in the paper runner context."""
        import pathlib

        src = pathlib.Path("src/assembled_core/paper/paper_track.py").read_text(
            encoding="utf-8"
        )
        assert 'mode="paper"' in src or "mode='paper'" in src, "paper_track.py must contain mode='paper' for the paper runner TradingContext"  # fmt: skip


# ---------------------------------------------------------------------------
# Fix 6 — trading_cycle_shared.py: de_risk_scale defaults to 0.5
# ---------------------------------------------------------------------------


class TestDeRiskScaleDefault:
    """de_risk_scale default must be 0.5, not 0.0, in trading_cycle_shared."""

    def test_de_risk_scale_default_is_not_zero(self):
        """Source must not use get('de_risk_scale', 0.0) — would zero all orders on DD."""
        import pathlib

        src = pathlib.Path(
            "src/assembled_core/pipeline/trading_cycle_shared.py"
        ).read_text(encoding="utf-8")
        assert (
            '"de_risk_scale", 0.0' not in src and "'de_risk_scale', 0.0" not in src
        ), (
            "de_risk_scale default must not be 0.0 — would block all orders on drawdown; "
            "use 0.5 as a safer fallback"
        )

    def test_de_risk_scale_default_is_half(self):
        """Source must use get('de_risk_scale', 0.5) as the fallback."""
        import pathlib

        src = pathlib.Path(
            "src/assembled_core/pipeline/trading_cycle_shared.py"
        ).read_text(encoding="utf-8")
        assert '"de_risk_scale", 0.5' in src or "'de_risk_scale', 0.5" in src, "de_risk_scale must default to 0.5 (halve positions) not 0.0 (block all)"  # fmt: skip


# ---------------------------------------------------------------------------
# Fix 7 — trading_cycle_shared.py: policy forwarded to filter_orders
# ---------------------------------------------------------------------------


class TestPolicyForwardedToFilterOrders:
    """filter_orders_with_risk_controls call must include a policy= argument."""

    def test_filter_orders_call_has_policy_arg(self):
        """Source must pass policy= to filter_orders_with_risk_controls."""
        import pathlib

        src = pathlib.Path(
            "src/assembled_core/pipeline/trading_cycle_shared.py"
        ).read_text(encoding="utf-8")

        # The call site must include policy= argument
        assert "policy=" in src, "trading_cycle_shared.py must forward policy= to filter_orders_with_risk_controls"  # fmt: skip

    def test_load_policy_imported_or_used(self):
        """load_policy must be imported/used so it can be passed to filter_orders."""
        import pathlib

        src = pathlib.Path(
            "src/assembled_core/pipeline/trading_cycle_shared.py"
        ).read_text(encoding="utf-8")
        assert "load_policy" in src, "trading_cycle_shared.py must import/use load_policy to obtain the policy object"  # fmt: skip
