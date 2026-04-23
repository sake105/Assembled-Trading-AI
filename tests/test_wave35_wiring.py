"""Tests for wave-35 module wiring into trading_cycle.py.

Covers:
  Step 3.89 — signals.plugin_loader (discover_signal_plugins)
  Step 8.24 — ops.health_check (HealthCheck / aggregate_overall_status)
  Step 8.25 — qa.robustness (compute_deflated_sharpe)
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pytest

from src.assembled_core.signals.plugin_loader import (
    discover_signal_plugins,
    load_signal_plugin,
)
from src.assembled_core.ops.health_check import (
    HealthCheck,
    HealthCheckResult,
    aggregate_overall_status,
    health_result_to_dict,
)
from src.assembled_core.qa.robustness import compute_deflated_sharpe


# ---------------------------------------------------------------------------
# discover_signal_plugins (Step 3.89)
# ---------------------------------------------------------------------------

def test_discover_plugins_nonexistent_dir_returns_empty(tmp_path):
    result = discover_signal_plugins(str(tmp_path / "no_such_dir"))
    assert isinstance(result, dict)
    assert len(result) == 0


def test_discover_plugins_empty_dir_returns_empty(tmp_path):
    plugin_dir = tmp_path / "plugins" / "signals"
    plugin_dir.mkdir(parents=True)
    result = discover_signal_plugins(str(plugin_dir))
    assert isinstance(result, dict)
    assert len(result) == 0


def test_discover_plugins_loads_valid_plugin(tmp_path):
    plugin_dir = tmp_path / "plugins" / "signals"
    plugin_dir.mkdir(parents=True)
    plugin_code = textwrap.dedent("""\
        import pandas as pd

        def signal_fn(prices):
            return pd.Series(0.0, index=prices.index if hasattr(prices, 'index') else [0])
    """)
    (plugin_dir / "test_plugin.py").write_text(plugin_code, encoding="utf-8")
    result = discover_signal_plugins(str(plugin_dir))
    assert "test_plugin" in result


def test_discover_plugins_skips_underscore_files(tmp_path):
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    (plugin_dir / "_private.py").write_text("def signal_fn(p): pass\n", encoding="utf-8")
    result = discover_signal_plugins(str(plugin_dir))
    assert "_private" not in result


def test_load_signal_plugin_nonexistent_returns_none(tmp_path):
    result = load_signal_plugin(tmp_path / "nonexistent.py")
    assert result is None


def test_load_signal_plugin_valid_file(tmp_path):
    plugin_file = tmp_path / "my_plugin.py"
    plugin_file.write_text(
        "def signal_fn(prices): return prices\n",
        encoding="utf-8",
    )
    fn = load_signal_plugin(plugin_file)
    assert fn is not None
    assert callable(fn)


def test_discover_plugins_multiple(tmp_path):
    plugin_dir = tmp_path / "signals"
    plugin_dir.mkdir()
    for name in ["alpha", "beta", "gamma"]:
        (plugin_dir / f"{name}.py").write_text(
            f"def signal_fn(p): return p\n", encoding="utf-8"
        )
    result = discover_signal_plugins(str(plugin_dir))
    assert len(result) == 3


# ---------------------------------------------------------------------------
# HealthCheck / aggregate_overall_status (Step 8.24)
# ---------------------------------------------------------------------------

def test_health_check_creates_ok():
    hc = HealthCheck(name="test_check", status="OK", value=1.0)
    assert hc.status == "OK"
    assert hc.name == "test_check"


def test_aggregate_all_ok():
    checks = [
        HealthCheck(name="c1", status="OK", value=1),
        HealthCheck(name="c2", status="OK", value=2),
    ]
    assert aggregate_overall_status(checks) == "OK"


def test_aggregate_with_warn():
    checks = [
        HealthCheck(name="c1", status="OK", value=1),
        HealthCheck(name="c2", status="WARN", value=0),
    ]
    result = aggregate_overall_status(checks)
    assert result in {"WARN", "CRITICAL"}


def test_aggregate_with_critical():
    checks = [
        HealthCheck(name="c1", status="OK", value=1),
        HealthCheck(name="c2", status="CRITICAL", value=-1),
    ]
    result = aggregate_overall_status(checks)
    assert result == "CRITICAL"


def test_aggregate_empty_list():
    result = aggregate_overall_status([])
    assert result in {"OK", "SKIP"}


def test_health_result_to_dict():
    import pandas as pd
    checks = [HealthCheck(name="c", status="OK", value=1.0)]
    result = HealthCheckResult(
        overall_status="OK",
        timestamp=pd.Timestamp.now(tz="UTC"),
        checks=checks,
    )
    d = health_result_to_dict(result)
    assert isinstance(d, dict)
    assert "overall_status" in d


# ---------------------------------------------------------------------------
# compute_deflated_sharpe (Step 8.25)
# ---------------------------------------------------------------------------

pytest.importorskip("scipy", reason="scipy required for deflated Sharpe")


def test_deflated_sharpe_n1_trial_equals_sharpe():
    sr = 1.2
    result = compute_deflated_sharpe(sharpe=sr, n_obs=252, n_trials=1)
    assert result == sr


def test_deflated_sharpe_multiple_trials_less():
    sr = 1.5
    single = compute_deflated_sharpe(sharpe=sr, n_obs=252, n_trials=1)
    multi = compute_deflated_sharpe(sharpe=sr, n_obs=252, n_trials=100)
    # Multiple trials should deflate (or equal) single-trial
    assert multi is None or multi <= (single + 1e-9)


def test_deflated_sharpe_returns_float_or_none():
    result = compute_deflated_sharpe(sharpe=0.8, n_obs=100, n_trials=5)
    assert result is None or isinstance(result, float)


def test_deflated_sharpe_insufficient_obs_returns_none():
    result = compute_deflated_sharpe(sharpe=1.0, n_obs=1, n_trials=1)
    assert result is None


def test_deflated_sharpe_zero_sharpe():
    result = compute_deflated_sharpe(sharpe=0.0, n_obs=252, n_trials=1)
    assert result == 0.0


def test_deflated_sharpe_negative_sharpe():
    result = compute_deflated_sharpe(sharpe=-0.5, n_obs=252, n_trials=1)
    assert result is None or isinstance(result, float)


def test_deflated_sharpe_invalid_alpha_returns_none():
    result = compute_deflated_sharpe(sharpe=1.0, n_obs=100, n_trials=5, alpha=1.5)
    assert result is None
