"""Hardening regression tests for config/feature_flags.py and config/policy_loader.py.

Covers four diagnostic findings (docs/Diagnostik.md):

  A19  feature_flags canary selection must be deterministic across process
       restarts (the built-in hash() is per-process salted). Verified via a
       cross-PYTHONHASHSEED subprocess determinism test.
  A53  feature_flags.is_active/is_shadow must fail loud on an unknown flag name
       instead of silently defaulting to "off" (inactive).
  A20  policy_loader.load_policy must RE-RAISE on a broken policy_schema import
       (validation cannot run) and only WARN (not DEBUG-swallow) on validation
       content errors.
  A52  policy_loader leverage-conflict guard must WARN — not silently pass — on
       a malformed policy_no_leverage.yaml.

Each assertion is written to FAIL against the pre-fix code.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from src.assembled_core.config.feature_flags import FeatureFlags
from src.assembled_core.config import policy_loader
from src.assembled_core.config.policy_loader import load_policy

# Repo layout: this file is tests/test_config_hardening.py → repo root is parent.parent.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"


# ---------------------------------------------------------------------------
# A19 — deterministic canary selection across PYTHONHASHSEED values
# ---------------------------------------------------------------------------


def _run_canary_in_subprocess(hashseed: str) -> str:
    """Run is_active("...","TICKER") for several tickers in a fresh interpreter.

    Returns a stable string like "AAPL=False,MSFT=True,...". With the built-in
    hash(), the result depends on PYTHONHASHSEED; with hashlib it does not.
    """
    code = textwrap.dedent(
        """
        from assembled_core.config.feature_flags import FeatureFlags
        flags = FeatureFlags(news_topic_clustering="canary")
        tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "META", "NVDA", "AMD"]
        parts = [t + "=" + str(flags.is_active("news_topic_clustering", t)) for t in tickers]
        print(",".join(parts))
        """
    )
    env = {
        # Minimal but sufficient child env. Inherit current env, then override.
        **_child_base_env(),
        "PYTHONHASHSEED": hashseed,
        "PYTHONPATH": str(_SRC_DIR),
    }
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(_REPO_ROOT),
    )
    assert proc.returncode == 0, (
        f"subprocess failed (seed={hashseed}): "
        f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    return proc.stdout.strip()


def _child_base_env() -> dict[str, str]:
    import os

    # Copy the parent env so the child can find the interpreter / DLLs on Windows,
    # but we will override PYTHONHASHSEED + PYTHONPATH explicitly.
    return dict(os.environ)


def test_a19_canary_deterministic_across_hashseeds():
    """Canary membership must be identical under different PYTHONHASHSEED values.

    Pre-fix (built-in hash()): the salted string hash differs per seed, so the
    two outputs diverge and this assertion fails. Post-fix (hashlib.md5): the
    digest is seed-independent, so both outputs match.
    """
    out_seed0 = _run_canary_in_subprocess("0")
    out_seed1 = _run_canary_in_subprocess("1")
    assert out_seed0 == out_seed1, (
        "Canary selection must be deterministic across PYTHONHASHSEED values.\n"
        f"  PYTHONHASHSEED=0 -> {out_seed0}\n"
        f"  PYTHONHASHSEED=1 -> {out_seed1}"
    )
    # Sanity: the output is the expected shape and at least one ticker resolved.
    assert "AAPL=" in out_seed0


def test_a19_canary_membership_matches_hashlib_reference():
    """Pin the exact hashlib bucket semantics so the implementation can't drift."""
    import hashlib

    flags = FeatureFlags(news_topic_clustering="canary")
    for ticker in ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "NVDA", "AMD", "INTC"]:
        expected = int(hashlib.md5(ticker.encode()).hexdigest(), 16) % 10 == 0
        assert flags.is_active("news_topic_clustering", ticker) is expected

    # Empty ticker is never a canary member (guard preserved).
    assert flags.is_active("news_topic_clustering", "") is False


# ---------------------------------------------------------------------------
# A53 — fail loud on unknown flag name
# ---------------------------------------------------------------------------


def test_a53_unknown_flag_raises_value_error():
    flags = FeatureFlags()
    with pytest.raises(ValueError, match="Unknown feature flag: totally_made_up"):
        flags.is_active("totally_made_up")


def test_a53_unknown_flag_raises_in_is_shadow():
    flags = FeatureFlags()
    with pytest.raises(ValueError, match="Unknown feature flag: typo_flag"):
        flags.is_shadow("typo_flag")


def test_a53_known_flag_still_returns_state():
    """Happy path is unchanged: declared flags resolve to their state."""
    flags = FeatureFlags(trend_baseline="on", regime_ml_model="shadow")
    assert flags.is_active("trend_baseline") is True
    assert flags.is_active("regime_ml_model") is False  # shadow -> inactive
    assert flags.is_shadow("regime_ml_model") is True
    assert flags.is_shadow("trend_baseline") is False


# ---------------------------------------------------------------------------
# A20 — policy schema validation: re-raise on import failure, warn on content
# ---------------------------------------------------------------------------


def _write_valid_policy(tmp_path: Path) -> Path:
    p = tmp_path / "policy.yaml"
    p.write_text(
        textwrap.dedent(
            """
            scope:
              leverage_allowed: false
            risk:
              max_drawdown: 0.2
            """
        ),
        encoding="utf-8",
    )
    return p


def test_a20_reraises_on_schema_import_error(tmp_path, monkeypatch):
    """A broken policy_schema import must propagate, not be swallowed.

    We force the import inside load_policy to fail by making the symbol lookup
    raise ImportError. Pre-fix this was caught by the broad `except Exception`
    and logged at DEBUG (load returned silently); post-fix it re-raises.
    """
    import src.assembled_core.config.policy_schema as schema_mod

    # load_policy does `from ...policy_schema import validate_policy, ...`.
    # Deleting the attribute makes that `from ... import` raise ImportError,
    # which post-fix must propagate out of load_policy.
    monkeypatch.delattr(schema_mod, "validate_policy", raising=True)

    policy_file = _write_valid_policy(tmp_path)
    policy_loader._POLICY_CACHE.clear()
    with pytest.raises(ImportError):
        load_policy(policy_file, validate=True)


def test_a20_warns_on_validation_content_error(tmp_path, monkeypatch, caplog):
    """A validator that raises a content error must WARN (not DEBUG) and proceed."""
    import src.assembled_core.config.policy_schema as schema_mod

    def _raise_value_error(_policy):
        raise ValueError("simulated validation content failure")

    monkeypatch.setattr(schema_mod, "validate_policy", _raise_value_error)

    policy_file = _write_valid_policy(tmp_path)
    policy_loader._POLICY_CACHE.clear()
    with caplog.at_level(logging.WARNING, logger=policy_loader.logger.name):
        result = load_policy(policy_file, validate=True)

    # Load proceeded and returned the parsed policy.
    assert isinstance(result, dict)
    assert result.get("scope", {}).get("leverage_allowed") is False
    # A WARNING (not a silent DEBUG) was emitted, mentioning validation.
    warnings = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "validation" in r.message
    ]
    assert warnings, (
        "Expected a WARNING about failed policy validation; "
        f"got records: {[(r.levelname, r.message) for r in caplog.records]}"
    )


def test_a20_valid_policy_loads_cleanly(tmp_path, caplog):
    """Successful validation path is unchanged: no WARNING/ERROR, policy returned."""
    policy_file = _write_valid_policy(tmp_path)
    policy_loader._POLICY_CACHE.clear()
    with caplog.at_level(logging.WARNING, logger=policy_loader.logger.name):
        result = load_policy(policy_file, validate=True)

    assert result.get("scope", {}).get("leverage_allowed") is False
    # No validation/import warning or error should be logged on the happy path.
    bad = [
        r
        for r in caplog.records
        if r.levelno >= logging.WARNING
        and ("validation" in r.message or "schema" in r.message)
    ]
    assert not bad, (
        f"Unexpected warnings/errors on valid policy: {[r.message for r in bad]}"
    )


# ---------------------------------------------------------------------------
# A52 — leverage-conflict guard must warn (not silent pass) on malformed file
# ---------------------------------------------------------------------------


def test_a52_malformed_no_leverage_file_warns(tmp_path, caplog):
    """A malformed policy_no_leverage.yaml must emit a WARNING, not silently pass.

    The guard reads `<active>.parent / 'policy_no_leverage.yaml'`. We make that
    sibling file malformed so the parse/compare inside the guard raises. Pre-fix
    the bare `except Exception: pass` hid this; post-fix it warns.
    """
    policy_file = _write_valid_policy(tmp_path)

    # Sibling no-leverage file that is valid YAML but NOT a mapping at .get() level:
    # a top-level list makes `_no_lev.get(...)` raise AttributeError inside the guard.
    no_lev = tmp_path / "policy_no_leverage.yaml"
    no_lev.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

    policy_loader._POLICY_CACHE.clear()
    with caplog.at_level(logging.WARNING, logger=policy_loader.logger.name):
        result = load_policy(policy_file, validate=False)

    assert isinstance(result, dict)
    guard_warnings = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "leverage-conflict guard" in r.message
    ]
    assert guard_warnings, (
        "Expected a WARNING from the leverage-conflict guard on a malformed "
        f"policy_no_leverage.yaml; got: {[(r.levelname, r.message) for r in caplog.records]}"
    )
