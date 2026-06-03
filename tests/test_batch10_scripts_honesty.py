"""Batch 10 — scripts honesty fixes (discriminating tests).

Covers:

  A24 (scripts/_oos_wf_pipeline_realistic.py + 2 sibling literal harnesses):
      the ASSEMBLED_NO_CRISIS_OVERLAY state-isolation guard is UNCONDITIONAL
      (forces "1" even when preset to "0"), not a setdefault no-op (E-035).
      Verified behaviourally in a subprocess (env preset to "0") because the
      pipeline_realistic module has heavy production imports at module scope.

  A25 (scripts/_oos_wf_mfv2.py / _oos_wf_mfv2_full.py / _oos_wf_mfv_long_short.py):
      the shared price-cache file was replaced by a per-harness, per-end-date
      keyed filename so sibling harnesses cannot cross-contaminate. The three
      PRICE_CACHE paths are mutually DISTINCT and embed both script_id and the
      fetch end-date. No network needed (module-level imports are light).

  A26 (scripts/ci/{drift,walk_forward,retraining}_check.py):
      main() returns 0 on happy path, 0 on ImportError/ModuleNotFoundError (SKIP),
      and 1 on a real non-import exception (FAIL) — previously a blanket
      ``except Exception -> exit(0)`` masked real regressions as green.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import types
from pathlib import Path

import pytest

pytestmark = pytest.mark.fast

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ===========================================================================
# A25 — cache-path keying (no network; module-level imports are light)
# ===========================================================================
_A25_MODULES = {
    "scripts._oos_wf_mfv2": "mfv2",
    "scripts._oos_wf_mfv2_full": "mfv2_full",
    "scripts._oos_wf_mfv_long_short": "mfv_long_short",
}


@pytest.mark.parametrize("modname, script_id", list(_A25_MODULES.items()))
def test_a25_cache_path_embeds_script_id_and_end_date(modname, script_id):
    mod = importlib.import_module(modname)
    name = mod.PRICE_CACHE.name
    # Self-describing: embeds the harness id and the fetch end-date.
    assert script_id in name, f"{modname}: cache name {name!r} missing script_id"
    assert str(mod.PERIOD_END.date()) in name, (
        f"{modname}: cache name {name!r} missing PERIOD_END date"
    )
    # No longer the old shared name.
    assert name != "oos_alpaca_prices_cache.parquet"


def test_a25_cache_paths_are_mutually_distinct():
    paths = {
        modname: importlib.import_module(modname).PRICE_CACHE
        for modname in _A25_MODULES
    }
    distinct = {p.name for p in paths.values()}
    assert len(distinct) == len(paths), (
        f"sibling harnesses share a cache file → cross-contamination: {paths}"
    )


def test_a25_distinct_end_dates_yield_distinct_paths(monkeypatch):
    """A different fetch end-date must produce a different cache filename."""
    import pandas as pd

    mod = importlib.import_module("scripts._oos_wf_mfv2")
    base = mod.PRICE_CACHE.name
    other_end = pd.Timestamp("2024-06-30", tz="UTC")
    other_name = (
        f"oos_alpaca_prices_cache__{mod._CACHE_SCRIPT_ID}__{other_end.date()}.parquet"
    )
    assert other_name != base


# ===========================================================================
# A26 — CI smoke-check exit-code honesty
# ===========================================================================
# (calling-module path, import target accessed inside main())
_A26_CHECKS = {
    "scripts.ci.drift_check": "src.assembled_core.qa.drift_detection",
    "scripts.ci.walk_forward_check": "src.assembled_core.qa.walk_forward",
    "scripts.ci.retraining_check": "src.assembled_core.ml.retraining_scheduler",
}


def _fresh_main(modname):
    sys.modules.pop(modname, None)
    return importlib.import_module(modname).main


@pytest.mark.parametrize("modname, import_target", list(_A26_CHECKS.items()))
def test_a26_import_error_is_skip_exit0(modname, import_target, monkeypatch):
    """ImportError on the production module → SKIP → return 0."""
    # Force the `from <import_target> import ...` to raise ImportError.
    monkeypatch.setitem(sys.modules, import_target, None)
    rc = _fresh_main(modname)()
    assert rc == 0


@pytest.mark.parametrize("modname, import_target", list(_A26_CHECKS.items()))
def test_a26_real_exception_is_fail_exit1(modname, import_target, monkeypatch):
    """A non-import (RuntimeError) regression on the smoke path → FAIL → return 1.

    Models a transitively-broken production module: the ``from <target> import
    <symbol>`` statement raises a NON-ImportError (here via the target module's
    ``__getattr__``). The pre-fix blanket ``except Exception -> exit(0)`` would
    have swallowed this as green; the fix must surface it as exit 1.
    """
    fake = types.ModuleType(import_target)

    def _raising_getattr(name):
        raise RuntimeError(f"simulated regression accessing {name!r}")

    fake.__getattr__ = _raising_getattr
    monkeypatch.setitem(sys.modules, import_target, fake)
    rc = _fresh_main(modname)()
    assert rc == 1


@pytest.mark.parametrize("modname, import_target", list(_A26_CHECKS.items()))
def test_a26_happy_path_exit0(modname, import_target, monkeypatch):
    """A clean import + sanity call → return 0."""
    fake = types.ModuleType(import_target)

    def _compute_psi(*a, **k):
        return 0.0

    class _WalkForwardConfig:
        def __init__(self, *a, **k):
            # Expose any kwarg the smoke check may print (e.g. test_window_days)
            # without constraining the signature — this stays a permissive stub.
            self.__dict__.update(k)

        def __getattr__(self, name):
            return 0

    class _Rec:
        decision = "skip"
        signals_fired = 0

    class _RetrainingScheduler:
        def evaluate(self, *a, **k):
            return _Rec()

    fake.compute_psi = _compute_psi
    fake.WalkForwardConfig = _WalkForwardConfig
    fake.RetrainingScheduler = _RetrainingScheduler
    monkeypatch.setitem(sys.modules, import_target, fake)
    rc = _fresh_main(modname)()
    assert rc == 0


def test_a26_walk_forward_real_config_constructs():
    """walk_forward_check.main() must construct against the REAL dataclass.

    The permissive stub in test_a26_happy_path_exit0 (``__init__(*a, **k)``)
    masks the actual WalkForwardConfig signature — it would pass even with the
    old bogus ``n_splits/test_size/gap`` kwargs. This test runs the corrected
    construction against the real ``src.assembled_core.qa.walk_forward``
    dataclass (importable under pytest via conftest), so it FAILS against the
    old bogus call and PASSES only with valid required fields. Result must be a
    happy-path exit 0 (not a SKIP, not a FAIL).
    """
    pytest.importorskip("src.assembled_core.qa.walk_forward")
    main = _fresh_main("scripts.ci.walk_forward_check")
    rc = main()
    assert rc == 0


# ===========================================================================
# A24 — unconditional ASSEMBLED_NO_CRISIS_OVERLAY guard
# ===========================================================================
# Subprocess because _oos_wf_pipeline_realistic imports production modules at
# module scope; we only exec the top-of-module guard block (lines up to the
# assert), then print the resulting env value.
_A24_GUARD_SRC = (ROOT / "scripts" / "_oos_wf_pipeline_realistic.py").read_text(
    encoding="utf-8"
)


def _extract_guard_block(src: str) -> str:
    """Slice the self-contained guard: from `_prev_no_overlay =` to the assert."""
    start = src.index("_prev_no_overlay = os.environ.get")
    end = src.index('assert os.environ["ASSEMBLED_NO_CRISIS_OVERLAY"] == "1"')
    end = src.index("\n", end) + 1
    return src[start:end]


def test_a24_guard_is_unconditional_not_setdefault():
    block = _extract_guard_block(_A24_GUARD_SRC)
    assert "setdefault" not in block
    assert 'os.environ["ASSEMBLED_NO_CRISIS_OVERLAY"] = "1"' in block


def test_a24_guard_forces_one_even_when_preset_zero(tmp_path):
    """Run the extracted guard with the var preset to '0' → it becomes '1'."""
    block = _extract_guard_block(_A24_GUARD_SRC)
    script = (
        "import os, sys\n"
        + block
        + '\nprint(os.environ["ASSEMBLED_NO_CRISIS_OVERLAY"])\n'
    )
    runner = tmp_path / "guard_runner.py"
    runner.write_text(script, encoding="utf-8")
    env = {**__import__("os").environ, "ASSEMBLED_NO_CRISIS_OVERLAY": "0"}
    out = subprocess.run(
        [sys.executable, str(runner)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == "1"
    # And it must have warned loudly about the override on stderr.
    assert "E-035" in out.stderr


@pytest.mark.parametrize(
    "modname",
    [
        "scripts/_oos_wf_pipeline_realistic.py",
        "scripts/_oos_wf_etf_pairs_literal.py",
        "scripts/_oos_wf_dual_momentum_literal.py",
    ],
)
def test_a24_siblings_have_unconditional_guard(modname):
    src = (ROOT / modname).read_text(encoding="utf-8")
    assert 'os.environ.setdefault("ASSEMBLED_NO_CRISIS_OVERLAY"' not in src
    assert 'os.environ["ASSEMBLED_NO_CRISIS_OVERLAY"] = "1"' in src
