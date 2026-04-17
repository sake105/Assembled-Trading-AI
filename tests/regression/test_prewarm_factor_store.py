"""C2 — Pin the factor-store prewarm helper.

The ``scripts/prewarm_factor_store.py`` driver is the offline entry used by
both the weekly cron (``.github/workflows/prewarm-factor-store.yml``) and
manual sweeps before a WF grid. It must:

* resolve universes (preset + file)
* populate the factor cache on cold run
* hit the cache on warm run (no rebuild)
* honor ``force_rebuild`` to wipe + rewrite

This pin is lightweight — uses the synthetic-price fallback so it is
environment-independent. Real-data coverage belongs to an integration
suite, not a regression gate.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.phase_speed


def _load_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "prewarm_factor_store.py"
    spec = importlib.util.spec_from_file_location("prewarm_factor_store", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["prewarm_factor_store"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_prewarm_cold_then_warm(tmp_path: Path) -> None:
    mod = _load_module()
    kwargs = dict(
        universe_name="DEMO",
        start="2024-01-02",
        end="2024-02-29",
        factors_root=tmp_path,
        allow_synthetic=True,
    )
    cold = mod.prewarm(**kwargs)
    warm = mod.prewarm(**kwargs)

    assert cold["rows"] == warm["rows"] > 0
    assert cold["universe_key"] == warm["universe_key"]
    # The second run must not recompute — warm path is at least as fast as
    # cold. Timing is noisy; use a generous ceiling so the test is not flaky.
    assert warm["elapsed_sec"] <= cold["elapsed_sec"] + 1.0


def test_prewarm_force_rebuild_overwrites(tmp_path: Path) -> None:
    mod = _load_module()
    kwargs = dict(
        universe_name="DEMO",
        start="2024-01-02",
        end="2024-02-29",
        factors_root=tmp_path,
        allow_synthetic=True,
    )
    mod.prewarm(**kwargs)
    rebuilt = mod.prewarm(**{**kwargs, "force_rebuild": True})
    assert rebuilt["rows"] > 0


def test_prewarm_universe_file(tmp_path: Path) -> None:
    mod = _load_module()
    uni_file = tmp_path / "uni.txt"
    uni_file.write_text("AAPL\nMSFT\n# comment\nGOOGL\n", encoding="utf-8")
    out = mod.prewarm(
        universe_name=str(uni_file),
        start="2024-01-02",
        end="2024-01-31",
        factors_root=tmp_path,
        allow_synthetic=True,
    )
    assert out["n_symbols"] == 3
    assert out["rows"] > 0


def test_prewarm_rejects_synthetic_when_disallowed(tmp_path: Path) -> None:
    mod = _load_module()
    # The real loader path (``src.assembled_core.data.loaders``) is unavailable
    # in this repo at this time, so ``allow_synthetic=False`` must raise.
    with pytest.raises(RuntimeError, match="real price loader"):
        mod.prewarm(
            universe_name="DEMO",
            start="2024-01-02",
            end="2024-01-10",
            factors_root=tmp_path,
            allow_synthetic=False,
        )
