# Dummy-Daten & Information-Flow Remediation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate silent dummy-data fallbacks in production import paths and reduce information-flow noise (script wildwuchs, config-dir duplication, stale worktrees) so future developers and agents can find the truth fast.

**Architecture:** Two independent sub-projects executed in any order. Sub-Project A (Dummy-Data Honest Tagging) hardens callers via fail-loud + explicit opt-in for sample/test data. Sub-Project B (Information-Flow Consolidation) physically consolidates duplicated/decayed structure (config dirs, throwaway scripts, stale worktrees, notebook split) without touching trading-core code.

**Tech Stack:** Python 3.11+, pytest, ruff/black pre-commit hooks. No new dependencies introduced.

**Scope guardrails (per CLAUDE.md):**
- Rule 30 (Risk/Execution sensitive zones): NOT touched in this plan.
- Rule 60 (one problem per change): Each task is one atomic commit.
- Rule 50 (architecture boundaries): No new cross-layer coupling.
- Improvements beyond Dummy + Info-Flow are catalogued in `KNOWN_ISSUES.md` (Section 6 — Strategic Quant Gaps), NOT executed here.

**Compass artifact reference:** Findings in `autonome_weiterarbeit/wichtig/compass_artifact_wf-738112f8-7a72-47d7-8553-dd9abf241d96_text_markdown.md`. Several claims verified outdated (test collection clean, sprint-legacy gone, drift-persistence done, pre-trade sector/region done, labeling outperformance done, trade-metrics done, ML __init__ clean). Only the genuinely-still-present problems are addressed below.

---

## Verified Current State (2026-05-17 baseline)

| Compass Claim | Verified Reality | Action |
|---|---|---|
| ~19 Test-Collection-Failures in `data/` | **CLEAN** — 7084 tests collected, no errors | None |
| Sprint9/10 legacy scripts | **REMOVED** | None |
| Insider/Shipping dummy generators | **CONFIRMED REAL** — `src/.../data/insider_ingest.py` + `shipping_routes_ingest.py` generate dummy data when `path=None`; 6 prod call sites | **Sub-Project A** |
| Monitoring API dummy fallback | **PARTIAL** — drift parquet now written, but missing-file fallback still returns dummy | **Sub-Project A** |
| Empty research notebooks | **3 OF 4 EMPTY** — `altdata/`, `meta/`, `risk/` have 1 cell each (2KB); only `trend/` has 14 cells | **Sub-Project A** |
| ~91 scripts | **232 SCRIPTS TOTAL / 140 TOP-LEVEL** (worse) — 8 `_append_batchN.py` throwaways at top-level | **Sub-Project B** |
| `config/` vs `configs/` | **CONFIRMED** — `config/env/` is the only thing in `config/`, `configs/` holds everything real | **Sub-Project B** |
| ML `__init__.py` NotImplementedError | **CLEAN** — file is empty `__all__: list[str] = []` | None |
| Phase-marker drift | **CONSOLIDATED** — phase4..13 are aliased to `fast` in pyproject.toml | None (KNOWN_ISSUES note only) |
| `notebooks/` vs `research/` | **REAL DUPLICATION** — `notebooks/` has 1 .py file, `research/` has 4 .ipynb | **Sub-Project B** |
| `.claude/worktrees/` stale | **70 MB** in 2 dead agent worktrees | **Sub-Project B** |

---

## Sub-Project A — Dummy-Data Honest Tagging

**Why this matters:** Per CLAUDE.md §7.4 "Datenprobleme nicht still verschlucken" — a code path that silently returns dummy data when no real data is provided produces false confidence in backtests, QA gates, and dataset builders. Make the dummy path **loud and opt-in only**, never the default.

**Why not "delete the dummies":** Removing the generators entirely breaks 6 production import sites. Instead: keep the generator functions but force callers to either pass a real path OR explicit `allow_sample=True`.

### Task A1: Insider ingest — fail-loud by default, explicit opt-in for sample

**Files (VERIFIED 2026-05-17 against actual source):**
- Modify: `src/assembled_core/data/insider_ingest.py` (function `load_insider_sample`)
- Modify (4 prod callers): `scripts/commands/ml.py:108-110`, `scripts/run_backtest_strategy.py:379-380`, `src/assembled_core/qa/dataset_builder.py:435-436`, plus 1 nested call in `scripts/commands/ml.py`
- Modify (1 research notebook caller): `research/altdata/insider_congress_shipping_exploration.ipynb` (Sub-Project A / Task A4 already touches this file — verify schema-flag added consistently)
- Create: `tests/test_insider_ingest_failloud.py` (NEW)
- Verify-only: `tests/test_signals_event_phase6.py` (existing file; has NO parameterless calls — no edit required)

**Note on Senior-Review correction (F-senior-1/2/3):** The earlier draft of this plan referenced `tests/test_features_events_phase6.py` (nonexistent — actual file is `test_signals_event_phase6.py` with zero parameterless callers) and `scripts/generate_sample_event_data.py` as a caller (actual: imports `normalize_insider`, not `load_insider_sample`). The corrected caller set is the 4 prod files + 1 notebook above. The actual dummy schema is `timestamp, symbol, trades_count, net_shares, role` (verified at `insider_ingest.py:73-80`) — NOT `ticker, date, insider_name, transaction_type, shares, value` as the earlier draft asserted.

- [ ] **Step 1: Write failing tests for the new contract**

In `tests/test_insider_ingest_failloud.py` (NEW):

```python
"""Sub-Project A / Task A1 — insider_ingest fail-loud contract.

Schema reference (verified at src/assembled_core/data/insider_ingest.py:73-80):
columns = timestamp (UTC), symbol, trades_count, net_shares, role.
"""

from __future__ import annotations

import pytest

from src.assembled_core.data.insider_ingest import load_insider_sample


def test_load_insider_sample_without_allow_sample_raises():
    """No real path + no explicit opt-in → ValueError (no silent dummy)."""
    with pytest.raises(ValueError, match="allow_sample=True"):
        load_insider_sample(path=None)


def test_load_insider_sample_with_allow_sample_returns_dummy():
    """Explicit opt-in → dummy data returned with the documented sample schema."""
    df = load_insider_sample(path=None, allow_sample=True)
    assert not df.empty
    # Schema per insider_ingest.py:73-80 (NOT ticker/date)
    assert "symbol" in df.columns
    assert "timestamp" in df.columns
    assert "trades_count" in df.columns
    assert "net_shares" in df.columns
    assert "role" in df.columns


def test_load_insider_sample_with_real_path_loads_file(tmp_path):
    """Real path overrides allow_sample (real data always wins)."""
    import pandas as pd

    real_path = tmp_path / "real_insider.parquet"
    pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01"], utc=True),
            "symbol": ["AAPL"],
            "trades_count": [3],
            "net_shares": [1000],
            "role": ["CEO"],
        }
    ).to_parquet(real_path)
    df = load_insider_sample(path=str(real_path), allow_sample=False)
    assert df.iloc[0]["symbol"] == "AAPL"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_insider_ingest_failloud.py -v`
Expected: FAIL — current `load_insider_sample(None)` returns dummy silently.

- [ ] **Step 3: Update `insider_ingest.py` to add `allow_sample` flag and fail-loud default**

In `src/assembled_core/data/insider_ingest.py`, modify `load_insider_sample`. **Concrete edit instructions for Sonnet (no ellipsis placeholders):**

1. Change the function signature from `def load_insider_sample(path: str | None = None) -> pd.DataFrame:` to `def load_insider_sample(path: str | None = None, *, allow_sample: bool = False) -> pd.DataFrame:`.
2. Update the docstring to describe the new `allow_sample` parameter (see existing param/return blocks for style; add a `Raises: ValueError` block).
3. Locate the line `# Generate dummy data` (currently at insider_ingest.py:65) — INSERT immediately before it:

```python
    if not allow_sample:
        raise ValueError(
            "load_insider_sample() received no path and no explicit "
            "allow_sample=True. Production callers must provide a real "
            "insider data file; tests/dev callers must pass allow_sample=True "
            "to opt into dummy sample data."
        )

```

4. **Do NOT delete** the existing dummy-data block (lines 66-85 in current file) and do **NOT** modify the file-load branch (`if path is not None:` at lines 36-63) or the helper functions `normalize_insider`, `validate_insider_schema`. Only the guard insertion and signature/docstring change are in scope.

- [ ] **Step 4: Run new tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_insider_ingest_failloud.py -v`
Expected: 3/3 PASS.

- [ ] **Step 5: Update production callers to pass `allow_sample=True` explicitly**

For each of (verified set):
- `scripts/commands/ml.py:108-110` — sample-context (no real feed wired)
- `scripts/run_backtest_strategy.py:379-380` — sample-context
- `src/assembled_core/qa/dataset_builder.py:435-436` — sample-context

In each call site, locate `load_insider_sample(...)` and add `allow_sample=True` as a keyword argument with a brief inline comment: `# intentional sample — no live insider feed wired (see KNOWN_ISSUES §6.5.5)`. Do NOT add the flag when a non-None `path` is being passed.

- [ ] **Step 6: Update the notebook caller**

`research/altdata/insider_congress_shipping_exploration.ipynb` imports and calls the dummy loader (verified via `grep -l load_insider_sample research/altdata/*.ipynb`). Edit the notebook JSON to add `allow_sample=True` to the call site. Use this Python snippet to avoid hand-editing JSON:

```python
import json
from pathlib import Path

nb_path = Path("research/altdata/insider_congress_shipping_exploration.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))
for cell in nb["cells"]:
    if cell.get("cell_type") != "code":
        continue
    for i, line in enumerate(cell.get("source", [])):
        if "load_insider_sample(" in line and "allow_sample" not in line:
            cell["source"][i] = line.replace(
                "load_insider_sample(", "load_insider_sample(allow_sample=True, "
            )
        if "load_shipping_sample(" in line and "allow_sample" not in line:
            cell["source"][i] = line.replace(
                "load_shipping_sample(", "load_shipping_sample(allow_sample=True, "
            )
nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
```

Note: `ensure_ascii=False` preserves any umlauts/special chars (Senior-Review F-senior-10).

- [ ] **Step 7: Run targeted test verification**

```
.venv/Scripts/python.exe -m pytest tests/test_insider_ingest_failloud.py tests/test_signals_event_phase6.py tests/test_run_backtest_strategy.py -v
```

Expected: PASS. (Note: `test_signals_event_phase6.py` has NO calls to `load_insider_sample` — it's listed only as a sanity check that nothing in the events suite regresses.)

- [ ] **Step 8: Commit**

```bash
git add src/assembled_core/data/insider_ingest.py \
        tests/test_insider_ingest_failloud.py \
        scripts/commands/ml.py \
        scripts/run_backtest_strategy.py \
        src/assembled_core/qa/dataset_builder.py \
        research/altdata/insider_congress_shipping_exploration.ipynb
git commit -m "fix(data): insider_ingest fail-loud + explicit allow_sample opt-in"
```

### Task A1b: Congress trades ghost-module — fail-loud, not silent-except

**Files (added per Senior-Review F-senior-6 + F-senior-15):**
- Modify: `src/assembled_core/pipeline/trading_cycle_shared.py:625-647` (narrow the exception)
- Update: `KNOWN_ISSUES.md` §6.5.5 to capture the ghost-module status

**Rationale:** `trading_cycle_shared.py:629` imports `congress_trades_ingest.load_congress_sample`, but `src/assembled_core/data/congress_trades_ingest.py` **does not exist** in the current repo (only stale `__pycache__` artifacts and a copy in `.claude/worktrees/agent-a700e54f/` survive). The import is wrapped in a bare `except Exception: logger.debug(...)` at line 645, so `include_congress=True` is silently a no-op. This matches the exact "silent fail-open in feature wiring" anti-pattern documented as E-019 in `docs/CLAUDE_CODING_ERRORS.md`.

**Scope-limit:** Do NOT restore the missing module here (that requires a real congress-trades data feed and is a multi-day effort tracked in KNOWN_ISSUES §6.5.5). The fix in this task is to **make the silence visible**, not to restore the feature.

- [ ] **Step 1: Locate the silent-except block at `trading_cycle_shared.py:625-647`**

The block currently reads (verified):

```python
        try:
            from src.assembled_core.data.congress_trades_ingest import (
                load_congress_sample,
            )
            from src.assembled_core.features.congress_features import (
                add_congress_features,
            )

            congress_path = getattr(feature_cfg_obj, "congress_data_path", None)
            congress_events = load_congress_sample(path=congress_path)
            if not congress_events.empty:
                prices_with_features = add_congress_features(
                    prices_with_features,
                    congress_events,
                    as_of=ctx.as_of,
                )
                logger.debug("[Features] Congress trading features merged")
        except Exception as e:
            logger.debug("[Features] Congress features skipped: %s", e)
```

- [ ] **Step 2: Replace the bare `except Exception` with narrow exception handlers**

```python
        except ModuleNotFoundError as e:
            logger.warning(
                "[Features] Congress features SILENTLY DISABLED — "
                "module congress_trades_ingest is not installed: %s. "
                "Set feature_cfg.include_congress=False to suppress this warning, "
                "or restore the module (see KNOWN_ISSUES.md §6.5.5).",
                e,
            )
        except ImportError as e:
            logger.warning(
                "[Features] Congress features import failed: %s", e
            )
```

Do NOT add a catch-all for other exceptions — they must propagate so real bugs surface.

- [ ] **Step 3: Write a regression test**

In `tests/test_pipeline_congress_failloud.py` (NEW):

```python
"""Sub-Project A / Task A1b — congress import surfaces missing module loudly."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from src.assembled_core.pipeline.trading_cycle_shared import (
    add_event_features_to_prices,  # adjust name if different in your repo
)


def test_congress_missing_module_logs_warning(caplog):
    """When include_congress=True but the module is absent → WARNING, not silent."""
    ctx = SimpleNamespace(as_of=pd.Timestamp("2024-01-01", tz="UTC"))
    feature_cfg = SimpleNamespace(include_congress=True, congress_data_path=None)
    prices = pd.DataFrame({"symbol": ["AAPL"], "close": [100.0]})

    with caplog.at_level(logging.WARNING):
        result = add_event_features_to_prices(prices, ctx, feature_cfg)

    # Result should be unchanged (feature was disabled), but warning must be emitted
    assert any(
        "Congress features SILENTLY DISABLED" in record.message
        or "Congress features import failed" in record.message
        for record in caplog.records
    ), f"Expected a Congress-related WARNING. Got: {[r.message for r in caplog.records]}"
```

**If the actual function name in `trading_cycle_shared.py` differs from `add_event_features_to_prices`**, grep first and adjust the import/call accordingly. Sonnet: do a quick grep `grep -n "def add.*features\|def _add_event" src/assembled_core/pipeline/trading_cycle_shared.py` before writing this test.

- [ ] **Step 4: Run test, then run pipeline tests for regression**

```
.venv/Scripts/python.exe -m pytest tests/test_pipeline_congress_failloud.py tests/test_trading_cycle_shared.py -v
```

Expected: PASS. The regression test should fail before Step 2, pass after.

- [ ] **Step 5: Update `KNOWN_ISSUES.md` §6.5.5**

Append a paragraph to §6.5.5 stating:
> **Concrete status (2026-05-17):** `src/assembled_core/data/congress_trades_ingest.py` does not exist in current repo (only stale `__pycache__` artifacts remain). `trading_cycle_shared.py:625-647` imports the module inside a try/except that now (post-Task A1b) emits a WARNING when the module is absent instead of silently swallowing the ModuleNotFoundError. Restoring the module requires a real congress-trades data source — track here, do not paper over.

- [ ] **Step 6: Commit**

```bash
git add src/assembled_core/pipeline/trading_cycle_shared.py \
        tests/test_pipeline_congress_failloud.py \
        KNOWN_ISSUES.md
git commit -m "fix(pipeline): narrow congress-module silent-except to ModuleNotFoundError WARN"
```

### Task A2: Shipping ingest — same pattern, explicit shipping schema

**Files (VERIFIED 2026-05-17):**
- Modify: `src/assembled_core/data/shipping_routes_ingest.py` (function `load_shipping_sample`)
- Modify: same prod callers as A1 (`scripts/commands/ml.py:109`, `scripts/run_backtest_strategy.py:380`, `src/assembled_core/qa/dataset_builder.py:436`)
- Modify: notebook caller already covered in A1 Step 6 (same notebook, single edit)
- Create: `tests/test_shipping_ingest_failloud.py` (NEW)

**Shipping schema (verified at `shipping_routes_ingest.py:98-108`):** `timestamp, route_id, port_from, port_to, symbol, ships, congestion_score`.

- [ ] **Step 1: Write failing tests in `tests/test_shipping_ingest_failloud.py`**

```python
"""Sub-Project A / Task A2 — shipping_routes_ingest fail-loud contract.

Schema reference (verified at src/assembled_core/data/shipping_routes_ingest.py:98-108):
columns = timestamp (UTC), route_id, port_from, port_to, symbol, ships, congestion_score.
"""

from __future__ import annotations

import pytest

from src.assembled_core.data.shipping_routes_ingest import load_shipping_sample


def test_load_shipping_sample_without_allow_sample_raises():
    with pytest.raises(ValueError, match="allow_sample=True"):
        load_shipping_sample(path=None)


def test_load_shipping_sample_with_allow_sample_returns_dummy():
    df = load_shipping_sample(path=None, allow_sample=True)
    assert not df.empty
    for col in (
        "timestamp",
        "route_id",
        "port_from",
        "port_to",
        "symbol",
        "ships",
        "congestion_score",
    ):
        assert col in df.columns, f"Missing column {col} in sample schema"


def test_load_shipping_sample_with_real_path_loads_file(tmp_path):
    import pandas as pd

    real_path = tmp_path / "real_shipping.parquet"
    pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01"], utc=True),
            "route_id": ["US-EU-001"],
            "port_from": ["NYC"],
            "port_to": ["HAM"],
            "symbol": ["MSFT"],
            "ships": [12],
            "congestion_score": [45],
        }
    ).to_parquet(real_path)
    df = load_shipping_sample(path=str(real_path), allow_sample=False)
    assert df.iloc[0]["route_id"] == "US-EU-001"
```

- [ ] **Step 2: Run, expect FAIL**

```
.venv/Scripts/python.exe -m pytest tests/test_shipping_ingest_failloud.py -v
```

- [ ] **Step 3: Update `shipping_routes_ingest.py`**

In `src/assembled_core/data/shipping_routes_ingest.py`:
1. Change signature from `def load_shipping_sample(path: str | None = None) -> pd.DataFrame:` to `def load_shipping_sample(path: str | None = None, *, allow_sample: bool = False) -> pd.DataFrame:`.
2. Update docstring (mirror insider_ingest after A1 step 3).
3. Locate the line `# Generate dummy data` (currently around shipping_routes_ingest.py:71). Insert immediately BEFORE it:

```python
    if not allow_sample:
        raise ValueError(
            "load_shipping_sample() received no path and no explicit "
            "allow_sample=True. Production callers must provide a real "
            "shipping data file; tests/dev callers must pass allow_sample=True."
        )

```

4. Do not modify the file-load branch or `normalize_shipping`.

- [ ] **Step 4: Run tests, expect PASS**

- [ ] **Step 5: Update prod callers**

For each of `scripts/commands/ml.py:109`, `scripts/run_backtest_strategy.py:380`, `src/assembled_core/qa/dataset_builder.py:436`, add `allow_sample=True` to the `load_shipping_sample(...)` call. Same inline comment style as A1 Step 5.

- [ ] **Step 6: Verify notebook caller already updated by A1 Step 6**

A1 Step 6 updates both insider AND shipping calls in the same notebook (the snippet handles both). Verify with:

```bash
grep -n "load_shipping_sample\|load_insider_sample" research/altdata/insider_congress_shipping_exploration.ipynb
```

Each occurrence should now contain `allow_sample=True`.

- [ ] **Step 7: Commit**

```bash
git add src/assembled_core/data/shipping_routes_ingest.py \
        tests/test_shipping_ingest_failloud.py \
        scripts/commands/ml.py \
        scripts/run_backtest_strategy.py \
        src/assembled_core/qa/dataset_builder.py
git commit -m "fix(data): shipping_routes_ingest fail-loud + explicit allow_sample opt-in"
```

### Task A3: Monitoring API drift docstring — match the (already-correct) 503 behavior

**Senior-Review correction (F-senior-4):** The earlier draft proposed replacing dummy fallback with HTTP 404. That premise was wrong. **The behavior is already correct** — `src/assembled_core/api/routers/monitoring.py:369-376` already raises `HTTPException(status_code=503, ...)` when the drift parquet is missing (audit C3-023 / C4-033 was previously addressed). The remaining residue is a **stale docstring** at lines 280-296 that still says *"Currently returns dummy/example data as drift analysis persistence is not yet implemented."*

**Files:**
- Modify: `src/assembled_core/api/routers/monitoring.py:280-296` (docstring only)
- Optionally extend: `tests/test_api_monitoring.py` to add an explicit 503-missing-file test if one is not already present

**Why 503 (not 404):** The endpoint is available; the *resource* the endpoint depends on is temporarily unavailable. Semantically correct per HTTP/RFC 9110 §15.6.4. Do **not** change the status code.

- [ ] **Step 1: Verify the missing-file behavior already returns 503**

```
grep -n "status_code=503\|status_code=404" src/assembled_core/api/routers/monitoring.py
```

Expected: a `status_code=503` near line 370. Confirm before proceeding.

- [ ] **Step 2: Check whether a missing-file test already exists**

```
grep -n "drift_status\|drift_results_file" tests/test_api_monitoring.py
```

If a test already covers the 503 path: no test addition needed in Step 4.

- [ ] **Step 3: Update the stale docstring**

In `src/assembled_core/api/routers/monitoring.py`, locate the `get_drift_status_summary` function (around line 274) and replace its docstring with:

```python
    """Get drift status summary for monitoring.

    Returns the status of the last feature drift analysis, showing which features
    have drifted and their severity. Reads persisted drift results written by
    `qa.drift_detection.save_drift_results()` to `output/drift_analysis_{freq}.parquet`.

    Args:
        freq: Trading frequency ("1d" or "5min"), default "1d"
        top_n: Number of top features with drift to return (default: 10, max: 50)

    Returns:
        DriftStatusSummary with overall severity, top features with drift, and total features checked

    Raises:
        HTTPException: 400 if `freq` is unsupported; 503 if no drift analysis has
            been persisted yet for this frequency; 500 for unexpected errors.
    """
```

(Remove the obsolete sentence about "dummy/example data".)

- [ ] **Step 4 (conditional): Add a 503 missing-file test if not present**

Only do this step if Step 2 found no existing coverage. Add to `tests/test_api_monitoring.py`:

```python
def test_drift_status_returns_503_when_parquet_missing(tmp_path, monkeypatch):
    """Senior-Review F-senior-4: missing drift parquet → 503 (service unavailable)."""
    monkeypatch.setattr(
        "src.assembled_core.api.routers.monitoring.OUTPUT_DIR", tmp_path
    )
    from fastapi.testclient import TestClient
    from src.assembled_core.api.app import app

    client = TestClient(app)
    r = client.get("/monitoring/drift_status?freq=1d")
    assert r.status_code == 503
    assert "drift analysis" in r.json()["detail"].lower()
```

- [ ] **Step 5: Run tests**

```
.venv/Scripts/python.exe -m pytest tests/test_api_monitoring.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/assembled_core/api/routers/monitoring.py tests/test_api_monitoring.py
git commit -m "docs(api): drift_status docstring matches actual 503 behavior (not dummy)"
```

### Task A4: Move skeleton research notebooks to `research/dead_ends/`

**Senior-Review correction (F-senior-10):** The earlier draft prepended a "STATUS: SKELETON" marker cell to in-place notebooks. That is a half-measure: a 2 KB skeleton with a banner is still a 2 KB skeleton. The directory `research/dead_ends/` **already exists** for exactly this purpose. Move the truly-empty notebooks there.

**Decision criteria (decide BEFORE moving):**
- A notebook with ≤ 2 cells and < 3 KB OR no markdown analysis cells → MOVE to `dead_ends/`.
- A notebook with concrete in-progress analysis (multi-cell, code+markdown) → leave in place.

**Verified state (2026-05-17):**
- `research/altdata/insider_congress_shipping_exploration.ipynb` — 1 cell, 2.2 KB → MOVE
- `research/meta/meta_model_calibration.ipynb` — 1 cell, 2.2 KB → MOVE
- `research/risk/scenario_and_risk_experiments.ipynb` — 1 cell, 2.0 KB → MOVE
- `research/trend/trend_baseline_experiments.ipynb` — 14 cells, 10 KB → LEAVE IN PLACE

**Caveat — interaction with Task A1 Step 6:** Task A1 Step 6 modifies `research/altdata/insider_congress_shipping_exploration.ipynb` (adds `allow_sample=True`). **Execute Task A1 BEFORE Task A4** so the move includes the updated file. Task ordering already specified in the Execution Notes.

- [ ] **Step 1: Create destination directory if missing and move the 3 thin notebooks**

```bash
mkdir -p research/dead_ends
git mv research/altdata/insider_congress_shipping_exploration.ipynb \
       research/dead_ends/altdata-insider_congress_shipping_exploration.ipynb
git mv research/meta/meta_model_calibration.ipynb \
       research/dead_ends/meta-meta_model_calibration.ipynb
git mv research/risk/scenario_and_risk_experiments.ipynb \
       research/dead_ends/risk-scenario_and_risk_experiments.ipynb
```

The flat naming `<originalsubdir>-<originalname>.ipynb` preserves provenance.

- [ ] **Step 2: Add a README to `research/dead_ends/` if not present**

```bash
test -f research/dead_ends/README.md || cat > research/dead_ends/README.md <<'EOF'
# research/dead_ends/

Notebooks and analyses that were started but never completed. Preserved for
provenance and to help future work avoid the same path. NOT to be treated
as conclusions.

Each filename uses the convention `<originalsubdir>-<originalname>.ipynb` so
the original location is recoverable from the name alone.

Tracked in `KNOWN_ISSUES.md` §6.7 (Research Notebook Completion).
EOF
```

- [ ] **Step 3: Search for and update any references to the moved paths**

```bash
grep -rn "altdata/insider_congress_shipping_exploration\|meta/meta_model_calibration\|risk/scenario_and_risk_experiments" \
    --include="*.py" --include="*.md" --include="*.yml" --include="*.yaml" \
    2>&1 | grep -v __pycache__ | grep -v ".claude/worktrees" | grep -v ".git/"
```

Update any hits with the new `research/dead_ends/...` paths. Skip hits in `archive/` or `.claude/worktrees/`.

- [ ] **Step 4: Commit**

```bash
git add -A research/
git commit -m "docs(research): move skeleton notebooks to dead_ends/ for honest provenance"
```

---

## Sub-Project B — Information-Flow Consolidation

**Why this matters:** Per CLAUDE.md §70 token-discipline + the user's stated "Überblick verloren" feeling: the more visual noise in scripts/, configs/, worktrees/, and the docs surface, the harder it is for any future contributor (human or agent) to find the truth. Reducing surface area is the highest-leverage non-functional change.

### Task B1: Delete `_append_batchN.py` throwaway scripts

**Files:**
- Delete: `scripts/_append_batch14.py` through `scripts/_append_batch21.py` (8 files)

**Rationale:** Underscore-prefixed, numbered batch scripts are clearly throwaway. Verify no callers, then delete.

- [ ] **Step 1: Verify no callers**

```bash
grep -rn "_append_batch" --include="*.py" --include="*.md" --include="*.yml" \
    scripts/ src/ tests/ docs/ .github/ 2>&1 | \
    grep -v __pycache__ | grep -v "scripts/_append_batch"
```

Expected: NO output (or only file headers).

- [ ] **Step 2: Delete the 8 files**

```bash
rm scripts/_append_batch14.py scripts/_append_batch15.py scripts/_append_batch16.py \
   scripts/_append_batch17.py scripts/_append_batch18.py scripts/_append_batch19.py \
   scripts/_append_batch20.py scripts/_append_batch21.py
```

- [ ] **Step 3: Verify test suite still collects cleanly**

```bash
.venv/Scripts/python.exe -m pytest --collect-only -q 2>&1 | tail -3
```

Expected: tests collected count is unchanged (these scripts aren't imported by tests).

- [ ] **Step 4: Commit**

```bash
git add -u scripts/
git commit -m "chore(scripts): remove throwaway _append_batchN.py scripts"
```

### Task B2: Audit + relocate other underscore-prefix scripts

**Files:**
- Audit: `scripts/_coverage_audit.py`, `scripts/_fix_duplicate_classes.py`
- Possibly delete or move to `scripts/dev/`

- [ ] **Step 1: Check if either is referenced**

```bash
grep -rn "_coverage_audit\|_fix_duplicate_classes" --include="*.py" --include="*.md" \
    --include="*.yml" --include="*.bat" --include="*.ps1" 2>&1 | grep -v __pycache__
```

- [ ] **Step 2: Read the file headers to understand intent**

```bash
head -20 scripts/_coverage_audit.py scripts/_fix_duplicate_classes.py
```

- [ ] **Step 3: Decision matrix**

- If header says "one-off audit" + no references → DELETE
- If header says "ongoing utility" + no references → move to `scripts/dev/`
- If references found → leave in place, file an issue to clean up later

- [ ] **Step 4: Execute the decision per file**

Either `git rm scripts/_coverage_audit.py` or `git mv scripts/_coverage_audit.py scripts/dev/coverage_audit.py` (drop underscore prefix).

- [ ] **Step 5: Commit**

Suggested message: `chore(scripts): clean up underscore-prefix utility scripts`

### Task B3: Consolidate `config/` into `configs/`

**Senior-Review correction (F-senior-9):** Verified-correct caller is at `src/assembled_core/config/env_settings.py:139` (module-level constant `_ENV_DIR = _REPO_ROOT / "config" / "env"`). The relevant test file is `tests/test_env_settings.py` (the earlier draft's `test_settings.py` / `test_config_loader.py` are fictional). Docstring references to `"config/env"` also exist at env_settings.py:15 and env_settings.py:151. Prose references exist in `.env.example`, `docs/SENIOR_REVIEW_AUDIT_2026-05-15.md`, and `docs/POINT_IN_TIME_AND_LATENCY.md`.

**Files:**
- Move: `config/env/` → `configs/env/`
- Delete: empty `config/` directory after move
- Modify: `src/assembled_core/config/env_settings.py` (constant + 2 docstring lines)
- Modify: `.env.example`, `docs/SENIOR_REVIEW_AUDIT_2026-05-15.md`, `docs/POINT_IN_TIME_AND_LATENCY.md` (prose refs only)

- [ ] **Step 1: Inventory `config/` contents and verify caller surface**

```bash
find config/ -type f 2>&1
grep -rn "config/env\|_ENV_DIR\|\"config\"" --include="*.py" --include="*.yml" --include="*.yaml" \
    --include="*.md" --include="*.toml" --include="*.env*" 2>&1 \
    | grep -v __pycache__ | grep -v "configs/" | grep -v ".claude/worktrees/" | head -40
```

- [ ] **Step 2: Move `config/env/` into `configs/env/`**

```bash
mkdir -p configs
git mv config/env configs/env
```

If `git mv` complains because `configs/env` already exists, abort and inspect — do not destructively overwrite.

- [ ] **Step 3: Update `src/assembled_core/config/env_settings.py`**

Replace `_REPO_ROOT / "config" / "env"` with `_REPO_ROOT / "configs" / "env"` at line 139. Update the two docstring strings `"config/env/..."` at lines 15 and 151 to `"configs/env/..."` for consistency.

- [ ] **Step 4: Update prose references**

Edit `.env.example` (if it mentions `config/env/...`), `docs/SENIOR_REVIEW_AUDIT_2026-05-15.md`, `docs/POINT_IN_TIME_AND_LATENCY.md`. Plain string replacement `config/env/` → `configs/env/`. Skip `archive/`, `.claude/worktrees/`, `.git/`.

- [ ] **Step 5: Remove empty `config/` directory**

```bash
rmdir config 2>&1 || ls config/
```

If `rmdir` fails because the directory is not empty, investigate before forcing.

- [ ] **Step 6: Smoke test**

```bash
.venv/Scripts/python.exe -m pytest tests/test_env_settings.py -v
```

Expected: PASS. (Use the verified test name — not the fictional `test_settings.py`.)

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "chore(config): consolidate config/ into configs/ (single source of truth)"
```

### Task B4: Document the `notebooks/` vs `research/` split

**Files:**
- Modify: `notebooks/README.md` (CREATE if missing) or `notebooks/operator_overview_example.py` header
- Modify: `research/README.md` (verify exists, update if needed)

**Rationale:** Both directories exist with different purposes. Make the split explicit so future contributors know where to put new notebooks.

- [ ] **Step 1: Read current `research/README.md`**

```bash
cat research/README.md 2>&1 | head -30
```

- [ ] **Step 2: Create or update `notebooks/README.md`** with this content:

```markdown
# notebooks/

Operator-facing example scripts and dashboards for live/paper usage.

NOT a place for research/experiments — those go in `research/` (one folder per topic).

## Current contents
- `operator_overview_example.py` — minimal operator dashboard example
```

- [ ] **Step 3: Append to `research/README.md`** (or create if missing) a similar one-paragraph header clarifying it is for research-experiment notebooks, not operator-facing.

- [ ] **Step 4: Commit**

```bash
git add notebooks/README.md research/README.md
git commit -m "docs: clarify notebooks/ vs research/ purpose split"
```

### Task B5: Remove stale `.claude/worktrees/` directories — git-safe

**Senior-Review correction (F-senior-5):** Earlier draft used `rm -rf` which would corrupt git's worktree admin state. **Both `.claude/worktrees/agent-*` directories ARE registered active git worktrees** (verified via `git worktree list`); one is `locked`. The correct sequence is: unlock → `git worktree remove --force` → `git worktree prune`.

**Verified state (2026-05-17):**
```
F:/Python_Projekt/Aktiengerüst/.claude/worktrees/agent-a700e54f   44c25ec [worktree-agent-a700e54f]
F:/Python_Projekt/Aktiengerüst/.claude/worktrees/agent-a73bbad9   5dc614d [worktree-agent-a73bbad9] locked
```

Also verified: `.gitignore` **already contains `.claude/worktrees/`** at line 64. No gitignore change needed.

- [ ] **Step 1: Confirm current worktree state**

```bash
git worktree list
```

Expected output: a `main` worktree plus the two `.claude/worktrees/agent-*` entries (one locked). If the agent-* entries are already gone, this task is a no-op — skip to Step 4.

- [ ] **Step 2: Unlock the locked worktree, then remove both**

```bash
git worktree unlock .claude/worktrees/agent-a73bbad9
git worktree remove --force .claude/worktrees/agent-a700e54f
git worktree remove --force .claude/worktrees/agent-a73bbad9
```

The `--force` flag is needed because each worktree has uncommitted changes from past agent runs. We accept loss of those changes (they were experimental and superseded by main).

- [ ] **Step 3: Prune stale admin metadata**

```bash
git worktree prune -v
```

Reports anything cleaned. Should output something like `Removing worktrees/agent-a700e54f: ...`.

- [ ] **Step 4: Verify clean state**

```bash
git worktree list   # should show only main + any active cursor worktrees, no agent-*
ls .claude/worktrees/  # should not exist, or be empty
```

- [ ] **Step 5: Commit only if there is anything to commit**

This task is mostly a git-admin operation — the worktrees were not tracked in main, so there is usually nothing to commit. Run `git status` and only `git commit --allow-empty -m "chore: prune stale agent worktrees"` if you want a marker commit. Otherwise just confirm clean state and move on.

### Task B7: Regenerate the architecture system map after scripts cleanup

**Senior-Review correction (F-senior-8):** Per CLAUDE.md §19.3, after structural changes the interactive system map must be regenerated to avoid stale phantom entries.

**Files (auto-generated, do not hand-edit):**
- Modify: `docs/architecture/system_map/data/system_map.json`
- Modify: `docs/architecture/system_map/data/system_map_data.js`

- [ ] **Step 1: Regenerate**

```bash
.venv/Scripts/python.exe scripts/architecture/generate_system_map.py
.venv/Scripts/python.exe scripts/architecture/validate_system_map.py
```

- [ ] **Step 2: Commit the regenerated artifacts**

```bash
git add docs/architecture/system_map/data/
git commit -m "chore(arch): regenerate system map after scripts + config cleanup"
```

**Run this AFTER Tasks B1, B2, B3** (which remove/move files the system map references).

### Task B6: Document the script-surface in `scripts/SCRIPTS_INDEX.md`

**Files:**
- Create: `scripts/SCRIPTS_INDEX.md` (or `docs/SCRIPTS_INDEX.md` if you prefer docs/)

**Rationale:** 140 top-level scripts make discovery hard. A one-page index categorizing them by purpose (data ingest, backtest, audit, ops, deprecated, etc.) is the cheapest navigational fix.

- [ ] **Step 1: Generate a categorized inventory**

```bash
.venv/Scripts/python.exe -c "
from pathlib import Path
scripts = sorted(Path('scripts').glob('*.py'))
print('Total top-level scripts:', len(scripts))
for s in scripts:
    print(f'  {s.name}')"
```

- [ ] **Step 2: Group by purpose**

Categories (suggested):
- **Entry-Point CLIs** (`cli.py`, `daily_pilot_review.py`, etc.) — installed via `pyproject.toml [project.scripts]`
- **Backtest runners** (`run_backtest_strategy.py`, `batch_backtest.py`, `batch_runner.py`, `benchmark_backtest*.py`)
- **Paper-trading** (`run_paper_pilot.py`, `daily_paper_trading.bat`, `run_live_paper.py`)
- **Data ingest / refresh** (`download_*.py`, `prewarm_price_cache.py`, `build_pre2020_panel.py`, `check_data_*.py`)
- **Audits & checks** (`audit_*.py`, `check_*.py`, `compare_*.py`, `debug_*.py`)
- **A/B / comparison** (`ab_compare_strategies.py`, `compare_strategies_trend_vs_event.py`)
- **Architecture tools** (`scripts/architecture/`)
- **Calibration** (`scripts/calibration/`)
- **Ops** (`scripts/ops/`)
- **Scheduled / batch utility** (`scripts/ci/`, `scripts/data/`)
- **Demo / seed** (`00_seed_demo_data.py`)
- **Dev / one-off** (anything left)

Write the index file with these sections. Each entry: `- \`name.py\` — one-line purpose.`

- [ ] **Step 3: Commit**

```bash
git add scripts/SCRIPTS_INDEX.md
git commit -m "docs(scripts): add SCRIPTS_INDEX.md categorising top-level scripts"
```

---

## Final Validation (after all tasks)

- [ ] **Step F1: Run hook test suite** — `.venv/Scripts/python.exe -m pytest tests/hooks/ -v` — expected 57/57 (no regression in the review-chain layer).
- [ ] **Step F2: Run new dummy-data + pipeline tests** — `.venv/Scripts/python.exe -m pytest tests/test_insider_ingest_failloud.py tests/test_shipping_ingest_failloud.py tests/test_pipeline_congress_failloud.py tests/test_api_monitoring.py -v` — expected all PASS.
- [ ] **Step F3: Run pytest --collect-only** — expected 7090+ tests collected (originally 7084 + new tests), no errors.
- [ ] **Step F4: Verify `config/` is gone and `configs/env/` exists with the same content as the old `config/env/`.**
- [ ] **Step F5: Verify `git worktree list` shows no `.claude/worktrees/agent-*` entries.**
- [ ] **Step F6: Verify `research/dead_ends/` contains the 3 moved skeleton notebooks with the renamed convention `<originalsubdir>-<originalname>.ipynb`.**
- [ ] **Step F7: Verify the system map artifacts were regenerated** — `docs/architecture/system_map/data/system_map.json` modification time matches B7's commit timestamp.

---

## Out-of-Scope (catalogued in `KNOWN_ISSUES.md` §6 — DO NOT execute in this plan)

The compass artifact identified additional improvements that the user explicitly told us NOT to execute now, only to track. They are appended to `KNOWN_ISSUES.md` Section 6 (Strategic Quant Gaps & Roadmap) as part of this plan's preparation but require their own future plans:

- **Markowitz / Risk-Parity / Kelly portfolio optimizers** (`src/assembled_core/portfolio/optimizers/`) — strategically critical but a multi-week project.
- **GARCH volatility module** (`src/assembled_core/risk/volatility/garch.py`) — `arch==8.0.0` already pinned in `requirements.txt`, never wired.
- **Monte-Carlo path simulation & trade shuffling** (`src/assembled_core/risk/monte_carlo/`) — `scenario_engine` exists but not dedicated MC.
- **HMM-based regime detection** (`src/assembled_core/risk/regime_hmm.py` is in source but Grid-Search found no edge → DISABLED, see `KNOWN_ISSUES.md` §4.3).
- **FinBERT / News-Sentiment ML** (`src/assembled_core/ml/nlp/`) — `transformers` is an Optional-Extra, not implemented.
- **Real Insider / Congress / Shipping data feeds** — the dummy generators in Task A1/A2 will be eliminated entirely once a real feed is wired. Track in §6.
- **Live broker routes** (the `oms.py:176` placeholder comment) — needs full broker integration story before code work.

---

## Execution Notes

- **Sub-projects A and B are independent.** A subagent can pick either order or run them in parallel branches.
- **Within Sub-Project A, tasks must run sequentially in order: A1 → A1b → A2 → A3 → A4** (per Senior-Review F-senior-12). Reason: A1 and A2 both touch the same notebook (`research/altdata/insider_congress_shipping_exploration.ipynb`); A4 then moves that notebook to `dead_ends/`. Within Sub-Project B, B1 → B2 → B3 → B4 → B5 → B7 → B6 (B7 must follow B1/B2/B3 because it regenerates the system map after structural changes).
- **Each task ends with a single commit.** Multi-file commits are OK *within* a task as long as the change is one logical thing. Run `git status` before starting a new task — working tree must be clean.
- **The Stop-hook review chain will fire after each commit** in protected paths (`src/`, `scripts/`). Allow it to run. Findings from Stage 1/2/3 must be addressed before declaring a task complete (per CLAUDE.md §20.3).
- **Use Sonnet for execution.** This plan is sized for Sonnet (mechanical edits, clear file paths, complete code blocks). Opus is overkill here.

---

## Plan Self-Review (writing-plans skill §"Self-Review")

**Spec coverage:** Each verified problem maps to a task. Compass-claimed problems already done (drift persistence, pre-trade checks, etc.) are explicitly excluded with evidence. ✓

**Placeholder scan:** No "TBD", "TODO", "implement later", or "similar to TaskN" without inline code. Each step has either a code block or an exact shell command. ✓ (Note: Task A2 / Task A3 / Task B5 were rewritten in the 2026-05-17 Senior-Review-Pass to replace earlier ellipsis placeholders with explicit instructions.)

**Type consistency:** Function signatures (`load_insider_sample`, `load_shipping_sample`, drift API) used consistently across step descriptions and test code. Schemas (`timestamp, symbol, trades_count, net_shares, role` for insider; `timestamp, route_id, port_from, port_to, symbol, ships, congestion_score` for shipping) match verified source code. ✓

**Scope check:** Two sub-projects, both narrowly defined, both achievable in one focused execution session. Out-of-scope items explicitly catalogued, not smuggled in. ✓

## Senior-Review-Pass (2026-05-17)

This plan was reviewed by `senior-code-reviewer` (Stage 2 of CLAUDE.md §20 review chain) **before** Sonnet execution. Verdict: **CONDITIONAL** with 6 BLOCKER + 4 MAJOR findings against the initial draft. All have been addressed in the now-current version:

| Finding | Severity | What was wrong | Fix |
|---|---|---|---|
| F-senior-1 | BLOCKER | Test asserted `ticker`/`date` columns; actual dummy schema is `symbol`/`timestamp`/`trades_count`/`net_shares`/`role` | Task A1 Step 1 + A2 Step 1 now use verified schemas |
| F-senior-2 | BLOCKER | Plan referenced fictional `tests/test_features_events_phase6.py` | Removed; actual file is `test_signals_event_phase6.py` (with no parameterless callers) |
| F-senior-3 | BLOCKER | `scripts/generate_sample_event_data.py` was wrongly listed as a `load_*_sample` caller | Removed; verified caller set is 4 prod files + 1 notebook |
| F-senior-4 | BLOCKER | Task A3 assumed dummy fallback; actual code already raises 503 at `monitoring.py:369` | Task A3 rewritten to update stale docstring only; preserve 503 behavior |
| F-senior-5 | BLOCKER | `rm -rf .claude/worktrees/agent-*` would corrupt git state (worktrees are registered, one locked) | Task B5 rewritten with `git worktree unlock` + `git worktree remove --force` + `git worktree prune` |
| F-senior-6 | BLOCKER | Missed `congress_trades_ingest` ghost-module silent-except in `trading_cycle_shared.py:625-647` | New Task A1b added; KNOWN_ISSUES §6.5.5 expanded with concrete status |
| F-senior-7 | MAJOR | Task A2 was "mirror A1" placeholder | Task A2 now has full inline shipping schema + tests |
| F-senior-8 | MAJOR | System map regeneration missing after structural changes | New Task B7 added with explicit step |
| F-senior-9 | MAJOR | Task B3 used fictional test names + missed `env_settings.py:139` constant | Task B3 rewritten with verified file/line references |
| F-senior-10 | MAJOR | Task A4 was tag-only half-measure with potential `ensure_ascii` emoji bug | Task A4 rewritten to MOVE skeletons to `research/dead_ends/` |
| F-senior-11 | MINOR | Script count "183" was wrong | Corrected to "232 total / 140 top-level" |
| F-senior-12 | MINOR | Task ordering unstated | Execution Notes now specify A1→A1b→A2→A3→A4 and B1→B2→B3→B4→B5→B7→B6 |
| F-senior-13 | MINOR | Ellipsis placeholders Sonnet might copy verbatim | Task A1 Step 3 + A2 Step 3 use explicit "INSERT BEFORE" instructions |
| F-senior-14 | MINOR | KNOWN_ISSUES §6.5.5 missed concrete congress status | Appended in this pass |
| F-senior-15 | MAJOR (repo bug) | `trading_cycle_shared.py:645` silent-except | Addressed by new Task A1b |
| F-senior-16 | INFO | KNOWN_ISSUES §6.7 phrased as if A4 already done | Rephrased to "vor Plan-Ausführung" + "Nach Plan-Ausführung (noch ausstehend)" |

All BLOCKERs and MAJORs closed before Stage 3 (task-completion-auditor) review. The plan is now ready for Sonnet execution.
