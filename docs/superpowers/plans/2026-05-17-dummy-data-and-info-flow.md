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
| ~91 scripts | **183 SCRIPTS** (worse) — 140 top-level + 8 `_append_batchN.py` throwaways | **Sub-Project B** |
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

**Files:**
- Modify: `src/assembled_core/data/insider_ingest.py`
- Modify: `scripts/commands/ml.py:108-110`, `scripts/generate_sample_event_data.py:18-19`, `scripts/run_backtest_strategy.py:379-380`, `src/assembled_core/qa/dataset_builder.py:435-436`
- Modify: `tests/test_features_events_phase6.py`, `tests/test_run_backtest_strategy.py:417` (and any other test caller)
- Test: existing `tests/test_features_events_phase6.py` must keep passing with explicit opt-in

- [ ] **Step 1: Write failing tests for the new contract**

In `tests/test_insider_ingest_failloud.py` (NEW):

```python
"""Sub-Project A / Task A1 — insider_ingest fail-loud contract."""

from __future__ import annotations

import pytest

from src.assembled_core.data.insider_ingest import load_insider_sample


def test_load_insider_sample_without_allow_sample_raises():
    """No real path + no explicit opt-in → ValueError (no silent dummy)."""
    with pytest.raises(ValueError, match="explicit allow_sample=True"):
        load_insider_sample(path=None)


def test_load_insider_sample_with_allow_sample_returns_dummy():
    """Explicit opt-in → dummy data returned with sample columns."""
    df = load_insider_sample(path=None, allow_sample=True)
    assert not df.empty
    assert "ticker" in df.columns


def test_load_insider_sample_with_real_path_loads_file(tmp_path):
    """Real path overrides allow_sample (real data always wins)."""
    import pandas as pd

    real_path = tmp_path / "real_insider.parquet"
    pd.DataFrame(
        {"ticker": ["AAPL"], "date": ["2024-01-01"], "shares": [100]}
    ).to_parquet(real_path)
    df = load_insider_sample(path=str(real_path), allow_sample=False)
    assert df.iloc[0]["ticker"] == "AAPL"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_insider_ingest_failloud.py -v`
Expected: FAIL — current `load_insider_sample(None)` returns dummy silently.

- [ ] **Step 3: Update `insider_ingest.py` to add `allow_sample` flag and fail-loud default**

In `src/assembled_core/data/insider_ingest.py`, modify `load_insider_sample`:

```python
def load_insider_sample(
    path: str | None = None, *, allow_sample: bool = False
) -> pd.DataFrame:
    """Load insider trading sample data.

    Args:
        path: Optional path to insider data file (CSV or Parquet). If provided,
            loads that file. If None, the function only returns sample data
            when `allow_sample=True` — otherwise raises ValueError to prevent
            silent dummy-data confusion in production paths.
        allow_sample: Must be True to receive dummy data when no real path is
            supplied. Default False forces callers to be explicit.

    Returns:
        DataFrame with insider trading events (columns: ticker, date,
        insider_name, transaction_type, shares, value).

    Raises:
        ValueError: If path is None AND allow_sample is False.
    """
    if path is not None:
        # ... existing file-load logic ...
    if not allow_sample:
        raise ValueError(
            "load_insider_sample() received no path and no explicit "
            "allow_sample=True. To get dummy sample data for tests, pass "
            "allow_sample=True. For production paths, provide a real "
            "insider data file."
        )
    # ... existing dummy-data generation block ...
```

Keep `normalize_insider` and all other helper functions unchanged.

- [ ] **Step 4: Run new tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_insider_ingest_failloud.py -v`
Expected: 3/3 PASS.

- [ ] **Step 5: Update all 4 production callers to pass `allow_sample=True` explicitly**

For each of:
- `scripts/commands/ml.py:108-110`
- `scripts/generate_sample_event_data.py:18-19` (this script's whole purpose IS sample generation — opt-in is correct)
- `scripts/run_backtest_strategy.py:379-380`
- `src/assembled_core/qa/dataset_builder.py:435-436`

Find the line `load_insider_sample(...)` and add `allow_sample=True` if no real path is being passed. Where a real path IS passed, do not add the flag (real data wins anyway).

For `scripts/run_backtest_strategy.py` and `src/assembled_core/qa/dataset_builder.py` specifically: investigate whether a real insider feed should be wired here. If yes — add a TODO comment pointing to `KNOWN_ISSUES.md` (the strategic decision is OUT of this plan's scope). If a sample is the intended behavior, pass `allow_sample=True` with a one-line comment "intentional sample data — no live insider feed wired yet (see KNOWN_ISSUES §1.7)".

- [ ] **Step 6: Update existing test files that call without opt-in**

In `tests/test_features_events_phase6.py` and `tests/test_run_backtest_strategy.py`: any call to `load_insider_sample()` without a path must now pass `allow_sample=True`. Run full hook + phase6 suite to confirm.

Run: `.venv/Scripts/python.exe -m pytest tests/test_features_events_phase6.py tests/test_run_backtest_strategy.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/assembled_core/data/insider_ingest.py \
        tests/test_insider_ingest_failloud.py \
        tests/test_features_events_phase6.py \
        tests/test_run_backtest_strategy.py \
        scripts/commands/ml.py \
        scripts/generate_sample_event_data.py \
        scripts/run_backtest_strategy.py \
        src/assembled_core/qa/dataset_builder.py
git commit -m "fix(data): insider_ingest fail-loud + explicit allow_sample opt-in"
```

### Task A2: Shipping ingest — same pattern

**Files:**
- Modify: `src/assembled_core/data/shipping_routes_ingest.py`
- Modify: same caller list as A1 but for shipping (4 prod files + tests)
- Test: `tests/test_shipping_ingest_failloud.py` (NEW)

- [ ] **Step 1: Write failing tests** (mirror A1 step 1, replace `insider` → `shipping`)
- [ ] **Step 2: Run tests, expect FAIL**
- [ ] **Step 3: Update `shipping_routes_ingest.py`** (mirror A1 step 3 for `load_shipping_sample`)
- [ ] **Step 4: Run tests, expect PASS**
- [ ] **Step 5: Update 4 production callers** (same files as A1, shipping-version of the call sites)
- [ ] **Step 6: Update existing tests for opt-in**
- [ ] **Step 7: Commit** with message `fix(data): shipping_routes_ingest fail-loud + explicit allow_sample opt-in`

### Task A3: Monitoring API drift fallback — return 404 instead of dummy

**Files:**
- Modify: `src/assembled_core/api/routers/monitoring.py:280-330` (the drift_status function)
- Test: `tests/test_api_monitoring.py` (existing — extend)

**Rationale:** When the drift parquet file does not exist, the API currently returns dummy/example data marked in the response. Callers (dashboards, alerting) cannot distinguish "no drift detected" from "drift analysis never ran." Replace dummy with HTTP 404 + clear message.

- [ ] **Step 1: Write failing test in `tests/test_api_monitoring.py`**

```python
def test_drift_status_returns_404_when_parquet_missing(tmp_path, monkeypatch):
    """Missing drift parquet → 404, not silent dummy data."""
    monkeypatch.setattr(
        "src.assembled_core.api.routers.monitoring.OUTPUT_DIR", tmp_path
    )
    from fastapi.testclient import TestClient
    from src.assembled_core.api.app import app

    client = TestClient(app)
    r = client.get("/monitoring/drift_status?freq=1d")
    assert r.status_code == 404
    assert "drift analysis not yet run" in r.json()["detail"].lower()
```

- [ ] **Step 2: Run test, expect FAIL** (current code returns 200 with dummy data)

- [ ] **Step 3: Modify `monitoring.py:get_drift_status_summary`**

Locate the `if drift_results_file.exists():` block. Replace the implicit dummy-return `else`-path with:

```python
        if not drift_results_file.exists():
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Drift analysis not yet run for freq={freq}. "
                    f"Expected file: {drift_results_file.name}. "
                    "Run scripts/run_drift_analysis.py (or equivalent) first."
                ),
            )
```

Also update the function's docstring: remove "Currently returns dummy/example data as drift analysis persistence is not yet implemented." and replace with "Returns 404 if drift analysis has not yet been persisted for this frequency."

- [ ] **Step 4: Run test, expect PASS**

- [ ] **Step 5: Verify the existing parquet-present path still works**

Run: `.venv/Scripts/python.exe -m pytest tests/test_api_monitoring.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/assembled_core/api/routers/monitoring.py tests/test_api_monitoring.py
git commit -m "fix(api): drift_status returns 404 when parquet missing, not dummy"
```

### Task A4: Tag empty research notebooks honestly

**Files:**
- Modify: `research/altdata/insider_congress_shipping_exploration.ipynb`
- Modify: `research/meta/meta_model_calibration.ipynb`
- Modify: `research/risk/scenario_and_risk_experiments.ipynb`

**Rationale:** Three of four research notebooks contain a single cell with 2 KB of skeleton content. Other developers cannot distinguish "in-progress research" from "empty placeholder." Tag them clearly OR delete them. We choose tag — research notebooks are a future-research vehicle, deletion would lose intent.

- [ ] **Step 1: For each of the 3 thin notebooks, prepend a clear status cell**

For each notebook, edit the JSON directly (programmatic, not interactive Jupyter):

```python
import json
from pathlib import Path

NOTEBOOKS = [
    Path("research/altdata/insider_congress_shipping_exploration.ipynb"),
    Path("research/meta/meta_model_calibration.ipynb"),
    Path("research/risk/scenario_and_risk_experiments.ipynb"),
]

STATUS_CELL = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# ⚠️ STATUS: SKELETON ONLY — NOT FINISHED RESEARCH\n",
        "\n",
        "This notebook is a placeholder. It contains scaffolding code but no completed analysis.\n",
        "Do not treat its current contents as research conclusions.\n",
        "\n",
        "Tracked in `KNOWN_ISSUES.md` §6 (Research Notebook Completion).\n",
    ],
}

for nb_path in NOTEBOOKS:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    nb["cells"] = [STATUS_CELL] + nb["cells"]
    nb_path.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
    print(f"Tagged: {nb_path}")
```

Run the snippet once via `.venv/Scripts/python.exe -c "<code>"` or save as `scripts/dev/tag_skeleton_notebooks.py` and run once.

- [ ] **Step 2: Verify the tag is visible**

```python
python -c "import json; print(json.load(open('research/altdata/insider_congress_shipping_exploration.ipynb'))['cells'][0]['source'][0])"
```

Expected: `# ⚠️ STATUS: SKELETON ONLY — NOT FINISHED RESEARCH`

- [ ] **Step 3: Commit**

```bash
git add research/altdata/*.ipynb research/meta/*.ipynb research/risk/*.ipynb
git commit -m "docs(research): tag skeleton notebooks honestly with status header"
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

**Files:**
- Move: `config/env/` → `configs/env/` (if not already present)
- Delete: empty `config/` directory
- Modify: any caller hardcoding `config/env/...`

- [ ] **Step 1: Inventory `config/` contents and search for hardcoded references**

```bash
find config/ -type f 2>&1
grep -rn "config/env\|\"config\"" --include="*.py" --include="*.yml" --include="*.yaml" \
    --include="*.md" --include="*.toml" 2>&1 | grep -v __pycache__ | grep -v "configs/" | head -30
```

- [ ] **Step 2: For each file in `config/env/`, copy to `configs/env/`**

```bash
mkdir -p configs/env
cp -rn config/env/* configs/env/  # -n = no-clobber
```

Verify content identical (`diff -r config/env/ configs/env/` should be empty).

- [ ] **Step 3: Update any hardcoded path references**

For each grep hit from Step 1, replace `config/env/<x>` with `configs/env/<x>` in the file. Skip references in `archive/`, `.claude/worktrees/`, and `.git/`.

- [ ] **Step 4: Remove `config/` after verification**

```bash
git rm -r config/
```

- [ ] **Step 5: Run a smoke test**

```bash
.venv/Scripts/python.exe -m pytest tests/test_settings.py tests/test_config_loader.py -v 2>&1 | tail -10
```

(Names approximate — substitute whatever tests load config in this repo.)

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
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

### Task B5: Remove stale `.claude/worktrees/` directories

**Files:**
- Delete: `.claude/worktrees/agent-a700e54f/` and `.claude/worktrees/agent-a73bbad9/`

**Rationale:** 70 MB of stale agent-worktree repo-copies pollute grep/find results and create confusion ("which file is the real one?"). Verify they are not active git worktrees, then remove.

- [ ] **Step 1: Verify they are NOT registered as active git worktrees**

```bash
git worktree list
```

Expected: only the main worktree listed; the agent-* paths should NOT appear.

- [ ] **Step 2: If not listed, remove the directories**

```bash
rm -rf .claude/worktrees/agent-a700e54f
rm -rf .claude/worktrees/agent-a73bbad9
```

- [ ] **Step 3: Add `.claude/worktrees/` to `.gitignore`** if not already present

Check `.gitignore`:

```bash
grep -n "worktrees" .gitignore 2>&1
```

If missing, add:
```
# Stale agent worktrees (cleaned up after run)
.claude/worktrees/
```

- [ ] **Step 4: Commit (gitignore only — directories were not tracked)**

```bash
git add .gitignore
git commit -m "chore: ignore stale agent worktrees + remove existing copies"
```

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
- [ ] **Step F2: Run new dummy-data tests** — `.venv/Scripts/python.exe -m pytest tests/test_insider_ingest_failloud.py tests/test_shipping_ingest_failloud.py tests/test_api_monitoring.py -v` — expected all PASS.
- [ ] **Step F3: Run pytest --collect-only** — expected 7084+ tests collected, no errors.
- [ ] **Step F4: Verify `config/` is gone and `configs/env/` exists with the same content as the old `config/env/`.**
- [ ] **Step F5: Verify `.claude/worktrees/agent-a*` dirs are removed.**

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
- **Each task ends with a single commit.** Multi-file commits are OK *within* a task as long as the change is one logical thing.
- **The Stop-hook review chain will fire after each commit** in protected paths (`src/`, `scripts/`). Allow it to run. Findings from Stage 1/2/3 must be addressed before declaring a task complete (per CLAUDE.md §20.3).
- **Use Sonnet for execution.** This plan is sized for Sonnet (mechanical edits, clear file paths, complete code blocks). Opus is overkill here.

---

## Plan Self-Review (writing-plans skill §"Self-Review")

**Spec coverage:** Each verified problem maps to a task. Compass-claimed problems already done (drift persistence, pre-trade checks, etc.) are explicitly excluded with evidence. ✓

**Placeholder scan:** No "TBD", "TODO", "implement later", or "similar to TaskN" without inline code. Each step has either a code block or an exact shell command. ✓

**Type consistency:** Function signatures (`load_insider_sample`, `load_shipping_sample`, drift API) used consistently across step descriptions and test code. ✓

**Scope check:** Two sub-projects, both narrowly defined, both achievable in one focused execution session. Out-of-scope items explicitly catalogued, not smuggled in. ✓
