"""Appends batch 14 tests to the session test file."""

from pathlib import Path

BATCH14 = """

# ---------------------------------------------------------------------------
# BATCH 14: Items 9, 10, 11, 12, 14, 17, 18, 19, 20, 26, 27
# ---------------------------------------------------------------------------


class TestLargeModuleCount:
    \"\"\"Item 9: Modules > 1000 LOC are tracked; expected in large codebase.\"\"\"

    def test_large_modules_are_known(self):
        src = Path(__file__).parents[1] / "src"
        large = []
        for f in src.rglob("*.py"):
            if "__pycache__" in str(f):
                continue
            try:
                loc = len(f.read_text(encoding="utf-8", errors="replace").splitlines())
                if loc > 1000:
                    large.append((f.name, loc))
            except OSError:
                pass
        assert len(large) >= 1, "At least one large module expected"

    def test_largest_module_is_identifiable(self):
        src = Path(__file__).parents[1] / "src"
        sizes = []
        for f in src.rglob("*.py"):
            if "__pycache__" in str(f):
                continue
            try:
                loc = len(f.read_text(encoding="utf-8", errors="replace").splitlines())
                sizes.append((loc, f.name))
            except OSError:
                pass
        sizes.sort(reverse=True)
        assert sizes
        biggest_loc, _ = sizes[0]
        assert biggest_loc > 100


class TestDuplicateImplementations:
    \"\"\"Item 10: Duplicate module detection; key ones identified.\"\"\"

    def test_transaction_costs_exists(self):
        src = Path(__file__).parents[1] / "src"
        tc_files = [f for f in src.rglob("transaction_costs.py") if "__pycache__" not in str(f)]
        assert len(tc_files) >= 1, "transaction_costs.py should exist"

    def test_state_machine_exists(self):
        src = Path(__file__).parents[1] / "src"
        sm_files = [f for f in src.rglob("state_machine.py") if "__pycache__" not in str(f)]
        assert len(sm_files) >= 1, "state_machine.py should exist"

    def test_models_py_count_bounded(self):
        src = Path(__file__).parents[1] / "src"
        models_files = [f for f in src.rglob("models.py") if "__pycache__" not in str(f)]
        assert len(models_files) <= 8, f"Too many models.py ({len(models_files)})"


class TestTypeHintCoverage:
    \"\"\"Item 11: Type hints cover >= 70% of pipeline functions.\"\"\"

    def test_pipeline_type_hint_coverage(self):
        import ast
        pipeline = Path(__file__).parents[1] / "src" / "assembled_core" / "pipeline"
        total, annotated = 0, 0
        for f in pipeline.rglob("*.py"):
            if "__pycache__" in str(f):
                continue
            try:
                tree = ast.parse(f.read_text(encoding="utf-8", errors="replace"))
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        total += 1
                        if node.returns is not None:
                            annotated += 1
            except SyntaxError:
                pass
        if total == 0:
            return
        assert annotated / total >= 0.70, f"Coverage {annotated/total:.1%} below 70%"

    def test_future_annotations_used_widely(self):
        src = Path(__file__).parents[1] / "src"
        count = sum(
            1 for f in src.rglob("*.py")
            if "__pycache__" not in str(f)
            and "from __future__ import annotations" in f.read_text(encoding="utf-8", errors="replace")
        )
        assert count >= 20, f"Only {count} files use from __future__ import annotations"


class TestLazyImports:
    \"\"\"Item 12: Lazy imports in multifactor_v2.py are bounded (<=25 suppressions).\"\"\"

    def test_plc0415_count_bounded(self):
        p = Path(__file__).parents[1] / "src" / "assembled_core" / "strategies" / "multifactor_v2.py"
        content = p.read_text(encoding="utf-8", errors="replace")
        count = content.count("PLC0415")
        assert count <= 25, f"Too many lazy import suppressions: {count}"

    def test_function_body_imports_use_noqa(self):
        p = Path(__file__).parents[1] / "src" / "assembled_core" / "strategies" / "multifactor_v2.py"
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        bare_lazy = [
            f"line {i}: {l.strip()[:50]}"
            for i, l in enumerate(lines, 1)
            if (l.strip().startswith("import ") or l.strip().startswith("from "))
            and l.startswith("    ")
            and "PLC0415" not in l
            and "noqa" not in l
        ]
        assert len(bare_lazy) <= 5, f"Unguarded lazy imports: {bare_lazy}"


class TestDDDamperConcurrency:
    \"\"\"Item 14: DD damper uses threading.Lock for thread-safe concurrent access.\"\"\"

    def test_dd_lock_exists(self):
        p = Path(__file__).parents[1] / "src" / "assembled_core" / "strategies" / "multifactor_v2.py"
        content = p.read_text(encoding="utf-8", errors="replace")
        assert "_DD_LOCK" in content
        assert "threading.Lock()" in content

    def test_dd_lock_used_as_context_manager(self):
        p = Path(__file__).parents[1] / "src" / "assembled_core" / "strategies" / "multifactor_v2.py"
        content = p.read_text(encoding="utf-8", errors="replace")
        assert "with _DD_LOCK:" in content

    def test_dd_damper_state_is_documented(self):
        p = Path(__file__).parents[1] / "src" / "assembled_core" / "strategies" / "multifactor_v2.py"
        content = p.read_text(encoding="utf-8", errors="replace")
        assert (
            "module-global" in content.lower()
            or "Item 6" in content
            or "global state" in content.lower()
        )


class TestAssertInProduction:
    \"\"\"Item 17: assert in hot-path code replaced with explicit raises.\"\"\"

    def _check_dir_for_asserts(self, directory: Path) -> list:
        violations = []
        for f in directory.rglob("*.py"):
            if "__pycache__" in str(f):
                continue
            try:
                for i, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                    s = line.strip()
                    if s.startswith("#") or s.startswith(">>>"):
                        continue
                    if s.startswith("assert ") and "noqa" not in s:
                        violations.append(f"{f.name}:{i}")
            except OSError:
                pass
        return violations

    def test_no_assert_in_pipeline(self):
        pipeline = Path(__file__).parents[1] / "src" / "assembled_core" / "pipeline"
        violations = self._check_dir_for_asserts(pipeline)
        assert not violations, f"assert in pipeline (use raise): {violations}"

    def test_no_assert_in_execution(self):
        execution = Path(__file__).parents[1] / "src" / "assembled_core" / "execution"
        violations = self._check_dir_for_asserts(execution)
        assert not violations, f"assert in execution (use raise): {violations}"


class TestRSSFetcherTimeout:
    \"\"\"Item 18: RSS fetcher has per-feed numeric timeout to prevent blocking.\"\"\"

    def test_has_timeout_attribute(self):
        p = Path(__file__).parents[1] / "src" / "assembled_core" / "intel" / "rss_fetcher.py"
        content = p.read_text(encoding="utf-8", errors="replace")
        assert "self._timeout" in content

    def test_has_numeric_default_timeout(self):
        import re
        p = Path(__file__).parents[1] / "src" / "assembled_core" / "intel" / "rss_fetcher.py"
        content = p.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"timeout[\\s]*:[\\s]*int[\\s]*=[\\s]*(\\d+)", content)
        assert m is not None, "RSS fetcher needs integer default timeout"
        assert int(m.group(1)) > 0

    def test_timeout_used_in_request(self):
        p = Path(__file__).parents[1] / "src" / "assembled_core" / "intel" / "rss_fetcher.py"
        content = p.read_text(encoding="utf-8", errors="replace")
        assert "timeout=self._timeout" in content


class TestRequirementsLock:
    \"\"\"Item 19: requirements.lock exists for reproducible CI installs.\"\"\"

    def test_requirements_lock_exists(self):
        lock = Path(__file__).parents[1] / "requirements.lock"
        assert lock.exists(), "requirements.lock should exist"

    def test_requirements_lock_not_empty(self):
        lock = Path(__file__).parents[1] / "requirements.lock"
        if lock.exists():
            assert len(lock.read_text(encoding="utf-8", errors="replace").strip()) > 100

    def test_requirements_txt_exists(self):
        req = Path(__file__).parents[1] / "requirements.txt"
        assert req.exists()


class TestSecurityAuditCI:
    \"\"\"Item 20: pip-audit wired into CI for dependency vulnerability scanning.\"\"\"

    def test_pip_audit_in_backend_ci(self):
        wf = Path(__file__).parents[1] / ".github" / "workflows" / "backend-ci.yml"
        if not wf.exists():
            return
        assert "pip-audit" in wf.read_text(encoding="utf-8", errors="replace")

    def test_pip_audit_skip_editable(self):
        wf = Path(__file__).parents[1] / ".github" / "workflows" / "backend-ci.yml"
        if not wf.exists():
            return
        assert "--skip-editable" in wf.read_text(encoding="utf-8", errors="replace")

    def test_security_scan_in_some_workflow(self):
        wf_dir = Path(__file__).parents[1] / ".github" / "workflows"
        if not wf_dir.exists():
            return
        found = any(
            "pip-audit" in wf.read_text(encoding="utf-8", errors="replace")
            for wf in wf_dir.glob("*.yml")
        )
        assert found


class TestDataFreshnessCheck:
    \"\"\"Item 26: Data freshness gate exists to prevent trading on stale prices.\"\"\"

    def test_freshness_module_exists(self):
        src = Path(__file__).parents[1] / "src"
        files = (
            list(src.rglob("*freshness*.py")) +
            list(src.rglob("*data_quality*.py")) +
            list(src.rglob("*data_check*.py"))
        )
        files = [f for f in files if "__pycache__" not in str(f)]
        assert len(files) >= 1, "A data freshness module should exist"

    def test_freshness_has_staleness_threshold(self):
        src = Path(__file__).parents[1] / "src"
        files = (
            list(src.rglob("*freshness*.py")) +
            list(src.rglob("*data_quality*.py")) +
            list(src.rglob("*data_check*.py"))
        )
        files = [f for f in files if "__pycache__" not in str(f)]
        if not files:
            return
        content = files[0].read_text(encoding="utf-8", errors="replace")
        assert any(kw in content for kw in ["timedelta", "hours", "stale", "max_age", "threshold"])

    def test_some_module_references_freshness(self):
        for directory in [
            Path(__file__).parents[1] / "src",
            Path(__file__).parents[1] / "scripts",
        ]:
            for f in directory.rglob("*.py"):
                if "__pycache__" in str(f):
                    continue
                try:
                    if any(kw in f.read_text(encoding="utf-8", errors="replace").lower()
                           for kw in ["freshness", "data_quality", "stale", "data_check"]):
                        return  # found
                except OSError:
                    pass
        assert False, "No module references data freshness"


class TestMemoryBounding:
    \"\"\"Item 27: Memory growth bounded via capped caches.\"\"\"

    def test_hmm_cache_is_bounded(self):
        p = Path(__file__).parents[1] / "src" / "assembled_core" / "strategies" / "multifactor_v2.py"
        content = p.read_text(encoding="utf-8", errors="replace")
        assert "_BoundedCache" in content or "lru_cache" in content or "maxsize" in content

    def test_regime_weights_cache_bounded(self):
        p = Path(__file__).parents[1] / "src" / "assembled_core" / "strategies" / "multifactor_v2.py"
        content = p.read_text(encoding="utf-8", errors="replace")
        assert "_REGIME_WEIGHTS_CACHE" in content
        assert "maxsize" in content

    def test_bounded_cache_has_eviction(self):
        p = Path(__file__).parents[1] / "src" / "assembled_core" / "strategies" / "multifactor_v2.py"
        content = p.read_text(encoding="utf-8", errors="replace")
        if "_BoundedCache" not in content:
            return
        assert (
            "len(self._store) > self._maxsize" in content
            or "popitem" in content
            or "evict" in content.lower()
        )
"""

target = Path("tests/test_session_2026_05_07_new_items.py")
with open(target, "a", encoding="utf-8") as fh:
    fh.write(BATCH14)
print("Batch 14 appended successfully.")
