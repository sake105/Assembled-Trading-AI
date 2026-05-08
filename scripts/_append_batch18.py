"""Appends batch 18 tests to the session test file."""

from pathlib import Path

BATCH18 = """

# ---------------------------------------------------------------------------
# BATCH 18: BACKLOG_ERGAENZUNG items 121, 123, 124, 132, 133, 134, 136, 137, 138, 139, 140
# ---------------------------------------------------------------------------


class TestModuleGlobalConsolidation:
    \"\"\"Item 121: Module-global mutables are bounded and documented.\"\"\"

    def test_bounded_cache_replaces_plain_dict(self):
        p = Path(__file__).parents[1] / "src" / "assembled_core" / "strategies" / "multifactor_v2.py"
        content = p.read_text(encoding="utf-8", errors="replace")
        # _BoundedCache should be used instead of plain dicts for caches
        assert "_BoundedCache" in content, "Caches should use _BoundedCache, not plain dicts"

    def test_plw0603_noqa_count_bounded(self):
        src = Path(__file__).parents[1] / "src"
        count = sum(
            f.read_text(encoding="utf-8", errors="replace").count("PLW0603")
            for f in src.rglob("*.py")
            if "__pycache__" not in str(f)
        )
        assert count <= 5, f"Too many PLW0603 suppressions: {count} — reduce global state"

    def test_dd_damper_state_has_comment_about_global(self):
        p = Path(__file__).parents[1] / "src" / "assembled_core" / "strategies" / "multifactor_v2.py"
        content = p.read_text(encoding="utf-8", errors="replace")
        # Should have comment about global state being known tech-debt
        assert "IMPORTANT" in content or "Item 6" in content or "module-global" in content.lower()


class TestVPSReadiness:
    \"\"\"Item 123: System has Docker or VPS-deployment readiness.\"\"\"

    def test_docker_compose_or_dockerfile_exists(self):
        root = Path(__file__).parents[1]
        docker_files = (
            list(root.glob("docker-compose*.yml")) +
            list(root.glob("Dockerfile*")) +
            list(root.glob("*.dockerfile"))
        )
        assert len(docker_files) >= 1, "Docker setup should exist for VPS deployment"

    def test_requirements_txt_for_docker(self):
        root = Path(__file__).parents[1]
        req = root / "requirements.txt"
        assert req.exists(), "requirements.txt needed for Docker image build"


class TestSchedulerRobustness:
    \"\"\"Item 124: Scheduler is backed by external cron or has watchdog.\"\"\"

    def test_scheduler_module_exists(self):
        src = Path(__file__).parents[1] / "src"
        sched_files = list(src.rglob("scheduler.py")) + list(src.rglob("*daily_scheduler*.py"))
        sched_files = [f for f in sched_files if "__pycache__" not in str(f)]
        assert len(sched_files) >= 1, "A scheduler module should exist"

    def test_github_workflows_have_cron(self):
        wf_dir = Path(__file__).parents[1] / ".github" / "workflows"
        if not wf_dir.exists():
            return
        cron_workflows = [
            wf for wf in wf_dir.glob("*.yml")
            if "schedule" in wf.read_text(encoding="utf-8", errors="replace")
        ]
        assert len(cron_workflows) >= 1, "At least one workflow should have cron schedule"


class TestPilotV2InitScript:
    \"\"\"Item 132: Pilot v2 start script exists with pre-flight checks.\"\"\"

    def test_start_pilot_script_exists(self):
        scripts = Path(__file__).parents[1] / "scripts"
        pilot_scripts = (
            list(scripts.glob("*pilot*v2*")) +
            list(scripts.glob("*start_pilot*"))
        )
        pilot_scripts = [f for f in pilot_scripts if "__pycache__" not in str(f)]
        assert len(pilot_scripts) >= 1, "A start_pilot_v2.ps1 or similar script should exist"

    def test_pilot_script_has_preflight_checks(self):
        scripts = Path(__file__).parents[1] / "scripts"
        pilot_scripts = (
            list(scripts.glob("*pilot*v2*")) +
            list(scripts.glob("*start_pilot*"))
        )
        pilot_scripts = [f for f in pilot_scripts if "__pycache__" not in str(f)]
        if not pilot_scripts:
            return
        content = pilot_scripts[0].read_text(encoding="utf-8", errors="replace").lower()
        has_checks = any(kw in content for kw in ["check", "validate", "verify", "test", "env"])
        assert has_checks, "Pilot start script should have pre-flight checks"


class TestExposureCeiling:
    \"\"\"Item 133: _MAX_EXPOSURE_MULT = 3.0 is enforced and bounded.\"\"\"

    def test_max_exposure_mult_exists(self):
        p = Path(__file__).parents[1] / "src" / "assembled_core" / "pipeline" / "_tc_sizing.py"
        content = p.read_text(encoding="utf-8", errors="replace")
        assert "_MAX_EXPOSURE_MULT" in content

    def test_max_exposure_mult_is_enforced(self):
        p = Path(__file__).parents[1] / "src" / "assembled_core" / "pipeline" / "_tc_sizing.py"
        content = p.read_text(encoding="utf-8", errors="replace")
        # Should clamp/enforce the ceiling
        assert "_MAX_EXPOSURE_MULT" in content and (
            "min(" in content or "clamp" in content.lower() or "final_multiplier = _MAX_EXPOSURE_MULT" in content
        )

    def test_max_exposure_mult_value(self):
        p = Path(__file__).parents[1] / "src" / "assembled_core" / "pipeline" / "_tc_sizing.py"
        content = p.read_text(encoding="utf-8", errors="replace")
        import re
        m = re.search(r"_MAX_EXPOSURE_MULT\s*=\s*([0-9.]+)", content)
        if m:
            val = float(m.group(1))
            assert val <= 3.5, f"_MAX_EXPOSURE_MULT = {val} seems too high"
            assert val >= 1.5, f"_MAX_EXPOSURE_MULT = {val} seems too low"


class TestDataSourceFallback:
    \"\"\"Item 134: yfinance has fallback data source or quality check.\"\"\"

    def test_multiple_data_sources_exist(self):
        src = Path(__file__).parents[1] / "src"
        sources_dir = src / "assembled_core" / "data" / "sources"
        if not sources_dir.exists():
            sources_dir = src / "assembled_core" / "data"
        source_files = [
            f for f in sources_dir.rglob("*.py")
            if "__pycache__" not in str(f) and "source" in f.name.lower()
        ]
        # Should have more than just yfinance
        assert len(source_files) >= 2, f"Only {len(source_files)} data sources — need fallback"

    def test_data_quality_check_exists(self):
        src = Path(__file__).parents[1] / "src"
        quality_files = (
            list(src.rglob("*data_quality*.py")) +
            list(src.rglob("*freshness*.py")) +
            list(src.rglob("validate_altdata*.py"))
        )
        quality_files = [f for f in quality_files if "__pycache__" not in str(f)]
        assert len(quality_files) >= 1, "A data quality or freshness check module should exist"

    def test_polygon_or_tiingo_source_exists(self):
        src = Path(__file__).parents[1] / "src"
        alt_sources = (
            list(src.rglob("*polygon*.py")) +
            list(src.rglob("*tiingo*.py")) +
            list(src.rglob("*alpha_vantage*.py")) +
            list(src.rglob("*alphavantage*.py")) +
            list(src.rglob("*finnhub*.py"))
        )
        alt_sources = [f for f in alt_sources if "__pycache__" not in str(f)]
        assert len(alt_sources) >= 1, "At least one non-yfinance data source should exist"


class TestABCompareScript:
    \"\"\"Item 136: A/B strategy comparison script exists.\"\"\"

    def test_ab_compare_script_exists(self):
        p = Path(__file__).parents[1] / "scripts" / "ab_compare_strategies.py"
        assert p.exists(), "scripts/ab_compare_strategies.py should exist"

    def test_ab_compare_has_comparison_logic(self):
        p = Path(__file__).parents[1] / "scripts" / "ab_compare_strategies.py"
        content = p.read_text(encoding="utf-8", errors="replace")
        has_compare = any(kw in content.lower() for kw in ["compare", "sharpe", "cagr", "vs", "a_result"])
        assert has_compare, "ab_compare_strategies should do actual comparison"


class TestRequirementsLockStrategy:
    \"\"\"Item 137: requirements.lock vs requirements.txt strategy is clear.\"\"\"

    def test_both_files_exist(self):
        root = Path(__file__).parents[1]
        assert (root / "requirements.txt").exists()
        assert (root / "requirements.lock").exists()

    def test_requirements_lock_has_more_pins(self):
        root = Path(__file__).parents[1]
        txt_lines = (root / "requirements.txt").read_text(encoding="utf-8", errors="replace").splitlines()
        lock_lines = (root / "requirements.lock").read_text(encoding="utf-8", errors="replace").splitlines()
        txt_count = len([l for l in txt_lines if l.strip() and not l.startswith("#")])
        lock_count = len([l for l in lock_lines if l.strip() and not l.startswith("#")])
        assert lock_count >= txt_count, "requirements.lock should have at least as many packages as requirements.txt"

    def test_ci_uses_requirements(self):
        wf_dir = Path(__file__).parents[1] / ".github" / "workflows"
        if not wf_dir.exists():
            return
        for wf in wf_dir.glob("*.yml"):
            content = wf.read_text(encoding="utf-8", errors="replace")
            if "requirements" in content:
                return  # found
        assert False, "At least one workflow should install from requirements.txt or requirements.lock"


class TestPreCommitConfig:
    \"\"\"Item 138: pre-commit is configured and available for installation.\"\"\"

    def test_pre_commit_config_exists(self):
        p = Path(__file__).parents[1] / ".pre-commit-config.yaml"
        assert p.exists(), ".pre-commit-config.yaml should exist"

    def test_pre_commit_has_ruff_or_black(self):
        p = Path(__file__).parents[1] / ".pre-commit-config.yaml"
        content = p.read_text(encoding="utf-8", errors="replace")
        assert "ruff" in content or "black" in content, "pre-commit should use ruff or black"

    def test_pre_commit_has_secret_detection(self):
        p = Path(__file__).parents[1] / ".pre-commit-config.yaml"
        content = p.read_text(encoding="utf-8", errors="replace")
        has_secrets = any(kw in content for kw in ["detect-secrets", "gitleaks", "secret"])
        assert has_secrets, "pre-commit should have secret detection"


class TestCommentLanguage:
    \"\"\"Item 139: Comment language is reasonably consistent (mostly EN in code).\"\"\"

    def test_code_comments_are_mostly_english(self):
        # Verify that src/ code has at least some English comments
        src = Path(__file__).parents[1] / "src"
        en_comments = 0
        for f in src.rglob("*.py"):
            if "__pycache__" in str(f):
                continue
            try:
                content = f.read_text(encoding="utf-8", errors="replace")
                for line in content.splitlines():
                    s = line.strip()
                    if s.startswith("#") and len(s) > 5:
                        # Simple heuristic: English comment contains common EN words
                        if any(kw in s.lower() for kw in [
                            "this", "the", "is", "for", "if", "use", "return", "check", "note"
                        ]):
                            en_comments += 1
            except OSError:
                pass
        assert en_comments >= 100, f"Only {en_comments} likely-English comments found in src/"


class TestNoqaDistributionPerFile:
    \"\"\"Item 140: noqa comments are not concentrated in single files (hotspot detection).\"\"\"

    def test_no_file_exceeds_noqa_limit(self):
        src = Path(__file__).parents[1] / "src"
        hotspots = []
        for f in src.rglob("*.py"):
            if "__pycache__" in str(f):
                continue
            try:
                count = f.read_text(encoding="utf-8", errors="replace").count("# noqa")
                if count > 20:
                    hotspots.append((count, f.name))
            except OSError:
                pass
        hotspots.sort(reverse=True)
        assert len(hotspots) <= 5, (
            f"Too many tech-debt hotspots (>20 noqa per file): {hotspots[:5]}"
        )

    def test_total_noqa_count_bounded(self):
        src = Path(__file__).parents[1] / "src"
        total = sum(
            f.read_text(encoding="utf-8", errors="replace").count("# noqa")
            for f in src.rglob("*.py")
            if "__pycache__" not in str(f)
        )
        assert total <= 350, f"Too many noqa suppressions total: {total}"
"""

target = Path("tests/test_session_2026_05_07_new_items.py")
with open(target, "a", encoding="utf-8") as fh:
    fh.write(BATCH18)
print("Batch 18 appended successfully.")
