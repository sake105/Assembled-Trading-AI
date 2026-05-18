"""Tests for scripts/ci/check_adversarial_reviewer_pattern.py (C2-051)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make scripts/ importable
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from scripts.ci.check_adversarial_reviewer_pattern import (
    check_pattern,
    expected_review_path,
    find_research_notebooks,
    main,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _touch_notebook(path: Path) -> None:
    """Create a minimal valid-ish .ipynb file (empty content but file exists)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '{"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}\n'
    )


# ---------------------------------------------------------------------------
# Helpers under test
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestExpectedReviewPath:
    def test_simple_swap(self, tmp_path: Path) -> None:
        r = tmp_path / "research_alpha_decay.ipynb"
        assert expected_review_path(r) == tmp_path / "review_alpha_decay.ipynb"

    def test_nested_directory(self, tmp_path: Path) -> None:
        r = tmp_path / "factors" / "research_momentum.ipynb"
        assert expected_review_path(r) == tmp_path / "factors" / "review_momentum.ipynb"

    def test_non_research_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="not a research notebook"):
            expected_review_path(tmp_path / "trend_baseline.ipynb")


@pytest.mark.fast
class TestFindResearchNotebooks:
    def test_empty_root_returns_empty(self, tmp_path: Path) -> None:
        result = find_research_notebooks(tmp_path)
        assert result == []

    def test_finds_research_notebooks(self, tmp_path: Path) -> None:
        _touch_notebook(tmp_path / "research_a.ipynb")
        _touch_notebook(tmp_path / "research_b.ipynb")
        _touch_notebook(tmp_path / "review_a.ipynb")  # not picked up
        _touch_notebook(tmp_path / "other.ipynb")  # not picked up
        result = find_research_notebooks(tmp_path)
        names = sorted(p.name for p in result)
        assert names == ["research_a.ipynb", "research_b.ipynb"]

    def test_recursive(self, tmp_path: Path) -> None:
        _touch_notebook(tmp_path / "topic" / "research_nested.ipynb")
        result = find_research_notebooks(tmp_path)
        assert len(result) == 1
        assert result[0].name == "research_nested.ipynb"

    def test_excludes_dead_ends(self, tmp_path: Path) -> None:
        """research_*.ipynb under dead_ends/ is intentionally excluded."""
        _touch_notebook(tmp_path / "research_active.ipynb")
        _touch_notebook(tmp_path / "dead_ends" / "research_old.ipynb")
        result = find_research_notebooks(tmp_path)
        assert len(result) == 1
        assert result[0].name == "research_active.ipynb"

    def test_nonexistent_root_returns_empty(self, tmp_path: Path) -> None:
        result = find_research_notebooks(tmp_path / "nope")
        assert result == []


# ---------------------------------------------------------------------------
# check_pattern
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestCheckPattern:
    def test_no_research_notebooks_no_missing(self, tmp_path: Path) -> None:
        research, missing = check_pattern(tmp_path)
        assert research == []
        assert missing == []

    def test_all_paired_no_missing(self, tmp_path: Path) -> None:
        _touch_notebook(tmp_path / "research_alpha.ipynb")
        _touch_notebook(tmp_path / "review_alpha.ipynb")
        _touch_notebook(tmp_path / "research_beta.ipynb")
        _touch_notebook(tmp_path / "review_beta.ipynb")
        research, missing = check_pattern(tmp_path)
        assert len(research) == 2
        assert missing == []

    def test_some_missing(self, tmp_path: Path) -> None:
        _touch_notebook(tmp_path / "research_alpha.ipynb")
        _touch_notebook(tmp_path / "review_alpha.ipynb")
        _touch_notebook(tmp_path / "research_beta.ipynb")
        # research_beta has NO matching review_beta.ipynb
        research, missing = check_pattern(tmp_path)
        assert len(research) == 2
        assert len(missing) == 1
        r, expected = missing[0]
        assert r.name == "research_beta.ipynb"
        assert expected.name == "review_beta.ipynb"


# ---------------------------------------------------------------------------
# main() CLI
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestMainCLI:
    def test_main_returns_0_on_empty_root(self, tmp_path: Path) -> None:
        code = main(["--root", str(tmp_path)])
        assert code == 0

    def test_main_returns_0_on_all_paired(self, tmp_path: Path) -> None:
        _touch_notebook(tmp_path / "research_alpha.ipynb")
        _touch_notebook(tmp_path / "review_alpha.ipynb")
        code = main(["--root", str(tmp_path)])
        assert code == 0

    def test_main_returns_1_on_missing_review(self, tmp_path: Path) -> None:
        _touch_notebook(tmp_path / "research_alpha.ipynb")
        # no review_alpha.ipynb
        code = main(["--root", str(tmp_path)])
        assert code == 1

    def test_main_returns_0_on_real_repo_root(self) -> None:
        """Sanity check against the actual research/ directory: no
        research_*.ipynb exist today, so the script must exit 0. If a
        research_*.ipynb is added later without a review_*.ipynb, this
        test will turn red — exactly the CI gate we want."""
        code = main(["--root", "research"])
        assert code == 0
