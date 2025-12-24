"""Unit tests for profiling functionality in profile_jobs.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import scripts.profile_jobs as pj

pytestmark = pytest.mark.advanced


@pytest.fixture
def dummy_job_func() -> callable:
    """Create a fast dummy job function for testing (no sleep, instant return)."""

    def _dummy_job() -> None:
        # Just do a trivial computation to have something to profile
        _ = sum(range(100))

    return _dummy_job


def test_profile_job_cprofile_output_generation(tmp_path: Path, dummy_job_func: callable) -> None:
    """Test that cProfile generates .prof and .txt files."""
    # Patch JOB_MAP to include our dummy job
    original_job_map = pj.JOB_MAP.copy()
    pj.JOB_MAP["TEST_DUMMY"] = dummy_job_func

    try:
        profile_dir = tmp_path / "profiles" / "TEST_DUMMY" / "20240101T000000Z"
        profile_dir.mkdir(parents=True, exist_ok=True)

        # Create a minimal profile_job call (we'll patch the job_func call)
        with patch("scripts.profile_jobs.JOB_MAP", {"TEST_DUMMY": dummy_job_func}):
            pj.profile_job(
                job_name="TEST_DUMMY",
                profiler="cprofile",
                profile_out=profile_dir,
                top_n=10,
            )

        # Check that output files exist
        prof_path = profile_dir / "profile_TEST_DUMMY.prof"
        stats_path = profile_dir / "profile_TEST_DUMMY_stats.txt"
        summary_path = profile_dir / "profile_summary.log"

        assert prof_path.exists(), "cProfile .prof file should exist"
        assert stats_path.exists(), "cProfile stats .txt file should exist"
        assert summary_path.exists(), "Profile summary log should exist"

        # Check that stats file contains expected content
        stats_content = stats_path.read_text(encoding="utf-8")
        assert "cProfile stats for TEST_DUMMY" in stats_content
        assert "Total runtime:" in stats_content

        # Check summary log
        summary_content = summary_path.read_text(encoding="utf-8")
        assert "job_name=TEST_DUMMY" in summary_content
        assert "profiler=cprofile" in summary_content
        assert "profile_file=profile_TEST_DUMMY.prof" in summary_content

    finally:
        # Restore original JOB_MAP
        pj.JOB_MAP.clear()
        pj.JOB_MAP.update(original_job_map)


def test_profile_job_pyinstrument_output_generation(
    tmp_path: Path, dummy_job_func: callable, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that pyinstrument generates HTML and optionally text files."""
    # Mock pyinstrument
    mock_profiler = MagicMock()
    mock_profiler.output_html.return_value = "<html>Test Profile</html>"
    mock_profiler.output_text.return_value = "Test Profile Text"

    # Create a mock pyinstrument module
    mock_pyinstrument_module = MagicMock()
    mock_pyinstrument_module.Profiler.return_value = mock_profiler

    monkeypatch.setattr("scripts.profile_jobs.pyinstrument", mock_pyinstrument_module)

    # Patch JOB_MAP
    original_job_map = pj.JOB_MAP.copy()
    pj.JOB_MAP["TEST_DUMMY"] = dummy_job_func

    try:
        profile_dir = tmp_path / "profiles" / "TEST_DUMMY" / "20240101T000000Z"
        profile_dir.mkdir(parents=True, exist_ok=True)

        pj.profile_job(
            job_name="TEST_DUMMY",
            profiler="pyinstrument",
            profile_out=profile_dir,
        )

        # Check that output files exist
        html_path = profile_dir / "profile_TEST_DUMMY.html"
        text_path = profile_dir / "profile_TEST_DUMMY.txt"
        summary_path = profile_dir / "profile_summary.log"

        assert html_path.exists(), "pyinstrument HTML file should exist"
        assert text_path.exists(), "pyinstrument text file should exist"
        assert summary_path.exists(), "Profile summary log should exist"

        # Check HTML content
        html_content = html_path.read_text(encoding="utf-8")
        assert "<html>" in html_content

        # Check summary log
        summary_content = summary_path.read_text(encoding="utf-8")
        assert "profiler=pyinstrument" in summary_content
        assert "html_file=profile_TEST_DUMMY.html" in summary_content

        # Verify profiler was called correctly
        mock_profiler.start.assert_called_once()
        mock_profiler.stop.assert_called_once()
        mock_profiler.output_html.assert_called_once()

    finally:
        # Restore original JOB_MAP
        pj.JOB_MAP.clear()
        pj.JOB_MAP.update(original_job_map)


def test_profile_job_none_profiler(tmp_path: Path, dummy_job_func: callable) -> None:
    """Test that profiler='none' just runs the job without generating profile files."""
    original_job_map = pj.JOB_MAP.copy()
    pj.JOB_MAP["TEST_DUMMY"] = dummy_job_func

    try:
        profile_dir = tmp_path / "profiles" / "TEST_DUMMY"
        profile_dir.mkdir(parents=True, exist_ok=True)

        pj.profile_job(
            job_name="TEST_DUMMY",
            profiler="none",
            profile_out=profile_dir,
        )

        # Check that only summary log exists (no .prof or .html files)
        summary_path = profile_dir / "profile_summary.log"
        assert summary_path.exists()

        # No .prof files should exist
        prof_files = list(profile_dir.glob("*.prof"))
        assert len(prof_files) == 0, "No .prof files should be created when profiler='none'"

        html_files = list(profile_dir.glob("*.html"))
        assert len(html_files) == 0, "No .html files should be created when profiler='none'"

    finally:
        pj.JOB_MAP.clear()
        pj.JOB_MAP.update(original_job_map)


def test_profile_job_pyinstrument_not_available(
    dummy_job_func: callable, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that pyinstrument raises an error when not available."""
    monkeypatch.setattr("scripts.profile_jobs.pyinstrument", None)

    original_job_map = pj.JOB_MAP.copy()
    pj.JOB_MAP["TEST_DUMMY"] = dummy_job_func

    try:
        with pytest.raises(RuntimeError, match="pyinstrument not available"):
            pj.profile_job(
                job_name="TEST_DUMMY",
                profiler="pyinstrument",
            )
    finally:
        pj.JOB_MAP.clear()
        pj.JOB_MAP.update(original_job_map)


def test_profile_job_output_directory_structure(tmp_path: Path, dummy_job_func: callable) -> None:
    """Test that profile output is written to the correct directory structure."""
    original_job_map = pj.JOB_MAP.copy()
    pj.JOB_MAP["TEST_DUMMY"] = dummy_job_func

    try:
        custom_out_dir = tmp_path / "custom_profiles" / "my_test"
        pj.profile_job(
            job_name="TEST_DUMMY",
            profiler="cprofile",
            profile_out=custom_out_dir,
        )

        # Check that files are in the custom directory
        assert custom_out_dir.exists()
        prof_path = custom_out_dir / "profile_TEST_DUMMY.prof"
        assert prof_path.exists()

    finally:
        pj.JOB_MAP.clear()
        pj.JOB_MAP.update(original_job_map)

