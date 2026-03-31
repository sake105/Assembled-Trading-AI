"""Smoke tests for CLI: import and parse without optional deps (e.g. tabulate)."""

from __future__ import annotations

import pytest


pytestmark = [pytest.mark.unit, pytest.mark.phase6]


def test_cli_create_parser_works() -> None:
    """create_parser() should succeed without optional packages (tabulate etc.)."""
    from scripts.cli import create_parser

    parser = create_parser()
    assert parser is not None


def test_cli_parse_run_paper_range_args() -> None:
    """Parsing run_paper_range args should not crash (no subcommand execution)."""
    from scripts.cli import create_parser

    parser = create_parser()
    args = parser.parse_args(
        [
            "run_paper_range",
            "--start",
            "2026-02-01",
            "--end",
            "2026-02-03",
            "--mode",
            "shadow",
        ]
    )
    assert args.func is not None
    assert getattr(args, "start", None) == "2026-02-01"
    assert getattr(args, "end", None) == "2026-02-03"
    assert getattr(args, "mode", None) == "shadow"
