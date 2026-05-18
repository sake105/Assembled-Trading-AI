"""Adversarial Reviewer Notebook Pattern check (audit C2-051).

Enforces the convention: every ``research_*.ipynb`` notebook (research
question / hypothesis exploration) MUST have a corresponding
``review_*.ipynb`` notebook (adversarial critique / failure-mode
analysis). The two notebooks live side-by-side and the review notebook
documents what the original would look like if it were WRONG — what
would have to change in the data, the method, or the assumptions for
the conclusion to flip.

Why: research notebooks are systematically biased toward confirming the
hypothesis (selection bias of which notebooks get cleaned up and shared).
The adversarial-review notebook is a Lopez-de-Prado-style discipline that
makes the failure-mode-search explicit and version-controlled. Without
the review notebook, only the success story survives.

The pattern is enforced AT CONVENTION-TIME (this CI hook) so no future
research notebook can land on main without its adversarial reviewer.

Usage::

    python scripts/ci/check_adversarial_reviewer_pattern.py
    python scripts/ci/check_adversarial_reviewer_pattern.py --root research/

Exit codes:
    0: all research notebooks have a corresponding review notebook (or
       no research_*.ipynb exist yet — convention not adopted)
    1: at least one research notebook lacks a review notebook
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def find_research_notebooks(root: Path) -> list[Path]:
    """Find all ``research_*.ipynb`` notebooks under ``root``, recursively.

    Excludes ``dead_ends/`` (research notebooks moved to dead-ends are no
    longer load-bearing; the convention applies to live research only).
    """
    if not root.exists():
        return []
    notebooks = []
    for p in root.rglob("research_*.ipynb"):
        if "dead_ends" in p.parts:
            continue
        notebooks.append(p)
    return sorted(notebooks)


def expected_review_path(research_path: Path) -> Path:
    """The matching ``review_*.ipynb`` for a given ``research_*.ipynb``.

    Same directory, name swapped: ``research_X.ipynb → review_X.ipynb``.
    """
    name = research_path.name
    if not name.startswith("research_"):
        raise ValueError(f"not a research notebook: {research_path}")
    review_name = "review_" + name[len("research_") :]
    return research_path.with_name(review_name)


def check_pattern(root: Path) -> tuple[list[Path], list[tuple[Path, Path]]]:
    """Return (all_research_notebooks, missing_pairs).

    missing_pairs is a list of (research_notebook, expected_review_path)
    tuples for which the review notebook does NOT exist.
    """
    research = find_research_notebooks(root)
    missing = []
    for r in research:
        expected = expected_review_path(r)
        if not expected.exists():
            missing.append((r, expected))
    return research, missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("research"),
        help="Directory to scan for research_*.ipynb notebooks (default: research/)",
    )
    args = parser.parse_args(argv)

    research, missing = check_pattern(args.root)

    if not research:
        msg = (
            f"[adversarial-reviewer] no research_*.ipynb under {args.root} — "
            "convention not yet adopted (this is acceptable)."
        )
        print(msg)
        return 0

    print(
        f"[adversarial-reviewer] {len(research)} research notebook(s) under {args.root}:"
    )
    for r in research:
        review = expected_review_path(r)
        status = "OK" if review.exists() else "MISSING"
        print(f"  - {r} → {review.name} [{status}]")

    if missing:
        print()
        print(
            f"[adversarial-reviewer] ERROR: {len(missing)} research notebook(s) "
            "lack an adversarial review notebook:"
        )
        for r, expected in missing:
            print(f"  - {r}  →  expected: {expected}")
        print()
        print(
            "Adversarial Reviewer Notebook Pattern (audit C2-051): every "
            "research_*.ipynb must be paired with a review_*.ipynb that "
            "documents the failure modes / what would have to change for "
            "the conclusion to flip. Add the missing review notebook(s) "
            "before merging."
        )
        return 1

    print(
        f"[adversarial-reviewer] OK: all {len(research)} research notebook(s) "
        "have a matching adversarial review notebook."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
