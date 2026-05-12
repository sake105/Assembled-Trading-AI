"""Audit C4-082 — pin the canonical latency constants so feature builders
cannot silently drift out of sync.
"""

from __future__ import annotations


def test_canonical_latency_constants() -> None:
    from src.assembled_core.data.source_latencies import (
        ACLED_DAYS,
        CONGRESS_DAYS,
        EARNINGS_DAYS,
        EDGAR_DAYS,
        FINRA_DAYS,
        GDELT_DAYS,
        INSIDER_DAYS,
        LATENCY_DAYS,
        WIKIPEDIA_DAYS,
        latency_for,
    )

    assert INSIDER_DAYS == 2
    assert CONGRESS_DAYS == 45
    assert EARNINGS_DAYS == 0
    assert ACLED_DAYS == 1
    assert GDELT_DAYS == 1
    # Audit C4-025/026/027 — see source_latencies.py rationale block.
    assert EDGAR_DAYS == 1
    assert FINRA_DAYS == 1
    assert WIKIPEDIA_DAYS == 1

    assert LATENCY_DAYS["insider"] == INSIDER_DAYS
    assert LATENCY_DAYS["congress"] == CONGRESS_DAYS
    assert LATENCY_DAYS["earnings"] == EARNINGS_DAYS
    assert LATENCY_DAYS["edgar"] == EDGAR_DAYS
    assert LATENCY_DAYS["finra"] == FINRA_DAYS
    assert LATENCY_DAYS["wikipedia"] == WIKIPEDIA_DAYS

    assert latency_for("Insider") == INSIDER_DAYS  # case-insensitive
    assert latency_for("CONGRESS") == CONGRESS_DAYS
    assert latency_for("Edgar") == EDGAR_DAYS
    assert latency_for("FINRA") == FINRA_DAYS


def test_unknown_source_raises_key_error() -> None:
    import pytest

    from src.assembled_core.data.source_latencies import latency_for

    with pytest.raises(KeyError):
        latency_for("nonexistent_source_xyz")


def test_feature_builder_defaults_match_canonical_constants() -> None:
    """The feature-builder default arguments MUST stay numerically equal to
    the canonical constants. If someone changes one without the other, this
    test catches the drift (audit C4-082 invariant).
    """
    import inspect

    from src.assembled_core.data.source_latencies import (
        CONGRESS_DAYS,
        INSIDER_DAYS,
    )
    from src.assembled_core.features.congress_features import add_congress_features
    from src.assembled_core.features.insider_features import add_insider_features

    insider_sig = inspect.signature(add_insider_features)
    congress_sig = inspect.signature(add_congress_features)

    assert insider_sig.parameters["disclosure_latency_days"].default == INSIDER_DAYS
    assert congress_sig.parameters["disclosure_latency_days"].default == CONGRESS_DAYS
