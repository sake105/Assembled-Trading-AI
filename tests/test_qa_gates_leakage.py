"""E2E unit tests for the check_leakage QA gate (qa_gates.py).

Regression guard for the E-059-class bug (2026-07): check_leakage used to
call assert_feature_zero_before_disclosure with a wrong signature
(df=/feature_col=/...), raising an uncaught TypeError for every non-empty
feature_df. The gate now implements the row-wise PIT check inline:
non-zero feature values before their disclosure date -> BLOCK.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.assembled_core.qa.qa_gates import QAResult, check_leakage

pytestmark = pytest.mark.fast


def _make_frame(values: list[float]) -> pd.DataFrame:
    """3-row frame: rows 0-1 before disclosure, row 2 after disclosure."""
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-10", "2024-01-12", "2024-01-20"], utc=True
            ),
            "disclosure_date": pd.to_datetime(["2024-01-15"] * 3, utc=True),
            "feature": values,
        }
    )


def test_leakage_free_frame_passes():
    # zero before disclosure, non-zero after -> clean
    res = check_leakage(feature_df=_make_frame([0.0, 0.0, 1.5]))
    assert res.result == QAResult.OK
    assert res.details["rows_checked"] == 3
    assert "skipped" not in res.details


def test_leak_before_disclosure_blocks():
    # row 1 (2024-01-12) is non-zero before disclosure (2024-01-15) -> leak
    res = check_leakage(feature_df=_make_frame([0.0, 2.0, 1.5]))
    assert res.result == QAResult.BLOCK
    assert res.details["violations"] == 1
    assert "LEAKAGE DETECTED" in res.reason
    sample = json.loads(res.details["sample_violations"])
    assert len(sample) == 1
    assert sample[0]["feature"] == "2.0"


def test_nan_before_disclosure_counts_as_zero():
    # NaN is "no value yet", not a leak (helper-consistent semantics)
    res = check_leakage(feature_df=_make_frame([float("nan"), float("nan"), 1.5]))
    assert res.result == QAResult.OK


def test_nat_disclosure_with_nonzero_value_blocks():
    # fail-closed: non-zero value without a known disclosure date
    df = _make_frame([0.0, 0.0, 1.5])
    df.loc[2, "disclosure_date"] = pd.NaT
    res = check_leakage(feature_df=df)
    assert res.result == QAResult.BLOCK
    assert res.details["violations"] == 1


def test_missing_columns_block():
    df = pd.DataFrame({"foo": [1.0]})
    res = check_leakage(feature_df=df)
    assert res.result == QAResult.BLOCK
    for col in ("feature", "disclosure_date", "timestamp"):
        assert col in res.details["missing_columns"]


def test_custom_column_names():
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(["2024-01-10", "2024-01-20"], utc=True),
            "disc": pd.to_datetime(["2024-01-15"] * 2, utc=True),
            "sent_score": [0.7, 0.9],
        }
    )
    res = check_leakage(
        feature_df=df,
        feature_col="sent_score",
        disclosure_col="disc",
        timestamp_col="ts",
    )
    assert res.result == QAResult.BLOCK  # row 0 leaks
    assert res.details["violations"] == 1


def test_none_and_empty_stay_fail_open_with_visible_skip():
    for df in (None, pd.DataFrame()):
        res = check_leakage(feature_df=df)
        assert res.result == QAResult.OK
        assert res.details.get("skipped") is True


def test_nat_timestamp_with_nonzero_value_blocks():
    """Fail-closed on BOTH unknown-time axes (F-senior-6): a non-zero value
    whose OBSERVATION time is unknown (NaT timestamp) cannot be proven
    PIT-safe and must be flagged, symmetric to the NaT-disclosure case."""
    import pandas as pd

    from src.assembled_core.qa.qa_gates import QAResult, check_leakage

    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime([pd.NaT, "2024-01-20"], utc=True),
            "disclosure_date": pd.to_datetime(["2024-01-15"] * 2, utc=True),
            "feature": [0.5, 0.5],
        }
    )
    res = check_leakage(feature_df=df)
    assert res.result == QAResult.BLOCK
    assert res.details["violations"] == 1  # only the NaT-timestamp row
