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

from src.assembled_core.qa.qa_gates import (
    QAResult,
    check_leakage,
    evaluate_all_gates,
)

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


# ---------------------------------------------------------------------------
# evaluate_all_gates wiring (E-059 follow-up, 2026-08-01)
#
# Before this, check_leakage existed but was in NO gate summary -> its state
# (checked / not checked / leaking) was invisible in every QA artifact. It is
# now part of the summary. Fail-open without a frame is UNCHANGED and
# deliberate; what changed is that "not checked" is now visible.
# ---------------------------------------------------------------------------


def _metrics():
    """Deterministic positive equity curve -> valid PerformanceMetrics.

    NO RNG on purpose: these tests assert gate outcomes (BLOCK / not-BLOCK),
    and NumPy guarantees stream stability only for legacy RandomState, not
    for Generator — an RNG-derived curve would make a core-contract test
    depend on the installed numpy version (Rule 40 drift class).
    """
    import numpy as np

    from src.assembled_core.qa.metrics import compute_all_metrics

    dates = pd.date_range("2020-01-01", periods=252, freq="D")
    equity = 10000.0 * (1.0003 ** np.arange(252, dtype=float))
    return compute_all_metrics(
        pd.DataFrame({"timestamp": dates, "equity": equity}), start_capital=10000.0
    )


def _leakage_gate(summary):
    return next(g for g in summary.gate_results if g.gate_name == "leakage_detection")


def test_evaluate_all_gates_includes_leakage_gate_as_visible_skip():
    """No feature_df -> gate present, OK, and explicitly marked NOT checked."""
    gate = _leakage_gate(evaluate_all_gates(_metrics()))
    assert gate.result == QAResult.OK
    assert gate.details.get("skipped") is True
    assert gate.details.get("skip_kind") == "no_frame"
    # The reason is the only honesty anchor that reaches the markdown report
    # and the backtest log (neither renders details) -> it must not read
    # like a clean bill of health.
    assert "NOT CHECKED" in gate.reason


def test_skip_kind_distinguishes_no_frame_from_empty_frame():
    """An EMPTY frame (e.g. failed altdata load) is a different finding than
    'caller passed nothing' — both skip, but they must be distinguishable."""
    assert check_leakage(feature_df=None).details["skip_kind"] == "no_frame"
    assert (
        check_leakage(feature_df=pd.DataFrame()).details["skip_kind"] == "empty_frame"
    )


def test_evaluate_all_gates_leakage_blocks_summary():
    """A leaking frame must BLOCK the overall summary, not just the gate."""
    summary = evaluate_all_gates(_metrics(), feature_df=_make_frame([0.0, 2.0, 1.5]))
    gate = _leakage_gate(summary)
    assert gate.result == QAResult.BLOCK
    assert gate.details["violations"] == 1
    assert summary.overall_result == QAResult.BLOCK
    assert summary.blocked_gates >= 1


def test_evaluate_all_gates_leakage_clean_frame_is_checked_not_skipped():
    """A clean frame -> OK WITHOUT the skipped flag (really checked)."""
    gate = _leakage_gate(
        evaluate_all_gates(_metrics(), feature_df=_make_frame([0.0, 0.0, 1.5]))
    )
    assert gate.result == QAResult.OK
    assert "skipped" not in gate.details
    assert gate.details["rows_checked"] == 3


def test_skipped_gate_is_not_counted_as_passed():
    """E-066: a not-checked gate must not inflate the green aggregate.

    Aggregate consumers (API gate_counts, '**Passed:** N' in the daily QA
    report, the backtest log line) read passed_gates and never see the
    reason string — so the skip has to be subtracted there, not only
    explained in details.
    """
    skipped = evaluate_all_gates(_metrics())
    checked = evaluate_all_gates(_metrics(), feature_df=_make_frame([0.0, 0.0, 1.5]))

    assert skipped.skipped_gates == 1
    assert checked.skipped_gates == 0
    # Same 8 gates in both, but only the really-checked one counts as passed.
    assert len(skipped.gate_results) == len(checked.gate_results) == 8
    assert checked.passed_gates == skipped.passed_gates + 1
    # Overall verdict must not change just because a gate was skipped.
    assert skipped.overall_result == checked.overall_result


def test_skipped_leakage_gate_writes_no_qa_block_flag(tmp_path):
    """CORE CONTRACT of the wiring step: adding the gate must NOT create a
    new live-pilot preflight halt. Without a feature_df the gate is OK ->
    summary not BLOCK -> write_qa_block_flag returns None and writes nothing."""
    from src.assembled_core.qa.qa_gates import write_qa_block_flag

    flag = tmp_path / "qa_block.json"
    summary = evaluate_all_gates(_metrics())
    assert summary.overall_result != QAResult.BLOCK
    assert write_qa_block_flag(summary, source="test", flag_path=flag) is None
    assert not flag.exists()


def test_leaking_frame_lands_in_qa_block_flag_payload(tmp_path):
    """Counterpart: a real leak must reach the operator-facing flag payload
    under its own gate name, not just the in-memory summary."""
    import json as _json

    from src.assembled_core.qa.qa_gates import write_qa_block_flag

    flag = tmp_path / "qa_block.json"
    summary = evaluate_all_gates(_metrics(), feature_df=_make_frame([0.0, 2.0, 1.5]))
    assert write_qa_block_flag(summary, source="test", flag_path=flag) is not None
    payload = _json.loads(flag.read_text(encoding="utf-8"))
    assert "leakage_detection" in [g["gate"] for g in payload["blocked_gates"]]


def test_column_mismatch_blocks_via_summary(tmp_path):
    """Documented halt footgun (Stage-1 MAJOR-3): a SCHEMA mismatch — not a
    leak — is fail-closed BLOCK and would arm the qa_block flag. Pinned as
    deliberate behaviour so the first real caller is not surprised."""
    from src.assembled_core.qa.qa_gates import write_qa_block_flag

    df = _make_frame([0.0, 0.0, 1.5]).rename(columns={"feature": "sent_score"})
    summary = evaluate_all_gates(_metrics(), feature_df=df)  # no column override
    gate = _leakage_gate(summary)
    assert gate.result == QAResult.BLOCK
    assert "feature" in gate.details["missing_columns"]
    assert summary.overall_result == QAResult.BLOCK
    flag = tmp_path / "qa_block.json"
    assert write_qa_block_flag(summary, source="test", flag_path=flag) is not None


def test_evaluate_all_gates_leakage_custom_columns():
    """Column overrides reach check_leakage (no silent default-column skip)."""
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(["2024-01-10", "2024-01-20"], utc=True),
            "disc": pd.to_datetime(["2024-01-15"] * 2, utc=True),
            "sent_score": [0.7, 0.9],
        }
    )
    gate = _leakage_gate(
        evaluate_all_gates(
            _metrics(),
            feature_df=df,
            leakage_feature_col="sent_score",
            leakage_disclosure_col="disc",
            leakage_timestamp_col="ts",
        )
    )
    assert gate.result == QAResult.BLOCK
    assert gate.details["violations"] == 1
