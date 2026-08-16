"""Tests for the per-request ingest protocol (E-112).

The single property that matters here: an EMPTY result must leave a trace.
That is the whole reason the module exists — without it, "no file on disk" is
indistinguishable from "never asked", and every later coverage claim is a
guess. Most of these tests exist to keep that property from quietly eroding.
"""

from __future__ import annotations

import json

import pytest

from src.assembled_core.data.pull_log import (
    DEFAULT_LOG_DIR,
    SCHEMA_VERSION,
    STATUS_EMPTY,
    STATUS_ERROR,
    STATUS_OK,
    STATUS_SKIPPED,
    PullLog,
)


def test_empty_result_is_recorded_not_dropped():
    """THE E-112 property: zero rows still produces an entry."""
    plog = PullLog(source="probe")
    plog.record("LEH", window=("2008-03-01", "2008-03-31"), http_status=200, n_rows=0)

    assert len(plog.entries) == 1
    entry = plog.entries[0]
    assert entry["key"] == "LEH"
    assert entry["status"] == STATUS_EMPTY
    assert entry["n_rows"] == 0
    assert entry["window_start"] == "2008-03-01"
    assert entry["window_end"] == "2008-03-31"


def test_status_is_derived_from_outcome():
    plog = PullLog(source="probe")
    plog.record("OK_SYM", n_rows=42)
    plog.record("EMPTY_SYM", n_rows=0)
    plog.record("ERR_SYM", n_rows=0, error="HTTP 401")

    assert [e["status"] for e in plog.entries] == [
        STATUS_OK,
        STATUS_EMPTY,
        STATUS_ERROR,
    ]


def test_error_wins_over_row_count():
    """An error with rows is still an error, never a success."""
    plog = PullLog(source="probe")
    plog.record("X", n_rows=5, error="partial read failed")
    assert plog.entries[0]["status"] == STATUS_ERROR


def test_summary_separates_empty_from_error():
    """Coverage questions need these two apart: 'not covered' != 'not reachable'."""
    plog = PullLog(source="probe")
    plog.record("A", n_rows=10)
    plog.record("B", n_rows=0)
    plog.record("C", n_rows=0, error="timeout")

    s = plog.summary()
    assert s["requested"] == 3
    assert s["ok"] == 1
    assert s["empty"] == 1
    assert s["error"] == 1
    assert s["returned_rows"] == 10
    assert s["empty_keys"] == ["B"]
    assert s["error_keys"] == ["C"]


def test_summary_can_only_speak_about_requested_keys():
    """A key never passed to record() must not appear anywhere.

    This is the E-112 lesson in assertion form: the protocol describes the
    requests it made, and nothing else. A symbol that was never asked about
    cannot show up as 'empty' and be mistaken for 'the vendor lacks it'.
    """
    plog = PullLog(source="probe")
    plog.record("ASKED", n_rows=0)

    s = plog.summary()
    assert s["requested"] == 1
    assert "NEVER_ASKED" not in s["empty_keys"]
    assert "NEVER_ASKED" not in s["error_keys"]


def test_write_produces_readable_protocol(tmp_path):
    plog = PullLog(source="eodhd", run_id="testrun")
    plog.record("AAPL", window=("2026-01-01", "2026-01-31"), http_status=200, n_rows=21)
    plog.record("LEH", window=("2008-03-01", "2008-03-31"), http_status=404, n_rows=0)

    out = plog.write(log_dir=tmp_path)
    assert out is not None and out.exists()

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema"] == SCHEMA_VERSION
    assert payload["source"] == "eodhd"
    assert payload["run_id"] == "testrun"
    assert payload["summary"]["requested"] == 2
    assert len(payload["entries"]) == 2
    # The failed/empty request must be in the written artifact, not just in memory.
    assert any(e["key"] == "LEH" and e["n_rows"] == 0 for e in payload["entries"])


def test_write_failure_never_raises(tmp_path, monkeypatch):
    """Bookkeeping must not be able to kill an ingest run that already fetched data."""
    plog = PullLog(source="probe")
    plog.record("A", n_rows=1)

    def _boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("pathlib.Path.write_text", _boom)
    assert plog.write(log_dir=tmp_path) is None


@pytest.mark.parametrize("bad_rows", ["not-a-number", object()])
def test_unparseable_row_count_degrades_to_zero(bad_rows):
    plog = PullLog(source="probe")
    plog.record("X", n_rows=bad_rows)
    assert plog.entries[0]["n_rows"] == 0
    assert plog.entries[0]["status"] == STATUS_EMPTY


def test_none_row_count_means_unknown_not_empty():
    """CONTRACT: n_rows=None is "not counted here", NOT "zero rows".

    A transport that cannot count rows (a text body, a dict envelope) must not
    be able to turn a successful request into an "empty" one — that would
    manufacture exactly the coverage confusion E-112 is about. It must equally
    not be able to claim rows it never counted, hence n_rows stays None.
    """
    plog = PullLog(source="probe")
    plog.record("X", http_status=200, n_rows=None)

    entry = plog.entries[0]
    assert entry["status"] == STATUS_OK
    assert entry["n_rows"] is None

    # An unknown count must not inflate the aggregate row total either.
    assert plog.summary()["returned_rows"] == 0
    assert plog.summary()["empty_keys"] == []


def test_unknown_status_is_recorded_as_error_not_silently_accepted():
    plog = PullLog(source="probe")
    plog.record("X", n_rows=5, status="totally-made-up")
    assert plog.entries[0]["status"] == STATUS_ERROR


# --- key aggregation (the E-148 fix) -------------------------------------
#
# summary() used to count LOG LINES and call them "requested". A puller that
# logged the same key from both the transport and the parse layer then made one
# currency pair appear as requested=2, ok=1, empty=1 — simultaneously successful
# and empty. Coverage ratios are built on that denominator, so it has to be a
# request count, not a row count.


def test_summary_aggregates_per_key_not_per_log_line():
    plog = PullLog(source="probe")
    plog.record("EURUSD", n_rows=None)  # transport layer: count unknown -> ok
    plog.record("EURUSD", n_rows=0)  # parse layer: nothing parsed -> empty

    s = plog.summary()
    assert s["requested"] == 1, "one key asked about, not two"
    assert s["log_lines"] == 2, "both lines are still kept"
    assert s["duplicate_keys"] == ["EURUSD"], "double logging must be visible"
    assert s["ok"] + s["empty"] + s["error"] + s["skipped"] == 1


def test_last_entry_wins_for_a_duplicated_key():
    plog = PullLog(source="probe")
    plog.record("X", n_rows=5)
    plog.record("X", n_rows=0)

    s = plog.summary()
    assert s["empty_keys"] == ["X"]
    assert s["ok"] == 0


def test_single_entry_per_key_reports_no_duplicates():
    plog = PullLog(source="probe")
    plog.record("A", n_rows=1)
    plog.record("B", n_rows=0)

    s = plog.summary()
    assert s["duplicate_keys"] == []
    assert s["requested"] == s["log_lines"] == 2


# --- skipped is its own state (E-112 contract) ---------------------------


def test_skipped_key_never_appears_in_empty_keys():
    """A key that was never requested must not read as "vendor has no data".

    That is the whole distinction E-112 exists for, and pressing such a key
    into `empty` would put it into a list this module documents as containing
    only requests it actually made.
    """
    plog = PullLog(source="probe")
    plog.record(
        "UNMAPPED", status=STATUS_SKIPPED, n_rows=0, skipped_reason="no mapping"
    )
    plog.record("ASKED", n_rows=0)

    s = plog.summary()
    assert s["skipped_keys"] == ["UNMAPPED"]
    assert s["empty_keys"] == ["ASKED"]
    assert "UNMAPPED" not in s["empty_keys"]
    assert "UNMAPPED" not in s["error_keys"], "skipped is not a vendor failure either"


# --- run_id ---------------------------------------------------------------


def test_run_id_defaults_to_a_timestamp_so_runs_do_not_overwrite(tmp_path):
    """Without a run id every run overwrites the previous one.

    The question E-112 was written for is retrospective ("were those eight
    symbols ever requested?"), and a single-slot file cannot answer it.
    """
    plog = PullLog(source="probe")
    assert plog.run_id, "run_id must not be empty by default"
    assert plog.run_id in str(plog.default_path(tmp_path))

    other = PullLog(source="probe", run_id="explicit")
    assert other.default_path(tmp_path) != plog.default_path(tmp_path)


def test_default_log_dir_is_repo_anchored_not_cwd_relative():
    """A CWD-relative default scatters protocols where nobody looks (E-146).

    Verified once by running a puller from /tmp and finding the protocol in
    /tmp/output/ops/. Without this assertion a regression back to
    ``Path("output") / "ops"`` would leave the whole suite green.
    """
    assert DEFAULT_LOG_DIR.is_absolute()
    assert DEFAULT_LOG_DIR.parts[-2:] == ("output", "ops")


# --- log-line arity (E-150) ----------------------------------------------
#
# A logging format string with the wrong number of placeholders HIDES ITSELF:
# logging catches the TypeError in emit() and calls handleError, so nothing
# propagates. No test can fail on it, the suite stays green, and the only
# symptom is "--- Logging error ---" on a console someone happens to read.
# That is how a 7-placeholder/8-argument line survived here until a full run
# with a RotatingFileHandler attached turned 13 tests red.
#
# record.getMessage() applies the %-formatting, so an arity mismatch raises
# here instead of disappearing.


def test_write_log_line_has_matching_placeholder_count(tmp_path, caplog):
    plog = PullLog(source="probe")
    plog.record("A", n_rows=5)
    plog.record("B", n_rows=0)
    plog.record("C", status=STATUS_SKIPPED, n_rows=0, skipped_reason="no mapping")

    with caplog.at_level("INFO", logger="src.assembled_core.data.pull_log"):
        assert plog.write(log_dir=tmp_path) is not None

    rendered = [r.getMessage() for r in caplog.records]
    assert any("requested" in m for m in rendered), "the summary line must be emitted"
    assert any("skipped" in m for m in rendered)


def test_duplicate_warning_line_has_matching_placeholder_count(tmp_path, caplog):
    """The duplicate-key WARNING is the other multi-placeholder line."""
    plog = PullLog(source="probe")
    plog.record("DUP", n_rows=1)
    plog.record("DUP", n_rows=0)

    with caplog.at_level("WARNING", logger="src.assembled_core.data.pull_log"):
        assert plog.write(log_dir=tmp_path) is not None

    rendered = [r.getMessage() for r in caplog.records]
    assert any("logged more than once" in m for m in rendered)


def test_non_finite_row_count_does_not_break_the_never_raises_contract(tmp_path):
    """OverflowError inherits from ArithmeticError, not ValueError.

    io_utils relies on record() never raising ("Kein try/except: PullLog.record
    ist per Vertrag nie-raise"), so a float('inf') arriving via **extra must not
    escape either record() or write().
    """
    plog = PullLog(source="probe")
    plog.record("X", n_rows=float("inf"))
    plog.record("Y", n_rows=float("nan"))

    # CONTRACT: non-finite means UNKNOWN, not zero. Mapping it to 0 would make
    # it "empty" and put it into empty_keys — the list coverage claims are
    # built from. An unknown count must never read as "the vendor has nothing".
    assert [e["n_rows"] for e in plog.entries] == [None, None]
    assert [e["status"] for e in plog.entries] == [STATUS_OK, STATUS_OK]
    assert plog.summary()["empty_keys"] == []
    assert plog.summary()["ok_rows_unknown"] == 2
    assert plog.write(log_dir=tmp_path) is not None
