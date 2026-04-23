"""Tests for wave-121 module wiring into trading_cycle.py.

Covers:
  Step 8.57 — events.disclosures.emit (emit_json_artifact)
  Step 8.58 — events.disclosures.evidence (summarize_evidence)
  Step 8.59 — events.disclosures.fetch_edgar (fetch_edgar_form4)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.assembled_core.events.disclosures.emit import emit_json_artifact
from src.assembled_core.events.disclosures.evidence import summarize_evidence
from src.assembled_core.events.disclosures.fetch_edgar import fetch_edgar_form4


# ---------------------------------------------------------------------------
# events.disclosures.emit (Step 8.57)
# ---------------------------------------------------------------------------

def test_emit_json_artifact_importable():
    assert emit_json_artifact is not None


def test_emit_json_artifact_writes_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "artifact.json"
        emit_json_artifact({"key": "value"}, path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data == {"key": "value"}


# ---------------------------------------------------------------------------
# events.disclosures.evidence (Step 8.58)
# ---------------------------------------------------------------------------

def test_summarize_evidence_importable():
    assert summarize_evidence is not None


def test_summarize_evidence_empty():
    result = summarize_evidence([], {})
    assert isinstance(result, dict)


def test_summarize_evidence_has_keys():
    result = summarize_evidence([], {})
    assert "evidence_ok" in result


def test_summarize_evidence_false_on_empty():
    result = summarize_evidence([], {})
    assert result["evidence_ok"] is False


# ---------------------------------------------------------------------------
# events.disclosures.fetch_edgar (Step 8.59)
# ---------------------------------------------------------------------------

def test_fetch_edgar_form4_importable():
    assert fetch_edgar_form4 is not None
