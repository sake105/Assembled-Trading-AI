"""Tests for wave-88 module wiring into trading_cycle.py.

Covers:
  Step 8.117 — intel.crisis_alpha_worker (CrisisStateConfig)
  Step 8.118 — intel.evidence_grade_writer (EvidenceGradeWriter)
  Step 8.119 — intel.news_newsapi_fetcher (NewsAPIFetcher)
"""

from __future__ import annotations

import json
import pytest

from src.assembled_core.intel.crisis_alpha_worker import CrisisStateConfig
from src.assembled_core.intel.evidence_grade_writer import EvidenceGradeWriter
from src.assembled_core.intel.news_newsapi_fetcher import NewsAPIFetcher


# ---------------------------------------------------------------------------
# crisis_alpha_worker (Step 8.117)
# ---------------------------------------------------------------------------

def test_crisis_state_config_creates():
    cfg = CrisisStateConfig()
    assert isinstance(cfg, CrisisStateConfig)


def test_crisis_state_config_watch_threshold():
    cfg = CrisisStateConfig()
    assert cfg.geo_score_watch_threshold > 0


def test_crisis_state_config_active_threshold():
    cfg = CrisisStateConfig()
    assert cfg.geo_score_active_threshold >= cfg.geo_score_watch_threshold


def test_crisis_state_config_has_risk_posture():
    cfg = CrisisStateConfig()
    posture = cfg.risk_posture_by_state
    assert isinstance(posture, dict)
    assert len(posture) > 0


# ---------------------------------------------------------------------------
# evidence_grade_writer (Step 8.118)
# ---------------------------------------------------------------------------

def test_evidence_grade_writer_creates(tmp_path):
    egw = EvidenceGradeWriter(output_dir=tmp_path)
    assert isinstance(egw, EvidenceGradeWriter)


def test_evidence_grade_writer_output_dir(tmp_path):
    from pathlib import Path
    egw = EvidenceGradeWriter(output_dir=tmp_path)
    assert egw._dir == Path(tmp_path)


def test_evidence_grade_writer_write(tmp_path):
    egw = EvidenceGradeWriter(output_dir=tmp_path)
    path = egw.write(run_id="test_run_001", grade="A")
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["run_id"] == "test_run_001"
    assert data["evidence_grade"] == "A"


def test_evidence_grade_writer_creates_dir(tmp_path):
    egw = EvidenceGradeWriter(output_dir=tmp_path / "new_subdir")
    egw.write(run_id="run_x", grade="B")
    assert (tmp_path / "new_subdir").exists()


# ---------------------------------------------------------------------------
# news_newsapi_fetcher (Step 8.119)
# ---------------------------------------------------------------------------

def test_newsapi_fetcher_creates():
    naf = NewsAPIFetcher()
    assert isinstance(naf, NewsAPIFetcher)


def test_newsapi_fetcher_disabled_without_key():
    naf = NewsAPIFetcher(api_key="")
    assert naf.enabled is False


def test_newsapi_fetcher_fetch_returns_list_when_disabled():
    naf = NewsAPIFetcher(api_key="")
    result = naf.fetch("Apple")
    assert isinstance(result, list)
    assert len(result) == 0
