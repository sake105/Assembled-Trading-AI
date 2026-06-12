"""Tests for the XBRL-fed PEAD/SUE seam in multifactor_v2._compute_pead_sue_factor
and the SHADOW (weight 0) invariant.

The previously broken Finnhub `batch_sue(earnings_df, symbols)` call is repointed
to the free SEC-XBRL path (`load_fundamentals_xbrl` -> `latest_sue_from_xbrl`).
The factor must COMPUTE (observable / logged) but its weight is held at 0 in both
DEFAULT_V2_WEIGHTS and every regime, pending a full OOS PEAD backtest.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.assembled_core.data import altdata_loader
from src.assembled_core.strategies import multifactor_v2 as mf
from tests.test_pead_sue_xbrl import _frame, _two_year_rows


def test_compute_pead_sue_factor_uses_xbrl_seam(monkeypatch):
    # Fake the loader to return a known 8-quarter ACME tall frame; the real
    # latest_sue_from_xbrl then computes the SUE end-to-end.
    frame = _frame(_two_year_rows())
    monkeypatch.setattr(
        altdata_loader, "load_fundamentals_xbrl", lambda symbols, as_of, **kw: frame
    )
    latest = pd.DataFrame({"symbol": ["ACME", "OTHER"]})
    out = mf._compute_pead_sue_factor(
        ["ACME", "OTHER"], latest, as_of=pd.Timestamp("2024-06-30")
    )
    assert "pead_sue_score" in out
    s = out["pead_sue_score"]
    assert list(s.index) == ["ACME", "OTHER"]
    # ACME has >=6 quarters -> a real, non-null, non-zero SUE
    assert pd.notna(s.loc["ACME"]) and s.loc["ACME"] != 0.0
    # OTHER has no XBRL data -> neutral 0.0 (not NaN)
    assert s.loc["OTHER"] == 0.0


def test_compute_pead_sue_factor_degrades_to_empty_on_no_data(monkeypatch):
    monkeypatch.setattr(
        altdata_loader,
        "load_fundamentals_xbrl",
        lambda symbols, as_of, **kw: pd.DataFrame(),
    )
    out = mf._compute_pead_sue_factor(["ACME"], pd.DataFrame({"symbol": ["ACME"]}))
    # Empty XBRL frame -> factor returns no series (graceful), never raises.
    assert out.get("pead_sue_score") is None or out["pead_sue_score"].empty


def test_pead_sue_weight_is_shadow_zero_in_default():
    assert mf.DEFAULT_V2_WEIGHTS["pead_sue_score"] == 0.0


def test_pead_sue_weight_is_shadow_zero_in_all_regimes():
    cfg = json.loads(
        Path("configs/factor_weights_by_regime.json").read_text(encoding="utf-8")
    )
    for regime in ("bull", "sideways", "bear", "crisis"):
        assert cfg[regime]["pead_sue_score"] == 0.0
