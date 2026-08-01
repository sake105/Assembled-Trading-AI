"""Regression guards for the 2026-07-22 GESAMTBEWERTUNG P5 honesty fixes.

P5-certify — verify_certificate must FAIL when artefacts are missing on
    both sides. Previously "NOT_FOUND" == "NOT_FOUND" counted as a hash
    match, so a certificate over MISSING artefacts verified as PASS
    (tautological reproducibility).
P5-leakage — check_leakage stays fail-open by design for feature_df=None,
    but the skipped state must be visible via details["skipped"] so a
    caller can distinguish "checked clean" from "not checked".
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.fast


def test_p5_certificate_over_missing_artifacts_fails(tmp_path):
    from src.assembled_core.certify.generator import (
        generate_certificate,
        save_certificate,
        verify_certificate,
    )

    out_dir = tmp_path / "out"
    out_dir.mkdir()  # empty: no equity_curve/trades/signals artefacts
    cert = generate_certificate(output_dir=out_dir)
    cert_path = tmp_path / "certificate.json"
    save_certificate(cert, cert_path)

    results = verify_certificate(cert_path, out_dir)
    assert results == {"equity_curve": False, "trades": False, "signals": False}, (
        "missing-on-both-sides must never verify as reproduced"
    )


def test_p5_check_leakage_skip_state_is_visible():
    from src.assembled_core.qa.qa_gates import QAResult, check_leakage

    res = check_leakage(feature_df=None)
    # Fail-open by design (documented): SKIPPED does not affect the overall
    # verdict. Since 2026-08-01 it is its OWN state instead of OK, so that a
    # consumer counting gate_results cannot report it as passed (E-066).
    assert res.result == QAResult.SKIPPED
    assert res.details.get("skipped") is True  # visibly NOT-CHECKED
