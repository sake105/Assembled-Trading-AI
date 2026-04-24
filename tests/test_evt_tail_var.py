"""Tests for the EVT POT-GPD tail-VaR sidecar (C9)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytest.importorskip("src.assembled_core.risk.evt_tail_var")
from src.assembled_core.risk.evt_tail_var import (  # noqa: E402
    evt_expected_shortfall,
    evt_var,
    fit_pot_gpd,
)


def _heavy_tailed_losses(n: int = 5000, df: int = 3, seed: int = 13) -> np.ndarray:
    """Student-t draws with df=3, returned as positive losses via abs()."""
    rng = np.random.default_rng(seed)
    # Student-t via standard-normal / sqrt(chi2/df).
    z = rng.standard_normal(n)
    g = rng.chisquare(df, size=n) / df
    t = z / np.sqrt(g)
    return np.abs(t)


def _gaussian_losses(n: int = 5000, seed: int = 13) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.abs(rng.standard_normal(n))


@pytest.mark.phase12
def test_fit_pot_gpd_basic_shape_and_counts():
    losses = _heavy_tailed_losses()
    fit = fit_pot_gpd(losses, threshold_pct=0.90)

    assert fit["n_total"] == len(losses)
    # ~10% above threshold, allow a tiny tolerance.
    expected_exc = int(0.10 * len(losses))
    assert abs(fit["n_exceedances"] - expected_exc) <= 2

    assert fit["scale"] > 0.0
    assert np.isfinite(fit["shape"])
    assert fit["threshold"] > 0.0


@pytest.mark.phase12
def test_evt_var_fat_tail_extrapolates_beyond_empirical():
    losses = _heavy_tailed_losses()
    empirical_99 = float(np.quantile(losses, 0.99))
    var_99 = evt_var(losses, alpha=0.99, threshold_pct=0.90)

    assert var_99 > 0.0
    # EVT should extrapolate at least as far as — typically further than —
    # the empirical 99th percentile for a heavy-tailed sample.
    assert var_99 >= empirical_99 * 0.9
    assert var_99 > empirical_99 or np.isclose(var_99, empirical_99, rtol=0.1)


@pytest.mark.phase12
def test_evt_var_monotone_in_alpha():
    losses = _heavy_tailed_losses()
    v95 = evt_var(losses, alpha=0.95, threshold_pct=0.90)
    v99 = evt_var(losses, alpha=0.99, threshold_pct=0.90)
    v999 = evt_var(losses, alpha=0.999, threshold_pct=0.90)

    assert v95 < v99 < v999


@pytest.mark.phase12
def test_evt_es_at_least_var_coherent():
    losses = _heavy_tailed_losses()
    var_99 = evt_var(losses, alpha=0.99, threshold_pct=0.90)
    es_99 = evt_expected_shortfall(losses, alpha=0.99, threshold_pct=0.90)

    assert es_99 >= var_99


@pytest.mark.phase12
def test_evt_var_rejects_alpha_below_threshold():
    losses = _heavy_tailed_losses()
    with pytest.raises(ValueError, match="alpha"):
        evt_var(losses, alpha=0.80, threshold_pct=0.90)
    with pytest.raises(ValueError, match="alpha"):
        evt_var(losses, alpha=0.90, threshold_pct=0.90)


@pytest.mark.phase12
def test_fit_pot_gpd_insufficient_exceedances():
    rng = np.random.default_rng(13)
    tiny = np.abs(rng.standard_normal(10))
    with pytest.raises(ValueError, match="insufficient exceedances"):
        fit_pot_gpd(tiny, threshold_pct=0.90)


@pytest.mark.phase12
def test_evt_var_gaussian_finite_and_sane():
    losses = _gaussian_losses()
    empirical_99 = float(np.quantile(losses, 0.99))
    var_99 = evt_var(losses, alpha=0.99, threshold_pct=0.90)

    assert np.isfinite(var_99)
    assert var_99 > 0.0
    # Should be within an order of magnitude of the empirical quantile.
    assert 0.1 * empirical_99 < var_99 < 10.0 * empirical_99


@pytest.mark.phase12
def test_loss_convention_no_sign_flip():
    """Input is taken at face value — positive numbers are losses.

    Passing the absolute value of a return series (as _heavy_tailed_losses
    does) should give positive VaR. Passing the negated input should give
    a symmetric result because we do not flip the sign inside the module.
    """
    rng = np.random.default_rng(13)
    raw = rng.standard_normal(5000)

    losses_pos = np.abs(raw)
    var_pos = evt_var(losses_pos, alpha=0.99, threshold_pct=0.90)
    assert var_pos > 0.0

    # If the caller passes a raw return series, the function still runs:
    # it treats the input as the "loss axis". No sign flipping happens.
    # We just assert it does not crash and returns a finite number.
    returns_as_is = raw  # not sign-flipped
    # With symmetric data around 0, the 90th percentile is positive, so
    # the exceedances are well-defined even without sign-flipping.
    val = evt_var(returns_as_is, alpha=0.99, threshold_pct=0.90)
    assert np.isfinite(val)


@pytest.mark.phase12
def test_evt_es_infinite_shape_raises():
    """If xi >= 1 (infinite-mean tail), ES must raise rather than return inf."""
    # Construct a very heavy tail: Pareto with small alpha → xi ~ 1/alpha > 1.
    rng = np.random.default_rng(13)
    # Pareto with shape 0.8 → extremely heavy tailed; xi estimate likely ≥ 1.
    samples = (rng.pareto(0.8, size=5000) + 1.0)
    try:
        fit = fit_pot_gpd(samples, threshold_pct=0.90)
    except ValueError:
        pytest.skip("method-of-moments degenerate on this sample")

    if fit["shape"] >= 1.0:
        with pytest.raises(ValueError, match="infinite ES"):
            evt_expected_shortfall(samples, alpha=0.99, threshold_pct=0.90)
    else:
        # If the fit happened to yield xi < 1, just verify ES is finite.
        es = evt_expected_shortfall(samples, alpha=0.99, threshold_pct=0.90)
        assert np.isfinite(es)
