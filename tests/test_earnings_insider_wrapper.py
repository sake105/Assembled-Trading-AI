"""Tests for B2.3 earnings surprise + insider activity factors.

Locks PIT safety, decay math, window boundaries, safe-divide and clipping.
Synthetic fixtures only — no external data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.features.earnings_insider_wrapper import (  # noqa: E402
    compute_earnings_insider_factors,
)


pytestmark = pytest.mark.phase12


def _empty_earnings() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["symbol", "filing_date", "eps_actual", "eps_estimate"]
    )


def _empty_insider() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["symbol", "filing_date", "transaction_type", "value_usd"]
    )


# ---------------------------------------------------------------------------
# PIT safety — the hard non-negotiable.
# ---------------------------------------------------------------------------


def test_pit_gate_drops_future_earnings_filings() -> None:
    """A filing on 2026-06-01 must NOT influence the factor computed
    as-of 2026-05-31. This is the core PIT invariant."""
    as_of = pd.Timestamp("2026-05-31")
    earnings = pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "filing_date": pd.Timestamp("2026-06-01"),
                "eps_actual": 2.0,
                "eps_estimate": 1.0,
            },
            {
                "symbol": "MSFT",
                "filing_date": pd.Timestamp("2026-05-15"),
                "eps_actual": 3.0,
                "eps_estimate": 2.5,
            },
        ]
    )
    out = compute_earnings_insider_factors(
        as_of, ["AAPL", "MSFT"], earnings, _empty_insider()
    )
    # AAPL filing is in the future → NaN. MSFT visible → not NaN (though
    # z-scoring across only one valid observation also returns NaN).
    assert pd.isna(out.loc["AAPL", "earnings_surprise_z"])


def test_pit_gate_drops_future_insider_filings() -> None:
    as_of = pd.Timestamp("2026-05-31")
    insider = pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "filing_date": pd.Timestamp("2026-06-15"),
                "transaction_type": "P",
                "value_usd": 5_000_000,
            },
            {
                "symbol": "MSFT",
                "filing_date": pd.Timestamp("2026-05-10"),
                "transaction_type": "P",
                "value_usd": 3_000_000,
            },
        ]
    )
    out = compute_earnings_insider_factors(
        as_of, ["AAPL", "MSFT"], _empty_earnings(), insider
    )
    assert pd.isna(out.loc["AAPL", "insider_activity_score"])


# ---------------------------------------------------------------------------
# Earnings surprise mechanics.
# ---------------------------------------------------------------------------


def test_earnings_surprise_directionality() -> None:
    """Positive surprise → higher z than negative surprise."""
    as_of = pd.Timestamp("2026-05-31")
    earnings = pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "filing_date": pd.Timestamp("2026-05-01"),
                "eps_actual": 2.5,
                "eps_estimate": 2.0,  # +25%
            },
            {
                "symbol": "MSFT",
                "filing_date": pd.Timestamp("2026-05-01"),
                "eps_actual": 1.0,
                "eps_estimate": 2.0,  # -50%
            },
            {
                "symbol": "NVDA",
                "filing_date": pd.Timestamp("2026-05-01"),
                "eps_actual": 2.0,
                "eps_estimate": 2.0,  # 0%
            },
        ]
    )
    out = compute_earnings_insider_factors(
        as_of, ["AAPL", "MSFT", "NVDA"], earnings, _empty_insider()
    )
    z = out["earnings_surprise_z"]
    assert z["AAPL"] > z["NVDA"] > z["MSFT"]


def test_earnings_decay_partial() -> None:
    """Filing 100 days old (10 past the 90-day mark) gets scaled by
    approximately (1 - 10/30) = 0.667."""
    as_of = pd.Timestamp("2026-05-31")
    filing = as_of - pd.Timedelta(days=100)
    earnings = pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "filing_date": filing,
                "eps_actual": 2.0,
                "eps_estimate": 1.0,  # +100% raw
            },
            # Sidekick so we can z-score something
            {
                "symbol": "MSFT",
                "filing_date": as_of - pd.Timedelta(days=10),
                "eps_actual": 1.0,
                "eps_estimate": 1.0,
            },
        ]
    )
    # We assert the decay by reading through the private raw helper since the
    # public API returns z-scored values. Compute the expected raw AAPL value:
    # surprise = 1.0, scale = 1 - 10/30 = 0.6667 → raw ≈ 0.6667.
    from src.assembled_core.features.earnings_insider_wrapper import (
        _earnings_surprise_raw,
    )

    raw = _earnings_surprise_raw(as_of, ["AAPL", "MSFT"], earnings)
    assert raw["AAPL"] == pytest.approx(2.0 / 3.0, abs=1e-6)


def test_earnings_decay_to_zero() -> None:
    """A filing 125 days old → factor = 0 (before z-scoring)."""
    as_of = pd.Timestamp("2026-05-31")
    filing = as_of - pd.Timedelta(days=125)
    earnings = pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "filing_date": filing,
                "eps_actual": 10.0,
                "eps_estimate": 1.0,  # massive raw surprise
            }
        ]
    )
    from src.assembled_core.features.earnings_insider_wrapper import (
        _earnings_surprise_raw,
    )

    raw = _earnings_surprise_raw(as_of, ["AAPL"], earnings)
    assert raw["AAPL"] == 0.0


def test_earnings_safe_divide_zero_estimate() -> None:
    """eps_estimate = 0.0 → symbol gets NaN, not inf."""
    as_of = pd.Timestamp("2026-05-31")
    earnings = pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "filing_date": pd.Timestamp("2026-05-01"),
                "eps_actual": 0.5,
                "eps_estimate": 0.0,
            },
            {
                "symbol": "MSFT",
                "filing_date": pd.Timestamp("2026-05-01"),
                "eps_actual": 2.0,
                "eps_estimate": 1.0,
            },
        ]
    )
    out = compute_earnings_insider_factors(
        as_of, ["AAPL", "MSFT"], earnings, _empty_insider()
    )
    assert pd.isna(out.loc["AAPL", "earnings_surprise_z"])
    assert not np.isinf(out.loc["AAPL", "earnings_surprise_z"])


# ---------------------------------------------------------------------------
# Insider activity mechanics.
# ---------------------------------------------------------------------------


def test_insider_window_boundary_60_days() -> None:
    """A purchase 30 days ago is counted; 70 days ago is not."""
    as_of = pd.Timestamp("2026-05-31")
    insider = pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "filing_date": as_of - pd.Timedelta(days=30),
                "transaction_type": "P",
                "value_usd": 1_000_000,
            },
            {
                "symbol": "MSFT",
                "filing_date": as_of - pd.Timedelta(days=70),
                "transaction_type": "P",
                "value_usd": 1_000_000,
            },
        ]
    )
    from src.assembled_core.features.earnings_insider_wrapper import (
        _insider_activity_raw,
    )

    raw = _insider_activity_raw(as_of, ["AAPL", "MSFT"], insider, None)
    assert not pd.isna(raw["AAPL"])
    assert pd.isna(raw["MSFT"])  # dropped by window


def test_insider_signing_purchase_vs_sale() -> None:
    """Purchases add, sales subtract. A matched pair with same USD value
    on the same symbol nets to zero."""
    as_of = pd.Timestamp("2026-05-31")
    recent = as_of - pd.Timedelta(days=10)
    insider = pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "filing_date": recent,
                "transaction_type": "P",
                "value_usd": 2_000_000,
            },
            {
                "symbol": "AAPL",
                "filing_date": recent,
                "transaction_type": "S",
                "value_usd": 2_000_000,
            },
            {
                "symbol": "MSFT",
                "filing_date": recent,
                "transaction_type": "P",
                "value_usd": 1_000_000,
            },
        ]
    )
    from src.assembled_core.features.earnings_insider_wrapper import (
        _insider_activity_raw,
    )

    raw = _insider_activity_raw(as_of, ["AAPL", "MSFT"], insider, None)
    assert raw["AAPL"] == pytest.approx(0.0, abs=1e-12)
    assert raw["MSFT"] > 0


def test_insider_purchases_positive_sales_negative_zscore() -> None:
    as_of = pd.Timestamp("2026-05-31")
    recent = as_of - pd.Timedelta(days=5)
    insider = pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "filing_date": recent,
                "transaction_type": "P",
                "value_usd": 5_000_000,
            },
            {
                "symbol": "MSFT",
                "filing_date": recent,
                "transaction_type": "S",
                "value_usd": 5_000_000,
            },
            {
                "symbol": "NVDA",
                "filing_date": recent,
                "transaction_type": "P",
                "value_usd": 100_000,
            },
        ]
    )
    out = compute_earnings_insider_factors(
        as_of, ["AAPL", "MSFT", "NVDA"], _empty_earnings(), insider
    )
    z = out["insider_activity_score"]
    assert z["AAPL"] > z["NVDA"] > z["MSFT"]


def test_insider_market_cap_normalization() -> None:
    """With market_cap_df provided, raw is divided by cap."""
    as_of = pd.Timestamp("2026-05-31")
    recent = as_of - pd.Timedelta(days=5)
    insider = pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "filing_date": recent,
                "transaction_type": "P",
                "value_usd": 10_000_000,
            },
            {
                "symbol": "MSFT",
                "filing_date": recent,
                "transaction_type": "P",
                "value_usd": 10_000_000,
            },
        ]
    )
    mcap = pd.DataFrame(
        [
            {"symbol": "AAPL", "market_cap": 1_000_000_000},  # small cap
            {"symbol": "MSFT", "market_cap": 10_000_000_000},  # big cap
        ]
    )
    out = compute_earnings_insider_factors(
        as_of, ["AAPL", "MSFT"], _empty_earnings(), insider, market_cap_df=mcap
    )
    # AAPL has same raw flow but smaller cap → bigger normalized flow → higher z.
    assert out.loc["AAPL", "insider_activity_score"] > out.loc[
        "MSFT", "insider_activity_score"
    ]


def test_insider_missing_market_cap_falls_back() -> None:
    """With no market_cap_df, the function must still work."""
    as_of = pd.Timestamp("2026-05-31")
    recent = as_of - pd.Timedelta(days=5)
    insider = pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "filing_date": recent,
                "transaction_type": "P",
                "value_usd": 5_000_000,
            },
            {
                "symbol": "MSFT",
                "filing_date": recent,
                "transaction_type": "S",
                "value_usd": 3_000_000,
            },
        ]
    )
    out = compute_earnings_insider_factors(
        as_of, ["AAPL", "MSFT"], _empty_earnings(), insider, market_cap_df=None
    )
    assert not out["insider_activity_score"].isna().all()


# ---------------------------------------------------------------------------
# Clipping and cross-section behaviour.
# ---------------------------------------------------------------------------


def test_clipping_bounds_plus_minus_three() -> None:
    """Extreme outlier z-score gets clipped to +/- 3.0."""
    as_of = pd.Timestamp("2026-05-31")
    filing = as_of - pd.Timedelta(days=5)
    # Construct a section where one symbol is a huge outlier. With n=12,
    # max achievable |z| ≈ sqrt(11) ≈ 3.316 → clipped to 3.0.
    rows = []
    balanced = [f"SYM{i}" for i in range(11)]
    for sym in balanced:
        rows.append(
            {
                "symbol": sym,
                "filing_date": filing,
                "eps_actual": 1.01,
                "eps_estimate": 1.0,  # ~1% surprise
            }
        )
    rows.append(
        {
            "symbol": "OUT",
            "filing_date": filing,
            "eps_actual": 1001.0,
            "eps_estimate": 1.0,  # 1000x surprise
        }
    )
    earnings = pd.DataFrame(rows)
    syms = balanced + ["OUT"]
    out = compute_earnings_insider_factors(as_of, syms, earnings, _empty_insider())
    assert out.loc["OUT", "earnings_surprise_z"] == pytest.approx(3.0, abs=1e-9)


def test_single_valid_observation_returns_nan() -> None:
    """One valid observation can't be z-scored → NaN (documented behaviour)."""
    as_of = pd.Timestamp("2026-05-31")
    earnings = pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "filing_date": pd.Timestamp("2026-05-01"),
                "eps_actual": 2.0,
                "eps_estimate": 1.0,
            }
        ]
    )
    out = compute_earnings_insider_factors(
        as_of, ["AAPL"], earnings, _empty_insider()
    )
    assert pd.isna(out.loc["AAPL", "earnings_surprise_z"])


# ---------------------------------------------------------------------------
# Validation.
# ---------------------------------------------------------------------------


def test_missing_required_column_in_earnings_raises() -> None:
    as_of = pd.Timestamp("2026-05-31")
    earnings = pd.DataFrame([{"symbol": "AAPL", "filing_date": as_of}])
    with pytest.raises(ValueError, match="earnings_df"):
        compute_earnings_insider_factors(
            as_of, ["AAPL"], earnings, _empty_insider()
        )


def test_missing_required_column_in_insider_raises() -> None:
    as_of = pd.Timestamp("2026-05-31")
    insider = pd.DataFrame([{"symbol": "AAPL", "filing_date": as_of}])
    with pytest.raises(ValueError, match="insider_df"):
        compute_earnings_insider_factors(
            as_of, ["AAPL"], _empty_earnings(), insider
        )


def test_non_timestamp_as_of_raises() -> None:
    with pytest.raises(ValueError, match="as_of_date"):
        compute_earnings_insider_factors(
            "2026-05-31", ["AAPL"], _empty_earnings(), _empty_insider()
        )
