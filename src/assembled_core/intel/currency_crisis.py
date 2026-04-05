"""Currency crisis modeling for geopolitical and macro risk analysis.

Models sovereign currency vulnerabilities, peg stress, carry trade unwinds,
and currency contagion risk.
"""

from __future__ import annotations

import logging

from .models import CurrencyProfile

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Currency Profile Database
# ---------------------------------------------------------------------------

CURRENCY_PROFILES: dict[str, CurrencyProfile] = {
    # --- Stable reserve currencies ---
    "USD": CurrencyProfile(
        currency="USD", nation="US",
        reserve_months_import=1.5,    # Low because USD IS the reserve currency
        real_interest_rate=2.0,       # Fed funds ~5.25% - inflation ~3.3%
        inflation_rate=3.3,
        current_account_gdp_pct=-3.8,
        peg_type="free_float",
        dollarization_pct=0.0,
        crisis_probability_12m=0.01,
    ),
    "EUR": CurrencyProfile(
        currency="EUR", nation="EUROZONE",
        reserve_months_import=3.0,
        real_interest_rate=1.2,       # ECB ~4.5% - inflation ~3.3%
        inflation_rate=3.3,
        current_account_gdp_pct=2.5,
        peg_type="free_float",
        dollarization_pct=0.0,
        crisis_probability_12m=0.02,
    ),
    "CHF": CurrencyProfile(
        currency="CHF", nation="SWITZERLAND",
        reserve_months_import=8.0,
        real_interest_rate=0.5,
        inflation_rate=1.5,
        current_account_gdp_pct=9.0,
        peg_type="free_float",        # SNB actively manages but no formal peg
        dollarization_pct=0.0,
        crisis_probability_12m=0.005,
    ),
    "JPY": CurrencyProfile(
        currency="JPY", nation="JAPAN",
        reserve_months_import=20.0,  # Massive reserves
        real_interest_rate=-2.0,     # BOJ at 0.1% - inflation ~2.9%
        inflation_rate=2.9,
        current_account_gdp_pct=3.5,
        peg_type="managed_float",
        dollarization_pct=0.0,
        crisis_probability_12m=0.03,  # Weak yen pressure but not crisis
    ),
    "GBP": CurrencyProfile(
        currency="GBP", nation="UK",
        reserve_months_import=2.5,
        real_interest_rate=2.5,       # BOE ~5.25%
        inflation_rate=4.0,
        current_account_gdp_pct=-3.5,
        peg_type="free_float",
        dollarization_pct=0.0,
        crisis_probability_12m=0.05,  # Post-Brexit structural vulnerabilities
    ),
    "CNY": CurrencyProfile(
        currency="CNY", nation="CHINA",
        reserve_months_import=16.0,
        real_interest_rate=0.5,       # PBOC 3.45% - low inflation
        inflation_rate=1.5,
        current_account_gdp_pct=1.5,
        peg_type="managed_band",      # ±2% daily band vs basket
        dollarization_pct=0.05,
        crisis_probability_12m=0.04,  # Property sector risk
    ),
    # --- Vulnerable / high-risk currencies ---
    "TRY": CurrencyProfile(
        currency="TRY", nation="TURKEY",
        reserve_months_import=3.5,
        real_interest_rate=-10.0,     # Nominal 40%+ but high inflation
        inflation_rate=65.0,
        current_account_gdp_pct=-5.5,
        peg_type="managed_float",
        dollarization_pct=0.55,       # 55% of deposits in FX
        crisis_probability_12m=0.35,
    ),
    "BRL": CurrencyProfile(
        currency="BRL", nation="BRAZIL",
        reserve_months_import=18.0,
        real_interest_rate=6.5,       # SELIC ~10.75% - inflation ~4.3%
        inflation_rate=4.3,
        current_account_gdp_pct=-2.5,
        peg_type="free_float",
        dollarization_pct=0.10,
        crisis_probability_12m=0.12,
    ),
    "INR": CurrencyProfile(
        currency="INR", nation="INDIA",
        reserve_months_import=10.0,
        real_interest_rate=2.0,       # RBI 6.5% - inflation ~4.5%
        inflation_rate=4.5,
        current_account_gdp_pct=-1.5,
        peg_type="managed_float",
        dollarization_pct=0.05,
        crisis_probability_12m=0.05,
    ),
    "RUB": CurrencyProfile(
        currency="RUB", nation="RUSSIA",
        reserve_months_import=12.0,   # But frozen/restricted reserves
        real_interest_rate=-5.0,      # CBR 16% - inflation 21%
        inflation_rate=21.0,
        current_account_gdp_pct=5.0,
        peg_type="managed_float",     # Post-sanctions capital controls
        dollarization_pct=0.30,
        crisis_probability_12m=0.25,
    ),
    "ARS": CurrencyProfile(
        currency="ARS", nation="ARGENTINA",
        reserve_months_import=1.0,
        real_interest_rate=-80.0,     # Deeply negative real rates
        inflation_rate=160.0,
        current_account_gdp_pct=-2.0,
        peg_type="managed_band",      # Multiple exchange rates
        dollarization_pct=0.75,       # Unofficial dollarization
        crisis_probability_12m=0.70,
    ),
    "ZAR": CurrencyProfile(
        currency="ZAR", nation="SOUTH_AFRICA",
        reserve_months_import=6.0,
        real_interest_rate=3.5,
        inflation_rate=5.5,
        current_account_gdp_pct=-1.5,
        peg_type="free_float",
        dollarization_pct=0.05,
        crisis_probability_12m=0.15,
    ),
}

# Contagion map: crisis in currency A affects currency B
_CONTAGION_MATRIX: dict[str, dict[str, float]] = {
    "TRY": {"ZAR": 0.40, "BRL": 0.30, "INR": 0.20, "ARS": 0.25},
    "ARS": {"BRL": 0.50, "TRY": 0.20, "ZAR": 0.15},
    "RUB": {"TRY": 0.30, "BRL": 0.10, "INR": 0.10},
    "CNY": {"INR": 0.30, "TRY": 0.20, "BRL": 0.20, "ZAR": 0.20},
    "JPY": {"USD": 0.15, "EUR": 0.10},  # Carry unwind affects all
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_currency_profile(currency: str) -> CurrencyProfile | None:
    """Return a currency profile by code."""
    return CURRENCY_PROFILES.get(currency.upper())


def compute_currency_stress_score(profile: CurrencyProfile) -> float:
    """Compute a currency stress score from 0 (stable) to 1 (crisis imminent).

    Weights:
    - Inflation rate (30%): high inflation = stress
    - Negative real rate (25%): negative real rate = erosion of value
    - Low reserves (25%): <3 months = critical
    - High dollarization (20%): flight from domestic currency
    """
    # Normalize inflation: 0% = 0, 10%+ = 1.0
    inflation_score = min(profile.inflation_rate / 50.0, 1.0)

    # Negative real rate: 0 or positive = 0 stress, -20% = 1.0
    real_rate_score = min(max(-profile.real_interest_rate, 0) / 20.0, 1.0)

    # Reserves: 12+ months = 0, 0 = 1.0
    reserve_score = max(0, 1.0 - profile.reserve_months_import / 12.0)

    # Dollarization
    dollar_score = min(profile.dollarization_pct, 1.0)

    stress = (
        inflation_score * 0.30
        + real_rate_score * 0.25
        + reserve_score * 0.25
        + dollar_score * 0.20
    )
    return min(stress, 1.0)


def detect_peg_break_risk(profile: CurrencyProfile) -> float:
    """Estimate probability of a currency peg breaking.

    Only relevant for managed_band and hard_peg currencies.
    Returns 0.0 to 1.0.
    """
    if profile.peg_type == "free_float":
        return 0.0

    stress = compute_currency_stress_score(profile)

    if profile.peg_type in ("managed_band", "hard_peg"):
        # High stress + managed peg = peg break risk
        reserve_inadequacy = max(0, 1 - profile.reserve_months_import / 6.0)
        current_account_stress = max(0, -profile.current_account_gdp_pct / 10.0)
        peg_risk = (stress * 0.5 + reserve_inadequacy * 0.3 + current_account_stress * 0.2)
        return min(peg_risk, 1.0)

    # managed_float: lower risk
    return min(stress * 0.3, 0.5)


def estimate_carry_trade_unwind_impact(
    high_yield_currency: str,
    safe_currency: str = "JPY",
) -> dict[str, float]:
    """Estimate carry trade unwind impact between currency pair.

    When high-yield currencies weaken, carry trades unwind — selling
    high-yield and buying safe (JPY, CHF, USD). Returns impact estimates.
    """
    hy_profile = CURRENCY_PROFILES.get(high_yield_currency)
    safe_profile = CURRENCY_PROFILES.get(safe_currency)

    if hy_profile is None or safe_profile is None:
        return {}

    hy_stress = compute_currency_stress_score(hy_profile)
    interest_differential = hy_profile.real_interest_rate - safe_profile.real_interest_rate

    # Higher interest differential = more carry trade built up = larger unwind
    carry_buildup = min(max(interest_differential, 0) / 10.0, 1.0)
    unwind_pressure = hy_stress * carry_buildup

    return {
        "carry_pair": f"{high_yield_currency}/{safe_currency}",
        "unwind_pressure": unwind_pressure,
        "high_yield_currency_depreciation_est_pct": unwind_pressure * 15.0,
        "safe_currency_appreciation_est_pct": unwind_pressure * 8.0,
        "global_risk_off_intensity": unwind_pressure * 0.7,
        "equities_impact": -unwind_pressure * 0.5,  # Negative = equity selloff
        "gold_impact": unwind_pressure * 0.3,
    }


def compute_contagion_currencies(crisis_currency: str) -> list[tuple[str, float]]:
    """Return currencies that would be affected if crisis_currency breaks.

    Returns [(currency, contagion_score)] sorted descending.
    """
    contagion = _CONTAGION_MATRIX.get(crisis_currency.upper(), {})
    return sorted(contagion.items(), key=lambda x: x[1], reverse=True)


def rank_currencies_by_risk() -> list[tuple[str, float]]:
    """Rank all tracked currencies by stress score, descending."""
    ranked = []
    for currency, profile in CURRENCY_PROFILES.items():
        score = compute_currency_stress_score(profile)
        ranked.append((currency, score))
    return sorted(ranked, key=lambda x: x[1], reverse=True)
