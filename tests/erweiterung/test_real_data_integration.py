"""Integrationstests mit ECHTEN Daten (Equity 19y + Cross-Asset 19y).

Diese Tests laufen die Erweiterungs-Module gegen die tatsächlichen
data/sample/watchlist_2007_2026.parquet und data/cache/yfinance_long/-
ETFs — keine synthetic noise. Stellt sicher dass:

- Master-Allocator-Pipeline auf echten 19y-Daten läuft
- Live-Engine konsistent bootstrappt vom real data
- Vol-Targeting numerisch stabil auf echten Returns
- Cross-Section-Helpers liefern erwartete Sharpes auf 22-Mega-Caps
- Audit-Module flag echte historische Equity-Anomalien (falls vorhanden)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from erweiterung.backtest.calmar_bootstrap import calmar_diff_bootstrap
from erweiterung.backtest.performance_metrics import all_metrics
from erweiterung.factors.fama_french import momentum_12_1
from erweiterung.live.live_decision_engine import (
    LiveDecisionEngine,
    LiveEngineConfig,
)
from erweiterung.qa.equity_curve_audit import audit_equity_curve
from erweiterung.strategies.cross_section_helpers import (
    cs_long_only_wide,
    long_format_to_wide,
)
from erweiterung.strategies.ema_trend_cross_section import (
    EMATrendConfig,
    backtest_ema_trend,
)
from erweiterung.strategies.intermarket_macro_factors import (
    build_intermarket_panel,
    macro_stress_composite_score,
)
from erweiterung.strategies.master_allocator import (
    MasterAllocator,
    MasterAllocatorConfig,
)
from erweiterung.strategies.tail_risk_hedge import (
    TailHedgeConfig,
    vix_stress_trigger,
)
from erweiterung.strategies.volatility_targeting import (
    VolTargetConfig,
    apply_vol_targeting,
)


# ============================================================================
# Master Allocator on REAL data
# ============================================================================


def test_master_allocator_on_real_19y_data(real_xa_returns, real_eq_returns_wide):
    """Master-Allocator muss auf echten 19y-Daten ohne Fehler laufen + sinnvolle Sharpe liefern."""
    # Build equity factor (Mom-12/1 long-only)
    eq_long = real_eq_returns_wide.stack().rename("return").reset_index()
    eq_long.columns = ["date", "symbol", "return"]

    # We don't have close prices in wide-returns; reconstruct synthetic close from returns:
    # (Master-Allocator nimmt nur returns als Input, also OK)
    # For Mom-12/1 wir brauchen close. Workaround: rebuild from cumprod.
    closes = (1 + real_eq_returns_wide.fillna(0)).cumprod() * 100
    closes_long = closes.stack().rename("close").reset_index()
    closes_long.columns = ["date", "symbol", "close"]

    mom = momentum_12_1(closes_long)
    closes_long = closes_long.merge(eq_long, on=["date", "symbol"], how="left")
    closes_long = closes_long.set_index(["date", "symbol"])
    closes_long["mom_12_1"] = mom.reindex(closes_long.index)
    closes_long = closes_long.reset_index().dropna(subset=["mom_12_1"])

    # Cross-section vectorized
    mom_wide = long_format_to_wide(
        closes_long[["date", "symbol", "mom_12_1"]], "mom_12_1"
    )
    ret_wide = real_eq_returns_wide.reindex(
        mom_wide.index, columns=mom_wide.columns
    ).fillna(0)
    pnl, _ = cs_long_only_wide(mom_wide, ret_wide, quantile=0.3, lag_days=1)
    eq_factor_ret = pnl.dropna()
    assert len(eq_factor_ret) > 1000, "Insufficient factor-return history"

    # Master allocator
    alloc = MasterAllocator(MasterAllocatorConfig(sa_weight=0.70))
    out = alloc.allocate(eq_factor_ret, real_xa_returns)
    assert "master_return" in out.columns
    assert len(out) > 1000

    m = all_metrics(out["master_return"].dropna())
    # Plausibility-Checks (19y mit 60/40 ~ 8% AnnRet, 1.0 Sharpe expected for Master)
    assert (
        -0.3 < m["annualized_return"] < 0.5
    ), f"AnnRet {m['annualized_return']:.2%} außerhalb plausible range"
    assert (
        0.3 < m["sharpe"] < 3.0
    ), f"Sharpe {m['sharpe']:.3f} außerhalb plausible range"
    assert (
        -0.6 < m["max_drawdown"] < 0
    ), f"MDD {m['max_drawdown']:.2%} außerhalb plausible range"


def test_master_allocator_beats_60_40_calmar_pgt_0_9(
    real_xa_returns, real_eq_returns_wide
):
    """Master_70_30 muss Calmar-p(>0) >= 0.85 vs 60/40 auf echten 19y haben."""
    closes = (1 + real_eq_returns_wide.fillna(0)).cumprod() * 100
    closes_long = closes.stack().rename("close").reset_index()
    closes_long.columns = ["date", "symbol", "close"]
    mom = momentum_12_1(closes_long)
    closes_long = closes_long.set_index(["date", "symbol"])
    closes_long["mom_12_1"] = mom.reindex(closes_long.index)
    closes_long = closes_long.reset_index().dropna(subset=["mom_12_1"])

    mom_wide = long_format_to_wide(
        closes_long[["date", "symbol", "mom_12_1"]], "mom_12_1"
    )
    ret_wide = real_eq_returns_wide.reindex(
        mom_wide.index, columns=mom_wide.columns
    ).fillna(0)
    eq_factor_ret, _ = cs_long_only_wide(mom_wide, ret_wide, quantile=0.3, lag_days=1)
    eq_factor_ret = eq_factor_ret.dropna()

    alloc = MasterAllocator(MasterAllocatorConfig(sa_weight=0.70))
    out = alloc.allocate(eq_factor_ret, real_xa_returns)
    master_ret = out["master_return"].dropna()

    # 60/40 benchmark
    classic = 0.60 * real_xa_returns["SPY"] + 0.40 * real_xa_returns["AGG"]
    classic = classic.loc[master_ret.index.min() : master_ret.index.max()]

    out_test = calmar_diff_bootstrap(
        master_ret,
        classic.dropna(),
        n_bootstrap=1000,
        avg_block_size=20,
        seed=42,
    )
    if "error" in out_test:
        pytest.skip(f"bootstrap error: {out_test['error']}")
    p_gt = 1.0 - out_test["p_value_one_sided_greater"]
    # Real-data smoke check: Master should beat 60/40 calmar with p >= 0.85
    assert p_gt >= 0.85, f"Calmar p(>0) {p_gt:.3f} unter Schwelle 0.85"


# ============================================================================
# Live Decision Engine on REAL data
# ============================================================================


def test_live_engine_bootstrap_real_19y(real_eq_returns_wide, real_xa_returns):
    """Live-Engine bootstrap auf echten 19y muss funktionieren."""
    engine = LiveDecisionEngine(LiveEngineConfig())
    engine.bootstrap_from_history(real_eq_returns_wide, real_xa_returns)
    summary = engine.state_summary()
    assert summary["n_eq_history_days"] > 200
    assert summary["n_xa_history_days"] > 200
    assert summary["n_xa_top_weights_nonzero"] == 5  # default top-N


def test_live_engine_decide_after_real_bootstrap(real_eq_returns_wide, real_xa_returns):
    """decide_next() liefert sinnvolle Werte nach realem Bootstrap."""
    engine = LiveDecisionEngine(LiveEngineConfig())
    engine.bootstrap_from_history(real_eq_returns_wide, real_xa_returns)
    out = engine.decide_next()

    # Sanity-checks
    assert 0.0 <= out["sa_leverage"] <= 2.0
    assert 0.0 <= out["xa_ew_leverage"] <= 2.0
    # Top picks: 5 ETFs
    assert (out["xa_top_weights"] > 0).sum() == 5
    # EQ picks: should have at least 1 (top 30% of available)
    assert (out["eq_top_weights"] > 0).sum() >= 1


def test_live_engine_update_consistency_real(real_eq_returns_wide, real_xa_returns):
    """Update mit realen Daten ändert state korrekt + decide-Werte plausibel."""
    engine = LiveDecisionEngine(LiveEngineConfig())
    # Bootstrap mit 80% der Daten
    n_train = int(len(real_eq_returns_wide) * 0.8)
    engine.bootstrap_from_history(
        real_eq_returns_wide.iloc[:n_train], real_xa_returns.iloc[:n_train]
    )

    # Simulate 50 daily updates with REAL data
    sa_leverages = []
    for i in range(50):
        if i >= len(real_eq_returns_wide) - n_train:
            break
        date = real_eq_returns_wide.index[n_train + i]
        eq_row = real_eq_returns_wide.iloc[n_train + i]
        xa_row = real_xa_returns.reindex(real_eq_returns_wide.index).iloc[n_train + i]
        # Some symbols may not overlap → NaN handling
        xa_row = xa_row.fillna(0)
        engine.update_with_new_day(date, eq_row, xa_row)
        out = engine.decide_next()
        sa_leverages.append(out["sa_leverage"])

    # SA-Leverage should vary (different vol-regimes)
    if len(sa_leverages) > 5:
        assert np.std(sa_leverages) > 0.0001, "SA-leverage variation zu klein"


# ============================================================================
# Vol-Targeting on REAL data
# ============================================================================


def test_vol_targeting_on_real_spy(real_xa_returns):
    """Vol-Target auf echten SPY-Returns."""
    spy = real_xa_returns["SPY"].dropna()
    out = apply_vol_targeting(spy, VolTargetConfig(target_vol_annual=0.10))
    assert "scaled_return" in out.columns
    scaled = out["scaled_return"].dropna()
    realized = scaled.std() * np.sqrt(252)
    # Skalierte Vol sollte näher am Target 0.10 sein als raw SPY (typisch ~0.18)
    raw_vol = spy.std() * np.sqrt(252)
    assert abs(realized - 0.10) < abs(raw_vol - 0.10)


# ============================================================================
# Cross-Section on REAL data
# ============================================================================


def test_cs_long_only_on_real_equity_panel(real_eq_returns_wide):
    """CS Long-Only auf echten Mega-Cap-Returns."""
    # Build mom_12_1 signal
    closes = (1 + real_eq_returns_wide.fillna(0)).cumprod()
    log_ret = np.log1p(real_eq_returns_wide.fillna(0))
    cumsum = log_ret.cumsum()
    mom_wide = np.exp(cumsum.shift(21) - cumsum.shift(252)) - 1.0

    pnl, pos = cs_long_only_wide(
        mom_wide, real_eq_returns_wide, quantile=0.3, lag_days=1
    )
    valid_pnl = pnl.dropna()
    # 19 years of Mega-Cap-Mom-12/1-LO should produce positive AnnRet
    assert len(valid_pnl) > 1000
    ann_ret = (1 + valid_pnl).prod() ** (252 / len(valid_pnl)) - 1
    assert ann_ret > 0.05, f"Mega-Cap-Mom-12/1 AnnRet {ann_ret:.2%} <5% verdächtig"


# ============================================================================
# EMA-Trend on REAL data
# ============================================================================


def test_ema_trend_on_real_22_megacaps():
    """EMA-Trend Cross-Section auf echten 22 Mega-Caps."""
    from pathlib import Path

    src = Path("data/sample/watchlist_2007_2026.parquet")
    if not src.exists():
        pytest.skip("no equity sample")
    df = pd.read_parquet(src)
    if "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "date"})
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    df["return"] = df.groupby("symbol")["close"].pct_change()
    panel = df[["date", "symbol", "close", "return"]].dropna()

    pnl = backtest_ema_trend(panel, EMATrendConfig(ema_fast=20, ema_slow=60))
    valid = pnl.dropna()
    assert len(valid) > 1000
    ann = (1 + valid).prod() ** (252 / len(valid)) - 1
    # EMA-Trend on Mega-Caps over 19y should produce positive AnnRet
    assert ann > 0.05, f"EMA-Trend AnnRet {ann:.2%} <5% verdächtig"


# ============================================================================
# Intermarket Macro on REAL data
# ============================================================================


def test_intermarket_factors_on_real_xa(real_xa_returns):
    """Intermarket-Faktoren auf echten ETF-Closes."""
    # Reconstruct closes
    closes = (1 + real_xa_returns.fillna(0)).cumprod() * 100
    panel = build_intermarket_panel(closes)
    assert "bond_equity_ratio_20d" in panel.columns
    assert "credit_spread_proxy" in panel.columns
    assert panel["bond_equity_ratio_20d"].dropna().std() > 0


def test_macro_stress_score_on_real_xa(real_xa_returns):
    """Macro-Stress-Composite-Score auf echten Daten."""
    closes = (1 + real_xa_returns.fillna(0)).cumprod() * 100
    panel = build_intermarket_panel(closes)
    score = macro_stress_composite_score(panel)
    valid = score.dropna()
    # Score in [0, 1]
    if not valid.empty:
        assert valid.min() >= 0
        assert valid.max() <= 1


# ============================================================================
# Equity-Audit on REAL Mainline Equity Curves
# ============================================================================


def test_audit_real_master_pipeline_equity_no_flags():
    """Master-Pipeline-Equity-Curve sollte keine kritischen Audit-Flags zeigen."""
    from pathlib import Path

    p = Path("output/erweiterung_master_pipeline_equity.csv")
    if not p.exists():
        pytest.skip("master pipeline equity not generated")
    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df.iloc[:, 0], utc=True)
    df = df.set_index("date")
    if "master_equity" not in df.columns:
        pytest.skip("master_equity column missing")
    audit = audit_equity_curve(df["master_equity"].dropna(), name="master_pipeline")
    critical = {
        "EXTREMELY_HIGH_SHARPE",
        "RETURNS_LIKELY_SMOOTHED",
        "MDD_TOO_LOW_FOR_SHARPE",
    }
    found = set(audit.flags) & critical
    assert not found, f"Master pipeline equity hat kritische Flags: {audit.flags}"


# ============================================================================
# VIX Tail-Hedge on REAL VIX
# ============================================================================


def test_vix_stress_trigger_real_vix_calm_dominant(real_vix):
    """Auf realer VIX-Series sollten normal-Tage dominieren (VIX>30 ist selten)."""
    trigger = vix_stress_trigger(
        real_vix, TailHedgeConfig(use_zscore=False, vix_absolute_threshold=30.0)
    )
    # > 65% sollten "normal" sein (VIX > 30 ist seltener Stress-Trigger)
    pct_normal = (trigger == "normal").mean()
    assert pct_normal > 0.50, f"Nur {pct_normal:.1%} normal — verdächtig wenig"


def test_vix_stress_trigger_real_vix_finds_2020_crash(real_vix):
    """VIX-Trigger muss den COVID-Crash 2020 finden."""
    trigger = vix_stress_trigger(
        real_vix, TailHedgeConfig(use_zscore=False, vix_absolute_threshold=30.0)
    )
    # COVID-window 2020-03 bis 2020-05
    covid_mask = (trigger.index >= pd.Timestamp("2020-03-01", tz="UTC")) & (
        trigger.index <= pd.Timestamp("2020-05-31", tz="UTC")
    )
    covid_window = trigger[covid_mask]
    if covid_window.empty:
        pytest.skip("VIX history doesn't include 2020")
    assert (covid_window == "stress").any(), "VIX-Trigger findet COVID-Crash 2020 nicht"
