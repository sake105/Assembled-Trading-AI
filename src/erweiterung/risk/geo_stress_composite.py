"""Composite Geo-Stress-Score — kombiniert GPR + GDELT zu einem Daily-Stress-Signal.

Idee
----
GPR (Caldara-Iacoviello, monthly) ist slow & broad (Newspaper-Coverage).
GDELT (event counts, conflict-CAMEO-Share, mean Goldstein-Score, monthly samples)
ist event-driven & tone-aware.

Zusammen geben sie ein robusteres Bild als jede Quelle allein:
- GPR-Spike + GDELT-Conflict-Share spike  →  reale Eskalation (z. B. Ukraine 2022)
- GPR-Spike alleine                       →  vage Bedrohung (z. B. Inauguration-Rhetorik)
- GDELT-Spike alleine                     →  isolierte Newsflut ohne breite Wahrnehmung

Composite-Definition
--------------------
score = w_gpr · z(GPR) + w_conflict · z(GDELT conflict_share) + w_tone · (-z(mean_tone))

(Tone ist negativ-orientiert: niedrigerer Tone = mehr Stress, daher Vorzeichenwechsel.)

State-Mapping (kompatibel zu GPROverlayPolicy)
----------------------------------------------
- PAUSE     (multiplier 0.50): composite > 2.0 (extremer Stress)
- ACTIVE    (multiplier 0.75): composite > 1.0
- WATCH     (multiplier 1.00): -1.0 <= composite <= 1.0
- COOLDOWN  (multiplier 1.05): composite < -1.0 (Post-Stress-Relief)

PR-Pfad
-------
Modul ist eigenständig — keine Mainline-Imports. Komplementiert das
bestehende ``src/assembled_core/risk/georisk_overlay.py`` durch externe
GPR+GDELT-Datenfundierung.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class GeoStressPolicy:
    """Composite-Policy."""

    enabled: bool = True
    # Gewichte für Composite-Score
    w_gpr: float = 0.50
    w_conflict: float = 0.30
    w_tone: float = 0.20
    # State-Thresholds
    pause_z: float = 2.0
    active_z: float = 1.0
    cooldown_z: float = -1.0
    # State-Multipliers
    state_multipliers: dict[str, float] = field(
        default_factory=lambda: {
            "PAUSE": 0.50,
            "ACTIVE": 0.75,
            "WATCH": 1.00,
            "COOLDOWN": 1.05,
        }
    )
    # Glättung gegen Whipsaw
    smoothing_days: int = 5
    # Z-Score-Lookback (in observed monthly samples ~= years)
    z_lookback_months: int = 36


def _load_gpr_monthly(
    cache_path: str = "data/cache/gpr/sheet1.parquet",
) -> pd.DataFrame:
    """Lade GPR-monthly als [date, gpr] long-format."""
    p = Path(cache_path)
    if not p.exists():
        raise FileNotFoundError(f"GPR cache not found at {cache_path}")
    df = pd.read_parquet(p)
    # Excel hat Multi-Row-Schema: nur Zeilen mit echtem GPR-Wert nehmen
    df = df[["date", "GPR"]].dropna(subset=["GPR"]).copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
    return df.rename(columns={"GPR": "gpr"})


def _load_gdelt_monthly(
    cache_path: str = "data/cache/gdelt/monthly_aggregates.parquet",
    biweekly_path: str | None = None,
    weekly_path: str | None = None,
) -> pd.DataFrame:
    """Lade GDELT-aggregates (monthly default, biweekly/weekly opt-in).

    Default: nur monthly (157 samples 2013-2026).
    Grid-Search-Befund (run_gdelt_resolution_grid.py):
    - Monthly-only:  AnnRet +16.40%, Sharpe 1.439, p(>0)=0.915 vs baseline
    - +Biweekly:     AnnRet +16.14%, Sharpe 1.415, p(>0)=0.746
    - +Biweekly+Weekly: AnnRet +15.72%, Sharpe 1.385, p(>0)=0.571
    Mehr Granularität bringt Noise statt Signal — geopolitische Risiken
    bewegen sich in monthly-cycles. Daher monthly als Default.

    Höhere Resolutionen via Parameter explizit aktivierbar (für Research).
    """
    p = Path(cache_path)
    if not p.exists():
        raise FileNotFoundError(f"GDELT cache not found at {cache_path}")
    df = pd.read_parquet(p)

    for extra_path in (biweekly_path, weekly_path):
        if extra_path is not None and Path(extra_path).exists():
            extra = pd.read_parquet(extra_path)
            df = pd.concat([df, extra], ignore_index=True).drop_duplicates(
                subset="sample_date", keep="last"
            )

    df["date"] = pd.to_datetime(df["sample_date"], format="%Y%m%d", utc=True)
    keep = ["date", "conflict_share", "mean_tone", "mean_goldstein", "n_events"]
    return df[keep].sort_values("date").reset_index(drop=True)


def _rolling_zscore(s: pd.Series, lookback: int) -> pd.Series:
    """Rolling z-score mit minimum periods. NaN-safe."""
    mu = s.rolling(lookback, min_periods=max(6, lookback // 4)).mean()
    sd = s.rolling(lookback, min_periods=max(6, lookback // 4)).std()
    sd = sd.replace(0.0, np.nan)
    return (s - mu) / sd


def compute_monthly_composite(
    policy: GeoStressPolicy | None = None,
    gpr_cache: str = "data/cache/gpr/sheet1.parquet",
    gdelt_cache: str = "data/cache/gdelt/monthly_aggregates.parquet",
) -> pd.DataFrame:
    """Berechne monthly Composite-Geo-Stress-Score.

    Returns
    -------
    DataFrame mit Spalten:
    - date
    - gpr, conflict_share, mean_tone (raw)
    - z_gpr, z_conflict, z_tone_inv (z-scores)
    - composite_z (gewichteter Composite)
    - state (PAUSE/ACTIVE/WATCH/COOLDOWN)
    """
    if policy is None:
        policy = GeoStressPolicy()

    gpr = _load_gpr_monthly(gpr_cache)
    gdelt = _load_gdelt_monthly(gdelt_cache)

    # Aggregate GDELT to monthly (mean) when multiple samples per month exist (biweekly)
    # tz_convert(None) drops UTC for Period conversion (info loss is intentional).
    gpr["yyyymm"] = gpr["date"].dt.tz_convert(None).dt.to_period("M")
    gdelt["yyyymm"] = gdelt["date"].dt.tz_convert(None).dt.to_period("M")
    gdelt_m = gdelt.groupby("yyyymm", as_index=False).agg(
        conflict_share=("conflict_share", "mean"),
        mean_tone=("mean_tone", "mean"),
        mean_goldstein=("mean_goldstein", "mean"),
        n_events=("n_events", "mean"),
    )
    merged = gpr.merge(gdelt_m, on="yyyymm", how="inner")
    merged = merged.drop(columns="yyyymm").sort_values("date").reset_index(drop=True)

    lb = policy.z_lookback_months
    merged["z_gpr"] = _rolling_zscore(merged["gpr"], lb)
    merged["z_conflict"] = _rolling_zscore(merged["conflict_share"], lb)
    # Tone inverted (low tone = high stress)
    merged["z_tone_inv"] = -_rolling_zscore(merged["mean_tone"], lb)

    merged["composite_z"] = (
        policy.w_gpr * merged["z_gpr"].fillna(0.0)
        + policy.w_conflict * merged["z_conflict"].fillna(0.0)
        + policy.w_tone * merged["z_tone_inv"].fillna(0.0)
    )

    def _state(z: float) -> str:
        if not np.isfinite(z):
            return "WATCH"
        if z > policy.pause_z:
            return "PAUSE"
        if z > policy.active_z:
            return "ACTIVE"
        if z < policy.cooldown_z:
            return "COOLDOWN"
        return "WATCH"

    merged["state"] = merged["composite_z"].apply(_state)
    return merged


def expand_composite_to_daily(
    monthly_df: pd.DataFrame,
    daily_index: pd.DatetimeIndex,
    policy: GeoStressPolicy | None = None,
) -> pd.DataFrame:
    """Forward-fill monthly composite auf daily index + Glättung."""
    if policy is None:
        policy = GeoStressPolicy()
    if monthly_df.empty:
        return pd.DataFrame(
            {"composite_z": 0.0, "state": "WATCH", "multiplier": 1.0},
            index=daily_index,
        )

    if monthly_df["date"].dt.tz is None:
        monthly_df = monthly_df.copy()
        monthly_df["date"] = monthly_df["date"].dt.tz_localize("UTC")

    m = monthly_df.set_index("date")[["composite_z", "state"]].sort_index()
    if daily_index.tz is None:
        daily_index = daily_index.tz_localize("UTC")

    daily = m.reindex(daily_index, method="ffill")
    # Smoothing on composite_z
    daily["composite_z_smoothed"] = (
        daily["composite_z"].rolling(policy.smoothing_days, min_periods=1).mean()
    )

    def _state(z: float) -> str:
        if not np.isfinite(z):
            return "WATCH"
        if z > policy.pause_z:
            return "PAUSE"
        if z > policy.active_z:
            return "ACTIVE"
        if z < policy.cooldown_z:
            return "COOLDOWN"
        return "WATCH"

    daily["state"] = daily["composite_z_smoothed"].apply(_state)
    daily["multiplier"] = daily["state"].map(policy.state_multipliers).fillna(1.0)
    return daily


def apply_geo_stress_overlay(
    returns: pd.Series,
    policy: GeoStressPolicy | None = None,
    gpr_cache: str = "data/cache/gpr/sheet1.parquet",
    gdelt_cache: str = "data/cache/gdelt/monthly_aggregates.parquet",
) -> dict:
    """Wendet Composite-Overlay auf eine Return-Serie an.

    Returns
    -------
    dict mit:
    - hedged_return: pd.Series (return × multiplier)
    - exposure_multiplier: pd.Series
    - state_series: pd.Series
    - composite_z: pd.Series
    - policy: GeoStressPolicy
    """
    if policy is None:
        policy = GeoStressPolicy()
    if not policy.enabled:
        return {
            "hedged_return": returns,
            "exposure_multiplier": pd.Series(1.0, index=returns.index),
            "state_series": pd.Series("WATCH", index=returns.index),
            "composite_z": pd.Series(0.0, index=returns.index),
            "policy": policy,
        }

    monthly = compute_monthly_composite(policy, gpr_cache, gdelt_cache)
    daily = expand_composite_to_daily(monthly, returns.index, policy)

    multiplier = daily["multiplier"].reindex(returns.index).fillna(1.0)
    hedged = returns * multiplier

    return {
        "hedged_return": hedged,
        "exposure_multiplier": multiplier,
        "state_series": daily["state"].reindex(returns.index).fillna("WATCH"),
        "composite_z": daily["composite_z_smoothed"].reindex(returns.index).fillna(0.0),
        "policy": policy,
    }
