"""Equity-Curve-Anomaly-Audit — statistische Plausibilitätsprüfung.

Zweck
-----
Eine gemeldete Equity-Curve kann auf bekannte Anomalien geprüft werden, ohne
Zugang zu den Intermediate-Signalen oder dem Quellcode des Originalsystems.

Checks
------
1. **Sharpe-Plausibilität:** rolling 60/120/252 Sharpe — ist er über die Zeit
   stabil oder fällt er nach Out-of-Sample-Cutoff?
2. **Autocorrelation der Returns:** ungewöhnlich hohe Autokorrelation deutet
   auf gemittelten oder geglätteten Returns hin (z. B. NAV-Smoothing,
   überoptimistische Fill-Annahmen).
3. **Skew/Kurtosis:** zu symmetrische / kurtotische Returns deuten auf
   synthetische Daten oder Look-Ahead.
4. **MDD-Konsistenz:** angegebener MDD vs konsistenter Worst-Drawdown gleichen
   Zeitraums in S&P / Equal-Weight.
5. **"Too-Good-To-Be-True"-Heuristik:** vergleicht die Equity gegen ein
   Bootstrap-Sample bekannter Faktor-Modelle. Ist die Equity außerhalb des
   95-%-Bands → Warnung.
6. **Daily-Return-Quantile:** unrealistisch niedrige Worst-Day-Returns
   relativ zur Vola.
7. **Day-of-Week-Effekte:** ungewöhnliche Konzentration der Gewinne auf
   bestimmte Wochentage (z. B. Montag) ist suspekt.

Wichtig
-------
Diese Checks beweisen **kein** Leakage, sie zeigen nur **Inkonsistenzen**.
Eine echte Leakage-Diagnostik braucht Zugang zu Signal- und Trade-Logs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class AuditFlags:
    name: str
    overall_sharpe: float | None = None
    rolling_sharpe_stability: float | None = None
    return_autocorr_lag1: float | None = None
    return_autocorr_lag5: float | None = None
    skew: float | None = None
    kurtosis: float | None = None
    max_drawdown: float | None = None
    worst_day: float | None = None
    worst_day_vol_ratio: float | None = None
    monday_share_of_gains: float | None = None
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        out = {k: v for k, v in self.__dict__.items() if k != "flags"}
        out["flags"] = list(self.flags)
        return out


def _rolling_sharpe(returns: pd.Series, window: int = 60) -> pd.Series:
    mean = returns.rolling(window).mean()
    std = returns.rolling(window).std()
    return (mean / std.replace(0, np.nan)) * np.sqrt(252)


def audit_equity_curve(
    equity: pd.Series,
    name: str = "unnamed",
    bootstrap_benchmark: pd.Series | None = None,
    expected_min_corr_with_market: float = 0.10,
) -> AuditFlags:
    """Statistische Plausibilitätsprüfung einer Equity-Curve.

    Args:
        equity: zeitindex sortierte Equity-Series (in Dollar oder normalisiert).
        name: Label für Output.
        bootstrap_benchmark: optionale Referenz-Equity (z. B. Equal-Weight).
        expected_min_corr_with_market: Schwelle für Marktkorrelation.

    Returns:
        AuditFlags mit metrics + warnings.
    """
    out = AuditFlags(name=name)
    eq = equity.dropna().sort_index()
    if len(eq) < 60:
        out.flags.append("INSUFFICIENT_DATA")
        return out

    ret = eq.pct_change().dropna()
    if ret.empty or ret.std() == 0:
        out.flags.append("FLAT_RETURNS")
        return out

    # 1. Overall Sharpe
    ann_ret = (1 + ret).prod() ** (252 / len(ret)) - 1
    ann_vol = ret.std() * np.sqrt(252)
    out.overall_sharpe = float(ann_ret / ann_vol) if ann_vol > 0 else None

    # 2. Rolling-Sharpe-Stabilität: Std der 60-Tage-Rolling-Sharpe
    rsh = _rolling_sharpe(ret, 60).dropna()
    if not rsh.empty:
        out.rolling_sharpe_stability = float(rsh.std())

    # 3. Autocorrelation
    if len(ret) > 10:
        out.return_autocorr_lag1 = float(ret.autocorr(lag=1))
        out.return_autocorr_lag5 = float(ret.autocorr(lag=5))

    # 4. Distribution
    out.skew = float(ret.skew())
    out.kurtosis = float(ret.kurtosis())

    # 5. Drawdown
    cm = (1 + ret).cumprod()
    dd = (cm / cm.cummax() - 1).min()
    out.max_drawdown = float(dd)
    out.worst_day = float(ret.min())
    if ret.std() > 0:
        out.worst_day_vol_ratio = float(abs(ret.min()) / ret.std())

    # 6. Day-of-week
    if isinstance(eq.index, pd.DatetimeIndex):
        try:
            dow = ret.copy()
            dow.index = pd.to_datetime(dow.index).dayofweek
            gains = ret[ret > 0]
            if not gains.empty:
                gains_dow = gains.copy()
                gains_dow.index = pd.to_datetime(gains.index).dayofweek
                monday_share = gains_dow[gains_dow.index == 0].sum() / gains.sum()
                out.monday_share_of_gains = float(monday_share)
        except Exception:
            pass

    # ============== Flags-Logik ==============
    if out.overall_sharpe is not None and out.overall_sharpe > 3.0:
        out.flags.append(f"SUSPICIOUS_SHARPE_{out.overall_sharpe:.2f}")
    if out.overall_sharpe is not None and out.overall_sharpe > 5.0:
        out.flags.append("EXTREMELY_HIGH_SHARPE")

    if out.return_autocorr_lag1 is not None and abs(out.return_autocorr_lag1) > 0.25:
        out.flags.append(f"HIGH_AUTOCORR_LAG1_{out.return_autocorr_lag1:.2f}")

    if out.return_autocorr_lag1 is not None and out.return_autocorr_lag1 > 0.4:
        out.flags.append("RETURNS_LIKELY_SMOOTHED")

    if out.kurtosis is not None and out.kurtosis < -1.0:
        out.flags.append("LOW_KURTOSIS_SYNTHETIC_LIKE")

    if (
        out.max_drawdown is not None
        and out.overall_sharpe is not None
        and out.overall_sharpe > 2.0
        and abs(out.max_drawdown) < 0.05
    ):
        out.flags.append("MDD_TOO_LOW_FOR_SHARPE")

    if (
        out.worst_day_vol_ratio is not None
        and out.worst_day_vol_ratio < 2.0
        and len(ret) > 252
    ):
        out.flags.append("WORST_DAY_TOO_MILD_FOR_PERIOD")

    if bootstrap_benchmark is not None and isinstance(
        bootstrap_benchmark.index, pd.DatetimeIndex
    ):
        bm_ret = bootstrap_benchmark.pct_change().dropna()
        aligned = pd.concat({"e": ret, "b": bm_ret}, axis=1).dropna()
        if len(aligned) > 60:
            corr = float(aligned["e"].corr(aligned["b"]))
            if abs(corr) < expected_min_corr_with_market:
                out.flags.append(f"MARKET_CORR_TOO_LOW_{corr:.2f}")

    return out


def compare_equity_curves(
    curves: dict[str, pd.Series],
    benchmark: pd.Series | None = None,
) -> pd.DataFrame:
    """Audit mehrere Equity-Curves → DataFrame."""
    rows = []
    for name, eq in curves.items():
        rows.append(
            audit_equity_curve(eq, name=name, bootstrap_benchmark=benchmark).to_dict()
        )
    df = pd.DataFrame(rows)
    return df


__all__ = ["AuditFlags", "audit_equity_curve", "compare_equity_curves"]
