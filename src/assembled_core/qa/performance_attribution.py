"""Performance Attribution — Faktor-Contribution-Dekomposition.

Zerlegt realisierten Portfolio-Return in Beiträge einzelner Faktoren:

    r_portfolio = Σ (β_i × f_i) + α + ε

- β_i: Exposure zum Faktor (aus Regression portfolio_return vs factor_returns)
- f_i: Realisierter Faktor-Return im Zeitraum
- α: Idiosynkratischer Return (nach Faktoren-Abzug)
- ε: Residual (unerklärter Rest)

Anwendung:
- Diagnose: welche Faktoren erklären unseren Return?
- Kontrolle: ist α positiv (echter Edge) oder alles Faktor-Exposure?
- Risk-Management: welche Faktoren tragen zum Drawdown bei?

PIT-Invariante: Faktor-Returns müssen zur gleichen Zeit wie Portfolio-Returns
realisiert sein. Rolling-Regression nutzt nur historische Daten.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class AttributionResult:
    """Ergebnis einer Performance-Attribution-Analyse."""

    factor_betas: dict[str, float]
    """β pro Faktor aus multipler Regression."""

    factor_contributions: dict[str, float]
    """Beitrag zum Portfolio-Return: β × mean(factor_return)."""

    alpha: float
    """Mittelwert der Residuen (nach Faktor-Abzug)."""

    alpha_t_stat: float
    """t-Statistik für α (Null-Hypothese: α=0)."""

    r_squared: float
    """Fraktion erklärter Varianz."""

    total_return: float
    """Mittelwert des Portfolio-Returns (für Reference)."""

    residual_std: float
    n_obs: int = 0

    factor_contribution_pct: dict[str, float] = field(default_factory=dict)
    """Beitrag in Prozent des Gesamt-Returns."""

    def summary(self) -> dict:
        return {
            "total_return": round(self.total_return, 6),
            "alpha": round(self.alpha, 6),
            "alpha_t_stat": round(self.alpha_t_stat, 3),
            "r_squared": round(self.r_squared, 4),
            "factor_betas": {k: round(v, 4) for k, v in self.factor_betas.items()},
            "factor_contributions": {
                k: round(v, 6) for k, v in self.factor_contributions.items()
            },
            "factor_contribution_pct": {
                k: round(v, 2) for k, v in self.factor_contribution_pct.items()
            },
            "n_obs": self.n_obs,
        }

    def has_significant_alpha(self, t_threshold: float = 2.0) -> bool:
        """True wenn α statistisch signifikant positiv (Edge-Nachweis)."""
        return self.alpha > 0 and self.alpha_t_stat > t_threshold


def compute_attribution(
    portfolio_returns: pd.Series,
    factor_returns: pd.DataFrame,
    min_obs: int = 20,
) -> AttributionResult:
    """Führt Performance-Attribution via OLS-Regression durch.

    Args:
        portfolio_returns: Zeit-indexierte Series der Portfolio-Returns
        factor_returns: DataFrame mit Faktor-Returns (Spalten = Faktoren)
        min_obs: Minimum Beobachtungen

    Returns:
        AttributionResult

    Raises:
        ValueError bei zu wenigen Beobachtungen oder Dimensions-Mismatch.
    """
    aligned = pd.concat(
        [portfolio_returns.rename("_portfolio"), factor_returns], axis=1
    ).dropna()
    if len(aligned) < min_obs:
        raise ValueError(
            f"[Attribution] Nur {len(aligned)} Beobachtungen (min: {min_obs})"
        )

    y = aligned["_portfolio"].values
    X_cols = [c for c in aligned.columns if c != "_portfolio"]
    X = aligned[X_cols].values

    # OLS mit Intercept: y = α + β·X + ε
    X_with_const = np.column_stack([np.ones(len(X)), X])
    try:
        coef, residuals_sum, rank, _ = np.linalg.lstsq(X_with_const, y, rcond=None)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"[Attribution] OLS failed: {exc}")

    alpha = float(coef[0])
    betas = {name: float(coef[i + 1]) for i, name in enumerate(X_cols)}

    # Residuen + R²
    y_pred = X_with_const @ coef
    residuals = y - y_pred
    ss_res = float((residuals**2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

    # t-Statistik für α
    n = len(y)
    k = X.shape[1] + 1  # Params incl. intercept
    dof = max(1, n - k)
    sigma_sq = ss_res / dof
    try:
        xtx_inv = np.linalg.inv(X_with_const.T @ X_with_const)
        alpha_var = float(sigma_sq * xtx_inv[0, 0])
        alpha_se = np.sqrt(alpha_var) if alpha_var > 0 else 1e-9
        alpha_t = alpha / alpha_se
    except np.linalg.LinAlgError:
        alpha_t = 0.0

    # Contribution = β × mean(factor_return)
    factor_means = {name: float(aligned[name].mean()) for name in X_cols}
    contributions = {name: betas[name] * factor_means[name] for name in X_cols}

    total_ret = float(y.mean())

    # Contribution in %
    pct: dict[str, float] = {}
    if abs(total_ret) > 1e-12:
        for name, c in contributions.items():
            pct[name] = float(c / total_ret * 100.0)

    result = AttributionResult(
        factor_betas=betas,
        factor_contributions=contributions,
        alpha=alpha,
        alpha_t_stat=float(alpha_t),
        r_squared=float(r_squared),
        total_return=total_ret,
        residual_std=float(residuals.std()),
        n_obs=n,
        factor_contribution_pct=pct,
    )
    logger.info(
        "[Attribution] n=%d α=%.4f (t=%.2f) R²=%.3f",
        n,
        alpha,
        alpha_t,
        r_squared,
    )
    return result


def rolling_attribution(
    portfolio_returns: pd.Series,
    factor_returns: pd.DataFrame,
    window: int = 60,
    min_obs: int = 30,
) -> pd.DataFrame:
    """Rolling-Attribution-Analyse.

    Gibt zeitindizierten DataFrame zurück mit α und β pro Faktor über die Zeit.
    Sichtbar machen wann Faktor-Exposures driften.

    Args:
        portfolio_returns: zeit-indexierte Portfolio-Returns
        factor_returns: zeit-indexierte Faktor-Returns (Spalten = Faktoren)
        window: Rolling-Fenster in Perioden
        min_obs: Minimum Beobachtungen pro Fenster

    Returns:
        DataFrame mit Spalten: alpha, r_squared, beta_<factor>
    """
    aligned = pd.concat(
        [portfolio_returns.rename("_portfolio"), factor_returns], axis=1
    ).dropna()
    if len(aligned) < window:
        raise ValueError(f"Datenlänge {len(aligned)} < Fenster {window}")

    records = []
    for end in range(window, len(aligned) + 1):
        slice_df = aligned.iloc[end - window : end]
        if len(slice_df) < min_obs:
            continue
        try:
            res = compute_attribution(
                slice_df["_portfolio"],
                slice_df.drop(columns="_portfolio"),
                min_obs=min_obs,
            )
            rec = {
                "timestamp": slice_df.index[-1],
                "alpha": res.alpha,
                "alpha_t_stat": res.alpha_t_stat,
                "r_squared": res.r_squared,
            }
            for name, b in res.factor_betas.items():
                rec[f"beta_{name}"] = b
            records.append(rec)
        except Exception as exc:
            logger.debug(
                "[Attribution] Rolling window @%s failed: %s", slice_df.index[-1], exc
            )

    return pd.DataFrame(records).set_index("timestamp") if records else pd.DataFrame()


def sector_attribution(
    position_returns: pd.DataFrame,
    sector_map: dict[str, str],
) -> dict[str, dict]:
    """Sektor-Attribution: Return pro Sektor aggregieren.

    Args:
        position_returns: DataFrame (zeit × symbol) mit Einzel-Position-Returns
        sector_map: {symbol: sector}

    Returns:
        {sector: {"total_return", "n_positions", "contribution_pct"}}
    """
    col_sums = position_returns.sum()
    col_sec = col_sums.index.map(lambda c: sector_map.get(c, "UNKNOWN"))
    sector_totals = col_sums.groupby(col_sec).sum()
    sector_counts = col_sums.groupby(col_sec).count()
    total = float(sector_totals.sum())

    out = {}
    for sec in sector_totals.index:
        s = float(sector_totals[sec])
        out[sec] = {
            "total_return": s,
            "n_positions": int(sector_counts[sec]),
            "contribution_pct": (s / total * 100.0) if abs(total) > 1e-12 else 0.0,
        }
    return out


def attribution_during_worst_drawdown(
    portfolio_returns: pd.Series,
    factor_returns: pd.DataFrame,
) -> dict:
    """Wrapper (Round 7H): führt compute_attribution nur auf DD-Periode aus.

    Verwendet `drawdown_decomposition.decompose_drawdown()` für konsistente Logik.
    """
    try:
        from src.assembled_core.qa.drawdown_decomposition import decompose_drawdown
    except ImportError:
        return {"error": "drawdown_decomposition module fehlt"}
    try:
        report = decompose_drawdown(portfolio_returns, factor_returns)
        return report.summary()
    except Exception as exc:
        logger.warning("[AttrDD] error: %s", exc)
        return {"error": str(exc)}


__all__ = [
    "AttributionResult",
    "compute_attribution",
    "rolling_attribution",
    "sector_attribution",
    "attribution_during_worst_drawdown",
]
