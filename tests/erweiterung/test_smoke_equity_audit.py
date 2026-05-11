"""Smoke-Tests: Equity-Curve-Anomaly-Audit gegen alle erzeugten Equity-CSVs.

Diese Tests laufen jede erzeugte Erweiterungs-Equity durch das Audit-Modul
und schlagen fehl, wenn kritische Flags gesetzt werden:

- EXTREMELY_HIGH_SHARPE: Sharpe > 5.0
- RETURNS_LIKELY_SMOOTHED: Lag-1 Autokorrelation > 0.4
- MDD_TOO_LOW_FOR_SHARPE: Sharpe > 2.0 und |MDD| < 5 %

Verhindert, dass die Erweiterung selbst Sharpe-4.6-Anomalien produziert
(wie sie im Mainline-System gefunden wurden, siehe EQUITY_AUDIT_FINDINGS.md).

Wird in CI automatisch ausgeführt nach `pytest tests/erweiterung`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from erweiterung.qa.equity_curve_audit import audit_equity_curve

CRITICAL_FLAGS = {
    "EXTREMELY_HIGH_SHARPE",
    "RETURNS_LIKELY_SMOOTHED",
    "MDD_TOO_LOW_FOR_SHARPE",
}


def _all_erweiterung_equities() -> list[tuple[str, Path]]:
    out_dir = Path("output")
    if not out_dir.exists():
        return []
    files = sorted(out_dir.glob("erweiterung_*equity*.csv"))
    return [(p.name, p) for p in files]


@pytest.mark.parametrize(
    "name,path", _all_erweiterung_equities() or [("none", Path("/dev/null"))]
)
def test_no_critical_audit_flags(name: str, path: Path):
    """Jede Erweiterungs-Equity-Curve darf keine kritischen Anomalie-Flags zeigen."""
    if name == "none":
        pytest.skip("No erweiterung equity files to audit")
    df = pd.read_csv(path)
    if df.empty or len(df.columns) < 2:
        pytest.skip(f"{name}: empty or single-column file")

    # Heuristic: first column = date, others = strategies
    first = df.columns[0]
    df[first] = pd.to_datetime(df[first], utc=True, errors="coerce")
    df = df.dropna(subset=[first]).set_index(first)

    numeric_cols = df.select_dtypes(include="number").columns
    if numeric_cols.empty:
        pytest.skip(f"{name}: no numeric columns")

    # Audit nur Spalten die echte Equity-Curves oder Returns sind.
    # Skip Multiplier/Score/Weight/State-Spalten (sind keine Equities).
    skip_patterns = ("multiplier", "score", "weight", "regime", "state",
                     "signal", "exposure", "leverage", "stress", "trigger",
                     "ensemble", "composite")
    audit_cols = [
        c for c in numeric_cols
        if not any(p in c.lower() for p in skip_patterns)
    ]
    if not audit_cols:
        pytest.skip(f"{name}: no auditable equity/return columns")
    # Audit jede numerische Spalte
    for col in audit_cols:
        eq_series = df[col].dropna()
        if len(eq_series) < 100:
            continue
        # Distinguish equity curves vs return series:
        # equity-like > 0, monotonic-ish growth; return-like centered at 0
        if eq_series.min() < 0:
            # Returns -> convert to equity
            eq_series = (1 + eq_series.fillna(0)).cumprod()
        if eq_series.iloc[-1] <= 0 or eq_series.std() == 0:
            continue
        audit = audit_equity_curve(eq_series, name=f"{name}::{col}")
        critical = set(audit.flags) & CRITICAL_FLAGS
        if critical:
            # Fail with detail
            raise AssertionError(
                f"{name}::{col} has critical flags {critical}. "
                f"Full flags: {audit.flags}. Sharpe={audit.overall_sharpe:.3f} "
                f"MDD={audit.max_drawdown:.3f}"
            )


def test_master_pipeline_equity_passes_audit():
    """Spezifischer Audit: Master-Pipeline Production-Equity darf keine Flags haben."""
    p = Path("output/erweiterung_master_pipeline_equity.csv")
    if not p.exists():
        pytest.skip("Master pipeline equity not generated yet")
    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df.iloc[:, 0], utc=True)
    df = df.set_index("date")
    if "master_equity" not in df.columns:
        pytest.skip("master_equity column missing")
    audit = audit_equity_curve(df["master_equity"].dropna(), name="master_pipeline")
    critical = set(audit.flags) & CRITICAL_FLAGS
    assert not critical, f"Master pipeline equity has critical flags: {audit.flags}"
