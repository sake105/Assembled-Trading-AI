"""Leakage-Frame-Assembly fuer das ``check_leakage``-QA-Gate (Gate 8).

STRUKTURVERSCHIEBUNG (2026-08-16, Audit-Plan 3.1):
  Alter Ort:  scripts/run_backtest_strategy.py::_build_leakage_frame
  Neuer Ort:  src/assembled_core/qa/leakage_frame.py
  Grund:      Der Orchestrator (src/) braucht dieselbe Logik — ein Import aus
              scripts/ waere eine Schichtverletzung (Rule 50), eine Kopie eine
              zweite Wahrheit. Call-Sites: scripts/run_backtest_strategy.py
              (bestehend) + tests/test_run_backtest_pit_wiring.py.
              Der Orchestrator-Anschluss (Audit-Plan 3.1) ist GEPLANT, aber
              NOCH NICHT gebaut — pipeline/ ist permissions-deny-geschuetzt;
              der Anschluss braucht einen freigegebenen Deny-Lift (M4,
              Stage-1-Review 2026-08-16: Plan != Implementierung).
  Folge:      Keine — Funktionskoerper unveraendert uebernommen (nur der
              blanket-except beim Parquet-Read wurde auf konkrete
              Fehlerklassen verengt, M2).

Warum das Earnings-Event-Frame und nicht das Feature-Panel: das Panel waere
der naheliegende Kandidat, ist aber unbrauchbar, weil
``altdata_news_macro_factors.py`` ``disclosure_date`` beim Merge verwirft.
Das Event-Frame ist das einzige Produktionsartefakt, das Ereignis- und
Offenlegungszeit nebeneinander traegt — genau was ein PIT-Leakage-Check
braucht.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

#: Spalten, die als Feature-Spalte fuer den PIT-Check in Frage kommen.
LEAKAGE_FEATURE_CANDIDATES = ("eps_surprise_pct", "eps_actual")


def build_leakage_frame(
    output_base: Path,
) -> tuple[pd.DataFrame | None, str | None, str]:
    """Assemble the frame for the ``check_leakage`` QA gate.

    Returns ``(frame, feature_col, reason)``. ``frame is None`` means the gate
    stays SKIPPED — which reads as "NOT checked", never as "clean" (E-066).
    ``reason`` always explains the outcome so the skip is visible rather than
    silent (E-142: a signal nobody can read is the same as no signal).

    This is deliberately conservative: any missing column yields ``None``
    rather than a guess. A wrong column name would make the gate BLOCK, and a
    BLOCK writes the QA block flag that stops the paper pilot until an
    operator acknowledges it (see the HALT warning in ``evaluate_all_gates``).
    Failing to *check* is recoverable; halting the pilot on a bookkeeping
    mistake is not.
    """
    earn_path = output_base / "events_earnings.parquet"
    if not earn_path.exists():
        return None, None, f"no leakage frame: {earn_path} does not exist"

    # Enger Except statt blanket (broad-except-Ratchet, M2 Stage-1 2026-08-16):
    # das sind die realistischen Lesefehler eines Parquet-Files. Jeder heisst
    # SKIP, nie BLOCK — ein unerwarteter Defekt darf dagegen laut scheitern.
    try:
        import pyarrow.lib as _pa_lib

        _arrow_error: type[Exception] = _pa_lib.ArrowException
    except ImportError:  # pragma: no cover - pyarrow ist Kerndependency
        _arrow_error = OSError
    try:
        frame = pd.read_parquet(earn_path)
    except (OSError, ValueError, KeyError, _arrow_error) as exc:
        return None, None, f"no leakage frame: could not read {earn_path}: {exc}"

    if frame.empty:
        return None, None, f"no leakage frame: {earn_path} is empty"

    for col in ("timestamp", "disclosure_date"):
        if col not in frame.columns:
            return None, None, f"no leakage frame: {earn_path} lacks column {col!r}"

    feature_col = next(
        (c for c in LEAKAGE_FEATURE_CANDIDATES if c in frame.columns), None
    )
    if feature_col is None:
        return (
            None,
            None,
            f"no leakage frame: none of {LEAKAGE_FEATURE_CANDIDATES} in {earn_path}",
        )

    return (
        frame,
        feature_col,
        f"leakage gate armed on {earn_path.name}: "
        f"feature={feature_col}, rows={len(frame)}",
    )
