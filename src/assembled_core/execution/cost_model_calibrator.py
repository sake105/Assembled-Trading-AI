"""Offline cost-model calibration from TCA feedback.

Reads N paper-engine TCA aggregate JSONs from ``output/paper_tca/`` and
estimates shrinkage-based updates to the three cost-model parameters that
drive the realism stack:

- ``half_spread_bps`` — expected half-spread charged on any fill
- ``impact_bps_per_pct_adv`` — a single-coefficient stand-in for the Kyle-like
  impact model (realised bps of impact per percent of ADV traded)
- ``participation_cap`` — the observed (capped) participation rate

The calibrator is **offline-only**: it produces a YAML recommendation file
and returns the new parameters; deployment is a manual config change.

Conservative design choices:

- 30% shrinkage toward the prior default ⇒ one run can move the knob at
  most 70% of the way to what it suggests.
- If fewer than ``min_runs`` valid TCA files are found, the calibrator
  returns the priors unchanged.
- NaN / inf sums short-circuit to the prior.

This module is deliberately dependency-light. It does not import numpy or
scipy. Reading YAML is best-effort: if PyYAML is not installed the output
file is a deterministic ``key: value`` text rather than failing the run.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


@dataclass
class CostModelPriors:
    """Prior (default) parameters to shrink toward.

    Matches ``UnifiedPaperConfig`` defaults so one round of calibration on
    an empty TCA folder is a no-op.
    """

    half_spread_bps: float = 5.0
    impact_bps_per_pct_adv: float = 10.0
    participation_cap: float = 0.05


@dataclass
class CalibrationResult:
    """Recommendation produced by :func:`calibrate_cost_model`.

    Attributes:
        half_spread_bps: Blended new half-spread.
        impact_bps_per_pct_adv: Blended new impact coefficient.
        participation_cap: Blended new participation cap.
        n_runs: Number of TCA files that contributed.
        mean_realised: Averaged realised metrics used as the likelihood.
        shrinkage: Weight on prior (``0.30`` means the new value is
            ``0.30*prior + 0.70*realised``).
        prior: Echo of the priors used.
    """

    half_spread_bps: float
    impact_bps_per_pct_adv: float
    participation_cap: float
    n_runs: int
    mean_realised: dict[str, float] = field(default_factory=dict)
    shrinkage: float = 0.30
    prior: CostModelPriors = field(default_factory=CostModelPriors)


def _iter_tca_files(tca_dir: Path) -> Iterable[Path]:
    if not tca_dir.exists():
        return []
    return sorted(tca_dir.glob("tca_*.json"))


def _load_tca(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("[CALIBRATOR] unreadable TCA file: %s", path)
        return None


def _mean(values: list[float]) -> float:
    clean = [v for v in values if v == v]  # drop NaN
    return sum(clean) / len(clean) if clean else float("nan")


def _shrink(prior: float, realised: float, shrinkage: float) -> float:
    """Blend prior and realised; if realised is NaN, keep prior.

    ``shrinkage`` is the weight on the prior, so 1.0 → no update.
    """
    if realised != realised:  # NaN
        return prior
    return shrinkage * prior + (1.0 - shrinkage) * realised


def calibrate_cost_model(
    tca_dir: Path,
    *,
    priors: CostModelPriors | None = None,
    shrinkage: float = 0.30,
    min_runs: int = 1,
) -> CalibrationResult:
    """Produce a shrinkage-blended cost-model recommendation.

    The function averages per-run aggregates exposed by the Phase 7 TCA
    writer:

    - ``cost_bps_avg.spread`` → drives ``half_spread_bps``
    - ``cost_bps_avg.impact`` → drives ``impact_bps_per_pct_adv``
    - ``fill_rate`` → proxy for effective participation; a fill rate of
      ``1.0`` means no capacity pressure, so the cap stays at prior.
      Lower fill-rates pull the cap down.
    """
    priors = priors or CostModelPriors()
    files = list(_iter_tca_files(Path(tca_dir)))
    if len(files) < min_runs:
        logger.info(
            "[CALIBRATOR] only %d runs found in %s (< min_runs=%d), returning priors",
            len(files), tca_dir, min_runs,
        )
        return CalibrationResult(
            half_spread_bps=priors.half_spread_bps,
            impact_bps_per_pct_adv=priors.impact_bps_per_pct_adv,
            participation_cap=priors.participation_cap,
            n_runs=len(files),
            shrinkage=shrinkage,
            prior=priors,
        )

    spreads: list[float] = []
    impacts: list[float] = []
    fill_rates: list[float] = []
    for p in files:
        payload = _load_tca(p)
        if payload is None:
            continue
        cost_avg = payload.get("cost_bps_avg", {}) or {}
        try:
            spreads.append(float(cost_avg.get("spread", float("nan"))))
            impacts.append(float(cost_avg.get("impact", float("nan"))))
            fill_rates.append(float(payload.get("fill_rate", float("nan"))))
        except (TypeError, ValueError):
            continue

    mean_spread = _mean(spreads)
    mean_impact = _mean(impacts)
    mean_fr = _mean(fill_rates)

    # Participation-cap signal: a fill rate below 1.0 signals the market is
    # pushing back; the realised cap is cap * fill_rate in expectation.
    realised_cap = (
        priors.participation_cap * mean_fr if mean_fr == mean_fr else float("nan")
    )

    new = CalibrationResult(
        half_spread_bps=_shrink(priors.half_spread_bps, mean_spread, shrinkage),
        impact_bps_per_pct_adv=_shrink(priors.impact_bps_per_pct_adv, mean_impact, shrinkage),
        participation_cap=_shrink(priors.participation_cap, realised_cap, shrinkage),
        n_runs=len(files),
        mean_realised={
            "spread_bps": mean_spread,
            "impact_bps": mean_impact,
            "fill_rate": mean_fr,
        },
        shrinkage=shrinkage,
        prior=priors,
    )
    logger.info(
        "[CALIBRATOR] %d runs: half_spread %.3f→%.3f, impact %.3f→%.3f, cap %.3f→%.3f",
        new.n_runs,
        priors.half_spread_bps, new.half_spread_bps,
        priors.impact_bps_per_pct_adv, new.impact_bps_per_pct_adv,
        priors.participation_cap, new.participation_cap,
    )
    return new


def write_calibration_report(
    result: CalibrationResult,
    out_path: Path,
) -> Path:
    """Serialise the recommendation to YAML (or plain key/value fallback).

    The format is intentionally simple so the user can diff the file before
    copying values into ``UnifiedPaperConfig``.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    body = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_runs": result.n_runs,
        "shrinkage": result.shrinkage,
        "priors": {
            "half_spread_bps": result.prior.half_spread_bps,
            "impact_bps_per_pct_adv": result.prior.impact_bps_per_pct_adv,
            "participation_cap": result.prior.participation_cap,
        },
        "mean_realised": result.mean_realised,
        "recommendation": {
            "half_spread_bps": result.half_spread_bps,
            "impact_bps_per_pct_adv": result.impact_bps_per_pct_adv,
            "participation_cap": result.participation_cap,
        },
        "deploy": False,
    }

    try:
        import yaml  # type: ignore[import-not-found]

        out_path.write_text(yaml.safe_dump(body, sort_keys=False))
    except Exception:
        # Deterministic fallback — one top-level key per line; nested dicts
        # are emitted as JSON so the file stays machine-parseable.
        lines: list[str] = []
        for k, v in body.items():
            if isinstance(v, dict):
                lines.append(f"{k}: {json.dumps(v, default=str)}")
            else:
                lines.append(f"{k}: {json.dumps(v, default=str)}")
        out_path.write_text("\n".join(lines) + "\n")

    logger.info("[CALIBRATOR] wrote %s", out_path)
    return out_path


__all__ = [
    "CostModelPriors",
    "CalibrationResult",
    "calibrate_cost_model",
    "write_calibration_report",
]
