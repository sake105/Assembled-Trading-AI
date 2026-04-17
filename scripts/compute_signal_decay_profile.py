"""D5/R4 — Offline signal-decay profile writer.

Wires ``qa/signal_decay.py`` (plan R4 — previously orphaned) into a periodic
offline CI that writes ``output/qa/signal_decay/latest.json``. That path is
the contract consumed by ``strategies/signal_decay_gate.py`` — see its
module docstring.

Execution model
---------------

* Reads a factor panel (CSV / parquet) with at least the columns the
  combiner cares about (timestamp, symbol, per-factor columns, and one of
  ``fwd_return_1m`` / ``fwd_return_5d`` / ``fwd_return_20d``).
* Runs ``analyze_all_signals`` for every requested factor.
* Writes ``{generated_at, universe, factors: {name: {ic_mean,
  ic_half_life_days, is_stale}}}`` to the target JSON.

If no real factor panel is available (e.g. first-time local run) we
deterministically synthesise one so the CI can prove the wiring works and
produce a non-empty artifact.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.assembled_core.qa.signal_decay import (  # noqa: E402
    SignalDecayProfile,
    analyze_all_signals,
)

logger = logging.getLogger("compute_signal_decay_profile")


DEFAULT_FACTOR_COLS = [
    "trend_ema_spread",
    "mom_rsi_centered",
    "mom_12_1",
    "low_vol_rank",
    "quality_score",
]


def _synthetic_factor_panel(
    n_days: int = 60,
    n_symbols: int = 30,
    factor_cols: list[str] | None = None,
    seed: int = 7,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    factor_cols = factor_cols or DEFAULT_FACTOR_COLS
    idx = pd.date_range("2024-01-02", periods=n_days, freq="B")
    symbols = [f"SYN{i:02d}" for i in range(n_symbols)]
    rows: list[dict[str, Any]] = []
    # Generate correlated factor + fwd-return structure so ICs are non-zero.
    ic_per_factor = {name: rng.uniform(0.02, 0.12) for name in factor_cols}
    for ts in idx:
        fwd = rng.normal(0.0, 0.02, size=n_symbols)
        row_base = {"timestamp": ts}
        for sym_i, sym in enumerate(symbols):
            row: dict[str, Any] = {**row_base, "symbol": sym, "fwd_return_1m": float(fwd[sym_i])}
            for name, ic in ic_per_factor.items():
                noise = rng.normal(0.0, 1.0)
                row[name] = ic * fwd[sym_i] + (1.0 - abs(ic)) * noise
            rows.append(row)
    return pd.DataFrame(rows)


def _profile_to_dict(profile: SignalDecayProfile) -> dict[str, Any]:
    return {
        "ic_mean": float(profile.ic_mean) if profile.ic_mean == profile.ic_mean else 0.0,
        "ic_ir": float(profile.ic_ir) if profile.ic_ir == profile.ic_ir else 0.0,
        "ic_half_life_days": (
            float(profile.ic_half_life_days)
            if profile.ic_half_life_days is not None
            else None
        ),
        "is_stale": bool(profile.is_stale),
    }


def build_report(
    factor_panel: pd.DataFrame,
    factor_cols: list[str],
    *,
    universe: str = "synthetic",
    forward_return_col: str = "fwd_return_1m",
) -> dict[str, Any]:
    profiles = analyze_all_signals(factor_panel, factor_cols, forward_return_col)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "universe": universe,
        "forward_return_col": forward_return_col,
        "factors": {p.factor_name: _profile_to_dict(p) for p in profiles},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panel",
        default=None,
        help="Optional CSV/Parquet factor panel. If omitted, a synthetic "
             "panel is generated so the wiring is always exercised.",
    )
    parser.add_argument(
        "--factors",
        default=",".join(DEFAULT_FACTOR_COLS),
        help="Comma-separated factor column names.",
    )
    parser.add_argument(
        "--forward-return-col",
        default="fwd_return_1m",
    )
    parser.add_argument(
        "--universe",
        default="synthetic",
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "output" / "qa" / "signal_decay" / "latest.json"),
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )

    factor_cols = [c.strip() for c in args.factors.split(",") if c.strip()]

    if args.panel:
        path = Path(args.panel)
        if path.suffix.lower() == ".parquet":
            panel = pd.read_parquet(path)
        else:
            panel = pd.read_csv(path, parse_dates=["timestamp"])
        logger.info("[signal_decay] loaded real panel: %s rows=%d", path, len(panel))
    else:
        panel = _synthetic_factor_panel(factor_cols=factor_cols)
        logger.info("[signal_decay] using synthetic panel rows=%d", len(panel))

    report = build_report(
        panel, factor_cols,
        universe=args.universe,
        forward_return_col=args.forward_return_col,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    n_stale = sum(1 for f in report["factors"].values() if f.get("is_stale"))
    print(
        f"[SIGNAL-DECAY] wrote {out_path} factors={len(report['factors'])} "
        f"stale={n_stale}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
