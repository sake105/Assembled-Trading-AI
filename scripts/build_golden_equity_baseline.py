"""B0 — Build the golden equity baseline JSON.

Runs the reference backtest (same config as scripts/profile_backtest.py) and
writes ``tests/regression/golden_equity_baseline.json`` in the schema expected
by ``scripts/compare_equity_curves.py``.

This file is committed so that the 1e-9 regression gate is bit-reproducible
across machines. Regenerate only when a **deliberate** semantic change is
accepted by the Plan's B0 gate — not for cosmetic CI reasons.

Usage:
    python scripts/build_golden_equity_baseline.py
    python scripts/build_golden_equity_baseline.py --out path/to/alt.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.profile_backtest import (  # noqa: E402
    build_reference_prices,
    equal_weight_sizing_fn,
    simple_signal_fn,
)
from src.assembled_core.qa.backtest_engine import run_portfolio_backtest  # noqa: E402


def run_reference(n_symbols: int, n_days: int, seed: int) -> dict:
    prices = build_reference_prices(n_symbols, n_days, seed)
    result = run_portfolio_backtest(
        prices=prices,
        signal_fn=simple_signal_fn,
        position_sizing_fn=equal_weight_sizing_fn,
        start_capital=100_000.0,
        include_costs=True,
        include_trades=False,
        include_ledger=False,
        strict_session_gate=False,
    )
    equity_rows = [
        {"timestamp": ts.isoformat(), "equity": float(eq)}
        for ts, eq in zip(
            result.equity["timestamp"].tolist(),
            result.equity["equity"].tolist(),
            strict=False,
        )
    ]
    return {
        "config": {
            "n_symbols": n_symbols,
            "n_days": n_days,
            "seed": seed,
            "start_capital": 100_000.0,
            "include_costs": True,
            "rebalance_schedule": "daily",
            "signal_fn": "scripts.profile_backtest.simple_signal_fn",
            "sizing_fn": "scripts.profile_backtest.equal_weight_sizing_fn",
        },
        "final_equity": float(result.equity["equity"].iloc[-1]),
        "n_bars": int(len(result.equity)),
        "equity": equity_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-symbols", type=int, default=25)
    parser.add_argument("--n-days", type=int, default=126)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "tests" / "regression" / "golden_equity_baseline.json",
    )
    args = parser.parse_args()

    payload = run_reference(args.n_symbols, args.n_days, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"[GOLDEN] wrote {args.out} bars={payload['n_bars']} "
        f"final_equity={payload['final_equity']:.9f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
