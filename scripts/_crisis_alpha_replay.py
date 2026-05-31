"""Crisis alpha state replay for 2022 Ukraine invasion period.

Replays the crisis alpha state machine day-by-day through 2022 using:
- GPR index from output/macro_gpr.parquet (PIT-safe, 32-day lag)
- Market stress computed from price data filtered to as_of (PIT fix applied)

Shows: state transitions, when ACTIVE fires, what positions would be added.

Usage:
    python -m scripts._crisis_alpha_replay
"""

import json
import logging
import pathlib

import pandas as pd

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

from src.assembled_core.data.prices_ingest import load_eod_prices
from src.assembled_core.config.policy_loader import load_policy
from src.assembled_core.risk.market_stress import compute_market_stress
from src.assembled_core.events.crisis_alpha.context import CrisisAlphaContext
from src.assembled_core.events.crisis_alpha.pipeline import run_crisis_alpha_pipeline

PRICE_FILE = "output/backtest_crisis_test.parquet"
GPR_FILE = "output/macro_gpr.parquet"
STATE_PATH = pathlib.Path("output/ops/crisis_alpha_state_replay.json")
RELEASE_LAG = 32  # days

START = "2022-01-01"
END = "2023-06-30"


def _reset_state() -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(
        json.dumps(
            {
                "state": "WATCH",
                "entered_at_utc": None,
                "last_evaluated_utc": None,
                "reason": "reset",
                "geo_score_at_entry": 0.0,
                "cooldown_start_utc": None,
            }
        ),
        encoding="utf-8",
    )


def _load_gpr(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["timestamp", "gpr_index"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def _get_gpr_value(gpr_df: pd.DataFrame, as_of: pd.Timestamp) -> float:
    """Return most recent GPR value available at as_of (with 32-day release lag)."""
    cutoff = as_of - pd.Timedelta(days=RELEASE_LAG)
    available = gpr_df[gpr_df["timestamp"] <= cutoff]
    if available.empty:
        return 0.0
    return float(available["gpr_index"].iloc[-1])


def main() -> None:
    print(f"Crisis Alpha Replay  {START} to {END}")
    print(f"Price file : {PRICE_FILE}")
    print(f"GPR file   : {GPR_FILE}")
    print()

    # Load data
    raw = load_eod_prices(price_file=PRICE_FILE)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    # Keep a full history window: go back 2 years from start for market-stress baseline
    full_start = pd.Timestamp(START, tz="UTC") - pd.Timedelta(days=730)
    prices_full = raw[raw["timestamp"] >= full_start].reset_index(drop=True)

    gpr_df = _load_gpr(GPR_FILE)
    policy = load_policy()

    # Build trading day sequence
    ts_start = pd.Timestamp(START, tz="UTC")
    ts_end = pd.Timestamp(END, tz="UTC")
    trading_days = pd.bdate_range(start=ts_start, end=ts_end, freq="B", tz="UTC")

    _reset_state()

    prev_state = "WATCH"
    print(f"{'Date':<12} {'State':8} {'Geo':5} {'GPR':7} {'Stress':7} {'Note'}")
    print("-" * 75)

    state_log = []

    for ts in trading_days:
        # PIT-filtered prices up to this date
        prices_pit = prices_full[prices_full["timestamp"] <= ts]

        # Market stress (PIT-correct)
        ms = compute_market_stress(prices_pit, policy)
        stress_ok = ms["stress_ok"]
        vol_z = ms["details"].get("vol_z") or 0.0

        # GPR → geo_score
        gpr_val = _get_gpr_value(gpr_df, ts)
        if gpr_val > 200:
            geo_score = 2.0
            geo_sources = 2
            trigger_items = [
                {"severity": 2, "topic": "gpr_index", "source": "Caldara-Iacoviello"}
            ]
        elif gpr_val > 150:
            geo_score = 1.0
            geo_sources = 2
            trigger_items = [
                {"severity": 1, "topic": "gpr_index", "source": "Caldara-Iacoviello"}
            ]
        else:
            geo_score = 0.0
            geo_sources = 0
            trigger_items = []

        ctx = CrisisAlphaContext(
            timestamp_utc=ts.to_pydatetime(),
            geo_score=geo_score,
            geo_sources=geo_sources,
            social_only=False,
            market_stress_ok=stress_ok,
            market_stress_score=int(ms["stress_score"]),
            health_ok=True,
            news_trigger_items=trigger_items,
            daily_pnl=0.0,
            daily_loss_limit=0.02,
            open_positions=[],
        )

        res = run_crisis_alpha_pipeline(
            ctx, policy, state_path=STATE_PATH, dry_run=False
        )
        new_state = res["state"]

        note = ""
        if new_state != prev_state:
            note = f"← TRANSITION from {prev_state}"

        # Only print rows where something notable happens (state change, ACTIVE days, or weekly)
        is_monday = ts.dayofweek == 0
        if (
            new_state != prev_state
            or new_state == "ACTIVE"
            or (is_monday and gpr_val > 150)
        ):
            state_str = new_state
            if new_state == "ACTIVE":
                targets = res.get("target_weights", {})
                if targets:
                    note = f"targets: {list(targets.keys())}"
            note = note.replace("←", "<-").replace("→", "->")
            print(
                f"{str(ts.date()):<12} {state_str:8} {geo_score:5.1f} {gpr_val:7.1f}"
                f" {str(stress_ok):7} {note}"
            )
            state_log.append(
                {
                    "date": str(ts.date()),
                    "state": new_state,
                    "geo_score": geo_score,
                    "gpr": gpr_val,
                    "stress_ok": stress_ok,
                    "vol_z": round(vol_z, 3),
                }
            )

        prev_state = new_state

    # Summary
    active_days = sum(1 for r in state_log if r["state"] == "ACTIVE")
    print()
    print("=== Summary ===")
    print(f"ACTIVE days logged: {active_days}")
    all_states = [r["state"] for r in state_log]
    for s in sorted(set(all_states)):
        print(f"  {s}: {all_states.count(s)} transitions/notable days")


if __name__ == "__main__":
    main()
