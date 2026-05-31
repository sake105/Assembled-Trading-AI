"""WITH vs. WITHOUT crisis alpha backtest comparison (2022-2023 Ukraine period).

Patches the policy _policy_cache to toggle crisis alpha without touching policy.yaml.

Usage:
    python -m scripts._crisis_alpha_backtest_compare
"""

import copy
import json
import logging
import pathlib

import pandas as pd

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

# fmt: off
from src.assembled_core.data.prices_ingest import load_eod_prices
from src.assembled_core.config.policy_loader import load_policy
from src.assembled_core.qa.backtest_engine import make_cycle_fn, run_portfolio_backtest
from src.assembled_core.pipeline.trading_cycle_shared import TradingContext
from src.assembled_core.pipeline.trading_cycle_v2 import run_trading_cycle
from src.assembled_core.features.ta_features import add_all_features
from src.assembled_core.strategies.multifactor_v2 import (
    compute_signals,
    compute_target_positions as mfv2_size,
)
# fmt: on

START = "2022-01-01"
END = "2023-12-31"
CAP = 100_000.0
PRICE_FILE = "output/backtest_crisis_test.parquet"
STATE_PATH = pathlib.Path("output/ops/crisis_alpha_state.json")


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


def _run_backtest(
    prices: pd.DataFrame,
    prices_with_features: pd.DataFrame,
    policy: dict,
    label: str,
) -> dict:
    _reset_state()

    def signal_fn(df: pd.DataFrame) -> pd.DataFrame:
        return compute_signals(df)

    def sizing_fn(sigs: pd.DataFrame, cap: float) -> pd.DataFrame:
        return mfv2_size(sigs, total_capital=cap)

    ctx_template = TradingContext(
        prices=prices,
        freq="1d",
        use_factor_store=False,
        write_outputs=False,
        enable_risk_controls=False,
        backtest_use_snapshot=False,
        precomputed_prices_with_features=prices_with_features,
    )
    try:
        object.__setattr__(ctx_template, "_policy_cache", policy)
    except (AttributeError, TypeError):
        ctx_template._policy_cache = policy  # type: ignore[attr-defined]

    cycle_fn = make_cycle_fn(
        ctx_template,
        signal_fn=signal_fn,
        position_sizing_fn=sizing_fn,
        capital=CAP,
        run_trading_cycle_fn=run_trading_cycle,
        enable_risk_controls=False,
    )

    result = run_portfolio_backtest(
        prices=prices,
        signal_fn=signal_fn,
        position_sizing_fn=sizing_fn,
        start_capital=CAP,
        commission_bps=10,
        spread_w=0.25,
        impact_w=0.5,
        include_trades=True,
        cycle_fn=cycle_fn,
        compute_features=False,
    )

    eq = result.equity
    metrics = result.metrics or {}
    trades_df = result.trades
    n_trades = len(trades_df) if trades_df is not None else 0
    cagr = metrics.get("cagr")
    sharpe = metrics.get("sharpe")
    mdd = metrics.get("max_dd")
    pf = metrics.get("profit_factor")
    final_eq = float(eq.iloc[-1]) if eq is not None and len(eq) > 0 else float("nan")

    print(f"\n{'=' * 62}")
    print(f"  {label}")
    print(f"{'=' * 62}")
    print(f"  Trades      : {n_trades}")
    print(f"  Final equity: ${final_eq:,.0f}")
    if cagr is not None:
        print(f"  CAGR        : {cagr:.1%}")
    if sharpe is not None:
        print(f"  Sharpe      : {sharpe:.3f}")
    if mdd is not None:
        print(f"  Max DD      : {mdd:.1%}")
    if pf is not None:
        print(f"  Profit Fact.: {pf:.3f}")

    return {
        "label": label,
        "n_trades": n_trades,
        "final_eq": final_eq,
        "cagr": cagr,
        "sharpe": sharpe,
        "max_dd": mdd,
        "profit_factor": pf,
    }


def main() -> None:
    print(f"Loading prices: {PRICE_FILE}  ({START} to {END})")
    raw = load_eod_prices(price_file=PRICE_FILE)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    mask = (raw["timestamp"] >= START) & (raw["timestamp"] <= END)
    prices = raw[mask].reset_index(drop=True)
    print(f"Prices: {len(prices)} rows, {prices['symbol'].nunique()} symbols")

    print("Pre-computing TA features once...")
    try:
        prices_feat = add_all_features(prices)
    except Exception as e:
        print(f"add_all_features failed ({e}), using raw prices")
        prices_feat = prices

    base_policy = load_policy()

    # Run A — no crisis alpha
    pol_no = copy.deepcopy(base_policy)
    pol_no.setdefault("intel", {}).setdefault("crisis_alpha", {})["enabled"] = False
    res_no = _run_backtest(prices, prices_feat, pol_no, "A — WITHOUT crisis alpha")

    # Run B — with crisis alpha
    pol_yes = copy.deepcopy(base_policy)
    pol_yes.setdefault("intel", {}).setdefault("crisis_alpha", {})["enabled"] = True
    res_yes = _run_backtest(prices, prices_feat, pol_yes, "B — WITH crisis alpha")

    # Delta
    print(f"\n{'=' * 62}")
    print("  Delta (B − A)")
    print(f"{'=' * 62}")
    for key, fmt in [("cagr", ".1%"), ("sharpe", ".3f"), ("max_dd", ".1%")]:
        a, b = res_no.get(key), res_yes.get(key)
        if a is not None and b is not None:
            rel = (b - a) / abs(a) * 100 if a != 0 else 0.0
            print(f"  {key:12}: {b - a:+{fmt}}  ({rel:+.1f}% rel)")
    dt = res_yes["n_trades"] - res_no["n_trades"]
    print(f"  trades      : {dt:+d}")


if __name__ == "__main__":
    main()
