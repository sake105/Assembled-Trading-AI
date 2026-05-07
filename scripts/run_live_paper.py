"""Live Paper Trading Runner — Execute paper trades via Alpaca broker API.

Modes:
  --once              Single cycle (for Task Scheduler / cron)
  --dry-run           Show everything without real orders
  --reconcile-only    Only sync positions, no new orders
  --rebuild-ledger    Emergency: rebuild ledger from Alpaca positions

Usage:
  python scripts/run_live_paper.py --once
  python scripts/run_live_paper.py --once --dry-run
  python scripts/run_live_paper.py --reconcile-only
  python scripts/run_live_paper.py --rebuild-ledger
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from src.assembled_core.logging_config import generate_run_id, setup_logging

logger = logging.getLogger(__name__)


HALT_FLAG_PATH = ROOT / "output" / "ops" / "halt_ack_required.json"


def _reconcile_policy(app_cfg: dict) -> dict:
    pol = (app_cfg.get("policy") or {}).get("reconciliation") or {}
    return {
        "halt_on_mismatch": bool(pol.get("halt_on_mismatch", True)),
        "cash_threshold_usd": float(pol.get("cash_threshold_usd", 100.0)),
        "cash_threshold_bps": float(pol.get("cash_threshold_bps", 10.0)),
    }


def _write_halt_flag(payload: dict) -> None:
    HALT_FLAG_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = HALT_FLAG_PATH.with_suffix(HALT_FLAG_PATH.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(HALT_FLAG_PATH)


def _mismatch_exceeds_threshold(
    cash_diff: float, broker_equity: float, policy: dict
) -> tuple[bool, str]:
    abs_diff = abs(cash_diff)
    usd_trip = abs_diff > policy["cash_threshold_usd"]
    bps_trip = False
    bps_observed = None
    if broker_equity > 0:
        bps_observed = abs_diff / broker_equity * 10_000.0
        bps_trip = bps_observed > policy["cash_threshold_bps"]
    if not (usd_trip or bps_trip):
        return False, ""
    reason_bits = []
    if usd_trip:
        reason_bits.append(
            f"cash_diff=${abs_diff:.2f} > {policy['cash_threshold_usd']:.2f}"
        )
    if bps_trip and bps_observed is not None:
        reason_bits.append(
            f"cash_diff={bps_observed:.2f}bps > {policy['cash_threshold_bps']:.2f}bps"
        )
    return True, " AND ".join(reason_bits)


def _load_app_cfg() -> dict:
    """Load app config from configs/app.yaml or return defaults."""
    cfg_path = ROOT / "configs" / "app.yaml"
    if cfg_path.exists():
        try:
            import yaml

            return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        except ImportError:
            return json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("[run_live_paper] failed to load app config: %s", exc)
    return {}


def _create_adapter():
    """Create and validate AlpacaAdapter instance."""
    from src.assembled_core.execution.broker_adapter import AlpacaAdapter

    adapter = AlpacaAdapter()
    health = adapter.health_check()
    if not health.get("ok"):
        logger.critical(
            "[run_live_paper] broker health check FAILED: %s",
            health.get("message", "unknown"),
        )
        sys.exit(1)

    logger.info(
        "[run_live_paper] broker connected — equity=%.2f",
        health.get("account_equity") or 0,
    )
    return adapter


def _load_prices(app_cfg: dict):
    """Load prices for live paper trading.

    Strategy: fetch fresh data via yfinance (1 year history for features),
    fall back to local parquet cache if yfinance fails.
    Non-US symbols (containing '.') are skipped for Alpaca compatibility.
    """
    import pandas as pd

    universe_path = ROOT / "watchlist.txt"
    if not universe_path.exists():
        logger.error("[run_live_paper] watchlist.txt not found")
        return pd.DataFrame()

    all_symbols = [
        s.strip()
        for s in universe_path.read_text(encoding="utf-8").splitlines()
        if s.strip() and not s.strip().startswith("#")
    ]
    # Filter to US symbols only (Alpaca paper supports US equities)
    symbols = [s for s in all_symbols if "." not in s]
    skipped = [s for s in all_symbols if "." in s]
    if skipped:
        logger.info(
            "[run_live_paper] skipping %d non-US symbols: %s",
            len(skipped),
            ", ".join(skipped[:5]) + ("..." if len(skipped) > 5 else ""),
        )

    if not symbols:
        logger.error("[run_live_paper] no tradeable US symbols in watchlist")
        return pd.DataFrame()

    logger.info("[run_live_paper] loading prices for %d US symbols", len(symbols))

    # --- Try 1: local parquet cache, but only if fresh ---
    # For live paper trading a cache more than ~3 calendar days old is stale
    # (weekend + one holiday = 3d). If stale, skip to yfinance so decisions
    # use the latest close. Cache still serves as a final fallback below.
    cache_prices = None
    cache_stale_reason: str | None = None
    try:
        from src.assembled_core.data.prices_ingest import load_eod_prices

        cache_prices = load_eod_prices(symbols=symbols)
        if not cache_prices.empty:
            cache_latest_ts = pd.Timestamp(cache_prices["timestamp"].max())
            if cache_latest_ts.tzinfo is None:
                cache_latest = cache_latest_ts.tz_localize("UTC")
            else:
                cache_latest = cache_latest_ts.tz_convert("UTC")
            today_utc = pd.Timestamp.now("UTC")
            age_days = (today_utc.normalize() - cache_latest.normalize()).days
            if age_days <= 3:
                n_syms = cache_prices["symbol"].nunique()
                logger.info(
                    "[run_live_paper] using cache — %d rows, %d symbols, "
                    "latest=%s, age=%dd",
                    len(cache_prices),
                    n_syms,
                    cache_latest.date(),
                    age_days,
                )
                return cache_prices
            cache_stale_reason = f"cache latest={cache_latest.date()} age={age_days}d"
            logger.warning(
                "[run_live_paper] cache is stale (%s) — fetching fresh from yfinance",
                cache_stale_reason,
            )
    except Exception as exc:
        logger.info("[run_live_paper] local cache unavailable: %s", exc)

    # --- Try 2: yfinance batch (authoritative when cache is stale) ---
    try:
        from src.assembled_core.data.sources.yfinance_source import (
            fetch_prices_yfinance,
        )

        end_date = (pd.Timestamp.now("UTC") + pd.DateOffset(days=1)).strftime(
            "%Y-%m-%d"
        )
        start_date = (pd.Timestamp.now("UTC") - pd.DateOffset(days=400)).strftime(
            "%Y-%m-%d"
        )

        logger.info(
            "[run_live_paper] fetching %d symbols via yfinance (%s to %s)",
            len(symbols),
            start_date,
            end_date,
        )
        prices = fetch_prices_yfinance(symbols, start_date, end_date)
        if not prices.empty:
            fresh_latest = pd.Timestamp(prices["timestamp"].max())
            logger.info(
                "[run_live_paper] fetched %d rows via yfinance — latest=%s",
                len(prices),
                fresh_latest.date() if hasattr(fresh_latest, "date") else fresh_latest,
            )
            return prices
    except Exception as exc:
        logger.warning("[run_live_paper] yfinance fetch failed: %s", exc)

    # --- Final fallback: stale cache if we have nothing else ---
    if cache_prices is not None and not cache_prices.empty:
        logger.warning(
            "[run_live_paper] yfinance unavailable — falling back to STALE cache (%s). "
            "Signals will be computed on out-of-date prices.",
            cache_stale_reason or "unknown age",
        )
        return cache_prices

    logger.error("[run_live_paper] no price data available from any source")
    return pd.DataFrame()


def _preflight_checks(adapter, app_cfg: dict) -> bool:
    """Run pre-flight safety checks. Returns True if safe to proceed."""
    from src.assembled_core.execution.kill_switch import is_kill_switch_engaged

    # Halt-ack gate: a prior cycle's reconcile mismatch may have written
    # HALT_FLAG_PATH. The operator must clear it via scripts/ack_halt.py
    # before the next run is permitted. Without this check the documented
    # halt-ack policy was de-facto unenforced.
    if HALT_FLAG_PATH.exists():
        logger.critical(
            "[run_live_paper] HALT FLAG present at %s — clear via scripts/ack_halt.py",
            HALT_FLAG_PATH,
        )
        return False

    # Kill switch
    if is_kill_switch_engaged():
        logger.critical("[run_live_paper] KILL SWITCH ENGAGED — aborting")
        return False

    # Drawdown check
    try:
        from src.assembled_core.execution.kill_switch import (
            check_drawdown_kill_switch,
        )

        account = adapter.get_account()
        equity = float(account.get("equity", 0))
        start_capital = float(
            (app_cfg.get("paper_runner") or {}).get("start_capital", 10000)
        )
        if equity > 0 and start_capital > 0:
            if check_drawdown_kill_switch(equity, start_capital):
                dd_pct = (equity - start_capital) / start_capital * 100
                logger.critical(
                    "[run_live_paper] DRAWDOWN KILL SWITCH — equity=%.2f dd=%.2f%%",
                    equity,
                    dd_pct,
                )
                return False
    except Exception as exc:
        logger.warning("[run_live_paper] drawdown check failed: %s", exc)

    # Stale open-order cleanup: cancel orders older than 5 minutes that survived
    # a prior crashed or interrupted run.  Orders submitted within the last 5
    # minutes are left alone — they may still be working normally.
    try:
        open_orders = adapter.get_open_orders()
        if open_orders:
            now_utc = datetime.now(timezone.utc)
            stale_ids: list[str] = []
            recent_count = 0
            for o in open_orders:
                submitted_str = o.submitted_at
                if submitted_str:
                    try:
                        submitted_dt = datetime.fromisoformat(
                            submitted_str.replace("Z", "+00:00")
                        )
                        if submitted_dt.tzinfo is None:
                            from datetime import timezone as _tz
                            submitted_dt = submitted_dt.replace(tzinfo=_tz.utc)
                        age_seconds = (now_utc - submitted_dt).total_seconds()
                        if age_seconds > 300:  # 5 minutes
                            stale_ids.append(o.order_id)
                        else:
                            recent_count += 1
                    except Exception:
                        # Cannot parse timestamp — treat as stale to be safe
                        stale_ids.append(o.order_id)
                else:
                    # No timestamp — treat as stale
                    stale_ids.append(o.order_id)

            if stale_ids:
                logger.warning(
                    "[run_live_paper] cancelling %d stale open order(s) (>5min old) — "
                    "%d recent order(s) left untouched",
                    len(stale_ids),
                    recent_count,
                )
                cancelled = adapter.cancel_all_orders()
                logger.warning(
                    "[run_live_paper] stale order cleanup: %d order(s) cancelled",
                    cancelled,
                )
            elif recent_count:
                logger.info(
                    "[run_live_paper] %d recent open order(s) found — all within 5min, "
                    "no cleanup needed",
                    recent_count,
                )
    except Exception as exc:
        logger.warning("[run_live_paper] open orders cleanup failed: %s", exc)

    # Pending intent check (crash recovery)
    try:
        from src.assembled_core.execution.intent_store import (
            find_pending_order_intents,
        )

        pending = find_pending_order_intents()
        if pending:
            logger.warning(
                "[run_live_paper] %d pending order intents from prior crash — "
                "reconcile manually before proceeding",
                len(pending),
            )
    except Exception as exc:
        logger.warning("[run_live_paper] intent store check failed: %s", exc)

    return True


def cmd_once(args):
    """Single execution cycle."""
    import pandas as pd
    from src.assembled_core.ops.paper_runner import run_paper_daily_one

    app_cfg = _load_app_cfg()
    adapter = _create_adapter()

    if not _preflight_checks(adapter, app_cfg):
        sys.exit(1)

    # Reset per-cycle counters
    adapter.reset_cycle_counters()

    execution_mode = "dry_run" if args.dry_run else "broker"
    as_of = pd.Timestamp.now("UTC")
    run_id = generate_run_id(prefix="live_paper")
    output_dir = ROOT / "output" / "runs" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "[run_live_paper] starting %s cycle — run_id=%s as_of=%s",
        execution_mode.upper(),
        run_id,
        as_of.isoformat(),
    )

    prices = _load_prices(app_cfg)
    if prices.empty:
        logger.error("[run_live_paper] no prices available — aborting")
        sys.exit(1)

    exit_code, reconcile_status = run_paper_daily_one(
        as_of_ts=as_of,
        output_dir=output_dir,
        mode="paper",
        app_cfg=app_cfg,
        prices=prices,
        root=ROOT,
        execution_mode=execution_mode,
        broker_adapter=adapter,
    )

    # Post-execution reconciliation
    if execution_mode == "broker" and exit_code == 0:
        try:
            from src.assembled_core.execution.position_sync import (
                sync_positions_from_broker,
            )
            from src.assembled_core.ops.paper_ledger import load_ledger_state

            paper_cfg = app_cfg.get("paper_runner") or {}
            ledger_path_str = (
                paper_cfg.get("ledger_path")
                or "output/runs/_paper_ledger/ledger_state.json"
            )
            ledger_path = (
                ROOT / ledger_path_str
                if not Path(ledger_path_str).is_absolute()
                else Path(ledger_path_str)
            )
            ledger_state = load_ledger_state(ledger_path)
            sync_result = sync_positions_from_broker(adapter, ledger_state)

            if not sync_result.ok:
                policy = _reconcile_policy(app_cfg)
                tripped, reason = _mismatch_exceeds_threshold(
                    sync_result.cash_diff,
                    sync_result.broker_equity,
                    policy,
                )
                logger.warning(
                    "[run_live_paper] POST-EXECUTION MISMATCH: %s",
                    sync_result.message,
                )
                if tripped and policy["halt_on_mismatch"]:
                    _write_halt_flag(
                        {
                            "triggered_at_utc": datetime.now(timezone.utc).isoformat(),
                            "run_id": run_id,
                            "cycle_date": as_of.strftime("%Y-%m-%d"),
                            "cash_diff": sync_result.cash_diff,
                            "broker_equity": sync_result.broker_equity,
                            "broker_cash": sync_result.broker_cash,
                            "ledger_cash": sync_result.ledger_cash,
                            "mismatches_count": len(sync_result.mismatches or []),
                            "policy": policy,
                            "reason": reason,
                            "message": sync_result.message,
                        }
                    )
                    logger.error(
                        "[run_live_paper] HALT engaged — %s; wrote %s. "
                        "Clear with: python scripts/ack_halt.py --reason=...",
                        reason,
                        HALT_FLAG_PATH,
                    )
                    reconcile_status = "halt"
                    exit_code = max(exit_code, 2)
        except Exception as exc:
            logger.warning("[run_live_paper] post-execution sync failed: %s", exc)

    # Write experience log entry
    try:
        from src.assembled_core.ops.experience_log import append_experience

        account = adapter.get_account()
        append_experience(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "cycle_date": as_of.strftime("%Y-%m-%d"),
                "execution_mode": execution_mode,
                "run_id": run_id,
                "exit_code": exit_code,
                "reconcile_status": reconcile_status,
                "broker_equity": float(account.get("equity", 0)),
                "broker_cash": float(account.get("cash", 0)),
            }
        )
    except Exception as exc:
        logger.warning("[run_live_paper] experience log write failed: %s", exc)

    logger.info(
        "[run_live_paper] %s cycle complete — exit_code=%d reconcile=%s",
        execution_mode.upper(),
        exit_code,
        reconcile_status,
    )
    sys.exit(exit_code)


def cmd_reconcile_only(args):
    """Position reconciliation only — no new orders."""
    from src.assembled_core.execution.position_sync import (
        sync_positions_from_broker,
    )
    from src.assembled_core.ops.paper_ledger import load_ledger_state

    app_cfg = _load_app_cfg()
    adapter = _create_adapter()

    paper_cfg = app_cfg.get("paper_runner") or {}
    ledger_path_str = (
        paper_cfg.get("ledger_path") or "output/runs/_paper_ledger/ledger_state.json"
    )
    ledger_path = (
        ROOT / ledger_path_str
        if not Path(ledger_path_str).is_absolute()
        else Path(ledger_path_str)
    )
    ledger_state = load_ledger_state(ledger_path)
    sync_result = sync_positions_from_broker(adapter, ledger_state)

    print(f"\n{'='*50}")
    print(f"Position Sync Result: {'OK' if sync_result.ok else 'MISMATCH'}")
    print(f"{'='*50}")
    print(f"Ledger cash:  ${sync_result.ledger_cash:,.2f}")
    print(f"Broker cash:  ${sync_result.broker_cash:,.2f}")
    print(f"Cash diff:    ${sync_result.cash_diff:,.2f}")
    print(f"Broker equity: ${sync_result.broker_equity:,.2f}")

    if sync_result.missing_in_ledger:
        print(f"\nMissing in ledger: {sync_result.missing_in_ledger}")
    if sync_result.missing_in_broker:
        print(f"Missing in broker: {sync_result.missing_in_broker}")
    if sync_result.mismatches:
        print("\nPosition mismatches:")
        for m in sync_result.mismatches:
            print(
                f"  {m['symbol']}: ledger={m.get('ledger_qty')}, broker={m.get('broker_qty')}"
            )

    if not sync_result.ok:
        print(f"\nMessage: {sync_result.message}")
        sys.exit(1)


def cmd_rebuild_ledger(args):
    """Emergency: rebuild ledger from Alpaca positions."""
    from src.assembled_core.execution.position_sync import (
        rebuild_ledger_from_broker,
    )
    from src.assembled_core.ops.paper_ledger import save_ledger_state

    app_cfg = _load_app_cfg()
    adapter = _create_adapter()

    paper_cfg = app_cfg.get("paper_runner") or {}
    ledger_path_str = (
        paper_cfg.get("ledger_path") or "output/runs/_paper_ledger/ledger_state.json"
    )
    ledger_path = (
        ROOT / ledger_path_str
        if not Path(ledger_path_str).is_absolute()
        else Path(ledger_path_str)
    )

    print("\n*** EMERGENCY LEDGER REBUILD ***")
    print("This will REPLACE the current ledger with broker state.")
    print("Historical equity curve data will be LOST.")
    confirm = input("Type 'REBUILD' to confirm: ")
    if confirm.strip() != "REBUILD":
        print("Aborted.")
        sys.exit(0)

    new_state = rebuild_ledger_from_broker(adapter)
    save_ledger_state(new_state, ledger_path)
    print(
        f"\nLedger rebuilt: cash=${new_state['cash']:,.2f}, "
        f"{len(new_state['positions'])} positions"
    )
    print(f"Saved to: {ledger_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Live Paper Trading Runner — Alpaca broker integration"
    )
    sub = parser.add_subparsers(dest="command")

    # --once / --dry-run
    once_p = sub.add_parser("once", help="Run a single trading cycle")
    once_p.add_argument(
        "--dry-run", action="store_true", help="Log orders but don't submit to broker"
    )
    once_p.set_defaults(func=cmd_once)

    # --reconcile-only
    recon_p = sub.add_parser("reconcile", help="Reconcile positions only (no orders)")
    recon_p.set_defaults(func=cmd_reconcile_only)

    # --rebuild-ledger
    rebuild_p = sub.add_parser(
        "rebuild-ledger", help="Emergency: rebuild ledger from broker"
    )
    rebuild_p.set_defaults(func=cmd_rebuild_ledger)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    run_id = generate_run_id(prefix="live_paper")
    setup_logging(run_id=run_id)
    logger.info("[run_live_paper] starting — command=%s", args.command)

    args.func(args)


if __name__ == "__main__":
    main()
