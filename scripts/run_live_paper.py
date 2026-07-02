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
from src.assembled_core.data.sources.yfinance_source import YFinanceRateLimitError  # noqa: E402

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
    try:
        from src.assembled_core.ops.alerting import AlertManager

        AlertManager().fire(
            "halt_flag_set",
            {"reason": payload.get("reason", "n/a"), "equity": "n/a"},
        )
    except Exception as _alert_exc:  # alerting must never break the halt write
        logger.error("[run_live_paper] halt alert failed: %s", _alert_exc)


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


def _drop_per_symbol_stale_rows(prices, max_age_days: int = 3):
    """Drop symbols whose own latest bar is older than ``max_age_days`` days.

    The global cache freshness check in ``_load_prices`` uses ``cache.max()``
    which lets a single fresh symbol mask 26 stale ones (audit 2026-05-21,
    F-RX-1): cache global max was 2026-05-18 (193 syms fresh), but 27 syms
    were 20–59 days stale (EXAS/HOLX delisted, KO/PEP/BRK-B/etc. live but
    unrefreshed). Without this filter, a dry-run today routed $60k of BUY
    orders priced on 20–59 day-old data.

    This filter is the per-symbol counterpart: drop the symbol entirely if
    its own latest timestamp is older than ``max_age_days``. Delisted
    symbols get pruned automatically; live-but-unrefreshed symbols surface
    as a loud WARN so the coverage gap can be addressed at the data layer
    instead of silently mispriced at the execution layer.
    """
    import pandas as pd

    if prices is None or prices.empty or "timestamp" not in prices.columns:
        return prices
    today = pd.Timestamp.now("UTC").normalize()
    ts = pd.to_datetime(prices["timestamp"], utc=True)
    prices = prices.assign(timestamp=ts)
    per_sym_latest = prices.groupby("symbol")["timestamp"].max()
    ages = (today - per_sym_latest.dt.normalize()).dt.days
    stale = per_sym_latest[ages > max_age_days]
    if not stale.empty:
        sample = ", ".join(sorted(stale.index)[:10]) + (
            "..." if len(stale) > 10 else ""
        )
        logger.warning(
            "[run_live_paper] dropping %d symbols with per-symbol staleness > %dd: %s",
            len(stale),
            max_age_days,
            sample,
        )
        prices = prices[~prices["symbol"].isin(stale.index)].reset_index(drop=True)
    return prices


def _load_prices(app_cfg: dict):
    """Load prices for live paper trading.

    Strategy: fetch fresh data via yfinance (1 year history for features),
    fall back to local parquet cache if yfinance fails.
    Non-US symbols (containing '.') are skipped for Alpaca compatibility.
    Per-symbol staleness filter (F-RX-1, 2026-05-21) drops symbols whose
    own latest bar is > 3 calendar days old, regardless of the source path.
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
                return _drop_per_symbol_stale_rows(cache_prices)
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
            return _drop_per_symbol_stale_rows(prices)
    except YFinanceRateLimitError as exc:
        logger.warning("[run_live_paper] yfinance rate-limited (HTTP 429): %s", exc)
    except Exception as exc:
        logger.warning("[run_live_paper] yfinance fetch failed: %s", exc)

    # --- Final fallback: BLOCK on stale cache (F-RX-7 §9.12 (e)) ---
    # Previously this path returned the stale cache with a WARN log, letting
    # the pilot trade on out-of-date prices when yfinance was down. With the
    # per-symbol staleness filter the worst-case shrinks to "trade on the
    # few syms that happen to be < 3d old", but partial-portfolio trading on
    # network-outage days is still undesirable. Block instead — operator can
    # see the run failed and either fix the data path or skip the day.
    if cache_prices is not None and not cache_prices.empty:
        logger.critical(
            "[run_live_paper] yfinance unavailable AND cache is stale (%s) — "
            "BLOCKING. No trades will be submitted on out-of-date prices. "
            "Resolve the upstream data issue or run scripts/ops/"
            "refresh_daily_cache_from_panel.py before retrying.",
            cache_stale_reason or "unknown age",
        )
        return pd.DataFrame()

    logger.error("[run_live_paper] no price data available from any source")
    return pd.DataFrame()


def _apply_adv_universe_filter(prices, paper_cfg: dict):
    """§9.6 (a) ADV universe filter — restrict to top-N most liquid symbols.

    Reads ``paper_runner.universe.{min_adv_top_n, adv_lookback_days}``.
    When ``min_adv_top_n`` is a positive int (default None = no filter),
    restricts the prices panel to the top-N symbols ranked by trailing
    dollar-volume. Default lookback = 20 trading days. Best-effort:
    failure logs WARNING and returns unfiltered prices.

    Motivation: backtest evidence (memory 2026-05-19) showed mfv2 Top-50
    daily = +5.70% CAGR vs all-195-daily = -7.74% CAGR — most of the
    delta was illiquidity drag on the long tail. Even for trend_baseline,
    restricting to liquid names reduces transaction-cost noise.
    """
    if prices is None or prices.empty:
        return prices

    universe_cfg = (paper_cfg or {}).get("universe") or {}
    top_n = universe_cfg.get("min_adv_top_n")
    if not top_n or int(top_n) <= 0:
        return prices

    lookback_days = int(universe_cfg.get("adv_lookback_days", 20))
    try:
        from src.assembled_core.data.universe import select_top_adv_symbols

        keep = select_top_adv_symbols(
            prices, top_n=int(top_n), lookback_days=lookback_days
        )
        if not keep:
            logger.warning(
                "[run_live_paper] ADV filter requested (top_n=%d) but produced "
                "empty universe — returning unfiltered prices",
                top_n,
            )
            return prices
        before = prices["symbol"].nunique() if "symbol" in prices.columns else 0
        filtered = prices[prices["symbol"].isin(keep)].reset_index(drop=True)
        after = filtered["symbol"].nunique() if "symbol" in filtered.columns else 0
        logger.info(
            "[run_live_paper] ADV universe filter: %d → %d symbols "
            "(top_n=%d, lookback=%dd)",
            before,
            after,
            top_n,
            lookback_days,
        )
        return filtered
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[run_live_paper] ADV filter failed (%s) — returning unfiltered prices",
            exc,
        )
        return prices


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

    # Drawdown stop (soft-halt): a breach writes the ack-clearable halt flag
    # (via _write_halt_flag), NOT the OPERATOR_KILL_TOKEN-gated kill switch.
    # Threshold + baseline are config-driven via
    # paper_runner.{dd_stop_pct, start_capital}. A missing dd_stop_pct falls
    # back to 0.30, preserving the pre-2026-07-02 30% behaviour.
    #
    # Evaluation and persistence are deliberately in SEPARATE scopes: a
    # transient failure to read/evaluate the stop is fail-closed-but-transient
    # (block this cycle, no flag, retry next cycle); a CONFIRMED breach whose
    # halt cannot be persisted must still block AND surface as an un-persisted
    # halt — never be misclassified as a self-recovering skip.
    dd_breach = False
    dd_equity = 0.0
    dd_baseline = 0.0
    dd_stop = 0.30
    try:
        from src.assembled_core.execution.kill_switch import (
            check_drawdown_kill_switch,
        )

        paper_cfg = app_cfg.get("paper_runner") or {}
        account = adapter.get_account()
        dd_equity = float(account.get("equity", 0))
        dd_baseline = float(paper_cfg.get("start_capital", 10000))
        dd_stop = float(paper_cfg.get("dd_stop_pct", 0.30))
        if dd_equity > 0 and dd_baseline > 0:
            # auto_activate=False -> detect only; a breach is persisted below as
            # the ack_halt-clearable flag, not the token-gated kill switch.
            dd_breach = check_drawdown_kill_switch(
                dd_equity, dd_baseline, kill_threshold=dd_stop, auto_activate=False
            )
    except Exception as exc:
        # Could not read/evaluate the stop (e.g. transient broker get_account()
        # error): fail-closed for THIS cycle only (return False) WITHOUT a halt
        # flag, so the next scheduled cycle retries once the read recovers —
        # avoiding a nuisance manual ack_halt on a transient network blip while
        # never trading the cycle with the drawdown stop unverified.
        logger.warning(
            "[run_live_paper] drawdown check failed (%s) — blocking this cycle "
            "(fail-closed, self-recovering; no halt flag written)",
            exc,
        )
        return False

    if dd_breach:
        dd_pct = (dd_equity - dd_baseline) / dd_baseline * 100
        logger.critical(
            "[run_live_paper] DRAWDOWN STOP — equity=%.2f dd=%.2f%% "
            "(limit=-%.0f%% of baseline %.2f) — writing halt flag",
            dd_equity,
            dd_pct,
            dd_stop * 100,
            dd_baseline,
        )
        try:
            _write_halt_flag(
                {
                    "ts_utc": datetime.now(timezone.utc).isoformat(),
                    "reason": (
                        f"drawdown stop: equity {dd_equity:.2f} breached "
                        f"-{dd_stop:.0%} of baseline {dd_baseline:.2f} "
                        f"(level {dd_baseline * (1 - dd_stop):.2f}). Next run "
                        f"halted until operator acks via scripts/ack_halt.py."
                    ),
                    "source": "run_live_paper._preflight_checks.drawdown_stop",
                }
            )
        except Exception as exc:
            # A CONFIRMED breach whose halt could not be persisted must STILL
            # block the cycle, and must NOT be reported as a transient
            # self-recovering skip: the ack gate did not arm. The next cycle
            # re-detects the still-breached equity and re-attempts the write.
            logger.critical(
                "[run_live_paper] DRAWDOWN STOP breached but halt-flag write "
                "FAILED (%s) — cycle BLOCKED; halt NOT persisted. Next cycle "
                "will re-detect; operator: check output/ops writability.",
                exc,
            )
        return False

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


_SOFT_TIMEOUT_TRIPPED: dict[str, bool] = {"flag": False}


def _arm_soft_timeout(seconds: float) -> "object":
    """F-RX-8 §9.12 (f): arm a soft-timeout that writes the halt-ack flag and
    flips an in-process gate BEFORE the Task Scheduler's hard kill.

    The Task Scheduler ExecutionTimeLimit (currently PT30M) hard-terminates
    the process without any chance for cleanup — exactly the failure mode
    that produced the stale pending intent on 2026-05-19 (mid-submission
    kill). This soft-timeout fires at ``seconds`` (default <= the OS limit)
    and:
      - writes ``output/ops/halt_ack_required.json`` so the NEXT run is
        blocked at the preflight gate (operator must clear via
        scripts/ack_halt.py)
      - flips ``_SOFT_TIMEOUT_TRIPPED`` which the main flow checks between
        steps to bail out gracefully instead of submitting more orders
      - logs CRITICAL so the per-day log captures the trip
    Returns the Timer handle so the caller can cancel it on normal exit.
    """
    import threading
    from datetime import datetime, timezone

    def _fire() -> None:
        _SOFT_TIMEOUT_TRIPPED["flag"] = True
        try:
            _write_halt_flag(
                {
                    "ts_utc": datetime.now(timezone.utc).isoformat(),
                    "reason": (
                        f"soft-timeout fired after {seconds:.0f}s in "
                        f"run_live_paper.cmd_once — task was about to be "
                        f"hard-killed by Task Scheduler. Next run is "
                        f"halted until operator acks via scripts/ack_halt.py."
                    ),
                    "source": "run_live_paper._arm_soft_timeout",
                }
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("[run_live_paper] halt-flag write failed: %s", exc)
        logger.critical(
            "[run_live_paper] SOFT TIMEOUT (%.0fs) — halt-ack flag set. "
            "No further orders will be submitted; main flow will exit at "
            "the next checkpoint.",
            seconds,
        )

    t = threading.Timer(seconds, _fire)
    t.daemon = True
    t.start()
    return t


def _check_soft_timeout(stage: str) -> None:
    """Exit gracefully if the soft-timeout has tripped. Called between major
    cmd_once stages so we never submit new orders after the trip."""
    if _SOFT_TIMEOUT_TRIPPED["flag"]:
        logger.critical(
            "[run_live_paper] soft-timeout already tripped before stage=%s "
            "— exiting gracefully with rc=2",
            stage,
        )
        sys.exit(2)


def cmd_once(args):
    """Single execution cycle."""
    import pandas as pd
    from src.assembled_core.ops.paper_runner import run_paper_daily_one

    # Soft-timeout default ~5min before the Task Scheduler PT30M hard kill,
    # so the in-process bail-out wins the race. Operators can tighten via
    # the --soft-timeout-seconds CLI flag below for short-window runs.
    soft_timeout_s = float(getattr(args, "soft_timeout_seconds", 1500.0))
    soft_timer = _arm_soft_timeout(soft_timeout_s) if soft_timeout_s > 0 else None

    app_cfg = _load_app_cfg()
    adapter = _create_adapter()

    if not _preflight_checks(adapter, app_cfg):
        if soft_timer is not None:
            soft_timer.cancel()
        sys.exit(1)
    _check_soft_timeout("post_preflight")

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
        if soft_timer is not None:
            soft_timer.cancel()
        sys.exit(1)
    # §9.6 (a): optional ADV-based universe restriction
    prices = _apply_adv_universe_filter(prices, app_cfg.get("paper_runner") or {})
    _check_soft_timeout("post_load_prices")

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
    # F-RX-FU-3: third checkpoint. If soft-timeout fired during order
    # generation/submission, skip post-execution reconciliation (it can
    # resume next run after operator clears halt-flag) and exit with rc=2.
    _check_soft_timeout("post_run_paper_daily")

    # Post-execution reconciliation
    if execution_mode == "broker" and exit_code == 0:
        try:
            from src.assembled_core.execution.position_sync import (
                sync_positions_from_broker,
            )
            from src.assembled_core.ops.paper_ledger import (
                LedgerCorruptionError,
                load_ledger_state,
            )

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
        except LedgerCorruptionError as exc:
            # R2-5: a corrupt ledger during post-exec reconcile must NOT be
            # downgraded to a warning by the broad handler below — that would
            # re-mask the corruption (E-025) one frame up. Surface it as a halt.
            logger.error(
                "[run_live_paper] post-execution reconcile aborted — corrupt ledger: %s",
                exc,
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
    if soft_timer is not None:
        soft_timer.cancel()
    sys.exit(exit_code)


def cmd_reconcile_only(args):
    """Position reconciliation only — no new orders."""
    from src.assembled_core.execution.position_sync import (
        sync_positions_from_broker,
    )
    from src.assembled_core.ops.paper_ledger import (
        LedgerCorruptionError,
        load_ledger_state,
    )

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
    try:
        ledger_state = load_ledger_state(ledger_path)
    except LedgerCorruptionError as exc:
        # R2-5: reconcile must not run against a silently-reset fresh state.
        logger.error("[run_live_paper] reconcile aborted — corrupt ledger: %s", exc)
        print(f"\n[ERROR] Corrupt ledger — reconcile aborted.\n{exc}")
        sys.exit(2)
    sync_result = sync_positions_from_broker(adapter, ledger_state)

    print(f"\n{'=' * 50}")
    print(f"Position Sync Result: {'OK' if sync_result.ok else 'MISMATCH'}")
    print(f"{'=' * 50}")
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
    # Fast-fail if required broker credentials are missing — before any
    # broker adapter or network call is attempted.
    try:
        from src.assembled_core.config.env_validator import validate_env

        validate_env()
    except RuntimeError as _env_err:
        print(f"[ENV] {_env_err}", file=sys.stderr)
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Live Paper Trading Runner — Alpaca broker integration"
    )
    sub = parser.add_subparsers(dest="command")

    # --once / --dry-run
    once_p = sub.add_parser("once", help="Run a single trading cycle")
    once_p.add_argument(
        "--dry-run", action="store_true", help="Log orders but don't submit to broker"
    )
    once_p.add_argument(
        "--soft-timeout-seconds",
        type=float,
        default=1500.0,
        help=(
            "F-RX-8 soft-timeout (default 1500s = 25min). Writes the "
            "halt-ack flag and exits at the next stage checkpoint, so the "
            "Task Scheduler hard-kill (PT30M default) does not interrupt "
            "mid-order. Set 0 to disable."
        ),
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
