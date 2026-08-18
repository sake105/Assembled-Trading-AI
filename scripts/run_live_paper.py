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

    # F-senior-8 / Pilot-Diagnose 2026-08-18: die crisis_alpha-Hedges
    # (SH/SHY/VIXY/XLU/...) brauchen PREISE, damit ihre Ziele zu Orders
    # werden — ohne Preis kann generate_orders_from_targets das Notional
    # nicht in Stueck umrechnen und der Hedge faellt im Ernstfall STILL aus
    # (gemessen: 5 Crisis-Ziele, 0 Crisis-Orders). Sie gehoeren aber NICHT
    # in watchlist.txt: dort wuerden sie Kandidaten der Core-Trend-Strategie
    # und gingen genau im Krisenfall LONG, waehrend das Overlay dieselben
    # Instrumente kauft (Doppelallokation, F-senior-8). Deshalb hier NUR
    # dem Preis-Frame beimischen; das Signal-Universum bleibt die Watchlist.
    try:
        from src.assembled_core.events.crisis_alpha.baskets import (
            get_basket_symbols,
        )

        _hedges = [h for h in get_basket_symbols() if "." not in h]
        _added = [h for h in _hedges if h not in symbols]
        if _added:
            symbols = symbols + _added
            logger.info(
                "[run_live_paper] added %d crisis-hedge symbols to the PRICE "
                "frame only (not to the signal universe): %s",
                len(_added),
                ", ".join(sorted(_added)),
            )
    except Exception as _hexc:  # noqa: BLE001 - Preisbeimischung ist best-effort
        logger.warning(
            "[run_live_paper] could not add crisis-hedge symbols to price "
            "frame (%s) — crisis targets may not become orders",
            _hexc,
        )

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
    # Initialised before the try so the BLOCK path below can report WHY the
    # feed failed instead of just "unknown age" (DAT-005 / E-142).
    feed_status = None
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

        # --- DAT-005 / E-142: read the feed stamp the source just wrote ---
        #
        # SCOPE, stated plainly: this makes the stamp VISIBLE. It does not yet
        # make it BINDING. fetch_prices_yfinance stamps the outcome into
        # DataFrame.attrs and nobody read it; now it is read and logged, and
        # the BLOCK path below reports the feed reason instead of "unknown age".
        # But the `if not prices.empty` decision twelve lines down is unchanged:
        # on a PARTIAL outage (say 40 of 200 symbols missing) the pilot still
        # trades the silently shrunken universe, now with a WARNING in the log.
        #
        # That is deliberate for this step, not an oversight. Deciding at what
        # share of missing symbols a live cycle must abort is a risk decision in
        # a protected path, and it does not belong as a side effect of a data
        # inventory. Tracked as an open item in KNOWN_ISSUES §0.06.
        #
        # attrs survive the path from source to here (assign / boolean mask /
        # reset_index all preserve them), so this read is reliable.
        # Cache-loaded frames carry no stamp at all -> get_feed_status returns
        # None, which means "unknown", NOT "ok". Do not treat it as ok.
        try:
            from src.assembled_core.data.feed_status import (
                FEED_ERROR,
                get_feed_status,
            )

            feed_status = get_feed_status(prices)
        except Exception as exc:  # pragma: no cover - never break the fetch path
            logger.debug("[run_live_paper] feed_status unavailable: %s", exc)
            FEED_ERROR = "error"  # noqa: N806

        if feed_status:
            _reason = feed_status.get("reason")
            if feed_status.get("status") == FEED_ERROR:
                logger.error(
                    "[run_live_paper] yfinance reported FEED OUTAGE "
                    "(reason=%s, rows=%s) - the result is an error, not an "
                    "empty window.",
                    _reason,
                    feed_status.get("n_rows"),
                )
            elif _reason == "partial_outage":
                _got = int(prices["symbol"].nunique()) if not prices.empty else 0
                logger.warning(
                    "[run_live_paper] yfinance PARTIAL OUTAGE - %d of %d "
                    "requested symbols returned data (%s rows). The universe "
                    "is silently smaller than intended for this cycle.",
                    _got,
                    len(symbols),
                    feed_status.get("n_rows"),
                )

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
        # Carry the feed's own reason into the CRITICAL line. Without it the
        # operator sees "unknown age" and has to guess whether the upstream
        # problem was rate-limiting, a credential failure or an empty window.
        _feed_reason = (
            f"{feed_status.get('status')}/{feed_status.get('reason')}"
            if feed_status
            else "no feed stamp (source never reached or cache path)"
        )
        logger.critical(
            "[run_live_paper] yfinance unavailable AND cache is stale (%s) — "
            "BLOCKING. No trades will be submitted on out-of-date prices. "
            "feed_status=%s. Resolve the upstream data issue or run scripts/ops/"
            "refresh_daily_cache_from_panel.py before retrying.",
            cache_stale_reason or "unknown age",
            _feed_reason,
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
                # K3 (2026-07-21, GESAMTBEWERTUNG): the old code claimed
                # "recent orders left untouched" but then called
                # cancel_all_orders(), cancelling recent orders too. Prefer a
                # per-order cancel when the adapter provides one; otherwise
                # only blanket-cancel when NO recent orders would be caught
                # in the blast radius — never silently contradict the log.
                cancel_one = getattr(adapter, "cancel_order", None)
                if callable(cancel_one):
                    cancelled = 0
                    for oid in stale_ids:
                        try:
                            cancel_one(oid)
                            cancelled += 1
                        except Exception as cexc:
                            logger.warning(
                                "[run_live_paper] cancel of stale order %s failed: %s",
                                oid,
                                cexc,
                            )
                    logger.warning(
                        "[run_live_paper] stale order cleanup: %d/%d stale order(s) "
                        "cancelled — %d recent order(s) left untouched",
                        cancelled,
                        len(stale_ids),
                        recent_count,
                    )
                elif recent_count == 0:
                    cancelled = adapter.cancel_all_orders()
                    logger.warning(
                        "[run_live_paper] stale order cleanup: %d order(s) cancelled "
                        "(blanket cancel — no recent orders present)",
                        cancelled,
                    )
                else:
                    # Stage-1 review M1 (2026-07-21): trading on top of live
                    # stale orders risks double exposure (the unbooked-fill
                    # class). If we can neither cancel them individually nor
                    # safely blanket-cancel, BLOCK the cycle instead of
                    # warn-and-trade.
                    logger.error(
                        "[run_live_paper] PREFLIGHT BLOCK: %d stale order(s) "
                        "cannot be cancelled (adapter has no per-order cancel; "
                        "%d recent order(s) would be caught by "
                        "cancel_all_orders()). Operator: cancel stale orders "
                        "manually, then re-run.",
                        len(stale_ids),
                        recent_count,
                    )
                    return False
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
            # W7 (2026-07-21, GESAMTBEWERTUNG): pending intents mean a prior
            # run crashed between submit-intent and fill-confirmation — the
            # broker may hold orders/positions the ledger never booked
            # (exactly the 2026-07-14 failure class). Trading on top of
            # unresolved crash residue is not safe: BLOCK instead of WARN.
            logger.error(
                "[run_live_paper] PREFLIGHT BLOCK: %d pending order intent(s) "
                "from a prior crashed run — resolve before trading. Operator: "
                "compare broker orders/positions vs ledger (scripts/"
                "ops_adopt_external_positions.py for unbooked fills), then "
                "clear/abandon the intents.",
                len(pending),
            )
            return False
    except ImportError as exc:
        # Stage-2 review F-senior-1 (2026-07-22): separate diagnosis — an
        # ImportError is an infrastructure/deploy problem (renamed module,
        # broken package), NOT crash residue. Still fail-closed (Rule 30),
        # but tell the operator the true cause.
        logger.error(
            "[run_live_paper] PREFLIGHT BLOCK: intent_store module unavailable "
            "(%s) — deploy/import problem, not crash residue. Fix the "
            "installation, then re-run.",
            exc,
        )
        return False
    except Exception as exc:
        # Stage-1 review M2 (2026-07-21): a crashed prior run is precisely
        # when the intent store may be corrupt — a broken checker must not
        # fail open into trading. Fail closed like the block it guards.
        logger.error(
            "[run_live_paper] PREFLIGHT BLOCK: intent store check failed (%s) — "
            "cannot prove there is no crash residue. Operator: inspect "
            "output/ops intent store, then re-run.",
            exc,
        )
        return False

    # W4 (2026-07-24, GESAMTBEWERTUNG Schritt 8): QA-block flag gate.
    # A BLOCK verdict from a root-output orchestrator run persists
    # output/ops/qa_block.json; while it exists (or is unreadable —
    # fail-closed on positive-but-corrupt evidence) the pilot refuses to
    # trade. ABSENCE means "no known QA block", not "QA passed" — the
    # orchestrator does not run in the daily cycle, so a freshness-gated
    # variant would dead-lock the pilot (E-054 lesson). Clearing is an
    # audited operator act: scripts/ops/ack_qa_block.py (reason-gated,
    # ledger-appended, flag archived — NOT a bare delete).
    # Runs AFTER the intent check (Stage-1 B5: crash residue is the more
    # urgent diagnosis; both blocks surface in order of urgency).
    # NOTE: distinct from ctx.qa_block_trading (in-cycle data-QC gate in
    # trading_cycle_shared) — this flag carries a cross-process BACKTEST
    # QA verdict.
    try:
        from src.assembled_core.qa.qa_gates import read_qa_block_flag

        _qa_flag = read_qa_block_flag()
    except Exception as exc:
        # Stage-1 H1 (2026-07-24): a failing flag READER is "evidence check
        # not performable", not "no evidence" — fail closed like the intent
        # check above (self-healing next cycle; no freshness dead-lock risk).
        logger.error(
            "[run_live_paper] PREFLIGHT BLOCK: qa-block flag check failed "
            "(%s) — infra failure, not a QA verdict. Fix and re-run.",
            exc,
        )
        return False
    if _qa_flag is not None:
        logger.error(
            "[run_live_paper] PREFLIGHT BLOCK: QA-block flag present "
            "(source=%s, written_at=%s, gates=%s) — review, then clear via "
            'scripts/ops/ack_qa_block.py --reason "...".',
            _qa_flag.get("source", "?"),
            _qa_flag.get("written_at_utc", "?"),
            _qa_flag.get("blocked_gates", "unreadable flag"),
        )
        return False

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


def _alert_on_run_gap(app_cfg: dict, *, max_gap_days: int = 3) -> None:
    """W12 (2026-07-21, GESAMTBEWERTUNG): detect missed pilot runs after the fact.

    The watchdog runs on the SAME host as the pilot — a powered-off machine
    alarms nothing (live-verified: runs 15.-17.07. missed silently). True
    external monitoring needs an off-host component; until then, the next
    successful start compares the ledger's last equity-curve date with today
    and alerts on a gap > ``max_gap_days`` calendar days (3 covers normal
    weekends; anything longer means missed trading days). Best-effort:
    never blocks the cycle.
    """
    try:
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
        if not ledger_path.exists():
            return
        state = load_ledger_state(ledger_path)
        curve = state.get("equity_curve") or []
        if not curve:
            return
        last_utc = str(curve[-1].get("utc", ""))[:10]
        if not last_utc:
            return
        last_day = datetime.strptime(last_utc, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        gap_days = (datetime.now(timezone.utc) - last_day).days
        if gap_days > max_gap_days:
            logger.warning(
                "[run_live_paper] RUN GAP: last ledger equity point is %s "
                "(%d calendar days ago) — pilot runs were missed silently. "
                "Check host uptime / Task Scheduler history.",
                last_utc,
                gap_days,
            )
            try:
                from src.assembled_core.ops.alerting import AlertManager

                AlertManager().fire(
                    "pilot_run_gap",
                    {"last_run_date": last_utc, "gap_days": str(gap_days)},
                )
            except Exception as _alert_exc:
                logger.error("[run_live_paper] run-gap alert failed: %s", _alert_exc)
    except Exception as exc:  # detection is best-effort, never block the cycle
        logger.warning("[run_live_paper] run-gap detection failed: %s", exc)


def _sync_trips_halt(sync_result, policy: dict) -> tuple[bool, str]:
    """W7b (2026-07-21, GESAMTBEWERTUNG): decide whether a failed broker
    sync trips the reconcile halt.

    The original gate only looked at cash_diff — pure position mismatches
    (qty diffs, symbols missing on either side) with a small cash_diff never
    halted (live-verified 2026-07-20: halt JSON reported mismatches_count=0
    while 5 symbols were missing from the ledger). Position truth is the
    core of the pilot: any non-dust mismatch (position_sync applies qty_tol)
    trips the halt as well.
    """
    tripped, reason = _mismatch_exceeds_threshold(
        sync_result.cash_diff,
        sync_result.broker_equity,
        policy,
    )
    if tripped:
        return tripped, reason
    # Stage-1 review B1 (2026-07-21): `mismatches` only carries qty diffs for
    # symbols present on BOTH sides (reconciliation.position_diffs_df).
    # Symbols the ledger does not know at all land in missing_in_ledger /
    # missing_in_broker — exactly the 2026-07-20 incident class (5 broker
    # positions unknown to the ledger). All three must trip.
    n_mismatch = len(sync_result.mismatches or [])
    n_missing_ledger = len(getattr(sync_result, "missing_in_ledger", []) or [])
    n_missing_broker = len(getattr(sync_result, "missing_in_broker", []) or [])
    if n_mismatch or n_missing_ledger or n_missing_broker:
        return True, (
            f"position_mismatches={n_mismatch} "
            f"missing_in_ledger={n_missing_ledger} "
            f"missing_in_broker={n_missing_broker} "
            f"(cash_diff ${sync_result.cash_diff:.2f} below threshold)"
        )
    return False, ""


def _market_open_for_submission(
    adapter, *, min_minutes_to_close: float = 10.0
) -> tuple[bool, str]:
    """K2a (2026-07-21, GESAMTBEWERTUNG): gate broker submissions on market hours.

    Root cause 2026-07-14: a delayed run submitted 5 market DAY orders at
    16:08 ET (after close). Alpaca queued them overnight and filled them at
    the next open, while the in-run 120s fill-wait timed out — the ledger
    never saw the fills (reconcile halt 2026-07-20). Submissions must only
    happen while the market is open AND far enough from the close that the
    fill-wait can complete.

    Primary source: the Alpaca clock via the adapter's underlying trading
    client (authoritative, holiday-aware; read-only private-attr access —
    the adapter exposes no public clock yet, see P4 follow-up). Fallback:
    local New-York wall-clock window Mon-Fri 09:30-15:50 ET (no holiday
    knowledge). Returns (is_open_for_submission, reason).
    """
    now_utc = datetime.now(timezone.utc)
    api = getattr(adapter, "_api", None)
    get_clock = getattr(api, "get_clock", None) if api is not None else None
    if callable(get_clock):
        try:
            clock = get_clock()
            if not bool(getattr(clock, "is_open", False)):
                return False, "alpaca_clock: market closed"
            next_close = getattr(clock, "next_close", None)
            if next_close is not None:
                try:
                    mins_to_close = (next_close - now_utc).total_seconds() / 60.0
                    if mins_to_close < min_minutes_to_close:
                        return False, (
                            f"alpaca_clock: only {mins_to_close:.1f}min to close "
                            f"(< {min_minutes_to_close:.0f}min buffer)"
                        )
                except TypeError:
                    logger.warning(
                        "[run_live_paper] market-hours gate: unparseable "
                        "next_close %r — ignoring close-buffer check",
                        next_close,
                    )
            return True, "alpaca_clock: open"
        except Exception as exc:
            logger.warning(
                "[run_live_paper] market-hours gate: Alpaca clock failed (%s) — "
                "falling back to local NY window",
                exc,
            )
    try:
        from zoneinfo import ZoneInfo

        ny = now_utc.astimezone(ZoneInfo("America/New_York"))
        if ny.weekday() >= 5:
            return False, "fallback: weekend (NY)"
        minutes = ny.hour * 60 + ny.minute
        if 570 <= minutes < 950:  # 09:30 <= t < 15:50 ET
            return True, "fallback: inside 09:30-15:50 ET window (no holiday check)"
        return False, f"fallback: outside 09:30-15:50 ET (NY time {ny:%H:%M})"
    except Exception as exc:
        # Cannot determine market state at all — fail closed for submissions.
        return False, f"fallback failed ({exc}) — fail-closed"


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
    _alert_on_run_gap(app_cfg)  # W12: detect silently missed runs

    # W15 (2026-07-23, GESAMTBEWERTUNG): book broker dividend payouts into
    # the ledger BEFORE the cycle/reconcile — otherwise Alpaca's real cash
    # dividends (TLT pays monthly) drift ledger<broker until the $100 gate
    # absorbs them into an "unexplained" halt contribution. Best-effort:
    # the script never raises on API failure; reconcile stays the backstop.
    # (args.dry_run directly: execution_mode is assigned further down.)
    if not args.dry_run:
        try:
            from scripts.ops.book_dividends import book_pending_dividends

            booked = book_pending_dividends()
            if booked:
                logger.info(
                    "[run_live_paper] booked %d broker dividend payout(s)", booked
                )
        except Exception as _div_exc:
            logger.warning(
                "[run_live_paper] dividend booking failed (non-blocking): %s",
                _div_exc,
            )

    _check_soft_timeout("post_preflight")

    # Reset per-cycle counters
    adapter.reset_cycle_counters()

    execution_mode = "dry_run" if args.dry_run else "broker"

    # K2a (2026-07-21): never submit into a closed or nearly-closed market —
    # root cause of the 2026-07-14 after-hours fills. Dry-run cycles may
    # proceed (no broker submission). Ordered skip = exit 0 (not a failure).
    if execution_mode == "broker" and not getattr(args, "allow_closed_market", False):
        market_open, clock_reason = _market_open_for_submission(adapter)
        if not market_open:
            logger.warning(
                "[run_live_paper] MARKET-HOURS GATE: skipping broker cycle — %s "
                "(override for testing: --allow-closed-market)",
                clock_reason,
            )
            if soft_timer is not None:
                soft_timer.cancel()
            sys.exit(0)
        logger.info("[run_live_paper] market-hours gate: %s", clock_reason)

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
                # W7b: cash thresholds OR non-dust position mismatches trip.
                tripped, reason = _sync_trips_halt(sync_result, policy)
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
    once_p.add_argument(
        "--allow-closed-market",
        action="store_true",
        help=(
            "K2a override: run a broker cycle even when the market-hours "
            "gate reports the market closed/near-close. Testing only — "
            "after-hours market DAY orders queue to the next open and "
            "bypass the in-run fill-wait (2026-07-14 incident)."
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
