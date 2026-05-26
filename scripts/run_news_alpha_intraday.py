"""Intraday news-alpha runner.

Polls RSS feeds every POLL_INTERVAL_SEC during US market hours (09:30–16:00 ET).
For each batch of new headlines:
  1. Classify via news_classifier → event_types + severity (0-10 float)
  2. Map to news_alpha topic_id (shipping_disruption / energy_crisis / etc.)
  3. Build trigger items for severity >= min_severity
  4. Run run_news_alpha_pipeline() — generates signals, checks exits
  5. If --live: submit Alpaca market orders; default: shadow mode (log only)

State (open signals, seen event IDs, day counter) is persisted in
output/news_alpha_state.json across poll cycles and restarts.

Usage:
    python scripts/run_news_alpha_intraday.py                # shadow mode
    python scripts/run_news_alpha_intraday.py --live         # real Alpaca orders
    python scripts/run_news_alpha_intraday.py --poll 120     # 2-min interval
    python scripts/run_news_alpha_intraday.py --min-severity 3  # critical only
    python scripts/run_news_alpha_intraday.py --no-market-hours-check  # dev/testing
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import fields as dc_fields
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import warnings

warnings.filterwarnings("ignore")

try:
    from zoneinfo import ZoneInfo

    _ET = ZoneInfo("America/New_York")
except ImportError:
    import pytz  # type: ignore[import]

    _ET = pytz.timezone("America/New_York")

import yfinance as yf

from src.assembled_core.events.news_alpha.asset_router import get_route
from src.assembled_core.events.news_alpha.models import NewsAlphaSignal
from src.assembled_core.events.news_alpha.pipeline import run_news_alpha_pipeline
from src.assembled_core.execution.broker_adapter import AlpacaAdapter, BrokerAdapter
from src.assembled_core.intel.news_classifier import classify_news_event
from src.assembled_core.intel.rss_fetcher import RSSFetcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# INFO-2: absolute path so the file lands in the right place regardless of CWD
_STATE_FILE = Path(__file__).parent.parent / "output" / "news_alpha_state.json"

_DEFAULT_POLICY: dict = {
    "news_alpha": {
        "enabled": True,
        "base_weight": 0.06,
        "leverage_etfs_allowed": False,
        "max_gross_exposure": 0.30,
    }
}

# Maximum seen-event-IDs to keep in state (rolling window)
_MAX_SEEN_IDS = 10_000


def _update_seen_ids(
    seen_ids_list: list[str],
    seen_ids_set: set[str],
    new_ids: list[str],
) -> tuple[list[str], set[str]]:
    """Add new_ids to the ordered list+set; trim if over _MAX_SEEN_IDS.

    Returns the updated (list, set). The list preserves insertion order so
    trim always retains the most recently added IDs.
    """
    for nid in new_ids:
        if nid not in seen_ids_set:
            seen_ids_list.append(nid)
            seen_ids_set.add(nid)
    if len(seen_ids_list) > _MAX_SEEN_IDS:
        seen_ids_list = seen_ids_list[-(_MAX_SEEN_IDS // 2) :]
        seen_ids_set = set(seen_ids_list)
    return seen_ids_list, seen_ids_set


# ---------------------------------------------------------------------------
# Topic classification: headline text → router topic_id
# Priority order matters — more specific topics come first.
# ---------------------------------------------------------------------------

_TOPIC_RULES: list[tuple[str, tuple[str, ...]]] = [
    (
        "shipping_disruption",
        (
            "hormuz",
            "strait of hormuz",
            "suez canal",
            "red sea attack",
            "tanker seized",
            "tanker attacked",
            "shipping lane",
            "houthi attack",
            "houthi missile",
            "bab el-mandeb",
            "piracy attack",
            "vessel seized",
            "oil tanker",
            "shipping route",
        ),
    ),
    (
        "taiwan_strait",
        (
            "taiwan strait",
            "taiwan crisis",
            "pla taiwan",
            "china taiwan",
            "tsmc blockade",
            "taipei tension",
            "military exercises taiwan",
            "chinese military taiwan",
        ),
    ),
    (
        "nuclear_risk",
        (
            "nuclear weapon",
            "nuclear threat",
            "nuclear strike",
            "nuclear war",
            "warhead deployed",
            "icbm launch",
            "nuclear arsenal",
            "dirty bomb",
            "radiation emergency",
            "nuclear deterrence",
        ),
    ),
    (
        "market_crash",
        (
            "market crash",
            "circuit breaker",
            "flash crash",
            "stock market crash",
            "market meltdown",
            "vix surges",
            "black monday",
            "dow plunges",
            "nasdaq crashes",
            "market panic",
            "trading halt",
            "limit down",
        ),
    ),
    (
        "energy_crisis",
        (
            "opec cuts",
            "opec production",
            "aramco attack",
            "aramco drone",
            "oil supply cut",
            "pipeline explosion",
            "gas shortage",
            "energy crisis",
            "brent crude",
            "wti crude",
            "oil embargo",
            "lng shortage",
            "refinery attack",
        ),
    ),
    (
        "geopolitical_conflict",
        (
            "invasion",
            "invades",
            "military offensive",
            "airstrike",
            "airstrikes",
            "missile strike",
            "war declared",
            "troops advance",
            "frontline",
            "ground offensive",
            "war escalat",
            "conflict escalat",
            "ceasefire collapse",
        ),
    ),
    (
        "central_bank",
        (
            "federal reserve raises",
            "fed raises",
            "fed hikes",
            "fomc hike",
            "fomc cut",
            "rate hike",
            "rate cut",
            "fed cuts",
            "emergency rate",
            "basis point",
            "interest rate decision",
            "ecb raises",
            "ecb cuts",
        ),
    ),
]


def _headline_to_topic_id(headline: str) -> str | None:
    """Map headline text to news_alpha router topic_id, or None if no match."""
    h = headline.lower()
    for topic_id, keywords in _TOPIC_RULES:
        if any(kw in h for kw in keywords):
            return topic_id
    return None


def _severity_float_to_int(s: float) -> int:
    """Convert 0-10 classifier severity to 1-3 router severity."""
    if s >= 7.0:
        return 3
    if s >= 4.0:
        return 2
    return 1


# ---------------------------------------------------------------------------
# Market hours
# ---------------------------------------------------------------------------


def _is_market_hours(now_et: datetime | None = None) -> bool:
    """Return True if current ET time is within NYSE regular session."""
    et = now_et or datetime.now(_ET)
    if et.weekday() >= 5:
        return False
    open_ = et.replace(hour=9, minute=30, second=0, microsecond=0)
    close_ = et.replace(hour=16, minute=0, second=0, microsecond=0)
    return open_ <= et < close_


# ---------------------------------------------------------------------------
# Price fetching (yfinance 1-minute bars, last close)
# ---------------------------------------------------------------------------


def _get_prices(symbols: list[str]) -> dict[str, float]:
    if not symbols:
        return {}
    unique = list(set(symbols))
    try:
        raw = yf.download(
            unique,
            period="1d",
            interval="1m",
            auto_adjust=True,
            progress=False,
        )
        if raw.empty:
            return {}

        # MAJOR-2: yfinance returns flat columns for a single symbol and
        # MultiIndex columns for multiple symbols.  Normalise to a DataFrame
        # with symbol columns in both cases.
        import pandas as pd  # already available via yfinance dep

        if isinstance(raw.columns, pd.MultiIndex):
            close_df = raw["Close"]
        else:
            # Single-symbol flat response: columns are ["Close", "High", ...]
            if "Close" not in raw.columns:
                return {}
            sym = unique[0]
            close_df = raw[["Close"]].rename(columns={"Close": sym})

        prices: dict[str, float] = {}
        for sym in unique:
            if sym in close_df.columns:
                series = close_df[sym].dropna()
                if len(series) > 0:
                    prices[sym] = float(series.iloc[-1])
        return prices
    except Exception as exc:
        logger.warning("[WARN] price fetch failed: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

_SIG_FIELDS = {f.name for f in dc_fields(NewsAlphaSignal)}


def _signal_to_dict(sig: NewsAlphaSignal) -> dict:
    return {f.name: getattr(sig, f.name) for f in dc_fields(sig)}


def _dict_to_signal(d: dict) -> NewsAlphaSignal:
    return NewsAlphaSignal(**{k: v for k, v in d.items() if k in _SIG_FIELDS})


def _load_state() -> dict:
    if _STATE_FILE.exists():
        try:
            return json.loads(_STATE_FILE.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("[WARN] state load failed (%s) — starting fresh", exc)
    return {
        "open_signals": [],
        "seen_event_ids": [],
        "day_counter": 0,
        "last_date": "",
    }


def _save_state(state: dict) -> None:
    _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write: write to .tmp then rename so a crash mid-write never corrupts state.
    _tmp = _STATE_FILE.with_suffix(".tmp")
    _tmp.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
    _tmp.replace(_STATE_FILE)


# ---------------------------------------------------------------------------
# Trigger classification from RSS events
# ---------------------------------------------------------------------------


def _events_to_triggers(
    events: list,
    seen_event_ids: set[str],
    min_severity: int,
) -> tuple[list[dict], list[str]]:
    """Classify RSS events into news_alpha trigger dicts.

    Returns (triggers, new_event_ids_to_mark_seen).
    New event IDs are returned separately so the caller can decide when to commit them.
    """
    triggers: list[dict] = []
    new_ids: list[str] = []

    for ev in events:
        eid = ev.content_hash or ev.url or ev.title[:64]
        if eid in seen_event_ids:
            continue

        # MAJOR-1: topic match first, BEFORE marking seen.
        # Non-matching events are NOT marked seen so they can be re-evaluated
        # on the next cycle (e.g., if the headline is updated by the source).
        # Events that do match are marked seen to prevent re-firing.
        topic_id = _headline_to_topic_id(ev.title)
        if topic_id is None:
            continue  # no route match — don't mark seen, try again next cycle

        # Use pre-classified severity if available, else run keyword classifier
        if ev.severity > 0:
            sev_float = ev.severity
        else:
            clf = classify_news_event(
                ev.title,
                geo_tags=list(ev.geo_tags) if ev.geo_tags else [],
                source_tier=str(ev.source_tier.value) if ev.source_tier else "T2",
            )
            sev_float = clf.severity

        # A keyword match from _headline_to_topic_id already signals relevance.
        # Apply a minimum so matched events are not silently filtered — the pipeline
        # then applies its own per-route min_severity gate.
        # urgency > 0 means "BREAKING / FLASH / URGENT" was in the title → floor=7
        urgency = getattr(ev, "urgency", 0.0) or 0.0
        if urgency > 0.5:
            sev_float = max(sev_float, 7.0)
        else:
            sev_float = max(sev_float, 4.0)  # matched keyword => at least sev=2

        int_sev = _severity_float_to_int(sev_float)
        if int_sev < min_severity:
            continue

        # Mark seen only after all checks pass — prevents permanently suppressing
        # events that matched topic but failed severity (could reclassify next cycle)
        new_ids.append(eid)
        triggers.append(
            {
                "severity": int_sev,
                "topic": topic_id,
                "source": str(ev.source_id or "rss"),
                # headline in details so split_central_bank_topic can use it
                "details": ev.title,
                "event_id": eid,
            }
        )

    return triggers, new_ids


# ---------------------------------------------------------------------------
# Order execution helpers
# ---------------------------------------------------------------------------


_MIN_PRICE_SANITY = 0.50  # M-2: skip order if price looks wrong
_MAX_SINGLE_POSITION_WEIGHT = 0.25  # M-2: per-symbol notional cap vs equity


def _execute_entries(
    target_weights: dict[str, float],
    signals: list[NewsAlphaSignal],
    prices: dict[str, float],
    broker: BrokerAdapter,
    account_equity: float,
) -> set[str]:
    """Submit entry orders. Returns symbols for which the order succeeded.

    MINOR-2: caller must only add result.signals to open_signals for symbols in
    the returned set — failed entries have no broker position to track.
    """
    entered: set[str] = set()
    for sym, weight in target_weights.items():
        price = prices.get(sym, 0.0)
        if price <= 0.0:
            logger.warning("[WARN] no price for %s — entry skipped", sym)
            continue
        # M-2: sanity floor — stale/bad tick guard
        if price < _MIN_PRICE_SANITY:
            logger.warning(
                "[WARN] price %.4f for %s looks stale — entry skipped", price, sym
            )
            continue
        # M-2: cap per-order notional as additional fat-finger guard
        capped_weight = min(abs(weight), _MAX_SINGLE_POSITION_WEIGHT)
        notional = capped_weight * account_equity
        qty = max(1.0, round(notional / price))
        try:
            order = broker.submit_market_order(
                sym, qty, "buy", comment=f"news_alpha_entry:{sym}"
            )
            logger.info(
                "[OK] ENTRY BUY %s x%.0f @ market | order_id=%s",
                sym,
                qty,
                order.order_id,
            )
            entered.add(sym)
            # Store entry price on signal for stop/tp checks
            for sig in signals:
                if sig.symbol == sym and sig.entry_price <= 0.0:
                    sig.entry_price = price
        except Exception as exc:
            logger.error("[ERROR] entry order %s failed — not tracked: %s", sym, exc)
    return entered


def _execute_exits(
    positions_to_exit: list[tuple[NewsAlphaSignal, str]],
    broker: BrokerAdapter,
) -> None:
    broker_positions = {p.symbol: p.qty for p in broker.get_positions()}
    for sig, reason in positions_to_exit:
        sym = sig.symbol
        qty = broker_positions.get(sym, 0.0)
        # M-3: mark inactive BEFORE submission attempt so that a network timeout
        # on the order call does not leave signal active for re-submission next cycle.
        sig.active = False
        if qty <= 0.0:
            logger.info(
                "[SKIP] EXIT %s: no broker position (orphaned signal deactivated)", sym
            )
            continue
        try:
            order = broker.submit_market_order(
                sym, qty, "sell", comment=f"news_alpha_exit:{reason[:40]}"
            )
            logger.info(
                "[OK] EXIT SELL %s x%.0f | reason=%s | order_id=%s",
                sym,
                qty,
                reason,
                order.order_id,
            )
        except Exception as exc:
            logger.error(
                "[ERROR] exit order %s failed (signal deactivated to prevent re-try): %s",
                sym,
                exc,
            )


# ---------------------------------------------------------------------------
# Main polling loop
# ---------------------------------------------------------------------------


def run_loop(
    *,
    shadow_only: bool = True,
    poll_interval: int = 300,
    min_severity: int = 2,
    enforce_market_hours: bool = True,
    policy: dict | None = None,
) -> None:
    effective_policy = policy or _DEFAULT_POLICY

    broker: AlpacaAdapter | None = None
    if not shadow_only:
        # B-1: --live submits real orders to the Alpaca PAPER trading account.
        # To use a live (non-paper) Alpaca account, set ALPACA_BASE_URL to the
        # live endpoint AND instantiate AlpacaAdapter(force_paper=False) manually.
        broker = AlpacaAdapter()  # defaults to paper URL + force_paper=True
        health = broker.health_check()
        if not health["ok"]:
            logger.error("[ERROR] Alpaca health check failed: %s", health["message"])
            sys.exit(1)
        target = "PAPER" if broker.is_paper else "LIVE"
        logger.info(
            "[OK] Alpaca connected | target=%s | equity=%s",
            target,
            health.get("account_equity"),
        )
        if not broker.is_paper:
            logger.warning("[WARN] Connected to LIVE Alpaca endpoint — orders are REAL")

    fetcher = RSSFetcher()
    state = _load_state()
    seen_ids_list: list[str] = list(state.get("seen_event_ids", []))
    seen_event_ids: set[str] = set(seen_ids_list)
    open_signals: list[NewsAlphaSignal] = [
        _dict_to_signal(d) for d in state.get("open_signals", [])
    ]
    day_counter: int = int(state.get("day_counter", 0))
    last_date: str = state.get("last_date", "")

    # M-1: warn about restored signals with no entry_price — they may be orphans
    # (e.g., entry order failed but signal was persisted before the error was caught)
    orphaned = [s for s in open_signals if s.active and s.entry_price <= 0.0]
    if orphaned:
        logger.warning(
            "[WARN] %d restored signal(s) have entry_price=0 (possible orphans): %s",
            len(orphaned),
            [s.symbol for s in orphaned],
        )

    mode = "LIVE" if not shadow_only else "SHADOW"
    logger.info(
        "[START] news_alpha intraday | mode=%s | poll=%ds | min_severity=%d | open_positions=%d",
        mode,
        poll_interval,
        min_severity,
        len(open_signals),
    )
    # MAJOR-B: warn when policy.yaml shadow_only=false but runner is in shadow mode
    # (two independent gates — operators reading policy.yaml may expect live orders).
    if shadow_only and not (effective_policy or {}).get("news_alpha", {}).get(
        "shadow_only", True
    ):
        logger.warning(
            "[WARN] policy.yaml news_alpha.shadow_only=false but runner is in SHADOW mode"
            " — the EOD pipeline will apply positions; this runner will NOT submit"
            " intraday orders. Pass --live to enable intraday order submission."
        )

    while True:
        now_utc = datetime.now(timezone.utc)
        now_et = datetime.now(_ET)

        # Day counter: increment when calendar date changes during market hours
        today_str = now_et.strftime("%Y-%m-%d")
        if last_date and last_date != today_str and _is_market_hours(now_et):
            day_counter += 1
            last_date = today_str
            logger.info("[OK] new trading day — day_counter=%d", day_counter)
        elif not last_date:
            last_date = today_str

        if enforce_market_hours and not _is_market_hours(now_et):
            logger.debug(
                "[WAIT] %s ET — outside market hours, sleeping %ds",
                now_et.strftime("%H:%M"),
                poll_interval,
            )
            time.sleep(poll_interval)
            continue

        try:
            # --- 1. Fetch RSS events ---
            events = fetcher.fetch_all()

            # --- 2. Classify into triggers ---
            triggers, new_ids = _events_to_triggers(
                events, seen_event_ids, min_severity
            )
            seen_ids_list, seen_event_ids = _update_seen_ids(
                seen_ids_list, seen_event_ids, new_ids
            )

            # --- 3. Collect symbols needed for price fetch ---
            price_syms: set[str] = {sig.symbol for sig in open_signals}
            for t in triggers:
                route = get_route(t.get("topic"))
                if route:
                    price_syms.update(route.get("long_etfs", []))
                    price_syms.update(route.get("long_etfs_2x", []))
                    price_syms.update(route.get("inverse_etfs", []))
                    price_syms.update(route.get("inverse_etfs_2x", []))

            prices = _get_prices(list(price_syms)) if price_syms else {}

            # --- 4. Run news_alpha pipeline ---
            result = run_news_alpha_pipeline(
                trigger_items=triggers,
                open_signals=[s for s in open_signals if s.active],
                current_day=day_counter,
                prices=prices,
                policy=effective_policy,
                shadow_only=shadow_only,
                timestamp_utc=now_utc,
            )

            # --- 5. Execute ---
            entered_symbols: set[str] = set()
            if not shadow_only and broker is not None:
                if result.target_weights:
                    acct = broker.get_account()
                    equity = float(acct.get("equity", 100_000))
                    entered_symbols = _execute_entries(
                        result.target_weights, result.signals, prices, broker, equity
                    )
                if result.positions_to_exit:
                    _execute_exits(result.positions_to_exit, broker)

            # --- 6. Update signal registry ---
            # MINOR-2: in live mode only track signals for which entry succeeded.
            # In shadow mode all signals are tracked (no real order placed).
            for sig in result.signals:
                if shadow_only or sig.symbol in entered_symbols:
                    open_signals.append(sig)
            open_signals = [s for s in open_signals if s.active]

            # --- 7. Persist state ---
            _save_state(
                {
                    "open_signals": [_signal_to_dict(s) for s in open_signals],
                    "seen_event_ids": seen_ids_list,
                    "day_counter": day_counter,
                    "last_date": last_date,
                    "last_poll_utc": now_utc.isoformat(),
                }
            )

            # --- 8. Summary log ---
            if triggers or result.signals or result.positions_to_exit:
                logger.info(
                    "[OK] %s ET | new_events=%d | triggers=%d | new_signals=%d | exits=%d | open=%d",
                    now_et.strftime("%H:%M:%S"),
                    len(new_ids),
                    len(triggers),
                    len(result.signals),
                    len(result.positions_to_exit),
                    len(open_signals),
                )
                if triggers:
                    for t in triggers:
                        logger.info(
                            "  >> trigger: topic=%s sev=%d | %s",
                            t["topic"],
                            t["severity"],
                            t.get("details", "")[:80],
                        )
            else:
                logger.debug(
                    "[OK] %s ET | new_events=%d | quiet",
                    now_et.strftime("%H:%M:%S"),
                    len(new_ids),
                )

        except KeyboardInterrupt:
            logger.info("[STOP] interrupted — saving state")
            _save_state(
                {
                    "open_signals": [_signal_to_dict(s) for s in open_signals],
                    "seen_event_ids": seen_ids_list[-(_MAX_SEEN_IDS // 2) :],
                    "day_counter": day_counter,
                    "last_date": last_date,
                }
            )
            break
        except Exception as exc:
            logger.exception("[ERROR] poll cycle failed: %s", exc)

        time.sleep(poll_interval)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Intraday news-alpha polling runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Submit real Alpaca orders (default: shadow mode, log only)",
    )
    parser.add_argument(
        "--poll",
        type=int,
        default=300,
        metavar="SECONDS",
        help="Poll interval in seconds",
    )
    parser.add_argument(
        "--min-severity",
        type=int,
        default=2,
        choices=[1, 2, 3],
        help="Minimum event severity to fire (1=watch, 2=elevated, 3=critical)",
    )
    parser.add_argument(
        "--leverage",
        action="store_true",
        help="Allow leveraged ETFs (UCO, TBT, UVXY) in signals",
    )
    parser.add_argument(
        "--no-market-hours-check",
        action="store_true",
        help="Poll even outside 09:30-16:00 ET (dev/testing mode)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # F-001 fix: load configs/policy.yaml so base_weight/max_gross_exposure stay
    # in sync with the EOD pipeline.  Fall back to _DEFAULT_POLICY if unavailable.
    _policy_path = Path(__file__).parent.parent / "configs" / "policy.yaml"
    try:
        import yaml  # noqa: PLC0415

        policy: dict = yaml.safe_load(_policy_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        logger.warning("[WARN] could not load policy.yaml (%s) — using defaults", exc)
        policy = _DEFAULT_POLICY.copy()

    # --leverage CLI flag overrides policy.yaml leverage_etfs_allowed so operators
    # can enable leverage for a single run without editing the config file.
    if args.leverage:
        policy.setdefault("news_alpha", {})["leverage_etfs_allowed"] = True

    # --live controls broker order submission; this is independent of policy.yaml
    # shadow_only (which gates the EOD pipeline path in _tc_sizing.py).
    run_loop(
        shadow_only=not args.live,
        poll_interval=args.poll,
        min_severity=args.min_severity,
        enforce_market_hours=not args.no_market_hours_check,
        policy=policy,
    )


if __name__ == "__main__":
    main()
