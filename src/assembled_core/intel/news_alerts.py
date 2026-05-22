"""Alert system for critical news events.

Emits structured alerts when incoming NewsEvents cross configurable thresholds
(severity, surge velocity, corroboration etc.). Handlers are pluggable so the
same engine can wire into logging, Telegram, email or a push webhook without
changing the core logic.

Handlers are simple callables: `handler(alert: NewsAlert) -> None`.
A default logging handler is registered unless disabled.

Usage:
    engine = AlertEngine(min_severity=7.5)
    engine.add_handler(lambda a: print(a.kind, a.message))
    alerts = engine.evaluate(events)

Never enable real outbound handlers from defaults — keep push/Telegram behind
explicit opt-in.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable

logger = logging.getLogger(__name__)

_ALERT_CRITICAL = "critical"
_ALERT_SURGE = "surge"
_ALERT_CORROBORATED = "corroborated"
_ALERT_CONTRADICTION = "contradiction"

AlertHandler = Callable[["NewsAlert"], None]


@dataclass
class NewsAlert:
    kind: str
    event_id: str
    source_id: str
    severity: float
    message: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    extra: dict = field(default_factory=dict)


def _default_log_handler(alert: NewsAlert) -> None:
    logger.warning(
        "[ALERT] %s kind=%s src=%s sev=%.1f msg=%s",
        alert.event_id,
        alert.kind,
        alert.source_id,
        alert.severity,
        alert.message,
    )


class AlertEngine:
    """Threshold-based alert emitter with pluggable handlers."""

    def __init__(
        self,
        min_severity: float = 8.0,
        min_corroboration_score: float = 0.75,
        min_corroboration_sources: int = 3,
        include_default_log_handler: bool = True,
        dedup_window_min: float = 30.0,
        rate_limit_per_min: int = 20,
    ) -> None:
        self._min_sev = min_severity
        self._min_corr_score = min_corroboration_score
        self._min_corr_n = min_corroboration_sources
        self._handlers: list[AlertHandler] = []
        if include_default_log_handler:
            self._handlers.append(_default_log_handler)
        # H3: de-dup of (kind, event_id) within a sliding window.
        self._dedup_window = timedelta(minutes=max(0.0, dedup_window_min))
        self._recent_keys: dict[tuple[str, str], datetime] = {}
        # Rate-limit: last N dispatch timestamps; drops if >= limit in last 60s.
        self._rate_limit = max(0, int(rate_limit_per_min))
        self._dispatch_times: deque[datetime] = deque(
            maxlen=max(1, self._rate_limit * 2)
        )
        # Observability counters.
        self.dropped_dedup = 0
        self.dropped_rate = 0

    def add_handler(self, handler: AlertHandler) -> None:
        self._handlers.append(handler)

    def clear_handlers(self) -> None:
        self._handlers.clear()

    # ----- evaluation -------------------------------------------------

    def evaluate(self, events: list) -> list[NewsAlert]:
        candidate: list[NewsAlert] = []
        for evt in events or []:
            try:
                candidate.extend(self._alerts_for(evt))
            except Exception as exc:
                logger.debug("[SKIP] AlertEngine.evaluate: %s", exc)
        # H3: filter dedup + rate limit before dispatch
        emitted: list[NewsAlert] = []
        for a in candidate:
            if self._should_suppress(a):
                continue
            self._dispatch(a)
            emitted.append(a)
        return emitted

    def _should_suppress(self, alert: NewsAlert) -> bool:
        now = alert.timestamp or datetime.now(tz=timezone.utc)
        key = (alert.kind, alert.event_id)
        # prune stale dedup entries
        if self._dedup_window.total_seconds() > 0:
            cutoff = now - self._dedup_window
            self._recent_keys = {
                k: t for k, t in self._recent_keys.items() if t >= cutoff
            }
            if key in self._recent_keys:
                self.dropped_dedup += 1
                return True
            self._recent_keys[key] = now
        # rate limit
        if self._rate_limit > 0:
            minute_ago = now - timedelta(seconds=60)
            while self._dispatch_times and self._dispatch_times[0] < minute_ago:
                self._dispatch_times.popleft()
            if len(self._dispatch_times) >= self._rate_limit:
                self.dropped_rate += 1
                return True
            self._dispatch_times.append(now)
        return False

    def _alerts_for(self, evt) -> list[NewsAlert]:
        out: list[NewsAlert] = []
        severity = float(getattr(evt, "severity", 0.0) or 0.0)
        event_id = getattr(evt, "event_id", "") or ""
        source_id = getattr(evt, "source_id", "") or ""
        event_types = list(getattr(evt, "event_types", []) or [])
        corr_score = float(getattr(evt, "corroboration_score", 0.0) or 0.0)
        corr_n = int(getattr(evt, "corroboration_n_sources", 0) or 0)

        if severity >= self._min_sev:
            out.append(
                NewsAlert(
                    kind=_ALERT_CRITICAL,
                    event_id=event_id,
                    source_id=source_id,
                    severity=severity,
                    message=f"High-severity event: {event_types[:3]}",
                    extra={"event_types": event_types},
                )
            )
        if corr_score >= self._min_corr_score and corr_n >= self._min_corr_n:
            out.append(
                NewsAlert(
                    kind=_ALERT_CORROBORATED,
                    event_id=event_id,
                    source_id=source_id,
                    severity=severity,
                    message=f"Widely corroborated story: n={corr_n} score={corr_score:.2f}",
                    extra={"n_sources": corr_n, "score": corr_score},
                )
            )
        return out

    def surge_alert(
        self, ticker: str, velocity: float, event_ref: str = ""
    ) -> NewsAlert:
        a = NewsAlert(
            kind=_ALERT_SURGE,
            event_id=event_ref or f"surge_{ticker}",
            source_id="ticker_velocity",
            severity=min(10.0, velocity),
            message=f"Ticker velocity surge: {ticker} v={velocity:.2f}",
            extra={"ticker": ticker, "velocity": velocity},
        )
        self._dispatch(a)
        return a

    def contradiction_alert(
        self,
        story_key: str,
        split: str,
        severity: float = 5.0,
    ) -> NewsAlert:
        a = NewsAlert(
            kind=_ALERT_CONTRADICTION,
            event_id=story_key,
            source_id="contradiction_detector",
            severity=severity,
            message=f"Source contradiction: {split}",
            extra={"split": split},
        )
        self._dispatch(a)
        return a

    # ----- dispatch ---------------------------------------------------

    def _dispatch(self, alert: NewsAlert) -> None:
        for handler in self._handlers:
            try:
                handler(alert)
            except Exception as exc:
                logger.debug("[SKIP] AlertEngine handler %s: %s", handler, exc)


__all__ = ["AlertEngine", "NewsAlert"]
