"""API key rotation pool for rate-limited external providers.

Background (2026-05-22): we use four keyed-API providers — Alpha Vantage,
Polygon, NewsAPI, Finnhub — each with single-key tight rate limits on the
free tier (25 req/day for AV, 100 req/day for NewsAPI, 60 req/min for
Finnhub, 5 req/min for Polygon). Single-key consumption regularly hits
limits during daily ingestion runs. Multi-key pool with 429-driven cooldown
rotation lets us linearly scale daily/minutely quota without scraping
work-arounds.

Env-var convention (two equivalent forms, both supported):
    Form A (comma-separated list):
        ALPHAVANTAGE_API_KEYS="k1,k2,k3"  # pragma: allowlist secret
        POLYGON_API_KEYS="kA,kB"  # pragma: allowlist secret
    Form B (numbered, .env-friendly):
        ALPHAVANTAGE_API_KEY=key1
        ALPHAVANTAGE_API_KEY_2=key2
        ALPHAVANTAGE_API_KEY_3=key3
The pool is the union of both forms. The original single-key var
(``ALPHAVANTAGE_API_KEY``) is always read first to preserve backward
compat with all existing call sites that read it directly.

State persistence:
    output/ops/api_key_usage.json — per-provider per-key cooldown
    timestamps so cooldowns survive process restarts. Best-effort: a
    failed read returns empty state (no false cooldown), a failed write
    logs WARNING (next rotation re-attempts).

Typical use:
    from src.assembled_core.utils.api_key_rotator import get_rotator
    rotator = get_rotator()
    key = rotator.get_key("alphavantage")
    response = httpx.get(URL, params={"apikey": key, ...})
    if response.status_code == 429:
        rotator.mark_rate_limited("alphavantage", key, cooldown_seconds=3600)
        key = rotator.get_key("alphavantage")  # try next
        response = httpx.get(URL, params={"apikey": key, ...})
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

_DEFAULT_STATE_PATH = (
    Path(__file__).resolve().parents[3] / "output" / "ops" / "api_key_usage.json"
)

# Canonical provider-name → env-var-prefix mapping. The single-key var
# `<PREFIX>_API_KEY` is read first for backward compat with existing call
# sites; additional keys come from `<PREFIX>_API_KEY_2/3/...` and/or the
# plural `<PREFIX>_API_KEYS` comma-separated form.
_PROVIDER_ENV_PREFIX: dict[str, str] = {
    "alphavantage": "ALPHAVANTAGE",
    "polygon": "POLYGON",
    "newsapi": "NEWSAPI",
    "finnhub": "FINNHUB",
}


#: Legacy aliases per provider — read AFTER the canonical _API_KEY but
#: BEFORE numbered/plural forms. Documented in `.env.example`.
_PROVIDER_LEGACY_ALIASES: dict[str, list[str]] = {
    "alphavantage": ["ALPHAVANTAGE_KEY"],
    "newsapi": ["NEWSAPI_KEY"],
    "finnhub": ["ASSEMBLED_FINNHUB_API_KEY"],
    "polygon": [],
}


def _discover_keys_for_provider(provider: str) -> list[str]:
    """Read keys for `provider` from env vars in the documented convention.

    Order of discovery (each candidate de-duplicated):
        1. ``<PREFIX>_API_KEY``                (canonical single)
        2. Legacy aliases (see ``_PROVIDER_LEGACY_ALIASES``)
        3. ``<PREFIX>_API_KEY_2`` … ``_API_KEY_31``  (numbered)
        4. ``<PREFIX>_API_KEYS``               (comma-separated plural)

    Returns a de-duplicated list preserving discovery order. Whitespace-only
    keys are dropped silently. Empty result means no keys configured.
    """
    prefix = _PROVIDER_ENV_PREFIX.get(provider)
    if prefix is None:
        return []

    keys: list[str] = []
    seen: set[str] = set()

    def _add(candidate: str | None) -> None:
        if candidate is None:
            return
        candidate = candidate.strip()
        if not candidate or candidate in seen:
            return
        keys.append(candidate)
        seen.add(candidate)

    # 1. Canonical single key
    _add(os.environ.get(f"{prefix}_API_KEY"))

    # 2. Legacy aliases (read second so the canonical wins on dedup)
    for alias in _PROVIDER_LEGACY_ALIASES.get(provider, []):
        _add(os.environ.get(alias))

    # 3. Numbered: _API_KEY_2, _API_KEY_3, ... up to a sane ceiling
    for i in range(2, 32):
        _add(os.environ.get(f"{prefix}_API_KEY_{i}"))

    # 4. Comma-separated plural
    plural_raw = os.environ.get(f"{prefix}_API_KEYS", "")
    if plural_raw:
        for token in plural_raw.split(","):
            _add(token)

    return keys


@dataclass
class _KeyState:
    """Per-key cooldown bookkeeping."""

    key: str
    cooldown_until_epoch: float = 0.0
    last_used_epoch: float = 0.0
    rate_limit_hits: int = 0

    def is_cooled_down(self, now_epoch: float) -> bool:
        return self.cooldown_until_epoch <= now_epoch

    def to_serializable(self) -> dict:
        # We never serialize the key itself to disk — only an HMAC-like
        # fingerprint (last 4 chars) so a leaked state file does not
        # expose secrets. The rotator re-resolves the actual key from
        # env on each process start, so the fingerprint is just an
        # alignment beacon.
        return {
            "key_suffix": self.key[-4:] if len(self.key) >= 4 else "***",
            "cooldown_until_epoch": self.cooldown_until_epoch,
            "last_used_epoch": self.last_used_epoch,
            "rate_limit_hits": self.rate_limit_hits,
        }


@dataclass
class ApiKeyRotator:
    """Pool-based key rotator for rate-limited providers.

    Single instance is fine for the entire process — use ``get_rotator()``.
    Thread-safe (internal lock) for the common pilot pattern of
    sequential calls across heterogeneous providers.
    """

    state_path: Path = field(default_factory=lambda: _DEFAULT_STATE_PATH)
    _pools: dict[str, list[_KeyState]] = field(default_factory=dict)
    _cursors: dict[str, int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _loaded: bool = False

    def _ensure_pool(self, provider: str) -> list[_KeyState]:
        """Lazy-initialize the pool from env on first access."""
        if provider in self._pools:
            return self._pools[provider]
        keys = _discover_keys_for_provider(provider)
        pool = [_KeyState(key=k) for k in keys]
        self._pools[provider] = pool
        self._cursors[provider] = 0
        if not pool:
            logger.debug(
                "[api-key-rotator] provider=%s has no keys configured "
                "(checked %s_API_KEY[_N] and %s_API_KEYS)",
                provider,
                _PROVIDER_ENV_PREFIX.get(provider, provider.upper()),
                _PROVIDER_ENV_PREFIX.get(provider, provider.upper()),
            )
        else:
            logger.info(
                "[api-key-rotator] provider=%s pool size=%d",
                provider,
                len(pool),
            )
        return pool

    def _load_state_once(self) -> None:
        """Merge persisted cooldown timestamps onto in-memory pools."""
        if self._loaded:
            return
        self._loaded = True
        try:
            if not self.state_path.exists():
                return
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[api-key-rotator] failed to read state from %s: %s — "
                "starting with empty cooldown state",
                self.state_path,
                exc,
            )
            return

        # Merge: for each provider, look up the in-memory keys by suffix
        # and copy cooldown_until_epoch from persisted entries.
        for provider, entries in (payload or {}).items():
            pool = self._ensure_pool(provider)
            if not pool:
                continue
            by_suffix: dict[str, _KeyState] = {
                ks.key[-4:] if len(ks.key) >= 4 else "***": ks for ks in pool
            }
            for entry in entries or []:
                suffix = entry.get("key_suffix") or "***"
                state = by_suffix.get(suffix)
                if state is None:
                    continue
                state.cooldown_until_epoch = float(
                    entry.get("cooldown_until_epoch") or 0.0
                )
                state.last_used_epoch = float(entry.get("last_used_epoch") or 0.0)
                state.rate_limit_hits = int(entry.get("rate_limit_hits") or 0)

    def _persist_state(self) -> None:
        """Write current cooldown state atomically."""
        try:
            payload = {
                provider: [ks.to_serializable() for ks in pool]
                for provider, pool in self._pools.items()
            }
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.state_path.with_name(self.state_path.name + ".tmp")
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp.replace(self.state_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[api-key-rotator] failed to persist state to %s: %s",
                self.state_path,
                exc,
            )

    def get_key(self, provider: str) -> str | None:
        """Return the next available (cooled-down) key for ``provider``.

        Returns None when the pool is empty OR every key is currently
        cooled down. Callers must handle the None case (defer, log,
        or fail-loud).
        """
        with self._lock:
            self._load_state_once()
            pool = self._ensure_pool(provider)
            if not pool:
                return None
            now = time.time()
            # Round-robin: start at the saved cursor, scan the whole pool.
            start = self._cursors.get(provider, 0) % len(pool)
            for offset in range(len(pool)):
                idx = (start + offset) % len(pool)
                state = pool[idx]
                if state.is_cooled_down(now):
                    state.last_used_epoch = now
                    self._cursors[provider] = (idx + 1) % len(pool)
                    return state.key
            # All keys cooled down — log once per call and return None.
            next_avail = min(ks.cooldown_until_epoch for ks in pool)
            wait_s = max(0.0, next_avail - now)
            logger.warning(
                "[api-key-rotator] provider=%s — all %d key(s) cooled down; "
                "next available in %.1fs",
                provider,
                len(pool),
                wait_s,
            )
            return None

    def mark_rate_limited(
        self,
        provider: str,
        key: str,
        *,
        cooldown_seconds: float = 3600.0,
    ) -> None:
        """Mark ``key`` cooled-down for ``cooldown_seconds``.

        Increments rate_limit_hits counter for observability. Best-effort
        persist — write failure logs WARNING but doesn't raise.

        Default cooldown 1h matches AV's typical daily-bucket reset cadence.
        Caller can override with a shorter value for rapid-rotation cases
        (e.g. Polygon's per-minute limit → cooldown_seconds=60).
        """
        with self._lock:
            self._load_state_once()
            pool = self._ensure_pool(provider)
            now = time.time()
            for state in pool:
                if state.key == key:
                    state.cooldown_until_epoch = now + float(cooldown_seconds)
                    state.rate_limit_hits += 1
                    logger.warning(
                        "[api-key-rotator] provider=%s key=…%s rate-limited; "
                        "cooled down for %.0fs (total hits=%d)",
                        provider,
                        key[-4:] if len(key) >= 4 else "***",
                        cooldown_seconds,
                        state.rate_limit_hits,
                    )
                    self._persist_state()
                    return
            logger.warning(
                "[api-key-rotator] mark_rate_limited called for provider=%s "
                "with unknown key …%s — ignored",
                provider,
                key[-4:] if len(key) >= 4 else "***",
            )

    def pool_size(self, provider: str) -> int:
        """Total keys configured for the provider (for diagnostics)."""
        with self._lock:
            return len(self._ensure_pool(provider))

    def available_count(self, provider: str) -> int:
        """Currently non-cooled-down keys (for diagnostics)."""
        with self._lock:
            self._load_state_once()
            pool = self._ensure_pool(provider)
            now = time.time()
            return sum(1 for ks in pool if ks.is_cooled_down(now))

    def reset(self) -> None:
        """Clear in-memory pools so next access re-reads env. Test-only."""
        with self._lock:
            self._pools.clear()
            self._cursors.clear()
            self._loaded = False


_singleton_lock = threading.Lock()
_singleton: ApiKeyRotator | None = None


def get_rotator() -> ApiKeyRotator:
    """Module-level singleton rotator. Re-uses the same pool across the
    process lifetime, which is necessary for cooldown bookkeeping to
    have any effect (each client invocation must consult the same state).
    """
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            _singleton = ApiKeyRotator()
        return _singleton


def reset_singleton() -> None:
    """Test-only helper: drop the module-level rotator so the next
    ``get_rotator()`` re-reads env vars. Production code should never
    need this."""
    global _singleton
    with _singleton_lock:
        _singleton = None


def known_providers() -> Iterable[str]:
    """List of provider names this rotator understands."""
    return tuple(_PROVIDER_ENV_PREFIX.keys())
