"""Signal plugin registry — loads BaseSignal implementations via entry-points.

From 33_EXECUTION_ORDERMANAGEMENT.md §33.13.

Entry-point group: "ata.signals"

Example pyproject.toml entry in a signal package:
    [project.entry-points."ata.signals"]
    pead = "ata_signal_pead.signal:PEADSignal"
"""

from __future__ import annotations

import logging
from importlib.metadata import entry_points
from typing import Iterator

from src.assembled_core.signals.base import BaseSignal

logger = logging.getLogger(__name__)


class SignalRegistry:
    """Loads and holds all registered BaseSignal plugins."""

    _EP_GROUP = "ata.signals"

    def __init__(self) -> None:
        self._signals: dict[str, BaseSignal] = {}
        self._load_errors: dict[str, str] = {}

    def load_all(self) -> int:
        """Load all entry-point signals.  Returns count of loaded signals."""
        loaded = 0
        try:
            eps = entry_points(group=self._EP_GROUP)
        except Exception as exc:
            logger.warning("entry_points query failed: %s", exc)
            return 0

        for ep in eps:
            try:
                cls = ep.load()
                inst: BaseSignal = cls()
                if inst.name in self._signals:
                    logger.warning("Duplicate signal name %r from %s — skipped", inst.name, ep.name)
                    continue
                self._signals[inst.name] = inst
                logger.info("Loaded signal: %s v%s", inst.name, inst.version)
                loaded += 1
            except Exception as exc:
                self._load_errors[ep.name] = str(exc)
                logger.error("Failed to load signal %s: %s", ep.name, exc)
        return loaded

    def register(self, signal: BaseSignal) -> None:
        """Manually register a signal instance (useful in tests)."""
        if signal.name in self._signals:
            raise ValueError(f"Signal {signal.name!r} already registered")
        self._signals[signal.name] = signal

    def get(self, name: str) -> BaseSignal | None:
        return self._signals.get(name)

    def all(self) -> list[BaseSignal]:
        return list(self._signals.values())

    def names(self) -> list[str]:
        return sorted(self._signals.keys())

    def errors(self) -> dict[str, str]:
        """Return load errors from the last load_all() call."""
        return dict(self._load_errors)

    def __iter__(self) -> Iterator[BaseSignal]:
        return iter(self._signals.values())

    def __len__(self) -> int:
        return len(self._signals)


_default_registry: SignalRegistry | None = None


def get_registry() -> SignalRegistry:
    """Return the process-level default registry (lazy singleton)."""
    global _default_registry
    if _default_registry is None:
        _default_registry = SignalRegistry()
        _default_registry.load_all()
    return _default_registry


__all__ = ["SignalRegistry", "get_registry"]
