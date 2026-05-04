"""Signal dispatcher with Strangler-Fig migration support.

From 60_MIGRATION_PLAYBOOK.md §0.1.

Provides three operating modes:
  - LEGACY:  routes all calls to the existing trading_cycle function
  - MODERN:  routes to the new plugin registry
  - SHADOW:  runs both, records diffs, returns LEGACY result

The mode is per-dispatcher instance (not global), so individual signal
types can be migrated independently.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class Pipeline(str, Enum):
    """Operating mode for the signal dispatcher."""

    LEGACY = "legacy"
    MODERN = "modern"
    SHADOW = "shadow"  # runs both, returns LEGACY, records diff


@dataclass
class DispatchRecord:
    """A single dispatch event with optional diff data."""

    mode: str
    inputs_repr: str
    legacy_result: Any = None
    modern_result: Any = None
    diff_keys: list[str] = field(default_factory=list)
    modern_error: str | None = None


class SignalDispatcher:
    """Routes signal computation between legacy and modern pipelines.

    Args:
        mode: Operating mode (LEGACY / MODERN / SHADOW).
        registry: Callable that runs the modern plugin pipeline.
            Must accept the same arguments as ``legacy_fn``.
        legacy_fn: Callable that runs the legacy trading_cycle.
        record_diffs: If True, append DispatchRecord for each call.
            Useful during shadow mode to track divergence rate.
    """

    def __init__(
        self,
        mode: Pipeline | str,
        registry: Callable | None = None,
        legacy_fn: Callable | None = None,
        record_diffs: bool = False,
    ) -> None:
        self.mode = Pipeline(mode)
        self.registry = registry
        self.legacy_fn = legacy_fn
        self.record_diffs = record_diffs
        self._records: list[DispatchRecord] = []

    def run(self, inputs: Any) -> Any:
        """Dispatch *inputs* according to the current mode.

        Returns the result from the active pipeline.
        Raises RuntimeError if the required callable is not set.
        """
        if self.mode == Pipeline.LEGACY:
            return self._run_legacy(inputs)
        elif self.mode == Pipeline.MODERN:
            return self._run_modern(inputs)
        elif self.mode == Pipeline.SHADOW:
            return self._run_shadow(inputs)
        raise ValueError(f"Unknown mode: {self.mode}")  # pragma: no cover

    def _run_legacy(self, inputs: Any) -> Any:
        if self.legacy_fn is None:
            raise RuntimeError("legacy_fn is not set — cannot run LEGACY mode")
        return self.legacy_fn(inputs)

    def _run_modern(self, inputs: Any) -> Any:
        if self.registry is None:
            raise RuntimeError("registry is not set — cannot run MODERN mode")
        return self.registry(inputs)

    def _run_shadow(self, inputs: Any) -> Any:
        """Run both pipelines; return LEGACY result; record any diff."""
        legacy_result = self._run_legacy(inputs)

        modern_result = None
        modern_error = None
        diff_keys: list[str] = []

        try:
            modern_result = self._run_modern(inputs)
            diff_keys = self._diff_keys(legacy_result, modern_result)
            if diff_keys:
                logger.debug(
                    "SHADOW diff: %d divergent keys: %s",
                    len(diff_keys),
                    diff_keys[:10],
                )
        except Exception as exc:
            modern_error = str(exc)
            logger.warning("Modern pipeline failed in shadow mode: %s", exc)

        if self.record_diffs:
            self._records.append(
                DispatchRecord(
                    mode="shadow",
                    inputs_repr=repr(inputs)[:200],
                    legacy_result=legacy_result,
                    modern_result=modern_result,
                    diff_keys=diff_keys,
                    modern_error=modern_error,
                )
            )

        return legacy_result  # LEGACY wins until explicit cutover

    @staticmethod
    def _diff_keys(a: Any, b: Any) -> list[str]:
        """Return keys that differ between two dict-like results."""
        if not (isinstance(a, dict) and isinstance(b, dict)):
            return [] if a == b else ["__root__"]
        all_keys = set(a) | set(b)
        return [k for k in all_keys if a.get(k) != b.get(k)]

    def divergence_rate(self) -> float:
        """Fraction of shadow calls with at least one differing key."""
        if not self._records:
            return 0.0
        diverged = sum(1 for r in self._records if r.diff_keys or r.modern_error)
        return diverged / len(self._records)

    def clear_records(self) -> None:
        self._records.clear()

    @property
    def records(self) -> list[DispatchRecord]:
        return list(self._records)

    def promote_to_modern(self) -> None:
        """Switch from SHADOW to MODERN after sufficient validation."""
        if self.mode not in (Pipeline.SHADOW, Pipeline.LEGACY):
            raise RuntimeError(f"Cannot promote from {self.mode.value}")
        logger.info(
            "Promoting dispatcher to MODERN (divergence_rate=%.1f%%)",
            self.divergence_rate() * 100,
        )
        self.mode = Pipeline.MODERN


__all__ = ["Pipeline", "SignalDispatcher", "DispatchRecord"]
