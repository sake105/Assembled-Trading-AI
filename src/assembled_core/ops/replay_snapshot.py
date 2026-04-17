"""Deterministic replay snapshots for paper runs.

Scope
-----
A *replay snapshot* freezes the inputs of a single paper-trading day so that
the same day can be re-run bit-identically later:

- the random seed used for stochastic components (fills, adversarial costs)
- the price frame fed into the engine
- an optional signals frame
- opaque context entries (regime label, kill-switch state, SOR venue spreads)

The snapshot is deliberately tiny and dependency-free: one JSON manifest and
two parquet files (prices, signals). That keeps snapshots cheap to store and
trivial to diff across runs.

Usage::

    from src.assembled_core.ops.replay_snapshot import RunSnapshot

    snap = RunSnapshot(
        run_id="paper_unified",
        as_of_date="2025-01-15",
        seed=42,
        prices=prices_df,
        signals=signals_df,
        context={"regime": "normal"},
    )
    path = snap.save(Path("output/replay_snapshots"))

    # later:
    loaded = RunSnapshot.load(path)
    assert loaded.seed == 42
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SNAPSHOT_SCHEMA_VERSION = 1


def derive_seed(run_id: str, as_of_date: str, base_seed: int | None) -> int:
    """Derive a stable 63-bit seed from ``run_id`` + ``as_of_date`` + base_seed.

    - When ``base_seed`` is not None, it is xor-mixed into the hash so that
      explicitly-seeded runs are reproducible but still unique per day.
    - When ``base_seed`` is None, only the run_id/date hash is used. This means
      two different engine instances with the same run_id and date still
      produce identical randomness. That is intentional: it is the
      "deterministic-by-default" mode for the paper engine.
    """
    material = f"{run_id}|{as_of_date}".encode("utf-8")
    digest = hashlib.sha256(material).digest()
    # Take the low 63 bits so we always fit in a signed 64-bit int.
    seed64 = int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)
    if base_seed is not None:
        seed64 ^= int(base_seed) & ((1 << 63) - 1)
    return seed64


def make_rng(run_id: str, as_of_date: str, base_seed: int | None) -> np.random.Generator:
    """Return a numpy Generator seeded from ``derive_seed``."""
    return np.random.default_rng(derive_seed(run_id, as_of_date, base_seed))


@dataclass
class RunSnapshot:
    """A single-day replay snapshot.

    Attributes:
        run_id: Paper run identifier.
        as_of_date: ISO date of the snapshotted day.
        seed: Base seed used by the engine for this day (None → deterministic
            by run_id+date only).
        prices: Input prices DataFrame (symbol, close, volume, adv, …).
        signals: Optional signals / targets DataFrame.
        context: Opaque dict of extra state (e.g. regime, kill-switch state,
            SOR venue spreads). Must be JSON-serialisable.
        schema_version: Bumped when the on-disk layout changes.
    """

    run_id: str
    as_of_date: str
    seed: int | None
    prices: pd.DataFrame
    signals: pd.DataFrame | None = None
    context: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SNAPSHOT_SCHEMA_VERSION

    # --- I/O ---------------------------------------------------------------

    def save(self, dir_path: Path | str) -> Path:
        """Persist the snapshot under ``dir_path`` and return the directory.

        Layout::

            dir_path / run_id / as_of_date /
                manifest.json
                prices.parquet
                signals.parquet   (optional)
        """
        base = Path(dir_path) / self.run_id / self.as_of_date
        base.mkdir(parents=True, exist_ok=True)

        manifest = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "as_of_date": self.as_of_date,
            "seed": self.seed,
            "derived_seed": derive_seed(self.run_id, self.as_of_date, self.seed),
            "context": self.context,
            "has_signals": self.signals is not None and not self.signals.empty,
            "price_row_count": int(len(self.prices)),
            "price_columns": list(map(str, self.prices.columns)),
        }
        (base / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        self.prices.to_parquet(base / "prices.parquet", index=False)
        if self.signals is not None and not self.signals.empty:
            self.signals.to_parquet(base / "signals.parquet", index=False)

        logger.debug("[REPLAY] Snapshot saved: %s", base)
        return base

    @classmethod
    def load(cls, dir_path: Path | str) -> RunSnapshot:
        """Load a snapshot previously written by :meth:`save`."""
        base = Path(dir_path)
        manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))

        prices = pd.read_parquet(base / "prices.parquet")
        signals = None
        sig_path = base / "signals.parquet"
        if sig_path.exists():
            signals = pd.read_parquet(sig_path)

        return cls(
            run_id=manifest["run_id"],
            as_of_date=manifest["as_of_date"],
            seed=manifest.get("seed"),
            prices=prices,
            signals=signals,
            context=dict(manifest.get("context", {})),
            schema_version=int(manifest.get("schema_version", SNAPSHOT_SCHEMA_VERSION)),
        )

    # --- Convenience -------------------------------------------------------

    def rng(self) -> np.random.Generator:
        """Return the Generator that the engine would use for this snapshot."""
        return make_rng(self.run_id, self.as_of_date, self.seed)
