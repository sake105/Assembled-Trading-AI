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


# ----------------------------------------------------------------------------
# E0.1 — deterministic paper replay
# ----------------------------------------------------------------------------


@dataclass
class ReplayResult:
    """Output of :func:`run_paper_replay`.

    Attributes:
        orders_df: Aggregated order stream across the replay. One row per
            order emitted by :func:`run_trading_cycle`, with a ``timestamp``
            column so rows from consecutive days remain in chronological
            order. Used by the E0.1 parity test to compare against
            ``run_portfolio_backtest`` output.
        n_days: Number of replay days visited.
        seed: Base seed used for this replay.
    """

    orders_df: pd.DataFrame
    n_days: int
    seed: int | None


def run_paper_replay(
    prices: pd.DataFrame,
    signal_fn: Any,
    position_sizing_fn: Any,
    *,
    start_capital: float = 10_000.0,
    seed: int | None = 42,
    as_of_dates: list[pd.Timestamp] | None = None,
    enable_risk_controls: bool = True,
    kill_switch_persist: bool = True,
) -> ReplayResult:
    """Replay the trading cycle day-by-day on the same inputs as a backtest.

    The replay drives :func:`src.assembled_core.pipeline.trading_cycle.run_trading_cycle`
    once per ``as_of_date`` using an evolving positions book. Order
    generation flows through exactly the same code path as
    :func:`run_portfolio_backtest` (via its ``cycle_fn``), which means the
    emitted order stream is bit-identical whenever risk gates, kill-switch
    behaviour and input fixtures match.

    This is the helper the E0.1 parity test relies on.

    Args:
        prices: Long-form price frame (``timestamp``, ``symbol``, ``close``).
        signal_fn: Callable ``(prices_df) -> signals_df`` (same contract as
            ``run_portfolio_backtest``).
        position_sizing_fn: Callable ``(signals_df, capital) -> targets_df``.
        start_capital: Replay seed capital (mirrors backtest ``start_capital``).
        seed: Base seed — used for deterministic slippage / adversarial cost
            hooks downstream. Replay itself is deterministic for any value.
        as_of_dates: Optional explicit schedule. When ``None`` the replay
            iterates every unique timestamp in ``prices``.
        enable_risk_controls: Propagated into ``TradingContext``; default
            ``True`` matches the E0.1 parity invariant.
        kill_switch_persist: Propagated into ``TradingContext``; default
            ``True`` matches the E0.1 parity invariant.

    Returns:
        ``ReplayResult`` whose ``orders_df`` is sorted by ``timestamp`` and
        carries every order emitted by the cycle across the replay window.
    """
    from src.assembled_core.pipeline.trading_cycle import (
        TradingContext,
        run_trading_cycle,
    )

    if prices is None or prices.empty:
        return ReplayResult(
            orders_df=pd.DataFrame(
                columns=["timestamp", "symbol", "side", "qty", "price"]
            ),
            n_days=0,
            seed=seed,
        )

    prices_sorted = prices.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    if as_of_dates is None:
        dates = list(prices_sorted["timestamp"].drop_duplicates().sort_values())
    else:
        dates = list(as_of_dates)

    universe = sorted(prices_sorted["symbol"].unique().tolist())
    positions: dict[str, float] = {}
    all_orders: list[pd.DataFrame] = []

    for ts in dates:
        window = prices_sorted[prices_sorted["timestamp"] <= ts]
        if window.empty:
            continue
        current_bar = prices_sorted[prices_sorted["timestamp"] == ts]
        if current_bar.empty:
            continue

        # order_timestamp=ts anchors the emitted orders' `timestamp` column to
        # the bar, not wall-clock. The default factory is `pd.Timestamp.now`,
        # which breaks replay determinism (P0 A8 sunset, parity_gap.md).
        ctx = TradingContext(
            prices=window,
            as_of=ts,
            universe=universe,
            signal_fn=signal_fn,
            position_sizing_fn=position_sizing_fn,
            capital=float(start_capital),
            current_positions=pd.DataFrame(
                [{"symbol": s, "qty": q} for s, q in positions.items() if q != 0]
            ),
            order_timestamp=ts,
            enable_risk_controls=enable_risk_controls,
            kill_switch_persist=kill_switch_persist,
            write_outputs=False,
            mode="backtest",
        )

        hooks = {
            "load_prices": lambda _ctx, _w=window, _b=current_bar: (_w.copy(), _b.copy()),
            "build_features": lambda _ctx, df: df,
        }
        result = run_trading_cycle(ctx, hooks=hooks)

        orders = result.orders_filtered if result.orders_filtered is not None else result.orders
        if orders is not None and not orders.empty:
            stamped = orders.copy()
            if "timestamp" not in stamped.columns:
                stamped["timestamp"] = ts
            all_orders.append(stamped)

            # Evolve positions so the next bar sees the updated book. Fill
            # model-independent: we only update based on *generated* orders,
            # which matches how the backtest loop advances between bars.
            # Replay is a correctness tool for E0.1 parity — defaulting a
            # missing ``side`` to BUY or a missing ``qty`` to 0 silently
            # diverges replay from backtest, so we fail-fast on malformed
            # rows instead.
            for _, row in stamped.iterrows():
                sym = str(row["symbol"])
                raw_side = row.get("side")
                raw_qty = row.get("qty")
                if raw_side is None or (isinstance(raw_side, float) and np.isnan(raw_side)):
                    raise ValueError(f"replay: order row for {sym} has null side")
                side = str(raw_side).upper()
                if side not in {"BUY", "SELL"}:
                    raise ValueError(f"replay: invalid side {side!r} for {sym}")
                if raw_qty is None:
                    raise ValueError(f"replay: order row for {sym} has null qty")
                qty = float(raw_qty)
                if not np.isfinite(qty) or qty <= 0:
                    raise ValueError(f"replay: non-positive/finite qty={qty} for {sym}")
                signed = qty if side == "BUY" else -qty
                positions[sym] = positions.get(sym, 0.0) + signed

    if all_orders:
        orders_df = (
            pd.concat(all_orders, ignore_index=True)
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
    else:
        orders_df = pd.DataFrame(columns=["timestamp", "symbol", "side", "qty", "price"])

    return ReplayResult(orders_df=orders_df, n_days=len(dates), seed=seed)
