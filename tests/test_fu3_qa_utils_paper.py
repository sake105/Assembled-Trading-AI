"""FU3 follow-ups: qa/utils/paper state-persistence + PIT corruption/leak guards.

Covers four targeted fixes (smallest-safe-change, behaviour preserved except the
corruption/leak being closed):

- FIX 1 (qa/backtest_engine): the per-bar cycle ctx sets ``kill_switch_persist=
  False`` so trading_cycle_v2's cross-bar restore (``_ks_restore_active =
  is_backtest and not kill_switch_persist``) fires in backtest — a backtest
  never persists a single-bar kill-switch trip across bars / into the live store.
- FIX 2 (utils/atomic_io): ``atomic_write_json`` uses a UNIQUE per-writer tmp
  (pid + uuid4); no shared-tmp collision; cleanup removes only its own tmp; a
  simulated half-written / concurrent case never corrupts the destination;
  single-writer output unchanged.
- FIX 3 (paper/paper_track): ``save_paper_state`` writes via a unique tmp +
  os.replace so a crash mid-write leaves the prior state file intact; ``.backup``
  is still produced from the consistent prior file; happy-path content unchanged.
- FIX 4 (paper/intel_context): a sector-rotation scores frame WITHOUT a timestamp
  column, with an ``as_of`` in play, does NOT silently use the dataset tail — it
  warns (observable) and skips, while a timestamped source is still as_of-filtered.
"""

from __future__ import annotations

import json
import logging
import os
from types import SimpleNamespace

import pandas as pd
import pytest

pytestmark = pytest.mark.advanced


# ---------------------------------------------------------------------------
# FIX 1 — qa backtest cycle ctx forces kill_switch_persist=False
# ---------------------------------------------------------------------------
class TestBacktestKillSwitchPersist:
    def test_cycle_ctx_sets_kill_switch_persist_false(self) -> None:
        """make_cycle_fn must build a ctx with kill_switch_persist=False so the
        backtest cross-bar restore (_ks_restore_active) fires."""
        from src.assembled_core.pipeline.trading_cycle_shared import TradingContext
        from src.assembled_core.qa.backtest_engine import make_cycle_fn

        ctx_template = TradingContext(prices=pd.DataFrame())
        # Default on the template is True (live/paper parity invariant).
        assert ctx_template.kill_switch_persist is True

        captured: dict[str, object] = {}

        def fake_run_trading_cycle(ctx):
            captured["kill_switch_persist"] = ctx.kill_switch_persist
            captured["mode"] = ctx.mode
            return SimpleNamespace(orders=pd.DataFrame())

        cycle_fn = make_cycle_fn(
            ctx_template=ctx_template,
            signal_fn=lambda *a, **k: pd.DataFrame(),
            position_sizing_fn=lambda df, cap: df,
            capital=100_000.0,
            run_trading_cycle_fn=fake_run_trading_cycle,
        )
        cycle_fn(pd.Timestamp("2025-01-15", tz="UTC"), pd.DataFrame())

        assert captured["mode"] == "backtest"
        assert captured["kill_switch_persist"] is False, (
            "backtest cycle ctx must set kill_switch_persist=False so the "
            "cross-bar daily-CB / kill-switch restore fires"
        )

    def test_ks_restore_active_condition_true_in_backtest(self) -> None:
        """Mirror trading_cycle_v2's gate: _is_backtest and not _ks_persist."""
        from src.assembled_core.pipeline.trading_cycle_shared import TradingContext
        from src.assembled_core.qa.backtest_engine import make_cycle_fn

        ctx_template = TradingContext(prices=pd.DataFrame())
        captured: dict[str, object] = {}

        def fake_run_trading_cycle(ctx):
            is_backtest = getattr(ctx, "mode", None) in ("backtest", "bt")
            ks_persist = bool(getattr(ctx, "kill_switch_persist", True))
            captured["ks_restore_active"] = is_backtest and not ks_persist
            return SimpleNamespace(orders=pd.DataFrame())

        cycle_fn = make_cycle_fn(
            ctx_template=ctx_template,
            signal_fn=lambda *a, **k: pd.DataFrame(),
            position_sizing_fn=lambda df, cap: df,
            capital=100_000.0,
            run_trading_cycle_fn=fake_run_trading_cycle,
        )
        cycle_fn(pd.Timestamp("2025-01-15", tz="UTC"), pd.DataFrame())

        assert captured["ks_restore_active"] is True


# ---------------------------------------------------------------------------
# FIX 2 — atomic_write_json unique per-writer tmp
# ---------------------------------------------------------------------------
class TestAtomicWriteJsonUniqueTmp:
    def test_tmp_name_is_unique_per_writer(self, tmp_path, monkeypatch) -> None:
        """The tmp path must embed pid + a uuid hex, never the fixed
        '<name>.tmp' shared name."""
        from src.assembled_core.utils import atomic_io

        target = tmp_path / "state.json"
        seen: list[str] = []
        real_replace = os.replace

        def recording_replace(src, dst):
            seen.append(os.path.basename(str(src)))
            return real_replace(src, dst)

        monkeypatch.setattr("os.replace", recording_replace)
        atomic_io.atomic_write_json(target, {"x": 1})

        assert len(seen) == 1
        tmp_name = seen[0]
        assert tmp_name != "state.json.tmp", "must not use the fixed shared tmp"
        assert tmp_name.startswith("state.json.")
        assert tmp_name.endswith(".tmp")
        assert str(os.getpid()) in tmp_name

    def test_two_writers_use_distinct_tmp_names(self, tmp_path) -> None:
        """Two concurrent writers (same pid) must not collide on the tmp name."""
        from src.assembled_core.utils.atomic_io import _unique_tmp_path

        target = tmp_path / "state.json"
        a = _unique_tmp_path(target)
        b = _unique_tmp_path(target)
        assert a != b

    def test_no_fixed_tmp_remains_after_success(self, tmp_path) -> None:
        from src.assembled_core.utils.atomic_io import atomic_write_json

        target = tmp_path / "state.json"
        atomic_write_json(target, {"x": 1})
        assert target.exists()
        # Neither the legacy fixed tmp nor any stray unique tmp survives.
        leftovers = [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
        assert leftovers == [], f"stray tmp files remain: {leftovers}"

    def test_concurrent_halfwritten_tmp_does_not_corrupt_dest(self, tmp_path) -> None:
        """A foreign half-written tmp sitting next to the target must NOT be
        the thing os.replace moves in — the writer uses its own unique tmp, so
        the destination is the fully-written payload."""
        from src.assembled_core.utils.atomic_io import atomic_write_json

        target = tmp_path / "state.json"
        # Simulate another writer's half-written fixed-name tmp lying around.
        (tmp_path / "state.json.tmp").write_text("{ this is half written")

        atomic_write_json(target, {"complete": True})

        assert json.loads(target.read_text()) == {"complete": True}

    def test_crash_during_replace_leaves_dest_intact_and_no_orphan(
        self, tmp_path, monkeypatch
    ) -> None:
        from src.assembled_core.utils.atomic_io import atomic_write_json

        target = tmp_path / "state.json"
        original = {"safe": "value"}
        target.write_text(json.dumps(original))

        def fail_replace(src, dst):
            raise OSError("Simulated crash during replace")

        monkeypatch.setattr("os.replace", fail_replace)

        with pytest.raises(OSError, match="Simulated crash"):
            atomic_write_json(target, {"unsafe": "value"}, retries=1)

        # Destination intact …
        assert json.loads(target.read_text()) == original
        # … and the writer's own tmp was cleaned up (no orphan).
        leftovers = [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
        assert leftovers == [], f"orphan tmp left behind: {leftovers}"

    def test_single_writer_output_unchanged(self, tmp_path) -> None:
        """Lag = single-writer: content byte-for-byte equals json.dumps(...)."""
        from src.assembled_core.utils.atomic_io import atomic_write_json

        target = tmp_path / "state.json"
        data = {"b": 2, "a": 1, "nested": {"z": [1, 2, 3]}}
        atomic_write_json(target, data, indent=2, sort_keys=False)
        expected = json.dumps(data, indent=2, sort_keys=False, default=str)
        assert target.read_text(encoding="utf-8") == expected


# ---------------------------------------------------------------------------
# FIX 3 — paper save_paper_state crash-safety + consistent backup
# ---------------------------------------------------------------------------
class TestSavePaperStateCrashSafety:
    @pytest.fixture
    def sample_state(self):
        from src.assembled_core.config.constants import PAPER_TRACK_STATE_VERSION
        from src.assembled_core.paper.paper_track import PaperTrackState

        return PaperTrackState(
            strategy_name="test_strategy",
            last_run_date=pd.Timestamp("2025-01-15", tz="UTC"),
            version=PAPER_TRACK_STATE_VERSION,
            positions=pd.DataFrame({"symbol": ["AAPL"], "qty": [10.0]}),
            cash=50000.0,
            equity=150000.0,
            seed_capital=100000.0,
            created_at=pd.Timestamp("2025-01-01", tz="UTC"),
            updated_at=pd.Timestamp("2025-01-15", tz="UTC"),
            total_trades=5,
            total_pnl=50000.0,
            last_equity=140000.0,
            last_positions_value=100000.0,
        )

    def test_crash_mid_write_leaves_prior_state_intact(
        self, sample_state, tmp_path, monkeypatch
    ) -> None:
        from src.assembled_core.paper.paper_track import save_paper_state

        state_path = tmp_path / "state" / "state.json"
        # First (good) write establishes the prior state file.
        save_paper_state(sample_state, state_path)
        prior_bytes = state_path.read_bytes()

        # Now simulate a crash during the replace of the SECOND write.
        sample_state.cash = 99999.0

        def fail_replace(src, dst):
            raise OSError("Simulated crash during replace")

        monkeypatch.setattr("os.replace", fail_replace)

        with pytest.raises(IOError, match="Failed to save state"):
            save_paper_state(sample_state, state_path)

        # Prior state file must be byte-identical (never half-written).
        assert state_path.read_bytes() == prior_bytes
        # No orphan tmp left from the crashed writer.
        state_dir = state_path.parent
        leftovers = [p.name for p in state_dir.iterdir() if p.name.endswith(".tmp")]
        assert leftovers == [], f"orphan tmp left behind: {leftovers}"

    def test_backup_still_produced_from_consistent_prior(
        self, sample_state, tmp_path
    ) -> None:
        from src.assembled_core.paper.paper_track import save_paper_state

        state_path = tmp_path / "state" / "state.json"
        save_paper_state(sample_state, state_path)  # cash=50000

        sample_state.cash = 60000.0
        save_paper_state(sample_state, state_path)  # creates .backup of prior

        backup_path = state_path.with_suffix(state_path.suffix + ".backup")
        assert backup_path.exists()
        backup = json.loads(backup_path.read_text(encoding="utf-8"))
        assert backup["cash"] == 50000.0, "backup must hold the prior, consistent state"
        current = json.loads(state_path.read_text(encoding="utf-8"))
        assert current["cash"] == 60000.0

    def test_happy_path_content_unchanged_and_roundtrips(
        self, sample_state, tmp_path
    ) -> None:
        from src.assembled_core.paper.paper_track import (
            load_paper_state,
            save_paper_state,
        )

        state_path = tmp_path / "state" / "state.json"
        save_paper_state(sample_state, state_path)

        # No fixed '<name>.tmp' and no stray unique tmp remains.
        assert not (state_path.with_suffix(state_path.suffix + ".tmp")).exists()
        leftovers = [
            p.name for p in state_path.parent.iterdir() if p.name.endswith(".tmp")
        ]
        assert leftovers == []

        loaded = load_paper_state(state_path, "test_strategy")
        assert loaded is not None
        assert loaded.cash == 50000.0
        assert loaded.equity == 150000.0
        assert list(loaded.positions["symbol"]) == ["AAPL"]


# ---------------------------------------------------------------------------
# FIX 4 — intel_context PIT-safe no-timestamp fallback
# ---------------------------------------------------------------------------
class TestSectorRotationPITFallback:
    def _make_prices(self) -> pd.DataFrame:
        """Build a prices frame with a 'timestamp' column and enough history /
        ETF coverage to pass the early guards in
        _populate_sector_rotation_scores."""
        from src.assembled_core.paper.intel_context import MIN_SECTOR_HISTORY_DAYS

        n = MIN_SECTOR_HISTORY_DAYS + 5
        idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
        symbols = ["SPY", "XLK", "XLF", "XLE", "XLV"]
        rows = []
        for sym in symbols:
            for i, ts in enumerate(idx):
                rows.append({"timestamp": ts, "symbol": sym, "close": 100.0 + i * 0.1})
        return pd.DataFrame(rows)

    @pytest.fixture(autouse=True)
    def _reset_warn_flag(self):
        import src.assembled_core.paper.intel_context as ic

        ic._SECTOR_ROTATION_PIT_WARNED = False
        yield
        ic._SECTOR_ROTATION_PIT_WARNED = False

    def test_no_ts_column_with_as_of_warns_and_skips(self, monkeypatch, caplog) -> None:
        """Scores frame without a timestamp column + as_of set => observable
        WARNING, no silent tail read, attribute not set."""
        import src.assembled_core.paper.intel_context as ic

        # compute_sector_scores returns a frame WITHOUT a 'timestamp' column.
        no_ts_scores = pd.DataFrame(
            {
                "XLK_score": [0.1, 0.2, 0.9],  # tail (0.9) would be the look-ahead
                "XLF_score": [0.0, 0.0, 0.0],
            }
        )
        monkeypatch.setattr(
            "src.assembled_core.signals.sector_rotation.compute_sector_scores",
            lambda sector_df, spy_df: no_ts_scores,
        )

        ctx = SimpleNamespace(
            prices=self._make_prices(),
            as_of=pd.Timestamp("2024-02-01", tz="UTC"),
        )

        with caplog.at_level(logging.WARNING, logger=ic.log.name):
            ic._populate_sector_rotation_scores(ctx)

        # Did NOT silently use the tail.
        assert not hasattr(ctx, "sector_rotation_scores")
        # Observable: a one-time WARNING was emitted.
        assert any(
            "no" in r.message.lower() and "timestamp" in r.message.lower()
            for r in caplog.records
            if r.levelno == logging.WARNING
        ), "expected an observable non-PIT WARNING"

    def test_no_ts_column_without_as_of_uses_tail(self, monkeypatch) -> None:
        """Live/EOD (as_of is None): tail == as_of, keep iloc[-1] behaviour."""
        import src.assembled_core.paper.intel_context as ic

        no_ts_scores = pd.DataFrame(
            {"XLK_score": [0.1, 0.2, 0.9], "XLF_score": [0.0, 0.0, 0.0]}
        )
        monkeypatch.setattr(
            "src.assembled_core.signals.sector_rotation.compute_sector_scores",
            lambda sector_df, spy_df: no_ts_scores,
        )

        ctx = SimpleNamespace(prices=self._make_prices(), as_of=None)
        ic._populate_sector_rotation_scores(ctx)

        assert hasattr(ctx, "sector_rotation_scores")
        assert ctx.sector_rotation_scores["XLK_score"] == 0.9

    def test_ts_column_still_as_of_filtered(self, monkeypatch) -> None:
        """Timestamped scores frame is still as_of-filtered (unchanged branch)."""
        import src.assembled_core.paper.intel_context as ic

        ts_scores = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2024-01-10", "2024-02-01", "2024-03-15"], utc=True
                ),
                "XLK_score": [0.1, 0.5, 0.9],  # 0.9 is AFTER as_of and must be cut
            }
        )
        monkeypatch.setattr(
            "src.assembled_core.signals.sector_rotation.compute_sector_scores",
            lambda sector_df, spy_df: ts_scores,
        )

        ctx = SimpleNamespace(
            prices=self._make_prices(),
            as_of=pd.Timestamp("2024-02-10", tz="UTC"),
        )
        ic._populate_sector_rotation_scores(ctx)

        assert hasattr(ctx, "sector_rotation_scores")
        # Latest row at/before as_of is the 2024-02-01 row (0.5), NOT the tail.
        assert ctx.sector_rotation_scores["XLK_score"] == 0.5
