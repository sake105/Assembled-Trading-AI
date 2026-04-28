# src/assembled_core/pipeline/orchestrator.py
"""Pipeline orchestration for EOD runs.

**Architecture note (B5):** This module is the *stateless EOD batch* pipeline,
called via ``scripts/run_eod_pipeline.py`` / ``assembled-run-daily``.

The *live/paper* pipeline (``trading_cycle_v2.py``) is called via
``scripts/run_daily.py`` and uses ``TradingContext`` + full risk overlays.

Both pipelines share signal generation via
``pipeline._shared_eod.compute_signals_by_mode``.  Full convergence (Option A)
is deferred — see ``autonome_weiterarbeit/AUDIT_2026-04-26_FINDINGS_AND_REMEDIATION_v2.md`` §B5.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.assembled_core.config import OUTPUT_DIR, SUPPORTED_FREQS
from src.assembled_core.costs import get_default_cost_model
from src.assembled_core.ema_config import get_default_ema_config
from src.assembled_core.logging_utils import get_logger
from src.assembled_core.pipeline.backtest import (
    compute_metrics,
    simulate_equity,
    write_backtest_report,
)
from src.assembled_core.pipeline.io import (
    load_orders,
    load_prices,
    load_prices_with_fallback,
)
from src.assembled_core.pipeline.orders import signals_to_orders, write_orders
from src.assembled_core.pipeline.portfolio import (
    simulate_with_costs,
    write_portfolio_report,
)
from src.assembled_core.qa.health import aggregate_qa_status
from src.assembled_core.qa.metrics import compute_all_metrics
from src.assembled_core.qa.qa_gates import QAResult, evaluate_all_gates
from src.assembled_core.reports.daily_qa_report import generate_qa_report

# Get logger (will use default logging if not configured)
logger = get_logger("assembled_core.pipeline")


def _manifest_path_str(path: str | Path | None, *, base_dir: Path) -> str | None:
    """Convert a path to a portable manifest string.

    Rules:
    - Prefer paths relative to base_dir (typically output/).
    - Always use POSIX slashes in the manifest (portable across OS).
    """
    if path is None:
        return None

    p = Path(path)
    try:
        rel = p.relative_to(base_dir)
        return rel.as_posix()
    except Exception:
        return p.as_posix()


def _write_manifest_json(manifest_path: Path, manifest: dict[str, Any]) -> None:
    """Write a JSON manifest deterministically (stable bytes for same inputs). Atomic: temp in same dir then replace."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(manifest, sort_keys=True, indent=2) + "\n"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            delete=False,
            dir=str(manifest_path.parent),
            prefix=manifest_path.name + ".tmp.",
            suffix=".json",
        ) as f:
            tmp_path = Path(f.name)
            f.write(content)
            f.flush()
        tmp_path.replace(manifest_path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            try:
                if tmp_path != manifest_path:
                    tmp_path.unlink(missing_ok=True)
            except OSError as _unlink_exc:
                logger.debug("[_write_manifest_json] tmp cleanup failed: %s", _unlink_exc)


def _backfill_evidence_index_manifest_path(
    *, base_dir: Path, ledger_result: dict[str, Any] | None, manifest_path: Path
) -> None:
    """Best-effort backfill of manifest_path into Evidence Index JSON.

    If ledger_result contains an evidence_index_path, this helper will:
    - Load the Evidence Index JSON from that path
    - Set paths.manifest_path to the relative POSIX path of the given manifest
      (relative to base_dir, typically output/)
    - Re-write the Evidence Index JSON deterministically (sort_keys=True, indent=2, trailing newline)

    Errors are logged as warnings and do not fail the pipeline.
    """
    if not ledger_result:
        return

    evidence_index_rel = ledger_result.get("evidence_index_path")
    if not evidence_index_rel:
        return

    evidence_index_path = base_dir / evidence_index_rel
    if not evidence_index_path.exists():
        logger.warning(
            "Evidence index not found for manifest backfill: %s", evidence_index_path
        )
        return

    try:
        with evidence_index_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        logger.warning(
            "Failed to read evidence index for manifest backfill: %s (%s)",
            evidence_index_path,
            exc,
        )
        return

    try:
        paths_block = data.setdefault("paths", {})
        # manifest_path is stored as relative POSIX path in Evidence Index
        rel_manifest = _manifest_path_str(manifest_path, base_dir=base_dir)
        paths_block["manifest_path"] = rel_manifest

        payload = json.dumps(data, sort_keys=True, indent=2) + "\n"

        # Atomic write: temp file in same directory, then replace
        tmp_path: Path | None = None
        try:
            from tempfile import NamedTemporaryFile

            with NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=evidence_index_path.parent,
                delete=False,
                suffix=".tmp.json",
            ) as tmp_file:
                tmp_file.write(payload)
                tmp_path = Path(tmp_file.name)

            tmp_path.replace(evidence_index_path)
        finally:
            if tmp_path is not None and tmp_path.exists():
                try:
                    # If replace failed and temp file is still there, best-effort cleanup
                    if tmp_path != evidence_index_path:
                        tmp_path.unlink(missing_ok=True)
                except OSError as _unlink_exc:
                    logger.debug(
                        "[evidence_backfill.manifest] tmp cleanup failed: %s",
                        _unlink_exc,
                    )

        logger.info(
            "Backfilled manifest_path into evidence index: %s -> %s",
            evidence_index_path,
            rel_manifest,
        )
    except Exception as exc:
        logger.warning(
            "Failed to backfill manifest_path into evidence index %s: %s",
            evidence_index_path,
            exc,
        )


def _backfill_evidence_index_accounting_path(
    *, base_dir: Path, ledger_result: dict[str, Any] | None
) -> None:
    """Best-effort backfill of accounting_report_path into Evidence Index JSON.

    If the Evidence Index exists and paths.accounting_report_path is missing/empty,
    set it from the known path in ledger_result (relative POSIX, no backslashes).
    Only sets when missing; never overwrites an existing value.
    """
    if not ledger_result:
        return

    evidence_index_rel = ledger_result.get("evidence_index_path")
    if not evidence_index_rel:
        return

    known_accounting = ledger_result.get("accounting_report_path")
    if not known_accounting:
        return

    evidence_index_path = base_dir / evidence_index_rel
    if not evidence_index_path.exists():
        return

    try:
        with evidence_index_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        logger.warning(
            "Failed to read evidence index for accounting backfill: %s (%s)",
            evidence_index_path,
            exc,
        )
        return

    paths_block = data.setdefault("paths", {})
    current = paths_block.get("accounting_report_path")
    if current is not None and (not isinstance(current, str) or current.strip()):
        return  # already set, do not overwrite

    rel_accounting = _manifest_path_str(known_accounting, base_dir=base_dir)
    if not rel_accounting:
        return

    # Normalize to POSIX (no backslashes)
    paths_block["accounting_report_path"] = Path(rel_accounting).as_posix()

    try:
        payload = json.dumps(data, sort_keys=True, indent=2) + "\n"
        tmp_path: Path | None = None
        try:
            from tempfile import NamedTemporaryFile

            with NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=evidence_index_path.parent,
                delete=False,
                suffix=".tmp.json",
            ) as tmp_file:
                tmp_file.write(payload)
                tmp_path = Path(tmp_file.name)

            tmp_path.replace(evidence_index_path)
        finally:
            if tmp_path is not None and tmp_path.exists():
                try:
                    if tmp_path != evidence_index_path:
                        tmp_path.unlink(missing_ok=True)
                except OSError as _unlink_exc:
                    logger.debug(
                        "[evidence_backfill.accounting] tmp cleanup failed: %s",
                        _unlink_exc,
                    )

        logger.info(
            "Backfilled accounting_report_path into evidence index: %s -> %s",
            evidence_index_path,
            paths_block["accounting_report_path"],
        )
    except Exception as exc:
        logger.warning(
            "Failed to backfill accounting_report_path into evidence index %s: %s",
            evidence_index_path,
            exc,
        )


def _metrics_to_dict(metrics) -> dict[str, Any] | None:
    """Convert PerformanceMetrics to dictionary for JSON serialization.

    Args:
        metrics: PerformanceMetrics instance or None

    Returns:
        Dictionary representation or None
    """
    if metrics is None:
        return None

    return {
        "final_pf": metrics.final_pf,
        "total_return": metrics.total_return,
        "cagr": metrics.cagr,
        "sharpe_ratio": metrics.sharpe_ratio,
        "sortino_ratio": metrics.sortino_ratio,
        "calmar_ratio": metrics.calmar_ratio,
        "max_drawdown": metrics.max_drawdown,
        "max_drawdown_pct": metrics.max_drawdown_pct,
        "current_drawdown": metrics.current_drawdown,
        "volatility": metrics.volatility,
        "var_95": metrics.var_95,
        "hit_rate": metrics.hit_rate,
        "profit_factor": metrics.profit_factor,
        "avg_win": metrics.avg_win,
        "avg_loss": metrics.avg_loss,
        "turnover": metrics.turnover,
        "total_trades": metrics.total_trades,
        "start_date": metrics.start_date.isoformat() if metrics.start_date else None,
        "end_date": metrics.end_date.isoformat() if metrics.end_date else None,
        "periods": metrics.periods,
        "start_capital": metrics.start_capital,
        "end_equity": metrics.end_equity,
    }


def _gate_result_to_dict(gate_result) -> dict[str, Any] | None:
    """Convert QAGatesSummary to dictionary for JSON serialization.

    Args:
        gate_result: QAGatesSummary instance or None

    Returns:
        Dictionary representation or None
    """
    if gate_result is None:
        return None

    passed = sum(1 for r in gate_result.gate_results if r.result.value == "ok")
    warnings = sum(1 for r in gate_result.gate_results if r.result.value == "warning")
    blocked = sum(1 for r in gate_result.gate_results if r.result.value == "block")

    return {
        "overall_result": gate_result.overall_result.value,
        "passed_gates": passed,
        "warning_gates": warnings,
        "blocked_gates": blocked,
        "gate_results": [
            {
                "gate_name": r.gate_name,
                "result": r.result.value,
                "reason": r.reason,
                "details": r.details,
            }
            for r in gate_result.gate_results
        ],
    }


def _enrich_signals_post_generation(
    signals: "pd.DataFrame",
    prices: "pd.DataFrame",
    policy: dict,
) -> None:
    """Apply optional post-generation signal enrichment steps in-place.

    Each enrichment step is individually guarded so that failures in one step
    do not block others or the main pipeline.  All imports are lazy so that
    optional dependencies do not cause import-time errors.

    Steps applied (when enabled in policy):
    - earnings_integration   (earnings_guard.enabled)
    - signal_confidence      (bayesian_confidence.enabled)
    - signal_diagnostics     (signal_diagnostics.enabled)

    Args:
        signals: Signal DataFrame (modified in-place where applicable).
        prices: Price DataFrame used by enrichment steps.
        policy: Parsed policy.yaml dict (may be empty).
    """
    if signals is None or signals.empty:
        return

    # --- Earnings integration ---
    try:
        earnings_cfg = (policy.get("earnings_guard") or {})
        if earnings_cfg.get("enabled", False):
            from src.assembled_core.features.event_features import apply_earnings_guard
            apply_earnings_guard(signals, policy=earnings_cfg)
            logger.debug("[ORCHESTRATOR] earnings_guard applied")
    except Exception as exc:
        logger.debug("[ORCHESTRATOR] earnings_guard skipped: %s", exc)

    # --- Bayesian signal confidence ---
    try:
        bayes_cfg = (policy.get("bayesian_confidence") or {})
        if bayes_cfg.get("enabled", False):
            from src.assembled_core.signals.signal_confidence import (
                apply_bayesian_confidence,
            )
            apply_bayesian_confidence(signals, prices=prices, config=bayes_cfg)
            logger.debug("[ORCHESTRATOR] bayesian_confidence applied")
    except Exception as exc:
        logger.debug("[ORCHESTRATOR] bayesian_confidence skipped: %s", exc)

    # --- Signal diagnostics ---
    try:
        diag_cfg = (policy.get("signal_diagnostics") or {})
        if diag_cfg.get("enabled", False):
            from src.assembled_core.qa.signal_diagnostics import run_signal_diagnostics
            run_signal_diagnostics(signals, prices=prices, config=diag_cfg)
            logger.debug("[ORCHESTRATOR] signal_diagnostics applied")
    except Exception as exc:
        logger.debug("[ORCHESTRATOR] signal_diagnostics skipped: %s", exc)


def _normalize_signals_schema(
    signals: pd.DataFrame, prices: pd.DataFrame
) -> pd.DataFrame:
    """Bridge multifactor_v2 `direction`/`score` schema to `sig`/`price`.

    EMA signals already carry `sig` and `price`. Multifactor v2 emits
    `direction ∈ {LONG, SHORT, FLAT, NEUTRAL}` and lacks `price`. Downstream
    `signals_to_orders` requires `sig` (int in {-1,0,+1}) and `price` (close).
    """
    if signals is None or signals.empty:
        return signals

    df = signals.copy()

    if "sig" not in df.columns:
        if "direction" in df.columns:
            mapping = {"LONG": 1, "BUY": 1, "SHORT": -1, "SELL": -1,
                       "FLAT": 0, "NEUTRAL": 0, "HOLD": 0}
            df["sig"] = (
                df["direction"].astype(str).str.upper().map(mapping).fillna(0).astype(int)
            )
        else:
            df["sig"] = 0

    if "price" not in df.columns:
        if prices is not None and not prices.empty and "close" in prices.columns:
            price_lookup = prices[["timestamp", "symbol", "close"]].copy()
            price_lookup["timestamp"] = pd.to_datetime(
                price_lookup["timestamp"], utc=True
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.merge(
                price_lookup.rename(columns={"close": "price"}),
                on=["timestamp", "symbol"],
                how="left",
            )
            # Fallback: for symbols with no exact ts match, use latest close ≤ ts.
            missing = df["price"].isna()
            if missing.any():
                latest_close = (
                    price_lookup.sort_values("timestamp")
                    .groupby("symbol")["close"]
                    .last()
                )
                df.loc[missing, "price"] = (
                    df.loc[missing, "symbol"].map(latest_close)
                )
        else:
            df["price"] = float("nan")

    return df


def run_execute_step(
    freq: str, output_dir: Path | None = None, price_file: str | None = None
) -> tuple[Path, pd.DataFrame]:
    """Run execution step: generate orders from policy-driven signals.

    Signal mode is read from ``configs/policy.yaml`` → ``signal_generation.mode``.
    Supported modes: ``ema`` (legacy default), ``multifactor``, ``ml_enhanced``.
    Falls back to ``ema`` if mode is unrecognised or the policy key is absent.

    Args:
        freq: Trading frequency ("1d" or "5min")
        output_dir: Base output directory (default: None, uses config.OUTPUT_DIR)
        price_file: Optional explicit path to price file

    Returns:
        Tuple of (orders_file_path, orders DataFrame)

    Side effects:
        Writes orders CSV file: output_dir/orders_{freq}.csv
    """
    base = output_dir if output_dir else OUTPUT_DIR

    # Load prices
    prices = load_prices(freq, price_file=price_file, output_dir=base)

    # Load policy for signal mode dispatch and enrichment
    _policy: dict = {}
    try:
        import yaml
        _policy_path = Path("configs/policy.yaml")
        if _policy_path.exists():
            with open(_policy_path, "r", encoding="utf-8") as _pf:
                _policy = yaml.safe_load(_pf) or {}
    except Exception as exc:
        logger.debug("Could not read policy.yaml: %s", exc)

    # Compute signals via shared canonical dispatch (B5: single source of truth).
    from src.assembled_core.pipeline._shared_eod import compute_signals_by_mode
    signals = compute_signals_by_mode(prices, _policy, freq=freq)

    # Post-signal enrichment (earnings guard, Bayesian confidence, diagnostics).
    try:
        _enrich_signals_post_generation(signals, prices, _policy)
    except Exception as exc:
        logger.debug("[ORCHESTRATOR] Post-signal enrichment skipped: %s", exc)

    # Normalize signal schema: multifactor_v2 / ml_enhanced emit
    # `direction`/`score` columns; signals_to_orders requires `sig`/`price`.
    # Bridge the two so downstream order-gen logic stays schema-stable.
    signals = _normalize_signals_schema(signals, prices)

    # Generate orders
    orders = signals_to_orders(signals)

    # Annotate orders with cost columns (slippage, commission, spread)
    try:
        from src.assembled_core.execution.transaction_costs import (
            add_cost_columns_to_trades,
        )
        from src.assembled_core.costs import get_default_cost_model
        cost_model = get_default_cost_model()
        orders = add_cost_columns_to_trades(orders, prices=prices, cost_model=cost_model)
    except Exception as _e:  # noqa: BLE001
        logger.debug("[orchestrator] Cost annotation skipped: %s", _e)

    # Write orders
    orders_path = write_orders(orders, freq, output_dir=base)

    return orders_path, orders


def run_backtest_step(
    freq: str,
    start_capital: float,
    output_dir: Path | None = None,
    price_file: str | None = None,
) -> tuple[Path, Path]:
    """Run backtest step: simulate equity without costs.

    Args:
        freq: Trading frequency ("1d" or "5min")
        start_capital: Starting capital
        output_dir: Base output directory (default: None, uses config.OUTPUT_DIR)
        price_file: Optional explicit path to price file

    Returns:
        Tuple of (equity_curve_path, report_path)

    Side effects:
        Writes equity_curve_{freq}.csv and performance_report_{freq}.md
    """
    base = output_dir if output_dir else OUTPUT_DIR

    # Load data
    if price_file:
        prices = load_prices(freq, price_file=price_file, output_dir=base)
    else:
        prices = load_prices_with_fallback(freq, output_dir=base)

    orders = load_orders(freq, output_dir=base, strict=False)

    # Simulate
    equity = simulate_equity(prices, orders, start_capital=start_capital)
    metrics = compute_metrics(equity)

    # Write results
    curve_path, rep_path = write_backtest_report(equity, metrics, freq, output_dir=base)

    return curve_path, rep_path


def run_portfolio_step(
    freq: str,
    start_capital: float,
    commission_bps: float | None = None,
    spread_w: float | None = None,
    impact_w: float | None = None,
    output_dir: Path | None = None,
) -> tuple[Path, Path, pd.DataFrame]:
    """Run portfolio step: simulate equity with costs.

    Args:
        freq: Trading frequency ("1d" or "5min")
        start_capital: Starting capital
        commission_bps: Commission in basis points (default: from cost model)
        spread_w: Spread weight (default: from cost model)
        impact_w: Impact weight (default: from cost model)
        output_dir: Base output directory (default: None, uses config.OUTPUT_DIR)

    Returns:
        Tuple of (equity_path, report_path, trades_df)
        trades_df: DataFrame with trades (includes fill_qty, fill_price, status, costs)

    Side effects:
        Writes portfolio_equity_{freq}.csv and portfolio_report_{freq}.md
    """
    base = output_dir if output_dir else OUTPUT_DIR

    # Get default cost model if not provided
    if commission_bps is None or spread_w is None or impact_w is None:
        cost_model = get_default_cost_model()
        commission_bps = (
            commission_bps if commission_bps is not None else cost_model.commission_bps
        )
        spread_w = spread_w if spread_w is not None else cost_model.spread_w
        impact_w = impact_w if impact_w is not None else cost_model.impact_w

    # Load orders
    orders = load_orders(freq, output_dir=base, strict=True)

    # Load prices for fill model pipeline
    prices = None
    try:
        prices = load_prices_with_fallback(freq, output_dir=base)
    except Exception:
        logger.warning("Could not load prices for fill model pipeline")

    # Simulate with costs (returns trades with fill_qty, fill_price, status, costs)
    equity, metrics, trades_df = simulate_with_costs(
        orders, start_capital, commission_bps, spread_w, impact_w, freq, prices=prices
    )

    # Write results
    eq_path, rep_path = write_portfolio_report(equity, metrics, freq, output_dir=base)

    return eq_path, rep_path, trades_df


# ---------------------------------------------------------------------------
# run_eod_pipeline helpers (_eo_*)
# ---------------------------------------------------------------------------

def _eo_load_prices(
    freq: str,
    symbols: list[str] | None,
    data_source: str | None,
    price_file: str | None,
    start_date: str | None,
    end_date: str | None,
    base: Path,
) -> tuple[pd.DataFrame, bool]:
    """Load price data from the configured source. Returns (prices, failed)."""
    try:
        from src.assembled_core.config.settings import get_settings
        from src.assembled_core.data.data_source import get_price_data_source

        settings = get_settings()
        source_type = data_source if data_source is not None else settings.data_source

        if symbols is None:
            if settings.watchlist_file.exists():
                symbols = []
                try:
                    with settings.watchlist_file.open("r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith("#"):
                                symbols.append(line.upper())
                except (IOError, OSError) as exc:
                    logger.warning(
                        "Failed to read watchlist file %s: %s. Using default universe.",
                        settings.watchlist_file, exc,
                    )
                    symbols = settings.default_universe
                if not symbols:
                    symbols = settings.default_universe
            else:
                symbols = settings.default_universe

        _start = start_date or "2020-01-01"
        _end = end_date or "today"

        price_source = get_price_data_source(
            settings=settings, data_source=source_type, price_file=price_file
        )
        logger.info("Loading prices from %s source...", source_type)
        logger.info(
            "Symbols: %s%s (%d total)",
            symbols[:10], "..." if len(symbols) > 10 else "", len(symbols),
        )
        logger.info("Date range: %s to %s", _start, _end)

        prices = price_source.get_history(
            symbols=symbols, start_date=_start, end_date=_end, freq=freq
        )
        logger.info(
            "Price data OK: %d rows, %d symbols", len(prices), prices["symbol"].nunique()
        )

        if source_type == "yahoo" and len(prices) > 0:
            cache_path = base / "aggregates" / f"{freq}_live_cache.parquet"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            prices.to_parquet(cache_path, index=False)
            logger.info("Cached live data to %s", cache_path)

        return prices, False
    except FileNotFoundError as e:
        logger.error("Price data not found: %s", e)
        return pd.DataFrame(), True
    except Exception as e:
        logger.error("Failed to load price data: %s", e, exc_info=True)
        return pd.DataFrame(), True


def _eo_step_ledger(
    *,
    freq: str,
    base: Path,
    started_at: datetime,
    start_capital: float,
    portfolio_trades_df: pd.DataFrame,
    broker_snapshot_policy: str,
    broker_snapshot_file: str | Path | None,
    broker_snapshot_date: str | None,
    broker_snapshot_run_id: str | None,
    write_evidence_pack: bool,
    write_paper_broker_snapshot: bool,
) -> dict[str, Any]:
    """Run Step 4b ledger/accounting. Returns {ledger_result, completed, failed}."""
    run_id = f"run_{started_at.strftime('%Y%m%d_%H%M%S')}"
    snapshot_run_id = broker_snapshot_run_id if broker_snapshot_run_id is not None else run_id

    try:
        from src.assembled_core.accounting.ledger_integration import build_ledger_from_trades

        orders_df = load_orders(freq, output_dir=base, strict=False)

        prices_df = None
        try:
            prices_df = load_prices_with_fallback(freq, output_dir=base)
        except Exception:
            logger.warning("Could not load prices for unrealized PnL calculation")

        if broker_snapshot_file:
            try:
                logger.info("Importing external broker snapshot from: %s", broker_snapshot_file)
                from src.assembled_core.accounting.broker_snapshot_importer import import_broker_snapshot

                if broker_snapshot_date is not None:
                    snapshot_date = pd.to_datetime(broker_snapshot_date, utc=True)
                elif (
                    not portfolio_trades_df.empty
                    and "timestamp" in portfolio_trades_df.columns
                ):
                    snapshot_date = pd.to_datetime(
                        portfolio_trades_df["timestamp"].max(), utc=True
                    )
                else:
                    snapshot_date = pd.Timestamp.now("UTC")

                import_result = import_broker_snapshot(
                    snapshot_path=Path(broker_snapshot_file),
                    run_id=snapshot_run_id,
                    snapshot_date=snapshot_date,
                    output_dir=base,
                    qty_tol=1e-8,
                    store_parquet=True,
                )
                logger.info(
                    "Imported broker snapshot: %s, cash=%s",
                    import_result["broker_snapshot_path"], import_result["cash"],
                )
            except Exception as e:
                logger.error("Failed to import broker snapshot: %s", e, exc_info=True)
                if broker_snapshot_policy == "require":
                    raise ValueError(
                        f"Broker snapshot import failed (policy=require): {e}"
                    ) from e

        ledger_result = build_ledger_from_trades(
            orders_df=orders_df,
            trades_df=portfolio_trades_df,
            run_id=run_id,
            output_dir=base,
            as_of_date=None,
            prices_df=prices_df,
            start_cash=start_capital,
            broker_snapshot_policy=broker_snapshot_policy,
            write_paper_broker_snapshot=write_paper_broker_snapshot,
            broker_snapshot_run_id=snapshot_run_id,
            write_evidence_pack=write_evidence_pack,
        )
        logger.info(
            "Ledger built: pack_path=%s, reconciliation_ok=%s",
            ledger_result["ledger_pack_path"], ledger_result["reconciliation_ok"],
        )
        return {"ledger_result": ledger_result, "completed": True, "failed": False}
    except ValueError:
        raise
    except Exception as e:
        logger.error("Ledger/Accounting step failed: %s", e, exc_info=True)
        return {"ledger_result": None, "completed": False, "failed": True}


def _eo_step_qa(
    freq: str,
    base: Path,
    start_capital: float,
    commission_bps: float | None,
    spread_w: float | None,
    impact_w: float | None,
) -> dict[str, Any]:
    """Run Step 5 QA (health + metrics + gates + report). Returns result dict."""
    out: dict[str, Any] = {
        "qa_result": None,
        "qa_metrics": None,
        "qa_gate_result": None,
        "qa_report_path_rel": None,
        "completed": False,
        "failed": False,
    }
    try:
        out["qa_result"] = aggregate_qa_status(freq, output_dir=base)
        qa_status = out["qa_result"].get("overall_status", "unknown")
        logger.info("QA health checks completed: overall_status=%s", qa_status)
        if qa_status == "error":
            logger.error("QA overall_status is 'error' - some checks failed")
        elif qa_status == "warning":
            logger.warning("QA overall_status is 'warning' - some checks have warnings")

        try:
            portfolio_equity_file = base / f"portfolio_equity_{freq}.csv"
            backtest_equity_file = base / f"equity_curve_{freq}.csv"

            if portfolio_equity_file.exists():
                equity_df = pd.read_csv(portfolio_equity_file, dtype={"timestamp": "string"})
                equity_df["timestamp"] = pd.to_datetime(equity_df["timestamp"], utc=True)
                logger.info("Using portfolio equity: %d rows", len(equity_df))
            elif backtest_equity_file.exists():
                equity_df = pd.read_csv(backtest_equity_file, dtype={"timestamp": "string"})
                equity_df["timestamp"] = pd.to_datetime(equity_df["timestamp"], utc=True)
                logger.info("Using backtest equity: %d rows", len(equity_df))
            else:
                logger.warning("No equity file found for metrics computation")
                equity_df = None

            orders_df = None
            try:
                orders_df = load_orders(freq, output_dir=base, strict=False)
                if orders_df.empty:
                    orders_df = None
            except Exception:
                pass

            if equity_df is not None and not equity_df.empty:
                qa_metrics = compute_all_metrics(
                    equity=equity_df,
                    trades=orders_df,
                    start_capital=start_capital,
                    freq=freq,
                    risk_free_rate=0.0,
                )
                out["qa_metrics"] = qa_metrics
                logger.info(
                    "Performance metrics computed: PF=%.4f, Sharpe=%s, CAGR=%s",
                    qa_metrics.final_pf, qa_metrics.sharpe_ratio, qa_metrics.cagr,
                )

                qa_gate_result = evaluate_all_gates(qa_metrics)
                out["qa_gate_result"] = qa_gate_result
                gate_status = qa_gate_result.overall_result.value
                passed = sum(1 for r in qa_gate_result.gate_results if r.result.value == "ok")
                warnings = sum(1 for r in qa_gate_result.gate_results if r.result.value == "warning")
                blocked = sum(1 for r in qa_gate_result.gate_results if r.result.value == "block")
                logger.info(
                    "QA gates: overall=%s (passed=%d, warnings=%d, blocked=%d)",
                    gate_status, passed, warnings, blocked,
                )
                if qa_gate_result.overall_result == QAResult.BLOCK:
                    logger.error("QA gates BLOCKED - strategy does not meet quality thresholds")
                elif qa_gate_result.overall_result == QAResult.WARNING:
                    logger.warning("QA gates WARNING - some quality thresholds not met")

                try:
                    cost_model = get_default_cost_model()
                    config_info = {
                        "strategy": "eod_pipeline_core",
                        "freq": freq,
                        "start_capital": start_capital,
                        "ema_fast": get_default_ema_config(freq).fast,
                        "ema_slow": get_default_ema_config(freq).slow,
                        "commission_bps": commission_bps if commission_bps is not None else cost_model.commission_bps,
                        "spread_w": spread_w if spread_w is not None else cost_model.spread_w,
                        "impact_w": impact_w if impact_w is not None else cost_model.impact_w,
                    }
                    equity_curve_path = (
                        portfolio_equity_file if portfolio_equity_file.exists()
                        else (backtest_equity_file if backtest_equity_file.exists() else None)
                    )
                    qa_report_path = generate_qa_report(
                        metrics=qa_metrics,
                        gate_result=qa_gate_result,
                        strategy_name="eod_pipeline_core",
                        freq=freq,
                        equity_curve_path=equity_curve_path,
                        data_start_date=qa_metrics.start_date,
                        data_end_date=qa_metrics.end_date,
                        config_info=config_info,
                        output_dir=base / "reports",
                    )
                    out["qa_report_path_rel"] = qa_report_path.relative_to(base)
                    logger.info("QA report written: %s", qa_report_path)
                except Exception as e:
                    logger.warning("QA report generation failed: %s", e, exc_info=True)
            else:
                logger.warning("Cannot compute QA metrics: no equity data available")
        except Exception as e:
            logger.warning("QA metrics/gates computation failed: %s", e, exc_info=True)

        out["completed"] = True
    except Exception as e:
        logger.error("ERROR in QA step: %s", e, exc_info=True)
        out["failed"] = True
    return out


def _eo_snapshot_id(
    prices: pd.DataFrame,
    freq: str,
    price_file: str | None,
    data_source: str | None,
) -> str | None:
    """Compute deterministic data snapshot ID for run manifest reproducibility."""
    try:
        from src.assembled_core.data.snapshot import compute_price_panel_snapshot_id

        source_meta: dict[str, str] = {}
        if price_file:
            source_meta["file"] = str(price_file)
        if data_source:
            source_meta["source"] = str(data_source)

        snap_id = compute_price_panel_snapshot_id(
            prices=prices,
            freq=freq,
            source_meta=source_meta if source_meta else None,
        )
        logger.info("Data snapshot ID computed: %s...", snap_id[:16])
        return snap_id
    except Exception as exc:
        logger.warning("Failed to compute data snapshot ID: %s", exc, exc_info=True)
        return None


def _eo_build_manifest(
    *,
    freq: str,
    start_capital: float,
    data_snapshot_id: str | None,
    completed_steps: list[str],
    qa: dict[str, Any],
    ledger_result: dict[str, Any] | None,
    started_at: datetime,
    finished_at: datetime,
    failure_flag: bool,
    base: Path,
) -> dict[str, Any]:
    """Build the run manifest dict from pipeline step outputs."""
    qa_result = qa.get("qa_result")
    qa_metrics = qa.get("qa_metrics")
    qa_gate_result = qa.get("qa_gate_result")
    qa_report_path_rel = qa.get("qa_report_path_rel")

    return {
        "schema_version": 1,
        "freq": freq,
        "start_capital": start_capital,
        "data_snapshot_id": data_snapshot_id,
        "completed_steps": completed_steps,
        "qa_overall_status": qa_result["overall_status"] if qa_result else None,
        "qa_checks": qa_result["checks"] if qa_result else [],
        "qa_metrics": _metrics_to_dict(qa_metrics) if qa_metrics else None,
        "qa_gate_result": _gate_result_to_dict(qa_gate_result) if qa_gate_result else None,
        "qa_report_path": (
            _manifest_path_str(qa_report_path_rel, base_dir=base) if qa_report_path_rel else None
        ),
        "robustness_pack_path": None,
        "wf_oos_metrics": None,
        "plateau_score": None,
        "sensitivity_summary": None,
        "crisis_summary": None,
        "deflated_sharpe": None,
        "multiple_testing_warning": None,
        "robustness_ok": None,
        "ledger_pack_path": (
            _manifest_path_str(ledger_result.get("ledger_pack_path"), base_dir=base)
            if ledger_result else None
        ),
        "reconcile_report_path": (
            _manifest_path_str(ledger_result.get("reconcile_report_path"), base_dir=base)
            if ledger_result else None
        ),
        "accounting_report_path": (
            _manifest_path_str(ledger_result.get("accounting_report_path"), base_dir=base)
            if ledger_result else None
        ),
        "evidence_index_path": (
            _manifest_path_str(ledger_result.get("evidence_index_path"), base_dir=base)
            if ledger_result else None
        ),
        "evidence_pack_path": (
            _manifest_path_str(ledger_result.get("evidence_pack_path"), base_dir=base)
            if ledger_result else None
        ),
        "evidence_pack_manifest_path": (
            _manifest_path_str(ledger_result.get("evidence_pack_manifest_path"), base_dir=base)
            if ledger_result else None
        ),
        "broker_snapshot_path": (
            _manifest_path_str(ledger_result.get("broker_snapshot_path"), base_dir=base)
            if ledger_result else None
        ),
        "reconciliation_ok": (
            ledger_result["reconciliation_ok"] if ledger_result else None
        ),
        "timestamps": {
            "started": started_at.isoformat(),
            "finished": finished_at.isoformat(),
        },
        "failure": failure_flag,
    }


def _eo_post_steps(base: Path) -> None:
    """Run non-blocking post-pipeline steps: feedback loop, news attribution, TCA."""
    try:
        from src.assembled_core.ml.feedback_loop import FeedbackLoopController  # type: ignore

        _fl = FeedbackLoopController()
        _fl_result = _fl.run_feedback_check(
            learning_store_path=base / "ops" / "learning_store.jsonl",
            current_model_path=base / "models" / "meta_model_current.joblib",
            panel_df=pd.DataFrame(),
        )
        logger.info(
            "[EOD][Feedback] signals=%d retrain_triggered=%s blocked=%s",
            _fl_result.active_signal_count,
            _fl_result.retrain_triggered,
            bool(_fl_result.blocked_reason),
        )
    except Exception as _fl_exc:
        logger.warning("[EOD][Feedback] Non-blocking Fehler (ignoriert): %s", _fl_exc)

    try:
        from src.assembled_core.intel.news_trade_attribution import NewsTradeAttributor  # type: ignore

        _ls_path = base / "ops" / "learning_store.jsonl"
        _news_path = base / "intel" / "news_event_store.jsonl"
        if _ls_path.exists() and _news_path.exists():
            _attributor = NewsTradeAttributor()
            _n_enriched = _attributor.enrich_learning_store(
                learning_store_path=_ls_path,
                news_events_path=_news_path,
            )
            logger.info("[EOD][NewsAttr] enriched %d Trade-Records mit news_links", _n_enriched)
    except Exception as _na_exc:
        logger.warning("[EOD][NewsAttr] Non-blocking Fehler: %s", _na_exc)

    try:
        import json as _json

        from src.assembled_core.qa.trade_tca import run_tca_from_learning_store  # type: ignore

        _ls_path = base / "ops" / "learning_store.jsonl"
        _tca_date_str: str | None = None
        try:
            if _ls_path.exists():
                _max_ts = None
                with _ls_path.open("r", encoding="utf-8") as _fh:
                    for _line in _fh:
                        _line = _line.strip()
                        if not _line:
                            continue
                        try:
                            _rec = _json.loads(_line)
                        except Exception:
                            continue
                        _t = _rec.get("timestamp") or _rec.get("execution_time") or _rec.get("closed_at")
                        if _t is None:
                            continue
                        _ts = pd.to_datetime(_t, utc=True, errors="coerce")
                        if pd.isna(_ts):
                            continue
                        if _max_ts is None or _ts > _max_ts:
                            _max_ts = _ts
                if _max_ts is not None:
                    _tca_date_str = _max_ts.strftime("%Y%m%d")
        except Exception:
            _tca_date_str = None

        if _tca_date_str is None:
            _tca_date_str = pd.Timestamp.now("UTC").strftime("%Y%m%d")
            logger.warning(
                "[EOD][TCA] Kein Trade-Timestamp im learning_store — fallback auf wall-clock-Datum %s",
                _tca_date_str,
            )

        _tca_out = base / "ops" / f"tca_report_{_tca_date_str}.json"
        if _ls_path.exists():
            _tca_result = run_tca_from_learning_store(_ls_path, _tca_out)
            if _tca_result:
                logger.info(
                    "[EOD][TCA] %d Trades analysiert, mean_impact_bps=%.2f",
                    _tca_result.get("n_trades", 0),
                    _tca_result.get("mean_impact_bps", 0.0),
                )
            try:
                from src.assembled_core.ops.report_retention import purge_old_dated_reports
                purge_old_dated_reports(_tca_out.parent, "tca_report_", ".json", keep_last_n=60)
            except OSError as _ret_exc:
                logger.debug("[EOD][TCA] Retention-Purge IO-Fehler: %s", _ret_exc)
    except Exception as _tca_exc:
        logger.warning("[EOD][TCA] Non-blocking Fehler: %s", _tca_exc)


def run_eod_pipeline(
    freq: str,
    start_capital: float = 10000.0,
    skip_backtest: bool = False,
    skip_portfolio: bool = False,
    skip_qa: bool = False,
    output_dir: Path | None = None,
    price_file: str | None = None,
    commission_bps: float | None = None,
    spread_w: float | None = None,
    impact_w: float | None = None,
    data_source: str | None = None,
    symbols: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    # Broker snapshot controls (Sprint 13 extension)
    broker_snapshot_policy: str = "prefer",
    write_paper_broker_snapshot: bool = False,
    broker_snapshot_run_id: str | None = None,
    broker_snapshot_file: str | Path | None = None,
    broker_snapshot_date: str | None = None,
    # Evidence pack controls
    write_evidence_pack: bool = False,
) -> dict[str, Any]:
    """Run full EOD pipeline for a given frequency.

    Args:
        freq: Trading frequency ("1d" or "5min")
        start_capital: Starting capital
        skip_backtest: If True, skip backtest step
        skip_portfolio: If True, skip portfolio step
        skip_qa: If True, skip QA step
        output_dir: Base output directory (default: None, uses config.OUTPUT_DIR)
        price_file: Optional explicit path to price file (for local source only)
        commission_bps: Commission in basis points (default: from cost model)
        spread_w: Spread weight (default: from cost model)
        impact_w: Impact weight (default: from cost model)
        data_source: Data source type ("local" or "yahoo"). If None, uses settings.data_source
        symbols: List of symbols to load. If None, uses default_universe from settings or watchlist
        start_date: Start date in format "YYYY-MM-DD" or "today". If None, uses all available data
        end_date: End date in format "YYYY-MM-DD" or "today". If None, uses all available data

    Returns:
        Dictionary with run manifest data

    Side effects:
        Executes pipeline steps and writes run_manifest_{freq}.json
    """
    if freq not in SUPPORTED_FREQS:
        raise ValueError(f"Unsupported frequency: {freq}. Supported: {SUPPORTED_FREQS}")

    base = output_dir if output_dir else OUTPUT_DIR
    started_at = datetime.now(tz=timezone.utc)
    completed_steps: list[str] = []
    failure_flag = False

    # Step 1: Load price data
    logger.info("Step 1: Load price data")
    prices, load_failed = _eo_load_prices(
        freq, symbols, data_source, price_file, start_date, end_date, base
    )
    if load_failed:
        failure_flag = True

    # Step 2: Execute
    try:
        logger.info("Step 2: Execute")
        orders_path, orders = run_execute_step(freq, output_dir=base, price_file=price_file)
        logger.info("Orders written: %s | rows=%d", orders_path, len(orders))
        completed_steps.append("execute")
    except Exception as e:
        logger.error("ERROR in execute step: %s", e, exc_info=True)
        failure_flag = True

    # Step 3: Backtest
    if not skip_backtest:
        try:
            logger.info("Step 3: Backtest")
            curve_path, rep_path = run_backtest_step(
                freq, start_capital, output_dir=base, price_file=price_file
            )
            logger.info("Backtest written: %s, %s", curve_path, rep_path)
            completed_steps.append("backtest")
        except Exception as e:
            logger.error("ERROR in backtest step: %s", e, exc_info=True)
            failure_flag = True
    else:
        logger.info("Step 3: Backtest (SKIPPED)")

    # Step 4: Portfolio
    portfolio_trades_df = None
    if not skip_portfolio:
        try:
            logger.info("Step 4: Portfolio")
            eq_path, rep_path, portfolio_trades_df = run_portfolio_step(
                freq, start_capital,
                commission_bps=commission_bps, spread_w=spread_w, impact_w=impact_w,
                output_dir=base,
            )
            logger.info("Portfolio written: %s, %s", eq_path, rep_path)
            completed_steps.append("portfolio")
        except Exception as e:
            logger.error("ERROR in portfolio step: %s", e, exc_info=True)
            failure_flag = True
    else:
        logger.info("Step 4: Portfolio (SKIPPED)")

    # Step 4b: Ledger/Accounting
    ledger_result = None
    if not skip_portfolio and portfolio_trades_df is not None and not portfolio_trades_df.empty:
        logger.info("Step 4b: Ledger/Accounting")
        _ledger = _eo_step_ledger(
            freq=freq, base=base, started_at=started_at, start_capital=start_capital,
            portfolio_trades_df=portfolio_trades_df,
            broker_snapshot_policy=broker_snapshot_policy,
            broker_snapshot_file=broker_snapshot_file,
            broker_snapshot_date=broker_snapshot_date,
            broker_snapshot_run_id=broker_snapshot_run_id,
            write_evidence_pack=write_evidence_pack,
            write_paper_broker_snapshot=write_paper_broker_snapshot,
        )
        ledger_result = _ledger["ledger_result"]
        if _ledger["completed"]:
            completed_steps.append("ledger")
        if _ledger["failed"]:
            failure_flag = True
    else:
        logger.info("Step 4b: Ledger/Accounting (SKIPPED - no trades available)")

    # Step 5: QA
    if not skip_qa:
        logger.info("Step 5: QA")
        _qa = _eo_step_qa(freq, base, start_capital, commission_bps, spread_w, impact_w)
        if _qa["completed"]:
            completed_steps.append("qa")
        if _qa["failed"]:
            failure_flag = True
    else:
        logger.info("Step 5: QA (SKIPPED)")
        _qa = {}

    finished_at = datetime.now(tz=timezone.utc)

    data_snapshot_id = _eo_snapshot_id(prices, freq, price_file, data_source)

    manifest = _eo_build_manifest(
        freq=freq, start_capital=start_capital, data_snapshot_id=data_snapshot_id,
        completed_steps=completed_steps, qa=_qa, ledger_result=ledger_result,
        started_at=started_at, finished_at=finished_at, failure_flag=failure_flag, base=base,
    )

    manifest_path = base / f"run_manifest_{freq}.json"
    try:
        _write_manifest_json(manifest_path, manifest)
        if ledger_result and ledger_result.get("evidence_index_path"):
            _backfill_evidence_index_manifest_path(
                base_dir=base, ledger_result=ledger_result, manifest_path=manifest_path,
            )
            _backfill_evidence_index_accounting_path(base_dir=base, ledger_result=ledger_result)
    except (IOError, OSError) as exc:
        logger.error("Failed to write manifest to %s: %s", manifest_path, exc)
        raise RuntimeError(f"Failed to write manifest to {manifest_path}") from exc
    except (TypeError, ValueError) as exc:
        logger.error("Failed to serialize manifest to JSON: %s", exc)
        raise ValueError(f"Failed to serialize manifest to JSON: {manifest_path}") from exc

    logger.info("Manifest written: %s", manifest_path)

    _eo_post_steps(base)

    return manifest
