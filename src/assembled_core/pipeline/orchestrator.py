# src/assembled_core/pipeline/orchestrator.py
"""Pipeline orchestration for EOD runs."""

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
from src.assembled_core.pipeline.signals import compute_ema_signals
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
            except Exception:
                pass


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
                except Exception:
                    # Ignore cleanup errors
                    pass

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
                except Exception:
                    pass

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

    # Determine signal mode from policy
    signal_mode = "ema"
    try:
        import yaml
        _policy_path = Path("configs/policy.yaml")
        if _policy_path.exists():
            with open(_policy_path, "r", encoding="utf-8") as _pf:
                _policy = yaml.safe_load(_pf) or {}
            signal_mode = (_policy.get("signal_generation") or {}).get("mode", "ema")
    except Exception as exc:
        logger.debug("Could not read signal_generation.mode from policy: %s", exc)

    # Compute signals based on mode. A multifactor config that silently
    # downgrades to EMA on any exception is a structurally different
    # strategy with a different risk profile — elevate the exception log
    # level from WARNING to ERROR so the fallback is visible in ops feeds.
    # Full silent-fallback attribution tracking requires the run_manifest
    # path (see run_eod_pipeline) — noted as follow-up.
    if signal_mode == "multifactor":
        try:
            from src.assembled_core.strategies.multifactor_v2 import (
                compute_signals as mf_compute_signals,
            )
            signals = mf_compute_signals(prices)
            logger.info("[ORCHESTRATOR] Signal mode: multifactor_v2 (%d signals)", len(signals))
        except Exception as exc:
            logger.error(
                "[ORCHESTRATOR] multifactor_v2 failed, falling back to EMA: %s",
                exc, exc_info=True,
            )
            ema_config = get_default_ema_config(freq)
            signals = compute_ema_signals(prices, ema_config.fast, ema_config.slow)
    elif signal_mode == "ml_enhanced":
        # Phase 2+: Meta-model enhanced signals (placeholder until meta-model trained)
        logger.info("[ORCHESTRATOR] Signal mode: ml_enhanced (not yet trained, using multifactor)")
        try:
            from src.assembled_core.strategies.multifactor_v2 import (
                compute_signals as mf_compute_signals,
            )
            signals = mf_compute_signals(prices)
        except Exception as exc:
            logger.error(
                "[ORCHESTRATOR] ml_enhanced -> multifactor fallback failed: %s",
                exc, exc_info=True,
            )
            ema_config = get_default_ema_config(freq)
            signals = compute_ema_signals(prices, ema_config.fast, ema_config.slow)
    else:
        # Default: legacy EMA signals
        ema_config = get_default_ema_config(freq)
        signals = compute_ema_signals(prices, ema_config.fast, ema_config.slow)
        logger.info("[ORCHESTRATOR] Signal mode: ema (fast=%d, slow=%d)", ema_config.fast, ema_config.slow)

    # --- Phase 9: Post-signal enrichment (same as trading_cycle) ---
    try:
        _policy_for_enrichment: dict = {}
        try:
            import yaml as _yaml
            _pe_path = Path("configs/policy.yaml")
            if _pe_path.exists():
                with open(_pe_path, "r", encoding="utf-8") as _pef:
                    _policy_for_enrichment = _yaml.safe_load(_pef) or {}
        except Exception as _pe_exc:
            logger.debug("[ORCHESTRATOR] Could not load policy for enrichment: %s", _pe_exc)
        _enrich_signals_post_generation(signals, prices, _policy_for_enrichment)
    except Exception as exc:
        logger.debug("[ORCHESTRATOR] Post-signal enrichment skipped: %s", exc)

    # Normalize signal schema: multifactor_v2 / ml_enhanced emit
    # `direction`/`score` columns; signals_to_orders requires `sig`/`price`.
    # Bridge the two so downstream order-gen logic stays schema-stable.
    signals = _normalize_signals_schema(signals, prices)

    # Generate orders
    orders = signals_to_orders(signals)

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

    completed_steps = []
    failure_flag = False

    # Step 1: Load price data (using data source abstraction)
    try:
        from src.assembled_core.config.settings import get_settings
        from src.assembled_core.data.data_source import get_price_data_source

        settings = get_settings()

        # Determine data source
        source_type = data_source if data_source is not None else settings.data_source

        # Determine symbols
        if symbols is None:
            # Try to read from watchlist file if it exists
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
                        f"Failed to read watchlist file {settings.watchlist_file}: {exc}. Using default universe."
                    )
                    symbols = settings.default_universe
                if not symbols:
                    symbols = settings.default_universe
            else:
                symbols = settings.default_universe

        # Determine date range
        if start_date is None:
            start_date = "2020-01-01"  # Default: use all available data
        if end_date is None:
            end_date = "today"  # Default: use current date

        # Get data source and load prices
        price_source = get_price_data_source(
            settings=settings, data_source=source_type, price_file=price_file
        )

        logger.info(f"Loading prices from {source_type} source...")
        logger.info(
            f"Symbols: {symbols[:10]}{'...' if len(symbols) > 10 else ''} ({len(symbols)} total)"
        )
        logger.info(f"Date range: {start_date} to {end_date}")

        prices = price_source.get_history(
            symbols=symbols, start_date=start_date, end_date=end_date, freq=freq
        )

        logger.info(
            f"Price data OK: {len(prices)} rows, {prices['symbol'].nunique()} symbols"
        )

        # If using online source, optionally save to local file for caching
        if source_type == "yahoo" and len(prices) > 0:
            cache_path = base / "aggregates" / f"{freq}_live_cache.parquet"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            prices.to_parquet(cache_path, index=False)
            logger.info(f"Cached live data to {cache_path}")

    except FileNotFoundError as e:
        logger.error(f"Price data not found: {e}")
        failure_flag = True
        prices = pd.DataFrame()  # Empty DataFrame to prevent downstream errors
    except Exception as e:
        logger.error(f"Failed to load price data: {e}", exc_info=True)
        failure_flag = True
        prices = pd.DataFrame()  # Empty DataFrame to prevent downstream errors

    # Step 2: Execute
    try:
        logger.info("Step 2: Execute")
        orders_path, orders = run_execute_step(
            freq, output_dir=base, price_file=price_file
        )
        logger.info(f"Orders written: {orders_path} | rows={len(orders)}")
        completed_steps.append("execute")
    except Exception as e:
        logger.error(f"ERROR in execute step: {e}", exc_info=True)
        failure_flag = True

    # Step 3: Backtest
    if not skip_backtest:
        try:
            logger.info("Step 3: Backtest")
            curve_path, rep_path = run_backtest_step(
                freq, start_capital, output_dir=base, price_file=price_file
            )
            logger.info(f"Backtest written: {curve_path}, {rep_path}")
            completed_steps.append("backtest")
        except Exception as e:
            logger.error(f"ERROR in backtest step: {e}", exc_info=True)
            failure_flag = True
    else:
        logger.info("Step 3: Backtest (SKIPPED)")

    # Step 4: Portfolio
    portfolio_trades_df = None
    if not skip_portfolio:
        try:
            logger.info("Step 4: Portfolio")
            eq_path, rep_path, portfolio_trades_df = run_portfolio_step(
                freq,
                start_capital,
                commission_bps=commission_bps,
                spread_w=spread_w,
                impact_w=impact_w,
                output_dir=base,
            )
            logger.info(f"Portfolio written: {eq_path}, {rep_path}")
            completed_steps.append("portfolio")
        except Exception as e:
            logger.error(f"ERROR in portfolio step: {e}", exc_info=True)
            failure_flag = True
    else:
        logger.info("Step 4: Portfolio (SKIPPED)")

    # Step 4b: Ledger/Accounting (Sprint 13 L5)
    ledger_result = None
    if (
        not skip_portfolio
        and portfolio_trades_df is not None
        and not portfolio_trades_df.empty
    ):
        try:
            logger.info("Step 4b: Ledger/Accounting")
            from src.assembled_core.accounting.ledger_integration import (
                build_ledger_from_trades,
            )

            # Generate run_id from timestamp
            run_id = f"run_{started_at.strftime('%Y%m%d_%H%M%S')}"

            # Load orders for ORDER_SUBMIT events
            orders_df = load_orders(freq, output_dir=base, strict=False)

            # Load prices for unrealized PnL
            prices_df = None
            try:
                prices_df = load_prices_with_fallback(freq, output_dir=base)
            except Exception:
                logger.warning("Could not load prices for unrealized PnL calculation")

            # Default: use run_id as snapshot namespace unless explicitly overridden
            snapshot_run_id = (
                broker_snapshot_run_id if broker_snapshot_run_id is not None else run_id
            )

            # Step 4b.1: Import external broker snapshot if provided
            if broker_snapshot_file:
                try:
                    logger.info(
                        f"Importing external broker snapshot from: {broker_snapshot_file}"
                    )
                    from src.assembled_core.accounting.broker_snapshot_importer import (
                        import_broker_snapshot,
                    )

                    # Determine snapshot date (use provided date, or last trade date, or today)
                    snapshot_date = broker_snapshot_date
                    if snapshot_date is None:
                        if (
                            not portfolio_trades_df.empty
                            and "timestamp" in portfolio_trades_df.columns
                        ):
                            snapshot_date = pd.to_datetime(
                                portfolio_trades_df["timestamp"].max(), utc=True
                            )
                        else:
                            snapshot_date = pd.Timestamp.now("UTC")
                    else:
                        snapshot_date = pd.to_datetime(snapshot_date, utc=True)

                    # Import snapshot
                    import_result = import_broker_snapshot(
                        snapshot_path=Path(broker_snapshot_file),
                        run_id=snapshot_run_id,
                        snapshot_date=snapshot_date,
                        output_dir=base,
                        qty_tol=1e-8,
                        store_parquet=True,
                    )
                    logger.info(
                        f"Imported broker snapshot: {import_result['broker_snapshot_path']}, "
                        f"cash={import_result['cash']}"
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to import broker snapshot: {e}", exc_info=True
                    )
                    # If policy is require, we should fail here
                    if broker_snapshot_policy == "require":
                        raise ValueError(
                            f"Broker snapshot import failed (policy=require): {e}"
                        ) from e
                    # Otherwise, log and continue (snapshot might still exist from previous import)

            # Build ledger
            ledger_result = build_ledger_from_trades(
                orders_df=orders_df,
                trades_df=portfolio_trades_df,
                run_id=run_id,
                output_dir=base,
                as_of_date=None,  # Use last trade timestamp
                prices_df=prices_df,
                start_cash=start_capital,
                broker_snapshot_policy=broker_snapshot_policy,
                write_paper_broker_snapshot=write_paper_broker_snapshot,
                broker_snapshot_run_id=snapshot_run_id,
                write_evidence_pack=write_evidence_pack,
            )

            logger.info(
                f"Ledger built: pack_path={ledger_result['ledger_pack_path']}, "
                f"reconciliation_ok={ledger_result['reconciliation_ok']}"
            )
            completed_steps.append("ledger")
        except ValueError:
            # Important: do NOT swallow fail-fast policy errors (e.g. policy="require")
            raise
        except Exception as e:
            logger.error(f"Ledger/Accounting step failed: {e}", exc_info=True)
            # Ledger is labeled "optional" at the pipeline level, but a
            # silent ledger failure produces downstream QA on inconsistent
            # equity/trades and can surface as green overall status. Set
            # failure_flag so the run manifest records the degradation,
            # even though the pipeline does not abort.
            ledger_result = None
            failure_flag = True
    else:
        logger.info("Step 4b: Ledger/Accounting (SKIPPED - no trades available)")

    # Step 5: QA
    qa_result = None
    qa_metrics = None
    qa_gate_result = None
    qa_report_path_rel = None
    if not skip_qa:
        try:
            logger.info("Step 5: QA")

            # 5a: Health checks (existing)
            qa_result = aggregate_qa_status(freq, output_dir=base)
            qa_status = qa_result.get("overall_status", "unknown")
            logger.info(f"QA health checks completed: overall_status={qa_status}")

            if qa_status == "error":
                logger.error("QA overall_status is 'error' - some checks failed")
            elif qa_status == "warning":
                logger.warning(
                    "QA overall_status is 'warning' - some checks have warnings"
                )

            # 5b: Performance metrics (new)
            try:
                logger.info("Step 5b: Computing performance metrics")
                # Load portfolio equity (preferred) or backtest equity
                portfolio_equity_file = base / f"portfolio_equity_{freq}.csv"
                backtest_equity_file = base / f"equity_curve_{freq}.csv"

                if portfolio_equity_file.exists():
                    equity_df = pd.read_csv(portfolio_equity_file)
                    equity_df["timestamp"] = pd.to_datetime(
                        equity_df["timestamp"], utc=True
                    )
                    logger.info(f"Using portfolio equity: {len(equity_df)} rows")
                elif backtest_equity_file.exists():
                    equity_df = pd.read_csv(backtest_equity_file)
                    equity_df["timestamp"] = pd.to_datetime(
                        equity_df["timestamp"], utc=True
                    )
                    logger.info(f"Using backtest equity: {len(equity_df)} rows")
                else:
                    logger.warning("No equity file found for metrics computation")
                    equity_df = None

                # Load trades if available
                orders_df = None
                try:
                    orders_df = load_orders(freq, output_dir=base, strict=False)
                    if orders_df.empty:
                        orders_df = None
                except Exception:
                    pass  # Orders optional for metrics

                if equity_df is not None and not equity_df.empty:
                    qa_metrics = compute_all_metrics(
                        equity=equity_df,
                        trades=orders_df,
                        start_capital=start_capital,
                        freq=freq,
                        risk_free_rate=0.0,
                    )
                    logger.info(
                        f"Performance metrics computed: PF={qa_metrics.final_pf:.4f}, Sharpe={qa_metrics.sharpe_ratio}, CAGR={qa_metrics.cagr}"
                    )

                    # 5c: QA gates (new)
                    logger.info("Step 5c: Evaluating QA gates")
                    qa_gate_result = evaluate_all_gates(qa_metrics)
                    gate_status = qa_gate_result.overall_result.value
                    passed = sum(
                        1 for r in qa_gate_result.gate_results if r.result.value == "ok"
                    )
                    warnings = sum(
                        1
                        for r in qa_gate_result.gate_results
                        if r.result.value == "warning"
                    )
                    blocked = sum(
                        1
                        for r in qa_gate_result.gate_results
                        if r.result.value == "block"
                    )
                    logger.info(
                        f"QA gates completed: overall_result={gate_status} (passed={passed}, warnings={warnings}, blocked={blocked})"
                    )

                    if qa_gate_result.overall_result == QAResult.BLOCK:
                        logger.error(
                            "QA gates BLOCKED - strategy does not meet quality thresholds"
                        )
                        # Don't set failure_flag here - gates are informational, not blocking
                    elif qa_gate_result.overall_result == QAResult.WARNING:
                        logger.warning(
                            "QA gates WARNING - some quality thresholds not met"
                        )

                    # 5d: Generate QA report
                    try:
                        logger.info("Step 5d: Generating QA report")

                        # Build config info for report
                        # Get actual cost parameters used (CLI overrides or defaults)
                        cost_model = get_default_cost_model()
                        final_commission_bps = (
                            commission_bps
                            if commission_bps is not None
                            else cost_model.commission_bps
                        )
                        final_spread_w = (
                            spread_w if spread_w is not None else cost_model.spread_w
                        )
                        final_impact_w = (
                            impact_w if impact_w is not None else cost_model.impact_w
                        )

                        ema_config = get_default_ema_config(freq)
                        config_info = {
                            "strategy": "eod_pipeline_core",
                            "freq": freq,
                            "start_capital": start_capital,
                            "ema_fast": ema_config.fast,
                            "ema_slow": ema_config.slow,
                            "commission_bps": final_commission_bps,
                            "spread_w": final_spread_w,
                            "impact_w": final_impact_w,
                        }

                        # Determine equity curve path for report
                        equity_curve_path = None
                        if portfolio_equity_file.exists():
                            equity_curve_path = portfolio_equity_file
                        elif backtest_equity_file.exists():
                            equity_curve_path = backtest_equity_file

                        # Generate report
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

                        # Convert to relative path for manifest (relative to base output dir)
                        qa_report_path_rel = qa_report_path.relative_to(base)
                        logger.info(f"QA report written: {qa_report_path}")

                    except Exception as e:
                        logger.warning(
                            f"QA report generation failed: {e}", exc_info=True
                        )
                        qa_report_path_rel = None
                        # Don't fail the pipeline if report generation fails - it's optional
                else:
                    logger.warning(
                        "Cannot compute QA metrics: no equity data available"
                    )
                    qa_report_path_rel = None

            except Exception as e:
                logger.warning(
                    f"QA metrics/gates computation failed: {e}", exc_info=True
                )
                # Don't fail the pipeline if metrics/gates fail - they're optional
                qa_report_path_rel = None

            completed_steps.append("qa")
        except Exception as e:
            logger.error(f"ERROR in QA step: {e}", exc_info=True)
            failure_flag = True
    else:
        logger.info("Step 5: QA (SKIPPED)")

    finished_at = datetime.now(tz=timezone.utc)

    # Compute data snapshot ID (D4)
    # Berechne genau einmal nachdem Preise geladen wurden (nicht pro Timestamp)
    data_snapshot_id = None
    try:
        from src.assembled_core.data.snapshot import compute_price_panel_snapshot_id

        # Auch bei leeren Preisen: Empty-Semantik aus D3 (stabiler Hash)
        # Build source_meta from available information (nur wenn vorhanden und deterministisch)
        source_meta = {}
        if price_file:
            source_meta["file"] = str(price_file)
        if data_source:
            source_meta["source"] = str(data_source)

        data_snapshot_id = compute_price_panel_snapshot_id(
            prices=prices,  # Kann leer sein (Empty-Semantik)
            freq=freq,
            source_meta=source_meta if source_meta else None,
        )
        logger.info(f"Data snapshot ID computed: {data_snapshot_id[:16]}...")
    except Exception as exc:
        logger.warning(f"Failed to compute data snapshot ID: {exc}", exc_info=True)
        # Bei Fehler: data_snapshot_id bleibt None (wird im Manifest als None gespeichert)

    # Build manifest
    manifest = {
        "schema_version": 1,
        "freq": freq,
        "start_capital": start_capital,
        "data_snapshot_id": data_snapshot_id,  # D4: Snapshot ID for reproducibility
        "completed_steps": completed_steps,
        "qa_overall_status": qa_result["overall_status"] if qa_result else None,
        "qa_checks": qa_result["checks"] if qa_result else [],
        "qa_metrics": _metrics_to_dict(qa_metrics) if qa_metrics else None,
        "qa_gate_result": (
            _gate_result_to_dict(qa_gate_result) if qa_gate_result else None
        ),
        "qa_report_path": (
            _manifest_path_str(qa_report_path_rel, base_dir=base)
            if qa_report_path_rel
            else None
        ),
        # Sprint 12: Robustness Pack fields (backward compatible: None if not run)
        "robustness_pack_path": None,
        "wf_oos_metrics": None,
        "plateau_score": None,
        "sensitivity_summary": None,
        "crisis_summary": None,
        "deflated_sharpe": None,
        "multiple_testing_warning": None,
        "robustness_ok": None,
        # Sprint 13 L5: Ledger/Accounting fields
        "ledger_pack_path": (
            _manifest_path_str(ledger_result.get("ledger_pack_path"), base_dir=base)
            if ledger_result
            else None
        ),
        "reconcile_report_path": (
            _manifest_path_str(
                ledger_result.get("reconcile_report_path"), base_dir=base
            )
            if ledger_result
            else None
        ),
        "accounting_report_path": (
            _manifest_path_str(
                ledger_result.get("accounting_report_path"), base_dir=base
            )
            if ledger_result
            else None
        ),
        "evidence_index_path": (
            _manifest_path_str(ledger_result.get("evidence_index_path"), base_dir=base)
            if ledger_result
            else None
        ),
        "evidence_pack_path": (
            _manifest_path_str(ledger_result.get("evidence_pack_path"), base_dir=base)
            if ledger_result
            else None
        ),
        "evidence_pack_manifest_path": (
            _manifest_path_str(
                ledger_result.get("evidence_pack_manifest_path"), base_dir=base
            )
            if ledger_result
            else None
        ),
        "broker_snapshot_path": (
            _manifest_path_str(ledger_result.get("broker_snapshot_path"), base_dir=base)
            if ledger_result
            else None
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

    # Write manifest
    manifest_path = base / f"run_manifest_{freq}.json"
    try:
        _write_manifest_json(manifest_path, manifest)
        # Best-effort backfill: ensure Evidence Index references the manifest if it exists.
        if ledger_result and ledger_result.get("evidence_index_path"):
            _backfill_evidence_index_manifest_path(
                base_dir=base,
                ledger_result=ledger_result,
                manifest_path=manifest_path,
            )
            _backfill_evidence_index_accounting_path(
                base_dir=base,
                ledger_result=ledger_result,
            )
    except (IOError, OSError) as exc:
        logger.error("Failed to write manifest to %s: %s", manifest_path, exc)
        raise RuntimeError(f"Failed to write manifest to {manifest_path}") from exc
    except (TypeError, ValueError) as exc:
        logger.error("Failed to serialize manifest to JSON: %s", exc)
        raise ValueError(
            f"Failed to serialize manifest to JSON: {manifest_path}"
        ) from exc

    logger.info(f"Manifest written: {manifest_path}")

    return manifest
