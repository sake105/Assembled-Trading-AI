"""_tc_execution — route_orders() and book_fills() extracted from trading_cycle_v2."""

from __future__ import annotations

import logging

import pandas as pd
from src.assembled_core.config.policy_loader import load_policy
from src.assembled_core.pipeline.trading_cycle_shared import (
    TradingContext,
    TradingCycleResult,
    _apply_group_exposure_caps,
    _apply_pre_trade_impact,
    _generate_orders_default,
)

logger = logging.getLogger(__name__)


def route_orders(
    targets: pd.DataFrame,
    ctx: TradingContext,
    *,
    prices_filtered: pd.DataFrame | None = None,
    prices_with_features: pd.DataFrame | None = None,
    prices_latest: pd.DataFrame | None = None,
    do_rebal: bool = True,
    log: logging.Logger | None = None,
) -> pd.DataFrame:
    """Generate orders from approved target positions.

    Steps kept:
      - Step 5: _generate_orders_default + price enrichment from prices_latest
      - Phase 17.8: Pre-Trade Impact estimate (modifies order qty)
      - Phase 17.9: Group-Exposure caps (modifies orders)

    Dropped (meta-only):
      - Phase 17.85 TWAP (only writes meta, does not replace orders)
      - Steps 5.5-5.14 (all meta-only)
    """
    if log is None:
        log = logger

    _empty = pd.DataFrame(columns=["timestamp", "symbol", "side", "qty", "price"])

    if not do_rebal or targets is None or targets.empty:
        return _empty

    policy = getattr(ctx, "_policy_cache", None)
    if policy is None:
        try:
            policy = load_policy()
        except Exception as _exc:
            logger.debug("[_tc_execution] load_policy failed, using empty policy: %s", _exc)
            policy = {}

    # Step 5: Generate orders
    try:
        orders = _generate_orders_default(ctx, targets)

        # Enrich with latest prices
        pwf = prices_with_features if prices_with_features is not None else pd.DataFrame()
        if not orders.empty and not pwf.empty:
            if prices_latest is not None and "close" in prices_latest.columns:
                latest_prices = prices_latest[["symbol", "close"]].rename(columns={"close": "price"})
            elif "close" in pwf.columns:
                latest_prices = pwf.groupby("symbol", group_keys=False)["close"].last().reset_index().rename(columns={"close": "price"})
            else:
                latest_prices = None
            if latest_prices is not None:
                orders = orders.merge(latest_prices, on="symbol", how="left", suffixes=("", "_latest"))
                if "price_latest" in orders.columns:
                    orders["price"] = orders["price_latest"].fillna(orders["price"])
                    orders = orders.drop(columns=["price_latest"])
    except Exception as e:
        log.warning("order generation failed: %s", e)
        return _empty

    # Phase 17.8: Pre-trade impact
    try:
        impact_cfg = (policy.get("execution", {}) or {}).get("pre_trade_impact", {}) or {}
        if impact_cfg.get("enabled", False) and not orders.empty:
            orders, _impact_meta = _apply_pre_trade_impact(orders, prices_filtered, impact_cfg)
    except Exception as e:
        log.debug("pre_trade_impact skipped: %s", e)

    # Phase 17.9: Group-exposure caps
    try:
        group_cfg = (policy.get("risk", {}) or {}).get("group_limits", {}) or {}
        if group_cfg.get("enabled", False) and not orders.empty:
            sec_meta = None
            try:
                from src.assembled_core.data.security_master import load_security_master
                sec_meta = load_security_master(group_cfg.get("security_master_path") or None)
            except Exception as _exc:
                logger.debug("[_tc_execution] security_master load failed, skipping group caps: %s", _exc)
            if sec_meta is not None:
                orders, _grp_meta = _apply_group_exposure_caps(orders, sec_meta, group_cfg)
    except Exception as e:
        log.debug("group_exposures skipped: %s", e)

    if not orders.empty and "qty" in orders.columns:
        orders = orders.copy()
        orders["qty"] = orders["qty"].abs()

    # Phase 17.85: RL execution quality annotation (live/paper only, never blocks)
    try:
        rl_cfg = (policy.get("execution") or {}).get("rl_executor") or {}
        if rl_cfg.get("enabled", False) and getattr(ctx, "mode", "") in ("live", "paper") and not orders.empty:
            from src.assembled_core.execution.rl_environment import ExecutionEnvConfig
            from src.assembled_core.execution.rl_execution import (
                RLExecutor,
                RuleBasedExecutor,
            )
            _rl_model_path = rl_cfg.get("model_path", "")
            _rl_n_steps = int(rl_cfg.get("n_steps", 20))
            _rl_min_qty = int(rl_cfg.get("min_qty_for_annotation", 100))
            _rl_shortfall_bps: list[float] = []
            for _rl_idx, _rl_row in orders.iterrows():
                _rl_qty = abs(int(_rl_row.get("qty", 0) or 0))
                _rl_price = float(_rl_row.get("price", 100.0) or 100.0)
                if _rl_qty >= _rl_min_qty and _rl_price > 0:
                    _rl_env_cfg = ExecutionEnvConfig(total_shares=_rl_qty, arrival_price=_rl_price, n_steps=_rl_n_steps)
                    if _rl_model_path:
                        _rl_exec: RLExecutor | RuleBasedExecutor = RLExecutor(config=_rl_env_cfg, model_path=_rl_model_path)
                        _rl_exec.load(_rl_model_path)
                    else:
                        _rl_exec = RuleBasedExecutor(config=_rl_env_cfg)
                    _rl_res = _rl_exec.execute(n_steps=_rl_n_steps)
                    orders.at[_rl_idx, "rl_avg_exec_price"] = _rl_res.get("avg_execution_price", _rl_price)
                    orders.at[_rl_idx, "rl_est_shortfall_bps"] = _rl_res.get("shortfall_bps", 0.0)
                    _rl_shortfall_bps.append(float(_rl_res.get("shortfall_bps", 0.0)))
            if _rl_shortfall_bps:
                log.debug("[RL-EXEC] annotated %d orders; avg shortfall %.1f bps", len(_rl_shortfall_bps), sum(_rl_shortfall_bps) / len(_rl_shortfall_bps))
    except Exception as e:
        log.debug("[RL-EXEC] rl_executor skipped: %s", e)

    return orders


def book_fills(
    result: TradingCycleResult,
    ctx: TradingContext,
    *,
    log: logging.Logger | None = None,
) -> TradingCycleResult:
    """Write outputs and return the final TradingCycleResult.

    Steps kept (write artifacts read by monitoring / downstream pipelines):
      - Step 7: write_outputs (safe_csv / equity_curve / state)
      - Step 7.6: write_run_kpis
      - Step 7.62: write_run_manifest
      - Step 7.63: append_run_index
      - Step 7.66: trade journal
      - Step 7.68: heartbeat
      - Phase 9: signal diagnostics (write signal_health.json)
      - Phase 11: KPI export (Prometheus metrics)

    Dropped (meta-only, 3-criteria rule):
      - Steps 7.5, 7.64, 7.65, 7.67, 7.69-7.71, 7.8, 7.9, 8.x,
        Phase 10 Monte Carlo, tail_hedge shadow, attribution shadow,
        portfolio_execution shadow, almgren_chriss shadow
    """
    if log is None:
        log = logger

    policy = getattr(ctx, "_policy_cache", None)
    if policy is None:
        try:
            policy = load_policy()
        except Exception as _exc:
            logger.debug("[_tc_execution] load_policy failed, using empty policy: %s", _exc)
            policy = {}

    # Ensure orders_filtered exists
    if result.orders_filtered is None:
        result.orders_filtered = result.orders.copy() if result.orders is not None else pd.DataFrame(columns=["timestamp", "symbol", "side", "qty", "price"])

    # A8: Apply cost annotation for backtest/paper modes
    if ctx.mode in ("backtest", "paper") and result.orders_filtered is not None and not result.orders_filtered.empty:
        try:
            from src.assembled_core.costs import get_default_cost_model
            from src.assembled_core.execution.transaction_costs import (
                CommissionModel,
                add_cost_columns_to_trades,
            )
            cost_model = get_default_cost_model()
            commission_model = CommissionModel(commission_bps=cost_model.commission_bps)
            prices = getattr(ctx, "prices", None)
            result.orders_filtered = add_cost_columns_to_trades(
                result.orders_filtered,
                commission_model=commission_model,
                prices=prices if prices is not None else pd.DataFrame(),
            )
        except Exception as _cost_err:
            log.debug("[book_fills] cost annotation skipped (mode=%s): %s", ctx.mode, _cost_err)

    # A8b: Derive total_cost_bps for Phase 11 slippage histogram
    try:
        if result.orders_filtered is not None and not result.orders_filtered.empty:
            if "total_cost_bps" not in result.orders_filtered.columns:
                if "total_cost_cash" in result.orders_filtered.columns:
                    _notional = (result.orders_filtered["qty"].abs() * result.orders_filtered["price"].abs())
                    result.orders_filtered["total_cost_bps"] = 0.0
                    _mask = _notional > 0
                    result.orders_filtered.loc[_mask, "total_cost_bps"] = (
                        result.orders_filtered.loc[_mask, "total_cost_cash"].abs() / _notional[_mask] * 10_000.0
                    )
                elif "expected_impact_bps" in result.orders_filtered.columns:
                    result.orders_filtered["total_cost_bps"] = result.orders_filtered["expected_impact_bps"]
    except Exception as _bps_err:
        log.debug("[book_fills] total_cost_bps derivation skipped: %s", _bps_err)

    # A8c: Drift monitoring (policy: drift_monitor.enabled + reference_path)
    try:
        dm_cfg = policy.get("drift_monitor") or {}
        if dm_cfg.get("enabled", False):
            ref_path = dm_cfg.get("reference_path")
            current_features = result.prices_with_features
            if ref_path and current_features is not None and not current_features.empty:
                from src.assembled_core.ops.drift_monitor import DriftMonitor
                _ref_df = (
                    pd.read_parquet(ref_path) if str(ref_path).endswith(".parquet")
                    else pd.read_csv(ref_path)
                )
                _monitor = DriftMonitor(
                    reference=_ref_df,
                    output_dir=ctx.output_dir if ctx.write_outputs else None,
                    psi_warn_threshold=float(dm_cfg.get("psi_warn", 0.25)),
                    psi_pause_threshold=float(dm_cfg.get("psi_pause", 0.35)),
                )
                _drift_report = _monitor.check_drift(
                    current=current_features,
                    report_date=ctx.as_of,
                )
                result.meta["drift_monitor"] = {
                    "max_psi": float(_drift_report.max_psi),
                    "action": _drift_report.action,
                    "drifted_features": _drift_report.drifted_features,
                    "n_drifted": len(_drift_report.drifted_features),
                }
                log.info(
                    "[DRIFT] max_psi=%.3f action=%s drifted=%d",
                    _drift_report.max_psi, _drift_report.action, len(_drift_report.drifted_features),
                )
    except Exception as _dm_err:
        log.debug("[book_fills] drift_monitor skipped: %s", _dm_err)

    # Step 7: Write outputs
    try:
        if ctx.write_outputs:
            if ctx.output_format == "safe_csv":
                from src.assembled_core.execution.safe_bridge import (
                    write_safe_orders_csv,
                )
                ctx.output_dir.mkdir(parents=True, exist_ok=True)
                out_path = write_safe_orders_csv(result.orders_filtered, output_path=ctx.output_dir / "orders_latest.csv")
                result.output_paths = {"safe_csv": out_path}
            else:
                result.output_paths = {}
    except Exception as e:
        result.status = "error"
        result.error_message = f"Error in write_outputs: {e}"
        return result

    # Step 7.6: KPI artifact
    try:
        if ctx.write_outputs:
            from src.assembled_core.ops.kpi_artifacts import write_run_kpis
            write_run_kpis(output_dir=ctx.output_dir, ctx=ctx, result=result, policy=policy, mode=ctx.execution_mode)
    except Exception as e:
        log.debug("[KPI] write_run_kpis skipped: %s", e)

    # Step 7.62: Run manifest
    try:
        if ctx.write_outputs and ctx.as_of is not None:
            from src.assembled_core.ops.run_manifest import write_run_manifest
            write_run_manifest(run_id=str(ctx.as_of.date()), date=str(ctx.as_of.date()), started_at_utc=ctx.as_of.isoformat(), status="success", metrics={"n_orders": len(result.orders_filtered), "n_signals": len(result.signals), "execution_mode": ctx.execution_mode}, manifests_dir=ctx.output_dir / "manifests")
    except Exception as e:
        log.debug("[MANIFEST] run_manifest skipped: %s", e)

    # Step 7.63: Run index
    try:
        if ctx.write_outputs and ctx.as_of is not None:
            from src.assembled_core.ops.run_index import append_run_index
            from src.assembled_core.ops.run_manifest import compute_config_hash
            append_run_index(run_id=str(ctx.as_of.date()), date=str(ctx.as_of.date()), status="success", metrics={"final_equity": float(getattr(ctx, "current_equity", ctx.equity)), "n_fills": len(result.orders_filtered)}, git_sha=result.meta.get("git_sha", ""), config_hash=compute_config_hash(policy) if policy else "", manifest_path=ctx.output_dir / "manifests" / str(ctx.as_of.date()) / "manifest.latest.json", index_path=ctx.output_dir / "manifests" / "index.csv")
    except Exception as e:
        log.debug("[RUN-INDEX] run_index skipped: %s", e)

    # Step 7.66: Trade journal
    try:
        if ctx.write_outputs and not result.orders_filtered.empty and ctx.as_of is not None:
            from src.assembled_core.ops.trade_journal import (
                append_trade_journal_entries,
            )
            _of = result.orders_filtered
            _qty_col = "quantity" if "quantity" in _of.columns else "qty"
            _price_col = "price" if "price" in _of.columns else "limit_price"
            _tj_fills = [{"symbol": str(r["symbol"]), "side": str(r["side"]), "qty": float(r[_qty_col] if pd.notna(r[_qty_col]) else 0), "price": float(r[_price_col] if pd.notna(r[_price_col]) else 0)} for r in _of[["symbol", "side", _qty_col, _price_col]].itertuples(index=False)]
            append_trade_journal_entries(_tj_fills, signal_context={"regime": result.meta.get("regime", {}).get("regime", ""), "execution_mode": ctx.execution_mode}, run_id=str(ctx.as_of.date()), journal_path=ctx.output_dir / "trade_journal.jsonl")
    except Exception as e:
        log.debug("[TRADE-JOURNAL] trade_journal skipped: %s", e)

    # Step 7.68: Heartbeat
    try:
        from src.assembled_core.ops.heartbeat import write_heartbeat
        _hb_path = ctx.output_dir / "state" / "heartbeat.json"
        write_heartbeat(path=_hb_path, status="ok", details={"cycle_date": str(ctx.as_of.date()) if ctx.as_of else "", "n_orders": len(result.orders_filtered), "execution_mode": str(ctx.execution_mode)})
        result.meta["heartbeat"] = {"status": "ok", "path": str(_hb_path)}
    except Exception as e:
        log.debug("[HEARTBEAT] heartbeat skipped: %s", e)

    # Phase 9: Signal diagnostics
    try:
        sd_cfg = (policy.get("signal_generation") or {}).get("signal_diagnostics") or {}
        if sd_cfg.get("enabled", False) and result.prices_with_features is not None and not result.prices_with_features.empty:
            from src.assembled_core.signals.signal_diagnostics import (
                compute_signal_health,
                generate_signal_health_alerts,
                save_signal_health_artifact,
            )
            fwd_col = sd_cfg.get("forward_returns_col", "return_1d")
            if fwd_col in result.prices_with_features.columns and "timestamp" in result.prices_with_features.columns:
                factor_cols = [c for c in result.prices_with_features.columns if c not in {"timestamp", "symbol", "open", "high", "low", "close", "volume", fwd_col} and result.prices_with_features[c].dtype in ("float64", "float32")][:20]
                if factor_cols:
                    health_df = compute_signal_health(result.prices_with_features, forward_returns_col=fwd_col, factor_cols=factor_cols)
                    alerts = generate_signal_health_alerts(health_df, ic_alert_threshold=float(sd_cfg.get("ic_alert_threshold", 0.0)))
                    save_signal_health_artifact(health_df, alerts, output_dir=str(ctx.output_dir / "diagnostics") if ctx.write_outputs else sd_cfg.get("output_dir", "output/diagnostics"), run_date=ctx.as_of.strftime("%Y-%m-%d") if ctx.as_of else None)
    except Exception as e:
        log.debug("[SIGNAL-DIAG] signal_diagnostics skipped: %s", e)

    # Phase 11: KPI export (Prometheus)
    try:
        kpi_cfg = policy.get("kpi_export") or {}
        if kpi_cfg.get("enabled", False):
            from src.assembled_core.ops.metrics_exporter import (
                export_metrics,
                slippage_histogram,
            )
            kpi_metrics: dict[str, float] = {
                "assembled_orders_generated_total": float(len(result.orders_filtered)),
                "assembled_targets_count": float(len(result.target_positions)),
                "assembled_signals_count": float(len(result.signals)),
            }
            tb_meta = result.meta.get("turnover_budget") or {}
            if "estimated_turnover" in tb_meta and tb_meta["estimated_turnover"] != float("inf"):
                kpi_metrics["assembled_turnover_estimated"] = float(tb_meta["estimated_turnover"])
            vt_meta = result.meta.get("vol_targeting") or {}
            if "realized_vol" in vt_meta:
                kpi_metrics["assembled_realized_vol"] = float(vt_meta["realized_vol"])
            # Slippage histogram: use cost-annotated orders if total_cost_bps column present
            kpi_histograms = None
            if result.orders_filtered is not None and "total_cost_bps" in result.orders_filtered.columns:
                _slip_obs = result.orders_filtered["total_cost_bps"].dropna().tolist()
                if _slip_obs:
                    kpi_histograms = {"assembled_slippage_bps": slippage_histogram(_slip_obs)}
            # Kill-switch state gauge (1 = engaged, 0 = inactive)
            try:
                from src.assembled_core.execution.kill_switch import (
                    is_kill_switch_engaged,
                )
                kpi_metrics["assembled_kill_switch_engaged"] = 1.0 if is_kill_switch_engaged() else 0.0
            except Exception as _exc:
                logger.debug("[_tc_execution] kill_switch KPI metric failed: %s", _exc)
            # Drift-PSI gauge from drift_monitor meta (if present)
            _drift_meta = result.meta.get("drift_monitor") or {}
            if "max_psi" in _drift_meta:
                kpi_metrics["assembled_drift_max_psi"] = float(_drift_meta["max_psi"])
            # Rejection counters (per reason)
            _rej_meta = result.meta.get("rejection_counts") or {}
            for _reason, _cnt in _rej_meta.items():
                _safe = str(_reason).replace("-", "_").replace(" ", "_").upper()
                kpi_metrics[f"assembled_rejections_{_safe}_total"] = float(_cnt)
            metrics_dir = ctx.output_dir / "metrics" if ctx.write_outputs else None
            export_metrics(kpi_metrics, histograms=kpi_histograms, labels={"strategy": ctx.strategy_name or "unknown", "mode": ctx.mode}, path=metrics_dir / "assembled.prom" if metrics_dir else None)
    except Exception as e:
        log.debug("[KPI] kpi_export skipped: %s", e)

    # Step 7.70: QuestDB write-through of fill prices (optional, never blocks cycle)
    try:
        qs_cfg = (policy.get("questdb") or {}).get("write_through") or {}
        if qs_cfg.get("enabled", False) and result.orders_filtered is not None and not result.orders_filtered.empty:
            from src.assembled_core.data.tick_store import OHLCVTick, TickStore
            _qs_store = TickStore(url=qs_cfg.get("url", ""))
            if _qs_store.ping():
                _qs_ts = pd.Timestamp.now("UTC")
                _qs_ticks: list[OHLCVTick] = []
                for _qs_row in result.orders_filtered.itertuples(index=False):
                    _qs_p = float(getattr(_qs_row, "price", 0) or 0)
                    if _qs_p > 0:
                        _qs_ticks.append(OHLCVTick(
                            symbol=str(_qs_row.symbol),
                            ts=_qs_ts,
                            open=_qs_p, high=_qs_p, low=_qs_p, close=_qs_p,
                            volume=abs(float(getattr(_qs_row, "qty", 0) or 0)),
                        ))
                if _qs_ticks:
                    written = _qs_store.write_ticks(_qs_ticks)
                    log.debug("[QUESTDB] wrote %d fill ticks", written)
    except Exception as e:
        log.debug("[QUESTDB] write_through skipped: %s", e)

    result.status = "success"
    return result
