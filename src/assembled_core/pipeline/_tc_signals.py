"""_tc_signals — generate_signals(), _apply_evidence_gate(), _compute_news_triggers()
extracted from trading_cycle_v2."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from src.assembled_core.config.policy_loader import load_policy
from src.assembled_core.pipeline.trading_cycle_shared import (
    TradingContext,
    _record_degraded_step,
)

logger = logging.getLogger(__name__)


def generate_signals(
    features: pd.DataFrame,
    ctx: TradingContext,
    *,
    log: logging.Logger | None = None,
    meta: dict | None = None,
) -> pd.DataFrame:
    """Apply signal_fn + real signal enrichment layers.

    Real steps kept:
      - Step 3: signal_fn (caller-provided) + reduce to latest bar
      - Zombie killer: force-FLAT signals for held positions past hold limit
      - Step 3.1: Intel signal layer (modifies scores for ctx-wired intel dims)
      - Step 3.2: Sector rotation signals (adds new rows)
      - Step 3.3: Earnings guard (suppresses pre-earnings signals)
      - Step 3.35: News→Signal bridge (modifies signals from news data)
      - Step 3.4 Bayesian: confidence scaling on scores
      - Step 3.5: Crash prediction + short signal generation
      - Step 3.4 MR: mean-reversion signal column (mr_signal)
      - Step 3.55: Multi-factor composite mf_score column
      - Step 3.6: Ranking hysteresis (anti-churn filter)
      - Step 3.62: MA-crossover trend_ma_score column

    Dropped (meta-only, no signal data changes):
      - Steps 3.45 (factor timing meta), 3.58 (signal correlation meta),
        3.7 (meta-model meta-only path), 3.75 (FDR meta), 3.8 (conformal meta),
        3.86-3.91 (all observability), 3.9 (adversarial validation meta)

    Returns signals DataFrame (columns: timestamp, symbol, direction, score).
    Raises ValueError if signals are missing required columns after generation.
    """
    if log is None:
        log = logger

    policy = getattr(ctx, "_policy_cache", None)
    if policy is None:
        try:
            policy = load_policy()
        except Exception:
            policy = {}

    from src.assembled_core.config.settings import get_settings
    from src.assembled_core.risk.zombie_killer import get_zombie_positions

    # --- Step 3: Call signal_fn ---
    signals = ctx.signal_fn(features)

    # Reduce to latest bar per symbol if configured
    settings = get_settings()
    if (
        settings.reduce_signals_to_latest_bar
        and ctx.mode in ("backtest", "eod", "paper", "live")
        and "timestamp" in signals.columns
        and not signals.empty
    ):
        signals["_ts"] = pd.to_datetime(signals["timestamp"], utc=True)
        signals = (
            signals.sort_values("_ts")
            .groupby("symbol", group_keys=False)
            .last()
            .reset_index()
            .drop(columns=["_ts"])
        )

    # Validate required columns
    required_cols = ["timestamp", "symbol", "direction"]
    missing = [c for c in required_cols if c not in signals.columns]
    if missing:
        raise ValueError(f"signals missing required columns: {', '.join(missing)}")

    # Coerce score column to float — signal_fn may return object dtype
    if "score" in signals.columns:
        signals = signals.copy()
        signals["score"] = pd.to_numeric(signals["score"], errors="coerce").fillna(0.0)

    log.debug("Signals generated: %d rows", len(signals))

    # --- Zombie killer: force-FLAT for positions held too long ---
    try:
        if (
            ctx.current_positions is not None
            and not ctx.current_positions.empty
            and not signals.empty
        ):
            _ref_ts = ctx.as_of if ctx.as_of is not None else pd.Timestamp.now("UTC")
            now_utc = (
                _ref_ts.to_pydatetime()
                if hasattr(_ref_ts, "to_pydatetime")
                else _ref_ts
            )
            zombies = get_zombie_positions(
                ctx.current_positions.to_dict("records"), now_utc, policy
            )
            if zombies:
                from src.assembled_core.ops.shadow_recorder import (
                    is_shadow_only,
                    record_shadow,
                )

                zk_shadow = is_shadow_only(policy, "zombie_killer")
                zombie_symbols = {pos["symbol"] for pos, _reason in zombies}
                record_shadow(
                    "zombie_killer",
                    {
                        "zombie_symbols": sorted(zombie_symbols),
                        "would_force_flat": sorted(zombie_symbols),
                    },
                    as_of=str(ctx.as_of) if ctx.as_of is not None else None,
                    meta={"zombies_found": len(zombies), "applied": not zk_shadow},
                )
                for _pos, reason in zombies:
                    log.warning(reason)
                if not zk_shadow:
                    mask = signals["symbol"].isin(zombie_symbols)
                    signals.loc[mask, "direction"] = "FLAT"
                    existing_syms = set(signals["symbol"].values)
                    missing_zombies = zombie_symbols - existing_syms
                    if missing_zombies:
                        zombie_rows = pd.DataFrame(
                            {
                                "timestamp": [ctx.as_of or pd.Timestamp.now("UTC")]
                                * len(missing_zombies),
                                "symbol": list(missing_zombies),
                                "direction": ["FLAT"] * len(missing_zombies),
                                "score": [0.0] * len(missing_zombies),
                            }
                        )
                        signals = pd.concat([signals, zombie_rows], ignore_index=True)
    except Exception as e:
        log.debug("zombie_killer check skipped: %s", e)

    # --- Step 3.1: Intel signal layer ---
    try:
        intel_sig_cfg = (policy.get("intel") or {}).get("signal_layer") or {}
        if intel_sig_cfg.get("enabled", False) and not signals.empty:
            from src.assembled_core.signals.intel_signal_adapter import (
                IntelSignalAdapter,
                compute_symbol_intel_scores,
            )

            sector_impacts = getattr(ctx, "intel_sector_impacts", None)
            supply_vuln = getattr(ctx, "intel_supply_vulnerability", None)
            sanctions_ben = getattr(ctx, "intel_sanctions_beneficiary", None)
            chokepoint_exp = getattr(ctx, "intel_chokepoint_exposure", None)
            intel_conf = getattr(ctx, "intel_confidence", None)

            if any(
                x is not None
                for x in [sector_impacts, supply_vuln, sanctions_ben, chokepoint_exp]
            ):
                raw_scores = compute_symbol_intel_scores(
                    sector_impacts=sector_impacts,
                    supply_chain_vulnerability=supply_vuln,
                    sanctions_beneficiary=sanctions_ben,
                    chokepoint_exposure=chokepoint_exp,
                    confidence=intel_conf,
                )
                if raw_scores and "score" in signals.columns:
                    intel_weight = float(intel_sig_cfg.get("weight", 0.15))
                    signals = signals.copy()
                    signals["score"] = signals["score"].astype(
                        float
                    ) + intel_weight * signals["symbol"].map(raw_scores).fillna(0.0)
                    log.info(
                        "[INTEL] signal layer applied: %d symbols scored",
                        len(raw_scores),
                    )

            active_shocks = getattr(ctx, "intel_active_shocks", None)
            if active_shocks:
                adapter = IntelSignalAdapter(
                    allow_short_signals=intel_sig_cfg.get("allow_short", False),
                    min_confidence=float(intel_sig_cfg.get("min_confidence", 0.50)),
                )
                shock_df = adapter.enrich_signals_with_shock_beneficiaries(
                    active_shocks,
                    base_confidence=float(intel_sig_cfg.get("shock_confidence", 0.60)),
                )
                if not shock_df.empty:
                    existing_syms = set(signals["symbol"].values)
                    new_shock = shock_df[~shock_df["symbol"].isin(existing_syms)].copy()
                    if not new_shock.empty:
                        new_shock["timestamp"] = ctx.as_of or pd.Timestamp.now("UTC")
                        new_shock["direction"] = "LONG"
                        signals = pd.concat(
                            [
                                signals,
                                new_shock[
                                    ["timestamp", "symbol", "direction", "score"]
                                ],
                            ],
                            ignore_index=True,
                        )
    except Exception as e:
        log.debug("[INTEL] intel_signal_layer skipped: %s", e)

    # --- Step 3.2: Sector rotation signals ---
    try:
        sr_cfg = (policy.get("signal_generation") or {}).get("sector_rotation") or {}
        if sr_cfg.get("enabled", False):
            from src.assembled_core.signals.sector_rotation import (
                generate_sector_rotation_signals,
                get_sector_weights,
            )

            scores_row = getattr(ctx, "sector_rotation_scores", None)
            if scores_row is not None:
                sr_signals = generate_sector_rotation_signals(scores_row)
                sr_weights = get_sector_weights(
                    sr_signals,
                    long_weight=float(sr_cfg.get("long_weight", 0.12)),
                    short_weight=float(sr_cfg.get("short_weight", 0.08)),
                )
                if sr_weights:
                    ts_now = ctx.as_of or pd.Timestamp.now("UTC")
                    existing_syms = (
                        set(signals["symbol"].values) if not signals.empty else set()
                    )
                    sr_rows = [
                        {
                            "timestamp": ts_now,
                            "symbol": sym,
                            "direction": "LONG" if w > 0 else "SHORT",
                            "score": round(w, 4),
                        }
                        for sym, w in sr_weights.items()
                        if sym not in existing_syms
                    ]
                    if sr_rows:
                        signals = pd.concat(
                            [signals, pd.DataFrame(sr_rows)], ignore_index=True
                        )
    except Exception as e:
        log.debug("[SIGNAL-DIAG] sector_rotation skipped: %s", e)

    # --- Step 3.3: Earnings guard ---
    try:
        eg_cfg = (policy.get("signal_generation") or {}).get("earnings_guard") or {}
        if eg_cfg.get("enabled", False) and not signals.empty:
            from src.assembled_core.signals.earnings_integration import (
                apply_earnings_integration,
            )

            earnings_calendar = getattr(ctx, "earnings_calendar", None)
            earnings_events = getattr(ctx, "earnings_events", None)
            if earnings_calendar is not None or earnings_events is not None:
                adjusted_signals, _eg_result = apply_earnings_integration(
                    signals,
                    earnings_calendar=earnings_calendar,
                    earnings_events=earnings_events,
                    as_of=ctx.as_of or pd.Timestamp.now("UTC"),
                    suppress_window=int(eg_cfg.get("suppress_window", 3)),
                    pead_window_days=int(eg_cfg.get("pead_window_days", 60)),
                    pead_weight=float(eg_cfg.get("pead_weight", 0.15)),
                )
                signals = adjusted_signals
    except Exception as e:
        log.debug("[SIGNAL-DIAG] earnings_guard skipped: %s", e)

    # --- Step 3.35: News→Signal bridge ---
    try:
        from src.assembled_core.signals.news_signal_bridge import (
            load_and_apply_news_signals,
        )

        root_for_news = (
            Path(ctx.data_root) if getattr(ctx, "data_root", None) else Path.cwd()
        )
        signals, _nsb_meta = load_and_apply_news_signals(
            signals, root=root_for_news, policy=policy, as_of=ctx.as_of
        )
    except Exception as e:
        log.debug("[SIGNAL-DIAG] news_signal_bridge skipped: %s", e)

    # --- Step 3.36: News IC-weights (BACKLOG — Phase 2c) ---
    # news_ml_bridge.get_event_type_ic_weights() was archived to
    # archive/observability_graveyard_2026q2/ml/news_ml_bridge.py.
    # IC weights require a calibrated historical event→alpha dataset.
    # Wire in once news_ground_truth labels (tests/news_gold/) are complete.
    # See: autonome_weiterarbeit/MIGRATION_TRADING_CYCLE_V2.md §Phase 2c

    # --- Step 3.4a: Bayesian signal confidence scoring ---
    try:
        bc_cfg = (policy.get("signal_generation") or {}).get(
            "bayesian_confidence"
        ) or {}
        if (
            bc_cfg.get("enabled", False)
            and not signals.empty
            and "score" in signals.columns
        ):
            from src.assembled_core.signals.signal_confidence import (
                compute_signal_confidence,
                confidence_position_scaler,
            )

            current_scores = signals.set_index("symbol")["score"].dropna()
            if len(current_scores) >= 2:
                confidences = compute_signal_confidence(
                    current_scores,
                    historical_scores=getattr(ctx, "signal_historical_scores", None),
                    ci_level=float(bc_cfg.get("ci_level", 0.90)),
                )
                signals = signals.copy()
                _max_scale = float(bc_cfg.get("max_scale", 1.5))
                _min_scale = float(bc_cfg.get("min_scale", 0.5))
                signals["score"] = signals["score"].astype(float) * signals[
                    "symbol"
                ].map(
                    {
                        sym: confidence_position_scaler(
                            conf, max_scale=_max_scale, min_scale=_min_scale
                        )
                        for sym, conf in confidences.items()
                    }
                ).fillna(1.0)
    except Exception as e:
        log.debug("[SIGNAL-DIAG] bayesian_confidence skipped: %s", e)

    # --- Step 3.5: Crash prediction + short signals ---
    try:
        shorts_policy = policy.get("shorts", {})
        if shorts_policy.get("enabled", False):
            from src.assembled_core.risk.short_risk import ShortRiskManager
            from src.assembled_core.signals.crash_prediction import (
                CrashPredictionEngine,
            )
            from src.assembled_core.signals.short_signals import ShortSignalGenerator

            macro_data: dict = {}
            if (
                ctx.prices is not None
                and not ctx.prices.empty
                and "VIX" in ctx.prices.columns
            ):
                # B-pipe-1 (latent/defensive): the production ctx.prices panel is
                # long-format (timestamp/symbol rows), so this wide "VIX"-column
                # branch does not fire in production — it is a latent path. To
                # avoid a future-VIX look-ahead if a wide panel ever reaches here
                # in backtest/replay, slice to the as_of window before taking the
                # tail (same idiom as the CB gate in trading_cycle_shared.py).
                # For live/eod (tail == as_of) this is byte-identical.
                _vix_src = ctx.prices
                _as_of = getattr(ctx, "as_of", None)
                if _as_of is not None and "timestamp" in _vix_src.columns:
                    _ts = pd.to_datetime(_vix_src["timestamp"], utc=True)
                    _as_of_utc = pd.Timestamp(_as_of)
                    if _as_of_utc.tzinfo is None:
                        _as_of_utc = _as_of_utc.tz_localize("UTC")
                    _vix_src = _vix_src.loc[_ts <= _as_of_utc]
                if not _vix_src.empty:
                    macro_data["vix"] = float(_vix_src["VIX"].iloc[-1])

            crash_engine = CrashPredictionEngine()
            crash_signal = crash_engine.predict(
                market_data=ctx.prices,
                regime=getattr(ctx, "regime_state", None),
                intel_state=getattr(ctx, "crisis_state_intel", None),
                macro_data=macro_data or None,
            )
            if crash_signal.crash_probability >= float(
                shorts_policy.get("min_crash_probability", 0.60)
            ):
                short_gen = ShortSignalGenerator(policy=shorts_policy)
                short_df = short_gen.generate_short_targets(
                    crash_signal=crash_signal,
                    universe=(
                        ctx.universe
                        if hasattr(ctx, "universe") and ctx.universe is not None
                        else pd.DataFrame()
                    ),
                    prices=ctx.prices,
                    regime=getattr(ctx, "regime_state", None),
                )
                risk_mgr = ShortRiskManager(policy=policy)
                risk_check = risk_mgr.validate_short_targets(
                    short_df, regime=getattr(ctx, "regime_state", None)
                )
                if risk_check.passed and not short_df.empty:
                    existing_syms = (
                        set(signals["symbol"].values) if not signals.empty else set()
                    )
                    short_rows = [
                        {
                            "timestamp": ctx.as_of or pd.Timestamp.now("UTC"),
                            "symbol": row.symbol,
                            "direction": getattr(row, "direction", "SHORT"),
                            "score": -abs(row.confidence),
                        }
                        for row in short_df.itertuples(index=False)
                        if row.symbol not in existing_syms
                    ]
                    if short_rows:
                        signals = pd.concat(
                            [signals, pd.DataFrame(short_rows)], ignore_index=True
                        )
    except Exception as e:
        log.debug("crash_prediction step skipped: %s", e)

    # --- Step 3.4b: MR signal column (mr_signal) ---
    try:
        mr_cfg = (policy.get("signals") or {}).get("mean_reversion") or {}
        if mr_cfg.get("enabled", False) and not features.empty:
            from src.assembled_core.signals.mean_reversion import (
                compute_mean_reversion_signals,
            )

            _mr_signals = compute_mean_reversion_signals(
                features, regime=str(getattr(ctx, "regime_state", "bull") or "bull")
            )
            if not _mr_signals.empty and not signals.empty:
                _mr_map = _mr_signals.set_index("symbol")["reversion_signal"].to_dict()
                signals = signals.copy()
                signals["mr_signal"] = signals["symbol"].map(_mr_map)
    except Exception as e:
        log.debug("[MR-SIGNAL] mean_reversion_signals skipped: %s", e)

    # --- Step 3.55: Multi-factor composite mf_score column ---
    try:
        mf_cfg = policy.get("multifactor_signal") or {}
        if mf_cfg.get("enabled", False) and not features.empty and not signals.empty:
            import pathlib as _pl

            from src.assembled_core.config.factor_bundles import load_factor_bundle
            from src.assembled_core.signals.multifactor_signal import (
                build_multifactor_signal,
            )

            _bundle_path = _pl.Path(
                mf_cfg.get(
                    "bundle_path",
                    "configs/factor_bundles/macro_world_etfs_core_bundle.yaml",
                )
            )
            if _bundle_path.exists():
                _mf_result = build_multifactor_signal(
                    features, load_factor_bundle(_bundle_path)
                )
                if not _mf_result.df.empty and "mf_score" in _mf_result.df.columns:
                    _mf_latest = _mf_result.df.groupby("symbol")["mf_score"].last()
                    signals = signals.copy()
                    signals["mf_score"] = signals["symbol"].map(_mf_latest)
    except Exception as e:
        log.debug("[MULTIFACTOR] multifactor_signal skipped: %s", e)

    # --- Step 3.6: Ranking hysteresis (anti-churn) ---
    try:
        anti_churn_cfg = policy.get("anti_churn") or {}
        if (
            anti_churn_cfg.get("ranking_hysteresis_enabled", False)
            and not signals.empty
        ):
            from src.assembled_core.paper.ranking_hysteresis import (
                apply_ranking_hysteresis,
            )

            held_symbols: set[str] = set()
            if (
                ctx.current_positions is not None
                and not ctx.current_positions.empty
                and "symbol" in ctx.current_positions.columns
            ):
                held_symbols = set(ctx.current_positions["symbol"].tolist())
            signals, _rh_meta = apply_ranking_hysteresis(
                signals,
                held_symbols,
                entry_n=int(anti_churn_cfg.get("entry_n", 5)),
                hold_n=int(anti_churn_cfg.get("hold_n", 7)),
            )
    except Exception as e:
        log.debug("[ANTI-CHURN] ranking_hysteresis skipped: %s", e)

    # --- Step 3.62: MA-crossover trend_ma_score column ---
    try:
        if not features.empty and not signals.empty:
            _req = {"timestamp", "symbol", "close"}
            if _req.issubset(set(features.columns)):
                from src.assembled_core.signals.rules_trend import (
                    generate_trend_signals,
                )

                _ts_signals = generate_trend_signals(features, ma_fast=20, ma_slow=50)
                if not _ts_signals.empty and "symbol" in _ts_signals.columns:
                    _ts_latest = (
                        _ts_signals.groupby("symbol")
                        .last()
                        .reset_index()[["symbol", "score"]]
                        .rename(columns={"score": "trend_ma_score"})
                    )
                    signals = signals.merge(_ts_latest, on="symbol", how="left")
    except Exception as e:
        log.debug("[TREND-MA] rules_trend skipped: %s", e)

    # --- Step 3.9: Evidence-grade gate (optional, filters news-backed signals) ---
    try:
        ev_cfg = policy.get("evidence_gate") or {}
        if ev_cfg.get("enabled", False):
            _news_ev_df = getattr(ctx, "news_events_df", None)
            signals, _ev_audit = _apply_evidence_gate(signals, _news_ev_df, policy)
            ctx.__dict__["_evidence_gate_audit"] = _ev_audit
    except Exception as e:
        log.debug("[EVIDENCE-GATE] skipped: %s", e)

    # --- Step 3.95: GNN graph signal blend (degrades to zero scores without PyG) ---
    try:
        gnn_cfg = (policy.get("signals") or {}).get("gnn_signal") or {}
        if (
            gnn_cfg.get("enabled", False)
            and not signals.empty
            and "symbol" in signals.columns
        ):
            from src.assembled_core.ml.gnn_signal import GNNConfig, GNNSignalModel

            _gnn_model_cfg = GNNConfig(
                n_node_features=int(gnn_cfg.get("n_node_features", 16)),
                hidden_dim=int(gnn_cfg.get("hidden_dim", 64)),
            )
            _gnn_model = GNNSignalModel(_gnn_model_cfg)
            _prices_for_gnn = getattr(ctx, "prices", None)
            if _prices_for_gnn is not None and not _prices_for_gnn.empty:
                _gnn_symbols = signals["symbol"].tolist()
                _gnn_result = _gnn_model.predict(_prices_for_gnn, symbols=_gnn_symbols)
                if _gnn_result.scores is not None and len(_gnn_result.scores) == len(
                    _gnn_symbols
                ):
                    _gnn_weight = float(gnn_cfg.get("blend_weight", 0.20))
                    _gnn_map = dict(
                        zip(_gnn_result.symbols, _gnn_result.scores.tolist())
                    )
                    if "score" in signals.columns:
                        signals = signals.copy()
                        signals["gnn_score"] = (
                            signals["symbol"].map(_gnn_map).fillna(0.0)
                        )
                        signals["score"] = (
                            signals["score"] * (1.0 - _gnn_weight)
                            + signals["gnn_score"] * _gnn_weight
                        )
                        log.debug(
                            "[GNN] blended gnn_score into score (weight=%.2f, backend=%s)",
                            _gnn_weight,
                            _gnn_result.backend,
                        )
    except Exception as e:
        log.debug("[GNN] gnn_signal skipped: %s", e)

    # --- Step 3.96: Meta-model score overlay (LightGBM v2, AUC=0.649, Prec@10%=44%) ---
    try:
        meta_cfg = (policy.get("signals") or {}).get("meta_model") or {}
        if meta_cfg.get("enabled", False) and not signals.empty:
            import joblib as _jl
            from pathlib import Path as _Path
            from src.assembled_core.signals.multifactor_signal import (
                apply_meta_model_filter,
            )

            _model_path = str(
                meta_cfg.get("model_path", "models/meta_model_lgbm_v2.joblib")
            )
            # Use bundle's calibrated threshold if policy doesn't override
            _policy_threshold = meta_cfg.get("confidence_threshold")
            if _policy_threshold is None:
                # parents[3] from src/assembled_core/pipeline/_tc_signals.py
                # = repo root. parents[4] was off-by-one (same pattern as
                # rss_fetcher / audit_trail). The except: pass below masked
                # the failure: bundle never loaded, threshold defaulted to
                # 0.58. Double-guarded today (meta_model.enabled=false AND
                # policy confidence_threshold=0.52), so no behavior change.
                _bundle_path = _Path(__file__).resolve().parents[3] / _model_path
                try:
                    _bundle = _jl.load(_bundle_path)
                    _threshold = float(_bundle.get("decision_threshold", 0.58))
                except Exception:
                    _threshold = 0.58
            else:
                _threshold = float(_policy_threshold)
            _scale = bool(meta_cfg.get("scale_by_confidence", True))
            # Enrich signals with panel features so meta-model v3 can find its columns
            _signals_for_meta = signals
            if not features.empty and "symbol" in features.columns:
                _extra = [c for c in features.columns if c not in signals.columns]
                if _extra:
                    _on = ["symbol"]
                    if (
                        "timestamp" in features.columns
                        and "timestamp" in signals.columns
                    ):
                        _on = ["timestamp", "symbol"]
                    _signals_for_meta = signals.merge(
                        features[_on + _extra], on=_on, how="left"
                    )
            signals = apply_meta_model_filter(
                _signals_for_meta,
                model_path=_model_path,
                confidence_threshold=_threshold,
                scale_by_confidence=_scale,
            )
            log.debug(
                "[META-MODEL] apply_meta_model_filter done (threshold=%.2f)", _threshold
            )
    except Exception as e:
        log.debug("[META-MODEL] meta_model skipped: %s", e)

    signals = _ensemble_signals_if_enabled(signals, features, ctx, log, meta)
    signals = _add_pairs_signals_if_enabled(signals, ctx, log)

    return signals


def _ensemble_signals_if_enabled(
    signals: pd.DataFrame,
    features: pd.DataFrame,
    ctx: "TradingContext",
    log: "logging.Logger",
    meta: dict | None = None,
) -> pd.DataFrame:
    """Optional multi-strategy ensemble layer (Plan 11/10 §1.1.2).

    Activated by policy.strategies.ensemble.enabled=true (default: off).
    When disabled this is a pure passthrough — no behavior change.
    """
    try:
        from src.assembled_core.config.policy_loader import load_policy

        _policy = load_policy()
        _ens_cfg = (_policy.get("strategies") or {}).get("ensemble") or {}
        if not _ens_cfg.get("enabled", False):
            return signals

        from src.assembled_core.portfolio.strategy_allocator import (
            AllocationConfig,
            StrategyAllocator,
        )

        _members = _ens_cfg.get("members", {})
        if not _members:
            log.debug("[ensemble] no members configured — skipping")
            return signals

        _method = _ens_cfg.get("method", "weighted_average")
        _weights = {k: float(v.get("weight", 1.0)) for k, v in _members.items()}
        _config = AllocationConfig(method=_method, weights=_weights)

        # Build strategy shims that delegate to ctx.signal_fn so each member
        # uses the same signal source (future: swap per-member signal_fn if wired)
        class _SignalFnShim:
            """Minimal duck-type for StrategyAllocator: wraps ctx.signal_fn.

            QUAL-15: a member whose signal_fn raises is fail-open (drops to an
            empty frame so the blend continues) but the drop is recorded via
            ``_record_degraded_step`` instead of vanishing silently — otherwise
            a failing member is indistinguishable from one that legitimately
            produced no signals.
            """

            def __init__(
                self,
                fn: object,
                name: str,
                meta_ref: dict | None,
                log_obj: "logging.Logger",
            ) -> None:
                self._fn = fn
                self._name = name
                self._meta = meta_ref
                self._log = log_obj

            def generate_signals(self, prices: "pd.DataFrame") -> "pd.DataFrame":
                try:
                    return self._fn(prices)  # type: ignore[operator]
                except Exception as _member_exc:
                    _record_degraded_step(
                        f"ensemble_member:{self._name}",
                        _member_exc,
                        meta=self._meta,
                        log_obj=self._log,
                    )
                    return pd.DataFrame()

        _signal_fn = ctx.signal_fn if hasattr(ctx, "signal_fn") else None
        if _signal_fn is None:
            log.debug("[ensemble] ctx.signal_fn not available — passthrough")
            return signals

        _strategy_shims = {
            name: _SignalFnShim(_signal_fn, name, meta, log) for name in _members
        }
        _allocator = StrategyAllocator(_strategy_shims, config=_config)
        _regime = (ctx.risk_state or {}).get("regime", None)
        _result = _allocator.generate_combined_signals(features, regime=_regime)
        if _result.combined_signals is not None and not _result.combined_signals.empty:
            log.info(
                "[ensemble] blended %d strategies -> %d signals",
                len(_strategy_shims),
                len(_result.combined_signals),
            )
            return _result.combined_signals
    except Exception as _e:
        _record_degraded_step("ensemble_layer", _e, meta=meta, log_obj=log)

    return signals


def _add_pairs_signals_if_enabled(
    signals: "pd.DataFrame",
    ctx: "TradingContext",
    log: "logging.Logger",
) -> "pd.DataFrame":
    """Append pairs-trading signals to the main signals DataFrame (opt-in).

    Activated by policy.pairs_trading.enabled=true (default: off).
    When disabled this is a pure passthrough — no behaviour change.

    Pairs signals are converted from (symbol_a, symbol_b, direction) to the
    standard (timestamp, symbol, direction, score) contract and appended as
    additional rows.  Each LONG_A or SHORT_A entry also adds the hedge leg
    when policy.pairs_trading.include_hedge_leg=true (default: true).

    PIT safety: ctx.prices is sliced to ctx.as_of before passing to the
    Kalman / z-score computation, preventing look-ahead bias in backtests.

    SHORT gate: SHORT-direction rows are only emitted when
    policy.scope.shorts_allowed=true AND the normalised score meets
    policy.scope.min_short_signal_confidence.
    """
    try:
        # Re-use the policy cache that ingest_data() populated to avoid per-bar
        # disk reads.  Use explicit None check — an empty dict is a valid cache.
        _cached = getattr(ctx, "_policy_cache", None)
        _policy = _cached if _cached is not None else load_policy()
        _pairs_cfg = _policy.get("pairs_trading") or {}
        if not _pairs_cfg.get("enabled", False):
            return signals

        from src.assembled_core.signals.pairs_trading import (
            generate_pairs_signals_from_panel,
        )

        # Build wide-format close prices from ctx.prices (long: timestamp, symbol, close).
        # PIT guard: slice to as_of so the Kalman filter never sees future bars.
        _prices = getattr(ctx, "prices", None)
        if _prices is None or _prices.empty:
            log.debug("[pairs] ctx.prices unavailable — skipping")
            return signals
        _required = {"timestamp", "symbol", "close"}
        if not _required.issubset(set(_prices.columns)):
            log.debug("[pairs] ctx.prices missing required columns — skipping")
            return signals

        _as_of = getattr(ctx, "as_of", None)
        if _as_of is None:
            # as_of is required for PIT-safe slicing in every operational mode.
            _mode = getattr(ctx, "mode", "unknown")
            log.warning(
                "[pairs] ctx.as_of is None (mode=%s) — skipping to avoid PIT contamination",
                _mode,
            )
            return signals
        # TZ-safe slice: normalise both sides to UTC to avoid tz-naive/tz-aware mismatch.
        _ts_utc = pd.to_datetime(_prices["timestamp"], utc=True)
        _cutoff = pd.Timestamp(_as_of)
        if _cutoff.tzinfo is None:
            _cutoff = _cutoff.tz_localize("UTC")
        _prices = _prices.loc[(_ts_utc <= _cutoff).values]

        close_wide = _prices.pivot_table(
            index="timestamp", columns="symbol", values="close", aggfunc="last"
        )
        _min_hist = int(_pairs_cfg.get("min_history", 120))
        if len(close_wide) < _min_hist:
            log.debug(
                "[pairs] insufficient history (%d rows) — skipping", len(close_wide)
            )
            return signals

        _pairs_explicit = _pairs_cfg.get("pairs") or None
        if _pairs_explicit:
            _pairs_explicit = [tuple(p) for p in _pairs_explicit]

        pairs_df = generate_pairs_signals_from_panel(
            close_wide,
            pairs=_pairs_explicit,
            coint_pval_threshold=float(_pairs_cfg.get("coint_pval_threshold", 0.05)),
            max_pairs=int(_pairs_cfg.get("max_pairs", 20)),
            entry_z=float(_pairs_cfg.get("entry_z", 2.0)),
            exit_z=float(_pairs_cfg.get("exit_z", 0.5)),
            stop_z=float(_pairs_cfg.get("stop_z", 4.0)),
            window=int(_pairs_cfg.get("window", 60)),
            delta=float(_pairs_cfg.get("delta", 1e-4)),
            min_history=_min_hist,
        )

        if pairs_df.empty:
            log.debug("[pairs] no pairs signals generated")
            return signals

        # SHORT gate: respect scope.shorts_allowed and min_short_signal_confidence.
        _scope_cfg = _policy.get("scope") or {}
        _shorts_allowed = bool(_scope_cfg.get("shorts_allowed", False))
        _min_short_conf = float(_scope_cfg.get("min_short_signal_confidence", 0.0))

        ts_now = _as_of
        include_hedge = bool(_pairs_cfg.get("include_hedge_leg", True))
        _stop_z = float(_pairs_cfg.get("stop_z", 4.0)) or 4.0
        new_rows: list[dict] = []

        # Pairs do not override existing main-signal decisions.  Symbols already
        # present in the upstream signals are skipped.  emitted_syms tracks
        # intra-pairs conflicts (same leg in multiple cointegrated pairs).
        existing_syms: set[str] = (
            set(signals["symbol"].values) if not signals.empty else set()
        )
        emitted_syms: set[str] = set()

        for _, row in pairs_df.iterrows():
            direction = str(row["direction"])
            z = float(row.get("z_score", 0.0))
            # Normalise |z| to a [0, 1] score capped at stop_z.
            score = min(1.0, abs(z) / _stop_z)
            sym_a = str(row["symbol_a"])
            sym_b = str(row["symbol_b"])

            if direction == "LONG_A":
                if sym_a not in existing_syms and sym_a not in emitted_syms:
                    new_rows.append(
                        {
                            "timestamp": ts_now,
                            "symbol": sym_a,
                            "direction": "LONG",
                            "score": score,
                        }
                    )
                    emitted_syms.add(sym_a)
                # Hedge leg: only when shorts are policy-allowed and meet confidence.
                if include_hedge and _shorts_allowed and score >= _min_short_conf:
                    if sym_b not in existing_syms and sym_b not in emitted_syms:
                        new_rows.append(
                            {
                                "timestamp": ts_now,
                                "symbol": sym_b,
                                "direction": "SHORT",
                                "score": score,
                            }
                        )
                        emitted_syms.add(sym_b)
            elif direction == "SHORT_A":
                # Both legs of the short pair require shorts_allowed — a one-sided
                # LONG without the short leg is not a pairs position.
                if _shorts_allowed and score >= _min_short_conf:
                    if sym_a not in existing_syms and sym_a not in emitted_syms:
                        new_rows.append(
                            {
                                "timestamp": ts_now,
                                "symbol": sym_a,
                                "direction": "SHORT",
                                "score": score,
                            }
                        )
                        emitted_syms.add(sym_a)
                    if (
                        include_hedge
                        and sym_b not in existing_syms
                        and sym_b not in emitted_syms
                    ):
                        new_rows.append(
                            {
                                "timestamp": ts_now,
                                "symbol": sym_b,
                                "direction": "LONG",
                                "score": score,
                            }
                        )
                        emitted_syms.add(sym_b)
            elif direction == "EXIT":
                # EXIT always emitted for each leg (risk-reducing; no conflict check).
                for _exit_sym in (sym_a, sym_b):
                    if _exit_sym not in emitted_syms:
                        new_rows.append(
                            {
                                "timestamp": ts_now,
                                "symbol": _exit_sym,
                                "direction": "FLAT",
                                "score": 0.0,
                            }
                        )
                        emitted_syms.add(_exit_sym)
            # HOLD: no new signal

        if not new_rows:
            return signals

        pairs_signals = pd.DataFrame(new_rows)
        signals = pd.concat([signals, pairs_signals], ignore_index=True)
        log.info(
            "[pairs] appended %d pairs signals from %d active pairs",
            len(new_rows),
            len(pairs_df),
        )
    except Exception as _e:
        log.warning(
            "[pairs] pairs signal layer skipped (%s): %s", type(_e).__name__, _e
        )

    return signals


def _apply_evidence_gate(
    signals: "pd.DataFrame",
    news_events: "pd.DataFrame | None",
    policy: dict,
) -> "tuple[pd.DataFrame, dict]":
    """Filter signals through evidence-grade gate. Returns (filtered_signals, audit_info).

    Applies only to signals whose symbol appears in ``news_events`` with
    insufficient evidence quality.  Signals for symbols not in news_events
    (i.e. non-news-driven signals) always pass through.

    Policy key: evidence_gate.require_grade (default "B").
    """
    import pandas as _pd
    from src.assembled_core.events.evidence_engine.action_gate import (
        check_evidence_grade_gate,
    )
    from src.assembled_core.events.evidence_engine.grader import grade_evidence
    from src.assembled_core.events.evidence_engine.misinfo_risk import (
        compute_misinfo_risk,
    )

    cfg = policy.get("evidence_gate") or {}

    audit: dict = {
        "enabled": bool(cfg.get("enabled", False)),
        "filtered_count": 0,
        "total_signals": len(signals),
        "filtered_symbols": [],
    }

    if not cfg.get("enabled", False):
        return signals, audit

    if (
        news_events is None
        or not isinstance(news_events, _pd.DataFrame)
        or news_events.empty
    ):
        audit["reason"] = "no_news_events"
        return signals, audit

    if signals.empty:
        return signals, audit

    require_grade = str(cfg.get("require_grade", "B")).upper()
    if require_grade not in {"A", "B", "C", "D"}:
        require_grade = "B"

    sym_col = "symbol" if "symbol" in news_events.columns else None
    tier_col = "source_tier" if "source_tier" in news_events.columns else None
    if sym_col is None or tier_col is None:
        audit["reason"] = "missing_required_columns"
        return signals, audit

    failing_symbols: set[str] = set()
    symbol_evidence: dict[str, dict] = {}

    for sym, grp in news_events.groupby(sym_col):
        tiers = grp[tier_col].str.upper().fillna("T3")
        n_src_a = int((tiers == "T1").sum())
        n_src_b = int((tiers == "T2").sum())
        n_src_b_ind = (
            int(grp["source_id"].nunique()) if "source_id" in grp.columns else n_src_b
        )
        evidence_summary = {
            "tierA_count": n_src_a,
            "tierB_count": n_src_b,
            "tierB_independent_count": n_src_b_ind,
            "evidence_ok": n_src_a >= 1 or n_src_b_ind >= 2,
        }
        social_only = (
            bool((tiers.isin({"T3", "SOCIAL"})).all()) if len(grp) > 0 else False
        )
        misinfo_score = compute_misinfo_risk(
            evidence_summary, social_only=social_only, event_count=len(grp)
        )
        grade = grade_evidence(evidence_summary, misinfo_risk_score=misinfo_score)
        ok, reason = check_evidence_grade_gate(grade, require_for_active=require_grade)
        symbol_evidence[str(sym)] = {"grade": grade.value, "ok": ok, "reason": reason}
        if not ok:
            failing_symbols.add(str(sym))

    audit["symbol_evidence"] = symbol_evidence

    if not failing_symbols:
        return signals, audit

    mask_keep = ~signals["symbol"].isin(failing_symbols)
    filtered = signals[mask_keep].reset_index(drop=True)
    audit["filtered_count"] = len(signals) - len(filtered)
    audit["filtered_symbols"] = list(failing_symbols)
    return filtered, audit


def _compute_news_triggers(
    news_events: "pd.DataFrame | None",
    policy: dict,
    *,
    as_of: "datetime | None" = None,
) -> "pd.DataFrame":
    """Convert processed news events into actionable triggers.

    Pipeline (order matters):
      1. Fingerprint deduplication (simhash64 + hamming distance — title-level)
      2. Greedy TF-IDF cosine clustering (groups semantically similar events)
      3. Rule-based trigger scoring (source_tier weight + burst bonus)

    Note: burst detection via compute_bursts_for_window requires pre-built cluster
    dicts and baseline counts; that integration is handled upstream by run_news_pipeline.
    Here we apply a lightweight burst bonus: events where published_utc is within
    ``burst_window_minutes`` of the most recent event get a +0.2 score boost.

    Returns a DataFrame with columns: symbol, trigger_score, cluster_id, dedup_kept.
    Returns empty DataFrame if news_events is None/empty or any step fails.
    """
    import pandas as _pd
    from src.assembled_core.events.news.fingerprint import hamming_distance, simhash64
    from src.assembled_core.events.news.tfidf import build_tfidf_vectors

    if (
        news_events is None
        or not isinstance(news_events, _pd.DataFrame)
        or news_events.empty
    ):
        return _pd.DataFrame()

    cfg = policy.get("news_triggers") or {}
    hamming_threshold = int(cfg.get("dedup_hamming_threshold", 3))
    cosine_threshold = float(cfg.get("cluster_cosine_threshold", 0.75))
    burst_window_minutes = int(cfg.get("burst_window_minutes", 60))

    text_col = next(
        (c for c in ("title", "headline", "text") if c in news_events.columns), None
    )
    sym_col = "symbol" if "symbol" in news_events.columns else None

    if text_col is None:
        return _pd.DataFrame()

    # Step 1: Fingerprint deduplication
    try:
        hashes = [simhash64(str(t)) for t in news_events[text_col].fillna("")]
        keep_mask = [True] * len(news_events)
        for i in range(len(hashes)):
            if not keep_mask[i]:
                continue
            for j in range(i + 1, len(hashes)):
                if (
                    keep_mask[j]
                    and hamming_distance(hashes[i], hashes[j]) <= hamming_threshold
                ):
                    keep_mask[j] = False
        deduped = news_events[keep_mask].copy().reset_index(drop=True)
        deduped["dedup_kept"] = True
    except Exception:
        deduped = news_events.copy()
        deduped["dedup_kept"] = True

    if deduped.empty:
        return _pd.DataFrame()

    # Step 2: TF-IDF cluster grouping (greedy, O(n²) — safe for <1000 events)
    try:
        texts = deduped[text_col].fillna("").tolist()
        vectors = build_tfidf_vectors(texts)
        cluster_ids: list[int] = [-1] * len(vectors)
        next_cid = 0
        for i, v_i in enumerate(vectors):
            if cluster_ids[i] >= 0:
                continue
            cluster_ids[i] = next_cid
            for j in range(i + 1, len(vectors)):
                if cluster_ids[j] >= 0:
                    continue
                v_j = vectors[j]
                dot = sum(v_i.get(k, 0.0) * v_j.get(k, 0.0) for k in v_i)
                norm_i = sum(x * x for x in v_i.values()) ** 0.5
                norm_j = sum(x * x for x in v_j.values()) ** 0.5
                if (
                    norm_i > 0
                    and norm_j > 0
                    and dot / (norm_i * norm_j) >= cosine_threshold
                ):
                    cluster_ids[j] = next_cid
            next_cid += 1
        deduped["cluster_id"] = cluster_ids
    except Exception:
        deduped["cluster_id"] = list(range(len(deduped)))

    # Step 3: Rule-based trigger scoring
    # Base score by source tier; burst bonus for events near the most-recent timestamp
    tier_weights = {"T1": 1.0, "T2": 0.7, "T3": 0.4}
    tier_col = "source_tier" if "source_tier" in deduped.columns else None
    if tier_col:
        deduped["trigger_score"] = (
            deduped[tier_col].str.upper().map(tier_weights).fillna(0.4)
        )
    else:
        deduped["trigger_score"] = 0.5

    time_col = "published_utc" if "published_utc" in deduped.columns else None
    if time_col:
        try:
            import pandas as _pd2

            times = _pd2.to_datetime(deduped[time_col], utc=True, errors="coerce")
            max_t = times.max()
            if _pd2.notna(max_t):
                burst_cutoff = max_t - _pd2.Timedelta(minutes=burst_window_minutes)
                deduped["trigger_score"] = (
                    deduped["trigger_score"]
                    + (times >= burst_cutoff).astype(float) * 0.2
                )
        except Exception as _exc:
            logger.debug("[trigger_score_burst] failed: %s", _exc)

    keep_cols = [
        c
        for c in (["symbol"] if sym_col else [])
        + ["trigger_score", "cluster_id", "dedup_kept"]
        if c in deduped.columns
    ]
    return deduped[keep_cols].reset_index(drop=True)
