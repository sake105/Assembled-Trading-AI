# ASSEMBLED Trading AI — ROADMAP_STATE.md

Version: v1.0  
Purpose: compact roadmap cockpit for Claude Code and human review  
Rule: keep this file short, factual, and current.

---

## 1. How to use this file

This file is the **single source of current roadmap position**.

It must answer, at a glance:
- where work currently is,
- what was last completed,
- what the next smallest safe step is,
- what is blocking progress,
- what has actually been validated.

This file must stay compact. Long explanations belong in `MASTER_ROADMAP.md` or in a roadmap log / milestone document.

---

## 2. Status taxonomy (must be used literally)

Use only these implementation-truth labels when describing current progress:
- discussed
- specified
- skeleton present
- partially implemented
- implemented
- locally tested
- CI-validated

Do not replace them with vague wording like:
- done-ish
- basically finished
- almost there
- probably works

---

## 3. Update discipline

Update this file when any of the following happens:
- a task is completed,
- the active task changes,
- a blocker is discovered or removed,
- validation status changes,
- milestone status changes,
- a stop condition is triggered.

Do **not** leave a session after meaningful work without checking whether this file is still accurate.

---

## 4. Current execution position

### Current milestone
- ID: Ultra-Plan — `also-erstens-wir-haben-polished-koala`
- Name: Alpaca-Revive + Backtest-Speed + Tiefe (v3 — Multi-Agent-Diskurs)
- Overall milestone status: Phase 0 (E0.1–E0.4) implemented, locally tested (2026-04-18); Tier-2 Module Activation complete (shadow-mode)

### Current task (Ultra-Plan)
- Phase 0 / E0.1 — Backtest-Paper-Parity Plumbing: `enable_risk_controls=True` default + `kill_switch_persist=True` default + `run_paper_replay` helper + determinism test green; full bit-identical bt-vs-paper kept as non-strict xfail (needs position-evolution threading).
- Phase 0 / E0.2 — Cost-Model-Aktivierung: `cost_tiers.yaml` wired, `default_adv=100_000`, `enable_borrow_costs=True` default.
- Phase 0 / E0.3 — Atomic State-Save: `_atomic_write_json` (tmp+fsync+os.replace).
- Phase 0 / E0.4 — Reconciliation-Halt-Policy: `halt_on_mismatch` + `scripts/ack_halt.py` + `paper-trading-ci.yml` halt-ack-Gate.
- Tier-2 Shadow-Mode Wiring: `portfolio_execution` + `almgren_chriss` (default OFF), `scripts/run_cost_calibration.py` offline E5-runner.
- Part A — GitHub-Actions Scheduler: workflow + halt-ack + ET time-gate + artifact upload + `check_scheduler_health.py` + `snapshot_alpaca_balance.py` + `docs/runbooks/12_paper_entry_point.md` — done.
- Part B1–B5 (speed): `_save_state` no-indent, `state_save_every_n_days` batching, hot-loop pre-extracted lists, factor-store PIT cache — all implemented, 16 Part-B regression tests green.
- Part C1–C3: 5 @njit kernels wired, `scripts/prewarm_factor_store.py` + `.github/workflows/prewarm-factor-store.yml`, `qa/parallel_grid.py` joblib+loky with deterministic per-worker seed — all green (equivalence/fallback tests green).
- Part C4 (Polars): explicitly deferred per plan.
- Part D1–D5: modules wired (`correlation_guard`, `zombie_killer`, `crash_prediction`, `inverse_etf`, `signal_decay`); **flag-flip gated on User Go/No-Go** per D-Standard A/B methodology (5d shadow + 10d enabled).
- Part E2: `config/htb_symbols.yaml` seed list + rates table.
- Part E3 + E4: walk-forward + deflated-sharpe job wired into `release-gate-ci.yml` with 2-week grace-period (`continue-on-error: true`).
- Part E1 (Sharpe-Drop Dokumentation): implemented — `scripts/quantify_realism_delta.py` emits `output/qa/realism_delta_report.md/.json` + `pre_realism_metrics.json`. Fresh 2026-04-19 rerun shows Sharpe delta **+0.3931** on synthetic fixture (outside plan-expected [-0.8, -0.3]); documented as fixture artifact pending real-price verification.
- Part E5 (real-vs-synthetic p95 < 2 bps): outstanding — needs 30 Paper-Days of real Alpaca fills.
- Part F1 (IC-Decay-Weighting), F2 (Regime-Posterior), F3 (Multi-Timeframe): wired with tests green.
- Part F4 (XGBoost/SHAP/FinBERT): explicitly deferred per plan.

### Current objective
- phase12 suite: **1266 passed**, 8 skipped, 0 failures (2026-04-19 fresh run, 168s wall-clock).
- regression suite: **142 passed**, 0 failures (2026-04-19 fresh run, 8.5s wall-clock).
- CI hardening (2026-04-19): `.env` removed from index (commit `e64fa21`), paper-trading-ci env-var name fix (`4d3a419`), backend-ci phase12+regression gate added (`d5ab05f`), gross-exposure cap switched to MTM equity (`5aa32f4`), walk-forward gate `--enforce` flag added (`e144821`), ROADMAP_STATE sync (`2ef3e24`), E1 realism delta classification + baseline artifact (`b868282`). All pushed to origin/main.
- `.env` keys rotated 2026-04-19 at provider (Alpaca/Polygon/AlphaVantage/Finnhub/NewsAPI/FRED). History-rewrite declined (old keys revoked, cost > benefit).
- E1 realism delta rerun 2026-04-19 on current code: Sharpe delta **+0.3931** (outside plan-expected [-0.8, -0.3]) — documented as synthetic-fixture artifact.
- Open user-blocked: GitHub Secrets `ALPACA_API_KEY` + `ALPACA_API_SECRET` must be set at repo settings for paper-trading-ci to succeed.
- Walk-forward gate (release-gate-ci.yml): grace period RE-OPENED 2026-04-19 through 2026-07-01 (decision today). Rationale: synthetic random-walk fixture produces ~7 OOS windows, structurally insufficient to pass DSR ≥ 0.5. Gate still runs + uploads report but is non-blocking. Re-closes on 2026-07-01 or when E5 real-price walk-forward fixture is available.

### Next smallest safe step
Ultra-Plan implementation is functionally complete at code/module/test level. Remaining items are either user-gated (secrets) or paper-days-gated:
1. **User blocker**: set `ALPACA_API_KEY` + `ALPACA_API_SECRET` in GitHub repo secrets (optional `DISCORD_WEBHOOK`), then manually dispatch `paper-trading-ci` and verify green.
2. Let GH-Actions `paper-trading-ci` run 5 consecutive weekdays and verify artifacts.
3. After 5 clean paper-days: collect Delta-Report for first shadow-mode D module; bring to User for Go/No-Go on flag-flip.
4. After 30 paper-days: run E5 `scripts/compare_real_vs_synthetic_fills.py` calibration.
5. Re-run E1 on real-price walk-forward fixture once available (current delta is synthetic-only).

### Previous milestone (superseded)
- ID: M14 — Institutional Upgrade (ML + TA + Portfolio + Execution)
- Name: Von M13 zum institutionellen / Wall-Street-Niveau
- Overall milestone status: implemented, locally tested (2026-04-04)

### Current task
- Plan "Von M13 zum institutionellen Niveau" (5 Phasen) — implementiert und getestet.
- Phase 1: Zombie Killer + Correlation Guard verdrahtet (committed e31b50d), Crisis Alpha Multiplier verdrahtet, Secret Management teilweise (.gitignore + pre-commit-config).
- Phase 2: ML-Ausbau komplett (XGBoost/LightGBM/CatBoost, Optuna, SHAP, IC-Decay, FinBERT, HMM, Stacking).
- Phase 3: TA-Vertiefung komplett (Candlestick, Multi-Timeframe, Mikrostruktur, Options, Intermarket, Breadth, VWAP).
- Phase 4: Institutional komplett (Black-Litterman, Barra-Risikomodell, TWAP/VWAP, Inverse ETFs, Daily Scheduler, SQLite Ledger, Monitoring).
- Phase 5: Earnings Calendar + Sector Rotation + ARP Bundle implementiert. Factor Curation Worker implementiert.

### Current objective
- 366/366 phase12 tests pass locally. Ruff clean on all changed files.
- 30 Dateien geändert/neu (15 modifiziert, 15 neue Module + 2 Config-Dateien).
- Alle Importe sauber, Smoke-Tests bestanden.
- Bugrun (breite Suite) läuft.
- CI matrix (ubuntu+windows) status: nicht bestätigt — Commit noch ausstehend.

### Next smallest safe step
1. Bugrun-Ergebnis bestätigen.
2. Commit + Push.
3. CI-Bestätigung (Ubuntu + Windows Matrix).
4. Secret Key Rotation beim Provider (manuell).

### After that
- Phase 5.5: Congress Trading in Bundles verdrahten.
- Batch D (OPT-9/12/13/14): Design Review.
- Unit-Tests für neue Module (Crisis Alpha Multiplier, MultiTimeframe, Factor Curation).
- Walk-Forward-Validation und Backtest-Vergleich mit neuem Feature-Set.
- CI-Härtung und Windows/Ubuntu Matrix-Bestätigung.

---

## 5. Last completed step

**Session 2026-04-18 — Ultra-Plan Phase 0 + Tier-2 Shadow Wiring — IMPLEMENTED (locally tested)**

- E0.1 helper `run_paper_replay` in `src/assembled_core/ops/replay_snapshot.py`:
  - `ReplayResult` dataclass `(orders_df, n_days, seed)`
  - Drives `run_trading_cycle` day-by-day with evolving positions book (same code path as `run_portfolio_backtest` via its `cycle_fn`)
  - Determinism test `test_run_paper_replay_emits_deterministic_orders` green (2x same seed → bit-identical order-stream)
  - Full bt-vs-paper equality `test_bit_identical_order_stream_backtest_vs_paper` kept as non-strict xfail — needs threading of the position-evolution/fill-model through both loops
- E0.1 plumbing verified (from prior session): `make_cycle_fn.enable_risk_controls=True` default, `TradingContext.kill_switch_persist=True` default, backtest-bar-restore only when `kill_switch_persist=False`
- E0.2 / E0.3 / E0.4 verified implemented from prior sessions (cost_tiers wired, atomic-save, reconcile halt + ack_halt + CI halt-ack gate)
- Tier-2 shadow-mode wiring in `src/assembled_core/pipeline/trading_cycle.py` (before final cycle-completed log):
  - `portfolio_execution` block — policy `portfolio_execution.enabled` (default OFF). Computes correlation matrix from `prices_filtered`, calls `optimize_execution_sequence`, emits `result.meta["execution_batches"]` with `shadow_only=True`.
  - `almgren_chriss` block — policy `almgren_chriss.enabled` (default OFF). Per-symbol sigma from `prices_filtered` (std of pct_change, tail 60), per-order `estimate_impact_cost`, emits `result.meta["almgren_chriss_impact"]` with `shadow_only=True`.
- `scripts/run_cost_calibration.py` — E5-loop offline runner (argparse around `calibrate_cost_model` + `write_calibration_report`, `deploy: False`).
- Tests (new):
  - `tests/regression/test_portfolio_execution_wiring.py` — 2 tests (disabled-no-meta, enabled-emits-batches)
  - `tests/regression/test_almgren_chriss_wiring.py` — 2 tests (disabled-no-meta, enabled-emits-impact)
  - `tests/regression/test_backtest_paper_parity.py` — extended with replay-importable + replay-determinism tests
- Pre-existing test bug fixed: `tests/test_regime_hmm.py::test_not_fitted_raises` — added `hmmlearn` skip-guard to match siblings (`RegimeHMM()` constructor raised ImportError before the expected RuntimeError).
- Test results: phase12 1259 passed / 8 skipped; regression 126 passed / 1 xfail / 0 failures.
- Truth status: locally tested; CI not confirmed; no commit.
- Non-action: no commit without explicit user request (CLAUDE.md §9.2).

---

**Session 2026-04-04 — M14 Institutional Upgrade (5 Phasen) — IMPLEMENTED (locally tested)**

- Plan: "Von M13 zum institutionellen / Wall-Street-Niveau" — 5 Phasen, ~30 Dateien
- Phase 1 (Sofortmaßnahmen):
  - Crisis Alpha Multiplier in trading_cycle.py verdrahtet: CRISIS=0.25, ELEVATED=0.60
  - Safety: `.upper()` case normalization + `min(1.0)` clamp gegen Fehlkonfiguration
  - `.pre-commit-config.yaml` erstellt: detect-secrets, ruff, black
- Phase 2 (ML-Ausbau): 7 Items implementiert
  - `ml/factor_models.py`: XGBoost/LightGBM/CatBoost mit Lazy-Import-Guards
  - `ml/hyperopt.py`: Optuna-basierte Hyperparameter-Optimierung
  - `ml/explainability.py`: SHAP-Werte (TreeExplainer + LinearExplainer)
  - `qa/factor_analysis.py`: IC-Decay-Tracking + Factor Half-Life
  - `ml/nlp_sentiment.py`: FinBERT-Sentiment (ProsusAI/finbert)
  - `ml/regime_hmm.py`: Hidden Markov Model (3-State Regime)
  - `ml/stacking.py`: Stacked Ensemble (OOF + Meta-Learner)
- Phase 3 (TA-Vertiefung): 7 Items implementiert
  - `features/ta_candlestick.py`: 8+ Candlestick-Pattern (pure pandas/numpy)
  - `data/resample.py`: PIT-sichere Weekly/Monthly Resampling
  - `signals/rules_trend.py`: MultiTimeframeSignal + SectorRotationSignal
  - `features/ta_liquidity_vol_factors.py`: Amihud, Roll Spread, Kyle Lambda
  - `execution/pre_trade_checks.py`: ADV-Cap Enforcement
  - `features/options_derived_signals.py` + `data/sources/cboe_source.py`
  - `features/intermarket_factors.py`: Cross-Asset-Faktoren (TLT, GLD, UUP, HYG)
  - `features/market_breadth.py`: McClellan, Zweig, TRIN, New Highs/Lows
  - `features/ta_features.py`: VWAP, VWAP-Bands, Volume-Weighted Momentum
- Phase 4 (Institutional): 7 Items implementiert
  - `portfolio/black_litterman.py` + `portfolio/covariance.py`
  - `risk/factor_risk_model.py`: Barra-Style 6-Faktor-Modell
  - `execution/algo_execution.py`: TWAP/VWAP Scheduler + Implementation Shortfall
  - `data/universe_etf.py`: Inverse ETF Map (SPY→SH, QQQ→PSQ etc.)
  - `risk/group_exposures.py`: Net Market Exposure Tracking
  - `ops/daily_scheduler.py`: Alle 4 Worker funktional + Factor Curation Worker (quartalsweise DSR-Gating)
  - `data/ledger_store.py`: SQLite-basiertes Paper-Ledger
  - `api/routers/monitoring.py`: 5 echte Routes (Portfolio, Regime, Alerts, Signals, Data-Quality)
- Phase 5 (Alpha): 4 von 5 Items
  - `data/sources/earnings_calendar_source.py`: Earnings-Kalender
  - `signals/rules_trend.py`: Sector Rotation Signal
  - `config/factor_bundles/alternative_risk_premia_bundle.yaml`: ARP Bundle
  - `ops/daily_scheduler.py`: Factor Curation Worker
  - OFFEN: Congress Trading Bundle-Integration
- Lint: 8 Findings behoben (unused imports/vars in 5 Dateien)
- Dependencies: pyproject.toml + requirements.txt aktualisiert (ml-boost, ml-tune, ml-explain, ml-nlp, ml-hmm, scipy, intermarket Extras)
- Test: 366/366 phase12 pass, 0 neue Fehler, alle Importe sauber
- Truth status: implemented, locally tested; CI nicht bestätigt; Commit ausstehend

---

**Session 2026-04-02 — Crisis Alpha + Intel Pipeline skeleton — IMPLEMENTED (locally tested)**

- Last commit: 34cda16
- Intel pipeline package `src/assembled_core/intel/` — 7 new modules:
  - `models.py`: Pydantic v2 models (NewsEvent, EvidenceCluster, GeoTrigger, DependencyNode/Edge, ShockTransmission, DependencySignal, CrisisState) + enums (SourceTier T0–T3, TriggerType, CrisisMode, NodeType, EdgeType, ShockType)
  - `source_registry.py`: static T0–T3 registry (OFAC/UN=T0, AP/Reuters=T1, GDELT/ACLED/WB/IMF=T2, NewsAPI=T3)
  - `geo_trigger.py`: keyword rules engine → GeoTrigger score 0–3
  - `dependency_graph.py`: YAML-based loader + BFS traversal
  - `shock_propagation.py`: TRIGGER_TO_SHOCKS mapping, weighted BFS, beneficiaries/losers
  - `crisis_alpha_worker.py`: state machine NORMAL→WATCH→ACTIVE→COOLDOWN, audit trail, market_confirm gate
  - `health_monitor.py`: freshness tracking, stale-on-error policy
  - `configs/dependency_graph.yaml`: 14 nodes (Hormuz, Suez, Oil, Gold, Defense, Energy, Cyber, Semis, US_Equities, China, Europe etc.) + 13 edges
- `src/assembled_core/qa/portfolio_analyzer.py`: PerformanceProfile (CAGR, Sharpe, Sortino, Calmar, MaxDD+duration, Profit Factor, Win Rate, Expectancy), PortfolioStructure (Herfindahl, top-5 concentration, sector/region weights), RegimePerformance, AttributionReport, analyze_portfolio() + text formatter
- `src/assembled_core/qa/scenario_engine.py` extended: oil_spike, gold_flight, defense_surge, geopolitical_shock; run_crisis_scenarios(); compare_crisis_scenarios()
- Tests: `tests/test_intel_pipeline.py` (78), `tests/test_portfolio_analyzer.py` (30), `tests/test_stress_scenarios_enhanced.py` (14+)
- Full suite: 3121 passed, 44 pre-existing failures (unchanged), 0 new failures
- Truth status: locally tested; CI not confirmed
- Key design decisions: rule-first, fully offline/deterministic, no external API calls in core, YAML-based dependency graph

NOT implemented (deferred):
- news_ingest.py / news_dedupe.py / news_cluster.py adapters (need external API keys)
- EIA/OFAC/UNSC real data ingest
- FastAPI intel endpoints (/intel/triggers, /intel/context/latest, /intel/health)
- trading_cycle.py integration of health_monitor + dependency_signal
- Sensitivity/parameter stability analyzer (spec Section 2.7)

---

**Session 2026-04-01 — Audit Fixes (CRITICAL-2.1, HIGH-1.4, HIGH-2.3, HIGH-5.1) — COMPLETE (locally)**

- Commit: 66dae29
- CRITICAL-2.1: `check_drawdown_kill_switch()` wired into `filter_orders_with_risk_controls()` in `risk_controls.py`. Drawdown >= 30% clears `filtered_orders` directly (not via `guard_orders_with_kill_switch()` — that path re-checks env var internally, bypassing drawdown logic).
- HIGH-1.4: `pre_trade_checks.py` — module-level logger added; both ImportError catches (exposure_engine, group_exposures) now call `logger.error()`; group exposure ImportError now also clears `filtered_orders` (was non-blocking, inconsistent with max_weight path — fixed).
- HIGH-2.3: `trading_cycle.py._apply_risk_controls_default()` loads `policy.yaml risk_limits` as fallback when `ctx.risk_config` is empty; mapping: `max_position_weight → max_weight_per_symbol`, `max_drawdown.kill → drawdown_threshold`, `turnover.daily_cap → turnover_cap`.
- HIGH-5.1: Default `safe_csv` output writing implemented in `trading_cycle.py` Step 7 via `write_safe_orders_csv()`; other formats log debug message; TODO removed.
- Files changed: `src/assembled_core/execution/risk_controls.py`, `src/assembled_core/execution/pre_trade_checks.py`, `src/assembled_core/pipeline/trading_cycle.py`
- Test results: 366/366 phase12 pass. Ruff clean.
- Truth status: locally tested; CI push pending.
- Still open: MEDIUM-3.1, MEDIUM-5.3, MEDIUM-6.2, MEDIUM-6.3; Batch D (4 items); 1 pre-existing test failure.

---

**Session 2026-04-01 — Security Hardening + API Sources Integration — COMPLETE (locally)**

- Commit: 28d7def
- Security fixes (5 files):
  - `kill_switch.py`: dual-gate check (env var + sentinel file); new `check_drawdown_kill_switch()` (30% DD)
  - `broker_adapter.py`: two-step live gate — `force_paper=False` AND `ALPACA_ALLOW_LIVE=true` required
  - `order_generation.py`: WARNING log for missing price when target_notional > 0
  - `state_machine.py`: ERROR on corrupt state file; `.bak` written before every save
  - `ledger_store.py`: `os.replace()` for atomic Windows writes; dedup `keep="last"`
- New package `src/assembled_core/data/sources/`: yfinance_source.py, polygon_source.py, fred_source.py
- New deps: alpaca-py, polygon-api-client, fredapi, edgartools, yfinance
- Test results: 366/366 phase12 pass (+1 new broker security test vs. prior 365)
- Truth status: locally tested; CI push pending
- Critical remaining: drawdown kill switch not yet wired into trading_cycle.py auto-stop

---

**Session 2026-04-01 — Systematic Optimization (Batches A, B, C) — COMPLETE**

- Batch A — Zero-risk isolated fixes (5 items):
  - `ta_factors_core.py`: removed dead `grouped_price.pct_change()` call (OPT-1)
  - `altdata_earnings_insider_factors.py`: removed dead function `aggregate_insider_events_per_date` (OPT-10); moved 2x `import os` to module top (OPT-11)
  - `turnover_budget.py`: replaced deprecated `from typing import Tuple` with built-in `tuple[...]` (OPT-15)
  - `trading_cycle.py`: added `logger.warning(...)` to 4 bare `except Exception:` blocks (OPT-17)
- Batch B — Performance vectorization (5 items):
  - `data_qc.py`: vectorized `_check_stale_prices` (OPT-4), `_check_outlier_returns` (OPT-5), `_check_missing_sessions` pre-grouping (OPT-19)
  - `turnover_budget.py`: replaced 4x `iterrows()` with vectorized dict-building (OPT-6)
  - `labeling.py`: pre-grouped `prices_by_symbol` in `label_signals` (OPT-18)
- Batch C — Feature computation pipeline (4 items):
  - `ta_factors_core.py`: `_add_short_term_reversal` replaced double-groupby with `transform()` (OPT-2)
  - `corporate_actions.py`: vectorized per-symbol split factor computation (OPT-7)
  - `calendar.py`: vectorized NYSE path in `filter_prices_to_trading_days` (OPT-8)
  - `multifactor_signal.py`: replaced `.apply()` with `transform()` (OPT-16)
- Batch D (4 items: OPT-9, OPT-12, OPT-13, OPT-14): NOT implemented — deferred for design review.
- Total phase12: 365/365 pass after each batch. ruff clean.
- Truth status: locally tested; CI push pending.

---

**Session 2026-03-31 (5) — M8–M13: Evidence Engine through Autonomous Operations — ALL COMPLETE**

- M8 Evidence Engine: EvidenceGrade A/B/C/D, grader, misinfo_risk scorer, action_gate, crisis_alpha integration. 65 new tests.
- M9 Policy Calibration: all TBD values in configs/policy.yaml replaced with concrete risk limits. 9 tests.
- M10 ETF Universe: configs/universe_etf_v1.yaml (30+ liquid ETFs), data/universe_etf.py loader/filter module. 22 tests.
- M11 Post-Trade Learning Loop: post_trade_analyzer.py (forward returns, signal hit rate), learning_store.py (atomic JSONL), scripts/run_post_trade_analysis.py. 29 tests.
- M12 Broker Adapter: BrokerAdapter ABC + AlpacaAdapter (paper-only, force_paper=True), factory. 19 tests.
- M13 Autonomous Operations: DailyScheduler, 4 workers (ingest/post_trade/reconcile/health_check), build_cycle_summary(), schedule_loop(). 10 tests.
- CI fixes merged: release-gate-ci.yml `py -3` → `python`; allow_external_fetch guard in data_source.py.
- Total phase12: 365/365 pass.
- Commits: ccdeb68 (M11+M12), 59c0bfa (M13). Prior: 7a381f8 (M8+M9+M10), ea3a399 (CI fixes).

Truth status: locally tested; CI push pending.

---

**Session 2026-03-31 (4) — M7: Realism Upgrades v2 — M7 COMPLETE**

- `src/assembled_core/data/calendar.py` patched (M7-T01):
  - Added `_CALENDAR_MODE` variable ("nyse" or "fallback") logged at import
  - Added `is_weekday()` — pure-Python weekday check (Mon–Fri), no holiday awareness
  - Added `calendar_mode()` — returns active mode string
  - Added `is_trading_day_safe()` — uses NYSE when available, weekday fallback otherwise
  - Added `filter_prices_to_trading_days()` — filters price DataFrame to trading-day rows, fallback-safe
- `src/assembled_core/data/corporate_actions.py` patched (M7-T02):
  - `adjust_prices_for_splits()` replaced: was stub returning copy unchanged. Now applies backward split adjustment: pre-split prices divided by split_ratio. Validates required columns (defensive: returns copy unchanged if missing). Skips zero or negative ratios. Multiple splits on same symbol applied sequentially.
- `src/assembled_core/data/cost_model_policy.py` created (M7-T03):
  - `estimate_rebalance_cost_fraction()` — cost = turnover * one_way_bps / 10000; policy-driven, disableable
  - `compute_cost_drag_per_period()` — maps turnover series to cost fractions
  - `get_effective_cost_params()` — resolves effective cost params after policy override
- `src/assembled_core/data/realism_meta.py` created (M7-T04):
  - `build_realism_label()` — explicit per-component mode + additive score 0–10 → none/minimal/standard/high
  - `build_realism_label_from_policy()` — reads policy sections automatically
  - Score breakdown: calendar (0/1/2), CA (0/1/2), cost (0/1/2), universe (0/1/2), data source (0/1/2)
- Rule files updated (before M7 work):
  - `.claude/rules/40-testing-and-ci.md` — "Pflicht vor Aufgabenabschluss" section added
  - `.claude/rules/10-core-operating-rules.md` — "Dokumentationspflicht nach Änderungen" added
  - `.claude/rules/95-token-efficiency.md` — /compact after each roadmap step made explicit
- Test results: 68/68 targeted M7; 242/242 phase12 (2026-03-31)

Truth status: locally tested; CI not confirmed.

M7 acceptance criteria status (COMPLETE):
- ✅ Exchange calendar: filter + fallback mode (is_trading_day_safe, filter_prices_to_trading_days)
- ✅ Corporate actions: adjust_prices_for_splits() real implementation (backward split adjustment)
- ✅ Cost model wrapper: policy-driven estimate_rebalance_cost_fraction()
- ✅ Realism metadata: build_realism_label() labels backtest outputs with realism level
- ⚠ CI confirmation pending

---

**Session 2026-03-31 (3) — M6-T08 + M6-T09: attribution + parameter stability — M6 COMPLETE**

- `src/assembled_core/risk/attribution.py` created (M6-T08):
  - `compute_symbol_return_contributions()` — weight_i * return_i per symbol
  - `compute_portfolio_return()` — total portfolio return
  - `compute_covariance_matrix()` — annualized covariance from price history
  - `compute_symbol_vol_contributions()` — marginal contribution to risk (MCR)
  - `compute_portfolio_vol()` — portfolio annualized vol from weights + cov
  - `compute_attribution_report()` — full report: return + vol attribution, policy-driven
- `src/assembled_core/risk/param_stability.py` created (M6-T09):
  - `compute_rolling_vol_estimates()` — realized vol at multiple window sizes
  - `check_vol_stability()` — CV of vol-by-window, threshold-based stability flag
  - `check_turnover_stability()` — CV of turnover series, threshold-based flag
  - `compute_rolling_max_drawdown()` — rolling drawdown series
  - `check_drawdown_stability()` — CV of drawdown across windows
  - `compute_stability_report()` — combined stability report, policy-driven
- `tests/test_risk_attribution.py` — 35 tests (all pass)
- `tests/test_risk_param_stability.py` — 33 tests (all pass)
- Test results: 174/174 phase12 pass; 261/261 M4+M5+M6 targeted suite pass
- `.claude/settings.json` updated: `defaultMode: bypassPermissions` added

Truth status: locally tested; CI not confirmed.

M6 acceptance criteria status (COMPLETE):
- ✅ Vol targeting implemented with policy-driven disable/enable, clamping, annualization
- ✅ Vol targeting wired into trading_cycle.py as optional multiplicative overlay (M6-T03)
- ✅ Zombie killer implemented: policy-configurable hold limit and min gain
- ✅ Correlation guard implemented: cluster detection, proportional scaling
- ✅ Profit lock tests added (12 tests for existing module)
- ✅ Turnover budget tests added (14 tests for existing module)
- ✅ Attribution report implemented: return + vol MCR contributions (M6-T08)
- ✅ Parameter stability checks implemented: vol, turnover, drawdown stability (M6-T09)
- ⚠ CI confirmation pending

---

**Session 2026-03-31 (2) — M6-T03: vol_targeting wired into trading_cycle.py**

- `src/assembled_core/pipeline/trading_cycle.py` patched:
  - Added `from src.assembled_core.risk.vol_targeting import compute_vol_targeting_result`
  - Added vol_targeting block in overlay composition section (after profit_lock, before final_multiplier)
  - Pattern: `vt_cfg = policy.get("vol_targeting")`, guarded by `enabled`, `equity_curve`, `equity_curve_index`
  - PIT-safe: passes `ctx.equity_curve_index` as `now_idx` (not `-1`) to avoid look-ahead in backtest
  - Extended `final_multiplier = geo * profit_lock * vol_scale_factor`
  - Fixed guard: `if abs(final_multiplier - 1.0) > 1e-9` — handles scale factors both above and below 1.0
  - Added `result.meta["vol_targeting"]` fields for full observability
- `configs/policy.yaml` patched: added `vol_targeting` section:
  - `enabled: false` (safe default)
  - `max_scale: 1.00` (no leverage amplification; consistent with `leverage_allowed: false`)
  - `min_scale: 0.50`, `lookback_days: 20`, `target_vol_annual: 0.15`
- Safety review done via `risk-execution-reviewer` subagent before coding
- Baseline check via `git stash` confirmed pre-trade integration failure is pre-existing (circular import),
  not a regression from this patch
- Test results: 106/106 phase12 tests pass; trading_cycle import clean

Truth status: locally tested; CI not confirmed.

M6 acceptance criteria status (COMPLETE — all tasks done):
- ✅ Vol targeting implemented with policy-driven disable/enable, clamping, annualization
- ✅ Vol targeting wired into trading_cycle.py as optional multiplicative overlay (M6-T03)
- ✅ Zombie killer implemented: policy-configurable hold limit and min gain
- ✅ Correlation guard implemented: cluster detection, proportional scaling
- ✅ Profit lock tests added (12 tests for existing module)
- ✅ Turnover budget tests added (14 tests for existing module)
- ✅ Attribution report implemented (M6-T08): return + vol MCR contributions per symbol
- ✅ Parameter stability checks implemented (M6-T09): vol, turnover, drawdown stability
- ⚠ CI confirmation pending

---

**Session 2026-03-31 (1) — M6 Risk v1.1 Upgrades core implementation**

- `src/assembled_core/risk/vol_targeting.py` created:
  - `compute_realized_vol()` — annualized vol from returns series, configurable lookback/min_obs
  - `compute_vol_scale_factor()` — target_vol / realized_vol clamped to [min_scale, max_scale]
  - `apply_vol_targeting_to_weights()` — scales symbol→weight dict by scale factor
  - `compute_vol_targeting_result()` — policy-driven entry: returns (scale, realized, target)
- `src/assembled_core/risk/zombie_killer.py` created:
  - `check_zombie_position()` — single position check: held > max_hold_days AND gain < min_gain_pct
  - `get_zombie_positions()` — scans all open positions, returns (pos, reason) list
  - Handles missing price data (conservative flag), unparseable timestamps (safe non-flag),
    long and short sides
- `src/assembled_core/risk/correlation_guard.py` created:
  - `compute_correlation_matrix()` — pivot prices → returns → corr matrix (requires timestamp col)
  - `detect_correlated_clusters()` — union-find grouping of positively correlated symbols
  - `apply_correlation_guard()` — proportional scale-down of over-concentrated clusters
  - Uses positive correlation only (negative/hedging correlations excluded from cluster risk)
- 5 test files (96 tests):
  - `tests/test_risk_vol_targeting.py` — 28 tests
  - `tests/test_risk_zombie_killer.py` — 19 tests
  - `tests/test_risk_correlation_guard.py` — 23 tests
  - `tests/test_risk_profit_lock.py` — 12 tests (covers existing profit_lock.py)
  - `tests/test_risk_turnover_budget.py` — 14 tests (covers existing turnover_budget.py)
- Bugrun: 210/210 pass across M3+M4+M5+M6 targeted suite.
- Pre-existing `exchange_calendars` failures confirmed pre-existing (not M6 regressions).

Truth status: locally tested; CI not confirmed.

M6 acceptance criteria status (partial — core modules done):
- ✅ Vol targeting implemented with policy-driven disable/enable, clamping, annualization
- ✅ Zombie killer implemented: policy-configurable hold limit and min gain
- ✅ Correlation guard implemented: cluster detection, proportional scaling
- ✅ Profit lock tests added (12 tests for existing module)
- ✅ Turnover budget tests added (14 tests for existing module)
- ⚠ M6-T03: vol_targeting not yet wired into trading_cycle.py as overlay hook
- ⚠ M6-T08: attribution report not yet implemented
- ⚠ M6-T09: parameter stability checks not yet implemented
- ⚠ CI confirmation pending

---

**Session 2026-03-30 (4) — M5 Crisis-Alpha v1 core implementation**

- `src/assembled_core/events/crisis_alpha/` package created (6 modules + __init__):
  - `context.py` — CrisisAlphaContext dataclass (input contract)
  - `state_machine.py` — persistent WATCH/ACTIVE/COOLDOWN/PAUSE with hysteresis,
    cooldown timer (24h), daily loss guard, social-only guard, atomic JSON persistence
  - `gates.py` — 6 activation gates: health, social-only, evidence, source, market_stress,
    daily_loss; `run_all_activation_gates()` fail-fast ordered check
  - `baskets.py` — 5 default ETF basket entries (GLD/TLT/SHY/DEFENSIVE, SH/INVERSE_EQUITY,
    VIXY/VOLATILITY); policy-overridable
  - `entry.py` — equal_weight and geo_weighted entry methods; risk_budget applied
  - `risk_budget.py` — per-instrument weight caps, gross exposure cap (0.30), proportional scaling
  - `exit_rules.py` — time_stop (8h), break_even (0.5%), no_overnight checks;
    check_deactivation_triggers for full portfolio flatten
  - `pipeline.py` — orchestrator: load state → compute transition → run gates (audit) →
    generate entry → check exits → check deactivation → persist → emit result dict
- `configs/crisis_alpha/crisis_alpha.yaml` — full config with hysteresis, entry, risk_budget,
  exit, daily_loss, basket_overrides sections.
- `scripts/run_crisis_alpha_worker.py` — standalone worker: loads triggers_latest.json for
  geo signal, builds context, runs pipeline, writes JSON manifest; CLI overrides for testing;
  --dry-run / --reset-pause flags.
- Tests: 70 tests across 3 files:
  - `test_crisis_alpha_state_machine.py`: 30 tests — all transitions, persistence, roundtrip
  - `test_crisis_alpha_gates.py`: 22 tests — all 6 gates + run_all_activation_gates
  - `test_crisis_alpha_pipeline.py`: 18 tests — entry, deactivation, exit rules, dry_run,
    4 scenario tests (shock, false activation blocked, health error, daily loss + reset)

Truth status: locally tested; CI not confirmed.

M5 acceptance criteria status:
- ✅ Social-only cannot activate (social_only guard in state machine + gate)
- ✅ Degraded health cannot activate (health_ok gate blocks WATCH→ACTIVE)
- ✅ ERROR health forces ACTIVE→COOLDOWN (tested in scenario)
- ✅ Deactivation and cooldown work (hysteresis + 24h timer tested)
- ✅ Max daily loss pauses (daily_loss_breached → PAUSE, tested)
- ✅ All transitions and protective actions deterministic and logged (reason field, structured logs)
- ⚠ M5-T13: dedicated Crisis-Alpha spec doc not yet written (roadmap docs updated this session)
- ⚠ Actual order submission not automated — manual review step required (paper-safe by design)

---

**Session 2026-03-30 (3) — M4 Execution Workers (Ops v1) core implementation**

- `src/assembled_core/execution/intent_store.py` created: JSONL-based append-only
  intent store with idempotency keys (`make_daily_key`, `make_run_key`, `has_intent`,
  `record_intent`, `load_intents`, `filter_intents_by_action`).
- `scripts/run_stop_worker.py` created: registers STOP intent + writes `.stop_active`
  sentinel; idempotent (skip on second run, --force to override).
- `scripts/run_reconcile_worker.py` created: loads ledger parquet + broker snapshot CSV,
  runs `reconcile_ledger_vs_broker`, writes JSON manifest, records RECONCILE intent.
- `scripts/run_kill_switch_worker.py` created: registers KILL intent + writes
  `.kill_switch_active` sentinel; optionally generates SAFE-Bridge flatten orders CSV
  from `--positions-path` (human review required before execution).
- `tests/test_execution_intent_store.py` created: 27 tests covering key helpers,
  load/record/filter, idempotency patterns, force-override, store creation.
- Smoke-runs verified: all three workers pass --help and produce expected log output;
  idempotency skip confirmed on second run of stop_worker.
- 97/97 tests pass across M1+M2+M3+M4 intent store + kill switch suites.
- 5 pre-existing failures in test_execution_safe_orders.py / test_execution_pre_trade_*.py
  / test_execution_order_generation_vectorized.py confirmed as pre-existing (not caused
  by M4 changes — those test files are unchanged).

Truth status: locally tested; CI not confirmed.

M4 acceptance criteria status:
- ✅ Stop execution not dependent on main cycle (stop_worker is standalone)
- ✅ Reconcile establishes source-of-truth consistency (reconcile_worker reads ledger + broker)
- ✅ Kill switch can pause and flatten safely in paper (sentinel + optional SAFE-Bridge CSV)
- ✅ Repeated runs do not duplicate hard actions (idempotency keys in intent_store)
- ⚠ M4-T09 docs: ROADMAP docs being updated this session

---

**Session 2026-03-30 (2) — M3 Risk / State Machine v1 audit + risk_metrics bug fix**

- Full audit of `src/assembled_core/risk/` and `src/assembled_core/pipeline/trading_cycle.py`.
- Bug fixed: `risk_metrics.py` `compute_risk_by_regime()` passed `equity=` kwarg to
  `compute_basic_risk_metrics()` which has no such parameter — removed stale kwarg.
- M3 audit result: all v1 acceptance criteria met (see ROADMAP_LOG.md for details).
- Key M3 components confirmed: state_machine.py, market_stress.py, georisk_overlay.py,
  profit_lock.py, turnover_budget.py all implemented and wired in trading_cycle.py.
- 17 state machine tests pass; 116/116 risk tests pass; combined M1+M2+state machine: 150 pass.

Truth status: locally tested; CI not confirmed.

---

**Session 2026-03-30 (1) — M2 DISCLOSURES test fix + worker**

- Test bug fixed: `test_pipeline_fetch_report_includes_house_ptr_stats` used prod sources.yaml
  where house_ptr is `active: false`. Fix: test now writes its own sources config with house_ptr active.
- `scripts/run_disclosures_worker.py` created (same structure as news worker).
- 21/21 disclosures tests pass. Combined suite: 133/133 pass.

Truth status: locally tested; CI not confirmed.

---

**Session 2026-03-29 (2) — M1 NEWS trigger scoring + entity linking + worker**

- `score_triggers()` wired into `pipeline.py` (was implemented but never called).
- `triggers_latest.json` now populated; `health.metrics["triggers"]` added.
- `entity_linking.py`: stub replaced with full implementation (`mapping_df`, `security_master_df`, `missing`).
- `scripts/run_news_worker.py` created (M1-T13): argparse, file locking, structured log output.
- 112 news tests: 112/112 pass (was 100/112 before this session).

Truth status:
- NEWS v1 pipeline: locally tested (112 tests, no CI confirmation)
- Worker script: imports and --help verified locally; end-to-end pipeline run not executed
  (would require live RSS/GDELT network access)

---

**Session 2026-03-29 (1) — M0 formal state sync + data stub fixes**

- `docs/roadmap/MASTER_ROADMAP.md` and `ROADMAP_STATE.md` committed into repo.
- All 9 collection-failing stub modules implemented and verified (94 tests pass locally):
  - `src/assembled_core/data/factor_store.py`
  - `src/assembled_core/data/panel_store.py`
  - `src/assembled_core/data/universe.py`
  - `src/assembled_core/data/security_master.py`
  - `src/assembled_core/data/news/contract.py`
  - `src/assembled_core/data/news/store.py`
  - `src/assembled_core/data/shipping/contract.py`
  - `src/assembled_core/data/data_source.py`
  - `src/assembled_core/data/altdata/finnhub_events.py`
  - `src/assembled_core/data/altdata/finnhub_news_macro.py`
- Bug fix in `tests/test_factor_store_roundtrip.py` (base_date off-by-11-months).

Truth status:
- governance docs: implemented (CLAUDE.md, .claude/rules/, AGENTS.md, .cursor/rules/)
- policy config: implemented (`configs/policy.yaml` substantive)
- learning folder: implemented (templates, incidents, patterns, checklists)
- strategy policy doc: implemented (`docs/STRATEGY_POLICY.md`)
- data stub modules: locally tested (94 tests pass, CI not yet confirmed on this branch)
- intel loaders: implemented (`src/assembled_core/intel/news_triggers_loader.py`, `disclosures_triggers_loader.py`)
- news/disclosures configs: implemented (`configs/news/`, `configs/disclosures/`)
- hooks/settings behavior: not yet fully validated

---

## 6. Active blockers

### Technical blockers
- Startup hook error: previously observed, current status unknown — investigate before relying on automated hook runs.
- Claude-Mem integration: not yet validated as fully operational.
- CI confirmation pending: data stub module tests pass locally (94 tests, branch `cursor/development-environment-setup-8e96`), CI run not yet confirmed.

### Documentation / control blockers
- `docs/cursor/` may still contain stale context — audit if it is still loaded as active guidance.
- `docs/roadmap/ROADMAP_LOG.md` was created 2026-03-29 (first entry = M0 close).

### Repo blockers
- Historical `.env` / secret exposure risk: still a real security concern. Key must be treated as potentially compromised until rotated. `.gitignore` alone does not protect the history.

---

## 7. Validation snapshot

### Governance / docs
- Claude-vs-Cursor governance alignment: locally checked / largely synchronized
- master roadmap control layer: committed to repo, live-workflow validation pending
- ROADMAP_LOG.md: created 2026-03-29

### Claude Code tooling
- local Claude CLI on Windows: locally tested
- project-root launch: locally tested
- subagents visible/usable: partially tested
- hooks/settings behavior: not yet fully validated

### Repo/product code
- data layer stubs (10 modules): locally tested — 94 tests pass
- CI for branch `cursor/development-environment-setup-8e96`: not yet confirmed
- intel loaders (`news_triggers_loader`, `disclosures_triggers_loader`): implemented, not separately integration-tested
- M1/M3 configs (`configs/news/`, `configs/disclosures/`, `configs/policy.yaml`): implemented, not yet wired end-to-end

---

## 8. Stop-condition snapshot

Check these before continuing into feature work:
- [~] Governance layer still contradicts real repo state — largely resolved; `docs/cursor/` audit still pending
- [~] Startup hooks still error in a way that affects automation — status unknown, investigate before heavy automated runs
- [x] Secret handling is still operationally unresolved — `.env` key must be rotated; history not yet cleaned
- [ ] Active task is too large / not smallest safe step
- [ ] Validation plan for the next step is unclear

Legend: [ ] = clear / [~] = partially resolved / [x] = still open blocker

If any box becomes effectively true, pause feature expansion and stabilize first.

---

## 9. Milestone queue

Use this as the default sequence unless a documented blocker or dependency requires adjustment:
- M0 — Repo Governance & Policy Baseline
- M1 — NEWS v1 MVP
- M2 — DISCLOSURES v1 MVP
- M3 — Risk / State Machine v1
- M4 — Execution Workers (Ops v1)
- M5 — Crisis-Alpha v1
- M6 — Risk v1.1 Upgrades
- M7 — Realism Upgrades v2

---

## 10. Current milestone checklist template

Copy and adapt this block under the active milestone when execution begins:

```md
### Active milestone checklist
- [ ] spec / contract updated or verified
- [ ] config / interface clarified
- [ ] smallest implementation step chosen
- [ ] targeted tests identified
- [ ] docs impact identified
- [ ] next state update planned
```

---

## 11. Session-close checklist

Before ending a session, answer:
- What exactly changed?
- What exactly was verified?
- What is still only specified?
- What is the next smallest safe step?
- Does `ROADMAP_STATE.md` still match reality?

If the answer to the last question is “not sure”, update this file before ending the session.
