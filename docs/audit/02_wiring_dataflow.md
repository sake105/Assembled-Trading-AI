# Audit 02 — Wiring & Data Integrity

- **Date:** 2026-05-30
- **Agent:** AGENT 2 of 5 (read-only system audit)
- **Scope:** Does data REALLY flow end-to-end (ingest → features → signal → risk → sizing → orders → ledger → reporting), or is it computed and silently dropped? Silent error masking. Boundary tz/dtype/index risks. Confirm/refute the "19 silent mfv2 factors" defect class.
- **Method:** Static read-only trace of the orchestrator (`trading_cycle_v2.py`) and every `_tc_*` step module, the mfv2 strategy, the price loader, the backtest engine (the actual ledger path), and the transaction-cost merge. Cross-checked against persisted project memory (obs #381, #395, #475, #614). No code was run; UNSURE is marked where a claim needs runtime confirmation.
- **CHANGES MADE: NONE.** This agent only created this file. `src/`, `scripts/`, `.github/`, `.claude/` were read-only. No file was edited, moved, deleted, or refactored.

---

## 1. E2E chain — hop-by-hop

Orchestrator: `src/assembled_core/pipeline/trading_cycle_v2.py :: run_trading_cycle()` (lines 532–789). It calls, in order: `ingest_data` → `build_features` → `generate_signals` → `size_positions` → `route_orders` → (audit/decision logging) → `check_risk` → `book_fills`. The **ledger/equity** hop is NOT inside this cycle — it lives in `qa/backtest_engine.py` (backtest) and `execution/unified_paper_engine.py` / `ops/paper_ledger.py` (paper). `book_fills` despite its name only writes *artifacts* (CSV/journal), it does not book positions or update cash.

| # | Hop | Status | Evidence (file:line) |
|---|-----|--------|----------------------|
| 1 | ingest_data → prices_filtered | FLOWS | `trading_cycle_v2.py:597-600`. `_filter_prices_for_as_of` (in `trading_cycle_shared.py`) returns `(prices_filtered, prices_latest)`, both stored on `result`. |
| 2 | build_features → prices_with_features | FLOWS | `trading_cycle_v2.py:602-609`. EOD/paper/live build via `_build_features_default(ctx, prices_for_features)` (`_tc_features.py:169`); features panel is the input to the signal_fn. |
| 3 | generate_signals → signals(score) | FLOWS | `trading_cycle_v2.py:611-614`; `_tc_signals.py:61` calls `ctx.signal_fn(features)`; composite written to `score`/`mf_score`, coerced to float at `_tc_signals.py:87-89`. |
| 4 | signals → size_positions(targets) | FLOWS | `trading_cycle_v2.py:616-627`. `_sp_dispatch_sizing` consumes `signals["score"]` (e.g. `_tc_sizing.py:116-123` BL path; kelly/risk_parity/vol_scaled all take `signals`). **Composite is genuinely consumed, not cosmetic** — see §4. |
| 5 | targets → route_orders(orders) | FLOWS | `trading_cycle_v2.py:629-640`; `_tc_execution.py:60-91`. Orders enriched with `price` from `prices_latest`. |
| 6 | orders → check_risk | FLOWS (can HALT) | `trading_cycle_v2.py:724-726`; `_tc_risk.py` may scale `qty` (e.g. EVT 0.80×, `_tc_risk.py:122-128`) or set `result.status="halted"`. |
| 7 | result → book_fills (artifacts) | FLOWS, but **misnamed** | `_tc_execution.py:226-698`. Writes `orders_latest.csv`, `trade_journal.jsonl`, KPIs, heartbeat. **Does NOT update positions/cash/equity.** No fill simulation here. |
| 8 | orders_filtered → positions/equity (BACKTEST) | FLOWS | `backtest_engine.py:756` reads `cycle_result.orders_filtered`; `:771` `_update_positions_vectorized`; cash updated `:762-767`; ledger `_pb_build_ledger` `:1076-1170`. |
| 8' | orders → positions/equity (PAPER) | UNSURE (separate path) | `execution/unified_paper_engine.py` + `ops/paper_ledger.py` own paper booking. Not traced line-by-line in this audit — flagged for runtime/Agent-X confirmation that `book_fills` output is the same frame the paper engine ingests. |
| 9 | observability sub-steps inside ingest/features/signals | DROPPED **by design** | `trading_cycle_v2.py:108-114` and `_tc_features.py:85-88`, `_tc_signals.py:39-43` explicitly enumerate meta-only steps removed in the v2 decomposition (the "3-criteria rule"). These are documented drops, not silent defects. |

### Genuinely DROPPED / IGNORED computed values (not by-design)

- **`pl_update` override silently widens `prices_filtered`** — `trading_cycle_v2.py:603-608`: in backtest-snapshot mode `build_features` returns a second frame that **overwrites both `result.prices_latest` AND `result.prices_filtered`** with the snapshot. Downstream risk checks (`check_risk(..., prices_filtered=result.prices_filtered)`) then see the snapshot, not the PIT-filtered history. Intentional per docstring (`_tc_features.py:68-72`) but it is a non-obvious data-substitution at a module boundary — flag for risk-reviewer.
- **`_impact_meta` / `_grp_meta` discarded** — `_tc_execution.py:99,122`: pre-trade-impact and group-cap helpers return a meta dict that is assigned to `_`-prefixed locals and never propagated to `result.meta`. The *orders* mutation flows; the *diagnostics* are dropped. Low impact (observability only).
- **RL exec annotation columns** — `_tc_execution.py:175-181` writes `rl_avg_exec_price` / `rl_est_shortfall_bps` onto `orders`, but these columns are not consumed by `book_fills`'s trade-journal projection (`:460-461` selects only `symbol/side/qty/price/algo_*/order_id`). Cosmetic columns; dropped at the journal boundary. UNSURE if any reporting reads them.

---

## 2. Silent masking table

Classification: **INTENTIONAL** = fail-open with a visible WARN/flag/sentinel, or best-effort cleanup where the primary exception already propagated. **SILENT** = swallows a real defect with no flag and changes a numeric result.

| file:line | pattern | masks-what | verdict |
|-----------|---------|-----------|---------|
| `_tc_signals.py:569-573` | `try: _bundle=jl.load(...) except Exception: _threshold = 0.58` | meta-model bundle load failure → silently uses hard-coded 0.58 threshold. Comment at `:562-567` admits the prior `except: pass` masked an off-by-one that left the bundle unloaded. | SILENT (acknowledged; double-guarded today by `meta_model.enabled=false`, so currently inert — but the masking pattern remains). |
| `_tc_signals.py:600-601` | `except Exception as e: log.debug("[META-MODEL]... skipped")` | any meta-model filter failure → signals pass through unfiltered, only at DEBUG. | SILENT-ish (DEBUG, not WARN; signal count silently differs). |
| `_tc_signals.py:651-654` | ensemble shim `try: return fn(prices) except Exception: return pd.DataFrame()` | a member strategy raising → contributes an empty frame to the ensemble average, diluting weights with no log. | SILENT. |
| `_tc_risk.py:101-102` | `except Exception: _shared_rets = None` | pivot of returns fails → EVT/copula tail-VaR gates are skipped entirely (guarded by `_shared_rets is not None`), so a risk *control* is silently disabled. | SILENT (risk-relevant; the disable is invisible). |
| `_tc_risk.py:113-116` | `except Exception: _evt_var_99 = None` | EVT solver failure → no qty reduction. | INTENTIONAL-ish (fail-open on a *reducer*; conservative direction is "don't cut" which is the riskier default — borderline). |
| `_tc_sizing.py:2062-2065` | `try: policy=load_policy() except Exception: policy={}` | policy load failure → every overlay reads empty config → all caps/limits default-off. | SILENT (sizing runs with NO risk limits; no WARN). High impact. |
| `_tc_execution.py:51-57`, `226-260` | same empty-policy fallback in route_orders/book_fills | group caps + pre-trade impact silently disabled | SILENT (DEBUG only). |
| `_tc_execution.py:89-91` | `except Exception: log.warning(...); return _empty` | order-generation failure → returns empty orders (no trades that bar). | INTENTIONAL (visible WARN; safe direction = no trade). |
| `transaction_costs.py:192-197` | `except: ... spread_bps = fallback` | ADV merge/spread calc failure → fallback flat spread. Pairs with the tz-merge risk in §3. | INTENTIONAL (visible WARN) but masks the §3 silent-NaN merge upstream. |
| `_tc_execution.py` (many) `:103,126,189,221,291,318,358,394,414,439,520,538,596,661,695` | `except Exception as e: log.debug(... skipped)` | each optional artifact/annotation step swallowed at DEBUG | INTENTIONAL (best-effort observability; none change orders/positions). |
| `event_bus.py:176`, `kill_switch.py:134`, `tick_store.py:123/169/230/253`, `state_machine.py:151`, `unified_paper_engine.py:975` | `except Exception: pass` | I/O cleanup / null-bus publish / tick-store best-effort | INTENTIONAL (cleanup; primary path already handled). |

**Counts:** Material SILENT (changes a numeric/risk result without a WARN): **~5** (`_tc_sizing.py:2064`, `_tc_risk.py:101`, `_tc_signals.py:572/600/653`). Intentional fail-open / cleanup: the large majority (~25+ across the pipeline). The single highest-severity is `_tc_sizing.py:2062-2065` — a silent empty-policy fallback that disables all sizing risk limits.

---

## 3. Boundary type / tz / index risks

| Boundary (file:line) | Risk | Severity |
|----------------------|------|----------|
| `transaction_costs.py:179-184` and `:220-229` | `trades.merge(adv_df, on=["timestamp","symbol"], how="left")`. Trade `timestamp` originates from `_generate_orders_default`; `adv_df` from `ctx.prices`. If tz-awareness or ns/us resolution differ, the join key never matches → `adv_usd` all-NaN → spread bucketed at fallback. The all-NaN outcome is then masked by the broad `except` at `:192`. **This is exactly the documented "pyarrow UTC round-trip / tz-naive vs tz-aware merge" defect class.** | HIGH — silent cost mis-estimation in backtest/paper P&L. UNSURE if a normalization upstream prevents it; needs runtime check on the two frames' `timestamp` dtype. |
| `backtest_engine.py:780` | `prices[prices["timestamp"] == timestamp]` — exact-equality timestamp match to refresh `_px_cache`. `timeline` is derived from the same `prices["timestamp"]`, so it is self-consistent *within a run*; but any upstream re-localization of `prices` (e.g. the `pl_update` snapshot substitution, §1) between timeline-build and this lookup would silently return empty → stale equity marks. | MEDIUM — fragile equality on datetime; depends on single-source invariant. |
| `_tc_signals.py:71-78` | latest-bar reduction does `pd.to_datetime(signals["timestamp"], utc=True)` then groupby-last. Forces UTC; OK if signal_fn emits tz-naive local times that should not be UTC-coerced — would shift the "latest bar". | LOW-MEDIUM. |
| `_tc_signals.py:583-590` | meta-model enrichment `signals.merge(features[_on+_extra], on=_on, how="left")` where `_on` is `["timestamp","symbol"]` only when both have `timestamp`. If `signals.timestamp` (post UTC-coercion at :71) and `features.timestamp` differ in tz, merge silently yields NaN features → meta-model sees missing columns. | MEDIUM. |
| `_tc_features.py:109-112` | precomputed-panel branch coerces `precomputed["timestamp"]` to `utc=True` only when `.dtype.tz is None`; mixed-tz panels would not be normalized. | LOW. |
| `prices_ingest.py:116-117` | nullable-dtype volume `.to_numpy(dtype=bool)` guard — this is a *fix* for a prior pd.NA bitwise bug; noted as correct handling, not a risk. | OK (defensive). |
| `polygon_source.py:116` | ms-since-epoch handling — comment present; not traced for int32 overflow on Windows. UNSURE. | UNSURE. |

No raw `astype(int)` on epoch ms was found in the core ingest path (`data/insider_ingest.py:140`, `shipping_routes_ingest.py:168` use `astype("int64")`, which is overflow-safe). The historical Windows `astype(int)` overflow does not appear to survive in the price loader.

---

## 4. mfv2 "19 silent factors" — current verdict

**Verdict: REFUTED for the original mechanism, but a different real gap exists.**

The original defect class was: *mfv2 computes factor columns into the panel, `load_eod_prices` strips all non-OHLCV columns (`prices_ingest.py:134-146`), so the factors never reach the strategy.*

Current code shows mfv2 does **NOT** depend on the panel carrying its computed factors through `load_eod_prices`. `compute_signals` (`multifactor_v2.py:1179`) re-computes virtually all 34 factors **inline at signal time**:

- Factors 1–15 from price/TA helpers on the `latest`/`df` frame (`:1250-1278`).
- Factors 16–34 from dedicated lookup functions that read their **own** data sources keyed by `(latest_symbols, as_of)` — `_compute_earnings_insider_factors` (`:1299`), `_compute_news_macro_factors` (`:1310`), `_compute_intermarket_factors` (`:1341`), `_compute_options_factors` (`:1365`), `_compute_congress_factors` (`:1382`), `_compute_geo_risk_composite` (`:1395`), `_compute_insider_cluster_factor` (`:1407`), `_compute_pead_sue_factor` (`:1419`), `_compute_buyback_drift_factor` (`:1429`).

So column-stripping in `load_eod_prices` cannot starve those factors. **However:**

1. **The column-strip is confirmed still present** — `prices_ingest.py:134-146` still drops every non-OHLCV column. The few factors that DO read panel columns via `_safe_col(latest, ...)` (e.g. `trend_ma200_position` reads `ta_ma_200_v1` `:1251`, `trend_adx_strength` reads `ta_adx_v1` `:1254`, `mom_volume_weighted` reads `ta_vol_weighted_mom_20d_v1` `:1259`, `vol_tick_imbalance` reads `tick_imbalance_20d` `:1270`, plus `vix`/`yield_curve_slope` caps at `:1511/:1538`) **silently fall back to a `default=` constant** when those columns are absent. That fallback is exactly the silent-masking shape — but it is per-factor `_safe_col(..., default=0.0)`, not a wholesale 19-factor wipe. Whether these panel columns survive into `compute_signals` depends on whether the feature build (`_build_features_default` / `ta_factors_core`) re-creates them after the strip — UNSURE without runtime column inspection.

2. **Multiple factors are structurally zero by data starvation, not wiring** — corroborated by project memory: `insider_activity_score` and `congress_activity` are intentionally zeroed (no data files exist); `earnings_surprise_z`, `sector_rotation_bias`, `news_sentiment`, options/VIX factors report ZERO in the latest mfv2 full-stack OOS (session memory 2026-05-28, obs context). These are **dead factors**, excluded from the composite by the zero-variance filter at `multifactor_v2.py:1490-1496` (`if factor_vals.abs().sum() > eps`). That filter is *correct* (prevents dilution) but means the advertised "34-factor" strategy effectively trades on ~9 live factors. This is a **data-integrity / honesty gap**, not a silent-drop bug.

3. **Bundle config is a no-op on mfv2** (memory obs #475): `configs/factor_bundles/*.yaml` only feed the auxiliary `mf_score` channel at `_tc_signals.py` step 3.55, NOT mfv2's `DEFAULT_V2_WEIGHTS`. So "tuning the bundle" does not change mfv2 sizing — a real wiring-perception gap.

**Net:** The literal "loader strips 19 factors → silently dropped" defect is **not** the current failure mode. The current truth is (a) the strip still exists and panel-sourced factors degrade silently via `_safe_col(default=...)`; (b) ~half the 34 factors are zero by missing data and removed by the dead-factor filter; (c) the composite that *does* survive IS consumed by sizing. Factors affected by genuine zero/degrade: confirmed ZERO at last OOS — earnings_surprise_z, insider_activity_score, congress_activity, sector_rotation_bias, news_sentiment_7d, options put/call + vix, (and insider_cluster/buyback data-dependent). Count of effectively-dead factors: **~8–10**, not 19, and the cause is data starvation + intentional zeroing, not the loader strip.

---

## 5. Is the composite actually USED in sizing? — CONFIRMED USED

Not cosmetic. The `signals["score"]` column produced by `generate_signals` flows directly into `_sp_dispatch_sizing` (`_tc_sizing.py:43-`), e.g. the Black-Litterman path builds `scores_dict` from `signals[["symbol","score"]]` (`:116-123`), and kelly/risk_parity/vol_scaled all receive `signals` as their first argument (`:59-96`). The composite then drives `target_positions`, which drive `route_orders` → `orders` → `_update_positions_vectorized`. The chain from composite to position qty is intact.

The *caveat*: the composite's signal quality is degraded by the dead-factor situation in §4, and any sizing risk-limit can be silently neutralized by the empty-policy fallback (`_tc_sizing.py:2064`, §2). So the wiring is real; the *content* flowing through it is thinner than the "34-factor" label implies.

---

## 6. Summary of highest-impact findings

1. **`_tc_sizing.py:2062-2065`** — silent `policy={}` on load failure disables ALL sizing risk limits with no WARN. (SILENT, risk-relevant.)
2. **`transaction_costs.py:179-229`** — trades↔ADV merge on `["timestamp","symbol"]` is a tz/dtype-mismatch trap that silently NaN-fills cost inputs; masked by `except` at `:192`. (HIGH boundary risk; matches documented defect class.)
3. **`prices_ingest.py:134-146`** — non-OHLCV column strip still present; panel-sourced mfv2 factors degrade silently via `_safe_col(default=...)`. (Original "19-factor" mechanism survives in weakened form.)
4. **mfv2 dead-factor reality** — ~8–10 of 34 factors are structurally ZERO (missing data / intentional zeroing), filtered out at `multifactor_v2.py:1490-1496`; effective strategy is ~9 live factors. Data-honesty gap.
5. **`book_fills` misnomer** (`_tc_execution.py:226`) — writes artifacts only; the real position/cash/equity update is in `backtest_engine.py:756-777` / paper engine. Anyone auditing "where fills are booked" by name will look in the wrong place.
6. **`_tc_risk.py:101-102`** — silent `_shared_rets=None` disables EVT/copula tail-VaR gates with no flag. (SILENT risk-control disable.)

**Tally:** BROKEN hops: 0. DROPPED (non-by-design) values: 3 (pl_update substitution, impact/grp meta, RL annotation columns) — all low/medium, none break the core trade decision. Material SILENT-masking sites: ~5. HIGH boundary risks: 1 (cost merge tz). mfv2: original strip-defect refuted as the live mechanism, but strip still present + ~8–10 dead factors → strategy is honest-label-deficient, not silently mis-wired.

**Nothing in the repo was changed by this audit.**
