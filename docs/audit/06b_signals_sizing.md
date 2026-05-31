# 06b — Signal Generation + Position Sizing (DEEP-AUDIT Round 2, Agent B)

- **Date:** 2026-05-30
- **Cluster:** SIGNAL GENERATION + POSITION SIZING (data → target positions)
- **Mode:** read-only analysis. Nothing changed.
- **Modules dissected:**
  - `src/assembled_core/pipeline/_tc_signals.py`
  - `src/assembled_core/pipeline/_tc_sizing.py`
  - `src/assembled_core/portfolio/position_sizing.py`
  - `src/assembled_core/portfolio/kelly_robust.py`
  - `src/assembled_core/portfolio/kelly_uncertainty.py`
  - `src/assembled_core/portfolio/turnover_penalty.py`
  - `src/assembled_core/portfolio/liquidity_aware_sizer.py`
  - `src/assembled_core/portfolio/cost_aware_wrapper.py`
  - `src/assembled_core/risk/vol_targeting.py`
  - `src/assembled_core/risk/profit_lock.py`
  - `src/assembled_core/risk/georisk_overlay.py` (`apply_exposure_multiplier_to_targets`)
  - `src/assembled_core/signals/multifactor_signal.py` (`apply_meta_model_filter`)
  - `src/assembled_core/events/news_alpha/signal_generator.py` (`signals_to_weights`)

Round-1 findings (M-1 `except: policy={}`, M-3 meta-model swallow, M-5 mfv2 data gap, target_qty=0.0 contract) are NOT repeated except where a deeper related defect was found.

---

## Findings table

| ID | Modul:Zeile | Fund | Snippet | Schwere | betrifft |
|----|-------------|------|---------|---------|----------|
| B2-01 | `_tc_sizing.py:2080–2125` | **Global exposure multiplier applied BEFORE crisis_alpha/news_alpha ADD their entries** — defensive + event-driven positions escape geo × vol_target × profit_lock × market_stress × HMM × crisis scaling. | `final_multiplier = _sp_compute_final_multiplier(...)` (2080) … `apply_exposure_multiplier_to_targets(...)` (2083) … *then later* `_sp_apply_crisis_alpha_cap(...)` (2124), `_sp_apply_news_alpha(...)` (2125) | HOCH | Live/Paper |
| B2-02 | `multifactor_signal.py:908` ∧ `_tc_signals.py:591–596` | **Meta-model score scaling silently no-ops.** `apply_meta_model_filter` defaults `score_col="mf_score"`; caller never passes `score_col`, but the live signal column is `score`. `scale_by_confidence` branch (`score_col in signals_df.columns`) is skipped → confidence never multiplies the working score. Drop-filter still works; the *scaling* half is dead. | `score_col: str = "mf_score"` (908) / call omits `score_col` (591) / `if scale_by_confidence and score_col in signals_df.columns:` (1010) | MITTEL | Live/OOS |
| B2-03 | `_tc_sizing.py:2168 → 2300, 2333–2375` | **No final gross re-check after crisis/news caps.** cost_aware shrink (2168), conformal multiplier `.clip(0.25,2.0)` (2271/2288), and buying-power (only if `buying_power>0`) can leave aggregate gross above `max_gross_exposure`. The two gross guards live *inside* `_sp_apply_crisis_alpha_cap` (1674) / `_sp_apply_news_alpha` (1908); nothing re-checks aggregate gross at the end of `size_positions`. | conformal `target_positions[_weight_col] = (… * _sym_mult).clip(-1.0, 1.0)` (2288) with no portfolio-gross renorm afterward | MITTEL | Live/Paper |
| B2-04 | `_tc_sizing.py:2099` | **CASH-row clip bug on upscale.** Per-symbol re-clamp builds `_is_cash` via `target_positions.get("symbol", target_positions.index) == "CASH"`. If `symbol` column is absent it compares the *index* to `"CASH"` (always False) → a CASH weight could be clipped to `±max_position_weight` like a risky leg. Edge case (symbol normally present) but unguarded. | `_is_cash = target_positions.get("symbol", target_positions.index) == "CASH"` | MITTEL (UNSURE) | Live/Paper |
| B2-05 | `position_sizing.py:201,207` | **Kelly win-prob proxy from raw score is uncapped at the low end and arbitrary.** `score_p = 0.5 + score*0.1` then `.clip(0,1)`. A negative `score` (shorts / penalised meta-score) yields `p<0.5`, negative edge, `kelly_raw<0`, then `.clip(lower=0.0)` → silently 0. Combined with `direction=="LONG"` filter (184) the score sign is double-counted; mapping `score→p` is a magic linear hack, not a calibrated win-rate. | `score_p = (0.5 + long_signals["score"].astype(float) * 0.1).clip(0.0, 1.0)` | MITTEL | Live/OOS |
| B2-06 | `_tc_sizing.py:2127 vs 2124–2125` | **Crisis/news entries added before the rebalance gate is evaluated, but the gate cannot veto them.** `_sp_check_rebalance` (2127) runs *after* crisis_alpha/news_alpha mutate `target_positions`; if `do_rebal=False` the caller (route_orders) returns empty orders, so a freshly-added crisis hedge on a no-rebalance bar is silently dropped unless the exit-override (2134) re-fires. Timing-dependent: crisis hedge may never reach orders on a non-rebalance day. | `_sp_apply_crisis_alpha_cap(...)` (2124) precedes `do_rebal, rebal_reason = _sp_check_rebalance(...)` (2127) | MITTEL | Live/Paper |
| B2-07 | `_tc_signals.py:572, 600, 653, 144, 209…` | **Meta-model + most enrichment layers swallow to `log.debug` (M-3 deeper).** `except Exception: _threshold = 0.58` (572) and `except Exception as e: log.debug("[META-MODEL] … skipped")` (600) hide model-load / inference failures at DEBUG. Pattern repeats across ~12 enrichment steps (intel 209, sector 248, earnings 272, bayesian 332, MR 414, multifactor 442, GNN 543, ensemble 672). A broken layer becomes an invisible no-op. | `except Exception:  _threshold = 0.58` | MITTEL | Live/OOS |
| B2-08 | `_tc_sizing.py:1390–1502` | **Crisis_alpha GPR fallback re-introduces a single-source override of "no crisis".** When `geo_score==0` AND no live triggers, GPR>200 forces `geo_score=2.0` AND `geo_sources=2` with a *synthetic* trigger item so the `min_sources=2` evidence gate passes. A single index value flips the system into CRISIS sizing; documented as intentional but is a single-point activation that the evidence gate was designed to prevent. | `_geo_sources = 2  # … lets the min_sources=2 gate pass by design` (1476) | MITTEL | OOS (backtest path); Live if no intel |
| B2-09 | `vol_targeting.py:44–45` | **Realized-vol uses `ddof=1` std on the *tail*, no shrinkage / EWMA by default.** On a 5-obs minimum window (`min_observations=5`) a benign calm patch yields tiny `realized_vol` → `target/realized` clamps to `max_scale` (1.0 / 1.5) — i.e. max leverage exactly into low-realized-vol regimes (vol-of-vol blind spot). Division-by-zero is guarded (`realized_vol<=0 → 1.0`, line 72), but near-zero vol is not floored. | `std = float(tail.std(ddof=1)); return std * (annualize_factor**0.5)` | MITTEL | Live/OOS |
| B2-10 | `_tc_sizing.py:1117, 2117–2118, 889` | **Several overlays scale `target_weight` but DO NOT re-sync `target_qty`.** correlation regime-shift `target_positions["target_weight"] *= exp_scale` updates qty (1118) — OK. But `_sp_apply_factor_risk` (889) scales only `target_weight`, never `target_qty`; downstream order-gen that reads `target_qty` would use the *un-scaled* qty. Inconsistent qty/weight pairs across overlays. | `target_positions["target_weight"] = target_positions["target_weight"] * scale` (888) with no qty branch | MITTEL | Live/Paper |
| B2-11 | `position_sizing.py:683–700` | **`apply_news_sentiment_weight_adjustment` (T4.4) is shadow_only=True by default and never wired into `size_positions`.** Computed-but-unused: the function exists, logs adjustments, but `size_positions` never calls it. Pure cosmetic unless an external caller flips it. | `shadow_only: bool = True` (645); no call site in `_tc_sizing.py` | NIEDRIG | neither |
| B2-12 | `_tc_signals.py:603, 609–675` | **`_ensemble_signals_if_enabled` re-reads policy from disk every bar** (`load_policy()` at 623) instead of `ctx._policy_cache`, and on success **replaces** the entire signals frame with the allocator output — discarding all prior enrichment (intel, news, meta-model) silently. Default-off, but a correctness landmine if enabled. | `return _result.combined_signals` (671) replaces enriched `signals` | MITTEL | Live/OOS (if enabled) |
| B2-13 | `_tc_sizing.py:1924–1929` | **news_alpha shadow log prints all target weights at INFO every cycle** including when targets would breach gross. Cosmetic noise but masks the "applied vs shadow" distinction in ops logs (a shadow line and an ACTIVE line look similar). | `log.info("[T4.2] news_alpha shadow_only=True — %d targets NOT applied: %s", …)` | NIEDRIG | neither |

---

## Per-overlay verdict: applied vs cosmetic/shadow/dropped

### Global exposure multiplier (`_sp_compute_final_multiplier` → `apply_exposure_multiplier_to_targets`)
**APPLIED** (line 2080–2088). Multiplicative chain `geo × profit_lock × vol_scale × market_stress × crisis_alpha × pm × hmm × edcl` (825–834), clamped to [0.05, 3.0] (837–850). Math is correct for the symbols *present at application time*. **DEFECT B2-01:** crisis_alpha and news_alpha entries are added *after* this, so they bypass the whole chain.

### Vol-targeting (`vol_targeting.py`)
**APPLIED** via the global multiplier. Formula `min(max_scale, max(min_scale, target/realized))` is correct (76). Div-by-zero guarded (72). **DEFECT B2-09:** near-zero realized vol → max leverage; no vol floor / shrinkage in the default `realized` method.

### Profit-lock (`profit_lock.py`)
**APPLIED** via global multiplier, but **only when `ctx.equity_curve` AND `ctx.equity_curve_index` are both non-None** (535–539). In backtest snapshot mode where these are unset, profit_lock is a silent no-op (multiplier stays 1.0). Math (lookback return ≥ trigger → clamp to [floor,1.0], cooldown via `trigger_idx`) is correct (79–82). Verdict: **applied when wired, silently skipped otherwise** — not an except-swallow, a guard-skip.

### Kelly (`compute_kelly_weights`, `kelly_robust`, `kelly_uncertainty`)
`compute_kelly_weights` is **APPLIED** only when `position_sizing.method=="kelly"` (dispatch 54–65). Half-Kelly `fraction=0.5` correct (230). `robust_kelly_fraction` (Browne-Whitt × fractional) and `compute_kelly_with_uncertainty` are **NOT called from `size_positions`** — research-tier / external-caller only (no dispatch branch references them). **DEFECT B2-05:** the `score→win_prob` proxy in `compute_kelly_weights` is an uncalibrated linear hack.

### Meta-model overlay (`apply_meta_model_filter`)
**PARTIALLY APPLIED.** Confidence *drop-filter* works (1007). Confidence *score-scaling* is **dead** because of the `score_col` default mismatch (**B2-02**). Failure path passes through unchanged (no zeroing) — confirms round-1 M-3 is "swallow→passthrough", not "swallow→zero".

### Crisis_alpha (`_sp_apply_crisis_alpha_cap`)
**APPLIED when `shadow_only=False`** (gated 1542–1547; policy.yaml has it active per memory). Adds defensive ETFs (1639–1659), caps overlaps via min-merge (1620–1634), own gross guard renorm (1674–1686). Flatten/exit commands are **logged but NOT consumed** — `§9.13 deferred; no consumer exists yet` (1576–1597): a confirmed should_flatten_all does NOT flatten anything. **DEFECTS B2-01, B2-06, B2-08.**

### News_alpha (`_sp_apply_news_alpha`)
**APPLIED when `shadow_only=False`** (1800; active per memory). Sign handling correct: longs +, shorts − in `signals_to_weights` (234); direction-conflict keeps existing (237–245); gross cap by `abs().sum()` (252). EOD min-merge cap + own gross guard (1908–1923). Exits **logged, not consumed by EOD cycle** (intraday runner owns them, 1822–1843). **DEFECTS B2-01, B2-06.**

### Liquidity (`_sp_apply_liquidity` + `liquidity_aware_sizer`)
`_sp_apply_liquidity` **APPLIED** when `liquidity_scoring.enabled` (480–520), scales `target_weight` only — does NOT re-sync `target_qty` (related to B2-10). `LiquidityAwareSizer` class is **NOT wired** into `size_positions` (qty-based, external).

### Factor-risk / trailing-stops / turnover / correlation / crash-cap / inverse-ETF / quantile / crowding / cost-aware / conformal
All **APPLIED conditionally** behind policy flags, each in its own `try/except … log.debug` swallow. `_sp_apply_factor_risk` scales weight only, not qty (**B2-10**). crash-cap, inverse-ETF, correlation-guard, crowding all correctly update both weight and qty where present. cost_aware (2168) and conformal (2170–2301) run **after** the rebalance decision and after crisis/news, with **no final gross renorm** (**B2-03**).

### Ensemble (`_ensemble_signals_if_enabled`)
**DROPPED-replacement risk** (B2-12): default-off; when on it discards all enrichment and re-reads policy from disk each bar.

### news_sentiment weight adj (T4.4)
**COSMETIC** — shadow_only default + no call site (B2-11).

---

## Sizing-math correctness summary

- **Half-Kelly halving:** correct (`position_sizing.py:230`, `kelly_robust.py:134`, `kelly_uncertainty.py:72`).
- **Vol-target formula:** correct `min(1,target/realized)` shape (`vol_targeting.py:76`); near-zero-vol leverage blind spot (B2-09).
- **Gross-exposure caps:** present in 4 places (georisk upscale 224–239, crisis 1674, news 1908, buying-power 2357) but **no single end-of-pipeline aggregate gross renorm** after the last weight-mutating overlays (B2-03).
- **Renormalization sign safety:** kelly/risk-parity/vol-scaled renorm only when `total>1.0` (preserves cash, no inflation). risk-parity clips then renorms (327–333) — OK. No sign-flip renorm bug found in the core sizers.
- **Sign errors:** none in news_alpha (long +/short −, B2 verified). Kelly score→p proxy can silently zero negative-score longs (B2-05) — suppression, not sign inversion.
- **PIT threading:** pairs (738), crisis GPR (1401–1466 with release-lag 32d), news_alpha price lookup (1777–1779), HMM panel (697) all slice `<= as_of`. No new PIT contamination found in sizing; signal-side enrichment relies on caller-provided `features`/`ctx.as_of` and is consistent.

---

## NEW count

- **HOCH:** 1 (B2-01)
- **MITTEL:** 9 (B2-02, B2-03, B2-04, B2-05, B2-06, B2-07, B2-09, B2-10, B2-12)
- **NIEDRIG:** 3 (B2-08 borderline MITTEL→listed MITTEL above; B2-11, B2-13)

> Note: B2-08 listed as MITTEL in the table (single-source crisis activation). Counting: HOCH=1, MITTEL=9, NIEDRIG=2 (B2-11, B2-13).
