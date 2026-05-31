# 07e — Strategy / Feature-Engineering / ML-Model Layer Audit (Round 3)

READ-ONLY quant-correctness audit. Nothing changed/deleted. Static source review only,
NOT CI-confirmed. Execution-dependent claims marked UNSURE.

Scope: strategy modules (pairs, crisis_alpha, news_alpha, vol_target, cross_asset_carry,
lppls_crash, multifactor variants, dual_momentum, low_max_lottery), feature engineering
(TA factors, cross_sectional, event/insider features, altdata), ML/model layer (meta_model,
model_registry, ml/*), determinism, factor_decay no-op.

Finding IDs prefixed `STR-`. Severity: KRITISCH / HOCH / MITTEL / NIEDRIG.
Verification tag: `[VERIFIED]` = read at source with file:line; `[PATTERN]` = grep/pattern only.

Categorisation:
- **(a)** distorts backtest/OOS results
- **(b)** live-only defect (no backtest impact)
- **(c)** dummy / dead / no-op module presented as functional

---

## Summary table

| ID | Sev | Cat | File:line | One-line |
|----|-----|-----|-----------|----------|
| STR-001 | HOCH | (a) | ta_factors_core.py:194-227 + _tc_features.py:278-301 + policy.yaml:740-748 | Forward-return labels (`returns_12m`, `momentum_12m_excl_1m`) co-mingled with trailing features; `momentum_12m_excl_1m` cross-sectionally ranked into a live `_xrank` feature column |
| STR-002 | MITTEL | (b) | cross_asset_carry.py:47-61 | Carry signals pull live `yfinance` "now" data, no as_of — not PIT-safe. Research-only (not wired into pipeline) → severity capped |
| STR-003 | MITTEL | (a/b) | multifactor_signal.py:1010 + _tc_signals.py:591 | `apply_meta_model_filter` confidence-SCALING is a no-op (gates on `mf_score`, live sizing col is `score`); FILTER still works |
| STR-004 | NIEDRIG | (c) | logic_tensor_network.py:108-122; temporal_fusion_transformer.py:117-139 | `fit`/`predict` raise `NotImplementedError` unconditionally — stubs |
| STR-005 | NIEDRIG | (c) | orchestrator.py:1431-1439 + factor_decay_reporter.py | Factor-decay monitor wired with `panel_df=None` → deliberate no-op every run (honest TODO) |
| STR-006 | NIEDRIG | (a) | (pairs cointegration selection) | Pair selection on full history = selection look-ahead — research-only, not in live path |
| STR-007 | NIEDRIG | (c) | exit_rules.py:7,26,91 | news_alpha "Reversal" exit documented (docstring item 4) but unimplemented; `new_trigger_items` param unused |
| STR-008 | NIEDRIG | (a) | risk_metrics.py:601 | Default `factor_groups["Trend"]` includes forward label `returns_12m` in a correlation diagnostic → spurious QA correlation |
| STR-009 | NIEDRIG | (a) | meta_model.py:312-326 | `train_meta_model` exclude_cols footgun: forward-return labels not excluded → auto-detect could pick them as features |

---

## STR-001 — HOCH — Forward-return labels co-mingled with features; one is ranked into a live feature  [VERIFIED]

**Category: (a) — distorts backtest/OOS if the leaking column reaches a model/composite.**

`src/assembled_core/features/ta_factors_core.py`, `_add_multi_horizon_returns` (lines 194-227):

```
194    # Forward returns (looking ahead)
196    horizons = {"returns_1m": 21, "returns_3m": 63, "returns_6m": 126, "returns_12m": 252}
204    grouped = result.groupby(group_col, group_keys=False)[price_col]
209        future_price = grouped.shift(-periods)   # shift(-N) = look FORWARD
214        result[factor_name] = log_return.astype("float64")
218    price_12m = grouped.shift(-252)
219    price_1m  = grouped.shift(-21)
223    result["momentum_12m_excl_1m"] = np.where(mask, np.log(...price_12m/price_1m...), nan)
```

`returns_1m/3m/6m/12m` and `momentum_12m_excl_1m` are **forward** returns (labels), built with
`shift(-N)`. The docstring (line 194) is honest ("looking ahead"). The danger is **naming
collision**: `momentum_12m_excl_1m` reads like a trailing momentum factor, but it is a label.
The genuinely causal trailing twins exist separately (`_add_trailing_momentum_factors`:
`trailing_returns_12m`, `trailing_momentum_12m_excl_1m`).

**Leak reach into a live feature — confirmed:**

`src/assembled_core/pipeline/_tc_features.py` (lines 278-301): default `rank_cols` includes
`momentum_12m_excl_1m` (line 290). When `enhanced_factors.enabled` is true (it is — policy.yaml:730)
and `cross_sectional_rank` defaults True (code line 273), `rank_cross_sectional` is called on it,
producing a new column `momentum_12m_excl_1m_xrank` (suffix `_xrank`,
`cross_sectional.py:30,48`). That `_xrank` column is a **forward-looking value carried as a
feature** in the enriched panel.

policy.yaml also lists `momentum_12m_excl_1m` explicitly in a `rank_cols` block (line 748) —
though note that block is nested under `behavioral_features` (line 734), not `enhanced_factors`
(line 729, which carries only `enabled: true`). `_tc_features.py` reads `rank_cols` from the
`enhanced_factors` sub-dict, so the YAML entry does not apply; the **code default** (line 290)
is what fires. Either way the forward column is ranked.

**Mitigations that limit blast radius (verified):**
- In backtest mode with precomputed features, the whole enhanced-enrichment block is skipped
  (`_using_precomputed`, `_tc_features.py:254-261`) — so the OOS walk-forward runs that used
  precomputed panels did NOT recompute/rank this column. The leak is **live / non-precomputed
  path** (or any backtest without precomputed panels).
- No downstream composite consumes `momentum_12m_excl_1m_xrank` **by name** (grep: zero hits for
  `_xrank` consumers in `src/` besides the producer). So today it sits unused in the panel —
  a latent leak, not an active one — UNLESS a model auto-detects feature columns (see STR-009).

**Verdict:** real PIT defect (forward data materialised as a feature column on the live path).
Currently contained because (1) precomputed backtests skip it, (2) no named consumer. The risk
is a future model/auto-feature-detector ingesting `*_xrank` or the raw `momentum_12m_excl_1m`.
Severity HOCH (not KRITISCH) because no active consumer was found at audit time.

---

## STR-002 — MITTEL — cross_asset_carry pulls live yfinance data, not PIT-safe  [VERIFIED]

**Category: (b) — live-only; research-only module, not wired into pipeline.**

`src/assembled_core/signals/cross_asset_carry.py`, `_get_returns` (lines 47-61):

```
47   def _get_returns(ticker, period="3mo"):
54       hist = yf.Ticker(ticker).history(period=period)   # always "now", no as_of
58       return prices.pct_change().dropna()
```

All carry primitives (`equity_carry` L64, `bond_carry` L85, `fx_carry_usd_eur` L103,
`commodity_roll_proxy` L124, aggregated by `cross_asset_carry_score` L138) call `_get_returns`.
There is **no `as_of` parameter anywhere** — every call fetches trailing-from-today data.
Using this inside a backtest would inject current-date prices into a historical bar → look-ahead.

**Severity capped at MITTEL:** grep across `src/` shows these functions are exported in
`signals/__init__.py` (L45-48) but **never called** by any orchestrator / pipeline / strategy
module — they are research/utility functions only. No OOS run consumes them. If wired into a
backtest later, this becomes KRITISCH. Fail-open `return 0.0` on missing yfinance (L73, L51) also
silently degrades to "no carry" rather than erroring.

---

## STR-003 — MITTEL — meta-model confidence SCALING is a dead no-op  [VERIFIED]

**Category: (a) if a model is active in backtest; (b) otherwise. Currently double-gated off.**

`src/assembled_core/signals/multifactor_signal.py`, `apply_meta_model_filter` (L903),
param `score_col: str = "mf_score"` (L908). Confidence path:

```
1007  signals_df = signals_df[signals_df["confidence_score"] >= confidence_threshold]   # FILTER — works
1010  if scale_by_confidence and score_col in signals_df.columns:                        # SCALE — gated
1011      signals_df[score_col] = signals_df[score_col] * signals_df["confidence_score"]
```

Live call site `src/assembled_core/pipeline/_tc_signals.py:591-596` passes **no `score_col`**
→ defaults to `"mf_score"`. The live sizing column is `score`, not `mf_score`. So when
`scale_by_confidence=True`, line 1010's guard `score_col in signals_df.columns` is **False** →
scaling silently skipped. The confidence **filter** (L1007) still trims rows, so the overlay is
not fully dead — but the documented "scale position size by model confidence" behaviour does
nothing. (Round-2 finding confirmed and re-verified.)

Note: `strategies/multifactor_v2.py:1642` has its own `_apply_meta_model_filter` (different
function, L1660) — that path is separate and uses `mf_score` natively, so it is not affected.

Currently masked further by `meta_model.enabled=false` in policy (overlay off). If enabled, the
scaling remains a no-op until `score_col="score"` is passed or the column is renamed.

---

## STR-004 — NIEDRIG — ML stubs raise NotImplementedError  [VERIFIED]

**Category: (c) — dummy modules.**

- `src/assembled_core/ml/logic_tensor_network.py`: `fit`/`predict` raise `NotImplementedError`
  (lines ~108-122). `satisfiability()` is a real numpy helper and works without the `ltn` package.
- `src/assembled_core/ml/temporal_fusion_transformer.py`: `fit`/`predict` raise
  `NotImplementedError` (lines ~117-139).

Both are honest research scaffolding (docstrings say so), but a caller treating them as functional
models would crash. No pipeline imports them. No OOS impact.

---

## STR-005 — NIEDRIG — factor_decay monitor is a deliberate no-op  [VERIFIED]

**Category: (c) — no-op wired in, produces nothing.**

`src/assembled_core/pipeline/orchestrator.py` (Step 2b, ~L1431-1439): the factor-decay reporter
is invoked with `panel_df=None` plus a TODO comment stating it is a stop-gap until
post-signal-computation wiring exists. With `panel_df=None`, `factor_decay_reporter.py` produces
zero output every run. This matches the memory note (9467b0ae: "no-op until post-signal wiring").
Honest and documented — flagged only so it is not mistaken for active monitoring.

---

## STR-006 — NIEDRIG — pairs cointegration selection look-ahead (research-only)  [PATTERN/VERIFIED-header]

**Category: (a) for the research script, not the live path.**

ETF/pairs mean-reversion selection that estimates cointegration / hedge ratios on the **full**
price history before backtesting the spread embeds selection look-ahead (the pair set is chosen
with knowledge of the whole sample). `strategies/etf_pairs_meanrev.py` header documents a
PIT-aware online (Kalman) hedge-ratio path; the selection-bias concern applies to any research
pair-discovery step run over full history. Not in the live trading cycle. Low impact unless a
pairs OOS claim relies on full-history-selected pairs.

---

## STR-007 — NIEDRIG — news_alpha reversal exit documented but unimplemented  [VERIFIED]

**Category: (c) — documented-but-missing behaviour.**

`src/assembled_core/events/news_alpha/exit_rules.py`:
- Docstring lists 4 exits; item 4 (line 7) is "Reversal: a new trigger with OPPOSITE direction".
- `check_exits` accepts `new_trigger_items` (line 26) but the function body (lines 39-91)
  **never references it** — only time (L49), take-profit (L64/L84) and stop-loss (L75/L87) are
  implemented; the function returns at L91.

So a position will not be flattened on an opposite news trigger despite the docstring. Cosmetic /
incomplete-feature; no leak. Inverse-ETF "short via long" routing and ambiguous-CB→None handling
elsewhere in the module are correct (see POSITIVES).

---

## STR-008 — NIEDRIG — forward label in risk_metrics correlation diagnostic  [VERIFIED]

**Category: (a) — QA distortion only.**

`src/assembled_core/risk/risk_metrics.py:601`: default `factor_groups` maps
`"Trend": ["returns_12m", ...]`. `returns_12m` is the **forward** label (STR-001). A
`correlation_with_returns`-style diagnostic that correlates a factor named `returns_12m` against
realised returns will show spuriously high/inflated correlation (it is partly the future return
itself). Diagnostic-only; does not feed sizing. Cosmetic but misleading in QA output.

---

## STR-009 — NIEDRIG — train_meta_model exclude_cols footgun  [VERIFIED]

**Category: (a) — would become KRITISCH if triggered.**

`src/assembled_core/signals/meta_model.py`, `train_meta_model` exclude_cols set (L312-326) does
**not** list `returns_1m/3m/6m/12m` or `momentum_12m_excl_1m`. If the training feature set is
auto-detected (all-columns-minus-excludes), these forward labels would be selected as **features**,
giving the meta-model direct access to the future return → catastrophic leakage / inflated OOS.

**Why NIEDRIG today:** the production training path observed uses
`qa/dataset_builder.py` prefix-based feature selection (L188-209), which excludes columns unless a
known feature prefix matches — so the forward labels are not picked there. The footgun is latent:
any caller that bypasses dataset_builder and relies on `train_meta_model` auto-detect would leak.
Pair with STR-001 (same forward columns).

---

## POSITIVE confirmations (verified safe / correct)

- **meta_model.py train/val split** (L343-351, L380-444): time-sorted with a monotonic assert,
  chronological validation split, and embargo — López de Prado-style. PIT-correct. [VERIFIED]
- **multifactor_v2.py dead-factor re-normalization** (L1487-1499): each factor added only if
  `factor_vals.abs().sum() > FACTOR_ZERO_VARIANCE_EPS`; `total_weight` accumulated; composite then
  divided by `total_weight` via `safe_divide(...,default=0.0)`. Structurally-ZERO factors
  (`insider_activity_score`=0.00 L266, `congress_activity`=0.00 L282) do **not** dilute the
  composite. Correct. [VERIFIED]
- **cross_sectional.py** (rank/zscore/neutralize): all use per-timestamp `groupby` → PIT-safe by
  construction (no cross-time leakage). Minor: `neutralize_cross_sectional` L148
  `result.index.isin(group.loc[mask].index)` is fragile under duplicate indices but not a leak.
  [VERIFIED]
- **news_alpha asset_router**: routing table sound, inverse ETFs use `direction="long"` (no naked
  shorts), `split_central_bank_topic` returns `None` when ambiguous (no trade). [VERIFIED]
- **lppls_crash.py**: causal numpy LPPLS, fits past window only. [VERIFIED header]
- **insider_features.py**: PIT-safe `disclosure_date` filtering. [VERIFIED]
- **vol_target_overlay.py**: rolling windows use `min_periods=window` (no partial-window leak),
  `pct_change` causal, clamps `[0,1]`. [VERIFIED]
- **dual_momentum.py**: EOM detection uses `year*12+month` keys (not bare `month.values`) →
  avoids E-031. ffill-only, no bfill → avoids E-030. Strictly causal lookback. [VERIFIED]
- **low_max_lottery.py**: explicit "No bfill (E-030 anti-pattern)"; `(year, month)` rebalance keys
  (avoids E-031); strictly-causal MAX window `[t-lookback+1, t]`; stale-signal guard in
  `compute_signals`. [VERIFIED]
- **model_registry.py verify_model_hash** (L78-90): fail-open (returns True) when registry empty /
  model absent; meta_model uses `strict=False`. Security note (other round), not a quant defect.

---

## Honest verdict

The strategy/feature/ML layer is mostly causal and well-documented. The one substantive quant
risk is **STR-001**: forward-return labels share a DataFrame with trailing features under
collision-prone names, and `momentum_12m_excl_1m` is cross-sectionally ranked into a live
`*_xrank` feature column. It is currently **contained** (precomputed backtests skip the
enrichment; no named downstream consumer) but is a latent leak that STR-009 (meta-model
auto-detect) could activate. STR-002/003 are real but gated/unwired. The rest are dead/no-op
modules honestly labelled. No evidence that completed OOS walk-forward results were distorted,
*provided* those runs used precomputed panels (which the code path enforces).

Static review only — NOT CI-confirmed. Execution-path coverage (which OOS runs used precomputed
vs live enrichment) is UNSURE without running the pipeline.
