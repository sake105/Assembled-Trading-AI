# Audit 07f — Error Handling, Contract Drift & Determinism (Cross-Cutting Census)

**Auditor role:** READ-ONLY code-quality auditor (Round 3, cross-cutting agent)
**Scope:** `src/assembled_core/**` (whole tree)
**Method:** Grep/Glob census + targeted Read verification. **Static-only, NOT CI-confirmed.**
**Date:** 2026-05-30
**Finding ID prefix:** `QUAL-`

> Breadth over depth. Every finding is backed by `file:line`. Counts are produced
> by a throwaway scan script (categorizes the body of every `except Exception` /
> bare `except`). Classification: KRITISCH / HOCH / MITTEL / NIEDRIG.

---

## CENSUS 1 — Silent-except / fail-open / silent-degradation (E-025 family)

### Quantification (whole `src/assembled_core/`)

Scanned every `except Exception[...]:` / `except BaseException:` / bare `except:` and
categorized the handler body. **997 handlers** total across ~250 files.

| Category | Meaning | Count | In sensitive dirs* |
|----------|---------|------:|-------------------:|
| `a_pass` | body is only `pass` / comment | 42 | 21 |
| `b_debugonly` | `log.debug(...)` only (silent in prod) | 298 | 130 |
| `c_return_swallow` | `return None/{}/[]/empty-frame/False` | 32 | 9 |
| `d_warn` | `log.warning/error` + continue (visible degradation) | 278 | 137 |
| `d_warn_return` | `log.warning/error` + `return` (visible, degrades) | 126 | 57 |
| `e_reraise` | re-raises (correct) | 71 | 22 |
| `f_continue_silent` | bare `continue` in a loop | 11 | 2 |
| `z_other` | other (assign sentinel, set flag, etc.) | 139 | 57 |

\* sensitive dirs = `pipeline`, `execution`, `risk`, `accounting`, `data`. **435 of 997** handlers (44%) live on these paths.

**Headline:** Only **7%** (71/997) re-raise. The dominant pattern in the trading
pipeline is `try: <protective-or-enrichment-step> ... except Exception: log.debug("... skipped")`.
At the default prod log level (INFO/WARNING), **every DEBUG-only swallow is invisible** —
a step that silently no-ops looks identical to a step that ran. The `_tc_*.py` pipeline
modules are built almost entirely from this idiom (see below).

### The structural danger: fail-OPEN risk reductions

The single most dangerous shape is a **protective size/qty reduction wrapped in
`try/except: log.debug(skipped)`**. If the protective computation raises, the reduction
never applies and **full-size orders pass through** — the system fails toward *more* risk,
logged only at DEBUG. This is the M-1/M-2 family generalized; it is pervasive, not isolated.

### Worst ~20 (extends prior M-1/M-2/R2-5 — does not repeat them)

| ID | file:line | Handler | Why dangerous (KLASSE) |
|----|-----------|---------|------------------------|
| QUAL-01 | `pipeline/_tc_risk.py:129` | `log.debug("evt_tail_var skipped")` | EVT-VaR qty-reduction (×0.80 when tail VaR > 2× hist) silently disabled → **fail-open** full size. **HOCH** |
| QUAL-02 | `pipeline/_tc_risk.py:148` | `log.debug("copula_tail_risk skipped")` | Copula tail-dependence risk gate silently off → fail-open. **HOCH** |
| QUAL-03 | `pipeline/_tc_risk.py:194` | `log.debug("barbell_strategy skipped")` | Barbell allocation guard silently off. **MITTEL** |
| QUAL-04 | `pipeline/_tc_risk.py:292` | `log.debug("anti_churn filters skipped")` | Turnover/churn filter silently off → uncapped churn & cost. **MITTEL** |
| QUAL-05 | `pipeline/_tc_sizing.py:2330` | `log.debug("halt-check skipped")` | Halted-symbol drop silently off → can size/trade a **halted** symbol. **HOCH** |
| QUAL-06 | `pipeline/_tc_sizing.py:2374` | `log.debug("buying-power pre-check skipped")` | Gross-weight cap vs buying power silently off → over-leverage past 95%. **HOCH** |
| QUAL-07 | `pipeline/_tc_sizing.py:2422` | `log.debug("pre-earnings check skipped")` | 50% pre-earnings size cut silently off → full size into earnings gap. **HOCH** |
| QUAL-08 | `pipeline/_tc_sizing.py:2471` | `log.debug("M&A filter skipped")` | M&A target filter silently off. **MITTEL** |
| QUAL-09 | `pipeline/_tc_sizing.py:996` | `log.debug("trailing_stops skipped")` | Trailing-stop overlay silently off → no stop protection. **HOCH** |
| QUAL-10 | `pipeline/_tc_sizing.py:1064` | `log.debug("turnover_budget gate skipped")` | Turnover budget gate silently off → uncapped turnover/cost. **MITTEL** |
| QUAL-11 | `pipeline/_tc_sizing.py:1120` | `logger.debug("correlation_guard skipped")` | Correlation/crowding guard silently off → concentrated correlated book. **HOCH** |
| QUAL-12 | `pipeline/_tc_sizing.py:572` | `log.debug("vol_targeting skipped")` | Vol-targeting scale silently off → un-vol-scaled sizing. **HOCH** |
| QUAL-13 | `pipeline/_tc_sizing.py:897` | `log.debug("factor_risk_model skipped")` | Factor-risk constraint silently off. **MITTEL** |
| QUAL-14 | `pipeline/_tc_sizing.py:1245` | `log.debug("inverse_etf hedge skipped")` | Crisis inverse-ETF hedge silently off → no hedge in stress. **MITTEL** |
| QUAL-15 | `pipeline/_tc_signals.py:653` | bare `return pd.DataFrame()` (**no log at all**) | Ensemble member signal-fn failure → empty frame blended as zero-signal; **silently drops a strategy's contribution**, zero log. **HOCH** |
| QUAL-16 | `risk/disclosures_confirm.py:103` | `logger.debug("[ERROR] ...")` | Entire disclosures-confirm overlay swallowed at DEBUG; geo-confidence boost silently not applied. **MITTEL** |
| QUAL-17 | `pipeline/_tc_sizing.py:2039` | `log.debug("cost_aware_wrapper skipped")` | Cost-aware sizing wrapper silently off → cost-blind sizing. **MITTEL** |
| QUAL-18 | `pipeline/_tc_execution.py:318` | `log.debug("total_cost_bps derivation skipped")` | Cost-bps annotation on fills silently off → cost reporting understated. **MITTEL** |
| QUAL-19 | `pipeline/_tc_execution.py:519` | `log.debug("trade_journal skipped")` | Trade-journal write silently off → **audit-trail gap** invisible in prod. **HOCH** |
| QUAL-20 | `pipeline/_tc_sizing.py:1170` | `logger.debug("crash_prediction equity cap skipped")` | Crash-prediction equity cap silently off → no crash de-risk. **HOCH** |

**Cross-reference (prior rounds, NOT re-listed in the table):**
- M-1 `pipeline/_tc_sizing.py:2062` — `except: policy = {}` → empty policy silently
  disables **every** policy-gated overlay below it. *Confirmed still present.*
- M-2 `pipeline/_tc_risk.py:101` — `except: _shared_rets = None` → makes QUAL-01/02
  unreachable (their `if _shared_rets is not None` guard short-circuits). *Confirmed.*
- R2-5 `ops/paper_ledger.py:55` — paper-ledger swallow. *Out of this scope, not re-verified.*

**Benign / correct uses spot-checked (NOT findings):**
- `risk/circuit_breaker.py:175` — `# pragma: no cover` stdlib `statistics` import guard.
- `execution/unified_paper_engine.py:975` & `risk/state_machine.py:151` — temp-file
  `.unlink()` cleanup after the real error is already logged. Correct `pass` usage.

### `a_pass` in sensitive dirs (21) — lower danger, mostly data fetch & temp cleanup

Concentrated in `data/sources/*` (alphavantage:54/80, fred:62/86, newsapi:57/89/112,
earnings_calendar:72/85/117, polygon:70) and `data/altdata/finnhub_*` and
`data/tick_store.py:123/169/230/253`. These swallow upstream-feed failures → degraded
data delivered silently. **Data-quality risk (MITTEL)** rather than execution risk, but
they violate the project's own "Datenprobleme nicht still verschlucken" rule.

---

## CENSUS 2 — Contract / schema drift (vs `docs/CONTRACTS.md`)

Reference contract (`docs/CONTRACTS.md`, §5.5 Trades): `status` is **lowercase**
(`filled`/`partial`/`rejected`); `side` is **UPPERCASE** (`BUY`/`SELL`); cost columns
are `commission_cash` / `spread_cash` / `slippage_cash` / `total_cost_cash`.

### Drift 2A — `status` casing diverges between producers/consumers — **HOCH**

| Site | Code | Casing |
|------|------|--------|
| `execution/fill_model.py:207/257/258/259` | `status == "filled"/"partial"/"rejected"` | lowercase (contract-correct) |
| `execution/fill_model.py:135` | `status...str.upper() == "REJECTED"` | defensive upper-compare |
| `execution/transaction_costs.py:272/279` | `status == "rejected"` | lowercase |
| `ops/rejection_collector.py:45` | `status == "rejected"` | lowercase |
| `execution/broker_execution.py:479` | `order.status == "filled"` | lowercase |
| **`api/routers/oms.py:129`** | **`order.status == "FILLED"`** | **UPPERCASE — diverges** |

`oms.py:129` filters OMS `Order` objects by `status == "FILLED"`. If the OMS `Order`
status is ever sourced from / compared against the lowercase trades-contract value (e.g.
a paper engine that emits `"filled"`), this filter returns **zero executions silently**.
Whether the OMS `Order` enum is independently uppercase by design is **unverified** — but
the coexistence of `"FILLED"` (oms) and `"filled"` (broker_execution) for an
`order.status` attribute is a latent producer/consumer casing split. **Needs owner check.**

### Drift 2B — `side` casing: two conventions coexist — **MITTEL**

UPPERCASE `BUY`/`SELL` (contract-correct): `paper_track.py`, `fill_model.py:80/765/766`,
`pre_trade_checks.py`, `ledger.py:198/200/332/375`, `unified_paper_engine.py` (many),
`trade_journal.py` (with `.upper()` guard), `pipeline/backtest.py:181/182`.

lowercase `buy`/`sell` (diverges from contract): `execution/round_trip_detector.py:44/49/50`,
`execution/order_gate.py:80/81`, `execution/order_management.py:190/193`,
`execution/limit_orders_v1.py:161/256/300/363`, `execution/broker_adapter.py:634/722/820`
(via `side_lower`). These modules form a lowercase island. As long as each island is
internally consistent it works, but any frame that crosses the boundary **without**
`.str.upper()` normalization gets mis-signed. Defensively-guarded crossings
(`fill_model.py:80`, `paper_ledger.py:222`, `trade_journal.py:217/218`,
`unified_paper_engine.py:1372/2073` all use `.str.upper()`) are safe; the lowercase
island modules that compare a *raw* `side` are the drift surface.

### Drift 2C — cost-column naming is consistent (no `impact_cash` leak) — **OK**

The prior-known `slippage_cash` vs `impact_cash` drift is **NOT present** in
`src/assembled_core`: all 30+ sites use `slippage_cash` (`ledger.py`,
`ledger_integration.py:368`, `accounting_report.py`, `portfolio.py:161`,
`fill_model.py`, `transaction_costs.py`). `impact_w` exists only as a *model weight*
input (`portfolio.py:160` comment "use impact_w for slippage_cash") — correctly mapped,
not a column-name drift. Contract §5.5 adherence is good here.

### Drift 2D — empty-orders edge schema (documented, but a real consumer trap) — **MITTEL**

`docs/CONTRACTS.md §5.5` documents that `simulate_with_costs` on empty orders returns a
**5-column minimal schema** (no `status`/`fill_qty`/cost columns). Any consumer that does
`trades["status"]` without a `"status" in trades.columns` guard will `KeyError` on the
empty path. This is a documented contract bifurcation, not a code bug, but it is a
recurring drift trap — consumers must branch on schema. Reference consumer that does it
right: `qa/backtest_engine.py:1570-1583`.

---

## CENSUS 3 — Determinism / reproducibility

**Verdict up front: determinism is largely well-handled in decision paths.** No
decision-affecting nondeterminism found. Details:

### Seeded RNG everywhere it matters — **OK**

- `signals/causal_ml.py:419`, `signals/lppls_crash.py:86/172` → `np.random.default_rng(seed)`.
- `ml/regime_hmm.py` → `random_state=42` default, multi-seed deterministic
  (`seeds = [self.random_state + i*7 ...]` :417), persisted/restored (:520/548).
- `ml/feature_selection.py:412`, `ml/lime_explainer.py:51`,
  `ml/temporal_fusion_transformer.py:62` → `random_state=42`.
- `portfolio/quantum_portfolio.py:162` → `sampler.sample(..., seed=cfg.random_state)`.
- **Zero** unseeded `np.random.rand/randn/choice/shuffle/normal` in
  `signals/strategies/portfolio/risk/execution/pipeline/ml`.

### `datetime.now()` / `time.time()` — NOT in pipeline decision paths — **OK**

No `datetime.now()` / `datetime.utcnow()` / `time.time()` in `src/assembled_core/pipeline`.
Time enters via injected `as_of` / ctx clock, consistent with the project PIT discipline.

### `os.environ` reads — NONE in core decision dirs — **OK**

No `os.environ` / `os.getenv` reads in
`pipeline/execution/risk/signals/strategies/portfolio`. Behavior is not env-switched on
the hot path.

### Set-iteration into ordered output — one COSMETIC instance (NOT decision-affecting)

`pipeline/trading_cycle_shared.py:1311` — `for grp in set(groups):`. The iteration order
of `set()` over strings varies with `PYTHONHASHSEED`, **but** the loop only (a) appends to
a diagnostic `scaled_groups` list (order-only) and (b) applies `np.minimum(scale_factors[mask], factor)`
which is commutative. Final `qty` is order-independent → **cosmetic, NOT decision-affecting.**
Honest distinction per the brief. (The prior round-2 F-002 `set`-trim nondeterminism in
`news_alpha` is reported fixed in memory; not re-found in this scope.)

---

## Opportunistic notes

- **TODO/FIXME in risk/exec/pipeline/accounting:** exactly **one** —
  `pipeline/orchestrator.py:1431` `# TODO: wire to post-signal-computation step when
  factor panel is available.` (matches known factor-decay no-op stop-gap, memory 9467b0ae).
  risk/, execution/, accounting/ are **TODO-free**. Clean.
- **Hardcoded-constant / dummy returns:** not systematically swept in this census; the
  `c_return_swallow` set above (esp. QUAL-15 empty-frame, M-1 empty `{}`) are the
  swallow-driven dummies. No standalone "returns hardcoded value where computation
  implied" instances surfaced incidentally beyond those.

---

## Honest verdict

**Silent-degradation is pervasive and structural, not incidental.** The four `_tc_*.py`
pipeline stages are architecturally built from `try: <step> except Exception: log.debug("skipped")`
(~130 DEBUG-only swallows in sensitive dirs; only 7% of all handlers re-raise). The
*intent* — "an optional overlay failing should not crash the cycle" — is defensible, but
the *implementation* is fail-OPEN for protective steps (halt-check, buying-power cap,
trailing stops, vol-targeting, correlation guard, crash cap all degrade toward MORE risk
when they raise) and is **invisible at prod log level**. Contract drift is contained:
cost columns are clean, the live risk is `status`/`side` casing islands (one concrete
`oms.py:129 "FILLED"` divergence needs owner confirmation). Determinism is the healthy
dimension — seeded throughout, no env/clock leaks in decision paths, the one `set()`
iteration is cosmetic.

**Highest-leverage remediation (not performed — read-only):** promote the protective-step
swallows (QUAL-01/05/06/07/09/11/12/20, M-1) from `log.debug` to `log.warning` AND emit a
per-cycle "degraded steps" QA artifact, so a silently-disabled protection is at minimum
*observable*. That is a behavior-visible change in a protected path — requires explicit
authorization and the risk-execution review chain.

*All findings static-only. NOT CI-confirmed. No files were modified.*
