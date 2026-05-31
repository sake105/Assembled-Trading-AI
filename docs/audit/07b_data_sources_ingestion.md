# 07b — Data Sources / Ingestion / Feed Layer Audit (Round 3)

READ-ONLY static audit. No files changed. Scope: `src/assembled_core/data/**`,
every external feed adapter, data validation, corporate actions, survivorship,
calendar/PIT, caching, and dummy/placeholder data.

Method: Grep/Glob to locate, Read to verify. Each finding carries file:line +
evidence. "Verified at source" = quoted code read directly. "Pattern-matched" =
inferred from grep without full-context read. Static-only — NOT CI-confirmed.
Anything needing execution is flagged UNSURE.

Severity scale: KRITISCH / HOCH / MITTEL / NIEDRIG.

---

## Executive verdict on feed reliability

The **individual feed adapters are mostly well-hardened** at the call-site level:
retry/backoff, 429-rotation, timeouts, and WARN-on-empty are present in the
modern adapters (yfinance, FRED, newsapi, finnhub, worldbank, edgar). The
**systemic weakness is the validation layer**: two fully-built OHLCV quality
gates (`data/quality_gate.py`, `dataquality/gate.py`) and the
`FreshnessMonitor` are **completely unwired** — no production code path imports
them. A corrupt or stale feed therefore flows from adapter → parquet → signals
with only the lightweight `validate_price_data` warning (which only logs, never
blocks). The pervasive **E-025 empty-DataFrame-on-error contract** means a total
feed outage degrades to "zero rows" indistinguishable from a legitimately empty
window. Net: adapters are good; the gate between adapters and signals is largely
decorative.

---

## KRITISCH

(none — no live-trading data corruption proven statically; the gate gaps are
HOCH because they are latent, not actively producing wrong fills)

---

## HOCH

### DAT-001 — OHLCV quality gate `data/quality_gate.py` is dead infrastructure (unwired)
- **File:** `src/assembled_core/data/quality_gate.py` (entire module)
- **Evidence:** `validate_ohlcv()` (line 170) implements full validation —
  monotonic timestamps (`_check_timestamps_monotonic`, line 114), null prices,
  zero/negative prices, high<low, close-outside-range, spikes, pandera schema.
  Grep for callers: `validate_ohlcv|quality_gate|QualityResult` across `src/`
  returns only **2 files: the gate itself + the unrelated `dataquality/gate.py`**.
  No ingestion path (`prices_ingest.py`, download scripts) imports it.
- **Impact:** The documented contract ("Every incoming OHLCV batch is validated
  before features are computed", module docstring line 5) is **false**. Nothing
  validates batches. Verified at source.
- **Secondary defect:** the gate checks capitalized columns `Open/High/Low/Close/
  Volume` (lines 60, 66-75) while the canonical panel in `prices_ingest.py` uses
  **lowercase** `open/high/low/close/volume` (line 136). Even if wired, the
  structure check `required - set(df.columns)` (line 61-62) would fail on the
  canonical frame → schema mismatch.

### DAT-002 — Second quality gate `DataQualityGate` also unwired (test-only)
- **File:** `src/assembled_core/dataquality/gate.py` — class `DataQualityGate`
  (line 26), `validate_ohlcv` (line 48), `run_anomaly_checks` (line 115),
  `_quarantine` (line 169).
- **Evidence:** Grep `DataQualityGate` → 5 hits, all non-production:
  `tests/test_data_quality_gate.py`, the module + its `__init__`, and 2
  system-map JSON/JS docs. No production caller.
- **Impact:** Two parallel, fully-implemented OHLCV gates exist; **neither is
  wired** into ingestion. Architecture-boundary smell (duplicate truth, Rule 50)
  plus the same latent "no validation actually runs" gap as DAT-001.
  Verified at source.

### DAT-003 — `FreshnessMonitor` is dead infrastructure; staleness never detected from caches
- **File:** `src/assembled_core/data/freshness_monitor.py`
- **Evidence:** It is an in-memory dataclass; `update(source)` stamps
  `datetime.now(timezone.utc)` (line 50-52) and `is_stale` compares against
  `max_age_hours`. It does **not** read parquet/cache file mtimes. Grep
  `FreshnessMonitor|freshness_monitor|register(` across `src/` → 7 hits, none in
  any data-download / ingestion path (hits are `model_registry`,
  `strategies/base`, `signals/registry`, `intel/health_monitor`, the module
  itself, `feature_store`, `snapshot` — and the latter two are name-collisions on
  `.update(`/`register(`, not the monitor).
- **Impact:** There is **no cache-mtime staleness gate** on the price/macro/news
  parquet caches. A frozen `output/macro.parquet` (FRED feed dead for weeks) is
  invisible — backtests/live run on stale macro with no alert. Verified at source.

### DAT-004 — `_fetched_at` staleness marker written but never read
- **Files:** writer `scripts/download_all_market_data.py:33-37` (`_stamp()` adds
  `df["_fetched_at"] = datetime.now(timezone.utc).isoformat()` "so factor_store
  can detect stale data"); consumer **absent**.
- **Evidence:** Grep `_fetched_at|stale|max_age|getmtime|st_mtime` in
  `data/factor_store.py` → **no matches**. The docstring claim that factor_store
  uses `_fetched_at` for staleness is unbacked. The `load_eod_prices` path in
  `prices_ingest.py` actively **drops** non-OHLCV columns (line 134-146), so
  `_fetched_at` is stripped before reaching consumers anyway.
- **Impact:** Staleness detection on stamped feeds is a no-op. Verified at source.

### DAT-005 — E-025 family: pervasive empty-DataFrame-on-error masks total feed failure
- **Files (verified at source):**
  - `sources/fred_source.py:144-228` — every failure path (`api_key None` 162,
    `ImportError` 168, client-init fail 174, all-fetch-fail 214) returns `_empty`.
  - `sources/yfinance_source.py:128-159` — per-symbol failures collapse to empty
    frames (`_fetch_single_symbol` returns `None`, dropped at 148-149); a 90%
    partial outage returns a silently-thin frame, full outage returns empty (152-159).
  - `sources/newsapi_source.py:224-228`, `sources/worldbank_source.py:64-141`,
    `altdata/finnhub_news_macro.py:96-104`, `data/altdata_loader.py:47-57,105-115`,
    `sources/cboe_source.py:81,135,176,182`.
- **Impact:** The contract "error → empty frame" is structurally indistinguishable
  from "legitimately no data in window." Downstream sees zero rows and either
  produces zero signals (silent no-trade) or, for macro/news factors, a silently
  dead factor. There IS a WARN log in most paths (good), but no caller is forced
  to react — the masking is at the *return-type* level, not the log level.
  This is the single most systemic reliability pattern in the feed layer.

### DAT-006 — `build_universe_history_from_prices` infers delisting from panel coverage (survivorship hazard)
- **File:** `src/assembled_core/data/universe.py:221-250+`
- **Evidence:** Membership windows are derived from each symbol's first/last
  timestamp in the price panel; "Symbols whose last row equals the panel maximum
  are treated as still listed (end_date = NaT). All others get end_date =
  last_ts + 1 business day" (docstring 226-232; logic 240-250).
- **Impact:** A symbol absent from the tail of the panel **for a data-coverage
  reason** (feed gap, ingestion failure, ticker rename) is mis-classified as
  **delisted**, and conversely a genuinely-delisted symbol that happens to share
  the panel's max date is mis-classified as active. Universe truth becomes a
  function of feed completeness rather than corporate reality. This is a real
  survivorship/look-ahead coupling, not pattern-matched. Verified at source.
  Mitigant: the PIT API (`get_universe_members_pit`) and the
  `require_active_status=True` default (line 85) are correctly conservative.

---

## MITTEL

### DAT-007 — No retry/backoff in newsapi / finnhub / worldbank / edgar fetch loops
- **Evidence:** Only `yfinance_source.py` has a real retry loop
  (`_RETRY_MAX=3`, exponential backoff `_RETRY_BACKOFF_BASE**attempt`, lines
  26-27, 100-110). `newsapi_source.fetch_news_headlines` (156-222) does a single
  POST per query then `continue` on error; `finnhub_news_macro.fetch_news`
  (58-84) single GET then `continue`; `worldbank_source` (80-88) single GET;
  `edgar_source` single GET. They have **429 key-rotation** but no
  same-key retry/backoff for transient 5xx/network blips.
- **Impact:** A transient network error or 503 silently drops that symbol/query's
  data for the run (folds into DAT-005). Lower severity because rotation +
  next-run recovery exist. Verified at source.

### DAT-008 — FRED 6h TTL cache is process-local and never invalidated by source age
- **File:** `sources/fred_source.py:42-43,178-210`
- **Evidence:** `_FRED_CACHE: dict` is module-global, keyed
  `series_id|start|end`, TTL 21600s. It correctly does **not** cache `None`
  (F-AKR2-10, line 207-210). But TTL is wall-clock since fetch, not tied to FRED
  release schedule; and the cache is per-process (lost on restart, not shared).
- **Impact:** Minor — within a long-running process a series can be up to 6h
  stale with no signal; cross-process there is no shared cache so no staleness
  carryover. Acceptable but undocumented as a freshness assumption. Verified.

### DAT-009 — `validate_price_data` only warns, never blocks; OHLC-invalid rows pass through
- **File:** `prices_ingest.py:148-172`
- **Evidence:** Invalid OHLC relationships (`high<low` etc.) are detected
  (149-156) and **logged at WARNING** (157-161); `validate_price_data` issues are
  logged (164-167) but the frame is returned regardless. The function's only hard
  failure is the volume-coercion breach (119-124, good) and missing-OHLCV (88-97).
- **Impact:** A feed that produces `high < low` bars (e.g. a bad split adjustment
  or a corrupt vendor row) flows straight into the panel — only a log line marks
  it. There is no quarantine. The real gates that *would* block (DAT-001/002) are
  unwired. Verified at source.

### DAT-010 — `incremental_update` dedup keeps `last` by (timestamp, symbol) — restatement-blind
- **File:** `prices_ingest.py:390-432`
- **Evidence:** `drop_duplicates(subset=[timestamp_col, symbol_col], keep="last")`
  (417-419). New data overwrites existing for the same (ts, sym).
- **Impact:** Correct for late-arriving corrections, but there is **no audit /
  diff** of what changed — a vendor restatement that silently rewrites a historical
  close is applied with no trace. Also reads/writes the whole file each call (no
  atomic temp-write — partial write on crash corrupts the cache; contrast
  factor_store's `_write_parquet_atomic`, line 107). Verified at source.

### DAT-011 — CBOE source docstring claims FRED; code uses yfinance (doc/code drift)
- **File:** `sources/cboe_source.py`
- **Evidence:** Class docstring (46-48): "VIX / VIX3M: Federal Reserve FRED API
  (no key required)"; `__init__` takes `fred_api_key` (54-55). Actual
  `fetch_vix` uses `yf.download([_YF_VIX, _YF_VIX3M], ...)` (87-93). `fred_api_key`
  is stored (55) but **never used**.
- **Impact:** Misleading provenance documentation (Rule "Doku ist Steuerung").
  Operationally fine but a maintainer reasoning about the VIX source path will be
  wrong about which feed/key is in play. Verified at source.

### DAT-012 — `apply_splits_for_research_prices` and `adjust_prices_for_splits` are not idempotent and use different schemas
- **File:** `corporate_actions.py:27-91` vs `143-218`
- **Evidence:** Both back-adjust pre-split rows by `1/split_ratio`. Neither
  records that an adjustment was already applied, so calling either twice on the
  same frame double-divides (e.g. a 10:1 split applied twice → 1/100). The two
  functions also differ: `apply_splits_for_research_prices` writes a separate
  `close_research` column (non-destructive, 90) while `adjust_prices_for_splits`
  mutates `close` in place (216). A caller that runs both, or runs the in-place
  one across re-ingestion, compounds the adjustment.
- **Impact:** Idempotency is **not guaranteed at the function level** — it depends
  entirely on the caller never re-applying. No `already_adjusted` guard exists.
  Round-2 reportedly verified corporate-action *file* idempotency; this is the
  *function* re-application hazard, which is distinct. Verified at source.
  UNSURE whether any production path double-applies — needs call-graph trace.

---

## NIEDRIG

### DAT-013 — `date.today()` (local-tz) in daily counters and default date windows
- **Evidence:** `newsapi_source.py:71` (`today = date.today().isoformat()` for the
  daily-call-limit rollover), `weather_source.py:74,76,218,219`,
  `wikipedia_views_source.py:78`, `stooq_source.py:59`, `fx.py:98,100`,
  `cboe_source.py:84` (`datetime.today()`).
- **Impact:** `date.today()` is server-local, not UTC. The newsapi daily counter
  can roll the limit window at local midnight rather than the API's reset moment
  (minor quota mis-accounting). Default macro/fx/weather windows shift by ≤1 day
  depending on server TZ — not a PIT violation (these are *fetch-window*
  defaults, not as_of features), but non-deterministic across hosts. Verified.

### DAT-014 — `factor_store._load_all_partitions` returns partial data on corrupt partitions (permissive by design)
- **File:** `factor_store.py:123-186`
- **Evidence:** Corrupt partitions are logged at **ERROR** (147, 149-155) and a
  manifest/data year-mismatch is logged (166-176), but the function **still
  returns the concatenation of the readable partitions** (184-186). The docstring
  explicitly states this is intentional ("still permissive ... returns partial
  data") for append-mode rebuilds.
- **Impact:** A silently-truncated factor panel (e.g. year 2021 partition with a
  bad footer) yields ERROR logs but a usable-looking partial panel; a backtest not
  watching logs computes IC/Sharpe on an incomplete series. The ERROR log is the
  only guardrail. Reasonable trade-off, flagged for completeness. Verified.

### DAT-015 — `is_market_open_now` ZoneInfo fallback hardcodes EDT (UTC-4)
- **File:** `calendar.py:130-141`
- **Evidence:** When `zoneinfo` import fails, `utc_offset_hours = -4 # assume EDT
  (DST active); will be wrong for EST period` (138). A WARN is logged (135-137).
- **Impact:** Only triggers if `zoneinfo` AND `backports.zoneinfo` are both
  unavailable (rare on supported Python). During the EST period (Nov–Mar) the
  market-open check would be off by 1h. Self-documented and WARN-guarded.
  Verified at source.

### DAT-016 — insider factor data is structurally dead (transaction_type='unknown')
- **File:** `data/altdata_loader.py:103` ("transaction_type is currently 'unknown'
  for all rows (data quality issue)").
- **Impact:** Confirms the known dead-factor: insider-trading factor is always 0
  because direction is unknown. Not new (documented in MEMORY), recorded here as a
  data-source-layer confirmation. Verified at source.

---

## POSITIVE confirmations (correct / hardened)

- **P-01 — Volume coercion breach raised, not swallowed.** `prices_ingest.py:104-124`
  distinguishes pre-existing NaN from coerce-from-junk and **raises** on feed
  corruption rather than silently → 0.0. Correct E-025 inversion. Verified.
- **P-02 — Corporate actions raise on schema drift.** `adjust_prices_for_splits`
  (176-187), `apply_splits_for_research_prices` (44-49) raise `ValueError` on
  missing columns; `compute_total_return_index` / `apply_delisting_exits` /
  `apply_spinoff` WARN on every silent no-op path (corporate_actions.py:248-281,
  367-372, 490-495). No silent unadjusted-return fork. Verified.
- **P-03 — Delisting exit refuses post-delisting fallback price.** `apply_delisting_exits`
  skips with WARN when no pre-delisting price exists rather than using `iloc[-1]`
  (F-B-9 fix, corporate_actions.py:414-428). PIT-correct. Verified.
- **P-04 — PIT universe API is strict.** `get_universe_members_pit` mandates `as_of`,
  raises on empty result, defaults `require_active_status=True`
  (universe.py:166-213). `get_universe_members(as_of=None)` WARNs that the
  watchlist fallback is not PIT-safe (107-119). Verified.
- **P-05 — Resample drops partial trailing period (no look-ahead).** `resample.py:205-225`
  removes the last period when `pit_cutoff` falls inside it. Verified.
- **P-06 — FRED does not cache failures.** `fred_source.py:207-210` only caches
  non-empty results; rotation retry is gated to genuine rate-limit signals only
  (189-206), not wasted on missing-series. Verified.
- **P-07 — EDGAR respects SEC rate limit + UA.** `edgar_source.py:30-34` enforces
  0.11s min interval and a descriptive User-Agent (SEC requirement). Verified.
- **P-08 — factor_store atomic writes + ERROR-on-corruption.** `_write_parquet_atomic`
  (107) and ERROR-level corrupt-partition + manifest cross-check
  (123-186). Verified.
- **P-09 — synthetic_generator is honestly labeled.** Module docstring "for stress
  testing" (line 2); all functions deterministic-seeded crisis/GARCH/jump models,
  not presented as real feeds; no production ingestion imports it. No
  dummy-as-real masquerade found in the data layer. Verified.
- **P-10 — No E-030/E-031/E-032/E-033 in the data layer.** Grep found **no** bfill-
  on-pivot, **no** `month.values` year-strip, **no** `astype(int)`/`int32`
  (the two int casts use `int64`: insider_ingest.py:140, shipping_routes_ingest.py:168),
  and tz-strip-before-parquet only in snapshot.py:47 (hash normalization, not a
  persisted panel — round-trip-safe). Verified by grep + targeted read.

---

## Items needing execution to confirm (UNSURE)

- **U-01 (DAT-012):** Whether any production call path double-applies split
  adjustment (re-ingestion + research path). Needs a call-graph / runtime trace.
- **U-02 (DAT-008):** Real FRED cache staleness behavior under a long-running
  pilot process — needs a running process to observe TTL boundary.
- **U-03:** Whether `prices_ingest.load_eod_prices` is ever fed a frame that the
  unwired gates *would* have rejected (high<low survivors). Needs sample data run.

---

## Coverage note

Read at source: prices_ingest, corporate_actions, universe, calendar,
yfinance_source, fred_source, newsapi_source, synthetic_generator, snapshot,
altdata_loader, freshness_monitor, finnhub_common, finnhub_news_macro,
quality_gate, resample, cboe_source, worldbank_source, factor_store, edgar_source,
download_all_market_data (head). Grep-swept the whole `data/` tree for the four
named anti-patterns, bare-except masking, `datetime.now()`/`date.today()` naive
usage, pivot/bfill, and gate/freshness wiring. Not deep-read (lower risk, time):
polygon/alphavantage/stooq/kalshi/polymarket/bls/finra/weather/wikipedia/
earnings_calendar source bodies, tier_processor, security_master, panel_store,
tick_store, ledger_store, crypto, fx internals — recommend a follow-up pass if
those feeds become live-path.
