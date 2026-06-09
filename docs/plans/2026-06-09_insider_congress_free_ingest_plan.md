# Plan — Replace paid QuiverQuant with two FREE ingesters (EDGAR Form 4 · free Congress)

Status: PLAN ONLY (Phase 1+2). No code changed. Read-only investigation done 2026-06-09 via Dynamic Workflow (11 agents).
Hard rule: Form 4 (insider) and Congress (STOCK-Act) are SEPARATE ingesters, modules, output files, factors. Never merge.

---

## 0. TWO PREMISE CORRECTIONS (read first)

1. **The "insider 100% unknown type" defect is NOT in `insider_ingest.py`.**
   `src/assembled_core/data/insider_ingest.py` is a Phase-6 skeleton (`load_insider_sample()` + `normalize_insider()`),
   schema `timestamp, symbol, trades_count, net_shares, role`, writes nothing to disk, and its dummy path is already
   fail-loud-gated (`allow_sample=False` → `ValueError`). The real "unknown" bug is in
   `scripts/download_all_market_data.py:601`, which hardcodes `transaction_type="unknown"`, `shares=None`, `price=None`
   in `_fetch_form4_for_cik` and writes `output/insider_trading.parquet` (KNOWN_ISSUES.md:280 = 59,506 rows all unknown;
   `multifactor_v2.py:267` zeroes the factor because of it). A SECOND defect compounds it:
   `altdata_loader.load_insider_filings()` (`altdata_loader.py:176`) strips the frame to `[symbol, filing_date, shares_delta]`,
   dropping `transaction_type`/`value_usd` — so even good data never reaches the live factor wrapper.

2. **Finnhub congressional-trading is PREMIUM-only, not free.** A free key returns HTTP 401
   ("You don't have access to this resource."). The "Congress via Finnhub Free-Tier" premise does not hold.
   Free path instead: extend the EXISTING official-disclosure scraper
   `src/assembled_core/events/disclosures/fetch_house_ptr.py` (House Periodic Transaction Reports), which already
   feeds `congress_features.py`. Note: `src/assembled_core/data/congress_trades_ingest.py` DOES NOT EXIST, and
   `pipeline/trading_cycle_shared.py:756` imports it under `try/except ModuleNotFoundError` → `include_congress=True`
   is currently a SILENT no-op (factor 30 always 0, weight 0.00 at `multifactor_v2.py:283`).

---

## 1. TARGET SCHEMA MAPPING

### (a) EDGAR Form 4 XML → our column (output/insider_form4.parquet)
Consumer `earnings_insider_wrapper.compute_earnings_insider_factors` REQUIRES `[symbol, filing_date, transaction_type, value_usd]`
with `transaction_type ∈ {P,S}`. We emit a superset so the time-series path (`insider_features.add_insider_features`:
`[timestamp, symbol, net_shares, trades_count, role]`) is also served.

| Form 4 XML (no namespace prefix) | our column | dtype | notes |
|---|---|---|---|
| `issuer/issuerTradingSymbol` | `symbol` | str | blank/NONE → fall back via `issuerCik`→company_tickers.json |
| `issuer/issuerCik` | `issuer_cik` | str(10) | join fallback; ≠ rptOwnerCik |
| `reportingOwner/reportingOwnerId/rptOwnerCik` | `reporting_owner_cik` | str(10) | iterate ALL owner blocks |
| `reportingOwnerRelationship/{isDirector,isOfficer,isTenPercentOwner,isOther}` | `role` | str | default 'Unknown' if all false |
| SGML header `ACCEPTANCE-DATETIME` | `available_at` | datetime64[ns,UTC] | **PIT anchor**; ET→America/New_York→UTC |
| SGML header `FILED AS OF DATE` | `filing_date` | datetime64[ns] (date) | what wrapper + loader gate on (`<=as_of`) |
| `nonDerivativeTransaction/transactionDate/value` | `transaction_date`/`event_date` | datetime64[ns] | economic date; NEVER `available_at` |
| `transactionCoding/transactionCode` | `transaction_code` | str(1) | raw SEC code, verbatim for audit |
| derived | `transaction_type` | str | P→P, S→S, **else `unknown`+WARNING** (no silent default) |
| `transactionAmounts/transactionShares/value` | `shares` | float64 | |
| `transactionAmounts/transactionPricePerShare/value` | `price` | float64 | |
| `shares*price` | `value_usd` | float64 | **REQUIRED by wrapper**; signed |
| `transactionAcquiredDisposedCode/value` | `acquired_disposed` | str A/D | exact tag `transactionAcquiredDisposedCode`; A=+ D=− |
| computed (A=+/D=−) | `net_shares` | float64 | for insider_features |
| computed | `trades_count` | int64 | for insider_features |

Fetch raw `ownership.xml` (root `<ownershipDocument>`); the `/xslF345X05/` sibling is rendered HTML — do not parse.

### (b) Free Congress field → our column (output/congress_trades.parquet)
Consumer `congress_features.add_congress_features` reads required `[timestamp, symbol]`, optional `amount, event_date, disclosure_date`.
Downstream factor 30 reads only pre-merged panel cols `congress_trade_count_90d` + `congress_total_amount_90d`.

| free source (House/Senate eFD; *-stock-watcher) | our column | dtype | notes |
|---|---|---|---|
| ticker | `symbol` | str | required |
| trade date | `transaction_date`/`event_date` | datetime64[ns] | economic date; NEVER `available_at` |
| "Notification Date" / disclosure | `disclosure_date` | datetime64[ns] | **PIT anchor**, `<=as_of` gate |
| disclosure_date (UTC) | `available_at` | datetime64[ns,UTC] | feature_store path only |
| amount bucket/range | `amount` | float64 | coarse RANGE → representative; flag DEGRADED |
| Sale/Purchase | `transaction_type` | str | normalize buy/sell |
| representative | `member` | str | audit/dedup |
| fallback when disclosure missing | `disclosure_date = transaction_date + CONGRESS_DAYS(45)` | | import constant from source_latencies; never inline |

---

## 2. available_at / PIT CONTRACT
- Default path does NOT use `feature_store` — insider→`output/insider_form4.parquet` (consumed by
  `load_insider_filings`/`insider_features`), congress→`output/congress_trades.parquet` (consumed by
  `add_congress_features`); both PIT-gate on `disclosure_date`/`filing_date <= as_of`. This sidesteps the
  `compute_event_betas` foot-gun (it wrote `inference_ts=max(event_date)` and never `available_at`, while the
  feature_store reader joins on `available_at` → silent None). If a feature_store view is ever added, write
  `available_at` explicitly and assert non-null UTC before `write_features` (which silently stamps `now()` if missing,
  `feature_store.py:114-119`).
- E-038 discipline: keep raw `transaction_date`/`event_date` immutable; compute `disclosure_date`/`available_at` as a
  SEPARATE derived column used ONLY for the `<= as_of` upper bound. The +45d congress fallback writes `disclosure_date`,
  never overwrites `transaction_date`.

## 3. EDGAR COMPLIANCE
- User-Agent REQUIRED, canonical SEC form `"Assembled-Trading-AI hans.oertel2@gmail.com"` (pulled from settings/env,
  no fake domain); undeclared UA → 403.
- ≤8 req/s, 0.12s spacing, exponential backoff on 403/429/503. Budget the 10/s cap GLOBALLY across
  www.sec.gov + efts.sec.gov + data.sec.gov.
- Bulk enumeration via daily-index `…/edgar/daily-index/YYYY/QTRn/form.YYYYMMDD.idx`, filter Form Type ∈ {4,4/A}.
  (EFTS `efts.sec.gov` JSON only as cross-check: capped 10k hits, indexes since 2001 only.)
- transactionCode classification with explicit `unknown` + per-code WARNING + end-of-run `% unknown` summary;
  do not silently coerce, do not drop the row.
- Multi-owner / multi-transaction: iterate all reportingOwner blocks + nonDerivative+derivative rows; dedupe by
  accession; 4/A amendments supersede.

## 4. CONGRESS PLAN
- Free fallback = extend `events/disclosures/fetch_house_ptr.py`; new `congress_trades_ingest.py` wraps it into the
  `add_congress_features` schema. (Finnhub premium path documented for later: reuse `finnhub_common`/`api_key_rotator`,
  but fix that a 401 entitlement-denial currently falls through `mark_finnhub_rate_limited` as a SILENT no-op.)
- `available_at` = disclosure date, never `transactionDate` (STOCK Act ≤45d lag).
- amount is a coarse range → representative mapping; flag low-resolution.

## 5. FILE PLAN
New (outside hard-protected paths): `src/assembled_core/data/edgar_form4_ingest.py`,
`src/assembled_core/data/congress_trades_ingest.py`, `tests/test_edgar_form4_ingest.py`,
`tests/test_congress_trades_ingest.py`, optional thin runner in `scripts/`.
Edit (advisory-sensitive): `src/assembled_core/config/settings.py` (SEC UA contact), docs.

PROTECTED / present separately with one-shot auth + risk-execution-reviewer:
- `src/assembled_core/pipeline/trading_cycle_shared.py` (HARD-PROTECTED) — congress wiring; without it factor 30 stays 0.
- `src/assembled_core/data/altdata_loader.py:176` keep-list — widen `+transaction_type,+value_usd`; without it the live
  insider factor stays dead even with good data.
- `scripts/download_all_market_data.py` (not protected) — legacy "unknown" writer; needs a retire/redirect/keep decision.
- `multifactor_v2.py` weights (:267,:283) — re-activating the 0.00 factors is a SEPARATE evidence-gated decision (out of scope).

## 6. PHASE-4 VERIFICATION
- Insider: 1 daily-index day (e.g. 2024-10-08) or 3-5 tickers/5-day window; print per record
  `symbol, reporting_owner_cik, transaction_date, filing_date, available_at(UTC), transaction_code, transaction_type,
  acquired_disposed, shares, price, value_usd`; batch summary incl. **% unknown** + observed req/s.
  Accept: available_at == ACCEPTANCE-DATETIME (ET→UTC, ≠ filing midnight); req/s ≤ 8; **% unknown ≪ 100%**;
  value_usd=shares*price; multi-owner filing → multiple rows.
- Congress: few House PTR filings; print `symbol, member, transaction_date, disclosure_date, available_at, amount, transaction_type`.
  Accept: available_at = real notification date (or +45d only when absent); `add_congress_features` emits non-zero
  count/amount; range→representative sane.
- All LOCAL smoke + offline fixture tests; NOT CI-confirmed until suite runs.

## 7. OPEN DECISIONS (user)
1. Congress source breadth (House-only free vs +Senate/mirrors vs Finnhub premium vs defer).
2. Protected wiring authorization (pipeline + altdata_loader keep-list) now vs separate-later.
3. Insider legacy file collision (new file vs redirect downloader vs retire legacy).
4. Congress amount-range mapping policy (geometric midpoint default).
5. Factor re-activation stays OUT OF SCOPE (weights 0.00; prior mfv2 full-stack showed Sharpe-Δ +0.00 — edge unproven).
6. Secret hygiene flag: `.env` holds real FINNHUB keys (gitignored = future only); if ever historically committed → rotate.
