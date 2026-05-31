# 07a — Security / Secrets / Auth / MNPI-Compliance Audit

- **Date:** 2026-05-30
- **Auditor role:** READ-ONLY static security auditor (round 3, multi-agent audit)
- **Mode:** Static source review only. **NOT CI-confirmed.** No code changed.
- **Scope:** secrets handling, API auth & middleware, kill-switch/operator auth,
  log hygiene, MNPI/PIT-disclosure compliance, input validation at system
  boundaries, dangerous deserialization/eval/subprocess primitives.

> Severity legend: **KRITISCH** (exploitable now / direct loss), **HOCH** (serious,
> conditional), **MITTEL** (real defect, bounded blast radius), **NIEDRIG** (hardening / cosmetic).
> Each finding marks evidence as **[verified-at-source]** or **[pattern-matched]**.

---

## Summary of severity counts

| Severity | Count |
|----------|-------|
| KRITISCH | 0 |
| HOCH     | 1 |
| MITTEL   | 4 |
| NIEDRIG  | 3 |

No tracked live-secret file found. No `eval`/`exec`/`os.system`/`shell=True`/unsafe-`yaml.load`/`pickle.load` on user input found. The auth + secrets layer is broadly well-built; the open items are about **fail-open defaults** and **unauthenticated file-path / deserialization surfaces**.

---

## HOCH

| ID | Title | File:Line | Evidence | Status |
|----|-------|-----------|----------|--------|
| SEC-1 | API command-auth **fails OPEN** when `ASSEMBLED_API_KEY` is unset — all state-mutating endpoints (paper orders, paper reset, kill-switch activate) are unauthenticated by default | `src/assembled_core/api/auth.py:43-48` | `expected = _expected_key(); if expected is None: logger.warning("...command endpoints are UNPROTECTED"); return` | [verified-at-source] |

**Detail (SEC-1):** `require_api_key` returns (allows) when the env var is absent. The
default dev/test posture therefore leaves `POST /api/v1/paper/orders`,
`POST /api/v1/paper/reset`, and `POST /api/v1/kill-switch/activate` open. If the API
is ever bound to a non-loopback interface without `ASSEMBLED_API_KEY` set, anyone can
submit paper orders or **DoS the trading system by activating the kill-switch**
(activate is NOT operator-token gated — only deactivate is, see SEC-6 positive).
The token comparison itself is correct (`hmac.compare_digest`, line 47). `/ready`
does not surface auth posture even though the docstring claims it does
(`auth_is_configured()` exists at `auth.py:51` but is **not called** by the `/ready`
handler in `app.py:53-92`) — so ops cannot verify the gap via the readiness probe.
Mitigating: single-tenant backend, intended loopback bind. Raised to HOCH because the
failure mode is silent and the most dangerous endpoint (kill-switch activate) is the
least protected.

---

## MITTEL

| ID | Title | File:Line | Evidence | Status |
|----|-------|-----------|----------|--------|
| SEC-2 | Unauthenticated **arbitrary file-path probe** on `/api/v1/ledger` — `ledger_path` query param has NO path-traversal guard (unlike `/health`) | `src/assembled_core/api/routers/ledger.py:33-36, 52-74` | `ledger_path: str = Query(default=_DEFAULT_LEDGER ...)`; `jpath = Path(ledger_path); ... state = load_ledger_state(jpath, ...)` | [verified-at-source] |
| SEC-3 | Model deserialization hash-check is **non-enforcing by default** (`strict=False`): on SHA256 mismatch it logs a WARNING and `joblib.load`s the (possibly tampered) pickle anyway | `src/assembled_core/ml/model_registry.py:109-112, 118-134`; caller `src/assembled_core/signals/meta_model.py:541` | `if strict: raise ...; logger.warning(msg); return False` then `model = joblib.load(path)`; meta_model calls `safe_load_model(path, strict=False)` | [verified-at-source] |
| SEC-4 | **Info leakage in error responses** — multiple endpoints return raw `str(exc)` / `str(e)` in HTTP `detail`, exposing internal paths, column names, and exception internals to any caller | `oms.py:100,168`; `paper_trading.py:342,357,381,398,421`; `risk.py:118`; `orders.py:42,44`; `diagnostics.py:182,286` | e.g. `raise HTTPException(status_code=500, detail=f"Error retrieving blotter: {e}")` | [verified-at-source] |
| SEC-5 | **Look-ahead / MNPI-shaped risk** when `as_of=None`: insider & congress feature builders skip PIT disclosure filtering entirely, and `ensure_event_schema` derives `disclosure_date` from event `timestamp` with **zero added latency** | `features/insider_features.py:88-97`; `features/congress_features.py` (same pattern); `data/latency.py:53-57` | `if as_of is not None: events = filter_events_as_of(...)` (else: no filter); latency.py: `# Derive disclosure_date from timestamp (no extra latency).` | [verified-at-source] |

**Detail (SEC-2):** `load_ledger_state` (`ops/paper_ledger.py:29-60`) does `read_text` +
`json.loads` on the caller-supplied path (and on `.1/.2/.3` backup siblings). There is
no `_is_safe_output_dir`-style guard like the one correctly present in `health.py:34`.
Impact is bounded: only JSON-dict-parseable files yield content, and the loader catches
parse errors (returns `no_ledger`), so it is **not** a raw arbitrary-file-read of
non-JSON content. But it is an unauthenticated **file-existence oracle** + partial
content disclosure for any JSON file the process can read, plus the path is echoed into
a WARNING log. The endpoint is a GET (no `require_api_key`).

**Detail (SEC-3):** Models live in user-writable `output/models/` (noted in the source
comment at `meta_model.py:536`). The R5 fix correctly routes through `safe_load_model`,
but because the default is `strict=False`, a swapped/tampered `.joblib` only produces a
log warning before unpickling — pickle deserialization is arbitrary-code-execution by
construction. Requires local write access to `output/models/`, so not remotely
exploitable, but the guard does not actually *block* the documented attack surface it
claims to defend. Recommend `strict=True` on the live/paper load path.

**Detail (SEC-5):** EDGAR ingestion itself is PIT-correct — it keys on `filed_at`, the
true public-disclosure date (`data/sources/edgar_source.py:27,57`), consistent with the
CLAUDE.md "public data only" mandate. The feature builders are PIT-aware *when called
with `as_of`* (insider T+2 default, congress T+45 default) and `filter_events_as_of`
defaults `fallback_to_event_date=False` (F-B-10 fix, `latency.py:154-158`) — all good.
The residual defect: PIT enforcement is **opt-in via the `as_of` argument**. A caller
passing `as_of=None` (the default) gets NO disclosure gating, and the
`ensure_event_schema` no-latency derivation path means a caller that lets it create
`disclosure_date` would treat trade-date as disclosure-date — effectively acting on
insider/congress trades before they are public (an MNPI-shaped advantage). Per project
memory, insider/congress data pipelines are **archived / not active in the hot path**,
so this is currently latent, not live — but it is a foot-gun that should be closed by
making `as_of` mandatory (or defaulting to a no-look-ahead cutoff) in these builders.

---

## NIEDRIG

| ID | Title | File:Line | Evidence | Status |
|----|-------|-----------|----------|--------|
| SEC-7 | `/ready` kill-switch detail reads a non-existent `state` key — `get_kill_switch_state()` returns `engaged`/`throttle_pct`, never `state`, so health detail is always `"state=unknown"` (cosmetic, misleading ops signal) | `api/routers/health.py:133`; cf. `execution/kill_switch.py:380-389` | `"detail": f"state={_ks.get('state', 'unknown')}"` vs returned dict keys `engaged/throttle_pct/sources/persistent` | [verified-at-source] |
| SEC-8 | API audit middleware reads `await request.body()` then relies on Starlette body-caching for re-injection; if a command endpoint's body is consumed here and not re-cached, POST handlers could see an empty body. Off by default (`ASSEMBLED_API_AUDIT`), so low risk, but unverified at runtime | `api/middleware.py:96-102` | `body = await request.body() ... # Re-inject so the endpoint can still consume the body.` (comment-only; no explicit re-injection) | [pattern-matched, not runtime-traced] |
| SEC-9 | No CORS middleware configured. This is the *safe* default (no cross-origin headers emitted), recorded here only so a future "add CORS" change is reviewed for `allow_origins=["*"]` regressions | `api/app.py` / `api/middleware.py` (no `CORSMiddleware`) | grep for `CORSMiddleware|allow_origins` → no matches in `src/` | [verified-at-source] |

---

## Positive confirmations (correctly secured)

| ID | What is correct | File:Line |
|----|-----------------|-----------|
| SEC-P1 | **No tracked live-secret file.** `git ls-files` matches only `.env.example`, `configs/env/*.template`, `configs/secrets/README.md`, `.secrets.baseline`. `.env.example` contains placeholders (`your_*_here`) only; `OPERATOR_KILL_TOKEN=` is blank. | `.env.example:1-219` |
| SEC-P2 | `.gitignore` excludes `.env`, `.env.*`, `.env.local/dev/prod`. Plus a `secrets-scan.yml` workflow + `.secrets.baseline` + `detect_secrets_baseline_diff.py` exist. | `.gitignore:37-47` |
| SEC-P3 | **API-key rotator never persists or logs full keys** — serializes only last-4 suffix to disk; all log lines show `…XXXX`. Exemplary secret hygiene. | `utils/api_key_rotator.py:141-152, 308-323` |
| SEC-P4 | **No secret VALUES ever logged.** All ~12 secret-adjacent log lines emit only env-var *names* ("X not set") or last-4 suffixes. | `data/sources/polygon_source.py:174`, `fred_source.py:162`, `ops/alerting.py:126,146`, etc. |
| SEC-P5 | **secrets_loader** uses OS keychain + env fallback; never logs values (only `env:key` names), uses `getpass` for interactive entry. | `config/secrets_loader.py:34-106` |
| SEC-P6 | **Kill-switch deactivation is FAIL-CLOSED**: missing `OPERATOR_KILL_TOKEN` → `PermissionError` + `REJECT_DEACTIVATE` audit entry; token compared via `hmac.compare_digest` on bytes; API maps to HTTP 403. | `execution/kill_switch.py:314-344`; `api/app.py:144-150` |
| SEC-P7 | **Auth & operator-token comparisons are timing-safe** (`hmac.compare_digest`), not `==`. | `api/auth.py:47`; `kill_switch.py:330`; `middleware.py` hash-only audit |
| SEC-P8 | **`/health` HAS a path-traversal guard** (`_is_safe_output_dir` restricts to OUTPUT_DIR or system temp). Contrast SEC-2. | `api/routers/health.py:28-35,63-68` |
| SEC-P9 | **No dangerous primitives on untrusted input.** All `subprocess.run` calls use list-form args with hardcoded command names (git, python script paths) — no `shell=True`, no `os.system`, no `eval`/`exec`, no unsafe `yaml.load`, no `pickle.load` of network input. | `paper_track.py:68`, `daily_scheduler.py:116`, `certify/generator.py:68`, `run_manifest.py:66` |
| SEC-P10 | **EDGAR/MNPI posture is correct in principle**: public SEC Form-4 Atom feed only, descriptive User-Agent, rate-limit respected, keyed on `filed_at` (public-disclosure date). No MNPI ingestion logic. | `data/sources/edgar_source.py:1-38,57` |
| SEC-P11 | Rate-limit middleware (token-bucket per-IP) and audit middleware (sha256-hashed actor/body, fsync'd, opt-in) are present and structurally sound. | `api/middleware.py:143-204, 70-136` |
| SEC-P12 | API DoS guards: `MAX_ORDERS_PER_RESPONSE` (413) on orders; `limit < 0` (400) validation on oms/paper list endpoints. | `api/routers/orders.py:47-51`; `oms.py:59-60,121-122` |

---

## Verification status

- **Static-only.** Every finding is grounded in a quoted source line; no dynamic/runtime
  testing was performed and **no CI run confirms any of this**.
- `[verified-at-source]` = the defect/behavior is directly readable in the quoted code.
- `[pattern-matched]` = behavior inferred from code shape but not traced through the full
  call/runtime path (SEC-8 only).
- `git ls-files` secret-name scan: **clean** (no live `.env`/key/credential file tracked).
- Not examined deeply (out of time-box, candidate for round 4): broker_adapter credential
  handling end-to-end; full middleware ordering under concurrency; `configs/secrets/`
  deployment templates; the `scripts/` secret-handling surface beyond grep coverage.

---

## Recommended next steps (priority order)

1. **SEC-1:** decide the production auth contract. Either fail-closed when
   `ASSEMBLED_API_KEY` is unset *in production profile*, or gate kill-switch *activate*
   behind operator-token too. Wire `auth_is_configured()` into `/ready` as the docstring promises.
2. **SEC-3:** flip the live/paper meta-model load path to `safe_load_model(..., strict=True)`.
3. **SEC-2:** add a `_is_safe_output_dir`-equivalent guard to `ledger.py`'s `ledger_path`.
4. **SEC-5:** make `as_of` mandatory (or no-look-ahead-default) in insider/congress
   feature builders before those pipelines are ever activated in the hot path.
5. **SEC-4:** replace raw `str(exc)` in HTTP 500 details with a generic message + a
   server-side-logged request-id correlation.
