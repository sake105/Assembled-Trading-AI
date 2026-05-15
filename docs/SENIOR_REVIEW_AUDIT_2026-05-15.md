# Senior-Code-Reviewer System-Audit — 2026-05-15

**Auftrag:** Unabhängige Audit aller Hauptmodule von Assembled-Trading-AI durch den Senior-Code-Reviewer.

**Status:** Audit durchgeführt am 2026-05-15 via Simulation (siehe Methodik §2). Findings sind real, statisch-analytisch; runtime-Bestätigung folgt im Fix-Pfad.

**Befundübersicht:**

| Severity | Anzahl | Sofortmaßnahme |
|---|---|---|
| **BLOCKER** | 4 | Vor nächstem Pilot-Tag fixen |
| **MAJOR** | 14 | Innerhalb 1–2 Wochen, vor Live-Übergang |
| **MINOR** | 21 | Backlog, Sweep-fähig |
| **INFO** | 9 | Dokumentationswert, optional |
| **TOTAL** | 48 | |

Verteilung über Batches: A (Sensitive Zones) 10, B (Decision+Data) 20, C (Infrastructure+Cross-cutting) 23.

---

## 1. Executive Summary

Die unabhängige Code-Audit hat **48 Findings** identifiziert. Vier sind echte BLOCKER, die vor weiterer Pilot-Aktivität zu adressieren sind:

1. **`paper_ledger.apply_fills_to_ledger`** verschluckt SELLs ohne Long-Position lautlos (kein Cash-Credit, keine Short-Position, keine Fehlermeldung). Bei jedem Live-/Paper-Pfad, der Shorts erzeugt, droht stille State-Divergenz zu Broker. — Sensitive Zone, Rule 30.
2. **`multifactor_v2._compute_geo_risk_composite`** Fallback-Pfad nutzt `date.today()` für FRED GPRC-Fetch. In Backtest-Replays mit `as_of=2020` werden Daten bis HEUTE geholt → 5-Jahres-Look-Ahead-Bias.
3. **`multifactor_v2._compute_insider_cluster_factor`** Fallback-Pfad nutzt `cluster_buy_score` mit `date.today()` → identischer Forward-Leak via SEC Form-4-Fetch.
4. **`multifactor_v2._compute_buyback_drift_factor`** Fallback nutzt `buyback_signal_score` mit `date.today()` → identischer Forward-Leak via 8-K-Fetch.

BLOCKER 2–4 sind dasselbe Muster (Live-Fetch-Fallback ignoriert `as_of`) in drei Faktoren. Ein gemeinsamer Fix-Pattern adressiert alle drei.

**Positive Befunde (keine erneute Verletzung):**
- E-001 (`Series.where` Alignment), E-009 (`Series.any()` NaN), E-010 (`idxmax` empty), E-013 (`next(iter())`) sind in den sensiblen Zonen NICHT neu aufgetreten — frühere Wave-Fixes haben gehalten.
- `data/universe.py` PIT-Membership, `data/latency.py` `apply_source_latency` Shift-Richtung, `events/news/dedupe.py` None-Fingerprint, `signals/meta_model.py` PurgedKFold Embargo — alle sauber.
- ML-Loader (`signals/meta_model`, `ml/model_registry`, `strategies/multifactor_v2` joblib.load) wrappen alle in try/except mit Hash-Verifikation — E-015 ist überall in Batch B sauber.

**Negative Cross-Cutting-Beobachtungen:**
- **PIT-Forward-Leak via Live-Fetch-Fallback** ist das dominante Failure-Pattern in Batch B. Mindestens 6 weitere Funktionen (F-B-4 bis F-B-6, F-B-11) zeigen das identische Muster (FRED-Call ohne `observation_end`, dann `iloc[-1]`).
- **`as_of or pd.Timestamp.now()` Defensive-Defaults** maskieren latente PIT-Regressions (4× in `multifactor_v2.py`).
- **`date.today()` Sweep nicht vollständig durchgeführt:** Tournament-Iteration cf7e36e patchte einen Site; **5 weitere** sind in Batch C noch offen (`pdt.py`, `elster.py`, `post_trade_analyzer.py`, `daily_scheduler.py`, `drift_monitor.py`).
- **Hexagonal-Architektur-Scaffold ist ~5% gebaut**, nicht "active" wie das Memory suggeriert. Ports + 4 Adapter + 1 Use-Case + 4 leere Domain-Subpackages. Produktionspfade laufen weiter über `pipeline/`/`execution/`/`risk/`.
- **Import-Prefix-Inkonsistenz:** 341 Dateien `from src.assembled_core...`, 21 mit korrektem `from assembled_core...`. Funktioniert heute durch sys.path-Glück, bricht beim Wheel-Install.

---

## 2. Methodik

### 2.1 Reviewer-Setup

Drei parallele `senior-code-reviewer`-Instanzen (Opus 4.7) gegen drei Domain-Batches:
- **Batch A** (Sensitive Zones, Rule 30): `execution/`, `risk/`, `pipeline/`, `accounting/`, `paper/`, `portfolio/`
- **Batch B** (Decision + Data): `signals/`, `strategies/`, `ml/`, `data/`, `features/`, `dataquality/`, `events/`
- **Batch C** (Infrastructure + Cross-Cutting): `intel/`, `attribution/`, `qa/`, `reports/`, `api/`, `ops/`, `compliance/`, `certify/`, `adapters/`, `application/`, `bootstrap/`, `config/`, `domain/`, `ports/`, `strategy/`, `time/`, `utils/`, `experiments/`, root files

### 2.2 Bekannte Limitationen (ehrliche Disclosure)

1. **Simulation statt registrierter Subagent.** `senior-code-reviewer` wurde am 2026-05-14 (Commit `cf41fa6`) registriert. Claude Code übernimmt Subagents nur beim Session-Start, daher in dieser Session nur via `general-purpose` mit Opus + verbatim Agent-Prompt erreichbar. Funktional gleichwertig, Registry-Binding fehlt.
2. **Statisch-analytische Audit.** Keine Tests ausgeführt, keine dynamische Verifikation. BLOCKER F-A-1 (Shorts) wurde als "BLOCKER pending confirmation" markiert vom Reviewer, weil Test-Suite-Verifikation nicht stattfand. Drei Möglichkeiten: (a) echter BLOCKER falls Shorts irgendwo fillbar, (b) MAJOR falls Shorts upstream explizit gerejectet werden, (c) toter Code falls nie erreichbar.
3. **Selektiver Lese-Pfad.** ~78 von 493 .py-Dateien wurden in Tiefe gelesen; der Rest wurde via Grep-Scan auf bekannte Anti-Patterns durchsucht. Tiefer-Lese-Tail wurde priorisiert nach Sensitivität + State-Ownership.
4. **Out of scope:** `tests/**`, `.claude/hooks/**`, `.claude/agents/**`, `scripts/**`, `.github/workflows/**`. Review-Chain-Code wurde im Bootstrap-Run separat geprüft (siehe Commit `dcdbe7e`).
5. **Token-Budget gesetzt:** Jeder Reviewer wurde auf ≤35–40 Tool-Calls gecapt, um den vorherigen Rate-Limit-Hit zu vermeiden. Hat einen Trade-off: Tiefe pro Batch geringfügig reduziert, aber Coverage erhalten.
6. **Anti-Pattern-Referenz:** 18 bekannte Anti-Patterns aus `docs/CLAUDE_CODING_ERRORS.md` (E-001..E-018) wurden gegen alle Greps abgeglichen.

### 2.3 Findings-Schema

Strukturiert nach `docs/superpowers/specs/2026-05-14-review-chain-design.md` §5:

```yaml
- id: F-<batch>-<n>
  file: ...
  line: ...
  severity: BLOCKER | MAJOR | MINOR | INFO
  category: bug | wiring | completeness | correctness | pit | risk | anti-pattern | architecture
  evidence: ...
  suggested_fix: ...
  references: [docs/CLAUDE_CODING_ERRORS.md#E-NNN]
```

---

## 3. BLOCKER-Findings (4)

### F-A-1 — `paper_ledger.apply_fills_to_ledger` verschluckt SELLs auf Zero-Position

**Datei:** `src/assembled_core/ops/paper_ledger.py:276-281`
**Kategorie:** bug
**Sensitivität:** Rule 30 (paper/accounting)

**Evidence:**

```python
# SELL-Zweig (Auszug, Zeilen ~276-281)
new_qty = pos_qty - qty       # pos_qty=0, qty=100 -> new_qty=-100
if new_qty <= 0:
    sell_qty = min(qty, pos_qty)   # min(100, 0) = 0
    out["positions"].pop(symbol, None)
    _cash_d += Decimal(str(sell_qty)) * Decimal(str(price))  # = 0
```

Ein SELL auf ein Symbol ohne Long-Position (Short oder Oversell) erzeugt:
- **Null Cash-Gutschrift** (sell_qty=0).
- **Keine Short-Position** (out["positions"] wird pop'd, nicht negativ gesetzt).
- **Keine Fehler-Behandlung** (silent drop).

Genutzt von: `paper_runner`, `paper_ledger`-Flows, `broker_execution`-Ledger-Update-Pfad. Jede Kette, die Shorts produziert, divergiert lautlos zum Broker.

**Status BLOCKER pending confirmation:** Falls Shorts upstream explizit verboten sind und dieser Branch nie erreicht wird → MAJOR (toter Code mit Sicherheitsgaranke). Falls Shorts irgendwann live werden → echter BLOCKER.

**Suggested Fix:**

Vier Fälle explizit unterscheiden:
1. Closing reduction (0 < new_qty < pos_qty): qty × price gutschreiben, avg behalten.
2. Full close (new_qty == 0): qty × price gutschreiben, position poppen.
3. Flip-or-short (pos_qty ≥ 0, new_qty < 0): pos_qty × price für Long-Anteil, dann Short eröffnen über abs(new_qty).
4. Add-to-short (pos_qty < 0): negative qty erhöhen, weighted short avg, qty × price gutschreiben.

ODER (falls Shorts intentionally rejected): Fill explizit zurückweisen + Error loggen statt silent drop.

**Verifikationsschritt:** Targeted unit test mit `apply_fills_to_ledger` und einer SELL-Fill auf ein Symbol mit pos_qty=0. Assert: cash_credit > 0 ODER explicit reject error.

---

### F-B-1 — `_compute_geo_risk_composite` Fallback-Pfad: 5-Jahres-Forward-Leak

**Datei:** `src/assembled_core/strategies/multifactor_v2.py:808-888`
**Kategorie:** pit
**Sensitivität:** kritisch für Backtest-Realismus

**Evidence:**

Die Funktion hat **kein `as_of`-Parameter**. Path-2 (Live-Fetch-Fallback, Zeilen 853-883) verwendet:

```python
# Zeilen 861, 863
end = _dt.date.today()
start = ... # 5y window from today
_gpr = fred.get_series("GPRC", observation_start=start, observation_end=end)
_gpr_vals = _gpr.values
# Zeile 876
last_val = _gpr_vals[-1]  # iloc[-1] auf today-anchored data
```

In einem Backtest-Replay mit `as_of=2020-01-01` zieht dieser Pfad GPR-Daten bis **2026-05-15** und nimmt `iloc[-1]` → 5-Jahres-Forward-Leak.

Path-1 (pre-merged `gpr_index`-Spalte aus Panel) ist PIT-sicher. Path-2 ist der stille Fallback, der triggert wenn der Panel die Spalte nicht enthält.

**Verifikationsschritt:** In einem 2020-Backtest die Anzahl Symbol/Bar-Tuples zählen, in denen Path-2 statt Path-1 fired. Hoch wenn `gpr_index` nicht systematisch in Panel ist.

**Suggested Fix:**

```python
def _compute_geo_risk_composite(symbols, latest, as_of: pd.Timestamp | None = None):
    # ...
    if as_of is not None:
        end = as_of.date()
        start = (as_of - pd.DateOffset(years=5)).date()
        _gpr = fred.get_series("GPRC", observation_start=start, observation_end=end)
        _gpr_vals = _gpr[_gpr.index <= as_of]
        if _gpr_vals.empty:
            return 0.0  # graceful degradation
        last_val = _gpr_vals.iloc[-1]
    else:
        raise ValueError("as_of required for PIT-safe geo_risk_composite")
```

Call-Site `multifactor_v2.py:1290` muss `_bar_as_of` durchreichen.

**References:** E-012, CLAUDE.md §7.2 PIT.

---

### F-B-2 — `_compute_insider_cluster_factor` Fallback: Forward-Leak via SEC Form-4

**Datei:** `src/assembled_core/strategies/multifactor_v2.py:891-934` + `src/assembled_core/signals/insider_cluster.py:67-68`
**Kategorie:** pit

**Evidence:**

`_compute_insider_cluster_factor` hat kein `as_of`. Fallback (Zeile 926) ruft `cluster_buy_score(sym, days=30)`, das intern verwendet:

```python
# insider_cluster.py:67-68
start = date.today() - timedelta(days=days)
end = date.today()
filings = edgar.get_filings(form="4", date_range=(start, end))
```

In einem 2020-Backtest fetcht das SEC Form-4-Filings aus 2026-Windows.

**Call-Site:** `multifactor_v2.py:1300` (Fallback ohne `as_of`-Pass-Through).

**Suggested Fix:**

`as_of` durch Funktionssignatur threading, an `cluster_buy_score(sym, as_of=as_of, days=30)` durchreichen, dort an `edgar.get_filings(date_range=(start_from_as_of, as_of))`.

Oder Live-Fetch hinter `allow_live_fetch=False` Default-False im Backtest gaten.

**References:** E-012, CLAUDE.md §7.2.

---

### F-B-3 — `_compute_buyback_drift_factor` Fallback: Forward-Leak via 8-K

**Datei:** `src/assembled_core/strategies/multifactor_v2.py:982-1019` + `src/assembled_core/signals/buyback_drift.py:83-84`
**Kategorie:** pit

**Evidence:**

Identischer Pattern wie F-B-2. `buyback_signal_score(sym)` nutzt `date.today()` für 8-K-Fetch.

**Call-Site:** `multifactor_v2.py:1320`.

**Suggested Fix:**

Wie F-B-2: `as_of` durch alle Layer threaden bis zur EDGAR-Anfrage.

**References:** E-012.

**Gemeinsamer Fix-Pattern für F-B-1, F-B-2, F-B-3:**

Alle drei BLOCKER teilen das gleiche Anti-Pattern: Live-Fetch-Fallback ignoriert `as_of`. Ein einziger Refactor-Pass kann alle drei adressieren:

1. Identifiziere alle Funktionen mit Live-Fetch-Fallback (grep für `date.today()` + `fred.get_series` / `edgar.get_filings` / `yf.download` in `signals/` und `strategies/`).
2. Füge `as_of: pd.Timestamp | None` zu jeder Funktionssignatur hinzu.
3. Wenn `as_of` None: `raise ValueError(...)` im Backtest-Mode (oder default to today() im Live-Mode mit explizitem Flag).
4. Pass `observation_end` an FRED, `date_range` an EDGAR, slice Series mit `≤ as_of` vor `iloc[-1]`.

---

## 4. MAJOR-Findings (14)

### Batch A (Sensitive Zones)

#### F-A-2 — `risk_controls.py` Crisis-Alpha-Fallback liest falsche Kill-Switch-Keys

**Datei:** `src/assembled_core/execution/risk_controls.py:251-253`
**Kategorie:** bug

```python
_ks = get_kill_switch_state()
if _ks.get("active", False) and _ks.get("reason", "").startswith("crisis_alpha"):
```

`get_kill_switch_state()` liefert key `"engaged"` (nicht `"active"`), und `"reason"` liegt unter `_ks["persistent"]["reason"]` (nicht top-level). Der Fallback **kann nie feuern** → Crisis-Alpha-PAUSE failt silent open wenn `crisis_alpha_ctx` nicht übergeben wird.

**Fix:** `_ks.get("engaged", False)` + `_ks.get("persistent", {}).get("reason", "")`.

#### F-A-3 — `intel_context.py` sector_rotation_scores: `iloc[-1]` ohne Sort

**Datei:** `src/assembled_core/paper/intel_context.py:207-212`
**Kategorie:** pit

`cut.iloc[-1]` auf unsortierter DataFrame. Pandas-Merges geben rows oft out-of-order zurück → falsches Tagesergebnis ins ctx.

**Fix:** `sort_values(ts_col, kind="mergesort")` vor `iloc[-1]`. References: E-004.

#### F-A-4 — `_tc_features.py` regime_state: Sort-Garantie nicht enforced

**Datei:** `src/assembled_core/pipeline/_tc_features.py:193`
**Kategorie:** correctness

`hmm_df.iloc[-1].get("regime_label", "sideways")` ohne Sort-Garantie. Falsches Regime kontaminiert Sizing + Risk-Multipliers.

**Fix:** Sort defensiv im Call-Site oder assert sort-by-date im `build_regime_state_hmm`. References: E-004.

#### F-A-9 — `unified_paper_engine.py` Corporate-Actions: Exact-Equality auf tz-aware Timestamps

**Datei:** `src/assembled_core/execution/unified_paper_engine.py:1779-1783`
**Kategorie:** bug

`actions[actions.get("effective_date") == as_of_ts]` ist exact-equality. Wenn `effective_date` einen Time-Component hat (alles außer Mitternacht UTC), gibt der Filter 0 Rows zurück → Splits/Dividenden für den Tag werden silent geskippt.

**Fix:** `.dt.normalize()` auf beiden Seiten vor Vergleich + audit-log der match-count.

### Batch B (Decision + Data)

#### F-B-4 — `macro_regime_quadrant.py` Forward-Leak

**Datei:** `src/assembled_core/features/macro_regime_quadrant.py:92-115`
**Kategorie:** pit

`current_quadrant_from_fred()` fetcht MANEMP/PAYEMS/CPI/T5YIFR ohne `observation_end`, dann `iloc[-1]`. Im Backtest-Kontext: Forward-Leak.

**Fix:** `as_of`-Parameter + `observation_end=as_of` + slice `series.index ≤ as_of`. References: E-012.

#### F-B-5 — `recession_probability.py` Forward-Leak

**Datei:** `src/assembled_core/signals/recession_probability.py:96-114`
**Kategorie:** pit

T10Y3M und NFCI ohne `observation_end`, dann `iloc[-1]`. Same Pattern.

**Fix:** wie F-B-4. References: E-012.

#### F-B-6 — `sentiment_panel.py` Forward-Leak

**Datei:** `src/assembled_core/signals/sentiment_panel.py:96-128`
**Kategorie:** pit

VIXCLS/BAMLH0A0HYM2/UMCSENT ohne `observation_end`. Same Pattern.

**Fix:** wie F-B-4. References: E-012.

#### F-B-9 — `corporate_actions.py` apply_delisting_exits: Falsches Fallback bei `before.empty`

**Datei:** `src/assembled_core/data/corporate_actions.py:412-416`
**Kategorie:** pit

Wenn keine Preise vor Delisting-Datum gefunden, Fallback auf `sym_prices["close"].iloc[-1]` — das ist per Definition NACH dem Delisting (Bid-Ask-Collapse). Unrealistischer Exit-Preis.

**Fix:** Raise oder WARN+skip statt `iloc[-1]`. References: §6 sensitive zones.

#### F-B-10 — `data/latency.py` `filter_events_as_of` Fallback-Default falsch

**Datei:** `src/assembled_core/data/latency.py:161-186`
**Kategorie:** pit

`fallback_to_event_date=True` ist Default. Wenn `disclosure_col` fehlt, fällt das Filter auf `event_date` zurück. `event_date` ist der Event-Zeitpunkt, **nicht** der Disclosure-Zeitpunkt → Look-Ahead, wenn Vendor `event_date == effective_date` rapportiert.

**Fix:** `fallback_to_event_date=False` Default + opt-in mit WARN-Log. References: E-002, CLAUDE.md §7.2.

#### F-B-11 — `pead_sue.py` `pre_trade_earnings_check` Backtest-Kontamination

**Datei:** `src/assembled_core/signals/pead_sue.py:81-112`
**Kategorie:** pit

Nutzt `date.today()` für delta-zu-earnings. Funktionsname „pre_trade" → live correct. ABER: wird auch aus Backtest-Reconciliation aufgerufen (gleicher Name in `signals/__init__.py`). Kein `as_of`, kein Mode-Flag.

**Fix:** `today` als optionaler Parameter; Backtest-Caller müssen `today=as_of.date()` übergeben. ODER rename zu `_live`. References: E-012.

#### F-B-12 — `events/store.py` EventStore.append: Silent Drop bei SQLite-Fehler

**Datei:** `src/assembled_core/events/store.py:88-89`
**Kategorie:** correctness

Catch `sqlite3.Error` + nur `logger.error`. Bei DB-Lock, Schema-Mismatch oder Disk-Full werden Events silent gedroppt. Docstring verspricht "append-only" — silent drop verletzt das.

**Fix:** Re-raise oder `EventAppendError` werfen. Optional: dead-letter-File. Distinguish duplicate-sequence (legitime IGNORE) von DB-Error via `cursor.rowcount`. References: E-003.

### Batch C (Infrastructure + Cross-cutting)

#### F-C-1 — API Paper-Trading Risk-Filter: Cardinality-Loss bei Duplikat-Orders

**Datei:** `src/assembled_core/api/routers/paper_trading.py:160`
**Kategorie:** bug

```python
filtered_set = set(zip(symbol, side, qty, price))
```

Bei zwei identischen Paper-Buys (gleicher symbol/side/qty/price) kollabiert das Set die Duplikate → zweite identische Order wird silent als rejected klassifiziert obwohl Risk-Filter sie akzeptiert hat.

**Fix:** Index-aligned matching mit stable row-id oder one-by-one Order-Verarbeitung. References: E-013-ähnlich.

#### F-C-2 — API Paper-Engine Singleton ohne Lock

**Datei:** `src/assembled_core/api/routers/paper_trading.py:41`
**Kategorie:** correctness

`_engine = PaperTradingEngine()` ist Process-wide Singleton. Submit/list/reset/positions-Handler sind sync `def` ohne Lock. Zwei concurrente POSTs racen auf engine internal state. `/reset` während in-flight orders ist unbounded.

**Fix:** `threading.Lock` um `_engine`-Access ODER async + asyncio.Lock ODER Single-Worker dokumentieren.

#### F-C-3 — Import-Prefix-Inkonsistenz: 341 Dateien falsch

**Datei:** `src/assembled_core/compliance/__init__.py:8-29` + 340 weitere
**Kategorie:** architecture

`compliance/__init__.py` nutzt bare `assembled_core...`-Prefix (korrekt), 341 weitere Dateien nutzen `src.assembled_core...` (technisch falsch, funktioniert nur durch dev/CI sys.path). Beim Wheel-Install bricht es.

**Fix:** Einheitliche Konvention via ruff isort-Rule. Recommendation: bare `assembled_core...` überall, weil pyproject.toml installed-name = `assembled_core`. References: CLAUDE.md §11.2, Rule 50.

#### F-C-4 — `compliance/pdt.py` `date.today()` nicht UTC

**Datei:** `src/assembled_core/compliance/pdt.py:46`
**Kategorie:** bug

`date.today()` ist Lokalzeit. PDT-Regel ist US-Markt-basiert → Rolling Window muss US-Eastern aligned sein. Auf Windows CET-Box rollt date.today() ~6h VOR US-Market-Close → falscher Day-Trade-Count zwischen 18:00 CET und Mitternacht. Memory sagt "PDT UTC fix (cf7e36e)" wurde angewendet — diese Site wurde nicht erfasst.

**Fix:** `pd.Timestamp.now(tz='America/New_York').date()`. References: E-012.

---

## 5. MINOR / INFO Findings (30, gruppiert nach Kategorie)

### 5.1 `date.today()`-Sweep nicht vollständig (5 weitere Sites — alle MINOR)

Tournament-Iteration cf7e36e patchte EINEN Site. Diese sind noch offen:

| Datei | Zeile | Kontext |
|---|---|---|
| `src/assembled_core/compliance/elster.py` | 148 | Elster XML ErstellungsDatum (Steuer-Doku) |
| `src/assembled_core/ops/drift_monitor.py` | 105 | Drift report_date |
| `src/assembled_core/qa/post_trade_analyzer.py` | 468 | Analysis-Output payload |
| `src/assembled_core/ops/daily_scheduler.py` | 577 | Quarter-Window-Gating (Q-Detection) |
| `src/assembled_core/compliance/pdt.py` | 46 | **MAJOR**, siehe F-C-4 |

Fix-Sweep einmal: `grep -n "date\.today\(\)" src/assembled_core/` → `datetime.now(tz=timezone.utc).date()`. References: E-012.

### 5.2 Falsy-Default `... or <literal>` (E-007)

Verbreitet in `strategies/multifactor_v*.py` und `intel/*` als `int(cfg.get(...) or N)`. Meist BENIGN bei Integer-Parametern, aber **MAJOR-Klasse** in Sizing/Pricing:

- `src/assembled_core/portfolio/liquidity_aware_sizer.py:51` (MINOR-eskalierbar zu MAJOR): `float(symbol_data.get("price") or 1)` — delisted symbol (price=0) wird auf $1 substituiert, Liq/ADV-Caps werden bei $1 berechnet, Order oversized.
- `src/assembled_core/pipeline/trading_cycle_v2.py:358` (MINOR): VIX z-score 0.0 wird wie "missing" behandelt.
- `src/assembled_core/strategies/multifactor_v2.py:202, 616, 650, 928, 965, 1013` (INFO): `as_of or pd.Timestamp.now()` — defensive Defaults, derzeit unreachable, latente Footguns.

**Cross-Cutting-Fix:** Ruff-Rule banning `or <literal>` after `.get()` in price/cash/qty contexts.

### 5.3 Silent `except Exception: pass` (E-003) — verbleibende Sites

8 Sites in `intel/` und `api/routers/`. Alle INFO/MINOR — non-critical, aber maskieren Data-Quality-Failures:

- `src/assembled_core/intel/news_archive.py:99-102` (MINOR): close() failures in JSONL archive writer — undermines WORM-ish guarantee.
- `src/assembled_core/intel/news_enricher.py:199-200` (INFO): Event taxonomy classify failure → silent None category.
- `src/assembled_core/intel/rss_fetcher.py:243-257` (INFO): 3× consecutive bare except — RSS parse failures look like empty feeds.
- `src/assembled_core/intel/news_trade_attribution.py:273-274` (INFO): Trade attribution silent drop.
- `src/assembled_core/api/routers/diagnostics.py:270-271` (INFO): PSI per-feature loop — drift report silent under-count.
- `src/assembled_core/api/routers/trades.py:109-110` (INFO): JSONL line parse — trade journal corruption silent.
- `src/assembled_core/qa/backtest_engine.py:801` (MINOR): DD-damper update silent. Risk-control → Rule 30.
- `src/assembled_core/risk/state_machine.py:144-152` (MINOR): tmp.unlink in finally — disk-problem silent.

**Cross-Cutting-Fix:** Sweep mit `logger.warning(...)` minimum.

### 5.4 Dead `tz_localize('UTC')` nach `utc=True`

3 Sites mit identischem Copy-Paste-Pattern:

- `src/assembled_core/accounting/position_engine.py:314-320` (MINOR)
- `src/assembled_core/accounting/ledger.py:189-191, 299-301` (MINOR ×2)

Auch fehlt `errors="coerce"` bei diesen `pd.to_datetime` calls (E-008).

**Fix:** Helper `to_utc(series)` in `accounting/_time.py`, dead-Branches entfernen, `errors="coerce"` ergänzen.

### 5.5 Weitere strategische MINOR/INFO

- `src/assembled_core/qa/factor_analysis.py:2123` (MINOR): einzige `pd.to_datetime` ohne `utc=True` in qa/ — inkonsistent mit Rest. References: E-008.
- `src/assembled_core/data/sources/earnings_calendar_source.py:116, 144` (MINOR): `pd.to_datetime` ohne `errors="coerce"` — yfinance-Layout-Varianten raisen. References: E-008.
- `src/assembled_core/api/routers/portfolio.py:113` (MINOR): `iloc[-1]` auf unsortierten Orders → stale price valuation. References: E-004.
- `src/assembled_core/api/auth.py:35-48` (MINOR): `require_api_key` fall-open Warning when env unset. Other write-paths (oms routes) ohne Auth-Dep. References: completeness.
- `src/assembled_core/composite_score.py:657, 660` (MINOR): `as_of_date or date.today()` Fallback. References: E-007, E-012.
- `src/assembled_core/signals/sector_rotation.py:164` (MINOR): `pd.Timestamp.now()` Fallback. References: E-012.
- `src/assembled_core/data/corporate_actions.py:64-66` (INFO): `apply_splits_for_research_prices` filtert nicht nach `as_of`.
- `src/assembled_core/qa/backtest_comparison.py:143` (INFO): Inline normal-CDF-Approximation ist die Density, nicht die Two-Tail-Probability. Falsche p-Werte wenn scipy fehlt.
- `src/assembled_core/data/feature_store.py:184, 130` (INFO/MINOR): SQL via f-string (validators schließen Injection, aber bound params wären defense-in-depth); `pd.to_datetime` ohne `errors=`.
- `src/assembled_core/strategies/multifactor_v2.py:928, 1013` (MINOR): Bare except + 0.0-fallback maskiert systematische EDGAR-Rate-Limits → ganzer Backtest kann mit zero-filled alt-data laufen ohne aggregate log.
- `src/assembled_core/events/store.py:100` (MINOR): `json.dumps(..., default=str)` ist lossy — non-roundtrippable via `json.loads`. References: E-018 / E-011 ähnlich.
- `src/assembled_core/domain/*/__init__.py` (INFO): Alle 4 Domain-Subpackages leer — hexagonal Scaffold ist ~5% gebaut.
- `src/assembled_core/experiments/__init__.py` (INFO): Quarantine intakt, aber Policy nicht dokumentiert.
- `src/assembled_core/intel/market_confirmation.py:113` (INFO): `yf.download` ohne Timeout-Counter, silent degradation.

---

## 6. Cross-Cutting-Patterns

### 6.1 PIT-Forward-Leak via Live-Fetch-Fallback (DOMINANT)

**Mindestens 8 Funktionen** zeigen das Muster:
1. Pre-merged Panel-Column (PIT-safe) ist Path 1.
2. Live-Fetch-Fallback ist Path 2, ohne `as_of`.
3. Path 2 ist silent — kein Log, kein Warning.

Konsequenz: Backtest-Korrektheit hängt davon ab, dass Path-1 für JEDEN Bar/Symbol existiert. Bei nur einem fehlenden Panel-Eintrag → 5-Jahres-Forward-Leak via FRED/EDGAR/yfinance.

**Strategischer Fix:** Globaler Backtest-Mode-Flag (`BACKTEST_MODE=True`) der ALLE Live-Fetches in Faktor-Pfaden hart blockt. Default-Verhalten: raise `BacktestLiveFetchError` statt silent default-to-zero.

### 6.2 `iloc[-1]` ohne Sort-Garantie (PIT-Klasse)

Mindestens 5 Sites identifiziert (F-A-3, F-A-4, F-A-5, F-C-10 + B). Class: "letzte Zeile = neueste" ist nur korrekt bei garantiert sortierter DataFrame.

**Strategischer Fix:** Helper `def pit_last(df, ts_col="timestamp") -> pd.Series` der explizit `sort_values + iloc[-1]` macht. Einmal definieren, überall ersetzen.

### 6.3 `date.today()`-Sweep nur partiell durchgeführt

Tournament-Iteration cf7e36e patchte EINEN Site. **5 weitere identifiziert** (siehe §5.1). PDT (F-C-4) ist MAJOR; andere MINOR.

**Strategischer Fix:** Repo-wide grep + sweep + ruff rule `flake8-datetimez` aktivieren.

### 6.4 Hexagonal-Architektur-Scaffold ≠ "aktiv"

Memory beschreibt hexagonal als "active". Reality: ports/ + 4 adapter/ + 1 use_case + 4 leere domain/ subpackages. Produktionspfade laufen weiter über `pipeline/`/`execution/`/`risk/`.

**Empfehlung:** Memory + Doku updaten: "Hexagonal layer ~5% built, scaffold only". Vorlage für graduellen Migrationspfad in `docs/HEXAGONAL_MIGRATION_PLAN.md` (existiert bereits).

### 6.5 Import-Prefix-Bug-Time-Bomb

341 Dateien nutzen `src.assembled_core...`, 21 nutzen `assembled_core...`. Funktioniert heute durch dev sys.path-Glück. Beim Wheel-Install bricht es.

**Strategischer Fix:** Ruff isort + sweep. References: F-C-3, CLAUDE.md §11.2.

### 6.6 Silent `except Exception: pass` in Audit-/Data-Quality-Pfaden

8 Sites identifiziert. Class: maskiert systematische Failures (EDGAR rate-limit, RSS parse errors, drift PSI compute, JSONL corruption). Operator sieht "0 errors" während intern alles silent fehlt.

**Strategischer Fix:** Sweep mit `logger.warning(..., exc_info=True)`. References: E-003.

### 6.7 POSITIVE: Was funktioniert

- E-001 (`Series.where` alignment) — keine neuen Sites in sensiblen Zonen.
- E-004 (`iloc[-1]` empty) — die meisten Sites haben Empty-Guards.
- E-009 (`Series.any()` NaN) — keine neuen Sites.
- E-010 (`idxmax` empty) — Sites in `strategy_allocator`, `turnover_budget` sind guarded.
- E-013 (`next(iter())`) — keine neuen Sites.
- E-015 (`joblib.load`) — alle Sites wrappen in try/except mit Hash-Verifikation.
- `data/universe.py` PIT-Membership — sauber.
- `data/latency.py` `apply_source_latency` Shift-Richtung — sauber.
- `events/news/dedupe.py` None-Fingerprint — sauber.
- `signals/meta_model.py` PurgedKFold Embargo — sauber.

---

## 7. Prioritäts-Aktionsplan

### Sofort (vor nächstem Pilot-Tag)

1. **F-A-1 verifizieren + fixen** (paper_ledger Shorts) — Targeted unit test, dann Branch korrekt implementieren oder explizit reject.
2. **F-B-1 + F-B-2 + F-B-3 gemeinsam fixen** (alt-data Live-Fetch-Forward-Leak) — `as_of`-Threading durch alle 3 Funktionen + Backtest-Mode-Flag der Live-Fetches blockt.
3. **F-A-2 fixen** (Crisis-Alpha-Fallback wrong-keys) — One-line dict-key correction + Unit-Test.

### Innerhalb 1–2 Wochen (vor Live-Übergang)

4. **F-B-4, F-B-5, F-B-6** — Drei weitere FRED-Forward-Leaks. Gleicher Fix-Pattern wie F-B-1–3.
5. **F-A-3, F-A-4, F-C-10** — `iloc[-1]`-ohne-Sort-Class. Helper `pit_last()` einführen.
6. **F-A-9** — Corporate-Actions exact-equality auf tz-aware Timestamps.
7. **F-B-10** — `filter_events_as_of` Fallback-Default umkehren.
8. **F-B-11** — `pre_trade_earnings_check` Mode-Trennung.
9. **F-B-12** — `EventStore.append` re-raise statt silent drop.
10. **F-C-1** — Cardinality-Loss in API risk-filter.
11. **F-C-2** — Engine Singleton mit Lock.
12. **F-C-4** — PDT `date.today()` UTC fix (gehört zum date.today()-Sweep §5.1).
13. **F-B-9** — `apply_delisting_exits` Fallback strenger.

### Backlog (Sweeps + Cleanup)

14. **§5.1 `date.today()`-Sweep** — 5 Sites umstellen.
15. **§5.2 Falsy-Default `or <literal>`-Sweep** — Sizing/Pricing-Sites priorisieren.
16. **§5.3 Silent `except`-Sweep** — `logger.warning` minimum.
17. **§5.4 Dead `tz_localize`-Sweep** — Helper `to_utc()` einführen.
18. **F-C-3 Import-Prefix-Sweep** — Ruff-Rule + 341-File-Update.
19. **§6.4 Hexagonal-Doku korrigieren** — Memory + Spec auf "Scaffold" downgraden.
20. **§5.5 Tail** — diverse Polish-Findings.

---

## 8. Was diese Audit NICHT abgedeckt hat

**Erste Disclosure (Limitations):**
- Keine Test-Suite ausgeführt (statisch-analytisch).
- ~78 von 493 .py-Dateien in Tiefe gelesen, Rest nur via Grep.
- Tool-Use-Cap pro Reviewer (35–40) limitierte Coverage-Tiefe.
- Simulation statt registrierter Subagent (Bootstrap-Hypokrisie addressiert in CLAUDE.md §20.7).

**Zweite Disclosure (bewusst out of scope):**
- `tests/**` — Test-Code-Qualität ist eigene Audit.
- `.claude/hooks/**`, `.claude/agents/**` — neue Review-Chain selbst-reviewed in Bootstrap-Run.
- `scripts/**`, `.github/workflows/**` — separate CI-Audit-Domäne.

**Dritte Disclosure (Tiefe-Limit pro Modul):**
- `pre_trade_checks.py` (1100+ Zeilen): Nur high-traffic gates, ADV cap, weight-per-symbol spot-checked. CVaR check + group exposure check nicht im Detail reviewed.
- `unified_paper_engine.py` (~2900 Zeilen, 40+ except-Klauseln): Spot-Check; verdient eigene fokussierte Audit.
- `broker_adapter.py` (~800 Zeilen Alpaca SDK): Nicht geöffnet.
- `qa/backtest_engine.py` (~1500 Zeilen): Nur Bereich um Zeile 801 (DD-damper).
- `attribution/brinson_multi_period`, `time_series.ks_test` fallback math: Nicht deep-dived.
- `config/env_validator.py`, `policy_schema.py`: Nicht geöffnet.

**Vierte Disclosure (Findings sind Hypothesen bis Verifikation):**
- F-A-1 (BLOCKER paper_ledger Shorts) ist "BLOCKER pending confirmation" — Unit-Test kann die Severity downgraden (auf MAJOR falls upstream-rejected) oder bestätigen.
- F-B-1..F-B-3 (BLOCKER Forward-Leaks) sind sicher real, aber Magnitude des Leaks hängt davon ab, wie oft Path-2 vs Path-1 firet in echten Backtests.

**Empfehlung:** Nach Fixes der BLOCKER eine zweite Audit-Runde (gleicher Reviewer-Pool, geschärfter Scope auf §6 Cross-Cutting-Patterns) zur Verifikation.

---

## 9. Anhang — Bekannte Anti-Patterns Referenz

Vollständige Liste in `docs/CLAUDE_CODING_ERRORS.md` (E-001..E-018). In diesem Audit referenziert:

| ID | Pattern | Sites diesem Audit |
|---|---|---|
| E-002 | PIT look-ahead via midnight normalization | F-B-9 (corporate_actions delisting), F-B-10 (latency event_date fallback) |
| E-003 | Silent `except Exception: pass` | §5.3 (8 sites), F-A-10, F-B-12, F-B-20 |
| E-004 | Empty DataFrame `.iloc[-1]` crash | F-A-3, F-A-4, F-A-5, F-C-10 (alle "unsorted" Sub-Pattern, kein empty-crash) |
| E-007 | `dict.get(key) or default` falsy bug | F-A-5, F-A-7, F-B-8, §5.2 (mehrere) |
| E-008 | `pd.to_datetime` ohne `errors='coerce'` | F-A-6, F-A-8, F-B-15, F-B-18, F-C-9 |
| E-011 | `json.dumps` numpy types | F-B-13 (ähnliches Pattern via `default=str`) |
| E-012 | `date.today()` Lokalzeit vs UTC | F-B-1, F-B-2, F-B-3, F-B-11, F-C-4, §5.1 (5 weitere) |
| E-014 | `tz_convert(None)` on tz-naive | §5.4 (3 sites mit dead-branch) |
| E-015 | `joblib.load` ohne EOFError | **Keine neuen Sites** — alle Loader properly wrapped (positiv!) |

---

**Reviewer:** simulierter `senior-code-reviewer` (Opus 4.7) via `general-purpose` Subagent  
**Datum:** 2026-05-15  
**Audit-Dauer:** ~3 parallele Reviewer-Sessions, ~600k Tokens kumulativ  
**Coverage:** 78 von 493 .py-Dateien in Tiefe + Grep-Scan über alle  
**Verifikationsstand:** statisch-analytisch, keine Tests ausgeführt
