# 06e — Costs / Reporting / QA / Config Precedence (Deep Audit Round 2, Agent E)

**Datum:** 2026-05-30
**Cluster:** Cost Models + Reporting/QA/Evidence + Config Precedence + H-1 Blast-Radius
**Modus:** Read-only. Nichts verändert außer dieser Datei.
**Round-1-Vorbedingung:** H-1 (`qa/metrics.py:215-217` `compute_drawdown` nutzt GLOBAL peak statt peak-to-date) als Bug BESTÄTIGT. Diese Datei klärt die **Reichweite**.

---

## 1. H-1 Consumer-Trace — HEADLINE-VERDICT: **MATERIAL**, nicht kosmetisch

### 1.1 Der Bug (unverändert bestätigt)

`src/assembled_core/qa/metrics.py:211-218`:

```python
rolling_max = equity.expanding().max()
drawdown_series = equity - rolling_max
max_drawdown = float(drawdown_series.min())
peak_equity = float(rolling_max.max())          # ← GLOBAL peak (Zeile 215)
max_drawdown_pct = (
    float((max_drawdown / peak_equity) * 100) if peak_equity > 0 else 0.0
)
```

Mechanismus: `peak_equity = rolling_max.max()` ist der **End-Peak** der ganzen Kurve. Textbuch-MDD% teilt den absoluten DD durch den **Peak-to-Date** an der Talsohle. Für jede Kurve, die nach dem Tiefpunkt ein neues Hoch macht, ist `peak_equity > peak_at_trough` ⇒ `|max_drawdown_pct|` zu **klein** ⇒ MDD optimistisch (zu gut). Richtung des Fehlers: **immer schönfärbend**, nie pessimistisch.

### 1.2 Propagationskette

`compute_drawdown` ist die EINZIGE MDD%-Quelle der QA-Schicht. Sie speist über `compute_equity_metrics`/`compute_all_metrics` (`qa/metrics.py:526`) das Feld `PerformanceMetrics.max_drawdown_pct` (`:564`) UND `calmar_ratio` (`:532-535`, `cagr / abs(max_drawdown_pct/100)` ⇒ Calmar zusätzlich **aufgebläht**).

### 1.3 Consumer (file:line) — wer liest das FALSCHE `max_drawdown_pct`

| # | Consumer | file:line | Nutzt MDD%? | Operator-/Gate-relevant? |
|---|----------|-----------|-------------|--------------------------|
| C1 | **QA-Gate `check_max_drawdown`** (BLOCK bei < -20%, WARN < -15%) | `src/assembled_core/qa/qa_gates.py:153-155` | **JA** (`metrics.max_drawdown_pct`) | **JA — Promotionsschwelle** |
| C2 | **`evaluate_all_gates`** ruft C1 | `qa/qa_gates.py:607-611` | JA (über C1) | JA |
| C3 | **EOD-Orchestrator-Gate** | `pipeline/orchestrator.py:911-937` | JA (über C2) | JA — loggt BLOCK/WARNING |
| C4 | **Daily-QA-Report** | `reports/daily_qa_report.py:68, 608` | JA (über C2 + metrics-Dump) | **JA — Operator-Tagesreport** |
| C5 | **`_metrics_to_dict` (Orchestrator JSON)** | `pipeline/orchestrator.py:296` | JA — schreibt `max_drawdown_pct` ins Run-JSON | JA — Evidence/Report |
| C6 | **Paper-Track Performance-Panel** | `paper/paper_track.py:602, 657` | **JA** (`compute_drawdown(window_equity)` → `max_drawdown_pct` pro Fenster) | **JA — Paper-Pilot-Report** |
| C7 | **Backtest-Runner QA** | `scripts/run_backtest_strategy.py:2864, 2967, 3022, 2878` | JA (Log + JSON + `evaluate_all_gates`) | JA — Backtest-Report |
| C8 | **API `/qa` Router** | `api/routers/qa.py:237, 381` | JA | JA — Operator-API |
| C9 | **API `/monitoring` Router** | `api/routers/monitoring.py:183, 260, 172` | JA | JA — Operator-API |
| C10 | Diverse Dev-/Compare-Skripte | `scripts/dev/run_strategy_benchmark.py`, `compare_strategies_trend_vs_event.py:127`, `run_full_system_backtests.py:413`, `analyze_backtest_results.py:139` | JA | Research-Reports |

### 1.4 Was H-1 NICHT erreicht (Gegenbeweis — wichtig für die Severity-Eingrenzung)

- **Promotion-Gate-Skript** `scripts/ops/check_promotion_gate.py:97-116`: rechnet MDD **selbst** mit `np.maximum.accumulate(equity)` (Peak-to-Date, **korrekt**) — H-1 ist hier irrelevant. Der „echte" Promotionsentscheid-Gate ist also sauber.
- **Pilot-Hard-Stop / Kill-Switch** (`evaluate_pilot_v2.py:144`, `daily_pilot_review.py:131`): nutzt MDD aus eigenem Pfad (`round(100.0*mdd,3)` mit lokal berechnetem `mdd`), nicht `compute_drawdown`. UNSURE ob jeder dieser `mdd`-Werte korrekt ist (eigene Audit-Spur), aber NICHT von H-1 betroffen.
- **Risk-Controls Live-DD-Cap** `execution/risk_controls.py:279-282` nutzt `compute_drawdown_risk_level` (anderer Pfad, kein MDD%).
- **`risk/risk_metrics.py:126`** / **`qa/risk_metrics.py:113,136`** entpacken nur `max_drawdown` (absolut, Tuple-Index 1), **nicht** `max_drawdown_pct` — H-1 trifft sie nicht.

### 1.5 Verdict

**H-1 ist MATERIAL.** Der falsche `max_drawdown_pct` erreicht (a) den **harten QA-BLOCK-Gate** (`qa_gates.py:153`, Schwelle -20%), (b) den **Daily-QA-Report** des Operators (C4), (c) das **Paper-Pilot-Performance-Panel** (C6), (d) **Run-JSON / Backtest-Reports** (C5/C7) und (e) die **Operator-API** (C8/C9). Konkrete Gefahr: Eine Strategie, deren echter MDD z.B. -22% (peak-to-date) wäre, kann durch den optimistischen Nenner als z.B. -18% rapportiert werden und so den -20%-BLOCK-Gate **fälschlich passieren** — und gleichzeitig Calmar überzeichnen. Der _formale_ Promotionsentscheid (`check_promotion_gate.py`) ist zwar sauber, aber die **operativen Tagesreports, das Paper-Panel und der Pipeline-Gate** sind es nicht. Severity bleibt damit auf HIGH; „nur kosmetisch" ist widerlegt.

**Fix-Ort (read-only, nicht ausgeführt):** `qa/metrics.py:215` — Nenner muss `rolling_max` an der Talsohle sein, z.B. `(drawdown_series / rolling_max).min() * 100`. Kein Caller-Eingriff nötig, da alle Consumer den korrigierten Wert transparent übernehmen.

---

## 2. Cost-Path-Consistency

### 2.1 Es gibt VIER Kostenflächen (Doppel-/Mehrfachstruktur)

| Pfad | Datei | Modell | bps-Granularität |
|------|-------|--------|------------------|
| A) Backtest/OOS Cost-Engine | `execution/transaction_costs.py` (via `pipeline/portfolio.py:164-190` `add_cost_columns`) | commission + ADV-bucket-Spread + Vol/Participation-Slippage | **Per-Symbol, ADV-/Vol-aware** |
| B) Tier-YAML | `costs.py:186-196` + `data/cost_model_policy.py` | per-symbol Tier (`config/cost_tiers.yaml`) | per-Symbol-Tier |
| C) **Live Paper-Ledger Fill** | `ops/paper_ledger.py:180-238` `simulate_fills` | **nur** commission_bps + flat slippage_bps (Fallback `spread_w+impact_w`) | **flach, KEIN ADV/Vol** |
| D) Reconcile | `ops/reconcile.py:65-67` | nur commission_bps | flach |
| (TCA-Analyse) | `risk/transaction_costs.py` | Backtest-TCA-Report (`estimate_per_trade_cost`, default 0.5/3.0 bps) | analytisch, separat |

**`costs/transaction_costs.py` existiert NICHT.** Es gibt `execution/transaction_costs.py` und `risk/transaction_costs.py` (TCA) — zwei verschiedene Module gleichen Namens (Architektur-Geruch).

### 2.2 Konsistenz-Verdict: **NICHT konsistent zwischen Live und OOS**

- **OOS/Backtest (A)** verwendet die ADV-/Vol-aware Engine (per-Symbol differenziert).
- **Live Paper (C)** `simulate_fills` (`paper_ledger.py:188-227`) verwendet **flach** `commission_bps + slippage_bps`, **ohne** Spread-/Impact-Bucket-Logik. Es ist ein anderer Code- und Modellpfad.
- Aktuelle Kalibrierung (`configs/policy.yaml:828-833`): `commission_bps: 10.0`, `spread_w: 0.25`, `impact_w: 0.5` ⇒ Live-Pfad rechnet **10.75 bps/Trade flach**. Der OOS-Pfad rechnet per-Symbol-Tier (typ. ~1–3 bps). ⇒ **Live ist deutlich konservativer (gut), aber nicht dieselbe bps-Zahl wie OOS** — Backtest-Reports und Live-Fills sind nicht 1:1 vergleichbar.

### 2.3 0-bps-Risiko (Historie: „0 bps incident")

`paper_ledger.py:189` `commission_bps = float(cost.get("commission_bps", 0) or 0)` ⇒ **Default 0**. Wenn `_resolve_cost_cfg` (`paper_runner.py:1136-1147`) ein leeres `cost_model` liefert (weder `policy.paper_pilot.cost_model` noch `app_cfg.paper_runner.cost_model` gesetzt), läuft die Live-Simulation **lautlos bei 0 bps**. **Aktuell abgewendet**, weil `policy.yaml:828` `cost_model` gesetzt ist. Aber: kein expliziter Guard/Log, der ein leeres cost_model meldet. Wer `policy.yaml` editiert/leert, bekommt stillschweigend kostenlose Fills. → Finding R-04.

### 2.4 Broad-`except` Cost-Fallback (Round-1-Trap bestätigt)

`execution/transaction_costs.py`:
- Spread-Merge-Block `:179-191`, `except Exception as _exc:` `:192-197` → fällt auf `fallback_spread_bps` zurück, nur `logger.warning`.
- Slippage-Merge-Block `:220-241`, `except Exception as _exc:` `:242-247` → `fallback_slippage_bps`.

Der Merge `trades.merge(adv_df, on=["timestamp","symbol"], how="left")` (`:179`, `:220`) ist tz-/dtype-empfindlich: wenn `timestamp`-dtypes (tz-naive vs tz-aware) zwischen `trades` und `adv_df` divergieren, wirft pandas → `except` fängt **breit** → Kosten degradieren still auf Fallback-bps statt ADV-aware. Round-1-Zeilen bestätigt (`:179-229` Bereich, `except` bei `:192` und `:242`). Severity MEDIUM: degradiert (kostenniedriger oder -höher je nach Fallback), nur WARN-Log, kein Block. → Finding R-05.

---

## 3. Config-Precedence-Findings

### 3.1 `_resolve_active_strategy` (M-7 aus Round 1)

`ops/paper_runner.py:1115-1133`:

```python
pol_name = str((policy.get("paper_pilot") or {}).get("active_strategy") or "").strip().lower()
if pol_name and pol_name != "none":
    if pol_name != app_name:
        log.info("[paper_runner] active_strategy overridden by policy: %r → %r", app_name, pol_name)
    return pol_name
return app_name
```

**Verdict: DOKUMENTIERT + GELOGGT, kein stiller Footgun.** Präzedenz `policy.paper_pilot.active_strategy > app_cfg.strategy.name` ist (a) im Docstring `:1116` benannt, (b) in `policy.yaml:820-822` kommentiert (`active_strategy: trend_baseline`), (c) **wird bei Divergenz geloggt** (`:1127-1131` `log.info`). Restrisiko: es ist nur `log.info` (kann in Log-Flut untergehen), und die Präzedenz ist „policy gewinnt" — d.h. der Operator, der den Test/`app_cfg` auf Strategie X stellt, fährt real Strategie Y aus `policy.yaml`, falls er den policy-Key übersieht. Severity LOW→MEDIUM (Observability-, kein Korrektheits-Bug). Konsequenz für M-7: Der Test-Override-Effekt ist **beabsichtigt und sichtbar**, nicht versteckt.

### 3.2 `_resolve_cost_cfg` (analoge Präzedenz)

`ops/paper_runner.py:1136-1147`: `policy.paper_pilot.cost_model > app_cfg.paper_runner.cost_model`, ebenfalls **geloggt** (`:1143-1145`) — aber nur „loaded from policy", **ohne** zu loggen, dass damit der app_cfg-Wert verworfen wurde, und **ohne** Log, wenn BEIDE leer sind (→ 0-bps, siehe R-04). Severity LOW.

### 3.3 Kein weiteres stilles Override gefunden

Keine andere `policy.get(...)`-Stelle überschreibt ein explizites Caller-Argument ohne Log im untersuchten Cost-/Report-/QA-Scope.

---

## 4. Report/QA/Evidence-Integrity — Findings

| ID | Modul:Zeile | Fund | Snippet | Schwere | betrifft |
|----|-------------|------|---------|---------|----------|
| **R-01** | `qa/metrics.py:215-217` | **H-1 MATERIAL**: MDD% über GLOBAL peak ⇒ MDD optimistisch, Calmar aufgebläht. Erreicht QA-BLOCK-Gate, Daily-QA-Report, Paper-Panel, Run-JSON, API. | `peak_equity = float(rolling_max.max())` | **HIGH** | qa_gates BLOCK, daily_qa_report, paper_track, orchestrator JSON, API |
| **R-02** | `qa/qa_gates.py:153-155` | Harter BLOCK-Gate (-20%) konsumiert R-01-Wert direkt ⇒ kann fälschlich passieren lassen. | `max_dd = metrics.max_drawdown_pct; if max_dd < max_dd_pct_limit:` | **HIGH** | Promotions-/QA-Schwelle |
| **R-03** | `paper/paper_track.py:602,657` | Rolling Performance-Panel schreibt R-01-MDD% je Fenster in Operator-Paper-Report. | `_, _, max_dd_pct, _ = compute_drawdown(window_equity)` | **HIGH** | Paper-Pilot-Report |
| **R-04** | `ops/paper_ledger.py:189` | Live-Fill commission default **0**; leeres `cost_model` ⇒ lautlose 0-bps-Fills. Aktuell durch policy.yaml abgewendet, kein Guard/Log. | `commission_bps = float(cost.get("commission_bps", 0) or 0)` | **MEDIUM** | Live Paper Cost-Realismus |
| **R-05** | `execution/transaction_costs.py:179-197, 220-247` | Breite `except Exception` um tz/dtype-empfindlichen ADV-Merge ⇒ stille Kosten-Degradation auf Fallback-bps, nur WARN. | `except Exception as _exc: logger.warning("[TC] spread calc failed, using fallback...")` | **MEDIUM** | OOS/Backtest Cost-Genauigkeit |
| **R-06** | Live `paper_ledger.py:180-238` vs OOS `pipeline/portfolio.py:164` + `execution/transaction_costs.py` | Cost-Modell-Inkonsistenz: Live = flach 10.75 bps; OOS = per-Symbol ADV/Vol-Tier (~1–3 bps). Backtest- und Live-Fills nicht 1:1 vergleichbar. | policy.yaml:828-833 `commission_bps: 10.0` vs `costs.py:195` Tier-bps | **MEDIUM** | Vergleichbarkeit Backtest↔Live |
| **R-07** | `ops/paper_runner.py:1115-1133` | active_strategy-Präzedenz (policy > app_cfg) nur `log.info`; korrekt + dokumentiert, aber Operator kann Override übersehen. | `log.info("...active_strategy overridden by policy: %r → %r"...)` | **LOW** | Strategiewahl-Transparenz (M-7) |
| **R-08** | `ops/paper_runner.py:1136-1147` | cost_model-Präzedenz geloggt, aber kein Log bei „beide leer" (→ R-04). | `if pol_cost: log.info("...cost_model loaded from policy")` | **LOW** | Cost-Config-Transparenz |
| R-09 | `accounting/accounting_report.py`, `evidence_pack.py`, `evidence_index.py` | **KEIN** Befund: echte Writer, sha256-Hash + `_validate_manifest_consistency` (`evidence_pack.py:54-60`), keine Stubs/No-Ops/hardcodierten Metriken. | `MANIFEST_FILE_KEYS = ("path","sha256","size_bytes","source_type")` | OK | — |
| R-10 | `scripts/ops/check_promotion_gate.py:97-116` | **KEIN** Befund (positiv): rechnet MDD korrekt peak-to-date selbst ⇒ H-1-immun. | `rm = np.maximum.accumulate(equity); dd = float((equity/rm - 1.0).min())` | OK | — |

### Notiz zu Sharpe/CAGR/Hit-Rate

`compute_sharpe_ratio`/`compute_cagr` (`qa/metrics.py`) wurden gegen R-01 gegengeprüft — sie nutzen **nicht** `compute_drawdown` und sind von H-1 nicht betroffen. CAGR-Edge-Case (`<1 Jahr → None`, `:244`) ist sauber. `calmar_ratio` ist die einzige _abgeleitete_ Metrik, die H-1 erbt (über `max_drawdown_pct`, `:534`). Keine hardcodierten Sharpe/CAGR/Hit-Rate-Werte gefunden.

---

**Geändert: nichts** (nur diese Datei geschrieben). Alle Befunde read-only verifiziert; keine Ausführung. R-04 (0-bps live) und R-05 (except-Fallback) als UNSURE-frei bestätigt über Code; die exakte bps-Differenz in R-06 ist konfig-abhängig und hier mit policy.yaml-Stand 2026-05-30 belegt.
