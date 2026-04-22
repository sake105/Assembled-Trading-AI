# System-Check Remediation Plan — 2026-04-22

## Kontext

Ergebnis des 30-Agent-Audits auf HEAD `8ad5990` (post ML-Rounds 4–7 + orphan-wiring).

- phase12 lokal: 1810 passed / 0 failed / 10 skipped
- Sensible Zonen (`execution/`, risk_state_machine, pre_trade_checks): **unverändert** — kein Angriff auf Kernlogik nötig.
- 4 konkrete Code-Bugs, 1 Rule-50-Verletzung, 1 schwere Governance-Drift.

## Priorisierung

1. **P0** — Code-Bugs (ISSUE-Level, können falsche Allokation / falsche Tests produzieren)
2. **P1** — Governance-Sync (Doku lügt über Projektzustand)
3. **P2** — Architektur-Konsolidierung (Doppel-Wahrheiten)
4. **P3** — CI / Dependency-Hygiene
5. **P4** — CONCERN-Liste (Cleanup, nicht blockierend)

Jeder Punkt: **ein** Commit, **ein** Thema. Kein Vermischen (CLAUDE.md §16).

---

## P0 — Code-Bugs (ISSUE)

### P0.1 — Kelly-Uncertainty Math-Bug
- **Datei:** `src/assembled_core/portfolio/kelly_uncertainty.py:61`
- **Problem:** Code invertiert Docstring-Semantik. `relative_uncertainty ≤ 1.0` → `scale=1.0` (kein Discount), Discount greift erst bei >100% Unsicherheit.
- **Fix:** Formel gemäß Docstring: `uncertainty_scale = 1.0 - clip(relative_uncertainty, 0, 1)`.
- **NaN-Guards** für `edge` und `variance` ergänzen.
- **Clip-Entscheidung:** symmetrisch `[-max_fraction, +max_fraction]` oder long-only `[0, max_fraction]` festlegen und in Docstring fixieren. Konsistenz mit `compute_kelly_weights`.
- **Test:** Neuer Case in `tests/test_ml_round7.py`:
  - `cw == ref_cw` → scale ≈ 0 (vollständig abgewertet)
  - `cw == 0` → scale ≈ 1 (kein Abzug)
  - `cw >> ref_cw` → scale ≈ 0
  - NaN-Inputs → definiertes Verhalten (0 oder Exception)
- **Risiko:** niedrig — Modul ist Shadow-only (kein Consumer in pipeline/execution). Fix vor erster Wiring kritisch.

### P0.2 — Position-Sizing Capital-Regression
- **Datei:** `src/assembled_core/portfolio/position_sizing.py:739`
- **Problem:** `result["target_qty"] = result["target_weight"]` lässt `total_capital` weg.
- **Fix:** `target_qty = target_weight * total_capital / price` (oder gemäß Basis-Funktion `compute_target_positions` angleichen).
- **Test:** `tests/test_portfolio_position_sizing.py` erweitern — Assertion, dass `target_qty * price ≈ target_weight * total_capital`.
- **Risiko:** niedrig, Modul ebenfalls Shadow.

### P0.3 — Nested-Meta Batch-Relative Size
- **Datei:** `src/assembled_core/ml/nested_meta_labeling.py:186`
- **Problem:** `max_abs = np.abs(size_raw).max()` — Batch-relativ. Zwei Aufrufe mit verschiedenen Batches → unterschiedliche Scales für dieselbe Observation.
- **Fix-Optionen:**
  - (a) Trainings-Zeit-Skalierung speichern (`self._size_scale_max`), in `predict` anwenden.
  - (b) Fixe Normalisierung via bekannte Target-Range.
  - Option (a) bevorzugt — deterministischer.
- **Zusatz:** Walk-forward-Split in `fit()` ergänzen (aktuell In-Sample-Training).
- **Test:** Round-4-Test `test_nested_meta_fit_predict` erweitern — zwei Predict-Calls mit unterschiedlichen Batch-Sizes, gleiche Observation → gleiche `size_scale`.
- **Risiko:** mittel — Consumer ist `MLSignalPipeline` (Research-Layer, nicht live).

### P0.4 — config/__init__ `__all__` inkonsistent
- **Datei:** `src/assembled_core/config/__init__.py:50-54`
- **Problem:** `FactorBundleConfig`, `FactorConfig` im `__all__`, Import wrapped in try/except mit stillem Warn-Log. Wenn Submodul fehlt → ImportError bei Consumer.
- **Fix-Optionen:**
  - (a) Try/except entfernen → hartes Requirement.
  - (b) `__all__` konditional aufbauen: Namen nur anhängen, wenn Import gelingt.
- **Empfehlung:** (b) — konsistent mit existierendem Fallback-Pattern.
- **Test:** Import-Smoke, dass `config` auch ohne `factor_bundles`-Modul importierbar bleibt.

### P0 — Completion Criteria
- 4 Commits, jeder mit Fix + Test.
- phase12 weiter grün, neue Tests pass.
- Kein behavior change in unveränderten Call-Sites (Shadow-Module).

---

## P1 — Governance-Sync (SEVERE-DRIFT)

### P1.1 — ROADMAP_STATE.md aktualisieren
- **Datei:** `docs/roadmap/ROADMAP_STATE.md`
- **Aktueller Stand:** endet bei Ultra-Plan „polished-koala" (2026-04-18/19), 64 Commits Rückstand.
- **Nachzutragen:**
  - paper-engine-upgrade (phases 0-11)
  - news engine P1-P8 + F1-F18
  - 40-point news upgrade
  - ML Rounds 4–7
  - orphan-wiring (160 Module, 18 Domains)
  - system-map galaxy-type
  - frontend-design plugin
- **Test-Zahlen aktualisieren:** von 1259/1266 → 1810 (phase12 local).
- **„Last completed step"** neu setzen auf ML-Round-7-Hardening (cffa1bd).
- **Milestone-Queue** auf aktuelle Realität.
- **§8 Stop-condition:** `.env`-Rotation Status prüfen und konsistent setzen.
- **§6:** Verweis auf `cursor/development-environment-setup-8e96`-Branch entfernen.

### P1.2 — MEMORY.md Active Milestone aktualisieren
- **Datei:** `C:\Users\hanso\.claude\projects\F--Python-Projekt-Aktienger-st\memory\MEMORY.md`
- **Aktueller Stand:** „Active Milestone" = News engine P1-P8+F1-F18, „Latest key commit" = `65e07ed`.
- **Neu:** ML-Round-7-Hardening bzw. System-Check-Remediation, Latest commit = HEAD post-Fix.
- **Neue Memory-Datei:** `ml-rounds-4-7-orphan-wiring-2026-04-22.md` mit Commits `4b1f77e` bis `cffa1bd` + audit-Befunde.

### P1.3 — CLAUDE.md §14.3 Memory-Pfad präzisieren
- **Datei:** `CLAUDE.md`
- **Problem:** §14.3 sagt „Memory-System via `memory/` + claude-mem (aktiv)" — suggeriert Repo-lokales Verzeichnis, das nicht existiert.
- **Fix:** klarstellen, dass Memory user-level unter `%USERPROFILE%\.claude\projects\…\memory\` liegt.

### P1.4 — docs/cursor/CONTEXT_PACK.md deprecaten
- **Datei:** `docs/cursor/CONTEXT_PACK.md`
- **Problem:** Beschreibt pre-assembled_core-Welt (sprint9/sprint10) als „heute produktiv" — Stand 2025-11-28.
- **Fix-Optionen:**
  - (a) Löschen (Risiko: evtl. noch von Cursor-Workflow gelesen).
  - (b) Header `⚠️ legacy/deprecated — pre-assembled_core era` hinzufügen, Inhalt unverändert lassen.
- **Empfehlung:** (b) — safer.

### P1 — Completion Criteria
- ROADMAP_STATE auf HEAD + phase12-Count.
- MEMORY.md Active Milestone stimmt mit letztem Commit überein.
- CLAUDE.md Memory-Pfad eindeutig.
- CONTEXT_PACK als legacy markiert.

---

## P2 — Architektur-Konsolidierung (Rule 50)

### P2.1 — TCA-Konsolidierung
- **Problem:** 3 TCA-Implementierungen nebeneinander:
  - `qa/tca.py` (cost_bps breakdown aus trades_df)
  - `qa/tca_arrival.py` (Sprint C11 arrival-IS sidecar)
  - `qa/trade_tca.py` (neu, dupliziert IS-Formel)
- **Fix-Optionen:**
  - (a) `trade_tca` auf `tca_arrival` delegieren (neue Wrapper-API, alte Sidecar-Logik wiederverwendet).
  - (b) `tca_arrival` in `trade_tca` absorbieren, alte als Deprecation markieren.
- **Entscheidung pending** — vor Implementierung Ownership/Call-Site-Analyse.
- **Zusätzlich:** Timing-Cost-Claim aus `trade_tca` Docstring entweder implementieren oder entfernen.

### P2.2 — Realized-P&L-Authority in accounting
- **Problem:** 3 FIFO-Implementierungen: `position_engine.py` (canonical), `round_trips.py`, `tax_lots.py`.
- **Fix:** In `accounting/__init__.py` + Modul-Docstrings klar markieren, welche die Autorität ist. Keine Code-Migration nötig, aber Doku-Hinweis.

### P2.3 — Orchestrator TCA-Filename auf `as_of`
- **Datei:** `src/assembled_core/pipeline/orchestrator.py` (Ende der neuen Blöcke)
- **Problem:** TCA-Report mit `pd.Timestamp.now()` — Wall-Clock, bricht Backtest/Replay-Parität.
- **Fix:** `manifest["as_of"]` oder äquivalenten PIT-Date verwenden.
- **Test:** Integration-Test, dass zwei EOD-Runs auf gleiches `as_of` denselben Filename produzieren.

### P2.4 — Orchestrator bare-except absichern
- **Datei:** gleiche Stelle in `orchestrator.py`
- **Problem:** 3× blanket `except Exception` + 1× `except: pass` auf Retention-Purge.
- **Fix:** `except: pass` → `except OSError: logger.debug(...)`. Nested ExceptionBreite einengen auf erwartete Typen.

### P2 — Completion Criteria
- Eine TCA-Wahrheit, andere deprecaten.
- Orchestrator-Filename deterministisch.
- Bare-excepts enger gefasst.

---

## P3 — CI / Dependencies

### P3.1 — `newsapi-python` in requirements.txt
- **Datei:** `requirements.txt`
- **Problem:** In `pyproject.toml` als Runtime-Dep deklariert, aber nicht in `requirements.txt`.
- **Fix:** Pin ergänzen (`newsapi-python==<current>`).
- **Alternative:** Aus pyproject nach extras umziehen, wenn wirklich optional.

### P3.2 — Windows Pip-Cache-Pfad
- **Dateien:** `.github/workflows/accounting-ci.yml:32`, `evidence-pack-ci.yml:32`, `ops-evidence-ci.yml:33`
- **Problem:** Cache auf `~/.cache/pip` (Linux-Pfad) auf `windows-latest`.
- **Fix:** `~\AppData\Local\pip\Cache` für Windows-Jobs.

### P3.3 — Install-Rezepte konsolidieren
- **Problem:** 3 unterschiedliche Install-Pfade:
  - `backend-ci.yml` → `pip install -e ".[dev]"`
  - `paper-trading-ci.yml`, `signal-decay-update.yml`, `prewarm-factor-store.yml`, `release-gate-ci.yml` walk-forward → `requirements.txt`
  - Windows presets → hand-picked Subset
- **Fix-Optionen:**
  - (a) Alle auf `pip install -e ".[dev]"`.
  - (b) `requirements.txt` als Single-Source-of-Truth, pyproject-extras nur für dev/test.
- **Entscheidung pending** — erst nach `newsapi-python`-Fix entscheiden.

### P3.4 — MEMORY-Eintrag „19 collection-fail files" veraltet
- **Quelle:** `memory/ci-stabilization-2026-03-31.md` + `.claude/rules/40-testing-and-ci.md`
- **Realität:** 0 Collection-Errors auf 5414 Tests.
- **Fix:** Rule-Text in `40-testing-and-ci.md` um überholte Aussage bereinigen; MEMORY markieren.

### P3 — Completion Criteria
- `pip install -r requirements.txt` installiert alle Runtime-Deps erfolgreich.
- Windows-CI-Cache hit-fähig.
- Rules/Memory reflektieren aktuelle Collection-Realität.

---

## P4 — CONCERN-Cleanup (nicht blockierend)

Diese Punkte sind keine Bugs, aber Erhöhen die Robustheit vor späterer Wiring. Einzelne kleine Commits, wenn Zeit oder bei Touch im gleichen Modul.

### Signals / ML
- `signals/meta_model.py`: Time-Series-Split in `train_meta_model` + Ordnungs-Guard in `predict_with_intervals`.
- `signals/ml_integration.py`: Explizites WARN-Log statt `debug` bei Stage-Fallback (Observability).
- `signals/risk_aware_combiner.py`:
  - Annualisierungsfaktor als Parameter (nicht hart 252).
  - CRISIS-Fallback auf Zero-Weights statt Equal-Weight.
  - Save/Load für Persistence.

### Risk
- `risk/intraday_monitor.py`:
  - Reset-Pfad für Kill-Switch-Flag.
  - Persistenz (State überlebt Prozess-Restart).
  - VaR-Z-Scores parametrisch statt hard-coded.
  - Entscheidung: bleibt im `__all__` oder versteckt bis Wiring?

### ML
- `ml/online_hmm_regime.py`: Umbenennen (nicht wirklich online) oder echten Streaming-Pfad implementieren.
- `ml/online_hpo.py`:
  - Atomic save (tmp+rename).
  - `observe_reward` dedup via call-id.
  - `_load()` errors nicht stillschweigend auf Prior resetten.
- `ml/feedback_loop.py`:
  - File-Lock auf `feedback_state.json` (fcntl/msvcrt).
  - Retention auf `report_{date}.json`.
  - Same-day-retrain dedup.
- `ml/lime_explainer.py`:
  - 3× `except Exception` narrow.
  - „lime-like"-Fallback als separate Funktion, nicht Silent-Fallback.
- `ml/signal_decay_tracker.py`:
  - Feld `halflife_days` → `halflife_snapshots` oder Zeit-Einheit dokumentieren.
  - Log-Floor `1e-9` überdenken (biased slope bei Zero-IC-Snapshots).

### QA
- `qa/performance_attribution.py`:
  - Reconciliation-Test: `α + Σcontribs ≈ total_return`.
  - `factor_contribution_pct` entweder inkl. α oder explizit dokumentieren.
- `qa/backtest_comparison.py`:
  - DM-Test vs. Sharpe-Diff korrekt kommunizieren oder echten Sharpe-Diff-Test ergänzen (Jobson-Korkie, bootstrap).
  - Multi-Testing: Holm oder BH statt Bonferroni.
- `qa/scenario_simulator.py`:
  - PSD-Check auf target_correlation-Covariance.
  - Historical-Replay-Pfad ergänzen (derzeit nur parametric).

### Intel
- `intel/news_trade_attribution.py`:
  - Window auf strikt pre-trade (`window_end = opened`) ODER Docstring anpassen, dass post-trade News bewusst einbezogen werden.
  - PIT-Leak-Test gegen post-window Events.

---

## Ausführungsprinzipien

- Jedes P0–P3 Item = **ein** Commit, **ein** Thema.
- Vor Commit: `pytest -m phase12 --tb=no` grün halten (Baseline 1810).
- Nach P0-Fixes: neue Round-7-Tests müssen pass, alte weiterhin pass.
- Bei Touch in Risk/Execution/Pipeline (nur P2.3, P2.4): vorher `risk-execution-reviewer`-Delegation.
- P4 nur opportunistisch — kein eigener Sprint, außer explizit angeordnet.

## Verifikations-Gates

- **Nach P0:** `pytest tests/test_ml_round7.py tests/test_portfolio_position_sizing.py tests/test_ml_round4.py -q` + phase12.
- **Nach P1:** Doku-Review ohne Tests.
- **Nach P2:** phase12 + gezielter Integration-Test auf orchestrator-Manifest.
- **Nach P3:** `pip install -r requirements.txt` von leerer venv, `pytest --collect-only` 0-Fehler.
- **Abschluss:** memory-tracker-Eintrag mit Commit-Hashes je Phase.

## Offene Entscheidungen

1. P2.1 — TCA-Konsolidierung: `trade_tca → tca_arrival` oder umgekehrt?
2. P3.1 — `newsapi-python`: runtime oder extra?
3. P3.3 — Install-Pfad: pyproject oder requirements als Single-Source?
4. P4/Risk — `intraday_monitor` im `__all__` belassen oder bis Wiring zurückziehen?

Diese Entscheidungen vor Phase-Start klären.
