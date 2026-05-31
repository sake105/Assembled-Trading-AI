# 01 Branch-Analyse
Stand: 2026-05-27 | Basis: main @ a7e01689

Zweck: Reine Analyse. Keine Aktionen durchgeführt.

---

## git worktree list (vollständig)

```
F:/Python_Projekt/Aktiengerüst                                    a7e01689 [main]
C:/Users/hanso/.cursor/worktrees/Aktienger_st/alj                 a93cfe5b (detached HEAD)
C:/Users/hanso/.cursor/worktrees/Aktienger_st/eiy                 a93cfe5b (detached HEAD)
C:/Users/hanso/.cursor/worktrees/Aktienger_st/feb                 a93cfe5b (detached HEAD)
F:/Python_Projekt/Aktiengerüst/.claude/worktrees/agent-a7b606cb13c522e1d  bb1ba02e [worktree-agent-a7b606cb13c522e1d] locked
F:/Python_Projekt/Aktiengerüst/.claude/worktrees/agent-a8530ae9d7d5595f6  9467b0ae [worktree-agent-a8530ae9d7d5595f6] locked
F:/Python_Projekt/Aktiengerüst/.claude/worktrees/agent-ac275289d3bf5b9ed  bb1ba02e [worktree-agent-ac275289d3bf5b9ed] locked
F:/Python_Projekt/Aktiengerüst/.claude/worktrees/agent-aff57adf12c4aadd1  9467b0ae [worktree-agent-aff57adf12c4aadd1] locked
```

Anmerkung zu den Cursor-Worktrees (`alj`, `eiy`, `feb`): Diese liegen außerhalb des Repos in `C:/Users/hanso/.cursor/worktrees/` und zeigen einen detached HEAD auf `a93cfe5b`. Sie sind keine Git-Branches im Repo — lediglich Cursor-IDE-interne Checkouts. Kein Handlungsbedarf via `git branch -d`.

---

## main

| Feld | Wert |
|------|------|
| Letzter Commit | `a7e01689` — 2026-05-27 00:57 |
| Commit-Message | `fix(pipeline): E-024 — wire algo_type/algo_n_slices into trade journal` |
| Ahead origin/main | 2 |
| Untracked Dateien | `docs/recherche/`, `scripts/_crisis_alpha_backtest_compare.py`, `scripts/_crisis_alpha_pit_verify.py`, `scripts/_crisis_alpha_replay.py`, `scripts/backtest_news_alpha.py` |

Anmerkung: main ist 2 Commits vor origin/main — lokale Commits noch nicht gepusht.

---

## ERWEITERUNG

| Feld | Wert |
|------|------|
| Letzter Commit | `2c902e63` — 2026-05-12 21:31 |
| Commit-Message | `feat(erweiterung): cDCC-GARCH — Aielli (2013) corrected DCC (audit C4-072)` |
| Commits ahead main | **52** |
| Commits behind main | **250** |
| Diff-Umfang | **354 Dateien, 51.176 Insertions** |

### Divergenz-Inhalt (git diff --stat main...ERWEITERUNG, Auszug)

Neue Verzeichnisse/Dateien, die exklusiv auf ERWEITERUNG existieren:
- `docs/erweiterung/` — 26 Markdown-Dokumente (ARCHITECTURE, CODE_AUDIT_FINDINGS, CROSS_ASSET_FINDINGS, DUPLICATE_AUDIT, EQUITY_AUDIT_FINDINGS, EXPANDED_UNIVERSE_BACKTEST, FINAL_EXECUTIVE_SUMMARY, FINAL_EXECUTIVE_SUMMARY_V2, GPR_GEORISK_FINDINGS, HONEST_ASSESSMENT, LIVE_PERFORMANCE_FINDINGS, LONG_HISTORY_FINDINGS, META_LABELING_FINDINGS, MULTI_FACTOR_VOL_TARGET_FINDINGS, PAID_DATA_WISHLIST, PR_MAPPING, REAL_BACKTEST_RESULTS, SIGNALS_REFERENCE, VOL_TARGETING_FINDINGS, WALK_FORWARD_VALIDATION, ...)
- `scripts/erweiterung/` — build_master_html_report.py, fetch_full_universe_long_history.py, fetch_gdelt_biweekly.py, fetch_gdelt_geo_aggregates.py, ...
- `tests/erweiterung/` — test_walk_forward.py (87 Zeilen), test_yfinance_cache_loader.py (88 Zeilen)
- `tests/test_erweiterung_cdcc.py` (92 Zeilen), `tests/test_erweiterung_hansen_spa.py` (90 Zeilen)
- `.github/workflows/erweiterung-tests.yml` (100 Zeilen)

### Kontext
- Branch entstand parallel zu main ab ca. 2026-05-11 (19y-Falsifikationsrunde)
- Kernbefund aus dem Branch selbst (`HONEST_ASSESSMENT.md`): Composite-Edge bei 19y Backtest FALSIFIZIERT (p=0.448 vs 0.915 auf 3.3y — Selection-Bias)
- cDCC-GARCH, Hansen SPA Test, Multi-Asset-Master, GDELT-Resolution-Grid gehören zum Research-Corpus
- Keine Tests dieser Forschung laufen in der Haupt-CI (erweiterung-tests.yml ist ein eigener Workflow)
- Diese Arbeit wurde NIE in main gemergt — 250 Commits hinter main

**Kategorie: ARCHIVIEREN**

Begründung: 52 Commits wertvoller Research-Code (cDCC-GARCH, SPA, 19y-Falsifikation, GDELT-Grid), der nicht in main gemergt werden soll, aber als historische Forschungsreferenz erhalten bleiben sollte. → Git-Tag setzen, Branch löschen.

---

## feat/edcl

| Feld | Wert |
|------|------|
| Letzter Commit | `f967cce3` — 2026-05-02 03:06 |
| Commit-Message | `feat(edcl): wire options_iv_skew_z into ctx — Phase H triple-confirmation now fully wired` |
| Commits ahead main | **0** |
| Commits behind main | **532** |
| Diff-Umfang | **leer** (kein Unterschied zu main) |

### Kontext
- EDCL = Event-Driven Conviction Layer, Phasen A–H
- 0 Commits ahead bedeutet: die Branch-Spitze ist direkter Ancestor von main
- Alle Änderungen sind vollständig in main enthalten (wurden in der main-Entwicklung integriert)

**Kategorie: LÖSCHEN**

Begründung: Vollständig in main enthalten — leere Diff-Ausgabe bestätigt keine einzige Zeile Unterschied.

---

## sprint1/blocker-core-safety

| Feld | Wert |
|------|------|
| Letzter Commit | `eb309923` — 2026-04-10 15:46 |
| Commit-Message | `C17a: add runbooks 1-5 (broker, ledger, kill_switch, drawdown, pit_violation)` |
| Commits ahead main | **13** |
| Commits behind main | **1059** |
| Diff-Umfang | 35 Dateien, 2.581 Insertions, 28 Deletions |

### Divergenz-Inhalt (git diff --stat main...sprint1, Auszug)
- `.github/workflows/backend-ci.yml` (+22 Zeilen)
- `.github/workflows/secrets-scan.yml` (+72 Zeilen)
- `.gitleaks.toml` (+29 Zeilen)
- `.pre-commit-config.yaml` (+10 Zeilen)
- `.secrets.baseline` (+179 Zeilen)
- `docs/runbooks/01_broker_api_unreachable.md` bis `05_pit_violation_detected.md`
- `src/assembled_core/data/pit_guard.py` (+72 Zeilen)
- `src/assembled_core/pipeline/trading_cycle.py` (+334 Zeilen) — **ALTE Version**
- `src/assembled_core/risk/risk_metrics.py` (+120 Zeilen)
- `src/assembled_core/strategies/multifactor_v1.py` (+85 Zeilen)
- diverse Tests

### Kontext
- Branch divergierte ca. Anfang April 2026 von einem alten main-Stand
- Alle genannten Artefakte sind BEREITS IN MAIN vorhanden: `.gitleaks.toml` ✓, `docs/runbooks/` (jetzt 10+ Runbooks) ✓, `secrets-scan.yml` ✓, `Dockerfile` ✓
- `trading_cycle.py` (hier +334 Zeilen) ist die veraltete Monolithen-Version — in main durch `trading_cycle_v2.py` + `trading_cycle_shared.py` ersetzt
- Sprint-1-Inhalte wurden einzeln und verbessert in main übernommen, nicht via Branch-Merge

**Kategorie: LÖSCHEN**

Begründung: Alle 13 Commits-ahead-Inhalte sind inhaltlich in main absorbiert worden (Runbooks, security files, risk_metrics). Die trading_cycle.py-Änderungen sind durch die v2-Refaktorierung überholt.

---

## sprint2/algo-phase1-exec-quality

| Feld | Wert |
|------|------|
| Letzter Commit | `8fe78fc4` — 2026-04-12 14:16 |
| Commit-Message | `feat(C5b): Monte-Carlo VaR + Component VaR (Euler decomposition)` |
| Commits ahead main | **57** |
| Commits behind main | **1059** |
| Diff-Umfang | 124 Dateien, 14.426 Insertions, 32 Deletions |

### Divergenz-Inhalt (git diff --stat main...sprint2, Auszug)
- `.github/workflows/backend-ci.yml`, `secrets-scan.yml`, `.gitleaks.toml`, `.pre-commit-config.yaml`, `.secrets.baseline`
- `Dockerfile` (+96 Zeilen), `docker-compose.yml` (+31 Zeilen)
- `configs/security_master.csv`, `configs/security_meta.csv`, `configs/stress_scenarios.yaml`
- `docs/adr/ADR-001` bis `ADR-007` (Unified Trading Cycle, Factor Store, Kill Switch, PIT Guard, Strategy Plugin, multifactor_v2 Regime Weights, Auto-Drawdown-Kill)
- `docs/runbooks/01–10` + Runbooks-Dateien
- `docs/regression_test_map.md`
- `docs/specs/correlation_guard.md`, `docs/specs/crash_prediction.md`
- `src/assembled_core/pipeline/trading_cycle.py` — ALTE Version
- `src/assembled_core/risk/risk_metrics.py`
- 40+ Testdateien

### Kontext
- 57 Commits ahead, aber 1059 behind — riesige Divergenz
- ALLE Kernressourcen dieser Branch sind bereits in main: `Dockerfile` ✓, `docker-compose.yml` ✓, `docs/adr/ADR-001..007` ✓, `docs/runbooks/01..10` ✓
- Monte-Carlo VaR, Component VaR, EVT-Tails: in main über die Audit-Sweep-Sessions (9 Waves, Mai 2026) integriert
- Die `trading_cycle.py`-Änderungen: veraltet (trading_cycle_v2.py in main)

**Kategorie: LÖSCHEN**

Begründung: 57 Commits, aber alle inhaltlich in main absorbiert — ADRs, Runbooks, Dockerfile, security files und VaR-Implementierungen sind alle in main vorhanden und dort weiterentwickelt worden.

---

## worktree-agent-a0a9c2c0
## worktree-agent-a2f080f2
## worktree-agent-a700e54f
## worktree-agent-ae6ea25d

(Alle vier zeigen identischen Stand — gemeinsam analysiert)

| Feld | Wert |
|------|------|
| Letzter Commit (alle 4) | `44c25ec8` — 2026-03-31 22:30 |
| Commit-Message | `Fix CI: backtest gate error messages + release-gate py -3 → python` |
| Commits ahead main | **0** |
| Commits behind main | **1101** |
| Diff-Umfang | **leer** (kein Unterschied zu main) |

### Kontext
- Alle 4 Branches zeigen auf denselben Commit `44c25ec8`
- Entstanden als Cursor-/Subagent-Worktrees im März 2026
- 0 ahead, leere Diff → vollständig in main enthalten
- Keine Worktree-Verzeichnisse mehr unter `.claude/worktrees/` für diese Branches (dort sind nur die 4 Mai-2026-Worktrees)

**Kategorie: LÖSCHEN** (alle 4)

Begründung: Vier identische Branches auf demselben alten Commit — vollständig in main enthalten, kein Worktree mehr aktiv.

---

## worktree-agent-a73bbad9

| Feld | Wert |
|------|------|
| Letzter Commit | `5dc614da` — 2026-04-26 10:54 |
| Commit-Message | `refactor(pipeline): Phase 4 — pipeline/__init__.py imports directly from shared + v2` |
| Commits ahead main | **0** |
| Commits behind main | **644** |
| Diff-Umfang | **leer** (kein Unterschied zu main) |

### Kontext
- Pipeline-Refaktorierungs-Session vom 26.04.2026: Phasen 0–4 der trading_cycle-Migration
- `trading_cycle.py` (9141 Zeilen → 58 Zeilen), Einführung von `trading_cycle_shared.py` + `trading_cycle_v2.py`
- 0 ahead, leere Diff → Refaktorierung vollständig in main übernommen
- Kein aktiver Worktree mehr

**Kategorie: LÖSCHEN**

Begründung: Pipeline-Refactor vollständig in main, keine einzige divergierende Zeile.

---

## worktree-agent-a7b606cb13c522e1d (AKTIVER WORKTREE)

| Feld | Wert |
|------|------|
| Worktree-Pfad | `.claude/worktrees/agent-a7b606cb13c522e1d` (locked) |
| Letzter Commit | `bb1ba02e` — 2026-05-26 05:54 |
| Commit-Message | `fix(news_alpha): address review-chain MAJORs — Ctrl-C handler, E-021 trim test, gate warning` |
| Commits ahead main | **0** |
| Commits behind main | **4** |
| Diff-Umfang | **leer** — merge-base IST der Branch-HEAD |

### Uncommittete Änderungen im Worktree

```
 M src/assembled_core/data/universe.py
 M src/assembled_core/pipeline/orchestrator.py
?? src/assembled_core/qa/factor_decay_reporter.py
?? tests/test_factor_decay_reporter.py
?? tests/test_universe_survivorship.py
```

### Kontext
- Branch ist vollständig in main aufgegangen (commit `bb1ba02e` = merge-base mit main)
- Die 5 Dateien im Worktree sind NICHT committed — es handelt sich um in-progress Arbeit die NICHT in main übernommen wurde
- `factor_decay_reporter.py` und `test_factor_decay_reporter.py`: In main vorhanden seit Commit `9467b0ae` (`feat(qa+data): factor-decay monitoring + survivorship-safe universe default`)
- `test_universe_survivorship.py`: In main vorhanden
- `universe.py` und `orchestrator.py` (modified): Unklar ob die lokalen Änderungen über den main-Stand hinausgehen oder Zwischen-Artefakte sind

**Kategorie: UNKLAR**

Begründung: Branch-Commits in main, aber 5 uncommittete Dateien im Worktree. Drei Dateien vermutlich bereits in main enthalten. Die zwei modified-Dateien (universe.py, orchestrator.py) brauchen manuelle Prüfung: Enthalten sie Work-in-Progress das verloren geht?

---

## worktree-agent-a8530ae9d7d5595f6 (AKTIVER WORKTREE)

| Feld | Wert |
|------|------|
| Worktree-Pfad | `.claude/worktrees/agent-a8530ae9d7d5595f6` (locked) |
| Letzter Commit | `9467b0ae` — 2026-05-26 17:49 |
| Commit-Message | `feat(qa+data): factor-decay monitoring + survivorship-safe universe default` |
| Commits ahead main | **0** |
| Commits behind main | **2** |
| Diff-Umfang | **leer** — merge-base IST der Branch-HEAD |

### Uncommittete Änderungen im Worktree

```
 M src/assembled_core/pipeline/orchestrator.py
?? tests/test_benchmark_attribution_wiring.py
```

### Kontext
- Branch-Commit `9467b0ae` ist direkter Ancestor von main (2 Commits hinter)
- `test_benchmark_attribution_wiring.py`: In main vorhanden seit Commit `72236ebb` (`feat(pipeline): wire TWAP/VWAP annotation + benchmark attribution`)
- `orchestrator.py` (modified): Muss geprüft werden — könnte Zwischen-Artefakt sein

**Kategorie: UNKLAR**

Begründung: Committed-Arbeit vollständig in main, aber 2 uncommittete Dateien. test_benchmark_attribution_wiring.py vermutlich schon in main. orchestrator.py-Modifikation unklar.

---

## worktree-agent-ac275289d3bf5b9ed (AKTIVER WORKTREE)

| Feld | Wert |
|------|------|
| Worktree-Pfad | `.claude/worktrees/agent-ac275289d3bf5b9ed` (locked) |
| Letzter Commit | `bb1ba02e` — 2026-05-26 05:54 |
| Commit-Message | `fix(news_alpha): address review-chain MAJORs — Ctrl-C handler, E-021 trim test, gate warning` |
| Commits ahead main | **0** |
| Commits behind main | **4** |
| Diff-Umfang | **leer** — identisch mit a7b606cb (gleicher Commit) |

### Uncommittete Änderungen im Worktree

```
 M configs/policy.yaml
?? scripts/dms_daemon.py
?? src/assembled_core/ops/dead_man_switch.py
?? tests/test_dead_man_switch.py
```

### Kontext
- Gleicher Branch-Stand wie `a7b606cb` (`bb1ba02e`)
- `dead_man_switch.py`, `dms_daemon.py`, `test_dead_man_switch.py`: In main vorhanden seit Commit `86468b0c` (`feat(ops): Dead-Man's Switch — passive auto-flat on stale heartbeat`)
- `policy.yaml` (modified): Unklar ob über main-Stand hinausgehende Änderungen

**Kategorie: UNKLAR**

Begründung: Alle 3 neuen Dateien bereits in main. policy.yaml-Modifikation braucht manuelle Prüfung. Vermutlich vollständig bereinigbar.

---

## worktree-agent-aff57adf12c4aadd1 (AKTIVER WORKTREE)

| Feld | Wert |
|------|------|
| Worktree-Pfad | `.claude/worktrees/agent-aff57adf12c4aadd1` (locked) |
| Letzter Commit | `9467b0ae` — 2026-05-26 17:49 |
| Commit-Message | `feat(qa+data): factor-decay monitoring + survivorship-safe universe default` |
| Commits ahead main | **0** |
| Commits behind main | **2** |
| Diff-Umfang | **leer** — identisch mit a8530ae9 (gleicher Commit) |

### Uncommittete Änderungen im Worktree

```
 M configs/policy.yaml
 M src/assembled_core/pipeline/_tc_execution.py
?? tests/test_twap_vwap_annotation.py
```

### Kontext
- Gleicher Branch-Stand wie `a8530ae9` (`9467b0ae`)
- `test_twap_vwap_annotation.py`: In main vorhanden seit Commit `72236ebb` (`feat(pipeline): wire TWAP/VWAP annotation + benchmark attribution`)
- `_tc_execution.py` (modified): Muss geprüft werden
- `policy.yaml` (modified): Muss geprüft werden

**Kategorie: UNKLAR**

Begründung: test_twap_vwap_annotation.py schon in main. Aber _tc_execution.py und policy.yaml sind modifiziert — brauchen Diff gegen main zur endgültigen Bewertung.

---

## Cursor-Worktrees (detached HEAD a93cfe5b)

| Feld | Wert |
|------|------|
| Pfade | `C:/Users/hanso/.cursor/worktrees/Aktienger_st/alj`, `.../eiy`, `.../feb` |
| HEAD | `a93cfe5b` — 2025-11-28 13:56 |
| Commit-Message | `docs(cursor): add rules + context pack` |
| Commits hinter main | **1334** |
| Kein Branch-Name | Detached HEAD — kein `git branch -d` möglich |

### Kontext
- Diese Worktrees werden von Cursor IDE verwaltet, nicht von Claude Code
- `a93cfe5b` ist ein very alter Commit (November 2025, 1334 Commits hinter main)
- Kein Repo-Branch zugeordnet — Cleanup erfolgt über Cursor-IDE oder `git worktree remove`

**Kategorie: LÖSCHEN (via Cursor-IDE oder git worktree remove)**

Begründung: 1334 Commits hinter main, November 2025, keine aktive Arbeit — Cursor-interne Worktrees ohne aktuellen Bezug.

---

## Empfehlungstabelle

| Branch / Worktree | Kategorie | Begründung (ein Satz) |
|---|---|---|
| `main` | — | Aktiver Hauptbranch, 2 Commits vor origin/main (noch nicht gepusht). |
| `ERWEITERUNG` | **ARCHIVIEREN** | 52 Commits wertvoller Research-Code (cDCC-GARCH, 19y-Falsifikation, GDELT-Grid) der nicht in main soll, aber als Forschungsreferenz erhalten werden sollte — Git-Tag setzen, dann Branch löschen. |
| `feat/edcl` | **LÖSCHEN** | 0 Commits ahead, leere Diff — EDCL-Arbeit vollständig in main enthalten. |
| `sprint1/blocker-core-safety` | **LÖSCHEN** | Alle 13 Commits-ahead-Inhalte (Runbooks, security files, risk_metrics) wurden einzeln und verbessert in main übernommen; trading_cycle.py-Änderungen durch v2-Refaktor obsolet. |
| `sprint2/algo-phase1-exec-quality` | **LÖSCHEN** | Alle 57 Commits-ahead-Inhalte (ADRs, Runbooks, Dockerfile, VaR, docker-compose) sind in main vorhanden und weiterentwickelt worden. |
| `worktree-agent-a0a9c2c0` | **LÖSCHEN** | 4 Branches auf identischem alten Commit `44c25ec8`, 0 ahead, leere Diff. |
| `worktree-agent-a2f080f2` | **LÖSCHEN** | Identisch mit a0a9c2c0 — gleicher Commit, vollständig in main. |
| `worktree-agent-a700e54f` | **LÖSCHEN** | Identisch mit a0a9c2c0 — gleicher Commit, vollständig in main. |
| `worktree-agent-ae6ea25d` | **LÖSCHEN** | Identisch mit a0a9c2c0 — gleicher Commit, vollständig in main. |
| `worktree-agent-a73bbad9` | **LÖSCHEN** | Pipeline-Refaktor (trading_cycle-Migration) vollständig in main, 0 ahead, leere Diff. |
| `worktree-agent-a7b606cb13c522e1d` | **UNKLAR** | Branch in main, aber 5 uncommittete Dateien im Worktree (universe.py, orchestrator.py modifiziert) — manuelle Prüfung der lokalen Änderungen nötig bevor Worktree entfernt wird. |
| `worktree-agent-a8530ae9d7d5595f6` | **UNKLAR** | Branch in main, aber orchestrator.py modifiziert im Worktree — manuelle Prüfung nötig. |
| `worktree-agent-ac275289d3bf5b9ed` | **UNKLAR** | Branch in main, aber policy.yaml modifiziert, 3 neue Dateien vermutlich alle schon in main — kurze Prüfung genügt. |
| `worktree-agent-aff57adf12c4aadd1` | **UNKLAR** | Branch in main, aber _tc_execution.py und policy.yaml modifiziert — Diff gegen main nötig. |
| Cursor alj/eiy/feb | **LÖSCHEN** | 3 Cursor-interne Worktrees auf Nov-2025-Commit, keine Branch-Namen, via `git worktree remove` oder Cursor-IDE bereinigen. |

---

## Statistik-Zusammenfassung

| Kategorie | Anzahl | Branches |
|---|---|---|
| MERGEN | 0 | — |
| ARCHIVIEREN | 1 | ERWEITERUNG |
| LÖSCHEN | 10 | feat/edcl, sprint1, sprint2, a0a9c2c0, a2f080f2, a700e54f, ae6ea25d, a73bbad9, Cursor×3 |
| UNKLAR | 4 | a7b606cb, a8530ae9, ac275289, aff57adf |

**Kritischer Hinweis zu den 4 UNKLAR-Worktrees:**
Alle 4 haben `locked`-Status in `git worktree list` — sie können nicht einfach per `git branch -d` gelöscht werden. Zuerst muss `git worktree remove --force <pfad>` ausgeführt werden. Danach erst `git branch -d <name>`. Vor dem `worktree remove` die uncommitteten Änderungen prüfen — entweder verwerfen (`git checkout .` im Worktree) oder in main überführen.
