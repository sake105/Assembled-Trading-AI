Jeder Prompt hat **(a) Plan, (b) umsetzen, (c) Tests/Lint, (d) Ergebnis + Status**.

---

## Prompt 1 — Paper-Track: Strategy-Adapter (trend + multifactor) sauber kapseln

```text
Für jeden Punkt: (a) kurz Plan, (b) Änderungen umsetzen, (c) Tests/Lint laufen lassen, (d) Ergebnis + kurzer Status)

Ziel: Paper-Track soll nicht nur trend_baseline, sondern auch multifactor_long_short (und perspektivisch weitere) ohne Copy/Paste unterstützen.

(a) Kurz Plan:
- Lies src/assembled_core/paper/paper_track.py: wo signal/sizing erzeugt werden.
- Identifiziere bestehende Strategy-Bausteine (trend und multifactor).
- Implementiere ein kleines Adapter-Pattern: get_signal_and_sizer(cfg, prices, as_of) -> (signal_fn, position_sizing_fn) ODER direkt daily target weights.

(b) Änderungen umsetzen:
- Erstelle neues Modul src/assembled_core/paper/strategy_adapters.py (oder innerhalb paper_track.py sauber getrennt).
- Implementiere Adapter für:
  - trend_baseline (bestehende Logik unverändert nutzen)
  - multifactor_long_short (Bundle laden, rebalance, exposure etc. aus bestehendem Code verwenden; KEINE Business-Logik duplizieren).
- PaperTrackConfig: validiere, dass multifactor einen bundle_path/factor_bundle braucht (falls so im Projekt).

(c) Tests/Lint:
- Unit-Test: Adapter liefert callable(s) und produziert nicht-leere Targets für synthetische Preise.
- Mini-E2E: paper_track 3 Tage multifactor mit gepatchten Loadern (wie bei CLI-Tests) -> outputs werden geschrieben.
- pytest targeted + ruff/py_compile.

(d) Ergebnis + kurzer Status:
- Welche Strategien unterstützt, welche Config keys neu/required, Tests grün.
```

---

## Prompt 2 — Paper-Track: Price-Panel Robustheit (Missing Data / Delisted / NaNs)

```text
Für jeden Punkt: (a) kurz Plan, (b) Änderungen umsetzen, (c) Tests/Lint laufen lassen, (d) Ergebnis + kurzer Status)

Ziel: Paper-Track soll robust mit fehlenden Preisen umgehen (NaNs, Symbole ohne Daten, Lücken).

(a) Kurz Plan:
- Prüfe, wie load_eod_prices_for_universe() zurückliefert (Index/Columns).
- Definiere Regeln:
  - Symbole ohne Daten am as_of -> keine Trades / Targets = 0
  - NaN Preise -> skip fill, WARN im daily_summary
  - Optional: Mindestanzahl Datenpunkte vor Trading

(b) Änderungen umsetzen:
- Implementiere helper: filter_tradeable_universe(prices, as_of, min_history_days).
- In run_paper_day: wende Filter an bevor Orders erzeugt werden.
- In daily_summary: schreibe counts (n_symbols_requested, n_tradeable, n_missing).

(c) Tests/Lint:
- Test: Universe enthält 3 Symbole, eins komplett NaN -> tradeable count = 2, keine Orders für NaN symbol.
- Test: as_of Tag Preis fehlt -> skip orders/fills, Status WARN (aber Run success).
- pytest + ruff/py_compile.

(d) Ergebnis + kurzer Status:
- Neue Regeln + Felder im daily_summary, Verhalten bei Missing Data, Tests grün.
```

---

## Prompt 3 — Paper-Track State: Versioniertes Schema + Migration (v1 -> v2)

```text
Für jeden Punkt: (a) kurz Plan, (b) Änderungen umsetzen, (c) Tests/Lint laufen lassen, (d) Ergebnis + kurzer Status)

Ziel: State-Schema soll versioniert migrierbar sein (z.B. neue Felder ohne Breaking).

(a) Kurz Plan:
- Lies load_paper_state/save_paper_state: wie version currently genutzt wird.
- Definiere v2: z.B. neue Felder metadata.run_history_counter, last_equity_snapshot, etc.
- Implementiere migrate_state_dict(state_dict) -> state_dict_v2.

(b) Änderungen umsetzen:
- Erweitere PaperTrackState minimal um 1-2 Felder (z.B. last_equity: float | None, last_positions_value: float | None).
- load_paper_state: wenn version fehlt/alt -> migration anwenden.
- save_paper_state: schreibt immer aktuelle version.

(c) Tests/Lint:
- Test: Lade v1 state fixture (ohne neue Felder) -> returns PaperTrackState mit defaults, version hochgesetzt.
- Test: Roundtrip bleibt stabil.
- pytest state_io tests + ruff/py_compile.

(d) Ergebnis + kurzer Status:
- Neue version, Migration vorhanden, backwards-compatible, Tests grün.
```

---

## Prompt 4 — Paper-Track: Weekly/Monthly Risk-Report Auto-Trigger aus Runner (optional)

```text
Für jeden Punkt: (a) kurz Plan, (b) Änderungen umsetzen, (c) Tests/Lint laufen lassen, (d) Ergebnis + kurzer Status)

Ziel: Runner kann optional Risk-Report erzeugen (z.B. jeden Freitag oder Monatsultimo).

(a) Kurz Plan:
- Prüfe scripts/run_paper_track.py: wo daily loop läuft.
- Prüfe generate_risk_report.py: welche Inputs benötigt werden.
- Definiere Config/CLI: --risk-report-frequency {off,weekly,monthly} + optional benchmark.

(b) Änderungen umsetzen:
- In run_paper_track_from_cli: nach erfolgreichem Tageslauf prüfen ob frequency match (weekday/month-end).
- Risk report als Python call (keine subprocess): import generate_risk_report, call function.
- Output in paper_track/<strategy>/risk_reports/YYYYMMDD/...

(c) Tests/Lint:
- Test: range 5 Tage inkl. Freitag -> risk report wird einmal erzeugt.
- Test: off -> kein report.
- pytest new tests + ruff/py_compile.

(d) Ergebnis + kurzer Status:
- Wie aktiviert, wann getriggert, Output-Pfade, Tests grün.
```

---

## Prompt 5 — Paper-Track: Performance Panel (rolling metrics + deflated sharpe) exportieren

```text
Für jeden Punkt: (a) kurz Plan, (b) Änderungen umsetzen, (c) Tests/Lint laufen lassen, (d) Ergebnis + kurzer Status)

Ziel: Zusätzlich zur equity_curve ein performance_metrics.csv/parquet erzeugen (rolling sharpe, maxdd, turnover, dsr).

(a) Kurz Plan:
- Prüfe vorhandene metrics utils (qa/metrics.py, risk_metrics).
- Definiere Rolling window default (z.B. 63/252) konfigurierbar.
- DSR für Paper-Track: n_tests aus config (oder 1), n_obs aus rolling window.

(b) Änderungen umsetzen:
- Implementiere compute_paper_performance_panel(equity_curve, trades, cfg) -> df.
- Schreibe als aggregated artifact.
- Update docs: Welche Spalten.

(c) Tests/Lint:
- Test: synthetische equity -> rolling sharpe finite ab warmup.
- Test: dsr monotonic bei n_tests (optional).
- pytest + ruff/py_compile.

(d) Ergebnis + kurzer Status:
- Neue Datei, neue Spalten, Defaults, Tests grün.
```

---

## Prompt 6 — Ops: Paper-Track Health-Checks pro Strategie mit Thresholds aus Config

```text
Für jeden Punkt: (a) kurz Plan, (b) Änderungen umsetzen, (c) Tests/Lint laufen lassen, (d) Ergebnis + kurzer Status)

Ziel: check_health soll thresholds je Strategie aus deren paper_track config ziehen (statt nur global CLI).

(a) Kurz Plan:
- Lies scripts/check_health.py: paper-track checks + CLI args.
- Decide precedence: CLI overrides config; config provides defaults.
- Implement load_strategy_thresholds(strategy_dir) -> dict.

(b) Änderungen umsetzen:
- Wenn paper-track strategy gefunden: lade config (falls vorhanden) und ziehe thresholds (max_gap_days, max_daily_pnl_pct, max_drawdown_min, etc.).
- Wende thresholds pro strategie an; Report zeigt verwendete thresholds im details Feld.

(c) Tests/Lint:
- Test: Zwei Strategien, config thresholds unterschiedlich -> Check-Status unterscheidet sich erwartbar.
- pytest check_health tests + ruff/py_compile.

(d) Ergebnis + kurzer Status:
- Threshold precedence dokumentiert, Tests grün.
```

---

## Prompt 7 — CLI UX: `cli.py paper_track list` + Auto-Discovery von Configs

```text
Für jeden Punkt: (a) kurz Plan, (b) Änderungen umsetzen, (c) Tests/Lint laufen lassen, (d) Ergebnis + kurzer Status)

Ziel: Nutzer kann verfügbare Paper-Track Configs/Strategien listen und per name starten.

(a) Kurz Plan:
- Prüfe scripts/cli.py paper_track subcommand structure.
- Implementiere option: paper_track --list (zeigt configs/paper_track/*.yaml und output_root strategies).
- Optional: --strategy-name nutzt default config path.

(b) Änderungen umsetzen:
- In run_paper_track.py: helper discover_paper_track_configs().
- CLI: --list Flag (exit 0) und --strategy-name (resolve config file).
- Dokumentation: Quickstart update.

(c) Tests/Lint:
- Test: create temp configs dir with 2 yaml -> --list prints both.
- Test: --strategy-name resolves correct file.
- pytest cli tests + ruff/py_compile.

(d) Ergebnis + kurzer Status:
- Neue Flags, Beispiel-Output, Tests grün.
```

---

## Prompt 8 — Config Templates: `configs/paper_track/` Beispiele + README/Docs verlinken

```text
Für jeden Punkt: (a) kurz Plan, (b) Änderungen umsetzen, (c) Tests/Lint laufen lassen, (d) Ergebnis + kurzer Status)

Ziel: Standard-Configs im Repo bereitstellen (trend + multifactor) inkl. comments.

(a) Kurz Plan:
- Prüfe aktuelles Config schema (PaperTrackConfig + validate_config_dict).
- Erstelle 2 Beispiel-yaml: trend_baseline.yaml, multifactor_long_short.yaml.
- Stelle sicher: relative paths funktionieren und sind realistisch.

(b) Änderungen umsetzen:
- Add configs/paper_track/*.yaml.
- Update docs/PAPER_TRACK_QUICKSTART.md mit Verweis + commands.
- Optional: add small note in README quick links.

(c) Tests/Lint:
- Test: load_paper_track_config on both templates passes validation (no run).
- pytest config validation tests + ruff (yaml not linted).

(d) Ergebnis + kurzer Status:
- Welche Templates, welche Felder, Tests grün.
```

---

## Prompt 9 — Logging/Tracing: Run-ID + structured summary (JSON) pro Runner-Lauf

```text
Für jeden Punkt: (a) kurz Plan, (b) Änderungen umsetzen, (c) Tests/Lint laufen lassen, (d) Ergebnis + kurzer Status)

Ziel: Jeder Runner-Lauf bekommt run_id (uuid oder timestamp) und schreibt run_summary.json mit per-day statuses.

(a) Kurz Plan:
- Prüfe scripts/run_paper_track.py: wo summary CSV geschrieben wird.
- Erweitere um JSON summary mit run_id, start/end, days attempted, successes/failures, skipped, rerun.
- In daily manifests: include run_id.

(b) Änderungen umsetzen:
- Generate run_id once at start.
- Add run_summary.json next to run_summary.csv.
- Ensure deterministic tests: in tests patch run_id generator.

(c) Tests/Lint:
- Test: range run -> run_summary.json exists and contains counts.
- pytest cli runner tests + ruff/py_compile.

(d) Ergebnis + kurzer Status:
- Neue Summary file, run_id usage, Tests grün.
```

---

## Prompt 10 — CI: GitHub Actions Pipeline (pytest + ruff + py_compile) minimal

```text
Für jeden Punkt: (a) kurz Plan, (b) Änderungen umsetzen, (c) Tests/Lint laufen lassen, (d) Ergebnis + kurzer Status)

Ziel: Minimaler CI-Workflow im Repo, damit PRs nicht regressieren.

(a) Kurz Plan:
- Prüfe ob .github/workflows bereits existiert.
- Definiere job: setup python, install deps, ruff (falls genutzt), pytest (mark advanced optional), py_compile.
- Cache pip optional.

(b) Änderungen umsetzen:
- Add .github/workflows/ci.yml.
- Ensure commands match repo (pyproject/requirements).
- Optional: split fast tests vs advanced marker.

(c) Tests/Lint:
- Lokal: ruff + pytest (subset) ausführen.
- Validate YAML syntax (basic).

(d) Ergebnis + kurzer Status:
- CI steps beschrieben, was laeuft bei push/PR, Status ok.
```

---

Wenn du willst, kann ich dir als nächstes auch **eine “Backlog-Reihenfolge”** empfehlen (damit du Merge-Konflikte minimierst): z.B. erst Prompt 1–3 (Core), dann 4–6 (Ops/Reports), dann 7–9 (UX/Configs), dann 10 (CI).
