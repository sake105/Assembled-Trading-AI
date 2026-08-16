# Claude Coding Errors — Anti-Pattern Register

> **Zweck:** Append-only Log von Coding-Anti-Patterns, die Claude in diesem Repo schon einmal produziert hat. Jeder Eintrag dient als Reminder, denselben Fehler nicht erneut zu machen. Beim Session-Start lädt ein Hook die 10 neuesten Einträge in den Kontext.
>
> **Pflege:** Neue Einträge werden vom `senior-code-reviewer` vorgeschlagen und nach Bestätigung vom Hauptagent appendiert. Niemals existierende Einträge editieren oder löschen — nur neue anhängen.
>
> **Schema:** Siehe `docs/superpowers/specs/2026-05-14-review-chain-design.md` §4.3.

---

## E-001 — pandas `.where(Series)` row-index alignment bug
**Datum:** 2026-05-05
**Kategorie:** pandas-pitfall
**Was passierte:** z-score-Berechnung in `multifactor_v1.py` und `multifactor_v2.py` verwendete `.where(series_condition)`. Bei pandas 2.x alignt das die Condition auf den Row-Index. Wenn die Series einen anderen Index als der Caller hat, werden Werte auf NaN/0 gesetzt → alle Signale werden 0.
**Warum falsch:** pandas alignt Series-Conditions auf Row-Index, nicht auf Position. Bei vom Caller-Index abweichendem Index entstehen stille Datenverluste.
**Wie vermeiden:** `.where(series_condition.values)` oder explizite numpy-Maske via `np.where(condition.values, x, y)`.
**Erkannt in:** `src/assembled_core/strategies/multifactor_v1.py`, `src/assembled_core/strategies/multifactor_v2.py`
**Referenzen:** `memory/session-2026-05-05-path-b-complete-pilot-started.md`

## E-002 — PIT look-ahead durch midnight normalization
**Datum:** 2026-05-09
**Kategorie:** pit-violation
**Was passierte:** In `src/assembled_core/data/latency.py` wurden Timestamps auf Mitternacht normalisiert. Dadurch landete intra-day-Information in „Vortag verfügbar"-Buckets → Look-Ahead-Bias in Backtests.
**Warum falsch:** Timestamp-Normalisierung darf niemals Information zeitlich nach vorne verschieben. PIT-Safety verlangt: Information ist erst verfügbar, NACHDEM sie tatsächlich verfügbar war.
**Wie vermeiden:** Normalisierung nur in Richtung „später" oder „nicht ändern". Bei Zweifel: expliziter `as_of`-Cutoff-Check.
**Erkannt in:** `src/assembled_core/data/latency.py`
**Referenzen:** `memory/session-2026-05-09-tournament-iteration-2-fixes.md`

## E-003 — Silent `except Exception: pass`
**Datum:** 2026-05-02
**Kategorie:** silent-except
**Was passierte:** Diverse Module enthielten `try: ... except Exception: pass` ohne Logging. Fehler verschwanden lautlos, Verhalten wurde non-deterministisch.
**Warum falsch:** Stille Exception-Schluckung versteckt Bugs, verhindert Debugging, untergräbt Determinismus-Garantien.
**Wie vermeiden:** Mindestens `except Exception as e: log.warning("context", exc_info=True)`. Wenn wirklich ignoriert werden soll, im Kommentar begründen.
**Erkannt in:** Multiple Module — siehe Audit-Wave 45-51.
**Referenzen:** `memory/session-2026-05-02-bug-scan-waves-45-51.md`

## E-004 — Empty DataFrame `.iloc[-1]` crash
**Datum:** 2026-05-02
**Kategorie:** logic-error
**Was passierte:** `pairs_trading.py` und `regime_hmm.py` verwendeten `.iloc[-1]` ohne Empty-Check. Bei leerem DataFrame → IndexError, ganzer Pipeline-Step bricht.
**Warum falsch:** `.iloc[-1]` setzt non-empty voraus. In Production-Pfaden mit unsicheren Daten-Quellen ist das eine harte Annahme.
**Wie vermeiden:** `if not df.empty: x = df.iloc[-1]` oder `df.iloc[-1] if len(df) else None`.
**Erkannt in:** `src/assembled_core/strategies/pairs_trading.py`, `src/assembled_core/risk/regime_hmm.py`
**Referenzen:** `memory/session-2026-05-02-bug-scan-waves-52-64.md`

## E-005 — Index-Alignment-Bug bei `set_index().assign()`
**Datum:** 2026-05-03
**Kategorie:** pandas-pitfall
**Was passierte:** Beim Vektorisieren mehrerer Schleifen wurde `df.set_index(col).assign(new_col=series)` benutzt. Die `series` hatte aber einen anderen Index → Index-Alignment zerstörte die Daten.
**Warum falsch:** `.assign()` mit Series alignt auf den DataFrame-Index. Wenn die Quell-Series einen anderen Index hat, kommen NaN raus oder Werte werden vertauscht.
**Wie vermeiden:** Vor `assign()` explizit `series.reindex(df.index)` oder via `.values` arbeiten. Bei Vektorisierungs-Refactors immer Index-Konsistenz-Test.
**Erkannt in:** Mehrere Vektorisierungs-Refactors in Waves 2–11.
**Referenzen:** `memory/session-2026-05-03-optimization-sweep-waves-2-11.md`

## E-006 — datetime64[ns] vs datetime64[us] Ubuntu/Windows mismatch
**Datum:** 2026-05-04
**Kategorie:** logic-error
**Was passierte:** `qa/event_study.py` erzeugte datetime64[us] Timestamps lokal, aber CI auf Ubuntu mit pandas 2.2 erwartete datetime64[ns] → Vergleichs- und Merge-Operationen schlugen fehl, aber nur in CI.
**Warum falsch:** Implizite dtype-Annahmen sind plattformabhängig. „Lokal grün" ist nicht „CI grün".
**Wie vermeiden:** Bei Timestamp-Vergleichen explizit `astype("datetime64[ns]")`. Tests sollen plattformrobuste dtypes verlangen.
**Erkannt in:** `src/assembled_core/qa/event_study.py`, `src/assembled_core/qa/post_trade_analyzer.py`
**Referenzen:** `memory/session-2026-05-04-session4-alles-machen.md`

## E-007 — float-NaN/None Mix in dict.get-Fallbacks
**Datum:** 2026-05-02
**Kategorie:** logic-error
**Was passierte:** `dict.get(key) or default` Pattern bricht wenn Value `0`, `False`, oder `NaN` ist → unbeabsichtigter Default. Auch `int(None)` / `float(None)` ohne Guard.
**Warum falsch:** Python's `or`-Truthiness behandelt 0/False/leere Strings/NaN als falsy. Bei numerischen Defaults entstehen so falsche Werte.
**Wie vermeiden:** `dict.get(key, default)` mit explizitem Default. Bei numerischen Casts immer `if v is not None: int(v)` oder `pd.notna(v)`.
**Erkannt in:** `intel/news_dedupe.py`, mehrere YAML-Loader.
**Referenzen:** `memory/session-2026-05-02-bug-scan-waves-86-101.md`

## E-008 — `pd.to_datetime` ohne `errors='coerce'` crasht auf schlechten Daten
**Datum:** 2026-05-02
**Kategorie:** pandas-pitfall
**Was passierte:** Datenpipeline rief `pd.to_datetime(series)` ohne `errors='coerce'` auf. Bei einem einzigen unparsbaren Wert → `ValueError`, ganzer Pipeline-Schritt bricht statt graceful zu degradieren.
**Warum falsch:** Externe Datenquellen (CSV, API-Responses, YAML) enthalten fast garantiert irgendwann malformed timestamps. Crash-Verhalten verhindert PIT-saubere Backtests.
**Wie vermeiden:** Immer `pd.to_datetime(series, errors='coerce')` außer in expliziten Validator-Pfaden. Nachgeschalteter NaT-Check entscheidet über Fortsetzung/Block.
**Erkannt in:** `data/altdata/earnings_calendar_source.py`, mehrere Loader.
**Referenzen:** `memory/session-2026-05-02-bug-scan-waves-86-101.md`

## E-009 — `Series.any()` mit NaN gibt NaN zurück, nicht bool
**Datum:** 2026-05-02
**Kategorie:** pandas-pitfall
**Was passierte:** `if series.any():` in Conditional verwendet. Series hatte NaN-Werte → `any()` lieferte `NaN` (truthy!) → Branch genommen, wo False richtig wäre.
**Warum falsch:** Pandas-Aggregationen propagieren NaN. `any()` mit NaN ist nicht False sondern unknown → in `if` als truthy interpretiert.
**Wie vermeiden:** `series.fillna(False).any()` oder `series.dropna().any()`. Bei numerischen Conditions: `(series > threshold).fillna(False).any()`.
**Erkannt in:** `qa/regime_analysis.py`, mehrere Signal-Module.
**Referenzen:** `memory/session-2026-05-02-bug-scan-waves-86-101.md`

## E-010 — `idxmax()` / `idxmin()` auf leerer Series → ValueError
**Datum:** 2026-05-02
**Kategorie:** logic-error
**Was passierte:** `df['col'].idxmax()` in Allocator-/Optimizer-Code ohne Guard. Bei leerem DataFrame → `ValueError: attempt to get argmax of an empty sequence`.
**Warum falsch:** `idxmax` setzt non-empty voraus. In Production-Pfaden mit Filter-Vorstufen kann der DataFrame leer werden, ohne dass es offensichtlich ist.
**Wie vermeiden:** `df['col'].idxmax() if not df.empty else default`. Oder besser: `df['col'].dropna().pipe(lambda s: s.idxmax() if not s.empty else default)`.
**Erkannt in:** `strategies/strategy_allocator.py`, `signals/ensemble.py`.
**Referenzen:** `memory/session-2026-05-02-bug-scan-waves-86-101.md`

## E-011 — `json.dumps` crasht auf numpy-Typen
**Datum:** 2026-05-02
**Kategorie:** logic-error
**Was passierte:** `json.dumps(d)` auf Dict mit `np.int64`/`np.float64`-Werten → `TypeError: Object of type int64 is not JSON serializable`. API-Endpoint crashed.
**Warum falsch:** numpy-Skalar-Typen sind keine Python-Builtins. Standard json-Encoder kennt sie nicht.
**Wie vermeiden:** Eigener Encoder: `json.dumps(d, default=lambda o: o.item() if hasattr(o, 'item') else str(o))`. Oder vor Serialisierung `pd.json_normalize`-äquivalent konvertieren.
**Erkannt in:** `api/routers/monitoring.py`, `events/store.py`, `attribution/storage.py`.
**Referenzen:** `memory/session-2026-05-02-bug-scan-waves-86-101.md`

## E-012 — `date.today()` ist Lokalzeit, nicht UTC
**Datum:** 2026-05-02
**Kategorie:** logic-error
**Was passierte:** PDT-Check verwendete `date.today()` für Tages-Vergleich. Lokal (Europe/Berlin) und UTC unterscheiden sich nach 00:00 lokal um einen Tag → Off-by-One in Tag-Zähler an Tageswechseln.
**Warum falsch:** Trading-System läuft mit UTC-basierten Daten. Lokalzeit-Boundary stimmt nicht mit Market-Boundary überein.
**Wie vermeiden:** Immer `datetime.now(timezone.utc).date()`. Für Markttag: `pd.Timestamp.now(tz='UTC').normalize()`. Module-Konstanten via `TODAY = datetime.now(timezone.utc).date()` nur einmal pro Modul-Load.
**Erkannt in:** `risk/pdt_tracker.py`, mehrere Tagesgrenzen-Checks.
**Referenzen:** `memory/session-2026-05-02-hmm-grid-complete.md`

## E-013 — `next(iter(d))` auf leerem Dict → StopIteration
**Datum:** 2026-05-02
**Kategorie:** logic-error
**Was passierte:** ECB-API-Loader nutzte `next(iter(response['data']))` als „erstes Element". Bei leerer Response → `StopIteration` unter dem Hood, lokal als Crash propagiert.
**Warum falsch:** `next()` ohne `default` Argument wirft `StopIteration` wenn der Iterator leer ist. In Python 3.7+ kein „bubble through generators" mehr, aber lokal immer noch ein Crash.
**Wie vermeiden:** `next(iter(d), default)` mit explizitem Default. Oder `if d: first = next(iter(d))` mit Guard.
**Erkannt in:** Mehrere API-Response-Parser.
**Referenzen:** `memory/session-2026-05-02-bug-scan-waves-52-64.md`

## E-014 — `tz_convert(None)` auf tz-naiver Series → TypeError
**Datum:** 2026-05-02
**Kategorie:** logic-error
**Was passierte:** Code verwendete `series.dt.tz_convert(None)` um in lokale Naive zu konvertieren. Bei bereits naiver Series → `TypeError: Cannot convert tz-naive timestamps, use tz_localize to localize`.
**Warum falsch:** `tz_convert` verlangt eine tz-aware Series als Input. Naive Series brauchen erst `tz_localize` bevor `tz_convert` funktioniert.
**Wie vermeiden:** `if series.dt.tz is not None: series = series.dt.tz_convert(None)`. Oder defensiver Helper: `def to_naive_utc(s): return s.dt.tz_convert('UTC').dt.tz_localize(None) if s.dt.tz else s`.
**Erkannt in:** Mehrere Zeitstempel-Handler.
**Referenzen:** `memory/session-2026-05-02-bug-scan-waves-52-64.md`

## E-015 — `joblib.load` ohne EOFError-Handling crasht auf truncated cache
**Datum:** 2026-05-02
**Kategorie:** logic-error
**Was passierte:** Model-Loader rief `joblib.load(cache_path)`. Wenn Cache-Datei beim letzten Run nicht vollständig geschrieben wurde → `EOFError`, Loader crashed statt cache zu invalidieren und neu zu generieren.
**Warum falsch:** Joblib serializes mit Streaming-Format. Bei Abbruch beim Schreiben (Strom, Kill) entstehen halbe Dateien, die kein `pickle` mehr parsen kann.
**Wie vermeiden:** `try: m = joblib.load(p) except (EOFError, pickle.UnpicklingError): p.unlink(missing_ok=True); m = regenerate()`. Cache als regenerierbar behandeln, nie als Wahrheit.
**Erkannt in:** `signals/meta_model.py`, `strategies/multifactor_v2.py`.
**Referenzen:** `memory/session-2026-05-02-bug-scan-waves-36-44-continuation.md`

## E-016 — `yaml.YAMLError` uncaught → ganze Config-Load crasht
**Datum:** 2026-05-02
**Kategorie:** logic-error
**Was passierte:** `yaml.safe_load(open(path))` ohne try/except. Bei minimal kaputter YAML (eingerücktes Tab, fehlendes `:`) → ganzer Boot-Prozess bricht, statt graceful degradation auf Defaults.
**Warum falsch:** Configs werden öfter manuell editiert als Code. Ein Syntax-Fehler in einer Config sollte nicht den ganzen Runner töten.
**Wie vermeiden:** `try: cfg = yaml.safe_load(...) except yaml.YAMLError as e: log.error("Config malformed: %s", e); cfg = DEFAULT_CONFIG`. Validation-Pfad separat.
**Erkannt in:** `batch_runner/`, `run_paper_track.py`, `strategy_config.py`.
**Referenzen:** `memory/session-2026-05-02-bug-scan-waves-36-44-continuation.md`

## E-017 — pandas `groupby().apply()` Deprecation: `include_groups=False`
**Datum:** 2026-05-02
**Kategorie:** pandas-pitfall
**Was passierte:** `df.groupby('x').apply(fn)` produzierte ab pandas 2.2 `DeprecationWarning`, ab 3.0 wird das Default-Verhalten geändert. Bei Update von pandas 2.0 → 2.2 → bricht teilweise.
**Warum falsch:** pandas-Versionswechsel ändern subtle Defaults. Tests laufen lokal mit 2.0, CI mit 2.2 → divergierendes Verhalten.
**Wie vermeiden:** `df.groupby('x', group_keys=False).apply(fn, include_groups=False)` explizit setzen. Auch im Repo aktiv suchen nach allen `groupby(...).apply` Sites.
**Erkannt in:** Mehrere QA- und Strategy-Module.
**Referenzen:** `memory/session-2026-05-02-bug-scan-waves-52-64.md`

## E-018 — `np.exp()` overflow auf großen Werten ohne Clip
**Datum:** 2026-05-02
**Kategorie:** logic-error
**Was passierte:** Scenario-Engine berechnete `np.exp(returns_sum)` für Long-Horizon-Aggregate. Bei extremen Returns → `RuntimeWarning: overflow encountered in exp` und `inf`-Werte, die alle nachgelagerten Aggregate ruinieren.
**Warum falsch:** `np.exp(710+)` overflows zu `inf` (float64-Limit). Bei langen Horizonten oder synthetischen Stress-Szenarien realistisch erreichbar.
**Wie vermeiden:** `np.exp(np.clip(x, -700, 700))` oder log-space arithmetic durchhalten. Bei finanziellen Returns: `np.expm1(np.clip(...))` für Stabilität nahe Null.
**Erkannt in:** `qa/scenario_engine.py`, `qa/synthetic_generator.py`.
**Referenzen:** `memory/session-2026-05-02-bug-scan-waves-52-64.md`

## E-019 — Silent fail-open in Enforcement-Schicht durch unvollständiges Input-Shape-Parsing
**Datum:** 2026-05-15
**Kategorie:** governance/automation-failure
**Was passierte:** Der Stop-Hook-Transcript-Parser (`.claude/hooks/hook_utils/transcript_parser.py`) lief vom Transkriptende rückwärts und brach bei jedem `type=user`-Eintrag ab. Claude Code speichert aber Tool-Ergebnisse ebenfalls als `type=user` (mit `content=[tool_result, ...]`). Folge: nach jedem Edit/Write kam ein Tool-Result-Wrapper, der Parser brach dort ab, `edited_paths_in_last_turn()` gab `[]` zurück, `classify_diff([])` setzte `run_full_chain=False`, der Hook ließ Stop durch. Die Review-Chain triggerte ~27 Stunden lang nicht, obwohl CLAUDE.md §20.7 sie als „vollständig aktiv" beschrieb. Tests passten weiterhin, weil die Fixture nur synthetische Shapes ohne Tool-Result-Wrapper enthielt (siehe CLAUDE.md §20.8).
**Warum falsch:** Eine Enforcement-Schicht darf bei unbekanntem Input nicht still ein leeres Ergebnis liefern. „Keine geänderten Pfade gefunden" und „Parser hat den Input nicht verstanden" sind zwei verschiedene Zustände, die getrennt signalisiert werden müssen. Sonst sieht ein kaputter Enforcer aus wie ein arbeitsloser Enforcer. Das Pattern ist breiter als der Tool-Result-Wrapper-Spezialfall: jeder Parser, der bei produktiven Input-Shapes still `None`/`[]`/leeres Ergebnis liefert, ohne dass Tests die Shape exerciert haben, kann das gleiche Failure-Mode auslösen.
**Wie erkennen:**
- Ein Enforcer, der „nie triggert", ist verdächtig — nicht erfolgreich.
- Marker-/Log-Dateien, die nie entstehen, obwohl sie sollten: roter Flag (`.claude/.review_markers/` existierte nie, `.review_skip_log.jsonl` nie angelegt).
- Tests passen, aber das Feature wirkt in Produktion nicht: Fixture-Shape prüfen gegen echte Produktions-Inputs.
**Wie vermeiden:**
- Hook-/Enforcer-Tests müssen mindestens einen anonymisierten Echtinput als Fixture nutzen, nicht nur synthetische Minimal-Shapes.
- Heartbeat-Log einbauen: jeder Hook-Aufruf schreibt eine Zeile, auch wenn er nichts zu tun hatte (z. B. `edited_paths=[]`). Absence-of-heartbeat ist dann selbst ein Signal.
- Bei Discriminator-basiertem Parsing (hier: `type=user`) explizite Allow-list + Deny-list testen (real user, tool_result wrapper, mixed text+tool_result, attachment, system) — nicht nur Happy Path.
- Bei `obj.get("message", {}).get("content")`: aufpassen auf `message=null` vs. `message` fehlend — `.get` koalesziert keinen `None`-Wert. Idiom: `(obj.get("message") or {}).get("content", "")`.
**Erkannt in:** `.claude/hooks/hook_utils/transcript_parser.py` (Commit `60c7ea2`), `tests/hooks/fixtures/transcript_with_edits.jsonl` (Fixture-Gap).
**Referenzen:** CLAUDE.md §20.8, Commit `60c7ea2`, Stage-2 Review-Findings F-senior-1/3/4/6.

## E-020 — Parallel-Subagent Dispatch + `git add -A` verliert Commit
**Datum:** 2026-05-17
**Kategorie:** wiring-gap / parallel-execution
**Was passierte:** Wave-1 der Plan-Ausführung (8 parallele Sonnet-Subagents) lief auf dem gleichen git-Working-Tree. Jeder Subagent endete mit `git add` + `git commit`. Mindestens einer (B4) verwendete eine breite Staging-Semantik (`git add -A` oder äquivalent), die nicht-committete Änderungen von einem Sibling-Agent (A2) mit aufnahm. Folge: A2's Main-Fix wurde unter der falschen Commit-Message (B4: "docs: clarify notebooks/ vs research/") committet — nur teilweise. A2's eigener Commit wurde leer (Empty-Commit `9070ff2 "style: ruff-format..."` mit 0 file changes). Recovery: A2 manuell re-committen (`e864584`). Zusätzlich: A2's Test-File `tests/test_shipping_ingest_failloud.py` blieb untracked (Senior F-senior-1, BLOCKER). Per Senior-Review entdeckt im Stage-2-Pass auf die ausgeführten Tasks.
**Warum falsch:** Parallele Subagents auf dem gleichen working tree teilen sich den git-Index. `git add -A` / `git add .` sind nicht semantisch atomar in einem Multi-Process-git-Kontext. Jeder Subagent denkt er staged "seine" Änderungen, aber er staged alles was Sibling-Agents auf Disk geschrieben aber noch nicht committed haben. Die atomare Einheit von git ist der Commit, nicht die Subagent-Grenze.
**Wie erkennen:**
- Nach paralleler Subagent-Welle: `git log --since=Xmin --stat` zeigt Commits, die Files enthalten die nicht zur Commit-Message passen.
- Ein Subagent meldet "DONE Commit X" aber X enthält 0 file-changes (Empty-Commit).
- `git status` zeigt untracked Files, die laut Plan committed sein müssten.
**Wie vermeiden:**
- Subagents MÜSSEN explizite File-Pfade in `git add` verwenden (niemals `-A`, niemals `.`).
- Dispatcher (Parent-Agent) sollte den COMMIT-Step serialisieren auch wenn der WORK-Step parallel lief — Subagents produzieren Patches, Parent applied sequenziell.
- Alternative: `git worktree` so jeder Subagent in isoliertem Tree arbeitet, dann sequenziell mergen.
- Minimum: Parent-Agent `git status` zwischen Wellen + untracked Files reconcilen BEVOR pushed wird.
- Stop-Hook könnte aborten wenn `git status` untracked Files unter `tests/` oder `src/` zeigt, die zum aktuellen Task-Scope gehören.
**Erkannt in:** `tests/test_shipping_ingest_failloud.py` (untracked nach Wave-1, recovered `24e4517`), Commit `9070ff2` (empty), Commit `4d95c32` (bundled B1+B4).
**Referenzen:** Plan `docs/superpowers/plans/2026-05-17-dummy-data-and-info-flow.md`, Senior-Review F-senior-1 (BLOCKER), Auditor F-auditor-3-1 (MAJOR), F-auditor-3-3 (MINOR bundled commit).

## E-021 — Self-verifying Logging-Test (Test emittiert eigene Assertion-Signal)
**Datum:** 2026-05-17
**Kategorie:** test-anti-pattern
**Was passierte:** `tests/test_pipeline_congress_failloud.py::test_narrowed_congress_except_emits_warning` testete eigentlich die Production-Handler-Warning in `_build_features_default`. Stattdessen reproduzierte der Test die `ModuleNotFoundError` MANUELL im Test-Body, fing sie selbst, emittierte SEINE EIGENE Warning via `logging.getLogger(__name__).warning(...)`, und assertete dann dass diese Warning in `caplog` ist. Production-Code wurde nie ausgeführt — der Test verifiziert sein eigenes Logging-Statement.
**Warum falsch:** Tests, die das Signal emittieren auf das sie asserten, sind tautologisch. Sie geben ein falsches Coverage-Gefühl: Pass-Status sagt nichts darüber aus, ob der Production-Code überhaupt erreicht wurde oder ob der Production-Code die richtige Message loggt. Wenn ein zukünftiger Refactor die Production-Warning entfernt, passt der Test weiterhin.
**Wie erkennen:**
- Test ruft `logger.warning(...)` (oder andere Log-Calls) DIREKT im Test-Body und assertet danach auf `caplog`.
- Test-Body enthält `try/except` der das Error-Pattern reproduziert das eigentlich die Production-Funktion auslösen sollte.
- Coverage-Reports zeigen das Production-File mit niedriger Coverage trotz scheinbar passendem Test.
**Wie vermeiden:**
- Niemals `logger.warning(...)` direkt im Test-Body wenn der Test caplog-Warnings verifiziert. Die Warning muss aus dem Code unter Test kommen.
- Wenn der Production-Pfad schwer zu konstruieren ist (z.B. erfordert vollen TradingContext): (a) Handler in kleine pure-Function extrahieren und die testen, oder (b) `unittest.mock.patch` zum Stubben der Dependencies aber RUFE die echte Production-Funktion auf.
- Fallback wenn (a) und (b) nicht praktikabel: Source-Text-Static-Check (lies das Production-File und assertet dass die erwartete Message-String enthalten ist). Verifiziert nur Static-Contract, aber wenigstens nicht selbstreferentiell.
**Erkannt in:** `tests/test_pipeline_congress_failloud.py::test_narrowed_congress_except_emits_warning` (entfernt in Commit `bc290fb`, ersetzt durch zwei Source-Text-Static-Checks).
**Referenzen:** Senior-Review F-senior-3, Commit `bc290fb`.

## E-022 — Dual-Import-Path durch unvollständigen Bare-Prefix-Sweep
**Datum:** 2026-05-17
**Kategorie:** wiring-gap / partial-refactor
**Was passierte:** Commit `e4e88cc` migrierte `from assembled_core.*` → `from src.assembled_core.*` in 12 `src/`-Files (Sub-Project `O-stage2-1`), aber NICHT in den Test-Files (22 Files) oder Scripts (8 Files), die dasselbe Modul importierten. Folge: `src/assembled_core/certify/` lud sich als `src.assembled_core.certify.X` (neu), aber `tests/test_certify.py` lud `from assembled_core.certify.X import EnvironmentFingerprint` (alt). Python betrachtet die beiden Pfade als **zwei verschiedene Module** mit zwei verschiedenen Class-Objekten. `isinstance(fp, EnvironmentFingerprint)` lieferte False (eine Klasse vs. andere), 2 Tests in `tests/test_certify.py` fielen aus.
**Warum falsch:** Python's Modul-Cache (`sys.modules`) keyed nach exakter Import-Path-String. `assembled_core.certify` und `src.assembled_core.certify` sind zwei distincte Cache-Einträge → zwei distincte Module → zwei distincte Klassen. Das ist KEINE Python-Idiosynkrasie sondern erwartetes Verhalten; der Fehler ist der unvollständige Sweep.
**Wie erkennen:**
- `isinstance()`-Fehler bei Code, der "offensichtlich passen müsste".
- Doppelte Class-Definitionen im Debugger (`type(obj) is not ExpectedClass` aber `obj.__class__.__name__ == 'ExpectedClass'`).
- Test-Regressions in seemingly-unrelated Tests nach Refactor (`grep -rln "^from assembled_core\."` zeigt nicht-migrierte Files).
**Wie vermeiden:**
- Bei Bare-Prefix-Sweep (oder anderer Import-Path-Migration) IMMER `src/` + `tests/` + `scripts/` in **einem Commit** oder sequenziell ohne dazwischenliegenden Push migrieren.
- Pre-commit grep-check `grep -rln "^from assembled_core\." src/ tests/ scripts/` → exit 1 wenn nicht-leeres Resultat.
- Test-Sweep nach Refactor: `pytest --collect-only` reicht nicht (Collection findet die Imports); echte Test-Runs auf den affected Modulen + isinstance-heavy Tests laufen lassen.
- Nie `git add -A` über teilweise-migrierten Stand committen.
**Erkannt in:** Commit `e4e88cc` (Sweep src/ only) → Stage-1 test-runner F-test-1 MAJOR → Fix in Commit `43299ce` (Sweep tests/ + scripts/ ergänzt). 32 Files total in der vollständigen Migration.
**Referenzen:** CLAUDE.md Rule 50 (Doppelstrukturen), Rule 60 (one problem per change), Stage-1 test-runner Findings 2026-05-17.

## E-023 — Too-broad Detection-Token führt zu Silent False-Positive Cooldown
**Datum:** 2026-05-22
**Kategorie:** detection-logic / silent-degradation
**Was passierte:** In commit `30d9c3d` (API key rotator) enthielt `_RATE_LIMIT_TOKENS` den Eintrag `"thank you for using alpha vantage"` — die Standard-Begrüßungs/Branding-Phrase, die AV in **vielen** Response-Bodys verwendet, nicht nur in Rate-Limit-Replies. Konkrete falsche Matches: Premium-Required-Replies, Invalid-API-Key-Replies, Deprecated-Endpoint-Notices. Folge: `is_rate_limit_signal()` lieferte für jede dieser Antworten True → der working Key wurde 3600s gecooled → für die nächste 1h waren alle nachfolgenden Symbole im Run abgeschnitten von AV, obwohl der Key nicht erschöpft war. Cooldown wurde zudem persistiert (`output/ops/api_key_usage.json`) → überlebte Process-Restart.
**Warum falsch:** Detection-Tokens, die auf Vendor-Branding/Greeting matchen statt auf das spezifische Fehler-Pattern, feuern auf ALLES vom Provider — nicht nur auf die Fehlerklasse, die gemeint war. Blast radius bei False-Positive: 1h × (pool size − 1) verlorene Quota plus persistente State-Pollution.
**Wie erkennen:**
- "Working Key wird scheinbar grundlos cooled-down, obwohl ich gerade erst Quota benutzt habe."
- Cooldown-Trigger-Logs feuern bei Non-429-Responses ("invalid key", "premium endpoint", "no data" etc.).
- `_RATE_LIMIT_TOKENS` (oder äquivalente Detection-Listen) enthält Wörter/Phrasen, die NICHT exklusiv für Rate-Limit-Fälle sind.
**Wie vermeiden:**
- Detection-Tokens müssen **spezifisch** für die Fehlerklasse sein, nicht für den Vendor. Bevorzugt: HTTP-Statuscode (`status_code == 429`, `code == 429`) als Primärsignal.
- Text-Match nur für Vendor-spezifische Throttle-Wordings, die NICHT in anderen Responses des gleichen Vendors auftreten (z.B. "api call frequency", "calls per minute", "calls per day" — alle exklusiv für AV-Throttle).
- Pflicht: **False-Positive-Regression-Tests**. Für jeden Token im Detector mindestens ein Test, der ein nicht-rate-limit-Response mit ähnlichem Wording prüft und `False` erwartet.
- Bei Token-Co-Occurrence-Heuristik (z.B. "429" + co-token): Co-Tokens müssen ebenfalls spezifisch sein (keine isolierten Wörter wie "rate"/"limit"/"quota" allein).
**Erkannt in:** Commit `30d9c3d` → Stage-1 risk-execution-reviewer F-AKR2-1 HIGH + F-AKR2-2 MAJOR (single-token false positives) → Fix in Commit `a39077b` (Tokens präzisiert + 6 false-positive Regression-Tests).
**Referenzen:** Stage-1 risk-review 2026-05-22, CLAUDE.md Rule 30 (Risk-Execution-Safeguards — silent-degradation), Rule 40 (Test-Honesty — positiv- AND negativ-cases).

## E-024 — Infrastructure shipped without consumer wiring
**Datum:** 2026-05-22
**Kategorie:** wiring-gap / completeness
**Was passierte:** Commit `30d9c3d` lieferte einen voll funktionsfähigen `ApiKeyRotator` mit `mark_rate_limited()`/`get_key()`/Cooldown-Persistence + 21 Unit-Tests. Aber: **kein einziger Client rief tatsächlich `mark_rate_limited()` bei 429 auf**. Das heißt: das Cooldown-Halbsystem des Rotators war in der echten Codebase toter Code. Round-Robin allein verteilte Last → User-Wunsch "wenn einer erschöpft, zum nächsten wechseln" war **nicht** funktional eingelöst, weil kein Client die Exhaustion zurückmeldete. Zusätzlich: `alphavantage_source.py` (Daily-OHLCV, schmerzhafteste Quota im System mit 25 req/Tag) war gar nicht via Rotator gewired. Tests gaben PASS (Rotator-Mechanik korrekt), aber das User-Intent war nicht erfüllt.
**Warum falsch:** "Infrastruktur gebaut" ≠ "User-Intent eingelöst". Helper-Funktionen, Pools, Detection-Mechanismen sind nur dann nützlich, wenn die Consumer-Pfade sie tatsächlich aufrufen. Tests, die nur die Infrastruktur direkt prüfen (Rotator-API), sagen nichts über die End-to-End-Funktionalität aus.
**Wie erkennen:**
- Stage-3 Auditor (oder Reviewer) fragt: "Funktioniert das User-Intent durch den Code, oder nur die Komponente?"
- Strukturierte Per-Call-Site-Enumeration: für jeden Provider/Pfad die Consumer-Surface auflisten und prüfen, ob jeder Call die neue Infrastruktur tatsächlich nutzt.
- Spotcheck: `grep -rn "mark_rate_limited\|<new_helper>" src/` — wenn die Anzahl der Aufruf-Stellen vergleichbar mit der Anzahl der Fetch-Call-Sites ist → wahrscheinlich gewired. Sonst → Wiring-Gap.
**Wie vermeiden:**
- "End-to-end wired" als Claim erfordert: pro Consumer-Pfad mindestens (a) `get_key()`-Call, (b) `mark_rate_limited()`-Call in 429-except-Branch, (c) re-resolve nach mark wenn loop-basiert.
- Pre-Commit-Checkliste: für jedes neue Infrastructure-Modul die Liste der erwarteten Consumer-Pfade explizit notieren + pro Pfad verifizieren.
- Tests sollten **mindestens einen** Test mit gemocktem HTTP-Layer haben, der einen vollen Rate-Limit-Loop simuliert (429 → mark → next-call uses different key) durch den echten Client-Pfad, nicht nur durch direkte Rotator-API-Calls.
- Commit-Message muss ehrlich trennen zwischen "Infrastructure built" und "Consumer wired" — bei Gap explizit dokumentieren.
**Erkannt in:** Commit `30d9c3d` (rotator built, clients not wired for 429-feedback) → Stage-2 senior-reviewer F-senior-AKR-2 MAJOR + F-senior-AKR-1 (AV not wired) → Fix in Commit `a39077b` (5 Clients × 5+ Call-Sites alle gewired, within-loop rotation für Finnhub/AV/Polygon/NewsAPI).
**Referenzen:** Stage-2 senior-review + Stage-3 auditor 2026-05-22, CLAUDE.md Rule 60 (one problem per change — Infrastruktur OHNE Wiring ist nicht "one problem complete").

---

## E-025 — Loader Fail-Open Masks Corruption in API Read Path
**Datum:** 2026-05-28
**Kategorie:** silent-except / silent-degradation
**Was passierte:** `GET /api/v1/ledger` rief `load_ledger_state()` auf, ohne zu prüfen ob der Loader stillschweigend auf `_fresh_state(start_capital=10000.0)` gefallen war. `load_ledger_state` fängt alle Parse-Fehler intern ab und gibt bei vollständigem Fallback `{updated_utc: None, cash: 10000.0, positions: {}, equity_curve: []}` zurück — ununterscheidbar vom Zustand eines frisch gestarteten Pilots. API hätte `status='ok'` mit $10k Startguthaben zurückgegeben, obwohl das Ledger korrumpiert war.
**Warum falsch:** Eine Observability-Schicht die bei korruptem Underlying-State "alles ok" meldet ist schlimmer als keine Schicht. Operators vertrauen dem grünen Signal. Dieselbe Anti-Pattern-Klasse wie CLAUDE.md §20.8 (silent fail-open in Enforcement-Schicht).
**Wie vermeiden:** Loaders die intentional fail-open sind, müssen einen erkennbaren Fallback-Zustand produzieren (Sentinel-Wert, `loaded_clean: bool`-Flag, oder explizites Exception-Raise bei Corruption). API-Endpoints über fail-open Loaders müssen den Fallback detektieren und korrekt reporten. Hier: `load_ledger_state(jpath, start_capital=-1.0)` + Check `cash < 0 AND updated_utc is None`.
**Erkannt in:** Paket 5 Commit `190fc1dd`, Stage-2 F-senior-1 MAJOR → Fix via Sentinel-Approach in `src/assembled_core/api/routers/ledger.py`.
**Referenzen:** Stage-2 senior-review 2026-05-28, CLAUDE.md §20.8 (fail-open anti-pattern).

---

## E-026 — Unauthenticated Health Endpoint Amplifies Upstream API Calls
**Datum:** 2026-05-28
**Kategorie:** risk-execution / resource-exhaustion
**Was passierte:** `GET /health` (unauthenticated, kein Rate-Limit) rief `AlpacaAdapter().health_check()` bei jedem Request auf — macht einen outbound API-Call zu api.alpaca.markets. Standard Kubernetes/Uptime-Monitoring schlägt `/health` alle 10-30s → 2880-8640 Broker-API-Calls/Tag, bevor überhaupt menschlicher Traffic eintrifft. Hätte Alpaca-Quota exhausted und potentiell 429-Locks während Live-Trading-Fenstern verursacht.
**Warum falsch:** Health-Endpoints sind extern exponiert und amplizieren jeden per-Request-Seiteneffekt. Ein 429-Lock bei Alpaca während eines Trading-Fensters blockiert Live-Orders — das ist ein Risk-Execution-Impact, nicht nur ein Performance-Problem.
**Wie vermeiden:** Health-Endpoints dürfen nur lokalen State lesen (Filesystem, In-Process). Jeder Check der einen externen Service berührt (Broker, FRED, IEX, etc.) muss entweder: (a) mit TTL gecacht werden, (b) opt-in via Query-Param sein (default: skip), oder (c) auf einem separaten authentifizierten `/health/deep`-Endpoint liegen. Hier: `?check_broker=true` opt-in, default returns `{ok: null, detail: "skipped"}`.
**Erkannt in:** Paket 5 Commit `190fc1dd`, Stage-2 F-senior-3 MAJOR → Fix via `check_broker: bool = Query(default=False)` in `src/assembled_core/api/routers/health.py`.
**Referenzen:** Stage-2 senior-review 2026-05-28, CLAUDE.md §6.1 (sensible Kernbereiche: execution).

---

## E-027 — Walk-Forward: Lagged Weights computed inside Test Slice discard Warmup-End Carry-In
**Datum:** 2026-05-28
**Kategorie:** logic-error / backtest-bias
**Was passierte:** In `_simulate_vol_target()` wurden `w_spy_lag = test_df["w_spy"].shift(1).fillna(0.0)` und `turnover = test_df["w_spy"].diff().abs().fillna(...)` NACH dem Slice `test_df = combined[test_mask]` berechnet. Damit trägt `shift(1)` am ersten Test-Bar NaN → fillna(0.0) → die Strategie startet jeden Fold von 0% SPY, unabhängig vom echten Position-Stand am Ende der Warmup-Phase. Turnover-diff am ersten Test-Bar wurde ebenfalls falsch berechnet (als ob von Cash-Zustand gestartet wird, nicht von Warmup-End-Gewicht).
**Warum falsch:** Ein OOS Walk-Forward repliziert den realen Betrieb: die Strategie läuft kontinuierlich, der Test-Window ist nur eine Beobachtungs-Periode. Weights und Turnover müssen den Zustand aus dem Warmup korrekt übernehmen. Ohne das erscheint jeder Fold als Cold-Start → Rendite und Kosten systematisch verzerrt (je nach Fold unterschiedlich, daher schwer diagnostizierbar).
**Wie erkennen:**
- `shift(1)` oder `.diff()` nach einem Slice-Filter statt auf dem vollen Frame.
- Fillna(0.0) für Lag-Weights in einem WF-Kontext (außer am echten ersten Bar des Gesamtdatenrahmens).
- Fold-1-Renditen unterscheiden sich stark je nach wie aggressiv die Warmup-End-Position ist.
**Wie vermeiden:**
- Lag-Weights, Turnover und Returns IMMER auf dem **vollen combined Frame** (warmup + test) berechnen, DANN erst auf den Test-Window slicen.
- `fillna` für Lags nur am allerersten Bar des gesamten Frames, nicht am ersten Bar jedes Folds.
- Assertion: `assert test_df["w_spy_lag"].iloc[0] != 0.0 or warmup_end_weight == 0.0` — der erste Test-Bar sollte nur dann 0 sein, wenn die Strategie tatsächlich am Ende der Warmup-Phase 0% hielt.
**Erkannt in:** `scripts/_oos_wf_vol_target_overlay.py` → Stage-2 F-senior-1 BLOCKER → Fix in Commit `90ec835c` (lag/turnover auf full combined frame).
**Referenzen:** Stage-2 senior-review 2026-05-28.

## E-028 — Silent Default Weights in WF Replay trivially satisfy Weight-Sum Assertion
**Datum:** 2026-05-28
**Kategorie:** silent-degradation / test-anti-pattern
**Was passierte:** In `_simulate_vol_target()` wurden Warmup-Rows im `combined` DataFrame mit `fillna(0.0)` für w_spy und `fillna(1.0)` für w_ief befüllt (Pre-Signal-Default: 100% IEF, 0% SPY). Die nachfolgende Assertion `(combined["w_spy"] + combined["w_ief"] - 1.0).abs().max() < 1e-6` passt trivialerweise: 0+1=1, egal was. Das bedeutet: wenn Signale im Test-Window ausgeblieben wären (z.B. durch Warmup-Fehler oder Strategie-Bug), hätte die Assertion NICHT gefeuert — die Strategie wäre still mit 100% IEF gelaufen ohne Warnung.
**Warum falsch:** Assertions die durch die Defaults, die sie schützen sollen, trivialerweise erfüllt werden, sind Alibis, keine Garantien. Eine Assertion die "nie feuert" gibt falsches Vertrauen.
**Wie erkennen:**
- Post-fillna-Assertion auf eine Summe/Constraint die durch die fillna-Defaults selbst trivialerweise erfüllt wird.
- `assert (a + b == 1.0)` wenn a=0 und b=1 die fillna-Defaults sind.
**Wie vermeiden:**
- Assertion vor fillna oder auf Basis der raw reindex-Werte (ohne Fallback) stellen.
- Zusätzlich: assertieren dass im Test-Window echte (nicht-default) Signale vorhanden sind. Z.B. `spy_sigs.reindex(test_rows).notna().any()`.
- Für zweiwertige Complement-Systeme (b = 1-a): nur eine der beiden Größen assertieren; die andere ist rechnerisch gebunden.
**Erkannt in:** `scripts/_oos_wf_vol_target_overlay.py` → Stage-2 F-senior-3 MAJOR → Fix via `spy_sigs_in_test.isna().all()` Guard in Commit nach `90ec835c`.
**Referenzen:** Stage-2 senior-review 2026-05-28, E-025 (Loader Fail-Open), CLAUDE.md §20.8.

## E-029 — Variable nur in einem Condition-Branch definiert → NameError in komplementärem Branch
**Datum:** 2026-05-28
**Kategorie:** logic-error / runtime-crash
**Was passierte:** In `_write_report()` war `dd_ratio_val` nur innerhalb `if not (np.isnan(...)) and abs(mean_dd_spy) > 1e-6:` definiert. Der `elif sharpe_criterion:` Branch referenzierte `dd_ratio_val` in einem f-string. Der `elif`-Branch ist exklusiv mit `if dd_criterion:` → d.h. er wird genau dann erreicht, wenn `dd_criterion=False` → genau dann, wenn `dd_ratio_val` NICHT definiert wurde → `NameError`. Der Fehler blieb unsichtbar, weil in den Test-Runs immer `dd_criterion=True` war (MaxDD-Verbesserung > 30%).
**Warum falsch:** `NameError` im Report-Writer bedeutet: wenn eine Kombination aus Strategie-Performance (Sharpe ok, DD nicht) vorkommt, crasht die Analyse ohne Output. Genau die Kombination, die ein interessantes "teilweise gut"-Signal wäre, produziert einen Fehler statt einen Report.
**Wie erkennen:**
- Variable in `if a:` definiert, in `elif not a:` (oder `else:`) referenziert.
- Linter (pylance, mypy) zeigt "possibly undefined" für die Variable.
- Test-Suite hat keine Tests die den `elif`-Branch mit den Daten ausüben, die `a=False` ergeben.
**Wie vermeiden:**
- Alle Variablen die in Conditional-Branches genutzt werden IMMER vor dem ersten Branch initialisieren (z.B. `dd_ratio_val = float("nan")`).
- Für Report-Writer: am Ende der Berechnung alle Template-Variablen auflisten und sicherstellen, dass jede in allen Code-Pfaden einen definierten Wert hat.
- Test: mindestens ein Test pro Branch-Kombination in Report-Writern mit variablen Metriken.
**Erkannt in:** `scripts/_oos_wf_vol_target_overlay.py` → Stage-2 F-senior-4 MAJOR → Fix: `dd_ratio_val = float("nan")` vor Branch in Commit `90ec835c`.
**Referenzen:** Stage-2 senior-review 2026-05-28.

## E-030 — bfill auf Pivot-Panel leakt Future-Prices bei gestaffelter Asset-Inception
**Datum:** 2026-05-29
**Kategorie:** pit-violation / look-ahead-bias
**Was passierte:** In `dual_momentum.py` wurde `pivot = pivot.ffill().bfill()` auf das Wide-Preis-Panel angewendet. Wenn ein Asset einen späteren Inception-Date hat als der Panel-Start (z.B. BIL ab 2007-05-25, Panel ab 2007-01-01), erzeugt das Pivoting leading NaN-Rows fuer dieses Asset. `.bfill()` fuellt diese NaN rueckwaerts auf mit dem ersten verfuegbaren Preis — einem Preis, der zum Zeitpunkt der NaN-Bars noch nicht existiert hat. Das korrumpiert 12M-Return-Berechnungen auf den fruehesten Rebalance-Bars: BIL erscheint als "flat" obwohl es noch gar kein Preissignal gab.
**Warum falsch:** `.bfill()` auf einem Panel mit gestaffelten Inceptions propagiert Zukunftsinformation in die Vergangenheit — klassischer Look-Ahead-Bias. Das ist besonders tückisch weil es nur die fruehesten Bars betrifft (vor dem spaetesten Inception-Date), die Tests mit gleichem Startdatum fuer alle Assets das Problem unsichtbar machen.
**Wie erkennen:**
- Preis-Panel mit unterschiedlichen Inception-Dates der Symbole (z.B. ETFs mit gestaffeltem Launch).
- `pivot.ffill().bfill()` auf einem solchen Panel.
- Test-Fixture alle Symbole mit gleichem Startdatum → Bug ist unsichtbar.
**Wie vermeiden:**
- Nur `.ffill()`, niemals `.bfill()` auf Multi-Asset-Pivots mit gestaffelten Inceptions.
- Test-Fixture explizit mit gestaffelten Startdaten bauen (PIT-Test 7 in `test_dual_momentum_pit_safety.py` als Vorlage).
- Wenn Leading-NaN-Rows ein Problem sind: erst nach dem spaetesten Inception-Date schneiden (explizit, mit Log-Warning), nicht blind bfill-en.
**Erkannt in:** `src/assembled_core/strategies/dual_momentum.py` → Stage-2 F-senior-1 MAJOR → Fix: `.bfill()` entfernt, Commit `feat(dual_momentum)`.
**Referenzen:** Stage-2 senior-review 2026-05-29.

## E-031 — EOM-Detection mit month.values strippt Jahresinfo → falsche Identifikation bei Luecken
**Datum:** 2026-05-29
**Kategorie:** logic-error / latent-bug
**Was passierte:** In `dual_momentum.py` wurde EOM (End-of-Month) via `months = dates.month.values; eom_flags[:-1] = months[:-1] != months[1:]` detektiert. Das funktioniert fuer taegliche Business-Day-Daten korrekt (Dezember→Januar immer 12!=1). Aber bei Lueckendaten oder nicht-taeglich abgetasteten Panels wuerde z.B. Jan-2021 → Jan-2022 als "kein Monatswechsel" behandelt (beide month=1), was EOM-Bars ausblendet. Auch Dezember-2021 → Dezember-2022 waere ein falsch-negativer Monatswechsel.
**Warum falsch:** `month.values` wirft Jahresinformation weg. Gleiches Monat in verschiedenen Jahren ist kein "selber Monat" fuer den Zweck der EOM-Detektion. Bei daily Business-Day-Data ist die Wahrscheinlichkeit des Fehlers klein (Wochen ohne Jahreswechsel ueberbruecken nie > 12 Monate), aber bei gappigen oder monats-gesampleten Daten ist es ein echter Bug.
**Warum nicht schon im echten Run aufgefallen:** Alpaca-Daily-Daten 2016-2025 haben keine Luecken gross genug, um das Problem zu triggern. Der Fehler ist latent fuer gappige Panels oder seltene Assets.
**Wie vermeiden:**
- `keys = dates.year.values * 12 + dates.month.values` statt `months = dates.month.values` fuer EOM-/Periodenvergleiche.
- Allgemein: bei Zeitreihen-Vergleichen nie Jahresinformation wegwerfen, auch wenn der "normale" Fall taegliche Daten sind.
**Erkannt in:** `src/assembled_core/strategies/dual_momentum.py` → Stage-2 F-senior-3 MAJOR → Fix: year*12+month, Commit `feat(dual_momentum)`.
**Referenzen:** Stage-2 senior-review 2026-05-29.

## E-032 — Windows astype(int) auf ms-Epoch overflows int32
**Datum:** 2026-05-29
**Kategorie:** pandas-pitfall / platform-divergence
**Was passierte:** In `scripts/crypto_funding_carry_backtest.py` wurden Binance-Timestamps (ms seit Epoch, ~1.57×10¹² ms fuer 2019-Daten) via `df["fundingTime"].astype(int)` konvertiert. Auf Windows mappt `astype(int)` zu numpy `int32` (max ~2.1×10⁹), was den Wert auf einen kleinen positiven Rest wrapt. `pd.to_datetime(..., unit="ms")` interpretierte die getruncten Werte als ~5 Tage nach Epoch → alle Timestamps landeten in 1970-01-06 statt 2019.
**Warum falsch:** Python's `int` ist unbegrenzt, aber numpy mappt `int` auf den Platform-Default-Integer-Typ. Auf Windows 64-bit ist `numpy.int_` = `int32`, nicht `int64`. Werte > ~2.1 Mrd. (inklusive alle ms-Epoch-Timestamps nach 1970-01-25) laufen ueber.
**Wie vermeiden:**
- Immer explizit `astype("int64")` (String-Notation) verwenden, wenn die Spalte ms- oder ns-Epoch-Werte halten koennte.
- Niemals `astype(int)` oder `astype(np.int_)` fuer Zeitstempel-Mathe auf Windows.
- Auf Linux ist `np.int_` = `int64`, daher taucht der Bug nur auf Windows auf → klassische Local-vs-CI-Divergenz bei gemischten Umgebungen.
**Erkannt in:** `scripts/crypto_funding_carry_backtest.py` → beide `astype(int)` in fetch_funding_rates() + fetch_klines() → Fix: `astype("int64")`.
**Referenzen:** Stage-2 senior-review 2026-05-29, Diagnostic-Run F-032.

## E-033 — pyarrow datetime64[ns, UTC] Round-Trip-Korruption auf Windows
**Datum:** 2026-05-29
**Kategorie:** pandas-pitfall / platform-divergence
**Was passierte:** In `scripts/crypto_funding_carry_backtest.py` wurden korrekte `datetime64[ns, UTC]`-Spalten via `df.to_parquet()` gespeichert und via `pd.read_parquet()` eingelesen. Auf Windows (bestimmte pyarrow-Versionen) lieferte der Read-Back Timestamps nahe Epoch-0 (1969-12-07 statt 2019-09-10). Die In-Memory-Werte waren korrekt; nur die Parquet-Persistenz war kaputt.
**Warum falsch:** pyarrow auf Windows behandelt `datetime64[ns, UTC]`-Daten in bestimmten Versionen anders als auf Linux (Endianness, tz-metadata, ns-precision). Stille Datenverfaelschung, kein Exception.
**Wie vermeiden:**
- tz-aware datetime-Spalten nicht direkt als solche in Parquet persistieren, wenn plattformuebergreifende Kompatibilitaet wichtig ist.
- Stattdessen: Timestamp als `int64 ms` (Epoch-Millisekunden) speichern, nach dem Load via `pd.to_datetime(col, unit="ms", utc=True)` rekonstruieren.
- Workaround-Pattern: `_save_parquet()` + `_load_parquet()` aus `scripts/crypto_funding_carry_backtest.py` als Template.
- Gegencheck: nach erstem Write sofort Read-Back und `assert df["timestamp"].min().year > 1972` o.ae.
**Erkannt in:** `scripts/crypto_funding_carry_backtest.py` → `load_or_fetch()` → Fix: `_save_parquet()` speichert `timestamp_ms: int64`, `_load_parquet()` rekonstruiert.
**Referenzen:** Stage-2 senior-review 2026-05-29, Fix in Commit `7909bd99`.

## E-034 — Governance-Restrukturierung lässt inbound §-Anker und cross-file Faktenbehauptungen verwaisen
**Datum:** 2026-05-30
**Kategorie:** governance-drift / docs-consistency
**Was passierte:** Bei der CLAUDE.md-Verschlankung (985→215 Zeilen) wurden Sektionsnummern entfernt und Inhalte in PROJEKT_STATUS.md / ARCHITECTURE_BACKEND.md / review_chain_disclosure.md ausgelagert; parallel wurden in `.cursorrules` nicht-existente Legacy-Script-Refs entfernt. Nicht mitgezogen: (a) AGENTS.md behauptete weiter, drei gelöschte Sprint-Scripts „existieren noch im Repo" und würden von `run_all_sprint10.ps1` referenziert (beides per Glob/Grep widerlegt); (b) `docs/review_chain_disclosure.md` verwies forward auf `§2.2`/`§20.6`, die in der verschlankten CLAUDE.md keine gültigen Anker mehr sind; (c) ~11 weitere Docs tragen inbound `CLAUDE.md §X.Y`-Refs (MNPI/§7.3, leverage/§3.5, PIT/§7.2), deren Ziel-Nummern entfielen. Kein Content verloren — nur Anker und Faktenbehauptungen drifteten.
**Warum falsch:** Governance-Dateien sollen *eine* Source of Truth bilden. Verwaiste Anker und cross-file Widersprüche untergraben genau die Steuerfunktion, für die diese Dateien existieren. Ein späterer Agent liest die falsche Behauptung („Script existiert") als Repo-Wahrheit. Die Review-Chain fing das (Stage 1+2+3, F-DGS-1/2, F-senior-1/2, F-auditor-1/2/5) — ohne Chain wäre es still durchgegangen.
**Wie vermeiden:**
- Bei jeder Restrukturierung, die Sektionsnummern oder Pfad-/Script-Listen ändert: `Grep "CLAUDE.md §"` über das gesamte Repo und `Grep` nach jedem entfernten Pfadnamen, *bevor* der Step als fertig gilt.
- Schwester-Governance-Dateien (AGENTS.md, `.cursorrules`, `.claude/rules/`) im selben Step synchronisieren, nicht später.
- Jede Faktenbehauptung über Dateiexistenz per `Glob`/`Grep` verifizieren statt aus Altdoku übernehmen.
- Inbound-Anker auf Überschriftennamen statt §-Nummern umstellen, wenn die Zielnummerierung wegfällt.
**Erkannt in:** `AGENTS.md` (Legacy-Scripts-Block), `docs/review_chain_disclosure.md` (Z.6/40), `.cursorrules` (Bootstrap-Liste) → Fixes im selben Step; ~11-Docs-Anker-Sweep als getrackter Follow-up (Rule 60). **Schlimmster Fall in EXECUTABLE CODE:** `system_check/runner/brief_builder.py` extrahierte Mission/Architektur/Sensible-Zonen per `### 1.3` / `5.1` / `6.1`-Regex aus CLAUDE.md; nach der Verschlankung lieferte der `6.1`-Fallback nur **2 von 6** Schutzpfaden an jeden Tournament-Agent — stille Sicherheits-Degradation, kein bloßer Doku-Widerspruch. Fix 2026-05-31: Regex auf Überschriften (`## Projekt` / `## Sensible Zonen`) bzw. `docs/ARCHITECTURE_BACKEND.md` umgestellt, beide Fallbacks auf alle 6 Pfade vervollständigt, 2 Regressions-Tests.
**Referenzen:** Review-Chain 2026-05-30 (Stage 1 docs-governance-sync, Stage 2 senior-code-reviewer, Stage 3 task-completion-auditor, Verdict CONDITIONAL → in-scope Fixes adressiert).

## E-035 — Research-Harness fährt `run_trading_cycle` und persistiert time-traveled Overlay-State in LIVE `output/ops/`
**Datum:** 2026-05-31
**Kategorie:** state-corruption / backtest-side-effect
**Was passierte:** Ein read-only OOS-Walk-Forward-Harness in `scripts/` (`_oos_wf_pipeline_realistic.py`) trieb den LITERALEN Produktions-Cycle (`run_portfolio_backtest(cycle_fn=make_cycle_fn(...))` → `run_trading_cycle`). Der Cycle ruft per Default die Crisis-Alpha-Overlay-Pipeline auf, deren Step-7 (`pipeline.py:209-211`, gated nur durch `if not dry_run`) den State in die LIVE-Datei `output/ops/crisis_alpha_state.json` schreibt (`_DEFAULT_STATE_PATH`). Da policy.yaml `intel.crisis_alpha.shadow_only=false` hat, war `dry_run=False` → jeder historische `as_of`-Schritt überschrieb den realen Paper-Pilot-State mit time-traveled Records (beobachtet: `entered_at_utc=2025-01-11` bei `last_evaluated_utc=2020-01-02`, COOLDOWN). Der Harness-Docstring behauptete gleichzeitig „touches NO production module, policy.yaml, or state" — nachweislich falsch. (`output/state/risk_state.json` blieb unberührt: Backtest-Mode triggert den live-gated `save_risk_state`-Pfad nicht — Datei trug echten `2026-05-10`-Stempel.)
**Warum falsch:** Ein Backtest darf NIE live operativen State mutieren. Hier war der Write ein inerter Side-Effect (Overlay ging bei `geo_score=0` nie ACTIVE → 0 Targets; `lowmax_lo`-Re-Run MIT Isolation reproduzierte byte-identische Pooled-Zahlen 0.92/7.2% → bewiesen kein Result-Input). Aber: hätte das Overlay Entries erzeugt, wäre der live State-Machine korrumpiert worden; ein anderer Harness könnte den time-traveled State als INPUT zurücklesen (Look-Ahead/Replay-Kontamination). Der DMS/Paper-Pilot liest diese Dateien.
**Wie vermeiden:**
- Vor dem Treiben von `run_trading_cycle` aus JEDEM `scripts/`-Harness: `os.environ.setdefault("ASSEMBLED_NO_CRISIS_OVERLAY", "1")` VOR den Produktions-Imports setzen (sanktionierter Escape-Hatch: `_tc_sizing.py:1632-1637` OR-t ihn in `shadow_only` → `dry_run` → kein Persist). Kein Prod-Edit, kein Monkeypatch.
- Generelle Regel: Harness, die den Live-Cycle treiben, müssen JEDEN Overlay-Persistenzpfad auditieren (nicht nur `risk_state.json`) und asserten, dass nichts unter `output/ops/` oder `output/state/` landet. Loader-Default prüfen: `load_crisis_state` re-init't auf WATCH bei fehlender/korrupter Datei (`state_machine.py:124-138`) — korrupte Datei daher löschen statt fabrizierten „korrekten" Vorzustand erfinden.
- Docstring-/Report-Claims über State-Berührung empirisch verifizieren (Run-Log auf `[CRISIS_STATE] saved` grepen), nicht behaupten.
**Erkannt in:** `scripts/_oos_wf_pipeline_realistic.py` (Stage-1 `risk-execution-reviewer` BLOCKER B1). Fix im selben Step: Env-Isolation gesetzt + korrupte `crisis_alpha_state.json` gelöscht (re-init WATCH) + Docstring/Report-Footer korrigiert + Re-Run-Verifikation (0 State-Writes, identische Zahlen).
**Referenzen:** Review-Chain 2026-05-31 (Stage 1 risk-execution-reviewer + test-runner, Stage 2 senior-code-reviewer E-NEW-1, Stage 3 task-completion-auditor).

## E-036 — `os.environ.setdefault` als State-Isolations-Guard ist ein stiller No-op, wenn die Variable bereits gesetzt ist
**Datum:** 2026-06-03
**Kategorie:** logic-error / state-isolation
**Was passierte:** Research/OOS-Harnesses nutzten `os.environ.setdefault("ASSEMBLED_NO_CRISIS_OVERLAY", "1")`, um Live-State-Writes zu blocken (der E-035-Schutz). `setdefault` ist aber ein No-op, sobald die Variable schon existiert (z. B. ein Shell-/CI-`export` von `"0"`) — der Guard reaktiviert dann genau den Side-Effect still wieder, den er verhindern soll.
**Warum falsch:** Ein defensiver Isolations-Guard, den die ambient Environment still aushebeln kann, ist schlimmer als gar kein Guard — er suggeriert eine Sicherheit, die nicht existiert. Der E-035-Schutz war damit unter realistischen CI-/Shell-Bedingungen wirkungslos.
**Wie vermeiden:** Für MANDATORY-Isolation unbedingte Zuweisung + Warn-on-Conflict verwenden, nicht `setdefault`: `if prev not in (None, "1"): warn(...); os.environ[k] = "1"`. `setdefault` ist nur korrekt, wenn Caller-Overrides *honoriert* werden sollen — das ist hier die gegenteilige Intention.
**Erkannt in:** `scripts/_oos_wf_pipeline_realistic.py`, `scripts/_oos_wf_etf_pairs_literal.py`, `scripts/_oos_wf_dual_momentum_literal.py`. Fix in Commit `4f23f532`.
**Referenzen:** E-035; senior-code-reviewer Batch 10 (Commit `4f23f532`).

## E-037 — Geteilter On-Disk-Cache mit reiner Symbol-Validitätsprüfung → Cross-Producer-Kontamination
**Datum:** 2026-06-03
**Kategorie:** logic-error / cache-correctness
**Was passierte:** Drei OOS-Harnesses mit unterschiedlichen Symbol-Sets / Datumsfenstern schrieben und lasen denselben Parquet-Preis-Cache (`oos_alpaca_prices_cache.parquet`), validiert nur über Symbol-Präsenz — so konnte Harness A still die Preise laden, die Harness B gefetcht hatte. Das Ergebnis einer Studie hing damit unsichtbar vom vorherigen Lauf einer anderen ab.
**Warum falsch:** Symbol-Präsenz ist keine hinreichende Cache-Validität; Datumsbereich und Producer-Identität sind entscheidend. Stille Wiederverwendung eines Caches mit falscher Coverage vergiftet OOS-Inputs unsichtbar — ein Falsifikationsergebnis kann so durch fremde Daten verfälscht werden.
**Wie vermeiden:** Geteilte On-Disk-Caches per Producer-Identität + Parametern keyen (z. B. `script_id` + End-Datum im Dateinamen) oder den abgedeckten Datumsbereich (min/max) gegen das benötigte Fenster validieren — nicht nur Symbol-Präsenz.
**Erkannt in:** `scripts/_oos_wf_mfv2.py`, `scripts/_oos_wf_mfv2_full.py`, `scripts/_oos_wf_mfv_long_short.py`. Fix in Commit `4f23f532`.
**Referenzen:** senior-code-reviewer Batch 10 (Commit `4f23f532`).

## E-038 — Timestamp vor einem zweiseitigen `[cutoff, as_of]`-PIT-Fenster forward-shiften zieht zu alte Beobachtungen herein
**Datum:** 2026-06-03
**Kategorie:** logic-error / pit-violation / look-ahead
**Was passierte:** Ein Release-/Publication-Lag wurde umgesetzt, indem der Beobachtungs-Timestamp um N Tage *forward* geschoben wurde (`timestamp += lag`), **bevor** ein zweiseitiger Filter `(timestamp <= as_of) & (timestamp >= cutoff)` (cutoff = as_of − lookback_days) lief. Die Annahme war „ein Forward-Shift kann Zeilen nur entfernen/verzögern, nie Signal injizieren". Falsch: derselbe Shift zieht an der **unteren** Grenze Beobachtungen, deren Roh-Datum *älter* als cutoff war (vorher korrekt ausgeschlossen), in das Lookback-Fenster **hinein** — und verändert damit die zurückgegebenen Zeilen und jede darauf berechnete History-Statistik (z. B. rollende z-Scores). Auf dem Live-mfv2-Pfad (macro-Faktoren w=0.03/0.02) hätte das stille Stale-History-Admission einen nicht-null-gewichteten Produktionsfaktor verschoben.
**Warum falsch:** Ein Availability-/Release-Lag ist ausschließlich ein **Obergrenzen-Konzept** (as_of). Die untere (Lookback-)Grenze muss weiterhin gegen das **Roh-Beobachtungsdatum** vergleichen. Ein monotoner Forward-Shift bewegt Zeilen relativ zu *beiden* Grenzen Richtung as_of: er entfernt oben, admittiert aber unten. Zusätzlich korrumpiert In-place-Mutation der Timestamp-Spalte das zurückgegebene Schema für jeden Downstream, der erneut auf dem Timestamp alignt (Double-Lag-Risiko).
**Wie vermeiden:** Grenzen trennen: `available = roh + lag` nur für den `(<= as_of)`-Test; das Roh-Datum für den `(>= cutoff)`-Test. Nie die Quell-Timestamp-Spalte für einen Filter mutieren — eine lokale Availability-Serie berechnen. Eine **untere**-Grenze-Regression ergänzen (Obs mit Roh-Datum vor cutoff, dessen geshifteter Wert ins Fenster fällt → muss ausgeschlossen bleiben) plus einen `lag=0`-Legacy-Äquivalenz-Test.
**Erkannt in:** `src/assembled_core/data/altdata_loader.py` (`load_macro_indicators`). Caught von der Review-Chain (Stage-2 senior + risk-execution-reviewer) als MAJOR, Fix in Commit `fd8a192c`.
**Referenzen:** E-030 (Look-Ahead-Klasse); E-GPR-1 (release_lag_days); Review-Chain Batch 12 (Workflows `wpt79crgz` + `wz8dz6v1a`).

## E-039 — Report-Sink rekonstruiert eine bereits gegradete Status-Größe aus dem Rohergebnis und überschreibt sie still
**Datum:** 2026-06-04
**Kategorie:** silent-degradation / status-integrity
**Was passierte:** `build_ledger_from_trades` (ledger_integration) gradet auf dem paper_view-Pfad (kein unabhängiger Broker-Snapshot) bewusst `reconciliation_ok=None` + `reconciliation_severity="unverified"` — eine Selbst-vs-Selbst-Reconcile liefert kein echtes Drift-Signal. Diese gegradete Größe wurde die Call-Chain hinabgereicht. Aber die SUMMARY-Zeile von `write_accounting_report_csv/json` rekonstruierte `reconciliation_ok` lokal neu aus `reconciliation_result.get("ok")` — dem Roh-Selbstvergleich, der trivial `True` ist. Das EOD-Artefakt zeigte dadurch einen „healthy pass" auf einem paper_view-Run, obwohl der authoritative Grade `None/unverified` war. Der Report-Sink überschrieb still die Wahrheit, die weiter oben in der Pipeline bereits korrekt bestimmt worden war.
**Warum falsch:** Wenn eine Status-Größe einmal autoritativ gegradet wurde (hier: in ledger_integration), darf ein nachgelagerter Sink sie nicht aus dem Rohergebnis neu ableiten. „Roh-Ergebnis vorhanden" und „gegradeter Status" sind zwei verschiedene Dinge; die Neuableitung am Sink ignoriert genau die Degradations-Logik, für die das Grading existiert. Besonders perfide: das Rohergebnis ist hier ein Selbstvergleich (Ledger gegen sich selbst) → `ok=True` ist tautologisch → das Override produziert ein falsches grünes Signal, dem Operators vertrauen.
**Wie erkennen:**
- Eine Größe wird oben in der Call-Chain explizit gegradet (None/unverified/severity), aber ein Report-/Artefakt-Writer berechnet `x = raw_result.get("ok")` erneut, statt den durchgereichten Grade zu verwenden.
- CSV- und JSON-Writer (oder zwei Sinks) leiten dieselbe Größe unabhängig ab → Divergenz-Risiko zwischen Artefakten.
- Ein „pass"-Artefakt auf einem Pfad, dessen Underlying-Vergleich strukturell ein Selbstvergleich ist (paper-vs-paper, A-vs-A) — der niemals `False` werden kann.
**Wie vermeiden:**
- Den gegradeten Wert explizit durch die Call-Chain bis zum Sink **threaden** und dort verwenden, nicht neu ableiten. Hier: `reconciliation_ok`-Parameter an `write_accounting_report_csv/json` durchgereicht.
- „Nicht übergeben" von „gegradet None" mit einem **Sentinel** (`_RECON_OK_UNSET = object()`) unterscheiden, damit ein bewusstes `None` nicht als „kein Wert → fallback auf Rohergebnis" missverstanden wird. Ein nackter `None`-Default kollabiert beide Fälle.
- Sinks, die dieselbe Größe schreiben (CSV/JSON), durch **einen gemeinsamen Resolver** speisen (Parität), nie zwei unabhängige Ableitungen.
- Den Block/die Zelle auch bei „graded-only call" (gegradeter Wert vorhanden, aber kein Rohergebnis-Dict) emittieren — sonst fällt der Status auf einem Sink still weg.
**Erkannt in:** `src/assembled_core/accounting/accounting_report.py` (`write_accounting_report_csv/json` SUMMARY-Zeile), gegradet in `ledger_integration.build_ledger_from_trades` → Stage-2 senior + risk-execution-reviewer (incomplete-fix: erster Pass ließ den Report-Sink noch auf `True`), Fix in Commit `58e6ca91` (Sentinel `_RECON_OK_UNSET` + `_resolve_summary_reconciliation_ok` + Thread durch die Call-Chain).
**Referenzen:** E-025 (Loader Fail-Open maskiert Korruption), E-028 (Default erfüllt Assertion trivial); CLAUDE.md §20.8 (silent fail-open); Rule 30 (accounting-Schutzzone).

## E-040 — `monkeypatch.delenv(..., raising=False)` auf einer abwesenden Variable registriert kein Undo → Forward-Leak in spätere Tests
**Datum:** 2026-06-04
**Kategorie:** test-anti-pattern / test-isolation
**Was passierte:** `test_set_deterministic_sets_env_and_seeds_numpy` löschte vor dem Aufruf drei Env-Keys via `monkeypatch.delenv(k, raising=False)`, um saubere Preconditions herzustellen. Der Code-under-Test (`set_deterministic`) mutiert `os.environ` aber DIREKT (nicht via monkeypatch) für FÜNF Keys. Problem: pytest-`monkeypatch.delenv(name, raising=False)` auf einem **bereits abwesenden** Key registriert **keinen Undo-Eintrag** (es gibt nichts wiederherzustellen). Der anschließend vom Production-Code geschriebene Wert (`os.environ[k]=v`) wird daher von monkeypatch beim Teardown NICHT zurückgerollt → er leakt in spätere Tests (verifiziert: `OMP_NUM_THREADS`/`MKL_CBWR`/`CUBLAS_WORKSPACE_CONFIG` blieben gesetzt) → order-abhängige Flakiness.
**Warum falsch:** `monkeypatch` rollt nur Mutationen zurück, die es selbst vorgenommen hat. Eine Direkt-Mutation von `os.environ` durch den getesteten Code ist für monkeypatch unsichtbar; ein vorausgehendes `delenv(raising=False)` auf einen abwesenden Key erzeugt KEINEN Snapshot, der diese spätere Direkt-Mutation neutralisieren würde. Der Test sieht hermetisch aus, ist es aber nicht — globaler Prozess-State (`os.environ`) leakt zwischen Tests, was Test-Reihenfolge-Abhängigkeit und Heisenbug-Flakiness erzeugt.
**Wie erkennen:**
- Test löscht Env-Keys via `monkeypatch.delenv(..., raising=False)`, aber der Code-under-Test schreibt `os.environ[...]` direkt (statt über monkeypatch).
- Ein Test, der isoliert grün ist, aber in bestimmter Reihenfolge (oder unter `-p no:randomly` vs. zufälliger Reihenfolge) einen anderen Test umkippt.
- Anzahl der vom Production-Code mutierten Keys > Anzahl der im Test `delenv`'ten Keys (hier 5 vs. 3).
**Wie vermeiden:**
- Bei Code, der globalen Prozess-State direkt mutiert, NICHT auf `monkeypatch.delenv` zur Restoration vertrauen. Stattdessen: alle betroffenen Keys explizit **snapshotten** (`{k: os.environ.get(k) for k in keys}`) und per `request.addfinalizer` zurücksetzen (`pop` wenn vorher abwesend, reassign wenn vorher gesetzt) — unabhängig davon, wie der Wert hineinkam.
- Die Key-Liste im Test muss die VOLLE Menge der vom Production-Code berührten Keys spiegeln (hier `reproducibility.desired_env`), nicht nur eine Teilmenge.
- Den Test gegen ein vorab verschmutztes ambient Env beweisen (before==after), nicht nur in einer sauberen Umgebung.
**Erkannt in:** `tests/test_audit_additions.py::test_set_deterministic_sets_env_and_seeds_numpy` (Code-under-Test: `src/assembled_core/reproducibility.set_deterministic`) → Fix in Commit `a31f0c5d` (5-Key-Snapshot + `request.addfinalizer`-Restore statt `monkeypatch.delenv`).
**Referenzen:** E-021 (Self-verifying Logging-Test), E-012 (`date.today()` Lokalzeit-Leak in Modul-State); Rule 40 (Test-Honesty).

## E-041 — Fehlende Ordering-/Timestamp-Spalte → `else`-Branch liest still den Dataset-TAIL ohne as_of-Filter (latenter Look-Ahead)
**Datum:** 2026-06-04
**Kategorie:** pit-violation / silent-degradation / look-ahead
**Was passierte:** `_populate_sector_rotation_scores` (paper/intel_context) wendete den as_of-PIT-Filter nur an, wenn die Scores-Frame eine Timestamp-Spalte (`_pit_ts`) trug. Fehlte diese Spalte (z. B. weil `compute_sector_scores` eine anders/nicht benannte ts-Spalte emittiert, obwohl die Input-Preise eine trugen), griff der `else`-Branch zu `scores_df.iloc[-1]` — also dem **Dataset-TAIL** ohne jeden as_of-Bezug. In einem Replay/Backtest ist dieser Tail das Dataset-ENDE, nicht der as_of-Bar → latenter Look-Ahead. Live/EOD war zufällig korrekt, weil dort Tail == as_of.
**Warum falsch:** Wenn die Spalte fehlt, auf der die PIT-Filterung beruht, ist der korrekte Zustand „kann nicht PIT-filtern", nicht „nimm das letzte verfügbare". Der stille Fallback auf `iloc[-1]` degradiert von „as_of-korrekt" zu „dataset-end" ohne jedes Signal — exakt in der Konstellation (Replay), in der die Differenz Look-Ahead bedeutet. Der Fehler ist latent: in Live/EOD (Tail == as_of) ist er unsichtbar; nur im Backtest/Replay mit echtem as_of < Dataset-Ende wird er aktiv. Das ist dieselbe „silent tail read"-Klasse wie `.iloc[-1]` generell, aber getriggert durch eine **fehlende Strukturspalte**, nicht durch leere Daten (E-004).
**Wie erkennen:**
- `if <ts-Spalte vorhanden>: <PIT-filter> else: df.iloc[-1]` — der `else`-Branch lässt den as_of-Filter still fallen.
- Eine PIT-Filterung, deren Vorbedingung (Spaltenpräsenz) von einer Upstream-Funktion abhängt, die das Schema nicht garantiert.
- Code, der in Live korrekt ist, weil Tail == as_of, aber im Replay den Dataset-Tail liest.
**Wie vermeiden:**
- Wenn die Ordering-Spalte fehlt UND `as_of` gesetzt ist (Replay/Backtest): **konservativ skippen** (kein Attribut setzen) + **einmalige WARNING** (`_..._WARNED`-Global), statt still den Tail zu lesen. Der nicht-PIT-Read wird damit beobachtbar statt unsichtbar.
- Den Live/EOD-Pfad (`as_of is None` oder `as_of == latest`) explizit byte-identisch erhalten (dort ist `iloc[-1]` korrekt), damit der Guard kein Verhalten in Produktion ändert.
- Idealerweise das Schema-Versprechen der Upstream-Funktion (ts-Spaltenname) verifizieren oder normalisieren, statt sich auf ein optionales `else` zu verlassen.
**Erkannt in:** `src/assembled_core/paper/intel_context.py` (`_populate_sector_rotation_scores`, no-timestamp `else`-Branch) → Stage-2 senior + risk-execution-reviewer, Fix in Commit `62358cbc` (as_of-gesetzt → one-time WARNING + skip; as_of None → `iloc[-1]` unverändert).
**Referenzen:** E-002 (PIT Midnight-Normalization), E-030 (bfill leakt Future-Prices), E-038 (Forward-Shift admittiert Stale-History), E-004 (`.iloc[-1]` Empty-Crash — invertierte Failure-Mode: Crash vs. stiller Fehlwert).

## E-042 — Per-Bar-Flag-Reset stromabwärts seines Consumers platziert → nur partieller Fix + Test, der die Produktions-Reihenfolge umkehrt und den Rest-Bug verdeckt
**Datum:** 2026-06-04
**Kategorie:** logic-error / ordering / test-anti-pattern
**Was passierte:** Um einen stale/latched `intel_disclosures_triggers="DEGRADED"`-Flag zu entlatchen (sticky über den ganzen Run wegen des per `dataclasses.replace` shared-by-reference `intel_health_flags`-Dicts, vgl. FU-2), wurde ein Clear-at-top in `_load_intel` (~L195) eingefügt. ABER der sicherheitskritische Consumer `compute_next_state` (state_machine, gated die WATCH/COOLDOWN→ACTIVE-Krisen-Eskalation) läuft in `ingest_data` (~L170/L188) **früher im selben Cycle** als `_load_intel`. Der Clear feuert also NACHDEM der State-Machine-Consumer den Flag bereits gelesen hat → für diesen Consumer wird der Latch nur von „ganzer Run" auf „ein Bar" verkürzt, NICHT beseitigt. Der zweite Consumer (`apply_disclosures_confirm`, später IN `_load_intel`) wird dagegen korrekt same-cycle entlatcht. Verschärfend: der End-to-End-Test rief `_load_intel` VOR `compute_next_state` auf — die **umgekehrte** Produktions-Reihenfolge — und „bewies" so eine Same-Cycle-Entlatchung, die es nicht gibt.
**Warum falsch:** Ein Flag-Reset wirkt nur für Consumer, die NACH ihm im selben Cycle laufen. Wird der Reset stromabwärts des frühesten (sicherheitskritischen) Consumers platziert, ist der Fix für genau diesen Pfad unvollständig — die Behauptung „Latch gefixt" ist überzogen. Ein Unit-Test, der Producer-dann-Consumer in der **Fix**-Reihenfolge statt der **Produktions**-Reihenfolge aufruft, verdeckt den Rest-Bug und erzeugt falsche Sicherheit (vgl. E-021, self-verifying). Die Richtung war hier fail-safe (Gate bleibt zu = keine fälschliche Eskalation), aber das ist Glück, kein Design.
**Wie erkennen:**
- Ein Flag/State wird an Stelle X resettet, aber mindestens ein Consumer liest ihn an Stelle Y < X im selben Cycle (frühere Zeile / frühere Funktion im Treiber).
- Ein Test, der die Consumer-Funktion direkt nach dem Producer aufruft, obwohl der reale Treiber sie VOR dem Producer/Reset aufruft.
- Eine „gefixt"-Behauptung, die nur für eine Teilmenge der Consumer eines Flags gilt.
**Wie vermeiden:**
- Vor dem Platzieren eines Resets die EXAKTE Intra-Cycle-Aufrufreihenfolge JEDES Consumers tracen (alle Call-Sites der konsumierenden Funktion im Treiber grepen, Zeilennummern notieren) und den Reset strikt VOR den frühesten Consumer setzen — oder offenlegen, dass der frühe Consumer den Vorgänger-Bar-Zustand by-design liest.
- Mindestens einen Treiber-/Ingest-Level-Test schreiben, der die REALE Reihenfolge über mehrere Bars reproduziert (shared-by-reference ctx), nicht eine isolierte Clear-dann-Consume-Sequenz. Das tatsächliche Verhalten (ggf. Ein-Bar-Lag) assertieren, nicht das gewünschte.
- Teil-Fixes ehrlich als solche labeln; die verbleibende Lücke als separaten scoped Follow-up benennen.
**Erkannt in:** `src/assembled_core/pipeline/trading_cycle_v2.py` (`_load_intel` Clear-at-top vs. `compute_next_state` in `ingest_data`) + `tests/test_fu2_sibling_intel_health_flags.py` (umgekehrte-Reihenfolge-Test). Von Stage-2 risk + senior + Stage-3 auditor als MAJOR gefangen; Behauptung/Test in der Honesty-Fix-Iteration korrigiert (Whole-Run-Latch beseitigt + ehrlicher Produktions-Ordering-Test), Same-Bar-Reorder als Follow-up. Commit `498c9216`.
**Referenzen:** E-036 (setdefault-State-Guard No-op), E-021 (self-verifying Test), FU-2 (`bdb8d0d1` daily_circuit_breaker non-trip reset — dasselbe Muster, dort korrekt platziert).

## E-043 — Import eines archivierten/verschobenen/falschen Moduls in breitem `try/except` degradiert ein Feature STILL zu einem permanenten No-op
**Datum:** 2026-06-04
**Kategorie:** silent-except / dead-feature / import-drift
**Was passierte:** Drei Instanzen in einer Session: (1) `signals/meta_model.py` `from src.assembled_core.ml.conformal import ConformalResult` in einem `try/except Exception` — beide conformal-Module waren nach `archive/observability_graveyard_2026q2/` verschoben, also schlug der Import bei JEDEM Aufruf fehl, der `except`-Zweig lieferte degenerierte Intervalle (`lower==upper`, `half_width=0`, `confidence=1`) — ein permanent totes Feature, das wie funktionierend aussah. (2) `execution/unified_paper_engine.py:106` `from accounting.ledger import store_ledger_events_parquet` — Symbol lebt in `accounting.ledger_store`, nicht `accounting.ledger`; Import schlug immer fehl → `_HAS_LEDGER=False` → der Ledger-Parquet-Write (gated by `_HAS_LEDGER`) lief NIE, obwohl `enable_ledger=True` default ist. (3) `:142` `from ops.experience_log import log_experience_entry` — Symbol existiert nirgends → `_HAS_EXPERIENCE_LOG=False` → Feature lief nie. Alle drei waren durch ein `_HAS_X=False`-Flag in einem breiten `except` maskiert.
**Warum falsch:** Ein Import, der fehlschlagen KANN (weil das Modul archiviert/verschoben/umbenannt wurde oder das Symbol nicht existiert), gehört auf Modul-Ebene, damit der Fehler beim Laden LAUT wird — nicht in ein breites `try/except Exception`, dessen generischer Handler einen harten `ModuleNotFoundError`/`ImportError` in ein still-degradiertes Ergebnis verwandelt, das von einem LEGITIMEN Fallback-Modus (z. B. „keine Kalibrierung", „Feature deaktiviert") ununterscheidbar ist. Das Feature ist dann dauerhaft tot und sieht funktionierend (oder bewusst-deaktiviert) aus — niemand merkt es, bis ein Type-Checker oder ein Audit den Import-Drift findet.
**Wie erkennen:**
- `from <archiviertes/umbenanntes Modul> import <Y>` in einem `try/except (Exception|ImportError)`, das ein `_HAS_X=False`-Flag setzt; das Feature läuft danach nie.
- Ein `_HAS_X`-Flag, das den einzigen Call-Site eines Features gated, kombiniert mit einem Default, der das Feature eigentlich AN haben will (`enable_X=True`).
- mypy `[import-not-found]` / `[attr-defined]` auf einem first-party-Import (genau das, was ein blockierender mypy-Gate fängt — E-043 war der Treiber, das mypy-Gate scharfzuschalten).
**Wie vermeiden:**
- Imports auf Modul-Ebene halten (fail-loud beim Laden). Für ECHT-optionale Abhängigkeiten: NARROWES `except ImportError` (nicht `except Exception`), das nur den Fehlbetrag-Import abfängt, nicht jeden Folgefehler.
- Einen Test/Assertion hinzufügen, der beweist, dass das Feature TATSÄCHLICH läuft, wenn das `_HAS_X`-Flag True ist (nicht nur, dass das Flag existiert) — bzw. einen Test, der das reale Zielmodul importiert, sodass ein Archiv-/Umbenennungs-Drift sofort rot wird.
- Bei `archive/`-Verschiebungen: alle inbound-Importe der verschobenen Module greppen und mit-migrieren oder lautstark entfernen — ein verwaister Import in `try/except` ist schlimmer als ein harter Fehler.
- Ein blockierender mypy-Gate fängt Import-Drift künftig (siehe Commit `2ca6bea4`).
**Erkannt in:** `src/assembled_core/signals/meta_model.py` (Fix `5266b1eb`, inline q-Intervall statt archiviertem Import), `src/assembled_core/execution/unified_paper_engine.py:106/142` (Fix `9b642ce5`: :142 dead-removal byte-identical; :106 `_HAS_LEDGER=False` gepinnt + Re-Aktivierung als Decision deferred). Von senior-code-reviewer (E-NEW-1-Vorschlag) + risk-execution-reviewer gefangen.
**Referenzen:** E-003 (silent `except Exception: pass` — generischer Vorläufer), E-025 (Loader Fail-Open maskiert Korruption), E-024 (Infra ohne Consumer-Wiring); Rule 20 (CI/mypy-Gate), CLAUDE.md „stille except-Pfadlogik die Fehler maskiert".

## E-044 — Cross-Source-Dedupe auf einem nicht-kanonisierten Roh-Label zählt still doppelt
**Datum:** 2026-06-09
**Kategorie:** logic-error / data-correctness / test-blind-spot
**Was passierte:** Ein Multi-Mirror-Congress-Ingester (kadoa + house-stock-watcher) deduplizierte überlappende House-Trades per `drop_duplicates(subset=[symbol, event_date, disclosure_date, transaction_type])` — auf dem ROHEN Quell-Label. Dieselbe ökonomische House-PTR-Verkaufstransaktion trägt je Mirror unterschiedliche Roh-Labels („Sale (Partial)"/„Sale (Full)" bei kadoa vs. „Sale" bei house-watcher); KÄUFE kollidierten zufällig (beide „Purchase") und deduplizierten korrekt — was Fixtures/Tests grün erscheinen ließ —, VERKÄUFE überlebten dagegen doppelt. Der gewirte Consumer `add_congress_features` summiert UNSIGNED `amount` → das live `congress_total_amount_*`-Feature wurde für jeden in beiden Mirrors vorhandenen House-Verkauf inflationiert. Eine normalisierte `type`-Spalte (buy/sell) existierte bereits, wurde aber NICHT als Dedupe-Key verwendet.
**Warum falsch:** Dedupe-/Merge-Keys müssen aus KANONISIERTEN Feldern gebaut werden, nicht aus rohen Per-Source-Strings. Zufällige Übereinstimmung bei genau einem Wert (Käufe) maskiert den Defekt und erzeugt falsche Test-Sicherheit; der Unsigned-Amount-Consumer inflationiert dann still.
**Wie erkennen:**
- `drop_duplicates`/Merge-Key, der eine Roh-Label-Spalte enthält, deren Werte sich zwischen Quellen für dasselbe Ereignis unterscheiden.
- Tests/Fixtures, in denen nur EINE Kategorie (z. B. Käufe) über die Quellen überlappt, sodass der Sell-Pfad ungetestet bleibt.
**Wie vermeiden:**
- Auf normalisierte/kanonische Spalten deduplizieren (hier `type` mit Roh-Label-Fallback für None-Sides, damit distinkte Unknown/Exchange-Trades nicht über-kollabieren).
- Regressionstest, der DASSELBE logische Record in JEDER Quell-Roh-Form füttert und genau 1 überlebende Zeile assertiert; zusätzlich opposite-side same-day → 2 Zeilen.
**Erkannt in:** `src/assembled_core/data/congress_trades_ingest.py` (`ingest_congress` → neue `dedupe_congress`). Von senior-code-reviewer (Stage 2) gefangen; Stage 1 (PIT-fokussiert) hatte es übersehen.
**Referenzen:** E-037 (geteilter Cache per Symbol-Validität → Cross-Producer-Kontamination), E-045 (selbe Session/Feature), E-039 (still überschriebene Status-Größe).

## E-045 — Binäres `np.where`-Sign-Mapping macht 'unknown/None' zu einem harten SELL (fail-open)
**Datum:** 2026-06-09
**Kategorie:** silent-degradation / fail-open / sign-fabrication
**Was passierte:** Ein Net-Buy-Score leitete das Vorzeichen via `np.where(side.isin(("buy","purchase")), +1, -1)` ab — zwei-zweigig. Ein Producer, der legitim `None` für unbekannte/Exchange-Sides emittiert, bekam diese Zeilen als SELL (−amount) gewertet statt neutral. Der vorgelagerte Mitigations-Fix (normalisierte `type`-Spalte beim Producer) verlagerte das fail-open lediglich von „Sale→buy" (vorher: `type` fehlt → Default +1) auf „unknown→sell" — beseitigte es NICHT.
**Warum falsch:** Ein Zwei-Zweig-`where` kollabiert eine Drei-Zustands-Domäne (buy/sell/unknown) auf zwei und fabriziert gerichtetes Signal aus fehlenden Daten — dieselbe fail-open-Klasse, die die Normalisierung eigentlich verhindern sollte. Ein Producer-seitiger Teilfix (Label normalisieren) schließt das Consumer-seitige fail-open NICHT, solange der Consumer einen binären Default hat.
**Wie erkennen:** `where(cond, a, b)`, dessen `else`-Zweig einem Wert ein gerichtetes Vorzeichen/Signal zuweist, obwohl der Input drei Zustände (inkl. unknown/None) haben kann.
**Wie vermeiden:** Drei-Zweig-Mapping `where(buy, +1, where(sell, -1, 0))` mit neutralem 0 für unknown; die 0/None-Zeilen vor der Aggregation droppen oder als 0 belassen. Nie einem Default-Zweig ein gerichtetes Vorzeichen für Unknown-Input geben.
**Erkannt in:** `src/assembled_core/features/congress_features.py` (`compute_congress_net_buy_score`). Von senior-code-reviewer (Stage 2) als MAJOR gefangen (Folge der M1-Mitigation). Fix: Drei-Zweig + neutral, Regressionstest unknown→0.
**Referenzen:** E-025 (Fail-Open maskiert), E-043 (silent dead-feature), E-044 (selbe Session/Feature).

## E-046 — Dedupe/Restatement-Groupby-Key lässt eine diskriminierende Dimension weg → kollabiert distinkte Fakten still
**Datum:** 2026-06-10
**Kategorie:** logic-error / data-correctness / silent-collapse
**Was passierte:** Die PIT-/Restatement-Selektion eines neuen XBRL-Company-Facts-Loaders (`select_pit_rows`) wählte je Gruppe die Zeile mit max. Verfügbarkeit, gruppierte aber auf `(symbol, namespace, tag, period_end)` — OHNE `period_start`. Ein einzelnes 10-K emittiert legitim ZWEI Fakten mit identischem `period_end 2023-12-31`: die Q4-Quartalszahl (`start=2023-10-01`) UND die FY-Jahreszahl (`start=2023-01-01`). Beide kollabierten in eine Gruppe; nur eine überlebte und überschrieb die andere still (FY 1.85 statt Q4 0.30). Dasselbe in `coalesce_field`. Das korrumpiert genau die Quartals-EPS-Zeitreihe, die der reanimierte PEAD/SUE-Consumer liest — kein Look-ahead, sondern ein Wrong-Value-Defekt. Offline-Fixtures waren grün, weil jedes `period_end` dort nur EINE Dauer hatte.
**Warum falsch:** Ein Dedupe-/Selektions-Groupby-Key muss ALLE Dimensionen enthalten, die zwei Beobachtungen fachlich unterscheiden. Bei XBRL-Duration-Fakten ist `period_start` (die Periodendauer: 3M vs. 6/9/12M) genauso diskriminierend wie `period_end`. Fehlt eine solche Dimension, kollabieren distinkte Records still und der „Überlebende" hängt von Selektionsregel + Zeilenreihenfolge ab.
**Wie erkennen:**
- `groupby([...]).tail/first/last` oder `drop_duplicates(subset=[...])`, dessen Key eine offensichtlich vorhandene, fachlich-unterscheidende Spalte auslässt (hier `period_start`/`fp`).
- Test-Fixtures, in denen die ausgelassene Dimension je Key zufällig eindeutig ist (jedes `period_end` hat nur eine Dauer) → der Collapse bleibt ungetestet.
**Wie vermeiden:**
- Den vollständigen fachlichen Schlüssel verwenden (`period_start` mit aufnehmen, `dropna=False`).
- Regressionstest mit zwei Zeilen, die sich NUR in der zusätzlichen Dimension unterscheiden (gleiches `period_end`, anderes `period_start`) → beide müssen überleben.
**Erkannt in:** `src/assembled_core/data/fundamentals_xbrl_ingest.py` (`select_pit_rows` + `coalesce_field`). Von risk-execution-reviewer + senior-code-reviewer (beide MAJOR, empirisch reproduziert) + task-completion-auditor (CONDITIONAL) gefangen. Fix: `period_start` im Key + FY/Q4-Regressionsfixture.
**Referenzen:** E-044 (Dedupe auf nicht-kanonischem Key — verwandte Key-Klasse), E-047 (selbe Session, Tie-Break-Determinismus).

## E-047 — Dedup via `sort_values(eine Spalte) + groupby.tail(1)` ist auf Gleichstand reihenfolge-abhängig
**Datum:** 2026-06-10
**Kategorie:** logic-error / non-determinism / data-correctness
**Was passierte:** Dieselbe `select_pit_rows`-Selektion wählte den „aktuellsten" Restatement via `sort_values("_eff").groupby(key).tail(1)` — Sortierung NUR nach effektiver Verfügbarkeit. Bei zwei Filings mit IDENTISCHER Verfügbarkeit (häufig auf dem Date-only-`filed_date+EDGAR_DAYS`-Fallback, wo alles auf dieselbe UTC-Mitternacht kollabiert) behielt `tail(1)` die im DataFrame zuletzt stehende Zeile — also die Input-Reihenfolge. Da ein Parquet-Round-Trip / Cross-Symbol-`concat` keine kanonische Reihenfolge garantiert, war der gewählte as-reported-Wert run-to-run / order-to-order nicht-deterministisch (Vorwärts → Amendment 1.25, rückwärts → Original 1.20).
**Warum falsch:** Wenn ein Sortier-Key die Gruppenmitglieder nicht eindeutig ordnet, ist die `tail(1)`/`first()`-Auswahl von der zufälligen Zeilenreihenfolge abhängig — nicht reproduzierbar und potenziell fachlich falsch (Original statt Amendment).
**Wie erkennen:** `sort_values(<einzelne/teilweise Spalte>)` gefolgt von `groupby(...).tail(1)`/`.first()`/`.last()`, wo der Sort-Key Gleichstände zulässt; besonders gefährlich, wenn ein Fallback (Date-only) Gleichstände zum Normalfall macht.
**Wie vermeiden:** Den Tie-Break deterministisch UND fachlich sinnvoll machen: hier `sort_values(["_eff", "is_amendment", "accession"])` → max. Verfügbarkeit, dann Amendment gewinnt, dann Accession — reihenfolge-unabhängig. Order-Invarianz-Regressionstest (gleiche Daten, umgekehrte Zeilenreihenfolge → identische Auswahl).
**Erkannt in:** `src/assembled_core/data/fundamentals_xbrl_ingest.py` (`select_pit_rows`). Von risk-execution-reviewer + adversarial-PIT-verifier + senior-code-reviewer gefangen (empirisch via Reihenfolge-Umkehr reproduziert). Fix: dreistufiger deterministischer Sort + Order-Invarianz-Test.
**Referenzen:** E-046 (selbe Session, fehlende Key-Dimension), E-036 (state-isolation no-op — verwandte „stiller-Default"-Klasse).

## E-048 — `.env`-Credential-Datei in-place rewrite ohne atomic swap / robusten Key-Match
**Datum:** 2026-06-23
**Kategorie:** data-loss-risk / config-mutation / test-blind-spot
**Was passierte:** `scripts/setup_telegram.py::_upsert_env` mutierte die `.env` (einzige Kopie der Broker-/Alert-Credentials, nicht in git) via `read_text` → `write_text` in-place, und matchte existierende Keys mit `ln.strip().startswith(f"{key}=")`. Zwei latente Defekte: (1) ein Crash/Interrupt zwischen Truncate und Write hätte die gesamte `.env` verloren; (2) ein nicht-kanonischer Bestands-Eintrag (`KEY = value` mit Spaces, `export KEY=value`) matchte nicht → ein zweiter `KEY=...` wurde appendiert (dotenv nimmt last-wins, also verhaltens-korrekt, aber stille Duplikat-Akkumulation). Pure-File-Logik, die die Credential-Datei umschreibt, war zudem ungetestet.
**Warum falsch:** Eine Funktion, die die einzige Kopie sensibler Credentials neu schreibt, braucht atomare Persistenz (sonst Datenverlust-Fenster) und einen Key-Match, der den geparsten Key vergleicht statt eines Literal-Präfixes (sonst Duplikate). Ungetestete Datei-Mutation an Secrets ist ein blinder Fleck.
**Wie vermeiden:** `.env`/Config-Mutation: tmp-Datei schreiben + `os.replace` (atomarer Swap); Key per `split('=',1)[0].strip()` vergleichen (+ optionales `export `-Präfix strippen, Kommentarzeilen überspringen); `tmp_path`-Unit-Test über append/replace/preserve-others/preserve-comments/spaces-around-eq.
**Erkannt in:** `scripts/setup_telegram.py` (`_upsert_env`). Von test-runner (Stage 1) + senior-code-reviewer (Stage 2, F-senior-1/2) gefangen; im selben Step gefixt (atomic + robust match + 6 Tests in `tests/test_setup_telegram.py`).
**Referenzen:** E-003 (silent except — verwandte „stiller-Fehler"-Klasse).

## E-049 — Fail-closed `except`, das auch den Safety-Persistenz-Write umschließt, verschluckt still einen echten Halt
**Datum:** 2026-07-02
**Kategorie:** silent-except / risk-control / fail-open-residue
**Was passierte:** Der neue −10 %-Drawdown-Soft-Stop in `run_live_paper._preflight_checks` erkannte einen Breach korrekt (`check_drawdown_kill_switch(..., auto_activate=False)` → True) und schrieb dann das ack_halt-Halt-Flag via `_write_halt_flag`. Der `_write_halt_flag`-Aufruf lag jedoch INNERHALB desselben breiten `try/except`, das eigentlich nur den transienten Fall „Equity nicht lesbar" fail-closed-self-recovering behandeln sollte. Ein FS-Fehler beim (ungeguardeten) Halt-Write wäre vom `except` gefangen und als „drawdown check failed … self-recovering; no halt flag written" geloggt worden → ein BESTÄTIGTER −10 %-Breach würde still zu einem transienten Zyklus-Skip degradiert: das Ack-Gate armt nie, der nächste Zyklus läuft erneut in denselben Breach statt zu halten.
**Warum falsch:** Ein `except` konfundiert zwei fachlich verschiedene Fehlerklassen: (a) „konnte den Guard nicht auswerten" (sicher, nächsten Zyklus retryen) vs. (b) „Guard hat ausgelöst, aber Persistenz schlug fehl" (darf NICHT self-recovern — der bestätigte Stop wäre nicht durchgesetzt). Die „self-recovering"-Formulierung ist dabei aktiv irreführend für den Operator.
**Wie erkennen:** Ein fail-closed/fail-open `try/except` um einen Safety-Check, dessen Body AUCH den Persistenz-/Enforcement-Write (Flag/State/Ledger) des ausgelösten Zustands enthält — Auswertung und Durchsetzung im selben Guard.
**Wie vermeiden:** Auswertung und Persistenz in GETRENNTE Scopes: der fail-closed `except` deckt nur den Read/Evaluate-Schritt; ein bestätigter Trigger schreibt sein Flag außerhalb dieses Guards, und ein Persistenz-Fehler auf einem bestätigten Trigger blockt WEITERHIN und wird als un-persisted CRITICAL sichtbar gemacht, nie als transient. Regressionstest: Breach + `_write_halt_flag` wirft → return False, kein „self-recovering"-Log.
**Erkannt in:** `scripts/run_live_paper.py` (`_preflight_checks` Drawdown-Block). Von senior-code-reviewer (Stage 2, F-senior-1 MAJOR) gefangen; im selben Step gefixt (Scope-Trennung eval/persist + Regressionstest #5 in `tests/test_run_live_paper_drawdown_stop.py`).
**Referenzen:** E-003 (silent except), E-045 (fail-open Default), E-039 (still überschriebener Status).

## E-050 — Security-Waiver-Begründung behauptet eine Nicht-Exposition, die der Launcher widerlegt
**Datum:** 2026-07-03
**Kategorie:** security-rationale / false-justification / audit-integrity
**Was passierte:** Ein pip-audit-`--ignore-vuln`-Waiver für 4 starlette-CVEs (multipart/urlencoded form-limit bypass, `request.url` host reconstruction) wurde in `backend-ci.yml` begründet mit „Paper-ops API runs locally only, not exposed to untrusted traffic". Der eigene Launcher `scripts/run_api.py:18` bindet aber `uvicorn.run(app, host="0.0.0.0", …)` (alle Interfaces) und es ist **keine** `TrustedHostMiddleware` registriert. Die Waiver-*Entscheidung* war substanziell trotzdem korrekt (die verwundbaren Pfade sind unerreichbar: kein multipart/form/UploadFile-Handler in `api/**`, `request.url` nur als `.path` gelesen) — aber die *geschriebene Begründung* war faktisch falsch.
**Warum falsch:** Die Prämisse eines Security-Waivers muss dem tatsächlichen Verhalten des Codes entsprechen. Ein späteres Re-Audit liest „local-only" und vertraut einer Mitigation, die der Code nicht erzwingt (0.0.0.0-Bind). Die belastbare Begründung ist Code-Pfad-**Unerreichbarkeit**, nicht eine Deployment-Annahme, die das Repo verletzt.
**Wie erkennen:** Ein CVE-Waiver, der „not exposed / runs locally" als Grund nennt, ohne die tatsächliche Bind-Adresse / Ingress / Host-Allowlist-Middleware geprüft zu haben.
**Wie vermeiden:** Vor „not exposed"-Begründungen die reale Bind-Adresse (`run_*.py`/deploy) + Vorhandensein einer Host-Allowlist verifizieren. Bevorzugt Unerreichbarkeit-des-verwundbaren-Code-Pfads als Waiver-Basis; „nicht exponiert" nur behaupten, wenn Launcher/Deploy es beweisbar erzwingen. Dep-Pin-Constraints (z. B. `fastapi==0.122.0` → `starlette<0.51.0,>=0.40.0`) autoritativ aus PyPI-Metadaten belegen, nicht aus einem evtl. gedrifteten lokalen venv.
**Erkannt in:** `.github/workflows/backend-ci.yml` (pip-audit-Waiver-Kommentar) + `scripts/run_api.py`. Von senior-code-reviewer (Stage 2, F-senior-1 MAJOR) gefangen; im selben Step gefixt (Begründung auf Code-Pfad-Unerreichbarkeit + PyPI-belegten starlette-Pin umformuliert).
**Referenzen:** E-039 (Status still rekonstruiert), E-043 (silent dead-feature — verwandte „stille-Fehlannahme"-Klasse).

## E-051 — Frozenset-Iterationsreihenfolge + `rank(method="first")`-Tie-Break = prozess-abhängiger Nichtdeterminismus
**Datum:** 2026-07-10
**Kategorie:** non-determinism / reproducibility / research-integrity
**Was passierte:** In `research/mandat/verdict_engine.py` (`run_verdict`) wurde das handelbare Universum als `tradable = [s for s in members if …]` gebaut, wobei `members` ein `frozenset` ist (aus `load_membership`/`band_membership`). Die Iterationsreihenfolge eines `frozenset` von Strings hängt von `PYTHONHASHSEED` ab → variiert zwischen Prozessen. Beim Ranking `order = m.rank(ascending=False, method="first")` bricht `method="first"` Gleichstände nach Position in `m` (= `tradable`-Reihenfolge). Im Low-Div-Signal (H-032) haben viele Titel Dividende = 0 → identischer Score → massenhaft Gleichstände → welche 50 Null-Div-Titel selektiert werden variierte pro Lauf → **±10 % Ergebnis-Swing, PASS/FAIL kippte zwischen identischen Läufen.** Zusätzlich hing das Steuer-Timing an der Verkaufsreihenfolge (`for sym in held - keep - set(entries)` — Set-Iteration), weil der Verlusttopf-Offset innerhalb eines Rebalance von der Sell-Reihenfolge abhängt (Verlust zuerst → Offset für späteren Gewinn).
**Warum falsch:** Ein Research-Backtest MUSS deterministisch/reproduzierbar sein — sonst ist kein Verdict belastbar. Nichtdeterminismus versteckte, dass der „H-032-PASS" ein Artefakt der zufälligen Tie-Break-Reihenfolge war.
**Wie erkennen:** Zwei identische Läufe → unterschiedliche Endwerte. Jede Iteration über `set`/`frozenset` in einem ergebnisrelevanten Pfad; jedes `rank(method="first")`/`argsort` auf einem Feld mit vielen Gleichständen.
**Wie vermeiden:** In ergebnisrelevanten Pfaden NIE direkt über `set`/`frozenset` iterieren — immer `sorted(...)`. Tie-Breaks deterministisch UND fachlich sinnvoll machen (vgl. E-047). Regressionstest: identische Inputs → byte-identischer Output über 2 Prozesse (`PYTHONHASHSEED` variieren). Fix hier: `sorted(members)` für `tradable` + `sorted()` der Verkaufs-Sets; verifiziert byte-identisch.
**Erkannt in:** `research/mandat/verdict_engine.py` (`run_verdict`). Betraf JEDE score_panel-Verdict-Hypothese (H-029/031/032/047; H-035/036-Momentum in geringerem Maß).
**Referenzen:** E-047 (nichtdeterministischer Tie-Break via `sort_values+tail`), `research/ledger.md` „KRITISCHE KORREKTUR 2026-07-10".

## E-052 — Abgeleiteter Daten-Loader überspringt den kanonischen Hygiene-Schritt des Quell-Loaders
**Datum:** 2026-07-12
**Kategorie:** data-hygiene / silent-corruption / derived-loader-drift
**Was passierte:** `h077_mega_search.month_panel()` las `prices_verdict.parquet` direkt und baute ein Monats-Panel — OHNE die in `verdict_engine.load_verdict_prices()` kanonisch etablierte Impossible-Jump-Trunkierung (|ret|>100 % & Vortag<$1 → NaN ab da). Selektive Baskets (Insider/Congress: meist reale Titel) blieben unauffällig, aber ein GANZ-UNIVERSUM-Basket (Insider-Sell-Avoidance-Filter) fing die +34.000x-Micro-Price-Artefakte delisteter Serien → Fake-Endwert 7,5×10³⁰.
**Warum falsch:** Wenn ein Quell-Datensatz einen dokumentierten Pflicht-Hygiene-Schritt hat, MUSS jeder abgeleitete Loader ihn übernehmen (oder den kanonischen Loader wiederverwenden). Selektive Tests maskieren die Korruption; erst Breitband-Nutzung explodiert.
**Wie erkennen:** Absurde Endwerte (>10^x) bei Ganz-Universum-Aggregaten; neuer Loader liest dieselbe Roh-Datei wie ein bestehender Loader mit Hygiene-Logik.
**Wie vermeiden:** Roh-Parquets mit bekannten Defekten nur über den kanonischen Loader (oder dessen Hygiene-Funktion) konsumieren; bei neuen Loadern grep nach existierenden Loadern derselben Datei. Plausibilitäts-Guard in Screens (net > 100×START → Artefakt-Alarm statt Ergebnis).
**Erkannt in:** `research/mandat/h077_mega_search.py` (month_panel), aufgefallen in H-080-Sell-Filter. Sofort gefixt; betroffene Strand-Läufe wiederholt.
**Referenzen:** H-036-Illiquiditäts-Artefakt (gleiche Datenquelle, verwandter Fang), E-041 (Look-Ahead durch fehlenden Filter im else-Branch).

## E-053 — „EOD"-Feed liefert forming Same-Day-Bar → Partial-Bar-Look-Ahead im Live-Cache
**Datum:** 2026-07-14
**Kategorie:** pit-violation / logic-error / ingest-drift
**Was passierte:** Ein neues Cache-Refresh-Skript (`refresh_daily_cache_from_eodhd.py`) filterte neue Bars nur per `timestamp > per_symbol_cache_max` — OHNE oberen Cutoff. EODHD `/eod/{sym}.US` liefert aber bereits WÄHREND der offenen US-Session einen Bar mit heutigem Datum (empirisch verifiziert: AAPL same-day-Bar intraday abrufbar). Beim geplanten Task-Lauf 21:10 CEST (= 15:10 ET, vor Close) wäre ein partieller/forming Tagesbar als neuester Per-Symbol-Bar in den LIVE-Cache gelangt; `multifactor_v2.py:501` nimmt via `groupby(...).tail(1)` exakt diesen Bar als Signal- UND Ausführungspreis.
**Warum falsch:** PIT-Verletzung: ein unfertiger Intraday-Preis wird als settled Close behandelt. Das Schwester-Skript `refresh_sector_etf_cache.py:204-208` hält gegen genau diese Klasse einen expliziten `timestamp < today_utc`-Cutoff — neue Ingest-Pfade erben solche Invarianten NICHT automatisch (Ingest-Drift, verwandt E-052).
**Wie erkennen:** Neuer Writer in einen live-genutzten Preis-Store ohne oberen Datums-Cutoff; „EOD"-Endpunkt-Vertrauen („liefert ja nur fertige Bars") ohne Intraday-Probe; Task-Läufe vor Session-Close.
**Wie vermeiden:** Jeder Ingest-Pfad in einen live genutzten Preis-Store MUSS einen harten `timestamp < today_utc` (session-close-aware) PIT-Cutoff selbst erzwingen — dem Feed-Namen „EOD" nie vertrauen; forming-Bar-Verhalten des Endpunkts einmal empirisch proben; Guard-Idiom des bestehenden Schwester-Writers spiegeln.
**Erkannt in:** `scripts/ops/refresh_daily_cache_from_eodhd.py` — von senior-code-reviewer (Stage 2, F-senior-1 BLOCKER) VOR dem ersten 21:10-Scheduled-Lauf gefangen; Ein-Zeilen-Fix + Drop-Logging im selben Step.
**Referenzen:** E-041 (Look-Ahead-Klasse), E-052 (Ingest-Hygiene-Drift), `scripts/ops/refresh_sector_etf_cache.py:204`.

## E-054 — Safety-Gate liest Artefakt-Pfad, den kein Writer bedient (armed-fail-closed-Deadlock)
**Datum:** 2026-07-22
**Kategorie:** wiring-gap / safety-control / deadlock-by-design
**Was passierte:** `ops/_paper_runner_gates.apply_reconcile_block_gate` (Reconcile-Block-Gate, default-off seit `54cc9026`) liest `<root>/output/reconcile_latest.json` und blockt ARMED fail-closed bei fehlendem/stalem Artefakt. Der einzige Writer (`ops/paper_runner._prd_paper_fills_and_ledger`) schrieb das Artefakt aber ausschließlich ins per-run-Verzeichnis `<root>/output/runs/<run_id>/`. Beim Scharfschalten hätte das Gate den Root-Pfad nie gefunden → jeder Zyklus geblockt; und weil das Gate VOR dem Zyklus läuft, hätte kein Zyklus das Artefakt je refreshen können — struktureller Deadlock. Default-off maskierte die Lücke seit Einführung.
**Warum falsch:** Ein fail-closed-Safety-Gate und der Writer seiner Freigabe-Bedingung müssen nachweislich denselben aufgelösten Pfad bedienen. „Gate liest X, Writer schreibt Y" wird erst beim Scharfschalten sichtbar — der gefährlichste Zeitpunkt.
**Wie vermeiden:** Beim Scharfschalten eines artefakt-basierten Gates den Writer-Pfad gegen den Reader-Pfad VERIFIZIEREN (identischer absoluter Pfad, Freshness-Feld vorhanden) und einen Operator-Recovery-Pfad bereitstellen, weil ein pre-cycle-Gate seinen eigenen Refresh nicht auslösen kann. Fix hier: stabile Root-Kopie in paper_runner (Layout-Guard `output_dir.parent.parent.name == "output"`), Seed dokumentiert, Recovery via `scripts/ops/rebuild_reconcile_artifact.py` (re-evaluiert Invarianten, kein Bypass).
**Erkannt in:** `src/assembled_core/ops/paper_runner.py`, `src/assembled_core/ops/_paper_runner_gates.py`, `configs/app.yaml` — GESAMTBEWERTUNG K5-Umsetzung 2026-07-21; von Stage-1 risk-execution-reviewer (M3/M4) und Stage-2 senior-code-reviewer bestätigt.
**Referenzen:** `docs/GESAMTBEWERTUNG.md` §5 Schritt 3, Commit `54cc9026` (default-off-Einführung).

## E-055 — Status-Schreibweisen-Drift: Alpaca „canceled" (ein L) vs. Terminal-Set „cancelled" (zwei L)
**Datum:** 2026-07-22
**Kategorie:** logic-error / vendor-enum-drift / silent-misclassification
**Was passierte:** `broker_execution._TERMINAL_STATUSES` und die Poll-Kategorisierung prüften nur „cancelled" (britisch, zwei L). `AlpacaAdapter._normalize_order` liefert den echten Alpaca-Enum aber als „canceled" (US, ein L). Eine broker-seitig gecancelte Order galt dadurch nie als terminal: sie pollte bis zum 120s-Timeout und landete in `timed_out` statt `rejected` — Latenzverschwendung pro Zyklus und semantisch falsche Kategorie in Result/Journal.
**Warum falsch:** Vendor-Enums sind die kanonische Wahrheit; ein lokal gewähltes Synonym driftet still. Der Fehler ist unsichtbar, solange kein Broker-Cancel auftritt, und maskiert sich dann als „Timeout".
**Wie vermeiden:** Vendor-Status-Strings EINMAL zentral normalisieren oder die Terminal-Menge mit ALLEN real emittierten Schreibweisen definieren und überall referenzieren (keine lokalen Status-Tupel). Beim Anfassen einer Status-Naht: die tatsächlich vom Adapter emittierten Werte gegen jede Vergleichsstelle grep-prüfen.

## E-056 — pip-freeze-Lock aus verschmutztem venv zieht dev/research-Toolchain in den Prod-Docker-Pfad
**Datum:** 2026-07-23
**Kategorie:** wiring-gap / dependency-hygiene / two-truths
**Was passierte:** `requirements.lock` (vom Dockerfile via `pip install -r requirements.lock` ins Runtime-Image konsumiert) wurde per `pip freeze` im ARBEITS-venv regeneriert. Das enthielt über requirements.txt hinaus dev-Extras (bandit, detect-secrets, pre-commit-Toolchain) und optionale Research-Libs (numba, llvmlite, lightgbm, hmmlearn, MAPIE) — alle wanderten in den Lock, obwohl der Lock-Header die Herleitung als „pip install -r requirements.txt && pip freeze" dokumentiert. Zusätzlich schrieb der Regen die pre-existierende alpaca-py-Divergenz (Lock 0.43.2 vs. txt 0.38.0) unkommentiert fort, statt sie zu reconcilen.
**Warum falsch:** Prod-Image bekommt Compiler-Stack + Security-Scanner, die es nie ausführt (Bloat + Attack-Surface); Lock über-spezifiziert gegen seine eigene dokumentierte Herleitung; zwei Wahrheiten zwischen CI-Datei (requirements.txt) und Container-Datei (requirements.lock).
**Wie vermeiden:** Locks IMMER in einem frischen venv aus exakt `pip install -r requirements.txt` freezen (kein dev-Extra, keine Research-Libs) — oder getrennte prod-/dev-Locks führen und im Dockerfile das prod-Lock referenzieren. Beim Regen jede Divergenz zur autoritativen Datei benennen statt fortschreiben (Rule 40).
**Erkannt in:** `requirements.lock`, `Dockerfile` — Stage-2 senior-code-reviewer (F-senior-1/F-senior-2) beim Rest-Batch-Review 2026-07-23; Fix: frisches Lock-venv, alpaca-py-Pin in requirements.txt auf die real laufende 0.43.2 gehoben.
**Referenzen:** E-024, Rule 40 (Dependency-Drift), GESAMTBEWERTUNG K8.
**Erkannt in:** `src/assembled_core/execution/broker_execution.py` — Stage-2 senior-code-reviewer beim P4-Review (pre-existing, nicht vom Paket eingeführt); Fix + Regressionstest `test_e055_broker_canceled_single_l_is_terminal` im Follow-up-Commit.
**Referenzen:** E-044 (nicht-kanonisierte Roh-Labels), GESAMTBEWERTUNG P4 Stage-2-Review.

## E-057 — mypy-Incremental-Cache meldet Fehler frueherer Einzelmodul-Laeufe im Follow-Graph erneut (falsche Env-Divergenz-Diagnose)
**Datum:** 2026-07-26
**Kategorie:** tooling-pitfall / false-diagnosis / ci-vs-local
**Was passierte:** Beim mypy-Sweep Tranche 1 meldete das lokale 5-Pfad-Gate-Kommando ploetzlich 72 Fehler in 19 followed Dateien (accounting/ops/pipeline/qa) — waehrend CI mit identischem Kommando gruen war. Zwei Reviewer reproduzierten die 72 unabhaengig und diagnostizierten eine Env-Divergenz (CI sieht weniger optionale Deps). Tatsaechliche Ursache: Unmittelbar davor waren qa/ops/accounting/api/pipeline EINZELN mit mypy gemessen worden — der Incremental-Cache (.mypy_cache) kannte deren Fehler und meldete sie bei nachfolgenden Laeufen erneut, sobald die Module im Follow-Import-Graph lagen. Ein frischer Lauf nach Cache-Konsolidierung: lokal == CI == gruen ("no issues in 248 files").
**Warum falsch:** Eine Tooling-Eigenheit wurde als Umgebungs-/Konfigurationsproblem fehlgedeutet und haette zu einem unnoetigen "Fix" (z.B. follow_imports=silent) fuehren koennen, der das Gate real geschwaecht haette.
**Wie vermeiden:** Vor jedem Lokal-vs-CI-Divergenz-Schluss bei mypy: frischen Lauf ohne Cache-Vorbelastung machen (Remove .mypy_cache oder --no-incremental) — besonders nach Einzelmodul-Messlaeufen. Erst wenn die Divergenz den Frischlauf ueberlebt, ist es ein Env-Thema (Rule 40).
**Erkannt in:** mypy-Sweep Tranche 1 (Commit 00ebf104); aufgeklaert durch Frischlauf nach CI-Gruen-Beweis; Stage-3-Auditor empfahl den Registereintrag.
**Referenzen:** Rule 40 (Dependency-Drift-Unterscheidungspflicht), backend-ci.yml mypy-Gate.

## E-058 — Einseitiges type: ignore fuer lokal-getypte / CI-fehlende Third-Party-Libs (polars-unused-ignore-Falle)
**Datum:** 2026-07-26
**Kategorie:** tooling-pitfall / ci-vs-local / type-ignore-asymmetrie
**Was passierte:** `qa/differential_testing.py` hatte im ImportError-Fallback `pl = None  # type: ignore[assignment]`. Lokal ist polars installiert UND getypt (py.typed) — der ignore ist noetig. In CI fehlt polars (nicht in requirements.txt) und der `[[tool.mypy.overrides]]`-Eintrag `ignore_missing_imports` macht `pl` zu Any — der ignore waere dort UNUSED und `warn_unused_ignores = true` haette das frisch erweiterte 12-Pfad-Gate ROT gemacht, obwohl lokal alles gruen war. Vom Stage-1 ci-debugger per Fake-Modul-Repro empirisch belegt, BEVOR der Commit gepusht war.
**Warum falsch:** Ein Paket, das lokal getypt vorliegt, aber in CI fehlt und im Override steht, erzeugt asymmetrische ignore-Notwendigkeit: ein einzelner Error-Code ist in genau EINER der beiden Umgebungen falsch. Gleiches Risiko fuer alle lokal-installierten, CI-fehlenden Override-Libs (numba etc.).
**Wie vermeiden:** Fuer solche Pakete den paarigen Code verwenden: `# type: ignore[<realer-code>, unused-ignore]` — der unused-ignore-Zusatz neutralisiert die Umgebung, in der der Hauptcode nicht feuert. Vor Merge beide Bedingungen pruefen (lokal mit Paket; CI-simuliert ohne, z.B. Fake-Env): 0 unused-ignore UND 0 unsilenced error. Nicht raten — empirisch verifizieren.
**Erkannt in:** `src/assembled_core/qa/differential_testing.py:64`, `pyproject.toml` — mypy-Sweep Tranche 4, Stage-1 ci-debugger (HIGH), Fix vor Commit.
**Referenzen:** E-057 (mypy-Cache-Falle), Rule 40 (Drift-Unterscheidungspflicht), backend-ci.yml mypy-Gate.

## E-059 — Import auf nicht-existentes Modul unter broad except = stiller Feature-Ausfall (config-gated dead paths)
**Datum:** 2026-07-26
**Kategorie:** silent-except / wiring-gap / plan-ne-implementierung
**Was passierte:** Der mypy-Sweep Tranche 3+4 deckte >20 config-gated Bloecke auf (in `_tc_signals`, `_tc_features`, `_tc_sizing`, `_tc_execution`, `orchestrator`), deren `from ...X import Y` auf nie existierende Module/Symbole zeigt (`ops.shadow_recorder`, `portfolio.risk_budgeting/bl_sizing/mvo_optimizer`, `risk.factor_risk_model`, `TickStore`-Klasse, falsche `rolling_imbalance_signal`-Nutzung, `ctx.execution_mode`/`ctx.equity` nie gesetzt). Bei aktiviertem Feature wirft der Import/Zugriff, wird vom umschliessenden `except Exception` verschluckt (bestenfalls debug-Log) — das Feature laeuft NIE: u.a. Zombie-Killer force-FLAT, ERC/MVO/BL-Sizing, cost_aware-Config, Factor-Risk-Overlay, OB-Imbalance-Merge, QuestDB-Write-Through, KPI/Manifest/Heartbeat-Sites.
**Warum falsch:** Ein per Policy aktivierbares Feature taeuscht „aktiviert" vor, ist real tot — klassischer Plan-≠-Implementierung-Drift, unsichtbar gemacht durch broad-except. Die frueher als „importability-only" geloeschten Wiring-Tests haetten genau diese Klasse gefangen.
**Wie vermeiden:** (1) Config-gated optionale Imports nicht unter broad `except Exception` verstecken: ImportError/ModuleNotFoundError separat fangen und mindestens WARN loggen, damit enabled-aber-tot sichtbar wird. (2) Pro aktivierbarem Feature ein Wiring-/Smoke-Test mit enabled=true, der prueft, dass der Pfad nicht still ge-excepted wird. (3) Blocking-mypy ueber vollstaendiges src/ beibehalten — es war der einzige strukturelle Fang.
**Erkannt in:** mypy-Sweep Tranchen 3+4 (2026-07-26), alle Stellen als FIXME(mypy-sweep) + type-ignore markiert (verhaltensneutral); Wiring-Fixes als separate Auftraege, Prio Zombie-Killer (`_tc_signals`).
**Referenzen:** E-054 (Wiring-Gap-Klasse), CLAUDE.md „Bekannte Problemzonen" (stille except-Pfadlogik), GESAMTBEWERTUNG §8.

## E-060 — Ratchet-Bump-Kommentar mit Cap-Wert statt Ist-Count begruendet (+1 statt +3)
**Datum:** 2026-07-27
**Kategorie:** test-anti-pattern / audit-trail-integrity / false-arithmetic
**Was passierte:** Beim Bump des Broad-Except-Ratchets (Cap 1035->1036 nach dem shadow_recorder-Restore) wurde der Delta-Kommentar mit dem alten CAP (1035) als Startpunkt geschrieben und "+1 broad except" behauptet. Tatsaechlich: Ist-Count vor Restore 1033, das restaurierte Modul enthaelt DREI broad-except-Substrings (zwei date-parse-Fallbacks + ein Swallow) = +3. Endzahl (1036) und Zero-Headroom stimmten nur durch Verrechnung zweier Fehler (Startwert -2, Delta +2). Stage-2-Review fing es (F-senior-1 MAJOR).
**Warum falsch:** Der Ratchet-Kommentar ist die append-only-Audit-Historie, auf die sich jede kuenftige Bump-Entscheidung stuetzt — eine falsche Zahl vergiftet spaetere Reviews. Substring-Zaehlung ("except Exception:"/"except Exception as") ist nicht die gefuehlte Anzahl except-Bloecke; geschachtelte Fallbacks im selben Modul werden leicht als "ein except" fehlwahrgenommen.
**Wie vermeiden:** Vor jedem Ratchet-Bump den Ist-Count am Parent-Commit UND am neuen Commit empirisch messen (git-Blob + exakt die Zaehlmethode des Tests), Delta aus (neu - alt) ableiten, im Kommentar Ist-Counts (nie Cap-Werte) als Start/Ende dokumentieren. Bei Modul-Restores: broad-Substrings im Modul per grep zaehlen, nicht aus dem Gedaechtnis.
**Erkannt in:** `tests/test_session_2026_05_07_new_items.py` (Ratchet-Kommentar), `src/assembled_core/ops/shadow_recorder.py` (Z.42/45/60) — Stage-2 senior-code-reviewer auf Commit 35b3ea95; Kommentar im Folge-Commit korrigiert.
**Referenzen:** Rule 40 (Testehrlichkeit), E-059 (shadow_recorder-Restore-Kontext).

## E-061 — call-arg-Mismatch per type-ignore stillgelegt statt als Bug erkannt (totes Safety-Gate)
**Datum:** 2026-07-27
**Kategorie:** silent-except / type-ignore-missbrauch / dead-safety-gate
**Was passierte:** `qa_gates.check_leakage` rief `assert_feature_zero_before_disclosure` mit df=/feature_col=/...-Kwargs auf; die reale Signatur ist `(prices, events, feature_fn, *, as_of_before, as_of_after)` (Re-Computation-Harness). Der TypeError wurde von den except-Klauseln (AssertionError/ValueError/ImportError) NICHT gefangen — das Leakage-Gate war fuer echte Inputs tot. Ein `# type: ignore[call-arg]` hatte den Mismatch vor mypy versteckt.
**Warum falsch:** Ein Safety-Gate, das fuer echten Input nie BLOCK liefern kann, ist schlimmer als kein Gate — es bewirbt PIT-Schutz, der nicht laeuft. Ein call-arg-Mismatch ist IMMER ein echter Bug, nie ein Typ-Rauschen.
**Wie vermeiden:** call-arg-Fehler NIE per type-ignore stummschalten — Kwargs, die nicht zur Zielsignatur passen, sind ein Realfehler. Wenn ein Gate einen bestehenden Helper nicht nutzen kann (falsche Input-Form), den Check inline gegen den tatsaechlichen Input implementieren statt eine unpassende API zu rufen und den Typfehler zu ignorieren.
**Erkannt in:** `src/assembled_core/qa/qa_gates.py` — mypy-Sweep T4 fand den Mismatch (als FIXME markiert), Fix in Commit 3a3ec42f (inline row-wise PIT-Check, fail-closed, 8 E2E-Tests).
**Referenzen:** E-059 (Silent-Except-Klasse), Rule 30 (keine Lockerung von Safety-Checks).

## E-062 — getattr mit Attribut-Zugriff als Default: eager evaluiert, AttributeError killt den Block
**Datum:** 2026-07-27
**Kategorie:** logic-error / eager-default / silent-except
**Was passierte:** `run_index`-Metrics nutzten `getattr(ctx, "current_equity", ctx.equity)`. `ctx.equity` existiert nicht — das DEFAULT-Argument wird bei getattr IMMER eager evaluiert, der AttributeError flog VOR dem getattr-Fallback und der umschliessende except uebersprang den gesamten run_index-Write. Der Zwischenfix `getattr(...) or ctx.capital` haette zusaetzlich ein legitimes equity==0.0 (Totalverlust) still in Startkapital verwandelt.
**Warum falsch:** getattr-Defaults sind nicht lazy; ein Attributzugriff als Default hebelt genau den Schutz aus, den getattr geben soll. Und `x or default` kollabiert valide falsy-Werte (0.0) — bei Finanzgroessen ein Ehrlichkeitsfehler.
**Wie vermeiden:** `getattr(obj, "x", None)` + expliziter `is not None`-Check; NIE einen Attributzugriff als getattr-Default; NIE `or` als Default-Mechanik fuer Numerik, die legitim 0 sein kann.
**Erkannt in:** `src/assembled_core/pipeline/_tc_execution.py` (run_index final_equity) — Stage-1 risk-execution-reviewer (MAJOR-2) auf dem E-059-#2-Diff; Fix in Commit 69e8f68d (final_equity nur bei echtem Wert, sonst leer; 0.0-Edge-Test).
**Referenzen:** E-049 (Scope-Trennung um Safety-Writes), Rule 40 (Testehrlichkeit).

## E-063 — Blockierender Netzwerk-Connect in einem "never-blocks"-except ist nicht wirklich geschuetzt
**Datum:** 2026-07-27
**Kategorie:** silent-except / latenter-cycle-stall / false-guarantee
**Was passierte:** Step 7.70 (QuestDB-Write-Through, nach E-059-Reparatur erstmals lauffaehig) wrappt `tick_store.ping()` in `except Exception -> log.debug(skipped)` und gilt damit als "never blocks the cycle". Aber `ping() -> _open_conn() -> _connect_fn()` setzt KEIN connect_timeout: Ein black-holed Host (Firewall-Drop statt Refuse) blockiert den Connect unbegrenzt — ein Haenger ist keine Exception, der except feuert nie. In einem synchronen Zyklus-Step (book_fills) waere das ein Cycle-Stall.
**Warum falsch:** "In try/except gewrappt" wird routinemaessig als "kann die Pipeline nicht anhalten" gelesen. Das gilt fuer raises, nicht fuer blockierende Syscalls.
**Wie vermeiden:** Jeder Netzwerk-Client, der aus einem synchronen Trading-Zyklus-Step erreichbar ist, MUSS ein explizites connect/read-Timeout auf Treiber-Ebene setzen; try/except allein begrenzt keine Wall-Clock-Zeit. Vor dem Vertrauen auf den except das Timeout verifizieren. Hier: Feature bleibt config-aus; connect_timeout in tick_store ist dokumentierte Enablement-Precondition (Kommentar am Step 7.70).
**Erkannt in:** `src/assembled_core/data/tick_store.py` (_get_conn_kwargs ohne Timeout), `src/assembled_core/pipeline/_tc_execution.py` (Step 7.70) — Stage-1 risk-execution-reviewer (H-1) + Stage-2 senior beim E-059-Rest-Review 2026-07-27.
**Referenzen:** E-059 (Silent-Except-Klasse), Rule 30 (Latenz-/Verfuegbarkeits-Invarianten).
**Status 2026-08-01 (autoritativ fuer diese Precondition-Liste):** connect_timeout implementiert (`src/assembled_core/data/tick_store.py`, env `QUESTDB_CONNECT_TIMEOUT_S`, akzeptierter Bereich (0, 30] — sonst Default; Kwarg-Namen empirisch verifiziert gegen psycopg2-binary 2.9.12 / pg8000 1.31.5, plus offline-Validierung `_validate_conn_kwargs` die bei Treiber-Drift LAUT fail-closed abschaltet statt unbegrenzt zu connecten). NOCH OFFEN vor `questdb.write_through.enabled`: (a) Read-/Query-Timeout im psycopg2-Zweig (libpq hat keins; Keepalives fangen nur den toten Peer, nicht den haengenden Server); pg8000 ist laut Treiber-Quelltext bounded (persistenter Socket-Timeout), im Repo aber NICHT nachgemessen — kein Treiber installiert/deklariert, die Vertragstests skippen dauerhaft, also vor Enablement einmal empirisch verifizieren; (b) DNS-Aufloesung ist in keinem Zweig beschraenkt → `QUESTDB_HOST` als IP-Literal setzen; (c) Fills landen in derselben "trades"-Tabelle wie die Marktdaten, mit ts=now() statt Fill-Zeit; (d) kein Dedup-Key (Re-Run von book_fills dupliziert Ticks); (e) kein PG-Treiber ist in requirements/pyproject deklariert — Enablement ist ein Dependency-/pip-audit-Ereignis; (f) Budget pro Zyklus sind DREI Connects (ping/ensure_table/write_ticks), nicht einer.
**Doku-Drift bewusst offen:** Der Precondition-Kommentar in `src/assembled_core/pipeline/_tc_execution.py` (Step 7.70, Block `ENABLEMENT PRECONDITIONS`) nennt Punkt (1) „kein connect_timeout" weiterhin als offen. Er ist seit 2026-08-01 falsch, liegt aber in einem geschuetzten Pfad (Edit-deny) und wird beim naechsten beauftragten Edit dort nachgezogen. Diese Datei ist bis dahin die Wahrheit.

## E-064 — enabled-Flip ohne Feld-Voraussetzung = stiller No-Op (falsche Beobachtungs-Evidenz)
**Datum:** 2026-07-27
**Kategorie:** wiring-gap / silent-no-op / false-negative-evidence
**Was passierte:** Der Operator-Flip `zombie_killer.enabled: true` waere allein wirkungslos gewesen: Die Pilot-Ledger-Positionen trugen kein `entry_ts`, `check_zombie_position` gab fuer jede Position `(False, "")` zurueck — ohne eine einzige Logzeile ununterscheidbar von „keine Zombies gefunden". Ein leeres `output/shadow/` haette als Beobachtungs-Evidenz („lief, nichts gefunden") gelesen werden koennen, obwohl die Pruefung datenbedingt tot war. Vom Stage-1 risk-execution-reviewer (H1) VOR dem Commit gefangen.
**Warum falsch:** Ein enabled-Flag suggeriert Aktivitaet. Ohne die vom Check gelesenen Input-Felder (`entry_ts`, `entry_price`, `current_price`) ist der aktivierte Check ein unsichtbarer No-Op, der falsche Negativ-Evidenz produziert.
**Wie vermeiden:** Vor jedem enabled-Flip pruefen, ob die Datenquelle die vom Feature gelesenen Felder real liefert (Feld-fuer-Feld gegen den Consumer-Code). Fehlende Inputs LAUT machen (warn-once), nie als Negativ-Ergebnis durchgehen lassen. Flip und Feld-Bereitstellung gehoeren in denselben Step. Fix hier: entry_ts-Schema in paper_ledger (Open/Flip stampt, Add/Partial preserved), Durchreichung in _prd_load_paper_state (+entry_price/current_price fuer den Gain-Check), warn-once in risk/zombie_killer fuer Legacy-Positionen ohne entry_ts.
**Erkannt in:** `configs/policy.yaml`, `src/assembled_core/ops/paper_ledger.py`, `src/assembled_core/ops/paper_runner.py`, `src/assembled_core/risk/zombie_killer.py` — Zombie-Killer-Aktivierung 2026-07-27.
**Referenzen:** E-054 (Gate liest un-bedienten Pfad — Geschwister-Klasse Writer-fehlt), E-059 (Silent-Except), E-062 (getattr-Defaults).

## E-065 — Read-modify-write auf den Paper-Ledger ohne Herkunfts-Guard maskiert Backup-Fallback-Verlust
**Datum:** 2026-07-27
**Kategorie:** silent-except / silent-data-loss / ops-tooling
**Was passierte:** Ein Operator-Tool (entry_ts-Backfill) lud den Ledger via `load_ledger_state`, mutierte ein Feld und schrieb via `save_ledger_state` zurueck. `load_ledger_state` faellt bei korrupter HAUPTdatei still auf einen AELTEREN .1/.2/.3-Backup-Stand zurueck — der anschliessende save haette diesen aelteren Stand (inkl. altem cash/positions) als neue Wahrheit persistiert, waehrend der Post-Write-Verify nur das gestempelte Feld pruefte und [OK] gemeldet haette. Vom Stage-2-Review VOR dem Commit gefangen (im Live-Run war die Hauptdatei intakt).
**Warum falsch:** Silent Data Loss auf cash/positions bei gleichzeitigem Erfolgs-Log — verletzt Datenrealismus (Datenprobleme nicht still verschlucken) und Rule 30 (keine stille Portfolio-Veraenderung). Die Backup-Fallback-Robustheit des LOADERS ist fuer Reader richtig, fuer Writer gefaehrlich.
**Wie vermeiden:** Bei read-modify-write auf den Ledger: (1) die HAUPTdatei vorab selbst parsen und bei jedem Parse-Problem abbrechen (kein Backup-Fallback fuer Schreib-Tools); (2) LedgerCorruptionError explizit fangen -> [ERROR] + Abbruch; (3) Post-Write-Verify muss auch die NICHT-mutierten Felder pruefen (cash + alle Positionsfelder ausser dem gestempelten), nicht nur das Ziel-Feld; (4) load->save ist trotz File-Lock im save nicht atomar gegen konkurrierende Writer — Ops-Tools ausserhalb des Scheduler-Fensters laufen lassen und das dokumentieren.
**Erkannt in:** `scripts/ops/backfill_position_entry_ts.py` — Stage-2 senior-code-reviewer (F-senior-1 MAJOR) beim Backfill-Review 2026-07-27; alle 4 Guards vor dem Commit eingebaut + Corrupt-Main-Regressionstest.
**Referenzen:** E-048 (atomare Config-Writes), E-064 (Feld-Voraussetzungen), Rule 30.

## E-066 — Nicht gepruefter Gate zaehlt im Aggregat als bestanden (Sichtbarkeit ohne Zaehler-Ehrlichkeit)
**Datum:** 2026-08-01
**Kategorie:** wiring-gap / false-positive-evidence / aggregate-vs-detail
**Was passierte:** `check_leakage` wurde als 8. Gate in `evaluate_all_gates` verdrahtet (E-059-Follow-up). Ohne `feature_df` liefert es `OK` + `details["skipped"]` — im Detail korrekt als „nicht geprueft" markiert, aber `passed_gates` stieg dadurch von 7 auf 8. Alle Aggregat-Konsumenten (API `gate_counts`, `**Passed:** N` im Daily-QA-Report, die Backtest-Logzeile) zeigten seither ein zusaetzliches Gruen fuer eine Pruefung, die im Betrieb nichts prueft, weil kein Produktions-Caller einen Frame uebergibt. Zusaetzlich wurde die Entscheidung „nur dokumentieren statt fixen" mit einer FALSCHEN Tatsachenbehauptung begruendet („alle Konsumenten leiten die Counts aus gate_results neu ab") — tatsaechlich lasen 4 von 5 das Summary-Attribut, der Fix lag komplett in der eigenen, ungeschuetzten Datei. Vom Stage-2 senior-code-reviewer VOR dem Commit gefangen (empirisch nachgemessen, nicht nur gelesen).
**Warum falsch:** Aggregate werden gelesen, Detail-Strings nicht. Ein hochgezaehlter Passed-Counter ist genau die falsche Sicherheit, gegen die E-064 geschrieben wurde — nur eine Ebene hoeher: nicht der Check ist still tot, sondern seine Erfolgsmeldung ist erfunden. Und: eine „geht nicht ohne Schutzpfad-Edit"-Begruendung ist selbst eine Behauptung, die belegt werden muss.
**Wie vermeiden:** (1) Wer ein Gate mit OPTIONALEM Input in ein Summary haengt, fuehrt im selben Step einen `skipped`-Zaehler und schliesst geskippte Gates aus dem Passed-Zaehler aus. (2) Vor der Entscheidung „nur dokumentierbar" JEDEN Konsumenten einzeln pruefen, ob er das Summary-Attribut liest oder aus `gate_results` neu ableitet — nicht schaetzen. (3) Der `reason`-String ist bei Report-/Log-Sinks oft der einzige Anker, der ankommt (`details` wird dort nicht gerendert) — er muss „NOT CHECKED" sagen, nie etwas, das wie ein sauberes Ergebnis klingt.
**Erkannt in:** `src/assembled_core/qa/qa_gates.py` (`evaluate_all_gates` Count-Block, `check_leakage` Skip-Branch) — Stage-1 risk-execution-reviewer (MAJOR-1/-2) + Stage-2 senior-code-reviewer (F-senior-1/-2) beim check_leakage-Wiring 2026-08-01.
**Referenzen:** E-064 (enabled-Flip ohne Feld-Voraussetzung), E-054 (Gate liest un-bedienten Pfad), CLAUDE.md „Keine falsche Sicherheit".
**Status 2026-08-01 (Zaehl-Ehrlichkeit behoben, Skip-Sichtbarkeit offen; nach Stage-2-Widerspruch):** Der erste Anlauf zaehlte den Skip nur im Dataclass-Feld heraus und liess die RE-derivierenden Konsumenten (`pipeline/orchestrator._gate_result_to_dict` -> Run-Manifest -> `api/routers/qa.py:327` + `monitoring.py:118`) weiter `passed_gates=8` melden — die Klasse waere also im selben Commit an einer Stelle behoben und an einer anderen NEU eingefuehrt worden. Ursache war die Modellierung des Skips als `QAResult.OK` + details-Flag. Fix: **`QAResult.SKIPPED` als eigener Enum-Zustand** — ein Konsument, der auf `result.value == "ok"` filtert, kann den Skip damit gar nicht als bestanden zaehlen, ohne dass ein geschuetzter Pfad angefasst werden muss. Mitgezogen: `reports/daily_qa_report` (❓-Icon + „Not checked"-Zeile), `scripts/run_backtest_strategy` („?" statt „✓" + NOT-CHECKED-Summenzeile). **NOCH OFFEN (eigener Follow-up, geschuetzter Pfad):** die SICHTBARKEIT des Skips im Artefakt — `_gate_result_to_dict` serialisiert `skipped_gates` nicht, `api/routers/qa.py:327/399` und `monitoring.py:118/217` bauen die counts ohne skipped-Key, und `QAStatusSummary` transportiert gar keine `gate_results`. Der Zaehler luegt nicht mehr, aber drei Sinks schweigen ueber den 8. Gate-Zustand. **Erkannt in** zusaetzlich: `pipeline/orchestrator.py`, `api/routers/qa.py`, `api/routers/monitoring.py`, `reports/daily_qa_report.py`, `scripts/run_backtest_strategy.py`.

## E-067 — strict-xfail auf eine im selben Step SELBST erzeugte Regression
**Datum:** 2026-08-01
**Kategorie:** test-anti-pattern / false-green / scope-rationalisierung
**Was passierte:** Nachdem Stage 1 das neue False-Green im Run-Manifest gefunden hatte (E-066-Rest), wurde die Luecke per `@pytest.mark.xfail(strict=True)` festgeschrieben und mit „Zielpfad `pipeline/` ist Edit-deny" begruendet. Stage 2 wies empirisch nach, dass die Begruendung falsch war: die Fehlzaehlung entstand nicht im geschuetzten Konsumenten, sondern in der eigenen Datenmodellierung (Skip als `QAResult.OK`). Der Fix lag komplett in `qa_gates.py`.
**Warum falsch:** strict-xfail ist ein Instrument fuer VORGEFUNDENE Luecken, die man nicht schliessen darf. Auf eine selbst erzeugte Regression angewendet, verwandelt es eine Verschlechterung in eine gruene Testzeile, dokumentiert sie als unvermeidbar — und die Review-Kette meldet PASS, waehrend das System schlechter ist als vorher.
**Wie vermeiden:** (1) Vor jedem xfail die Frage beantworten: „Bestand diese Luecke VOR meinem Diff?" Nur bei JA ist xfail zulaessig. (2) Bei NEIN zuerst pruefen, ob der Konsument die Luecke hat oder die eigene Modellierung sie erzeugt — ein Zustand, den ein re-derivierender Konsument nicht falsch zaehlen KANN (eigener Enum-Zustand statt OK+Flag, oder den No-op-Eintrag gar nicht emittieren), ist fast immer im eigenen Modul herstellbar. (3) „Geschuetzter Pfad" ist erst dann ein Argument, wenn die Alternativen in der eigenen Datei durchgezaehlt und schriftlich verworfen wurden.
**Erkannt in:** `tests/test_qa_gates_leakage.py`, `src/assembled_core/qa/qa_gates.py` — Stage-2 senior-code-reviewer (F-senior-1, E-NEW-1) am 2026-08-01; xfail vor dem Commit wieder entfernt und durch den echten Fix ersetzt.
**Referenzen:** E-066 (die Klasse, um die es ging), E-061 (Symptom stilllegen statt Bug erkennen), CLAUDE.md „Keine falsche Sicherheit".

## E-068 — Common-mode-Verzerrung wird entscheidungsrelevant, sobald eine feste Konstante zum Parameter wird
**Datum:** 2026-08-01
**Kategorie:** logic-error / false-comparison / altcode-uebernahme
**Was passierte:** Mandat I entnahm bei Dividenden im total-return-adjustierten Preispanel nur die Steuer und liess die Lot-Basis unveraendert. Derselbe Dividenden-Euro wurde dadurch ZWEIMAL besteuert — einmal am Ex-Tag und ein zweites Mal im Veraeusserungsgewinn, weil er im Kurspfad steckt. Bei EINEM Satz (26,375 % fuer Kursgewinn UND Dividende) war das common-mode: alle Kandidaten gleich betroffen, Verdicts unberuehrt. Mandat II machte den Satz zum Parameter (GmbH: 1,49 % Kursgewinn / 29,83 % Dividende). Die Doppelbesteuerung skaliert mit dem KURSGEWINN-Satz — gemessen 52,75 % effektiv bei PRIVAT_DE gegen 31,32 % bei der GmbH. Damit machte das Modell GmbH-Dividenden 41 % BILLIGER, obwohl sie fachlich 13 % TEURER sind: **das Vorzeichen der Kernasymmetrie war gedreht**, die die ganze Kampagne messen soll. Vom Stage-2-Review VOR dem ersten Backtest gefangen (empirisch nachgemessen, nicht erschlossen).
**Warum falsch:** Beim Uebernehmen von Alt-Logik wird geprueft, ob sie sich identisch verhaelt (Regressionstest) — nicht, ob ihre stillschweigende RECHTFERTIGUNG („ist common-mode, kuerzt sich raus") unter der neuen Parametrisierung noch traegt. Genau diese Rechtfertigung ist die erste, die beim Pluralisieren eines Parameters bricht. Verschaerfend: die isolierten Unit-Tests auf dem Regime-Objekt waren GRUEN, waehrend das Portfolio-Verhalten invertiert war.
**Wie vermeiden:** (1) Wenn eine fest verdrahtete Konstante zum Parameter wird, fuer JEDE bekannte Vereinfachung des Altcodes nachrechnen, ob sie ueber den neuen Parameterbereich common-mode BLEIBT. (2) Effektive ERGEBNISgroessen end-to-end messen (hier: effektiver Steuersatz pro Dividenden-Euro ueber Kauf-Halten-Verkauf), nicht nur Komponenten unit-testen. (3) Bei Doppelerfassungs-Risiko explizit fragen: wird dieselbe wirtschaftliche Groesse an zwei Stellen erfasst, und hebt sich das nur zufaellig auf?
**Erkannt in:** `research/mandat2/portfolio.py` (book_dividend), Ursprung `research/mandat/verdict_engine.py` — Stage-2 senior-code-reviewer (F-senior-1 BLOCKER) beim Mandat-II-P0-Review 2026-08-01. Fix: Lot-Basis um die Bruttodividende anheben + vier End-to-End-Satztests.
**Referenzen:** E-051 (Determinismus-Artefakt in denselben Verdicts), E-066 (Aggregat luegt, Detail stimmt), Rule 40.

## E-069 — Steuer-/Kostenregime auf den Kandidaten angewandt, aber nicht auf die Instrumentenklasse des Benchmarks
**Datum:** 2026-08-01
**Kategorie:** false-comparison / benchmark-bias
**Was passierte:** Der Mandat-II-Plan fixierte „Benchmark: SPY Total Return … identisches Steuerregime". Das GmbH-Regime bildet §8b KStG ab (Anteile an Kapitalgesellschaften, ~1,49 %). SPY ist aber ein Investmentfonds und faellt unter §20 InvStG (Teilfreistellung 80 % KSt / 40 % GewSt -> ~11,57 % fuer eine Koerperschaft). Dasselbe Regime auf beide Seiten haette dem Einzelaktien-Kandidaten rund 10 Prozentpunkte gegenueber seinem Benchmark geschenkt — ein PASS waere ein Rechtsform-/Instrumentenklassen-Artefakt gewesen, kein Alpha. Mandat I hatte die Asymmetrie modelliert (`ETF_TAX = 0.185`), Mandat II liess sie beim Neubau ersatzlos weg.
**Warum falsch:** „Gleiche Behandlung von Kandidat und Benchmark" ist nur fair, wenn beide dieselbe rechtliche/oekonomische Klasse sind. Bei Steuer, Finanzierungskosten, Wertpapierleihe und Spread ist das oft NICHT der Fall — der Fairness-Reflex erzeugt dann genau den Bias, den er verhindern soll.
**Wie vermeiden:** (1) In Steuer-/Kostenmodellen die INSTRUMENTENKLASSE als expliziten Parameter fuehren (Aktie / Fonds / Derivat), nicht den Satz. (2) Beim Neubau eines Vorgaengermodells auflisten, welche Differenzierungen es hatte, und jeden Wegfall begruenden — ein Neubau ist der haeufigste Ort, an dem eine muehsam erarbeitete Unterscheidung still verschwindet. (3) Gilt analog fuer Hebel: Derivate fallen nicht unter §8b.
**Erkannt in:** `research/mandat2/PLAN.md`, `research/mandat2/tax_regimes.py` — Stage-2 senior-code-reviewer (F-senior-2 MAJOR) 2026-08-01. Fix: `AssetClass`-Enum + instrumentengerechte Saetze + Tests, die den ~10-pp-Unterschied pinnen.
**Referenzen:** E-068 (Geschwisterbefund desselben Reviews), Mandat I `h011_kandidat_a.py:47`.

## E-070 — Zugangs-Gate deckte nur den Hauptdatensatz; `clip` tarnte den Leak als gueltige Daten
**Datum:** 2026-08-01
**Kategorie:** wiring-gap / holdout-leak / silent-data-corruption
**Was passierte:** `research/mandat2/campaign_data.py` wurde gebaut, um die Holdout-Sperre technisch zu erzwingen, und routete `prices_verdict.parquet` korrekt durch `load_search()`. `dividends.parquet` und `sp500_historical_constituents.csv` wurden daneben direkt mit `pd.read_parquet`/`csv` gelesen — an derselben Sperre vorbei, in derselben Datei, deren Docstring „Jeder Phasen-Code holt seine Daten hier — nicht mit pd.read_parquet" vorschreibt. Verschaerfend: `_load_div_panel` snappte Ex-Daten per `clip(searchsorted(...), 0, len-1)` auf den Index. Alle 22.507 Dividendenzeilen aus dem Holdout (2017-2027) landeten dadurch nicht „irgendwo spaeter", sondern GEBUENDELT auf dem letzten Suchtag: 728 Symbole an einem Tag gegen einen Median von 5, SPY 57,93 statt 1,33. Gemessene Wirkung 7-8 % Endvermoegen beim Kandidaten, 0 % beim Benchmark — der Leak traf also ASYMMETRISCH.
**Warum falsch:** Eine Sperre, die nur den prominentesten Datensatz abdeckt, erzeugt genau die „unbeweisbare Behauptung", gegen die sie geschrieben wurde — und sie tut es unsichtbar, weil der gesperrte Pfad korrekt aussieht. Das `clip`-Idiom macht aus einem Bereichsfehler eine plausibel aussehende Randzeile statt eines Crashs oder eines offensichtlichen Index-Ueberlaufs: der Leak versteckt sich in gueltig aussehenden Daten.
**Wie vermeiden:** (1) Beim Bau eines Zugangs-Gates ALLE Rohquellen des Moduls auflisten und einzeln durchzaehlen, welche durch das Gate laufen — nicht nur die, um die es beim Schreiben ging. (2) `searchsorted`-Treffer am oberen Rand VERWERFEN statt klemmen, wenn der Rand „ausserhalb des erlaubten Fensters" bedeutet. (3) Panel-Plausibilitaetsguard als Test: keine Zeile darf ein Vielfaches der typischen Belegung tragen.
**Erkannt in:** `research/mandat2/campaign_data.py` — Stage-2 senior-code-reviewer (F-senior-1 BLOCKER) beim P1-Erstlauf-Review 2026-08-01.
**Referenzen:** E-052 (abgeleiteter Loader umgeht kanonischen Schritt), E-068.

## E-071 — Der Fix wirkte am Endpunkt, die entscheidende Kennzahl las ihn nie
**Datum:** 2026-08-01
**Kategorie:** logic-error / false-fixed / metrik-vs-mechanik
**Was passierte:** `engine.py` fuehrte eine End-Liquidation ein und begruendete sie ausdruecklich mit Mandat I BUG 2 (unversteuerte Buchgewinne drehten das Vorzeichen des Kernbefunds). Die Liquidation wirkte aber nur auf `equity.iloc[-1]`. Die gesperrte Zielfunktion in `metrics.py` bewertet ausschliesslich rollierende 10-Jahres-Fenster — von denen KEINES am letzten Index endet (letztes Fenster 2006-12-01..2016-12-01, Panelende 2016-12-30). Der umschichtende Kandidat trug seine Steuer laufend, Buy-and-Hold nie: Fenster-Median des Benchmarks fiel bei Steuer nur um 2,0 %, sein reales Endvermoegen aber um 15,9 %. Die Zielfunktion schenkte Buy-and-Hold rund 14 pp Steuerstundung.
**Warum falsch:** Ein Fix, der nur an einem einzigen Datenpunkt der Kurve wirkt, ist kein Fix, wenn die entscheidende Kennzahl diesen Punkt nie liest. Der Fehler ueberlebt den Umbau, weil er in eine ANDERE SCHICHT wandert — und die Doku weist ihn als behoben aus, was ihn schwerer auffindbar macht als vorher.
**Wie vermeiden:** Bei jedem „wir haben X behoben" pruefen, welche KENNZAHL X liest — nicht welche Zeile X schreibt. Fuer Steuer-/Kostenkorrekturen konkret: die Korrektur gehoert in die Kurve, die die Zielfunktion konsumiert, nicht in ihren Endpunkt. Fix hier: `Portfolio.latente_steuer()` + eine zweite Serie `equity_netto`, auf der ausgewertet wird. Kopplungstest: ein Regimewechsel muss den Fenster-Median in derselben Groessenordnung bewegen wie den Endwert.
**Erkannt in:** `research/mandat2/metrics.py`, `research/mandat2/engine.py` — Stage-2 senior-code-reviewer (F-senior-2 BLOCKER) 2026-08-01.
**Referenzen:** E-068 (dieselbe Familie: Korrektur an der falschen Stelle), E-066.

## E-072 — Stellparameter ohne Wirkung, aber mit voller Kostenseite
**Datum:** 2026-08-01
**Kategorie:** logic-error / false-mechanism / selbsterfuellende-bestaetigung
**Was passierte:** `run_momentum` akzeptierte `hebel` und belastete `finanzierung_pa` taeglich auf die berechnete Leihsumme. `Portfolio.buy` deckelte die Ausgabe aber auf verfuegbares Cash — es wurde NIE etwas geliehen. `hebel=2` und `hebel=3` zahlten 21k bzw. 41k Zinsen auf exakt die Exponierung von `hebel=1`. Der geplante P2-Hebel-Sweep haette daraus „Hebel ist tot" abgeleitet — und das BEFUND-Dokument sagte dieses Ergebnis bereits vorab voraus. Die Bestaetigung waere rein mechanisch gewesen.
**Warum falsch:** Gefaehrlicher als ein fehlender Parameter: die KOSTENseite war implementiert und die Diagnose (`finanzierung_gezahlt > 0`) meldete Aktivitaet, der Mechanismus sah im Log also lebendig aus. Kombiniert mit einer vorab notierten Ergebniserwartung ist das eine selbsterfuellende Bestaetigung.
**Wie vermeiden:** Fuer jeden neuen Stellparameter im selben Step einen Test schreiben, der die WIRKUNGSseite misst (Exponierung, Positionszahl, Haltedauer) — nicht die Kostenseite; die ist typischerweise der leichtere Teil und wird zuerst fertig. Vorab notierte Ergebniserwartungen ERHOEHEN die Beweislast fuer den Mechanismus, sie senken sie nicht.
**Erkannt in:** `research/mandat2/engine.py`, `research/mandat2/portfolio.py` — Stage-2 senior-code-reviewer (F-senior-4 MAJOR) 2026-08-01. Fix: `Portfolio.max_kredit` + Test, der bei `hebel=2` reale Mehr-Exponierung nachweist.
**Referenzen:** E-064 (enabled-Flip ohne Feld-Voraussetzung), E-066.

## E-073 — Plausibilitaetscheck mit dem kontaminierten Wert kalibriert
**Datum:** 2026-08-01
**Kategorie:** false-evidence / sanity-check-theater
**Was passierte:** Die neue Dividendenskalierung wurde mit „SPY-Rendite jetzt 2,59/1,18/2,54/2,37 % — plausible Bandbreite" belegt, und das Toleranzband im Docstring auf „grob 1,3-3,5 %" gesetzt. Die WAHREN Werte sind 2,33/1,05/2,23/2,04 %; die gemeldeten waren der verbliebene 10-20-%-Skalenfehler selbst. Beide Zahlenreihen lagen bequem im selben Band.
**Warum falsch:** Ein Sanity-Check, dessen Toleranz breiter ist als der Fehler, den er finden soll, erzeugt keine Evidenz, sondern eine Quittung. Er ist GEFAEHRLICHER als kein Check, weil er im Commit-Text als Beleg auftritt und die Frage fuer den naechsten Leser schliesst.
**Wie vermeiden:** Skalen-Checks gegen eine EXTERNE, unabhaengig bekannte Groesse pinnen (hier: EODHD-Rohkurs 45,7813 fuer SPY am 1995-01-03), nicht gegen ein Erwartungsband. Wenn nur ein Band verfuegbar ist: seine Breite gegen die erwartete Fehlergroesse pruefen und den Check ausdruecklich als schwach kennzeichnen. Und: ein Check ohne Aufrufer ist kein Check — `implizite_jahresrendite` hatte weder Test noch Runner.
**Erkannt in:** `research/mandat2/dividenden.py`, `research/mandat2/BEFUND_P1_ERSTLAUF.md` — Stage-2 senior-code-reviewer 2026-08-01.
**Referenzen:** E-066 (Aggregat luegt), E-060 (Cap-Wert statt Ist-Count).

## E-074 — Rueckwaerts-Rekonstruktion auf einem gefensterten Panel: der Anker liegt ausserhalb des Fensters
**Datum:** 2026-08-01
**Kategorie:** logic-error / normalisierungs-artefakt / holdout-inkonsistenz
**Was passierte:** `raw(t-1) = (raw(t)+d_t)*adj(t-1)/adj(t)` wurde korrekt hergeleitet, mit dem korrekten Anker „am letzten Kurs ist adj == raw". Aufgerufen wurde die Funktion aber auf dem BEREITS per Holdout-Gate abgeschnittenen Panel, dessen Adjustierung ueber die volle Historie bis 2026 normiert ist. Gemessen: SPY 1995-01-03 raw = 45,80 auf dem vollen Panel (EODHD-Ist 45,7813, Fehler 0,04 %) gegen 41,52 auf dem Suchfenster (-9,3 %). Folge: Dividendenpanel im Suchfenster in Summe +20,4 % ueberzeichnet, heterogen je Symbol (SPY +10..16 %, KO +19..34 %, XOM +30..51 %) und monoton wachsend zum Fensterrand. Zweiter, schwererer Effekt: im HOLDOUT-Fenster faellt Panelende und Datenende zusammen, dort ist der Anker korrekt — Suche und Holdout liefen auf UNTERSCHIEDLICH skalierten Dividenden, der eine Schuss waere nicht mit der Kalibrierung vergleichbar gewesen.
**Warum falsch:** Der Anker einer Rueckwaertsrekursion ist eine Eigenschaft der DATENQUELLE, nicht des Auswertungsfensters. Ein Gate, das Zeilen entfernt, verschiebt still den Anker — und weil Renditen von der Normierung unberuehrt bleiben, faellt es in KEINER return-basierten Pruefung auf.
**Wie vermeiden:** Niveau-abhaengige Groessen (Rohkurse, Dividende je Stueck, Nominalbetraege) auf der VOLLEN Quelle bestimmen und erst danach fenstern. Test: derselbe Faktor muss auf vollem und geschnittenem Panel herauskommen. Und ausdruecklich begruenden, warum das kein Leck ist — der Rohkurs von 1995 war 1995 bekannt; gefenstert werden Zeilen, nicht Skalen.
**Erkannt in:** `research/mandat2/dividenden.py`, `research/mandat2/campaign_data.py` — Stage-2 senior-code-reviewer (BLOCKER) 2026-08-01.
**Referenzen:** E-070 (Gate deckte nur den Hauptdatensatz), E-068.

## E-075 — Zwei Epsilon-Schwellen fuer dieselbe Groesse: unsterbliches Lot + nicht terminierender Zwangsverkauf
**Datum:** 2026-08-01
**Kategorie:** logic-error / non-termination / metrik-inflation
**Was passierte:** `Portfolio.sell` prueft `qty <= 0` als Frueh-Return, verkauft aber in `while rest > 1e-12`. Mengen zwischen 0 und 1e-12 passierten den Guard, wurden nie verkauft, das Lot blieb bestehen. Der spaeter eingebaute Delisting-Zwangsverkauf triggert auf „Symbol ist noch in lots" — und lief daraufhin JEDEN TAG neu: 7.986 von 8.001 Delisting-Triggern entfielen auf vier Staublots (PVN 2.841, TXU 2.332, BRL 2.029, TLAB 784), nur 15 waren echte Liquidationen. Der publizierte Trade-Count 16.293 bestand aus ~8.353 echten plus ~7.940 Phantom-Trades; das BEFUND-Dokument argumentierte mit „38.899 EUR Kosten (16.293 Trades)".
**Warum falsch:** Zwei Schwellen fuer dieselbe Groesse erzeugen ein Fenster, in dem eine Operation als erfolgreich gemeldet wird, ohne stattzufinden. Der Aufrufer glaubt, die Position sei glattgestellt — und ein Retry-Mechanismus darueber terminiert nie. Die Inflation landet in einer publizierten Kennzahl.
**Wie vermeiden:** Eine Epsilon-Konstante je Groesse (hier `Portfolio.EPS_QTY`), identisch in Frueh-Return, Schleife und Aufraeumzweig. Bei jedem „Zwangs"-Mechanismus einen Terminierungstest: nach der Aktion darf die Ausloesebedingung nicht mehr wahr sein.
**Erkannt in:** `research/mandat2/portfolio.py`, `research/mandat2/engine.py` — Stage-2 senior-code-reviewer 2026-08-01.
**Referenzen:** E-072 (Mechanismus sieht im Log lebendig aus), E-066.

## E-076 — Kosten im Fliesstext subtrahiert statt im Lauf gemessen (Zinseszins fehlt)
**Datum:** 2026-08-01
**Kategorie:** false-evidence / handrechnung-statt-messung
**Was passierte:** Der P1-Befund verglich die vermoegensverwaltende GmbH mit dem Privatanleger und schrieb: „Selbst nach 22 Jahren Rechtsformkosten (3.500 EUR/J = 77.000 EUR) bleiben +65.000 EUR." Der Parameter `fixkosten_pa` EXISTIERTE und floss real ab — er wurde nur nicht gesetzt. Mit ihm gemessen: 73.500 EUR eingezahlte Fixkosten (21 Jahreswechsel, nicht 22) kosten ueber die Laufzeit **137.663 EUR** Endvermoegen; die entgangene Verzinsung ist fast so gross wie der Nominalbetrag noch einmal. Der GmbH-Vorsprung schrumpfte von +141.742 auf +4.079 EUR — und auf der GESPERRTEN Zielfunktion (Median ueber rollierende Fenster) war die GmbH mit Fixkosten SCHLECHTER als privat (1,0524 gegen 1,0910). Der publizierte Befund 3 war damit durch die eigene Engine widerlegt. Der eigene Plan hatte es woertlich vorhergesagt: „fuer die Frage 'GmbH oder privat?' muss er gesetzt werden, sonst ist das Ergebnis geschoent."
**Warum falsch:** Eine Nominalsubtraktion im Text unterschlaegt jeden Pfadeffekt (Zinseszins, Steuerzeitpunkte, Positionsgroessen). Sie sieht aus wie eine Rechnung, ist aber eine Schaetzung — und sie erscheint im Dokument mit derselben Autoritaet wie eine gemessene Zahl. Verschaerfend: der Fehler ging in dieselbe Richtung wie ein bereits einmal gekippter Befund, also genau dorthin, wo die Erwartung lag.
**Wie vermeiden:** (1) **Keine Zahl in einem Befund-Dokument, die nicht aus dem Ergebnis-Artefakt stammt.** Existiert ein Parameter, wird er GESETZT und gemessen, nicht im Text nachgebildet. (2) Bei Kosten-/Steuerabzuegen ueber lange Zeitraeume den Lauf wiederholen statt zu subtrahieren — bei 20+ Jahren ist der Zinseszins groesser als der Nominalbetrag. (3) Ist ein Befund schon einmal gekippt, die naechste Fassung in derselben Richtung besonders skeptisch pruefen.
**Erkannt in:** `research/mandat2/BEFUND_P1_ERSTLAUF.md`, `research/mandat2/p1_baseline.py` — Stage-3 task-completion-auditor 2026-08-01, per Gegenrechnung mit der vorhandenen Engine.
**Referenzen:** E-073 (Sanity-Check mit dem kontaminierten Wert kalibriert), E-060, CLAUDE.md „Keine falsche Sicherheit".

## E-077 — DSR mit grossem N und kleinem V: Varianz aus der eigenen Klonfamilie geschaetzt
**Datum:** 2026-08-01
**Kategorie:** false-evidence / statistik-fehlanwendung
**Was passierte:** Der Deflated Sharpe Ratio wurde fuer einen Kandidaten mit `n_trials=2144` (kumulierter Kampagnen-Zaehler) und `variance_across_trials` aus der EMPIRISCHEN Streuung von 37 Varianten gerechnet — die aber Fast-Klone derselben Strategie mit leicht verschobenem Gate-Fenster waren. Ihre Sharpes liegen naturgemaess eng beieinander: gemessene Varianz 3,7e-05 gegen IID-Naeherung 1,8e-04, also 4,8-fach zu klein. Ergebnis: p = 0,9988 (bestanden). Mit der konservativen IID-Naeherung und demselben N: p = 0,8783 (durchgefallen). Dieselbe Struktur schoente auch PBO: CSCV setzt unterschiedliche Strategien in den Spalten voraus; bei Klonen ueberlebt der In-sample-Sieger fast immer OOS, was 20 % PBO als Struktur-Artefakt statt als Robustheitsnachweis erzeugt.
**Warum falsch:** `n_trials` und `variance_across_trials` muessen dieselbe Bezugsgroesse haben — beide beschreiben DIE SUCHE. Ein grosses N aus der ganzen Kampagne mit einem kleinen V aus einer engen Unterfamilie zu kombinieren, senkt die Schwelle kuenstlich und macht die Korrektur wirkungslos. Verschaerfend: der Fehler wirkt genau dort, wo die Korrektur schuetzen soll, und er sieht wie sorgfaeltige Arbeit aus („empirische statt genaeherter Varianz").
**Wie vermeiden:** (1) V aus Sharpes ueber HETEROGENE Strategiefamilien schaetzen (verschiedene Signale, gegatet und ungegatet, Zufallskontrollen), nie ueber Parametervarianten einer Strategie. (2) Bei Zweifel BEIDE Varianten rechnen und die konservative als Entscheidungsgrundlage nehmen. (3) Dasselbe Prinzip fuer PBO: die CSCV-Matrix braucht heterogene Spalten, sonst misst sie Aehnlichkeit statt Ueberanpassung.
**Erkannt in:** `research/mandat2/p7_dsr_pbo.py` — Eigenkontrolle 2026-08-01, bevor der Holdout-Schuss ausgeloest wurde. Folge: KEIN Schuss, Holdout bleibt versiegelt.
**Referenzen:** E-073 (Check mit dem kontaminierten Wert kalibriert), E-076 (Handrechnung statt Messung), Rule 40.

## E-078 — Effektive Stichprobe der ENTSCHEIDUNG verwechselt mit der Zahl der Beobachtungen
**Datum:** 2026-08-02
**Kategorie:** false-evidence / statistik-fehlanwendung / scheinpraezision
**Was passierte:** Der DSR fuer den Trendfilter-Kandidaten wurde auf 5.548 Tagesrenditen gerechnet — und die Zahl sah dadurch praezise aus (p = 0,9512 gegen eine 0,95-Schwelle). Die Forensik zeigte: das Gate wird nur an 264 Monatsenden gelesen und trifft ueber 22 Jahre **12 bis 18 wirksame Regimewechsel**, von denen im Ergebnis etwa VIER zaehlen (Ausstieg und Wiedereinstieg in je einem der beiden Baerenmaerkte). Die drei getesteten Trend-Definitionen stimmen an 86,4 % der Rebalance-Termine ueberein; sie unterscheiden sich an einer Handvoll Tage, und genau die entscheidet ueber Bestehen oder Durchfallen. Passend dazu ist die Zahl gerissener Fenster ueber alle 72 Robustheitslaeufe fast binaer (0 / 64 / 69) — ein einzelner Kall kippt eine ganze Fensterklasse.
**Warum falsch:** Die Stichprobengroesse einer Aussage ist nicht die Zahl der DATENPUNKTE, sondern die Zahl der unabhaengigen ENTSCHEIDUNGEN, die das Ergebnis erzeugen. Ein Mechanismus mit vier wirkenden Kalls, gemessen auf 5.548 Tagen, liefert eine Zahl mit drei Nachkommastellen und der Aussagekraft eines Wuerfelwurfs. Keine Varianzkorrektur (E-077) behebt das — sie korrigiert die Selektion ueber Varianten, nicht die Duennheit des Mechanismus selbst.
**Wie vermeiden:** Vor jeder Signifikanzaussage die effektive Stichprobe des MECHANISMUS zaehlen, nicht die der Zeitreihe: Wie oft trifft er tatsaechlich eine Entscheidung? Wie viele davon veraendern das Ergebnis? Bei Regime-/Timing-Modellen ist das fast immer eine ein- bis zweistellige Zahl. Konkret: Zahl der Zustandswechsel AN DEN AUSWERTUNGSTERMINEN ausweisen (nicht die taeglichen Flips — hier 140 gegen 18) und die Uebereinstimmung konkurrierender Signalvarianten an denselben Terminen messen.
**Erkannt in:** `research/mandat2/p9_gate_forensik.py` — Eigenkontrolle 2026-08-02, nach dem bereits negativen P8-Verdikt. Das Ergebnis verstaerkt es: der Kandidat ist nicht knapp gescheitert, er war nie belastbar genug, um knapp zu sein.
**Referenzen:** E-077 (DSR-Varianz aus der Klonfamilie), E-073, Rule 40.

## E-079 — Benchmark mit anderer Gewichtungsmethode als der Kandidat: Faktorexposition als Alpha gemessen
**Datum:** 2026-08-02
**Kategorie:** false-comparison / benchmark-bias
**Was passierte:** Die gesamte Mandat-II-Kampagne (P1 bis P8) mass GLEICHgewichtete Kandidaten (20 Namen, je 1/20) gegen SPY, einen KAPITALgewichteten Index. Erst die nachgereichte Kontrolle (P11) baute den passenden Massstab: ein gleichgewichteter Index desselben Universums erreicht Median 2,594 gegen SPY 1,948 — er schlaegt SPY also um Faktor 1,33 OHNE jede Auswahl, ohne Signal, ohne Timing. Damit verschwand der P3-Zufallsbefund fast vollstaendig: 20 zufaellige Namen liegen gegen SPY bei 1,40x, gegen den EW-Index nur noch bei 1,05x (2 von 8 Seeds sogar darunter). Auch der Hauptkandidat verlor rund ein Viertel seines gemessenen Vorsprungs (3,43x -> 2,57x).
**Warum falsch:** Gleichgewichtung gegenueber Kapitalgewichtung ist eine bekannte, gut dokumentierte Faktorexposition (kleinere Namen hoeher gewichtet) — kein Alpha. Wer sie nicht im Benchmark neutralisiert, misst sie und nennt sie Ueberrendite. Hier ging es glimpflich aus, weil die Verdicts ohnehin negativ waren und gegen einen HAERTEREN Massstab negativ bleiben; in die andere Richtung waere es fatal gewesen: ein „bestandener" Kandidat haette womoeglich nur die Gewichtung abgebildet.
**Wie vermeiden:** Der Benchmark muss dieselbe Gewichtungsmethode haben wie der Kandidat — gleichgewichtete Strategie gegen gleichgewichteten Index, kapitalgewichtete gegen kapitalgewichteten. Bei abweichender Methode BEIDE Massstaebe ausweisen und die Differenz als das benennen, was sie ist. Gilt analog fuer Universum (Large vs. Small), Waehrung und Rebalancing-Frequenz. Faustregel: jede Dimension, in der sich Kandidat und Benchmark unterscheiden, wird gemessen — ob man will oder nicht.
**Erkannt in:** `research/mandat2/p11_gleichgewicht.py` — Eigenkontrolle 2026-08-02, aus einer in P3 selbst notierten offenen Frage.
**Referenzen:** E-069 (Instrumentenklasse des Benchmarks), E-078, CLAUDE.md „Keine falsche Sicherheit".

## E-080 — Datenverfuegbarkeit aus dem Gedaechtnis behauptet statt an der API geprueft
**Datum:** 2026-08-03
**Kategorie:** false-evidence / ungeprueft-uebernommen
**Was passierte:** Der Mandat-II-Abschluss erklaerte den Intraday-Strang fuer nicht durchfuehrbar: „braeuchte das EODHD-Intraday-Paket (ab ca. Okt 2020)". Beide Teile waren falsch. (1) Das Paket ist laengst freigeschaltet UND wird im Repo bereits verwendet — 452k 5-Minuten-Bars (4 ETFs, 2020-2026) und 246k 1-Minuten-Bars (20 Titel, 2024-2026) lagen auf der Platte. (2) Die Grenze „ab Okt 2020" stammte aus einem Memory-Eintrag und gilt nur fuer den 1h-Endpunkt; der 1m-Endpunkt liefert Einzelaktien ab 2004 — also 22 Jahre und damit genug fuer die 10-Jahres-Fenster der Zielfunktion. Der Nutzer hat den Fehler bemerkt („eigentlich gab es Zugriff auf Intraday-Daten und wir hatten diese runtergeladen").
**Warum falsch:** Eine Verfuegbarkeitsaussage ist eine EMPIRISCHE Aussage. Sie aus einem Memory-Eintrag zu uebernehmen — der zudem eine Teilaussage ueber EINEN Endpunkt war — und daraus ein „datenblockiert" im Abschlussbericht zu machen, schliesst einen ganzen Forschungsstrang auf einer ungeprueften Grundlage. Verschaerfend: die Gegenevidenz lag als Datei im Repo, ein `find` haette gereicht.
**Wie vermeiden:** (1) Bevor ein Strang als „datenblockiert" deklariert wird: erst `find`/`ls` ueber das Repo, dann ein Live-Probe-Call gegen die API mit mehreren Jahren und mehreren Endpunkten. (2) Memory-Eintraege zu Entitlements sind Momentaufnahmen und oft endpunkt-spezifisch — als Hinweis behandeln, nie als Beleg. (3) „Blockiert" ist die teuerste aller Aussagen, weil sie Arbeit beendet; sie braucht deshalb die staerkste Evidenz, nicht die schwaechste.
**Erkannt in:** `research/mandat2/ABSCHLUSS.md` — vom Nutzer korrigiert 2026-08-03; danach empirisch nachgeprueft und der Ingest gebaut (`research/mandat2/intraday_pull.py`).
**Referenzen:** E-076 (Handrechnung statt Messung), CLAUDE.md „Keine falsche Sicherheit", Rule 10.

## E-081 — Vor-Gate-Sichtung erzeugt Wissen, das dann ins Dokument leckt
**Datum:** 2026-08-03
**Kategorie:** holdout-leak / disziplinbruch / explorations-kontamination
**Was passierte:** Beim ersten Sichten der Intraday-Rohdaten habe ich `data/raw/intraday_1h/*.parquet` direkt gelesen, um Splits zu finden — zu diesem Zeitpunkt existierte `intraday_data.load_intraday()` (das Gate) noch nicht. Eines der gefundenen Beispiele, „AFL 2018-03-19 −49,7 % (Split 2:1)", stammt aus dem HOLDOUT-Zeitraum (Cutoff 2016-12-31). Es landete anschliessend im Modul-Docstring UND im versionierten Befund-Dokument. Die Auswertungen selbst waren sauber — alle drei Skripte lesen ausschliesslich ueber das Gate, das Panel endet nachweislich am 2016-12-30 und enthaelt 0 Bars danach. Es gab also keinen verwerteten Informationsvorteil, aber eine publizierte Holdout-Beobachtung.
**Warum falsch:** Ein Zugangs-Gate schuetzt nur ab dem Moment seiner Existenz. Die Sichtungsphase DAVOR ist genau der Zeitraum, in dem man das Datenmaterial am unbefangensten anschaut — und alles, was man dort sieht, wandert als „Beispiel", „Plausibilitaetscheck" oder „Motivation" in Docstrings und Befunde. Der Leak entsteht nicht im Code, sondern in der Prosa. Verschaerfend: von Hand abgeschriebene Beispieltabellen entziehen sich jeder spaeteren automatischen Pruefung, weil sie in keinem `results/*.json` stehen (vgl. E-073).
**Wie vermeiden:** (1) Das Gate ZUERST bauen, dann sichten — auch fuer explorative Blicke. (2) Ist doch vor dem Gate gesichtet worden, gilt jede dabei entstandene Beobachtung als kontaminiert und ist nicht zitierfaehig; sie muss aus dem gegateten Fenster neu erhoben werden. (3) Diagnose-Tabellen (Splits, Ausreisser, Abdeckung) als versioniertes Artefakt aus dem gegateten Panel erzeugen, nicht als Prosa abschreiben — dann ist der Cutoff strukturell erzwungen statt erinnert. Fix hier: `_split_diagnose()` berechnet die Tabelle auf dem gegateten Fenster; die 2018er Zeile verschwand dadurch von selbst.
**Erkannt in:** Stage-1-Review (F-test-6) zu `research/mandat2/BEFUND_P12_INTRADAY.md` + `intraday_data.py`.
**Referenzen:** E-070 (Gate deckte nur den Hauptdatensatz), E-073/E-076 (Zahlen nur aus results/*.json), E-080.

## E-082 — NaN-Vorlauf macht den Startzeitpunkt zur zweiten, unbemerkten Variable
**Datum:** 2026-08-03
**Kategorie:** false-comparison / ceteris-paribus-bruch
**Was passierte:** Ein Haltedauer-Sweep (1 Stunde bis 2 Jahre) sollte laut Dokumentation genau EINEN Parameter variieren. Der Rueckblick des Signals war als „20x Haltedauer" definiert, also mitskaliert; am Anfang des Panels ist ein solcher Score NaN, und der fail-closed-Pfad haelt dann Cash. Ergebnis: die 1-Stunden-Variante war nach 20 Bars investiert, die 2-Jahres-Variante erst nach 8.064 Bars — **vier Jahre** eines 13-Jahres-Fensters, also 31 % der Zeit in Cash. Die Zufallskontrolle hatte gar keine NaN und war ab Bar 0 investiert. Verglichen wurden damit Startzeitpunkte, nicht Haltedauern. Zusaetzlich schoente die flache Cash-Phase den ausgewiesenen MaxDD der langen Zeilen.
**Warum falsch:** „Alles andere bleibt fest" ist eine Behauptung ueber den REALISIERTEN Lauf, nicht ueber die Parameterliste. Ein Warm-up ist ein stiller Nebenparameter: er wird nicht uebergeben, nicht geloggt und taucht in keiner Ergebnisspalte auf — aber er verschiebt den effektiven Anlagehorizont um Jahre. Die Richtung war hier zufaellig konservativ (der Fehler benachteiligte das lange Ende, die Schlussfolgerung hielt trotzdem); das ist Glueck, kein Verfahren.
**Wie vermeiden:** (1) In jedem Parameter-Sweep einen GEMEINSAMEN Startindex erzwingen — der laengste Vorlauf ueber alle Varianten, angewandt auf alle, inklusive Benchmark und Kontrollen. (2) Kontroll-Scores (Zufall, Konstante) muessen die NaN-Maske des echten Signals ERBEN, sonst vergleicht die Kontrolle einen anderen Zeitraum. (3) Als Pflichtdiagnose je Sweep-Zeile ausweisen: erster investierter Bar und Anteil der Zeit in Cash — steht das nicht in der Tabelle, ist „ceteris paribus" unbelegt. (4) Laesst man den Rueckblick mitskalieren, ist es kein Ein-Parameter-Sweep mehr; entweder Rueckblick fixieren oder beide Familien getrennt ausweisen.
**Erkannt in:** Stage-1-Review (F-test-2/F-test-3) zu `research/mandat2/p12_intraday_haltedauer.py`.
**Referenzen:** E-079 (Benchmark-Methode), E-072 (Stellparameter ohne Wirkung).

## E-083 — Das Bereinigungsverfahren erzeugt genau den Effekt, den die Studie sucht
**Datum:** 2026-08-03
**Kategorie:** false-mechanism / methodenartefakt / annahme-statt-messung
**Was passierte:** Intraday-Stundenbars wurden ueber `faktor(tag) = adj_close(tag) / roh_close(letzte Bar des Tages)` auf den tagesadjustierten Anker gehoben. Der Docstring behauptete, damit wuerden „Uebernacht-Renditen korrekt um Split UND Dividende bereinigt". Die Stage-2-Review hat den Faktor NACHGEMESSEN statt geglaubt: er ist keine reine Kapitalmassnahmen-Treppe, sondern absorbiert zusaetzlich die Differenz zwischen Vendor-Tagesschluss und letzter Stundenbar (Schlussauktion, fehlende Bars — 26-27 unvollstaendige Tage je Symbol). Gemessen ueber acht Symbole: Streuung von d log f 11-31 bps/Tag bei lag-1-Autokorrelation -0,16 bis -0,47. Das ist **reversierendes** Rauschen, das per Konstruktion in JEDE Uebernacht-Rendite eingeht — also gleichgerichtet mit dem Short-Term-Reversal, den ein Intraday-Test am kurzen Ende zu finden hofft, und mit derselben Horizontabhaengigkeit (bei 1 Stunde gross, ab 1 Tag < 7 %).
**Warum falsch:** Ein Bereinigungsschritt, dessen Residuum dieselbe Signatur traegt wie der gesuchte Effekt, ist keine Bereinigung, sondern eine Signalquelle. Besonders tueckisch, weil die Bereinigung fachlich RICHTIG und sogar notwendig ist (ohne sie misst man Splits, vgl. E-080-Kontext) — der Fehler liegt nicht im Verfahren, sondern in der ungepruefeten Annahme ueber seine Feinstruktur. Auf Horizonten unterhalb der Faktorfrequenz (hier: stuendlich unter taeglich) ist diese Feinstruktur nicht mehr vernachlaessigbar.
**Wie vermeiden:** (1) Wo ein Faktor als Stufenfunktion GEDACHT ist, seine Stufigkeit messen — Verteilung und Autokorrelation von `d log f` — statt sie anzunehmen. (2) Eine Gegenprobe mit erzwungen stufigem Faktor rechnen und BEIDE Ergebnisse ausweisen; die Differenz ist die Artefaktschranke des Verfahrens. (3) Liegt der Untersuchungshorizont unterhalb der Faktorfrequenz, ist diese Schranke Pflichtangabe im Befund, nicht Fussnote. (4) Formulierungen wie „korrekt bereinigt" brauchen eine Messung als Beleg, sonst „auf Tagesebene bereinigt; Feinstruktur ungeprueft".
**Erkannt in:** Stage-2-Review (F-senior-7) zu `research/mandat2/intraday_data.py`. Fix: `_stufig_machen()` + `load_intraday(stufig=True)` als Gegenprobe, Ergebnisse beider Laeufe getrennt versioniert.
**Referenzen:** E-070, E-074 (Rekonstruktionsanker), E-082.

## E-084 — Deckelung eines mitskalierten Parameters erzeugt Pseudo-Zeilen, und die gewinnen
**Datum:** 2026-08-03
**Kategorie:** logic-error / false-comparison / stiller-parameter
**Was passierte:** In einem Haltedauer-Sweep war der Signal-Rueckblick als `min(bars * 20, WARMUP)` definiert. Fuer die drei laengsten Haltedauern (Quartal, Jahr, 2 Jahre) griff die Deckelung, alle drei bekamen denselben Rueckblick von 4.355 Bars. Diese drei Zeilen unterschieden sich damit nur noch in der Umschichtungsfrequenz eines identischen Signals — und lieferten ausgerechnet die drei besten Netto-Werte des gesamten Sweeps (2,752x / 3,519x / 3,696x) sowie die einzigen, die den Buy-and-Hold-Benchmark (3,138x) schlugen. Der Spitzenwert des Experiments war ein Deckelungsartefakt. Verschaerfend: der Modul-Docstring fuehrte genau diese Deckelung als BEHOBENEN Fehler eines frueheren Entwurfs auf — behoben war nur der Deckelwert, nicht der Mechanismus.
**Warum falsch:** Eine Deckelung ist eine stille Parameteraenderung. Sie wird nicht uebergeben, steht in keiner Ergebnisspalte und macht aus einem Sweep an der Obergrenze eine Wiederholung derselben Konfiguration. Wenn die Bestwerte aus dem gedeckelten Bereich stammen, misst man den Deckel und schreibt es dem Sweep-Parameter zu — der Verdacht faellt immer zuerst auf den Parameter, der laut Tabelle variiert.
**Wie vermeiden:** (1) Den Sweep-Bereich so waehlen, dass keine Zeile die Deckelung erreicht — oder die Grenze anheben; wo beides nicht geht, die betroffenen Zeilen NICHT als Sweep-Ergebnis ausweisen. (2) Der tatsaechlich verwendete Parameterwert gehoert in jede Ergebniszeile und ins JSON, dann faellt Identitaet beim Lesen sofort auf. (3) Vor jeder Interpretation pruefen, ob der Bestwert aus dem gedeckelten Bereich stammt. (4) Wird ein Nebenparameter mitskaliert, ist es kein Ein-Parameter-Sweep — entweder fixieren oder beide Familien getrennt ausweisen (vgl. E-082).
**Erkannt in:** Stage-2-Review (F-senior-4) zu `research/mandat2/p12_intraday_haltedauer.py`.
**Referenzen:** E-082 (NaN-Vorlauf als zweite Variable), E-079.

## E-085 — Befund und Lauf driften auseinander, sobald der Code NACH dem Schreiben korrigiert wird
**Datum:** 2026-08-03
**Kategorie:** false-evidence / dokumentations-drift / review-ketten-spezifisch
**Was passierte:** Der Befund zu einer Forschungsphase wurde aus Lauf 1 geschrieben — korrekt, jede Zahl aus `results/*.json`. Die anschliessende Review-Remediation aenderte Warm-up, Benchmark-Logik, Sitzungsfilter und Rueckblick-Familien; der Lauf wurde wiederholt, das Befund-Dokument nicht. Ergebnis: eine versionierte Tabelle, deren jede Zeile dem eigenen JSON widersprach — „1 Stunde brutto 0,159x" gegen tatsaechlich 2,297x, „Halten 5,066x" gegen 3,138x — und **zwei von drei Schlussfolgerungen hatten sich umgekehrt**. Ich hatte die alten Zahlen zu diesem Zeitpunkt bereits an den Nutzer berichtet. Zusaetzlich stuetzte ein Folgemodul seine gesamte Existenzbegruendung im Docstring auf die veraltete Richtung.
**Warum falsch:** E-073/E-076 verlangen, dass Zahlen aus `results/*.json` STAMMEN. Das ist beim Schreiben erfuellbar und danach nie wieder ueberprueft. Der teure Fall ist nicht die erfundene Zahl, sondern die einmal korrekt abgeschriebene, die durch einen spaeteren Fix veraltet. In einer Review-Kette tritt dieser Fall SYSTEMATISCH auf, weil die Remediation per Definition nach dem Schreiben kommt — und je gruendlicher die Review, desto groesser die Drift.
**Wie vermeiden:** (1) Befund-Tabellen aus dem JSON RENDERN, nicht abschreiben — dann ist Drift strukturell unmoeglich. (2) Wo das nicht praktikabel ist: vor jedem Commit einen maschinellen Abgleich Befund-Zahl gegen JSON-Feld ausfuehren und das Ergebnis melden. (3) Jede Remediation, die einen Lauf wiederholt, hat „Befund neu erzeugen" als TEIL des Fixes, nicht als Folgeaufgabe. (4) Wurden veraltete Zahlen bereits berichtet, gehoert die Korrektur unaufgefordert und explizit an den Empfaenger — nicht stillschweigend in die naechste Version.
**Erkannt in:** Stage-2-Review (F-senior-1/2/3) zu `research/mandat2/BEFUND_P12_INTRADAY.md` und `p12c_reversal_kostenschwelle.py`.
**Referenzen:** E-073, E-076, E-081.

## E-086 — Erzeuger gefixt, Artefakt nicht neu erzeugt: Finding gilt als erledigt, Deliverable zeigt den alten Wert
**Datum:** 2026-08-03
**Kategorie:** wiring-gap / false-fixed / artefakt-drift
**Was passierte:** Ein Review-Finding bemaengelte die Formatierung eines Verwurfsgrunds (`Abdeckung nur 54.5%` im deutschen Fliesstext). Ich korrigierte den ERZEUGER (`intraday_data.py`) und meldete das Finding als adressiert. Das committete `results/*.json` und der daraus gerenderte Befund trugen den alten String unveraendert weiter — die Korrektur wirkt erst nach einem Neulauf, der 110 Trials kostet. Wer nur den Diff liest, sieht ein geschlossenes Finding; wer das Dokument liest, sieht den alten Wert. Eine zweite Naht derselben Klasse entstand beim Versuch, eine hartkodierte Schwelle zu entkoppeln: der Renderer importierte die Code-Konstante `MIN_ABDECKUNG` — damit haette ein Rendern gegen ein AELTERES Ergebnis-JSON eine Zahl behauptet, die fuer diesen Lauf nie galt.
**Warum falsch:** Das ist die Umkehrung von E-085. Dort veraltete die Doku gegen den Lauf, hier veraltet das ARTEFAKT gegen den Code. Beide Male behauptet ein Dokument etwas ueber einen Lauf, das dieser Lauf nicht hergibt. Besonders tueckisch, weil der Fix technisch korrekt ist und im Diff ueberzeugend aussieht — nur seine WIRKUNG fehlt. Und: eine Code-Konstante zu importieren sieht wie Entkopplung aus, ist aber eine neue Kopplung an den jetzigen Stand.
**Wie vermeiden:** (1) Bei jeder Aenderung an einem Erzeuger pruefen, ob ein committetes Artefakt davon abhaengt — und ob ein Neulauf bezahlbar ist. (2) Ist er es nicht, den Fix in die RENDERschicht legen: der wirkt sofort und rueckwirkend auf Altartefakte. (3) Parameter, die ein Ergebnis erklaeren (Schwellen, Kosten, top_k), gehoeren INS Ergebnis-Artefakt; der Renderer darf nur das Artefakt lesen, nie den Live-Code. (4) Fehlt das Feld im Altartefakt, muss der Renderer das SAGEN, nicht eine Zahl aus dem Code einsetzen. (5) Nie als erledigt melden, was nur im Diff erledigt ist.
**Erkannt in:** Stage-2-Review (F-senior-1/2) zu `research/mandat2/`.
**Referenzen:** E-085 (Gegenrichtung), E-073, E-076.

## E-087 — abs() wirft genau die Information weg, auf der die Behauptung beruht
**Datum:** 2026-08-03
**Kategorie:** logic-error / unpruefbare-behauptung
**Was passierte:** Ein Befund behauptete, die Abweichung zwischen Haupt- und Kontrolllauf sei „nicht systematisch gerichtet (das Vorzeichen wechselt)" — und berechnete die zugehoerigen Deltas mit `abs()`. Der Generator konnte seine eigene Kernbehauptung damit nicht pruefen; sie stand fest im Code, waehrend die Zahl daneben aus den Daten kam. Nachgerechnet stimmte sie zufaellig (signierte Deltas -3,0 / +5,4 / -2,4 / +12,0 %), aber nur zufaellig.
**Warum falsch:** „Nicht systematisch gerichtet" IST eine Aussage ueber Vorzeichen. Eine Kennzahl zu bilden, die das Vorzeichen verwirft, und daneben eine Vorzeichenaussage zu treffen, ist ein Selbstwiderspruch im Datenfluss — die Behauptung ueberlebt jede Datenaenderung unveraendert. Generell: wo eine Aggregation eine Dimension kollabiert (abs, max, mean), darf keine Aussage ueber genau diese Dimension danebenstehen.
**Wie vermeiden:** (1) Vor jeder Aggregation fragen, welche Aussage sie tragen soll — und ob die Aggregation die dafuer noetige Information erhaelt. (2) Vorzeichenaussagen aus signierten Werten ableiten und im Generator VERZWEIGEN (`min >= 0 or max <= 0`), damit die Gegenaussage bei anderen Daten automatisch erscheint. (3) Faustregel: jede pruefbare Wertung im Text braucht den Ausdruck, der sie prueft, unmittelbar daneben.
**Erkannt in:** Stage-2-Review (F-senior-3) zu `research/mandat2/render_befund_p12.py`.
**Referenzen:** E-085, E-089.

## E-088 — Extremwerte aus verschiedenen Zeilen im selben Satz als ein Fall gelesen
**Datum:** 2026-08-03
**Kategorie:** false-comparison / scheinpraezision
**Was passierte:** Ein Befundsatz stellte die Artefaktschranke (`max` ueber alle Zeilen = 12,0 %) neben den Break-even (`min` ueber alle Zeilen = 1 bps) und schloss, beide seien „von derselben Groessenordnung wie der verbleibende Spielraum". Die 12,0 % stammten aus der 1-Tag-Zeile — derjenigen, die gar keinen Break-even hat. Die 1 bps stammten aus den kurzen Zeilen. Fuer die tragende Zeile (1 Stunde) betraegt die Schranke 3,0 %. Der Satz kombinierte zwei nicht zusammengehoerige Extremwerte zu einer Aussage ueber einen Fall, den es nicht gibt. Die vorherige Fassung hatte die Zahl hartkodiert; die „Verbesserung" per `min()`/`max()` zementierte den Fehler und liess ihn zugleich gerechnet aussehen.
**Warum falsch:** Zwei Aggregate ueber dieselbe Tabelle sind keine zwei Eigenschaften desselben Objekts. `max(A)` und `min(B)` beschreiben im Allgemeinen VERSCHIEDENE Zeilen; sie in einem Satz zu verbinden erzeugt einen Fall, der in den Daten nicht vorkommt. Dass beide Zahlen aus dem JSON kommen, macht die Verbindung nicht wahr — Herkunft ersetzt keinen Zeilenbezug.
**Wie vermeiden:** (1) Sollen zwei Kennzahlen zusammen gelesen werden, aus DERSELBEN Zeile ziehen und die Zeile im Satz benennen. (2) Die relevante Zeile explizit waehlen (hier: die mit dem hoechsten Bruttowert) statt sie durch `min`/`max` implizit entstehen zu lassen. (3) Alternativ je Zeile ausweisen und auf die Gesamtaussage verzichten. (4) Warnsignal: ein Satz mit zwei Aggregaten unterschiedlicher Richtung.
**Erkannt in:** Stage-2-Review (F-senior-5) zu `research/mandat2/render_befund_p12.py`.
**Referenzen:** E-078 (Scheinpraezision), E-079.

## E-089 — Generator schuetzt die Zahlen vor Drift, nicht die Schlussfolgerung
**Datum:** 2026-08-03
**Kategorie:** false-evidence / halbe-absicherung
**Was passierte:** Nach E-085 wurde das Befund-Dokument generiert statt geschrieben, damit keine Zahl gegen ihren Lauf driften kann. Die WERTUNGEN blieben aber feste Strings: „Das kurze Ende traegt nicht", „also im Plus", „der Abstand ist klein gegenueber der Streuung". Nur eine von drei Kernaussagen hatte eine Datenverzweigung. Haetten sich die Ergebnisse umgedreht, waeren die Zahlen korrekt und die Schlussfolgerungen falsch gewesen — dasselbe Dokument, dieselbe Klasse Fehler, gegen die der Generator gebaut wurde.
**Warum falsch:** Gelesen wird die Schlussfolgerung, nicht die Tabelle. Genau so entstand E-085: nicht die Zahlen waren das Problem, sondern dass zwei von drei Schlussfolgerungen sich umgekehrt hatten. Ein Generator, der nur Zahlen absichert, verlagert das Risiko vom auffaelligen Teil (Tabelle) in den unauffaelligen (Prosa) — und erzeugt zugleich das Vertrauen, das Problem sei geloest.
**Wie vermeiden:** (1) Jede datenABHAENGIGE Wertung im Generator verzweigen, mit ausformulierter Gegenaussage — nicht nur die Zahl einsetzen. (2) Testen, dass die Verzweigung greift: Eingabedaten umdrehen, pruefen dass der Text kippt. Ein Generator, der immer dasselbe schreibt, ist eine Attrappe. (3) Absolutwertungen ohne Rechengrundlage („klein", „deutlich", „vernachlaessigbar") entweder rechnen oder streichen. (4) Auch tote Zweige testen — der Zweig fuer den umgekehrten Befund feuert genau dann, wenn es darauf ankommt.
**Erkannt in:** Stage-2-Review (F-senior-6/13) zu `research/mandat2/render_befund_p12.py`.
**Referenzen:** E-085, E-087, E-073.

## E-090 — Wiederholungslauf zur Artefakt-Hygiene inkrementiert die Mehrfachtest-Buchhaltung
**Datum:** 2026-08-03
**Kategorie:** metrik-verwaesserung / statistik-buchhaltung
**Was passierte:** Ein Ergebnis-JSON war eine Skript-Revision alt (fehlendes Feld, alle Zahlen bit-identisch). Der Neulauf zur Bereinigung erhoehte den Trial-Zaehler um 44 — 44 gezaehlte „Hypothesen" ohne eine einzige neue Hypothese. Ein weiterer faelliger Regenerationslauf haette 110 addiert.
**Warum falsch:** Der Zaehler steuert den Deflated-Sharpe-Haircut und bedeutet „Zahl gepruefter Hypothesen". Zaehlt er Wiederholungen mit, verliert er genau diese Bedeutung. Die Richtung ist konservativ (der Haircut wird zu streng), das Problem ist die Interpretierbarkeit — und bei routinemaessigen Regenerationen waechst der Fehler mit der Sorgfalt, mit der man Artefakte pflegt. Ein Zaehler, dessen Wert von Hygiene-Massnahmen abhaengt, misst nicht mehr, was er messen soll.
**Wie vermeiden:** (1) Regenerationslaeufe explizit kennzeichnen (`--regen`) und vom Increment ausnehmen. (2) Den Zaehler NIEMALS still zurueckschreiben — er ist append-only Buchhaltung; stattdessen die faelschlich gezaehlte Differenz offenlegen. (3) Beim Bau eines solchen Zaehlers von Anfang an trennen: Hypothesen zaehlen, nicht Prozessaufrufe.
**Erkannt in:** Stage-2-Review (F-senior-8) zu `research/mandat2/trials.json`.
**Referenzen:** E-077, E-078.

## E-091 — CI zwei Tage rot, unbemerkt, weil die Arbeit „nur in research/" stattfand
**Datum:** 2026-08-03
**Kategorie:** false-green / prozess-luecke / lokal-vs-ci
**Was passierte:** Ab Commit `b9656969` (2026-08-01 13:36) waren die Workflows `CI` und `Backend CI` rot — fuenf Commits und zwei Tage lang. Aufgefallen ist es erst, als ich nach einer Nutzerfrage („hast du schon alles gefunden?") den CI-Status zum ersten Mal aktiv abgefragt habe. Ursache: die Mandat-II-Tests rufen `load_campaign()` auf, das `research/mandat/data/prices_verdict.parquet` liest — ein Verzeichnis, das per `.gitignore` ausgeschlossen ist (~4,1 GB, jederzeit nachziehbar). Lokal liegen die Daten, in CI nicht. Jeder dieser Tests scheiterte dort mit einem Dateifehler.
**Warum falsch:** Zwei Fehler zugleich. (1) **Der Test-Entwurf:** ein Test, der gitignorierte Daten braucht, muss einen expliziten Skip haben — sonst meldet er in CI einen Fehler, wo eine Umgebungsvoraussetzung fehlt. Rule 40 verlangt genau diese Unterscheidung; das Repo hat sie fuer optionale PAKETE (`pytest.importorskip`), hatte sie aber nicht fuer optionale DATEN. (2) **Der Prozess:** ich habe ueber mehrere Steps „lokal gruen" berichtet und den CI-Status als unveraendert angenommen, weil die Aenderungen „nur in `research/`" lagen. `research/` ist nicht im Lint- und Typing-Scope der CI — aber `pytest` sammelt `tests/**` vollstaendig ein, und dort lagen die neuen Tests. Die Annahme „mein Bereich ist nicht CI-relevant" war schlicht falsch.
**Wie vermeiden:** (1) Jeder Test, der auf gitignorierte Daten zugreift, bekommt beim SCHREIBEN einen Skip-Guard mit Klartext-Begruendung — nicht spaeter. Verifizieren, indem der Datenpfad einmal versaeuert und der Skip beobachtet wird. (2) Vor der Aussage „CI-Status unveraendert" den Status ABFRAGEN. Ein Bereich gilt nicht deshalb als CI-frei, weil er ausserhalb des Lint-Scopes liegt — die Testsammlung ist repo-weit. (3) Neue Testdateien sind IMMER CI-relevant, egal wo der Produktivcode liegt. (4) Nach jedem Push den Lauf abwarten, statt ihn zu unterstellen; „lokal gruen" und „CI gruen" sind zwei Aussagen (Rule 40).
**Erkannt in:** `tests/test_mandat2_data_gate.py`, `tests/test_mandat2_dividenden.py`. Fix: `tests/mandat2_daten_guard.py` mit `braucht_kampagnendaten`; 6 Tests markiert, 112 laufen in CI weiter. Skip-Verhalten gegen einen versaeuerten Pfad verifiziert.
**Referenzen:** Rule 40 (Optionale Dependencies / lokal vs. CI), E-066 (nicht geprueftes Gate zaehlt als bestanden), E-086.
**Korrigiert durch E-092** (Praemisse und Zahlen dieses Eintrags sind dort richtiggestellt).

## E-092 — Skip-Marke gesetzt, ohne den Datenbedarf je Test zu messen — Safety-Test in CI stillgelegt
**Datum:** 2026-08-03
**Kategorie:** false-green / weichspuelen / ungemessene-annahme
**Was passierte:** Nachdem CI wegen fehlender gitignorierter Daten rot war (E-091), habe ich sechs Tests mit einer `braucht_kampagnendaten`-Marke versehen. Die Auswahl entstand per `grep` nach `load_campaign|load_search|prices_verdict` — also nach ERWAEHNUNG, nicht nach tatsaechlichem Bedarf. Die Stage-1-Review hat es gemessen: **fuenf der sechs Tests laufen ohne die Daten einwandfrei.** `load_search(df)` nimmt den DataFrame als Argument und liest nie von Platte; die betroffenen Tests arbeiten auf synthetischen Fixtures. Nur der SPY-Bandtest brauchte die Daten wirklich. Schwerer: die drei faelschlich markierten `data_gate`-Tests waren die **einzige CI-Abdeckung der Holdout-Suchsperre** — des P0-Sicherungsmechanismus der Kampagne, der laut eigenem Modul-Docstring gerade deshalb als CODE existiert und nicht als Vorsatz. Ich hatte in derselben Commit-Message ausdruecklich geschrieben: „Das ist KEIN Weichspuelen."
**Warum falsch:** Genau das Muster, das CLAUDE.md verbietet — „keine Lockerung von Safety-Checks, um Tests einfach gruen zu bekommen" — und zwar begangen, waehrend ich das Gegenteil behauptete. Die Behauptung selbst ist der teure Teil: sie beruhigt den naechsten Leser und macht den Fehler unauffindbar. Ursache war ein Verfahrensfehler: die Marke wurde nach TEXTSUCHE vergeben statt nach Messung. Ein Test, der eine Datenquelle erwaehnt, braucht sie nicht zwangslaeufig; ein Test, der sie nicht erwaehnt, kann sie ueber einen Aufrufpfad trotzdem brauchen. Beides ist nur empirisch zu klaeren.
**Wie vermeiden:** (1) **Vor** dem Setzen einer Umgebungs-Skip-Marke den Bedarf MESSEN: Datenpfad wegschieben, Test einzeln laufen lassen, Ergebnis notieren. Wer nicht gemessen hat, setzt die Marke nicht. (2) Nach dem Markieren gegenpruefen: mit entfernten Marken und fehlenden Daten muessen genau die markierten Tests fallen — faellt weniger, ist zu viel markiert. (3) Bei jedem Skip fragen, WAS dadurch in CI ungeprueft bleibt; Safety- und PIT-Gates duerfen nie stillschweigend wegskippen. (4) Formulierungen wie „das ist kein Weichspuelen" sind belegpflichtig — ohne Messung nicht schreiben.
**Erkannt in:** Stage-1-Review (F-test-1 BLOCKER, F-test-2 MAJOR) zu Commit `7ccedca3`. Fix: fuenf der sechs Marken ersatzlos entfernt; Guard zusaetzlich auf alle drei von `load_campaign()` gelesenen Dateien ausgeweitet.
**Zwei Richtigstellungen an E-091** (dort nicht editiert, Register ist append-only):
1. E-091 behauptet, das Repo habe das `exists()`+`skip`-Muster nur fuer optionale PAKETE gehabt, nicht fuer DATEN. **Falsch.** Es war laengst etabliert — Dutzende Testdateien nutzen es gegen gitignorierte Pfade (u. a. `test_cli_ml_dataset.py`, `test_io_smoke.py`, `test_data_prices_ingest.py`, `test_forensic_survivorship.py`). Die zutreffende und unangenehmere Lehre lautet nicht „das Muster fehlte“, sondern „das etablierte Muster wurde hier nicht angewandt“.
2. E-091 nennt als Ergebnis „6 Tests markiert, 112 laufen in CI weiter“. Nach der Korrektur ist **1 Test markiert und 117 laufen**.

**Verlauf, unbeschoenigt:** der Defekt war bereits als `7ccedca3` auf `origin/main` gepusht, als Stage 1 ihn fand — die Review lief post-commit, entgegen der Projektregel „Review-Chain proaktiv vor erstem Commit“. Zwischen Push und Fund stand ein Safety-Test der Kampagne in CI still.

**Messung nach der Korrektur** (Scope `tests/test_mandat2_*.py`, 124 gesammelt, lokal Windows, **nicht CI-bestaetigt**):
- alle Kampagnendaten vorhanden → **124 passed**
- alle drei Kampagnendateien entfernt → **123 passed, 1 skipped**
- nur `dividends.parquet` entfernt (Teildatenlage, Scope data_gate+dividenden = 22) → 21 passed, 1 skipped; vor der Guard-Erweiterung ergab dieser Fall einen `FileNotFoundError`
- vor der Korrektur waren es 6 Marken → 112 laufend; jetzt 1 Marke → 123 laufend

*Nachtrag:* die erste Fassung dieses Absatzes nannte 118/117 — Zahlen aus einem Lauf VOR den sechs neuen Guard-Tests, ausgewiesen als „nach der Korrektur“. Genau die Klasse Fehler, die dieser Eintrag anprangert; von Stage 3 gefunden (F-auditor-1) und auf dem aktuellen Baum neu gemessen.

**Referenzen:** E-091 (der ausloesende CI-Bruch), E-066 (nicht geprueftes Gate zaehlt als bestanden), E-067 (xfail auf selbst erzeugte Regression), Rule 40.

## E-093 — Guard kopiert die Abhaengigkeitsliste seines Ziels statt sie abzuleiten
**Datum:** 2026-08-03
**Kategorie:** wiring-gap / stille-drift
**Was passierte:** Der Skip-Guard `tests/mandat2_daten_guard.py` hielt eine handgepflegte Kopie der Dateien, die `campaign_data.load_campaign()` liest — plus eine eigene Kopie des Datenwurzelpfads. Die erste Fassung listete nur eine von drei. Fehlte `dividends.parquet` bei vorhandenen Preisen, lief der Test in genau den `FileNotFoundError`, den der Guard verhindern sollte.
**Warum falsch:** Ein Guard, der die Abhaengigkeiten seines Ziels dupliziert statt sie abzuleiten, veraltet lautlos. Die naechste zusaetzliche Datenquelle im Loader macht ihn wieder zu kurz, und der Fehlermodus kehrt in exakt der Form zurueck, gegen die er gebaut wurde. Sichtbar wird die Luecke nur bei TEILWEISE fehlenden Voraussetzungen — also praktisch nie lokal, wo entweder alles da ist oder nichts.
**Wie vermeiden:** (1) Den Pfad aus dem Produktionsmodul IMPORTIEREN (`from ... import DATA`) statt ihn neu zusammenzusetzen. (2) Die Dateiliste per Regressionstest an die tatsaechlichen Lesezugriffe koppeln — in beide Richtungen: keine gelesene Datei ohne Guard, kein Guard-Eintrag ohne Lesezugriff. (3) Guards, die eine Liste fuehren, immer gegen TEILWEISE fehlende Voraussetzungen testen, nicht nur gegen „alles weg". (4) Den Regressionstest gegen eine Mutation verifizieren, sonst ist er selbst nur eine Behauptung.
**Erkannt in:** Stage-2-Review (F-senior-3) zu `tests/mandat2_daten_guard.py`. Fix: `DATA` importiert, `tests/test_mandat2_daten_guard.py` koppelt die Liste beidseitig an `campaign_data`; gegen Mutation verifiziert (Eintrag entfernt -> Test faellt).
**Referenzen:** E-091, E-092, E-070 (Gate deckte nur den Hauptdatensatz).

## E-094 — Korrektur eines Anti-Pattern-Eintrags per In-Place-Edit in einem append-only Register
**Datum:** 2026-08-03
**Kategorie:** prozess-verstoss / register-integritaet
**Was passierte:** Ein Review-Finding widerlegte eine Praemisse in E-091. Ich habe die Korrektur direkt in den „Warum falsch"-Absatz von E-091 hineingeschrieben — obwohl der Header derselben Datei ausdruecklich sagt: „Niemals existierende Eintraege editieren oder loeschen — nur neue anhaengen." Verschaerfend: die Korrektur war unvollstaendig. Der Abschnitt „Erkannt in" desselben Eintrags blieb auf dem widerlegten Stand („6 Tests markiert, 112 laufen"), sodass ein Eintrag zwei einander widersprechende Aussagen trug.
**Warum falsch:** Ein append-only Register lebt davon, dass ein Eintrag den Wissensstand SEINES Datums festhaelt. Wird er nachtraeglich teilweise ueberschrieben, entsteht ein Mischtext, aus dem weder der urspruengliche Irrtum noch der Korrekturzeitpunkt rekonstruierbar ist — und genau der Irrtum ist der Lernwert des Registers. Halbe Korrekturen sind dabei schlechter als keine: sie erzeugen einen Eintrag, der sich selbst widerspricht. Zusaetzlich laedt der SessionStart-Hook die zehn neuesten Eintraege; ein stehengebliebener falscher Zahlwert wird damit aktiv weitertransportiert.
**Wie vermeiden:** (1) Korrekturen als NEUEN Eintrag mit Rueckverweis fuehren; im alten Eintrag hoechstens EINE angehaengte Zeile „korrigiert durch E-0xx". (2) Die Pflegeregel steht im Header derselben Datei — vor dem Edit lesen. (3) Wird ausnahmsweise doch editiert, den GESAMTEN Eintrag auf Konsistenz pruefen, nicht nur die Stelle, auf die das Finding zeigte.
**Erkannt in:** Stage-2-Review (F-senior-1 MAJOR, F-senior-2) zu `docs/CLAUDE_CODING_ERRORS.md`. Fix: E-091 auf den Originalstand zurueckgesetzt, eine Verweiszeile angehaengt, beide Richtigstellungen (Praemisse + Zahlen) nach E-092 verlagert.
**Referenzen:** E-085 (Doku driftet gegen den Lauf), E-086.

## E-095 — Wirkung dem falschen Mechanismus zugeschrieben, weil zwei Fixes gleichzeitig kamen
**Datum:** 2026-08-03
**Kategorie:** false-mechanism / ungemessene-kausalitaet
**Was passierte:** Eine Kennzahl eskalierte auf 10^81 (Rendite-Produkt ueber ein Aktienuniversum). Ich vermutete Renditen ueber NaN-Luecken hinweg, baute eine Lueckenmaske ein — die Eskalation blieb. Dann baute ich einen Glitch-Filter gegen Vendor-Preisfehler (MEL 7,73 -> 141.630) ein, und das Ergebnis wurde plausibel. Im Code stand anschliessend ein Kommentar, die LUECKENMASKE habe die Eskalation verhindert. Ein Mutationstest zeigte das Gegenteil: Maske abschwaechen -> alle Tests bleiben gruen. Nachgemessen liefert `pct_change(fill_method=None)` ueber eine NaN-Luecke ohnehin NaN — die Maske ist vollstaendig redundant und hat nie etwas bewirkt.
**Warum falsch:** Zwei Aenderungen, ein beobachteter Effekt, und die Zuschreibung folgte der Reihenfolge des Einbaus statt einer Messung. Das ist doppelt teuer: (1) der Kommentar erklaert kuenftigen Lesern einen Mechanismus, den es nicht gibt, und (2) der WIRKLICH tragende Mechanismus (Glitch-Filter) erscheint dadurch weniger wichtig, als er ist — wer spaeter aufraeumt, entfernt womoeglich den falschen. Verschaerfend: der zugehoerige Test prueft die Eigenschaft („keine Scheinrendite nach einer Luecke"), nicht die Codezeile — er kann die Fehlzuschreibung also nicht aufdecken und suggeriert trotzdem Deckung.
**Wie vermeiden:** (1) Nie zwei Fixes gleichzeitig einbauen, wenn danach eine Ursachenaussage im Code stehen soll — einzeln einbauen und nach jedem messen. (2) Ist es doch passiert: den vermuteten Mechanismus per Mutation abschwaechen und pruefen, ob der Effekt zurueckkehrt. Kehrt er nicht zurueck, war es der andere. (3) Kommentare der Form „X verhindert Y" sind belegpflichtig wie jede andere Behauptung. (4) Redundante Absicherungen duerfen bleiben — aber als solche gekennzeichnet („traegt kein Ergebnis"), nicht als Ursache.
**Erkannt in:** `research/mandat2/p12d_survivorship_schranke.py` — beim Mutationstest der eigenen neuen Tests, also durch die Verifikationsregel aus E-092 (3).
**Referenzen:** E-089 (Behauptung neben gerechneter Zahl), E-072 (Stellparameter ohne Wirkung), E-087.

## E-096 — Taeglich rebalanciertes Portfolio als „Buy-and-Hold" ausgegeben; der Rebalancing-Bonus trug die Entwarnung
**Datum:** 2026-08-03
**Kategorie:** false-mechanism / false-comparison / entwarnung-auf-artefakt
**Was passierte:** Um die Survivorship-Verzerrung eines Universums zu beziffern, sollten zwei Behandlungen des Delisting-Erloeses verglichen werden: Erloes liegen lassen vs. auf die Ueberlebenden umschichten. Die zweite habe ich als `(1 + r.mean(axis=1)).cumprod()` implementiert — das ist aber kein Buy-and-Hold mit Umschichtung, sondern ein **taeglich gleichgewichtet rebalanciertes Portfolio**. Es weicht auch dann ab, wenn ueberhaupt nichts delistet (synthetisch: -9,1 % bei zwei gegenlaeufigen Titeln ohne ein einziges Delisting), und sein Rebalancing-Bonus waechst mit der Zahl der Namen: +2,5 % bei n=20, +21,5 % bei n=261, +36,6 % bei n=418. Da die verglichenen Universen genau in n stark differieren, ging der Bonus vollstaendig in die Differenz ein. Ergebnis: eine gemessene Ueberhoehung von **+0,14 % p. a.**, aus der ich schloss, das Verdikt der Vorphase kippe nicht — und das dem Nutzer so berichtete. Korrekt gerechnet sind es **+2,36 %** (umgeschichtet) bzw. **+2,90 %** (gehalten), beide **oberhalb** des Entscheidungsabstands von 1,5 % p. a. Die richtige Aussage lautet damit nicht „das Verdikt haelt", sondern „dieser Datensatz kann die Frage nicht entscheiden".
**Warum falsch:** Zwei Portfoliokonstruktionen, die sich in einer Dimension unterscheiden SOLLEN, unterschieden sich in zweien — und die zweite Dimension (Rebalancing-Frequenz) hat einen Effekt, der mit der Universumsgroesse skaliert, also genau entlang der Achse, die verglichen wurde. Das ist E-079 in neuer Gestalt: nicht die Gewichtungsmethode, sondern die Rebalancing-Frequenz war die stille Zusatzvariable. Besonders teuer, weil das Artefakt in die BERUHIGENDE Richtung zeigte: eine zu kleine Verzerrung laesst ein Verdikt sicherer aussehen, als es ist.
**Wie vermeiden:** (1) Vergleicht man zwei Behandlungen EINER Annahme, muss ein Nullfall existieren, in dem beide **identisch** sein muessen — hier: kein Delisting im Universum. Diesen Nullfall als Test schreiben, bevor die Zahlen interpretiert werden. Er faengt den Fehler sofort und ist billig. (2) `r.mean(axis=1).cumprod()` ist NIE Buy-and-Hold. Wer Positionen halten will, simuliert Positionswerte, nicht Renditemittel. (3) Wenn ein Effekt mit einer Groesse skaliert (hier n), die zwischen den verglichenen Faellen variiert, ist er per Konstruktion in der Differenz — vor dem Vergleich pruefen, ob die Kennzahl groessenneutral ist. (4) Entwarnungen sind belegpflichtiger als Warnungen: eine Zahl, die ein Problem kleinredet, verdient die Gegenprobe zuerst.
**Erkannt in:** Stage-1-Review (F-test-1, BLOCKER) zu `research/mandat2/p12d_survivorship_schranke.py`. Fix: wertbasierte Simulation mit Pro-rata-Verteilung nur am Delisting-Tag; Regressionsanker `test_ohne_delisting_sind_beide_varianten_identisch` gegen die alte Implementierung mutationsgeprueft.
**Referenzen:** E-079 (Benchmark-Methode), E-089 (Wertung ohne Rechengrundlage), E-095 (Wirkung dem falschen Mechanismus zugeschrieben).

## E-097 — Finding an zwei Fundstellen, nur die genannte korrigiert — und als erledigt gemeldet
**Datum:** 2026-08-03
**Kategorie:** wiring-gap / false-fixed
**Was passierte:** Ein Review-Finding bemaengelte die Behauptung „P1-P11 sind survivorship-frei" als zu stark. Ich habe sie im Generator und im erzeugten Befund sauber umformuliert und das Finding als erledigt gemeldet. Dieselbe Behauptung stand aber noch zweimal im Quellmodul: fett im Modul-Docstring und in einem `print`-Block, der sie bei JEDEM Lauf auf die Bedienkonsole schreibt. Die schwaechere, korrekte Fassung landete im Erzeugnis, die staerkere blieb in der Quelle — und die wirkt autoritativer.
**Warum falsch:** Der Erledigt-Vermerk deckte die FUNDSTELLE ab, auf die das Finding zeigte, nicht die BEHAUPTUNG. Ein generiertes Dokument schuetzt gegen Zahlendrift (E-085), aber nicht gegen Prosa, die ausserhalb des Generators liegt. Wer spaeter das Modul liest oder den Lauf startet, bekommt die zurueckgenommene Aussage.
**Wie vermeiden:** (1) Betrifft ein Finding eine BEHAUPTUNG, vor dem Erledigt-Vermerk nach dem WORTLAUT greppen, nicht nach der Zeilennummer. (2) Konsolenausgaben zaehlen als Fundstelle — sie sind das, was der Bediener tatsaechlich liest. (3) Wo Generator-Prosa und Modul-Docstring dieselbe Aussage tragen, gehoert eine davon weg oder auf die andere verwiesen.
**Erkannt in:** Stage-2-Review (F-senior-1, BLOCKER) zu `research/mandat2/p12d_survivorship_schranke.py`.
**Referenzen:** E-085, E-086 (Erzeuger gefixt, Artefakt nicht), E-092.

## E-098 — Docstring der ersetzten Implementierung ueberlebt den Rewrite und behauptet das Gegenteil
**Datum:** 2026-08-03
**Kategorie:** logic-error / false-mechanism
**Was passierte:** Nach dem Umbau einer Funktion von renditebasierter auf wertbasierte Simulation blieb der Docstring des zugehoerigen Tests stehen. Er schrieb den Schutz gegen Scheinrenditen `pct_change(fill_method=None)` zu und schloss daraus, der Test fange eine Abschwaechung der expliziten NaN-Maske NICHT. In der neuen Implementierung existiert kein `pct_change` mehr; die Maske IST der Mechanismus, und der Mutationstest zeigt, dass genau dieser Test ihre Entfernung faengt.
**Warum falsch:** Eine Mechanismus-Aussage, die beim Schreiben stimmte, wird durch einen Rewrite falsch — unbemerkt, weil Docstrings nicht ausgefuehrt werden. Die Richtung ist besonders teuer: der Text redet die eigene Testabdeckung KLEINER, als sie ist. Wer spaeter aufraeumt, haelt den Test fuer wirkungslos und schwaecht oder entfernt ihn. Das ist E-095 mit umgekehrtem Vorzeichen — dort wurde ein Mechanismus zu Unrecht gelobt, hier zu Unrecht abgeschrieben. Verschaerfend: der falsche Docstring stand im selben Commit, der E-095 protokolliert.
**Wie vermeiden:** (1) Wird eine Funktion ersetzt, gilt ihr gesamter erklaerender Text als ungeprueft, bis er neu belegt ist — auch der in den Tests. (2) Saetze der Form „dieser Test faengt X nicht" sind belegpflichtig wie jede andere Behauptung: per Mutation nachweisen, sonst nicht schreiben. (3) Nach einem Rewrite nach Symbolnamen greppen, die im Text stehen, aber im Code nicht mehr vorkommen.
**Erkannt in:** Stage-2-Review (F-senior-2) zu `tests/test_mandat2_survivorship.py`.
**Referenzen:** E-095, E-089.

## E-099 — Fixture mit nur einem Empfaenger kann keine Verteilungsregel pruefen
**Datum:** 2026-08-03
**Kategorie:** test-anti-pattern
**Was passierte:** Ein Test namens `test_erloes_wird_pro_rata_verteilt` prueft „gegen die Handrechnung", dass ein Delisting-Erloes pro rata auf die Ueberlebenden verteilt wird. Im Fixture ueberlebte genau EIN Titel — der bekommt unter Pro-rata, Gleichverteilung und jeder anderen Regel denselben Betrag. Mutation zu Gleichverteilung: alle Tests blieben gruen. Die Handrechnung war korrekt, aber fuer beide Regeln dieselbe Zahl.
**Warum falsch:** Ein Test unterscheidet nur, was sein Fixture unterscheidet. Der Testname wirkt zugleich wie eine Zusicherung und wird beim spaeteren Lesen als Beleg fuer die Regel genommen. Verwandt mit dem Schwellen-Test desselben Moduls, der seine Fixtures aus der zu pruefenden Konstanten baute und damit gegen deren Aenderung immun war: beide Male war die Groesse, um die es geht, im Fixture gar nicht variiert.
**Wie vermeiden:** (1) Vor dem Schreiben fragen: welche ALTERNATIVE Regel liefert dieselbe Zahl? Liefert eine, ist das Fixture zu klein. (2) Verteilungsregeln brauchen mindestens zwei Empfaenger mit UNGLEICHEN Anteilen — und eine Bewegung danach, weil sich die Regeln erst im Folgeverlauf unterscheiden. (3) Jede Regel, deren Name im Testtitel steht, per Mutation gegen ihre naechstliegende Alternative pruefen.
**Erkannt in:** Stage-2-Review (F-senior-3) zu `tests/test_mandat2_survivorship.py`. Fix verifiziert: Mutation zu Gleichverteilung -> Test faellt (2,833 statt 2,917).
**Referenzen:** E-092, E-072.

## E-100 — Ungleichungs-Schluss ueber zwei Messbasen
**Datum:** 2026-08-03
**Kategorie:** false-comparison / unbelegte-belastbarkeit
**Was passierte:** Die Kernaussage eines Befunds lautete: Verzerrung (2,4-2,9 % p. a.) > Entscheidungsmarge (1,5 % p. a.), also kann der Datensatz die Frage nicht entscheiden. Die Marge stammte aus dem Intraday-Artefakt (Stundenbars), die Verzerrung aus dem Tagespanel. Dasselbe Buy-and-Hold, dieselben 20 Namen, dasselbe Fenster liefert auf beiden Panels 11,48 % gegen 11,06 % — 0,42 Prozentpunkte, also 28 % der Marge. Tabelle und Prosa nannten beide Werte unkommentiert nebeneinander.
**Warum falsch:** Eine Ungleichung zwischen zwei Groessen aus verschiedenen Quellen traegt nur, wenn die Quellendifferenz klein gegen den Abstand ist. Hier ist sie es nicht offensichtlich, und der Leser bekommt keinen Hinweis — er sieht zwei verschiedene Zahlen fuer dieselbe Sache ohne Erklaerung. Der Schluss selbst haelt (beide Differenzen sind je intern konsistent gerechnet), aber seine Belastbarkeit war nicht ausgewiesen.
**Wie vermeiden:** (1) Bevor zwei Zahlen in eine Ungleichung gehen, pruefen, ob sie aus derselben Messbasis stammen. (2) Tun sie es nicht, die Basisdifferenz an einer GEMEINSAMEN Groesse messen (hier: identisches Buy-and-Hold auf beiden Panels) und neben den Schluss schreiben. (3) Zwei sichtbar verschiedene Werte fuer dieselbe Groesse im selben Abschnitt sind immer erklaerungspflichtig, auch wenn beide richtig sind.
**Erkannt in:** Stage-2-Review (F-senior-4) zu `research/mandat2/render_befund_p12.py`.
**Referenzen:** E-079, E-088 (Extremwerte aus verschiedenen Zeilen), E-078.

## E-101 — Ungleichung behauptet statt gerechnet, im entwarnenden Sinn — und E-097 verschrieb ein Mittel, das den Rueckfall nicht verhindert
**Datum:** 2026-08-03
**Kategorie:** false-evidence / entwarnung-ohne-rechnung / register-korrektur
**Was passierte:** Zwei Rueckfaelle in derselben Runde, in der die zugehoerigen Regeln formuliert wurden.

(1) Die Ueberhoehung eines Benchmarks wurde in zwei Kanaele zerlegt (Dauermitgliedschaft 1,69 pp, Intraday-Verfuegbarkeit 1,21 pp) und dazu geschrieben: „aber der erste Kanal allein ueberschreitet die Marge nicht". Die Marge betraegt 1,47 pp — der erste Kanal ueberschreitet sie also sehr wohl, um 0,23 pp. Beide Zahlen standen gerundet im selben Abschnitt („1,7 %" und „1,5 %"), der Widerspruch war fuer jeden Leser sichtbar. Der Satz war handgeschrieben neben zwei gerechneten Werten und zeigte in die ENTWARNENDE Richtung (E-089, E-096).

(2) E-097 hatte als Gegenmittel formuliert: „nach dem WORTLAUT greppen, nicht nach der Zeilennummer". Genau das habe ich getan — und eine dritte Fundstelle uebersehen, weil sie dieselbe Behauptung in ANDEREN Worten trug („betroffen ist ausschliesslich der Intraday-Strang" statt „P1-P11 sind survivorship-frei"). Sie stand in einem `print`-Block drei Zeilen unter der korrigierten Stelle und widersprach dem Docstring derselben Datei.
**Warum falsch:** (1) Jede Ungleichung zwischen zwei berechneten Groessen gehoert gerechnet, nicht behauptet — besonders wenn die Behauptung beruhigt. Ein handgeschriebenes „X ueberschreitet Y nicht" neben `X` und `Y` ist die teuerste Form von Prosa: sie sieht belegt aus. (2) Eine Behauptung ist keine Zeichenkette. Wer nach dem Wortlaut sucht, findet Kopien, keine Umformulierungen — und Umformulierungen sind der Normalfall, weil derselbe Gedanke an verschiedenen Stellen anders formuliert wird.
**Wie vermeiden:** (1) Vergleiche zwischen berechneten Groessen im Generator VERZWEIGEN (`"über" if k > marge else "unter"`), nie ausformulieren. Dann kann der Satz nicht gegen seine eigenen Zahlen laufen. (2) **Korrektur an E-097:** nach der BEHAUPTUNG suchen, nicht nach ihrem Wortlaut — also nach den Traegerbegriffen (hier `P1-P11`, `Intraday-Strang`, `survivorship`) und dann jede Fundstelle lesen. Ein Wortlaut-Grep ist notwendig, aber nicht hinreichend. (3) Konsolenausgaben und Docstrings derselben Datei gegeneinander pruefen: widersprechen sie sich, ist eine der beiden ein Rueckfall.
**Erkannt in:** Stage-3-Review (F-auditor-1 und F-auditor-2, beide BLOCKER, Verdict FAIL) zu `research/mandat2/`.
**Referenzen:** E-097 (dessen Lehre hiermit korrigiert wird), E-089, E-096, E-100.

## E-102 — Proxy fuer die AUSWAHL als Proxy fuer das HALTEN benutzt; die Entwarnung war das Ergebnis
**Datum:** 2026-08-04
**Kategorie:** false-mechanism / entwarnung-auf-proxy / nicht-instrumentiert
**Was passierte:** Zu klaeren war, ob korrumpierte Kursserien in die Ergebnisse der Kampagne eingegangen sind. Ich unterschied dafuer zwei Kanaele — HALTEN ueber den Fehlertag und AUSWAHL im kontaminierten Momentum-Fenster — und mass beide aus derselben Quelle: der Top-20-Auswahlliste. Fuer den Halte-Kanal galt „letzter Auswahltermin <= 31 Tage vor dem Fehlertag". Ergebnis: „4 Namen beruehrt, **keiner ueber den Halte-Kanal**, 0,38 % der Auswahlplaetze". Das war eine Entwarnung, und sie war falsch.

Die Engine verkauft nicht beim Verlassen der Top-20, sondern erst bei `rang > rank_out` (Default 60) bzw. nach `min_haltetage` — die HALTEmenge ist echt groesser als die AUSWAHLmenge. An der instrumentierten Engine gemessen (Bestand je Handelstag mitgeschrieben): **GPS lag am 1996-12-20 im Portfolio**, dem Tag seines Vendor-Fehlers, mit einer Portfolio-Tagesrendite von **+12,4 %** — dem zweitgroessten Einzeltag der gesamten 21 Jahre, rein aus einem Datenfehler. Der Proxy hatte diesen Fall als „81 Tage vor dem Glitch, Halteperiode vorbei" eingestuft, und ein Test zementierte den Fehlschluss sogar. Korrekt: 2 Namen an 11 korrupten Handelstagen ueber den Halte-Kanal, 5 Namen / 27 Plaetze ueber den Auswahl-Kanal.
**Warum falsch:** Auswahl und Bestand sind verschiedene Mengen, sobald eine Turnover-Bremse existiert — und sie existiert in fast jedem realistischen Backtest. Einen Proxy zu benutzen ist zulaessig; ihn zu benutzen, OHNE die Groesse einmal direkt zu messen, ist es nicht. Verschaerfend: der Proxy zeigte in die beruhigende Richtung, und genau dort ist die Beweislast am hoechsten (E-096). Zweiter Fehler derselben Wurzel: der Fehlertag wurde als `idxmax` gespeichert, also als EIN Tag — tatsaechlich ist die Korruption ein Fenster (CFC: 21 Tage). Wer einen Skalar speichert, wo ein Intervall vorliegt, verfehlt die Ueberlappung mit der Haltedauer systematisch.
**Wie vermeiden:** (1) Wenn die Frage „wurde X gehalten" lautet, den BESTAND messen, nicht die Auswahl. Die Engine laesst sich ohne Eingriff instrumentieren (`Portfolio.set_date` umschliessen und `lots` je Tag mitschreiben) — das kostet einen Lauf und ersetzt jede Schaetzung. (2) Bei jedem Proxy einmal fragen: welcher Mechanismus koennte Proxy und Zielgroesse auseinandertreiben? Turnover-Bremsen, Mindesthaltedauern und Risk-off-Gates tun genau das. (3) Ereignisse, die sich ueber Tage ziehen, als Intervall speichern, nicht als Extremwert. (4) Eine Kennzahl, die ein Problem kleinredet, braucht die direkte Messung, bevor sie berichtet wird.
**Erkannt in:** Stage-1-Review (F-test-1 BLOCKER, F-test-2 BLOCKER) zu `research/mandat2/p12e_panel_hygiene.py`. Der Reviewer hat die echte Engine instrumentiert, statt meinem Proxy zu glauben.
**Referenzen:** E-096 (Entwarnung auf Artefakt), E-072 (Stellparameter ohne Wirkung), E-095, E-101.

## E-103 — Die Korrektur behielt die Fail-Open-Richtung des Fehlers
**Datum:** 2026-08-04
**Kategorie:** silent-except / false-green
**Was passierte:** Nach der Korrektur von E-102 kam der Halte-Kanal aus dem echten Bestand. Der Lookup blieb aber still: `bestand.get(t, ())` und `rendite.get(t, nan)` liefern bei jedem Key-Format- oder tz-Drift die leere Menge bzw. 0,0 %. Das Ergebnis waere erneut „kein korrumpierter Name wurde gehalten" gewesen — dieselbe Entwarnung, nur diesmal aus einem Verdrahtungsfehler statt aus einem Denkfehler. Ein Test hielt das Schweigen sogar als Sollverhalten fest (`test_fehlende_rendite_kippt_nicht`).
**Warum falsch:** Eine Messung, deren Ausfallmodus die beruhigende Antwort ist, ist keine Messung. Der Fix adressierte den Mechanismus, nicht die RICHTUNG, in die der Code bei Stoerung faellt — und genau die Richtung war das urspruengliche Problem. Ein Test, der „kippt nicht" prueft, zementiert diese Richtung zusaetzlich.
**Wie vermeiden:** (1) Bei jeder Kennzahl, die ein Risiko beziffert, fragen: was liefert sie, wenn die Verdrahtung bricht? Faellt sie auf „unauffaellig", muss sie fail-loud werden. (2) Konkret: Existenz der Keys pruefen und bei fehlendem Wert zu einem als betroffen erkannten Fall abbrechen statt zu 0 zu degradieren. (3) Tests von „kippt nicht" auf „meldet" umschreiben — ein Test, der Schweigen absichert, ist ein Test gegen die eigene Messung.
**Erkannt in:** Stage-2-Review (F-senior-1) zu `research/mandat2/p12e_panel_hygiene.py`.
**Referenzen:** E-102, E-096, E-066.

## E-104 — Zwei-Punkt-Kennzahl als Fenster modelliert; beide Korrekturen lagen daneben
**Datum:** 2026-08-04
**Kategorie:** false-mechanism / modell-statt-formel
**Was passierte:** Zu bestimmen war, welche Auswahltermine einen kontaminierten Momentum-Score hatten. Ich modellierte das als Fenster: „jeder Termin im Intervall nach einem Fehlertag". `momentum_score` ist aber `close.shift(21) / close.shift(252)` — ein Quotient aus GENAU ZWEI Stuetzstellen. Es gibt kein Fenster; nur die beiden Beine zaehlen. Bei einem Ein-Tages-Spike markierte mein Modell ~230 Termine zu viel.

Die naheliegende Korrektur waere ebenfalls falsch gewesen: „ein Bein liegt auf einer falschen Skala" uebersieht, dass sich der Skalenfaktor herauskuerzt, wenn BEIDE Beine auf DERSELBEN falschen Skala liegen. Richtig ist: **kontaminiert, wenn die beiden Beine auf VERSCHIEDENEN Skalen liegen.** Das deckt den dauerhaften Niveaubruch (ein Bein davor, eins danach) und den Einzelspike (ein Bein trifft den Spike) in einer Bedingung ab.

Zusaetzlich hatte ich beim Umstellen der Ungleichung „g in (t-252, t-21]" nach t beide Grenzen gekippt: korrekt ist t in [g+21, g+252), geschrieben hatte ich (g+21, g+252]. Ausgeschlossen wurde damit ausgerechnet der maximal kontaminierte Termin.
**Warum falsch:** Ein Modell ist keine Formel. Wer „Rueckblickfenster" denkt, wo eine Zwei-Punkt-Kennzahl steht, baut eine plausible Struktur um die falsche Mechanik — und praezisiert sie dann sogar noch (Kalendertage -> Handelstage), ohne die Praemisse zu pruefen. Beim Aufloesen einer Ungleichung nach der anderen Variablen kehren sich offen/geschlossen zusaetzlich um; das sieht wie eine Umbenennung aus und ist eine Rechnung.
**Wie vermeiden:** (1) Vor jedem Kontaminationsmodell die FORMEL der Kennzahl hinschreiben und fragen, welche Eingaenge sie wirklich hat. Ein Quotient aus zwei Kursen hat zwei Stuetzstellen, kein Fenster. (2) Umgestellte Intervalle als Ungleichungskette ausschreiben, nicht als Prosa. (3) Beide RAENDER testen, nicht „knapp drin / knapp draussen" — die Grenze ist die einzige Stelle, an der ein Randfehler sichtbar wird. (4) Bei einer vorgeschlagenen Korrektur pruefen, ob sie die Formel trifft: hier war auch der Review-Vorschlag („ein Bein falsch") noch nicht richtig.
**Erkannt in:** Stage-2-Review (F-senior-2, F-senior-3) zu `research/mandat2/p12e_panel_hygiene.py`. Wirkung: Kanal B von 27 auf 22 Auswahlplaetze.
**Referenzen:** E-102, E-095, E-087.

## E-105 — Review-Befund uebernommen, ohne die Fachlogik zu pruefen (Beinahe-Fall)
**Datum:** 2026-08-04
**Kategorie:** ungeprueft-uebernommen
**Was passierte:** Ein Review-Finding (F-senior-10) lautete, der Halte-Kanal unterschaetze die Reichweite: GPS habe zwei Uebergangstage, aber ein ganzes JAHR auf falscher Skala gelegen, also seien „hunderte Tage" betroffen statt elf. Der Befund klang zwingend und kam von einem Reviewer, der zuvor zwei echte BLOCKER gefunden hatte.

Er ist trotzdem falsch — fuer den Halte-Kanal. Dieser misst verzerrte TAGESRENDITEN. Liegen Vortag und Tag auf derselben (falschen) Skala, kuerzt sich der Faktor im Quotienten heraus und die Rendite ist korrekt. Verzerrt ist ausschliesslich der Uebergangstag. Haette ich den Befund uebernommen, waere die Kennzahl von 11 auf ueber 32.000 Tage gesprungen — eine Alarmzahl ohne Substanz, in einem Dokument, das gerade um Ehrlichkeit ringt.
**Warum das ein Anti-Pattern ist:** Ein Reviewer mit hoher Trefferquote erzeugt Autoritaetsdruck. Genau dann ist die Versuchung am groessten, einen Befund zu uebernehmen, statt ihn zu pruefen — und ein uebernommener falscher Befund ist schlimmer als ein uebersehener richtiger, weil er mit fremder Autoritaet auftritt und in die scheinbar sichere Richtung (mehr Alarm) zeigt.
**Wie vermeiden:** (1) Jeden Review-Befund gegen die Fachlogik pruefen, auch und gerade wenn die vorherigen richtig waren. (2) Widerspruch begruenden und belegen, nicht behaupten — hier: Renditen sind Quotienten, ein konstanter Skalenfaktor kuerzt sich. (3) Trefferquote ist kein Argument fuer den EINZELNEN Befund. (4) Die Richtung des Befunds mitdenken: „mehr Alarm" fuehlt sich sicher an und ist es nicht, wenn die Zahl falsch ist.
**Erkannt in:** Eigene Pruefung von Stage-2-Finding F-senior-10 zu `research/mandat2/p12e_panel_hygiene.py`. Der Reviewer hatte fuer Kanal B recht (E-104), fuer Kanal A nicht.
**Referenzen:** E-104, E-102.

## E-106 — Offenes Intervall mit dem letzten Index als Ende: der letzte Tag faellt still heraus
**Datum:** 2026-08-04
**Kategorie:** logic-error / off-by-one
**Was passierte:** Korruptionsspannen wurden als `(start, ende)` gespeichert und ueberall mit der Maske `>= start & < ende` ausgewertet. Fuer eine offene Spanne — ein dauerhafter Skalenbruch, der bis zum Panelende laeuft — setzte ich `ende = index[-1]`. Damit faellt der LETZTE Handelstag aus der Maske: er wird nicht mitbereinigt und behaelt seinen kuenstlichen Sprung, waehrend alles davor gespleisst ist. Die Bereinigung erzeugt an dieser Stelle also selbst einen Sprung, wo vorher keiner war. Betroffen waren die drei Namen mit dauerhaftem Bruch (TWX, SWKS, RHT) und die Zaehlung `n_tage_falsch`, die um je einen Tag zu niedrig lag.
**Warum falsch:** Ein halboffenes Intervall `[a, b)` braucht als „bis zum Ende" ein `b` HINTER dem letzten Element, nicht das letzte Element selbst. Der Fehler ist unauffaellig, weil er nur ein einziges Datum betrifft — und tueckisch, weil er ausgerechnet den Rand trifft, an dem die Endwerte der Backtests gebildet werden. Im schlimmsten Fall haette die „Bereinigung" den Ergebnisvergleich verzerrt, den sie gerade ermoeglichen sollte.
**Wie vermeiden:** (1) Bei halboffenen Intervallen das offene Ende explizit als Sentinel hinter dem letzten Element setzen (`index[-1] + 1 Tag`), nicht als letztes Element. (2) Jede „bis zum Ende"-Spanne mit einem Test pruefen, der genau den letzten Eintrag inspiziert — Tests, die die Mitte pruefen, sehen den Fehler nie. (3) Bei Bereinigungsfunktionen zusaetzlich die Gegenprobe fahren: nach dem Eingriff darf die Auffaelligkeit, gegen die bereinigt wurde, nicht mehr messbar sein. Genau diese Gegenprobe hat den Fehler hier gefunden.
**Erkannt in:** Ein selbst geschriebener Test (`test_renditen_innerhalb_der_spanne_bleiben_exakt`) schlug fehl — nicht wegen eines Testfehlers, sondern weil er den Produktionsfehler traf. `research/mandat2/p12d_survivorship_schranke.py`.
**Referenzen:** E-104 (Intervallgrenzen beim Umstellen gekippt), E-070 (`clip` statt verwerfen am Rand).

## E-107 — Die Bereinigung erzeugte schlimmere Fehler als die, gegen die sie bereinigte
**Datum:** 2026-08-04
**Kategorie:** false-repair / gegenprobe-einseitig
**Was passierte:** Um zu pruefen, ob Vendor-Preisfehler ein Verdikt drehen, wurden die Skalenbrueche gespleisst: innerhalb einer Korruptionsspanne `[a, b)` werden die Kurse durch den Uebergangsfaktor `f` geteilt. Die Spanne wurde bei der ERSTEN Gegenbewegung unter -66,7 % geschlossen — ohne zu pruefen, ob deren Betrag zu `f` passt.

Bei YRCW oeffnete ein Sprung mit `f = 302,8`. Geschlossen wurde bei einem **echten** Kurssturz von -77,2 %. Die Division der davorliegenden Spanne durch 302,8 zog den Rand um Faktor 69 daneben und **erzeugte** eine Tagesrendite von **+6.802 %** — groesser als jeder Vendor-Fehler im Originalpanel. Dazu skalierte die erste Fassung nur `close`, nicht das abgeleitete `div_panel`: die implizite Dividendenrendite stieg in der Spanne um genau `f`, bei WIN von 26 % auf 274 %. Das Ergebnis lief durch und lieferte ein sauber aussehendes „das Verdikt dreht nicht".

**Warum falsch:** Die Gegenprobe zaehlte nur, was VERSCHWINDET (Sprung weg = Erfolg), nie, was ENTSTEHT. Eine Reparatur ohne Gegenrichtung ist keine Reparatur, sondern eine zweite, unbeobachtete Datenquelle — und zwar eine, die genau dort eingreift, wo die Fragestellung entschieden wird. Verschaerfend: das Ergebnis war das erwartete („dreht nicht"), also gab es keinen Anlass, misstrauisch zu werden. Ein Fehler, der die erwartete Antwort liefert, wird nicht gesucht.

Zweitens: Beim Spleissen sind zwei Groessen betroffen, nicht eine. `div_panel` ist ein aus `close` ABGELEITETES Feld in Panel-Einheiten. Wer nur die primaere Groesse korrigiert, verschiebt still das Verhaeltnis zwischen beiden — hier ausgerechnet auf der Achse Dividende-gegen-Kursgewinn, um die es in der GmbH-Frage geht (29,83 % gegen 1,49 %).

**Wie vermeiden:**
1. **Jede Datenreparatur braucht eine Gegenprobe in BEIDE Richtungen**, als Code, nicht als Blick: nach dem Eingriff darf die Auffaelligkeit nicht mehr messbar sein UND es darf keine neue entstanden sein. Zweiseitig — ein falscher Faktor verbiegt den Rand nach oben ODER unten, und ein kuenstlicher -80-%-Tag faellt in einem Momentum-Backtest niemandem auf, er verschiebt nur still die Rangliste.
2. **Fail-closed statt Warnung.** Die Gegenprobe bricht ab, wenn sie anschlaegt. Eine Warnung in einem Lauf mit 144 Backtests liest niemand.
3. **Paarungsbedingung statt Schwellwert.** Eine Rueckkehr aus einem Sprung um `f` ist ein Fall um `1/f - 1`. Passt der Betrag nicht, ist es ein echter Kurssturz. Geschlossen wird nur bei `|(1+r)*f - 1| <= Toleranz`.
4. **Nicht aufloesbare Faelle melden, nicht raten.** Bei verschraenkten Skalen (zweiter Sprung in offener Spanne) ist nicht bestimmbar, welcher Kurs auf welcher Skala liegt; dazu kommen Vendor-Sentinels. **13 der 25** auffaelligen Namen fallen darunter. Sie bleiben unberuehrt und stehen im Protokoll. Eine Bereinigung, die **12 statt 25** Namen repariert und das sagt, ist ehrlicher als eine, die 25 „repariert". (Die Zahlen standen hier zunaechst vertauscht — ausgerechnet die, die dieser Schritt korrigiert hat. Stage-2-Finding F-senior-7.)
5. **Abgeleitete Felder mitfuehren.** Vor jedem Eingriff auflisten, welche anderen Groessen in denselben Einheiten stehen, und die Invariante benennen, die erhalten bleiben muss (hier: Dividende je Kurseinheit).
6. **Ein Ergebnis, das die erwartete Antwort liefert, ist kein Beleg fuer korrekten Code** — es ist der Fall, in dem am wenigsten geprueft wird.

**Erkannt in:** Stage-1-Review (F-test-1, F-test-3) zu `research/mandat2/panel_bereinigt.py` und `p12f_neulauf_bereinigt.py`. Der Lauf war zu diesem Zeitpunkt bereits durchgelaufen und das Ergebnis formuliert; es wurde zurueckgehalten und nach der Reparatur neu erhoben.
**Referenzen:** E-106 (Off-by-one am Spannenende, von der Gegenprobe gefunden), E-083 (Bereinigungsverfahren erzeugt den gesuchten Effekt), E-096, E-102.

## E-108 — Der Waechter filterte nach einer Kennzahl, die die Reparatur selbst verschiebt
**Datum:** 2026-08-04
**Kategorie:** false-green / waechter-blind / fail-open-im-fail-closed
**Was passierte:** Der Fail-Closed-Waechter der Panel-Bereinigung meldet neue Ausreisser: `((r > 1.0) | (r < -0.5)) & (df.shift(1) > 1.0)`. Die 1-USD-Schwelle traegt eine fachliche Aussage — unter einem Dollar sind Verzehnfachungen real, dort ist ein Sprung kein Vendor-Fehler. Angewendet wurde sie aber auf das **bereinigte** Panel. Und die Bereinigung teilt Kurse durch den Spleissfaktor: sie schiebt sie systematisch unter die eigene Schwelle.

Im echten Lauf fielen dadurch **379 Symbol-Tage in fuenf Namen** aus dem Blickfeld des Waechters. RHT lag im Original nie unter 3,02 USD und rutschte an 86 Tagen darunter. Reproduziert wurde es mit dem bestehenden Guard-Test, alle Kurse durch 20 geteilt: derselbe Fehler, den der Test bei Kursen ueber 1 USD zuverlaessig faengt, laeuft bei kleineren Kursen still durch — eingefuegt wird eine Rendite von **+2400 %**, gemeldet wird `neu_entstanden: 0`.

Verschaerfend: die Testbegruendung behauptete ausdruecklich, der Detektor arbeite „auf dem ORIGINALPANEL". Fuer den Detektor stimmte das, fuer den Waechter nicht — und die beiden teilen sich die Funktion.

**Warum falsch:** Ein Waechter, der pruefen soll, ob eine Transformation Schaden angerichtet hat, darf seine Filterbedingung nicht aus dem **Ergebnis** der Transformation ziehen. Sonst kann die Transformation ihre eigene Ueberwachung abschalten — und zwar genau dort, wo sie am staerksten eingreift, denn je groesser der Spleissfaktor, desto tiefer rutscht der Kurs unter die Schwelle. Das ist ein Fail-Open im Innern eines Fail-Closed: die Struktur sieht sicher aus, die Richtung des Ausfalls ist Entwarnung.

**Wie vermeiden:**
1. **Jede Filter- oder Schwellenbedingung eines Waechters gehoert an den UNVERAENDERTEN Zustand.** Konkret: die Frage „war das ein Penny-Stock?" beantwortet der Kurs, wie er wirklich war — nicht der, den die Reparatur daraus gemacht hat. Referenzpanel explizit uebergeben, nicht implizit das Ergebnis nehmen.
2. **Beim Schreiben eines Waechters durchgehen, welche seiner Eingaenge die ueberwachte Operation veraendert.** Alles, was sie veraendert, ist als Filter disqualifiziert.
3. **Guards mit veraenderter Groessenordnung testen**, nicht nur mit veraenderter Form. Derselbe Fall mal 1/20 gerechnet haette den Fehler sofort gezeigt — die bestehende Testreihe variierte Morphologien, nie Skalen.
4. **Mutationsprobe auf die Schwelle selbst.** „Schwelle entfernen" muss einen Test zum Fallen bringen; tut es das nicht, ist die Schwelle unbelegt — hier ueberlebte sie die gesamte Suite.
**Erkannt in:** Stage-1-Review (F1, BLOCKER) zu `research/mandat2/panel_bereinigt.py`.
**Referenzen:** E-107 (dieselbe Bereinigung, einseitige Gegenprobe), E-103 (Fail-Open-Richtung nach der Korrektur beibehalten), E-066.

## E-109 — Der Text nannte einen anderen Vergleichsmassstab als der Code rechnete
**Datum:** 2026-08-04
**Kategorie:** false-evidence / bezugsgroesse
**Was passierte:** Die Kernzahl eines Befund-Dokuments lautete „**Ueberhoehung: 2,90 Prozentpunkte p. a.** gegenueber SPY". Gerechnet hatte der Code `intraday.cagr - pit_2004.cagr` — die Differenz zum survivorship-freien PIT-Universum, nicht zu SPY. Gegen SPY waeren es 3,36 pp gewesen. Das Skript selbst druckte korrekt „gegenueber dem PIT-Universum"; der Renderer schrieb SPY, weil SPY in derselben Tabelle stand. Die Zahl wanderte dann handgeschrieben ins Abschlussdokument der Kampagne.

Beide Zahlen sind legitim, aber sie messen Verschiedenes: der PIT-Vergleich isoliert die **Auswahl** (gleiches Verfahren, gleiche Gewichtung, nur Ausscheider mehr), der SPY-Vergleich enthaelt zusaetzlich Gewichtung und Indexkonstruktion — genau der Fehler, den dieselbe Kampagne zwei Tage zuvor als E-079 registriert hatte.

**Warum falsch:** Eine Differenz ohne korrekt benannten Bezug ist keine Messung, sondern eine Zahl. Der Fehler ist besonders zaeh, weil beide Werte plausibel sind und in dieselbe Richtung zeigen — es faellt niemandem auf, und der Befund bleibt „richtig", waehrend seine Begruendung falsch ist. Beim naechsten Lauf mit anderen Daten kann genau das kippen.

**Wie vermeiden:**
1. **Der Bezug gehoert in denselben Ausdruck wie die Zahl.** Wo eine Differenz berechnet wird, den Namen der Vergleichsgroesse aus den Daten ziehen (`pit["universum"]`), nicht im Fliesstext wiederholen.
2. **Wenn zwei Bezugsgroessen im Dokument vorkommen, beide ausweisen** und den Unterschied benennen — hier: 2,36–2,90 pp gegen PIT, 3,36 pp gegen SPY, und warum PIT der ehrlichere Massstab ist.
3. **Bereits registrierte Fehlerklassen gegen den eigenen neuen Text pruefen.** E-079 („der Benchmark war ein Drittel des Vorsprungs") stand im selben Dokument, drei Abschnitte darueber.
4. **Eine Zahl, die von einem Skript in ein Dokument wandert, braucht denselben Test wie Code** — hier faengt jetzt ein Renderer-Test den Bezugsnamen.
**Erkannt in:** Stage-1-Review (F4) zu `render_befund_datenqualitaet.py` und `ABSCHLUSS.md`.
**Referenzen:** E-079, E-085 (Zahlendrift zwischen Quelle und Dokument), E-081.

## E-110 — Ein reparierter Detektor entwertet die Artefakte seines Vorgaengers
**Datum:** 2026-08-04
**Kategorie:** wiring-gap / artefakt-drift
**Was passierte:** `korruptions_spannen` wurde nach einem BLOCKER-Fund repariert (Paarungsbedingung, Sentinel-Erkennung, Verschraenkungs-Guard). Zwei Skripte rufen die Funktion auf. Eines wurde neu ausgefuehrt, das andere nicht — sein Ergebnis-JSON blieb vom alten Lauf stehen. Der generierte Befund zog Abschnitt 2 aus dem alten und Abschnitt 4 aus dem neuen Artefakt: **246 gegen 25 Uebergangstage, 32.332 gegen 48.380 Tage auf falscher Skala**, im selben Dokument, ohne Warnung.

Aufgefallen ist es nur, weil ein Reviewer die Zeitstempel der Dateien mit dem Zeitstempel des Moduls verglich. Das Dokument selbst war syntaktisch einwandfrei, beide Zahlenreihen plausibel, jede einzelne Zahl korrekt gegen ihr Artefakt.

**Warum falsch:** Ein generiertes Dokument garantiert Konsistenz zwischen **Zahl und Satz** — nicht zwischen **Artefakt und Code**. Das ist ein anderer Vertrag, und der Generator kann ihn nicht einhalten, weil er die Artefakte nur liest. Sobald mehrere Skripte dieselbe Analysefunktion aufrufen, ist jede Aenderung an ihr eine **Invalidierung aller ihrer Artefakte**. Ohne Gegenmassnahme wandert der alte Stand still ins Dokument, und zwar in dem Moment, in dem die groesste Sorgfalt herrscht: direkt nach einer Reparatur.

**Wie vermeiden:**
1. **Nach jeder Aenderung an einer geteilten Analysefunktion die Konsumentenliste bestimmen** (`grep -rn "funktionsname" research/ scripts/`) und deren Artefakte neu erzeugen — vor dem Rendern, nicht danach.
2. **Artefakte tragen einen Strukturmarker.** Hier reichte es, ein Feld zu verlangen, das nur der neue Detektor schreibt (`unaufloesbar_grund`). Der Renderer bricht ab, wenn es fehlt, statt zu rendern.
3. **Querbezuege zwischen Artefakten pruefen.** Kennt Lauf B einen Namen als unauflösbar, den Lauf A gar nicht meldet, haben beide verschiedene Panels gesehen — das ist billig zu pruefen und faengt Drift, die kein Einzelfeld zeigt.
4. **Mtime-Regel als Faustformel:** kein Artefakt darf aelter sein als das Modul, das es erzeugt hat. Taugt zur Diagnose, nicht als alleiniger Guard (Checkouts setzen mtimes neu).
**Erkannt in:** Stage-2-Review (F-senior-1, BLOCKER) zu `render_befund_datenqualitaet.py` / `results/p12e_panel_hygiene.json`.
**Referenzen:** E-085 (Zahlendrift zwischen Quelle und Dokument), E-081, E-109.

## E-111 — Reparierbarkeit in die Messung geschrieben: „nicht heilbar" wurde zu „nicht krank"
**Datum:** 2026-08-04
**Kategorie:** logic-error / entwarnung-durch-konstruktion
**Was passierte:** Namen mit verschraenkten Preisskalen oder Vendor-Sentinel sollten von der Bereinigung nicht angefasst werden. Umgesetzt wurde das, indem der Detektor fuer sie `spannen: []`, `uebergaenge: []` und `n_tage_falsch: 0` zurueckgab — die Reparatur findet dann nichts zu tun.

Ein zweiter Konsument misst mit **denselben Feldern**, wie stark das Panel kontaminiert ist. Er bekam fuer die **13 kaputtesten Namen null gemeldet**: WFT mit 1.375 Sentinel-Tagen, YRCW mit 1.456, MCIC mit 844. Nach der Korrektur (Messfelder immer befuellen, nur ein Flag transportiert die Nicht-Reparierbarkeit) sprang die gemessene Kontamination von 24.123 auf **48.380 Tage** — die Haelfte des Schadens war unsichtbar, und zwar exakt die schlimmere Haelfte.

Die Bereinigung selbst brauchte das Leeren nie: sie prueft das Flag ohnehin selbst.

**Warum falsch:** Detektor und Reparatur beantworten verschiedene Fragen — „wie kaputt ist es?" und „was kann ich reparieren?". Wer das Ergebnis der einen benutzt, um die andere zu steuern, koppelt zwei Semantiken aneinander und schaltet die Messung genau dort ab, wo der Schaden am groessten ist. Der Ausfall zeigt dabei immer in die **Entwarnungsrichtung**, denn unbehandelbare Faelle sind typischerweise die schwersten. Ein Befund, der auf so einer Messung steht, ist systematisch zu optimistisch — und wirkt umso solider, je kaputter die Daten sind.

**Wie vermeiden:**
1. **Nicht-Verarbeitbarkeit immer als eigenes Flag transportieren**, nie durch Leeren der Messfelder. Der Konsument entscheidet, was er damit tut.
2. **Merksatz fuer den Test:** fuer einen als unbehandelbar markierten Fall muss der gemessene Befund **groesser** sein als null, nicht kleiner. Ein Test, der `n_tage_falsch == 0` fuer einen unauflösbaren Namen fordert, zementiert den Fehler — genau so ein Test existierte hier und musste mitgedreht werden.
3. **Bei jedem Feld, das zwei Konsumenten hat, die Frage stellen:** beantwortet es fuer beide dieselbe Frage? Wenn nicht, aufspalten.
4. **Sanity-Summe:** die Gesamtkontamination muss mit der Zahl der betroffenen Namen wachsen, nicht schrumpfen, wenn mehr Namen als unbehandelbar erkannt werden.
**Erkannt in:** Stage-2-Review (F-senior-2, BLOCKER) zu `p12d_survivorship_schranke.py`.
**Referenzen:** E-102 (Entwarnung auf Proxy), E-103 (Fail-Open-Richtung), E-083, E-066.

## E-112 — Abwesenheit im Dateisystem als Eigenschaft des Datenanbieters ausgegeben
**Datum:** 2026-08-04
**Kategorie:** false-evidence / ls-statt-call
**Was passierte:** Nach einem Datenpull sollte gemessen werden, ob der Intraday-Endpunkt die Survivorship-Luecke schliessen kann. Gemessen wurde: welche Symbole haben eine Parquet-Datei? Daraus entstand ein „Anreicherungsfaktor 3,06" — wer eine Datei hat, ueberlebt dreimal so haeufig — und der Schluss „mehr Anfragen erhoehen die Praezision, nicht die Unverzerrtheit".

Eine Stichprobe gegen dieselbe API widerlegte das in einem Zug: **AMZN, GILD, VRSN, ADBE, NVDA, COST, ROST und PAYX — acht von acht Namen aus genau der Fehlgruppe — liefern 7.000–8.000 Bars im Suchfenster.** Sie waren nie angefragt worden. `intraday_pull.py` schreibt bei Leerergebnis weder Datei noch Protokoll; damit sind „nie angefragt", „angefragt und leer" und „Anfrage fehlgeschlagen" auf der Platte ununterscheidbar.

Der Faktor mass also die Zusammensetzung einer **nicht protokollierten Anfrageliste** — und die stammte mit hoher Wahrscheinlichkeit aus dem Tagespanel, dessen eigene Abdeckungsluecke dieselbe Kampagne zwei Abschnitte weiter oben als survivorship-behaftet ausweist. Die Kennzahl mass teilweise die Verzerrung ihrer eigenen Quelle.

**Warum falsch:** Ein `ls` beantwortet keine Frage ueber einen Anbieter. Die Abwesenheit einer Datei hat mindestens drei Ursachen, und genau eine davon ist die behauptete. Verschaerfend: derselbe Anti-Pattern steht seit drei Tagen als **E-080** im selben Katalog („Datenverfuegbarkeit aus dem Gedaechtnis behauptet statt an der API geprueft") — die Wiederholung kam zustande, weil die Evidenzquelle diesmal *lokal und konkret* aussah. Ein Verzeichnis fuehlt sich wie eine Messung an; es ist eine.

**Wie vermeiden:**
1. **Jeder Ingest schreibt ein Anfrage-Protokoll** — Symbol, Fenster, HTTP-Status, Bar-Zahl —, auch und gerade fuer Leerergebnisse. Ohne das ist jede spaetere Abdeckungsaussage unbelegbar.
2. **Abdeckungskennzahlen nur ueber ANGEFRAGTE Symbole bilden.** Wer die Anfrageliste nicht kennt, kann keine Aussage ueber den Anbieter treffen — hoechstens eine ueber den eigenen Pull, und die muss dann auch so heissen.
3. **Aussagen ueber einen Anbieter brauchen einen Call**, und der Call braucht eine **Kontrollgruppe**: ohne Namen, die liefern, ist ein Negativbefund nicht von einem kaputten Aufruf zu unterscheiden. Hier: 6 Ausscheider mit 0 Bars gegen 8 Ueberlebende mit 7.000+.
4. **Bei jeder „X liefert nicht"-Aussage fragen: habe ich X gefragt?** Die Antwort steht im Zweifel nicht auf der Platte.
**Erkannt in:** Stage-2-Review (F-senior-1/2, BLOCKER) zu `research/mandat2/p12g_pull_bilanz.py`.
**Referenzen:** E-080 (dieselbe Klasse, drei Tage zuvor), E-109, E-111.

## E-113 — Delisting-Coverage unter dem Post-Insolvenz-Ticker geprueft
**Datum:** 2026-08-04
**Kategorie:** logic-error / test-kann-nicht-widerlegen
**Was passierte:** Der Haertetest fuer „fuehrt die Quelle Ausscheider?" fragte CCTYQ, EKDKQ, LEHMQ, MTLQQ und WAMUQ ab — alle mit 0 Bars, was als Beleg gewertet wurde. Diese Q-Symbole entstehen aber **erst mit dem Chapter-11-Handel**. Waehrend der Index-Mitgliedschaft hiessen die Namen LEH, WM, EK, GM, CC.

Ein Negativbefund auf dem Q-Ticker ist damit fast garantiert, unabhaengig davon, ob der Anbieter die Historie des Namens fuehrt. Der Test konnte die These **nicht widerlegen** — also stuetzte er sie auch nicht. (Die Nachpruefung unter den richtigen Symbolen ergab ebenfalls 0 Bars: das Verdikt hielt, der Beleg fehlte.)

Dieselbe Klasse trifft retroaktive Ticker in Membership-Serien: GOOGL liefert 0 Bars fuer 2006, GOOG 20.340 — der Name stand damals unter einem anderen Symbol im Index. Ebenso TPR (vorher COH) und BF.B (HTTP 404 wegen des Punkts). Solche Faelle erzeugen **Scheinluecken**, die wie Anbieter-Grenzen aussehen.

**Warum falsch:** Ein Test, dessen Ergebnis unabhaengig vom geprueften Sachverhalt feststeht, ist kein Test. Er hat dieselbe Struktur wie eine gesaettigte Messung (vgl. den Kipp-Abstand in P12f): das Ergebnis ist richtig und die Begruendung leer.

**Wie vermeiden:**
1. **Coverage immer unter dem zum `as_of` gueltigen Handelssymbol pruefen.** Der Ticker ist ein Zeitreihen-Attribut, kein Schluessel.
2. **Symbol-Change-Historie mitfuehren** (EODHD: `get_symbol_change_history`), wenn ein Universum ueber Jahrzehnte laeuft. Wo sie fehlt, beide Kandidaten pruefen und beide Zahlen ausweisen.
3. **Vor jedem Negativtest fragen: unter welchen Umstaenden waere das Ergebnis anders?** Faellt die Antwort schwer, misst der Test nichts.
4. **Q-Suffix, Punkte im Symbol und retroaktive Umbenennungen** sind die drei bekannten Fallen in US-Indexhistorien.
**Erkannt in:** Stage-2-Review (F-senior-3, MAJOR) zu `research/mandat2/p12g_pull_bilanz.py`.
**Referenzen:** E-112, E-080, E-104.

## E-114 — Der Ticker als Schluessel: eine Panel-Spalte fuehrte zwei Unternehmen
**Datum:** 2026-08-04
**Kategorie:** false-mechanism / identitaets-annahme
**Was passierte:** Beim Versuch, die Survivorship-Luecke ueber Tagesdaten zu schliessen, fiel auf: 99 PIT-Mitglieder liegen in der Rohdatei, haben dort aber Kurse **nur ausserhalb des Suchfensters**. Der Grund ist keine Luecke, sondern eine Verwechslung: der Anbieter liefert unter einem Symbol die **heutige** Firma. ABI (Anheuser-Busch, uebernommen 2008) beginnt 2025-06-26; ABS (Albertsons, 2006) beginnt 2018; ALTR (Altera, 2015) beginnt 2017.

Wo die Neuvergabe INNERHALB des Fensters liegt, steht die Historie von Firma A, eine mehrjaehrige Luecke, dann Firma B — in derselben Spalte. Betroffen: **29 Spalten (Schwelle 500 Handelstage), alle 29 waren Index-Mitglieder** (CTXS an 205 Terminen, TOY an 114).

Zwei Schaeden, beide am echten Bestand gemessen:

1. **Kurssprung am Wiedereinstieg.** Die Engine bewertet eine in A gekaufte Position erstmals mit dem Kurs von B: CGP −3,49 % Portfolio-Tagesrendite (Rang 5.419 von 5.548).
2. **Ausfall der Delisting-Hygiene** — der groessere. Der Zwangsverkauf prueft `last_valid < t`; bei einem recycelten Ticker liegt `last_valid` am Ende der Serie von **B**, die Bedingung ist nie erfuellt. CGP lief **3.264 Handelstage (13 Jahre)** im Bestand weiter, bewertet auf dem eingefrorenen letzten Kurs von 2001. Ueber drei Namen zusammen 5.035 tote Haltetage.

Keiner der bestehenden Detektoren sah das: `korruptions_spannen` arbeitet auf `pct_change(fill_method=None)`, und ueber eine NaN-Luecke ist die Rendite NaN. Der Sprung war fuer die Preisfehler-Pruefung unsichtbar und fuer die Engine (`close_ff`) sichtbar — genau andersherum, als man es braucht.

**Warum falsch:** Ein Ticker ist ein **Zeitreihen-Attribut**, kein Schluessel. Ueber zwei Jahrzehnte werden Symbole neu vergeben, umbenannt und getauscht. Wer sie als Identitaet behandelt, verknuepft Unternehmen — und zwar bevorzugt dort, wo eines verschwunden ist, also genau in der Teilmenge, um die es bei Survivorship geht.

**Wie vermeiden:**
1. **Vor jeder Panel-Konstruktion pruefen, ob eine Spalte durchgehend dasselbe Unternehmen fuehrt.** Signatur: eine Luecke, nach der die Serie weiterlaeuft. Ein delistetes Unternehmen kommt nicht nach Jahren zurueck.
2. **Die Delisting-Regel darf nicht am Serienende haengen**, wenn die Serie mehrere Emittenten enthaelt. Besser zusaetzlich: eine Position ohne Kurs ueber N Tage ist nicht haltbar — unabhaengig von der Ursache.
3. **Detektoren, die auf Renditen arbeiten, sehen keine Luecken.** Wer Preisfehler und Luecken beide sucht, braucht zwei Pruefungen; `pct_change` ueber NaN liefert NaN und schweigt.
4. **Symbol-Change-Historie des Anbieters nutzen, wo verfuegbar.** Bei EODHD beginnt sie 2022-07-22 und deckt aeltere Fenster NICHT ab (geprueft) — dann ist jede Zuordnung eine dokumentationspflichtige Heuristik.
**PRAEZISIERT DURCH E-115 UND E-117:** „29 Spalten fuehren zwei Unternehmen" ist die Formulierung nach der VERMUTETEN Ursache. Belegt ist nur die **Signatur** (Luecke >= 500 Handelstage); 8 der 30 Schnitte liegen im Faktorband, in dem der Kurs fortsetzt, und sind wahrscheinliche Fehltreffer. Wer diesen Eintrag zitiert, muss E-117 mitlesen.

**Erkannt in:** eigener Pruefung beim Anschluss an P12g; `research/mandat2/p12h_ticker_recycling.py`.
**Referenzen:** E-113 (dieselbe Klasse, Intraday), E-112, E-079, **E-115**, **E-117**.

## E-115 — „Unterbrochen" mit „recycelt" verwechselt: die Reparatur zerlegte eine intakte Firma
**Datum:** 2026-08-04
**Kategorie:** false-repair / klasse-nicht-getrennt
**Was passierte:** Der Detektor aus E-114 findet *unterbrochene* Serien. Die erste Trennung behandelte jede Unterbrechung ab 120 Handelstagen als Firmenwechsel. Beim Lauf zeigte sich: **CCE (Coca-Cola Enterprises) hat sechs Unterbrechungen, jaehrlich im November** — die Firma existierte durchgehend, der Feed hat Loecher. Die Trennung haette sie in **sieben Stuecke** zerlegt und damit eine intakte Historie zerstoert, um einen Fehler zu beheben, den sie dort nicht hatte.

Aufgefallen ist es nur, weil die Segmentliste im Lauf ausgegeben wurde und das jaehrliche Muster ins Auge sprang.

Gemessen ueber alle 57 Luecken >= 60 Handelstage, Anteil mit Faktor 0,5..2,0 („Kurs setzt fort"):
60–120: 100 % · 120–250: 86 % · 250–500: 36 % · 500–1000: 42 % · >= 1000: 17 %.

Unter 250 Tagen dominieren Datenloecher, ab 500 der Firmenwechsel. Die Schwelle liegt jetzt bei 500; der Bereich 120–500 bleibt unangetastet und wird als eigenes, ungeloestes Problem ausgewiesen.

**Warum falsch:** Ein Detektor findet eine **Signatur**, keine **Ursache**. Zwischen beiden liegt eine Klassifikation, und wer sie ueberspringt, behandelt alle Traeger der Signatur gleich. Der Schaden ist dabei asymmetrisch: eine unbehandelte Korruption verfaelscht ein Ergebnis, eine falsch behandelte intakte Serie erzeugt ein neues — und sieht dabei aus wie Sorgfalt.

**Wie vermeiden:**
1. **Vor jeder Reparatur die Signatur nach Ursachen aufschluesseln** und die Verteilung ausweisen. Hier genuegte eine Tabelle „Lueckenlaenge gegen Kursfaktor", um zwei Klassen sichtbar zu machen.
2. **Die Schwelle aus den Daten begruenden, nicht aus dem Bauch** — und die Herleitung dorthin schreiben, wo die Konstante steht.
3. **Faelle im unklaren Bereich NICHT mitbehandeln**, sondern als offen ausweisen. Eine Reparatur, die 29 von 37 Faellen loest und das sagt, ist besser als eine, die 37 „loest".
4. **Die Segmentliste eines Reparaturlaufs immer ausgeben.** Genau sie hat den Fehler gezeigt; eine stille Reparatur haette ihn nicht.
**Erkannt in:** eigener Pruefung des ersten P12i-Laufs.
**Referenzen:** E-107 (Bereinigung schlimmer als der Fehler), E-114, E-083.

## E-116 — Artefakt vor der letzten Kennzahl geschrieben: die Aufloesungszahl war nur stdout
**Datum:** 2026-08-04
**Kategorie:** wiring-gap / nicht-persistiert
**Was passierte:** `p12i_neulauf_getrennt.py` schrieb das Ergebnis-JSON, BEVOR `kipp_abstand` berechnet und an `ergebnis` gehaengt wurde. Der Schluessel landete nie in der Datei; die Zahl existierte ausschliesslich im Terminal.

Folge: im Abschlussdokument stand „kein Kandidat kam dem Deckel je naeher als rund 26 pp" — die Zahl des VORHERIGEN Laufs (P12f). Der aktuelle Lauf ergab **28,6 pp**. Kein Artefakt konnte widersprechen, weil die Kennzahl in keinem stand. Genau die Zahl, die aus „dreht nicht" erst eine Robustheitsaussage macht, war die einzige nicht persistierte.

**Warum falsch:** Ein Ergebnis-Artefakt hat zwei Aufgaben — das Ergebnis zu speichern und **spaetere Behauptungen darueber pruefbar zu machen**. Fehlt eine Kennzahl darin, ist die zweite Aufgabe fuer genau diese Kennzahl nicht erfuellt, und der Fehler faellt erst auf, wenn jemand von Hand nachrechnet. Die Reihenfolge „schreiben, dann weiterrechnen" sieht harmlos aus, weil der Lauf korrekt durchlaeuft und die Konsole alles zeigt.

**Wie vermeiden:**
1. **Ergebnisdateien immer als LETZTE Anweisung schreiben.** Kein `write_text` mitten in der Funktion, auch nicht „schon mal sichern".
2. **Regel fuer Befundtexte: was nicht im Artefakt steht, gehoert nicht in den Befund.** Jede Zahl im Dokument muss ueber einen Schluessel im JSON auffindbar sein.
3. **Beim Uebernehmen einer Kennzahl aus einem anderen Lauf** (hier: dieselbe Funktion in P12f und P12i) pruefen, ob die neue Zahl auch neu berechnet wurde — Zahlen wandern leichter zwischen Dokumenten als zwischen Laeufen.
**NACHTRAG — die Regel im selben Zug gebrochen:** Nach dem Fix wurden die Befundzahlen erneut aus **stdout eines noch laufenden Prozesses** uebernommen; das Artefakt auf der Platte war zu diesem Zeitpunkt noch die aeltere Fassung ohne `kipp_abstand`. Die Zahlen stimmten am Ende — aber das war nicht garantiert, und nachgeprueft wurde es erst durch die Stage-1-Validierung. Regel 2 gilt also auch fuer den **Zeitpunkt**: erst wenn der Lauf beendet UND das Artefakt geschrieben ist, darf eine Zahl ins Dokument. Eine frisch aufgeschriebene Regel schuetzt nicht davor, sie eine Stunde spaeter zu brechen.

**Erkannt in:** Stage-2-Review (F-senior-1/2) zu `p12i_neulauf_getrennt.py`. Derselbe Defekt bestand in `p12f_neulauf_bereinigt.py`. Der Nachtrag stammt aus der Stage-1-Validierung desselben Steps.
**Referenzen:** E-085, E-110.

## E-117 — Eindimensionale Regel auf zweidimensionaler Evidenz
**Datum:** 2026-08-04
**Kategorie:** logic-error / klassifikation-verkuerzt
**Was passierte:** Die Trennschwelle fuer recycelte Ticker (500 Handelstage) wurde aus einer Tabelle **Lueckenlaenge gegen Kursfaktor** hergeleitet — zwei Merkmale, weil eines allein die beiden Ursachen (Firmenwechsel gegen Vendor-Datenloch) nicht trennt. Die Regel benutzt danach nur noch die Lueckenlaenge.

Ergebnis: **8 der 30 Schnitte** liegen im Faktorband 0,5–2,0, das dieselbe Herleitung als „Kurs setzt fort" definiert (HSH 0,80 · MWI 1,80 · MYG 1,15 · NLC 0,73 · RX 0,59 · RYC 1,20 · WLL 0,83 und 0,91). Der Befundtext lautete trotzdem „29 Spalten fuehren zwei Unternehmen".

Der Folgeschaden ist **asymmetrisch**: fuer jede falsch getrennte Spalte greift danach der Zwangsverkauf, weil `last_valid` nun am kuenstlichen Segmentende liegt. Die Reparatur erzeugt also ein **fabriziertes Delisting** — bei WLL (Faktor 0,91) waere das 2010, obwohl die Firma weiterlief.

**Praezisierung (Stage-3-Finding F-auditor-3):** Hier stand zunaechst „WLL wird 2010 aus dem Portfolio geworfen" — eine Aussage ueber echtes Portfolioverhalten, die der Lauf nicht stuetzt. Die Instrumentierung derselben Session meldet genau drei je gehaltene Namen (CGP, NGH, NVLS); WLL ist nicht darunter, seine Indexmitgliedschaft endet 2002-01. Ausgerechnet ein Eintrag ueber „mehr behaupten als klassifiziert" behauptete selbst mehr als gemessen. Gemessen gilt: **alle 8 fraglichen Schnitte liegen nach dem letzten Mitgliedschaftstermin** ihres Symbols — der Schaden ist damit auf Namen begrenzt, die zum Schnittzeitpunkt nur noch ueber die Mindesthaltedauer erreichbar waren, und die Richtung ist konservativ (ein Zwangsverkauf kann keinen PASS erzeugen).

**Warum falsch:** Wer zwei Merkmale erhebt, um zwei Klassen zu trennen, und dann nach einem entscheidet, uebernimmt die Fehlerrate des weggelassenen — und verliert das Recht, das Ergebnis als **Klassifikation** zu bezeichnen. Es bleibt eine Signaturliste. Die Verkuerzung ist besonders verfuehrerisch, weil die Herleitung sauber aussieht: die Tabelle steht da, sie ist richtig gerechnet, und trotzdem geht die Haelfte ihrer Information beim Uebergang zur Regel verloren.

**Wie vermeiden:**
1. **Benutzt die Herleitung zwei Achsen, muss die Regel beide benutzen** — oder die zweite Achse als bezifferte Restfehlerquelle ausweisen, je Fall, im Artefakt.
2. **Das Ergebnis nach der Signatur benennen, nicht nach der vermuteten Ursache.** „Spalten mit Luecke >= 500", nicht „Spalten mit zwei Unternehmen".
3. **Bei asymmetrischem Schaden** — wenn die Reparatur selbst Ereignisse erzeugt — die wahrscheinlichen Fehltreffer explizit zaehlen und in den Befund schreiben. Eine Reparatur, die 22 von 30 Faellen richtig trifft und das sagt, ist besser als eine, die 30 „loest".
4. **Die Merkmale, die in die Entscheidung eingehen, gehoeren ins Protokoll** — nicht nur in den Kommentar ueber der Konstante.
**Erkannt in:** Stage-2-Review (F-senior-3) zu `panel_getrennt.py` und `ABSCHLUSS.md`.
**Referenzen:** E-115 (dieselbe Klasse, eine Stufe grober), E-107, E-083.

## E-118 — Kontrollgruppe zerstoerte zwei Dinge und sollte nur eines pruefen
**Datum:** 2026-08-05
**Kategorie:** false-control / zwei-ursachen
**Was passierte:** Die Zufallskontrolle zu `preis>SMA200` sollte genau eine Frage beantworten: zaehlt das TIMING der Marktpausen, oder nur die Zeit ausserhalb? Dafuer wurden die An/Aus-Bloecke des echten Gates genommen und ihre Reihenfolge permutiert.

Beim Permutieren koennen zwei gleichwertige Bloecke nebeneinander landen und **verschmelzen**. Die Kontrollgruppe bekam dadurch weniger und laengere Episoden als das echte Gate — die Laengenmultimenge `[1,1,1,2,3,3,...]` wurde zu `[1,3,3,3,4,4]`. Der investierte Anteil blieb exakt gleich, die Episodenstruktur nicht.

Der Begleittext im Befund behauptete trotzdem woertlich „gleiche Zahl Episoden, gleiche Laengenverteilung, nur anderes Timing". Zwei der drei Zusicherungen waren falsch.

**Warum falsch:** Eine Kontrollgruppe ist nur dann eine Kontrolle, wenn sie sich in **genau einer** Eigenschaft vom Kandidaten unterscheidet. Aendert sie zwei, misst die Differenz beide gleichzeitig und laesst sich keiner Ursache zuordnen. Besonders tueckisch ist der Fall hier, weil die Richtung des zweiten Effekts unklar ist: laengere Pausen koennen die Kontrolle staerker oder schwaecher machen: das Ergebnis war also weder konservativ noch optimistisch, sondern uninterpretierbar.

Die korrigierte Fassung (Wertfolge behalten, nur die Laengen INNERHALB jeder Wertklasse permutieren) erhaelt Anteil, Episodenzahl und Laengenverteilung exakt und verschiebt ausschliesslich, wann die langen und kurzen Episoden liegen. Sie fiel schaerfer aus als die fehlerhafte (0/60 statt 1/60 erreichen den Kandidaten) — der Fehler hatte also gegen den Kandidaten gewirkt, was ihn nicht harmloser macht, nur zufaellig unauffaellig.

**Wie vermeiden:**
1. **Die erhaltenen Groessen einer Kontrolle als Zusicherung TESTEN**, nicht im Docstring behaupten. Ein Test auf „Anteil, Anzahl und Laengenverteilung unveraendert" hat den Fehler hier gefunden, bevor die Zahl in ein Dokument wanderte.
2. **Bei jeder Permutation fragen, ob benachbarte Elemente verschmelzen koennen.** Laufkodierte Daten (Runs, Bloecke, Segmente) tun das fast immer.
3. **Formulieren, was die Kontrolle NICHT veraendert, und jedes Element der Liste einzeln pruefen.** Wer drei Dinge zusichert, braucht drei Assertions.
4. **Ein Fehler, der gegen den eigenen Kandidaten wirkt, ist kein kleinerer Fehler** — er faellt nur nicht durch ein zu gutes Ergebnis auf.
**Erkannt in:** eigenem Test `test_mischen_erhaelt_anteil_anzahl_und_laengen` waehrend der Testerstellung zu P13d.
**Referenzen:** E-083 (Bereinigung erzeugt den gesuchten Effekt), E-107, E-079.

## E-119 — Ein Rueckgabefeld, das wie ein Messwert heisst und eine Konstante ist
**Datum:** 2026-08-05
**Kategorie:** false-evidence / konstante-als-messwert
**Was passierte:** `run_buy_and_hold` in `research/mandat2/engine.py` gibt `n_trades=1` **fest verdrahtet** zurueck — die Schwesterfunktion `run_strategy` zaehlt an derselben Stelle echt mit. Der erste Entwurf von P13 uebernahm `r.n_trades` in Ausgabe und Artefakt.

Im Trockenlauf stand deshalb „Trades 1" unter einem SMA200-Filter ueber 22 Jahre, der nachweislich neunmal kauft und neunmal verkauft. Die Zahl war nicht falsch gerundet oder ungenau, sie war ueberhaupt keine Messung.

**Warum falsch:** Ein benanntes Feld in einem Ergebnisobjekt ist eine Zusicherung. Wer es uebernimmt, berichtet eine Konstante als Befund, und niemand sieht es dem Artefakt spaeter an. Die Konstante ist im Kontext ihrer eigenen Funktion sogar vertretbar (Buy-and-Hold kauft einmal) — sie wird erst falsch, sobald dieselbe Funktion einen optionalen Parameter bekommt, der das Verhalten aendert (`risk_off_gate`). Das Feld ist mit dem Parameter nicht mitgewachsen.

**Wie vermeiden:**
1. **Vor der Uebernahme eines Rueckgabefelds einmal die Quelle lesen.** Ein `grep` auf den Feldnamen in der erzeugenden Funktion kostet Sekunden.
2. **Plausibilitaet gegen die Fachlogik pruefen:** ein Trendfilter mit einem einzigen Trade ueber 22 Jahre ist kein Trendfilter. Auffaellige Werte sind billiger zu pruefen als stille.
3. **Statt die Logik nachzubauen, die Engine instrumentieren** (E-102): den Bestand vor und nach dem Aufruf vergleichen. Eine Nachbildung waere eine zweite Buchungswahrheit gewesen und haette die internen Frueh-Returns (`notional <= 0`, `EPS_QTY`) nicht gekannt.
4. **Optionale Parameter, die das Verhalten aendern, verpflichten zur Pruefung aller Rueckgabefelder** — nicht nur derer, die man gerade braucht.
**Erkannt in:** Trockenlauf vor dem eigentlichen P13-Lauf (auffaellige Zahl, dann Quelle gelesen).
**Referenzen:** E-102 (instrumentieren statt schaetzen), E-109 (Text nennt anderen Massstab als der Code).

## E-120 — Ueberholte Methodik kopiert, weil sie das naechstliegende Geschwistermodul war
**Datum:** 2026-08-05
**Kategorie:** logic-error / verworfene-methode-verlaengert
**Was passierte:** `p13e_dsr_pbo_spy.py` wurde bewusst als Kopie von `p7_dsr_pbo.py` gebaut, mit der ausdruecklichen Begruendung im eigenen Docstring: „ein eigener Weg waere hier eine dritte Wahrheit und wuerde die Vergleichbarkeit zerstoeren".

P7 war zu diesem Zeitpunkt bereits durch `p8_dsr_heterogen.py` ersetzt und im eigenen Fehlerlog als **E-077** protokolliert: Varianz aus der Klonfamilie, CSCV auf Klonspalten. Die Kopie erbte exakt den Fehler, der die Kampagne drei Tage zuvor davon abgehalten hatte, den Holdout-Schuss auszuloesen — und veroeffentlichte darauf ein „DSR bestanden ✅, p = 0,9974".

Auf dem gueltigen Schaetzer (heterogenes V aus P8, 1,328e-04 statt 1,503e-05) steigt die Schwelle von 0,0140 auf 0,0415 und p faellt auf **0,7838** — durchgefallen. Die Klonvarianz war 8,8-fach zu klein.

Verschaerfend: Der Befundtext stellte die eigenen 0,9974 aus der Klonfamilie neben die 0,9512 des Aktien-Kandidaten aus der HETEROGENEN Familie und leitete daraus eine „Symmetrie" ab („jeder hat eine Haelfte bestanden"). Zwei verschiedene Schaetzer, verglichen zugunsten des eigenen Kandidaten. Auf gleichem Fundament liegt der neue Kandidat schlechter.

**Warum falsch:** Vergleichbarkeit mit einem frueheren Lauf ist nur dann ein Wert, wenn der fruehere Lauf noch gilt. Rule 50 („keine zweite Wahrheit") schuetzt vor divergierenden **Implementierungen**, nicht vor der Wiederverwendung einer verworfenen **Methode**. Wer den ueberholten Stand kopiert, erzeugt keine Konsistenz, sondern verlaengert einen bekannten Fehler in einen neuen Befund — und tarnt ihn als Disziplin. Die Begruendung klingt dabei genau wie die richtige.

**Wie vermeiden:**
1. **Vor dem Kopieren eines Moduls das Fehlerlog nach dem Dateinamen greppen** (`Erkannt in:`-Zeilen). Ein Treffer ist ein Stoppschild, kein Hinweis.
2. **Pruefen, ob ein Nachfolgemodul existiert** (`p7` -> `p8`) und ob dessen Docstring eine Entscheidungsregel festschreibt. Die gilt dann auch fuer den neuen Kandidaten.
3. **Bei Korrekturverfahren muessen `n_trials` und `variance_across_trials` dieselbe Bezugsgroesse haben.** Das ist eine Eigenschaft der FAMILIE, nicht des Kandidaten, und wird beim Kopieren nicht automatisch mitgeprueft.
4. **Kennzahlen zweier Kandidaten nur zitieren, wenn sie mit demselben Schaetzer entstanden sind.** Sonst ist der Vergleich eine Aussage ueber die Schaetzer.
5. **Bei Zweifel alle Varianten rechnen und die konservativste als Entscheidungsgrundlage kennzeichnen** — hier: drei Schaetzer im Artefakt, `entscheidungsfaehig` als Flag je Eintrag.
**Erkannt in:** Stage-2-Review (F-senior-1, BLOCKER) zu `p13e_dsr_pbo_spy.py`.
**Referenzen:** E-077 (der geerbte Fehler), E-078, Rule 50.

## E-121 — Invariante auf der Signalebene getestet, auf der Portfolioebene behauptet
**Datum:** 2026-08-05
**Kategorie:** test-anti-pattern / ebenenwechsel
**Was passierte:** Nach E-118 wurde die Zufallskontrolle korrigiert und mit einem Test abgesichert, der Anteil, Blockzahl und Laengenverteilung der GATE-Serie prueft. Der Befundtext uebernahm die Zusicherung woertlich fuer die Strategie: „Exakt erhalten bleiben Anteil investierter Tage, Zahl der Episoden und deren Laengenverteilung."

Die Engine liest das Gate aber nur an Monatsenden. Zwischen Signal und Wirkung liegt ein Sampling-Schritt, der die Erhaltung nicht durchreicht. Nachgemessen (Instrumentierung von `Portfolio.set_date`): der echte Filter war an 66,3 % der Tage investiert, die 60 Zufallslaeufe an 62,4–67,3 %. Die Groesse streut also um rund 5 pp, statt exakt zu sein.

Der Befund wurde daraufhin getrennt: „auf Signalebene exakt erhalten" gegen „auf Portfolioebene gemessen 62,4–67,3 %". Inhaltlich stuetzt die Messung das Ergebnis sogar — der echte Filter ist MEHR im Markt als der typische Zufallslauf, sein Vorsprung stammt also nicht aus zusaetzlicher Abwesenheit. Das war vorher aber nicht gewusst, sondern angenommen.

**Warum falsch:** Eine Invariante gilt an der Stelle, an der sie gemessen wurde. Ein Test, der die Zusicherung eine Ebene unterhalb der Behauptung prueft, erzeugt Sicherheit fuer eine Aussage, die er nicht deckt — schaedlicher als gar kein Test, weil er den Vorbehalt aktiv verdraengt. Genau dieselbe Klasse wie E-118, eine Ebene tiefer.

**Wie vermeiden:**
1. **Die Zusicherung dort messen, wo sie im Text steht.** Bei Portfolio-Aussagen die realisierte Groesse aus dem Lauf ins Artefakt schreiben, nicht die Signalgroesse.
2. **Bei jeder Kontrollgruppe die Kette Signal -> Konsum -> Wirkung durchgehen** und fragen, an welcher Stelle ein Sampling, ein Guard oder ein Frueh-Return die Erhaltung bricht. Rebalance-Termine, Mindesthaltedauern und Kostenschwellen tun das regelmaessig.
3. **Faustregel: steht im Befund „exakt erhalten", muss es genau dafuer eine Assertion auf DERSELBEN Ebene geben.** Sonst heisst es „auf Ebene X erhalten" — mit Nennung von X.
**Erkannt in:** Stage-2-Review (F-senior-4) zu `p13d_zufallstiming.py`.
**Referenzen:** E-118 (dieselbe Klasse, eine Ebene hoeher), E-078, E-102.

## E-122 — Halb generiertes Dokument: Tabelle nachgerechnet, Fliesstext getippt
**Datum:** 2026-08-05
**Kategorie:** other / falsche-zusicherung
**Was passierte:** `BEFUND_SPY_TREND.md` wird generiert und traegt im Kopf die Zusicherung „jede Zahl hier hat einen Schluessel dort". Ein starker Drift-Guard rechnet die Bandtabelle Zeile fuer Zeile aus den Artefakten nach.

Im Fliesstext daneben standen handgetippte Zahlen: „12 bis 56 Buchungen" (Artefakte: 6…56 ohne, 6…62 mit Verzoegerung — die getippte Spanne war die einer einzigen Definition, waehrend die Schlussfolgerung ueber alle drei lief), „144" viermal, „acht der zwoelf" obwohl `MINDEST_KETTE` eigens als Konstante extrahiert worden war, und „3.529" direkt neben dem gerenderten Feld mit demselben Wert. Zusaetzlich las das Dokument den LIVE-Trial-Zaehler statt des im Artefakt eingefrorenen.

Die einzige Zahl, die tatsaechlich gegen die Artefakte driftete, lag im ungeschuetzten Teil — und war von dort bereits nach `ABSCHLUSS.md` weiterkopiert worden.

**Warum falsch:** Ein Dokument mit Generierungs-Kopfzeile und Drift-Guard sagt dem Leser: hier ist nichts von Hand. Trifft das nur fuer einen Teil zu, ist die Zusicherung schlechter als keine — sie lenkt die Pruefung genau auf den bereits geschuetzten Bereich. Fliesstextzahlen wandern zudem erfahrungsgemaess in Zusammenfassungen weiter, wo sie keinen Guard mehr haben.

**Wie vermeiden:**
1. **Im Renderer keine Ziffer als Literal** — auch nicht „144", nicht als Wort („acht") und nicht als Tausenderpunkt-Duplikat einer bereits gerenderten Zahl.
2. **Abgeleitete Groessen ableiten, nicht tippen:** Fensterzahl aus dem Artefakt, Ueberlappung aus `FENSTER_JAHRE * 12`, Spannen aus `min`/`max` ueber die Zeilen.
3. **Live gelesene Zustaende gehoeren nicht in ein Dokument, das aus Artefakten reproduzierbar sein soll.** Ein Zaehler, der zwischen Lauf und Rendern weiterlaeuft, macht das Dokument unreproduzierbar.
4. **Drift-Guard ueber den Fliesstext ausdehnen** oder die Kopfzusicherung auf das einschraenken, was tatsaechlich geprueft wird.
**Erkannt in:** Stage-2-Review (F-senior-5) zu `render_befund_p13.py` und `ABSCHLUSS.md`.
**Referenzen:** E-085, E-116, E-117.

## E-123 — Gleicher Spaltenname, andere Wertemenge: zweite Wahrheit ohne Schemabruch
**Datum:** 2026-08-05
**Kategorie:** wiring-gap / stille-nullkante
**Was passierte:** Ein zweiter Form-4-Bestand (`pull_form4_dera.py`) schrieb `transaction_type` als `{"buy","sell","unknown"}`. Der bestehende Bestand `form4_broad` fuehrt unter DEMSELBEN Spaltennamen `{"P","S","unknown"}`, weil er `classify_transaction_code()` des Core-Ingesters benutzt.

Schema, dtype und Spaltenzahl waren identisch — kein Import-, Lade- oder Typfehler. Sechs vorhandene Konsumenten filtern hart auf `transaction_type == "P"` (`fable_exploration/experiment/engine.py`, `insider_congress_recheck.py`, `mill/insider_buy_mill.py`, `mill/insider_timing_mill.py`, zwei Scratch-Proben). Jeder haette auf dem neuen Bestand **null Zeilen** geliefert.

Verschaerfend: der Modul-Docstring behauptete an genau dieser Stelle woertlich „Das ist dieselbe Konvention wie im bestehenden Ingester; sie hier zu lockern wuerde zwei Wahrheiten erzeugen." Die Absicht war formuliert, die Umsetzung widersprach ihr.

**Warum falsch:** Ein leeres Filterergebnis ist im Research nicht von einem echten Null-Befund zu unterscheiden. Der Fehler bleibt unsichtbar, bis jemand eine Nullkante als Ergebnis meldet — und dann sieht er aus wie eine Erkenntnis. Rule 50 („keine zweite Wahrheit") wird ueblicherweise als Schutz vor divergierenden IMPLEMENTIERUNGEN gelesen; hier war die Implementierung getrennt und die WERTEMENGE das Problem.

**Wie vermeiden:**
1. **Uebernimmt ein neuer Bestand einen bestehenden Spaltennamen, die Klassifikationsfunktion IMPORTIEREN statt nachzubauen.** Ein Inline-`map` ist genau dann verlockend, wenn er trivial aussieht.
2. **Bei bewusst anderer Semantik den SPALTENNAMEN aendern, nicht die Werte.** Ein `transaction_side` neben `transaction_type` bricht laut, ein anderer Wert bricht still.
3. **Fuer jede enum-artige Spalte ein Test, der die Wertemenge gegen den bestehenden Bestand vergleicht** — nicht gegen eine im Test wiederholte Literalliste.
4. **Eine Konventionsbehauptung im Docstring ist eine Zusicherung und braucht eine Assertion.** Wer sie hinschreibt, ohne sie zu pruefen, erzeugt falsche Sicherheit statt Klarheit.
**Erkannt in:** Stage-2-Review (F-senior-1, BLOCKER) zu `research/mandat/pull_form4_dera.py`.
**Referenzen:** E-109, E-119, Rule 50.

## E-124 — One-to-many-Merge ohne Primaerschluessel macht den Fan-out irreversibel
**Datum:** 2026-08-05
**Kategorie:** pandas-pitfall / informationsverlust
**Was passierte:** Transaktionen (`NONDERIV_TRANS`) wurden per Left-Merge mit den Meldepflichtigen (`REPORTINGOWNER`) verbunden. Hat ein Filing N Insider, wird jede Transaktionszeile N-fach kopiert. Das war **bewusst so gewollt** — die Zahl der Insider je Titel ist genau das Cluster-Signal, um das es fachlich ging.

Nicht mitgezogen wurde der Primaerschluessel der Transaktionstabelle (`NONDERIV_TRANS_SK`). Damit war die Aufblaehung nicht mehr rueckgaengig zu machen. Storeweit ueber alle 81 Quartale gemessen (8.310.850 Zeilen gegen 7.086.284 verschiedene Transaktionen): **2,87 %** der Filings haben mehrere Owner (max 10), die Zeilenzahl liegt **17,3 %**, die Stueckzahlsumme **37,8 %** ueber der echten — und bei den KAUFzeilen, um die es fachlich geht, sind es **52,7 %** (1.278.080 Zeilen gegen 837.272 verschiedene Kauftransaktionen). Ein `drop_duplicates` ueber die Fachspalten (Accession, Datum, Code, Stueck, Preis) waere kein Ausweg — es kollabiert auch echte Mehrfachausfuehrungen und ueberkorrigiert auf 52,1 %.

**Nachtrag zur eigenen Messdisziplin:** Die erste Fassung dieses Eintrags nannte 21,7 % und 2 %. Beide Werte stammten aus **2006Q1 allein** und wurden storeweit generalisiert. Die Streuung ueber die Quartale ist gross (2015Q1: +69,5 %), die Verallgemeinerung war also nicht nur ungenau, sondern methodisch falsch — derselbe Fehler wie E-078 in klein: eine Stichprobe fuer die Grundgesamtheit genommen.

**Warum falsch:** Aus einer bewussten Entscheidung wurde ein irreversibler Informationsverlust. Der Zaehl-Anwendungsfall war korrekt bedient, der Summen-Anwendungsfall war ohne Neuziehen nicht mehr herstellbar — und beim naechsten Leser haette nichts darauf hingedeutet, weil Zeilenzahl und Schema unauffaellig sind. Bemerkenswert ist auch die Groessenordnung: 11 % mehr ZEILEN waren 22 % mehr VOLUMEN, weil die Filings mit vielen Insidern die grossen sind.

**Wie vermeiden:**
1. **Vor jedem one-to-many-Merge den Primaerschluessel der „einen" Seite in die Spaltenauswahl aufnehmen** — auch wenn er fuer den aktuellen Zweck nutzlos wirkt. Er kostet eine Spalte und rettet den Bestand.
2. **Merge-Ergebnisse mit `validate=` oder einem Zeilenzahl-Assert absichern.** Ein unbeabsichtigter Fan-out sieht sonst wie mehr Daten aus.
3. **Aufblaehung immer auf der WERT-Summe messen, nicht auf der Zeilenzahl** — hier waren 17 % mehr Zeilen 38 % mehr Volumen, weil die Filings mit vielen Insidern die grossen sind.
4. **Ueber den GANZEN Bestand messen, nicht ueber die erste Partition.** Eine Quartalszahl als Storezahl auszugeben ist eine Verallgemeinerung ohne Beleg.
5. **Wenn Duplizierung gewollt ist, gehoert die Entblaehungsanleitung ins Manifest** — welcher Schluessel, fuer welche Kennzahl. Ein Kommentar im Code erreicht den Datenkonsumenten nicht.
**Erkannt in:** Stage-2-Review (F-senior-2, BLOCKER) zu `research/mandat/pull_form4_dera.py`.
**Referenzen:** E-116, E-122.

## E-125 — Zahl nur in der Commit-Message, kein erzeugender Code im Repo
**Datum:** 2026-08-05
**Kategorie:** test-anti-pattern / nicht-reproduzierbar
**Was passierte:** „Abdeckung des handelbaren Ziel-Universums 723 -> 6.146 von 8.876 (69,2 %)" stand als Ergebnis in einer Commit-Message. Ein `grep` nach der Zahl im Repo fand keinen erzeugenden Code; die Zwischendatei lag unter dem gitignorten `research/mandat/data/`.

Inhaltlich war die Zahl zusaetzlich in BEIDE Richtungen unbeschraenkt, weil sie ein roher Ticker-String-Vergleich war: zu niedrig, weil das Preispanel Vendor-Suffixe fuehrt (`AAC_old2`), die im SEC-Bestand nie so heissen; zu hoch, weil recycelte Ticker falsch matchen — Panel-Ticker X (Firma A, 2003–2008) gegen SEC-Ticker X (Firma B, 2015) zaehlte als abgedeckt. Das ist exakt Befund 7, und zwar in demselben Modul, das den ersten Pull-Ansatz mit genau dieser Begruendung verworfen hatte.

**Warum falsch:** Eine Zahl, die kein Skript reproduziert, ist keine Messung, sondern eine Erinnerung. Sie wird spaeter zitiert, obwohl niemand sie nachrechnen kann, und ihre Fehlerrichtung ist unbekannt. Commit-Messages sind append-only und werden nie korrigiert — eine falsche Zahl dort ueberlebt jede spaetere Berichtigung im Code.

**Wie vermeiden:**
1. **Jede Zahl, die in eine Commit-Message oder ein Befund-Dokument geht, muss von committetem Code erzeugt werden.** Sonst gehoert sie mit ihrer Fehlerrichtung gekennzeichnet oder gar nicht genannt.
2. **Liegt die Datengrundlage unter einem gitignorten Pfad, muss wenigstens das ERGEBNIS-Artefakt committet sein** (hier: `abdeckung_form4.json`).
3. **Bei jedem Abgleich ueber Ticker die Fehlerrichtung mitschreiben** — und wo vorhanden den stabilen Schluessel danebenstellen (hier: 17.134 Emittenten ueber CIK statt 21.860 Ticker).
**Erkannt in:** Stage-2-Review (F-senior-6) zu `research/mandat/pull_form4_dera.py`.
**Referenzen:** E-085, E-116, E-122, Befund 7.

## E-126 — Ein Artefakt zu committen erzeugte einen zweiten Fehlzustand, den der Guard nicht kannte
**Datum:** 2026-08-05
**Kategorie:** false-green / guard-unvollstaendig / ci-only
**Was passierte:** Testfixtures schuetzten sich mit dem naheliegenden Guard:

```python
if not p.exists():
    pytest.skip("Daten noch nicht gezogen")
return pd.read_parquet(p)
```

Lokal und in CI gruen — solange die Parquets **nicht** committet waren. Mit dem Commit der Artefakte kippte CI auf beiden Plattformen: `.gitattributes` schickt jede `*.parquet` durch Git LFS, und der CI-Checkout holt die LFS-Objekte nicht (kein `lfs: true`). Am Pfad der Daten liegt dort eine rund 130 Byte grosse Textdatei mit dem Kopf `version https://git-lfs.github.com/spec/v1`.

`Path.exists()` sagt True, der Guard laesst durch, `pd.read_parquet` platzt am Pointer. Der Fehler existiert ausschliesslich in CI und ist lokal nicht reproduzierbar, weil dort die echten Objekte liegen.

**Warum falsch:** Der Guard war fuer die Zustaende korrekt, die es zum Zeitpunkt seines Entwurfs gab — „Datei da" oder „Datei fehlt". Das **Committen des Artefakts** hat einen dritten Zustand eingefuehrt: „Datei da, aber nicht die Datei". Eine Aenderung an der Versionierung hat damit die Gueltigkeit einer Annahme im Testcode gebrochen, ohne den Testcode zu beruehren — genau die Art Kopplung, die man beim Review nicht sieht, weil beide Seiten fuer sich richtig aussehen.

Verschaerfend: `exists()` ist der Standardgriff und fuehlt sich vollstaendig an. Er prueft aber nur die Existenz eines Pfades, nicht die Brauchbarkeit des Inhalts.

**Wie vermeiden:**
1. **Bei einem Guard fragen, welche Zustaende die Datei annehmen kann** — nicht nur „da/nicht da". Bei LFS-verwalteten Pfaden gehoert „Zeigerdatei" dazu, bei Downloads „halb geschrieben", bei Netzlaufwerken „nicht lesbar".
2. **Wenn ein Artefakt neu unter Versionskontrolle kommt, `.gitattributes` pruefen.** Ein LFS-Filter aendert, was in CI ankommt, ohne dass ein Diff im Code das zeigt.
3. **Den Fehlzustand simulieren, statt auf den naechsten CI-Lauf zu warten**: eine echte Zeigerdatei an die Stelle des Artefakts legen, Tests laufen lassen, per Pruefsumme zuruecksetzen. Das kostet eine Minute und ersetzt einen roten Lauf.
4. **Faustregel: ein Guard, der aus einem harten Fehler ein `skip` machen soll, muss den Inhalt anfassen, nicht nur den Pfad.**
5. **Kein Erfolg melden, bevor CI durch ist.** Der rote Lauf kam nach der Meldung „alles gepusht" — die Meldung war zum Zeitpunkt des Schreibens ungedeckt.
**Erkannt in:** CI-Lauf zu `2d2ca2e6` (Lint & Test, ubuntu-latest und windows-latest, Schritt „Run fast tests").
**Referenzen:** E-103 (Fail-Open-Richtung), E-085, Rule 40.

## E-127 — Assertion auf die Repraesentation statt auf die Invariante: lokal gruen, CI rot
**Datum:** 2026-08-05
**Kategorie:** false-green / dependency-drift / test-fragil
**Was passierte:** Ein Test sollte sicherstellen, dass `available_at` **zeitzonenbehaftet** ist — die Invariante, die verhindert, dass ein `concat` mit dem Core-Bestand still eine object-Spalte erzeugt. Geschrieben wurde `assert str(df["available_at"].dtype) == "datetime64[ns, UTC]"`.

Lokal gruen, in CI rot auf beiden Plattformen: `AssertionError: assert 'datetime64[us, UTC]' == 'datetime64[ns, UTC]'`.

Ursache ist die Dependency-Drift aus Rule 40, hier im Test statt im Code: `ci.yml` installiert `pip install -e ".[dev]"`, und die Ranges in `pyproject.toml` sind offen — CI zieht **pandas 3.0.5**, dessen Standardaufloesung **Mikrosekunden** ist. Lokal pinnt `requirements.txt` **pandas 2.2.3**, also Nanosekunden.

Der Test prueft damit die **Aufloesung**, obwohl es um die **Zeitzone** ging. Die Aufloesung gleicht pandas beim `concat` an; naiv gegen tz-aware gleicht es NICHT an — genau das war der Punkt. Die Assertion war also nicht nur fragil, sie mass die falsche Eigenschaft.

Verschaerfend kam eine zweite Verdeckung dazu: der gruene Job `backend-ci.yml` installiert aus `requirements.txt` und laeuft nur `-m "fast and not slow"`. Die neue Testdatei hat **keinen Marker** und laeuft deshalb ausschliesslich im roten Job — mit der abweichenden Installation. Zwei Unterschiede fielen zusammen, und der gruene Nachbarjob vermittelte falsche Sicherheit.

**Warum falsch:** Eine Assertion auf einen `repr`- oder `str`-Wert testet die Darstellung eines Objekts, nicht seine Eigenschaft. Darstellungen sind Implementierungsdetails der Bibliothek und aendern sich zwischen Versionen; Eigenschaften nicht. In einem Repo mit dokumentierter Drift zwischen Pins und Ranges ist das kein theoretisches Risiko, sondern ein Termin.

**Wie vermeiden:**
1. **Die Eigenschaft pruefen, nicht ihren Text.** Hier: `df[c].dt.tz is not None` und `str(tz) == "UTC"` statt eines dtype-Strings. Analog `is_numeric_dtype()` statt `== "float64"`.
2. **Vor dem Schreiben fragen, welche Eigenschaft die Invariante wirklich ist.** Wer die Frage beantwortet, schreibt die fragile Assertion gar nicht erst.
3. **Neue Testdateien laufen in einem anderen Job als gedacht**, wenn sie keinen Marker tragen. Marker bewusst setzen — oder wissen, welcher Job die Datei aufsammelt und mit welcher Installation.
4. **Bei Verdacht auf Drift die CI-Installation nachbauen** (`python -m venv` plus `pip install -e ".[dev]"` im Klon) statt gegen die lokale zu testen. Das kostet Minuten und ersetzt Rateschleifen: hier zeigte es pandas 3.0.5 gegen 2.2.3 sofort.
5. **Ein gruener Nachbarjob ist kein Beleg.** Unterschiedliche Jobs koennen unterschiedliche Testmengen mit unterschiedlichen Abhaengigkeiten fahren.
**Erkannt in:** CI-Lauf zu `64266967`, reproduziert in einer nachgebauten CI-Umgebung.
**Referenzen:** E-126 (derselbe Lauf, anderer Mechanismus), E-103, Rule 40.

## E-128 — Symptom im CI geflickt statt die Ursache entfernt; dabei einen weichen Fehler gegen einen harten getauscht
**Datum:** 2026-08-05
**Kategorie:** false-fix / abhaengigkeit-eingebaut
**Was passierte:** Nachdem LFS-Zeigerdateien Tests in CI zum Ueberspringen brachten (E-126), lautete meine Empfehlung: `lfs: true` in den Checkout von `ci.yml`. Ich hatte die Seiteneffekte sogar geprueft — 3 Objekte, 1,0 MiB, zwei Matrix-Jobs, also ~2 MB Traffic pro Lauf — und daraus "unbedenklich" geschlossen.

Uebersehen hatte ich das Entscheidende: **das Repo ist oeffentlich.** GitHub bucht LFS-Bandbreite dem Eigentuemer, auch bei anonymen Clones, und sperrt LFS bei Erschoepfung repo-weit statt zu drosseln. Damit haette `lfs: true` aus einem **weichen** Fehler (Pointer -> Test skippt) einen **harten** gemacht: der Checkout-Step scheitert, beide Jobs sind rot, bevor ein Test laeuft — ausgeloest durch Traffic, den das Repo nicht kontrolliert. Dazu: `actions/checkout@v4` cached keine LFS-Objekte, die Regel `*.parquet filter=lfs` gilt repo-weit, der Traffic waere still mitgewachsen.

Die billigere Loesung lag eine Ebene tiefer: die drei Dateien aus LFS **herausnehmen**. 0,97 MiB — dafuer zahlt sich LFS nirgends aus. Das wirkt fuer jeden Workflow und jeden Clone, braucht den permissions-geschuetzten Pfad `.github/workflows/` gar nicht und beseitigt die Fehlerklasse an der Quelle, statt sie in jedem Leser einzeln abzufangen.

**Zwei Fallen dabei, beide gemessen statt vermutet:**
1. **`binary` setzt `filter=lfs` NICHT zurueck.** Das Makro expandiert zu `-diff -merge -text`. Nach `git rm --cached` + `git add` lagen weiterhin 131-Byte-Zeiger im Index. Noetig ist explizit `-filter -diff -merge binary`.
2. **`binary` ist trotzdem Pflicht.** Ohne die text-Ausnahme kann `core.autocrlf` auf dem windows-latest-Runner CRLF in die Binaerdatei schreiben — Korruption auf genau einem der beiden Matrix-Jobs.

**Warum falsch:** Ein Fix, der CI an eine externe, erschoepfbare und von Dritten beeinflussbare Ressource koppelt, ist teurer als er aussieht — und der Preis faellt zu einem Zeitpunkt an, den niemand waehlt. Meine Seiteneffekt-Pruefung war nicht falsch, sondern zu eng: ich hatte das Volumen gemessen und die Sichtbarkeit des Repos nicht.

**Wie vermeiden:**
1. **Bei jedem CI-Fix fragen: koppele ich hier an eine Ressource, die ausgehen kann?** Quoten, Rate-Limits, externe Endpunkte. Wenn ja: was passiert im Erschoepfungsfall — weich oder hart?
2. **Ursache vor Symptom.** Bevor eine Pipeline etwas nachladen soll, pruefen, ob das Artefakt ueberhaupt dort liegen muss, wo es liegt.
3. **Repo-Sichtbarkeit ist ein Sicherheits- UND ein Kostenparameter.** Bei oeffentlichen Repos ist jeder verbrauchbare Kontingentposten von aussen ausloesbar.
4. **`.gitattributes`-Aenderungen am Index verifizieren, nicht am Arbeitsverzeichnis:** `git cat-file -s $(git rev-parse :pfad)` und `git check-attr filter diff merge text -- pfad`. Ein Zeiger im Index sieht im Working Tree wie die echte Datei aus.
**Erkannt in:** Stage-1-Review (`ci-debugger`, F-ci-1/2/4/5) zur Frage, ob `lfs: true` die richtige Loesung ist. Meine Empfehlung war die teuerste der drei moeglichen.
**Referenzen:** E-126 (der Ausloeser), E-127 (derselbe CI-Lauf), Rule 40.

## E-129 — CLI-Flag deklariert, nie ausgewertet: der Schutzmechanismus war eine Attrappe
**Datum:** 2026-08-05
**Kategorie:** wiring-gap / false-green
**Was passierte:** `h087_trendfilter_lang.py` deklarierte `--regen` mit dem Hilfetext „Artefakt neu erzeugen, OHNE den Trial-Zaehler zu erhoehen (E-090)". Das Ergebnis von `parse_args` wurde an `args` gebunden und **nie gelesen**; `TrialCounter().increment(1)` lief unbedingt.

Folge: der Regenerationslauf zaehlte H-087 ein zweites Mal (`trials.json` 1566 -> 1567), waehrend die Commit-Message desselben Commits „KEIN zusaetzlicher Trial" behauptete. Die Schwestern-Skripte (p13, p13b, p12*) verdrahten das Flag korrekt — dieses war der einzige Ausreisser.

**Warum falsch:** Ein Flag, das im Hilfetext eine Garantie gibt und sie nicht einloest, ist schlimmer als kein Flag: es erzeugt genau die falsche Sicherheit, gegen die der zitierte Eintrag geschrieben wurde. Es faellt in keinem Test auf, weil die **Abwesenheit** von Verhalten keine Fehlermeldung erzeugt, und die Behauptung in der Commit-Message wirkt wie eine Verifikation. Besonders bitter: es war E-090s eigenes Vermeidungsmuster, das hier zur Attrappe wurde.

Linter helfen nicht: `args` sieht formal benutzt aus (es wird zugewiesen), also greift weder `F841` noch eine ARG-Regel.

**Wie vermeiden:**
1. **Nach jedem `parse_args` abzaehlen:** `grep -c 'args\.' datei.py` gegen die Zahl der `add_argument`. Steht links 0, ist das Flag tot.
2. **Jedes Flag, das eine Buchhaltung beeinflusst, braucht eine Ausgabezeile** („[REGEN] Zaehler unveraendert bei N") — dann ist die Wirkung im Laufprotokoll sichtbar statt nur im Quelltext behauptet.
3. **Behauptungen ueber Zaehlerstaende gegen die tatsaechliche Diff-Zeile pruefen**, nicht gegen die Absicht. `git diff -- trials.json` kostet eine Sekunde.
4. **Append-only-Zaehler nicht zurueckschreiben.** Der falsch gezaehlte Trial bleibt stehen und wird offengelegt — eine stille Korrektur waere ein zweiter Fehler.
**Erkannt in:** Stage-2-Review (F-senior-1, BLOCKER) zu `h087_trendfilter_lang.py`.
**Referenzen:** E-090 (dessen Massnahme hier zur Attrappe wurde), E-085, E-119.

## E-130 — Cash zu 0 % im Timing-Backtest: einseitiger Nachteil, die Zinsreihe lag eine Spalte daneben
**Datum:** 2026-08-05
**Kategorie:** logic-error / konvention-unbenannt
**Was passierte:** H-087 vergleicht ueber 1926–2026 einen Trendfilter gegen dieselbe Reihe ohne Filter. Der Filter ist an **28 % der Auswertungstermine in Cash** und bekommt dafuer 0 %. Die Spalte `rf` liegt in genau der Parquet-Datei, aus der die Kursreihe gelesen wird — `pd.read_parquet(p)["index_crsp_vw"]` verwirft sie. `Portfolio` verzinst Cash nirgends (nur Kredit).

Ueber den Zeitraum ist das nicht neutral: RF-Median 2,55 % p. a., Maximum 14,66 %, und die Hochzinsphasen (1974er, 1987er) fallen ausgerechnet in die krisenfreien Bloecke, gegen die der Befund argumentiert. Gemessen: mit Cash zum risikolosen Satz schrumpft der Rueckstand von −97,6 auf −55,9 pp.

**Warum falsch:** Der Nachteil trifft **nur den Kandidaten** — der Benchmark ist immer investiert. Er ist ausserdem **epochenabhaengig**: im alten Suchfenster 1995–2016 war er ein Bruchteil dessen, was er in der neuen Reihe ist. Damit ist die Konvention im NEUEN Test teurer als im alten und arbeitet systematisch in Richtung des erwarteten negativen Verdikts. Das Verdikt hielt hier in allen drei gerechneten Konventionen — aber es hielt, ohne dass die Frage gestellt worden war.

**Wie vermeiden:**
1. **Jede Strategie mit Cash-Zustand braucht eine explizite Cash-Verzinsungs-Annahme im Artefakt** — auch wenn sie 0 % lautet. Unbenannt ist sie kein Modell, sondern ein Versehen.
2. **Beim Anlegen eines Backtests auf einer neuen Quelle die VOLLE Spaltenliste ansehen**, bevor eine Serie herausgegriffen wird. Was nicht gelesen wird, wird auch nicht als Luecke sichtbar.
3. **Bei Vergleichen ueber Zinsepochen pruefen, ob eine Konvention in verschiedenen Epochen verschieden teuer ist.** Nur dann ist es „dieselbe Konvention".
4. **Loesung ohne Engine-Eingriff:** auf der UEBERSCHUSSREIHE rechnen. Dort ist „nicht investiert" per Definition „zum risikolosen Satz angelegt", beide Seiten liegen im selben Raum, und keine gemeinsame Komponente muss geaendert werden (Rule 50).
**Erkannt in:** Stage-2-Review (F-senior-2) — nachdem ich denselben Verdacht selbst geaeussert, aber nicht geprueft hatte.
**Referenzen:** E-079 (Massstab-Asymmetrie), E-085.

## E-131 — Vorregistriertes Entscheidungskriterium im Artefakt nicht nachpruefbar gemacht
**Datum:** 2026-08-05
**Kategorie:** test-anti-pattern / registrierung-wirkungslos
**Was passierte:** Die Registrierung (Welle 47) legte vorab fest: „Massgeblich sind die 4 disjunkten Bloecke, nicht die 338 Fenster. Ein Ergebnis, das nur in einem Block auftritt, ist eine Episode, kein Mechanismus, und wird so berichtet."

Das Artefakt berichtete die **Blockzahl** (4) und aggregierte Kennzahlen ueber alle Fenster — aber **kein Ergebnis je Block**. Aus dem Artefakt heraus liess sich das vorregistrierte Kriterium nicht anwenden. Nachgetragen zeigte sich: 0 gewonnene Fenster in **allen vier** Bloecken (241 / 40 / 41 / 16) — der Befund war also staerker, als er berichtet wurde.

**Warum falsch:** Eine Vorregistrierung schuetzt nur, solange ihr Kriterium hinterher **ueberpruefbar** ist. Ist es das nicht, bleibt sie eine Absichtserklaerung, und die Auswertung faellt still auf die aggregierte Zahl zurueck, die sie ausdruecklich verworfen hatte. Der Schaden ist symmetrisch: hier war das Ergebnis besser als berichtet, im naechsten Fall waere es schlechter — und niemand koennte es sehen.

**Wie vermeiden:**
1. **Beim Schreiben des Laufskripts die Pass/Fail-Saetze der Registrierung durchgehen** und je Satz fragen: liefert das Artefakt das Feld, das ihn entscheidet?
2. **Gruppierungskriterien muessen die Kennzahlen MITFUEHREN, nicht nur gezaehlt werden.** Eine Blockzahl ohne Ergebnis je Block ist eine Zahl ueber die Daten, keine ueber den Befund.
3. **Faustregel:** steht in der Registrierung ein „massgeblich ist X", muss X ein Schluessel im Ergebnis-JSON sein.
**Erkannt in:** Stage-2-Review (F-senior-3) zu `h087_trendfilter_lang.py` und `registry.md` Welle 47.
**Referenzen:** E-078, E-085, E-116.

## E-132 — Der Guard schloss einen Ausfallpfad und liess die gleichgerichteten Nachbarn offen
**Datum:** 2026-08-08
**Kategorie:** silent-except / fail-open-nachbarn
**Was passierte:** Im Odd-Lot-Scanner wurde ein still beruhigender Ausfallmodus geschlossen (HTTP 200 ohne `hits` → SystemExit statt „[OK] 0 Treffer"). In derselben Funktion blieben zwei Pfade mit identischer Fail-Richtung offen: die 100er-Deckelung der EDGAR-Antwort (`hits.total.value` wurde nie gelesen — live gemessen: 273 Treffer vorhanden, 100 geliefert, Ausgabe waere „[OK] 100 Treffer") und das ungedeckte Zeitfenster bei einem Lauf-Aussetzer laenger als `RUECKBLICK_TAGE` (`stand` wurde geschrieben, aber nie gelesen).

**Warum falsch:** Ein Guard gegen einen konkreten Ausfall erzeugt den Eindruck, die Fehlerklasse sei erledigt. Erledigt ist nur der Pfad, den der ausloesende Vorfall beleuchtet hat. Ausfallmodi, die auf „unauffaellig" fallen, treten selten einzeln auf — sie teilen die Ursache (unvalidierte Fremdantwort, implizite Betriebsannahme).

**Wie vermeiden:**
1. Beim Schliessen eines Fail-Open-Pfades die **uebrigen Ausgaenge derselben Funktion durchzaehlen** und je Ausgang fragen: welche Ausgabe entsteht bei Stoerung?
2. Jede Zaehlgroesse gegen die **vom Anbieter gemeldete Gesamtzahl** pruefen (total vs. geliefert).
3. Jedes geschriebene Zustandsfeld **entweder auch lesen oder nicht schreiben**.
**Erkannt in:** Stage-2-Review (F-senior-1/2) zu `scripts/ops/oddlot_forward_scanner.py`.
**Referenzen:** E-103.

## E-133 — Erzaehlung committet, Zaehlerstand nicht: die Zahl ist im Repo nicht herleitbar
**Datum:** 2026-08-08
**Kategorie:** wiring-gap / zaehler-uncommittet
**Was passierte:** Commit `b0bc8b8c` schrieb „Welle 48b … N=3.539" in Registry und Commit-Message, liess aber `research/mandat2/trials.json` (die erzeugende Zaehlerdatei) uncommittet. Ein Folge-Commit uebernahm 3.539 in den Ledger als Autoritaetsaussage. Im committeten Stand stand der Zaehler auf 3.533.

**Warum falsch:** Bei einem monoton wachsenden Zaehler ist der **Dateizustand** die Wahrheit, nicht der Prosatext. Bleibt er uncommittet, divergieren beide lautlos, und ein Verwerfen des Working Tree vergibt bereits benutzte Nummern erneut — dieselbe Kollisionsklasse (H-086), gegen die der betroffene Ledger-Nachtrag gerade die Regel festschrieb.

**Wie vermeiden:**
1. Zaehler-, Sequenz- und Manifestdateien gehoeren **in denselben Commit** wie jede daraus zitierte Zahl.
2. Vor dem Committen einer abgeleiteten Zahl den Zaehlerstand **aus dem INDEX lesen** (`git show HEAD:pfad`), nicht aus dem Working Tree.
**Erkannt in:** Stage-2-Review (F-senior-4) zu `research/ledger.md` + `research/mandat2/trials.json`.
**Referenzen:** E-090, E-125, E-129.

## E-134 — Entwarnung im Freigabe-Dokument, ohne das zitierte Audit-Artefakt zu oeffnen
**Datum:** 2026-08-08
**Kategorie:** false-evidence / erinnerung-als-befund
**Was passierte:** Der Pilot-Umstellungsplan erklaerte 12 Reconcile-FAILs als „unterhalb des Schwellen-Gates, bekanntes Verhalten". `output/ops/reconciliation_audit.jsonl` enthaelt 431 fail-Zeilen mit Werten AUF oder UEBER den Fail-Schwellen (cash_diff_bps bis 526,3 gegen Gate 25,0; max_qty_diff 15,0 gegen 10,0; `reconciliation.py:66-69`); die Zahl 12 kommt darin nicht vor, und der zitierte saubere SLO-Eval war drei Tage aelter als das Dokument.

**Warum falsch:** Die Formulierung stammte aus einer **korrekten Projekt-Erinnerung** („Reconcile-CLI-FAILED ist nicht das Halt-Kriterium", Juli-Vorfall) und wurde auf einen neuen, ungeprueften Sachverhalt uebertragen. Ein plausibles Muster aus dem Gedaechtnis liest sich wie ein Befund, ist aber keiner — und in einem Dokument, das jemand unterschreiben soll, wird aus der Erinnerung eine Zusicherung. Genau die Entwarnung ohne Evidenz, die CLAUDE.md („Keine falsche Sicherheit") verbietet.

**Wie vermeiden:**
1. Jede Entwarnung ueber ein Gate nennt **drei Dinge aus der Datei**: Zeitstempel der juengsten Auswertung, gemessenen Wert, geltenden Schwellwert mit Fundstelle im Code.
2. Faellt eines davon aus, wird die Lage als **offen** beschrieben, nicht als „bekannt".
3. Erinnerungen (Memory, fruehere Sessions) sind Hypothesen ueber den Ist-Zustand — vor dem Zitieren in Freigabe-Dokumenten **gegen das aktuelle Artefakt pruefen**.
**Erkannt in:** Stage-2-Review (F-senior-3, BLOCKER) zu `docs/PILOT_UMSTELLUNG_PLAN.md`.
**Referenzen:** E-085, E-122.

## E-135 — Feature auf einem Mechanismus gebaut, ohne zu pruefen, ob er je gefeuert hat
**Datum:** 2026-08-09
**Kategorie:** wiring-gap / inerter-schutz
**Was passierte:** Die DD-Treppe -10/-15/-20 galt seit Monaten als aktiver Pilot-Schutz (policy.yaml, Statusdokumente, Review-Sprache). Erst beim Umstellen des aeusseren Halts fiel auf: `_evaluate_auto_dd_kill_switch` liest seine Equity per getattr, und im Paper-Pfad setzte NIEMAND `ctx.current_equity` — dd war immer None, die Treppe hat nie ausgeloest. Zusaetzlich haette die damalige soft-Stufe den persistenten Kill-Switch aktiviert, also im Broker-Pfad einen Vollstopp statt einer 50-%-Drossel bedeutet — unbemerkt, weil der Mechanismus inert war.

**Warum falsch:** Ein konfigurierter Schwellwert ist kein Beweis, dass der Codepfad erreichbar ist. getattr-Leser mit None-Fallback verschlucken fehlende Verdrahtung geraeuschlos: der Schutz „existiert", produziert aber nie ein Log, nie ein meta-Feld, nie einen Testfall — und niemand vermisst etwas.

**Wie vermeiden:**
1. Bei jeder Aussage „Schutz X ist aktiv": den **Produzenten der Eingangsgroesse benennen** (grep auf Zuweisung, nicht auf Lesen).
2. Mindestens eine **im Normalbetrieb sichtbare Spur** verlangen (Log der ausgewerteten Groesse oder meta-Feld auch im Nicht-Ausloesefall).
3. `getattr(ctx, X, None)` in Risk-Pfaden ist ein Verdachtsmoment, kein Komfort.
**Erkannt in:** DD-Umbau 2026-08-09 (`trading_cycle_shared.py`, `paper_runner.py`).
**Referenzen:** E-059, E-121.

## E-136 — Der Leerlauf-Fallback verkehrte die neue Drossel in Vollgas
**Datum:** 2026-08-09
**Kategorie:** logic-error / fail-open-fallback
**Was passierte:** `paper_runner.py`: `orders_for_fills = orders_filtered if not orders_filtered.empty else orders`. Der Ausdruck deutete „leer" als „nicht befuellt" und fiel auf den ungefilterten Batch zurueck. Jede Gate-Stufe, die orders_filtered auf null filtert (VaR-Gate, Circuit Breaker, min_notional, fail-closed-excepts, neu: DD-Drossel + min_notional), lieferte damit das GEGENTEIL ihrer Absicht — die neue qty-Drossel machte den Fall wahrscheinlicher, je staerker sie griff. Gefunden im Stage-2-Review, einen Commit bevor die Drossel den Pfad erstmals real haette treffen koennen.

**Warum falsch:** „Leer" und „nicht gesetzt" sind zwei verschiedene Zustaende. Ein Guard, der bei totaler Filterung auf die Rohmenge zurueckfaellt, ist fail-open in genau dem Moment, fuer den er gebaut wurde. Verschaerfend: `meta["risk_gate_error"]` wird an fuenf Stellen gesetzt und repo-weit von niemandem gelesen — das Blocken hing allein am Leeren des DataFrames, das der Fallback rueckgaengig machte.

**Wie vermeiden:**
1. Nicht-Vorhandensein per **None** modellieren, nie per empty.
2. Beim Einbau eines neuen Filters/Skalierers den **Weg der gefilterten Menge bis zur Ausfuehrung abgehen** und den Grenzfall „filtert alles weg" mit einem Test fixieren.
3. Fail-closed-Flags brauchen einen Konsumenten, sonst sind sie Dekoration.
**Erkannt in:** Stage-2-Review (F-senior-1) zu `ops/paper_runner.py` + `pipeline/_tc_risk.py`.
**Referenzen:** E-103, E-132.

## E-137 — Eine Artefakt-Spalte, zwei Produzenten, zwei Bedeutungen
**Datum:** 2026-08-09
**Kategorie:** wiring-gap / artefakt-semantik
**Was passierte:** `run_index.csv` fuehrt `final_equity`. `unified_paper_engine` schrieb dort die Post-Fill-Equity; der Pipeline-Pfad fuellte dieselbe Spalte im ersten Wurf mit der Start-of-Cycle-Equity, weil dort erstmals ueberhaupt ein Wert verfuegbar war. Der Code-Kommentar war ehrlich, das Artefakt nicht: eine Zeile war ohne Kenntnis des Schreibers nicht interpretierbar. Fix: eigener Schluessel `equity_start_of_cycle`, `final_equity` bleibt im Pipeline-Pfad leer.

**Warum falsch:** Eine frei gewordene Spalte ist keine Einladung, sie mit der naechstbesten verfuegbaren Groesse zu befuellen. Ehrlichkeit im Kommentar ersetzt keine Ehrlichkeit im Feldnamen — der Kommentar reist nicht mit dem Artefakt.

**Wie vermeiden:**
1. Vor dem Befuellen eines bestehenden Artefaktfelds die **anderen Produzenten greppen**.
2. Weicht die Semantik ab: **neues Feld, nicht Umdeutung.**
**Erkannt in:** Stage-2-Review (F-senior-6) zu `pipeline/_tc_execution.py` vs. `execution/unified_paper_engine.py`.
**Referenzen:** E-122, E-133.

## E-138 — Schemafixe Senke verwirft den neuen Schluessel still (korrigiert E-137 „Wie vermeiden")
**Datum:** 2026-08-09
**Kategorie:** wiring-gap / schema-fixe-senke
**Was passierte:** Die E-137-Remediation gab der Start-of-Cycle-Equity korrekt einen eigenen Schluessel `equity_start_of_cycle` — schrieb ihn aber in das metrics-Dict von `append_run_index`. `run_index.csv` hat mit `INDEX_COLUMNS` ein fixes Schema und mappt nur bekannte Keys; der neue Key wurde ohne Warnung fallengelassen. Doppelt maskiert: der Aufruf liegt in `except Exception: log.debug(...)`. Sichtbar wurde es nur, weil ein bestehender Test auf der alten (falschen) Semantik bestand und CI auf `bb19531d` rot lief. Fix: Wert liegt jetzt in `run_kpis.json` (schemafrei), der Run-Index bekommt vom Pipeline-Pfad gar nichts, und der Pipeline-Level-Test assertet den Wert in der NEUEN Senke.

**Warum falsch:** „Neues Feld statt Umdeutung" (E-137) setzt voraus, dass die Senke neue Felder aufnehmen KANN. CSV/Parquet/Manifest mit Spaltenkonstante nehmen sie nicht auf und melden das nicht — der Schreibvorgang meldet Erfolg, der Wert existiert nicht.

**Wie vermeiden:**
1. Vor dem Schreiben eines neuen Keys pruefen, ob die Senke eine **Spalten-/Feldkonstante** fuehrt (`INDEX_COLUMNS`, dtype-Schema, `schema_version`). Wenn ja: Schema bewusst erweitern oder eine schemafreie Senke (JSON) waehlen.
2. Neuen Artefakt-Wert **dort assertieren, wo der Produktionspfad ihn erzeugt** (Pipeline-/Runner-Level), nicht nur im Unit-Test des Writers.
3. Wandert ein Wert von Senke A nach Senke B, **wandert der Wiring-Test mit** — ein Test, der nur noch Abwesenheit in A prueft, laesst B ungeschuetzt.
**Erkannt in:** CI-Rot auf `bb19531d`; Kompakt-Review (F-senior-1/2, E-NEW-1) zu `_tc_execution.py`/`run_index.py`/`kpi_artifacts.py`.
**Referenzen:** E-137 (dessen „Wie vermeiden #2" ist hiermit um die Schema-Vorpruefung ergaenzt), E-047, E-136.

## E-139 — Test-Isolation gegen Betriebsstores wird immer nur um die Datei erweitert, die gerade zugebissen hat
**Datum:** 2026-08-09
**Kategorie:** test-anti-pattern / reaktive-isolation
**Was passierte:** Die autouse-Fixture `_isolate_operational_stores` dokumentiert in ihrem eigenen Docstring drei Vorfaelle (intent_store 2026-07-24, order_lifecycle 2026-07-22, crisis_alpha_state) und wurde am 2026-08-09 zum vierten Mal erweitert — nachdem ein Testlauf den REALEN persistenten Kill-Switch scharfgeschaltet hatte (auto_dd_kill, dd=-90 % aus Test-Equity; entschaerfbar nur per OPERATOR_KILL_TOKEN, Pilot waere geblockt gewesen). Noch im selben Review wurden zwei weitere offene Kanaele derselben Klasse belegt: `output/ops/reconciliation_audit.jsonl` bekam synthetische severity=fail-Eintraege aus einem Verifikationslauf, und `tests/hooks/test_stop_hook.py` konsumierte den echten One-Shot-Skip `.claude/.review_skip` und schrieb ins echte Skip-Audit-Log.

**Warum falsch:** Jede Erweiterung war korrekt und jede war zu klein. Der Fehler ist nicht die einzelne fehlende Zeile, sondern die reaktiv gefuehrte Liste: isoliert wird, was schon einmal geblutet hat. Die Eskalation ist qualitativ — von stoerendem Residuum zu Testlaeufen, die Sicherheitsmechanismen SCHARFSCHALTEN und Audit-Trails verunreinigen, auf die sich Freigabeentscheidungen stuetzen. Ein Audit-Eintrag, dem man Testherkunft nicht ansieht, entwertet das ganze Log.

**Wie vermeiden:**
1. Die Liste einmal vollstaendig ERZEUGEN statt fortschreiben: alle Default-Schreibpfade unter `output/ops` und `.claude` enumerieren (grep auf `Path("output/ops...` und die `_*_path()`-Resolver) und jeden isolieren oder mit Begruendung als bewusst-real markieren.
2. Harte Grenze: **Ein Test darf keinen Zustand erzeugen, dessen Aufraeumen ein Operator-Secret braucht.** Ab da ist Isolation nicht optional.
3. Regressionsschutz statt Gedaechtnis: ein Wachhund-Test, der `output/ops` und `.claude` vor/nach einem Suite-Lauf per mtime+size vergleicht, faengt die naechste Wiederholung selbst.
**Erkannt in:** Stage-2-Review 2026-08-09 (`tests/conftest.py`, `accounting/reconciliation.py`, `tests/hooks/test_stop_hook.py`).
**Referenzen:** E-135; conftest-Docstring-Historie 2026-07-22/24.

## E-140 — Der Registrierungstext im Forschungsartefakt beschreibt einen anderen Lauf als den gelaufenen
**Datum:** 2026-08-09
**Kategorie:** false-evidence / registrierung-drift
**Was passierte:** `check_15m_breakout.json` archivierte als „registriert" den Docstring-Kopf mit „EODHD 5m -> 15m resampled" und „-USD.CC"-Paaren. Gelaufen ist native Binance-15m auf USDT-Paaren; der (preisblinde, legitime) Quellenwechsel stand nur als Code-KOMMENTAR unterhalb des Docstrings — und reiste damit nicht ins Artefakt. Verschaerfend schnitt in beiden Krypto-Skripten ein `__doc__.split(...)` genau die Teile weg, die man spaeter braucht: einmal die Signaldefinition, einmal den Asset-Klassen-Vorbehalt. Beide Laeufe waren bereits im Trial-Zaehler gebucht.

**Warum falsch:** Die Vorabregistrierung ist der einzige Schutz gegen nachtraegliches Zurechtruecken — sie ist nur so viel wert, wie sie den tatsaechlichen Lauf beschreibt. Ein Docstring-Split ist eine stille Schere: er wirkt wie Kuratierung, entfernt aber genau den Vorbehalt bzw. die Regel, ohne die das Verdikt spaeter breiter zitiert wird, als es traegt. Ein Kommentar reist nicht mit dem Artefakt.

**Wie vermeiden:**
1. Aenderungen an Quelle, Frequenz, Symbolen, Zeitraum gehoeren **in den Docstring** (das, was gebucht wird), nicht in einen Kommentar daneben.
2. Kein `__doc__.split()` fuer Artefaktfelder — ganzen Docstring schreiben oder explizit benannte Felder (registrierung/signal/vorbehalte).
3. Vor dem Buchen das erzeugte JSON einmal LESEN und gegen die tatsaechlich verwendeten Konstanten pruefen (Symbolliste und Intervall stehen im selben File).
**Erkannt in:** Stage-2-Review (F-senior-1/2) zu `research/krypto/check_*_userstrategie.py`; Fix ohne Neubuchung via `--regen`, Verdicts unveraendert FAIL.
**Referenzen:** E-085, E-090, E-131.

## E-141 — itertuples-Attributzugriff auf selbstgebaute Unterstrich-Hilfsspalten wirft IMMER
**Datum:** 2026-08-11
**Kategorie:** pandas-pitfall / itertuples-underscore
**Was passierte:** In `_sp_apply_trailing_stops` wurden Hilfsspalten `_sym`/`_entry` angelegt und per `row._sym` gelesen. `DataFrame.itertuples` benennt jeden Bezeichner mit fuehrendem Unterstrich positional um (`_1`, `_2`, ...). `row._sym` warf in jeder Runde AttributeError; der umschliessende except-Pfad machte daraus einen DEGRADED-Log — ein konfiguriertes Risk-Overlay lief seit Einfuehrung NIE (beide pandas-Versionen, `_fields=('_0','_1')` reproduziert).

**Warum falsch:** Der Unterstrich-Praefix ist im Repo die uebliche Konvention fuer temporaere Hilfsspalten — genau diese Konvention ist mit itertuples-Attributzugriff unvereinbar. Der Fehler ist weder versions- noch datenabhaengig: er trifft bei jedem Aufruf, in jedem Env.

**Wie vermeiden:**
1. Hilfsspalten, die per itertuples gelesen werden, NIE mit `_` praefixen (`sym_norm` statt `_sym`).
2. Besser gleich `zip(df[a], df[b])` statt itertuples-Attributzugriff.
3. Review-grep: `itertuples` im Diff -> pruefen, ob eine gelesene Spalte mit `_` beginnt (Repo-Sweep 2026-08-11: alle 68 uebrigen Aufrufe sauber).
**Erkannt in:** CI-Log-Fund nach gh-Auth (`pipeline/_tc_sizing.py`).
**Referenzen:** E-059, E-135.

## E-142 — Lautstaerke ohne Konsument ist Stille: der DEGRADED-Log lief monatelang ins Leere
**Datum:** 2026-08-11
**Kategorie:** wiring-gap / loudness-ohne-leser
**Was passierte:** Der E-141-Fehler war NICHT still — `_record_degraded_step` schrieb pro Runde ein WARN `[DEGRADED] trailing_stops skipped` plus `meta['degraded_steps']`, exakt die nach E-059 eingebaute Loudness. Gefunden wurde er trotzdem erst, als gh-Auth die CI-Logs erstmals lesbar machte.

**Warum falsch:** Ein WARN ohne Leser und ein meta-Feld ohne Gate sind operativ identisch mit einem debug-Log. Ein degraded_step, der in JEDER Runde erscheint, erzeugt nie ein Delta — genau deshalb bemerkt ihn niemand.

**Wie vermeiden:**
1. Fuer jedes `meta['degraded_steps']` einen benannten Konsumenten (QA-Gate, Report-Zeile, Alert).
2. Dauer-Degradation-Check: gleicher Step in N aufeinanderfolgenden Laeufen degraded = Bug, kein fail-soft.
3. Bei neu erschlossener Log-Sichtbarkeit zuerst die WIEDERHOLTEN Meldungen auswerten, nicht die auffaelligen.
**Erkannt in:** Kompakt-Review 2026-08-11.
**Referenzen:** E-059, E-135.

## E-143 — Exception behoben ist nicht Feature wirksam: der Konsument fehlte am anderen Ende
**Datum:** 2026-08-11
**Kategorie:** wiring-gap / ausgang-ohne-konsument
**Was passierte:** Nach dem itertuples-Fix lief das Trailing-Stops-Overlay durch und reduzierte `target_weight` (verifiziert 0.5 -> 0.125). Die Ordergenerierung liest aber ausschliesslich `target_qty` — die Reduktion erreichte keine einzige Order. Zusaetzlich ist der Trigger-Pfad ohne `prior_states`/`current_bar` strukturell unerreichbar (bars_since_entry immer 0 < grace_period).

**Warum falsch:** E-135 beschreibt fehlende Verdrahtung am EINGANG; das hier ist das Spiegelbild am AUSGANG. Ein behobener Stacktrace fuehlt sich wie ein reaktiviertes Feature an und ist es nicht — die Versuchung, „Risk-Feature wieder scharf" zu melden, ist maximal genau dann, wenn die Evidenz am duennsten ist.

**Wie vermeiden:**
1. Beim Reaktivieren eines toten Pfades die Wirkung am KONSUMENTEN messen (Order/qty/Fill), nicht an der Fix-Stelle.
2. Der Regressionstest muss die WIRKUNG mitfuehren, nicht nur die Erreichbarkeit.
3. Overlays, die `target_weight` aendern, muessen `target_qty` synchron halten (Nachbar-Overlays im selben Modul tun das) — jede Ausnahme ist begruendungspflichtig.
**Erkannt in:** Kompakt-Review 2026-08-11 (F-senior-1/2).
**Referenzen:** E-135, E-136.

## E-144 — NaN als „lauter Sentinel" auf einer Praemisse, die nie gemessen wurde
**Datum:** 2026-08-15
**Kategorie:** false-safety / sentinel-ohne-messung
**Was passierte:** Zwei Cache-Writer schrieben `adj_close = NaN` fuer angehaengte Zeilen (F-RX-3 §9.12 (a)). Begruendung im Kommentar: die Quelle habe kein adj_close, und `adj_close = close` wuerde Ex-Dividenden-Tage still falsch behandeln. Die Praemisse war falsch. `close` in diesem Cache IST bereits total-return-adjustiert (`pipeline/io.py:33-41`; gemessen: AAPLs 4:1-Split am 2020-08-31 zeigt keinen Sprung, 15 von 279.013 Zeilen haben close ausserhalb [low, high], und alle 15 sind Float-Epsilon (max 2,8e-14 absolut, 1,5e-16 relativ)), und der yfinance-Pfad laeuft mit `auto_adjust=True` (`data/sources/yfinance_source.py:73`). Ueberall, wo adj_close belegt war, war es EXAKT gleich close: 180.734 von 180.734 Zeilen, max. Differenz 0.0. Der Sentinel hinterliess 98.279 NaN — 35,2 % der Spalte.

**Warum falsch:** Ein Sentinel ist eine Aussage ueber die Datenlage („hier fehlt Information"). Wird er gesetzt, ohne die Datenlage zu messen, kehrt er sich um: er ERZEUGT den Fehlzustand, den er anzeigen sollte. „Loud failure" schuetzt nur, wenn der stille Fehler real existiert; sonst ist es Vergiftung mit guter Absicht. Verschaerfend: der Test hiess `test_..._sets_adj_close_nan` und zementierte die Fehlannahme als Vertrag — die Repraesentation war getestet, nicht die Invariante.

**Wie vermeiden:**
1. Vor jedem Sentinel die Praemisse EINMAL am realen Feed messen (hier haette ein Blick auf `auto_adjust=True` genuegt), nicht aus dem Spaltennamen ableiten.
2. Sentinel-Anteil an der Zielspalte nach dem ersten Produktionslauf zaehlen. Zweistellige Prozentwerte sind kein Schutzsignal, sondern ein Defekt.
3. Tests auf die INVARIANTE schreiben („adj_close ist adjustiert"), nicht auf die Repraesentation („adj_close ist NaN").
**Erkannt in:** Datenbestandsbewertung 2026-08-15 (`scripts/ops/refresh_daily_cache_from_panel.py`, `scripts/ops/refresh_sector_etf_cache.py`).
**Referenzen:** E-111, E-135.

## E-145 — Default-Pfad auf eine nie existierende Datei: Risikokontrolle im Code, abwesend im Verhalten
**Datum:** 2026-08-15
**Kategorie:** silent-disable / default-pfad-ins-leere
**Was passierte:** `get_default_security_master_path()` lieferte `data/security_master.csv`. Diese Datei existiert nicht und hat nie existiert — der Security Master liegt git-getrackt unter `configs/security_master.csv`. Alle drei Aufrufer (`scripts/run_backtest_strategy.py`, `scripts/run_daily.py`, `pipeline/_tc_execution.py`) behandeln eine fehlende Datei als „skip" mit einer Zeile in der Art „group exposure limits will be skipped". Netto liefen Sektor-, Regions- und FX-Gruppenlimits in der Default-Konfiguration nie.

**Warum falsch:** Der Ausfall war doppelt getarnt. Der Default zeigte auf einen plausiblen Ort, und jeder Aufrufer degradierte einzeln und leise — kein Pfad war fuer sich auffaellig, die Kombination deaktivierte eine Risikokontrolle vollstaendig. Ein „skip", das in JEDEM Lauf eintritt, erzeugt nie ein Delta und wird deshalb nie bemerkt.

**Wie vermeiden:**
1. Jeder Default-Pfad zu einer Konfigurationsdatei braucht einen Test, der die Existenz am DEFAULT-Ort prueft — nicht den Ladevorgang mit tmp_path.
2. Fehlt eine Datei, von der eine RISIKOKONTROLLE abhaengt, ist „skip" der falsche Default: fail-closed, oder mindestens WARN/ERROR mit dem gesuchten absoluten Pfad im Text.
3. Bei jeder „... will be skipped"-Zeile einmal fragen: kann dieser Zweig der Normalfall sein? Wenn ja, ist die Kontrolle eine Attrappe.
**Erkannt in:** Datenbestandsbewertung 2026-08-15 (`src/assembled_core/data/security_master.py`).
**Referenzen:** E-135, E-142.

## E-146 — CWD-relativer Pfad liefert ein leeres Universum, das wie ein gueltiges Ergebnis aussieht
**Datum:** 2026-08-15
**Kategorie:** cwd-abhaengigkeit / leeres-ergebnis-als-antwort
**Was passierte:** Der Watchlist-Fallback in `get_universe_members()` las `Path("watchlist.txt")` — relativ zum Arbeitsverzeichnis. Aus jedem Prozess, der nicht im Repo-Root startete (Scheduler, Task-Runner, Tests mit chdir), lieferte die Funktion stillschweigend `[]`. Gemessen: 0 statt 195 Symbole aus fremdem CWD. Zweiter Fehler in derselben Zeile: `line.startswith("#")` wurde auf der UNgestrippten Zeile geprueft, ein eingerueckter Kommentar waere als Symbol durchgerutscht.

**Warum falsch:** Der Rueckgabewert trug keine Information ueber den Grund. Leer aus „Datei nicht gefunden" und leer aus „heute keine Mitglieder aktiv" sind operativ voellig verschiedene Zustaende, die denselben Wert erzeugten — dieselbe Klasse wie `signal=0.0` fuer „Feed ausgefallen" und „Markt ruhig". Deployment-abhaengige Fehler dieser Art sind lokal unreproduzierbar, weil man im Repo-Root steht.

**Wie vermeiden:**
1. Repo-interne Ressourcen NIE ueber CWD-relative Pfade aufloesen. Reihenfolge: konfigurierter Override -> Anker ab `Path(__file__).resolve().parents[N]` -> relativer Pfad nur als Rueckwaertskompatibilitaet.
2. Fehlt eine erwartete Quelle, mindestens WARN mit dem tatsaechlich GESUCHTEN Pfad, und im Text ausdruecklich sagen, dass „leer" nicht „keine Mitglieder" bedeutet.
3. Bei Text-Watchlists immer `strip()` VOR dem Kommentar-Test.
**Erkannt in:** Datenbestandsbewertung 2026-08-15 (`src/assembled_core/data/universe.py`).
**Referenzen:** E-112, E-142.

## E-147 — Protokoll nur am Happy-Path-Ende geschrieben: der Ausfall loescht seine eigene Evidenz
**Datum:** 2026-08-15
**Kategorie:** logic-error / evidenz-hinter-dem-erfolgspfad
**Was passierte:** Zwei Puller sammelten pro Anfrage eine Protokollzeile und schrieben das Protokoll
mit einem `plog.write()` am Ende von `main()`. Bei einem Anbieterausfall raist der erste Request
durch `main()` hindurch — `write()` wurde nie erreicht. Verifiziert mit gestubbtem HTTPError 503:
die Fehlerzeilen standen im Speicher, auf Platte landete nichts. Zweite Stufe desselben Fehlers:
nach dem `try/finally`-Fix existierte die Datei zwar, war aber leer (`requested: 0`), weil die
Transportschicht raist, bevor der Aufrufer — die einzige Ebene, die den Schluessel kennt —
protokollieren konnte.

**Warum falsch:** Ein Anfrage-Protokoll wird fuer genau den Lauf gebaut, in dem die Quelle nicht
liefert. Steht der Schreibvorgang hinter dem Erfolgspfad, existiert es in allen Laeufen ausser dem
einen, der zaehlt — und der Nachweis „wir haben gefragt" fehlt weiterhin. E-112 ist dann nicht
repariert, nur teurer geworden.

**Wie vermeiden:**
1. Jede Evidenz-Artefakt-Schreibung gehoert in `finally`, nicht ans Funktionsende.
2. Den Fehler auf der Ebene protokollieren, die den SCHLUESSEL kennt (der Aufrufer), nicht nur dort,
   wo die Exception entsteht (der Transport).
3. Gegenprobe im Test: Transport gezielt raisen lassen und pruefen, dass die Datei existiert UND
   den angefragten Schluessel enthaelt. „Datei da" allein genuegt nicht.
**Erkannt in:** Stage-2-Review der Datenbestandsbewertung 2026-08-15 (`scripts/data/pull_ecb_fx.py`,
`pull_coingecko_ohlc.py`).
**Referenzen:** E-112, E-142.

## E-148 — Aggregat zaehlt Log-Zeilen und nennt sie „requested"
**Datum:** 2026-08-15
**Kategorie:** logic-error / falscher-nenner
**Was passierte:** `PullLog.summary()` zaehlte `len(self.entries)` als `requested`. Ein Puller
protokollierte pro Symbol zweimal — einmal auf Transportebene (`n_rows=None` -> `ok`), einmal auf
Parse-Ebene (`n_rows=0` -> `empty`). Ein einzelnes Waehrungspaar erschien damit als
`requested=2, ok=1, empty=1`: gleichzeitig erfolgreich und leer.

**Warum falsch:** Coverage-Aussagen werden aus `requested` und `ok` gebaut. Ist der Nenner eine
Zeilenzahl statt einer Anfragezahl, ist die Quote frei erfunden — dieselbe Fehlerform wie der
widerrufene Anreicherungsfaktor 3,06 aus E-112, nur eine Ebene hoeher. Besonders tueckisch, weil
das Protokoll fachlich korrekt befuellt war; erst die Aggregation log.

**Wie vermeiden:**
1. Aggregate ueber ENTITAETEN bilden, nicht ueber Zeilen: pro Schluessel den finalen Status
   bestimmen, dann zaehlen.
2. Genau EINE Ebene darf pro Schluessel protokollieren; alles Weitere gehoert als Zusatzfeld in
   dieselbe Zeile.
3. Mehrfacheintraege sichtbar machen (`duplicate_keys`) statt sie still in die Summen zu falten.
**Erkannt in:** Stage-2-Review der Datenbestandsbewertung 2026-08-15
(`src/assembled_core/data/pull_log.py`, `scripts/data/pull_ecb_fx.py`).
**Referenzen:** E-112, E-147.

## E-149 — Zusatzfeld an die Transportschicht gehaengt statt an den Protokoll-Recorder
**Datum:** 2026-08-15
**Kategorie:** wiring-gap / falsche-ebene
**Was passierte:** Ein Review-Finding verlangte, Waehrung und Intervall aus dem zweckentfremdeten
`window_end`-Feld in eigene Felder zu ziehen. Der Fix landete am Aufruf der HTTP-Hilfsfunktion
(`http_get_json(..., quote_ccy=ccy)`) statt am Recorder (`PullLog.record`, der `**extra` hat).
Die Signatur von `http_get_json` nimmt kein `**extra` — **TypeError bei jedem Symbol, vor jedem
Netzwerkzugriff**. Der Puller war ab diesem Commit vollstaendig tot. Verschaerfend: das im selben
Schritt eingebaute breite `except Exception` im Symbol-Loop fing den eigenen TypeError und schrieb
ihn als *Anbieterfehler* ins Anfrage-Protokoll — das Artefakt sah aus wie ein sauber
dokumentierter Vendor-Ausfall. Der Schwester-Puller bekam dasselbe Finding korrekt umgesetzt
(`plog.record(..., interval=...)`): zwei Umsetzungen, eine davon nie ausgefuehrt.

**Warum falsch:** Extra-Felder gehoeren an die Ebene, die das Protokoll SCHREIBT, nicht an die,
die es durchreicht. Zwischenschichten mit fixer Signatur schlucken keine freien kwargs. Und ein
Evidenzartefakt, das eigene Programmierfehler als Anbieterfehler fuehrt, vergiftet genau die
Aussage, fuer die es gebaut wurde.

**Wie vermeiden:**
1. Nach jeder Signatur-Erweiterung den geaenderten Aufruf EINMAL ausfuehren — ein gestubbter
   Transport genuegt, keine Netzverbindung noetig.
2. Wird dasselbe Finding an mehreren Call-Sites umgesetzt, pruefen, ob alle dieselbe Ebene treffen.
3. Jeder Ingest, der ein Evidenzartefakt schreibt, braucht einen Smoke-Test mit gestubbtem
   Transport. Dieser BLOCKER ueberlebte SECHS Review-Durchlaeufe, weil kein Test den Puller je
   aufrief — die Kette prueft Diffs, nicht Lauffaehigkeit.
4. Im Ingest-Loop Programmierfehler von Transportfehlern trennen (TypeError/AttributeError/
   NameError durchlassen oder als `error_class="internal"` markieren), damit Coverage-Auswertungen
   interne Fehler ausschliessen koennen.
**Erkannt in:** Stage-2-Review der Datenbestandsbewertung 2026-08-15
(`scripts/data/pull_coingecko_ohlc.py`).
**Referenzen:** E-112, E-147, E-148.

## E-150 — Logging-Formatstring mit falscher Platzhalterzahl versteckt sich selbst
**Datum:** 2026-08-15
**Kategorie:** logic-error / selbstverdeckender-defekt
**Was passierte:** In `PullLog.write()` wurde ein Argument (`summary["skipped"]`) ergaenzt, ohne den
Formatstring mitzuziehen — 7 Platzhalter, 8 Argumente. Ergebnis: `TypeError: not all arguments
converted during string formatting` bei JEDEM erfolgreichen `write()`. Symptom war ausschliesslich
`--- Logging error ---` auf der Konsole eines Standalone-Laufs. In der isolierten Testdatei blieb
alles gruen; erst ein Vollauf, bei dem `tests/test_logging_config.py` (alphabetisch davor) einen
`RotatingFileHandler` am Root-Logger hinterliess, machte 13 von 14 Tests rot — dessen
`shouldRollover` formatiert den Record und loest den Fehler aus.

Zweiter, gleichartiger Vorfall in derselben Sitzung: der Nachbesserungs-`replace` fuer denselben
Formatstring traf nicht, weil `ruff format` die Zeile zwischenzeitlich umgebrochen hatte. Der Fix
galt als erledigt und war es nicht.

**Warum falsch:** `logging` faengt Exceptions in `emit()` ab und ruft `handleError` — der Fehler
propagiert NICHT. Kein Test kann anschlagen, die Suite bleibt gruen, und der Defekt ist nur
sichtbar, wenn ein Mensch zufaellig die Konsole liest. Eine Statuszeile, die genau dann
verschwindet, wenn sie gebraucht wird, ist schlimmer als keine.

**Wie vermeiden:**
1. Jede Log-Zeile mit >= 4 Platzhaltern bekommt einen `caplog`-Test, der `record.getMessage()`
   aufruft — das wendet die %-Formatierung an und wirft bei Arity-Mismatch.
2. Nach jeder Erweiterung eines `summary()`-Dicts die zugehoerige Log-Zeile mitzaehlen.
3. Nach einem `sed`/`replace`-Fix auf einer Datei, die anschliessend formatiert wird, das
   Ergebnis GEGENLESEN — ein nicht getroffener Replace meldet sich nicht.
**Erkannt in:** Vollauf auf eingefrorenem Baum, Datenbestandsbewertung 2026-08-15
(`src/assembled_core/data/pull_log.py`).
**Referenzen:** E-142, E-149.

## E-151 — Nicht-ASCII im print eines Produktionspfads: Erfolg wird als Anbieterfehler protokolliert
**Datum:** 2026-08-15
**Kategorie:** logic-error / umgebungsabhaengige-quittung
**Was passierte:** `print(f"[INTRA] OK {s} → {fp}")` mit U+2192. Unter Windows-cp1252 wirft das einen
`UnicodeEncodeError` — und zwar NACH `df.to_parquet(fp)`, aber VOR `plog.record(...)`. Der breite
per-Symbol-`except` fing den eigenen Encoding-Fehler und schrieb den **vollstaendig erfolgreichen**
Pull als Anbieterfehler ins Anfrage-Protokoll; der Lauf endete mit `rc=2`. Gemessen: Erfolgsfall
lieferte `requested=2, ok=0, error=2` bei zwei korrekt geschriebenen Parquets auf der Platte.

**Warum falsch:** Die Umgebungsabhaengigkeit sitzt zwischen der Arbeit und ihrer Quittung. Das
Evidenzartefakt behauptet danach das Gegenteil dessen, was passiert ist — ausgerechnet in dem
Artefakt, das eine Coverage-Frage beantworten soll. Unter Linux (UTF-8) waere der Defekt nie
aufgetreten; die Ops-Tasks laufen unter Windows.

**Wie vermeiden:**
1. Ausgabestrings in Skripten ASCII-only (`→` -> `->`, `—` -> `--`).
2. Buchhaltung VOR der Konsolenausgabe, nie danach.
3. Im Ingest-Loop Programmierfehler von Transportfehlern trennen, damit ein eigener Bug nicht als
   Vendor-Fehler in der Evidenz landet (vgl. E-149).
**Erkannt in:** Stage-1-Review der Datenbestandsbewertung 2026-08-15
(`scripts/data/pullers/pull_alpha_vantage_intraday.py`).
**Referenzen:** E-112, E-147, E-149.

## E-152 — Union-Kalender erzeugt Phantomtage: die Simulation liest „kein Kurs" als Ereignis
**Datum:** 2026-08-16
**Kategorie:** logic-error / union-kalender-phantomtage
**Was passierte:** Das Preispanel entstand als Union aller Symbolkalender. An 9 US-Feiertagen im
Suchfenster trugen nur wenige (fehlerhafte) Reihen Kurse, die Mehrheit hatte NaN. Die
H-090-Tagesschleife interpretierte NaN als Delisting und verkaufte an diesen Tagen ALLE offenen
Positionen zwangsweise — in jeder der 8 Varianten. Der komplette erste Volllauf war kontaminiert.

**Warum falsch:** Ein Union-Kalender ist kein Handelskalender. Jede Regel, die „heute kein Kurs"
als Ereignis liest (Delisting, Halt, Datenausfall), feuert auf reinen Kalenderartefakten. Der
Fehler ist unsichtbar: die Zahlen bleiben plausibel, die Trade-Zahl steigt sogar.

**Wie vermeiden:**
1. Simulationskalender IMMER an einem vollstaendigen Ankerinstrument festmachen
   (`close = close[close["SPY"].notna()]`).
2. Die Zahl der entfernten Tage ins Ergebnis-Artefakt schreiben (`n_phantom_days_removed`).
3. NaN im Panel nie ohne Kalenderpruefung als Ereignis interpretieren.
**Erkannt in:** Stage-1-Review H-090 Welle 50 (`research/mandat2/h090_momentum_exits.py`).
**Referenzen:** E-085, E-144.

## E-153 — last_valid als Delisting-Detektor: der Fallback-Preis kommt aus der Zukunft
**Datum:** 2026-08-16
**Kategorie:** logic-error / fensterwissen-als-preisquelle
**Was passierte:** Bei fehlendem Kurs verkaufte die Schleife zum „letzten verfuegbaren Kurs" via
`last_valid` ueber das GESAMTE Fenster. Bei temporaeren Kursluecken (892/1037 Symbole auf dem
rohen Kalender) lag dieser Kurs im Median 501 Handelstage NACH dem Verkaufstag. Verdict-relevant
gemessen: BASIS PF-H2 1,755 -> 1,517 allein durch die Preiskorrektur (Zwischenmessung ohne den
Phantomtag-Fix, nicht als Artefakt persistiert; Endwert nach beiden Fixes 1,471).

**Warum falsch:** `last_valid` ist eine Fenster-Aggregation, kein Zeitpunkt-Fakt. Als Preisquelle
liefert es Look-ahead; als Ereignisdetektor verwechselt es Datenluecke mit Delisting.

**Wie vermeiden:**
1. Zwangsverkauf nur wenn `i > last_valid[col]` — der Kurs ist dann garantiert Vergangenheit.
2. Temporaere Luecken: Position halten, Fills verschieben, Signalpruefung ueberspringen.
3. Verbleibende Fensterwissen-Nutzung (die UNTERSCHEIDUNG Luecke vs. Delisting) als Restannahme
   im Docstring benennen, nicht verschweigen. Bei Ticker-Recycling (E-114) zusaetzlich pruefen,
   ob eine „Luecke" in Wahrheit zwei Firmen unter einem Symbol sind.
**Erkannt in:** Stage-1-Review H-090 Welle 50 (`research/mandat2/h090_momentum_exits.py`).
**Referenzen:** E-114, E-152.

## E-154 — PASS gegen die Nullmessung, ohne die unveraenderte Basis als zweite Referenz
**Datum:** 2026-08-16
**Kategorie:** research-anti-pattern / pass-label-ohne-basis-referenz
**Was passierte:** Das vorregistrierte Primaerkriterium verglich jede Exit-Variante gegen die
Zeit-Referenz V7 (Nullmessung). 5 Varianten bekamen „PASS_PRIMAER" — waehrend KEINE die
unveraenderte Basisstrategie schlug (beats_basis false in 7/7, beide Haelften).

**Warum falsch:** Ein Label gegen eine bewusst schwache Nullmessung wird beim Zusammenschreiben
zwangslaeufig als „die Exits wirken" gelesen. Der Befund kann das Vorzeichen wechseln, ohne dass
eine einzige Zahl falsch ist.

**Wie vermeiden:**
1. Bei „Zusatzregel verbessert Strategie X"-Hypothesen IMMER zwei Referenzen ins Ergebnis-JSON:
   die vorregistrierte Nullmessung (Label, unveraenderlich) UND die unveraenderte
   Ausgangsstrategie als eigenes Flag (`beats_basis`).
2. Das Flag bis in den Endbefund und alle Folgephasen durchreichen.
**Erkannt in:** Stage-2-Review H-090 Welle 50 (`research/mandat2/h090_momentum_exits.py`).
**Referenzen:** E-085.

## E-155 — Replay-Reihenfolge verschluckt Zero-Day-Trades und friert Kapital ein
**Datum:** 2026-08-16
**Kategorie:** logic-error / replay-verschluckt-zero-day
**Was passierte:** Beim Abspielen einer Trade-Liste auf ein Portfolio liefen Sells vor Buys (wie
in der Simulationsschleife). Ein Trade mit entry_date == exit_date (Delisting am Einstiegstag,
MEE 2011-06-01) traf beim Sell auf qty==0, der Sell wurde still verworfen, der Buy danach
ausgefuehrt — die Position blieb bis zum Fensterende liegen, ~1/5 des Portfolios war ab 2011 tot.
Wirkung: BASIS-Endwert PRIVAT_DE 508.830 statt 687.867 (Bias GEGEN den Kandidaten), PBO und DSR
auf kontaminierten Kurven.

**Warum falsch:** Ein Replay ist kein Simulator: die Trade-Liste enthaelt bereits gepaarte
Ein-/Ausstiege, deren Reihenfolge innerhalb eines Tages nicht der Signalreihenfolge folgt. Der
Fehler ist still — die Trade-Zahl stimmt, die Kurve nicht.

**Wie vermeiden:**
1. Sells auf qty==0 zurueckstellen und NACH dem Buy-Loop desselben Tages erneut ausfuehren;
   bleibt qty==0, hart abbrechen (SystemExit) statt still verwerfen.
2. Tripwire am Fensterende: keine offene Position (`rest == []`) — als harter Guard, nicht als
   ungelesenes Rueckgabefeld.
3. Unit-Test mit synthetischem Zero-Day-Trade (`tests/test_h090_run_variant.py`).
**Erkannt in:** Stage-1-Review (Runde 2) H-090 Welle 50 (`research/mandat2/h090_phase2_sekundaer.py`).
**Referenzen:** E-147, E-150.

## E-156 — enabled:true ohne Implementierung: die Konfigurations-Illusion
**Datum:** 2026-08-16
**Kategorie:** wiring-gap / konfigurations-illusion
**Was passierte:** Neun Policy-Bloecke standen auf enabled:true. Vier davon (shorts,
intel.signal_layer, news_signal_bridge, earnings_guard) gateten Codebloecke, deren erster Import
ein nicht existierendes Modul/Symbol lud; der except-Zweig loggte auf DEBUG — der Ausfall lief
JEDEN Zyklus, seit Einfuehrung. Fuenf weitere Bloecke (macro_event_calendar mit FOMC-Exposure-
Scaling!, quant_gates mit DSR/PSR-Block-Gates!, quarter_end_guard, ma_exclusion,
freshness_monitor) hatten ueberhaupt keinen Code-Leser. Ein CI-Workflow (earnings-calendar-
refresh) fuetterte taeglich einen Cache fuer einen Guard, der nie lief.

**Warum falsch:** Die Konfiguration wurde zur Autoritaetsquelle ueber das Systemverhalten, obwohl
sie es nicht steuert. Jede Aussage der Form „Feature X ist aktiv, steht in policy.yaml" war
falsch — inklusive der Aussagen, die Menschen und Agenten daraus ableiteten. log.debug macht den
Ausfall formal sichtbar und praktisch unsichtbar (E-142-Verwandtschaft). Die Schema-Validierung
verstaerkte die Illusion („der Block wird validiert, also wird er gelesen").

**Wie vermeiden:**
1. Ein enabled:true-Flag braucht einen Test, der beweist, dass der Zweig bei true ANDERS laeuft
   als bei false (Mutationsprobe).
2. Feature-Modul-Imports gehoeren VOR den except-Vorhang oder mindestens auf WARN, nie auf DEBUG.
3. Bei Flag-Inventuren pruefen, ob der GELESENE Key derselbe ist wie der GESCHRIEBENE —
   earnings_guard existierte doppelt (Top-Level ohne Datei-Eintrag, verschachtelt mit Eintrag).
**Erkannt in:** Nutzungsaudit 2026-08-16 (`configs/policy.yaml`,
`src/assembled_core/pipeline/_tc_signals.py`, `orchestrator.py`).
**Referenzen:** E-142, E-145.

## E-157 — Glob-Praefix faengt die Nachbarquelle — und Lexikographie stellt sie nach vorn
**Datum:** 2026-08-16
**Kategorie:** logic-error / glob-namensfilter-statt-quellenfilter
**Was passierte:** Ein neuer Konsument suchte das juengste Protokoll per
glob("pull_log_yfinance_*.json") + sorted(reverse=True)[0]. Ein zweiter Puller schreibt
pull_log_yfinance_intraday_*.json — und weil "i" lexikographisch NACH jeder Ziffer kommt, stuende
die Fremddatei in reverse-Sortierung IMMER an erster Stelle. Der Check haette nie das Protokoll
des Live-Preispfads gesehen, den er ueberwachen soll.

**Warum falsch:** Ein Praefix-Glob ist ein Namensfilter, kein Quellenfilter; und „neueste Datei
per Namenssortierung" setzt EIN Namensschema voraus. Beides bricht lautlos: der Check laeuft,
findet eine Datei, liefert eine Zahl — nur ueber die falsche Quelle.

**Wie vermeiden:**
1. Quelle aus dem INHALT verifizieren (payload["source"]), nie aus dem Dateinamen.
2. Glob so eng wie moeglich (Ziffern-Praefix "pull_log_yfinance_2*.json").
3. Das Glob gegen ein reales Verzeichnis auflisten, nicht gegen die Vorstellung davon.
**Erkannt in:** Stage-1-Review Audit-Paket 2 (`scripts/ops_watchdog.py`).
**Referenzen:** E-146, E-152.

## E-158 — Fehlerquote ueber einen Nenner, den der Abbruch selbst gekuerzt hat
**Datum:** 2026-08-16
**Kategorie:** logic-error / abbruch-kuerzt-nenner
**Was passierte:** Ein Ratio-Alarm rechnet error/requested aus einem Anfrageprotokoll. Beim
Rate-Limit-Abbruch enthielt das Protokoll nur die ERREICHTEN Symbole: Abbruch bei Symbol 3 von
220 ergab requested=3, error=1, ratio=0,33 — KEIN Alarm, obwohl 217 Symbole nie gefragt wurden.
Je katastrophaler der Ausfall, desto stiller.

**Warum falsch:** Der Nenner soll „wie viele wollten wir" bedeuten, misst aber „wie viele haben
wir geschafft". Der Ausfall unterdrueckt genau die Kennzahl, die ihn melden soll. Eine Quote ist
nur so ehrlich wie ihr Nenner (E-112/E-148 eine Ebene weiter).

**Wie vermeiden:**
1. Beim Abbruch die nie angefragten Keys als STATUS_SKIPPED protokollieren (der Status existiert
   genau dafuer) und skipped in die Quote zaehlen.
2. Testfall: Abbruch bei Element 1 UND bei Element n-1 muessen beide alarmieren.
**Erkannt in:** Stage-2-Review Audit-Paket 2 (`src/assembled_core/data/sources/yfinance_source.py`,
`scripts/ops_watchdog.py`).
**Referenzen:** E-112, E-148.

## E-159 — Producer-Consumer-Fix ohne Bindungstest: derselbe blinde Fleck, neu lackiert
**Datum:** 2026-08-16
**Kategorie:** test-anti-pattern / fix-ohne-bindung
**Was passierte:** Zwei Monitoring-Alarme lasen jahrelang einen Dateinamen, den kein Producer je
schrieb (zombie_report_*.json — 0 Existenzen bei 901 Shadow-Dateien). Der Fix korrigierte Pfad
und Envelope-Schema — zunaechst OHNE Test, der Producer-Ausgabe und Consumer-Lesung aneinander
bindet.

**Warum falsch:** Der Originaldefekt war unsichtbar, WEIL nichts das Paar band. Ein Fix, der
diese Eigenschaft erbt, ist nicht verifiziert, sondern nur neu behauptet — der naechste
Schema-Drift schlaegt genauso still zu.

**Wie vermeiden:** Jeder Producer-Consumer-Fix bekommt einen Test, der mit dem ECHTEN
Producer-Writer schreibt (record_shadow, nicht handgebautes JSON) und mit dem echten Consumer
liest. Handgebaute Fixtures beweisen nur, dass der Consumer sein eigenes Wunschschema lesen kann.
**Erkannt in:** Stage-2-Review Audit-Paket 5.1 (`src/assembled_core/api/routers/monitoring.py`).
**Referenzen:** E-143, E-145.

## E-160 — Skript archiviert, Zeiger stehen gelassen
**Datum:** 2026-08-16
**Kategorie:** wiring-gap / archiv-ohne-zeigerpflege
**Was passierte:** Ein Legacy-Skript wurde nach archive/ verschoben; zwei In-Repo-Stellen wiesen
weiter auf den alten Pfad — eine davon als FEHLERTEXT-Anweisung an den Bediener („Bitte zuerst X
ausfuehren").

**Warum falsch:** Eine Handlungsanweisung auf einen nicht mehr existierenden Pfad ist schlimmer
als keine: sie kostet Debugzeit und suggeriert, der (gefaehrliche) Pfad sei noch der richtige.

**Wie vermeiden:** Beim Verschieben/Archivieren grep auf den DATEINAMEN ueber *.py, *.bat, *.yml,
*.md — nicht nur auf Imports, auch auf Freitext in Fehlermeldungen und Docstrings.
**Erkannt in:** Stage-2-Review Audit-Paket 6.1 (`scripts/features/build_daily_features.py`,
`scripts/data/pull_yfinance_eod.py`).
**Referenzen:** E-140, E-152.

## E-161 — Ein Zaehler wird fortgeschrieben, ohne die Grundgesamtheit neu zu messen
**Datum:** 2026-08-16
**Kategorie:** logic-error / fortgeschriebener-zaehler
**Was passierte:** Eine Coverage-Zahl in KNOWN_ISSUES §0.06(c) wurde beim Schliessen einer Luecke
um 1 erhoeht („5 → 6") statt mit dem dokumentierten Kriterium neu gezaehlt. Doppelt falsch:
(a) die Basis „5" war bereits eine Fehlzaehlung (Neuzaehlung mit dem eigenen Kriterium ergab 6),
(b) das neue Element (yfinance_source.py, nutzt `yf.Ticker`) lag GAR NICHT im 77er-Nenner der
Marker-Heuristik (`yf.download` etc.) — ein „davon"-Inkrement fuer etwas ausserhalb des „davon".

**Warum falsch:** Ein inkrementierter Zaehler erbt jeden Fehler seiner Basis und verdeckt ihn,
weil die Fortschreibung wie eine Bestaetigung aussieht. Eine Zahl, die aus ihrem eigenen
genannten Kriterium nicht reproduzierbar ist, ist die E-112-Form in Miniatur — und sie stand
ausgerechnet in dem Dokument, das der ehrliche Kassensturz sein soll (E-144: Messung vor jeder
Zahl frisch).

**Wie vermeiden:**
1. Beim Fortschreiben JEDER Coverage-Zahl das Zaehl-Kommando erneut ausfuehren; Zaehler UND
   Nenner frisch setzen.
2. Pruefen, ob das neue Element ueberhaupt im Nenner liegt — bei grep-Marker-Heuristiken ist
   das der Regelfall des Irrtums.
3. Die Zaehlweise (Kommando/Kriterium) neben die Zahl schreiben, damit jeder Leser
   reproduzieren kann.
**Erkannt in:** Kompakt-Review des Auditor-Fix-Deltas (`KNOWN_ISSUES.md` §0.06c).
**Referenzen:** E-112, E-144, E-148.

## E-162 — Das Ehrlichkeitsfeld wird zur neuen Luege: producer_exists:false im Commit, der den Producer baut
**Datum:** 2026-08-17
**Kategorie:** wiring-gap / veralteter-ehrlichkeitsmarker
**Was passierte:** Audit-Paket 5.2 stempelte producer-lose Endpunkte mit `producer_exists: false`
+ „NO producer writes this artifact anywhere in the repo". Paket 5.3 baute IM SELBEN DELTA den
Producer fuer signal_scores_* — die Aussage blieb stehen und war ab dem Moment falsch.

**Warum falsch:** Ein Ehrlichkeitsmarker ist ein Zustandsbefund mit Verfallsdatum. Bleibt er
stehen, ist er schaedlicher als die alte vage Meldung: er behauptet STRUKTURLOSIGKEIT statt
Datenlosigkeit und stoppt jede Fehlersuche („steht doch da: kein Producer").

**Wie vermeiden:**
1. „Kein Producer" und „noch keine Daten" sind ZWEI Zustaende — nie ein Feld teilen.
2. producer_exists immer mit benannter Producer-Referenz setzen.
3. Beim Bau eines Producers: grep auf `producer_exists`/„NO producer" ueber das GANZE Delta.
**Erkannt in:** Stage-2-Review Audit-Paket 5.2/5.3 (`src/assembled_core/api/routers/monitoring.py`).
**Referenzen:** E-140, E-159.

## E-163 — Frische aus dem Datei-mtime statt aus dem letzten Bar
**Datum:** 2026-08-17
**Kategorie:** logic-error / mtime-ist-kein-datendatum
**Was passierte:** /monitoring/data-quality meldete freshness aus `st_mtime`. Das operative Panel
wurde am 15.08. neu geschrieben (adj_close-Backfill), endet aber am 05.08. — der Endpoint haette
„fresh" fuer 11 Tage alte Daten gemeldet. `last_bar` wurde mitgeliefert, aber nicht bewertet.

**Warum falsch:** Ein Rewrite (Backfill, Spaltenmigration, PIT-Umbau) ist ein DATEI-Ereignis,
kein DATEN-Ereignis. mtime misst den Schreiber, nicht den Markt.

**Wie vermeiden:** Freshness IMMER aus der juengsten Beobachtung im Datensatz ableiten
(max(timestamp)); mtime nur als getrenntes Feld (file_age_hours) ausweisen.
**Erkannt in:** Stage-2-Review Audit-Paket 5.2 (`src/assembled_core/api/routers/monitoring.py`).
**Referenzen:** E-144, E-161.

## E-164 — 503-Guard nur gegen die fehlende Datei, nicht gegen das leere Artefakt
**Datum:** 2026-08-17
**Kategorie:** logic-error / container-guard-statt-inhalts-guard
**Was passierte:** Der neue walk_forward-503-Guard prueffte nur `if not data` (Datei fehlt). Ein
vorhandenes Artefakt mit leerem walk_forward-Block (Teilschreibung, abgebrochener Lauf) lieferte
weiter HTTP 200 mit Perzentilen aus mean=0/std=1 — exakt dem Muster, dessen Beseitigung der
frisch geschriebene Kommentar behauptete.

**Warum falsch:** Abgebrochene/teilgeschriebene Laeufe sind der HAEUFIGERE Fall als „nie
gelaufen". Ein Guard auf Dateiexistenz deckt die schwaechere Haelfte und erzeugt falsche
Sicherheit im Kommentar.

**Wie vermeiden:**
1. Guards auf den BENOETIGTEN INHALT setzen (Block vorhanden, n_splits > 0), nicht auf den
   Container.
2. Jede Kommentar-Behauptung „X kann nicht mehr passieren" gegen einen Probe-Aufruf mit
   Teilartefakt testen.
**Erkannt in:** Stage-2-Review Audit-Paket 5.2, per Scratch-Probe belegt
(`src/assembled_core/api/routers/qa.py`).
**Referenzen:** E-147, E-158.
