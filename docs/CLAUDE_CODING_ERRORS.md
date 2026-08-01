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
