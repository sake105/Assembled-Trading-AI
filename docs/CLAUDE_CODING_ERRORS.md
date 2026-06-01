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
