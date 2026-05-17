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
