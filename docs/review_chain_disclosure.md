# Review-Chain — Ablauf, Schema und ehrliche Bypass-Disclosures

> **Herkunft:** Ausgegliedert aus `CLAUDE.md` §20.1–20.5 / §20.7 / §20.8 am **2026-05-30**
> im Zuge der CLAUDE.md-Restrukturierung (Verschlankung der Verfassung auf Kernregeln).
> Diese Datei ist die autoritative Langfassung. `CLAUDE.md` enthält nur noch die
> kompakte Review-Chain-Zusammenfassung inkl. One-Shot-Skip (Abschnitt „Review-Chain"
> in der verschlankten `CLAUDE.md`; die alte §20.6-Nummerierung existiert dort nicht mehr).
> Erzwungen wird die Kette durch den Stop-Hook `.claude/hooks/stop_review_chain.py`.

---

## Review-Chain (automatisch erzwungen)

Nach jedem Coding-Step mit Edits in geschützten Pfaden (`src/`, `scripts/`, `.github/workflows/`, `.claude/rules/`, `CLAUDE.md`) läuft eine Review-Kette **zwingend**, erzwungen durch den Stop-Hook (`.claude/hooks/stop_review_chain.py`).

### 20.1 Ablauf

1. **Stage 1 (parallel):** relevante Spezialisten — `risk-execution-reviewer` (bei sensiblen Zonen), `test-runner` (immer bei `src/`/`scripts/`), `ci-debugger` (bei Workflows), `docs-governance-sync` (bei Governance-Docs).
2. **Stage 2:** `senior-code-reviewer` (Opus) — breiter Code-Review auf Bugs, Wiring, Vollständigkeit, Korrektheit, bekannte Anti-Patterns.
3. **Stage 3:** `task-completion-auditor` (Opus) — prüft Task-Erfüllung mit Tiefe, flaggt Adjacent als Follow-up, vergibt Verdict PASS/CONDITIONAL/FAIL.

### 20.2 Findings-Schema

Strukturiertes YAML mit Feldern `file`, `line`, `severity` (BLOCKER/MAJOR/MINOR/INFO), `category`, `evidence`, `suggested_fix`. Details: `docs/superpowers/specs/2026-05-14-review-chain-design.md` §5.

### 20.3 Step-Abschluss-Regel

Ein Step gilt **erst dann als abgeschlossen**, wenn:
- Verdict = PASS, oder
- Verdict = CONDITIONAL und MAJOR-Findings sind adressiert oder dokumentiert akzeptiert.

Vorher wird der User nicht informiert „fertig". BLOCKER müssen immer adressiert werden.

### 20.4 Anti-Pattern-Register

Wiederholungs-würdige Fehler werden in `docs/CLAUDE_CODING_ERRORS.md` (append-only) festgehalten. SessionStart-Hook (`.claude/hooks/session_start_load_errors.py`) lädt Top-10 in den Initial-Kontext. Volle Datei bei Bedarf lesen.

### 20.5 Was diese Kette NICHT ändert

- Grundsatz „kleinster sicherer Schritt" (CLAUDE.md-Abschnitt gleichen Namens) bleibt bindend.
- Rule 10 „kein großer Refactor ohne Auftrag" bleibt bindend.
- Rule 60 „ein Problem pro Änderung" bleibt bindend.
- Auditor darf Adjacent nur als Follow-up-Vorschlag flaggen, nie als Pflicht im aktuellen Step.

> **Hinweis:** Der One-Shot-Skip für Mid-Task-Pausen (vormals §20.6) ist in `CLAUDE.md`
> verblieben, weil er operativ jede Session betrifft. Mechanismus dort: nicht-leere
> Begründung in `.claude/.review_skip` schreiben; one-shot, wird nach Konsum gelöscht;
> Audit-Log `.claude/.review_skip_log.jsonl`. Skip ersetzt die Kette nicht.

---

## 20.7 Bootstrap-Lücke (v1) — ehrliche Disclosure

Die Commits, die die Review-Chain **selbst** gebaut haben (Range `1547fb7..772d395`, ca. 30 Commits zwischen 2026-05-14 16:00–19:20 GMT+2), sind **nicht** durch die Chain gelaufen, weil:

- die beiden Stage-2/3-Subagents (`senior-code-reviewer`, `task-completion-auditor`) erst während dieser Session geschrieben wurden und Claude Code Subagent-Typen nur beim Session-Start registriert,
- der Stop-Hook erst gegen Ende der Session in `.claude/settings.json` registriert wurde, weil er sonst während des Aufbaus auf die eigenen Build-Edits getriggert hätte (Chicken-and-Egg).

**Stattdessen lief am Sessionende eine Bootstrap-Review-Kette:**
- Stage 1: `docs-governance-sync` (echt, war seit Session-Start verfügbar)
- Stage 2 + 3: simuliert via `general-purpose`-Subagent mit den verbatim Agent-Prompts

Diese Bootstrap-Kette fand 3 MAJOR-Findings (F-senior-1, F-senior-2, F-auditor-1), die in Commits nach `772d395` adressiert wurden.

**Konsequenz für zukünftige Sessions:** Die Chain SOLLTE ab `2026-05-14 19:20 GMT+2` vollständig aktiv sein — eine zweite, getrennte Bypass-Lücke (Parser-Bug, siehe §20.8) hat das aber zwischen `dcdbe7e` und `2925a72` (~36 Commits, ~27 Stunden) unterlaufen. Die Aussage „vollständig aktiv" gilt erst ab Commit `60c7ea2` (2026-05-15 22:56 GMT+2). Die Bootstrap-Commits gelten als **akzeptiert** mit dokumentierter retroaktiver Review. Ein nachgelagerter Echtbetriebs-Re-Review (mit den nun registrierten Stage-2/3-Subagents) ist als Follow-up-Option dokumentiert, aber nicht zwingend.

**Lessons learned für künftige Meta-Workflow-Erweiterungen:** Wenn eine neue Komponente die Chain modifiziert (z. B. neue Stages, neuer Trigger-Pfad), gilt dieselbe Bootstrap-Lücke. Saubere Lösung: Komponente in eigenem Branch implementieren, dort von der **aktuellen** Chain reviewen lassen, dann mergen. **Zweite Lesson (siehe §20.8):** „Bootstrap-Lücke" ist nur ein Failure Mode. Eine zweite Klasse ist „silent fail-open in der Enforcement-Schicht selbst" — der Code existiert, läuft, aber gibt bei produktiven Input-Shapes still ein leeres Ergebnis zurück. Eine Enforcement-Schicht, die „nie triggert", ist verdächtig, nicht erfolgreich.

## 20.8 Parser-Bug-Bypass (v2) — ehrliche Disclosure

Zwischen Commit `dcdbe7e` (2026-05-14 19:22:44 GMT+2) und Commit `2925a72` (2026-05-15 21:33:30 GMT+2) hat die Review-Chain **nicht automatisch getriggert**, obwohl §20.7 sie als „vollständig aktiv" beschrieb.

**Ursache:** Bug in `.claude/hooks/hook_utils/transcript_parser.py`. Claude Code speichert Tool-Ergebnisse als `type=user`-Einträge mit `content=[tool_result, ...]`. Der Parser lief vom Transkriptende rückwärts und brach bei jedem `type=user` ab — also immer beim ersten Tool-Ergebnis nach einem Edit/Write. Die trailing Assistant-Turn sah dadurch leer aus, `edited_paths_in_last_turn()` lieferte `[]`, `classify_diff([])` setzte `run_full_chain=False`, der Stop-Hook ließ den Stop durch. Kein Marker wurde je geschrieben.

**Evidenz, dass der Hook nie gefeuert hat:**
- `.claude/.review_markers/` existierte nie
- `.claude/.review_skip_log.jsonl` wurde nie angelegt
- Der `.review_skip` aus dem Bootstrap-Session-Ende (2026-05-14 19:23, verbatim Inhalt: *„Bootstrap-Session-Ende: Chain via Simulation … durchgelaufen, 3 MAJOR-Findings adressiert in Commit dcdbe7e. Echte Stage-2/3-Subagents … sind ab nächster Session registriert und greifen automatisch."*) lag bis zum Parser-Fix unkonsumiert auf Platte (§20.6: One-Shot-Skips werden bei der nächsten Stop-Auswertung gelöscht)

**Umfang:** ~36 Commits, ca. 27 Stunden. Mehrere davon in Schutzpfaden (`src/assembled_core/execution/`, `src/assembled_core/risk/`, paper-ledger F-A-1 BLOCKER, mfv2 F-B-1/2/3 BLOCKERs, FRED helpers, broker-adapter, pre-trade, risk_controls, georisk). Die sensiblen Änderungen wurden durch **manuell aufgerufene** `senior-code-reviewer`-Runs (Audit Rounds 1–6) reviewt — aber nicht durch die hook-erzwungene Chain.

**Audit Rounds 1–6 ≠ chain-validiert:** Beide Verfahren prüfen Code, aber unterschiedliche Surfaces. Audit Rounds gingen tief auf einzelne Hochrisiko-Module (risk, execution, paper, FRED) durch einen Senior-Reviewer in einem Pass. Die Chain prüft pro Commit breiter (Stage-1-Spezialisten je Pfadart, Stage-2 Senior, Stage-3 Auditor mit strukturiertem Verdict). Konkrete Gaps der Audit Rounds gegenüber der Chain: kein `ci-debugger` für den `.github/workflows/`-Edit (Commit `3562f0b` autonomous Task Scheduler), kein `test-runner` programmatisch je Commit, kein `task-completion-auditor` mit PASS/CONDITIONAL/FAIL pro Commit. Die Bypass-Commits gelten als **akzeptiert** mit dokumentiertem Restrisiko. Ein optionaler Re-Review (mindestens für den Workflow-Edit, wo Audit-Coverage null war) ist als Follow-up dokumentiert.

**Fix:** Commit `60c7ea2` (2026-05-15 22:56 GMT+2). `_is_real_user_message()` unterscheidet echte User-Eingaben (string content oder content-Liste mit text-Block) von Tool-Result-Wrappern. Regression-Tests in `tests/hooks/test_transcript_parser.py`. Ergänzungen (im gleichen Step, siehe Folge-Commits): `message=null`-Crash-Guard (F-senior-1), Whitespace-only-Empty-String-Guard (F-senior-3).

**Pflichtprüfungen-Härtung (siehe neuer Anti-Pattern-Eintrag E-019 in `docs/CLAUDE_CODING_ERRORS.md`):**
1. Hook-Layer-Tests müssen Fixtures mit Produktions-Shape verwenden (nicht nur synthetische Minimal-Shapes).
2. Enforcement-Schichten dürfen nicht still fail-open gehen — bei unbekanntem Input-Shape ist explizites Logging Pflicht, nicht leere Rückgabe.
3. „Enforcer triggert nie" = roter Flag. Heartbeat-Log einbauen (Follow-up §20.8.1) oder regelmäßig manuell prüfen, ob `.claude/.review_markers/` aktuell ist.

**Akzeptierte Follow-ups (nicht Teil von Commit `60c7ea2`, separat zu tracken):**
- F-senior-2: `.claude/hooks/` zu `PROTECTED_PREFIXES` hinzufügen + Default-Specialist-Fallback in `specialists_for_paths` für test-only Diffs. Heute fällt die Chain bei Edits in der Enforcement-Schicht selbst auf Stage 1 = leere Specialist-Liste, weil `.claude/hooks/` nicht protected ist. Chain blockt trotzdem (über test-only Fallback), aber Stage 1 ist no-op. Adjacent per Rule 60.
- F-senior-5 / F-DGS-3: Fixture `tests/hooks/fixtures/transcript_with_edits.jsonl` an Produktions-Shape anpassen (oder zweite Fixture-Datei mit tool_result Wrappern) + Hook-Test-Realismus-Regel in `.claude/rules/90-subagents-hooks-and-automation.md`.
- F-senior-7: Heartbeat-Log für den Stop-Hook (jeder Invoke schreibt eine Zeile, auch bei leerem Result). Absence-of-heartbeat wird dann selbst zum Signal.
- Optionaler Echtbetriebs-Re-Review für `.github/workflows/`-Edit (`3562f0b`).
