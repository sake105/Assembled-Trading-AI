# Review-Chain Design — Senior-Reviewer + Task-Completion-Auditor

**Datum:** 2026-05-14
**Status:** Spec (genehmigt, noch nicht implementiert)
**Scope:** Meta-Workflow-Erweiterung für Claude Code in Assembled-Trading-AI
**Spec-Autor:** Claude Code (Opus 4.7), nach Brainstorming-Session mit User

---

## 1. Motivation

Heute ist in diesem Repo keine strukturelle Erzwingung vorhanden, dass nach jedem Coding-Step ein systematischer Review läuft. Die 5 bestehenden Spezialist-Subagents (`risk-execution-reviewer`, `test-runner`, `ci-debugger`, `docs-governance-sync`, `memory-tracker`) sind alle **domänenspezifisch**. Es gibt:

- keinen **breiten Senior-Code-Reviewer** für Bugs/Wiring/Vollständigkeit/Korrektheit
- keinen **Task-Completion-Auditor**, der prüft, ob die Task wirklich erledigt wurde (Tiefe statt Mindeststandard)
- keinen **Anti-Pattern-Log** für „Fehler, die Claude beim Coden gemacht hat und nicht nochmal machen soll"
- keine **automatische Erzwingung** der Review-Empfehlungen aus `.claude/rules/90-subagents-hooks-and-automation.md` — sie sind „proaktiv empfohlen", nicht durchgesetzt

Konsequenz: Fehlerklassen wie der pandas `.where(Series)` Row-Index-Bug (2026-05-05) oder PIT-Look-Ahead-Probleme können wiederkehren, weil keine Instanz konsequent „hat das schonmal passiert?" prüft.

## 2. Ziele

1. **Erzwingung:** Nach jedem logischen Step mit Code-Änderungen läuft eine Review-Kette automatisch, nicht skip-able durch Vergessen.
2. **Schichtung:** Bestehende Spezialisten bleiben unverändert und sind Stage 1; zwei neue Opus-Reviewer sind Stage 2 und 3.
3. **Strukturierte Findings:** Alle Reviewer liefern ein einheitliches Findings-Schema, keine Freitext-Konversation.
4. **Anti-Pattern-Lernen:** Jeder erkannte wiederholungswürdige Fehler wird in `docs/CLAUDE_CODING_ERRORS.md` festgehalten und beim nächsten Sessionstart geladen.
5. **Keine Regelverletzung:** CLAUDE.md §2.2 „kleinster sicherer Schritt", Rule 10, Rule 60 bleiben bindend. Auditor flaggt Adjacent als Follow-up, nicht als Pflicht im aktuellen Step.

## 3. Was außerhalb des Scope ist (v1)

- Zusätzliche Reviewer (security-reviewer, performance-reviewer) — später möglich, jetzt nicht.
- Auto-Update von `KNOWN_ISSUES.md` — bleibt manuell.
- Cross-Session-Statistik über Fehlerkategorien — später.
- Strafsystem für wiederholte Errors-Log-Verletzungen — später.
- Frontend, UI, Dashboards für die Findings — nicht Teil des Backends.

## 4. Komponenten

### 4.1 Neue Subagents

#### 4.1.1 `senior-code-reviewer` (Opus)

**Trigger:** Stage 2 der Review-Kette. Wird vom Stop-Hook nach Stage-1-Spezialisten dispatched.

**Input:**
- Diff der im Step geänderten Dateien
- Originale Task-Beschreibung (aus letzter User-Message oder TodoWrite)
- Alle Stage-1-Findings (strukturiert, siehe §5)
- Top-10 Einträge aus `docs/CLAUDE_CODING_ERRORS.md` (über SessionStart-Hook bereits im Kontext)

**Aufgabe:**
- Eigener Pass auf: Bugs, Wiring-Status (Imports, Registry-Einträge, Tests, Doku), Vollständigkeit (keine TODOs/Stubs hinterlassen?), Korrektheit (passt die Logik zur Task?), bekannte Anti-Patterns aus dem Errors-Log.
- Stage-1-Findings bestätigen, widersprechen oder ergänzen.
- Wenn ein neuer wiederholungs-würdiger Fehler erkannt wird: Errors-Log-Eintrag vorschlagen (siehe §6).
- **Keine** Forderung nach Scope-Erweiterung — Adjacent ist nicht sein Job, das ist der Auditor.

**Output:** Strukturierte Findings nach §5 + Liste vorgeschlagener Errors-Log-Einträge.

#### 4.1.2 `task-completion-auditor` (Opus)

**Trigger:** Stage 3 der Review-Kette. Läuft nach `senior-code-reviewer`.

**Input:**
- Originale Task-Beschreibung
- Diff
- Alle bisherigen Findings (Stage 1 + Stage 2)

**Aufgabe:**
- Prüft Qualitätstiefe **innerhalb** der Task: Edge-Cases bedacht? Tests vorhanden und passen sie zur Logik? Wiring komplett (z. B. wenn Funktion neu, ist sie in Registry/Imports/Tests/Doku?)? Hinterlässt der Step versteckte TODOs oder halbfertige Stubs?
- Flaggt Adjacent-Probleme im selben Bereich als **Follow-up-Vorschläge** (nicht als Pflicht): „im selben Modul gibt es noch X, das könnte in eigener Task adressiert werden".
- Produziert ein **Verdict** für den aktuellen Step:
  - `PASS` — Task erledigt, keine offenen BLOCKER/MAJOR
  - `CONDITIONAL` — Task im Kern erledigt, aber MAJOR-Findings müssen vor Abschluss adressiert werden
  - `FAIL` — Task nicht erledigt oder BLOCKER vorhanden

**Output:** Strukturierte Findings nach §5 (Kategorie `completeness` und `scope`) + Verdict + Follow-up-Vorschläge separat markiert.

### 4.2 Neue Hooks

#### 4.2.1 `.claude/hooks/stop_review_chain.py` (Stop-Hook)

**Trigger:** Stop-Event (Assistant beendet Antwort).

**Logik:**
1. Lies Transcript-Excerpt des letzten Turns.
2. Prüfe: gab es `Edit`-, `Write`- oder `Bash`-Tool-Calls mit Schreibwirkung auf Pfade in:
   - `src/**`
   - `scripts/**`
   - `.github/workflows/**`
   - `.claude/rules/**`
   - `CLAUDE.md`
3. Wenn nein: skip, kein Review nötig.
4. Wenn ja:
   a. Identifiziere relevante Spezialisten basierend auf Pfaden:
      - `src/assembled_core/execution/**` ∪ `risk/**` ∪ `pipeline/**` ∪ `accounting/**` ∪ `portfolio/**` → `risk-execution-reviewer`
      - jede Änderung in `src/**` oder `scripts/**` → `test-runner`
      - `.github/workflows/**` → `ci-debugger`
      - `CLAUDE.md` oder `.claude/rules/**` → `docs-governance-sync`
   b. Dispatche Stage-1-Spezialisten **parallel** mit identischem Diff-Input und Findings-Schema-Spec.
   c. Sammle Stage-1-Findings.
   d. Dispatche `senior-code-reviewer` (Stage 2) mit Diff + Stage-1-Findings + Errors-Log-Top-10.
   e. Dispatche `task-completion-auditor` (Stage 3) mit Diff + Original-Task + alle bisherigen Findings.
   f. Konsolidiere alle Findings → Markdown-Block → an Hauptagent zurück.
5. Hauptagent (ich) verarbeitet:
   - BLOCKER → muss vor User-Rückmeldung gefixt werden
   - MAJOR → muss adressiert oder explizit dokumentiert/akzeptiert werden
   - MINOR/INFO → optional, dokumentieren
   - Verdict `FAIL` → Step gilt als nicht abgeschlossen, weiter arbeiten
   - Errors-Log-Vorschläge → in `docs/CLAUDE_CODING_ERRORS.md` appenden

**Wichtig:** Hook darf den Stop-Event blockieren bis Review-Kette läuft. Findings werden als System-Message in den Hauptkontext zurückgespielt.

**Performance-Guards:**
- Wenn Diff > 2000 Zeilen: Warnung im Findings-Block (zu groß für einen Step, vermutlich Regel-10-Verstoß). Volle Kette läuft trotzdem.
- Wenn **nur** Doku-Dateien geändert (`docs/**`, `*.md` außer CLAUDE.md/Rules): Stage 1 = `docs-governance-sync` falls Governance-Datei, sonst skip. Stage 2+3 werden **übersprungen** (kein Code-Risiko).
- Wenn **nur** Test-Dateien geändert (`tests/**`): Stage 1 = `test-runner`. Stage 2 läuft (test-correctness ist Code-Quality). Stage 3 läuft (Auditor prüft ob Tests die Original-Task abdecken).
- Wenn **nur** Lint-/Format-Änderungen erkannt (ruff/black-typische Whitespace-/Reorder-Diffs ohne Semantik-Änderung): kurzer „lint-only" Pass, keine Stage 2+3. Erkennung: Diff enthält ausschließlich Indent/Whitespace/Import-Reorder.
- Wenn Diff in beliebiger Kombination geschützter Pfade Code-Semantik berührt: **volle Kette** (Stage 1 + 2 + 3) — Default, keine Ausnahme.

#### 4.2.2 `.claude/hooks/session_start_load_errors.py` (SessionStart-Hook)

**Trigger:** SessionStart-Event.

**Logik:**
1. Lies `docs/CLAUDE_CODING_ERRORS.md`.
2. Parse Einträge (Markdown-Headings `## E-NNN`).
3. Sortiere nach Datum descending.
4. Wähle Top-10.
5. Output als System-Reminder-Block:
   ```
   # Claude Coding Errors — Top 10 (vermeide diese Muster!)
   - E-042: pandas .where(Series) row-index alignment — verwende .where(Series.values)
   - E-041: PIT look-ahead durch midnight normalization — siehe data/latency.py:...
   ...
   ```
6. Vollständige Datei bleibt on-demand lesbar.

### 4.3 Neue Datei: `docs/CLAUDE_CODING_ERRORS.md`

**Format:** Append-only Markdown. Eintrag-Schema:

```markdown
## E-NNN — Kurzer Titel
**Datum:** YYYY-MM-DD
**Kategorie:** <pandas-pitfall | pit-violation | silent-except | wiring-gap | test-anti-pattern | logic-error | other>
**Was passierte:** <Konkrete Beschreibung>
**Warum falsch:** <Mechanismus, warum es ein Bug ist>
**Wie vermeiden:** <Konkreter Vermeidungs-Pattern oder Test>
**Erkannt in:** <Dateipfade>
**Referenzen:** <Session-Memory-File, Commit-Hash, etc.>
```

**Initiale Befüllung:** Bei Implementierung des Hooks werden bekannte Anti-Patterns aus den letzten Memory-Files seeding-weise eingetragen (z. B. pandas .where(Series), PIT look-ahead, silent except).

**Pflege:** Senior-Code-Reviewer schlägt neue Einträge im Findings-Block vor. Hauptagent appendet sie nach Bestätigung.

## 5. Findings-Schema (gemeinsam für alle Reviewer)

Jeder Reviewer liefert eine Liste von Findings im folgenden YAML-Format:

```yaml
findings:
  - id: F-<reviewer-prefix>-<n>            # z. B. F-senior-1, F-auditor-2
    file: "src/assembled_core/foo/bar.py"
    line: 42                                # oder line_range: [42, 58]
    severity: BLOCKER | MAJOR | MINOR | INFO
    category: bug | wiring | completeness | correctness | pit | risk | test | docs | scope | anti-pattern
    evidence: "Konkrete Zeile, konkretes Verhalten, konkrete fehlende Kopplung"
    suggested_fix: "Was zu tun ist (kurz, präzise, kein 'warum')"
    references: ["docs/CLAUDE_CODING_ERRORS.md#E-042", "session-...md"]
verdict: PASS | CONDITIONAL | FAIL          # nur task-completion-auditor
errors_log_proposals: []                    # nur senior-code-reviewer; Liste neuer Anti-Pattern-Einträge
follow_ups: []                              # nur task-completion-auditor; Liste angrenzender Themen
```

**Severity-Definition:**
- `BLOCKER` — Step kann nicht als abgeschlossen gelten. Beispiele: Build bricht, kritischer Bug in Risk/Execution, falsches Verhalten zur Task.
- `MAJOR` — Step kann nur unter Vorbehalt abgeschlossen werden. Beispiele: fehlende Tests für neuen Pfad, fehlende Doku bei Public-API-Änderung, suboptimale aber funktionierende Lösung in sensiblen Zonen.
- `MINOR` — Sollte gefixt werden, blockiert aber nicht. Beispiele: Lint-Hinweise, naming, fehlende Type-Hints.
- `INFO` — Reine Information, keine Aktion erforderlich. Beispiele: System-Map veraltet, „im Modul gibt es zusätzlich X" (Auditor-Follow-up).

**Konsolidierter Block am Ende der Kette:**

```markdown
# Review-Chain Result — Step <hash/timestamp>

**Verdict:** PASS | CONDITIONAL | FAIL
**Files changed:** <count>, +<added>/-<removed> lines

## BLOCKER (<n>)
- F-... in <file>:<line> — <evidence> → <suggested_fix>

## MAJOR (<n>)
- ...

## MINOR (<n>)
- ...

## INFO (<n>)
- ...

## Follow-up vorgeschlagen (angrenzende Themen, separate Tasks)
- ...

## Errors-Log-Vorschläge (neue Anti-Patterns)
- E-NNN: <titel> → wird nach Bestätigung in CLAUDE_CODING_ERRORS.md appendiert.
```

## 6. Ablauf (End-to-End-Beispiel)

**Szenario:** Ich habe gerade `src/assembled_core/execution/order_router.py` um eine neue Pre-Trade-Check-Funktion erweitert.

1. Ich beende meine Antwort → Stop-Hook feuert.
2. Hook erkennt Edit in `src/assembled_core/execution/**` → Dispatch:
   - Stage 1 parallel: `risk-execution-reviewer` (weil execution/) + `test-runner` (immer wenn src/)
3. Stage 1 liefert Findings: z. B. `risk-execution-reviewer` markiert eine fehlende Notional-Validation als MAJOR.
4. Stage 2: `senior-code-reviewer` läuft.
   - Sieht die Stage-1-Findings.
   - Macht eigenen Pass: erkennt, dass die neue Funktion nicht in `__init__.py` exportiert ist (Wiring-Gap, MAJOR) und dass kein Test in `tests/test_order_router.py` existiert (test-coverage-gap, MAJOR).
   - Erkennt, dass das Pattern „Pre-Trade-Check ohne Notional-Validation" wiederholungswürdig ist → schlägt Errors-Log-Eintrag vor.
5. Stage 3: `task-completion-auditor` läuft.
   - Original-Task war: „füge Pre-Trade-Notional-Cap-Check hinzu".
   - Sieht: Funktion existiert, aber Notional-Validation fehlt (vom risk-reviewer markiert), Wiring-Gap (vom senior-reviewer markiert).
   - Verdict: `CONDITIONAL` — Kern erledigt, aber Notional-Validation und Export-Wiring vor Step-Abschluss nötig.
   - Flaggt angrenzend: „im selben Modul gibt es noch `route_order_v1`, das hat ähnliche Lücke — separate Task".
6. Konsolidierter Block kommt zu mir zurück. Ich:
   - Adressiere die zwei MAJOR-Findings.
   - Bestätige Errors-Log-Eintrag → append.
   - Melde an User: „Step abgeschlossen, Verdict PASS nach Fix der zwei MAJORs. Follow-up `route_order_v1` empfohlen, möchtest du das als nächste Task?"

## 7. Integration mit bestehenden Regeln

### 7.1 `.claude/rules/90-subagents-hooks-and-automation.md` Update

Die existierende Routing-Policy wird ergänzt um die Erzwingungs-Regel:

> Nach jedem Step mit Code-Änderungen in geschützten Pfaden läuft die Review-Kette (Stop-Hook) automatisch. Spezialist-Aufrufe sind nicht mehr „proaktiv empfohlen", sondern automatisch durch den Hook.

### 7.2 `CLAUDE.md` Update

Neuer Abschnitt §20 „Review-Chain":
- Verweis auf diese Spec
- Verweis auf `docs/CLAUDE_CODING_ERRORS.md`
- Kurzregel: „Step gilt erst als abgeschlossen wenn Verdict = PASS oder CONDITIONAL mit dokumentierter Akzeptanz."

### 7.3 Beziehung zu `KNOWN_ISSUES.md`

- `KNOWN_ISSUES.md` bleibt für **System-/Code-Issues** (z. B. „Symbol XYZ delisted", „Feature ABC noch nicht implementiert").
- `CLAUDE_CODING_ERRORS.md` ist **NEU** für **Claude-Anti-Patterns** beim Coden (z. B. „pandas .where(Series) Row-Index-Bug").
- Keine Überlappung — wenn ein System-Issue durch ein Anti-Pattern entstanden ist, kann in beiden Dateien ein Cross-Ref stehen.

## 8. System-Map-Currency-Check (Bonus, in v1 enthalten)

Im Stop-Hook zusätzliche Logik:

```
Wenn Edits in src/**:
  Lies docs/architecture/system_map/system_map.json mtime
  Wenn mtime > 30 Tage alt:
    Füge INFO-Finding hinzu: "System Map veraltet — Regeneration empfohlen via scripts/architecture/generate_system_map.py"
```

Kein Blocker, nur Reminder.

## 9. Token-Realität (offene Diskussion mit User)

- Stage 1 (Spezialisten): bereits etabliert, kein Extra-Kosten gegenüber Status quo wenn sie ohnehin laufen sollten.
- Stage 2 (senior-code-reviewer Opus): pro Step zusätzlich ca. 5k–15k Output-Token + Input-Diff.
- Stage 3 (task-completion-auditor Opus): pro Step zusätzlich ca. 3k–8k Output-Token.
- Erwartete Mehrkosten: ~10k–25k Token pro Step mit Code-Änderungen.
- User hat explizit gesagt: „es wird in die Tokenkosten gehen, das nehmen wir aber gerne in Kauf, weil das wichtig ist."

**Mitigation:**
- Performance-Guards in §4.2.1 (lint-only-Pfad, doku-only-Pfad, große-Diff-Warnung).
- Trivial-Edits (keine Code-Änderung in geschützten Pfaden) werden vom Hook gefiltert.

## 10. Was diese Spec NICHT festlegt

- Konkrete Prompt-Texte für die zwei neuen Subagents — kommen in Implementierungs-Plan.
- Genauer Python-Code der Hooks — kommt in Implementierungs-Plan.
- Migrations-Strategie für bestehende Memory-Einträge / KNOWN_ISSUES — kommt in Implementierungs-Plan.
- Test-Strategie für die Hooks selbst — kommt in Implementierungs-Plan.

## 11. Akzeptanz-Kriterien für v1

Die v1-Implementierung gilt als erfolgreich wenn:

1. `senior-code-reviewer` und `task-completion-auditor` Subagent-Definitionen existieren unter `.claude/agents/`.
2. `.claude/hooks/stop_review_chain.py` und `.claude/hooks/session_start_load_errors.py` existieren und sind in `.claude/settings.json` registriert.
3. `docs/CLAUDE_CODING_ERRORS.md` existiert mit ≥5 Seeding-Einträgen aus bekannten Memory-Files.
4. CLAUDE.md §20 und `.claude/rules/90-...` aktualisiert.
5. Manueller End-to-End-Test: ein bewusster kleiner Code-Change in `src/` triggert die volle Kette, Findings-Block erscheint, Verdict wird ausgegeben.
6. Trivial-Test: ein reiner Doku-Change triggert die Kette **nicht** (Filter funktioniert).
7. Performance-Guard-Test: ein bewusst sehr großer Diff (>2000 Zeilen) erzeugt entsprechende Warnung.

## 12. Risiken

| Risiko | Wahrscheinlichkeit | Mitigation |
|---|---|---|
| Token-Kosten höher als geschätzt | mittel | Performance-Guards, ggf. Opus → Sonnet für senior-reviewer wenn Diff klein |
| Hook blockt Stop-Event zu lange (UX-Problem) | mittel | Hook async, max-timeout setzen, bei Timeout Stage 2/3 als degradiert markieren |
| Findings-Theater (höfliche Konsens-Texte) | niedrig | strukturiertes Schema erzwingt evidence + suggested_fix |
| Auditor fordert Scope-Erweiterung trotz Regel | mittel | Prompt explizit auf Follow-up-Trennung, Tests dafür |
| Errors-Log wird zu lang und unübersichtlich | niedrig (langfristig mittel) | später: Kategorisierung + Archivierung alter Einträge |
| Konflikt zwischen Stage-1- und Stage-2-Findings | niedrig | senior-reviewer hat explizit „bestätigen/widersprechen"-Auftrag |
| Hook bricht Claude-Code-Workflow wenn buggy | mittel | gründliche Tests, Fail-Open (bei Hook-Fehler: warne, aber blockiere nicht) |

---

**Ende der Spec.**
