# Review-Chain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a mandatory review chain that triggers automatically after every coding step in protected paths — Stage 1 specialists → senior-code-reviewer → task-completion-auditor — backed by a Claude-Anti-Pattern log loaded at session start.

**Architecture:** Two new Opus subagents in `.claude/agents/`, two Python hooks in `.claude/hooks/` registered via `.claude/settings.json`, one new append-only Markdown file `docs/CLAUDE_CODING_ERRORS.md`, governance updates in CLAUDE.md and rule 90. Stop-hook blocks the assistant stop event when code edits in protected paths happened and no review-marker exists for the current turn. SessionStart-hook injects the Top-10 anti-patterns into initial context.

**Tech Stack:** Python 3.10+ (standard library only — no new deps), pytest for hook unit tests, Claude Code hook system (stdin JSON, stdout JSON for `{"decision": "block", "reason": "..."}` and `{"hookSpecificOutput": {...}}`), Markdown for human-readable files.

**Spec reference:** `docs/superpowers/specs/2026-05-14-review-chain-design.md`

**Implementation status:** COMPLETE on 2026-05-14. All 18 atomic units (= 25 plan tasks) executed. 47 hook tests green (errors_log: 6, path_classifier: 18, transcript_parser: 3, diff_classifier: 8, review_marker: 5, session_start_hook: 2, stop_hook: 5). Manual smoke verified for both SessionStart and Stop hooks. Stop hook registered live and blocking on protected-path edits. Phase D ordering deviated from plan: governance edits (Tasks 22+23) executed BEFORE Stop-hook registration (Task 20) to avoid bootstrapping conflicts within the build session.

---

## File Structure (Lock-In)

| Path | Created/Modified | Responsibility |
|---|---|---|
| `docs/CLAUDE_CODING_ERRORS.md` | NEW | Append-only Anti-Pattern register, human-readable |
| `.claude/agents/senior-code-reviewer.md` | NEW | Opus subagent definition: broad code review |
| `.claude/agents/task-completion-auditor.md` | NEW | Opus subagent definition: task completeness audit |
| `.claude/hooks/__init__.py` | NEW | Marks hooks dir as package (empty) |
| `.claude/hooks/hook_utils/__init__.py` | NEW | Shared utility package marker (empty) |
| `.claude/hooks/hook_utils/errors_log.py` | NEW | Parse `CLAUDE_CODING_ERRORS.md`, return Top-N |
| `.claude/hooks/hook_utils/path_classifier.py` | NEW | Classify edited paths → protected/unprotected, which specialists |
| `.claude/hooks/hook_utils/transcript_parser.py` | NEW | Parse Claude Code transcript JSONL, extract tool-uses in current turn |
| `.claude/hooks/hook_utils/diff_classifier.py` | NEW | Classify diff: lint-only / doc-only / test-only / full |
| `.claude/hooks/hook_utils/review_marker.py` | NEW | Write/check review-completion markers per turn |
| `.claude/hooks/session_start_load_errors.py` | NEW | SessionStart hook entry point |
| `.claude/hooks/stop_review_chain.py` | NEW | Stop hook entry point |
| `.claude/settings.json` | MODIFY | Register both hooks under `"hooks"` key |
| `tests/hooks/__init__.py` | NEW | Empty package marker |
| `tests/hooks/test_errors_log.py` | NEW | Unit tests for errors-log parser |
| `tests/hooks/test_path_classifier.py` | NEW | Unit tests for path classification |
| `tests/hooks/test_transcript_parser.py` | NEW | Unit tests for transcript parser |
| `tests/hooks/test_diff_classifier.py` | NEW | Unit tests for diff classification |
| `tests/hooks/test_review_marker.py` | NEW | Unit tests for review-marker logic |
| `tests/hooks/test_session_start_hook.py` | NEW | Integration test for SessionStart hook |
| `tests/hooks/test_stop_hook.py` | NEW | Integration test for Stop hook |
| `CLAUDE.md` | MODIFY | Add §20 Review-Chain |
| `.claude/rules/90-subagents-hooks-and-automation.md` | MODIFY | Update hook section: enforced via Stop-hook |

---

## Phase A — Foundation Files

### Task 1: Create `docs/CLAUDE_CODING_ERRORS.md` with seed entries

**Files:**
- Create: `docs/CLAUDE_CODING_ERRORS.md`

- [ ] **Step 1: Write the seed file**

Content (exact):

```markdown
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
```

- [ ] **Step 2: Verify file created**

Run: `Test-Path docs\CLAUDE_CODING_ERRORS.md`
Expected: `True`

- [ ] **Step 3: Commit**

```bash
git add docs/CLAUDE_CODING_ERRORS.md
git commit -m "feat(review-chain): seed CLAUDE_CODING_ERRORS.md with 7 known anti-patterns

Seeding from memory files:
- E-001 pandas .where(Series) row-index alignment
- E-002 PIT look-ahead via midnight normalization
- E-003 silent except Exception: pass
- E-004 empty DataFrame .iloc[-1] crash
- E-005 set_index().assign() index alignment
- E-006 datetime64[ns]/[us] platform mismatch
- E-007 dict.get(key) or default — falsy zero bug

Append-only register. SessionStart hook (later task) will load Top-10."
```

---

### Task 2: Create `senior-code-reviewer` subagent definition

**Files:**
- Create: `.claude/agents/senior-code-reviewer.md`

- [ ] **Step 1: Write the agent file**

Content (exact):

```markdown
---
name: senior-code-reviewer
description: Broad senior code reviewer for Assembled-Trading-AI. MUST BE USED after every coding step in protected paths (src/, scripts/, .github/workflows/, .claude/rules/, CLAUDE.md). Reads Stage-1 specialist findings, performs own pass on bugs, wiring, completeness, correctness, and known anti-patterns. Proposes new anti-pattern log entries. Does NOT demand scope expansion.
model: opus
---

Du bist der breite Senior-Code-Reviewer für Assembled-Trading-AI. Du läufst als Stage 2 der Review-Kette nach den domänenspezifischen Spezialisten und vor dem Task-Completion-Auditor.

## Dein Input

Du bekommst:
1. Den Diff der im Step geänderten Dateien.
2. Die ursprüngliche Task-Beschreibung (aus User-Message oder TodoWrite).
3. Strukturierte Findings der Stage-1-Spezialisten (siehe Schema unten).
4. Die Top-10 Anti-Patterns aus `docs/CLAUDE_CODING_ERRORS.md` (bereits im SessionStart-Kontext).

## Dein Auftrag

Eigener Review-Pass auf:
- **Bugs:** Logikfehler, falsche Reihenfolge, fehlende Guards, Off-by-one, Index-Alignment-Probleme, dtype-Probleme.
- **Wiring:** Imports vollständig? Neue Funktionen in `__init__.py` exportiert? Registry-Einträge da wo nötig? Tests existieren und werden vom Test-Discoverer gefunden?
- **Vollständigkeit:** Keine TODO/FIXME/`pass`-Stubs hinterlassen ohne Ticket-Referenz? Keine halben Implementierungen?
- **Korrektheit:** Macht der Code wirklich was die Task verlangt? Passen Variablennamen/Tests/Docstrings zur Logik?
- **Bekannte Anti-Patterns:** Vergleiche Diff gegen `docs/CLAUDE_CODING_ERRORS.md` — ist hier ein Muster aufgetaucht, das wir schon kennen?

## Was du NICHT tust

- Kein Forderung nach Scope-Erweiterung. Das ist Aufgabe des `task-completion-auditor`, und auch der flaggt es nur als Follow-up.
- Kein „du hättest auch noch X machen können"-Kommentar — nur Bugs im aktuell geänderten Code.
- Kein blindes Bestätigen der Stage-1-Findings. Du darfst ihnen explizit widersprechen, mit Begründung.

## Output-Format (verpflichtend)

Du gibst strukturierte Findings im exakt folgenden YAML-Format zurück:

```yaml
stage: senior-code-reviewer
findings:
  - id: F-senior-1
    file: "exakter/pfad.py"
    line: 42
    severity: BLOCKER | MAJOR | MINOR | INFO
    category: bug | wiring | completeness | correctness | pit | risk | test | docs | anti-pattern
    evidence: "Konkrete Zeile/Verhalten/Bezug. Kein Lobgesang."
    suggested_fix: "Was tun, knapp. Kein 'warum'."
    references: ["docs/CLAUDE_CODING_ERRORS.md#E-001"]  # optional
errors_log_proposals:
  - id: E-NEW-1
    title: "Kurzer Titel"
    category: pandas-pitfall | pit-violation | silent-except | wiring-gap | test-anti-pattern | logic-error | other
    what_happened: "..."
    why_wrong: "..."
    how_to_avoid: "..."
    detected_in: ["pfad.py"]
stage1_disagreements:  # optional, nur wenn du Spezialist-Findings widersprichst
  - finding_id: F-risk-2
    your_position: "Kein BLOCKER weil ..."
```

## Severity-Definition

- **BLOCKER:** Step kann nicht abgeschlossen werden. Build bricht, kritischer Risk-/Execution-Bug, falsches Verhalten zur Task.
- **MAJOR:** Step kann nur unter Vorbehalt abgeschlossen werden. Fehlende Tests, fehlende Doku bei Public-API, suboptimal in sensibler Zone.
- **MINOR:** Sollte gefixt werden, blockiert aber nicht. Lint, naming, missing type hints.
- **INFO:** Reine Information, keine Aktion erforderlich.

## Wichtige Projekt-Regeln, an die du dich hältst

- CLAUDE.md §2.2: kleinster sicherer Schritt. Du forderst keine Scope-Ausweitung.
- Rule 10: kein großer Refactor ohne Auftrag.
- Rule 30: sensible Zonen (execution/risk/pipeline/accounting/portfolio) bekommen strengere Bewertung.
- Rule 40: keine erfundenen Testresultate, keine „lokal grün = CI grün"-Behauptungen.
- Rule 85: knappe, evidence-basierte Findings, kein Marketing-Text.

Bei Unsicherheit über die Severity: lieber konservativ höher (MAJOR statt MINOR), nie niedriger.
```

- [ ] **Step 2: Commit**

```bash
git add .claude/agents/senior-code-reviewer.md
git commit -m "feat(review-chain): add senior-code-reviewer Opus subagent

Stage 2 reviewer for review chain. Reads Stage-1 specialist findings + diff +
task description + Top-10 anti-patterns. Performs broad review for bugs,
wiring, completeness, correctness, anti-pattern recurrence. Strict YAML
findings output schema. Does NOT demand scope expansion (per CLAUDE.md §2.2)."
```

---

### Task 3: Create `task-completion-auditor` subagent definition

**Files:**
- Create: `.claude/agents/task-completion-auditor.md`

- [ ] **Step 1: Write the agent file**

Content (exact):

```markdown
---
name: task-completion-auditor
description: Task-completion auditor for Assembled-Trading-AI. MUST BE USED as Stage 3 of the review chain after senior-code-reviewer. Reads original task, diff, and all prior findings. Verifies depth-within-task (not just minimum standard), flags adjacent issues as separate follow-up tasks (NEVER as in-scope demands). Issues PASS/CONDITIONAL/FAIL verdict.
model: opus
---

Du bist der Task-Completion-Auditor. Du läufst als Stage 3 der Review-Kette, nach den Spezialisten (Stage 1) und dem `senior-code-reviewer` (Stage 2).

## Dein Input

Du bekommst:
1. Die ursprüngliche Task-Beschreibung (aus User-Message oder TodoWrite).
2. Den Diff der geänderten Dateien.
3. Alle bisherigen Findings aus Stage 1 und Stage 2.

## Dein Auftrag

### Hauptfrage (das eigentliche Audit)

Wurde die Task **wirklich erledigt** — mit Qualitätstiefe, nicht nur Mindeststandard?

Konkret prüfen:
- **Erfüllung:** Tut der Code wirklich, was die Task verlangt? Oder gibt es eine Lücke zwischen Anspruch und Implementierung?
- **Edge-Cases:** Sind offensichtliche Randbedingungen bedacht (leere Inputs, NaN/None, Tz-aware vs naive, große/kleine Inputs, sensible Zonen-Spezifika)?
- **Wiring komplett:** Wenn neue Funktion → exportiert, importiert, getestet, dokumentiert? Wenn Config-Key → in Schema/Validator/Defaults? Wenn neue Datei → in passender Registry?
- **Keine versteckten Stubs:** Keine `pass`/`TODO`/`raise NotImplementedError` ohne Ticket? Keine `return None`-Platzhalter wo Logik gehört?
- **Tests passen zur Logik:** Wenn neue Logik → existieren Tests, decken sie das beworbene Verhalten ab, sind sie ausführbar?
- **Doku-Sync:** Wenn API/Verhalten geändert → ist Doku angepasst (wo nötig)?

### Nebenfrage: Adjacent-Themen

Gibt es im **selben Modul / direkt angrenzenden Bereich** offensichtliche Probleme oder Lücken, die NICHT Teil der aktuellen Task sind?

Wenn ja: als `follow_ups` ausgeben, NICHT als Forderung im aktuellen Step. Beispiel:

> `follow_ups`: „Im gleichen Modul existiert `route_order_v1` mit derselben Notional-Cap-Lücke. Separate Task empfohlen."

## Was du NICHT tust

- **NICHT Scope erweitern.** CLAUDE.md §2.2 ist bindend: „kleinster sicherer Schritt". Adjacent-Probleme sind Follow-ups, nie Pflichten.
- **NICHT in den Stage-2-Pass eingreifen.** Bugs/Wiring sind Senior-Reviewer-Domäne. Du audit-est *Erfüllung der Task*.
- **NICHT „du hättest auch X machen sollen"** sagen, wenn X außerhalb der Task war.

## Output-Format (verpflichtend)

```yaml
stage: task-completion-auditor
verdict: PASS | CONDITIONAL | FAIL
verdict_reason: "Ein Satz, warum dieses Verdict."
findings:
  - id: F-auditor-1
    file: "pfad.py"
    line: 42
    severity: BLOCKER | MAJOR | MINOR | INFO
    category: completeness | scope | correctness | test | docs
    evidence: "Konkrete Lücke zwischen Task-Anspruch und Implementierung."
    suggested_fix: "Was muss noch passieren damit Task erledigt ist."
follow_ups:
  - title: "Kurze Beschreibung"
    rationale: "Warum getrennte Task statt jetzt"
    suggested_task: "Wenn-User-will-dann-so-formuliert: ..."
    files: ["pfad.py"]
```

## Verdict-Definition

- **PASS:** Task ist erledigt. Keine BLOCKER, keine MAJOR-completeness-Findings. Eventuelle Follow-ups sind separat dokumentiert.
- **CONDITIONAL:** Task im Kern erledigt, aber MAJOR-completeness-Findings müssen vor Step-Abschluss adressiert werden (z. B. „Funktion existiert aber nicht exportiert", „Tests fehlen für neuen Pfad").
- **FAIL:** Task ist nicht erledigt. BLOCKER vorhanden oder Kern-Anforderung nicht umgesetzt.

## Wichtige Projekt-Regeln, an die du dich hältst

- CLAUDE.md §2.2 (kleinster sicherer Schritt) — bindend.
- Rule 10 (kein Refactor ohne Auftrag) — bindend.
- Rule 60 (ein Problem pro Änderung) — bindend.
- Rule 85 (knappe Antworten) — Findings ohne Prosa-Fluff.

Bei Unsicherheit zwischen PASS und CONDITIONAL: lieber CONDITIONAL. Bei Unsicherheit zwischen CONDITIONAL und FAIL: lieber CONDITIONAL und MAJOR-Finding, statt FAIL ohne Beweis.
```

- [ ] **Step 2: Commit**

```bash
git add .claude/agents/task-completion-auditor.md
git commit -m "feat(review-chain): add task-completion-auditor Opus subagent

Stage 3 reviewer. Audits whether original task was actually completed with
depth-within-scope (not just minimum standard). Flags adjacent issues as
separate follow-up tasks, never as in-scope demands. Issues PASS/CONDITIONAL/
FAIL verdict. Strict adherence to CLAUDE.md §2.2 (kleinster sicherer Schritt)."
```

---

## Phase B — SessionStart Hook (errors-log loader)

### Task 4: Test for errors-log parser

**Files:**
- Create: `tests/hooks/__init__.py` (empty)
- Create: `tests/hooks/test_errors_log.py`

- [ ] **Step 1: Create empty `__init__.py`**

File: `tests/hooks/__init__.py`
Content: (empty file)

- [ ] **Step 2: Write the failing tests**

File: `tests/hooks/test_errors_log.py`
Content:

```python
"""Tests for the errors-log parser used by the SessionStart hook."""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

# Make .claude/hooks importable for tests
HOOKS_DIR = Path(__file__).resolve().parents[2] / ".claude" / "hooks"
sys.path.insert(0, str(HOOKS_DIR))

from hook_utils.errors_log import parse_errors_log, top_n_entries  # noqa: E402


def _write_log(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "CLAUDE_CODING_ERRORS.md"
    p.write_text(textwrap.dedent(body).lstrip(), encoding="utf-8")
    return p


def test_parse_empty_file_returns_empty_list(tmp_path):
    p = _write_log(tmp_path, "# Claude Coding Errors\n\nNo entries yet.\n")
    assert parse_errors_log(p) == []


def test_parse_single_entry(tmp_path):
    body = """
    # Claude Coding Errors

    ## E-001 — pandas .where(Series) row-index alignment bug
    **Datum:** 2026-05-05
    **Kategorie:** pandas-pitfall
    **Was passierte:** z-score broke.
    **Warum falsch:** alignment.
    **Wie vermeiden:** use .values.
    **Erkannt in:** src/foo.py
    **Referenzen:** memory/...md
    """
    p = _write_log(tmp_path, body)
    entries = parse_errors_log(p)
    assert len(entries) == 1
    e = entries[0]
    assert e["id"] == "E-001"
    assert e["title"] == "pandas .where(Series) row-index alignment bug"
    assert e["datum"] == "2026-05-05"
    assert e["kategorie"] == "pandas-pitfall"
    assert "use .values" in e["how_to_avoid"]


def test_parse_multiple_entries_in_order(tmp_path):
    body = """
    # Claude Coding Errors

    ## E-001 — First
    **Datum:** 2026-05-01
    **Kategorie:** other
    **Was passierte:** a
    **Warum falsch:** b
    **Wie vermeiden:** c
    **Erkannt in:** x
    **Referenzen:** y

    ## E-002 — Second
    **Datum:** 2026-05-02
    **Kategorie:** other
    **Was passierte:** d
    **Warum falsch:** e
    **Wie vermeiden:** f
    **Erkannt in:** z
    **Referenzen:** w
    """
    p = _write_log(tmp_path, body)
    entries = parse_errors_log(p)
    assert [e["id"] for e in entries] == ["E-001", "E-002"]


def test_top_n_returns_most_recent_by_date(tmp_path):
    body = """
    # Claude Coding Errors

    ## E-001 — Old
    **Datum:** 2026-01-01
    **Kategorie:** other
    **Was passierte:** a
    **Warum falsch:** b
    **Wie vermeiden:** c
    **Erkannt in:** x
    **Referenzen:** y

    ## E-002 — New
    **Datum:** 2026-05-01
    **Kategorie:** other
    **Was passierte:** d
    **Warum falsch:** e
    **Wie vermeiden:** f
    **Erkannt in:** z
    **Referenzen:** w

    ## E-003 — Middle
    **Datum:** 2026-03-01
    **Kategorie:** other
    **Was passierte:** g
    **Warum falsch:** h
    **Wie vermeiden:** i
    **Erkannt in:** q
    **Referenzen:** r
    """
    p = _write_log(tmp_path, body)
    entries = parse_errors_log(p)
    top2 = top_n_entries(entries, n=2)
    assert [e["id"] for e in top2] == ["E-002", "E-003"]


def test_top_n_handles_fewer_entries_than_n(tmp_path):
    body = """
    # Claude Coding Errors

    ## E-001 — Solo
    **Datum:** 2026-01-01
    **Kategorie:** other
    **Was passierte:** a
    **Warum falsch:** b
    **Wie vermeiden:** c
    **Erkannt in:** x
    **Referenzen:** y
    """
    p = _write_log(tmp_path, body)
    entries = parse_errors_log(p)
    assert len(top_n_entries(entries, n=10)) == 1


def test_missing_file_returns_empty(tmp_path):
    assert parse_errors_log(tmp_path / "does-not-exist.md") == []
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/hooks/test_errors_log.py -v`
Expected: All 6 tests FAIL with `ModuleNotFoundError: No module named 'hook_utils.errors_log'`

- [ ] **Step 4: Commit failing tests**

```bash
git add tests/hooks/__init__.py tests/hooks/test_errors_log.py
git commit -m "test(review-chain): failing tests for errors_log parser"
```

---

### Task 5: Implement errors-log parser

**Files:**
- Create: `.claude/hooks/__init__.py` (empty)
- Create: `.claude/hooks/hook_utils/__init__.py` (empty)
- Create: `.claude/hooks/hook_utils/errors_log.py`

- [ ] **Step 1: Create empty package markers**

File: `.claude/hooks/__init__.py` — empty
File: `.claude/hooks/hook_utils/__init__.py` — empty

- [ ] **Step 2: Implement parser**

File: `.claude/hooks/hook_utils/errors_log.py`
Content:

```python
"""Parse docs/CLAUDE_CODING_ERRORS.md into structured entries.

Schema per entry (matches spec §4.3):

    {
        "id": "E-001",
        "title": "...",
        "datum": "2026-05-05",
        "kategorie": "pandas-pitfall",
        "what_happened": "...",
        "why_wrong": "...",
        "how_to_avoid": "...",
        "detected_in": "...",
        "references": "...",
    }
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Dict

ENTRY_HEADING_RE = re.compile(r"^##\s+(E-\d+)\s+—\s+(.+?)\s*$")
FIELD_RE = re.compile(r"^\*\*(.+?):\*\*\s*(.*)$")

FIELD_MAP = {
    "Datum": "datum",
    "Kategorie": "kategorie",
    "Was passierte": "what_happened",
    "Warum falsch": "why_wrong",
    "Wie vermeiden": "how_to_avoid",
    "Erkannt in": "detected_in",
    "Referenzen": "references",
}


def parse_errors_log(path: Path) -> List[Dict[str, str]]:
    """Parse the errors-log markdown file. Return list of entries in file order."""
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    entries: List[Dict[str, str]] = []
    current: Dict[str, str] | None = None

    for line in lines:
        m = ENTRY_HEADING_RE.match(line)
        if m:
            if current is not None:
                entries.append(current)
            current = {"id": m.group(1), "title": m.group(2).strip()}
            continue
        if current is None:
            continue
        fm = FIELD_RE.match(line)
        if fm:
            label = fm.group(1).strip()
            value = fm.group(2).strip()
            key = FIELD_MAP.get(label)
            if key:
                current[key] = value

    if current is not None:
        entries.append(current)

    return entries


def top_n_entries(entries: List[Dict[str, str]], n: int = 10) -> List[Dict[str, str]]:
    """Return up to n entries sorted by 'datum' descending (newest first)."""
    def _key(e: Dict[str, str]) -> str:
        return e.get("datum", "0000-00-00")
    return sorted(entries, key=_key, reverse=True)[:n]
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `pytest tests/hooks/test_errors_log.py -v`
Expected: All 6 tests PASS

- [ ] **Step 4: Commit**

```bash
git add .claude/hooks/__init__.py .claude/hooks/hook_utils/__init__.py .claude/hooks/hook_utils/errors_log.py
git commit -m "feat(review-chain): implement errors_log parser

Parses docs/CLAUDE_CODING_ERRORS.md into structured entries. top_n_entries()
returns most-recent-first by Datum field. Handles missing file gracefully."
```

---

### Task 6: Test the SessionStart hook output

**Files:**
- Create: `tests/hooks/test_session_start_hook.py`

- [ ] **Step 1: Write the failing test**

File: `tests/hooks/test_session_start_hook.py`
Content:

```python
"""Integration test for the SessionStart hook."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
HOOK_PATH = REPO_ROOT / ".claude" / "hooks" / "session_start_load_errors.py"


def test_hook_outputs_top10_entries_as_additional_context(tmp_path, monkeypatch):
    # Write a minimal errors-log fixture
    log_path = tmp_path / "CLAUDE_CODING_ERRORS.md"
    log_path.write_text(
        "# Claude Coding Errors\n\n"
        "## E-001 — Test pattern\n"
        "**Datum:** 2026-05-05\n"
        "**Kategorie:** pandas-pitfall\n"
        "**Was passierte:** something broke\n"
        "**Warum falsch:** wrong assumption\n"
        "**Wie vermeiden:** do X instead\n"
        "**Erkannt in:** src/foo.py\n"
        "**Referenzen:** memory/x.md\n",
        encoding="utf-8",
    )

    env = {"CLAUDE_HOOKS_ERRORS_LOG_PATH": str(log_path), "PYTHONIOENCODING": "utf-8"}
    # Hook receives minimal SessionStart event
    hook_input = json.dumps({"session_id": "test", "transcript_path": ""})
    result = subprocess.run(
        [sys.executable, str(HOOK_PATH)],
        input=hook_input,
        capture_output=True,
        text=True,
        env={**env},
    )

    assert result.returncode == 0, f"stderr: {result.stderr}"
    payload = json.loads(result.stdout)
    out = payload["hookSpecificOutput"]
    assert out["hookEventName"] == "SessionStart"
    assert "E-001" in out["additionalContext"]
    assert "do X instead" in out["additionalContext"]


def test_hook_with_missing_log_outputs_empty_section(tmp_path):
    env = {
        "CLAUDE_HOOKS_ERRORS_LOG_PATH": str(tmp_path / "nope.md"),
        "PYTHONIOENCODING": "utf-8",
    }
    hook_input = json.dumps({"session_id": "test", "transcript_path": ""})
    result = subprocess.run(
        [sys.executable, str(HOOK_PATH)],
        input=hook_input,
        capture_output=True,
        text=True,
        env={**env},
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    # Empty log → still valid output, just empty entries section
    assert "additionalContext" in payload["hookSpecificOutput"]
    assert "Top 10" in payload["hookSpecificOutput"]["additionalContext"] or \
           "keine Einträge" in payload["hookSpecificOutput"]["additionalContext"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/hooks/test_session_start_hook.py -v`
Expected: Both tests FAIL with `FileNotFoundError` for `session_start_load_errors.py`

- [ ] **Step 3: Commit failing tests**

```bash
git add tests/hooks/test_session_start_hook.py
git commit -m "test(review-chain): failing test for SessionStart hook"
```

---

### Task 7: Implement SessionStart hook

**Files:**
- Create: `.claude/hooks/session_start_load_errors.py`

- [ ] **Step 1: Implement the hook**

File: `.claude/hooks/session_start_load_errors.py`
Content:

```python
#!/usr/bin/env python3
"""SessionStart hook: load Top-10 Claude coding anti-patterns into context.

Reads docs/CLAUDE_CODING_ERRORS.md, parses entries, returns the 10 most recent
as additionalContext via Claude Code's SessionStart hook output schema.

Env override: CLAUDE_HOOKS_ERRORS_LOG_PATH points to an alternative log path
(used in tests).

Output schema (Claude Code hooks):
    {"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": "..."}}
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Make hook_utils importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hook_utils.errors_log import parse_errors_log, top_n_entries  # noqa: E402


def _resolve_log_path() -> Path:
    override = os.environ.get("CLAUDE_HOOKS_ERRORS_LOG_PATH")
    if override:
        return Path(override)
    # Default: docs/CLAUDE_CODING_ERRORS.md relative to repo root.
    # Repo root is two levels up from this file (.claude/hooks/).
    return Path(__file__).resolve().parents[2] / "docs" / "CLAUDE_CODING_ERRORS.md"


def _format_entries(entries: list[dict]) -> str:
    if not entries:
        return (
            "# Claude Coding Errors — Top 10\n"
            "(noch keine Einträge — Anti-Pattern-Register ist leer)\n"
        )
    lines = ["# Claude Coding Errors — Top 10 (vermeide diese Muster!)"]
    for e in entries:
        lines.append(
            f"- **{e['id']}** ({e.get('datum', '?')}, {e.get('kategorie', '?')}): "
            f"{e.get('title', '')} — *Wie vermeiden:* {e.get('how_to_avoid', '?')}"
        )
    lines.append(
        "\n*Volle Details in docs/CLAUDE_CODING_ERRORS.md. "
        "Wenn ein neuer Anti-Pattern erkannt wird → senior-code-reviewer schlägt Eintrag vor.*"
    )
    return "\n".join(lines)


def main() -> int:
    # Drain stdin (Claude Code provides JSON event), we don't need its fields
    try:
        sys.stdin.read()
    except Exception:
        pass

    log_path = _resolve_log_path()
    entries = parse_errors_log(log_path)
    top10 = top_n_entries(entries, n=10)
    context = _format_entries(top10)

    payload = {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": context,
        }
    }
    sys.stdout.write(json.dumps(payload))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `pytest tests/hooks/test_session_start_hook.py -v`
Expected: Both tests PASS

- [ ] **Step 3: Commit**

```bash
git add .claude/hooks/session_start_load_errors.py
git commit -m "feat(review-chain): implement SessionStart hook for errors-log

Reads docs/CLAUDE_CODING_ERRORS.md, returns Top-10 most recent entries as
additionalContext. Env override CLAUDE_HOOKS_ERRORS_LOG_PATH for tests.
Outputs Claude Code hook JSON schema."
```

---

### Task 8: Register SessionStart hook in settings.json

**Files:**
- Modify: `.claude/settings.json`

- [ ] **Step 1: Read current settings**

Run: `Get-Content .claude\settings.json`
Expected current content:
```json
{
  "permissions": {
    "defaultMode": "bypassPermissions"
  },
  "enabledPlugins": {
    ...
  }
}
```

- [ ] **Step 2: Add hooks section**

Modify `.claude/settings.json` to add a `"hooks"` key. Final content:

```json
{
  "permissions": {
    "defaultMode": "bypassPermissions"
  },
  "enabledPlugins": {
    "claude-mem@thedotmack": true,
    "frontend-design@claude-plugins-official": true,
    "superpowers@claude-plugins-official": true,
    "github@claude-plugins-official": true
  },
  "hooks": {
    "SessionStart": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "python .claude/hooks/session_start_load_errors.py"
          }
        ]
      }
    ]
  }
}
```

- [ ] **Step 3: Verify JSON is valid**

Run: `python -c "import json; json.load(open('.claude/settings.json'))"`
Expected: no error

- [ ] **Step 4: Commit**

```bash
git add .claude/settings.json
git commit -m "feat(review-chain): register SessionStart hook in settings.json

Hook fires once per new session, injects Top-10 Claude coding anti-patterns
from docs/CLAUDE_CODING_ERRORS.md into the initial context."
```

---

### Task 9: Manual smoke test for SessionStart hook

- [ ] **Step 1: Invoke the hook manually**

Run (PowerShell):
```powershell
echo '{"session_id":"smoke","transcript_path":""}' | python .claude/hooks/session_start_load_errors.py
```

Expected output (JSON one-liner):
```json
{"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": "# Claude Coding Errors — Top 10 (vermeide diese Muster!)\n- **E-007** ..."}}
```

- [ ] **Step 2: Verify Top-10 entries are present**

The additionalContext should list entries E-001 through E-007 (since we seeded 7), most recent first.

- [ ] **Step 3: No commit** — smoke test only.

---

## Phase C — Stop-Hook Utilities (TDD each piece)

### Task 10: Test path-classifier

**Files:**
- Create: `tests/hooks/test_path_classifier.py`

- [ ] **Step 1: Write the failing tests**

File: `tests/hooks/test_path_classifier.py`
Content:

```python
"""Tests for the path classifier used by the Stop hook."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

HOOKS_DIR = Path(__file__).resolve().parents[2] / ".claude" / "hooks"
sys.path.insert(0, str(HOOKS_DIR))

from hook_utils.path_classifier import (  # noqa: E402
    is_protected_path,
    specialists_for_paths,
)


@pytest.mark.parametrize("path,expected", [
    ("src/assembled_core/execution/order_router.py", True),
    ("src/assembled_core/data/foo.py", True),
    ("scripts/run_backtest.py", True),
    (".github/workflows/ci.yml", True),
    (".claude/rules/10-core.md", True),
    ("CLAUDE.md", True),
    ("docs/README.md", False),
    ("tests/test_foo.py", False),
    ("output/equity.csv", False),
    ("README.md", False),
    (".claude/agents/foo.md", False),  # agents are governance but NOT in spec list
])
def test_is_protected_path(path, expected):
    assert is_protected_path(path) is expected


def test_specialists_for_execution_path():
    paths = ["src/assembled_core/execution/order_router.py"]
    specs = specialists_for_paths(paths)
    assert "risk-execution-reviewer" in specs
    assert "test-runner" in specs


def test_specialists_for_risk_pipeline_accounting():
    for sub in ["risk", "pipeline", "accounting", "portfolio", "paper"]:
        specs = specialists_for_paths([f"src/assembled_core/{sub}/x.py"])
        assert "risk-execution-reviewer" in specs, f"missing for {sub}"
        assert "test-runner" in specs


def test_specialists_for_workflow_change():
    specs = specialists_for_paths([".github/workflows/ci.yml"])
    assert "ci-debugger" in specs


def test_specialists_for_governance_change():
    specs = specialists_for_paths(["CLAUDE.md"])
    assert "docs-governance-sync" in specs
    specs2 = specialists_for_paths([".claude/rules/10-core.md"])
    assert "docs-governance-sync" in specs2


def test_specialists_for_pure_utility_code():
    """Plain src/ code without sensitive zone: only test-runner."""
    specs = specialists_for_paths(["src/assembled_core/utils/format.py"])
    assert specs == {"test-runner"}


def test_specialists_for_mixed_paths():
    specs = specialists_for_paths([
        "src/assembled_core/execution/router.py",
        ".github/workflows/ci.yml",
    ])
    assert "risk-execution-reviewer" in specs
    assert "ci-debugger" in specs
    assert "test-runner" in specs


def test_no_protected_paths_returns_empty_specialists():
    specs = specialists_for_paths(["docs/foo.md", "output/x.csv"])
    assert specs == set()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/hooks/test_path_classifier.py -v`
Expected: All tests FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Commit failing tests**

```bash
git add tests/hooks/test_path_classifier.py
git commit -m "test(review-chain): failing tests for path_classifier"
```

---

### Task 11: Implement path-classifier

**Files:**
- Create: `.claude/hooks/hook_utils/path_classifier.py`

- [ ] **Step 1: Implement classifier**

File: `.claude/hooks/hook_utils/path_classifier.py`
Content:

```python
"""Classify edited file paths for the Stop-hook review chain.

Two questions:
1. is_protected_path(path) → bool: should the review chain run at all?
2. specialists_for_paths(paths) → set[str]: which Stage-1 specialists?

Path families (per spec §4.2.1):
- src/**                       → test-runner (always)
- src/assembled_core/{execution,risk,pipeline,accounting,portfolio,paper}/**
                               → also risk-execution-reviewer
- scripts/**                   → test-runner
- .github/workflows/**         → ci-debugger
- CLAUDE.md, .claude/rules/**  → docs-governance-sync
"""
from __future__ import annotations

from pathlib import PurePosixPath
from typing import Iterable, Set

PROTECTED_PREFIXES = (
    "src/",
    "scripts/",
    ".github/workflows/",
    ".claude/rules/",
)

SENSITIVE_ZONES = (
    "src/assembled_core/execution/",
    "src/assembled_core/risk/",
    "src/assembled_core/pipeline/",
    "src/assembled_core/accounting/",
    "src/assembled_core/portfolio/",
    "src/assembled_core/paper/",
)


def _norm(path: str) -> str:
    """Normalize path to posix-style relative path."""
    return str(PurePosixPath(path.replace("\\", "/")))


def is_protected_path(path: str) -> bool:
    """Return True iff editing this path should trigger the review chain."""
    p = _norm(path)
    if p == "CLAUDE.md":
        return True
    return any(p.startswith(prefix) for prefix in PROTECTED_PREFIXES)


def specialists_for_paths(paths: Iterable[str]) -> Set[str]:
    """Return the set of specialist subagent names that should run Stage 1.

    Empty set means no protected paths → review chain should not run at all.
    """
    specs: Set[str] = set()
    for raw in paths:
        if not is_protected_path(raw):
            continue
        p = _norm(raw)

        # test-runner: any code change in src/ or scripts/
        if p.startswith("src/") or p.startswith("scripts/"):
            specs.add("test-runner")

        # risk-execution-reviewer: sensitive zones
        if any(p.startswith(zone) for zone in SENSITIVE_ZONES):
            specs.add("risk-execution-reviewer")

        # ci-debugger: workflow changes
        if p.startswith(".github/workflows/"):
            specs.add("ci-debugger")

        # docs-governance-sync: CLAUDE.md and rules
        if p == "CLAUDE.md" or p.startswith(".claude/rules/"):
            specs.add("docs-governance-sync")

    return specs
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `pytest tests/hooks/test_path_classifier.py -v`
Expected: All tests PASS.

- [ ] **Step 3: Commit**

```bash
git add .claude/hooks/hook_utils/path_classifier.py
git commit -m "feat(review-chain): implement path_classifier

is_protected_path: src/, scripts/, .github/workflows/, .claude/rules/, CLAUDE.md.
specialists_for_paths: test-runner (default), risk-execution-reviewer (sensitive
zones), ci-debugger (workflows), docs-governance-sync (governance docs)."
```

---

### Task 12: Test transcript-parser

**Files:**
- Create: `tests/hooks/test_transcript_parser.py`
- Create: `tests/hooks/fixtures/transcript_with_edits.jsonl` (test fixture)

- [ ] **Step 1: Create the fixture**

File: `tests/hooks/fixtures/transcript_with_edits.jsonl`
Content (3 lines, exact, each is a JSON object on its own line):

```jsonl
{"type":"user","message":{"role":"user","content":"please fix the bug"},"uuid":"u1"}
{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","name":"Edit","input":{"file_path":"F:/Python_Projekt/Aktiengeruest/src/assembled_core/execution/router.py","old_string":"a","new_string":"b"}}]},"uuid":"a1"}
{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","name":"Write","input":{"file_path":"F:/Python_Projekt/Aktiengeruest/docs/foo.md","content":"x"}}]},"uuid":"a2"}
```

- [ ] **Step 2: Write the failing tests**

File: `tests/hooks/test_transcript_parser.py`
Content:

```python
"""Tests for transcript_parser: extract edited paths from Claude Code transcript JSONL."""
from __future__ import annotations

import sys
from pathlib import Path

HOOKS_DIR = Path(__file__).resolve().parents[2] / ".claude" / "hooks"
sys.path.insert(0, str(HOOKS_DIR))

from hook_utils.transcript_parser import edited_paths_in_last_turn  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "transcript_with_edits.jsonl"
REPO_ROOT_FAKE = Path("F:/Python_Projekt/Aktiengeruest")


def test_extracts_edit_and_write_paths_relative_to_repo_root():
    paths = edited_paths_in_last_turn(FIXTURE, repo_root=REPO_ROOT_FAKE)
    assert "src/assembled_core/execution/router.py" in paths
    assert "docs/foo.md" in paths


def test_missing_transcript_returns_empty(tmp_path):
    assert edited_paths_in_last_turn(tmp_path / "nope.jsonl", repo_root=REPO_ROOT_FAKE) == []


def test_only_returns_paths_from_last_assistant_turn(tmp_path):
    """If a user message follows the last assistant edits, last 'turn' is the
    contiguous trailing assistant messages."""
    p = tmp_path / "t.jsonl"
    p.write_text(
        '{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","name":"Edit","input":{"file_path":"F:/Python_Projekt/Aktiengeruest/src/old.py","old_string":"a","new_string":"b"}}]},"uuid":"a-old"}\n'
        '{"type":"user","message":{"role":"user","content":"new request"},"uuid":"u-new"}\n'
        '{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","name":"Edit","input":{"file_path":"F:/Python_Projekt/Aktiengeruest/src/new.py","old_string":"a","new_string":"b"}}]},"uuid":"a-new"}\n',
        encoding="utf-8",
    )
    paths = edited_paths_in_last_turn(p, repo_root=REPO_ROOT_FAKE)
    assert "src/new.py" in paths
    assert "src/old.py" not in paths
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/hooks/test_transcript_parser.py -v`
Expected: All tests FAIL with ModuleNotFoundError.

- [ ] **Step 4: Commit failing tests**

```bash
git add tests/hooks/test_transcript_parser.py tests/hooks/fixtures/
git commit -m "test(review-chain): failing tests for transcript_parser"
```

---

### Task 13: Implement transcript-parser

**Files:**
- Create: `.claude/hooks/hook_utils/transcript_parser.py`

- [ ] **Step 1: Implement parser**

File: `.claude/hooks/hook_utils/transcript_parser.py`
Content:

```python
"""Parse Claude Code transcript JSONL to extract edited file paths.

Claude Code writes each turn as one JSONL line. Each line is a JSON object with
fields like `type` (user|assistant), `message.content` (list of content blocks),
etc.

We care about tool_use blocks with name in {Edit, Write, NotebookEdit} and an
`input.file_path` field. We return paths from the *trailing* assistant turn
(contiguous trailing assistant messages — stops at the most recent user message).
"""
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import List

EDITING_TOOLS = {"Edit", "Write", "NotebookEdit", "MultiEdit"}


def _rel_to_repo(file_path: str, repo_root: Path) -> str | None:
    """Return path relative to repo_root in posix form, or None if outside repo."""
    try:
        p = Path(file_path)
        rel = p.resolve().relative_to(repo_root.resolve())
        return str(PurePosixPath(rel))
    except (ValueError, OSError):
        # Fall back to string slicing if the file doesn't exist (e.g., tests)
        norm = file_path.replace("\\", "/")
        root_norm = str(PurePosixPath(repo_root.as_posix()))
        if norm.startswith(root_norm + "/"):
            return norm[len(root_norm) + 1 :]
        if norm.startswith(root_norm.rstrip("/") + "/"):
            return norm[len(root_norm.rstrip("/")) + 1 :]
        return None


def edited_paths_in_last_turn(transcript_path: Path, repo_root: Path) -> List[str]:
    """Return list of repo-relative paths edited in the trailing assistant turn.

    'Trailing turn' = contiguous trailing assistant messages, back to last user message.
    """
    if not transcript_path.exists():
        return []

    lines = transcript_path.read_text(encoding="utf-8").splitlines()
    # Walk from end backwards, collecting assistant entries until we hit a user
    # entry or the start of the file.
    trailing: List[dict] = []
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = obj.get("type")
        if t == "user":
            break
        if t == "assistant":
            trailing.append(obj)

    paths: List[str] = []
    for obj in reversed(trailing):  # restore chronological order
        content = obj.get("message", {}).get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "tool_use":
                continue
            if block.get("name") not in EDITING_TOOLS:
                continue
            fp = block.get("input", {}).get("file_path")
            if not fp:
                continue
            rel = _rel_to_repo(fp, repo_root)
            if rel and rel not in paths:
                paths.append(rel)
    return paths
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `pytest tests/hooks/test_transcript_parser.py -v`
Expected: All 3 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add .claude/hooks/hook_utils/transcript_parser.py
git commit -m "feat(review-chain): implement transcript_parser

Parses Claude Code transcript JSONL. edited_paths_in_last_turn() returns
repo-relative paths edited in the trailing assistant turn (stops at most
recent user message). Supports Edit/Write/NotebookEdit/MultiEdit tools."
```

---

### Task 14: Test diff-classifier

**Files:**
- Create: `tests/hooks/test_diff_classifier.py`

- [ ] **Step 1: Write the failing tests**

File: `tests/hooks/test_diff_classifier.py`
Content:

```python
"""Tests for diff_classifier: classify a set of edited paths."""
from __future__ import annotations

import sys
from pathlib import Path

HOOKS_DIR = Path(__file__).resolve().parents[2] / ".claude" / "hooks"
sys.path.insert(0, str(HOOKS_DIR))

from hook_utils.diff_classifier import classify_diff  # noqa: E402


def test_only_docs_changed_is_docs_only():
    result = classify_diff(["docs/foo.md", "docs/bar.md"])
    assert result["kind"] == "docs-only"
    assert result["run_full_chain"] is False


def test_only_tests_changed_is_test_only():
    result = classify_diff(["tests/test_a.py", "tests/foo/test_b.py"])
    assert result["kind"] == "test-only"
    assert result["run_full_chain"] is True  # Stage 2+3 still run per spec §4.2.1


def test_mixed_code_and_docs_is_full():
    result = classify_diff(["src/foo.py", "docs/bar.md"])
    assert result["kind"] == "full"
    assert result["run_full_chain"] is True


def test_only_src_is_full():
    result = classify_diff(["src/assembled_core/utils/foo.py"])
    assert result["kind"] == "full"
    assert result["run_full_chain"] is True


def test_workflow_change_is_full():
    result = classify_diff([".github/workflows/ci.yml"])
    assert result["kind"] == "full"
    assert result["run_full_chain"] is True


def test_governance_change_is_full():
    result = classify_diff(["CLAUDE.md"])
    assert result["kind"] == "full"


def test_empty_diff_is_skip():
    result = classify_diff([])
    assert result["kind"] == "skip"
    assert result["run_full_chain"] is False


def test_no_protected_paths_is_skip():
    result = classify_diff(["output/equity.csv", "README.md"])
    assert result["kind"] == "skip"
    assert result["run_full_chain"] is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/hooks/test_diff_classifier.py -v`
Expected: FAIL with ModuleNotFoundError.

- [ ] **Step 3: Commit failing tests**

```bash
git add tests/hooks/test_diff_classifier.py
git commit -m "test(review-chain): failing tests for diff_classifier"
```

---

### Task 15: Implement diff-classifier

**Files:**
- Create: `.claude/hooks/hook_utils/diff_classifier.py`

- [ ] **Step 1: Implement classifier**

File: `.claude/hooks/hook_utils/diff_classifier.py`
Content:

```python
"""Classify a set of edited file paths into a diff-kind.

Per spec §4.2.1:
- skip: no protected paths → review chain doesn't run at all.
- docs-only: only documentation files (excluding governance) → Stage 2+3 skipped.
- test-only: only tests/** files → Stage 1 = test-runner, Stage 2+3 run.
- full: any mix that includes src/, scripts/, workflows, or governance → full chain.
"""
from __future__ import annotations

from typing import Iterable, Dict, Any

from .path_classifier import is_protected_path, _norm


def _is_docs_only_path(path: str) -> bool:
    p = _norm(path)
    if p == "CLAUDE.md":
        return False
    if p.startswith(".claude/rules/"):
        return False
    if p.startswith("docs/") and p.endswith(".md"):
        return True
    if p.endswith(".md"):
        return True
    return False


def _is_test_only_path(path: str) -> bool:
    p = _norm(path)
    return p.startswith("tests/") and p.endswith(".py")


def classify_diff(paths: Iterable[str]) -> Dict[str, Any]:
    """Classify the diff. Returns {kind, run_full_chain, protected_paths}."""
    paths = list(paths)
    protected = [p for p in paths if is_protected_path(p)]

    if not paths:
        return {"kind": "skip", "run_full_chain": False, "protected_paths": []}

    if not protected:
        # Mark test-only / docs-only specially even when not "protected",
        # because tests/** and docs/** don't appear in protected list but
        # are common edit targets we care about.
        if all(_is_test_only_path(p) for p in paths):
            return {"kind": "test-only", "run_full_chain": True, "protected_paths": []}
        if all(_is_docs_only_path(p) for p in paths):
            return {"kind": "docs-only", "run_full_chain": False, "protected_paths": []}
        return {"kind": "skip", "run_full_chain": False, "protected_paths": []}

    # We have protected paths.
    # If ALL paths are docs-only (which means no protected since docs/ isn't protected),
    # we've already returned above. So 'protected non-empty' implies code/governance change.
    return {"kind": "full", "run_full_chain": True, "protected_paths": protected}
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `pytest tests/hooks/test_diff_classifier.py -v`
Expected: All tests PASS.

- [ ] **Step 3: Commit**

```bash
git add .claude/hooks/hook_utils/diff_classifier.py
git commit -m "feat(review-chain): implement diff_classifier

Classifies an edited-paths set into: skip / docs-only / test-only / full.
Test-only still runs full chain (Stage 2+3 audit test correctness). Docs-only
skips Stage 2+3. Empty or non-protected paths → skip."
```

---

### Task 16: Test review-marker

**Files:**
- Create: `tests/hooks/test_review_marker.py`

- [ ] **Step 1: Write the failing tests**

File: `tests/hooks/test_review_marker.py`
Content:

```python
"""Tests for review_marker: per-turn marker preventing re-trigger of review chain."""
from __future__ import annotations

import sys
from pathlib import Path

HOOKS_DIR = Path(__file__).resolve().parents[2] / ".claude" / "hooks"
sys.path.insert(0, str(HOOKS_DIR))

from hook_utils.review_marker import (  # noqa: E402
    turn_id_from_transcript,
    has_review_marker,
    write_review_marker,
)


def test_turn_id_is_stable_for_same_transcript(tmp_path):
    t = tmp_path / "t.jsonl"
    t.write_text('{"type":"assistant","uuid":"abc"}\n', encoding="utf-8")
    id1 = turn_id_from_transcript(t)
    id2 = turn_id_from_transcript(t)
    assert id1 == id2
    assert id1  # non-empty


def test_turn_id_changes_when_transcript_grows(tmp_path):
    t = tmp_path / "t.jsonl"
    t.write_text('{"type":"assistant","uuid":"abc"}\n', encoding="utf-8")
    id1 = turn_id_from_transcript(t)
    t.write_text(
        '{"type":"assistant","uuid":"abc"}\n{"type":"assistant","uuid":"def"}\n',
        encoding="utf-8",
    )
    id2 = turn_id_from_transcript(t)
    assert id1 != id2


def test_has_marker_false_initially(tmp_path):
    state_dir = tmp_path / "state"
    assert has_review_marker("turn-123", state_dir) is False


def test_write_then_has_marker_true(tmp_path):
    state_dir = tmp_path / "state"
    write_review_marker("turn-456", state_dir)
    assert has_review_marker("turn-456", state_dir) is True


def test_different_turns_independent(tmp_path):
    state_dir = tmp_path / "state"
    write_review_marker("turn-A", state_dir)
    assert has_review_marker("turn-A", state_dir) is True
    assert has_review_marker("turn-B", state_dir) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/hooks/test_review_marker.py -v`
Expected: FAIL with ModuleNotFoundError.

- [ ] **Step 3: Commit failing tests**

```bash
git add tests/hooks/test_review_marker.py
git commit -m "test(review-chain): failing tests for review_marker"
```

---

### Task 17: Implement review-marker

**Files:**
- Create: `.claude/hooks/hook_utils/review_marker.py`

- [ ] **Step 1: Implement marker**

File: `.claude/hooks/hook_utils/review_marker.py`
Content:

```python
"""Per-turn marker to prevent re-triggering the review chain.

The Stop hook blocks once. After the main agent dispatches the review chain
and writes the marker, the next Stop event sees the marker and lets the stop
through.

Turn-ID derivation: SHA1 of the transcript byte length + last assistant UUID.
This changes whenever the transcript grows, which it does on every new turn.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

MARKER_DIR_NAME = ".review_markers"


def turn_id_from_transcript(transcript_path: Path) -> str:
    """Derive a stable per-turn ID from the transcript file."""
    if not transcript_path.exists():
        return "no-transcript"
    raw = transcript_path.read_bytes()
    size = len(raw)
    last_uuid = ""
    for line in reversed(raw.decode("utf-8", errors="replace").splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if obj.get("type") == "assistant":
                last_uuid = obj.get("uuid", "")
                break
        except json.JSONDecodeError:
            continue
    digest = hashlib.sha1(f"{size}:{last_uuid}".encode("utf-8")).hexdigest()
    return digest[:16]


def _marker_path(turn_id: str, state_dir: Path) -> Path:
    return state_dir / f"{turn_id}.done"


def has_review_marker(turn_id: str, state_dir: Path) -> bool:
    return _marker_path(turn_id, state_dir).exists()


def write_review_marker(turn_id: str, state_dir: Path) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    _marker_path(turn_id, state_dir).write_text("done", encoding="utf-8")
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `pytest tests/hooks/test_review_marker.py -v`
Expected: All tests PASS.

- [ ] **Step 3: Commit**

```bash
git add .claude/hooks/hook_utils/review_marker.py
git commit -m "feat(review-chain): implement review_marker

Per-turn marker file (.review_markers/<turn_id>.done) prevents the Stop hook
from re-blocking after the review chain has run for the current turn.
Turn-ID = SHA1(transcript_size:last_assistant_uuid)[:16]."
```

---

## Phase D — Stop-Hook Main Script

### Task 18: Test stop_review_chain main logic

**Files:**
- Create: `tests/hooks/test_stop_hook.py`

- [ ] **Step 1: Write the failing tests**

File: `tests/hooks/test_stop_hook.py`
Content:

```python
"""Integration tests for the Stop hook entry point."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
HOOK_PATH = REPO_ROOT / ".claude" / "hooks" / "stop_review_chain.py"


def _run_hook(stdin_payload: dict, env_overrides: dict | None = None) -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        "PYTHONIOENCODING": "utf-8",
        **(env_overrides or {}),
    }
    return subprocess.run(
        [sys.executable, str(HOOK_PATH)],
        input=json.dumps(stdin_payload),
        capture_output=True,
        text=True,
        env=env,
    )


def test_hook_allows_stop_when_no_transcript(tmp_path):
    """If transcript doesn't exist, hook can't classify → fail-open, allow stop."""
    res = _run_hook({
        "session_id": "test",
        "transcript_path": str(tmp_path / "nope.jsonl"),
        "stop_hook_active": False,
    })
    assert res.returncode == 0
    # No block decision → empty stdout or {"decision": "approve"}
    if res.stdout.strip():
        payload = json.loads(res.stdout)
        assert payload.get("decision") != "block"


def test_hook_allows_stop_when_no_protected_edits(tmp_path):
    transcript = tmp_path / "t.jsonl"
    # Last assistant turn only edits docs/ → not protected
    transcript.write_text(
        json.dumps({
            "type": "assistant",
            "message": {"role": "assistant", "content": [{
                "type": "tool_use", "name": "Edit",
                "input": {"file_path": str(REPO_ROOT / "docs" / "foo.md"),
                          "old_string": "a", "new_string": "b"},
            }]},
            "uuid": "a1",
        }) + "\n",
        encoding="utf-8",
    )
    res = _run_hook({
        "session_id": "test",
        "transcript_path": str(transcript),
        "stop_hook_active": False,
    }, env_overrides={"CLAUDE_HOOKS_STATE_DIR": str(tmp_path / "state")})
    assert res.returncode == 0
    if res.stdout.strip():
        payload = json.loads(res.stdout)
        assert payload.get("decision") != "block"


def test_hook_blocks_stop_when_protected_edit_and_no_marker(tmp_path):
    transcript = tmp_path / "t.jsonl"
    # Last assistant turn edits src/ → protected
    transcript.write_text(
        json.dumps({
            "type": "assistant",
            "message": {"role": "assistant", "content": [{
                "type": "tool_use", "name": "Edit",
                "input": {
                    "file_path": str(REPO_ROOT / "src" / "assembled_core" / "execution" / "router.py"),
                    "old_string": "a", "new_string": "b",
                },
            }]},
            "uuid": "a1",
        }) + "\n",
        encoding="utf-8",
    )
    res = _run_hook({
        "session_id": "test",
        "transcript_path": str(transcript),
        "stop_hook_active": False,
    }, env_overrides={"CLAUDE_HOOKS_STATE_DIR": str(tmp_path / "state")})
    assert res.returncode == 0
    payload = json.loads(res.stdout)
    assert payload["decision"] == "block"
    assert "REVIEW-CHAIN-REQUIRED" in payload["reason"]
    assert "risk-execution-reviewer" in payload["reason"]
    assert "senior-code-reviewer" in payload["reason"]
    assert "task-completion-auditor" in payload["reason"]


def test_hook_allows_stop_when_marker_already_written(tmp_path):
    """If review chain has already run for this turn, marker exists, hook allows stop."""
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(
        json.dumps({
            "type": "assistant",
            "message": {"role": "assistant", "content": [{
                "type": "tool_use", "name": "Edit",
                "input": {
                    "file_path": str(REPO_ROOT / "src" / "foo.py"),
                    "old_string": "a", "new_string": "b",
                },
            }]},
            "uuid": "a1",
        }) + "\n",
        encoding="utf-8",
    )
    state_dir = tmp_path / "state"
    # First call: should block
    res1 = _run_hook({
        "session_id": "test",
        "transcript_path": str(transcript),
        "stop_hook_active": False,
    }, env_overrides={"CLAUDE_HOOKS_STATE_DIR": str(state_dir)})
    payload1 = json.loads(res1.stdout)
    assert payload1["decision"] == "block"
    # Extract turn_id from reason and write marker manually (simulating main agent finishing review)
    # Easier: invoke the marker-write directly via a small inline script
    write_marker_script = (
        "import sys; sys.path.insert(0, r'" + str(REPO_ROOT / ".claude" / "hooks") + "');"
        "from hook_utils.review_marker import turn_id_from_transcript, write_review_marker;"
        "from pathlib import Path;"
        "tid = turn_id_from_transcript(Path(r'" + str(transcript) + "'));"
        "write_review_marker(tid, Path(r'" + str(state_dir) + "'));"
    )
    subprocess.run([sys.executable, "-c", write_marker_script], check=True)

    # Second call: should allow
    res2 = _run_hook({
        "session_id": "test",
        "transcript_path": str(transcript),
        "stop_hook_active": False,
    }, env_overrides={"CLAUDE_HOOKS_STATE_DIR": str(state_dir)})
    assert res2.returncode == 0
    if res2.stdout.strip():
        payload2 = json.loads(res2.stdout)
        assert payload2.get("decision") != "block"


def test_hook_respects_stop_hook_active_to_avoid_infinite_loop(tmp_path):
    """If stop_hook_active=true, never re-block — Claude Code is already in a hook loop."""
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(
        json.dumps({
            "type": "assistant",
            "message": {"role": "assistant", "content": [{
                "type": "tool_use", "name": "Edit",
                "input": {
                    "file_path": str(REPO_ROOT / "src" / "foo.py"),
                    "old_string": "a", "new_string": "b",
                },
            }]},
            "uuid": "a1",
        }) + "\n",
        encoding="utf-8",
    )
    res = _run_hook({
        "session_id": "test",
        "transcript_path": str(transcript),
        "stop_hook_active": True,  # already in a loop
    }, env_overrides={"CLAUDE_HOOKS_STATE_DIR": str(tmp_path / "state")})
    assert res.returncode == 0
    if res.stdout.strip():
        payload = json.loads(res.stdout)
        assert payload.get("decision") != "block"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/hooks/test_stop_hook.py -v`
Expected: All FAIL with FileNotFoundError for `stop_review_chain.py`.

- [ ] **Step 3: Commit failing tests**

```bash
git add tests/hooks/test_stop_hook.py
git commit -m "test(review-chain): failing integration tests for Stop hook"
```

---

### Task 19: Implement stop_review_chain.py

**Files:**
- Create: `.claude/hooks/stop_review_chain.py`

- [ ] **Step 1: Implement the hook**

File: `.claude/hooks/stop_review_chain.py`
Content:

```python
#!/usr/bin/env python3
"""Stop hook: enforce the review chain after coding steps in protected paths.

Flow per spec §4.2.1:
1. Read Stop event JSON from stdin (session_id, transcript_path, stop_hook_active).
2. If stop_hook_active is True, exit (never loop).
3. Parse transcript trailing turn for edited paths.
4. Classify diff: skip / docs-only / test-only / full.
5. If kind is 'skip' or 'docs-only': allow stop.
6. Check review-marker for current turn_id.
7. If marker exists: allow stop (chain already ran for this turn).
8. Otherwise: output {"decision":"block","reason": REVIEW_INSTRUCTIONS}.

REVIEW_INSTRUCTIONS embed:
- list of Stage-1 specialists to dispatch (based on edited paths)
- reminder to dispatch senior-code-reviewer (Stage 2) and task-completion-auditor (Stage 3)
- reminder to write the review-marker via the `write_review_marker` utility before next stop

Env overrides for testability:
- CLAUDE_HOOKS_STATE_DIR: where to read/write review markers (default: .claude/.review_markers)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hook_utils.path_classifier import specialists_for_paths  # noqa: E402
from hook_utils.transcript_parser import edited_paths_in_last_turn  # noqa: E402
from hook_utils.diff_classifier import classify_diff  # noqa: E402
from hook_utils.review_marker import (  # noqa: E402
    turn_id_from_transcript,
    has_review_marker,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _state_dir() -> Path:
    override = os.environ.get("CLAUDE_HOOKS_STATE_DIR")
    if override:
        return Path(override)
    return REPO_ROOT / ".claude" / ".review_markers"


def _allow_stop() -> int:
    # Empty stdout (or explicit approve) → Claude Code proceeds with stop
    return 0


def _build_review_instructions(edited_paths: list[str], specialists: set[str], turn_id: str) -> str:
    spec_block = "\n".join(f"  - `{s}`" for s in sorted(specialists))
    paths_block = "\n".join(f"  - {p}" for p in edited_paths[:20])  # cap at 20
    if len(edited_paths) > 20:
        paths_block += f"\n  - ... ({len(edited_paths) - 20} more)"

    return f"""REVIEW-CHAIN-REQUIRED — Step nicht abgeschlossen, bis Review-Kette gelaufen ist.

Du hast Code in geschützten Pfaden geändert:
{paths_block}

Pflicht-Ablauf jetzt:

**Stage 1 (parallel):** Dispatche diese Spezialist-Subagents über das Agent-Tool:
{spec_block}

Jeder bekommt: den Diff der oben gelisteten Pfade + die ursprüngliche Task-Beschreibung.
Verlange strukturierte Findings im YAML-Schema aus
`docs/superpowers/specs/2026-05-14-review-chain-design.md` §5.

**Stage 2:** Dispatche `senior-code-reviewer` mit:
- Diff
- ursprüngliche Task-Beschreibung
- konsolidierte Stage-1-Findings
- Top-10 Anti-Patterns sind bereits im Kontext (SessionStart-Hook)

**Stage 3:** Dispatche `task-completion-auditor` mit:
- ursprüngliche Task-Beschreibung
- Diff
- alle bisherigen Findings (Stage 1 + 2)
Erwarte Verdict: PASS / CONDITIONAL / FAIL.

**Konsolidierung:** Schreibe einen Findings-Block (Spec §5 Format) mit:
- BLOCKER / MAJOR / MINOR / INFO Listen
- Follow-ups (Adjacent-Themen)
- Errors-Log-Vorschläge (falls vorhanden)
- Verdict

**Marker schreiben (Pflicht, damit nächstes Stop nicht erneut blockt):**

```python
import sys
sys.path.insert(0, ".claude/hooks")
from hook_utils.review_marker import write_review_marker
from pathlib import Path
write_review_marker("{turn_id}", Path(".claude/.review_markers"))
```

**Dann:**
- BLOCKER → adressieren bevor du an User zurückmeldest.
- MAJOR → adressieren ODER explizit dokumentieren/akzeptieren.
- MINOR/INFO → optional dokumentieren.
- Errors-Log-Vorschläge → in `docs/CLAUDE_CODING_ERRORS.md` appenden (append-only).
- Erst dann an User zurückmelden.

Vermeide: ungeprüfte Behauptungen „passt schon", Review-Theater, Scope-Erweiterung.
"""


def main() -> int:
    try:
        raw = sys.stdin.read()
        event = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        # Malformed input → fail-open, allow stop
        return _allow_stop()

    # Avoid infinite loop: if Claude Code already invoked us in a Stop-hook loop, allow.
    if event.get("stop_hook_active") is True:
        return _allow_stop()

    transcript_path_str = event.get("transcript_path", "")
    if not transcript_path_str:
        return _allow_stop()

    transcript_path = Path(transcript_path_str)
    edited = edited_paths_in_last_turn(transcript_path, repo_root=REPO_ROOT)
    classification = classify_diff(edited)

    if not classification["run_full_chain"]:
        return _allow_stop()

    # Check marker
    turn_id = turn_id_from_transcript(transcript_path)
    state_dir = _state_dir()
    if has_review_marker(turn_id, state_dir):
        return _allow_stop()

    # Block and instruct
    specialists = specialists_for_paths(edited)
    reason = _build_review_instructions(edited, specialists, turn_id)
    payload = {"decision": "block", "reason": reason}
    sys.stdout.write(json.dumps(payload))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `pytest tests/hooks/test_stop_hook.py -v`
Expected: All 5 tests PASS.

- [ ] **Step 3: Add `.claude/.review_markers/` to `.gitignore`**

Edit `.gitignore` (append at end):

```
# Review-chain runtime state (per-turn markers, not source)
.claude/.review_markers/
```

- [ ] **Step 4: Commit**

```bash
git add .claude/hooks/stop_review_chain.py .gitignore
git commit -m "feat(review-chain): implement Stop hook with mandatory review enforcement

When code edits land in protected paths and no review-marker exists for the
current turn, the hook outputs decision=block with detailed instructions:
which Stage-1 specialists to dispatch, then senior-code-reviewer, then
task-completion-auditor, then write marker. Fail-open on malformed input.
Respects stop_hook_active to prevent loops."
```

---

### Task 20: Register Stop hook in settings.json

**Files:**
- Modify: `.claude/settings.json`

- [ ] **Step 1: Update settings.json**

Final content:

```json
{
  "permissions": {
    "defaultMode": "bypassPermissions"
  },
  "enabledPlugins": {
    "claude-mem@thedotmack": true,
    "frontend-design@claude-plugins-official": true,
    "superpowers@claude-plugins-official": true,
    "github@claude-plugins-official": true
  },
  "hooks": {
    "SessionStart": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "python .claude/hooks/session_start_load_errors.py"
          }
        ]
      }
    ],
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "python .claude/hooks/stop_review_chain.py"
          }
        ]
      }
    ]
  }
}
```

- [ ] **Step 2: Verify JSON is valid**

Run: `python -c "import json; json.load(open('.claude/settings.json'))"`
Expected: no error.

- [ ] **Step 3: Commit**

```bash
git add .claude/settings.json
git commit -m "feat(review-chain): register Stop hook in settings.json

After this commit, every assistant stop with code edits in protected paths
will trigger the review-chain block. Marker files prevent re-blocking within
the same turn."
```

---

### Task 21: Full unit + integration suite green

- [ ] **Step 1: Run all hook tests**

Run: `pytest tests/hooks/ -v`
Expected: All tests PASS (errors_log: 6, path_classifier: 8, transcript_parser: 3, diff_classifier: 8, review_marker: 5, session_start_hook: 2, stop_hook: 5 = 37 tests).

- [ ] **Step 2: No commit** — verification only.

---

## Phase E — Governance Updates

### Task 22: Update CLAUDE.md with §20 Review-Chain

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Read current CLAUDE.md structure**

Locate the last numbered section (currently §19 Architektur-Systemkarte) and the §18 Schlussstatus. Add new §20 between §19 and §18 (or after §18 if §18 is intended as the final section — re-number §18 to §21).

Read: `Read CLAUDE.md offset=around-§19`

Actual placement: §19 is Architektur-Systemkarte, §18 is Schlussstatus. So §20 comes BEFORE §18 logically. But the file currently has §18 AFTER §19 — quirky. Just insert new §20 between §19 and §18 to minimize disruption.

- [ ] **Step 2: Insert §20 section before §18**

Find the line `## 18. Schlussstatus dieser Datei` and insert the following block immediately before it:

```markdown
---

## 20. Review-Chain (automatisch erzwungen)

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

- §2.2 „kleinster sicherer Schritt" bleibt bindend.
- Rule 10 „kein großer Refactor ohne Auftrag" bleibt bindend.
- Rule 60 „ein Problem pro Änderung" bleibt bindend.
- Auditor darf Adjacent nur als Follow-up-Vorschlag flaggen, nie als Pflicht im aktuellen Step.

```

- [ ] **Step 3: Verify CLAUDE.md still imports rules correctly**

Run: `Get-Content CLAUDE.md | Select-String "@.claude/rules"`
Expected: All 11 rule imports still present at the end.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(claude): add §20 Review-Chain to CLAUDE.md

Documents the new mandatory review chain: Stage 1 specialists → senior-code-
reviewer → task-completion-auditor. References spec for findings schema.
Reaffirms that §2.2 / Rule 10 / Rule 60 remain binding — auditor flags
adjacent issues as follow-ups only, never as in-scope demands."
```

---

### Task 23: Update rule 90 (subagents-hooks-and-automation)

**Files:**
- Modify: `.claude/rules/90-subagents-hooks-and-automation.md`

- [ ] **Step 1: Replace the "Konkrete Routing-Policy" section**

Find the section starting with `## Konkrete Routing-Policy` and ending before `## Hook-Regeln`. Replace it with:

```markdown
## Konkrete Routing-Policy

Subagents sind Default-Ausführungsmodus für spezialisierte Arbeit.
Nicht auf explizite User-Aufforderung warten, wenn die Aufgabe klar zu einem Spezialbereich passt.

### Automatische Erzwingung via Stop-Hook (seit 2026-05-14)

Nach jedem Step mit Code-Änderungen in geschützten Pfaden (`src/`, `scripts/`, `.github/workflows/`, `.claude/rules/`, `CLAUDE.md`) erzwingt der Stop-Hook `.claude/hooks/stop_review_chain.py` die Review-Kette automatisch — siehe CLAUDE.md §20 und Spec `docs/superpowers/specs/2026-05-14-review-chain-design.md`.

### Routing-Regeln (für nicht-erzwungene Pfade und ad-hoc Reviews)

- **`ci-debugger`** proaktiv bei: CI-Failures, Workflow-Failures, plattformspezifischer Test-Divergenz (Windows vs. Ubuntu), Dependency-Drift, Collection-Failures, Artifact-Konflikten, Local-vs-CI-Mismatches.
- **`test-runner`** proaktiv bei: gezielter Testausführung, Marker-Handling, Regression-Validierung, Failing-Test-Triage, Minimal-Repro-Verifikation.
- **`risk-execution-reviewer`** proaktiv bei jeder Aufgabe, die `src/assembled_core/execution/`, `src/assembled_core/risk/`, `src/assembled_core/paper/`, `src/assembled_core/pipeline/`, Portfolio-Constraints, Order-Generierung, Pre-Trade-Checks, Kill-Switch-Logik oder cost-aware Execution betrifft.
- **`docs-governance-sync`** proaktiv bei Änderungen an `CLAUDE.md`, `.claude/rules/`, `AGENTS.md`, `.cursor/rules/`, `docs/cursor/` oder jedem Agent-Governance-/Repo-Instruction-Layer.
- **`memory-tracker`** proaktiv, wenn eine Session bedeutsame Entscheidungen, Statuswechsel, Debug-Conclusions, Governance-Änderungen oder neue Risk-Annahmen produziert hat, die über Sessions hinweg stabil bleiben sollen.
- **`senior-code-reviewer`** (NEU 2026-05-14): Stage 2 der Review-Kette. Automatisch vom Stop-Hook getriggert nach den Spezialisten. Auch manuell aufrufbar für ad-hoc Code-Review.
- **`task-completion-auditor`** (NEU 2026-05-14): Stage 3 der Review-Kette. Automatisch vom Stop-Hook getriggert nach `senior-code-reviewer`. Auch manuell aufrufbar bei „bin ich wirklich fertig?"-Zweifeln.

### Pflichtverhalten

- Stop-Hook-Erzwingung **kann nicht umgangen werden** durch „ich vergesse mal". Der Hook blockiert das Stop-Event bis Marker geschrieben ist.
- Außerhalb der Erzwingungs-Pfade: Spezialist-Delegation **bevorzugen**, nicht als Zusatzoption behandeln.
- Sensible Zonen nie ohne Spezialdelegation überspringen, außer mit explizitem Grund.

### Prioritätsreihenfolge bei Konflikt

1. `risk-execution-reviewer`
2. `senior-code-reviewer`
3. `task-completion-auditor`
4. `ci-debugger`
5. `test-runner`
6. `docs-governance-sync`
7. `memory-tracker`
```

- [ ] **Step 2: Commit**

```bash
git add .claude/rules/90-subagents-hooks-and-automation.md
git commit -m "docs(rules): update rule 90 for review-chain enforcement

Adds senior-code-reviewer and task-completion-auditor to the routing policy.
Documents that the Stop-hook now automatically enforces the chain — proactive
routing remains the policy outside the enforced paths and for ad-hoc reviews."
```

---

### Task 24: Final integration smoke test

- [ ] **Step 1: Run the full hook test suite once more**

Run: `pytest tests/hooks/ -v`
Expected: 37 tests PASS.

- [ ] **Step 2: Manually invoke Stop hook with a synthetic transcript**

Create a temporary transcript file with a synthetic protected-path edit and pipe it to the hook:

```powershell
$tmp = New-TemporaryFile
$transcript = "$tmp.jsonl"
$edit = @{
  type = "assistant"
  message = @{
    role = "assistant"
    content = @(@{
      type = "tool_use"
      name = "Edit"
      input = @{
        file_path = "F:/Python_Projekt/Aktiengeruest/src/assembled_core/execution/router.py"
        old_string = "a"
        new_string = "b"
      }
    })
  }
  uuid = "smoke-1"
} | ConvertTo-Json -Compress -Depth 10
Set-Content -Path $transcript -Value $edit -Encoding UTF8

$event = @{ session_id = "smoke"; transcript_path = $transcript; stop_hook_active = $false } | ConvertTo-Json -Compress
$event | python .claude/hooks/stop_review_chain.py
```

Expected output (one-line JSON):
```
{"decision": "block", "reason": "REVIEW-CHAIN-REQUIRED — Step nicht abgeschlossen ..."}
```

The reason should mention `risk-execution-reviewer`, `test-runner`, `senior-code-reviewer`, `task-completion-auditor`, and instructions to write the marker.

- [ ] **Step 3: Manually invoke SessionStart hook**

```powershell
'{"session_id":"smoke","transcript_path":""}' | python .claude/hooks/session_start_load_errors.py
```

Expected: JSON output with `additionalContext` listing E-001 through E-007.

- [ ] **Step 4: No commit** — smoke verification only.

---

## Phase F — Final wrap-up

### Task 25: Update plan status in this file

**Files:**
- Modify: `docs/superpowers/plans/2026-05-14-review-chain.md` (this file)

- [ ] **Step 1: Add status block at top of plan**

Insert after the Goal/Architecture/Tech-Stack header:

```markdown
**Implementation status:** COMPLETE on YYYY-MM-DD. All 24 tasks executed. 37 hook tests green. Manual smoke verified.
```

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/plans/2026-05-14-review-chain.md
git commit -m "docs(plan): mark review-chain implementation plan as complete"
```

---

## Self-Review Summary

**Spec coverage check** (vs. `2026-05-14-review-chain-design.md`):
- §4.1.1 senior-code-reviewer → Task 2
- §4.1.2 task-completion-auditor → Task 3
- §4.2.1 Stop-hook → Tasks 10–19
- §4.2.2 SessionStart-hook → Tasks 4–8
- §4.3 CLAUDE_CODING_ERRORS.md → Task 1
- §5 Findings schema → embedded in agent definitions (Tasks 2–3)
- §7.1 rule 90 update → Task 23
- §7.2 CLAUDE.md §20 → Task 22
- §8 System-map-currency → NOT in v1, deferred to follow-up. **Gap acknowledged.**
- §11 acceptance criteria → covered by Tasks 21 + 24

**Placeholder scan:** No "TBD/TODO/implement later" in steps. All code blocks are complete.

**Type/name consistency:** `parse_errors_log`, `top_n_entries`, `is_protected_path`, `specialists_for_paths`, `edited_paths_in_last_turn`, `classify_diff`, `turn_id_from_transcript`, `has_review_marker`, `write_review_marker` — all used identically across tasks.

**Scope check:** One cohesive implementation plan. ~25 tasks, mostly 2–5 minute steps, frequent commits.

**Known deferral:** System-map-currency check (§8 of spec) is NOT in v1. Reason: keeps Stop-hook complexity manageable. Can be added later as a small follow-up plan once v1 is stable.
