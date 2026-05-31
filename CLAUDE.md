# CLAUDE.md

Operativer Arbeitskontext (die „Verfassung") für Claude Code im Repo **Assembled-Trading-AI**.
Keine README, kein Marketing, kein Zukunftsroman — eine Steuerdatei für sicheres, präzises,
branch- und CI-bewusstes Arbeiten. Detailregeln liegen modular in den `@`-Imports am Ende.

**Ausgelagerte Langfassungen (am 2026-05-30 aus dieser Datei migriert):**
- Strategischer Kontext, Ziele, Constraints, Entwicklungsrichtung → `PROJEKT_STATUS.md`
- Schichtenlogik, Datenfluss, Backtest-Grundsatz, Systemkarte → `docs/ARCHITECTURE_BACKEND.md`
- Review-Chain-Langfassung + Bypass-Disclosures (§20.1–20.5/20.7/20.8) → `docs/review_chain_disclosure.md`

---

## Grundsatz

In diesem Projekt ist **technische Ehrlichkeit wichtiger als Tempo**.
**Plan ≠ Implementierung.** Was unklar ist, ist unklar. Was nur lokal getestet ist, ist nicht
CI-bestätigt. Was nur spezifiziert ist, ist nicht implementiert. Was branch-spezifisch ist, ist
nicht automatisch Repo-Wahrheit.

## Projekt

Assembled-Trading-AI ist ein **modulares Python-Backend** (Research, Backtests, Paper-/Simulation,
Risk-Overlays, QA/Evidence/Reporting, API/OMS-light/Paper-Routing, schrittweise
Intel-/News-/Disclosure-/GeoRisk-Integration). Leitidee: **Risk-first statt Rendite-first**, kein
Leverage im frühen Betrieb, kontrollierte State-Machine-Logik. Vollständige Ziel-/Constraint-/
Architekturbeschreibung: `PROJEKT_STATUS.md` und `docs/ARCHITECTURE_BACKEND.md`.

## Plan ≠ Implementierung

Es existieren viele Ebenen gleichzeitig — sie dürfen **nie automatisch gleichgesetzt** werden:
diskutiert · spezifiziert · Skeleton/Stub · teilweise implementiert · implementiert · lokal
getestet · CI-bestätigt · branch-spezifisch · überholt · offen.

- Ein Plan ist keine Implementierung.
- Ein lokaler Test ist keine CI-Bestätigung.
- Ein branch-spezifischer Fix ist keine Repo-Wahrheit.
- Ein TODO ist keine Funktion. Ein Stub ist keine Integration.
- Ein grüner Teiltest ist keine globale Entwarnung.

## Kleinster sicherer Schritt

Standard für jede Änderung:

1. Problem exakt lokalisieren.
2. Branch-/CI-Relevanz prüfen.
3. Betroffene Dateien eingrenzen.
4. Kleinste sichere Änderung planen.
5. Nur zielrelevante Dateien ändern.
6. Gezielte Tests / Lint ausführen.
7. Ehrlich dokumentieren, was wirklich geprüft wurde.

Keine Rundumschläge. Keine Nebenbei-Refactors. Keine stillen Strukturumbauten ohne Auftrag.

## Keine falsche Sicherheit

Claude darf niemals formulieren:

- „alles grün", wenn nur Teiltests liefen
- „ist implementiert", wenn es nur im Spec oder Prompt steht
- „ist im Repo", wenn es nur chat- oder branch-spezifisch beschrieben wurde
- „ist bestätigt", wenn nur lokales Verhalten beobachtet wurde
- „ist sicher", wenn dazu keine Evidenz vorliegt

## Sensible Zonen (Tabu ohne Auftrag)

Diese Pfade sind **hart geschützt** — Edit/Write werden über `permissions.deny` in
`.claude/settings.json` und destruktive Bash-Schreibzugriffe über den PreToolUse-Hook
`.claude/hooks/protected_paths_guard.py` **technisch blockiert** (nicht nur advisory), auch unter
`bypassPermissions`:

- `src/assembled_core/execution/`
- `src/assembled_core/risk/`
- `src/assembled_core/accounting/`
- `src/assembled_core/pipeline/`
- `src/assembled_core/paper/`
- `.github/workflows/`

Änderungen hier nur mit explizitem Auftrag: Scope eng halten, Seiteneffekte aktiv suchen, gezielte
Tests priorisieren, klar berichten, was verifiziert wurde. Der breitere advisory-sensible Bereich
(u. a. `data/` bei PIT-/Timing-Bezug, `features/event_features.py`, `data/corporate_actions.py`,
risk-/paper-nahe Scripts) folgt `@.claude/rules/30-risk-execution-safeguards.md`.

## Datenrealismus

Keine Ergebnisse als produktionsnah darstellen, wenn sie zu schwach abgesichert sind —
insbesondere bei: synthetischen Daten, Survivorship-Bias, Look-Ahead-Bias, fehlenden Corporate
Actions, fragwürdiger Feed-Qualität, unklaren Kalendern/Verfügbarkeitszeiten.

Datenprobleme **nicht still verschlucken**: sichtbar machen; blocken, warnen oder degradieren; im
Report/QA-Artefakt kenntlich machen; nicht still weiterlaufen, wenn das Ergebnis dadurch
unzuverlässig wird.

Bei datenpfadbezogener Arbeit klar angeben: welcher Preisdatenpfad benutzt wurde · real vs.
synthetisch · Coverage · Qualitätsstatus / bekannte Ausfälle / Delistings.

## PIT-Regel

Features und Signale dürfen nur Informationen nutzen, die zum jeweiligen `as_of`-Zeitpunkt wirklich
verfügbar waren. Bei Event-/Disclosure-Daten unterscheiden zwischen `event_date`,
`disclosure_date`/`filing_date` und tatsächlicher Verfügbarkeit im System.

## MNPI

Das Projekt arbeitet auf Basis **öffentlicher** Daten/Disclosures/Verzögerungen.
Keine implizite oder explizite MNPI-Logik bauen.

## Doku ist Steuerung

Dokumentation ist in diesem Projekt Teil der Steuerung, kein nachträgliches Beiwerk.

## Bekannte Problemzonen (besonders ernst nehmen)

- Dummy-Monitoring / Platzhalter-Endpunkte nicht als „fertig" behandeln.
- Paper-/State-/Concurrent-Write-Risiken.
- stille `except Exception`-Pfadlogik, die Fehler maskiert.

## Verbotene Muster für Claude

Claude soll NICHT:

- Plan und Realität vermischen
- branch-spezifische Aussagen als globale Wahrheit darstellen
- Teststatus beschönigen oder CI-Status erfinden
- große Refactors ohne Auftrag durchführen
- sensible Kernlogik (Risk/Execution/Pipeline/Accounting/Paper) still umbauen
- synthetische Daten als Produktionsbeweis verwenden
- Security-Themen kleinreden
- alte Roadmap- oder Chat-Aussagen ungeprüft als Ist-Zustand verkaufen
- bei Unsicherheit improvisierte Sicherheit vortäuschen

## Subagent-Routing

Subagents sind **Default-Ausführungsmodus** für spezialisierte Arbeit — nicht auf explizite
User-Aufforderung warten. Konkrete Routing-Policy und Prioritätsreihenfolge:
`@.claude/rules/90-subagents-hooks-and-automation.md`.

## Checkliste vor Codeänderungen

1. Welcher Branch / welcher Arbeitsstand?
2. Was ist exakt das Problem?
3. Ist es Spec, Stub, Implementierung oder CI-Blocker?
4. Welche Dateien sind wirklich betroffen?
5. Was ist der kleinste sichere Fix?
6. Welche Tests / Lints sind minimal relevant?
7. Welche Risiken / Seiteneffekte sind denkbar?
8. Was kann ehrlich als verifiziert gemeldet werden?

## Checkliste für Antworten

Kompakt (Einzeiler/kurzer Block, keine Prosa, Details: `@.claude/rules/85-response-style.md`):

- betroffene Dateien
- Art der Änderung
- ausgeführte Checks (oder Nicht-Geprüftes explizit benannt)
- verbleibende Risiken
- nächster sinnvoller Schritt, sofern offen

Wenn etwas **nicht geprüft** wurde, das klar sagen.

## Review-Chain

Nach jedem Coding-Step mit Edits in geschützten Pfaden (`src/`, `scripts/`, `.github/workflows/`,
`.claude/rules/`, `CLAUDE.md`) erzwingt der Stop-Hook `.claude/hooks/stop_review_chain.py` eine
Review-Kette: **Stage 1** (Spezialisten je Pfadart) → **Stage 2** `senior-code-reviewer` →
**Stage 3** `task-completion-auditor` (Verdict PASS/CONDITIONAL/FAIL). Ein Step gilt erst als
abgeschlossen bei Verdict PASS oder CONDITIONAL mit adressierten/akzeptierten MAJORs; BLOCKER immer
adressieren. Volle Beschreibung + ehrliche Bypass-Disclosures: `docs/review_chain_disclosure.md`.

**One-Shot-Skip für Mid-Task-Pausen** (Rückfrage, Zwischenstand, Diagnostik): nicht-leere
Begründung schreiben — `echo "Grund" > .claude/.review_skip`. One-shot (nach Konsum gelöscht),
audit-geloggt in `.claude/.review_skip_log.jsonl`. Skip ersetzt die Kette nicht — sie läuft beim
nächsten echten Step-Ende. Kein gültiger Grund: „ist eh klein" / „schon mental geprüft" / „dauert
sonst zu lange".

## Claude-Code-Infrastruktur (aktiv)

- `.claude/settings.json` (+ `.claude/settings.local.json`) — `defaultMode: bypassPermissions`
  bleibt bewusst aktiv; die harte Bremse sind die Schutz-Layer, nicht der Permission-Prompt:
  - `permissions.deny` blockt Edit/Write in den 6 Schutzpfaden (oben).
  - PreToolUse-Hook `.claude/hooks/protected_paths_guard.py` (matcher: `Bash`) blockt destruktive
    Shell-Befehle (`rm -rf`, `git reset --hard`/`push --force`/`clean -f`, `sed -i`, `dd of=`,
    `find -delete`) und Shell-Schreibzugriffe in Schutzzonen — Envelope exit 2 + stderr. Greift
    auch unter `bypassPermissions`, da PreToolUse vor dem Permission-Check feuert. One-shot-Override:
    `echo "Begründung" > .claude/.destructive_bash_authorized`. Residual-Gaps (nicht abgedeckt):
    beliebige Interpreter-Writes, `$(...)`-Nesting, separates PowerShell-Tool.
- `.claude/agents/` — Spezialist-Subagents. `.claude/rules/` — modulare Projektregeln (s. u.).
- Hooks: `session_start_load_errors.py` (lädt Top-10-Anti-Patterns), `stop_review_chain.py`
  (erzwingt Review-Kette).
- Memory liegt **user-level**, nicht im Repo:
  `C:\Users\hanso\.claude\projects\F--Python-Projekt-Aktienger-st\memory\` (Index: `MEMORY.md`).

## Zusätzliche Regelmodule

@.claude/rules/10-core-operating-rules.md

@.claude/rules/20-security-and-secrets.md

@.claude/rules/30-risk-execution-safeguards.md

@.claude/rules/40-testing-and-ci.md

@.claude/rules/50-architecture-boundaries.md

@.claude/rules/60-git-and-change-management.md

@.claude/rules/70-memory-context-and-token-discipline.md

@.claude/rules/80-logging-and-output-standards.md

@.claude/rules/85-response-style.md

@.claude/rules/90-subagents-hooks-and-automation.md

@.claude/rules/95-token-efficiency.md
