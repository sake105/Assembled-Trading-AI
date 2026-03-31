# Assembled-Trading-AI Claude Rules

Diese Dateien sind **eine projektinterne Modularisierung** für Claude Code.
Sie sind **kein offiziell magischer Claude-Code-Ordner**. Damit Claude Code sie sicher nutzt,
werden sie aus der Root-`CLAUDE.md` per `@pfad` importiert.

## Empfohlene Ablage

Diese Dateien gehören nach:

`.claude/rules/`

also in deinem Projekt konkret nach:

`F:\Python_Projekt\Aktiengerüst\.claude\rules\`

## Aktivierung

Füge in der Root-Datei `CLAUDE.md` diese Zeilen ein:

```md
## Zusätzliche Regelmodule
@.claude/rules/10-core-operating-rules.md
@.claude/rules/20-security-and-secrets.md
@.claude/rules/30-risk-execution-safeguards.md
@.claude/rules/40-testing-and-ci.md
@.claude/rules/50-architecture-boundaries.md
@.claude/rules/60-git-and-change-management.md
@.claude/rules/70-memory-context-and-token-discipline.md
@.claude/rules/80-logging-and-output-standards.md
@.claude/rules/90-subagents-hooks-and-automation.md
```

## Ziel dieser Struktur

Die Root-`CLAUDE.md` bleibt die **Verfassung**.
Die Dateien hier sind die **vertieften Spezialregeln**.

So bleibt der Hauptkontext klar, während Claude bei Bedarf die präzisen Regeln nachlädt.

## Priorität

1. `10-core-operating-rules.md`
2. `20-security-and-secrets.md`
3. `30-risk-execution-safeguards.md`
4. `40-testing-and-ci.md`
5. `50-architecture-boundaries.md`
6. `60-git-and-change-management.md`
7. `70-memory-context-and-token-discipline.md`
8. `80-logging-and-output-standards.md`
9. `90-subagents-hooks-and-automation.md`
