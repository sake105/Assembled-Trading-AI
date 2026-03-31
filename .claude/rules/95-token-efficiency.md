# 95 Token Efficiency

## Zweck

Diese Regeln reduzieren unnötigen Tokenverbrauch in jeder Sitzung, ohne die Projektarbeit zu beeinflussen.
Sie gelten für Kommunikation, Dateiarbeit und Werkzeugwahl — nicht für fachliche Entscheidungen.

---

## Kommunikation

- Antworten so kurz wie nötig, nie länger als nötig.
- Keine Wiederholung der Nutzerfrage oder des Nutzerbefehls am Anfang der Antwort.
- Keine abschließenden Zusammenfassungen nach Tool-Aufrufen, wenn die Ergebnisse selbsterklärend sind.
- Keine Ankündigungen für triviale Aktionen („Ich werde jetzt die Datei lesen…").
- Keine Aufzählung aller möglichen Optionen, wenn eine klare Empfehlung möglich ist.
- Statusblöcke maximal 5–8 Zeilen; nur was operativ relevant ist.

---

## Dateilesen

- Vor dem Öffnen einer Datei: prüfen, ob der benötigte Teil bereits im Kontext ist.
- `Read` mit `offset` + `limit` verwenden, wenn nur ein Abschnitt benötigt wird.
- Für Struktursuche: `Grep` oder `Glob` statt vollständige Dateien öffnen.
- Für Codestruktur: `claude-mem:smart-explore` statt mehrere Dateien vollständig lesen.
- Dateien nicht mehrfach lesen, wenn ihr Inhalt im aktuellen Kontext bereits bekannt ist.

---

## claude-mem (installiert v10.6.2)

- **Sitzungsbeginn:** `claude-mem:mem-search` verwenden, um relevantes Vorwissen zu laden — statt ROADMAP_STATE, ROADMAP_LOG und andere große Docs von Grund auf neu einzulesen.
- **Sitzungsende:** `memory-tracker` Subagent aufrufen, wenn bedeutsame Entscheidungen, Status-Wechsel oder neue Annahmen entstanden sind.
- **Während der Arbeit:** Keine doppelte Ablage: was in claude-mem steht, nicht nochmal in den Hauptkontext laden.
- **Niemals:** memory-Inhalte pauschal in Antworten ausgeben — nur auf Anfrage oder wenn direkt relevant.

---

## Tool-Auswahl

- `Grep`/`Glob` vor `Read` — immer prüfen ob eine Suche ausreicht.
- `Bash` nur wenn kein dediziertes Tool verfügbar ist.
- Subagents für isolierte Research-Aufgaben: schützen den Hauptkontext vor Bulkdaten.
- Kein paralleler Einsatz mehrerer Subagents, wenn ein einzelner ausreicht.

---

## Kontext-Hygiene

- Große Logs, ROADMAP_LOG.md und ähnliche historische Dokumente nur gezielt nachladen — nie pauschal für kleine Fixes.
- Bei kleinen Bugfixes: nur betroffene Datei lesen, nicht das gesamte Modul.
- Nach jedem abgeschlossenen Roadmap-Schritt `/compact` ausführen, um Tokenverbrauch für Folgeaufgaben zu reduzieren.
- Wenn der Kontext voll läuft: `/compact` ausführen, bevor Arbeit blockiert wird.
- Keine Policy-/Config-Dateien (policy.yaml, pyproject.toml etc.) vollständig lesen, wenn nur ein Abschnitt benötigt wird.

---

## Was diese Regeln NICHT ändern

- Fachliche Entscheidungen im Projekt bleiben unberührt.
- Sensible Zonen (risk, execution, pipeline) behalten ihre vollen Schutzregeln.
- Test- und CI-Disziplin wird nicht gelockert.
- CLAUDE.md Grundregeln haben immer Vorrang vor diesen Effizienzregeln.
