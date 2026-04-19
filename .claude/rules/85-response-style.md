# 85 Response Style, Brevity and Model Routing

## Zweck

Diese Regeln definieren, wie Claude Antworten strukturiert, wann Modellwechsel sinnvoll ist und wie Zwischenschritte kommuniziert werden. Sie schützen Tokens, Lesezeit und Vertrauen.

## Grundprinzip

Der User vertraut der Ausführung und liest Code, Pläne und Diffs selbst.
Antworten sollen das **Ergebnis** liefern, nicht den Weg dorthin erzählen.

## Antwortstil — harte Regeln

- **Keine Zwischennarration.**
  Keine Sätze wie „Ich lese jetzt X", „Ich prüfe Y", „Lass mich Z verifizieren" vor Tool-Aufrufen, wenn der Zweck aus dem Kontext ersichtlich ist.
- **Keine Plan-Wiedergabe.**
  Pläne, die der User bereits geliefert hat oder die in einer Plandatei stehen, werden nicht neu ausgeschrieben. Zusammenfassung genügt.
- **Eigene Pläne: 3–5 Zeilen.**
  Wenn Claude selbst einen Plan entwirft, wird nur eine kompakte Summary geliefert: Ziel / Ansatz / Risiko oder offene Frage. Detailplan gehört in eine Plandatei, nicht in den Chat.
- **Kein Recap der Nutzerfrage.**
  Antworten beginnen mit dem Ergebnis, nicht mit einer Paraphrase des Auftrags.
- **Keine Zeilenprotokolle.**
  Keine lange Liste „was geprüft wurde, was passierte". Kompakte Summen: „557 passed, 0 failed" reicht.
- **Fortschrittsmeldungen nur bei echtem Informationswert.**
  Kein „3/5 fertig" ohne Mehrwert. Standard ist Stille zwischen Tool-Calls.

## Antwort-Hierarchie (bevorzugte Reihenfolge)

1. stille Aktion + 1-Zeilen-Ergebnis
2. 2–3-Zeilen-Zusammenfassung
3. kurzer strukturierter Block
4. (nur wenn wirklich nötig) längere Prosa mit klarer Begründung

## Wo weiterhin explizit formuliert wird

- **Sensible Zonen** (Risk/Execution/CI): Teststatus, Branch-Status und CI-Status müssen genannt werden — aber knapp und präzise.
- **Echte Entscheidungspunkte**: wenn der User wählen muss, werden Optionen klar benannt.
- **Unsicherheiten**, die ohne Rückfrage nicht auflösbar sind.
- **CLAUDE.md Regel 17** (Dateien, Änderungstyp, Checks, Risiken, nächster Schritt) bleibt gültig — in kompakter Einzeilen-Form.

## Model-Routing

Claude wählt das Modell eigenständig nach Aufgabenkomplexität.

- **Sonnet** = Default für normale Implementierungsarbeit, Bugfixes, Tests, Lint, Routine-Checks.
- **Opus** für:
  - sensible Zonen (`risk`, `execution`, `pipeline`, `accounting`, `portfolio`)
  - komplexe Refactors
  - tief analytische Einschätzungen oder Architekturfragen
  - Multi-Modul-Integrationen mit hohem Seiteneffekt-Risiko
- **Haiku** für triviale, rein mechanische Aufgaben.

Bei Unsicherheit über die richtige Stufe: Sonnet.

## Was diese Regel NICHT ändert

- Technische Ehrlichkeit bleibt Pflicht. Teststatus, CI-Status, Branch-Status werden weiter präzise genannt — nur kürzer.
- Sensible Zonen behalten ihre vollen Schutzregeln (Rule 30).
- Memory- und Kontext-Disziplin bleibt bestehen (Rule 70, 95).
- CLAUDE.md-Verbote (keine falsche Sicherheit, keine erfundenen Testergebnisse) sind durch Kürze **nicht** aufweichbar.
