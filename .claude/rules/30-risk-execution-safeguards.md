# 30 Risk and Execution Safeguards

## Zweck

Diese Regeln schützen die sensibelsten Funktionsbereiche des Trading-Systems.

## Hochsensible Domänen

Als besonders schützenswert gelten:

- `src/assembled_core/execution/**`
- `src/assembled_core/pipeline/**`
- `src/assembled_core/portfolio/**`
- `src/assembled_core/accounting/**`
- `src/assembled_core/qa/**`
- `src/assembled_core/data/**` sofern PIT-, Timing- oder Backtest-Realismus betroffen ist
- alle Kill-Switch-, Pre-Trade-, Risk-Control-, Order-Generation- und Exposure-bezogenen Komponenten

## Schutzprinzip

In diesen Bereichen gelten strengere Standards als in normalen Utility-Dateien.
Claude darf hier nicht kreativ umstrukturieren, sondern muss konservativ, nachvollziehbar und testorientiert arbeiten.

## Pflichtregeln

- Keine Verhaltensänderung in Risk-/Execution-Pfaden ohne explizite Begründung.
- Keine Änderung an Risk-Grenzen, Limits, Checks, State-Machine-Verhalten oder Guardrails ohne klaren Auftrag.
- Keine stille Änderung von Signatur, Reihenfolge, Side Effects oder Fallback-Verhalten.
- Keine Lockerung von Safety-Checks, um Tests „einfach grün zu bekommen“.
- Keine Änderung an PIT-Safety, Latenzlogik, Fill-/Ledger-/Reconcile-Verhalten ohne gezielte Prüfung.

## Wenn in diesen Bereichen gearbeitet wird

Immer zusätzlich prüfen:

1. Welche Invarianten sollen erhalten bleiben?
2. Welche Downstream-Module hängen daran?
3. Ist das Verhalten deterministisch oder bewusst zustandsbehaftet?
4. Welche Tests oder Smoke-Checks müssen gezielt laufen?
5. Besteht Gefahr einer stillen Portfolio-, Cost-, Fill- oder Exposure-Veränderung?

## Berichtsregeln

Bei Änderungen in diesen Bereichen muss Claude im Ergebnis explizit nennen:

- was fachlich verändert wurde
- was bewusst nicht verändert wurde
- welche Schutzannahmen gelten
- welche Tests oder Checks ausgeführt wurden
- ob Risiko auf Business-Logik-Ebene besteht

## Standardpräferenz

Wenn ein Fix möglich ist durch:

- lokalen Guard-Fix
- zusätzliche Validierung
- besseren Fehlerpfad
- klarere Preconditions

bevorzuge diese Wege vor größerer Neuverdrahtung.
