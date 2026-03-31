# 80 Logging and Output Standards

## Zweck

Diese Regeln standardisieren Prints, Logs, Statusmeldungen und Ausgabeverhalten.

## Leitidee

Logs und Ausgaben sollen für Debugging, CI, Review und spätere Agentenarbeit nützlich sein.
Nicht laut, sondern strukturiert.

## Pflichtregeln

- Bevorzuge deterministische, knappe und klar lesbare Statusmeldungen.
- Jede wichtige Ausgabe soll eine klare Kategorie oder Bedeutung haben.
- Keine unnötigen Textwände in Laufzeit-Logs.
- Keine irrelevanten Debug-Prints dauerhaft im Produktivpfad.
- Keine Secrets, Tokens, Keys oder sensiblen Daten in Logs.

## Gute Statusstruktur

Wenn Statusmeldungen eingeführt oder angepasst werden, bevorzuge Muster wie:

- `[START] <schritt>`
- `[OK] <schritt>`
- `[WARN] <schritt>`
- `[ERROR] <schritt>`
- `[SKIP] <schritt>`

## Für Tests und CI

- Fehlermeldungen sollen konkret genug sein, um schnell lokalisieren zu können, aber nicht so breit, dass sie den Signalwert senken.
- Assertions sollen fachlich aussagekräftig sein.
- Debug-Ausgaben in Tests nur dann dauerhaft behalten, wenn sie echten Diagnosewert haben.

## Für Agentenfreundlichkeit

Ausgaben sollen so gebaut sein, dass ein späterer Agent oder Mensch leicht erkennen kann:

- welcher Schritt lief
- was geprüft wurde
- warum etwas scheiterte
- ob ein Fehler fachlich oder infrastrukturell ist

## Vermeide

- lockere Freitext-Prints ohne Prefix
- unstrukturierte Dump-Ausgaben
- „success“/„done“ ohne Kontext
- rauscharme und dennoch bedeutungslose Dauerlogs
