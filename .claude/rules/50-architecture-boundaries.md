# 50 Architecture Boundaries

## Zweck

Diese Regeln schützen die Modulgrenzen und vermeiden architektonische Drift.

## Leitidee

Assembled-Trading-AI ist kein einzelnes Script, sondern ein modulares System mit Kernlogik, Entry Points, API, QA und Reporting.

## Architektur-Regeln

- Respektiere vorhandene Modulgrenzen, solange kein expliziter Re-Architecture-Auftrag vorliegt.
- Verschiebe Code nicht zwischen Schichten, nur weil es lokal bequemer erscheint.
- Vermeide neue Kopplung zwischen `scripts/` und internem Core, wenn es sauberere Entry-Point-Nutzung gibt.
- Vermeide zusätzliche `sys.path`-Workarounds, wenn ein Paket- oder Importfix die bessere Lösung ist.
- Schaffe keine zweite Wahrheit für bestehende Funktionalität.

## Besonders sensible Strukturgrenzen

- `src/assembled_core/**` ist der bevorzugte Kernbereich.
- `scripts/**` sind primär Entry Points, Runner oder Hilfsskripte, nicht der Ort für neue Kernlogik.
- `tests/**` spiegeln und schützen Verträge; Tests dürfen nicht zur Umgehung fachlicher Probleme missbraucht werden.
- `.github/workflows/**` sind Betriebslogik, nicht Experimentierfläche.

## Keine Doppelstrukturen

Vermeide:

- parallele zweite Implementierungen ohne Migrationsplan
- Legacy-Kopie plus neue Kopie ohne klare Autorität
- ähnliche Hilfsfunktionen an mehreren Orten
- neue Konfigurationsorte ohne Bedarf
- neue „temporary“ Dateien, die faktisch dauerhaft werden

## Änderungsregel

Wenn du Code platzierst oder verschiebst, frage immer:

1. Ist das wirklich die richtige Schicht?
2. Entsteht dadurch eine zweite Wahrheit?
3. Wird Import-, Test- oder Verantwortungslogik unklarer?
4. Erschwert das spätere Packaging oder CI?

## Dokumentationsregel

Bei Strukturänderungen immer explizit benennen:

- alter Ort
- neuer Ort
- Grund für den Wechsel
- betroffene Imports / Call-Sites
- notwendige Folgearbeiten
