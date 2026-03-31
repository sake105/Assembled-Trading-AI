# 70 Memory, Context and Token Discipline

## Zweck

Diese Regeln verhindern Kontextdrift, doppelte Wissensspeicherung und unnötigen Tokenverbrauch.

## Grundprinzip

Claude soll in langen Projekten nicht versuchen, alles ständig komplett im aktiven Prompt zu tragen.
Stattdessen gelten Kompression, Modularisierung und präziser Kontextbezug.

## Pflichtregeln

- Halte die Root-`CLAUDE.md` knapp und grundlegend.
- Lege Detailregeln modular in importierten Dateien ab.
- Wiederhole Statuswissen nicht unnötig in mehreren Dateien.
- Erzeuge keine zweite Memory-Wahrheit, wenn die Information bereits in einem autoritativen Dokument liegt.
- Verwende lieber präzise Verweise auf Pfade, Module und Verträge als breite Wiederholungen.

## Kontextökonomie

Bevorzuge:

- diff-first Denken
- gezielte Dateinennung statt unscharfer Gesamtbeschreibungen
- kleine, spezialisierte Subagents statt riesiger Hauptkontexte
- kurze Statusblöcke mit klaren Kategorien
- imports und modulare Referenzen statt riesiger Monolith-Dateien

## Memory-Hygiene

Wenn neue Projektregeln oder Lessons Learned entstehen:

1. prüfen, ob sie dauerhaft sind
2. prüfen, ob sie global oder domänenspezifisch sind
3. nur an einem autoritativen Ort ablegen
4. veraltete oder doppelte Regeln konsolidieren

## Statusdarstellung

Arbeite mit kompakten Formulierungen wie:

- aktuelle Wahrheit
- offene Unsicherheit
- nächste kleinste Aktion
- belegte Prüfung
- ungeprüfte Annahme

## Token-Schutz

Vermeide:

- unnötige Wiederholung der gesamten Projekthistorie
- überlange Statusmemos ohne operative Relevanz
- ungefilterte Logmassen als Standardkontext
- das vollständige Nachladen irrelevanter Roadmaps für kleine Bugfixes
