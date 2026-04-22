# 40 Testing and CI

## Zweck

Diese Regeln erzwingen Testehrlichkeit, saubere CI-Disziplin und realistische Aussagen zum Prüfstatus.

## Grundprinzip

Claude darf nie so tun, als sei etwas vollständig geprüft, wenn nur ein Teil der Prüfungen lief.

## Pflichtregeln

- Vor dem Vorschlag von Codeänderungen immer überlegen, welche minimale Prüfung fachlich notwendig ist.
- Tests so klein und gezielt wie möglich wählen, aber nicht so klein, dass relevante Risiken ungetestet bleiben.
- Zwischen lokalem Test, Teiltest, Marker-abhängigem Test und CI-Run klar unterscheiden.
- Collection-Fehler, Skips und optionale Dependencies immer explizit benennen.

## Aussagen zum Status

Erlaube nur präzise Aussagen wie:

- „lokal nicht ausgeführt“
- „nur statische Prüfung gedacht“
- „gezielter Test für Modul X ausgeführt“
- „vollständige CI nicht bestätigt“
- „wegen bekannter Collection-Probleme nicht als global grün interpretierbar“

Vermeide Aussagen wie:

- „alles grün“
- „fertig getestet“
- „CI-safe“

wenn diese nicht wirklich belegt sind.

## Standard-Teststrategie für dieses Repo

- Bei kleinen Utility-Änderungen: gezielter Unit-/Modultest.
- Bei Änderungen in `pipeline`, `execution`, `portfolio`, `accounting`, `qa`, `data`: gezielte Tests plus relevante angrenzende Checks.
- Bei Änderungen an CI, Packaging, Imports, CLI oder Projektstruktur: mindestens gezielte Aufruf-/Importprüfung und möglichst passender Workflow-/Smoke-Bezug.

## Optionale Dependencies

Wenn Tests von optionalen Paketen abhängen:

- Nicht so tun, als sei das ein normaler Fehler, wenn es ein erwarteter Skip ist.
- Marker, Extras und Installationsannahmen sauber benennen.
- Keine Scheinreparaturen nur für lokale Umgebungsprobleme.

## Bekannte Projektrealität

Wenn Collection-Probleme oder historische Testartefakte existieren, muss Claude diese Realität in seine Bewertung einbeziehen.

## CI-Änderungen

Bei Änderungen an Workflows oder CI-Skripten muss Claude zusätzlich nennen:

- ob die Änderung Ubuntu oder Windows betrifft
- ob Blocking-Jobs betroffen sind
- ob der Effekt nur lokal gedacht oder auch CI-relevant ist

## Dependency-Drift zwischen pyproject.toml und requirements.txt

### Grundregel

Lokale Installation via `pip install -e ".[dev]"` und CI-Installation via `requirements.txt` können
unterschiedliche Paketversionen auflösen. Diese Abweichung ist in diesem Repo nachweislich vorhanden
(z. B. `pandas>=2.0.0` vs. `pandas==2.3.3`, `numpy>=1.24.0` vs. `numpy==2.3.3`).

Claude darf bei Installations-, Import-, Typing-, Test- oder CI-Fehlern nicht stillschweigend annehmen,
dass beide Installationspfade identische Versionen liefern.

### Pflichtregeln

- Wenn ein Problem lokal nicht reproduzierbar ist, aber in CI auftritt — oder umgekehrt —
  ist Dependency-Drift eine der ersten Ursachen, die zu prüfen ist.
- `pyproject.toml` (Ranges) und `requirements.txt` (Pins) müssen in solchen Fällen
  explizit verglichen und Abweichungen namentlich benannt werden.
- Drift darf nicht als Nebenthema behandelt werden, wenn sie plausibel CI-relevant ist.
- Die Aussage „lokal läuft es, also passt CI auch" ist in diesem Repo nicht zulässig.

### Unterscheidungspflicht

Drift, optionale Dependencies und bekannte Collection-Probleme sind drei verschiedene Ursachen
und dürfen nicht vermischt werden:

- **Dependency-Drift:** `pyproject.toml` erlaubt eine Version, `requirements.txt` pinnt eine andere —
  führt zu unterschiedlichem Verhalten bei gleichen Tests in unterschiedlichen Environments.
- **Optionale Dependencies:** Pakete wie `scipy` oder `scikit-learn` sind nicht in allen Environments
  installiert — führt zu erwarteten Skips (`requires_scipy`, `requires_sklearn`), nicht zu Drift.
- **Collection-Probleme:** Historisch gab es Phasen mit ungefähr 19 Testdateien, die bei der
  Collection fehlschlugen (unfertige Stubs). Dieser Stand ist seit April 2026 aufgeräumt —
  Stand 2026-04-22: 5417 Tests werden ohne Collection-Errors gesammelt. Der Zahlwert „19" gilt
  daher als veraltet; bei aktuellem Verdacht immer frisch via `pytest --collect-only` prüfen.

### Aussagenpflicht bei Drift-Verdacht

Wenn Drift als mögliche Ursache erkannt wird, muss Claude explizit benennen:

- welche Pakete betroffen sind
- welche Version-Range `pyproject.toml` erlaubt
- welche Version `requirements.txt` pinnt
- ob das konkrete Problem durch diese Spanne erklärbar ist
- ob der Fix in der Range, im Pin oder im Code liegt

## Pflicht vor Aufgabenabschluss

Bevor eine Roadmap-Aufgabe oder ein Feature als „abgeschlossen" gemeldet wird, müssen folgende Schritte ausgeführt worden sein:

1. Gezielte Tests für die betroffenen Module laufen (pytest, kein Blindflug).
2. Ein Bugrun über die relevante Suite (mindestens phase12 oder äquivalent) ist ausgeführt worden.
3. Alle neuen Testfälle pass. Keine neu eingeführten Fehler.
4. Testergebnis explizit gemeldet: Anzahl pass/fail, Suite-Name, Datum.

„Lokal getestet" ist nur zulässig, wenn diese Schritte tatsächlich ausgeführt wurden.
