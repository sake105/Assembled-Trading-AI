# Hygiene-Sprint: Quick-Wins nach der trading_cycle-Migration

**Ziel:** In einem Sprint (~3–4 Stunden) die Hygiene-Schulden abräumen, die der Audit aufgedeckt hat. Keine Architektur-Änderungen, keine neuen Features — nur Aufräumen.

**Hintergrund:** In `autonome weiterarbeit/AUDIT_TEIL3_Rest.md` und `REPO_AUDIT_ASSEMBLED_TRADING_AI.md` hast du viele dieser Punkte bereits dokumentiert. Diese Anleitung ist die operative Umsetzungs-Reihenfolge.

**Zeitlich:** Punkte 1–4 in einem Vormittag erledigbar. Punkt 5 (Bilder/System-Map) ist optional und kann später dran.

---

## Punkt 1: Ein-Zeiler-Hygiene (~20 Minuten)

Die kleinsten Änderungen mit dem höchsten Aufräum-Effekt. Alles in einem Commit zusammenfassbar: `chore: repo hygiene — remove profile output, update stale dates`.

### 1.1 Profile-Output löschen

```powershell
# Datei mit Sondernamen (U+F03A) — Bash quotet nicht zuverlässig, daher git rm direkt
git rm "FPython_ProjektAktiengerüst__profile_out.txt"
```

Falls Git über das Sonderzeichen meckert: in Cursor/VSCode den Datei-Explorer rechts-klicken → Delete → committen.

### 1.2 `.gitignore` erweitern

Vor das Block "OS/Tools" einfügen:

```gitignore
# Profile / benchmark outputs
*profile_out*.txt
*.prof
benchmark_output_*.txt
profile_*.json
```

### 1.3 Stale Datums-Header korrigieren

In `KNOWN_ISSUES.md` Zeile 3:
```diff
-**Letzte Aktualisierung:** 2025-01-15
+**Letzte Aktualisierung:** 2026-04-26
```

In `PROJEKT_STATUS.md` Zeile 3:
```diff
-**Letzte Aktualisierung:** 2025-01-15
+**Letzte Aktualisierung:** 2026-04-26
```

**Besser für die Zukunft:** Setze einen Pre-Commit-Hook, der diese Datei-Header automatisch auf `git log -1 --format=%ad` aktualisiert. Aber das ist eigener Sprint, jetzt nur die manuellen Werte fixen.

### 1.4 Commit

```powershell
git add .gitignore KNOWN_ISSUES.md PROJEKT_STATUS.md
git commit -m "chore: repo hygiene — remove profile output, update stale dates

- Delete 207 KB FPython_*profile_out.txt (Windows-Pfad-Output, gehört nicht ins Repo)
- Add *profile_out*, *.prof, benchmark_output_*, profile_*.json to .gitignore
- Update stale 2025-01-15 dates in KNOWN_ISSUES.md and PROJEKT_STATUS.md"
```

---

## Punkt 2: Verzeichnisse mit Leerzeichen umbenennen (~45 Minuten)

Drei Pfade mit Leerzeichen, die du selbst schon im Audit (`AUDIT_TEIL3_Rest.md` §11) zur Umbenennung vorgeschlagen hast.

### 2.1 Vor der Umbenennung — Referenzen finden

```powershell
# In PowerShell (Windows):
Get-ChildItem -Recurse -Include *.md,*.py,*.yaml,*.yml,*.toml,*.ini,*.ps1,*.bat,*.txt | `
  Select-String "autonome weiterarbeit|stand 3-12-2025|datensammlungen" | `
  Select-Object Path,LineNumber,Line | Format-Table -Wrap

# In Bash (Linux/WSL):
rg -l "autonome weiterarbeit|stand 3-12-2025|datensammlungen" \
  --type-add 'cfg:*.{md,py,yaml,yml,toml,ini,ps1,bat,txt}' --type cfg
```

Speichere die Trefferliste — du brauchst sie für Schritt 2.3.

### 2.2 Umbenennungen mit `git mv`

Damit Git die Historie als Rename und nicht als delete+add erkennt:

```powershell
git mv "autonome weiterarbeit" autonome_weiterarbeit
git mv "datensammlungen/altdaten/stand 3-12-2025" datensammlungen/altdaten/2025-12-03
git mv "docs/architecture/system_map/screenshots/system_map .png" docs/architecture/system_map/screenshots/system_map.png
```

**Wichtig — Cursor-Workspace und Indexer:** Cursor/VSCode haben gelegentlich Probleme, wenn `.cursorrules` oder Workspace-Dateien auf alte Pfade zeigen. Nach dem Rename einmal Cursor neu starten und Indexer durchlaufen lassen.

### 2.3 Referenzen aktualisieren

Aus Schritt 2.1 hast du eine Liste. Die wichtigsten Stellen aus dem Repo:

**Doku-Dateien (relativ harmlos, aber dranbleiben):**
- `archive/observability_graveyard_2026q2/README.md:7` → `autonome_weiterarbeit/AUDIT_TEIL2_...`
- `tests/test_free_stack_modules.py:1` Docstring → `autonome_weiterarbeit`
- `tests/test_non_paid_modules.py:1` Docstring → `autonome_weiterarbeit`

**Selbstreferenzen innerhalb der Audit-Dokumente:** Du wirst in `autonome_weiterarbeit/REPO_AUDIT_ASSEMBLED_TRADING_AI.md` und `AUDIT_TEIL3_Rest.md` selbst Erwähnungen der alten Pfade finden — die werden zu historisch korrekten Belegen, dass die Umbenennung **passiert ist**. Die kannst du stehen lassen, oder mit einem Hinweis ergänzen:
```markdown
> Hinweis: Diese Pfade wurden am 2026-04-26 umbenannt — siehe Commit `<hash>`.
```

**Doku-Dateien mit hardcodierten Windows-Pfaden:**
- `docs/ADVANCED_ANALYTICS_FACTOR_LABS.md:781`
- `docs/ARCHITECTURE_REVIEW_SUMMARY.md:232`
- `docs/DOWNLOAD_STRATEGY.md:29`

Alle drei zeigen auf:
```
F:\Python_Projekt\Aktiengerüst\datensammlungen\altdaten\stand 3-12-2025
```

Update zu:
```
F:\Python_Projekt\Aktiengerüst\datensammlungen\altdaten\2025-12-03
```

### 2.4 Tests laufen lassen

```powershell
pytest -m fast -q
```

Das ist die schnelle Suite (frühere phase4). Wenn die grün ist, sind keine Code-Pfade gebrochen.

### 2.5 Commit

```powershell
git add -A
git commit -m "refactor(repo): rename whitespace-paths to underscore/ISO-date

- 'autonome weiterarbeit/' → 'autonome_weiterarbeit/'
- 'datensammlungen/altdaten/stand 3-12-2025/' → 'datensammlungen/altdaten/2025-12-03/'
- 'system_map .png' → 'system_map.png' (no leading space before extension)

Whitespace in paths breaks shell scripts without quoting and is documented as
tech debt in AUDIT_TEIL3_Rest.md §11. Internal references in docs/tests
updated; Windows env vars in docs/ADVANCED_ANALYTICS_FACTOR_LABS.md and
similar files updated to new path."
```

---

## Punkt 3: Pandas-FutureWarning fixen (~10 Minuten)

Beim Audit stellte sich heraus: **alle 48 Warnungen** im `test_trading_cycle_v2`-Lauf kommen aus einer einzigen Zeile in `multifactor_signal.py`. Der Fix ist ein Zweizeiler — und sobald er drin ist, läuft die Test-Suite warnungsfrei (zumindest für diese Quelle).

### 3.1 Den Fix einbauen

Datei: `src/assembled_core/signals/multifactor_signal.py`, Zeile 133.

**Vorher:**
```python
zscores = result.groupby(timestamp_col, group_keys=False).apply(zscore_group)
```

**Nachher:**
```python
zscores = result.groupby(timestamp_col, group_keys=False).apply(
    zscore_group, include_groups=False
)
```

Das ist exakt das, was die FutureWarning vorschlägt: pandas wird ab einer kommenden Version die Group-Spalten standardmäßig nicht mehr in den Apply-Call reichen. `include_groups=False` macht das jetzt schon explizit, wodurch der Code zukunftssicher wird **und** die Warnung verschwindet.

### 3.2 Funktionalität verifizieren

```powershell
pytest tests/test_trading_cycle_v2.py -q
# Erwartung: 57 passed, 0 warnings (statt vorher 48 warnings)
```

Falls Tests rot werden: das Verhalten von `apply` mit/ohne `include_groups` ist subtil verschieden, wenn die Group-Spalte selbst in der Aggregations-Logik verwendet wird. Prüfe `zscore_group`-Funktion — wenn sie auf `timestamp_col` zugreift, musst du sie nach dem Fix anpassen (z.B. via `group.name` oder Index).

### 3.3 Commit

```powershell
git add src/assembled_core/signals/multifactor_signal.py
git commit -m "fix(signals): silence pandas FutureWarning in multifactor_signal

groupby().apply() with include_groups=False makes the existing behavior
explicit and future-proof for pandas 3.x. Eliminates 48 FutureWarnings
per test run of tests/test_trading_cycle_v2.py."
```

---

## Punkt 4: Strict-Warnings-Mode aktivieren (~30 Minuten)

Jetzt, wo die offensichtlichste Warnungsquelle weg ist, kannst du die Test-Suite so konfigurieren, dass **neue** Warnungen sichtbar werden. Das ist ein billiger Hebel gegen schleichenden Verfall — denn jede unkommentiert akzeptierte Warning ist ein potenzieller Bug-in-Wartung.

### 4.1 `pytest.ini` anpassen

**Aktueller Stand:**
```ini
addopts =
    -q
    --strict-markers
    -m "not external"
    --tb=short
    --disable-warnings
```

`--disable-warnings` versteckt Warnungen — das war der Grund, warum 48 FutureWarnings monatelang unbemerkt blieben.

**Empfohlener neuer Stand:**
```ini
addopts =
    -q
    --strict-markers
    -m "not external"
    --tb=short

filterwarnings =
    # Default: alle Warnings als Warnings anzeigen
    default
    # Von Drittanbietern bekannte, nicht behebbare Warnings ignorieren
    ignore::DeprecationWarning:pkg_resources
    ignore::DeprecationWarning:pydantic.*
```

`--disable-warnings` entfernt; stattdessen `filterwarnings = default` plus gezieltes Whitelisting für Drittanbieter, die du nicht selbst fixen kannst.

**Wichtig — kein `error::DeprecationWarning` als Standard.** Das hatte ich in der ursprünglichen Migrations-Anleitung empfohlen, ist aber zu radikal: jede neue pandas/numpy-Version würde dann ohne Eigenverschulden CI rot machen. Stattdessen: einmal pro Sprint einen Probelauf mit `pytest -W error::DeprecationWarning`, fixen, was geht, Rest dokumentieren.

### 4.2 Verifizieren

```powershell
pytest tests/test_trading_cycle_v2.py -q
# Erwartung: 57 passed (ohne 48-warnings-Suffix)
# Wenn doch noch Warnings auftauchen, sind das echte neue Funde
```

Wenn unerwartete Warnings auftauchen: Quelle prüfen. Häufige Verdächtige: `numpy.NaN` vs `numpy.nan`, deprecated `datetime.utcnow()`, `pkg_resources`-Aufrufe in Sub-Dependencies.

### 4.3 Commit

```powershell
git add pytest.ini
git commit -m "chore(test): show warnings instead of suppressing them

--disable-warnings hid 48 pandas FutureWarnings that should have surfaced
months ago. Switch to filterwarnings=default with explicit ignore-list for
third-party noise we cannot fix locally.

Future pre-release sprint: run 'pytest -W error::DeprecationWarning' to
catch new deprecations before they become errors in pandas/numpy major
releases."
```

---

## Punkt 5 (optional): System-Map und 12-MB-Screenshot (~1–2 Stunden)

Das ist der unangenehmste der Befunde, weil es eine echte Entscheidung verlangt: **gehören generierte Architektur-Visualisierungen ins Git oder nicht?**

### Optionen — eine wählen

**Option A: In Git LFS migrieren (empfohlen, wenn die Bilder wertvoll sind)**

Vorteile: Repo bleibt schlank für Klone, Files bleiben versioniert.

Nachteile: Setup-Aufwand, Git-LFS muss in CI verfügbar sein, manche Hoster (z.B. ältere GitHub-Free-Tiers) haben LFS-Quoten.

```powershell
# Git LFS installieren falls nicht da
git lfs install

# Tracking aktivieren
git lfs track "docs/architecture/system_map/screenshots/*.png"
git lfs track "docs/architecture/system_map/data/system_map.json"
git lfs track "docs/architecture/system_map/data/system_map_data.js"

git add .gitattributes

# Vorhandene Files migrieren — VORSICHT, schreibt Historie um
git lfs migrate import \
  --include="docs/architecture/system_map/screenshots/*.png,docs/architecture/system_map/data/system_map.json,docs/architecture/system_map/data/system_map_data.js" \
  --include-ref=refs/heads/main
```

**Achtung:** `git lfs migrate import` schreibt Git-History um. Das geht nur sauber, wenn du der einzige Mitwirkende bist (was bei dir aktuell zutrifft). Sonst per `--no-rewrite` als reguläres LFS-Add ohne History-Migration.

**Option B: Aus dem Repo entfernen, in CI generieren**

Wenn die System-Map ein **generierter Output** ist (was sie laut Header ist: `"generator_version": "0.2.0", "source_commit": "e226ef2"`), gehört sie nicht ins Repo. Stattdessen: ein CI-Job, der bei jedem Push die System-Map generiert und z.B. als GitHub-Pages-Site oder Workflow-Artifact hostet.

```powershell
# Files entfernen (bleiben in Git-History, aber neue Klone bekommen sie nicht mehr)
git rm docs/architecture/system_map/screenshots/system_map.png
git rm docs/architecture/system_map/data/system_map.json
git rm docs/architecture/system_map/data/system_map_data.js

# In .gitignore aufnehmen, damit lokale Generierungen nicht versehentlich
# wieder reingeraten
echo "docs/architecture/system_map/screenshots/*.png" >> .gitignore
echo "docs/architecture/system_map/data/system_map.json" >> .gitignore
echo "docs/architecture/system_map/data/system_map_data.js" >> .gitignore
```

Dann einen neuen Workflow `.github/workflows/system-map-build.yml`:
```yaml
name: Build System Map

on:
  push:
    branches: [main]
    paths: ['src/**']

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.10' }
      - run: pip install -e ".[dev]"
      - name: Generate system map
        run: python scripts/generate_system_map.py
      - uses: actions/upload-artifact@v4
        with:
          name: system-map
          path: docs/architecture/system_map/
          retention-days: 30
```

(Den `generate_system_map.py`-Generator hast du offenbar schon — du musst nur den Pfad anpassen.)

**Option C: Status quo behalten, dokumentieren**

Wenn keiner der Aufwände sich lohnt: einen Hinweis in `docs/architecture/system_map/README.md` anlegen, dass die Files generiert werden und nur committet werden, weil das einfacher als CI-Setup war. Damit ist es zumindest kein versteckter Schmerz mehr, sondern eine bewusste Entscheidung.

### Meine Empfehlung

**Option C jetzt, Option B in 2–3 Sprints.** Du bist gerade aus einer großen Migration raus; kein neuer Großumbau. Dokumentiere die Entscheidung, mach einen Issue in `KNOWN_ISSUES.md` mit Verweis auf Option B als Ziel, und mach weiter.

---

## Reihenfolge und Definition of Done

**Reihenfolge — strikt einhalten:**

1. Punkt 1 (20 min) → committen → Tests laufen
2. Punkt 2 (45 min) → committen → Tests laufen
3. Punkt 3 (10 min) → committen → Tests laufen
4. Punkt 4 (30 min) → committen → Tests laufen
5. Punkt 5 (optional, später)

**Nach Punkt 4 — Sprint-DoD:**

- `git status` ist sauber.
- Mindestens 4 atomare Commits, jeder mit aussagekräftiger Message.
- `pytest -m fast -q` läuft grün ohne FutureWarning-Block.
- `KNOWN_ISSUES.md` bekommt einen neuen Abschnitt:
  ```markdown
  ## 6. Hygiene-Sprint 2026-04-26 — abgeschlossen

  - [x] Profile-Output entfernt + .gitignore erweitert
  - [x] Whitespace-Pfade umbenannt (3 Stück)
  - [x] pandas FutureWarning in multifactor_signal.py gefixt
  - [x] pytest.ini auf filterwarnings=default umgestellt

  Offen (Punkt 5):
  - [ ] System-Map (1.5 MB JSON + 12 MB PNG) — Migrationsentscheidung
        siehe docs/MIGRATION_HYGIENE_2026Q2.md §5
  ```
- Ein Tag setzen: `git tag hygiene-2026q2 -m "Quick-Wins after trading_cycle migration"`.

---

## Was du dabei NICHT machen solltest

1. **Keine inhaltlichen Code-Änderungen.** Wenn du beim Aufräumen Bug X siehst, der dich juckt: Issue in `KNOWN_ISSUES.md`, weitergehen. Mischst du Hygiene mit Bugfixes, wird das Diff unlesbar.

2. **Keine `cli.py`-Modularisierung in diesem Sprint.** Das ist Punkt 5 aus meinem letzten Review (4007 Zeilen, 26 Subcommands). Eigener Sprint, eigene Anleitung.

3. **Keine `unified_paper_engine.py`-Refactor in diesem Sprint.** Das ist Punkt 6 — ähnliche Größenordnung wie die `trading_cycle`-Migration. Erst stabilisieren, dann angehen.

4. **Nicht versuchen, die alten Audit-Dokumente in `autonome_weiterarbeit/` aufzuräumen.** Die sind dein Backlog. Lass sie stehen, auch wenn sie alte Pfade referenzieren — das ist Geschichte, kein technischer Schuldenstand.

---

## Wenn etwas schiefgeht

**Tests werden rot nach Schritt 3 (FutureWarning-Fix):** Das `include_groups=False` ändert die übergebenen Spalten subtil. Schau in `zscore_group`-Funktion — falls sie `group[timestamp_col]` referenziert, gehört das in `group.name` oder einen separaten Parameter.

**Cursor indexiert nach Schritt 2 falsch:** Cursor neu starten, dann `Cmd/Ctrl+Shift+P` → "Reload Window". Wenn der Indexer dann immer noch alte Pfade ausspuckt, in `.cursorrules` und `.cursorignore` selbst nach alten Pfaden suchen.

**Du verlierst die Übersicht:** Punkt 1 abschließen und committen, bevor du mit Punkt 2 anfängst. Atomare Commits sind die Sicherheitsleinen — wenn ein Schritt schiefgeht, kannst du `git reset --hard HEAD~1` und fängst neu an, ohne andere Änderungen mit zu verlieren.
