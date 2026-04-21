# System Map — Assembled-Trading-AI

Interaktive HTML-Systemkarte für alle ~514 Module über 22 Kerndomänen.

## Schnellstart

```bash
# 1. Vendor-Libs einmalig herunterladen (Internet nötig)
python docs/architecture/system_map/assets/vendor/download_vendors.py

# 2. Karte aus Codebase generieren
python scripts/architecture/generate_system_map.py --report

# 3. Validierung
python scripts/architecture/validate_system_map.py

# 4. index.html im Browser öffnen (Chrome/Firefox)
```

## Regenerieren

```bash
python scripts/architecture/generate_system_map.py
```

Danach F5 im Browser.

## Manuelle Korrekturen

`data/system_map_overrides.yaml` — Status-Korrekturen, Duplikat-Gruppen, Orphan-Overrides.

## Keyboard-Shortcuts

| Key | Aktion |
|---|---|
| `/` | Suche |
| `F` | Fit |
| `R` | Reset Zoom |
| `M` | Mini-Map |
| `H` | Heat-Map |
| `D` | Dark/Light |
| `?` | Alle Shortcuts |

## Dateien

| Datei | Zweck |
|---|---|
| `index.html` | Viewer |
| `data/system_map.json` | Kartendaten (generiert) |
| `data/system_map_overrides.yaml` | Manuelle Korrekturen |
| `scripts/architecture/generate_system_map.py` | Generator |
| `scripts/architecture/validate_system_map.py` | Validator |
| `scripts/architecture/diff_system_map.py` | Delta-Vergleich |
