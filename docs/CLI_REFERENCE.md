# CLI Reference - Assembled Trading AI

## Übersicht

Das zentrale CLI (`scripts/cli.py`) bietet eine einheitliche Schnittstelle für die wichtigsten Backend-Operationen.

## Installation & Verwendung

```powershell
# Im Projektverzeichnis, venv aktiviert
cd F:\Python_Projekt\Aktiengerüst
.\.venv\Scripts\Activate.ps1

# CLI aufrufen
python scripts/cli.py <subcommand> [options]
```

---

## Globale Optionen

### `--version`
Zeigt Version und Projekt-Informationen:
```powershell
python scripts/cli.py --version
```

**Ausgabe:**
```
cli.py 0.0.1
```

### `--help`
Zeigt Hilfe für das CLI oder einen Subcommand:
```powershell
python scripts/cli.py --help
python scripts/cli.py <subcommand> --help
```

---

## Subcommands

### 1. `info` - Projekt-Informationen

Zeigt eine Übersicht über verfügbare Subcommands und Dokumentation.

**Verwendung:**
```powershell
python scripts/cli.py info
```

**Ausgabe:**
- Liste aller Haupt-Subcommands
- Links zu Dokumentation
- Beispiel-Aufrufe

---

### 2. `run_daily` - EOD-Pipeline

Führt die vollständige End-of-Day-Pipeline aus: Execute, Backtest, Portfolio-Simulation, QA-Checks.

**Verwendung:**
```powershell
python scripts/cli.py run_daily --freq 1d
```

**Argumente:**

| Argument | Typ | Erforderlich | Beschreibung |
|----------|-----|--------------|--------------|
| `--freq` | `{1d,5min}` | ✅ Ja | Trading-Frequenz |
| `--universe` | `FILE` | ❌ Nein | Pfad zur Universe-Datei (default: watchlist.txt) |
| `--price-file` | `FILE` | ❌ Nein | Expliziter Pfad zur Preis-Datei |
| `--start-date` | `YYYY-MM-DD` | ❌ Nein | Startdatum für Preis-Datenfilterung |
| `--end-date` | `YYYY-MM-DD` | ❌ Nein | Enddatum für Preis-Datenfilterung |
| `--start-capital` | `AMOUNT` | ❌ Nein | Startkapital in USD (default: 10000.0) |
| `--skip-backtest` | Flag | ❌ Nein | Backtest-Schritt überspringen |
| `--skip-portfolio` | Flag | ❌ Nein | Portfolio-Simulation überspringen |
| `--skip-qa` | Flag | ❌ Nein | QA-Checks überspringen |
| `--commission-bps` | `BPS` | ❌ Nein | Commission in Basis-Punkten (überschreibt Default) |
| `--spread-w` | `WEIGHT` | ❌ Nein | Spread-Weight für Cost-Model (überschreibt Default) |
| `--impact-w` | `WEIGHT` | ❌ Nein | Market-Impact-Weight (überschreibt Default) |
| `--out` | `DIR` | ❌ Nein | Output-Verzeichnis (default: config.OUTPUT_DIR) |

**Beispiele:**
```powershell
# Standard-Lauf (1d)
python scripts/cli.py run_daily --freq 1d

# Mit Universe-Datei
python scripts/cli.py run_daily --freq 1d --universe watchlist.txt

# Mit angepasstem Startkapital
python scripts/cli.py run_daily --freq 1d --start-capital 50000

# Mit expliziter Preis-Datei
python scripts/cli.py run_daily --freq 5min --price-file data/sample/eod_sample.parquet

# Ohne Backtest
python scripts/cli.py run_daily --freq 1d --skip-backtest
```

---

### 3. `run_backtest` - Strategy-Backtest

Führt einen Strategy-Backtest mit dem Portfolio-Level-Backtest-Engine aus.

**Verwendung:**
```powershell
python scripts/cli.py run_backtest --freq 1d
```

**Argumente:**

| Argument | Typ | Erforderlich | Beschreibung |
|----------|-----|--------------|--------------|
| `--freq` | `{1d,5min}` | ✅ Ja | Trading-Frequenz |
| `--price-file` | `FILE` | ❌ Nein | Expliziter Pfad zur Preis-Datei |
| `--universe` | `FILE` | ❌ Nein | Pfad zur Universe-Datei (default: watchlist.txt) |
| `--strategy` | `NAME` | ❌ Nein | Strategy-Name (default: trend_baseline) |
| `--start-capital` | `AMOUNT` | ❌ Nein | Startkapital in USD (default: 10000.0) |
| `--with-costs` | Flag | ❌ Nein | Transaktionskosten einbeziehen (default: True) |
| `--no-costs` | Flag | ❌ Nein | Transaktionskosten deaktivieren |
| `--commission-bps` | `BPS` | ❌ Nein | Commission in Basis-Punkten (überschreibt Default) |
| `--spread-w` | `WEIGHT` | ❌ Nein | Spread-Weight für Cost-Model (überschreibt Default) |
| `--impact-w` | `WEIGHT` | ❌ Nein | Market-Impact-Weight (überschreibt Default) |
| `--out` | `DIR` | ❌ Nein | Output-Verzeichnis (default: config.OUTPUT_DIR) |
| `--generate-report` | Flag | ❌ Nein | QA-Report nach Backtest generieren |

**Beispiele:**
```powershell
# Standard-Backtest
python scripts/cli.py run_backtest --freq 1d --universe watchlist.txt

# Mit QA-Report
python scripts/cli.py run_backtest --freq 1d --universe watchlist.txt --generate-report

# Ohne Kosten
python scripts/cli.py run_backtest --freq 1d --no-costs

# Mit angepasstem Startkapital
python scripts/cli.py run_backtest --freq 5min --start-capital 50000 --generate-report
```

---

### 4. `run_phase4_tests` - Phase-4-Test-Suite

Führt die Phase-4-Regression-Test-Suite aus (~13s, 110 Tests).

**Verwendung:**
```powershell
python scripts/cli.py run_phase4_tests
```

**Argumente:**

| Argument | Typ | Erforderlich | Beschreibung |
|----------|-----|--------------|--------------|
| `--verbose` | Flag | ❌ Nein | Detaillierte Test-Ausgabe (-vv statt -q) |
| `--durations` | `N` | ❌ Nein | Zeige N langsamste Tests (z.B. 5 für --durations=5) |

**Beispiele:**
```powershell
# Standard-Testlauf
python scripts/cli.py run_phase4_tests

# Mit detaillierter Ausgabe
python scripts/cli.py run_phase4_tests --verbose

# Mit Dauer-Informationen
python scripts/cli.py run_phase4_tests --durations 5

# Beides kombiniert
python scripts/cli.py run_phase4_tests --verbose --durations 10
```

**PowerShell-Wrapper:**
```powershell
# Über PowerShell-Skript (dünner Wrapper)
.\scripts\run_phase4_tests.ps1
.\scripts\run_phase4_tests.ps1 -Verbose -Durations
```

---

## Vereinheitlichte Argumente

### Häufig verwendete Flags

| Flag | Beschreibung | Unterstützt in |
|------|--------------|----------------|
| `--freq` | Trading-Frequenz (`1d` oder `5min`) | `run_daily`, `run_backtest` |
| `--universe` | Pfad zur Universe-Datei | `run_daily`, `run_backtest` |
| `--price-file` | Expliziter Pfad zur Preis-Datei | `run_daily`, `run_backtest` |
| `--start-capital` | Startkapital in USD | `run_daily`, `run_backtest` |
| `--out` | Output-Verzeichnis | `run_daily`, `run_backtest` |
| `--commission-bps` | Commission in Basis-Punkten | `run_daily`, `run_backtest` |
| `--spread-w` | Spread-Weight | `run_daily`, `run_backtest` |
| `--impact-w` | Market-Impact-Weight | `run_daily`, `run_backtest` |

### Test-spezifische Flags

| Flag | Beschreibung | Unterstützt in |
|------|--------------|----------------|
| `--verbose` | Detaillierte Ausgabe | `run_phase4_tests` |
| `--durations` | Zeige langsamste Tests | `run_phase4_tests` |

---

## Typische Workflows

### 1. Täglicher EOD-Lauf

```powershell
# Vollständiger Pipeline-Lauf
python scripts/cli.py run_daily --freq 1d

# Mit angepasstem Startkapital
python scripts/cli.py run_daily --freq 1d --start-capital 50000
```

### 2. Strategy-Backtest mit QA-Report

```powershell
# Backtest mit Report
python scripts/cli.py run_backtest --freq 1d --universe watchlist.txt --generate-report
```

### 3. Test-Suite ausführen

```powershell
# Schneller Testlauf
python scripts/cli.py run_phase4_tests

# Mit Details
python scripts/cli.py run_phase4_tests --verbose --durations 5
```

---

## Exit-Codes

| Code | Bedeutung |
|------|-----------|
| `0` | Erfolgreich |
| `1` | Fehler (z.B. Pipeline-Fehler, Backtest-Fehler) |
| `> 1` | Pytest-Exit-Code (bei `run_phase4_tests`) |

---

## Weitere Dokumentation

- **Backend-Architektur:** `docs/ARCHITECTURE_BACKEND.md`
- **Legacy-Übersicht:** `docs/LEGACY_OVERVIEW.md`
- **Legacy-Mapping:** `docs/LEGACY_TO_CORE_MAPPING.md`
- **PowerShell-Wrapper:** `docs/POWERSHELL_WRAPPERS.md`
- **Testing-Commands:** `docs/TESTING_COMMANDS.md`

---

## Troubleshooting

### CLI nicht gefunden
```powershell
# Prüfe, ob du im richtigen Verzeichnis bist
cd F:\Python_Projekt\Aktiengerüst

# Prüfe, ob venv aktiviert ist
.\.venv\Scripts\Activate.ps1
```

### Subcommand nicht erkannt
```powershell
# Zeige verfügbare Subcommands
python scripts/cli.py --help
```

### Hilfe zu einem Subcommand
```powershell
# Detaillierte Hilfe
python scripts/cli.py <subcommand> --help
```

