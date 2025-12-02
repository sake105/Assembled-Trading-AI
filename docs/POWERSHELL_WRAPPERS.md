# PowerShell Wrapper - Übersicht

## Ziel

Dieses Dokument listet alle PowerShell-Skripte (`.ps1`) im Projekt auf und zeigt, welche als **dünne Wrapper** um das zentrale Python-CLI (`scripts/cli.py`) fungieren und welche noch **Legacy-Skripte** sind.

---

## ✅ Umgestellt auf Python-CLI

### `scripts/run_phase4_tests.ps1`

**Status:** ✅ Umgestellt auf Python-CLI

**Ruft auf:** `python scripts/cli.py run_phase4_tests [--verbose] [--durations N]`

**Beschreibung:**
- Thin wrapper für die Phase-4-Test-Suite
- Mappt PowerShell-Parameter (`-Verbose`, `-Durations`) auf CLI-Argumente (`--verbose`, `--durations`)
- Aktiviert venv und ruft das Python-CLI auf

**Verwendung:**
```powershell
.\scripts\run_phase4_tests.ps1
.\scripts\run_phase4_tests.ps1 -Verbose -Durations
```

---

## 🚧 Legacy-Skripte (noch nicht umgestellt)

### `scripts/run_all_sprint10.ps1`

**Status:** 🚧 Legacy (komplex, ruft mehrere Legacy-Python-Skripte auf)

**Ruft auf:**
- `scripts/live/pull_intraday.ps1` (Daten-Pull)
- Inline Python-Code (Resampling 1m → 5m)
- `scripts/sprint9_execute.py` (Legacy Execute)
- `scripts/sprint9_backtest.py` (Legacy Backtest)
- `scripts/sprint10_portfolio.py` (Legacy Portfolio)

**Beschreibung:**
- Komplexes Orchestrator-Skript für Sprint 10
- Führt vollständigen Pipeline-Durchlauf aus (Pull → Resample → Execute → Backtest → Portfolio)
- **Zukünftige Migration:** Könnte auf `python scripts/cli.py run_daily` umgestellt werden, erfordert aber größere Refaktorierung

**Verwendung:**
```powershell
.\scripts\run_all_sprint10.ps1 -Symbols "AAPL,MSFT" -Days 2 -Freq 5min
```

---

### `scripts/run_live_pipeline.ps1`

**Status:** 🚧 Legacy

**Ruft auf:**
- `scripts/live_download.py` (falls vorhanden)
- `scripts/run_sprint8_rehydrate.ps1` (Legacy)
- `scripts/sprint8_execution.py` (Legacy)

**Beschreibung:**
- Orchestriert Live-Pipeline-Schritte
- **Zukünftige Migration:** Könnte auf `python scripts/cli.py run_daily` umgestellt werden

---

### `scripts/live/pull_intraday.ps1`

**Status:** 🚧 Legacy

**Ruft auf:** `scripts/live/pull_intraday.py`

**Beschreibung:**
- Wrapper für Intraday-Daten-Pull
- **Zukünftige Migration:** Könnte auf `python scripts/cli.py pull_data` (zukünftig) umgestellt werden

---

### `scripts/31_assemble_intraday.ps1`

**Status:** 🚧 Legacy (enthält eingebetteten Python-Code)

**Beschreibung:**
- Erzeugt assembled_intraday.parquet aus synthetischen Daten
- Enthält Python-Code als Here-String
- **Zukünftige Migration:** Könnte in Python-Modul ausgelagert werden

---

### `scripts/50_resample_intraday.ps1`

**Status:** 🚧 Legacy (enthält eingebetteten Python-Code)

**Beschreibung:**
- Resampling von Intraday-Daten zu höheren Frequenzen
- Enthält Python-Code in temporärer Datei
- **Zukünftige Migration:** Könnte auf `python scripts/cli.py resample_data` (zukünftig) umgestellt werden

---

### `scripts/51_qc_intraday_gaps.ps1`

**Status:** 🚧 Legacy (enthält eingebetteten Python-Code)

**Beschreibung:**
- QC-Check für Lücken in Intraday-Zeitreihen
- Enthält Python-Code als Here-String
- **Zukünftige Migration:** Könnte in Python-Modul ausgelagert werden

---

### `scripts/52_make_acceptance_intraday_sprint7.ps1`

**Status:** 🚧 Legacy

**Beschreibung:**
- Erstellt Acceptance-Report für Sprint 7
- Liest JSON-Reports und generiert Markdown
- **Zukünftige Migration:** Könnte in Python-Modul ausgelagert werden

---

### Weitere Legacy-Skripte

- `scripts/sprint9_dashboard.ps1` - Legacy Dashboard-Generierung
- `scripts/sprint9_cost_grid.ps1` - Legacy Cost-Grid-Analyse
- `scripts/sprint9_backtest.ps1` - Legacy Backtest-Wrapper
- `scripts/sprint10_portfolio.ps1` - Legacy Portfolio-Wrapper
- `scripts/sprint10_portfolio_simple.ps1` - Legacy Portfolio (robuste Version)
- `scripts/sprint10_param_sweep.ps1` - Legacy Parameter-Sweep
- `scripts/sprint8_cost_model.ps1` - Legacy Cost-Model
- `scripts/run_sprint8_rehydrate.ps1` - Legacy Rehydrate

---

## 🛠️ Tool-Skripte (keine Migration geplant)

Diese Skripte sind reine Utilities und werden nicht auf das CLI umgestellt:

- `scripts/tools/activate_python.ps1` - Venv-Aktivierung
- `scripts/tools/git_sync.ps1` - Git-Sync
- `scripts/tools/init_git.ps1` - Git-Initialisierung
- `scripts/tools/notify_discord.ps1` - Discord-Benachrichtigungen
- `scripts/tools/package_artifacts.ps1` - Artefakt-Packaging
- `scripts/tools/convert_heredocs.ps1` - Heredoc-Konvertierung
- `scripts/tools/fix_indent.ps1` - Indentation-Fix
- `scripts/tools/fix_all_project.ps1` - Projekt-Fixes
- `scripts/ps/ps_py_utils.ps1` - PowerShell-Python-Utilities
- `scripts/ps/fix_heredocs.ps1` - Heredoc-Fixes

---

## Zusammenfassung

| Kategorie | Anzahl | Beispiele |
|----------|--------|-----------|
| ✅ Umgestellt auf CLI | 1 | `run_phase4_tests.ps1` |
| 🚧 Legacy (Migration geplant) | ~15 | `run_all_sprint10.ps1`, `run_live_pipeline.ps1`, etc. |
| 🛠️ Tool-Skripte | ~10 | `tools/*.ps1`, `ps/*.ps1` |

---

## Migrations-Strategie

1. **Einfache Wrapper zuerst:** Skripte, die nur ein Python-Skript aufrufen, werden zuerst umgestellt (z. B. `run_phase4_tests.ps1` ✅).

2. **Komplexe Orchestratoren später:** Skripte wie `run_all_sprint10.ps1` erfordern größere Refaktorierung, da sie mehrere Legacy-Python-Skripte orchestrieren.

3. **Eingebetteter Python-Code:** Skripte mit Here-Strings (z. B. `31_assemble_intraday.ps1`) sollten in Python-Module ausgelagert werden, bevor sie auf das CLI umgestellt werden.

4. **Tool-Skripte bleiben:** Reine Utilities bleiben als PowerShell-Skripte erhalten.

---

## Nächste Schritte

- [ ] `run_all_sprint10.ps1` analysieren und auf `run_daily` umstellen (nach Refaktorierung der Legacy-Python-Skripte)
- [ ] `run_live_pipeline.ps1` auf `run_daily` umstellen
- [ ] Eingebetteten Python-Code aus PS-Skripten in Python-Module auslagern
- [ ] Weitere einfache Wrapper identifizieren und umstellen

