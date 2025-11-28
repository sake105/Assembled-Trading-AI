# EOD Pipeline Orchestration

## Übersicht

Das EOD-Pipeline-Skript (`scripts/run_eod_pipeline.py`) führt die gesamte Pipeline in einem einzigen Lauf aus und erstellt ein maschinenlesbares Run-Manifest.

**Zweck:** Automatisierte End-to-End-Ausführung der Trading-Pipeline mit konsistenter Fehlerbehandlung und Manifest-Generierung.

---

## Verwendung

### Basis-Kommando

```bash
python scripts/run_eod_pipeline.py --freq 1d --start-capital 10000
```

### Optionen

**Erforderlich:**
- `--freq`: Trading-Frequenz (`"1d"` oder `"5min"`)

**Optional:**
- `--start-capital`: Startkapital (default: 10000.0)
- `--skip-backtest`: Backtest-Schritt überspringen
- `--skip-portfolio`: Portfolio-Schritt überspringen
- `--skip-qa`: QA-Schritt überspringen
- `--price-file`: Expliziter Pfad zur Preis-Datei
- `--commission-bps`: Commission in Basis-Punkten (default: aus Cost-Model)
- `--spread-w`: Spread-Weight (default: aus Cost-Model)
- `--impact-w`: Impact-Weight (default: aus Cost-Model)
- `--out`: Output-Verzeichnis (default: aus config.OUTPUT_DIR)

### Beispiele

**Vollständiger Lauf (1d):**
```bash
python scripts/run_eod_pipeline.py --freq 1d --start-capital 10000
```

**5-Minuten-Pipeline:**
```bash
python scripts/run_eod_pipeline.py --freq 5min --start-capital 10000
```

**Ohne Backtest:**
```bash
python scripts/run_eod_pipeline.py --freq 1d --skip-backtest
```

**Mit custom Cost-Parametern:**
```bash
python scripts/run_eod_pipeline.py --freq 1d --commission-bps 0.5 --spread-w 0.3 --impact-w 0.6
```

---

## Pipeline-Schritte

Die Pipeline führt folgende Schritte in dieser Reihenfolge aus:

### 1. Preis-Daten-Prüfung

- Prüft, ob Preis-Daten für die gegebene Frequenz existieren
- Bei Fehlern: Setzt `failure_flag`, fährt aber fort (schreibt Manifest)

### 2. Execute

- Lädt Preise
- Berechnet EMA-Signale
- Generiert Orders
- Schreibt `output/orders_{freq}.csv`

### 3. Backtest

- Simuliert Equity-Kurve ohne Kosten
- Berechnet Performance-Metriken
- Schreibt:
  - `output/equity_curve_{freq}.csv`
  - `output/performance_report_{freq}.md`

### 4. Portfolio

- Simuliert Portfolio mit Transaktionskosten
- Schreibt:
  - `output/portfolio_equity_{freq}.csv`
  - `output/portfolio_report_{freq}.md`

### 5. QA

- Führt alle QA-Checks aus (prices, orders, portfolio)
- Aggregiert Status

---

## Run-Manifest

Nach jedem Lauf wird ein Run-Manifest als JSON geschrieben:

**Pfad:** `output/run_manifest_{freq}.json`

**Beispiel-Inhalt:**
```json
{
  "freq": "1d",
  "start_capital": 10000.0,
  "completed_steps": [
    "execute",
    "backtest",
    "portfolio",
    "qa"
  ],
  "qa_overall_status": "ok",
  "qa_checks": [
    {
      "name": "prices",
      "status": "ok",
      "message": "Price file OK: 100 rows, 5 symbols",
      "details": {
        "file": "output/aggregates/daily.parquet",
        "rows": 100,
        "symbols": 5
      }
    },
    {
      "name": "orders",
      "status": "ok",
      "message": "Orders file OK: 10 orders",
      "details": {
        "file": "output/orders_1d.csv",
        "rows": 10
      }
    },
    {
      "name": "portfolio",
      "status": "ok",
      "message": "Portfolio file OK: 50 rows",
      "details": {
        "file": "output/portfolio_equity_1d.csv",
        "rows": 50
      }
    }
  ],
  "timestamps": {
    "started": "2025-11-28T16:00:00Z",
    "finished": "2025-11-28T16:05:00Z"
  },
  "failure": false
}
```

**Felder:**
- `freq`: Trading-Frequenz
- `start_capital`: Verwendetes Startkapital
- `completed_steps`: Liste der erfolgreich abgeschlossenen Schritte
- `qa_overall_status`: QA-Overall-Status ("ok", "warning", "error")
- `qa_checks`: Liste der QA-Check-Ergebnisse
- `timestamps`: Start- und End-Zeitstempel (ISO 8601)
- `failure`: Boolean, ob Fehler aufgetreten sind

---

## Exit-Codes

- `0`: Erfolgreich (alle Schritte abgeschlossen, keine Fehler)
- `1`: Fehler aufgetreten (ein oder mehrere Schritte fehlgeschlagen)

---

## Integration mit FastAPI

Das Run-Manifest kann von der FastAPI-API gelesen werden, um den Status des letzten Pipeline-Laufs zu ermitteln:

```python
import json
from pathlib import Path

manifest_path = Path("output/run_manifest_1d.json")
if manifest_path.exists():
    with open(manifest_path) as f:
        manifest = json.load(f)
    print(f"Last run: {manifest['timestamps']['finished']}")
    print(f"QA status: {manifest['qa_overall_status']}")
```

---

## Orchestrator-Modul

Die Pipeline-Logik ist in `src/assembled_core/pipeline/orchestrator.py` implementiert:

**Funktionen:**
- `run_execute_step()`: Führt Execute-Schritt aus
- `run_backtest_step()`: Führt Backtest-Schritt aus
- `run_portfolio_step()`: Führt Portfolio-Schritt aus
- `run_eod_pipeline()`: Führt gesamte Pipeline aus

Diese Funktionen können auch programmatisch verwendet werden (z.B. in Tests oder anderen Skripten).

---

## Fehlerbehandlung

- **Preis-Daten fehlen:** Fehler wird geloggt, `failure_flag` gesetzt, Pipeline fährt fort
- **Execute-Fehler:** Fehler wird geloggt, `failure_flag` gesetzt, Pipeline fährt fort
- **Backtest-Fehler:** Fehler wird geloggt, `failure_flag` gesetzt, Portfolio wird trotzdem versucht
- **Portfolio-Fehler:** Fehler wird geloggt, `failure_flag` gesetzt, QA wird trotzdem versucht
- **QA-Fehler:** Fehler wird geloggt, `failure_flag` gesetzt, Manifest wird trotzdem geschrieben

**Prinzip:** Pipeline versucht so viele Schritte wie möglich auszuführen, auch wenn frühere Schritte fehlschlagen. Das Manifest dokumentiert, welche Schritte erfolgreich waren.

---

## Tests

Tests befinden sich in `tests/test_run_eod_pipeline.py`:

- `test_run_eod_pipeline_smoke()`: Vollständiger Pipeline-Lauf mit synthetischen Daten
- `test_run_eod_pipeline_skip_steps()`: Pipeline mit übersprungenen Schritten

**Ausführen:**
```bash
pytest tests/test_run_eod_pipeline.py
```

---

## Beispiel-Workflow

### 1. Pipeline ausführen

```bash
python scripts/run_eod_pipeline.py --freq 1d --start-capital 10000
```

### 2. Manifest prüfen

```bash
cat output/run_manifest_1d.json
```

### 3. API abfragen

```bash
# QA-Status
curl http://localhost:8000/api/v1/qa/status?freq=1d

# Portfolio-Zustand
curl http://localhost:8000/api/v1/portfolio/1d/current
```

---

## Nächste Schritte

- **Scheduling:** Cron-Job oder Task-Scheduler für automatische EOD-Läufe
- **Manifest-Historie:** Mehrere Manifeste pro Frequenz (mit Timestamp im Dateinamen)
- **Notifications:** E-Mail/Slack-Benachrichtigungen bei Fehlern
- **Retry-Logik:** Automatische Wiederholung fehlgeschlagener Schritte

---

## Weiterführende Dokumentation

- [Backend Architecture](ARCHITECTURE_BACKEND.md) - Gesamtarchitektur und Datenfluss
- [Backend Modules](BACKEND_MODULES.md) - Detaillierte Modulübersicht
- [Backend API](backend_api.md) - FastAPI-Endpoints
- [Backend Core](backend_core.md) - Konfiguration & Testing
- [Data Sources](DATA_SOURCES_BACKEND.md) - Datenquellen-Übersicht
- [Backend Roadmap](BACKEND_ROADMAP.md) - Entwicklungs-Roadmap

