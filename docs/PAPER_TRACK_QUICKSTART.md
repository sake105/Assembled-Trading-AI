# Paper-Track Quickstart

**Kurzanleitung für den Paper-Track Runner - 5 Minuten zum ersten Lauf.**

---

## Was ist Paper-Track?

Paper-Track simuliert Trading-Strategien mit echten Marktdaten, ohne echtes Kapital zu riskieren. Es ist die Brücke zwischen Backtesting und Live-Trading.

---

## Voraussetzungen

- Python 3.11+
- Installierte Dependencies (`pip install -e .[dev]`)
- Preis-Daten für deine Universe (z.B. in `data/raw/1min/` oder `data/raw/eod/`)

---

## Schnellstart (3 Kommandos)

### 1. Verfügbare Strategien auflisten

```bash
# List all available paper track configs/strategies
python scripts/cli.py paper_track --list
```

### 2. Strategie ausführen (mit Auto-Discovery)

```bash
# Run with strategy name (config auto-discovered)
python scripts/cli.py paper_track \
  --strategy-name trend_baseline \
  --as-of 2025-01-15
```

### Alternative: Eigene Config erstellen

```bash
# Kopiere eine Standard-Template-Config
cp configs/paper_track/trend_baseline.yaml configs/paper_track/my_strategy.yaml

# Edit the config (universe file, strategy params, etc.)
# Then run with explicit config file:
python scripts/cli.py paper_track \
  --config-file configs/paper_track/my_strategy.yaml \
  --as-of 2025-01-15
```

### Verfügbare Standard-Configs

- **`trend_baseline.yaml`**: Trend-following strategy template (moving averages)
- **`multifactor_long_short.yaml`**: Multi-factor long/short strategy template
- **`trend_baseline_example.yaml`**: Example config for trend baseline (ready to use)
- **`multifactor_long_short_example.yaml`**: Example config for multi-factor (ready to use)

### 3. Output prüfen

```bash
# Output-Verzeichnis
ls output/paper_track/my_strategy/runs/20250115/

# Equity-Kurve ansehen
cat output/paper_track/my_strategy/aggregates/equity_curve.csv
```

---

## Häufige Workflows

### Daily Catch-up (automatisch fehlende Tage nachholen)

```bash
python scripts/cli.py paper_track \
  --config-file configs/paper_track/my_strategy.yaml \
  --catch-up
```

### Date Range (mehrere Tage)

```bash
python scripts/cli.py paper_track \
  --config-file configs/paper_track/my_strategy.yaml \
  --start-date 2025-01-01 \
  --end-date 2025-01-31
```

### Dry-Run (ohne Dateien zu schreiben)

```bash
python scripts/cli.py paper_track \
  --config-file configs/paper_track/my_strategy.yaml \
  --as-of 2025-01-15 \
  --dry-run
```

### Re-run eines existierenden Tages

```bash
python scripts/cli.py paper_track \
  --config-file configs/paper_track/my_strategy.yaml \
  --as-of 2025-01-15 \
  --rerun
```

---

## Output-Struktur

Nach einem Run findest du:

```
output/paper_track/my_strategy/
├── state/
│   └── state.json              # Portfolio-State (Positions, Cash, Equity)
├── runs/
│   └── 20250115/               # Tages-spezifische Outputs
│       ├── manifest.json       # Run-Metadaten (Config-Hash, Git-Commit, etc.)
│       ├── equity_snapshot.json
│       ├── daily_summary.json
│       ├── daily_summary.md
│       ├── positions.csv
│       └── orders_today.csv
└── aggregates/                 # Aggregierte Zeitreihen
    ├── equity_curve.csv        # Equity-Kurve über alle Tage
    ├── trades_all.csv          # Alle Trades
    └── positions_history.csv   # Positions-Snapshots
```

---

## Config-Beispiel (Minimal)

```yaml
strategy_name: my_strategy
strategy_type: trend_baseline
universe:
  file: watchlist.txt
trading:
  freq: 1d
portfolio:
  seed_capital: 100000.0
costs:
  commission_bps: 0.5
  spread_w: 0.25
  impact_w: 0.5
strategy:
  params:
    ma_fast: 20
    ma_slow: 50
    top_n: 5
```

---

## Troubleshooting

### "No price data available for date"

- Prüfe: Existieren Preis-Daten für diesen Tag?
- Pfade: `data/raw/eod/` (für `freq: 1d`) oder `data/raw/1min/` (für `freq: 5min`)

### "Universe file not found"

- Prüfe: Pfad in Config relativ zum Config-File oder zum Repo-Root?
- Beispiel: `universe: file: watchlist.txt` sucht zuerst in `configs/paper_track/`

### "Run directory already exists"

- Default: Skipped (safe)
- Mit `--rerun`: Wird neu ausgeführt (alte Verzeichnisse werden gesichert)

### Health Checks

```bash
# Prüfe Paper-Track Health
python scripts/cli.py check_health \
  --paper-track-root output/paper_track/ \
  --paper-track-days 3
```

---

## Weiterführende Dokumentation

- **Playbook (Detailliert):** [PAPER_TRACK_PLAYBOOK.md](PAPER_TRACK_PLAYBOOK.md)
- **Design-Dokument:** [PAPER_TRACK_RUNNER_A5_DESIGN.md](PAPER_TRACK_RUNNER_A5_DESIGN.md)
- **Operations:** [OPERATIONS_BACKEND.md](OPERATIONS_BACKEND.md)

---

## Nächste Schritte

1. ✅ Erster Run erfolgreich
2. ⏭️ Daily Catch-up einrichten (Cron/Task Scheduler)
3. 📊 Risk-Reports generieren (weekly/monthly)
4. 🔍 Health Checks automatisieren

---

*Zuletzt aktualisiert: 2025-01*

