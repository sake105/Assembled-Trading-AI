# Phase 7: Meta Layer - Labeling & ML Dataset Builder

## Übersicht

Phase 7 fügt einen Meta-Layer für Machine Learning hinzu:
- **Labeling**: Automatische Label-Generierung für Trades und Equity-Kurven
- **Dataset Builder**: Kombiniert Features und Labels zu ML-ready Datasets

## Komponenten

### 1. Labeling (`src/assembled_core/qa/labeling.py`)

#### `label_trades()`

Generiert Labels (0/1) für Trades basierend auf P&L-Performance:

```python
from src.assembled_core.qa.labeling import label_trades

labeled_trades = label_trades(
    trades=trades_df,
    prices=prices_df,  # Optional, wenn pnl_pct fehlt
    horizon_days=10,
    success_threshold=0.02  # 2% P&L = erfolgreich
)
```

**Features:**
- Unterstützt Trades mit vorhandener `pnl_pct`-Spalte
- Kann P&L aus Preisdaten rekonstruieren (wenn `pnl_pct` fehlt)
- Label = 1, wenn `pnl_pct >= success_threshold`, sonst 0

#### `label_daily_records()`

Generiert Labels für tägliche Equity-Kurven:

```python
from src.assembled_core.qa.labeling import label_daily_records

labeled_equity = label_daily_records(
    df=equity_df,
    horizon_days=10,
    success_threshold=0.02,
    price_col="equity"
)
```

**Features:**
- Prüft, ob Equity innerhalb `horizon_days` um mindestens `success_threshold` steigt
- Label = 1, wenn Bedingung erfüllt, sonst 0

### 2. Dataset Builder (`src/assembled_core/qa/dataset_builder.py`)

#### `build_ml_dataset_from_backtest()`

Kombiniert Features und Labels zu einem flachen ML-Dataset:

```python
from src.assembled_core.qa.dataset_builder import build_ml_dataset_from_backtest

ml_dataset = build_ml_dataset_from_backtest(
    prices_with_features=prices_df,
    trades=trades_df,
    label_horizon_days=10,
    success_threshold=0.02,
    feature_prefixes=("ta_", "insider_", "shipping_", "news_")
)
```

**Output-Schema:**
- `label`: 0/1 (erfolgreich/nicht erfolgreich)
- `symbol`, `open_time`, `open_price`: Trade-Metadaten
- `pnl_pct`, `horizon_days`: P&L-Informationen
- Feature-Spalten: Alle Spalten, die mit `feature_prefixes` beginnen (z.B. `ta_ma_20`, `insider_net_buy_20d`)

**Feature-Filterung:**
- Unterstützt TA-Features (via `"ta_"` Prefix oder explizite Patterns: `"ma_"`, `"rsi_"`, `"atr_"`, `"log_return"`)
- Unterstützt Event-Features (`"insider_"`, `"shipping_"`, `"congress_"`, `"news_"`)
- Nur numerische Features werden inkludiert

#### `save_ml_dataset()`

Speichert Dataset als Parquet:

```python
from src.assembled_core.qa.dataset_builder import save_ml_dataset

save_ml_dataset(ml_dataset, "output/ml_datasets/my_dataset.parquet")
```

## CLI-Integration

### `build_ml_dataset` Subcommand

```bash
# Standard-Dataset (Trend-Baseline)
python scripts/cli.py build_ml_dataset --strategy trend_baseline --freq 1d --price-file data/sample/eod_sample.parquet

# Event-Strategie mit angepassten Parametern
python scripts/cli.py build_ml_dataset --strategy event_insider_shipping --freq 1d --label-horizon-days 5 --success-threshold 0.03

# Mit explizitem Output-Pfad
python scripts/cli.py build_ml_dataset --strategy trend_baseline --freq 1d --out output/ml_datasets/my_dataset.parquet
```

**Argumente:**
- `--strategy`: `trend_baseline` oder `event_insider_shipping`
- `--freq`: `1d` oder `5min`
- `--price-file`: Optional, expliziter Pfad zu Preis-Datei
- `--universe`: Optional, Pfad zu Universe-Datei
- `--start-capital`: Startkapital (default: 10000.0)
- `--with-costs` / `--no-costs`: Transaktionskosten ein/aus
- `--label-horizon-days`: Horizon für Label-Generierung (default: 10)
- `--success-threshold`: P&L-Schwelle für erfolgreiche Trades (default: 0.02 = 2%)
- `--out`: Output-Pfad (default: `output/ml_datasets/<strategy>_<freq>.parquet`)

**Workflow:**
1. Lädt Preisdaten
2. Führt Backtest mit gewählter Strategie aus
3. Berechnet Features (TA + Event, je nach Strategie)
4. Generiert Trades
5. Labelt Trades basierend auf P&L
6. Joined Trades mit Features
7. Speichert flaches Dataset als Parquet

## Tests

### Phase-7-Tests

```bash
# Alle Phase-7-Tests
pytest -m phase7

# Spezifische Test-Dateien
pytest tests/test_qa_labeling.py -q
pytest tests/test_qa_dataset_builder.py -q
pytest tests/test_cli_ml_dataset.py -q
```

**Test-Dateien:**
- `tests/test_qa_labeling.py`: Labeling-Funktionen (11 Tests)
- `tests/test_qa_dataset_builder.py`: Dataset-Builder (9 Tests)
- `tests/test_cli_ml_dataset.py`: CLI-Integration (2 Tests)

**Erwartete Ausgabe:**
- ~22 Tests in < 5 Sekunden
- Alle Tests sollten grün sein ✅

## Verwendung in ML-Pipelines

### Beispiel: Scikit-Learn

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load ML dataset
df = pd.read_parquet("output/ml_datasets/trend_baseline_1d.parquet")

# Extract features and labels
feature_cols = [c for c in df.columns if c.startswith(("ta_", "insider_", "shipping_"))]
X = df[feature_cols]
y = df["label"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
score = model.score(X_test, y_test)
print(f"Accuracy: {score:.2%}")
```

## Best Practices

1. **Horizon-Auswahl**: 
   - Kürzere Horizon (5-10 Tage) für kurzfristige Strategien
   - Längere Horizon (20-30 Tage) für langfristige Strategien

2. **Success-Threshold**:
   - 2% (0.02) ist ein guter Startpunkt
   - Anpassen basierend auf Strategie-Risiko-Profil

3. **Feature-Auswahl**:
   - Verwende `feature_prefixes` um nur relevante Features zu inkludieren
   - Reduziert Dataset-Größe und verbessert Modell-Performance

4. **Datenqualität**:
   - Prüfe Label-Verteilung (sollte nicht zu unbalanciert sein)
   - Prüfe auf fehlende Werte in Features
   - Verwende Feature-Engineering für bessere Modell-Performance

## Weitere Informationen

- **Labeling-Modul**: `src/assembled_core/qa/labeling.py`
- **Dataset-Builder**: `src/assembled_core/qa/dataset_builder.py`
- **CLI-Integration**: `scripts/cli.py` (Subcommand `build_ml_dataset`)
- **Tests**: `tests/test_qa_labeling.py`, `tests/test_qa_dataset_builder.py`, `tests/test_cli_ml_dataset.py`

