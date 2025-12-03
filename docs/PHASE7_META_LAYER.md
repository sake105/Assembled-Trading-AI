# Phase 7: Meta Layer - Labeling & ML Dataset Builder

## Übersicht

Phase 7 fügt einen Meta-Layer für Machine Learning hinzu:
- **Labeling**: Automatische Label-Generierung für Trades basierend auf zukünftigen Preisbewegungen
- **Dataset Builder**: Kombiniert Features und Labels zu ML-ready Datasets
- **Strategy-based Builder**: High-Level-Funktion zum direkten Erstellen von ML-Datasets für Strategien

## Komponenten

### 1. Labeling (`src/assembled_core/qa/labeling.py`)

#### `generate_trade_labels()` (Sprint 7.1)

Generiert Labels für Trading-Signale basierend auf zukünftigen Preisbewegungen:

```python
from src.assembled_core.qa.labeling import generate_trade_labels

labeled_signals = generate_trade_labels(
    prices=prices_df,  # OHLCV oder mind. 'close' pro Ticker & Tag
    signals=signals_df,  # DataFrame mit Signalen (timestamp, symbol, direction, score)
    horizon_days=10,
    threshold_pct=0.05,  # 5% Preissteigerung = erfolgreich
    label_type="binary_absolute"
)
```

**Features:**
- **Binary Absolute**: Label = 1, wenn innerhalb `horizon_days` der maximale Close >= EntryClose * (1 + threshold_pct), sonst 0
- Unterstützt verschiedene Label-Typen (`binary_absolute`, `binary_outperformance`, `multi_class` - letztere sind geplant)
- Robust gegenüber fehlenden Daten (NaN, Lücken)
- Klare Fehler, wenn wichtige Spalten fehlen
- Deterministische Labels (keine Zufallskomponenten)

**Output-Schema:**
- `label`: 0/1 (erfolgreich/nicht erfolgreich)
- `realized_return`: Tatsächliche Rendite innerhalb des Horizonts
- `entry_price`: Preis zum Zeitpunkt des Signals
- `exit_price`: Preis am Ende des Horizonts (oder wenn Ziel erreicht)
- `horizon_days`: Verwendeter Horizont

#### Legacy-Funktionen (vor Sprint 7.1)

Die folgenden Funktionen existieren weiterhin für Backward-Compatibility:

- `label_trades()`: Generiert Labels für Trades basierend auf P&L-Performance
- `label_daily_records()`: Generiert Labels für tägliche Equity-Kurven

### 2. Dataset Builder (`src/assembled_core/qa/dataset_builder.py`)

#### `build_ml_dataset_for_strategy()` (Sprint 7.1 - Neu)

High-Level-Funktion zum direkten Erstellen von ML-Datasets für eine Strategie:

```python
from src.assembled_core.qa.dataset_builder import build_ml_dataset_for_strategy

ml_dataset = build_ml_dataset_for_strategy(
    strategy_name="trend_baseline",  # oder "event_insider_shipping"
    start_date="2024-01-01",
    end_date="2024-12-31",
    universe=["AAPL", "MSFT", "GOOGL"],  # Optional
    universe_file=None,  # Optional, Pfad zu Universe-Datei
    label_params={
        "horizon_days": 10,
        "threshold_pct": 0.05,
        "label_type": "binary_absolute"
    },
    price_file=None,  # Optional, expliziter Pfad zu Preis-Datei
    freq="1d"
)
```

**Workflow:**
1. Lädt Preis- und Feature-Daten (TA, ggf. Event-Daten wie Insider/Shipping)
2. Generiert Signale für die gegebene Strategie
3. Erzeugt Labels über `generate_trade_labels()`
4. Joint Features + Labels zu einem ML-Tabellenformat

**Output-Schema:**
- `label`: 0/1 (erfolgreich/nicht erfolgreich)
- `timestamp`, `symbol`: Signal-Metadaten
- `realized_return`, `entry_price`, `exit_price`: Label-Metadaten
- Feature-Spalten: Alle Feature-Spalten (TA, Insider, Shipping, etc.)

#### `export_ml_dataset()` (Sprint 7.1 - Neu)

Exportiert Dataset in verschiedenen Formaten:

```python
from src.assembled_core.qa.dataset_builder import export_ml_dataset

export_ml_dataset(
    df=ml_dataset,
    output_path="output/ml_datasets/my_dataset.parquet",
    format="parquet"  # oder "csv"
)
```

**Unterstützte Formate:**
- `parquet`: Standard, effizient für große Datasets
- `csv`: Für Kompatibilität mit anderen Tools

#### Legacy-Funktionen (vor Sprint 7.1)

Die folgenden Funktionen existieren weiterhin für Backward-Compatibility:

- `build_ml_dataset_from_backtest()`: Kombiniert Features und Labels aus Backtest-Ergebnissen
- `save_ml_dataset()`: Speichert Dataset als Parquet (Legacy-Alias für `export_ml_dataset`)

## CLI-Integration

### `build_ml_dataset` Subcommand

```bash
# Standard-Dataset (Trend-Baseline) - Neue High-Level-Methode
python scripts/cli.py build_ml_dataset --strategy trend_baseline --freq 1d --start-date 2024-01-01 --end-date 2024-12-31 --symbols AAPL MSFT GOOGL

# Event-Strategie mit angepassten Parametern
python scripts/cli.py build_ml_dataset --strategy event_insider_shipping --freq 1d --start-date 2024-01-01 --end-date 2024-12-31 --label-horizon-days 5 --success-threshold 0.03

# Mit explizitem Output-Pfad und Format
python scripts/cli.py build_ml_dataset --strategy trend_baseline --freq 1d --start-date 2024-01-01 --end-date 2024-12-31 --out output/ml_datasets/my_dataset.csv --format csv

# Legacy-Modus (Backtest-basiert, für Backward-Compatibility)
python scripts/cli.py build_ml_dataset --strategy trend_baseline --freq 1d --price-file data/sample/eod_sample.parquet
```

**Argumente (Sprint 7.1 - Neu):**
- `--strategy`: `trend_baseline` oder `event_insider_shipping`
- `--freq`: `1d` oder `5min`
- `--start-date`: Startdatum im Format "YYYY-MM-DD" (für neue High-Level-Methode)
- `--end-date`: Enddatum im Format "YYYY-MM-DD" (für neue High-Level-Methode)
- `--symbols`: Liste von Symbolen (z.B. `AAPL MSFT GOOGL`)
- `--universe`: Optional, Pfad zu Universe-Datei (Text-Datei, ein Symbol pro Zeile)
- `--label-horizon-days`: Horizon für Label-Generierung (default: 10)
- `--success-threshold`: Preissteigerungs-Schwelle für erfolgreiche Signale (default: 0.02 = 2%)
- `--label-type`: Label-Typ (`binary_absolute`, `binary_outperformance`, `multi_class` - default: `binary_absolute`)
- `--out`: Output-Pfad (default: `output/ml_datasets/<strategy>_<freq>.parquet`)
- `--format`: Export-Format (`parquet` oder `csv`, default: `parquet`)

**Legacy-Argumente (für Backtest-basierte Methode):**
- `--price-file`: Optional, expliziter Pfad zu Preis-Datei
- `--start-capital`: Startkapital (default: 10000.0)
- `--with-costs` / `--no-costs`: Transaktionskosten ein/aus

**Workflow (Neue High-Level-Methode):**
1. Lädt Preis- und Feature-Daten (TA, ggf. Event-Daten)
2. Generiert Signale für die gegebene Strategie
3. Erzeugt Labels basierend auf zukünftigen Preisbewegungen
4. Joint Features + Labels zu einem ML-Tabellenformat
5. Exportiert Dataset als Parquet oder CSV

**Workflow (Legacy Backtest-basierte Methode):**
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
- `tests/test_ml_dataset_builder.py`: ML-Dataset-Builder und Labeling (Sprint 7.1)
  - Tests für `generate_trade_labels()`: Happy Path, Edge Cases
  - Tests für `build_ml_dataset_for_strategy()`: Mini-Setup, Feature/Label-Prüfung
  - Tests für `export_ml_dataset()`: Parquet/CSV-Export
- `tests/test_qa_labeling.py`: Legacy Labeling-Funktionen (11 Tests)
- `tests/test_qa_dataset_builder.py`: Legacy Dataset-Builder (9 Tests)
- `tests/test_cli_ml_dataset.py`: CLI-Integration (2 Tests)

**Erwartete Ausgabe:**
- ~40+ Tests in < 5 Sekunden
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

## Sprint 7.1 - Status

✅ **Abgeschlossen:**
- `generate_trade_labels()`: Neue Labeling-Funktion für Signale basierend auf zukünftigen Preisbewegungen
- `build_ml_dataset_for_strategy()`: High-Level-Funktion zum direkten Erstellen von ML-Datasets
- `export_ml_dataset()`: Flexible Export-Funktion (Parquet/CSV)
- CLI-Integration: Erweiterte Argumente für Strategy-basierte Dataset-Erstellung
- Umfassende Test-Suite: `tests/test_ml_dataset_builder.py`

## Sprint 7.2 - Meta-Modelle

✅ **Abgeschlossen:**
- `MetaModel` Klasse: Wrapper für trainiertes Modell mit Feature-Namen
- `train_meta_model()`: Trainiert Meta-Modell auf ML-Dataset (GradientBoostingClassifier oder RandomForestClassifier)
- `save_meta_model()` / `load_meta_model()`: Speichern/Laden von trainierten Modellen (joblib)
- `evaluate_meta_model()`: Evaluationsmetriken (ROC-AUC, Brier Score, Log Loss)
- `plot_calibration_curve()`: Calibration-Plot für Modell-Validierung
- CLI-Integration: `train_meta_model` Subcommand für Training und Evaluation
- Umfassende Test-Suite: `tests/test_meta_model.py`

**Dependencies:**
- `scikit-learn>=1.3.0` (optional, install with: `pip install scikit-learn`)
- `joblib>=1.3.0` (optional, install with: `pip install joblib`)
- Install both: `pip install -e .[ml]` (from pyproject.toml optional-dependencies)

**Workflow-Beispiel:**
```bash
# 1. ML-Dataset bauen
python scripts/cli.py build_ml_dataset --strategy trend_baseline --freq 1d --start-date 2024-01-01 --end-date 2024-12-31

# 2. Meta-Modell trainieren
python scripts/cli.py train_meta_model --dataset-path output/ml_datasets/trend_baseline_1d.parquet

# Oder: Dataset bauen und trainieren in einem Schritt
python scripts/cli.py train_meta_model --strategy trend_baseline --freq 1d --start-date 2024-01-01 --end-date 2024-12-31
```

**Verwendung in Python:**
```python
from src.assembled_core.signals.meta_model import train_meta_model, load_meta_model, save_meta_model
from src.assembled_core.qa.ml_evaluation import evaluate_meta_model
import pandas as pd

# Load dataset
df = pd.read_parquet("output/ml_datasets/trend_baseline_1d.parquet")

# Train model
model = train_meta_model(df, model_type="gradient_boosting")

# Predict confidence scores
X = df[model.feature_names]
confidence_scores = model.predict_proba(X)

# Evaluate
y_true = df["label"]
metrics = evaluate_meta_model(y_true, confidence_scores)
print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
print(f"Brier Score: {metrics['brier_score']:.4f}")

# Save model
save_meta_model(model, "models/meta/trend_baseline_meta.joblib")

# Load model later
loaded_model = load_meta_model("models/meta/trend_baseline_meta.joblib")
```

## Sprint 7.3 - Ensemble-Layer

✅ **Abgeschlossen:**
- `apply_meta_filter()`: Filtert Signale mit niedriger Meta-Model-Confidence (setzt auf "FLAT")
- `apply_meta_scaling()`: Skaliert Signal-Scores/Positionen basierend auf Meta-Model-Confidence
- Integration in `run_portfolio_backtest()`: Optionaler Meta-Ensemble-Schritt zwischen Signal-Generierung und Position-Sizing
- CLI-Integration: `--use-meta-model`, `--meta-model-path`, `--meta-min-confidence`, `--meta-ensemble-mode`
- Umfassende Test-Suite: `tests/test_signals_ensemble.py` (funktioniert ohne sklearn-Dependency)

**Workflow-Beispiel:**
```bash
# 1. ML-Dataset bauen
python scripts/cli.py build_ml_dataset --strategy trend_baseline --freq 1d --start-date 2024-01-01 --end-date 2024-12-31

# 2. Meta-Modell trainieren
python scripts/cli.py train_meta_model --dataset-path output/ml_datasets/trend_baseline_1d.parquet

# 3. Backtest mit Meta-Model-Ensemble
python scripts/cli.py run_backtest --freq 1d --strategy trend_baseline --use-meta-model --meta-model-path models/meta/trend_baseline_meta_model.joblib --meta-min-confidence 0.5

# 4. Vergleich: Backtest ohne Meta-Model
python scripts/cli.py run_backtest --freq 1d --strategy trend_baseline
```

**Ensemble-Modi:**
- **Filter-Modus** (`--meta-ensemble-mode filter`): Entfernt Signale mit Confidence < `min_confidence` (setzt `direction` auf "FLAT")
- **Scaling-Modus** (`--meta-ensemble-mode scaling`): Skaliert Signal-Scores und Positionen proportional zur Confidence

**Verwendung in Python:**
```python
from src.assembled_core.signals.ensemble import apply_meta_filter, apply_meta_scaling
from src.assembled_core.signals.meta_model import load_meta_model
import pandas as pd

# Load meta-model
meta_model = load_meta_model("models/meta/trend_baseline_meta_model.joblib")

# Load signals and features
signals = pd.DataFrame({
    "timestamp": [...],
    "symbol": [...],
    "direction": ["LONG", "FLAT", ...],
    "score": [0.8, 0.2, ...]
})
features = prices_with_features[meta_model.feature_names]

# Apply meta-filter (remove signals with confidence < 0.5)
filtered_signals = apply_meta_filter(
    signals=signals,
    meta_model=meta_model,
    features=features,
    min_confidence=0.5
)

# Or apply meta-scaling (scale positions by confidence)
scaled_signals = apply_meta_scaling(
    signals=signals,
    meta_model=meta_model,
    features=features,
    min_confidence=0.3,
    max_scaling=1.0
)
```

**Nächste Schritte (geplant):**
- `binary_outperformance` Labeling (vs. Benchmark)
- `multi_class` Labeling (stark buy, buy, hold, sell, strong sell)
- Feature-Engineering-Pipeline (Normalisierung, Skalierung)

## Weitere Informationen

- **Labeling-Modul**: `src/assembled_core/qa/labeling.py`
- **Dataset-Builder**: `src/assembled_core/qa/dataset_builder.py`
- **CLI-Integration**: `scripts/cli.py` (Subcommands `build_ml_dataset`, `train_meta_model`)
- **Ensemble-Layer**: `src/assembled_core/signals/ensemble.py`
- **Backtest-Integration**: `src/assembled_core/qa/backtest_engine.py` (Meta-Model-Parameter)
- **Tests**: 
  - `tests/test_ml_dataset_builder.py` (Sprint 7.1)
  - `tests/test_meta_model.py` (Sprint 7.2)
  - `tests/test_signals_ensemble.py` (Sprint 7.3)
  - `tests/test_qa_labeling.py`, `tests/test_qa_dataset_builder.py`, `tests/test_cli_ml_dataset.py`

