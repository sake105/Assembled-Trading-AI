# Architecture Layering - Assembled Trading AI

**Zweck:** Verbindliche Regelbasis für Modul-Imports und Architektur-Layering.

**Status:** Verbindlich (muss bei Code-Änderungen befolgt werden)

**Letzte Aktualisierung:** 2025-01-04

---

## Zielbild: Layer-Hierarchie

```
┌─────────────────────────────────────────────────────────┐
│                    pipeline/                            │
│              (Orchestrator - Top Layer)                 │
│  Importiert: data, features, signals, portfolio, exec   │
└─────────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
┌───────▼──────┐  ┌───────▼──────┐  ┌───────▼──────┐
│  execution/  │  │  portfolio/   │  │   signals/   │
│  (Orders)    │  │  (Sizing)     │  │  (Signals)   │
│              │  │               │  │              │
│  Importiert:│  │  Importiert: │  │  Importiert: │
│  portfolio,  │  │  signals      │  │  features,   │
│  signals     │  │               │  │  data        │
└──────────────┘  └───────────────┘  └───────┬──────┘
                                             │
                                    ┌────────▼────────┐
                                    │   features/     │
                                    │  (TA Features)  │
                                    │                 │
                                    │  Importiert:    │
                                    │  data           │
                                    └────────┬────────┘
                                             │
                                    ┌────────▼────────┐
                                    │     data/       │
                                    │  (Data Ingest)  │
                                    │                 │
                                    │  Importiert:    │
                                    │  (nichts)       │
                                    └─────────────────┘

┌─────────────────────────────────────────────────────────┐
│                    qa/ (Sidecar)                        │
│  Importiert: pipeline (nur Outputs), config, utils      │
│  KEINE Zyklen: qa importiert nicht in Pipeline-Layer    │
└─────────────────────────────────────────────────────────┘
```

### Layer-Definitionen

1. **`data/`** (Bottom Layer)
   - **Zweck:** Daten-Ingest (Preise, Events, Alt-Data)
   - **Importiert:** Nichts (nur externe Libraries: pandas, numpy, pathlib)
   - **Exportiert:** DataFrames mit standardisierten Schemas

2. **`features/`** (Layer 2)
   - **Zweck:** Feature-Engineering (TA-Indikatoren, Event-Features)
   - **Importiert:** `data/` (nur)
   - **Exportiert:** DataFrames mit Features

3. **`signals/`** (Layer 3)
   - **Zweck:** Signal-Generierung (Trend, Event, Multi-Factor)
   - **Importiert:** `features/`, `data/` (nur)
   - **Exportiert:** DataFrames mit Signalen

4. **`portfolio/`** (Layer 4)
   - **Zweck:** Position-Sizing (Target-Positionen berechnen)
   - **Importiert:** `signals/` (nur)
   - **Exportiert:** DataFrames mit Target-Positionen

5. **`execution/`** (Layer 5)
   - **Zweck:** Order-Generierung, Risk-Controls, Execution
   - **Importiert:** `portfolio/`, `signals/` (nur)
   - **Exportiert:** DataFrames mit Orders

6. **`pipeline/`** (Top Layer - Orchestrator)
   - **Zweck:** Orchestrierung aller Layer (Trading Cycle)
   - **Importiert:** Alle Layer (`data/`, `features/`, `signals/`, `portfolio/`, `execution/`)
   - **Exportiert:** TradingCycleResult (Orders, Equity, Reports)

7. **`qa/`** (Sidecar - Keine Zyklen)
   - **Zweck:** Backtesting, Metriken, QA-Gates, Reports
   - **Importiert:** `pipeline/` (nur Outputs/Results), `config/`, `utils/`
   - **NICHT erlaubt:** `qa/` importiert nicht in Pipeline-Layer (keine Zyklen)

---

## Erlaubte Imports (konkret)

### `data/` Module

**Erlaubt:**
```python
# Externe Libraries
import pandas as pd
import numpy as np
from pathlib import Path

# Config/Utils (shared)
from src.assembled_core.config import OUTPUT_DIR
from src.assembled_core.config.settings import get_settings
```

**Verboten:**
```python
# ❌ KEINE Imports von features, signals, portfolio, execution, pipeline
from src.assembled_core.features import ...  # VERBOTEN
from src.assembled_core.signals import ...   # VERBOTEN
from src.assembled_core.pipeline import ...   # VERBOTEN
```

**Beispiele:**
- ✅ `src/assembled_core/data/prices_ingest.py` importiert nur pandas, pathlib, config
- ✅ `src/assembled_core/data/insider_ingest.py` importiert nur pandas, pathlib

---

### `features/` Module

**Erlaubt:**
```python
# Externe Libraries
import pandas as pd
import numpy as np

# data/ Layer (Bottom)
from src.assembled_core.data.prices_ingest import load_eod_prices
from src.assembled_core.data.insider_ingest import load_insider_sample

# Config/Utils (shared)
from src.assembled_core.config import OUTPUT_DIR
```

**Verboten:**
```python
# ❌ KEINE Imports von signals, portfolio, execution, pipeline
from src.assembled_core.signals import ...   # VERBOTEN
from src.assembled_core.portfolio import ... # VERBOTEN
from src.assembled_core.pipeline import ...  # VERBOTEN
```

**Beispiele:**
- ✅ `src/assembled_core/features/ta_features.py` importiert nur pandas, numpy, data
- ✅ `src/assembled_core/features/insider_features.py` importiert nur data/insider_ingest

---

### `signals/` Module

**Erlaubt:**
```python
# Externe Libraries
import pandas as pd
import numpy as np

# Lower Layers
from src.assembled_core.features.ta_features import add_all_features
from src.assembled_core.data.prices_ingest import load_eod_prices

# Config/Utils (shared)
from src.assembled_core.config import OUTPUT_DIR
```

**Verboten:**
```python
# ❌ KEINE Imports von portfolio, execution, pipeline
from src.assembled_core.portfolio import ... # VERBOTEN
from src.assembled_core.execution import ... # VERBOTEN
from src.assembled_core.pipeline import ...  # VERBOTEN
```

**Beispiele:**
- ✅ `src/assembled_core/signals/rules_trend.py` importiert nur features, data
- ✅ `src/assembled_core/signals/rules_event_insider_shipping.py` importiert nur features, data

---

### `portfolio/` Module

**Erlaubt:**
```python
# Externe Libraries
import pandas as pd
import numpy as np

# Lower Layers
from src.assembled_core.signals.rules_trend import generate_trend_signals_from_prices

# Config/Utils (shared)
from src.assembled_core.config import OUTPUT_DIR
```

**Verboten:**
```python
# ❌ KEINE Imports von execution, pipeline
from src.assembled_core.execution import ... # VERBOTEN
from src.assembled_core.pipeline import ...  # VERBOTEN
```

**Beispiele:**
- ✅ `src/assembled_core/portfolio/position_sizing.py` importiert nur signals

---

### `execution/` Module

**Erlaubt:**
```python
# Externe Libraries
import pandas as pd
import numpy as np

# Lower Layers
from src.assembled_core.portfolio.position_sizing import compute_target_positions
from src.assembled_core.signals.rules_trend import generate_trend_signals_from_prices

# Config/Utils (shared)
from src.assembled_core.config import OUTPUT_DIR
```

**Verboten:**
```python
# ❌ KEINE Imports von pipeline
from src.assembled_core.pipeline import ...  # VERBOTEN
```

**Beispiele:**
- ✅ `src/assembled_core/execution/order_generation.py` importiert nur portfolio, signals
- ✅ `src/assembled_core/execution/risk_controls.py` importiert nur portfolio, signals

---

### `pipeline/` Module (Orchestrator)

**Erlaubt:**
```python
# Externe Libraries
import pandas as pd
import numpy as np

# ALLE Lower Layers (Orchestrator darf alles importieren)
from src.assembled_core.data.prices_ingest import load_eod_prices
from src.assembled_core.features.ta_features import add_all_features
from src.assembled_core.signals.rules_trend import generate_trend_signals_from_prices
from src.assembled_core.portfolio.position_sizing import compute_target_positions
from src.assembled_core.execution.order_generation import generate_orders_from_targets

# Config/Utils (shared)
from src.assembled_core.config import OUTPUT_DIR
```

**Verboten:**
```python
# ❌ KEINE zirkulären Imports (pipeline importiert nicht in Lower Layers)
# (Dies ist automatisch erfüllt, da Lower Layers pipeline nicht importieren)
```

**Beispiele:**
- ✅ `src/assembled_core/pipeline/trading_cycle.py` importiert alle Layer
- ✅ `src/assembled_core/pipeline/orchestrator.py` importiert alle Layer

---

### `qa/` Module (Sidecar)

**Erlaubt:**
```python
# Externe Libraries
import pandas as pd
import numpy as np

# pipeline/ Outputs (nur Results, keine internen Funktionen)
from src.assembled_core.pipeline.trading_cycle import TradingContext, TradingCycleResult
# ✅ Erlaubt: Import von Dataclasses/Results aus pipeline

# Config/Utils (shared)
from src.assembled_core.config import OUTPUT_DIR
from src.assembled_core.utils import ...
```

**Verboten:**
```python
# ❌ KEINE Imports von pipeline internen Funktionen (die Lower Layers importieren)
from src.assembled_core.pipeline.orders import signals_to_orders  # VERBOTEN
from src.assembled_core.pipeline.signals import ...               # VERBOTEN

# ❌ KEINE Imports in Pipeline-Layer (keine Zyklen)
# pipeline/ darf qa/ nicht importieren (außer für Reports)
```

**Beispiele:**
- ✅ `src/assembled_core/qa/backtest_engine.py` importiert nur pipeline Results
- ✅ `src/assembled_core/qa/metrics.py` importiert nur pandas, numpy
- ⚠️ `src/assembled_core/qa/backtest_engine.py` importiert `pipeline.trading_cycle` (OK, nur Results)

---

## Anti-Patterns (Verboten)

### 1. Zirkuläre Imports zwischen Layern

**❌ VERBOTEN:**
```python
# features/ta_features.py
from src.assembled_core.signals.rules_trend import generate_trend_signals  # VERBOTEN

# signals/rules_trend.py
from src.assembled_core.features.ta_features import add_all_features  # OK
```

**Problem:** `features/` importiert `signals/`, aber `signals/` importiert `features/` → Zirkulärer Import

**Fix:** `features/` darf nur `data/` importieren, nicht `signals/`

---

### 2. Pipeline importiert in Lower Layers

**❌ VERBOTEN:**
```python
# signals/rules_trend.py
from src.assembled_core.pipeline.trading_cycle import TradingContext  # VERBOTEN
```

**Problem:** Lower Layer (`signals/`) importiert Top Layer (`pipeline/`) → Verletzt Layering

**Fix:** `signals/` darf nur `features/` und `data/` importieren

---

### 3. QA importiert Pipeline-Interna

**❌ VERBOTEN:**
```python
# qa/backtest_engine.py
from src.assembled_core.pipeline.orders import signals_to_orders  # VERBOTEN
from src.assembled_core.pipeline.signals import generate_signals   # VERBOTEN
```

**Problem:** `qa/` importiert Pipeline-Interna, die wiederum Lower Layers importieren → Indirekter Zyklus

**Fix:** `qa/` darf nur `pipeline/` Results/Dataclasses importieren, nicht interne Funktionen

---

### 4. Lower Layer importiert Higher Layer

**❌ VERBOTEN:**
```python
# data/prices_ingest.py
from src.assembled_core.features.ta_features import add_all_features  # VERBOTEN

# features/ta_features.py
from src.assembled_core.signals.rules_trend import generate_trend_signals  # VERBOTEN
```

**Problem:** Verletzt Layer-Hierarchie (Bottom → Top)

**Fix:** Imports nur in Richtung Bottom → Top

---

### 5. QA importiert in Pipeline-Layer (Zyklen)

**❌ VERBOTEN:**
```python
# pipeline/trading_cycle.py
from src.assembled_core.qa.backtest_engine import run_portfolio_backtest  # VERBOTEN (wenn qa pipeline importiert)
```

**Problem:** Wenn `qa/` `pipeline/` importiert und `pipeline/` `qa/` importiert → Zirkulärer Import

**Fix:** `qa/` ist Sidecar - `pipeline/` darf `qa/` nur für Reports importieren, nicht für Core-Logik

---

## Shared Modules (Ausnahme)

**Erlaubt für alle Layer:**
- `config/` - Zentrale Konfiguration (OUTPUT_DIR, SUPPORTED_FREQS, Settings)
- `utils/` - Utility-Funktionen (keine Business-Logik)
- `costs.py` - Cost-Model (shared, keine Layer-Abhängigkeiten)
- `logging_utils.py` - Logging (shared)

**Beispiele:**
```python
# Erlaubt in allen Layern
from src.assembled_core.config import OUTPUT_DIR
from src.assembled_core.config.settings import get_settings
from src.assembled_core.costs import get_default_cost_model
from src.assembled_core.logging_utils import get_logger
```

---

## How to Check Imports (Lokal)

### Methode 1: Python Import-Check

```python
# scripts/check_imports.py
import ast
from pathlib import Path
from collections import defaultdict

ROOT = Path("src/assembled_core")

LAYERS = {
    "data": 1,
    "features": 2,
    "signals": 3,
    "portfolio": 4,
    "execution": 5,
    "pipeline": 6,
    "qa": 99,  # Sidecar
}

def check_imports(file_path: Path) -> list[tuple[str, str, int, int]]:
    """Check imports in a file and return violations."""
    violations = []
    with open(file_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=str(file_path))
    
    current_layer = None
    for part in file_path.parts:
        if part in LAYERS:
            current_layer = part
            break
    
    if current_layer is None:
        return violations
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("src.assembled_core."):
                imported_layer = None
                for part in node.module.split("."):
                    if part in LAYERS:
                        imported_layer = part
                        break
                
                if imported_layer and LAYERS[imported_layer] > LAYERS[current_layer]:
                    violations.append((
                        str(file_path),
                        node.module,
                        LAYERS[current_layer],
                        LAYERS[imported_layer]
                    ))
    
    return violations

# Run check
for py_file in ROOT.rglob("*.py"):
    violations = check_imports(py_file)
    if violations:
        for file, module, current, imported in violations:
            print(f"VIOLATION: {file} (layer {current}) imports {module} (layer {imported})")
```

### Methode 2: Ruff Check (Static Analysis)

```bash
# Install ruff if not available
pip install ruff

# Check for circular imports
ruff check --select F401 src/assembled_core/
```

### Methode 3: Manual Import-Graph

```python
# scripts/generate_import_graph.py
import ast
from pathlib import Path
from collections import defaultdict

def extract_imports(file_path: Path) -> set[str]:
    """Extract all imports from a file."""
    imports = set()
    with open(file_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=str(file_path))
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("src.assembled_core."):
                imports.add(node.module)
    
    return imports

# Build import graph
graph = defaultdict(set)
for py_file in Path("src/assembled_core").rglob("*.py"):
    imports = extract_imports(py_file)
    for imp in imports:
        graph[str(py_file.relative_to(Path("src/assembled_core")))].add(imp)

# Print graph (check for cycles manually)
for file, imports in graph.items():
    print(f"{file}:")
    for imp in sorted(imports):
        print(f"  -> {imp}")
```

### Methode 4: Runtime Import-Check (Tests)

```python
# tests/test_imports_layering.py
"""Test that imports follow layering rules."""

import pytest
from pathlib import Path
import ast

LAYERS = {
    "data": 1,
    "features": 2,
    "signals": 3,
    "portfolio": 4,
    "execution": 5,
    "pipeline": 6,
}

def test_no_circular_imports():
    """Test that no circular imports exist."""
    violations = []
    for py_file in Path("src/assembled_core").rglob("*.py"):
        if py_file.name == "__init__.py":
            continue
        
        current_layer = None
        for part in py_file.parts:
            if part in LAYERS:
                current_layer = part
                break
        
        if current_layer is None:
            continue
        
        with open(py_file, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=str(py_file))
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith("src.assembled_core."):
                    imported_layer = None
                    for part in node.module.split("."):
                        if part in LAYERS:
                            imported_layer = part
                            break
                    
                    if imported_layer and LAYERS[imported_layer] > LAYERS[current_layer]:
                        violations.append((str(py_file), node.module))
    
    assert not violations, f"Found {len(violations)} layering violations: {violations}"
```

---

## Zusammenfassung

### Layer-Hierarchie (Bottom → Top)

1. **`data/`** → Importiert: Nichts (nur externe Libraries)
2. **`features/`** → Importiert: `data/`
3. **`signals/`** → Importiert: `features/`, `data/`
4. **`portfolio/`** → Importiert: `signals/`
5. **`execution/`** → Importiert: `portfolio/`, `signals/`
6. **`pipeline/`** → Importiert: Alle Layer (Orchestrator)
7. **`qa/`** → Importiert: `pipeline/` (nur Results), `config/`, `utils/` (Sidecar)

### Shared Modules (Alle Layer)

- `config/` - Zentrale Konfiguration
- `utils/` - Utility-Funktionen
- `costs.py` - Cost-Model
- `logging_utils.py` - Logging

### Verboten

- ❌ Zirkuläre Imports zwischen Layern
- ❌ Lower Layer importiert Higher Layer
- ❌ `qa/` importiert Pipeline-Interna (nur Results)
- ❌ `pipeline/` importiert `qa/` für Core-Logik (nur Reports)

---

**Hinweis:** Diese Regeln sind verbindlich. Bei Code-Änderungen müssen Import-Violations behoben werden.
