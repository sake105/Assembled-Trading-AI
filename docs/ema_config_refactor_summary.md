# EMA Config Refactor - Zusammenfassung

## Übersicht

Implementierung einer zentralen EMA-Konfiguration mit frequency-basierten Defaults für die Sprint-9-Strategie.

**Aktuelle Defaults (beide Frequenzen):**
- `fast = 20`
- `slow = 60`

**Alternative für Experimente (5min):**
- `fast = 10`, `slow = 30` (als Konstante definiert, kann einfach aktiviert werden)

---

## Neue Dateien

### `src/assembled_core/ema_config.py` (NEU)
```python
# src/assembled_core/ema_config.py
"""EMA (Exponential Moving Average) configuration for trading strategies."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EmaConfig:
    """EMA crossover strategy parameters.
    
    Attributes:
        fast: Fast EMA period (shorter-term trend)
        slow: Slow EMA period (longer-term trend)
    """
    fast: int
    slow: int

    def __post_init__(self) -> None:
        """Validate EMA configuration."""
        if self.fast <= 0:
            raise ValueError(f"fast must be > 0, got {self.fast}")
        if self.slow <= 0:
            raise ValueError(f"slow must be > 0, got {self.slow}")
        if self.fast >= self.slow:
            raise ValueError(f"fast ({self.fast}) must be < slow ({self.slow})")


# Default EMA configurations per frequency
# Based on current working pipeline values
DEFAULT_EMA_1D = EmaConfig(fast=20, slow=60)
DEFAULT_EMA_5MIN = EmaConfig(fast=20, slow=60)

# Alternative configurations for experimentation (5min)
# These can be easily switched by changing DEFAULT_EMA_5MIN above
ALTERNATIVE_EMA_5MIN_FAST = EmaConfig(fast=10, slow=30)


def get_default_ema_config(freq: str) -> EmaConfig:
    """Return the default EMA configuration for a given frequency.
    
    Args:
        freq: Frequency string ("1d" or "5min")
    
    Returns:
        EmaConfig instance with default parameters for the frequency.
    
    Raises:
        ValueError: If freq is not supported.
    """
    if freq == "1d":
        return DEFAULT_EMA_1D
    elif freq == "5min":
        return DEFAULT_EMA_5MIN
    else:
        raise ValueError(f"Unsupported frequency: {freq}. Use '1d' or '5min'.")
```

---

## Geänderte Dateien

### `scripts/sprint9_execute.py`

**Änderungen:**

1. **Import hinzugefügt:**
```python
import sys
from pathlib import Path

OUT_DIR = Path("output")

# Import EMA configuration
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.assembled_core.ema_config import get_default_ema_config, EmaConfig
```

2. **parse_args() Funktion angepasst:**
```python
def parse_args() -> ExecArgs:
    p = argparse.ArgumentParser(description="EMA-Crossover Execution (orders csv)")
    p.add_argument("--freq", choices=["1d", "5min"], required=True, help="Zeitebene")
    
    # Parse freq first to get frequency-based defaults
    args_partial, remaining = p.parse_known_args()
    default_ema = get_default_ema_config(args_partial.freq)
    
    # Add remaining arguments with frequency-based defaults
    p.add_argument("--ema-fast", type=int, default=default_ema.fast,
                   help=f"Fast EMA period (default: {default_ema.fast} for {args_partial.freq})")
    p.add_argument("--ema-slow", type=int, default=default_ema.slow,
                   help=f"Slow EMA period (default: {default_ema.slow} for {args_partial.freq})")
    p.add_argument("--price-file", type=str, default=None,
                   help="Optional eigener Pfad zu Preisen (Parquet mit timestamp,symbol,close)")
    p.add_argument("--out", type=str, default=str(OUT_DIR), help="Output-Ordner (default: output)")
    
    # Parse all arguments
    args = p.parse_args()
    
    return ExecArgs(
        freq=args.freq,
        ema_fast=int(args.ema_fast),
        ema_slow=int(args.ema_slow),
        price_file=args.price_file,
        out_dir=Path(args.out),
    )
```

**Vorher:**
```python
p.add_argument("--ema-fast", type=int, default=20)
p.add_argument("--ema-slow", type=int, default=60)
```

**Nachher:**
```python
# Parse freq first
args_partial, remaining = p.parse_known_args()
default_ema = get_default_ema_config(args_partial.freq)

# Then add EMA args with frequency-based defaults
p.add_argument("--ema-fast", type=int, default=default_ema.fast,
               help=f"Fast EMA period (default: {default_ema.fast} for {args_partial.freq})")
p.add_argument("--ema-slow", type=int, default=default_ema.slow,
               help=f"Slow EMA period (default: {default_ema.slow} for {args_partial.freq})")
```

---

## Wo wird die EMA-Konfiguration definiert und verwendet?

### 1. Definition
**Datei:** `src/assembled_core/ema_config.py`
- Zeile 30-31: `DEFAULT_EMA_1D` und `DEFAULT_EMA_5MIN` mit Werten (20, 60)
- Zeile 34: `ALTERNATIVE_EMA_5MIN_FAST` mit Werten (10, 30) für Experimente
- Zeile 37-50: `get_default_ema_config(freq: str)` gibt die Default-Instanz zurück

### 2. Verwendung in der Pipeline
**Datei:** `scripts/sprint9_execute.py`
- Zeile 15: Import von `get_default_ema_config`
- Zeile 161: Aufruf `default_ema = get_default_ema_config(args_partial.freq)` nach Parsing von `--freq`
- Zeilen 164-167: Verwendung als CLI-Defaults für `--ema-fast` und `--ema-slow`
- Zeile 175: Werte werden an `ExecArgs` weitergegeben
- Zeile 145: Werte werden an `make_orders()` weitergegeben

### 3. Verhalten
- **Ohne CLI-Argumente:** Verwendet Defaults aus `get_default_ema_config(freq)` (20, 60 für beide Frequenzen)
- **Mit CLI-Argumenten:** CLI-Argumente überschreiben die Defaults
  - Beispiel: `--ema-fast 10 --ema-slow 30` überschreibt die Defaults

---

## CLI-Kompatibilität

### ✅ Bestehende Befehle funktionieren weiterhin

**Beispiel 1: Ohne EMA-Argumente (verwendet Defaults)**
```bash
python scripts/sprint9_execute.py --freq 5min
# Verwendet: fast=20, slow=60 (aus DEFAULT_EMA_5MIN)
```

**Beispiel 2: Mit expliziten EMA-Argumenten (überschreibt Defaults)**
```bash
python scripts/sprint9_execute.py --freq 5min --ema-fast 10 --ema-slow 30
# Verwendet: fast=10, slow=30 (CLI-Override)
```

**Beispiel 3: Nur ein Parameter überschreiben**
```bash
python scripts/sprint9_execute.py --freq 5min --ema-fast 15
# Verwendet: fast=15, slow=60 (nur fast überschrieben)
```

**Beispiel 4: Tägliche Frequenz**
```bash
python scripts/sprint9_execute.py --freq 1d
# Verwendet: fast=20, slow=60 (aus DEFAULT_EMA_1D)
```

---

## Experimentieren mit alternativen Werten

Um schnell alternative EMA-Werte für 5min zu testen, kann man in `src/assembled_core/ema_config.py` ändern:

```python
# Vorher:
DEFAULT_EMA_5MIN = EmaConfig(fast=20, slow=60)

# Nachher (für mehr Trades):
DEFAULT_EMA_5MIN = ALTERNATIVE_EMA_5MIN_FAST  # fast=10, slow=30
```

Oder direkt in der Funktion:
```python
def get_default_ema_config(freq: str) -> EmaConfig:
    if freq == "1d":
        return DEFAULT_EMA_1D
    elif freq == "5min":
        return ALTERNATIVE_EMA_5MIN_FAST  # Experiment: mehr Trades
    else:
        raise ValueError(f"Unsupported frequency: {freq}. Use '1d' or '5min'.")
```

---

## Rückwärtskompatibilität

### ✅ Verhalten bleibt gleich
- Die bisherigen Defaults (20, 60) sind identisch mit den neuen Defaults
- CLI-Argumente funktionieren weiterhin wie vorher
- Alle bestehenden Skripte und PowerShell-Wrapper funktionieren ohne Änderungen
- Der Befehl `python scripts/sprint9_execute.py --freq 5min --ema-fast 10 --ema-slow 30` funktioniert weiterhin

---

## Zusammenfassung

✅ **Zentrale Konfiguration:** `EmaConfig`-Dataclass in `src/assembled_core/ema_config.py`  
✅ **Frequency-basierte Defaults:** `get_default_ema_config(freq)` für "1d" und "5min"  
✅ **Integration:** `sprint9_execute.py` nutzt die zentrale Konfiguration  
✅ **CLI-Override:** Funktioniert weiterhin (z.B. `--ema-fast 10 --ema-slow 30`)  
✅ **Rückwärtskompatibel:** Keine Verhaltensänderung  
✅ **Experimentierfreundlich:** Alternative Konfigurationen leicht aktivierbar

Die Implementierung ist abgeschlossen und bereit für die Verwendung.

