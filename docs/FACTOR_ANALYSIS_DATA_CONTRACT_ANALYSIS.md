# Factor Analysis Data Contract - Analyse

**Datum:** 2025-12-09  
**Sprint:** C1 – Factor-IC/IR-Engine  
**Status:** Analyse abgeschlossen, Dokumentation aktualisiert

---

## Zusammenfassung

Diese Analyse dokumentiert die aktuellen Datenstrukturen und Data Contracts für die Factor-Analyse (Sprint C1). Die Dokumentation wurde in `docs/ADVANCED_ANALYTICS_FACTOR_LABS.md` unter Phase C1 aktualisiert.

---

## Datenstrukturen

### 1. Faktor-Berechnung (Phase A)

**Format:** Panel-DataFrame (kein MultiIndex)

**Struktur:**
- **Index:** Standard Integer-Index
- **Spalten:**
  - `timestamp`: UTC-Zeitstempel (datetime64[ns, UTC])
  - `symbol`: Symbol-Name (string)
  - `close`: Schlusskurs (float64)
  - Optional: `open`, `high`, `low`, `volume`, `freefloat`
  - **Faktor-Spalten:** Je nach Modul (siehe unten)

**Sortierung:** Nach `symbol`, dann `timestamp` (aufsteigend)

**Module:**

#### `ta_factors_core.build_core_ta_factors()`
- **Faktoren:**
  - `returns_1m`, `returns_3m`, `returns_6m`, `returns_12m` (Multi-Horizon Returns)
  - `momentum_12m_excl_1m` (12M-Momentum ohne letzten Monat)
  - `trend_strength_20`, `trend_strength_50`, `trend_strength_200` (Trend-Stärke)
  - `reversal_1d`, `reversal_2d`, `reversal_3d` (Short-Term Reversal)
- **Input:** Panel mit `timestamp`, `symbol`, `close` (+ optional: `high`, `low`, `volume`)
- **Output:** Gleiche Struktur + Faktor-Spalten

#### `ta_liquidity_vol_factors.add_realized_volatility()`
- **Faktoren:**
  - `rv_20`, `rv_60` (annualized realized volatility)
- **Input:** Panel mit `timestamp`, `symbol`, `close`
- **Output:** Gleiche Struktur + `rv_*` Spalten

#### `ta_liquidity_vol_factors.add_turnover_and_liquidity_proxies()`
- **Faktoren:**
  - `volume_zscore` (normalized volume)
  - `spread_proxy` ((high - low) / close)
  - `turnover` (volume / freefloat, nur wenn `freefloat_col` vorhanden)
- **Input:** Panel mit `timestamp`, `symbol`, `volume` (+ optional: `high`, `low`, `close`, `freefloat`)
- **Output:** Gleiche Struktur + Liquidity-Spalten

#### `market_breadth.compute_market_breadth_ma()`
- **Format:** Universe-Level DataFrame (eine Zeile pro Timestamp)
- **Spalten:** `timestamp`, `fraction_above_ma_{ma_window}`, `count_above_ma`, `count_total`
- **Hinweis:** Nicht direkt für IC-Berechnung verwendbar (universe-level, nicht symbol-level)

---

### 2. Forward-Returns/Labels

**Format:** Panel-DataFrame (kein MultiIndex)

**Erzeugung:** `add_forward_returns()` in `qa/factor_analysis.py`

**Struktur:**
- **Input:** Panel mit `timestamp`, `symbol`, `close` (oder `price_col`)
- **Output:** Gleiche Struktur + `fwd_return_{horizon_days}d` (z.B. `fwd_return_5d`)
- **Berechnung:**
  - Log-Returns: `ln(price[t+h] / price[t])`
  - Simple-Returns: `(price[t+h] / price[t]) - 1`
- **Look-Ahead-Bias:** Verhindert durch `shift(-horizon_days)`
- **NaN-Handling:** Letzte `horizon_days` Zeilen pro Symbol haben NaN

**Beispiel:**
```python
from src.assembled_core.qa.factor_analysis import add_forward_returns

# Forward-Returns für 5 Tage hinzufügen
df_with_returns = add_forward_returns(
    prices,
    horizon_days=5,
    col_name="fwd_return_5d",
    return_type="log"
)
```

---

### 3. IC-Berechnung (Cross-Sectional)

**Format:** DataFrame (eine Zeile pro (timestamp, factor))

**Erzeugung:** `compute_factor_ic()` oder `compute_rank_ic()` in `qa/factor_analysis.py`

**Input-Struktur:**
- Panel mit `timestamp`, `symbol`, `factor_*`, `fwd_return_*`
- Beispiel:
  ```
  timestamp              symbol  returns_12m  trend_strength_200  fwd_return_5d
  2020-01-01 00:00:00+00:00  AAPL      0.15             0.5           0.02
  2020-01-01 00:00:00+00:00  MSFT      0.12             0.3           0.01
  2020-01-02 00:00:00+00:00  AAPL      0.16             0.6           0.03
  ...
  ```

**Output-Struktur:**
- DataFrame mit Spalten: `timestamp`, `factor`, `ic`, `count`
- Beispiel:
  ```
  timestamp              factor              ic    count
  2020-01-01 00:00:00+00:00  returns_12m        0.15    50
  2020-01-01 00:00:00+00:00  trend_strength_200 0.08    50
  2020-01-02 00:00:00+00:00  returns_12m        0.18    50
  ...
  ```

**Berechnung:**
- Pro Timestamp: Cross-Sectional Korrelation zwischen Faktor-Werten und Forward-Returns
- Methoden: Pearson (linear) oder Spearman (rank)
- Minimum: 3 Symbole pro Timestamp für Korrelation

---

### 4. IC-Aggregation

**Format:** DataFrame (eine Zeile pro Faktor)

**Erzeugung:** `summarize_factor_ic()` in `qa/factor_analysis.py`

**Input:** IC-DataFrame von `compute_factor_ic()` oder `compute_rank_ic()`

**Output-Struktur:**
- DataFrame mit Spalten: `factor`, `mean_ic`, `std_ic`, `ic_ir`, `hit_ratio`, `count`, `min_ic`, `max_ic`
- Sortierung: Nach `ic_ir` (absteigend)
- Beispiel:
  ```
  factor              mean_ic  std_ic  ic_ir   hit_ratio  count
  returns_12m        0.12     0.08    1.50    0.65       1000
  trend_strength_200  0.08     0.06    1.33    0.60       1000
  rv_20               0.05     0.05    1.00    0.55       1000
  ...
  ```

---

### 5. Factor-Report-Workflow

**High-Level-Funktion:** `run_factor_report()` in `qa/factor_analysis.py`

**Workflow:**
1. Faktor-Berechnung (je nach `factor_set`: "core", "vol_liquidity", "all")
2. Forward-Returns hinzufügen
3. IC und Rank-IC berechnen
4. IC-Statistiken aggregieren

**Output-Struktur:**
```python
{
    "factors": DataFrame,        # Panel mit Faktoren + Forward-Returns
    "ic": DataFrame,             # IC pro (timestamp, factor)
    "rank_ic": DataFrame,         # Rank-IC pro (timestamp, factor)
    "summary_ic": DataFrame,      # Aggregierte IC-Statistiken
    "summary_rank_ic": DataFrame  # Aggregierte Rank-IC-Statistiken
}
```

**CLI-Integration:**
- `scripts/cli.py factor_report`
- Lädt Daten, ruft `run_factor_report()` auf, gibt Summary-Tabellen aus

---

## Wichtige Erkenntnisse

### 1. Kein MultiIndex
- Alle DataFrames verwenden **normale Spalten**, kein hierarchischer Index
- Sortierung erfolgt über `sort_values([symbol, timestamp])`

### 2. Panel-Format
- Faktoren und Forward-Returns sind im **Panel-Format** (mehrere Symbole pro Timestamp)
- IC-Berechnung erfolgt **cross-sectional** (pro Timestamp über alle Symbole)

### 3. UTC-Zeitzone
- Alle Timestamps müssen **UTC-aware** sein (`datetime64[ns, UTC]`)
- Wird automatisch durch `pd.to_datetime(..., utc=True)` sichergestellt

### 4. NaN-Handling
- Forward-Returns: Letzte `horizon_days` Zeilen pro Symbol haben NaN
- IC-Berechnung: Zeilen mit NaN in Faktor oder Forward-Return werden übersprungen
- Minimum: 3 Symbole pro Timestamp für Korrelation

### 5. Sortierung
- **Faktoren:** Nach `symbol`, dann `timestamp` (für korrekte Forward-Return-Berechnung)
- **IC-Output:** Nach `timestamp`, dann `factor`
- **Summary:** Nach `ic_ir` (absteigend)

---

## Integration mit bestehenden Modulen

### `qa/labeling.py`
- **Zweck:** Trade-Labeling für ML (nicht für IC-Berechnung)
- **Format:** Ähnlich (Panel mit `timestamp`, `symbol`), aber andere Logik
- **Unterschied:** Labeling verwendet P&L-basierte Thresholds, IC verwendet Korrelation

### `qa/dataset_builder.py`
- **Zweck:** ML-Dataset-Builder (Features + Labels für ML-Training)
- **Format:** Panel mit `timestamp`, `symbol`, Features, Labels
- **Unterschied:** Dataset-Builder fokussiert auf Trade-Labels, IC-Engine fokussiert auf Forward-Returns

---

## Nächste Schritte

### Implementiert (Sprint C1):
- ✅ Forward-Returns-Berechnung
- ✅ Cross-Sectional IC (Pearson)
- ✅ Rank-IC (Spearman)
- ✅ IC-Aggregation (mean, std, IC-IR, hit_ratio)
- ✅ High-Level Factor-Report-Workflow
- ✅ CLI-Integration

### Zukünftige Erweiterungen (C1 extension):
- ⏳ Rolling-IC (IC über rollierendes Fenster, z.B. 60 Tage)
- ⏳ IC-Decay-Analyse (wie lange bleiben Faktoren prädiktiv?)
- ⏳ IC-Verteilungs-Analyse (Skew, Kurtosis)
- ⏳ Visualisierungs-Tools (IC-Zeitreihen-Plots, IC-Verteilungs-Histogramme)

---

## Dokumentation

Die vollständige Dokumentation wurde in `docs/ADVANCED_ANALYTICS_FACTOR_LABS.md` unter Phase C1, Abschnitt "C1 – Factor-IC/IR-Engine: Zielbild & Data Contract" aktualisiert.

**Status:** ✅ Analyse abgeschlossen, Dokumentation aktualisiert

