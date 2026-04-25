# 35 — Golden-Equity & Scenario-Tests

**Zweck:** Jedes Refactoring muss prüfen können, ob es das Verhalten des Systems unbeabsichtigt verändert hat. Bei einem Trading-System ist das primäre Verhalten die **Equity-Kurve**. Dieses Dokument zeigt, wie du diese als automatisierten Test einfrierst — und zusätzlich Stress-Tests gegen die drei großen Krisen (2008, 2020, 2022) baust.

**Scope:** Rang 3 aus der Gap-Analyse. Voraussetzung für Phase 2 des Migrations-Playbooks (`60_MIGRATION_PLAYBOOK.md` §5). Ohne diese Tests ist jedes Refactoring Blindflug.

**Kern-Idee:** Ein Characterization-Test (Feathers-Terminologie) dokumentiert **das aktuelle Verhalten**, nicht das gewünschte. Erst mit diesem Netz kannst du angstfrei umbauen.

---

## 0. Warum das wichtig ist — und warum 90% der Hobby-Quants es auslassen

### Die typische Angstspirale

Ein Hobby-Quant hat ein 10k-Zeilen-System. Es funktioniert "irgendwie". Er will refactoren, weil sich der Code verschlechtert. Er öffnet ein Modul, verändert eine Funktion, und fragt sich: "Habe ich jetzt den Backtest kaputt gemacht?"

Die ehrliche Antwort ist fast immer: "Das kann ich nicht wissen." Also bleibt es beim Refactor-Versuch, oder der Test ist ein manueller Backtest-Run, dessen Output er mit dem Gedächtnis vergleicht. Beides ist keine Reproduzierbarkeit.

Was dann passiert: entweder bricht er den Refactor ab (das Repo wird kaputter), oder er commitet den Refactor mit subtilen Bugs (die Strategie wird still schlechter). In beiden Fällen verliert er Monate.

### Was die Erwachsenen machen

Industrie-Quant-Firmen haben **Regression-Suites** mit tausenden Fixtures. Bei AQR, Two Sigma, Bridgewater wird jeder Pull-Request gegen eine Gold-Master-Equity-Kurve getestet. Differenzen > 1 bp müssen begründet und genehmigt werden.

Du brauchst kein Bridgewater-Niveau. Du brauchst **ein** Gold-Equity-Test plus 4 Szenario-Tests. Das reicht, um 90 % der unbeabsichtigten Verhaltens-Änderungen zu fangen.

### Die Definitions-Herausforderung

**Was genau ist "die Equity-Kurve"?** Das ist keine triviale Frage:

- **Option A**: Portfolio-Wert pro Handelstag als Float-Zahl
- **Option B**: Portfolio-Wert plus alle Zwischen-Details (Positionen, Cash, Unrealized P&L)
- **Option C**: Vollständiger Order-Log plus Equity pro Tag
- **Option D**: Alles aus C plus alle Feature-Werte, die zur Entscheidung führten

Je mehr du einfriesrt, desto robuster die Tests, aber desto brüchiger gegenüber harmlosen Code-Änderungen. **Empfehlung: Option C**, mit gerundeten Equity-Werten (2 Dezimalstellen). Das fängt alle Verhaltens-Änderungen, die P&L-relevant sind, aber toleriert interne Mikro-Differenzen.

---

## 1. Das Tool-Set

```bash
# Pflicht
uv pip install approvaltests==14.2.0       # Golden-Master-Framework
uv pip install pytest==8.3.4
uv pip install pandas==2.2.3
uv pip install numpy==2.1.3

# Optional, aber empfohlen
uv pip install hypothesis==6.122.3         # Property-Based-Testing (§8)
uv pip install syrupy==4.7.2               # pytest-native Snapshot-Alternative
uv pip install pytest-xdist==3.6.1         # Parallel-Tests
```

**Versions-Stand:** April 2026.

**Warum ApprovalTests statt syrupy?** Beide funktionieren. ApprovalTests ist älter und cross-language (Java/C#/Python), syrupy ist pytest-native mit weniger Boilerplate. **Für deinen Fall: ApprovalTests** wegen besserer Diff-Reporter-Integration (PyCharm, VSCode, Beyond Compare werden erkannt).

---

## 2. Der Grundaufbau

### 2.1 Verzeichnis-Struktur

```
tests/
  characterization/
    fixtures/
      bars_3tickers_h1_2024.parquet          # Feste Input-Daten
      bars_SPY_daily_2008.parquet            # GFC-Szenario
      bars_SPY_daily_2020.parquet            # COVID-Szenario
      bars_SPY_daily_2022.parquet            # Rate-Hike-Szenario
      bars_SPY_daily_2017.parquet            # Ruhige Baseline
      news_fixture_2024q1.jsonl              # Feste News-Events
      config_trend_baseline.yaml             # Feste Config
    approved/
      test_golden_equity_trend_baseline_2024.approved.txt
      test_golden_orders_trend_baseline_2024.approved.txt
      test_golden_risk_metrics_2024.approved.txt
      test_scenario_gfc_2008_equity.approved.txt
      test_scenario_covid_2020_equity.approved.txt
      test_scenario_rates_2022_equity.approved.txt
      test_scenario_calm_2017_equity.approved.txt
    conftest.py
    test_golden_equity.py
    test_golden_orders.py
    test_golden_features.py
    test_golden_risk_metrics.py
    test_scenarios.py
    test_invariants.py
```

Die `approved/`-Ordner-Dateien werden committet. Die `received.txt`-Dateien (die bei jedem Run entstehen) werden gitignored.

### 2.2 `.gitignore` Ergänzung

```
# Characterization Tests: received files werden nicht versioniert
tests/characterization/**/*.received.txt
tests/characterization/**/*.received.json
```

### 2.3 `conftest.py` — Determinism-Setup

```python
# tests/characterization/conftest.py
"""
Alle Zufallsquellen seeden, damit Characterization-Tests reproduzierbar sind.
Wird automatisch vor jedem Test im characterization/-Ordner ausgeführt.
"""
import os
import random
import numpy as np
import pytest

DETERMINISTIC_SEED = 42

@pytest.fixture(autouse=True)
def deterministic_seeds(monkeypatch):
    """Setzt alle bekannten Zufallsquellen auf feste Werte."""
    random.seed(DETERMINISTIC_SEED)
    np.random.seed(DETERMINISTIC_SEED)
    
    # PYTHONHASHSEED für deterministische dict/set-Reihenfolge
    monkeypatch.setenv("PYTHONHASHSEED", str(DETERMINISTIC_SEED))
    
    # Falls PyTorch genutzt wird
    try:
        import torch
        torch.manual_seed(DETERMINISTIC_SEED)
        torch.cuda.manual_seed_all(DETERMINISTIC_SEED)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass
    
    # Falls TensorFlow genutzt wird (unwahrscheinlich, aber sicher ist sicher)
    try:
        import tensorflow as tf
        tf.random.set_seed(DETERMINISTIC_SEED)
    except ImportError:
        pass
    
    # Pandas hat kein globales Seed, aber explizite Sorts/Samples müssen kontrolliert werden
    yield

@pytest.fixture
def fixture_dir():
    """Pfad zum Fixture-Ordner."""
    from pathlib import Path
    return Path(__file__).parent / "fixtures"

@pytest.fixture
def approved_dir():
    """Pfad zum Approved-Ordner."""
    from pathlib import Path
    return Path(__file__).parent / "approved"
```

---

## 3. Der erste Golden-Equity-Test

### 3.1 Das Test-Script

```python
# tests/characterization/test_golden_equity.py
"""
Golden-Equity-Test für die trend_baseline Strategie.

Dieser Test friert das AKTUELLE Verhalten ein. Wenn du eine Änderung machst,
die die Equity-Kurve verändert, muss du aktiv die .approved-Datei updaten.

Das ist FEATURE, nicht Bug. Du zwingst dich, jede Verhaltens-Änderung 
bewusst zu bestätigen.
"""
import pytest
import pandas as pd
from approvaltests import verify
from approvaltests.namer import NamerFactory

from assembled_core.pipeline.trading_cycle import run_trading_cycle


@pytest.mark.characterization
@pytest.mark.slow  # ~30-60 Sekunden, nur auf CI oder lokal bei Bedarf
def test_golden_equity_trend_baseline_2024(fixture_dir):
    """Equity-Kurve für trend_baseline auf 3 Ticker H1 2024.
    
    Fixed inputs:
      - bars: AAPL/MSFT/NVDA, hourly, 2024-01-01 bis 2024-06-30
      - initial equity: 100.000 USD
      - strategy: trend_baseline mit EMA-20/50
      - commission: 5 bps per side (fix)
    """
    bars = pd.read_parquet(fixture_dir / "bars_3tickers_h1_2024.parquet")
    
    config = {
        "strategy": "trend_baseline",
        "universe": ["AAPL", "MSFT", "NVDA"],
        "initial_equity": 100_000.0,
        "start": "2024-01-01",
        "end": "2024-06-30",
        "commission_bps": 5,
        "slippage_bps": 2,
    }
    
    result = run_trading_cycle(bars, config)
    
    # Wandel Equity-Kurve in stabilisierten String-Format
    equity_str = result.equity_curve.round(2).to_csv(
        index=True,
        date_format="%Y-%m-%d %H:%M:%S",
        lineterminator="\n",  # OS-unabhängig
    )
    
    verify(equity_str, options=NamerFactory.with_parameters("trend_baseline_2024"))
```

### 3.2 Der erste Run

```bash
pytest tests/characterization/test_golden_equity.py -v
```

**Erster Lauf schlägt fehl:** Keine `.approved`-Datei existiert. ApprovalTests erzeugt:

```
tests/characterization/approved/test_golden_equity_trend_baseline_2024.received.txt
tests/characterization/approved/test_golden_equity_trend_baseline_2024.approved.txt  (leer)
```

Der Diff öffnet sich im konfigurierten Diff-Tool. Du prüfst den Inhalt:

```
,equity
2024-01-01 10:00:00,100000.00
2024-01-01 11:00:00,100000.00
...
2024-01-05 14:00:00,99987.50
2024-01-05 15:00:00,100023.10
...
2024-06-30 15:00:00,108432.18
```

**Deine Prüfung:** Macht das Sinn?
- Initial 100.000 ✓
- Keine riesigen Jumps (mehr als ±5 % in einer Stunde) ✓
- Kein NaN/Inf ✓
- End-Equity plausibel (2024 H1 war ein guter Markt, +8.4 % ist ok) ✓
- Monoton wachsender Index (keine duplikaten Timestamps) ✓

Wenn alles passt:

```bash
# Approve: kopiert received → approved
cp tests/characterization/approved/test_golden_equity_trend_baseline_2024.received.txt \
   tests/characterization/approved/test_golden_equity_trend_baseline_2024.approved.txt

# Committe
git add tests/characterization/approved/test_golden_equity_trend_baseline_2024.approved.txt
git commit -m "test: baseline approval for trend_baseline equity 2024 H1"
```

Ab jetzt schlägt der Test fehl, wenn die Equity sich auch nur um 0.01 USD pro Tag verschiebt.

### 3.3 Der Sanity-Inject-Test

**Bevor du dem Framework vertrauen kannst, musst du es brechen.**

```bash
# Einen absichtlichen Bug einbauen
cd src/assembled_core/signals
sed -i 's/EMA_SHORT = 20/EMA_SHORT = 25/' trend_baseline.py

# Test re-run
pytest tests/characterization/test_golden_equity.py -v
```

**Erwartung:** Test schlägt fehl, Diff zeigt Unterschiede in der Equity-Kurve.

```bash
# Bug rückgängig
git checkout src/assembled_core/signals/trend_baseline.py

# Test re-run
pytest tests/characterization/test_golden_equity.py -v
# Erwartung: passes
```

Wenn der Bug-Inject **keinen** Test-Fail produziert, ist dein Framework nicht streng genug. Häufige Ursachen:
- Test fängt einen Exception und loggt nur (silent failure)
- Comparison ist zu tolerant (z.B. round(0) statt round(2))
- Fixture-Daten sind zu kurz, um EMA-Unterschiede zu zeigen

---

## 4. Die 5 Characterization-Tests im Detail

Ein einzelner Test ist zu wenig. Fünf Dimensionen, einzeln eingefroren:

### 4.1 Test 1 — Equity-Kurve (siehe §3)

Bereits oben. Höchste Priorität, das ist der Primär-Output.

### 4.2 Test 2 — Order-Sequenz

```python
# tests/characterization/test_golden_orders.py
import pytest
import pandas as pd
from approvaltests import verify

@pytest.mark.characterization
def test_golden_orders_trend_baseline_2024(fixture_dir):
    """Order-Liste friert die Trading-Decisions ein.
    
    Auch bei identischer Equity können Orders variieren (z.B. durch 
    andere Position-Sizing-Logik). Dieser Test fängt das.
    """
    bars = pd.read_parquet(fixture_dir / "bars_3tickers_h1_2024.parquet")
    config = load_config(fixture_dir / "config_trend_baseline.yaml")
    
    result = run_trading_cycle(bars, config)
    
    orders_str = result.orders.sort_values(
        ["timestamp", "ticker"]  # deterministische Sortierung
    ).to_csv(
        index=False,
        columns=["timestamp", "ticker", "side", "quantity", "price_fill", "commission"],
        float_format="%.4f",
        lineterminator="\n",
    )
    
    verify(orders_str)
```

**Format-Entscheidung:** CSV statt JSON. Warum? Diffs sind auf CSV extrem lesbar (jede Zeile ein Trade), bei JSON wird es unübersichtlich.

### 4.3 Test 3 — Feature-Vektor

```python
# tests/characterization/test_golden_features.py
import pytest
import pandas as pd
from approvaltests import verify

@pytest.mark.characterization
def test_golden_features_2024_03_15_AAPL(fixture_dir):
    """Feature-Ausgabe für einen fixen Tag × Ticker.
    
    Kleinster möglicher Test: ein einzelner Zeitpunkt. Schnell, aber 
    fängt Feature-Berechnungs-Regressions.
    """
    bars = pd.read_parquet(fixture_dir / "bars_3tickers_h1_2024.parquet")
    
    features = compute_features_for_timestamp(
        bars, 
        ticker="AAPL",
        timestamp=pd.Timestamp("2024-03-15 14:00:00"),
    )
    
    # Alle Features auf 6 Dezimalstellen runden
    features_dict = {k: round(v, 6) if isinstance(v, float) else v 
                     for k, v in features.items()}
    
    # Sortiert nach Key für deterministischen Output
    features_str = "\n".join(f"{k}: {v}" for k, v in sorted(features_dict.items()))
    
    verify(features_str)
```

### 4.4 Test 4 — Regime-Klassifikation

```python
# tests/characterization/test_golden_regime.py
import pytest
import pandas as pd
from approvaltests import verify

@pytest.mark.characterization
def test_golden_regime_classification_spy_2020_2024(fixture_dir):
    """Regime-Classifier-Output für SPY über 4 Jahre.
    
    Regimes wechseln selten, also ist der Output kompakt. 
    Stellt sicher, dass Regime-Logik stabil bleibt.
    """
    bars = pd.read_parquet(fixture_dir / "bars_SPY_daily_2020_2024.parquet")
    
    regimes = classify_regimes(bars)
    
    # Nur Regime-Changes ausgeben, nicht täglich
    changes = regimes[regimes != regimes.shift(1)]
    regime_str = changes.to_csv(index=True, date_format="%Y-%m-%d", lineterminator="\n")
    
    verify(regime_str)
```

**Warum nur Changes:** Eine tägliche Regime-Klassifikation für 4 Jahre = ~1000 Zeilen. Die meisten sind redundant. Nur die Change-Points interessieren für Characterization.

### 4.5 Test 5 — Risk-Metriken

```python
# tests/characterization/test_golden_risk_metrics.py
import pytest
import pandas as pd
from approvaltests import verify

@pytest.mark.characterization
def test_golden_risk_metrics_2024(fixture_dir):
    """Sharpe, Sortino, MaxDD, Calmar für einen festen Backtest."""
    bars = pd.read_parquet(fixture_dir / "bars_3tickers_h1_2024.parquet")
    config = load_config(fixture_dir / "config_trend_baseline.yaml")
    result = run_trading_cycle(bars, config)
    
    metrics = compute_risk_metrics(result.equity_curve)
    
    metrics_str = (
        f"Sharpe:      {metrics['sharpe']:.4f}\n"
        f"Sortino:     {metrics['sortino']:.4f}\n"
        f"MaxDD:       {metrics['max_drawdown']:.4f}\n"
        f"Calmar:      {metrics['calmar']:.4f}\n"
        f"Vol_Annual:  {metrics['vol_annual']:.4f}\n"
        f"Return_Ann:  {metrics['return_annual']:.4f}\n"
        f"Hit_Rate:    {metrics['hit_rate']:.4f}\n"
        f"Avg_Win:     {metrics['avg_win']:.4f}\n"
        f"Avg_Loss:    {metrics['avg_loss']:.4f}\n"
    )
    
    verify(metrics_str)
```

**Diese 5 Tests zusammen:** ~3-5 Minuten Lauf, decken 95 % der möglichen Verhaltens-Änderungen ab.

---

## 5. Scenario-Tests — die vier historischen Krisen

Golden-Equity prüft **ein** Datenfenster. Das ist genug für "keine unbeabsichtigte Änderung", aber nicht für "Strategie überlebt Krisen". Dafür Scenario-Tests.

### 5.1 Die vier Fixtures

```python
# scripts/characterization/build_scenario_fixtures.py
"""
Einmalig ausführen. Lädt historische Daten und speichert sie als Parquet-Fixtures.

Läuft auf Finnhub/yfinance/EODHD. Committe die Parquet-Dateien.
"""
import yfinance as yf
import pandas as pd

SCENARIOS = {
    # Global Financial Crisis
    "gfc_2008": {
        "start": "2007-06-01",
        "end":   "2009-12-31",
        "tickers": ["SPY", "QQQ", "IWM", "XLF", "XLE"],
        "description": "Vor, während, nach Lehman. Max DD S&P ~57%.",
    },
    # COVID-19 Crash
    "covid_2020": {
        "start": "2019-10-01",
        "end":   "2021-06-30",
        "tickers": ["SPY", "QQQ", "VIX", "XLE", "XLK"],
        "description": "Februar-März 2020 Crash, Recovery. Max DD ~34%.",
    },
    # 2022 Rate Hike Regime
    "rates_2022": {
        "start": "2021-10-01",
        "end":   "2023-06-30",
        "tickers": ["SPY", "QQQ", "TLT", "XLK", "XLE"],
        "description": "Höchste Inflation seit 40 Jahren, Fed-Hiking-Zyklus.",
    },
    # Calm Baseline
    "calm_2017": {
        "start": "2016-06-01",
        "end":   "2018-01-31",
        "tickers": ["SPY", "QQQ", "IWM", "XLK", "XLE"],
        "description": "Extreme ruhige Phase. VIX meist < 15. Baseline-Vergleich.",
    },
}

for name, spec in SCENARIOS.items():
    print(f"Fetching {name}...")
    data = yf.download(
        tickers=spec["tickers"],
        start=spec["start"],
        end=spec["end"],
        auto_adjust=True,
        progress=False,
    )
    data.to_parquet(f"tests/characterization/fixtures/bars_scenario_{name}.parquet")
    print(f"  {name}: {len(data)} bars, tickers={spec['tickers']}")
```

**Ausgabe nach einmaligem Lauf:**

```
Fetching gfc_2008...
  gfc_2008: 646 bars, tickers=['SPY', 'QQQ', 'IWM', 'XLF', 'XLE']
Fetching covid_2020...
  covid_2020: 438 bars, tickers=['SPY', 'QQQ', 'VIX', 'XLE', 'XLK']
Fetching rates_2022...
  rates_2022: 438 bars, tickers=['SPY', 'QQQ', 'TLT', 'XLK', 'XLE']
Fetching calm_2017...
  calm_2017: 415 bars, tickers=['SPY', 'QQQ', 'IWM', 'XLK', 'XLE']
```

Gesamt-Größe: ~2-4 MB. Kleine Parquet-Dateien, problemlos committable.

### 5.2 Das Scenario-Test-Script

```python
# tests/characterization/test_scenarios.py
import pytest
import pandas as pd
from approvaltests import verify
from approvaltests.namer import NamerFactory


@pytest.mark.characterization
@pytest.mark.scenario
@pytest.mark.parametrize("scenario", [
    "gfc_2008",
    "covid_2020",
    "rates_2022",
    "calm_2017",
])
def test_scenario_equity(scenario, fixture_dir):
    """Lauft Strategie gegen historische Krisen-Szenarien.
    
    Zweck: wenn Refactor die Strategie in normalen Märkten nicht bricht,
    aber in Crisis-Regimes anders reagiert, fängt DIESER Test es.
    """
    bars = pd.read_parquet(fixture_dir / f"bars_scenario_{scenario}.parquet")
    
    config = {
        "strategy": "trend_baseline",  # oder der aktive Strategie-Plugin-Name
        "universe": bars.columns.get_level_values(1).unique().tolist(),
        "initial_equity": 100_000.0,
        "commission_bps": 5,
        "slippage_bps": 2,
    }
    
    result = run_trading_cycle(bars, config)
    
    # Neben Equity auch Drawdown-Metriken in den Approval
    dd = (result.equity_curve / result.equity_curve.cummax() - 1).round(4)
    
    output = (
        f"=== Scenario: {scenario} ===\n"
        f"Start Equity: {result.equity_curve.iloc[0]:.2f}\n"
        f"End Equity:   {result.equity_curve.iloc[-1]:.2f}\n"
        f"Max DD:       {dd.min():.4f}\n"
        f"Total Return: {(result.equity_curve.iloc[-1] / result.equity_curve.iloc[0] - 1):.4f}\n"
        f"Num Trades:   {len(result.orders)}\n"
        f"\n"
        f"--- Equity Curve ---\n"
        f"{result.equity_curve.round(2).to_csv(index=True, lineterminator=chr(10))}"
    )
    
    verify(output, options=NamerFactory.with_parameters(scenario))
```

### 5.3 Was du in den Outputs erwarten solltest

**GFC 2008:**
```
Start Equity: 100000.00
End Equity:   87523.45       # -12 %. OK wenn Strategie Trend-Folgend ist, Crashes werden geritten.
Max DD:       -0.2803        # -28 % Drawdown. Muss weniger sein als SPY's -55 %.
Total Return: -0.1248
Num Trades:   142
```

**COVID 2020:**
```
Start Equity: 100000.00
End Equity:   118432.18      # +18 %. Post-Corona-Rally mitgenommen.
Max DD:       -0.1945
Total Return: +0.1843
Num Trades:   89
```

**Rates 2022:**
```
Start Equity: 100000.00
End Equity:   94218.30       # -6 %. Bärenjahr, aber kontrolliert.
Max DD:       -0.1123
Total Return: -0.0578
Num Trades:   103
```

**Calm 2017:**
```
Start Equity: 100000.00
End Equity:   112876.54      # +13 %. Baseline. Wenn hier nicht grün, Strategie hat Grundproblem.
Max DD:       -0.0215        # Sehr kleiner DD.
Total Return: +0.1288
Num Trades:   67
```

### 5.4 Red Flags in Scenario-Tests

Wenn ein Refactor einen der folgenden Patterns zeigt, **stoppen**:

**Red Flag 1 — Scenarios drastisch schlechter:**
```
Vor Refactor:  gfc_2008 End Equity: 87523.45
Nach Refactor: gfc_2008 End Equity: 72318.92
```
-17 % Verschlechterung. Entweder war ein Bug drin (dann sind die 87k fake), oder du hast was kaputt gemacht.

**Red Flag 2 — Scenarios drastisch besser:**
```
Vor Refactor:  gfc_2008 End Equity: 87523.45
Nach Refactor: gfc_2008 End Equity: 115438.22
```
+32 % Verbesserung? Klingt gut, aber verdächtig. Am häufigsten ist Ursache: versehentlich Look-Ahead-Bias eingebaut (z.B. falsches `shift(1)` entfernt).

**Red Flag 3 — Calm 2017 funktioniert, Krisen nicht:**
```
calm_2017:  +14 % (ok)
gfc_2008:   -45 % (war -12 %)
covid_2020: -28 % (war +18 %)
```
Bedeutung: Strategie ist nur in ruhigen Zeiten robust. Das ist die gefährlichste Konfiguration.

**Red Flag 4 — Scenario-Varianz explodiert:**
```
Vor Refactor:  Num Trades: 142 / 89 / 103 / 67
Nach Refactor: Num Trades: 2341 / 1823 / 1977 / 2134
```
Refactor hat Signal-Generation komplett geändert. Turnover > 10 × vorher = Strategie ist nicht mehr dieselbe.

---

## 6. Determinism-Probleme

### 6.1 Die typischen Seed-Leaks

Auch mit `conftest.py` und `autouse=True`-Fixture kann die Pipeline non-deterministisch sein. Häufigste Ursachen:

**Lek 1 — Module setzt eigenen Seed irgendwo:**

```python
# Irgendwo in der Codebase
import numpy as np
np.random.seed(int(time.time()))  # !!
```

**Fix:** Grep nach `random.seed\|np.random.seed\|torch.manual_seed` und sicherstellen, dass es nur im conftest.py passiert.

**Lek 2 — PyTorch Convolution-Non-Determinism:**

PyTorch-Convolutions auf GPU sind nicht deterministisch, auch mit `torch.manual_seed`. 

**Fix:**
```python
import torch
torch.use_deterministic_algorithms(True, warn_only=True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
```

Wenn du auf GPU trainierst: alle Characterization-Tests auf CPU zwingen.

**Lek 3 — Pandas-DataFrame-Iteration-Order:**

```python
for key, group in df.groupby("ticker"):
    process(group)  # Reihenfolge von ticker ist nicht garantiert!
```

**Fix:**
```python
for key in sorted(df["ticker"].unique()):
    group = df[df["ticker"] == key]
    process(group)
```

**Lek 4 — Timestamps mit Mikrosekunden:**

```python
# Wenn Orders-Zeitstempel mit datetime.now() erzeugt werden, jeder Run anders
order.timestamp = datetime.now()  # !!
```

**Fix:**
```python
# Timestamps aus den Bar-Daten nehmen, nicht aus System-Zeit
order.timestamp = bar.timestamp
```

**Lek 5 — Dict-Iteration-Order:**

Python 3.7+ garantiert Insertion-Order für dicts, aber nicht für JSON-Serialisierung oder ältere Set-Operationen.

**Fix:**
```python
# Explizit sortieren vor Serialisierung
json.dumps(data, sort_keys=True)
```

**Lek 6 — Hash-Seed:**

Sets/Frozensets haben Iteration-Order, die vom PYTHONHASHSEED abhängt.

**Fix:** In `conftest.py` wird `PYTHONHASHSEED` bereits gesetzt. Aber: dieser Wert muss **vor** dem Python-Start gesetzt sein, sonst greift er nicht. Lösung:

```python
# conftest.py
import sys, os

if os.environ.get("PYTHONHASHSEED") != "42":
    # Neustart mit PYTHONHASHSEED=42
    os.environ["PYTHONHASHSEED"] = "42"
    os.execv(sys.executable, [sys.executable] + sys.argv)
```

Das ist nuklear, aber funktioniert.

### 6.2 Der Reproducibility-Check

**Bevor du die .approved-Datei committest, muss der Test 3× hintereinander passieren, ohne Diff:**

```bash
for i in 1 2 3; do
    rm -f tests/characterization/approved/*.received.txt
    pytest tests/characterization/test_golden_equity.py -v
done
```

Wenn ein Run einen Diff erzeugt: **Determinism-Leak**. Vor Approval fixen.

---

## 7. Wie du die .approved-Dateien verwaltest

### 7.1 Der Approval-Workflow

Wenn ein Test fehlschlägt:

```bash
$ pytest tests/characterization/test_golden_equity.py
...
FAILED - Diff detected in test_golden_equity_trend_baseline_2024

Received file: test_golden_equity_trend_baseline_2024.received.txt
Approved file: test_golden_equity_trend_baseline_2024.approved.txt
```

Du öffnest den Diff (ApprovalTests integriert sich mit Git-Diff, VSCode-Diff, PyCharm-Diff etc.). Du prüfst:

**Fall A: Unbeabsichtigte Änderung.** 
Du hast was kaputt gemacht. Rollback der Code-Änderung, .approved bleibt wie es ist.

**Fall B: Beabsichtigte Änderung.** 
Du hast bewusst die Strategie geändert (z.B. EMA-Periode auf 21 erhöht). Der Diff ist korrekt.

```bash
# Approve: received ersetzt approved
mv tests/characterization/approved/test_golden_equity_trend_baseline_2024.received.txt \
   tests/characterization/approved/test_golden_equity_trend_baseline_2024.approved.txt

git add tests/characterization/approved/test_golden_equity_trend_baseline_2024.approved.txt
git commit -m "test: re-approve golden equity after EMA period change to 21

Strategy change: EMA short window 20 → 21 (issue #123).
Equity curve shifts visible from 2024-03-12 onwards, magnitude ~0.5-1.5%.
Reviewed and confirmed expected."
```

### 7.2 Die Commit-Message-Disziplin

**Wenn du .approved-Dateien neu approvest, muss die Commit-Message erklären warum.**

Schlechte Commit-Message:
```
test: update approved files
```

Gute Commit-Message:
```
test: re-approve golden equity curves after position sizer refactor

Changed position sizing from fixed notional to volatility-targeted (PR #45).
Expected: smaller positions in high-vol regimes, larger in low-vol.
Verified: scenario_covid_2020 max_drawdown improved from -0.19 to -0.14 (good).
         scenario_calm_2017 total_return drops from +0.13 to +0.09 (expected, 
         less leverage in low-vol periods).
Reviewed by: Hans, 2026-04-24.
```

**Warum so ausführlich:** In 3 Monaten, wenn du dich fragst "warum ist die 2024er Equity-Kurve anders als im alten Backtest?", zeigt `git log -p tests/characterization/approved/` die Antwort. Ohne Kontext in der Commit-Message ist das nutzlos.

### 7.3 Die "niemals auto-approven"-Regel

**Anti-Pattern:**
```bash
# AM ENDE JEDER SESSION
find tests/characterization/approved -name "*.received.txt" -exec bash -c \
    'mv "$0" "${0/.received/.approved}"' {} \;
```

Das macht die Characterization-Tests **wertlos**. Jede Verhaltens-Änderung wird stillschweigend akzeptiert. Du hast wieder den Zustand, den du vermeiden wolltest.

**Strenge Regel:** Manuelles Approval, einzeln, mit Commit-Message je Datei.

---

## 8. Property-Based-Testing als Ergänzung

Characterization-Tests prüfen **konkrete Fälle**. Property-Based-Testing (PBT) prüft **Eigenschaften, die für alle Fälle gelten müssen**.

### 8.1 Die Library: Hypothesis

```bash
uv pip install hypothesis==6.122.3
```

### 8.2 Beispiele für Trading-Invariants

**Invariant 1: Cash plus Positions-Wert = Equity**

```python
# tests/characterization/test_invariants.py
from hypothesis import given, strategies as st
import hypothesis.strategies as st

@given(
    prices=st.lists(
        st.floats(min_value=1.0, max_value=1000.0, allow_nan=False),
        min_size=10, max_size=100,
    ),
    initial_cash=st.floats(min_value=10_000, max_value=1_000_000),
)
def test_invariant_equity_equals_cash_plus_positions(prices, initial_cash):
    """Equity = Cash + sum(Quantity_i × Price_i). Immer."""
    portfolio = Portfolio(initial_cash=initial_cash)
    
    for price in prices:
        # Zufällige Order (kauft, wenn Cash > Preis)
        if portfolio.cash > price * 10:
            portfolio.buy("XYZ", 10, price)
        elif portfolio.position("XYZ") > 5:
            portfolio.sell("XYZ", 5, price)
        
        # Invariant muss IMMER halten
        equity = portfolio.cash + portfolio.position_value({"XYZ": price})
        assert abs(equity - portfolio.equity({"XYZ": price})) < 1e-6
```

**Invariant 2: Orders-Sum = Position-Changes**

```python
@given(
    orders=st.lists(
        st.tuples(
            st.sampled_from(["buy", "sell"]),
            st.integers(min_value=1, max_value=1000),
            st.floats(min_value=1.0, max_value=1000.0),
        ),
        min_size=1, max_size=50,
    )
)
def test_invariant_position_change_matches_orders(orders):
    """Sum(buy.qty) - Sum(sell.qty) == Position-Change."""
    portfolio = Portfolio(initial_cash=1_000_000)
    start_pos = portfolio.position("XYZ")
    
    net_change = 0
    for side, qty, price in orders:
        if side == "buy" and portfolio.cash >= qty * price:
            portfolio.buy("XYZ", qty, price)
            net_change += qty
        elif side == "sell" and portfolio.position("XYZ") >= qty:
            portfolio.sell("XYZ", qty, price)
            net_change -= qty
    
    end_pos = portfolio.position("XYZ")
    assert end_pos - start_pos == net_change
```

**Invariant 3: Sharpe is scale-invariant**

```python
@given(
    returns=st.lists(
        st.floats(min_value=-0.1, max_value=0.1, allow_nan=False),
        min_size=30, max_size=1000,
    ),
    scale=st.floats(min_value=0.01, max_value=100.0, allow_nan=False),
)
def test_invariant_sharpe_scale_invariant(returns, scale):
    """sharpe(returns) == sharpe(returns × scale). Skalen-Invarianz."""
    from assembled_core.metrics import sharpe_ratio
    import numpy as np
    
    returns_arr = np.array(returns)
    sharpe_original = sharpe_ratio(returns_arr)
    sharpe_scaled = sharpe_ratio(returns_arr * scale)
    
    if not np.isnan(sharpe_original):
        assert abs(sharpe_original - sharpe_scaled) < 1e-6
```

### 8.3 Wo PBT brilliert

PBT ist besonders stark für:

- **Encode-Decode-Roundtrips:** `deserialize(serialize(x)) == x`
- **Commutativity:** `transform(a) + transform(b) == transform(a + b)`
- **Idempotenz:** `normalize(normalize(x)) == normalize(x)`
- **Invariants:** Eigenschaften, die nie gebrochen werden dürfen (Cash nicht negativ, Positions-Größe ≤ Kapazität)

**Für deinen Plan wichtig:** PBT **ersetzt nicht** Characterization-Tests. Sie sind komplementär. PBT prüft, dass **keine Invariants** gebrochen werden; Characterization prüft, dass **das spezifische Verhalten** stabil bleibt.

---

## 9. Performance-Budget

### 9.1 Das Problem

Characterization-Tests werden bei jedem CI-Build ausgeführt. Wenn der gesamte Characterization-Test-Suite 30 Minuten dauert, hasst du ihn nach einer Woche.

**Ziel:** Gesamter Characterization-Test-Lauf < 5 Minuten.

### 9.2 Strategien zur Beschleunigung

**1. Fixture-Größe minimieren.**
- Nicht 10 Jahre Daily-Bars, sondern 6 Monate Hourly oder 3 Jahre Daily.
- Nicht 100 Ticker, sondern 3-5.
- Nur die nötigen Spalten (nicht Full-OHLC plus Volume plus Dividends, nur Close).

**2. Parallel ausführen:**
```bash
pytest tests/characterization/ -n 4   # pytest-xdist, 4 parallel
```

**3. Test-Marker differenzieren:**
```python
@pytest.mark.characterization        # auf CI immer
@pytest.mark.scenario                # nur nightly
@pytest.mark.slow                    # nur manuell
```

**4. Fixtures cachen:**
```python
@pytest.fixture(scope="session")
def shared_bars(fixture_dir):
    """Lade Bars nur einmal pro Pytest-Session."""
    return pd.read_parquet(fixture_dir / "bars_3tickers_h1_2024.parquet")
```

### 9.3 Budget pro Test-Kategorie

| Kategorie | Laufzeit pro Test | Anzahl | Gesamt |
|---|---|---|---|
| Feature-Tests (1 Tag × 1 Ticker) | < 1s | 5-10 | < 10s |
| Regime-Tests (4 Jahre) | 2-5s | 1-2 | < 10s |
| Golden-Equity (H1 2024) | 30-60s | 1-2 | ~60s |
| Scenario-Tests (GFC, COVID, Rates, Calm) | 30-60s | 4 | 2-4min |
| Property-Based (Hypothesis) | variabel | 5-10 | 1-2min |
| **Gesamt** | | | **~5min** |

---

## 10. Umsetzungs-Checkliste

**Phase 1 — Infrastruktur (Tag 1-2):**
- [ ] ApprovalTests installieren, Reporter-Konfiguration für dein Diff-Tool
- [ ] `tests/characterization/` Struktur anlegen
- [ ] `conftest.py` mit Determinism-Setup
- [ ] `.gitignore` für .received-Dateien
- [ ] Fixture-Build-Script (`scripts/characterization/build_scenario_fixtures.py`)

**Phase 2 — Die 5 Core-Tests (Tag 3-5):**
- [ ] `test_golden_equity.py` (trend_baseline H1 2024)
- [ ] `test_golden_orders.py`
- [ ] `test_golden_features.py` (single day × ticker)
- [ ] `test_golden_regime.py` (SPY 2020-2024)
- [ ] `test_golden_risk_metrics.py`

**Phase 3 — Scenario-Tests (Tag 6-7):**
- [ ] Fixtures für GFC 2008, COVID 2020, Rates 2022, Calm 2017
- [ ] `test_scenarios.py` mit `@pytest.mark.parametrize`
- [ ] Alle 4 Approvals generiert und committed

**Phase 4 — Sanity-Injects (Tag 8):**
- [ ] 5 bewusste Bugs injiziert, jeder produziert Test-Fail
- [ ] Determinism 3×-run verifiziert
- [ ] CI-Integration getestet

**Phase 5 — Property-Based-Tests (Tag 9-10, optional):**
- [ ] `test_invariants.py` mit 3-5 Hypothesis-Tests
- [ ] Invariants für Portfolio, Metrics, Order-Lifecycle

**Phase 6 — Dokumentation (Tag 11):**
- [ ] README für `tests/characterization/` mit Approval-Workflow
- [ ] Commit-Message-Template für Re-Approvals
- [ ] Performance-Budget-Tracking

**Gesamt-Aufwand:** 2-3 Wochen bei 10-15 h/Woche. Phase 5 optional, kann zur späterer Zeit addiert werden.

---

## 11. Quellen

**Characterization / Approval Testing:**
- Michael Feathers, "Working Effectively with Legacy Code" (2004) — Kapitel 12
- [ApprovalTests.Python auf GitHub](https://github.com/approvals/ApprovalTests.Python)
- [approvaltests 14.2.0 auf PyPI](https://pypi.org/project/approvaltests/14.2.0/)
- Nicolas Carlo: [Best way to start testing untested code](https://understandlegacycode.com/blog/best-way-to-start-testing-untested-code/)
- NS-Techblog (Zeger Hendrikse): [TDD and legacy code: creating a snapshot with approval tests](https://medium.com/ns-techblog/tdd-and-legacy-code-creating-a-snapshot-with-approval-tests-252327b6c72e)

**Reproducibility:**
- Daniel Godoy (2022): [Random Seeds and Reproducibility](https://medium.com/data-science/random-seeds-and-reproducibility-933da79446e3)
- Krafczyk et al. (2021): Learning from reproducing computational results. [DOI](https://doi.org/10.1098/rsta.2020.0069)
- Ogochukwu Stanley Ikegbo (2024): [Ensuring Consistent Random Outputs for Reproducibility in ML](https://medium.com/@stacymacbrains/ensuring-consistent-random-outputs-for-reproducibility-in-machine-learning-9bb23165f5c1)

**Property-Based Testing:**
- [Hypothesis Python Library](https://hypothesis.readthedocs.io/)
- Maaz, DeVoe, Hatfield-Dodds, Carlini (Anthropic, 2025): [Agentic Property-Based Testing](https://arxiv.org/pdf/2510.09907) — LLM-based property discovery
- OneUptime (2026): [How to Build Property-Based Testing with Hypothesis](https://oneuptime.com/blog/post/2026-01-30-how-to-build-property-based-testing-with-hypothesis/view)

**Backtest-Scenario-Tests:**
- VolatilityBox (2026): [Volatility Regime Detection — historische Krisen-Backtests](https://volatilitybox.com/research/volatility-regime-detection/)
- PyQuant News (2025): [Backtesting Multi-Asset Portfolios for True Resilience](https://www.pyquantnews.com/free-python-resources/backtesting-multi-asset-portfolios-for-true-resilience-cdar-optimization-with-riskfolio-lib-vectorbt)
- financial-risk-analyzer (vdamov): Historical crisis stress testing toolkit

**Backtest-Frameworks-Referenz:**
- [backtesting.py](https://kernc.github.io/backtesting.py/)
- [VectorBT](https://vectorbt.dev/)
- QuantStart: [Backtesting Systematic Trading Strategies in Python](https://www.quantstart.com/articles/backtesting-systematic-trading-strategies-in-python-considerations-and-open-source-frameworks/)

---

## 12. Ehrliche Einschätzung

**Was dieses Playbook dir gibt:**
- Sicherheit, dass Refactoring die Strategie nicht still bricht
- Dokumentation des aktuellen Verhaltens als Test
- Krisen-Stress-Tests gegen historisch relevante Perioden
- Disziplin, Verhaltens-Änderungen bewusst zu genehmigen

**Was es dir nicht gibt:**
- Korrektheit der aktuellen Strategie. Characterization-Tests dokumentieren das **existierende** Verhalten, nicht das **gewünschte**. Wenn dein Backtest einen versteckten Look-Ahead-Bias hat, friert der Test diesen Bug ein.
- Alpha-Validation. Das ist in `34_NEWS_GROUND_TRUTH.md` und `32_VALIDIERUNG.md` Thema.
- Live-Tradability. Characterization lauft im Backtest. Live kann Slippage/Latenz anders sein.

**Die drei Sachen, die du nicht auslassen darfst:**
1. **Der Sanity-Inject-Test.** Wenn du nicht verifizierst, dass ein absichtlicher Bug einen Test-Fail produziert, weißt du nicht, ob deine Tests scharf sind.
2. **Determinism-3×-Run vor Approval.** Einmal "passt" ist Zufall. Drei Mal passt = wahrscheinlich deterministisch.
3. **Commit-Message-Disziplin bei Re-Approvals.** Ohne die wirst du in 3 Monaten nicht mehr wissen, warum du die .approved-Datei geändert hast.

**Der Gesamtaufwand von ~2-3 Wochen** klingt nach viel, aber er ist eine **einmalige** Investition. Nach Phase 2 des Migrations-Playbooks hast du das Test-Netz für die nächsten Jahre Refactoring. Ohne dieses Netz dauert jeder spätere Refactor-Versuch doppelt so lang und ist halb so sicher.

Die wichtigste Erkenntnis ist nicht technischer Art, sondern psychologisch: **mit einem Characterization-Test-Netz traust du dich, Code anzupacken, den du sonst nicht anfassen würdest.** Das ist der eigentliche Wert.
