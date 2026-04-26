# 38 — Feature-Attribution-Dashboard

**Zweck:** Wenn ein Signal "BUY AAPL" generiert wird — warum? Welche Features haben wie viel zu dieser Entscheidung beigetragen? Ohne Attribution ist dein System eine Blackbox, und du verlierst die Fähigkeit, schlechte Trades zu lernen.

**Scope:** Rang 6 aus der Gap-Analyse. Bezieht sich auf Composite-Score-Architektur aus `31_COMPOSITE_SCORE.md` und News-Features aus `30_NEWS_TA_FUSION.md`.

**Kern-Idee:** Jede Trading-Entscheidung bekommt einen Attribution-Report. Entweder analytisch (bei linearem Composite-Score) oder via SHAP (bei ML-basierten Signalen). Daraus entsteht ein Dashboard, das zeigt: "Warum hat das System heute so gehandelt?"

---

## 0. Warum das wichtig ist — und warum die meisten Hobby-Quants es ignorieren

### Das Szenario, das dich kostet

Du hast eine 3-Wochen-Verlust-Serie. Dein Composite-Score hat 9 Dimensionen (News, Sentiment, Trend, Momentum, Volatility, Regime, Volume, Microstructure, Fundamentals). Jede hat ein Gewicht zwischen 0 und 1. Du willst wissen: **Welche der 9 Dimensionen hat die falschen Trades getrieben?**

Ohne Attribution:
- Du gucks dir die Gesamt-Score-Zeitreihe an
- Du siehst, dass der Composite-Score für AAPL am 15.03. bei +0.72 lag → Long-Signal
- Am 18.03. war AAPL -4 %. Verlust.
- Du weißt: Signal war zu hoch. **Nicht**: welche Dimension den Fehler gemacht hat.

Mit Attribution:
- Score-Breakdown am 15.03.: News +0.45, Trend +0.25, Momentum +0.15, Rest negativ zusammen -0.13
- Du siehst: News-Dimension hat 63 % zum Signal beigetragen
- Die News vom 14.03. war "AAPL CEO considering retirement" — von FinBERT als positiv klassifiziert (Fehler)
- **Fehler gefunden:** FinBERT versteht das "considering" nicht als Unsicherheit, klassifiziert die Headline positiv statt neutral

Ohne die Attribution hättest du Wochen debuggen können, ohne den eigentlichen Bug zu finden.

### Die zwei Hauptgründe, warum niemand das macht

**Grund 1 — "Mein Modell ist linear, ich brauche kein SHAP."** 
Das stimmt halb. Ein linearer Composite-Score braucht kein SHAP für lokale Attribution — der Contribution-Plot ist direkt ablesbar. Aber: die **meisten** Signale sind nicht rein linear. News-Sentiment durchläuft FinBERT (nicht-linear), Volatilität wird via GARCH geschätzt (nicht-linear), Regime ist diskret. Du hast schnell eine Mischung.

**Grund 2 — "SHAP ist zu langsam für Live-Trading."** 
Auch halb richtig. SHAP auf einem Random-Forest mit 500 Features ist zu langsam für Milliseconden-Entscheidungen. Aber bei 9 Dimensionen und stündlicher Entscheidung hast du Millionen Sekunden Zeit. TreeSHAP auf einem 9-Feature-Modell: <10 ms.

Die wahre Ursache ist: **der Aufwand zahlt sich nur aus, wenn du dein System debugst.** Und solange du keine Verlust-Serie hast, fehlt der Druck. Das Dashboard baust du **bevor** die Verlust-Serie kommt, nicht danach.

---

## 1. Das Architektur-Modell

### 1.1 Zwei Ebenen der Attribution

**Ebene A — Composite-Score-Level:**
- Frage: welche der 9 Dimensionen hat wie viel beigetragen?
- Methode: direkte Contribution-Zerlegung, weil der Composite eine gewichtete Summe ist
- Output: Bar-Chart, Waterfall, Time-Series
- Latenz: < 1 ms
- Zeitpunkt: bei jeder Signal-Generation

**Ebene B — Innerhalb einer Dimension (Feature-Level):**
- Frage: **innerhalb** der News-Dimension, welche News-Sub-Features (Sentiment, Volume-Spike, Uncertainty, Corroboration etc.) trieben den Score?
- Methode: SHAP-Values auf dem Sub-Modell (z.B. Random-Forest, der aus 6 News-Sub-Features den News-Score berechnet)
- Output: SHAP-Waterfall, Force-Plot, Beeswarm
- Latenz: 10-100 ms
- Zeitpunkt: on-demand oder in Post-Mortem-Analysen

### 1.2 Der Datenfluss

```
┌────────────────────────────────────┐
│  Feature-Engineering               │
│  (9 Dimensionen × N Sub-Features)  │
└────────────┬───────────────────────┘
             │
             ▼
┌────────────────────────────────────┐
│  Dimension-Scoring                 │
│  News-Model → News-Score           │
│  Trend-Model → Trend-Score         │
│  ... (9 Modelle)                    │
└────────────┬───────────────────────┘
             │
             ▼
┌────────────────────────────────────┐
│  Composite-Score (linear combine)  │
│  composite = Σ w_i × dim_score_i   │
└────────────┬───────────────────────┘
             │
             ├──────► Signal → Execution
             │
             └──────► Attribution-Ledger (DB)
                      │
                      ▼
             ┌────────────────────────────────┐
             │  Attribution-Dashboard         │
             │  (Streamlit oder Flask)        │
             │  - Live: aktuelle Entscheidung │
             │  - Post: historische Analyse   │
             └────────────────────────────────┘
```

---

## 2. Ebene A: Composite-Score-Attribution

### 2.1 Die Datenstruktur

```python
# src/assembled_core/attribution/schemas.py
from dataclasses import dataclass
from datetime import datetime
from typing import Dict


@dataclass
class CompositeAttribution:
    """Attribution für einen einzelnen Composite-Score."""
    timestamp: datetime
    ticker: str
    composite_score: float
    
    # Pro Dimension
    dimension_contributions: Dict[str, float]  # dim_name → weighted contribution
    dimension_raw_scores: Dict[str, float]     # dim_name → raw score pre-weight
    dimension_weights: Dict[str, float]         # dim_name → weight
    
    # Context
    strategy_id: str
    model_version: str
    regime: str                                 # aktuelle Markt-Regime-Klassifikation
    
    def top_contributors(self, n: int = 3) -> Dict[str, float]:
        """Die n stärksten (absolut) Beitragenden."""
        sorted_contribs = sorted(
            self.dimension_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        return dict(sorted_contribs[:n])
```

### 2.2 Die Generator-Funktion

```python
# src/assembled_core/attribution/composite.py
import logging
from datetime import datetime
from typing import Dict

from ..features.registry import FeatureRegistry
from ..scoring.composite import CompositeScorer
from .schemas import CompositeAttribution
from .storage import AttributionStore

logger = logging.getLogger(__name__)


def compute_composite_with_attribution(
    ticker: str,
    features: Dict[str, float],
    scorer: CompositeScorer,
    strategy_id: str,
    timestamp: datetime,
    regime: str,
) -> tuple[float, CompositeAttribution]:
    """Berechnet Composite-Score UND gibt Attribution mit.
    
    Die Attribution ist ein Nebenprodukt ohne zusätzlichen Latenz-Kostenfaktor.
    """
    # Pro Dimension den Score berechnen
    dim_scores = {}
    for dim_name, dim_model in scorer.dimension_models.items():
        dim_features = features.get(dim_name, {})
        dim_scores[dim_name] = dim_model.score(dim_features)
    
    # Composite ist gewichtete Summe
    weights = scorer.weights  # z.B. {"news": 0.15, "trend": 0.20, ...}
    contributions = {
        dim: weights[dim] * score
        for dim, score in dim_scores.items()
    }
    composite = sum(contributions.values())
    
    # Attribution-Objekt
    attribution = CompositeAttribution(
        timestamp=timestamp,
        ticker=ticker,
        composite_score=composite,
        dimension_contributions=contributions,
        dimension_raw_scores=dim_scores,
        dimension_weights=weights,
        strategy_id=strategy_id,
        model_version=scorer.version,
        regime=regime,
    )
    
    return composite, attribution


def attribution_to_dict(attr: CompositeAttribution) -> dict:
    """Serialisierung für DB-Storage."""
    return {
        "timestamp": attr.timestamp.isoformat(),
        "ticker": attr.ticker,
        "composite_score": attr.composite_score,
        "dimension_contributions": attr.dimension_contributions,
        "dimension_raw_scores": attr.dimension_raw_scores,
        "dimension_weights": attr.dimension_weights,
        "strategy_id": attr.strategy_id,
        "model_version": attr.model_version,
        "regime": attr.regime,
    }
```

### 2.3 Storage-Schicht

```python
# src/assembled_core/attribution/storage.py
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .schemas import CompositeAttribution


class AttributionStore:
    """SQLite-basierter Attribution-Storage.
    
    Für Hetzner-Setup reicht SQLite. Bei >10 Mio Rows auf Postgres wechseln.
    """
    
    def __init__(self, db_path: str = "data/attributions.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
    
    def _init_schema(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS attributions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    composite_score REAL NOT NULL,
                    dimension_contributions_json TEXT NOT NULL,
                    dimension_raw_scores_json TEXT NOT NULL,
                    dimension_weights_json TEXT NOT NULL,
                    strategy_id TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    regime TEXT NOT NULL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ts_ticker ON attributions(timestamp, ticker)"
            )
    
    def save(self, attr: CompositeAttribution):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO attributions 
                (timestamp, ticker, composite_score, 
                 dimension_contributions_json, dimension_raw_scores_json, 
                 dimension_weights_json, strategy_id, model_version, regime)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                attr.timestamp.isoformat(),
                attr.ticker,
                attr.composite_score,
                json.dumps(attr.dimension_contributions),
                json.dumps(attr.dimension_raw_scores),
                json.dumps(attr.dimension_weights),
                attr.strategy_id,
                attr.model_version,
                attr.regime,
            ))
    
    def load_for_ticker(
        self, ticker: str, 
        start: Optional[datetime] = None, 
        end: Optional[datetime] = None,
    ) -> List[CompositeAttribution]:
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT * FROM attributions WHERE ticker = ?"
            params = [ticker]
            if start:
                query += " AND timestamp >= ?"
                params.append(start.isoformat())
            if end:
                query += " AND timestamp <= ?"
                params.append(end.isoformat())
            query += " ORDER BY timestamp"
            
            cursor = conn.execute(query, params)
            return [self._row_to_attribution(row) for row in cursor]
    
    def _row_to_attribution(self, row) -> CompositeAttribution:
        return CompositeAttribution(
            timestamp=datetime.fromisoformat(row[1]),
            ticker=row[2],
            composite_score=row[3],
            dimension_contributions=json.loads(row[4]),
            dimension_raw_scores=json.loads(row[5]),
            dimension_weights=json.loads(row[6]),
            strategy_id=row[7],
            model_version=row[8],
            regime=row[9],
        )
```

### 2.4 Integration in den Trading-Cycle

```python
# src/assembled_core/pipeline/cycle.py (Ausschnitt)
from ..attribution.composite import compute_composite_with_attribution
from ..attribution.storage import AttributionStore

attribution_store = AttributionStore()

def trading_cycle(...):
    for ticker in universe:
        features = compute_features(ticker, bars)
        regime = classify_regime(bars)
        
        composite, attribution = compute_composite_with_attribution(
            ticker=ticker,
            features=features,
            scorer=scorer,
            strategy_id="trend_news_v3",
            timestamp=datetime.utcnow(),
            regime=regime,
        )
        
        # Save immer, unabhängig vom Signal
        attribution_store.save(attribution)
        
        # Signal-Entscheidung
        if composite > 0.6:
            submit_order(ticker, "BUY")
        elif composite < -0.6:
            submit_order(ticker, "SELL")
```

**Wichtig:** Attribution wird **immer** gespeichert, auch wenn kein Signal generiert wird. Für Post-Mortem ist auch "warum KEIN Signal?" eine wertvolle Frage.

---

## 3. Ebene B: SHAP-Attribution innerhalb einer Dimension

### 3.1 Wann braucht man SHAP

**Brauchst du SHAP nicht:**
- Dein News-Score ist `w_sentiment * sentiment + w_volume * volume_spike + ...` (linear) → direkte Zerlegung möglich
- Dein Trend-Score ist `(ema_short - ema_long) / ema_long` (deterministisch) → Erklärung ist per Definition klar

**Brauchst du SHAP:**
- News-Score kommt aus einem Random-Forest mit 6 Sub-Features → Zerlegung nicht-linear
- Regime-Klassifikation ist ein XGBoost-Modell mit 15 Features
- Sentiment-Aggregation nutzt ein neuronales Netz

**Ehrlicher Rat für deinen Stack:** Wahrscheinlich brauchst du SHAP nur für 2-3 der 9 Dimensionen. Nicht überall einbauen.

### 3.2 SHAP-Installation und Setup

```bash
uv pip install shap==0.46.0
```

**Version 0.46 ist der aktuelle Stand April 2026**, mit aktiver Wartung. Die SPEC-0-Konformität garantiert Python 3.11+ Support.

### 3.3 TreeSHAP für Random-Forest-basierte Sub-Scores

```python
# src/assembled_core/attribution/shap_explainer.py
import shap
import numpy as np
import pandas as pd
from typing import Dict, List


class DimensionExplainer:
    """SHAP-Explainer für eine einzelne Composite-Dimension.
    
    Bindet sich an ein trainiertes Tree-Modell (sklearn/xgboost/lightgbm).
    """
    
    def __init__(self, model, feature_names: List[str], background_data: pd.DataFrame):
        self.model = model
        self.feature_names = feature_names
        self.background_data = background_data
        
        # TreeExplainer ist für sklearn/xgboost/lightgbm optimiert
        # Latenz: ~1-10 ms pro Prediction
        self.explainer = shap.TreeExplainer(
            model, 
            data=background_data,
            feature_perturbation="interventional",
        )
    
    def explain_single(self, feature_values: Dict[str, float]) -> Dict[str, float]:
        """SHAP-Werte für einen einzelnen Input.
        
        Returns:
            dict feature_name → SHAP-value
        """
        x = np.array([[feature_values[name] for name in self.feature_names]])
        shap_values = self.explainer.shap_values(x)
        
        # Für binäre Klassifikation: shap_values ist Liste [class0, class1]
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # positive class
        
        return dict(zip(self.feature_names, shap_values[0].tolist()))
    
    def explain_batch(self, X: pd.DataFrame) -> pd.DataFrame:
        """SHAP-Werte für eine Batch. Für Post-Mortem-Analysen."""
        shap_values = self.explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        return pd.DataFrame(shap_values, columns=self.feature_names, index=X.index)
```

### 3.4 Der News-Dimension-Explainer (konkret)

```python
# src/assembled_core/news/explainer.py
import pickle
from pathlib import Path
from typing import Dict

from ..attribution.shap_explainer import DimensionExplainer


NEWS_SUB_FEATURES = [
    "sentiment_score",          # FinBERT/Claude output
    "news_volume_spike",        # z.B. 3× 20-day avg
    "source_quality_weight",    # Reuters > Seeking Alpha
    "headline_uncertainty",     # LLM-based score
    "topic_cluster_signal",     # HDBSCAN-based
    "cross_source_corroboration",
]


class NewsDimensionExplainer:
    """SHAP-basierte Erklärung für News-Composite-Dimension."""
    
    def __init__(self, model_dir: Path):
        # Modell wurde im Training-Job erzeugt und als .pkl gespeichert
        self.model = pickle.load(open(model_dir / "news_model_v3.pkl", "rb"))
        
        # Background-Data: ein repräsentatives Sample aus Training
        # Muss im selben Verzeichnis liegen
        self.background = pd.read_parquet(model_dir / "news_background_sample.parquet")
        
        self.explainer = DimensionExplainer(
            model=self.model,
            feature_names=NEWS_SUB_FEATURES,
            background_data=self.background,
        )
    
    def explain(self, feature_values: Dict[str, float]) -> Dict[str, float]:
        """Erklärt, warum das News-Modell einen bestimmten Score liefert."""
        return self.explainer.explain_single(feature_values)
    
    def top_drivers(self, feature_values: Dict[str, float], n: int = 3) -> Dict[str, float]:
        """Die n wichtigsten Treiber (absolut)."""
        shap_values = self.explain(feature_values)
        sorted_vals = sorted(
            shap_values.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        return dict(sorted_vals[:n])
```

---

## 4. Das Dashboard (Streamlit)

### 4.1 Warum Streamlit statt Grafana

- **Grafana** ist super für Metriken (Counter, Gauge, Histogram)
- **Streamlit** ist super für interaktive Analyse mit Python-nativen Widgets (SHAP-Plots funktionieren direkt)

Für Attribution brauchst du interaktive Analyse. Streamlit.

### 4.2 Setup

```bash
uv pip install streamlit==1.41.1
uv pip install plotly==5.24.1
```

### 4.3 Das Basis-Dashboard

```python
# dashboards/attribution_app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

from assembled_core.attribution.storage import AttributionStore


@st.cache_resource
def get_store():
    return AttributionStore()


def main():
    st.set_page_config(page_title="ATA Attribution", layout="wide")
    st.title("Feature-Attribution-Dashboard")
    
    store = get_store()
    
    # Sidebar: Filters
    with st.sidebar:
        st.header("Filters")
        ticker = st.text_input("Ticker", value="AAPL")
        days_back = st.slider("Days back", min_value=1, max_value=90, value=7)
        
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days_back)
    
    # Data-Load
    attributions = store.load_for_ticker(ticker, start=start_date, end=end_date)
    
    if not attributions:
        st.warning(f"Keine Attribution-Daten für {ticker} im gewählten Zeitraum.")
        return
    
    # Convert to DataFrame für Plotting
    df = pd.DataFrame([{
        "timestamp": a.timestamp,
        "composite_score": a.composite_score,
        "regime": a.regime,
        **{f"contrib_{k}": v for k, v in a.dimension_contributions.items()},
        **{f"raw_{k}": v for k, v in a.dimension_raw_scores.items()},
    } for a in attributions])
    
    # Section 1: Composite-Score Time-Series
    st.header(f"Composite-Score für {ticker}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["composite_score"],
        mode="lines+markers", name="Composite",
        line=dict(width=2),
    ))
    # Regime als farbiger Hintergrund
    fig.add_hline(y=0.6, line_dash="dot", annotation_text="BUY threshold")
    fig.add_hline(y=-0.6, line_dash="dot", annotation_text="SELL threshold")
    st.plotly_chart(fig, use_container_width=True)
    
    # Section 2: Dimension-Contribution Stacked Bar
    st.header("Beiträge pro Dimension (gestapelt)")
    contrib_cols = [c for c in df.columns if c.startswith("contrib_")]
    
    fig2 = go.Figure()
    for col in contrib_cols:
        dim_name = col.replace("contrib_", "")
        fig2.add_trace(go.Bar(
            x=df["timestamp"], y=df[col],
            name=dim_name,
        ))
    fig2.update_layout(barmode="relative", height=500)
    st.plotly_chart(fig2, use_container_width=True)
    
    # Section 3: Waterfall für letzten Zeitpunkt
    st.header("Waterfall für jüngste Entscheidung")
    latest = attributions[-1]
    
    fig3 = go.Figure(go.Waterfall(
        orientation="v",
        measure=["relative"] * len(latest.dimension_contributions) + ["total"],
        x=list(latest.dimension_contributions.keys()) + ["Composite"],
        y=list(latest.dimension_contributions.values()) + [latest.composite_score],
        connector=dict(line=dict(color="rgb(63, 63, 63)")),
    ))
    fig3.update_layout(title=f"Attribution at {latest.timestamp}")
    st.plotly_chart(fig3, use_container_width=True)
    
    # Section 4: Top-Drivers-Table
    st.header("Top-5 Drivers je Stunde")
    driver_rows = []
    for a in attributions[-24:]:  # letzte 24 Entscheidungen
        top = sorted(
            a.dimension_contributions.items(),
            key=lambda x: abs(x[1]), reverse=True,
        )[:5]
        driver_rows.append({
            "timestamp": a.timestamp,
            "composite": a.composite_score,
            **{f"top{i+1}": f"{name}: {val:+.3f}" 
               for i, (name, val) in enumerate(top)},
        })
    st.dataframe(pd.DataFrame(driver_rows))


if __name__ == "__main__":
    main()
```

### 4.4 Ausführen

```bash
streamlit run dashboards/attribution_app.py --server.port 8501
```

Auf Hetzner via SSH-Tunnel:
```bash
ssh -L 8501:localhost:8501 ata@hetzner-server
# Dann im Browser: http://localhost:8501
```

**Kein öffentlicher Zugang.** Dashboard zeigt sensitive Trading-Daten. Niemals auf öffentlicher Port.

---

## 5. Der Post-Mortem-Workflow

### 5.1 Wenn ein Trade schief geht

```python
# scripts/attribution/post_mortem.py
"""
Post-Mortem: Warum wurde dieser Trade gemacht?

Usage:
    python -m scripts.attribution.post_mortem \
        --ticker AAPL \
        --entry-time 2026-03-15T10:00:00 \
        --exit-time 2026-03-18T15:30:00
"""
import argparse
from datetime import datetime
from assembled_core.attribution.storage import AttributionStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--entry-time", required=True)
    parser.add_argument("--exit-time", required=True)
    args = parser.parse_args()
    
    entry = datetime.fromisoformat(args.entry_time)
    exit_ = datetime.fromisoformat(args.exit_time)
    
    store = AttributionStore()
    
    # 1. Attribution zum Entry-Zeitpunkt
    entry_attrs = store.load_for_ticker(args.ticker, 
                                          start=entry - pd.Timedelta(minutes=5),
                                          end=entry + pd.Timedelta(minutes=5))
    if not entry_attrs:
        print(f"KEINE Attribution gefunden für {args.ticker} bei {entry}")
        return
    
    entry_attr = entry_attrs[0]
    
    print(f"\n=== ENTRY-ATTRIBUTION ({args.ticker} @ {entry_attr.timestamp}) ===\n")
    print(f"Composite-Score: {entry_attr.composite_score:+.4f}")
    print(f"Regime: {entry_attr.regime}")
    print(f"Strategy: {entry_attr.strategy_id}")
    print(f"Model: {entry_attr.model_version}")
    print()
    
    # Top Contributors
    sorted_contribs = sorted(
        entry_attr.dimension_contributions.items(),
        key=lambda x: abs(x[1]), reverse=True,
    )
    print("Contributors (sortiert):")
    for name, contrib in sorted_contribs:
        raw = entry_attr.dimension_raw_scores[name]
        weight = entry_attr.dimension_weights[name]
        pct = abs(contrib) / sum(abs(c) for c in entry_attr.dimension_contributions.values()) * 100
        print(f"  {name:20s} {contrib:+.4f}  "
              f"(raw: {raw:+.4f}  × w: {weight:.3f})  [{pct:.1f}%]")
    
    # 2. Was hat sich zwischen Entry und Exit verändert?
    exit_attrs = store.load_for_ticker(args.ticker,
                                         start=exit_ - pd.Timedelta(minutes=5),
                                         end=exit_ + pd.Timedelta(minutes=5))
    if exit_attrs:
        exit_attr = exit_attrs[0]
        print(f"\n=== DELTA ENTRY → EXIT ({(exit_ - entry).total_seconds()/3600:.1f}h) ===\n")
        for dim in entry_attr.dimension_raw_scores:
            delta_raw = exit_attr.dimension_raw_scores[dim] - entry_attr.dimension_raw_scores[dim]
            print(f"  {dim:20s}  Δ raw: {delta_raw:+.4f}")
    
    # 3. Action-Items
    print("\n=== ACTION-ITEMS ===\n")
    top_driver = sorted_contribs[0]
    print(f"Top-Driver: {top_driver[0]} ({top_driver[1]:+.4f})")
    print(f"→ Prüfe: waren die Sub-Features der {top_driver[0]}-Dimension korrekt?")
    print(f"→ Bei News-Dimension: Headline-Archivierung prüfen")
    print(f"→ Bei Trend-Dimension: Bar-Daten prüfen (wegen Data-Quality-Gate)")


if __name__ == "__main__":
    main()
```

### 5.2 Die wöchentliche Attribution-Review

**Jeden Freitag Abend:** Review aller Trades der Woche mit Attribution.

```python
# scripts/attribution/weekly_review.py
"""Wöchentlicher Attribution-Review.

Generiert Bericht: Welche Dimensionen haben in dieser Woche welche Trades getrieben?
"""
import pandas as pd
from datetime import datetime, timedelta


def weekly_review():
    store = AttributionStore()
    orders_db = OrdersStore()
    
    end = datetime.utcnow()
    start = end - timedelta(days=7)
    
    orders = orders_db.get_orders(start, end)
    
    report_rows = []
    for order in orders:
        attr = store.load_for_ticker(
            order.ticker,
            start=order.submitted_at - timedelta(minutes=5),
            end=order.submitted_at + timedelta(minutes=5),
        )
        if not attr:
            continue
        
        a = attr[0]
        top = max(a.dimension_contributions.items(), key=lambda x: abs(x[1]))
        
        report_rows.append({
            "ticker": order.ticker,
            "side": order.side,
            "pnl_pct": order.pnl_pct,
            "composite": a.composite_score,
            "top_driver": top[0],
            "top_driver_pct": abs(top[1]) / sum(abs(c) for c in a.dimension_contributions.values()),
            "regime": a.regime,
        })
    
    df = pd.DataFrame(report_rows)
    
    # Aggregations
    print("=== Attribution Weekly Review ===\n")
    
    print("Trades by top-driver:")
    print(df.groupby("top_driver").size().sort_values(ascending=False))
    print()
    
    print("PnL by top-driver:")
    print(df.groupby("top_driver")["pnl_pct"].agg(["mean", "sum", "count"]))
    print()
    
    print("PnL by regime:")
    print(df.groupby("regime")["pnl_pct"].agg(["mean", "sum", "count"]))
    print()
    
    # Schlechteste Trades
    worst = df.nsmallest(5, "pnl_pct")
    print("5 worst trades this week:")
    print(worst[["ticker", "side", "pnl_pct", "top_driver", "regime"]])
```

**Das Muster, auf das du achten willst:**

Wenn in der PnL-by-top-driver-Tabelle ein Dimension-Name auftaucht, der konstant negativ ist → diese Dimension sabotiert deine Strategie. Beispiel:

```
PnL by top-driver:
                mean       sum      count
top_driver                              
news          -0.024    -0.43        18   ← News als Top-Driver = Verluste!
trend         +0.018    +0.19        11
volume        +0.031    +0.09         3
sentiment     -0.011    -0.05         5
```

**Interpretation:** Wenn News-Dimension dominant ist, verliert die Strategie im Schnitt 2.4 % pro Trade. Nicht ignorieren.

---

## 6. Detection: "Das Modell verhält sich anders als letzte Woche"

### 6.1 Attribution-Distribution-Shift

Wenn dein Modell heute stark andere Dimensionen gewichtet als letzte Woche, ist etwas **strukturell** passiert. Data-Drift, Concept-Drift, oder Bug.

```python
# src/assembled_core/attribution/drift_detection.py
import pandas as pd
import numpy as np
from scipy import stats


def detect_attribution_drift(
    recent_attrs: list,
    baseline_attrs: list,
    threshold_ks_p: float = 0.01,
) -> dict:
    """Kolmogorov-Smirnov-Test pro Dimension: hat sich die Distribution 
    der Contributions verschoben?
    
    Returns:
        dict dim_name → (ks_statistic, p_value, is_drift)
    """
    if not recent_attrs or not baseline_attrs:
        return {}
    
    # Extract contributions per dimension
    def to_frame(attrs):
        rows = []
        for a in attrs:
            rows.append(a.dimension_contributions)
        return pd.DataFrame(rows)
    
    recent_df = to_frame(recent_attrs)
    baseline_df = to_frame(baseline_attrs)
    
    results = {}
    for dim in recent_df.columns:
        if dim not in baseline_df.columns:
            continue
        
        stat, p = stats.ks_2samp(recent_df[dim], baseline_df[dim])
        results[dim] = {
            "ks_statistic": stat,
            "p_value": p,
            "is_drift": p < threshold_ks_p,
            "recent_mean": recent_df[dim].mean(),
            "baseline_mean": baseline_df[dim].mean(),
        }
    
    return results
```

**Anwendung:**

```python
# Täglich, automatisch
recent = store.load_all(start=today-1d, end=today)
baseline = store.load_all(start=today-30d, end=today-7d)

drifts = detect_attribution_drift(recent, baseline)
for dim, info in drifts.items():
    if info["is_drift"]:
        send_alert(f"Attribution drift in {dim}: p={info['p_value']:.4f}, "
                  f"recent_mean={info['recent_mean']:.3f}, "
                  f"baseline_mean={info['baseline_mean']:.3f}")
```

**Wenn Drift gefunden:**
1. Prüfe ob Bug oder echter Markt-Regime-Wechsel
2. Shadow-Mode neuer Gewichte einführen
3. Kein sofortiges Live-Deploy — Drift ist Hinweis, nicht Ursache

---

## 7. SHAP für News-Sub-Features (konkret)

Wenn du in Ebene A siehst, dass News die Top-Dimension ist, willst du in Ebene B schauen.

### 7.1 Der Training-Job für das News-Sub-Modell

```python
# scripts/train/news_sub_model.py
"""
Training des News-Sub-Modells (Random-Forest) mit den 6 Sub-Features.

Der Sub-Modell ist, was Pandera-validierte News-Features zu einem 
einzelnen News-Score zusammenführt. SHAP kann das dann erklären.
"""
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
import pickle


def train_news_sub_model():
    # Training-Data: historische News-Features + Target (forward 5-day return)
    df = pd.read_parquet("data/news_training_2024_2025.parquet")
    
    X = df[[
        "sentiment_score",
        "news_volume_spike",
        "source_quality_weight",
        "headline_uncertainty",
        "topic_cluster_signal",
        "cross_source_corroboration",
    ]]
    y = df["fwd_5d_return"]
    
    # Time-Series-CV für Hyperparameter
    tscv = TimeSeriesSplit(n_splits=5)
    
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,           # Wichtig: begrenzte Tiefe für Interpretability
        min_samples_leaf=50,
        random_state=42,
    )
    
    # Train auf full data
    model.fit(X, y)
    
    # Background-Sample (1000 Rows für SHAP)
    background = X.sample(1000, random_state=42)
    
    # Speichern
    pickle.dump(model, open("models/news_sub_model_v3.pkl", "wb"))
    background.to_parquet("models/news_background_sample.parquet")
    
    print(f"Trained. OOB R²: {model.oob_score_ if hasattr(model, 'oob_score_') else 'N/A'}")
```

### 7.2 Die SHAP-Visualisierung im Dashboard

```python
# Ergänzung zu attribution_app.py
import shap
import matplotlib.pyplot as plt
from assembled_core.news.explainer import NewsDimensionExplainer


def render_news_shap_section(st, news_features: dict):
    """SHAP-Waterfall für News-Sub-Features."""
    st.header("News-Sub-Features SHAP")
    
    explainer = NewsDimensionExplainer(model_dir="models/")
    shap_values = explainer.explain(news_features)
    
    # Waterfall-Plot mit matplotlib (SHAP's native)
    fig, ax = plt.subplots(figsize=(10, 5))
    shap.plots.waterfall(
        shap.Explanation(
            values=np.array(list(shap_values.values())),
            base_values=explainer.explainer.explainer.expected_value,
            data=np.array(list(news_features.values())),
            feature_names=list(shap_values.keys()),
        ),
        max_display=10,
        show=False,
    )
    st.pyplot(fig)
    plt.close(fig)
```

---

## 8. Umsetzungs-Checkliste

**Phase 1 — Attribution-Infrastructure (Woche 1):**
- [ ] `CompositeAttribution` Schema
- [ ] `AttributionStore` (SQLite)
- [ ] Integration in Trading-Cycle
- [ ] Unit-Tests

**Phase 2 — Streamlit-Dashboard (Woche 2):**
- [ ] Basis-App mit Filters
- [ ] Composite-Score-Time-Series
- [ ] Dimension-Contribution-Stacked-Bar
- [ ] Waterfall
- [ ] Top-Drivers-Table

**Phase 3 — SHAP für 1-2 Sub-Dimensionen (Woche 3):**
- [ ] News-Sub-Model trainieren
- [ ] `DimensionExplainer` mit TreeSHAP
- [ ] SHAP-Visualisierung im Dashboard

**Phase 4 — Post-Mortem & Review-Tools (Woche 4):**
- [ ] `post_mortem.py` CLI
- [ ] `weekly_review.py` Automation
- [ ] Attribution-Drift-Detection

**Phase 5 — Monitoring (Woche 5):**
- [ ] Drift-Alerts via Telegram
- [ ] Attribution-Dashboard-Deployment auf Hetzner

**Gesamt:** 4-5 Wochen bei 10-15 h/Woche.

---

## 9. Quellen

**SHAP:**
- [SHAP Documentation](https://shap.readthedocs.io/) und [GitHub](https://github.com/shap/shap)
- Lundberg & Lee (2017): Ursprungspaper "A Unified Approach to Interpreting Model Predictions"
- Christoph Molnar (Interpretable ML Book): [SHAP Chapter](https://christophm.github.io/interpretable-ml-book/shap.html)
- DataCamp (2023): [Introduction to SHAP Values](https://www.datacamp.com/tutorial/introduction-to-shap-values-machine-learning-interpretability)
- Chathuraishara (Dez 2025): [From What to Why: SHAP Feature Importance](https://medium.com/@chathuraishara63/from-what-to-why-feature-importance-analysis-using-shap-f84a060d3fad)
- SciPy Proceedings 2025: [Explaining ML predictions with SHAP](https://proceedings.scipy.org/articles/mhum9729)

**SHAP in Finance (mit kritischer Reflexion):**
- Journal of Operational Research Society (April 2025): [On ML Explainability in Banking: The Case of SHAP](https://www.tandfonline.com/doi/full/10.1080/01605682.2025.2485263) — SHAP ist **nicht portfolio-invariant**
- GeeksforGeeks (Juli 2025): [SHAP Comprehensive Guide](https://www.geeksforgeeks.org/machine-learning/shap-a-comprehensive-guide-to-shapley-additive-explanations/)

**Streamlit + SHAP:**
- Sundar Krishnan (TDS): [Real-time Model Interpretability API using SHAP, Streamlit and Docker](https://medium.com/data-science/real-time-model-interpretability-api-using-shap-streamlit-and-docker-e664d9797a9a)
- Taylor Amarel (Juni 2025): [Building Interactive XAI Dashboards with SHAP and Gradio](https://tayloramarel.com/2025/06/building-interactive-explainable-ai-dashboards-with-shap-and-gradio/)
- Streamlit Forum: [Display SHAP diagrams with Streamlit](https://discuss.streamlit.io/t/display-shap-diagrams-with-streamlit/1029)

**Trading-Dashboards:**
- Jaydeep Patel: [Algo Trading Dashboard using Streamlit](https://jaydeep4mgcet.medium.com/algo-trading-dashboard-using-python-and-streamlit-live-index-prices-current-positions-and-payoff-f44173a5b6d7)
- Artefact Blog: [Streamlit Prophet for Time-Series Forecasting](https://www.artefact.com/blog/visual-time-series-forecasting-with-streamlit-prophet/)

---

## 10. Ehrliche Einschätzung

**Was dieses Playbook dir gibt:**
- Ein-Klick-Erklärung **jeder** Trading-Entscheidung
- Post-Mortem in Minuten statt Stunden
- Früherkennung von Model-Drift
- Konkrete Datenbasis für "welche Dimension sabotiert deine Strategie?"

**Was es dir nicht gibt:**
- **Kausalität.** SHAP zeigt Korrelation zwischen Feature und Output. Nicht ob das Feature das Output **verursacht**.
- **Ground-Truth.** Wenn das Trainings-Label falsch ist (z.B. fwd-return als Target ohne Kosten-Adjustment), optimiert SHAP für die falsche Zielfunktion. Garbage-in, garbage-out.
- **Portfolio-Invarianz.** Der Tandfonline-Paper 2025 zeigt: SHAP-Values verändern sich wenn sich das Portfolio ändert. Für Einzel-Signal-Attribution ist das ok, für Portfolio-Level-Analysen nicht ohne Adjustment.

**Die drei Sachen, die du nicht auslassen darfst:**
1. **Attribution-Store bei jedem Signal**, nicht nur wenn Order entsteht. Du willst auch "warum KEIN Signal?" beantworten können.
2. **Wöchentlicher Review** mit PnL-by-Driver. Das ist der schärfste Debug-Mechanismus für Strategie-Probleme.
3. **Drift-Detection** mit KS-Test auf Contribution-Distribution. Früher als jeder PnL-Alarm.

**Der psychologische Aspekt:** Ein Attribution-Dashboard **zwingt dich zur Ehrlichkeit** über dein System. Ohne es kannst du dir einreden, der letzte Verlust sei Pech. Mit es siehst du, dass die News-Dimension seit 3 Wochen schlecht performed — aber du hast sie weiter benutzt. Das ist nicht angenehm, aber es ist der einzige Weg, langfristig besser zu werden.
