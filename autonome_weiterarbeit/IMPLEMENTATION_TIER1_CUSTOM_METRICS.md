# Tier 1 #1: Custom Trading-Metrics — konkreter Implementierungsplan

**Datum:** 2026-04-27
**Repo-Stand:** `/home/claude/Assembled-Trading-AI-fresh/` @ commit 3ea173e
**Aufwand:** 10-16h verteilt auf 4 Schritte
**Ziel:** Volle Trading-Observability ohne Architektur-Bruch

---

## Befund: was du heute hast

### `ops/metrics_exporter.py` (177 LOC)

**Bewusste Architektur-Entscheidung** in deinem Code:
- Nutzt **bewusst nicht** `prometheus_client` (Kommentar Zeile 12: "no mandatory dependency on `prometheus_client` — the text format is small enough to emit by hand")
- Stateless: jeder Call erwartet einen vollen Snapshot `{metric_name: value}`
- Schreibt Prometheus-Text-Format zu Datei (`output/metrics/assembled.prom`)
- Optional: Push-Gateway via HTTP

**Diese Architektur ist gut.** Sie hat zwei wichtige Properties:
1. Keine Process-State zwischen Runs (Container-friendly)
2. Geringe Dependencies (du musst kein `prometheus_client` mitschleppen)

### Aktueller Aufruf-Punkt: `trading_cycle_v2.py:2092-2110`

```python
# Phase 11: KPI export (Prometheus)
kpi_metrics: dict[str, float] = {
    "assembled_orders_generated_total": float(len(result.orders_filtered)),
    "assembled_targets_count": float(len(result.target_positions)),
    "assembled_signals_count": float(len(result.signals)),
}
tb_meta = result.meta.get("turnover_budget") or {}
if "estimated_turnover" in tb_meta and tb_meta["estimated_turnover"] != float("inf"):
    kpi_metrics["assembled_turnover_estimated"] = float(tb_meta["estimated_turnover"])
vt_meta = result.meta.get("vol_targeting") or {}
if "realized_vol" in vt_meta:
    kpi_metrics["assembled_realized_vol"] = float(vt_meta["realized_vol"])
export_metrics(kpi_metrics, labels={"strategy": ctx.strategy_name or "unknown", "mode": ctx.mode}, ...)
```

**Was hier fehlt:**
- Slippage (du hast `risk/slippage.py`, `execution/transaction_costs.py` — wird nicht exportiert)
- Drawdown (du hast `qa/walk_forward.py` mit max_dd-Berechnungen — wird nicht exportiert)
- Order-Rejections nach Reason (du hast `execution/pre_trade_checks.py` (1261 LOC!) — wird nicht exportiert)
- Drift-PSI (du hast `ops/drift_monitor.py` + `qa/drift_detection.py` — wird nicht exportiert)
- Kill-Switch-Status (du hast `risk/state_machine.py` (595 LOC) — wird nicht exportiert)
- Per-Symbol-Latencies / Stale-Data-Ages

---

## Plan: 4 Schritte, jeder einzeln deploybar

### Schritt 1: Erweitere `metrics_exporter.py` um Histogram-Support (~2-3h)

Dein aktueller `export_metrics(dict[str, float])` reicht für Gauges und Counters, aber nicht für **Histograms** (Slippage-Verteilungen). Erweitere die API minimal-invasiv:

```python
# ops/metrics_exporter.py — Erweiterung

from dataclasses import dataclass

@dataclass
class HistogramSnapshot:
    """Pre-computed histogram bucket counts + sum + count."""
    buckets: dict[float, int]   # {upper_bound_or_inf: count}
    sum: float
    count: int

def render_prometheus_text(
    metrics: dict[str, float | int],
    *,
    histograms: dict[str, HistogramSnapshot] | None = None,  # NEU
    labels: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> str:
    """..."""
    # ... existierender Code ...
    
    if histograms:
        for hist_name, snap in histograms.items():
            if not _METRIC_NAME_RE.match(hist_name):
                continue
            lines.append(f"# TYPE {hist_name} histogram")
            for upper, count in sorted(snap.buckets.items()):
                bucket_label = "+Inf" if upper == float("inf") else f"{upper}"
                bucket_labels = {**(labels or {}), "le": bucket_label}
                lines.append(
                    f"{hist_name}_bucket{_format_labels(bucket_labels)} {count}"
                )
            lines.append(f"{hist_name}_sum{label_block} {snap.sum}")
            lines.append(f"{hist_name}_count{label_block} {snap.count}")
    
    return "\n".join(lines) + "\n"
```

**Plus:** kleine Helper-Funktion für Slippage-Histogramme:

```python
def slippage_histogram(slippage_bps: list[float]) -> HistogramSnapshot:
    """Build histogram snapshot from a list of slippage observations."""
    buckets = [-200, -50, -20, -10, -5, -1, 0, 1, 5, 10, 20, 50, 200, float("inf")]
    counts = {b: 0 for b in buckets}
    for s in slippage_bps:
        for b in buckets:
            if s <= b:
                counts[b] += 1
                break
    return HistogramSnapshot(
        buckets=counts,
        sum=sum(slippage_bps),
        count=len(slippage_bps),
    )
```

**Test (passt zu deinem `tests/`-Pattern):**
```python
# tests/ops/test_metrics_exporter_histogram.py
def test_histogram_renders_correctly():
    hist = slippage_histogram([-15, -3, 0, 1, 8, 25])
    text = render_prometheus_text(
        metrics={},
        histograms={"trading_slippage_bps": hist},
        labels={"strategy": "test"},
    )
    assert "trading_slippage_bps_bucket" in text
    assert "trading_slippage_bps_sum" in text
    assert "trading_slippage_bps_count" in text
```

**Aufwand:** 2-3h inkl. Tests.

---

### Schritt 2: Sammle Slippage in `execution/` & exportiere (~3-4h)

Deine `execution/transaction_costs.py` ist 1008 LOC groß und hat sicher schon Slippage-Berechnung. Ergänze einen **Collector**, der die Slippage pro Trade in einer Run-lokalen Liste sammelt:

```python
# ops/slippage_collector.py — neue Datei

from collections import defaultdict
import threading

class SlippageCollector:
    """Run-local accumulator for slippage observations.
    
    Flushed at end of trading cycle into metrics export.
    """
    
    def __init__(self):
        self._observations: dict[tuple[str, str, str], list[float]] = defaultdict(list)
        self._lock = threading.Lock()
    
    def record(
        self,
        strategy: str,
        symbol: str,
        side: str,  # 'buy' / 'sell'
        decision_price: float,
        executed_price: float,
    ):
        """Record one fill's slippage."""
        if decision_price <= 0:
            return  # invalid input
        slippage_bps = (executed_price - decision_price) / decision_price * 10000.0
        if side == "sell":
            slippage_bps = -slippage_bps  # signed: negative = adverse
        
        with self._lock:
            self._observations[(strategy, symbol, side)].append(slippage_bps)
    
    def snapshot(self) -> dict[tuple[str, str, str], list[float]]:
        """Return current observations (for export)."""
        with self._lock:
            return {k: list(v) for k, v in self._observations.items()}
    
    def reset(self):
        with self._lock:
            self._observations.clear()


# Module-level singleton (your codebase uses this pattern in heartbeat.py)
_collector: SlippageCollector | None = None

def get_collector() -> SlippageCollector:
    global _collector
    if _collector is None:
        _collector = SlippageCollector()
    return _collector
```

**Integration in deine bestehenden Execution-Module:**

In `execution/unified_paper_engine.py` (2710 LOC) gibt es definitiv eine Stelle, wo Fills verarbeitet werden. Dort einbauen:

```python
# In unified_paper_engine.py, irgendwo bei fill_order(...)

from src.assembled_core.ops.slippage_collector import get_collector

def _on_fill(self, order, fill_price):
    decision_price = order.decision_price  # mid at signal time
    get_collector().record(
        strategy=self.strategy_name,
        symbol=order.symbol,
        side=order.side,
        decision_price=decision_price,
        executed_price=fill_price,
    )
    # ... existierende fill-handling logik
```

**Export in `trading_cycle_v2.py` Phase 11:**

```python
# trading_cycle_v2.py Phase 11 — erweitert

from src.assembled_core.ops.slippage_collector import get_collector
from src.assembled_core.ops.metrics_exporter import slippage_histogram

# After existing kpi_metrics setup:
slippage_observations = get_collector().snapshot()
slippage_histograms = {}
for (strategy, symbol, side), obs in slippage_observations.items():
    if obs:
        # One histogram per strategy (aggregated over symbols)
        # Or finer-grained if you want per-symbol histograms
        key = f"trading_slippage_bps_{strategy}"
        if key not in slippage_histograms:
            slippage_histograms[key] = []
        slippage_histograms[key].extend(obs)

slippage_histograms_rendered = {
    name: slippage_histogram(obs)
    for name, obs in slippage_histograms.items()
}

export_metrics(
    kpi_metrics,
    histograms=slippage_histograms_rendered,  # Schritt 1 erweitert die API
    labels={"strategy": ctx.strategy_name or "unknown", "mode": ctx.mode},
    path=metrics_dir / "assembled.prom" if metrics_dir else None,
)

get_collector().reset()  # clean for next run
```

**Tests:**
```python
# tests/ops/test_slippage_collector.py
def test_records_buy_slippage_correctly():
    coll = SlippageCollector()
    coll.record("strat_a", "AAPL", "buy", decision_price=100.0, executed_price=100.05)
    snap = coll.snapshot()
    assert snap[("strat_a", "AAPL", "buy")] == [5.0]  # 5 bps adverse

def test_signs_sell_slippage_inverted():
    coll = SlippageCollector()
    coll.record("strat_a", "AAPL", "sell", decision_price=100.0, executed_price=99.95)
    snap = coll.snapshot()
    # Sold at 99.95 vs 100 -> adverse -> stored as +5 bps
    assert snap[("strat_a", "AAPL", "sell")] == [5.0]
```

**Aufwand:** 3-4h.

---

### Schritt 3: Order-Rejections, Kill-Switch-State, Drift-PSI (~3-4h)

Diese sind alle "Counter" und "Gauge" — keine Histogramm-Erweiterung nötig.

**Order-Rejections nach Reason:**

`execution/pre_trade_checks.py` (1261 LOC) blockt Orders aus verschiedenen Gründen. Ergänze einen Counter-Collector:

```python
# ops/rejection_collector.py — neue Datei

from collections import defaultdict
import threading

class RejectionCollector:
    """Counts order rejections by reason."""
    
    def __init__(self):
        self._counts: dict[tuple[str, str], int] = defaultdict(int)
        self._lock = threading.Lock()
    
    def record(self, strategy: str, reason: str):
        """reason: 'pdt', 'fat_finger', 'kill_switch', 'no_liquidity', etc."""
        with self._lock:
            self._counts[(strategy, reason)] += 1
    
    def snapshot(self) -> dict[tuple[str, str], int]:
        with self._lock:
            return dict(self._counts)


_rejection_collector: RejectionCollector | None = None

def get_rejection_collector() -> RejectionCollector:
    global _rejection_collector
    if _rejection_collector is None:
        _rejection_collector = RejectionCollector()
    return _rejection_collector
```

**Integration in `pre_trade_checks.py`:** an jeder Stelle wo eine Order abgelehnt wird, eine Zeile dazu:

```python
# In pre_trade_checks.py
from src.assembled_core.ops.rejection_collector import get_rejection_collector

# An jedem return False / raise / discard:
get_rejection_collector().record(strategy=ctx.strategy_name, reason="pdt_violation")
```

**Kill-Switch-Status aus `risk/state_machine.py`:**

```python
# trading_cycle_v2.py Phase 11 — weitere Erweiterung

from src.assembled_core.risk.state_machine import get_current_state

current_state = get_current_state()  # was auch immer du heute nutzt
kpi_metrics["trading_kill_switch_active"] = float(
    1 if current_state.is_kill_switch_active else 0
)
kpi_metrics["trading_state_machine_state"] = float(current_state.state_id)
```

**Drift-PSI aus `ops/drift_monitor.py`:**

Wenn dein Drift-Monitor einen aktuellen PSI-Wert pro Feature liefert (sehr wahrscheinlich), dann:

```python
# trading_cycle_v2.py Phase 11

from src.assembled_core.ops.drift_monitor import get_latest_psi_snapshot

psi_snapshot = get_latest_psi_snapshot()  # {feature_name: psi_value}
for feature, psi in psi_snapshot.items():
    # Sanitize feature name for Prometheus
    safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", feature)
    kpi_metrics[f"trading_drift_psi_{safe_name}"] = float(psi)
```

**Aufwand:** 3-4h.

---

### Schritt 4: Grafana-Dashboards als JSON in deinem Repo (~3-5h)

Du brauchst kein eigenes Hosting — Grafana hat einen **Dashboard-as-Code**-Workflow. Speichere die Dashboards als JSON in deinem Repo:

```
ops/
├── grafana/
│   ├── dashboards/
│   │   ├── trading_overview.json       # Strategie-Health auf einen Blick
│   │   ├── slippage_analysis.json      # Slippage-Histogramme pro Strategie/Symbol
│   │   ├── order_flow.json             # Submitted/Filled/Rejected mit Reasons
│   │   ├── drift_monitor.json          # PSI pro Feature über Zeit
│   │   └── README.md                   # How to import
│   └── prometheus_config_example.yml
```

**Dashboard 1: Trading Overview** (das wichtigste — du würdest es jeden Morgen anschauen)

```json
{
  "title": "Trading Overview",
  "panels": [
    {
      "title": "Active Strategies",
      "type": "stat",
      "targets": [{"expr": "count(count by (strategy) (assembled_signals_count))"}]
    },
    {
      "title": "Kill Switch Status",
      "type": "stat",
      "targets": [{"expr": "trading_kill_switch_active"}],
      "thresholds": [{"value": 0.5, "color": "red"}]
    },
    {
      "title": "Orders Generated Today",
      "type": "graph",
      "targets": [
        {"expr": "rate(assembled_orders_generated_total[1h])", "legendFormat": "{{strategy}}"}
      ]
    },
    {
      "title": "Realized Volatility",
      "type": "graph",
      "targets": [
        {"expr": "assembled_realized_vol", "legendFormat": "{{strategy}}"}
      ]
    },
    {
      "title": "Drift Alerts (PSI > 0.2)",
      "type": "alertlist",
      "targets": [
        {"expr": "trading_drift_psi_{feature_name=~\".*\"} > 0.2"}
      ]
    }
  ]
}
```

**Dashboard 2: Slippage Analysis** (für Execution-Quality-Tracking)

Die zwei wichtigsten Panels:
- **Heatmap**: Slippage-Verteilung pro Strategie über Zeit (zeigt wenn eine Strategie plötzlich schlechter ausgeführt wird)
- **Histogram**: aktuelle Slippage-Verteilung pro Strategie (von gestern oder so)

**Dashboard 3: Order Flow** (Reject-Reasons)

```promql
# Top 5 rejection reasons last hour
topk(5, sum by (reason) (rate(trading_orders_rejected_total[1h])))
```

**Aufwand:** 3-5h. Die JSON-Files sind groß, aber 80% kannst du aus Grafana exportieren nachdem du eine erste Visualisierung gebaut hast.

---

## Reihenfolge & Validierung

| Tag | Schritt | Validierung |
|---|---|---|
| 1 | Schritt 1 (Histogram-API) | Test rendering, manueller `cat output/metrics/assembled.prom` |
| 2 | Schritt 2 (Slippage-Collector + Export) | Mit `--paper-mode`-Run testen, Slippage-Histogram im Output sehen |
| 3 | Schritt 3 (Rejections + Kill-Switch + Drift) | Pre-trade-check trigger, Counter sollte hochzählen |
| 4 | Schritt 4 (Grafana) | Lokal Grafana mit Docker, Dashboards importieren, Daten sehen |

**Risiko:** Niedrig. Alle Änderungen sind additiv. Wenn `metrics_exporter` mal einen Fehler wirft, fällt nur die Metric raus — der Trade-Cycle läuft weiter (du hast `try/except` in Phase 11).

**Rückwärtskompatibilität:** Ja. `render_prometheus_text(metrics, labels=...)` bleibt 1:1 funktional. Histogram-Param ist optional.

---

## Was du nach diesen 10-16h hast

1. **Slippage-Tracking pro Strategie/Symbol** mit Verteilung — wenn du morgen einen Slippage-Spike hast, siehst du es sofort
2. **Order-Rejection-Stats nach Reason** — wenn dein Pre-Trade-Check plötzlich 80% deiner Orders blockt, siehst du **warum**
3. **Live Kill-Switch-Status** in Grafana — eine Lampe die rot wird wenn was schief läuft
4. **Drift-PSI pro Feature über Zeit** — der Pfad eines schleichenden Modell-Niedergangs ist sichtbar
5. **Grafana-Dashboards als Code** im Repo — reproduzierbar deploybar

**Visibility-Multiplier**: jede andere Empfehlung in v3 (Brinson-Attribution, toraniko, VPIN, ...) wird durch diese Observability **viel wertvoller**, weil du ihre Outputs auch siehst.

---

## Wenn du das ausführen willst

Sag einfach **"go"** oder **"Schritt 1 zuerst"** und ich:
1. Lese den exakten Code in `ops/metrics_exporter.py`
2. Generiere den Diff für Schritt 1 (mit Tests)
3. Du committest, ich gehe zu Schritt 2

Oder wenn du erst Tier 1 #2 (Brinson-Attribution) oder #3 (MLflow) machen willst — auch OK, sag Bescheid.
