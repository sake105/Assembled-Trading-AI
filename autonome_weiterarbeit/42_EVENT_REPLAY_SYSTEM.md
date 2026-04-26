# 42 — Event-Replay-System

**Zweck:** Du willst beantworten können: "Wenn ich den Code jetzt auf die Session von letztem Dienstag anwende, was passiert?" Nicht "was ist passiert" (das steht im Log), sondern "was **würde** jetzt mit demselben Input passieren". Diese Fähigkeit ist der Unterschied zwischen "ich habe meinen Bug gefixt" und "ich **weiß**, dass ich meinen Bug gefixt habe".

**Scope:** Rang 9 aus der Gap-Analyse. Verknüpft mit `35_GOLDEN_EQUITY_SCENARIO_TESTS.md` (Characterization-Tests) und `10_FREE_DATEN.md` (Data-Layer). Hauptnutzen: Debug, Regression-Test, Strategy-Development.

**Kern-Idee:** Alle externen Inputs (Market-Data-Ticks, News-Events, Fill-Confirmations) werden beim Live-Betrieb als Event-Stream in einer append-only Datenbank gespeichert. Das System kann diesen Stream zu jedem Zeitpunkt "replayen" — dieselben Inputs, derselbe Code (oder neuer Code), deterministisches Ergebnis.

---

## 0. Warum das wichtig ist

### Das Szenario, das Replay löst

**Dienstag 14:23 Uhr:** Dein System macht eine Entscheidung, die dich überrascht. Buy AAPL bei 182.40 — aber du dachtest, der Filter sollte das unterdrücken. Du investigierst, findest einen Bug in `src/filters/volume_filter.py`. Du fixt den Bug.

**Mittwoch Morgen:** Wie verifizierst du, dass der Fix funktioniert?

Ohne Event-Replay:
- Du schreibst einen Unit-Test mit handgestrickten Input-Daten → aber der testet nur dein künstliches Szenario, nicht das echte
- Du wartest auf das nächste Live-Auftreten → kann Wochen dauern
- Du backtestest → aber Backtest-Daten sind nicht dieselben wie Live-Daten (andere Feeds, andere Timings)

Mit Event-Replay:
- Du lädst die Event-Session von Dienstag 14:20-14:25
- Du rennst den gefixten Code gegen diesen Event-Stream
- Die Signal-Entscheidung ist jetzt "keine Order" statt "Buy AAPL" → Fix verifiziert
- Diese Session wird zu einem permanenten Regression-Test

### Der zweite Nutzen: "was wäre gewesen"

Du entwickelst eine neue Feature-Dimension (z.B. "Options-Flow"). Du willst wissen: wäre mein Live-System in den letzten 4 Wochen anders, wenn dieses Feature bereits aktiv gewesen wäre?

- Replay der letzten 4 Wochen mit der neuen Feature-Dimension
- Vergleich der generierten Signals mit den tatsächlichen Trades
- Quantitative Antwort auf "Feature X hätte Y Trades verhindert, Z Trades neu getriggert, Netto-PnL-Delta W"

Das ist Backtest-Qualität auf **realen Live-Daten** statt rekonstruierten Historical-Daten. Unterschied: keine Data-Quality-Abweichung zwischen Research und Live.

### Der dritte Nutzen: Forensik

**Incident:** Eine Order wurde submitted, die nicht submitted werden sollte. Der Compliance-Teil deines Systems (oder du selbst in 5 Jahren, wenn du verklagt wirst) muss rekonstruieren: was hat das System um 13:47:23 gesehen, und warum hat es so entschieden?

Ohne Event-Sourcing: Log-Files, die vielleicht ausreichen, vielleicht nicht.
Mit Event-Sourcing: exakte Wiederherstellung des Zustands inklusive aller Inputs zum Entscheidungszeitpunkt.

---

## 1. Die Architektur: Event-Sourcing für Einzel-Quants

### 1.1 Was ist Event-Sourcing

**Klassisches State-Management:** Du speicherst den **aktuellen Zustand** (Portfolio: AAPL 100 shares @ 182.40, Cash $8,200, ...). Jede Änderung überschreibt den alten Zustand.

**Event-Sourcing:** Du speicherst **alle Events** (14:23:01 OrderFilled AAPL 100 @ 182.40, 14:25:00 MarketTick AAPL 182.35, ...). Der aktuelle Zustand wird durch **Anwendung aller Events** berechnet.

**Implikation:** Der aktuelle Zustand ist **eine abgeleitete Größe**, nicht die primäre. Immer rekonstruierbar. Eine Bug im State-Management kann nicht zu Datenverlust führen — die Events bleiben unverändert.

### 1.2 Was NICHT Event-Sourcing ist

Nicht jeder `log.info("Order submitted")`-Eintrag ist ein Event. Events sind **domain-events** — strukturierte, typisierte Fakten mit klarer Semantik:

| Log-Eintrag | Event |
|---|---|
| `"Starting cycle"` | Nein — ist Meta-Info |
| `"Got 152 news items"` | Nein — ist Summary |
| `"NewsReceived(ticker=AAPL, source=Reuters, headline='...', ts=14:23:01)"` | **Ja** |
| `"MarketTickReceived(AAPL, 182.35, volume=500, ts=14:25:00)"` | **Ja** |
| `"SignalGenerated(AAPL, BUY, composite_score=0.72, ts=14:25:03)"` | **Ja** |
| `"OrderSubmitted(AAPL, BUY, 100, ts=14:25:04)"` | **Ja** |
| `"OrderFilled(order_id=abc, AAPL, 100 @ 182.40, ts=14:25:06)"` | **Ja** |

### 1.3 Die zwei Event-Kategorien

Für Replay musst du zwei Kategorien unterscheiden:

**Input-Events (von außen, müssen gespeichert werden):**
- `MarketTickReceived` — aus Data-Feed
- `NewsReceived` — aus News-Feed
- `OrderFilled` / `OrderRejected` — von Broker
- `AccountUpdate` — Equity-Änderungen
- `ClockTick` — Zeitfortschritt (für deterministisches Replay wichtig)

**Output-Events (vom System generiert, werden abgeleitet):**
- `SignalGenerated`
- `FeatureComputed`
- `OrderSubmitted`
- `AttributionComputed`

**Warum die Unterscheidung?** Beim Replay werden **Input-Events wieder abgespielt**, und das System generiert **neue Output-Events**. Wenn der Code sich nicht geändert hat, sind die neuen Output-Events identisch mit den alten. Wenn der Code sich geändert hat, divergieren die Outputs — und das ist genau das, was du messen willst.

---

## 2. Der Tool-Stack

### 2.1 Die Entscheidung

**Optionen:**

| Tool | Stärke | Für dich? |
|---|---|---|
| **eventsourcing** (Python-Lib) | Python-native, gutes Framework | Möglich, aber overkill |
| **NautilusTrader** | Production-grade, nanosecond-deterministic | Ja — aber Lock-In in deren Framework |
| **Apache Kafka** | Industrie-Standard für Event-Streams | Overkill für Einzel-Nutzer |
| **SQLite append-only** | Simpel, keine Infra-Kosten | **Ja — der Einstieg** |
| **Postgres mit event_store Tabelle** | Mehr als SQLite, aber auch mehr Ops | Später |

**Empfehlung für Hans:**
- **Start:** SQLite mit append-only Events-Tabelle. Pragmatisch, läuft sofort.
- **Upgrade-Pfad:** Postgres wenn Volume > 1 Mio Events/Tag oder Multi-Prozess-Zugriff nötig.
- **Wenn du ernsthaft NautilusTrader in Erwägung ziehst:** Separate Architektur-Entscheidung. Für jetzt eigenständig bleiben.

### 2.2 Installation

```bash
# Minimaler Stack
uv pip install pydantic==2.9.2          # Event-Schemas
# SQLite ist built-in

# Optional für größere Installs
uv pip install psycopg2-binary==2.9.10  # Postgres-Adapter
```

---

## 3. Der Event-Store

### 3.1 Das Schema

```python
# src/assembled_core/events/schema.py
from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel, Field
from enum import Enum
from uuid import UUID, uuid4


class EventSource(str, Enum):
    MARKET_DATA = "market_data"
    NEWS = "news"
    BROKER = "broker"
    CLOCK = "clock"
    ACCOUNT = "account"
    SYSTEM = "system"


class BaseEvent(BaseModel):
    """Basis für alle Events. Jedes Event hat Identity, Zeit, Source."""
    event_id: UUID = Field(default_factory=uuid4)
    session_id: str            # z.B. "live_20260424"
    sequence_no: int           # monoton steigend in der Session
    timestamp: datetime        # wann das Event in der echten Welt passierte
    received_at: datetime      # wann unser System es aufgezeichnet hat
    source: EventSource
    event_type: str            # Discriminator für Deserialisierung
    
    class Config:
        frozen = True          # Immutable


class MarketTickReceived(BaseEvent):
    event_type: Literal["MarketTickReceived"] = "MarketTickReceived"
    source: Literal[EventSource.MARKET_DATA] = EventSource.MARKET_DATA
    ticker: str
    price: float
    volume: int
    bid: Optional[float] = None
    ask: Optional[float] = None


class NewsReceived(BaseEvent):
    event_type: Literal["NewsReceived"] = "NewsReceived"
    source: Literal[EventSource.NEWS] = EventSource.NEWS
    news_source: str               # z.B. "Reuters"
    headline: str
    body: Optional[str] = None
    tickers: list[str]             # affected tickers
    external_id: Optional[str] = None


class OrderFilled(BaseEvent):
    event_type: Literal["OrderFilled"] = "OrderFilled"
    source: Literal[EventSource.BROKER] = EventSource.BROKER
    order_id: str
    ticker: str
    side: Literal["buy", "sell"]
    quantity: int
    price: float
    commission: float = 0.0


class OrderRejected(BaseEvent):
    event_type: Literal["OrderRejected"] = "OrderRejected"
    source: Literal[EventSource.BROKER] = EventSource.BROKER
    order_id: str
    ticker: str
    reason: str
    http_status: Optional[int] = None


class ClockTick(BaseEvent):
    """Explizites Zeit-Event. Wichtig für Replay-Determinismus."""
    event_type: Literal["ClockTick"] = "ClockTick"
    source: Literal[EventSource.CLOCK] = EventSource.CLOCK
    # kein zusätzliches Feld, timestamp ist die ganze Information


# Union-Type für Discriminated-Union
from typing import Union
Event = Union[MarketTickReceived, NewsReceived, OrderFilled, OrderRejected, ClockTick]
```

### 3.2 Der Store

```python
# src/assembled_core/events/store.py
import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

from .schema import BaseEvent, Event


class EventStore:
    """Append-only Event-Store mit SQLite-Backend.
    
    Sessions gruppieren Events. Reihenfolge garantiert durch sequence_no.
    """
    
    def __init__(self, db_path: str = "data/events.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
    
    def _init_schema(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    sequence_no INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    received_at TEXT NOT NULL,
                    source TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    UNIQUE(session_id, sequence_no)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_session ON events(session_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON events(timestamp)")
    
    def append(self, event: BaseEvent):
        """Append ein einzelnes Event. Append-only — niemals UPDATE oder DELETE."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO events 
                (event_id, session_id, sequence_no, timestamp, received_at, source, 
                 event_type, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(event.event_id),
                event.session_id,
                event.sequence_no,
                event.timestamp.isoformat(),
                event.received_at.isoformat(),
                event.source.value,
                event.event_type,
                event.model_dump_json(),
            ))
    
    def append_batch(self, events: list[BaseEvent]):
        """Batch-insert. Transaktional: entweder alle oder keine."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany("""
                INSERT INTO events 
                (event_id, session_id, sequence_no, timestamp, received_at, source, 
                 event_type, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [(
                str(e.event_id),
                e.session_id,
                e.sequence_no,
                e.timestamp.isoformat(),
                e.received_at.isoformat(),
                e.source.value,
                e.event_type,
                e.model_dump_json(),
            ) for e in events])
    
    def load_session(
        self, 
        session_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Iterator[Event]:
        """Iteriere Events einer Session in Original-Reihenfolge (sequence_no)."""
        query = "SELECT payload_json, event_type FROM events WHERE session_id = ?"
        params: list = [session_id]
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())
        
        query += " ORDER BY sequence_no ASC"
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            for payload_json, event_type in cursor:
                yield self._deserialize(payload_json, event_type)
    
    def _deserialize(self, payload_json: str, event_type: str) -> Event:
        data = json.loads(payload_json)
        # Discriminated-Union-Dispatch
        from .schema import (
            MarketTickReceived, NewsReceived, OrderFilled, 
            OrderRejected, ClockTick
        )
        type_map = {
            "MarketTickReceived": MarketTickReceived,
            "NewsReceived": NewsReceived,
            "OrderFilled": OrderFilled,
            "OrderRejected": OrderRejected,
            "ClockTick": ClockTick,
        }
        cls = type_map.get(event_type)
        if not cls:
            raise ValueError(f"Unknown event_type: {event_type}")
        return cls(**data)
    
    def session_stats(self, session_id: str) -> dict:
        """Summary einer Session."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total,
                    MIN(timestamp) as start,
                    MAX(timestamp) as end,
                    COUNT(DISTINCT event_type) as type_count
                FROM events WHERE session_id = ?
            """, (session_id,))
            row = cursor.fetchone()
            if row[0] == 0:
                return {"session_id": session_id, "total": 0}
            return {
                "session_id": session_id,
                "total_events": row[0],
                "start_time": row[1],
                "end_time": row[2],
                "event_types": row[3],
            }
```

### 3.3 Integration in Live-System

```python
# src/assembled_core/pipeline/live.py (Ausschnitt)
from datetime import datetime
from ..events.store import EventStore
from ..events.schema import MarketTickReceived, NewsReceived, OrderFilled

event_store = EventStore()
session_id = f"live_{datetime.utcnow():%Y%m%d}"
_sequence_counter = 0


def next_seq():
    global _sequence_counter
    _sequence_counter += 1
    return _sequence_counter


def on_market_tick(tick):
    """Handler für Alpaca's WebSocket-Ticks."""
    # 1. Event aufnehmen
    event = MarketTickReceived(
        session_id=session_id,
        sequence_no=next_seq(),
        timestamp=tick.timestamp,
        received_at=datetime.utcnow(),
        ticker=tick.ticker,
        price=tick.price,
        volume=tick.volume,
        bid=tick.bid,
        ask=tick.ask,
    )
    event_store.append(event)
    
    # 2. An Trading-Pipeline weitergeben
    pipeline.process_market_tick(tick)


def on_news(news_item):
    """Handler für News-Feed."""
    event = NewsReceived(
        session_id=session_id,
        sequence_no=next_seq(),
        timestamp=news_item.timestamp,
        received_at=datetime.utcnow(),
        news_source=news_item.source,
        headline=news_item.headline,
        body=news_item.body,
        tickers=news_item.tickers,
        external_id=news_item.external_id,
    )
    event_store.append(event)
    
    pipeline.process_news(news_item)
```

**Wichtig:** Events werden **bevor** die Pipeline sie verarbeitet gespeichert. Wenn die Pipeline crasht, der Event bleibt. Wenn du replayst, hast du den genauen State vor dem Crash.

---

## 4. Der Replayer

### 4.1 Die Kern-Abstraktion

```python
# src/assembled_core/events/replayer.py
from datetime import datetime
from typing import Callable, Optional
import logging

from .store import EventStore
from .schema import (
    Event, MarketTickReceived, NewsReceived, 
    OrderFilled, OrderRejected, ClockTick
)

logger = logging.getLogger(__name__)


class Replayer:
    """Spielt Event-Stream gegen einen Handler ab.
    
    Der Handler ist typischerweise die Trading-Pipeline, aber mit:
    - mocked Broker (keine echten Orders werden submitted)
    - mocked Clock (Zeit wird aus Events genommen)
    """
    
    def __init__(self, store: EventStore, handlers: dict[str, Callable]):
        self.store = store
        self.handlers = handlers  # event_type → Callable
    
    def replay_session(
        self,
        session_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        speed: str = "as_fast_as_possible",  # or "realtime"
    ):
        """Replayt alle Events einer Session in Original-Reihenfolge."""
        events_iter = self.store.load_session(session_id, start_time, end_time)
        
        event_count = 0
        for event in events_iter:
            handler = self.handlers.get(event.event_type)
            if handler is None:
                logger.warning(f"No handler for {event.event_type}, skipping")
                continue
            
            try:
                handler(event)
                event_count += 1
            except Exception as e:
                logger.error(f"Handler failed on event {event.event_id}: {e}")
                raise
        
        logger.info(f"Replayed {event_count} events from session {session_id}")
        return event_count
```

### 4.2 Die Mock-Clock

Beim Replay darf das System **nicht** `datetime.utcnow()` aufrufen — das würde "jetzt" statt "replay-time" liefern und Non-Determinismus einführen.

```python
# src/assembled_core/time/clock.py
from datetime import datetime
from typing import Protocol


class Clock(Protocol):
    def now(self) -> datetime: ...


class RealClock:
    """Für Live-Betrieb."""
    def now(self) -> datetime:
        return datetime.utcnow()


class ReplayClock:
    """Für Replay. Wird von ClockTick-Events gesteuert."""
    def __init__(self, start_time: datetime):
        self._current = start_time
    
    def now(self) -> datetime:
        return self._current
    
    def advance_to(self, timestamp: datetime):
        if timestamp < self._current:
            raise ValueError(f"Cannot go backwards: {timestamp} < {self._current}")
        self._current = timestamp
```

**Alle zeit-abhängigen Code-Stellen müssen den Clock nehmen:**

```python
# FALSCH:
now = datetime.utcnow()

# RICHTIG:
now = self.clock.now()
```

Das ist ein systematischer Refactor. Aber essentiell. Ohne saubere Clock-Abstraktion ist Replay nie wirklich deterministisch.

### 4.3 Der Mock-Broker

```python
# src/assembled_core/execution/mock_broker.py
"""Broker-Interface, der beim Replay NICHT echte Orders submitted,
sondern die passenden OrderFilled/OrderRejected-Events aus dem Stream konsumiert."""

from collections import deque
from typing import Optional


class ReplayBroker:
    """Der Broker beim Replay.
    
    Statt echte API-Calls zu machen, pickt er die OrderFilled/OrderRejected-
    Events aus dem Stream, die dem submitted Order entsprechen.
    """
    
    def __init__(self):
        self._pending_fills: deque = deque()
        self._submitted_orders: dict[str, dict] = {}
    
    def inject_broker_event(self, event):
        """Wird vom Replayer aufgerufen wenn ein Broker-Event im Stream kommt."""
        self._pending_fills.append(event)
    
    def submit_order(self, ticker: str, side: str, qty: int) -> str:
        """Wird von der Pipeline aufgerufen. Returned order_id."""
        # Suche das nächste passende Fill/Reject-Event
        for i, event in enumerate(self._pending_fills):
            if event.ticker == ticker and event.event_type in ("OrderFilled", "OrderRejected"):
                # Match — remove from queue
                matched = self._pending_fills[i]
                del self._pending_fills[i]
                
                self._submitted_orders[matched.order_id] = {
                    "ticker": ticker,
                    "side": side,
                    "qty": qty,
                    "status": matched.event_type,
                    "matched_event": matched,
                }
                return matched.order_id
        
        # Kein Match — der Code hat eine Order submitted, die im Live-Run nicht passiert ist
        # (z.B. weil der alte Code diese Order nicht submitted hat, aber der neue Code tut es)
        # → Order wird künstlich als "rejected-in-replay" gemarkt
        return "REPLAY_NO_MATCH"
```

**Der Casus Knacksus:** Im Replay muss jede Order-Submission gegen den bekannten Stream validiert werden. Wenn der **neue Code** eine Order submitted, die der **alte Code** nicht submitted hat, hast du ein Divergenz-Event — genau das, was du entdecken willst.

---

## 5. Der Replay-Workflow

### 5.1 Basis-Workflow

```python
# scripts/replay/replay_session.py
"""
Replayt eine Session mit dem aktuellen Code.
Vergleicht Output-Events mit dem Original-Lauf.

Usage:
    python -m scripts.replay.replay_session \
        --session live_20260422 \
        --output reports/replay_20260422.json
"""
import argparse
import json
from datetime import datetime

from assembled_core.events.store import EventStore
from assembled_core.events.replayer import Replayer
from assembled_core.time.clock import ReplayClock
from assembled_core.execution.mock_broker import ReplayBroker
from assembled_core.pipeline import TradingPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    
    store = EventStore()
    
    # Session-Info holen
    stats = store.session_stats(args.session)
    print(f"Replaying: {stats}")
    
    # Replay-Clock auf Session-Start setzen
    clock = ReplayClock(datetime.fromisoformat(stats["start_time"]))
    
    # Mock-Broker
    broker = ReplayBroker()
    
    # Trading-Pipeline mit Replay-Abhängigkeiten
    pipeline = TradingPipeline(
        clock=clock,
        broker=broker,
        # ... andere Deps (Data-Feeds sind durch Events ersetzt)
    )
    
    # Output-Events werden gesammelt für Vergleich
    replay_outputs: list = []
    
    def on_signal(signal):
        replay_outputs.append({
            "type": "signal",
            "timestamp": clock.now().isoformat(),
            "ticker": signal.ticker,
            "composite": signal.composite_score,
            "decision": signal.decision,
        })
    
    pipeline.on_signal_generated = on_signal
    
    # Handlers-Dispatch
    handlers = {
        "MarketTickReceived": lambda e: (clock.advance_to(e.timestamp), 
                                          pipeline.process_market_tick(e)),
        "NewsReceived": lambda e: (clock.advance_to(e.timestamp), 
                                    pipeline.process_news(e)),
        "OrderFilled": lambda e: broker.inject_broker_event(e),
        "OrderRejected": lambda e: broker.inject_broker_event(e),
        "ClockTick": lambda e: clock.advance_to(e.timestamp),
    }
    
    replayer = Replayer(store, handlers)
    replayer.replay_session(args.session)
    
    # Output schreiben
    with open(args.output, "w") as f:
        json.dump({
            "session_id": args.session,
            "replayed_at": datetime.utcnow().isoformat(),
            "replay_outputs": replay_outputs,
        }, f, indent=2)
    
    print(f"Replay complete. {len(replay_outputs)} output events. Output: {args.output}")


if __name__ == "__main__":
    main()
```

### 5.2 Der Vergleich alt vs. neu

```python
# scripts/replay/compare_replays.py
"""
Vergleicht zwei Replay-Outputs (z.B. alter Code vs. neuer Code).
Zeigt Divergenzen.
"""
import json
from pathlib import Path


def load_outputs(path: Path):
    data = json.load(open(path))
    # Key by (timestamp, ticker) for matching
    return {(o["timestamp"], o["ticker"]): o for o in data["replay_outputs"]}


def compare(original_path: Path, new_path: Path):
    original = load_outputs(original_path)
    new = load_outputs(new_path)
    
    all_keys = set(original) | set(new)
    
    only_in_original = set(original) - set(new)
    only_in_new = set(new) - set(original)
    in_both = set(original) & set(new)
    
    divergences = []
    for key in in_both:
        o_val = original[key]
        n_val = new[key]
        if o_val["decision"] != n_val["decision"] or \
           abs(o_val["composite"] - n_val["composite"]) > 0.001:
            divergences.append({
                "key": key,
                "original": o_val,
                "new": n_val,
            })
    
    print(f"Total events: {len(all_keys)}")
    print(f"Only in original (old code triggered, new doesn't): {len(only_in_original)}")
    print(f"Only in new (new code triggers, old didn't): {len(only_in_new)}")
    print(f"Decision/score divergence: {len(divergences)}")
    
    return {
        "total": len(all_keys),
        "only_original": list(only_in_original),
        "only_new": list(only_in_new),
        "divergences": divergences,
    }
```

**Interpretations-Regel:** Eine Code-Änderung ist erst "verifiziert", wenn du eine klare Erwartung über die Divergenzen hast UND das Tool genau diese Divergenzen zeigt — nicht mehr und nicht weniger.

---

## 6. Non-Determinismus-Fallen

### 6.1 Die häufigsten Bugs

**Falle 1 — `datetime.utcnow()` im Code:**
Jede nicht-durch-Clock-abstrahierte Zeit-Zugriff bricht Replay. Grep:
```bash
grep -rn "datetime.utcnow\|datetime.now\|time.time\b" src/ --exclude-dir=__pycache__
```

Systematischer Refactor: jeder Zeit-Zugriff geht durch `self.clock.now()`.

**Falle 2 — Random ohne seeding:**
```python
# FALSCH:
import random
val = random.random()

# RICHTIG:
self.rng = random.Random(seed=42)  # oder seed aus Session-Metadaten
val = self.rng.random()
```

**Falle 3 — External HTTP-Calls während Replay:**
Irgendwo in deinem Code ruft du `requests.get(...)` auf, um z.B. einen Ticker-Meta-Datensatz zu holen. Beim Replay triggert das HTTP — Netzwerk-Latenz, API-Rate-Limit, andere Ergebnisse als damals.

**Lösung:** alle externen Calls durch Events ersetzen, die zum Stream gehören. Oder HTTP-Response-Cache (z.B. `vcr.py`) pro Session speichern.

**Falle 4 — Dict-Iteration-Order (vor Python 3.7 relevant, jetzt weniger):**
Python 3.7+ hat insertion-ordered dicts, also ok. Aber `set()` hat immer noch non-deterministische Iteration. Wenn Reihenfolge matters, `sorted()`.

**Falle 5 — Floating-Point-Akkumulation:**
`sum([0.1, 0.2, 0.3])` kann minimal unterschiedliche Werte produzieren je nach CPU-Pfad. Für Replay-Vergleiche: Toleranz (`abs(diff) < 1e-6`) statt exakter Gleichheit.

### 6.2 Die Determinismus-Checkliste

```python
# tests/determinism/test_replay_determinism.py
"""Property-Test: derselbe Event-Stream → identische Output-Events."""

def test_replay_twice_same_result():
    session_id = "test_session_001"
    
    # Erstes Replay
    outputs_a = run_replay(session_id)
    
    # Zweites Replay
    outputs_b = run_replay(session_id)
    
    # Identisch?
    assert len(outputs_a) == len(outputs_b)
    for a, b in zip(outputs_a, outputs_b):
        assert a == b, f"Divergenz: {a} != {b}"
```

Wenn dieser Test jemals fehlschlägt, hast du einen Non-Determinismus-Bug. **Immer** finden, bevor es weitergeht. Replay ist wertlos, wenn es nicht deterministisch ist.

---

## 7. Storage-Management

### 7.1 Event-Volumen

**Grobrechnung für Hans:**
- 20 Ticker × 1 Tick/Minute × 390 Minuten/Tag = 7.800 Market-Ticks/Tag
- ~50 News/Tag
- ~5 Orders/Tag (plus Fills/Rejects)
- ~2 ClockTicks/Minute = 780/Tag

Total: **~8.500 Events/Tag**

Bei ~200 Byte pro Event (pydantic-JSON): **~1.7 MB/Tag**, **~600 MB/Jahr**.

SQLite kommt damit problemlos klar. Erst bei 100+ Mio Events (mehrere Jahre × mehr Ticker) wird Postgres interessant.

### 7.2 Retention-Policy

**Empfehlung:**
- **Live-Event-Stream:** 90 Tage in Primary-DB
- **Archive:** Sessions älter als 90 Tage als parquet gezippt in `data/event_archive/`
- **Golden-Sessions (siehe `35_GOLDEN_EQUITY_SCENARIO_TESTS.md`):** permanent behalten

```python
# scripts/maintenance/archive_old_sessions.py
"""Läuft nächtlich. Archiviert Sessions älter als 90 Tage."""

def archive_session(store: EventStore, session_id: str, archive_dir: Path):
    events = list(store.load_session(session_id))
    
    import pandas as pd
    df = pd.DataFrame([e.model_dump() for e in events])
    
    output = archive_dir / f"{session_id}.parquet.gz"
    df.to_parquet(output, compression="gzip")
    
    # Delete from primary store
    with sqlite3.connect(store.db_path) as conn:
        conn.execute("DELETE FROM events WHERE session_id = ?", (session_id,))
```

---

## 8. Integration mit anderen Playbooks

**Mit `35_GOLDEN_EQUITY_SCENARIO_TESTS.md`:**
Jede "Golden Session" ist ein archivierter Event-Stream + erwartetes Signal-Output. Regression-Test:
```python
def test_golden_session_2026_02_15():
    outputs = run_replay("golden_2026_02_15")
    expected = json.load(open("tests/golden/2026_02_15_expected.json"))
    assert_outputs_match(outputs, expected, tolerance=1e-6)
```

**Mit `38_FEATURE_ATTRIBUTION_DASHBOARD.md`:**
Jede Attribution wird als Event geloggt. Post-Mortem zeichnet Attribution-Events aus Session nach.

**Mit `39_HYPERPARAMETER_GOVERNANCE.md`:**
Jeder Replay-Run bekommt Metadata: Git-SHA, Strategy-Config-Version, MLflow-Model-Version. So kannst du sagen: "Replay mit Code-SHA a3f2b1c vs. e4d5c2b, Session golden_2026_02_15, Divergenzen: ...".

**Mit `36_MULTI_ENVIRONMENT_SETUP.md`:**
Replay läuft nur in Dev-Environment. Prod-Event-Store wird nach Dev kopiert.

---

## 9. Umsetzungs-Checkliste

**Phase 1 — Event-Schema (Woche 1):**
- [ ] Pydantic-Schemas für alle Event-Typen
- [ ] `EventStore` mit SQLite-Backend
- [ ] Unit-Tests für Append + Load
- [ ] Property-Test: Round-Trip Serialisierung

**Phase 2 — Live-Integration (Woche 2):**
- [ ] Event-Recording im Market-Data-Handler
- [ ] Event-Recording im News-Handler
- [ ] Event-Recording im Broker-Handler
- [ ] Sequence-Counter-Management
- [ ] Performance-Test: kann das Live-System 10k Events/Sekunde schreiben?

**Phase 3 — Clock-Abstraktion (Woche 3):**
- [ ] `Clock`-Protocol, `RealClock`, `ReplayClock`
- [ ] Systematischer Refactor aller `datetime.utcnow()`-Stellen
- [ ] Dependency-Injection in Pipeline
- [ ] grep-Check: kein `datetime.utcnow()` mehr ausser in `RealClock`

**Phase 4 — Replayer (Woche 4):**
- [ ] `Replayer`-Klasse
- [ ] `ReplayBroker` für Broker-Mocking
- [ ] `replay_session.py` Script
- [ ] Determinismus-Test

**Phase 5 — Vergleich-Tools (Woche 5):**
- [ ] `compare_replays.py`
- [ ] Golden-Session-Archive
- [ ] Regression-Tests für Golden-Sessions

**Phase 6 — Maintenance (Woche 6):**
- [ ] Archive-Script
- [ ] Retention-Policy in cron
- [ ] Monitoring: Event-Rate, DB-Size

**Gesamt-Aufwand:** 5-6 Wochen bei 10-15 h/Woche. **Größtes Risiko:** Determinismus-Bugs, die erst Wochen später auftauchen.

---

## 10. Quellen

**Event Sourcing in Python:**
- [pyeventsourcing/eventsourcing](https://github.com/pyeventsourcing/eventsourcing) — Referenz-Library
- [eventsourcing auf PyPI](https://pypi.org/project/eventsourcing/)

**Event-Sourcing in Trading:**
- Durga Analytics: [Event Sourcing & Audit Trail Design for Trading Systems](https://durgaanalytics.com/event_sourcing_audit_trading) — mit Replay-Harness-Konzept
- arxiv 2602.23193 (Feb 2026): [ESAA: Event Sourcing for Autonomous Agents](https://arxiv.org/html/2602.23193) — SHA-256-Hash-Validierung für deterministisches Replay

**Deterministische Trading-Engines:**
- [NautilusTrader](https://nautilustrader.io/) — Rust-native, nanosecond-deterministic, Research-to-Live-Parity
- [NautilusTrader GitHub](https://github.com/nautechsystems/nautilus_trader)
- [NautilusTrader Backtest Documentation](https://nautilustrader.io/docs/nightly/getting_started/backtest_low_level/)

**Event-Driven Backtesting:**
- Timothy Kimutai (Oktober 2025): [How I Built an Event-Driven Backtesting Engine in Python](https://timkimutai.medium.com/how-i-built-an-event-driven-backtesting-engine-in-python-25179a80cde0) — ABC-Pattern für DataHandler
- QuantStart: [Event-Driven Backtesting with Python](https://www.quantstart.com/articles/Event-Driven-Backtesting-with-Python-Part-I/) — Klassiker der Architektur-Patterns

---

## 11. Ehrliche Einschätzung

**Was dieses Playbook dir gibt:**
- Bit-genaues Replay einer Live-Session
- Regression-Testing gegen reale Daten
- Forensik-Fähigkeit für Incidents
- "What-if"-Analysen auf echten Daten statt rekonstruierten

**Was es dir nicht gibt:**
- **Live-Debugging.** Replay hilft **nach** dem Bug, nicht in dem Moment, wo er passiert.
- **Full-Fidelity-Broker-Simulation.** Der `ReplayBroker` picks Fill-Events aus dem Stream — wenn dein neuer Code eine Order submitted, die im Original nicht passiert ist, gibt es keinen Fill im Stream zum matchen. Das ist fundamentaler: was hätte der echte Broker geantwortet? Unmöglich zu wissen. Du kannst nur validieren, dass Code-Änderungen **keine neuen Orders** generieren.
- **Sub-Millisekunden-Präzision.** Wenn deine Strategy auf Microsecond-Latenz angewiesen ist, ist SQLite nicht genug. Für dich irrelevant.

**Die drei Sachen, die du nicht auslassen darfst:**
1. **Clock-Abstraktion von Anfang an.** Wenn du `datetime.utcnow()` erst später rausrefaktorierst, findest du jahrelang Bugs. Mach's früh, einmal, richtig.
2. **Events sind immutable. Append-only. Niemals UPDATE.** Klingt trivial, aber es gibt die Versuchung, z.B. einen Typo in der Headline zu korrigieren. Nein — die Event-Historie bleibt wie sie war. Korrekturen sind neue Events.
3. **Determinismus-Test als CI-Check.** Ein nicht-deterministischer Replayer ist wertlos. Der Test, der zweimal Replay laufen lässt und identische Outputs fordert, muss in der Pipeline sein.

**Der entscheidende Punkt für einen Einzel-Quant:** Du willst nicht NautilusTrader-Niveau. Du willst das **minimale Replay-System**, das deine häufigsten Debug-Szenarien abdeckt. SQLite + Pydantic + Clock-Abstraktion = 90 % des Werts mit 10 % des Aufwands. Wenn du später auf Kafka oder NautilusTrader migrierst, hast du zumindest das Schema schon. Das ist ein guter Trade.

**Ein wichtiger Nebeneffekt, der oft unterschätzt wird:** Event-Sourcing **zwingt dich, deine Architektur klar zu strukturieren**. Welche Events sind inputs, welche sind outputs? Was ist State, was ist Derivation? Nach Einführung von Events wirst du feststellen, dass 30-40 % deines bisherigen Codes gar nicht notwendig war — er war Workaround für fehlende Event-Abstraktion. Das ist der größte Code-Quality-Gewinn des ganzen Playbooks.
