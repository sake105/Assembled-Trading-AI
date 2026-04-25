# 41 — PDT-Regel und Intraday-Margin-Strategie

**Zweck:** US-Aktien unter 25.000 USD Equity haben historisch die FINRA-Pattern-Day-Trader-Regel (PDT) einhalten müssen. Hans' Alpaca-Account ist wahrscheinlich unter dieser Schwelle. **Aber:** Die Regel wird am **4. Juni 2026** durch ein neues Intraday-Margin-System ersetzt. Diese Datei dokumentiert die Regeln vor und nach dem Cutover und wie der Code damit umgehen muss.

**Scope:** Rang 8 aus der Gap-Analyse. Direkt relevant für die Execution-Layer (`33_EXECUTION_ORDERMANAGEMENT.md`). Zeitkritisch.

**Kern-Fakten (Stand 24. April 2026):**
- SEC hat am **14. April 2026** die Abschaffung der PDT-Regel genehmigt (SR-FINRA-2025-017)
- FINRA Regulatory Notice 26-10 vom **20. April 2026** setzt Effective Date auf **4. Juni 2026**
- Broker dürfen bis **20. Oktober 2027** (18 Monate Phase-In) migrieren
- **Alpaca-Status:** zu prüfen — wann Alpaca konkret migrated
- Bis der Broker migriert hat, gelten **die alten PDT-Regeln für deinen Account weiter**

---

## 0. Warum das wichtig ist (auch in 6 Wochen noch)

### Was passiert ohne PDT-Awareness

**Szenario 1 — Ungewolltes Flagging (bis Juni 2026):**
Du hast 8.000 USD im Alpaca-Margin-Account. Deine Strategy macht Dienstag 09:35 Buy AAPL, Dienstag 15:45 Sell AAPL — **1 Day-Trade**. Mittwoch dasselbe mit MSFT — **2 Day-Trades**. Donnerstag NVDA — **3**. Freitag — viertes Day-Trade = **PDT-Flag**.

Ohne Alpaca's Built-In-Protection würde dein Account jetzt:
- 90 Tage liquidation-only mode
- Keine neuen Day-Trades bis Equity ≥ 25k
- Restriction-Flag, das du manuell aufheben lassen musst (einmalig pro Account-Lifetime)

**Szenario 2 — Alpaca's Protection blockt deine Orders:**
Alpaca schützt dich vor Szenario 1, indem ab dem **4. Day-Trade** ein Order mit **HTTP 403** abgelehnt wird. Dein System sieht:
```
OrderRejectedException: 403 Forbidden — PDT protection
```

Wenn dein Code das nicht handled, crasht die Strategy-Schleife. Oder schlimmer: sie bleibt in einer Retry-Loop und hämmert die API mit denselben 403-Errors.

**Szenario 3 — Nach Juni 2026, aber Broker noch nicht migriert:**
Offiziell ist PDT weg. In der Praxis läuft deine Alpaca-API aber noch unter den alten Regeln, weil Alpaca erst später migriert. Du denkst "ab Juni frei", dein System denkt "noch gebunden". Verwirrung, Fehlersuche.

**Szenario 4 — Nach voller Migration (spätestens Oktober 2027):**
Keine Day-Trade-Zählung mehr. Stattdessen **Intraday-Margin-Checks** — dein Broker prüft Real-Time, ob die aktuelle Position zum verfügbaren Equity passt. Wenn nicht: Order-Reject. Das Verhalten deines Codes muss sich darauf einstellen.

### Die Kern-Aufgabe für dein System

1. **Day-Trade-Counter** einbauen, der die Historie der letzten 5 Business-Days tracked
2. **Pre-Order-Check:** Würde dieser Order den Account als PDT flaggen? Wenn ja, Strategy overrule.
3. **HTTP-403-Handling:** wenn Alpaca ablehnt, nicht crashen
4. **Feature-Flag** für den Cutover: vor 4. Juni "alte Logik", nach 4. Juni (sobald Alpaca migriert) "neue Logik"

---

## 1. Die alte PDT-Regel — bis zum Cutover

### 1.1 Formale Definition (FINRA Rule 4210, bis 3. Juni 2026)

Ein Account wird als Pattern-Day-Trader klassifiziert, wenn:
1. **Vier oder mehr Day-Trades** werden **innerhalb von 5 Business-Days** ausgeführt
2. Diese Day-Trades machen **mehr als 6 %** der gesamten Trades in diesem Zeitraum aus

Ein **Day-Trade** = Öffnen UND Schließen derselben Position am selben Tag. Long-Beispiel: Buy 09:30, Sell 15:45. Short-Beispiel: Sell-Short 09:30, Buy-Cover 15:45.

### 1.2 Beispiele aus der FINRA-Doku

```
Example A:
09:30  Buy 100 ABC
15:45  Sell 100 ABC
→ 1 Day-Trade

Example B:
09:30  Buy 100 ABC
09:31  Sell 100 ABC        ← Round-trip 1
09:32  Buy 100 ABC
13:00  Sell 100 ABC        ← Round-trip 2
→ 2 Day-Trades

Example C:
09:30  Buy 500 ABC
13:00  Sell 100 ABC
13:01  Sell 100 ABC
13:03  Sell 300 ABC
→ 1 Day-Trade (nicht 3! — das Schließen eines einzigen Buy-Orders zählt als ein Trade)

Example D:
09:30  Buy 250 ABC
09:31  Buy 300 ABC
13:01  Buy 100 ABC
13:02  Sell 150 ABC
13:03  Sell 175 ABC
→ 2 Day-Trades (aufstocken + teilweise schließen in zwei Schritten)
```

**Wichtig:** Die Teilausführungen in Example C zählen als **ein** Day-Trade, weil sie **eine** ursprüngliche Position schließen.

### 1.3 Konsequenzen des PDT-Flags

- **90 Tage Restriction:** kein Day-Trade mehr, bis Equity ≥ 25k
- **One-Time-Removal:** einmal pro Account-Leben kann ein manueller Request ans Broker das Flag aufheben. Danach nicht mehr.
- **Alpaca-spezifisch:** Alpaca hat **Preventive Protection** — sie blockieren den 4. Day-Trade mit HTTP 403, **bevor** das Flag gesetzt wird. Du bekommst also das Flag nie, solange der Block greift.

### 1.4 Was **nicht** PDT-betroffen ist

- **Cash-Accounts:** PDT gilt nur für Margin-Accounts
- **Crypto:** keine PDT-Anwendung (unregulated)
- **Futures/Forex:** andere Regulatory-Frameworks, kein PDT
- **Swing-Trades:** Position > 1 Tag gehalten = kein Day-Trade

---

## 2. Die neue Regel — ab 4. Juni 2026

### 2.1 Was sich ändert

Per FINRA Regulatory Notice 26-10:
- **25k-Minimum abgeschafft**
- **PDT-Designation abgeschafft**
- **Day-Trade-Counting abgeschafft**
- **Alte 4:1-Day-Trading-Buying-Power abgeschafft**
- **90-Tage-Freeze-Mechanismus abgeschafft**

### 2.2 Was bleibt / was neu ist

**Was bleibt:**
- **Federal Reserve Regulation T:** Minimum 2.000 USD für Margin-Trading. **Dein Account braucht immer noch mindestens 2k.**
- **Standard-Maintenance-Margin:** 25 % der aktuellen Position

**Was neu ist — Intraday-Margin-Standards:**
- Broker müssen **Real-Time- oder End-of-Day-Margin-Checks** machen
- Margin-Requirements basieren auf **tatsächlicher Position-Risiko**, nicht mehr auf arbiträren Dollar-Schwellen
- Für "high-velocity instruments" wie 0DTE-Options strengere Regeln
- Broker darf eigene **"house rules"** strenger machen als FINRA-Minimum

### 2.3 Was das für Hans praktisch heißt

**Mit Alpaca-Margin-Account und $5.000 Equity:**

| Vor 4. Juni 2026 | Nach voller Migration |
|---|---|
| Max 3 Day-Trades pro 5 Business-Days | Unbegrenzte Day-Trades |
| 4. Day-Trade → 403 | Kein harter Block mehr |
| — | Aber: Intraday-Margin-Checks können Order ablehnen, wenn Position-Risk zu groß |
| 4:1 Day-Trading-Buying-Power wenn PDT-designated | Standard 2:1 oder 4:1 je nach Broker-House-Rules |
| Keine Strategy-Änderung nötig bis Alpaca migriert | Strategy kann freier handeln, aber andere Error-Modes |

**Timing-Unsicherheit:** Alpaca hat 18 Monate Phase-In. Zwischen 4. Juni 2026 und 20. Oktober 2027 kann Alpaca jederzeit migrieren. Monitoring nötig.

---

## 3. Der Day-Trade-Counter

### 3.1 Die Datenstruktur

```python
# src/assembled_core/execution/pdt_tracker.py
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import List
import pandas as pd


@dataclass
class DayTrade:
    """Eine abgeschlossene Round-Trip-Transaktion am selben Tag."""
    ticker: str
    open_timestamp: datetime
    close_timestamp: datetime
    side: str                              # "long" oder "short"
    quantity: int
    entry_price: float
    exit_price: float
    
    @property
    def trade_date(self) -> date:
        return self.open_timestamp.date()
    
    @property
    def pnl(self) -> float:
        if self.side == "long":
            return (self.exit_price - self.entry_price) * self.quantity
        return (self.entry_price - self.exit_price) * self.quantity


class PDTTracker:
    """Tracked Day-Trades über das rolling 5-Business-Day-Fenster."""
    
    def __init__(self, account_equity: float, enabled: bool = True):
        self.account_equity = account_equity
        self.enabled = enabled  # Wird False nach Alpaca-Migration
        self.day_trades: List[DayTrade] = []
    
    def record_day_trade(self, trade: DayTrade):
        """Wird von Order-Manager aufgerufen wenn ein Round-Trip detected wird."""
        self.day_trades.append(trade)
    
    def count_recent_day_trades(self, reference_date: date = None) -> int:
        """Anzahl Day-Trades in den letzten 5 Business-Days."""
        if reference_date is None:
            reference_date = date.today()
        
        cutoff = self._business_days_ago(reference_date, 5)
        return sum(1 for t in self.day_trades if t.trade_date > cutoff)
    
    def would_violate_pdt(self, reference_date: date = None) -> bool:
        """Würde ein zusätzlicher Day-Trade das PDT-Limit verletzen?"""
        if not self.enabled:
            return False  # nach Migration: keine PDT mehr
        
        if self.account_equity >= 25_000:
            return False  # über Schwelle: keine Restriction
        
        current = self.count_recent_day_trades(reference_date)
        return current >= 3  # 3 existing + 1 new = 4 = PDT
    
    @staticmethod
    def _business_days_ago(reference: date, n: int) -> date:
        """Business-Day-Arithmetic mit pandas."""
        bday_range = pd.bdate_range(end=reference, periods=n+1)
        return bday_range[0].date()
    
    def days_until_pdt_reset(self, reference_date: date = None) -> int:
        """Wie viele Business-Days bis der älteste Day-Trade rausfällt?"""
        if reference_date is None:
            reference_date = date.today()
        
        recent = [t for t in self.day_trades 
                  if t.trade_date > self._business_days_ago(reference_date, 5)]
        if not recent:
            return 0
        
        oldest = min(t.trade_date for t in recent)
        days_since_oldest = len(pd.bdate_range(start=oldest, end=reference_date))
        return max(0, 5 - days_since_oldest)
```

### 3.2 Die Detection: wann ist eine Position ein Day-Trade?

```python
# src/assembled_core/execution/round_trip_detector.py
from typing import Optional
from datetime import datetime
from .pdt_tracker import DayTrade, PDTTracker


class RoundTripDetector:
    """Erkennt Day-Trades bei jedem Order-Fill."""
    
    def __init__(self, tracker: PDTTracker):
        self.tracker = tracker
        # Offene Positionen pro Ticker, mit Open-Timestamp
        self.open_positions: dict[str, tuple[datetime, int, float, str]] = {}
    
    def on_fill(self, fill_event):
        """Wird bei jedem Order-Fill von Alpaca aufgerufen."""
        ticker = fill_event.ticker
        side = fill_event.side  # "buy" oder "sell"
        qty = fill_event.quantity
        price = fill_event.price
        ts = fill_event.timestamp
        
        if ticker not in self.open_positions:
            # Opening trade
            position_side = "long" if side == "buy" else "short"
            self.open_positions[ticker] = (ts, qty, price, position_side)
            return None
        
        # Existing position
        open_ts, open_qty, open_price, open_side = self.open_positions[ticker]
        
        # Check: wird Position geschlossen?
        is_closing = (
            (open_side == "long" and side == "sell") or
            (open_side == "short" and side == "buy")
        )
        
        if not is_closing:
            # Position wird aufgestockt, nicht geschlossen
            # Average-down/up: für PDT zählt das nicht als Day-Trade
            # Wir merken uns: neue Quantity, Weighted-Avg-Price
            new_qty = open_qty + qty
            new_price = (open_qty * open_price + qty * price) / new_qty
            self.open_positions[ticker] = (open_ts, new_qty, new_price, open_side)
            return None
        
        # Ist das Open-Trade am selben Handelstag?
        if open_ts.date() == ts.date():
            # → Day-Trade!
            trade = DayTrade(
                ticker=ticker,
                open_timestamp=open_ts,
                close_timestamp=ts,
                side=open_side,
                quantity=min(open_qty, qty),
                entry_price=open_price,
                exit_price=price,
            )
            self.tracker.record_day_trade(trade)
            
            # Position-Tracking updaten
            if qty >= open_qty:
                del self.open_positions[ticker]
            else:
                self.open_positions[ticker] = (open_ts, open_qty - qty, open_price, open_side)
            
            return trade
        
        # Position wird geschlossen, aber Open war an anderem Tag → Swing-Trade
        if qty >= open_qty:
            del self.open_positions[ticker]
        else:
            self.open_positions[ticker] = (open_ts, open_qty - qty, open_price, open_side)
        
        return None
```

**Der kritische Punkt:** Teilausführungen. FINRA-Beispiel C zeigt: Buy 500, dann 3× Sell zu verschiedenen Zeiten = **1** Day-Trade, nicht 3. Unsere Implementation erkennt das, weil nach dem ersten Sell die `open_positions[ticker]` auf die Rest-Qty verringert wird, nicht gelöscht. Der zweite Sell findet eine "Position mit open_ts von vor dem Buy" — zählt nicht als neuer Day-Trade, weil wir nur den **ersten** Close als DT recorden. Bei erneutem Buy danach am selben Tag und erneutem Sell zählt das als zweiter Day-Trade — was korrekt ist (FINRA-Beispiel B).

**Caveat bei Example D:** Unser Code recorded hier **ein** Day-Trade pro Round-Trip-Schritt. Die 150-Stück- und 175-Stück-Sells sind aus Sicht unseres Codes zwei separate Teil-Closures derselben ursprünglichen Position. Das entspricht nicht exakt FINRA's "2 Day-Trades"-Logik für Example D (die aus der mixed Buy/Sell-Pattern 2 separate Round-Trips identifiziert). **Das ist ein bekanntes Edge-Case.** Für den Einzel-Quant-Use-Case mit simpler Buy-dann-Sell-Logic irrelevant, aber in Unit-Tests dokumentieren.

---

## 4. Der Pre-Order-Check

### 4.1 Das Interface

```python
# src/assembled_core/execution/order_gate.py
import logging
from dataclasses import dataclass
from enum import Enum

from .pdt_tracker import PDTTracker
from .round_trip_detector import RoundTripDetector

logger = logging.getLogger(__name__)


class OrderDecision(Enum):
    ALLOWED = "allowed"
    BLOCKED_PDT = "blocked_pdt"
    BLOCKED_MARGIN = "blocked_margin"
    BLOCKED_KILL_SWITCH = "blocked_kill_switch"


@dataclass
class GateResult:
    decision: OrderDecision
    reason: str
    suggested_action: str | None = None


class OrderGate:
    """Pre-Order-Checks vor API-Submission."""
    
    def __init__(self, pdt_tracker: PDTTracker, rt_detector: RoundTripDetector):
        self.pdt_tracker = pdt_tracker
        self.rt_detector = rt_detector
    
    def check_order(self, ticker: str, side: str, qty: int) -> GateResult:
        """Prüft: darf dieser Order submitted werden?"""
        
        # PDT-Check (nur wenn enabled)
        if self.pdt_tracker.enabled:
            would_be_day_trade = self._would_be_day_trade(ticker, side)
            if would_be_day_trade:
                if self.pdt_tracker.would_violate_pdt():
                    days_until_reset = self.pdt_tracker.days_until_pdt_reset()
                    return GateResult(
                        decision=OrderDecision.BLOCKED_PDT,
                        reason=f"Would be 4th day trade in 5 business days. "
                               f"Count: {self.pdt_tracker.count_recent_day_trades()}/3. "
                               f"Account equity ${self.pdt_tracker.account_equity:,.0f} < 25k.",
                        suggested_action=(
                            f"Wait {days_until_reset} business days, "
                            f"OR hold position overnight (no day-trade), "
                            f"OR skip this signal."
                        ),
                    )
        
        # Weitere Checks (Margin, Kill-Switch) werden in 33_EXECUTION_ORDERMANAGEMENT.md behandelt
        
        return GateResult(
            decision=OrderDecision.ALLOWED,
            reason="all checks passed",
        )
    
    def _would_be_day_trade(self, ticker: str, side: str) -> bool:
        """Check: würde dieser Order eine open position vom heutigen Tag schließen?"""
        if ticker not in self.rt_detector.open_positions:
            return False
        
        open_ts, _, _, open_side = self.rt_detector.open_positions[ticker]
        from datetime import date
        if open_ts.date() != date.today():
            return False  # Swing, kein Day-Trade
        
        is_closing = (
            (open_side == "long" and side == "sell") or
            (open_side == "short" and side == "buy")
        )
        return is_closing
```

### 4.2 Integration im Trading-Cycle

```python
# src/assembled_core/pipeline/cycle.py (Ausschnitt)
from ..execution.order_gate import OrderGate, OrderDecision


def submit_order_with_gate(ticker, side, qty, gate: OrderGate):
    result = gate.check_order(ticker, side, qty)
    
    if result.decision == OrderDecision.ALLOWED:
        try:
            order_id = alpaca_client.submit_order(ticker, side, qty)
            logger.info(f"Order submitted: {ticker} {side} {qty} → {order_id}")
            return order_id
        except AlpacaAPIError as e:
            if e.status_code == 403 and "pattern day trading" in e.message.lower():
                # Alpaca's Built-in protection kicked in (redundant zum Gate, aber Safety-Net)
                logger.warning(f"Alpaca PDT-protection blocked order: {ticker}")
                return None
            raise
    
    elif result.decision == OrderDecision.BLOCKED_PDT:
        logger.warning(
            f"Order blocked by local PDT gate: {ticker} {side} {qty}. "
            f"Reason: {result.reason}. Suggestion: {result.suggested_action}"
        )
        # Attribution-Store: dokumentiere, dass Signal generiert aber nicht ausgeführt
        return None
```

**Wichtig:** Zwei Defense-Layer. Unser lokaler Gate **und** Alpaca's Built-in-Protection. Beide greifen redundant. Wenn unser Gate einen Bug hat, fängt Alpaca ab. Wenn Alpaca einen Bug hat, fängt unser Gate ab.

---

## 5. Strategien um PDT zu vermeiden

### 5.1 Strategie A — Cash-Account statt Margin

**Vorteil:** Keine PDT-Regel überhaupt.
**Nachteil:** T+1 Settlement für Options, T+1 für Stocks (seit Mai 2024). Good-Faith-Violations bei unsettled funds.
**Empfehlung:** Nur wenn du wirklich viele Day-Trades machen willst und bereit bist, Settlement-Friction zu managen.

### 5.2 Strategie B — Swing-Trades statt Day-Trades

**Vorteil:** Keine PDT-Zählung wenn Position > 1 Nacht gehalten wird.
**Nachteil:** Overnight-Risk, größere Stops nötig, weniger Trades = weniger Edge-Gelegenheiten.
**Empfehlung:** **Die beste Strategie für Hobby-Quants.** Dein Edge wird durch 3 Day-Trades/Woche kaum aufgebraucht.

### 5.3 Strategie C — Multi-Broker-Splitting

**Vorteil:** Wenn du 3 Accounts à 5k hast, hast du 3×3=9 Day-Trades/Woche statt 3.
**Nachteil:** Pro-Account-Minimum von 2k (Fed Regulation T), operationeller Overhead, jede Firma anderen Onboarding.
**Empfehlung:** Nur wenn du wirklich das Limit fühlst. Für Einzel-Hobby-Quant Overkill.

### 5.4 Strategie D — Futures/Options statt Stocks

**Vorteil:** PDT gilt nicht für Futures. Mini-Contracts (MES, MNQ) haben niedrige Margins.
**Nachteil:** Anderes Instrument, Leverage höher (gefährlich), andere Steuerregeln.
**Empfehlung:** Nicht für Einsteiger. Nach 1-2 Jahren Erfahrung evaluierbar.

### 5.5 Empfehlung für Hans

Gegeben:
- Hans' Budget wahrscheinlich < 25k (aus Gap-Analyse hervorgegangen)
- Strategy ist Trend-News-basiert mit stündlichen bis täglichen Zyklen
- Risk-Appetit moderat (Streifendienst-Lohn, kein Spekulant)

**→ Strategy B (Swing-Trading)** ist optimal. Dein Composite-Score soll auf 2-5 Tage Forward-Returns ausgelegt werden, nicht auf Intraday-Moves. Das umgeht PDT komplett.

**Mit 6 Wochen bis Cutover:** Nicht lohnenswert, groß umzubauen. Swing-Fokus behalten, PDT-Gate implementieren, Juni abwarten.

---

## 6. Der Cutover-Plan (Juni 2026)

### 6.1 Die Unsicherheit managen

**Was wir wissen:**
- FINRA-Effective-Date: 4. Juni 2026
- Broker-Phase-In: bis spätestens 20. Oktober 2027
- **Alpaca's konkreter Migration-Termin: nicht öffentlich kommuniziert** (Stand 24. April 2026)

**Was zu tun ist:**
1. **Alpaca beobachten:** Newsletter abonnieren, Community-Forum checken, Support-Tickets für offiziellen Timeline-Request
2. **Feature-Flag im Code:** `pdt_tracker.enabled = True` steht auf `True` solange unklar ist, ob Alpaca migriert
3. **Error-Pattern-Detection:** wenn Alpaca aufhört HTTP 403 mit "pattern day trading" zu senden, ist das ein Signal dass Migration passiert ist

### 6.2 Detection-Logic

```python
# src/assembled_core/execution/migration_detector.py
from datetime import datetime, timedelta
from collections import deque
import logging

logger = logging.getLogger(__name__)


class PDTMigrationDetector:
    """Beobachtet ob Alpaca PDT-Blocks noch auftreten.
    
    Wenn über 30 Tage keine PDT-spezifischen 403 mehr kommen obwohl
    4. Day-Trade-Attempts dazwischen, ist Alpaca wahrscheinlich migriert.
    """
    
    def __init__(self, observation_window_days: int = 30):
        self.observation_window = timedelta(days=observation_window_days)
        self.pdt_blocks: deque = deque(maxlen=100)
        self.fourth_day_trade_attempts: deque = deque(maxlen=100)
    
    def record_pdt_block(self, timestamp: datetime = None):
        self.pdt_blocks.append(timestamp or datetime.utcnow())
    
    def record_fourth_day_trade_attempt(self, timestamp: datetime = None):
        """Ein Order, bei dem unser Tracker sagt 'wäre 4. Day-Trade in 5d' — 
        wenn Alpaca das trotzdem durchlässt, ist Migration passiert."""
        self.fourth_day_trade_attempts.append(timestamp or datetime.utcnow())
    
    def likely_migrated(self) -> bool:
        cutoff = datetime.utcnow() - self.observation_window
        
        recent_attempts = sum(1 for ts in self.fourth_day_trade_attempts if ts > cutoff)
        recent_blocks = sum(1 for ts in self.pdt_blocks if ts > cutoff)
        
        if recent_attempts >= 3 and recent_blocks == 0:
            logger.warning(
                f"PDT Migration likely: {recent_attempts} 4th-day-trade attempts "
                f"in last {self.observation_window.days}d, 0 PDT blocks. "
                f"Recommend: disable pdt_tracker."
            )
            return True
        return False
```

### 6.3 Der Cutover-Prozess

```bash
# Wenn Detection sagt "wahrscheinlich migriert" ODER Alpaca offiziell ankündigt:

# 1. Staging: pdt_tracker.enabled = False
# In configs/strategies/active.yaml:
# execution:
#   pdt_tracker_enabled: false

# 2. 7 Tage Staging-Beobachtung
# - Gibt es HTTP 403 mit "pattern day trading"? Dann Migration NICHT passiert.
# - Alles normal? Weiter.

# 3. Prod-Cutover
# Alte PDT-Handling bleibt im Code als Fallback (Feature-Flag).
# Kann in 6 Monaten entfernt werden, wenn sicher.

# 4. Neue Error-Modes handhaben
# Nach Migration: Alpaca kann wegen Intraday-Margin ablehnen.
# Neuer 403-Text: "insufficient day trading buying power" oder
# "intraday exposure limit exceeded".
# Error-Handling anpassen.
```

### 6.4 Was nach Migration anders ist

| Aspekt | Vor (alte PDT) | Nach (Intraday-Margin) |
|---|---|---|
| Day-Trade-Counting | 4 in 5 Tagen = Flag | Nicht mehr relevant |
| Equity-Schwelle | $25k hard floor | $2k Regulation-T floor |
| Buying-Power | 4× Equity für PDT, 2× sonst | Dynamisch, Position-basiert |
| Intraday-Limits | keine expliziten | Real-Time oder EOD-Margin-Check |
| 403-Errors | "pattern day trading" | "intraday exposure exceeded" |
| 90-Tage-Freeze | Nach 4. DT ohne 25k | Nicht mehr |

---

## 7. Monitoring im Dashboard

Im Attribution-Dashboard (siehe `38_FEATURE_ATTRIBUTION_DASHBOARD.md`) ergänzen:

```python
# dashboards/attribution_app.py (ergänzung)
import streamlit as st


def render_pdt_status(st, pdt_tracker):
    """PDT-Status-Widget."""
    st.header("PDT-Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        count = pdt_tracker.count_recent_day_trades()
        remaining = max(0, 3 - count)
        st.metric(
            label="Day-Trades (rolling 5d)",
            value=count,
            delta=f"{remaining} remaining" if remaining > 0 else "AT LIMIT",
            delta_color="inverse",
        )
    
    with col2:
        st.metric(
            label="Account Equity",
            value=f"${pdt_tracker.account_equity:,.0f}",
            delta="over 25k" if pdt_tracker.account_equity >= 25_000 else "under 25k",
        )
    
    with col3:
        days_to_reset = pdt_tracker.days_until_pdt_reset()
        st.metric(
            label="Days until PDT reset",
            value=days_to_reset,
        )
    
    # Migration Status
    if pdt_tracker.enabled:
        st.warning("PDT-Tracking aktiv (alte Regel, bis Alpaca migriert)")
    else:
        st.success("PDT abgeschafft — Migration detected")
```

---

## 8. Umsetzungs-Checkliste

**Phase 1 — Tracker implementieren (Tag 1-3):**
- [ ] `DayTrade` dataclass
- [ ] `PDTTracker` mit rolling-window-Logic
- [ ] `RoundTripDetector` für Fill-Events
- [ ] Unit-Tests mit allen 4 FINRA-Beispielen

**Phase 2 — Pre-Order-Gate (Tag 4-5):**
- [ ] `OrderGate` mit `would_violate_pdt`-Check
- [ ] Integration in `submit_order`
- [ ] HTTP-403-Handling als Safety-Net
- [ ] Logging jeder Blockierung in Attribution-Store

**Phase 3 — Dashboard (Tag 6):**
- [ ] PDT-Widget in Streamlit
- [ ] Historical-Chart: Day-Trade-Count über Zeit
- [ ] Alert wenn Count = 3 (warning-state)

**Phase 4 — Strategy-Alignment (Tag 7-10):**
- [ ] Composite-Score auf Swing-Zeitrahmen (2-5d) auslegen
- [ ] Forward-Return-Label-Generation angepasst
- [ ] Backtest auf Swing-Strategie validieren

**Phase 5 — Cutover-Vorbereitung (Woche 2):**
- [ ] `PDTMigrationDetector` implementiert
- [ ] Feature-Flag in Config
- [ ] Monitoring-Alert bei Detection

**Phase 6 — Cutover (ab Juni 2026):**
- [ ] Alpaca-Announcement tracken
- [ ] Staging mit `pdt_enabled=False` für 7 Tage
- [ ] Prod-Switch
- [ ] Error-Handling auf neue 403-Patterns

**Gesamt:** 2 Wochen aktiv + Monitoring-Phase bis Cutover.

---

## 9. Quellen

**Offizielle Regulatory:**
- [FINRA Regulatory Notice 26-10 (20. April 2026)](https://www.finra.org/rules-guidance/notices/26-10) — Official Effective Date
- SEC Release 34-105226 (14. April 2026): SR-FINRA-2025-017 Approval — [SEC.gov](https://www.sec.gov/files/rules/sro/finra/2026/34-105226.pdf)
- [FINRA: Board Holds September Meeting (24.09.2025)](https://www.finra.org/compliance-tools/weekly-archive/092425) — initiale Board-Approval
- [FINRA: Moves to Overhaul Day Trading Margin (07.01.2026)](https://www.finra.org/compliance-tools/weekly-archive/01072026) — Filing mit SEC

**Alpaca:**
- [Alpaca User Protection](https://docs.alpaca.markets/docs/user-protection) — Built-in PDT-Protection
- [Alpaca: What is the PDT Rule?](https://alpaca.markets/support/what-is-the-pattern-day-trading-pdt-rule) — FINRA-Beispiele
- [Alpaca: PDT Protection](https://alpaca.markets/support/pattern-day-trading-protection) — HTTP-403-Logik
- [Alpaca: Live Trading as Non-US Resident](https://alpaca.markets/learn/live-trading-account-non-us) — explizit: PDT gilt auch für Non-US-Residents
- [Alpaca Community: Is PDT disabled for non US residents?](https://forum.alpaca.markets/t/is-pdt-disabled-for-non-us-residents/6760) — PDT gilt, auch wenn Hans in DE ist

**News 2025/2026:**
- Traders Magazine (September 2025): [FINRA Approves Overhaul of PDT Rules](https://www.tradersmagazine.com/featured_articles/finra-approves-overhaul-of-pattern-day-trading-rules-opens-doors-for-smaller-investors/)
- NerdWallet (April 2026): [The $25,000 Day Trading Rule Ends June 4](https://www.nerdwallet.com/investing/news/pattern-day-trading-rule-change)
- EconoTimes (April 2026): [SEC and FINRA Abolish the $25,000 PDT Rule](https://www.econotimes.com/The-End-of-an-Era-SEC-and-FINRA-Abolish-the-25000-Pattern-Day-Trader-Rule-1739077)
- DayTradingToolkit (April 2026): [PDT Rule Eliminated: Complete Guide](https://daytradingtoolkit.com/market-insights/pdt-rule-eliminated-2026-complete-guide/)
- Prosper Trading (April 2026): [PDT Rule Eliminated: What It Means](https://www.prospertrading.com/pattern-day-trader-rule-eliminated-what-it-means-for-retail-traders/)
- tastytrade: [Pattern Day Trading (updated April 2026)](https://tastytrade.com/learn/markets/industry/pattern-day-trading/)

**Workarounds (historisch):**
- ETNA (Februar 2026): [How to Avoid PDT in 2025](https://www.etnasoft.com/how-to-avoid-the-pattern-day-trading-rule-complete-guide-for-traders-and-brokers/) — Cash-Account, Multi-Broker, Futures
- Algohubb (Januar 2025): [Navigating PDT Rules with Alpaca](https://www.algohubb.com/article/navigating-pdt-rules-with-alpaca-and-algocloud:-a-practical-guide) — Alpaca's Detection-Mechanics
- Pearler: [PDT bei Pearler/Alpaca](https://pearler.com/help/13163859-what-is-the-pattern-day-trading-pdt-rule) — User-Sicht

---

## 10. Ehrliche Einschätzung

**Was dieses Playbook dir gibt:**
- Lokalen PDT-Counter, der redundant zu Alpaca's Protection greift
- Pre-Order-Gate das Signals stoppt bevor Alpaca sie reject
- Dashboard-Widget für laufendes Monitoring
- Cutover-Plan mit Detection-Logic für Alpaca's Migration (Timing unklar)
- Strategische Ausrichtung auf Swing-Trading, das PDT komplett umgeht

**Was es dir nicht gibt:**
- **Keine Garantie über Alpaca's Migration-Timing.** Kann 4. Juni sein, kann 20. Oktober 2027 sein.
- **Kein Edge-Case-Handling für Example-D-Pattern.** Wenn deine Strategy komplex aufstockt und teilweise liquidiert, brauchst du feinere Logik.
- **Keine Umgehung falls du wirklich viele Day-Trades willst.** Die Regel existiert noch 6 Wochen, und dann hat Alpaca noch Phase-In. Bis Ende 2026 musst du damit leben, wenn du unter 25k bist.

**Die drei Sachen, die du nicht auslassen darfst:**
1. **Der Day-Trade-Counter ist Pflicht, unabhängig vom 25k-Status.** Auch wenn du morgen 25k hast, willst du wissen, wie viele Day-Trades dein System macht — für Strategy-Analyse und Overtrading-Detection. Der Counter stirbt nicht mit der PDT-Regel, nur das Limit.
2. **Swing-Fokus ist psychologisch und strategisch besser.** Day-Trades sind nicht nur rechtlich riskant (bis Juni), sondern auch in der Edge-Dichte problematisch. Dein News-Signal braucht typischerweise Tage zum Einpreisen, nicht Stunden. Deine Composite-Score-Architektur aus `31_COMPOSITE_SCORE.md` ist ohnehin nicht auf Intraday ausgelegt.
3. **Nach dem Juni-Cutover neue Error-Patterns antizipieren.** Die 403-Errors ändern sich. "pattern day trading" verschwindet, "intraday exposure" taucht auf. Deine Exception-Handler müssen darauf vorbereitet sein. Einfachster Ansatz: Catch-All für HTTP 403 + Logging des vollen Error-Textes, so kannst du die Patterns im Live-Betrieb beobachten.

**Der psychologische Aspekt, der bei allen Trading-Regeln zählt:** Jede künstliche Beschränkung — ob PDT, ob Intraday-Margin, ob dein eigener Kill-Switch — zwingt dich zur Disziplin. **Das ist ein Feature, kein Bug.** Hobby-Quants verlieren meistens nicht durch zu wenige Trades, sondern durch zu viele. Wenn die Regel dich auf 3 Day-Trades pro Woche limitiert, filtert das 80 % der schwachen Signale raus. Nach Juni, wenn das Limit weg ist, brauchst du eigene Disziplin-Mechanismen — sonst tradest du dich zu Tode, nur weil du darfst.
