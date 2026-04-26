# 33 — Execution und Order-Management

**Zweck:** Layer 4 aus dem Master-Plan — wie aus einem Signal ein Order wird, wie Orders nachverfolgt werden, was bei Störungen passiert. Das ist die Schicht, an der Solo-Quant-Systeme real Geld verlieren.

**Regel:** Der Signal-Stack kann perfekt sein — wenn Execution bricht, verlierst du trotzdem. Diese Datei ist nicht optional.

**Scope:** Alpaca Paper-Trading, US-Equity + optional Options. Live-Trading nur nach 6+ Monaten sauberem Paper-Track-Record.

---

## Die Kern-Probleme die hier gelöst werden

1. **Idempotenz** — bei Retry nicht doppelt ordern
2. **Partial-Fill-Handling** — was wenn nur 300 von 1000 Shares gefüllt werden?
3. **Order-Rejections** — Wash-Sale, PDT, Insufficient-Buying-Power, Halt
4. **Reconciliation** — interner State vs. Broker-State synchron halten
5. **Kill-Switch** — wann und wie stoppen?
6. **Position-Sizing** — von Composite-Score zu Shares

---

## 33.1 Order-Lifecycle

### Der Zustandsbaum

```
Signal         ───► SizedIntent
                         │
                         ▼ (Risk-Gate bestanden)
               PendingSubmission
                         │
                         ▼ (Idempotent Submit)
                   Submitted ─────► Rejected (Terminal, Grund loggen)
                         │
                         ▼
                   Accepted
                         │
                         ├─► PartiallyFilled ──► Filled (Terminal)
                         │
                         ├─► Filled (Terminal)
                         │
                         ├─► Canceled (Terminal)
                         │
                         └─► Expired (Terminal)
```

**Jeder Zustandsübergang wird in Postgres persistiert** (Event-Sourcing-light), nicht nur der aktuelle Zustand.

### Die Tabelle `orders`

```sql
CREATE TABLE orders (
    id UUID PRIMARY KEY,
    client_order_id TEXT UNIQUE NOT NULL,  -- Idempotency-Key
    broker_order_id TEXT UNIQUE,           -- Alpaca-ID, null bis accepted
    symbol TEXT NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('buy', 'sell')),
    qty NUMERIC NOT NULL,
    order_type TEXT NOT NULL,              -- market, limit, stop, stop_limit
    time_in_force TEXT NOT NULL,           -- day, gtc, ioc, fok
    limit_price NUMERIC,
    stop_price NUMERIC,
    status TEXT NOT NULL,                  -- siehe Zustandsbaum
    filled_qty NUMERIC DEFAULT 0,
    filled_avg_price NUMERIC,
    signal_source_id UUID REFERENCES signals(id),
    composite_score NUMERIC,
    intent_hash TEXT,                      -- SHA256 über Signal-Payload
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    submitted_at TIMESTAMPTZ,
    accepted_at TIMESTAMPTZ,
    filled_at TIMESTAMPTZ,
    canceled_at TIMESTAMPTZ,
    rejection_reason TEXT,
    reject_code TEXT,
    raw_broker_response JSONB,
    UNIQUE (signal_source_id, intent_hash)  -- verhindert doppelte Orders pro Signal
);

CREATE INDEX idx_orders_status ON orders(status) WHERE status NOT IN ('filled', 'canceled', 'expired', 'rejected');
CREATE INDEX idx_orders_symbol_status ON orders(symbol, status);
CREATE INDEX idx_orders_created_at ON orders(created_at);
```

### Tabelle `order_events` (Audit-Trail)

```sql
CREATE TABLE order_events (
    id BIGSERIAL PRIMARY KEY,
    order_id UUID NOT NULL REFERENCES orders(id),
    event_type TEXT NOT NULL,   -- submitted, accepted, partial_fill, filled, rejected, canceled, expired
    event_timestamp TIMESTAMPTZ NOT NULL,
    payload JSONB NOT NULL,
    sequence_number BIGINT NOT NULL,
    UNIQUE (order_id, sequence_number)
);
```

Jedes Broker-Event (Websocket oder Polling) erzeugt einen Event-Row. Der aktuelle Order-Status wird aus dem letzten Event abgeleitet oder via Projection gespeichert.

---

## 33.2 Idempotenz via client_order_id

**Das Problem:** Dein Worker crashed nach `submit_order()` aber vor dem DB-Write. Beim Neustart: ordnest du doppelt? **Niemals.**

**Die Lösung:** Jeder Order bekommt eine deterministische `client_order_id` vor dem Submit.

```python
import hashlib
import uuid
from datetime import datetime, timezone

def build_client_order_id(signal_id: str, intent_hash: str, attempt: int = 0) -> str:
    """
    Deterministisch: gleicher Signal + gleicher Intent → gleiche client_order_id.
    Attempt erlaubt gezieltes Retry mit neuer ID, wenn intendiert.
    """
    raw = f"{signal_id}:{intent_hash}:{attempt}"
    h = hashlib.sha256(raw.encode()).hexdigest()[:20]
    return f"ata-{h}"

def compute_intent_hash(symbol: str, side: str, qty: float, 
                        order_type: str, limit_price: float | None) -> str:
    """SHA256 über die intent-definierenden Felder."""
    canon = f"{symbol}|{side}|{qty:.6f}|{order_type}|{limit_price or '-'}"
    return hashlib.sha256(canon.encode()).hexdigest()
```

**Alpaca verhält sich korrekt bei doppelter `client_order_id`:** es lehnt den zweiten Call mit Fehler `duplicate client_order_id` ab. Dein Worker fängt das ab:

```python
from alpaca.common.exceptions import APIError

def submit_with_idempotency(client, intent):
    coid = build_client_order_id(intent.signal_id, intent.intent_hash)
    try:
        resp = client.submit_order(
            symbol=intent.symbol,
            qty=intent.qty,
            side=intent.side,
            type=intent.order_type,
            time_in_force=intent.tif,
            client_order_id=coid,
            limit_price=intent.limit_price,
        )
        return ("submitted", resp, None)
    except APIError as e:
        msg = str(e).lower()
        if "duplicate" in msg and "client_order_id" in msg:
            # Schon submitted — hole den existierenden Order
            existing = client.get_order_by_client_order_id(coid)
            return ("already_submitted", existing, None)
        elif "insufficient" in msg:
            return ("rejected", None, "insufficient_buying_power")
        elif "wash" in msg:
            return ("rejected", None, "wash_sale_block")
        else:
            return ("error", None, str(e))
```

**Regel:** Vor jedem Live-Submit prüfst du zuerst DB: `SELECT 1 FROM orders WHERE client_order_id = $1`. Wenn da → kein Submit, nur Status aktualisieren.

---

## 33.3 Partial-Fill-Handling

**Typische Fehl-Annahme:** "Ich sende Market-Order, bekomme sofort Full-Fill." **Falsch** bei 500+ Tickern mit variabler Liquidität.

### Die Policy

```python
class PartialFillPolicy:
    """
    Was tun, wenn nur ein Teil gefüllt wurde?
    """
    # Nach 2 Minuten, wenn nur partial gefüllt
    CANCEL_AFTER_SECONDS = 120
    
    # Mindest-Fill-Quote, unter der wir den Trade als failed werten
    MIN_FILL_RATIO = 0.5
    
    # Was tun mit teilweisem Fill bei Cancel?
    # "keep" = Teilposition behalten, "liquidate" = mit Market schließen
    ON_CANCEL = "keep"

async def handle_partial_fill(order, current_status):
    elapsed = (datetime.now(timezone.utc) - order.submitted_at).total_seconds()
    fill_ratio = float(order.filled_qty) / float(order.qty)
    
    if fill_ratio >= 1.0:
        return "complete"
    
    if elapsed < PartialFillPolicy.CANCEL_AFTER_SECONDS:
        return "wait"
    
    # Zeit überschritten
    if fill_ratio >= PartialFillPolicy.MIN_FILL_RATIO:
        # Akzeptable Teilfüllung — cancel rest
        await cancel_order(order.broker_order_id)
        await audit_log("partial_accepted", order.id, 
                        {"fill_ratio": fill_ratio})
        return "partial_accepted"
    else:
        # Zu wenig — cancel und abschreiben
        await cancel_order(order.broker_order_id)
        if PartialFillPolicy.ON_CANCEL == "liquidate":
            await submit_flatten_order(order.symbol, order.filled_qty, order.side)
        await audit_log("partial_failed", order.id, 
                        {"fill_ratio": fill_ratio})
        return "partial_failed"
```

### Für Signal-Entscheidungen

**Wichtig:** Wenn ein Signal nur zu 40% gefüllt wurde, darf das Composite-Score-System **nicht** das nächste Mal "den Rest" nachholen — das führt zu Position-Kreep. Erst im nächsten Signal-Zyklus neu bewerten.

```python
async def position_reconcile_before_signal(symbol, target_size, current_size):
    delta = target_size - current_size
    if abs(delta) < MIN_TRADE_SIZE:
        return None  # kein Trade
    if abs(delta / target_size) < 0.1:
        return None  # weniger als 10% Abweichung — nicht nachziehen
    # Sonst: neue Order auf Delta
    return build_intent(symbol, delta)
```

---

## 33.4 Order-Rejections systematisch behandeln

Die häufigsten Rejection-Gründe bei Alpaca und was zu tun ist:

| Grund | Bedeutung | Aktion |
|---|---|---|
| `insufficient_buying_power` | Equity + Margin reicht nicht | Position-Sizing-Bug — Signal zu groß skaliert |
| `wash_sale_block` | Verkauf mit Verlust, Rückkauf <30 Tage | Signal überspringen, Log-Eintrag für Tax-Reconcile |
| `pdt_restriction` | Pattern-Day-Trader unter 25k Equity | Day-Trade-Counter checken vor Submit |
| `position_halted` | Handel ausgesetzt | 60s warten, dann neu bewerten |
| `market_closed` | außerhalb Handelszeiten | Order mit `extended_hours=true` neu einreichen oder verwerfen |
| `invalid_symbol` | Ticker delisted oder typo | Symbol aus Universum entfernen |
| `exceeds_price_bands` | Limit zu weit vom NBBO | Limit anpassen oder als Market |
| `short_not_available` | Kein Borrow verfügbar | Short-Trade überspringen, als "unshortable" cachen für 24h |

```python
async def handle_rejection(order, reason, details):
    await db.execute("""
        UPDATE orders SET status = 'rejected', 
                          rejection_reason = $1, 
                          reject_code = $2,
                          raw_broker_response = $3
        WHERE id = $4
    """, reason, details.get("code"), details, order.id)
    
    if reason == "insufficient_buying_power":
        # Kritisch: Sizing-Bug oder race condition
        await alert_slack("🚨 Buying-Power-Reject", order=order)
        await pause_trading_for_symbol(order.symbol, minutes=30)
    elif reason == "wash_sale_block":
        await mark_wash_sale(order.symbol, order.signal_source_id)
    elif reason == "short_not_available":
        await cache_set(f"unshortable:{order.symbol}", "1", ttl=86400)
    elif reason == "pdt_restriction":
        # Day-Trade-Counter im Worker war veraltet
        await alert_slack("⚠ PDT-Reject — Counter-Bug?", order=order)
        await refresh_daytrade_counter()
```

**Wash-Sale-Precheck vor Submit** (Alpaca rejected erst, kostet Roundtrip):

```python
async def has_recent_loss_close(symbol: str, days: int = 30) -> bool:
    row = await db.fetchrow("""
        SELECT COUNT(*) as n FROM positions_closed 
        WHERE symbol = $1 
          AND closed_at > NOW() - INTERVAL '%s days'
          AND realized_pnl < 0
    """, symbol, days)
    return row["n"] > 0
```

---

## 33.5 Reconciliation

**Kern-Invariante:** Der interne State (Postgres-Positions-Tabelle) muss jederzeit dem Broker-State (Alpaca `/v2/positions`) entsprechen. Drift führt zu falschen Signal-Entscheidungen.

### Rekonziliations-Frequenzen

```python
RECONCILE_SCHEDULE = {
    "fast":   30,    # Sekunden — während Trading-Hours
    "normal": 300,   # 5 Min — während Trading-Hours, wenn idle
    "slow":   3600,  # 1 Stunde — außerhalb Trading-Hours
}
```

### Reconcile-Worker

```python
async def reconcile_positions():
    broker_positions = {p.symbol: p for p in await alpaca.list_positions()}
    internal_positions = await db.fetch("""
        SELECT symbol, qty, avg_entry_price FROM positions 
        WHERE qty != 0
    """)
    internal = {r["symbol"]: r for r in internal_positions}
    
    all_symbols = set(broker_positions.keys()) | set(internal.keys())
    drifts = []
    
    for sym in all_symbols:
        b = broker_positions.get(sym)
        i = internal.get(sym)
        b_qty = float(b.qty) if b else 0.0
        i_qty = float(i["qty"]) if i else 0.0
        
        if abs(b_qty - i_qty) > 1e-6:
            drifts.append({
                "symbol": sym,
                "broker_qty": b_qty,
                "internal_qty": i_qty,
                "delta": b_qty - i_qty
            })
    
    if drifts:
        await alert_slack(f"🔄 Position-Drift: {len(drifts)} Symbols", drifts)
        for d in drifts:
            await correct_internal_position(d)
    
    await metric_gauge("reconcile.drifts", len(drifts))
    await metric_gauge("reconcile.last_run_timestamp", time.time())
```

### Cash-Reconcile

```python
async def reconcile_cash():
    account = await alpaca.get_account()
    broker_cash = float(account.cash)
    broker_equity = float(account.equity)
    
    internal_cash = await db.fetchval("SELECT value FROM account WHERE key = 'cash'")
    drift = broker_cash - internal_cash
    
    if abs(drift) > 1.0:  # 1 USD Toleranz
        await alert_slack(f"💰 Cash-Drift: ${drift:.2f}", 
                         {"broker": broker_cash, "internal": internal_cash})
        await db.execute("UPDATE account SET value = $1 WHERE key = 'cash'", 
                        broker_cash)
```

### Die goldene Regel

**Bei jedem Drift ist der Broker die Wahrheit.** Nie umgekehrt. Der Broker hat tatsächlich Geld, dein System hat nur Buchungen.

---

## 33.6 Kill-Switch

**Wann:** Wenn Vertrauen in das System verloren geht. Dann sofort stoppen, nicht debuggen während Geld auf dem Spiel ist.

### Trigger-Regeln

```python
KILL_SWITCH_TRIGGERS = {
    # Hard-Kill (sofort alle Orders stoppen, existierende abbrechen)
    "hard": {
        "drawdown_daily_pct":     3.0,   # mehr als 3% Tages-Drawdown
        "drawdown_weekly_pct":    7.0,   # mehr als 7% Wochen-DD
        "consecutive_losses":     10,    # 10 Verlust-Trades in Folge
        "reconcile_drifts":       5,     # 5+ Drift-Positionen
        "reject_rate_1h":         0.3,   # >30% Orders abgelehnt/1h
        "data_feed_stale_sec":    300,   # News-Pipeline liefert 5min nichts
    },
    # Soft-Pause (keine neuen Entries, existierende bleiben)
    "soft": {
        "drawdown_daily_pct":     2.0,
        "consecutive_losses":     5,
        "psi_drift_features":     3,     # 3 Features mit PSI > 0.35
        "anthropic_budget_spent": 0.9,   # 90% Monatsbudget verbraucht
    },
}
```

### Implementierung

```python
class KillSwitch:
    def __init__(self, redis):
        self.redis = redis
    
    async def state(self) -> str:
        # "normal" | "soft" | "hard"
        return await self.redis.get("ks:state") or "normal"
    
    async def trip(self, level: str, reason: str, context: dict):
        await self.redis.set("ks:state", level)
        await self.redis.set("ks:tripped_at", datetime.now(timezone.utc).isoformat())
        await self.redis.set("ks:reason", reason)
        await db.execute("""
            INSERT INTO kill_switch_events (level, reason, context, tripped_at)
            VALUES ($1, $2, $3, NOW())
        """, level, reason, json.dumps(context))
        await alert_slack(f"🛑 Kill-Switch {level.upper()}: {reason}", context)
        
        if level == "hard":
            await cancel_all_open_orders()
    
    async def guard(self) -> bool:
        """True = safe to proceed, False = blocked."""
        s = await self.state()
        return s == "normal"
    
    async def guard_entry(self) -> bool:
        s = await self.state()
        return s in ("normal",)  # soft blockiert Entries
    
    async def reset(self, human_confirmation_token: str):
        expected = await self.redis.get("ks:reset_token")
        if human_confirmation_token != expected:
            raise PermissionError("Kill-Switch Reset braucht validen Token")
        await self.redis.set("ks:state", "normal")
        await alert_slack("✅ Kill-Switch zurückgesetzt")
```

**Entscheidende Regel:** Der Kill-Switch kann **nie automatisch zurückgesetzt werden**. Nach Hard-Trip muss ein Mensch den Token ausstellen. Das verhindert Restart-Loops.

### Daily-Check-Worker

```python
async def kill_switch_monitor():
    """Läuft jede Minute."""
    daily_dd = await compute_daily_drawdown_pct()
    weekly_dd = await compute_weekly_drawdown_pct()
    consec_losses = await count_consecutive_losses()
    drift_count = await count_reconcile_drifts()
    reject_rate = await compute_reject_rate_1h()
    stale_feed = await news_feed_staleness_seconds()
    
    ctx = {"daily_dd": daily_dd, "weekly_dd": weekly_dd, 
           "consec": consec_losses, "drifts": drift_count,
           "reject_rate": reject_rate, "stale_sec": stale_feed}
    
    # Hard-Triggers zuerst
    if daily_dd >= 3.0:
        await ks.trip("hard", "daily_drawdown_exceeded", ctx)
    elif weekly_dd >= 7.0:
        await ks.trip("hard", "weekly_drawdown_exceeded", ctx)
    elif consec_losses >= 10:
        await ks.trip("hard", "consecutive_losses", ctx)
    elif drift_count >= 5:
        await ks.trip("hard", "reconcile_drift", ctx)
    elif reject_rate >= 0.3:
        await ks.trip("hard", "high_reject_rate", ctx)
    elif stale_feed >= 300:
        await ks.trip("hard", "data_feed_stale", ctx)
    
    # Soft-Triggers
    elif daily_dd >= 2.0:
        if await ks.state() == "normal":
            await ks.trip("soft", "daily_drawdown_warning", ctx)
    elif consec_losses >= 5:
        if await ks.state() == "normal":
            await ks.trip("soft", "losses_warning", ctx)
```

---

## 33.7 Position-Sizing: Von Score zu Shares

Der Composite-Score liefert [-1, +1]. Die Meta-Labeling-Probability liefert [0, 1]. Conformal-Intervalle liefern Konfidenz. Daraus Shares zu berechnen ist die letzte Meile.

```python
class PositionSizer:
    """
    Baut aus Signal-Output eine konkrete Shares-Zahl.
    """
    def __init__(self,
                 max_position_pct=0.05,     # max 5% Equity pro Position
                 max_sector_pct=0.20,       # max 20% pro Sektor
                 max_leverage=1.0,          # kein Leverage Phase 1-3
                 vol_target_annual=0.15,    # 15% Ziel-Vola
                 min_trade_usd=200,         # unter 200 USD kein Trade
                 kelly_fraction=0.25):      # 25% Kelly
        self.max_position_pct = max_position_pct
        self.max_sector_pct = max_sector_pct
        self.max_leverage = max_leverage
        self.vol_target = vol_target_annual
        self.min_trade_usd = min_trade_usd
        self.kelly_fraction = kelly_fraction
    
    def size(self, signal, equity, current_positions, sector_exposures, 
             asset_vol_annual, conformal_width=None):
        """
        Returns: target_shares (signed: + for long, - for short)
        """
        if abs(signal.composite_score) < 0.1:
            return 0
        
        # 1. Edge aus Meta-Probability (Kalibrierte Kelly)
        p = signal.meta_probability
        b = 1.0  # assume 1:1 payoff vs SL via Triple-Barrier
        kelly = (p * b - (1 - p)) / b if p > 0.5 else 0
        kelly_sized = kelly * self.kelly_fraction
        
        # 2. Vol-Target-Sizing
        vol_scale = self.vol_target / max(asset_vol_annual, 0.05)
        
        # 3. Signal-Magnitude-Multiplier (aus 2D-Matrix)
        mag_mult = signal.matrix_multiplier  # [0.5, 1.5]
        
        # 4. Conformal-Discount bei breiten Intervallen
        if conformal_width is not None:
            # Enge Bänder → voll, breite Bänder → skaliert
            width_discount = np.clip(1.0 - conformal_width, 0.5, 1.0)
        else:
            width_discount = 1.0
        
        # 5. Dollar-Target
        dollar_target = (equity * kelly_sized * vol_scale * mag_mult * width_discount)
        dollar_target = np.sign(signal.composite_score) * abs(dollar_target)
        
        # 6. Cap pro Position
        max_dollar = equity * self.max_position_pct
        dollar_target = np.clip(dollar_target, -max_dollar, max_dollar)
        
        # 7. Cap pro Sektor
        sector = signal.sector
        current_sector_dollar = sector_exposures.get(sector, 0)
        sector_budget = equity * self.max_sector_pct
        sector_room = sector_budget - abs(current_sector_dollar)
        if abs(dollar_target) > sector_room:
            dollar_target = np.sign(dollar_target) * sector_room
        
        # 8. Von Dollar zu Shares
        price = signal.current_price
        shares = dollar_target / price
        shares = round(shares)  # whole shares (bei Alpaca Fractional optional)
        
        # 9. Min-Trade-Check
        if abs(shares * price) < self.min_trade_usd:
            return 0
        
        # 10. Delta zu aktueller Position
        current = current_positions.get(signal.symbol, 0)
        delta = shares - current
        if abs(delta * price) < self.min_trade_usd:
            return 0
        
        return delta
```

### Pre-Trade-Risk-Check (Gate vor Submit)

```python
async def pre_trade_risk_check(intent) -> tuple[bool, str | None]:
    # 1. Kill-Switch
    if not await ks.guard_entry():
        return False, "kill_switch_blocks"
    
    # 2. Halted Stock
    if await is_halted(intent.symbol):
        return False, "symbol_halted"
    
    # 3. Earnings in <24h und Size > 50% Normal
    if await earnings_within_hours(intent.symbol, 24):
        if abs(intent.qty * intent.price) > 0.5 * NORMAL_POSITION_USD:
            return False, "oversized_pre_earnings"
    
    # 4. Wash-Sale
    if intent.side == "buy" and await has_recent_loss_close(intent.symbol, 30):
        return False, "wash_sale_risk"
    
    # 5. Margin-Check (Simuliert)
    bp = await current_buying_power()
    if intent.qty * intent.price > bp * 0.9:
        return False, "insufficient_buying_power_predicted"
    
    # 6. Daily-Trade-Count (PDT)
    if await equity_below_pdt_threshold():
        if await day_trades_today() >= 3:
            return False, "pdt_risk"
    
    # 7. Symbol-Blacklist (unshortable, delisted, etc.)
    if intent.side == "sell_short" and await cache_get(f"unshortable:{intent.symbol}"):
        return False, "symbol_unshortable"
    
    return True, None
```

---

## 33.8 Alpaca-spezifische Fallen

**Aus dem aktuellen Alpaca-Verhalten, das in Doku unterschlägt:**

1. **Fractional Shares brauchen `qty` als Float, nicht Decimal.** `0.5` ja, `Decimal("0.5")` führt zu Rejection bei manchen SDK-Versionen.
2. **`time_in_force="gtc"` bei Market-Orders wird ignoriert** — Market ist immer IOC-artig.
3. **Extended-Hours-Orders** brauchen `extended_hours=True` UND `type="limit"` — Market in Extended-Hours gibt Rejection.
4. **`client_order_id` ist 48 Zeichen limitiert** — SHA256-full (64 chars) bricht. Nutze `[:20]` wie oben.
5. **Stop-Orders: `stop_price` muss ≥0.01 vom aktuellen Preis entfernt sein.** Zu nah → Rejection `stop_price_invalid`.
6. **Short-Sells: Alpaca prüft Borrow erst bei Submit**, nicht vorher. Kein `/v2/assets`-Flag für "aktuell shortbar".
7. **Options haben eigene Order-Endpoints** (`/v2/options/orders`), nicht gemischt mit Equity.
8. **Paper-Account-Balance ist separat von Live** — keine Migration möglich.
9. **WebSocket `trade_updates`-Stream** kann Nachrichten verlieren bei Reconnect. **Immer mit Polling als Safety-Net kombinieren.**
10. **PDT-Counter im Alpaca-Account** ist nicht real-time — 5-10 Sek Lag. Eigener Counter im System halten.

---

## 33.9 Websocket-Stream + Polling-Backup

```python
class OrderStatusStream:
    """
    Primär WebSocket, Polling alle 30s als Fallback.
    Reconciliation-Check alle 5min.
    """
    async def start(self):
        asyncio.create_task(self._ws_loop())
        asyncio.create_task(self._poll_loop())
        asyncio.create_task(self._reconcile_loop())
    
    async def _ws_loop(self):
        while True:
            try:
                async with alpaca.stream_trade_updates() as stream:
                    async for event in stream:
                        await self._apply_event(event)
            except Exception as e:
                await metric_counter("ws.reconnect")
                await asyncio.sleep(5)
    
    async def _poll_loop(self):
        """Polling der letzten 24h-Orders alle 30s."""
        while True:
            try:
                orders = await alpaca.list_orders(
                    status="all",
                    after=datetime.now(timezone.utc) - timedelta(hours=24),
                    limit=500
                )
                for o in orders:
                    await self._sync_order_status(o)
            except Exception:
                await metric_counter("poll.error")
            await asyncio.sleep(30)
    
    async def _reconcile_loop(self):
        while True:
            await reconcile_positions()
            await reconcile_cash()
            await asyncio.sleep(300)
```

**Warum Polling trotz WebSocket:** Bei Alpaca-Paper-API sind WebSocket-Gaps dokumentiert. Live-API ist stabiler, aber auch nicht perfekt. Polling als Netz ist billig.

---

## 33.10 EOD-Reconciliation und Reporting

**Jeden Tag nach Close** (16:30 ET):

```python
async def eod_reconciliation():
    # 1. Position-Reconcile hart
    await reconcile_positions()
    await reconcile_cash()
    
    # 2. Trades des Tages mit Broker-Trades abgleichen
    our_fills = await db.fetch("""
        SELECT * FROM order_events 
        WHERE event_type = 'filled' 
          AND event_timestamp::date = CURRENT_DATE
    """)
    broker_fills = await alpaca.get_activities(activity_types=["FILL"],
                                                date=date.today())
    
    our_count = len(our_fills)
    broker_count = len(broker_fills)
    if our_count != broker_count:
        await alert_slack(f"⚠ EOD-Mismatch: {our_count} intern, {broker_count} Broker")
    
    # 3. P&L berechnen
    realized = sum(f.realized_pnl for f in broker_fills)
    unrealized = sum(p.unrealized_pl for p in await alpaca.list_positions())
    total_equity = float((await alpaca.get_account()).equity)
    
    await db.execute("""
        INSERT INTO daily_pnl (date, realized, unrealized, equity, trade_count)
        VALUES ($1, $2, $3, $4, $5)
        ON CONFLICT (date) DO UPDATE SET 
          realized = EXCLUDED.realized,
          unrealized = EXCLUDED.unrealized,
          equity = EXCLUDED.equity,
          trade_count = EXCLUDED.trade_count
    """, date.today(), realized, unrealized, total_equity, our_count)
    
    # 4. Audit-Export für Steuer (später relevant)
    await export_trades_csv(date.today())
```

---

## 33.11 Position-Management

### Tabelle `positions`

```sql
CREATE TABLE positions (
    symbol TEXT PRIMARY KEY,
    qty NUMERIC NOT NULL,
    avg_entry_price NUMERIC NOT NULL,
    opened_at TIMESTAMPTZ NOT NULL,
    last_update_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    strategy_tag TEXT,                    -- welche Signal-Strategie hat geöffnet
    origin_signal_id UUID,
    stop_price NUMERIC,
    profit_target_price NUMERIC,
    max_holding_days INTEGER,             -- vertical barrier
    opened_on_regime TEXT,                -- Regime zum Open-Zeitpunkt
    original_thesis TEXT                  -- 1-2 Sätze, warum geöffnet
);

CREATE TABLE positions_closed (
    id UUID PRIMARY KEY,
    symbol TEXT NOT NULL,
    qty NUMERIC NOT NULL,
    avg_entry_price NUMERIC NOT NULL,
    avg_exit_price NUMERIC NOT NULL,
    realized_pnl NUMERIC NOT NULL,
    opened_at TIMESTAMPTZ NOT NULL,
    closed_at TIMESTAMPTZ NOT NULL,
    exit_reason TEXT,                     -- stop_hit, pt_hit, vertical_barrier, manual, regime_change
    strategy_tag TEXT,
    origin_signal_id UUID,
    shap_attribution JSONB               -- P&L-Attribution pro Feature
);
```

### Exit-Logik

```python
class ExitManager:
    """
    Drei Exit-Pfade: Stop-Loss, Profit-Target, Vertical-Barrier.
    Plus: Regime-Change-Exit.
    """
    async def check_exits(self):
        positions = await db.fetch("SELECT * FROM positions WHERE qty != 0")
        for p in positions:
            current = await get_current_price(p["symbol"])
            
            # Stop-Loss
            if p["stop_price"] and self._stop_hit(p, current):
                await self._close_position(p, "stop_hit", current)
                continue
            
            # Profit-Target
            if p["profit_target_price"] and self._pt_hit(p, current):
                await self._close_position(p, "pt_hit", current)
                continue
            
            # Vertical Barrier
            days_held = (datetime.now(timezone.utc) - p["opened_at"]).days
            if p["max_holding_days"] and days_held >= p["max_holding_days"]:
                await self._close_position(p, "vertical_barrier", current)
                continue
            
            # Regime-Change
            current_regime = await regime_classifier.classify(now)
            if current_regime == "crisis" and p["opened_on_regime"] != "crisis":
                # Long-Positionen aus Nicht-Crisis → skalieren runter
                if float(p["qty"]) > 0:
                    await self._reduce_position(p, 0.5, "regime_change", current)
```

---

## 33.12 Backtest vs. Live-Parität

**Regel:** Was im Backtest als Fill angenommen wird, muss in Live realistisch sein. Sonst wandert Paper-Sharpe Live nach unten.

### Realistische Kosten-Annahmen im Backtest

```python
class ExecutionCostModel:
    """
    Für Backtest: Konservative Fills.
    """
    def estimate_fill(self, side, symbol, qty, bar):
        """Liefert angenommenen Fill-Preis."""
        # 1. Spread-Kosten
        spread_bps = self._get_spread(symbol, bar)  # von ADV+Volatility-Modell
        
        # 2. Slippage: Market-Impact
        adv = self._get_adv(symbol, bar)
        participation_pct = qty * bar.close / adv
        slippage_bps = 10 * np.sqrt(participation_pct)  # Almgren-Chriss-Light
        
        # 3. SEC/TAF/FINRA-Fees für Sell
        fees_bps = 0.2 if side == "sell" else 0.0
        
        # 4. Summe
        total_bps = spread_bps/2 + slippage_bps + fees_bps
        
        # 5. Fill relativ zu Close
        direction = 1 if side == "buy" else -1
        fill_price = bar.close * (1 + direction * total_bps / 10000)
        
        return fill_price
```

**Spread-Proxy aus ADV und Vola:**

```python
def _get_spread(self, symbol, bar):
    adv = self._get_adv(symbol, bar)
    vol = bar.realized_vol_20d
    # Roll-1984-artiger Schätzer
    if adv > 100_000_000:     # Large-Cap
        base = 1.0  # 1 bp
    elif adv > 10_000_000:    # Mid-Cap
        base = 3.0
    else:                      # Small-Cap
        base = 10.0
    return base * (1 + 5 * vol)  # Vola-Aufschlag
```

### Live-vs-Backtest-Drift-Monitor

```python
async def track_execution_drift():
    """
    Vergleicht tatsächliche Fill-Preise mit Backtest-Erwartung.
    """
    fills_24h = await db.fetch("""
        SELECT o.symbol, o.side, o.filled_avg_price, o.composite_score, b.close
        FROM orders o 
        JOIN bars b ON b.symbol = o.symbol AND b.timestamp::date = o.filled_at::date
        WHERE o.filled_at > NOW() - INTERVAL '24 hours'
    """)
    
    drifts = []
    for f in fills_24h:
        expected = cost_model.estimate_fill(f.side, f.symbol, f.qty, f.close)
        drift_bps = (f.filled_avg_price - expected) / expected * 10000
        drifts.append(drift_bps)
    
    if drifts:
        mean_drift = np.mean(drifts)
        if abs(mean_drift) > 3.0:  # >3bp durchschnittlich
            await alert_slack(f"⚠ Execution-Drift: {mean_drift:.1f} bps vs. Backtest")
    
    await metric_gauge("exec.drift_mean_bps", np.mean(drifts) if drifts else 0)
```

---

## 33.13 BaseSignal-Plugin-Architektur

**Warum hier:** Die Execution-Ebene ist der Ort, an dem neue Signale eingehängt werden. Plugin-System macht das ohne Core-Änderung möglich.

```toml
# pyproject.toml in einem Signal-Package
[project]
name = "ata-signal-pead"
version = "0.1.0"

[project.entry-points."ata.signals"]
pead = "ata_signal_pead.signal:PEADSignal"
```

```python
# Basis-Interface in src/signals/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

@dataclass
class SignalOutput:
    symbol: str
    score: float                # [-1, +1]
    confidence: float           # [0, 1]
    metadata: dict
    features_used: list[str]
    computed_at: datetime

class BaseSignal(ABC):
    name: str = "base"
    version: str = "0.0.0"
    required_features: list[str] = []
    horizon_days: int = 5
    
    @abstractmethod
    async def compute(self, symbol: str, feature_store, now: datetime) -> SignalOutput | None:
        ...
    
    async def healthcheck(self) -> bool:
        return True
```

```python
# Loader in src/signals/registry.py
from importlib.metadata import entry_points

class SignalRegistry:
    def __init__(self):
        self._signals: dict[str, BaseSignal] = {}
    
    def load_all(self):
        for ep in entry_points(group="ata.signals"):
            try:
                cls = ep.load()
                inst = cls()
                if inst.name in self._signals:
                    raise ValueError(f"Duplicate signal name: {inst.name}")
                self._signals[inst.name] = inst
                log.info(f"Loaded signal: {inst.name} v{inst.version}")
            except Exception as e:
                log.error(f"Failed to load {ep.name}: {e}")
                # Einzelner Plugin-Crash darf System nicht killen
    
    def all(self) -> list[BaseSignal]:
        return list(self._signals.values())
    
    def get(self, name: str) -> BaseSignal | None:
        return self._signals.get(name)
```

**Regel:** Plugin-System von Tag 1 einführen. Später nachzurüsten kostet Wochen.

---

## 33.14 Umsetzungs-Checkliste

**Phase 1 (Monat 1-3):**
- [ ] Tabellen `orders`, `order_events`, `positions`, `positions_closed` erstellt
- [ ] `client_order_id` deterministisch via SHA256
- [ ] Idempotenter Submit mit Duplicate-Handling
- [ ] WebSocket `trade_updates` + Polling-Fallback
- [ ] Reconcile-Worker (fast/normal/slow)
- [ ] Cash-Reconcile täglich
- [ ] Pre-Trade-Risk-Check mit allen 7 Gates
- [ ] Kill-Switch mit Hard+Soft-Triggers
- [ ] EOD-Reconciliation + Daily-P&L-Tabelle
- [ ] BaseSignal-Plugin-Interface + Registry

**Phase 2 (Monat 4-6):**
- [ ] Partial-Fill-Policy mit Cancel-After-Timeout
- [ ] Wash-Sale-Precheck
- [ ] PDT-Counter eigenständig im System
- [ ] Position-Management mit Stop/PT/Vertical
- [ ] Regime-Change-Exit
- [ ] Position-Sizer mit Kelly + Vol-Target + Sektor-Cap
- [ ] Execution-Cost-Model im Backtest
- [ ] Live-vs-Backtest-Drift-Monitor

**Phase 3 (Monat 7-9):**
- [ ] Fractional Shares für kleine Tickets
- [ ] Options-Order-Path separat
- [ ] Extended-Hours-Handling
- [ ] Canary-Deployment-Integration (aus 32_VALIDIERUNG)

---

## 33.15 Standard-Fehler die hier verhindert werden

**Was Solo-Quant-Systeme üblicherweise falsch machen:**

1. **Kein `client_order_id`** → Retry verdoppelt Order.
2. **Status nur aus WebSocket** → Nachricht verloren, Order hängt.
3. **Keine Reconciliation** → Broker hat 100 Shares, System glaubt 80. Nächstes Signal kauft 20 nach, obwohl voll.
4. **Market-Orders bei Open/Close** → 30-50 bps Slippage statt 5.
5. **Kein Kill-Switch** → System traded weiter, während Data-Feed tot ist.
6. **Wash-Sale erst durch Broker-Reject** → unnötige API-Roundtrips, statistik-verzerrte Order-Log.
7. **Signal-Engine ruft direkt `alpaca.submit_order()`** → keine Abstraction-Schicht, Testing unmöglich.
8. **Backtest mit Close-Fills** → Paper-Sharpe 1.0, Live-Sharpe 0.4.

---

## Ehrliche Einschätzung

**Die Execution-Ebene ist die unsexy Schicht, aber die kritischste.** Signale kann man immer verbessern, Execution-Bugs kosten real Geld.

**Zwei Dinge sind die häufigsten Blindflecken:**
- Idempotenz wird oft vergessen, bis ein Retry-Sturm passiert.
- Reconciliation wird als "brauch ich später" abgetan, bis der erste Drift eine falsche Entry-Entscheidung erzeugt.

**Meine Regel:** Die Execution-Ebene ist "fertig", wenn du 48 Stunden lang den Server abschießen und neu starten kannst, ohne dass ein einziger Cent Diskrepanz zu Alpaca entsteht. Das ist das Benchmark.

**Was Paper-Trading dir nicht zeigt:**
- Borrow-Availability (Shorts können live rejecten, Paper nicht)
- Wash-Sale-Realität (Paper trackt nicht über Monate)
- Extended-Hours-Slippage (Paper ist sauber, Live brutal)

Diese drei lernst du erst Live. Deshalb: **mindestens 6 Monate sauberer Paper-Track-Record, bevor Live-Capital**. Und selbst dann: mit 10% anfangen, nicht Full-Size.
