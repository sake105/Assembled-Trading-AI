# 50 — Compliance und Recht (Personal-Use-only)

**Zweck:** Rechtliche Grundlagen für dich als **Einzelperson, die mit eigenem Geld tradet**. Keine B2B-SaaS-Themen, keine Lizenz-Fragen für Signal-Weiterverkauf — das wird erst relevant, wenn das System fertig ist.

**Stand:** April 2026, Deutschland. Ich bin kein Anwalt — für formale Rechtsberatung einen Fachanwalt konsultieren.

---

## Die gute Nachricht zuerst

**Als Einzelperson, die ausschließlich eigenes Geld handelt, brauchst du keine Lizenz.** Nicht von der BaFin, nicht vom Finanzamt (außer Steuererklärung), von niemandem. Automatisiertes Trading für den eigenen Account ist in Deutschland nicht erlaubnispflichtig.

**Was du brauchst:**
1. Die korrekte **steuerliche Behandlung** der Trading-Gewinne
2. **DSGVO-Hygiene** bei Nutzung von Drittdaten (Reddit, News)
3. **Daten-Lizenz-Bewusstsein** (yfinance, Scraping, ToS)

Das wars. Keine BaFin, kein §32 KWG, kein Haftungsdach — solange du nicht fremdes Geld verwaltest und keine Signale verkaufst.

---

## 50.1 Steuer: Abgeltungsteuer auf Kapitaleinkünfte

### Das Basis-Setup

**Alpaca ist ein US-Broker.** Das bedeutet:

- Keine automatische Abführung deutscher Kapitalertragsteuer
- **Du musst jeden Gewinn selbst in der Steuererklärung angeben** (Anlage KAP)
- Abgeltungsteuer: 25 % + 5,5 % Solidaritätszuschlag + ggf. Kirchensteuer
- Sparer-Pauschbetrag 2026: 1.000 EUR (für Alleinstehende) / 2.000 EUR (Ehepaare)

**Praktische Konsequenz:** Du musst sauber Buch führen, denn Alpaca liefert keine deutschen Steuerbescheinigungen.

### Was dein System tracken muss

Die `positions_closed`-Tabelle aus `33_EXECUTION_ORDERMANAGEMENT.md` reicht nicht aus. Du brauchst zusätzlich:

```sql
CREATE TABLE tax_lots (
    id UUID PRIMARY KEY,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,                    -- buy = opening
    qty NUMERIC NOT NULL,
    price_usd NUMERIC NOT NULL,
    price_eur NUMERIC NOT NULL,            -- EZB-Referenzkurs am Trade-Tag
    usd_eur_rate NUMERIC NOT NULL,
    trade_date DATE NOT NULL,
    trade_timestamp TIMESTAMPTZ NOT NULL,
    fees_usd NUMERIC DEFAULT 0,
    fees_eur NUMERIC DEFAULT 0,
    matched_against UUID REFERENCES tax_lots(id),  -- bei Close: Referenz auf Open-Lot
    realized_pnl_eur NUMERIC,              -- bei Close gefüllt
    holding_days INTEGER,                  -- bei Close gefüllt
    status TEXT NOT NULL                   -- open | closed
);

CREATE INDEX idx_tax_lots_symbol_status ON tax_lots(symbol, status);
CREATE INDEX idx_tax_lots_year ON tax_lots(EXTRACT(YEAR FROM trade_date));
```

### FIFO-Matching bei Schließung

Deutsche Steuerregel: **First-In-First-Out**. Bei Teilschließung einer Position wird die älteste Teilmenge zuerst geschlossen.

```python
async def match_fifo_on_close(symbol: str, qty_closed: float, 
                               exit_price_usd: float, usd_eur_rate: float,
                               trade_date: date):
    remaining = qty_closed
    open_lots = await db.fetch("""
        SELECT * FROM tax_lots 
        WHERE symbol = $1 AND status = 'open' AND side = 'buy'
        ORDER BY trade_date ASC, trade_timestamp ASC
    """, symbol)
    
    close_results = []
    for lot in open_lots:
        if remaining <= 0:
            break
        
        match_qty = min(float(lot["qty"]), remaining)
        
        # P&L in EUR, nicht USD!
        entry_eur = match_qty * float(lot["price_eur"])
        exit_eur = match_qty * exit_price_usd * usd_eur_rate
        pnl_eur = exit_eur - entry_eur - float(lot["fees_eur"])
        
        holding_days = (trade_date - lot["trade_date"]).days
        
        close_results.append({
            "lot_id": lot["id"],
            "qty": match_qty,
            "pnl_eur": pnl_eur,
            "holding_days": holding_days,
        })
        
        if match_qty == float(lot["qty"]):
            # komplett geschlossen
            await db.execute("""
                UPDATE tax_lots SET status = 'closed', 
                                    realized_pnl_eur = $1,
                                    holding_days = $2
                WHERE id = $3
            """, pnl_eur, holding_days, lot["id"])
        else:
            # teilweise — Split
            new_remaining = float(lot["qty"]) - match_qty
            await db.execute("""
                UPDATE tax_lots SET qty = $1 WHERE id = $2
            """, new_remaining, lot["id"])
            await db.execute("""
                INSERT INTO tax_lots (symbol, side, qty, price_usd, price_eur,
                                      usd_eur_rate, trade_date, trade_timestamp,
                                      fees_usd, fees_eur, status, realized_pnl_eur,
                                      holding_days)
                VALUES ($1, 'buy', $2, $3, $4, $5, $6, $7, 0, 0, 'closed', $8, $9)
            """, symbol, match_qty, float(lot["price_usd"]), float(lot["price_eur"]),
                float(lot["usd_eur_rate"]), lot["trade_date"], lot["trade_timestamp"],
                pnl_eur, holding_days)
        
        remaining -= match_qty
    
    return close_results
```

### USD/EUR-Umrechnung

**Regel:** Jeder USD-Trade wird mit dem **EZB-Referenzkurs des Trade-Tages** in EUR umgerechnet. Nicht mit dem Tageskurs irgendeiner Börse, nicht mit dem Jahresdurchschnitt.

```python
import httpx

async def get_ecb_usd_eur_rate(d: date) -> float:
    """EZB-Referenzkurs. Wochenends/Feiertage → letzter Handelstag."""
    # Cache, weil EZB rate-limiting hat
    cached = await redis.get(f"ecb:usd_eur:{d.isoformat()}")
    if cached:
        return float(cached)
    
    # EZB SDW API
    url = f"https://data-api.ecb.europa.eu/service/data/EXR/D.USD.EUR.SP00.A"
    params = {"startPeriod": d.isoformat(), "endPeriod": d.isoformat(), "format": "jsondata"}
    async with httpx.AsyncClient() as c:
        r = await c.get(url, params=params, timeout=10)
        # Bei Wochenende: letzten verfügbaren Wert holen
        if r.status_code == 404 or len(r.json().get("dataSets", [])) == 0:
            # Fallback: 3 Tage zurückgehen
            for offset in range(1, 5):
                prev = d - timedelta(days=offset)
                return await get_ecb_usd_eur_rate(prev)
    
    data = r.json()
    # EZB gibt EUR→USD, wir wollen 1 USD = x EUR
    usd_per_eur = float(data["dataSets"][0]["series"]["0:0:0:0:0"]["observations"]["0"][0])
    eur_per_usd = 1.0 / usd_per_eur
    
    await redis.setex(f"ecb:usd_eur:{d.isoformat()}", 86400 * 30, eur_per_usd)
    return eur_per_usd
```

### Jahresabschluss-Report

Einmal im Jahr (Dezember oder spätestens vor der Steuererklärung):

```python
async def generate_tax_report(year: int) -> dict:
    closed_lots = await db.fetch("""
        SELECT symbol, qty, price_eur, realized_pnl_eur, holding_days,
               trade_date, matched_against
        FROM tax_lots 
        WHERE status = 'closed' 
          AND EXTRACT(YEAR FROM trade_date) = $1
        ORDER BY trade_date
    """, year)
    
    total_pnl = sum(float(l["realized_pnl_eur"]) for l in closed_lots)
    wins = [l for l in closed_lots if float(l["realized_pnl_eur"]) > 0]
    losses = [l for l in closed_lots if float(l["realized_pnl_eur"]) < 0]
    
    return {
        "year": year,
        "total_realized_pnl_eur": total_pnl,
        "total_wins_eur": sum(float(l["realized_pnl_eur"]) for l in wins),
        "total_losses_eur": sum(float(l["realized_pnl_eur"]) for l in losses),
        "trade_count": len(closed_lots),
        "wins_count": len(wins),
        "losses_count": len(losses),
        "after_sparer_pauschbetrag": max(0, total_pnl - 1000),
        "estimated_tax_eur": max(0, (total_pnl - 1000) * 0.26375),  # 25% + Soli
    }
```

Dieser Report wandert in die Anlage KAP. Details wie Broker-Name, Kontonummer, Einzel-Trades **müssen nicht** in der Erklärung stehen, aber auf Rückfrage des Finanzamts vorzeigbar sein.

### Wichtige Besonderheiten

**Verlustverrechnungstopf für Aktien:** Verluste aus Aktien-Trades können nur gegen Gewinne aus Aktien-Trades verrechnet werden (nicht gegen Zinsen oder Dividenden). Dein Tax-Report muss das trennen.

**Stillhaltergeschäfte / Optionen:** Wenn du Options-Trading aufnimmst (Phase 3), andere Kategorie — **Termingeschäfte** mit 20.000 EUR Verlustverrechnungs-Obergrenze pro Jahr (seit 2021, mehrfach vor Gerichten, Stand 2026 noch gültig). Das ist ein Grund, Options-Trading erst ernsthaft anzugehen, wenn du weißt, was du tust.

**Dividenden:** Alpaca zahlt US-Dividenden aus, 15 % US-Quellensteuer wird einbehalten (DBA). Die 15 % werden in Deutschland auf die Abgeltungsteuer angerechnet. Dein System sollte Dividenden-Events tracken.

```sql
CREATE TABLE dividends (
    id UUID PRIMARY KEY,
    symbol TEXT NOT NULL,
    pay_date DATE NOT NULL,
    gross_usd NUMERIC NOT NULL,
    us_withholding_usd NUMERIC NOT NULL,
    net_usd NUMERIC NOT NULL,
    usd_eur_rate NUMERIC NOT NULL,
    gross_eur NUMERIC NOT NULL,
    us_withholding_eur NUMERIC NOT NULL,
    net_eur NUMERIC NOT NULL
);
```

---

## 50.2 DSGVO: Drittdaten korrekt behandeln

**Dein System speichert potenziell personenbezogene Daten:**

- Reddit-Posts mit Usernamen (bei PRAW-Scraping)
- News-Autoren
- Social-Media-Cashtags mit User-IDs (Bluesky, Stocktwits)
- CEO-Namen aus SEC-Form-4

**Solange du Einzelperson bist und nur für dich verarbeitest**, greift die **Haushaltsausnahme nach Art. 2 Abs. 2 lit. c DSGVO** — private Nutzung ist ausgenommen. Aber:

1. Die Ausnahme ist eng auszulegen.
2. Sobald du Daten strukturiert verarbeitest, in Cloud speicherst, oder auch nur an einen Freund weitergibst, wird es schnell keine rein private Nutzung mehr.

### Empfohlenes Vorgehen

**Nicht mehr speichern, als nötig.** Der Plan macht das ohnehin:

- Reddit: Mention-Velocity als Aggregat, **nicht** User-Posts in DB.
- News: Sentiment-Score + Cluster-ID, Headline und URL als Hash, **kein** Author-Name persistiert.
- Social: Cashtag-Counts, **keine** User-Identifikatoren.

**Konkrete Datenhaltungs-Regeln:**

```python
# NICHT TUN:
# CREATE TABLE reddit_posts (user_id, username, post_body, timestamp...)

# STATT DESSEN:
# Aggregat-Tabelle, keine personenbezogenen Daten
CREATE TABLE reddit_sentiment_daily (
    date DATE NOT NULL,
    ticker TEXT NOT NULL,
    mention_count INTEGER,
    bullish_ratio NUMERIC,
    bearish_ratio NUMERIC,
    subreddit TEXT,
    PRIMARY KEY (date, ticker, subreddit)
);

# Raw-Posts nur im Redis-Cache mit TTL 24h, danach löschen
await redis.setex(f"reddit:post:{post_id}", 86400, json.dumps(post_anonymized))
```

**Hash-Anonymisierung für Author-Tracking** (falls mal nötig):

```python
import hashlib

def pseudonymize_user(user_id: str) -> str:
    """Irreversibler Hash mit Salz."""
    salt = os.environ["ATA_PSEUDO_SALT"]
    return hashlib.sha256(f"{salt}:{user_id}".encode()).hexdigest()[:16]
```

### Auskunfts- und Löschanfragen

**Realistische Szenario:** Ein Reddit-User fordert Löschung seiner Daten. Wenn du nur aggregierte Daten hältst → kein Problem, er ist nicht identifizierbar. Wenn du Rohdaten hältst → du musst reagieren können.

**Der saubere Weg:** Keine Rohdaten speichern, die zur Identifikation führen. Damit erübrigt sich die Frage.

---

## 50.3 Daten-Lizenzen: ToS-Grauzonen

Verschiedene Datenquellen haben verschiedene Lizenzen. Für **reine Privatnutzung** sind die Regeln deutlich liberaler als bei Weiterverkauf.

### Die Stufen

| Stufe | Was erlaubt | Beispiele |
|---|---|---|
| **Public Domain** | alles | SEC EDGAR, FRED, NOAA, USPTO |
| **Explizit kommerziell OK** | alles für Private + Gewerblich | Alpaca, Finnhub Free (mit Limits), CBOE CSVs |
| **CC-BY-SA** | alles mit Attribution | Wikipedia |
| **Privat toleriert** | eigenes Trading OK, kein Resale | yfinance (Yahoo-ToS), Stooq, Stocktwits Public-Endpoints |
| **Commercial License erforderlich** | für Resale | Reddit (seit Juli 2023), EODHD-SaaS, Finnhub-SaaS |
| **Scraping verboten** | keinesfalls nutzen | LinkedIn, Indeed, Glassdoor |

### Was das praktisch für dich heißt

**Solange du ausschließlich eigenes Geld handelst und keine Daten weiterverkaufst**, bist du hier in der grünen Zone:

- ✅ yfinance für EU-Ticker-Fallback
- ✅ Stooq für EOD-Fallback
- ✅ Reddit via PRAW für Sentiment-Analyse (private use)
- ✅ Wikipedia-Page-Views
- ✅ Stocktwits Public-Endpoints
- ✅ SEC/FRED/FINRA
- ✅ Alle Commercial-Free-Tiers (Alpaca, Finnhub, Alpha Vantage, EODHD Free)

**Was du trotzdem vermeiden solltest:**

- ❌ LinkedIn/Indeed/Glassdoor-Scraping (klarer ToS-Verstoß, rechtlich riskant auch privat)
- ❌ Nitter-Instances für Twitter-Ersatz (blockiert, oft gehackt)
- ❌ Seeking Alpha-Scraping (aggressive Abuse-Detection)
- ❌ `pytrends` (Google liefert bekanntermaßen gefälschte Daten an Bot-Detection)

### Attribution für Wikipedia

Wikipedia-Daten sind CC-BY-SA. In deinem internen Dashboard reicht ein Footer:
> "Attention-Daten via Wikimedia, CC-BY-SA"

Das ist nicht für die Öffentlichkeit, aber hygienisch.

### Rate-Limits und technische Compliance

Selbst bei tolerierten Quellen: **respektiere Rate-Limits**. Exzessives Scraping führt zu IP-Ban, selbst wenn es technisch erlaubt wäre. Dein System soll nach 5 Jahren immer noch laufen, nicht nach 3 Monaten im Ban-Sumpf.

```python
# Faustregeln
YFINANCE_MAX_REQ_PER_HOUR = 500        # konservativ
STOOQ_MAX_REQ_PER_MINUTE = 10
WIKIPEDIA_MAX_REQ_PER_SEC = 100        # offiziell, kein Problem
REDDIT_MAX_REQ_PER_MINUTE = 60         # PRAW enforced
SEC_EDGAR_MAX_REQ_PER_SEC = 10         # offiziell, bei Verstoß ban
```

---

## 50.4 Paper vs. Live: Der Übergang

**Paper-Trading ist kein Trading.** Aber wenn du irgendwann auf Live umstellst:

### Checkliste vor dem ersten Live-Cent

- [ ] **Mindestens 6 Monate Paper-Track-Record** mit Sharpe > 0.5 nach realistischen Kosten
- [ ] **Tax-Lot-Tabelle** produktiv und getestet
- [ ] **USD/EUR-Rate-Pipeline** läuft täglich
- [ ] **Kill-Switch-Trigger** in den letzten 60 Tagen mindestens einmal ausgelöst und korrekt behandelt
- [ ] **Reconciliation** null Drift über 30 Tage
- [ ] **Disaster-Recovery-Test** bestanden (siehe `51_INCIDENT_PLAYBOOK.md`)
- [ ] **Klares Start-Kapital-Limit** — nicht dein gesamtes Erspartes
- [ ] **Max-Loss-Regel** schriftlich — bei X% Gesamtverlust → Live-Trading stoppen, Review
- [ ] **Steuerberater-Kontakt** — bevor der erste Gewinn realisiert wird

### Start-Kapital-Regel für Privatanleger

Eine bewährte Regel: **Starte mit 10 % deines theoretischen Ziel-Kapitals**. Das System hat noch keine echten Live-Daten gesehen, es wird Drifts geben. Bei 10 % bleibt der maximale Schaden verkraftbar.

**Nach 3 Monaten Live ohne Probleme:** auf 30 % hochziehen. Nach weiteren 3: auf 60 %. Nach einem Jahr: Vollkapital. Wer nach 2 Wochen "alles reinballert", verliert statistisch mit sehr hoher Wahrscheinlichkeit.

### Was rechtlich beim Übergang zu Live passiert

**Nichts Neues.** Du brauchst keine Genehmigung, kein Lizenz-Upgrade, keine Meldung. Paper zu Live ist ein technischer Switch (API-Key), kein rechtlicher.

**Einzige Änderung:** Ab jetzt entstehen echte steuerpflichtige Einkünfte. Tax-Lot-Pipeline muss sauber laufen.

---

## 50.5 PDT-Regel (Alpaca-spezifisch)

**Pattern Day Trader Rule (SEC):** Wer innerhalb von 5 Handelstagen 4 oder mehr Day-Trades macht **und** unter 25.000 USD Equity hat, wird als PDT eingestuft und gelockt.

**Konsequenz:** 90 Tage lang nur Cash-Account (T+1 Settlement, kein echtes Day-Trading möglich).

**Für dich relevant:**

1. Dein System muss **Day-Trade-Counter** führen (siehe `33_EXECUTION_ORDERMANAGEMENT.md` §33.6).
2. Wenn Equity < 25.000 USD: **max 3 Day-Trades pro rolling 5-Tage-Fenster**.
3. **Overnight-Holds** zählen nicht als Day-Trade.

```python
async def can_daytrade() -> tuple[bool, str]:
    equity = float((await alpaca.get_account()).equity)
    if equity >= 25_000:
        return True, "pdt_threshold_met"
    
    count = await db.fetchval("""
        SELECT COUNT(*) FROM orders
        WHERE filled_at::date > CURRENT_DATE - INTERVAL '5 days'
          AND EXISTS (
            SELECT 1 FROM orders AS closer
            WHERE closer.symbol = orders.symbol
              AND closer.side != orders.side
              AND closer.filled_at::date = orders.filled_at::date
              AND closer.filled_at > orders.filled_at
          )
    """)
    if count >= 3:
        return False, f"pdt_limit_reached ({count}/3)"
    return True, f"pdt_ok ({count}/3)"
```

**Strategische Konsequenz:** Mit Start-Kapital unter 25k USD ist deine Strategie **notwendigerweise swing-orientiert** (Overnight-Holds, nicht Intraday). Das System aus dem Plan ist darauf ausgelegt.

---

## 50.6 Dokumentation für den Notfall

**Falls dir etwas passiert** (Krankheit, Unfall): Jemand muss das System stoppen können.

### Das Wallet-Testament

Eine Datei `EMERGENCY_STOP.md`, verschlüsselt, einer vertrauten Person gegeben:

```markdown
# Emergency Stop — Assembled Trading AI

Wenn du das liest: Hans kann das System gerade nicht bedienen. 
Das System tradet automatisch auf Alpaca.

## Sofort-Maßnahme: Alle Positionen schließen

1. Gehe zu https://app.alpaca.markets/paper/dashboard (oder /live)
2. Login: [E-Mail], Passwort: [in Password-Manager-Export]
3. 2FA-Recovery-Codes: [in safe]
4. Tab "Positions" → "Close All Positions"
5. Tab "Orders" → "Cancel All"

## Server stoppen

1. SSH auf Hetzner-Server:
   - Host: [IP]
   - Key: [Pfad zu SSH-Key in Backup]
2. `docker compose -f /home/trading/ata/compose.yml down`
3. Fertig. System ist aus.

## Kontaktinformationen

- Steuerberater: [Name, Tel]
- Broker-Support Alpaca: support@alpaca.markets
- Falls Unklarheiten: [Trusted Friend mit Finanz-Know-How]
```

Diese Datei **nicht** in Git. Nicht in Cloud-Speicher ohne Verschlüsselung. Ausdruck im Tresor oder bei Vertrauensperson.

---

## 50.7 Umsetzungs-Checkliste

**Phase 1 (sofort):**
- [ ] Tax-Lot-Tabelle `tax_lots` erstellt
- [ ] USD/EUR-Rate-Pipeline (EZB) mit Cache
- [ ] FIFO-Matching bei Position-Close
- [ ] `dividends`-Tabelle für Dividenden-Tracking
- [ ] Keine personenbezogenen Daten in persistenten DBs
- [ ] Pseudonymisierungs-Helper für Social-Daten
- [ ] Rate-Limit-Compliance pro Datenquelle dokumentiert

**Phase 2 (Monat 3-4):**
- [ ] Jahresabschluss-Report-Generator
- [ ] Tax-Lot-Einträge für alle historischen Trades rückwirkend (falls nötig)
- [ ] Reddit-Scraping nur noch als Aggregat
- [ ] Wikipedia-Attribution im internen Dashboard
- [ ] PDT-Counter eigenständig und live-accurate

**Phase 3 (vor Live-Trading):**
- [ ] Steuerberater-Gespräch, Tax-Report-Format abgestimmt
- [ ] 6-Monats-Paper-Track-Record nachgewiesen
- [ ] 10%-Start-Kapital-Regel dokumentiert und eingeplant
- [ ] Max-Loss-Regel schriftlich (z.B. "bei -15% Gesamtverlust Live-Stop")
- [ ] `EMERGENCY_STOP.md` bei Vertrauensperson hinterlegt

**Jährlich (Dezember):**
- [ ] Tax-Report für aktuelles Jahr generiert
- [ ] In Anlage KAP übertragen
- [ ] Broker-Kontoauszüge als Nachweis-PDF archiviert
- [ ] DBA-Anrechnung US-Quellensteuer geprüft

---

## 50.8 Was du NICHT brauchst

Zur Klarstellung — alles das gilt **nicht** für dich als Privatperson:

- ❌ §32 KWG Erlaubnis (nur bei Fremdverwaltung relevant)
- ❌ Anzeige nach WpIG
- ❌ Haftungsdach
- ❌ BaFin-Meldung
- ❌ MiFID-II-Kompetenznachweis
- ❌ AGB oder Widerrufsrecht (du hast keine Kunden)
- ❌ Impressum auf deinem Dashboard
- ❌ Datenschutzerklärung auf deinem Dashboard (solange intern)
- ❌ IT-Sicherheits-Zertifizierung
- ❌ Compliance-Officer

Alles das wird erst relevant, wenn das System für andere läuft. Solange du die einzige Nutzerin des Systems bist, ist das rechtliche Setup minimal.

---

## Ehrliche Einschätzung

**Der rechtliche Aufwand für Personal-Quant ist klein.** Das ist bewusst so von deutschen Gesetzgeber — was Privatpersonen mit eigenem Geld tun, ist in weiten Teilen Vertragsfreiheit + Steuerrecht.

**Die beiden einzigen Fallen:**

1. **Steuer-Dokumentation vergessen** — und dann im März für 2.000 Trades rückwirkend die Tax-Lots nachpflegen. Das ist die häufigste Ursache für Stress bei Retail-Algotradern.
2. **Sich zu früh "professionell" fühlen** — Freund fragt "läuft dein System? Kannst du für mich auch...?" und du sagst ja. Ab da brauchst du BaFin-Genehmigung. Wir halten uns daran: **nur eigenes Geld**.

**Regel zum Mitschreiben:** Bis das System fertig ist, existiert **keine** Kundenschnittstelle. Keine API-Keys für Freunde, keine Telegram-Signale an Bekannte, keine Beta-Tester. Auch nicht "nur als Test". Das sind die Momente, in denen Privat zu Regulated wird.
