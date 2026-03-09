Strategy Policy (v1.x) — Assembled Trading AI
=============================================

Diese Datei definiert die **Strategy Policy** als Vertrag zwischen System und Operator. Sie beschreibt das Zielbild, Scope v1, Risk-Kernregeln, State-Machine-Prinzip, Health/QC-Gates sowie bewusst aufgeschobene Security-Themen.

`configs/policy.yaml` ist die parametrisierbare Single Source of Truth; dieses Dokument beschreibt die Logik auf Policy-Ebene.

---

## 1) Zielbild

- **Renditeband als Benchmark**:
  - Ziel-CAGR (Benchmark): **20–30% p.a.**
  - Dieses Band ist **kein Stop-Kriterium**, sondern ein **Vergleichsmaßstab** für Strategien/Varianten.
- **Steuerung primär über Risiko-Grenzen**:
  - Maximaler Drawdown (Soft/Hard/Kill)
  - Ziel-Volatilität des Portfolios (annualisiert)
  - Turnover-Budget (wie viel Portfolio pro Zeitraum „umgedreht“ werden darf)

---

## 2) Scope v1

- **Instrumente v1.x**
  - Fokus auf **hochliquide ETFs** (Core-Bausteine).
  - Keine Hebelprodukte (kein Margin, keine Leveraged ETFs, keine Derivate) in v1.x.
- **Zeitebene**
  - **EOD-first** (End-of-Day Backtests und Signale).
  - Intraday (z.B. 5min) ist optional und wird später als Erweiterung betrachtet.

---

## 3) Risk-Kernregeln (Policy, keine Implementierung)

### 3.1 Max Drawdown (DD)

- **Soft DD**:
  - Ab einem bestimmten Drawdown (z.B. -15%) wird die Brutto-Exposure reduziert (z.B. durch Skalierung der Zielgewichte).
- **Hard DD**:
  - Ab einem tieferen Drawdown (z.B. -25%) wechselt das System in einen **Crisis-/PAUSE-nahen** Zustand (z.B. nur noch Defensiv/Watch).
- **Kill DD**:
  - Ab einem kritischen Drawdown (z.B. -35%) wird das System in **PAUSE** versetzt, bis eine manuelle Review stattgefunden hat.

### 3.2 Ziel-Volatilität

- Das Portfolio zielt auf eine **annualisierte Volatilität** im Zielband (z.B. 15–20%).
- Risiko-Steuerung erfolgt über Exposure-Skalierung (z.B. Position-Weights, Cash-Anteil), nicht über Stoppen bei Erreichen der Rendite.

### 3.3 Turnover-Budget & Klumpen/Korrelation

- **Turnover-Budget**:
  - Pro Zeitraum (z.B. Woche/Tag) gibt es ein Budget, wie viel Notional gehandelt werden darf (Turnover-Grenzen).
  - Exzessiver Turnover wird begrenzt, um Kosten und Slippage zu kontrollieren.
- **Klumpen-/Korrelation-Guard**:
  - Maximalgewicht pro Sektor / Korrelations-Cluster, um Konzentrationsrisiken zu vermeiden.
  - Beispiel: Max Sektor-Gewicht, Max Cluster-Gewicht für hoch korrelierte ETFs.

### 3.4 Risk Overlays (GeoRisk, Profit Lock, Turnover Budget)

**Gemeinsames Prinzip:**

- Alle Overlays sind **rein skalierend/gate-basiert**:
  - Sie erzeugen **keine eigenen Signale**.
  - Sie verändern **nur Exposure / Trades**, die aus der Kern-Strategie kommen.
  - Reihenfolge (konzeptionell): **Signale → Position Sizing → Risk Overlays → Orders**.

- **GeoRisk Overlay (intel-driven, news-basiert)**:
  - Quelle: News-/Geo-Intel (`news_geo`, `geo_score`, `geo_confidence`, Risk-State).
  - Policy: `georisk_overlay.*` in `configs/policy.yaml`.
  - Wirkung: Berechnet einen **Exposure-Multiplikator** \\(m_{geo} \in [0,1]\\), der die Zielgewichte/Target-Qtys der Strategie skaliert.
  - Typische Beispiele:
    - WATCH: \\(m_{geo} \approx 1.0\\) (keine Anpassung).
    - ACTIVE mit hohem Geo-Risk: \\(m_{geo} < 1.0\\) (Exposure-Reduktion).
    - PAUSE: \\(m_{geo} = 0.0\\) (faktisch Flat, keine neuen Risikopositionen).

- **Profit Lock (equity-driven, PnL-basiert)**:
  - Quelle: **Equity Curve** des Portfolios (Backtest/Paper: Equity-Historie; Live: später).
  - Policy: `profit_lock.*` in `configs/policy.yaml` (z.B. `lookback_days`, `trigger_return`, `multiplier_on_trigger`, `floor`, `cooldown_days`).
  - Logik:
    - Wenn die Equity über das Lookback-Fenster (z.B. 20 Tage) mindestens `trigger_return` (z.B. +8%) erreicht,
      wird ein **Exposure-Multiplikator** \\(m_{pl} \le 1.0\\) aktiviert.
    - Dieser Multiplikator gilt für eine **Cooldown-Periode** (`cooldown_days`), auch wenn das Portfolio zwischenzeitlich leicht zurückkommt.
    - Ein `floor` stellt sicher, dass das Overlay allein die Exposure nicht unter einen Minimalwert (z.B. 50%) drückt.
  - Wirkung: **Soft Lock-In** von Gewinnen durch temporäre Reduktion der Brutto-Exposure; keine Zwangs-Realisation, keine Stops.

- **Turnover Budget (cost-driven, trade-basiert)**:
  - Quelle: Delta zwischen aktuellen Positionen und Ziel-Positionen (inkl. Preise).
  - Policy: `turnover_budget.*` in `configs/policy.yaml` (z.B. `mode = daily|weekly`, `cap`, `behavior = scale|block`, QC-Einstellungen).
  - Logik:
    - Es wird ein **Turnover** geschätzt (z.B. \\(\\sum |\\Delta \\text{Weight}| / 2\\) oder notional-basiert).
    - Liegt der geschätzte Turnover **unterhalb** des Caps: Overlay ist **no-op**.
    - Liegt er **oberhalb** des Caps:
      - `behavior = "scale"`: die Trade-Deltas werden proportional skaliert, so dass der effektive Turnover ≈ Cap ist.
      - `behavior = "block"`: Ziel-Positionen werden auf aktuelle Positionen zurückgesetzt (≈ keine Trades).
    - QC: fehlende Preise können z.B. dazu führen, dass das Overlay konservativ „auf 0“ skaliert oder blockt.
  - Wirkung: **Kosten-Gate** vor Order-Generierung; schützt vor exzessivem Turnover durch Signalrauschen oder Rebalancing-Spikes.

### 3.4 Event / Earnings Filter (optional)

- Optionaler Guard, der Positionen/Orders rund um Earnings/Events ausdünnt oder pausiert.
- Wird als Policy-Eintrag geführt (z.B. „event_filter_enabled“), Implementierung erfolgt später.

---

## 4) State-Machine Prinzip

Das System folgt einer einfachen, aber expliziten **State Machine**:

- **WATCH**:
  - Beobachtungsmodus; Signale werden gesammelt, aber Handelsaktivität ist reduziert oder aus.
  - Typisch bei „DEGRADED“ Data/News/Disclosures.
- **ACTIVE**:
  - Normaler Trading-Modus (innerhalb aller Risk-Limits).
- **COOLDOWN**:
  - Reduzierte Exposure nach starken Bewegungen / nach Exit aus ACTIVE; dient zur Stabilisierung.
- **PAUSE**:
  - Kein neues Risiko; nur Monitoring, keine neuen Orders (bis manuell aufgehoben).

### 4.1 Aktivierung (ACTIVE)

- **Trigger**:
  - Kombination aus **Geo-Trigger / News-Regime / Macro-Regime** (z.B. „Risk-On“) und
  - **Market-Stress-Confirmation** (z.B. Volatilität, Spreads, Drawdown-Metriken).
- **Bedingungen**:
  - Health-Gates (News/Disclosures/MarketData) nicht im Modus „kritisch“.
  - Risk-Limits nicht verletzt (MaxDD, Turnover, Konzentration).

### 4.2 Deaktivierung / Cooldown

- **Deaktivierung** (ACTIVE → WATCH/PAUSE):
  - Bei Verletzung von Hard-DD, Health-Gates (z.B. MARKETDATA=DEGRADED), oder schwerwiegenden Operational-Issues.
- **Cooldown** (ACTIVE → COOLDOWN):
  - Nach Phasen sehr hoher Aktivität oder außerordentlicher Gewinne/Verluste.
  - Cooldown definiert eine Zeitspanne oder Anzahl Bars, in der Exposure begrenzt wird, bevor wieder ACTIVE erlaubt ist.

---

## 5) Health / QC Gates

Health-/QC-Gates definieren, wie sich das System verhält, wenn Eingabedaten/Feeds degradiert sind.

- **News-Health (z.B. NEWS_V1)**
  - Wenn Health-Status **DEGRADED**: nur noch **WATCH-only** (keine neuen aggressiven Positionen).
- **Disclosures-Health**
  - Wenn DEGRADED: ebenfalls **WATCH-only** oder strengere Filter auf Event-Signale.
- **MarketData-Health**
  - Wenn DEGRADED: konservativer Modus bis hin zu **PAUSE** (je nach Schwere).

**Audit-Logging Pflicht:**
- Jede State-Transition (WATCH/ACTIVE/COOLDOWN/PAUSE) und jede Health-Gate-Entscheidung muss auditierbar sein (Log/Artifact).
- Policy- und Config-Snapshots werden pro relevanter Run-ID gesichert.

---

## 6) Deferred: Security / Secrets / .env

Folgende Themen sind **bewusst aufgeschoben** und werden in späteren Sprints adressiert:

- Hardening von **Secrets / .env / Credentials**.
- Einheitliche **Secret-Scanning-Regeln in CI** (z.B. GitHub Actions, pre-commit).
- Klare Trennung von:
  - Code/Repo (öffentlich oder geteilt),
  - Konfiguration,
  - Secrets (getrennt verwaltet).

Bis zur Umsetzung dieser Themen gilt:

- Keine zusätzlichen Secrets im Repo ablegen.
- Bestehende .env-/Secrets-Themen werden in `KNOWN_ISSUES.md` als TODO geführt.

---

## Change Log

- v1.0: Initiale Strategy Policy (Zielbild, Scope v1, Risk-Kernregeln, State-Machine, Health/QC, Security-TODO)


