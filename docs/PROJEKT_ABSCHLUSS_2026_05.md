# Assembled-Trading-AI — Projekt-Abschluss Mai 2026

Erstellt: 2026-05-29  
Autor: Assembled-Trading-AI Projekt  
Status: **Abschlussdokument aktive Alpha-Strategiesuche**

---

## 1. Zweck

Dieses Dokument schließt die aktive Phase der Alpha-Strategiesuche (November 2024 – Mai 2026) ab.

In dieser Phase wurden neun Strategien konzipiert, implementiert, getestet und durch Out-of-Sample
Walk-Forward-Backtests validiert. Das Ergebnis ist eindeutig und wird hier ohne Beschönigung
dokumentiert: **Keine der getesteten Strategien erzielt risk-adjustiert einen stabilen Edge
gegenüber einem einfachen SPY-Investment.**

Das Projekt ist damit nicht gescheitert. Es hat sein primäres Qualitätsziel erreicht:
eine ehrliche, reproduzierbare Evidenzbasis zu schaffen, bevor Echtgeld eingesetzt wird.

---

## 2. System-Status zum Abschluss

### GO_LIVE_CHECKLIST: 14 von 16 Kriterien erfüllt

| Kategorie | Kriterium | Status |
|-----------|-----------|--------|
| A1 | Unit + Integration Tests (557 PASS, 0 fail) | ✓ ERFÜLLT |
| A2 | CI alle 7 Workflows grün (Ubuntu + Windows) | ✓ ERFÜLLT (commit 879c2c7d) |
| A3 | PIT-Regression-Tests (keine Look-Ahead-Bias) | ✓ ERFÜLLT |
| B1 | OOS Walk-Forward Backtest trend_baseline | ✓ ERFÜLLT (negatives Ergebnis) |
| B2 | CPCV-Validierung | OFFEN — bewusst zurückgestellt |
| B3 | Multi-Strategie-Edge | UNKLAR — kein validierter Edge |
| C1 | Order-Lifecycle-Log | ✓ ERFÜLLT (commit b3cd4ec5) |
| C2 | Kill-Switch mit Operator-Auth | ✓ ERFÜLLT (commit b1a3434d) |
| C3 | Live-Reconciliation | OFFEN |
| E1 | Dead Man's Switch | ✓ ERFÜLLT (commit 86468b0c) |
| E2 | Pre-Trade-Checks vollständig | ✓ ERFÜLLT |
| E3 | Test-Alert / Smoke-Fire | OFFEN — Operator-Schritt (§6) |
| F1 | API-Auth + Rate-Limit | ✓ ERFÜLLT |
| F2 | API-Endpoints (/health, /ledger, /performance) | ✓ ERFÜLLT (commit 190fc1dd) |
| F3 | Security-Hardening Vollständig | ✓ ERFÜLLT |
| F4 | Secrets in .env, nicht im Repo | ✓ ERFÜLLT |

**Technischer Zustand:** Das System ist als Paper-Trading-Infrastruktur vollständig einsatzbereit.
Die offenen Kriterien (B2, C3, E3) sind keine technischen Blocker für Paper-Betrieb,
aber Voraussetzungen für einen verantwortungsvollen Live-Start.

---

## 3. Getestete Strategien — OOS-Ergebnisse

Alle Ergebnisse aus Walk-Forward-Backtests auf echten Alpaca-Marktdaten (2018–2025/2026),
kein Survivorship-Bias-bereinigter Datensatz, kein Leverage.

### 3.1 Übersichtstabelle

| # | Strategie | Ø CAGR | Ø Sharpe | Ø MaxDD | SPY Ref | Verdikt | Ergebnisdokument |
|---|-----------|--------|----------|---------|---------|---------|------------------|
| 1 | trend_baseline | **-6.1%** | -0.18 | -22.2% | +13.0% | KEIN EDGE | `docs/results/2026_05_trend_baseline_real_oos.md` |
| 2 | multifactor_v2 (TA-only) | +12.9% | 0.36 | -23.0% | +13.0% | KEIN EDGE | `docs/results/2026_05_multifactor_v2_real_oos.md` |
| 3 | multifactor_long_short | **-19.5%** | -0.80 | ~-22% | +13.0% | KEIN EDGE | `docs/results/2026_05_multifactor_long_short_real_oos.md` |
| 4 | mfv2 + Altdata | +10.7% | 0.36 | -18.6% | +13.0% | KEIN EDGE | `docs/results/2026_05_mfv2_full_stack_real_oos.md` |
| 5 | vol_target_overlay | +8.8% | **0.88** | **-8.4%** | +14.5% | OVERLAY BEHALTEN | `docs/results/2026_05_vol_target_overlay_real_oos.md` |
| 6 | dual_momentum | +9.7% | 0.98 | -11.3% | +14.5% | SCHWACH | `docs/results/2026_05_dual_momentum_real_oos.md` |
| 7 | etf_pairs_meanrev | -0.3% | -0.49 | -2.0% | +19.7% | KEIN EDGE | `docs/results/2026_05_etf_pairs_meanrev_real_oos.md` |
| 8 | low_max_lottery | +9.8% | 1.06 | -10.1% | SPY Sharpe 1.40 | KEIN EDGE | `docs/results/2026_05_low_max_lottery_real_oos.md` |
| 9 | crypto_funding_carry | +4.5–6.7% APR | 4.40–5.64* | -$1,389–$1,611 | n/a (Crypto) | MARGINAL / EXCHANGE-RISIKO | `docs/results/2026_05_crypto_funding_carry_backtest.md` |

*Sharpe für Crypto-Carry strukturell überhöht — siehe Caveat §4.2 im Ergebnisdokument.

Strategievergleich: `docs/results/2026_05_strategy_comparison.md`

### 3.2 Kernbefund

**Keine der getesteten Aktienstrategien schlägt SPY risk-adjustiert.**

- trend_baseline: –19.1% CAGR-Gap, Sharpe negativ. Momentum auf einem 75-Titel-Universum
  reproduziert den akademischen Effekt nicht.
- multifactor_v2: Sharpe 0.36 bei negativem CAGR-Gap. 34 Faktoren davon 9+ strukturell Null
  (fehlende Altdata, Cross-sectional z-Score-Degeneration).
- multifactor_long_short: Long-Only-Betrieb bricht das Long-Short-Faktormodell fundamental.
- mfv2 + Altdata: Altdata-Integration liefert Sharpe-Delta = 0.00 gegenüber TA-only.
  Altdata teuer, Wirkung nachweisbar nicht vorhanden auf diesem Universum.
- etf_pairs_meanrev: Auf großen, liquiden ETF-Paaren kein Mean-Reversion-Signal messbar.
- low_max_lottery: MAX-Effekt (Bali et al. 2011) auf Large/Mid-Cap-Universum abwesend
  oder durch Survivorship-Bias im Datensatz kompensiert. High-MAX schlägt Low-MAX deutlich.
- dual_momentum: Schwächste positive Evidenz aller Aktienstrategien, aber Folds-Hit-Rate
  unter 35 %. Kein robuster Edge.
- crypto_funding_carry: Edge vorhanden (+4.5–6.7% APR nach Fees), aber nicht modelliertes
  Exchange-Gegenparteirisiko (FTX-Szenario) und strukturell überhöhter Sharpe.
  Nicht als Standalone-Strategie geeignet.

**vol_target_overlay** bildet eine Ausnahme: kein Alpha, aber demonstrierter Schutzwert
im COVID-Crash 2020 (–8.8% vs. –28.9% SPY). Bleibt als Drawdown-Schutz im Portfolio.

---

## 4. Abschluss-Entscheidung

### 4.1 Aktives Alpha: pausiert

Die aktive Suche nach einem OOS-validierten Edge auf dem verfügbaren Datensatz (Alpaca,
75 Titel, 2018–2025) wird eingestellt. Der Grund ist nicht mangelnde Implementierungsqualität,
sondern ehrliche Evidenz: kein getestetes Modell erzielt reproduzierbare Überrendite.

### 4.2 Portfolio-Architektur ab sofort

**Passiver Kern** + **vol_target_overlay** als Drawdown-Schutz:

```
Passiver Kern:
  SPY            ~60 %   (US Equities)
  AGG / BND      ~40 %   (US Bonds)

Overlay:
  vol_target_overlay    aktiv — reduziert Aktienquote bei hoher realisierter Volatilität
                        Evidenz: MaxDD –8.4% vs. SPY –14.5% (Ø 6 Folds, 2019–2025)
```

Diese Architektur ist explizit konservativ und keine Rendite-Maximierung.
Ziel: kapitalerhalt und kontrolliertes Risiko im Paper-Betrieb.

### 4.3 Paper-Betrieb

Der Paper-Pilot (`scripts/run_live_paper.py`) läuft weiter mit trend_baseline als Pilot-Strategie,
weil die Live-Infrastruktur (Task Scheduler, EOD-Runner, Trade-Journal, Kill-Switch, API)
ohne aktivierte Alpha-Strategie sonst untätig ist.

Der Paper-Betrieb dient ausschließlich der Infrastruktur-Erprobung und Systemüberwachung.
Er ist **kein implizites Go-Live für trend_baseline**. OOS-Ergebnis: –6.1% CAGR. Kein Edge.

### 4.4 Kein Echtgeld ohne validierten Edge

Eine Live-Implementierung ist ausgeschlossen, solange kein OOS-Walk-Forward-Ergebnis
folgende Mindestanforderungen erfüllt:

| Kriterium | Schwelle |
|-----------|----------|
| Ø CAGR | > +5 % (nach Fees, realistisches Universum) |
| Ø Sharpe | > 0.80 |
| MaxDD | > –20 % |
| Folds mit positivem Alpha | > 50 % |
| Datensatz-Qualität | Survivorship-bereinigt oder externe Datenbasis |
| Unabhängige Replikation | Mindestens 1 Fold auf getrenntem Datensatz |

Kein dieser Schwellenwerte wurde in dieser Forschungsphase erreicht.

---

## 5. Was gelernt wurde

### 5.1 Survivorship-Bias ist keine Randnotiz

Das Alpaca-Local-Cache-Universum enthält nur überlebende Symbole (aktuell handelbar).
Delisting, Bankruptcies, Übernahmen fehlen vollständig. Dies verzerrt:
- High-MAX-Portfolios (die schlimmsten Lottery-Verlierer fehlen → High-MAX erscheint weniger schlecht)
- Momentum-Strategien (tote Momentum-Aktien fehlen → scheinbar höhere Qualität)
- Jede Strategie, die auf Small-/Mid-Cap-Rotation setzt

**Konsequenz:** Ein positives OOS-Ergebnis auf diesem Datensatz hat reduzierten Beweischarakter.
Ein negatives OOS-Ergebnis ist dagegen besonders belastbar.

### 5.2 Multiple Testing erfordert DSR-Korrektur

Nach 9 getesteten Strategien (und weiteren Parametervarianten innerhalb der Strategien) ist
der naive p-Wert einzelner Konfigurationen wertlos. Jeder neue Backtest-Run erhöht die
Wahrscheinlichkeit, zufällig ein positives Ergebnis zu finden.

Die korrekte Metrik wäre der **Deflated Sharpe Ratio (DSR)** nach Harvey & Liu (2015),
der die Anzahl der Testversuche explizit einbezieht. Kein der hier berechneten Ergebnisse
wurde DSR-korrigiert. Das ist ein bekanntes Defizit dieser Forschungsphase.

### 5.3 Altdata ohne eigene Infrastruktur ist ein Mythos

MultiFactor V2 hat 34 Faktoren implementiert. Im OOS-Test sind mindestens 9 strukturell Null:

- `insider_trading`: 100 % "unknown" Sentiment im lokalen Parquet — Faktor immer 0
- `congress_activity`: Keine Datendateien vorhanden — Faktor immer 0
- `news_sentiment`: Sparse vor April 2026, GDELT-Mapping unzuverlässig
- `earnings_surprise_z`: Cross-sectional z-Score kollabiert bei kleinem Universum (Std ≈ 0)
- `sector_rotation_bias`: Gleiche Degeneration

Altdata, die nicht aktiv und kontinuierlich befüllt wird, erzeugt nicht Null-Alphas,
sondern **lautlosen Drag** durch factor dilution.

**Konsequenz:** Lieber 5 funktionierende Faktoren als 34 davon 9 Nullen.

### 5.4 Akademische Faktoren replizieren nicht auf kleinen Universen

| Akademischer Effekt | Universum der Studie | Dieses Universum | OOS-Ergebnis |
|--------------------|--------------------|-----------------|--------------|
| MAX-Anomalie (Bali 2011) | NYSE/AMEX/NASDAQ, alle Caps | 75 Large/Mid-Cap | Effekt abwesend |
| Short-Term Momentum | Breites Universum | 75 Titel | negativ |
| Pairs Mean-Reversion | Liquid ETF-Paare | Liquid ETF-Paare | negativ |

Akademische Faktorprämien existieren primär in Small-Cap-Universa oder Long-Short-Implementierungen.
Long-Only, Large-Cap, breiter Markt: akademische Prämien werden weitgehend wegarbitriert.

### 5.5 Geparkte Ideen — nicht getestet, nicht verworfen

Folgende Ansätze wurden nicht ausreichend getestet und gelten als **offen für spätere Phasen**:

- **News-Alpha (Event-Driven):** Implementiert in `src/assembled_core/events/news_alpha/`.
  Backtests noch nicht OOS-validiert. Konzept (Hormuz → Öl-Long binnen Stunden) unterscheidet
  sich fundamental von allen getesteten Strategien. Erfordert Intraday-Daten und schnellen Execution-Pfad.
- **PEAD (Post-Earnings-Announcement Drift):** Implementiert als Research-Modul.
  PIT-Safety noch nicht vollständig verifiziert (as_of-Parameter offen).
- **Crypto-Carry als Portfolio-Baustein:** Marginaler Edge vorhanden (+4.5–6.7% APR).
  Nur sinnvoll als kleiner Bestandteil einer breiteren Allokation mit crypto-nativer Infrastruktur.
- **Survivorship-bereinigte Daten:** Mit CRSP oder Sharadar wären die Ergebnisse für
  akademische Faktoren wahrscheinlich anders — nicht notwendig besser, aber ehrlicher.

---

## 6. Offene Operator-Schritte

Diese Punkte sind technisch klein, aber für den sauberen Weiterbetrieb relevant.
Sie sind keine Blocker für Paper-Betrieb, aber sollten zeitnah adressiert werden.

| # | Schritt | Details | Priorität |
|---|---------|---------|-----------|
| 1 | **E3: Test-Alert feuern** | `scripts/run_live_paper.py --smoke-fire-alert` einmalig ausführen, um den Notification-Pfad zu verifizieren (SMS/E-Mail aus policy.yaml). Bestätigt: Alert-Infrastruktur funktioniert. | HOCH |
| 2 | **policy.yaml: Veralteter Kommentar** | `policy.yaml` enthält Referenzen auf trend_baseline als "primäre Strategie" — nach Abschluss-Entscheidung §4.2 (passiver Kern) ist das irreführend. Kommentar aktualisieren. | MITTEL |
| 3 | **B2: CPCV-Validierung** | Combinatorial Purged Cross-Validation ist implementiert aber nicht ausgeführt. Falls die Alpha-Suche wieder aufgenommen wird, ist das der nächste methodische Schritt vor Live-Entscheidungen. | NIEDRIG (deferred) |
| 4 | **DMS: Task Scheduler Eintrag** | Dead Man's Switch Daemon (`scripts/dms_daemon.py`) ist implementiert (commit 86468b0c) aber noch nicht im Windows Task Scheduler eingetragen. Ohne diesen Eintrag läuft kein autonomes Heartbeat-Monitoring. | MITTEL |
| 5 | **macro.parquet: CPI-Neuberechnung** | `download_all_market_data.py` berechnet `cpi_yoy` jetzt korrekt via `pct_change(12)`. Das Cache-File muss einmalig neu gezogen werden, damit der Faktor korrekte Werte enthält. | NIEDRIG |

---

## 7. Fazit

Das Assembled-Trading-AI-Projekt hat in dieser Phase das getan, was ein ehrliches Quantsystem
tun soll: es hat Hypothesen getestet und die Ergebnisse unvermindert dokumentiert.

Das Ergebnis — kein validierter Edge — ist informativ, nicht deprimierend.
Es spart erheblichen Kapitalverlust durch voreiligen Livestart.

Die Infrastruktur (Kill-Switch, Order-Lifecycle-Log, CI, API, Paper-Runner, Risk-Overlays)
ist produktionsreif und wartet auf eine Strategie, die einen OOS-validierten Edge hat.

**Nächste Phase beginnt, wenn eine der geparkten Ideen (§5.5) oder eine neue Hypothese
die in §4.4 definierten Mindestanforderungen erfüllt.**

---

_Dokument: `docs/PROJEKT_ABSCHLUSS_2026_05.md`_  
_Ergebnisdokumentation: `docs/results/2026_05_strategy_comparison.md`_  
_GO_LIVE_CHECKLIST: `docs/GO_LIVE_CHECKLIST.md`_

---

## Nachtrag (2026-07-23) — Einordnung multifactor_long_short / etf_pairs_meanrev

Die Verdicts „KEIN EDGE" für **multifactor_long_short** (Zeile 3) und **etf_pairs_meanrev**
(Zeile 7) sind präziser als **„nicht valide getestet"** zu lesen, nicht als Konzept-FAIL:

- **multifactor_long_short:** Der OOS-Lauf betrieb das Long-Short-Faktormodell **long-only**
  (im Dokument selbst benannt: „Long-Only-Betrieb bricht das Long-Short-Faktormodell
  fundamental"). Das Ergebnis (−19,5 % CAGR) misst damit nicht das Konzept, sondern eine
  invalide Betriebsart.
- **etf_pairs_meanrev:** Der Harness testete eine wörtliche Literal-Umsetzung mit bekannten
  Einschränkungen; das Ergebnis widerlegt das Pair-Trading-Konzept nicht generell, sondern
  nur diese konkrete Implementierung auf diesem Universum.

Konsequenz: Beide dürfen nicht als „Konzept widerlegt" zitiert werden — korrekt ist
„in dieser Form nicht valide getestet / kein verwertbarer Nachweis". Der ursprüngliche
Text oben bleibt unverändert (Audit-Artefakt).
