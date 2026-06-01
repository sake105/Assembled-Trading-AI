# Strategiesuche & Universen-Recherche — Kann das Projekt SPY schlagen?

**Erstellt:** 2026-05-31
**Modus:** Reine Recherche. KEINE Code-Änderung, KEIN Backtest ausgeführt, KEIN Push.
Einziges beschriebenes Artefakt ist diese Datei. Alle anderen Pfade nur gelesen.
**Auftrag:** Suche nach neuen Strategien und anderen Universen (insb. Small/Mid Caps),
Ziel: **mindestens SPY schlagen**. Fortsetzung der Linie aus `docs/Überprüfung.md`
(gleiche Ehrlichkeits-Disziplin).

---

## §0 Quellentrennung (Legende)

Jede belastbare Aussage ist markiert. Das ist bewusst streng, weil im Trading-Kontext
die Verwechslung von „belegt" und „vermutet" der teuerste Fehler ist.

- **[V] = Verifiziert** — direkt aus Code/Daten/Ergebnisdokument dieses Repos, mit `Datei:Zeile`.
  - **[V✓]** = in DIESER Session von mir persönlich erneut geöffnet und gelesen.
  - **[V·agent]** = von einem read-only Recherche-Subagenten mit `Datei:Zeile` zitiert, von mir
    nicht erneut geöffnet. Plausibel, aber eine Stufe weniger hart als [V✓].
- **[Z] = Zitiert** — externe Literatur/Quelle mit URL (§9). Aussage Dritter, nicht von mir nachgerechnet.
- **[H] = Hypothese** — meine eigene Schlussfolgerung/Spekulation. KEIN Beleg. Explizit als unsicher markiert.

**Keine Zahl in diesem Dokument ist erfunden.** Wo eine Zahl fehlt, steht „unbekannt".

---

## §1 Ehrliches Fazit zuerst

**Es gibt eine *bedingte* Aussicht — aber keine fertige Lösung, und in reinem Recherche-Modus
ist keine möglich.**

1. **Alle bereits OOS-getesteten Strategien des Repos schlagen SPY NICHT** — weder absolut
   (CAGR) noch risikoadjustiert (Sharpe). Das gilt auch für die zwei, die ein Subagent
   zunächst als „vielversprechend" meldete (`vol_target_overlay`, `low_max_lottery`): nach
   eigener Prüfung verlieren beide gegen SPY. [V✓ §3]

2. **Der einzige scheinbare „Sieg"** (Low-MAX Equal-Weight-Universum +25.6 %/Sharpe 1.46 vs
   SPY +19.7 %/1.40) **ist ein Artefakt**, kein Edge: 0 Trades, reines Halten von 75
   *überlebenden* Symbolen gleichgewichtet → Survivorship- + Equal-Weight-Effekt, Korrelation
   0.92 zu SPY. [V✓ §3.4]

3. **Die akademischen Anomalien, auf die wir hoffen (Size, MAX/Lottery), leben in Small Caps —
   nicht in Large Caps.** Das Repo testet sie auf einem Large-Cap-Survivor-Universum, also genau
   dort, wo sie *erwartungsgemäß* versagen. Im Low-MAX-Test war der Effekt sogar **umgekehrt**
   (MAX-Spread −37.8 %). [V✓ §3.4] + [Z §5.1]

4. **Die logische nächste Richtung (Small/Mid Caps) ist genau dort am gefährlichsten, wo wir am
   schwächsten sind: Datenqualität.** Small-Cap-Survivorship-Bias überschätzt Renditen massiv
   (eine Studie: +4.94 Prozentpunkte/Jahr, +9.1 % Sharpe). [Z §5.2] Das Repo hat **keine
   survivorship-saubere Small-Cap-Datenquelle** und sein Kostenmodell **unterschätzt
   Small-Cap-Handelskosten** in den OOS-Skripten. [V·agent §4] Ein naiver Small-Cap-Backtest
   auf heutigem Stand würde **systematisch zu optimistische** Ergebnisse liefern.

**Konsequenz:** Es ist *zu früh* für „keine Aussicht", aber auch *unredlich* für „Lösung gefunden".
Der ehrliche Stand ist: **mehrere literatur-gestützte Kandidatenpfade [H §7], deren Validierung
zwingend (a) survivorship-saubere Small/Mid-Cap-Daten, (b) liquiditätsbewusste Kosten und (c)
Multiple-Testing-Korrektur (DSR) voraussetzt** — alles drei außerhalb dessen, was reine Recherche
liefern kann. Der eine entscheidende Engpass ist **Datenbeschaffung** (§7.1, §8).

---

## §2 Was heißt „SPY schlagen"? (Die Zielgröße ehrlich gemacht)

Der Auftrag „mindestens SPY schlagen" zerfällt in zwei sehr unterschiedlich schwere Ziele:

| Ziel | Definition | Schwierigkeit 2016–2025 |
|------|-----------|--------------------------|
| **Absolut** | höhere CAGR als SPY | **sehr schwer.** SPY lief ~13–14.5 %/Jahr (historischer Large-Cap-Bull). [V✓ §3] Ohne Leverage/Konzentration kaum schlagbar. |
| **Risikoadjustiert** | höhere Sharpe **oder** Calmar (Rendite je Risiko / je MaxDD) | **schwer, aber die verteidigbare Zielgröße.** Hier liefert die Literatur (Low-Vol, Quality, Trend) realen Mehrwert. [Z §5.5] |

**[H]** Das absolute Ziel ist 2016–2025 fast eine Falle: Der US-Large-Cap-Bull war so stark, dass
ihn zu „schlagen" entweder mehr Risiko (Leverage/Konzentration) oder Glück erforderte. Small-Cap-
*Beta* hat in genau diesem Zeitraum **schlechter** abgeschnitten als Large Cap (Russell 2000
~9.5 %/Jahr vs S&P 500 ~13.5 %/Jahr 2010–2024 [Z §5.2]). Ein naiver „in Small Caps gehen"-Trade
hätte also **verloren**, nicht gewonnen.

**[H]** Die einzige intellektuell ehrliche SPY-Schlag-These lautet daher:
> *risikoadjustiert* (Sharpe/Calmar), via *qualitätsgefilterten* Small/Mid-Cap-Faktoren oder
> Low-Vol/Trend-Overlays, auf *survivorship-sauberen* Daten, mit *realistischen* Kosten — und
> selbst dann mit Demut, weil Faktor-Decay nach Publikation real ist.

Diese Präzisierung zieht sich durch den Rest des Dokuments.

---

## §3 [V✓] Was das Repo BEREITS getestet hat — und alles verliert gegen SPY

**Wichtiger Befund dieser Session:** Das Repo hat nicht nur `trend_baseline`/`mfv2` getestet,
sondern **8 Strategien** per echtem OOS-Walk-Forward (Alpaca-Daten, Mai 2026). Die Ergebnisse
liegen als Artefakte in `docs/results/*.md`. Ich habe die vier wichtigsten (vol_target,
dual_momentum, low_max, crypto-carry) in dieser Session **persönlich** geöffnet [V✓]; die
übrigen sind dokumentiert [V·doc].

### §3.1 Übersicht (aggregierte OOS-Mittel, je Dokument)

| Strategie | Ø CAGR | Ø Sharpe | Ø MaxDD | Benchmark SPY (gleicher Lauf) | Schlägt SPY? | Quelle |
|-----------|--------|----------|---------|-------------------------------|--------------|--------|
| trend_baseline | −6.1 % | −0.18 | −22.2 % | +13.0 % / — | **Nein** | `2026_05_trend_baseline_real_oos.md:53-72` [V✓ Vorsession] |
| multifactor_v2 | ~−7.7 % | ~−0.05 | — | +13.0 % | **Nein** | `2026_05_multifactor_v2_real_oos.md` [V·agent] |
| multifactor_long_short | nur long-only getestet | — | — | — | **unklar/Nein** | `2026_05_multifactor_long_short_real_oos.md` [V·agent] |
| etf_pairs_meanrev | −0.3 % (full) / +3.1 % (long-only) | −0.49 / +0.71 | — | — | **Nein** | `2026_05_etf_pairs_meanrev_real_oos.md` [V·agent] |
| dual_momentum | **+9.7 %** | **+0.98** | −11.3 % | +14.5 % / **1.26** | **Nein** (30.8 % Folds) | `2026_05_dual_momentum_real_oos.md:64-73` [V✓] |
| vol_target_overlay | **+8.8 %** | **+0.88** | −8.4 % | +14.5 % / **1.22** | **Nein** (8.3 % Folds) | `2026_05_vol_target_overlay_real_oos.md:65-71` [V✓] |
| low_max_lottery (bottom) | +9.8 % | +1.06 | −10.1 % | +19.7 % / **1.40** | **Nein** | `2026_05_low_max_lottery_real_oos.md:38,113` [V✓] |
| crypto_funding_carry | BTC +4.5 % / ETH +6.7 % APR | „strukturell irreführend" | — | n/a (kein Equity) | **Nein, nicht standalone** | `2026_05_crypto_funding_carry_backtest.md:208-249` [V✓] |

**Kernaussage:** Die besten risikoadjustierten Kandidaten (`dual_momentum` 0.98, `vol_target` 0.88)
liegen **klar unter** SPY (~1.2). Es gibt im aktuellen Bestand **keinen** SPY-Schläger.

### §3.2 [V✓] `vol_target_overlay` — Korrektur eines Subagent-Fehlers

Ein Subagent meldete „Ø CAGR 5.8 %, Sharpe 0.82". **Falsch.** Das Dokument selbst sagt:
Ø CAGR **8.8 %**, Ø Sharpe **0.88**, Ø MaxDD **−8.4 %** vs SPY 14.5 %/1.22/−12.3 %; nur **8.3 %**
der Folds schlagen SPY-Sharpe (`vol_target_overlay_real_oos.md:65-71`). Verdict im Dokument:
*„MaxDD-Kriterium erfüllt (0.68x, 32 % weniger Drawdown), Sharpe-Kriterium NICHT erfüllt"*
(Zeile 84-85). **Ehrlich:** Das ist ein **Drawdown-Reduzierer**, kein Outperformer — er senkt
das Risiko, aber auch Rendite *und* Sharpe unter SPY. Nützlich als Defensiv-Baustein, nicht als
SPY-Schläger. [V✓]

### §3.3 [V✓] `dual_momentum` — sauber getestet, schlägt nicht

13/13 Folds erfolgreich, GFC 2008 + COVID 2020 abgedeckt. Ø CAGR 9.7 % / Sharpe 0.98 / MaxDD
−11.3 % vs SPY 14.5 %/1.26/−11.7 % (`dual_momentum_real_oos.md:64-73`). Nur 30.8 % der Folds
schlagen SPY. Verdict: *„Kein Kriterium erfüllt … kein messbarer Mehrwert gegenüber einfachen
passiven Benchmarks"* (Zeile 100-101). Defensiv-tauglich (MaxDD ≈ SPY bei etwas weniger Rendite),
aber kein Edge. [V✓]

### §3.4 [V✓] `low_max_lottery` — der lehrreichste Fall

Drei Varianten getestet (`low_max_lottery_real_oos.md`):
- **Bottom-Quintil (die eigentliche Strategie):** +9.8 % / Sharpe 1.06 vs SPY +19.7 %/1.40 →
  **verliert** (Zeile 38). Das Dokument-eigene Kriterium: *„Beat SPY Sharpe: +1.06 vs +1.40 ✗"*
  (Zeile 113).
- **MAX-Spread (Low minus High):** Ø **−37.8 %** (Zeile 74) → **negativ**, d. h. High-MAX-Aktien
  haben *outperformt*. Das Dokument: *„Lottery effect absent or reversed"* (Zeile 86-87). Der
  akademische Effekt **repliziert nicht** auf diesem Large-Cap-Survivor-Universum.
- **Equal-Weight-Universum:** +25.6 % / Sharpe 1.46 vs SPY +19.7 %/1.40 → **schlägt SPY**, ABER
  **Trades/yr = 0** (Zeile 44-50). Das ist keine Strategie, sondern simples Halten der 75
  überlebenden Namen gleichgewichtet. Korrelation 0.92. **[H] Das ist Survivorship-Bias +
  Equal-Weight-Tilt, kein Alpha** — und das Dokument warnt selbst explizit vor genau diesen
  beiden Verzerrungen (Zeile 91-100).

**[H] Warum das der wichtigste Befund ist:** Genau die Anomalie, auf die man für Small Caps hofft,
ist auf Large Caps *verschwunden/umgekehrt*. Das ist **konsistent** mit der Literatur (§5.1) und
sagt voraus: ein *sauberer* Small-Cap-Test ist nötig, um zu wissen, ob hier etwas ist — der
Large-Cap-Test ist nicht aussagekräftig (weder positiv noch negativ entscheidend).

---

## §4 [V·agent] Kann das Repo überhaupt neue Universen / Small Caps testen?

Read-only-Rekonstruktion durch einen Subagenten (Datei:Zeile zitiert; von mir nicht alle erneut
geöffnet → [V·agent]). Die qualitativen Schlüsse sind die belastbaren.

### §4.1 Daten & Universum
- Preisquellen: **Alpaca** (primär), yfinance (Fallback). `data/data_source.py:79-141`,
  `scripts/_oos_wf_trend_baseline.py:79-108`.
- `watchlist.txt`: **195 Symbole, ausschließlich Large Cap**. `free_universe.py:31-76`: 35 Core-ETFs,
  IWM (Russell-2000-ETF) *als ETF* gelistet, aber **keine Russell-2000-/S&P-600-Einzelwerte**.
- **Kein** Small/Mid-Cap-Konstituenten-File, **kein** existierender Small-Cap-Backtest/Parquet im Repo.

### §4.2 Survivorship — der wunde Punkt
- PIT-Universum-API existiert und ist gehärtet: `data/universe.py:166-200` (`get_universe_members_pit`),
  Survivorship-Warnung Zeile 114-119. **ABER:** Die OOS-Skripte rufen sie **nicht** auf — sie lesen
  `watchlist.txt` direkt (`_oos_wf_trend_baseline.py:269-273`), also nur *lebende* Ticker.
- **Alpaca Free Tier liefert keine delisteten Symbole.** Es gibt im Repo **keinen**
  survivorship-sauberen Datenpfad. [V·agent]

### §4.3 Harness-Anpassbarkeit — die gute Nachricht
- `qa/walk_forward.py:384-399` + `qa/backtest_engine.py` sind **universums-agnostisch** (nehmen
  generische `prices`-DataFrames + Callables). Ein neues Universum bräuchte **keinen Code-Umbau**,
  nur eine neue Symbolliste + Preis-Parquet. [V·agent]

### §4.4 Kostenmodell — die schlechte Nachricht für Small Caps
- Es **existiert** ein liquiditätsbewusstes Modell: `execution/transaction_costs.py` mit
  ADV-bucketed `SpreadModel` (~335-376) und vol/participation `SlippageModel`.
- **ABER** die OOS-Skripte nutzen es nicht — sie verwenden **flache** Legacy-Gewichte
  (`spread_w=0.25`, `impact_w=0.5`) → effektiv **~85 bps pauschal, unabhängig von der Liquidität**
  (`pipeline/portfolio.py:90-91,159-162`, `_oos_wf_*.py`). [V·agent]
- **[H] Folge:** Ein Small-Cap mit ADV 10 Mio. USD zahlt im Backtest dasselbe wie eine Mega-Cap.
  Reale Small-Cap-Kosten (breite Spreads, Market Impact bei 100k+-Orders) wären **deutlich höher**.
  Jeder Small-Cap-Backtest auf heutigem Stand würde **Alpha überschätzen**. Das ist *der* Grund,
  warum ein naiver Small-Cap-Lauf gefährlich ist.

---

## §5 [Z] Externe Literatur — was wirklich SPY (risikoadjustiert) schlägt

### §5.1 Size-Prämie: nur mit Quality-Filter belastbar
Asness, Frazzini, Israel, Moskowitz, Pedersen — *„Size Matters, If You Control Your Junk"* (2018,
J. Financial Economics). Kernaussage: Die Size-Prämie galt als schwach, instabil und auf Microcaps
konzentriert — **diese Schwächen verschwinden, wenn man für Qualität kontrolliert.** Dann ist sie
*signifikant, zeitstabil, robust, NICHT auf Microcaps konzentriert*, über 30 Branchen und 24
Märkte. [Z] → **[H] Naives Small-Cap-Long ≠ Edge; qualitätsgefiltertes Small-Cap = die eigentliche
These.**

### §5.2 Small-Cap-Survivorship ist groß und teuer
- NIFTY-Smallcap-250-Studie (arXiv 2603.19380, 2016–2025): Survivor-only-Backtest **überschätzt**
  Jahresrendite um **+4.94 pp** (23.3 % relativ), Sharpe um **+0.097** (9.1 %); 82.5 % Removal-Rate. [Z]
- Recent regime: S&P 500 ~13.5 %/Jahr vs Russell 2000 ~9.5 %/Jahr (2010–2024). [Z]
  → **[H] Small-Cap-Beta war zuletzt ein Verlierer vs SPY; nur ein *Faktor* (Size+Quality), nicht
  das Beta, hat eine Chance.**

### §5.3 Faktor-Decay (aus `docs/Überprüfung.md`, Quelle McLean & Pontiff)
Faktoren verlieren nach Publikation ~26 % (OOS) bis ~58 % (post-publication) ihrer Prämie;
Effekte konzentrieren sich in *small-cap/illiquide/high-idio-vol* Werten. [Z] → **[H] Demut:
selbst ein literatur-validierter Faktor ist heute schwächer als im Paper.**

### §5.4 Trend-Following / Managed Futures: Diversifikation, nicht Outperformance
- „A Century of Evidence on Trend-Following" (Hurst/Ooi/Pedersen, AQR): TSMOM war in **8 von 10**
  der größten 60/40-Drawdowns der letzten 137 Jahre positiv (Crisis Alpha). [Z]
- Aktuell (2025): TTU-Trend-Index Langfrist-CAGR ~7.06 % vs S&P deutlich höher. [Z]
- → **[H] Trend schlägt SPY NICHT standalone (niedrigere CAGR), verbessert aber risikoadjustiert
  in einem *Multi-Asset*-Kontext.** Passt zu unseren `vol_target`/`dual_momentum`-Befunden (§3).

### §5.5 Low-Vol + Quality/Value/Momentum: der belastbarste risikoadjustierte Pfad
- USMV (Low-Vol-ETF): ~gleiche 10-Jahres-Rendite bei **~20 % weniger Vol** → bessere Sharpe. [Z]
- Reine Low-Vol allein kann underperformen; **Low-Vol + Net-Payout-Yield + 12M-Momentum** lieferte
  in einer Untersuchung **+2.8 pp** Rendite, **+47 %** Sharpe, **−13 pp** MaxDD vs simple Low-Vol. [Z]
- Low-Vol und Trend gelten als **echte Anomalien**, nicht reine Risikoprämien. [Z]
- „Doing Nothing beat the S&P" (Morningstar): die meisten *aktiven* Ansätze underperformen — Demut. [Z]

### §5.6 Survivorship-saubere Datenquellen (der Engpass, §7.1/§8)
- **Sharadar** (via Nasdaq Data Link): ~25 J. Historie, aktive **+ delistete** Werte, „nahezu
  survivorship-frei", Fundamentals **+** Tages-Preise, **retail-bezahlbar**. [Z]
- **Norgate Data** (Platinum): inkl. delisteter Werte (mit Delisting-Monat), US zurück bis 1950/1990,
  **PIT-Index-Konstituenten** (Russell/S&P) — ideal gegen Survivorship. [Z]
- **CRSP**: akademischer Goldstandard, listed + delisted, aber teuer/institutionell. [Z]

---

## §6 Synthese — warum „einfach Small Caps" nicht reicht, und was bleibt

Die Befunde greifen ineinander:

1. Unsere getesteten Strategien verlieren auf Large Caps [V✓ §3]. Das ist **erwartbar**, weil die
   Faktoren dort schwach/abwesend sind [Z §5.1].
2. Die Faktoren leben in Small Caps — aber **nur qualitätsgefiltert** [Z §5.1], und Small-Cap-
   *Beta* war zuletzt ein Verlierer [Z §5.2].
3. Small-Cap-Tests sind nur mit **survivorship-sauberen Daten** ehrlich [Z §5.2]; die haben wir
   nicht [V·agent §4.2].
4. Selbst mit sauberen Daten würde unser **Kostenmodell Small-Cap-Alpha überschätzen** [V·agent §4.4].
5. Risikoadjustiert (nicht absolut) ist die verteidigbare Zielgröße [H §2], und dort liefert die
   Literatur real (Low-Vol+Quality+Momentum) [Z §5.5].

**[H] Schlussfolgerung:** Der Weg „SPY schlagen" führt **nicht** über mehr Large-Cap-Signal-Tuning
und **nicht** über naives Small-Cap-Beta. Er führt — *falls überhaupt* — über
**qualitätsgefilterte Small/Mid-Cap-Faktoren bzw. Low-Vol/Quality-Overlays, risikoadjustiert
gemessen, auf survivorship-sauberen Daten mit liquiditätsbewussten Kosten.** Ohne diese drei
Voraussetzungen ist jede Small-Cap-Zahl, die wir heute erzeugen würden, **systematisch zu schön**.

---

## §7 [H] Kandidaten-Lösungspfade (Hypothesen, un-backtestet, gerankt)

> **Disclaimer:** Alles in §7 ist [H] — Hypothese. Kein Pfad ist hier validiert; reine Recherche
> kann das nicht. Jeder Pfad nennt: These · was zur Validierung nötig ist · Ziel (absolut vs
> risikoadj.) · Hauptrisiko.

### Pfad A — Qualitätsgefiltertes Small/Mid-Cap-Faktor-Portfolio ★ stärkste Literaturbasis
- **These:** Size+Quality (Asness) + ggf. Value/Momentum auf Small/Mid-Cap-Universum liefert eine
  *stabile* Prämie, die SPY **risikoadjustiert** schlagen kann. [Z §5.1]
- **Nötig zur Validierung:** (1) survivorship-saubere Small/Mid-Cap-Daten (Sharadar/Norgate, §5.6),
  (2) Quality-Score (Profitabilität, Verschuldung, Earnings-Stabilität — Daten teils via Sharadar
  Fundamentals), (3) **liquiditätsbewusste Kosten** statt der 85-bps-Pauschale (§4.4),
  (4) DSR/Deflated-Sharpe wegen Multiple Testing.
- **Ziel:** primär risikoadjustiert; absolut nur mit Demut.
- **Hauptrisiko:** Datenbeschaffung (Engpass §8) + Faktor-Decay [Z §5.3] + Kosten fressen Small-Cap-
  Edge [Z §5.2].

### Pfad B — Low-Vol + Quality/Momentum-Overlay (risikoadjustierter Direktangriff)
- **These:** „Enhanced Low-Vol" (500 niedrigste Vol → Top-100 nach Payout-Yield + 12M-Momentum)
  schlägt SPY **risikoadjustiert** (bessere Sharpe, niedrigerer MaxDD). [Z §5.5]
- **Nötig:** großes Liquid-Universe (S&P 500/1500 reicht — **kein** Small-Cap-Datenproblem!),
  Payout-Yield-Daten, sauberer Vol-Schätzer. **Geringste Daten-Hürde aller Pfade.**
- **Ziel:** explizit risikoadjustiert (Sharpe/Calmar), nicht absolute CAGR.
- **Hauptrisiko:** Low-Vol kann in starken Bull-Märkten absolut zurückbleiben (war 2016–2025 so);
  „SPY schlagen" gelingt hier eher auf Calmar/Sharpe als auf CAGR. Ehrlich benennen.
- **[H] Pragmatischer Erstkandidat**, weil er das Survivorship-/Daten-Problem *umgeht* (Large/Mid-Cap,
  liquide) und trotzdem die belastbarste risikoadjustierte Evidenz hat.

### Pfad C — Trend-Following als Multi-Asset-Diversifikator (nicht standalone)
- **These:** TSMOM/Managed-Futures-Overlay verbessert **Portfolio**-Sharpe/Calmar via Crisis Alpha,
  schlägt SPY aber **nicht** standalone. [Z §5.4]
- **Nötig:** Multi-Asset-Daten (Aktien/Bonds/Commodities/FX-Proxies via ETFs — teils vorhanden),
  Kombination mit einem Equity-Kern.
- **Ziel:** risikoadjustiert, nur im Verbund. Als Solo-SPY-Schläger **ungeeignet** — ehrlich.
- **Hauptrisiko:** Overlay-Komplexität; das Repo hat `vol_target`/`dual_momentum`/`crisis_alpha`
  bereits, alle schlagen solo nicht (§3).

### Pfad D — Equal-Weight / Mid-Cap-Struktur-Tilt (billig, aber ehrlich = Beta-Wette)
- **These:** Equal-Weight oder Mid-Cap-Tilt kann SPY zeitweise schlagen.
- **Realitätscheck:** Genau das „gewann" in unserem Test nur als **Survivorship-Artefakt** (§3.4),
  und Equal-Weight (RSP) hat cap-weighted zuletzt mehrheitlich **unterperformt**. [H/Z §5.2]
- **Ziel:** absolut, aber **schwache** Evidenz.
- **Hauptrisiko:** Das ist eine Beta-/Faktor-Tilt-Wette, kein Alpha. **Niedrigste Priorität.**

### Pfad E — Krypto-Funding-Carry (nur als kleiner Portfolio-Baustein)
- **These:** persistenter Funding-Carry (BTC +4.5 %, ETH +6.7 % netto p.a.). [V✓ §3, Doku]
- **Realitätscheck:** Sharpe „strukturell irreführend" (Steamroller-Profil), Counterparty-/
  Liquidations-Risiko **nicht modellierbar**, MiCA/BaFin-Retail-Fragen offen. Dokument-Verdict:
  *„not as a standalone strategy"*. [V✓]
- **Ziel:** allenfalls Diversifikations-Baustein, **kein** SPY-Ersatz.
- **Hauptrisiko:** Tail-/Exchange-Risiko (FTX-Klasse). Niedrige Priorität, andere Asset-Klasse.

### Ranking (nach Aussicht × Machbarkeit, [H])
1. **Pfad B** (Low-Vol+Quality/Momentum) — beste Evidenz *ohne* Small-Cap-Datenproblem.
2. **Pfad A** (Quality-Small-Cap) — stärkste These, aber Daten-Engpass.
3. **Pfad C** (Trend-Diversifikator) — solide, aber nur im Verbund.
4. Pfad D / E — schwach / Spezialfall.

---

## §8 [H] „Aussicht oder nicht?" — die ehrliche Landung

Der Auftrag: *aufhören erst, wenn keine Aussicht mehr **oder** eine/mehrere Lösungen.*

**Mein Stand:** Es gibt **Aussicht**, aber **keine fertige Lösung** — und mehr als
Kandidatenpfade [§7] ist in *reinem Recherche-Modus* nicht erreichbar, weil die drei
entscheidenden Validierungs-Voraussetzungen alle **außerhalb reiner Recherche** liegen:

| Voraussetzung | Status | Warum außerhalb reiner Recherche |
|---------------|--------|----------------------------------|
| Survivorship-saubere Small/Mid-Cap-Daten | **fehlt** [V·agent §4.2] | Erfordert **Daten-Abo-Entscheidung** (Sharadar/Norgate, Kosten) + Download-Wiring → Code/Daten-Änderung. |
| Liquiditätsbewusste Kosten in OOS | **vorhanden, aber ungenutzt** [V·agent §4.4] | Erfordert **Code-Änderung** in den OOS-Skripten. |
| DSR / Multiple-Testing-Korrektur | **fehlt in OOS** | Erfordert **Code-Änderung** + neuen Lauf. |

**[H] Konkrete Empfehlung für den nächsten (nicht-reinen-Recherche-)Schritt — Entscheidung liegt
beim User:**

1. **Pfad B zuerst** (Low-Vol+Quality/Momentum auf S&P-500/1500): umgeht das Survivorship-Problem,
   nutzt bereits beschaffbare Liquid-Daten, beste risikoadjustierte Evidenz. Schnellster ehrlicher
   Erst-Test. Zielmetrik **Sharpe/Calmar vs SPY**, nicht CAGR.
2. **Parallel die Daten-Frage klären** (User-Business-Entscheidung): Sharadar (retail-bezahlbar,
   survivorship-frei, Fundamentals für Quality) **oder** Norgate (PIT-Index-Konstituenten). Ohne
   eine dieser Quellen bleibt **Pfad A unehrlich**.
3. **Vor jedem Small-Cap-Ergebnis**: Kostenmodell auf ADV-bewusst umstellen + DSR anwenden, sonst
   ist das Ergebnis nicht belastbar.

**Was ich NICHT behaupte:** dass einer dieser Pfade SPY *tatsächlich* schlägt. Das ist
unbewiesen — alle §7-Pfade sind [H]. Die existierende Evidenz [V✓ §3] sagt: alles bisher
Getestete verliert. Die Aussicht ist **konditional**, nicht belegt.

---

## §9 Externe Quellen (mit URL)

- Asness, Frazzini, Israel, Moskowitz, Pedersen — *Size Matters, If You Control Your Junk* (SSRN 2553889):
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2553889 ·
  AQR: https://www.aqr.com/Insights/Research/Working-Paper/Size-Matters-If-You-Control-Your-Junk ·
  ScienceDirect (S0304405X18301326): https://www.sciencedirect.com/science/article/pii/S0304405X18301326
- Survivorship Bias, NIFTY Smallcap 250 (arXiv 2603.19380): https://arxiv.org/abs/2603.19380
- Hurst, Ooi, Pedersen — *A Century of Evidence on Trend-Following Investing* (AQR/Yale PDF):
  https://fairmodel.econ.yale.edu/ec439/hurst.pdf
- Moskowitz, Ooi, Pedersen — *Time Series Momentum* (Yale PDF): https://fairmodel.econ.yale.edu/ec439/jpde.pdf
- Top Traders Unplugged — Trend Following Performance Report, Aug 2025:
  https://www.toptradersunplugged.com/trend-following-performance-report-august-2025/
- Quantpedia — Time Series Momentum Effect: https://quantpedia.com/strategies/time-series-momentum-effect ·
  Small-Cap-Tag: https://quantpedia.com/strategy-tags/small-cap/
- The Evidence-Based Investor — Low-Volatility Investing (latest research):
  https://www.evidenceinvestor.com/post/low-volatility-investing
- Invesco — *A „pure" approach to the low-volatility potential advantage*:
  https://www.invesco.com/us-rest/contentdetail?contentId=f59a60d6cefaa610VgnVCM1000006e36b50aRCRD
- Morningstar — *What Beat the S&P 500 Over the Past Three Decades? Doing Nothing*:
  https://www.morningstar.com/stocks/what-beat-sp-500-over-past-three-decades-doing-nothing
- Grid Oasis — Small Cap vs Large Cap (Daten 2010–2024):
  https://gridoasis.com/guides/value-investing/small-cap-vs-large-cap/
- Sharadar (Fundamentals + Prices, survivorship-frei): https://www.sharadar.com/
- Norgate Data Review (Alvarez Quant): https://alvarezquanttrading.com/blog/norgate-data-review/ ·
  PIT-Konstituenten (Concretum): https://concretumgroup.com/historical-constituents-of-an-equity-index-in-python-norgate-data/
- CRSP (Survivor-Bias-Free): https://www.crsp.org/research/crsp-survivor-bias-free-us-mutual-funds/
- EODHD — Survivorship-bias-free analysis / delisted integration:
  https://eodhd.com/financial-academy/financial-faq/survivorship-bias-free-financial-analysis

## §10 Interne Quellen (Datei:Zeile)

- `docs/results/2026_05_vol_target_overlay_real_oos.md:65-71,84-85` [V✓]
- `docs/results/2026_05_dual_momentum_real_oos.md:64-73,100-101` [V✓]
- `docs/results/2026_05_low_max_lottery_real_oos.md:38,74,86-100,113` [V✓]
- `docs/results/2026_05_crypto_funding_carry_backtest.md:208-249` [V✓]
- `docs/results/2026_05_trend_baseline_real_oos.md:53-72` [V✓ Vorsession]
- `docs/results/2026_05_multifactor_v2_real_oos.md`, `…_long_short_real_oos.md`, `…_etf_pairs_meanrev_real_oos.md` [V·agent]
- `src/assembled_core/data/universe.py:114-119,166-200` (PIT-API, ungenutzt in OOS) [V·agent]
- `src/assembled_core/data/free_universe.py:31-76`; `watchlist.txt` (195 Large Caps) [V·agent]
- `src/assembled_core/qa/walk_forward.py:384-399`; `qa/backtest_engine.py` (universums-agnostisch) [V·agent]
- `src/assembled_core/execution/transaction_costs.py:335-399` (ADV/Slippage-Modell vorhanden) [V·agent]
- `src/assembled_core/pipeline/portfolio.py:90-91,159-162`; `scripts/_oos_wf_*.py` (flache 85-bps-Kosten in OOS) [V·agent]
- Kontext: `docs/Überprüfung.md` (Vorgänger-Bericht, OOS/CAGR-Ehrlichkeitsprüfung), `docs/PROJEKT_ABSCHLUSS_2026_05.md` (§5.1–5.5 Lessons, §4.4 Go-Live-Schwellen)

---

_Reine Recherche. Keine der §7-Aussagen ist backtest-validiert. Existierende Evidenz (§3): alle
bisher getesteten Strategien schlagen SPY nicht. Aussicht (§7/§8) ist konditional auf
survivorship-saubere Daten + realistische Kosten + DSR — alles außerhalb reiner Recherche._
