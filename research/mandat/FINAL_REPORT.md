# FORSCHUNGSMANDAT — Abschluss-Dossier (Stand 2026-07-12, N=1.964)

**Auftrag (Hans, 2026-07-05):** Systematische Strategien entwickeln/testen, die den S&P 500 Total
Return **nach deutschen Steuern** über 10-Jahres-Fenster schlagen. Protokoll: Registry vor Lauf,
Test-Budget-Ledger, DSR/PBO-Pflicht, ehrliche Verdicts, autonomes Arbeiten.

**Umfang:** 80+ Hypothesen-Familien, **1.964 registrierte Trials**, 41+ Assets, 10 Strategie-Stränge,
2 Zeitebenen, 48 Portfolio-Konstruktionen, echte Monte-Carlo-Simulationen (1.000 Bootstrap-Pfade,
Seed 42), survivorship-freie Universen (S&P-PIT 1.167 Namen; Broad-US 15.101 inkl. Delisted).
Alle Läufe reproduzierbar (research/mandat/h0*.py + results/*.json). Ledger führt die Wahrheit.

---

## 1. Kernbefund (mehrfach korrigiert, final)

> **Keine deployable Strategie schlägt den passiven thesaurierenden S&P-ETF absolut nach ehrlicher
> deutscher Steuer.** Kein Signal (Technik/Insider/Congress/Whale/News/Geopolitik/Kalender/Faktor),
> kein Wrapper (wikifolio/Direct-Indexing/Faktor-ETF), kein Timing (Asset/Zustand/Regime/Sleeve),
> keine Zeitebene (Daily/Intraday), kein Universum (S&P/Small-Cap/Welt/EU/FX/Krypto-Timing).

**Drei strukturelle Gründe (empirisch isoliert):**
1. **Kein Brutto-Alpha:** Ohne jede Steuer gewinnt der ETF trotzdem (höchster Brutto-CAGR 11,21 %
   im 31,5-J-Vergleich) — Cap-Weighting hält die Mega-Gewinner ohne Selektion/Rotation/Prognose.
2. **Steuer-Asymmetrie:** ETF zahlt 18,46 % (Teilfreistellung) END-gestundet (−0,69 pp/J);
   aktive Direktanlagen 26,375 % laufend realisiert (−2 bis −3,7 pp/J = 3–5×).
3. **Jede Aktivität kostet:** Timing = Steuer-Event; §23-Assets verlieren bei <1 J die
   Steuerfreiheit (44 %); Intraday-Signale existieren nicht mal brutto.

## 2. Was ROBUST übrig bleibt (das deployable Endergebnis)

**Endspezifikation** (MC-verifiziert über 48 Konstruktionen, Floor-optimal im ehrlichen Szenario):
> **~65–70 % thesaurierender Aktien-ETF · ~25 % Xetra-Gold · 5–10 % Krypto (BTC/ETH-Split, §23-
> Disziplin >1 J)** · Ansparphase: **Cash-Flow-Rebalancing** (Rate→Untergewicht, NIE verkaufen) ·
> Haltephase: **2-Jahres-Verkaufs-Rebalancing** (nie jährlich — §23-Uhr + Steuer) · keine Bonds,
> kein Silber, kein Vol-Targeting, kein Timing.

**Ehrliche Charakterisierung:** Der Gold-Sleeve ist **Versicherung, kein Renditebringer**
(Maximin auf Endvermögen = 0 % Gold; kostet in ~2/3 der 10-J-Fenster Endvermögen). Was er kauft:
Floor +41 % (MC-5 %-Quantil), MaxDD −0,55→−0,35 (GFC in EUR: −50 %→−27 %), **Ruin-Risiko in der
Entnahmephase 0 % bis 5 %-Entnahme** (SPY: 2 %). In EUR-Sicht stärker (hedgt USD). Krypto-Sleeve
= dimensionierte Wette (½-Kelly des Pessimist-Szenarios ≈ 5 %; im Tot-Szenario −4 % Median-Kosten).

## 3. Die letzte Tür — GESCHLOSSEN (H-081, echte CBOE-Historie, Vorschlag Hans)

**Stillhalter-/Vol-Risk-Prämie:** War nach drei Modelltests (H-046/062/077) unentscheidbar
(H-079-Band). **Aufgelöst mit 40 Jahren ECHTER SPX-Optionspreise (CBOE BXMD/PUT/BXM, 1986–2026):**
- BXMD (bestes Design, 30Δ-OTM): brutto 10,90 vs SPXTR 11,52 %/J — trailt schon VOR Steuern;
  Sharpe 0,90/DD −0,43 besser = **Versicherungs-Profil**, keine Outperformance.
- Regime: gewinnt NUR in der Lost Decade (2000er +3,7 pp), verliert jede Bull-Dekade um 4–6 pp/J.
- Deutsches Steuer-Overlay (Stillhalter-Asymmetrie): alle drei KLAR unter dem ETF-Pfad
  (BXMD 3,56M vs 5,15M). **VERDICT: FAIL für den steuerpflichtigen deutschen Anleger.**
- Synthese: BXMDs Risiko-Verbesserung ist dieselbe Familie wie der §23-Gold-Sleeve — der sie mit
  **0 % Steuer** liefert statt 26,375 % asymmetrisch → **Gold-Sleeve dominiert BuyWrite strikt.**

**Damit hat das Mandat keine offene Alpha-Tür mehr auf erreichbaren Daten.**

**Nachtrag — Versicherungs-Duell (H-082, PPUT/CLL/CNDR, echte CBOE-Historie 1986–2026):** Der
§23-Gold-Sleeve dominiert auch jede optionsbasierte ABSICHERUNG strikt: Protective Puts kosten real
−3,5 pp/J (38,4 J), verschlechtern die Sharpe und halfen selbst in der Crash-Dekade nicht (2000er:
PPUT −1,39 % vs SPX −0,95 %); netto-DE 545k vs Gold-Sleeve 890k (> ungesichert 767k) bei besserem
Drawdown. Gold = einzige Versicherung mit positivem Erwartungswert und 0 % Steuer. Frage geschlossen.

## 4. Verdicts nach Feldern (Kurzform; Details im Ledger)

| Feld | Verdict | Kern-Evidenz |
|---|---|---|
| Technische Analyse (1.032 Configs, 41 Assets) | **TOT** | 11/950 Welt < Zufall; SPY/GLD 0/25 |
| Intraday (Retail, 5m) | **TOT (brutto!)** | alle 7 Strategien −1,7…−10 bps/Tag vor Steuern |
| Insider (Käufe/Verkäufe/Cluster/×Technik, 186 Cfg) | **TOT** | 0 Survivors, alle Grids |
| Congress / Whale-13F (168 Cfg) | **TOT** | 0 Survivors |
| News / Geopolitik / Social (132 Cfg + W13/14/16a) | **TOT** | brutto real intraday, netto nie |
| Kalender (TOM/DoW/Halloween) | **TOT** | TOM brutto schwach, netto gefressen; Montag invertiert |
| Faktoren (direkt + ETF-Wrapper) | **TOT** | senken DD, schließen Gewinner aus; MTUM/SPMO-Widerspruch |
| Small-Cap (Momentum/Size) | **TOT/Artefakt** | Momentum vernichtet Kapital; „Size" = Illiquiditäts-Artefakt |
| Short/LS, FX, Hebel (117 Cfg, Research-Override) | **TOT** | Short zahlt nie; FX trivial; Hebel = BTC-Hindsight |
| wikifolio/Copy-Trading (Wrapper) | **DOMINIERT** | braucht 2–5 %/J Brutto-Alpha nur für Gleichstand |
| Direct-Indexing + TLH | **TOT** | dt. TLH-Alpha real aber ~+5 % ≪ Teilfreistellungs-Nachteil |
| Krypto-Timing | **TOT** | §23-Keil: HODL schlägt aktiv (BTC-Trend = nicht-replizierende Ausnahme, ETH 0/25) |
| Vol-Targeting / Regime / Glide / Krisen-Rebal | **TOT** | De-Risking = Steuer-Event; stur schlägt clever |
| **Portfolio-Aufstellung + §23-Sleeve** | **✓ ROBUST** | Abschnitt 2 |
| **Cash-Flow-Rebalancing** | **✓ PASS** | dominiert Verkaufs-Rebal strikt (Steuer 0) |
| **Optionen (Stillhalter)** | **TOT (real-data)** | Abschnitt 3 — BXMD 38 J trailt brutto; DE-Steuer verbreitert |

## 5. Methodische Integrität (Fänge & Korrekturen — Teil des Ergebnisses)

- **E-051:** Frozenset-Nichtdeterminismus (±10 % Swing, PASS/FAIL kippte) → `sorted()`, byte-identisch verifiziert.
- **E-052:** Ganz-Universum-Basket fing Delisting-Micro-Price-Artefakte (Fake 10³⁰) → 2-Schicht-Hygiene-Fix.
- **Mark-to-market-Korrektur:** frühere „PASSes" (H-024/032) waren End-Steuer-frei gerechnet → unter
  End-Liquidation schlägt NICHTS den ETF; als Steuer-Mechanik-Demos reklassifiziert.
- **Steuer-Engine:** 26,375 % (Beamter, Soli bestätigt), Sparerpauschbetrag 1.000 €, FIFO+Verlusttopf,
  Div-Steuer am Ex-Tag, §23 (0 %/44 %), ETF-Teilfreistellung; Vorabpauschale bewusst konservativ weggelassen.
- Survivorship-frei überall wo möglich; PIT/available_at; DSR-Latte wächst ehrlich mit N (jetzt ~1.964).

## 6. Offene Punkte / Operator-Entscheidungen

1. ~~Echte Optionsdaten~~ → **ERLEDIGT via CBOE-Historie (H-081): Stillhalter-Feld FAIL, geschlossen.**
2. EU-PIT-Index-Membership → All-World-Selektion verdict-fähig (Prior: FAIL, vgl. GEM/H-052).
3. Form-4 für echte Small-Caps (großer EDGAR-Pull) → letzte Insider-Lücke (Prior: FAIL).
4. Live: Shadow-Track 13F_k10 läuft (2026-07-12: 101.787 vs SPY 101.281); Paper-Pilot separat.

*Reproduzierbarkeit: alle Skripte research/mandat/, Seeds fixiert, Registry/Ledger append-only.*
