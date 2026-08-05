# Hypothesen-Registry — FORSCHUNGSMANDAT (append-only)

Regel (Mandat §4.1): Jeder Test wird VOR Ausführung hier registriert. Ein Test ohne
Registry-Eintrag ist ungültig. Pflichtfelder: H-ID, Datum, Hypothese (falsifizierbar),
Begründung (Literatur/ökonomische Logik), Parameter (vorab fixiert), Pass/Fail-Kriterium,
kumulatives N nach dem Test.

H-IDs H-001–H-010 sind rückwirkend die 10 Closure-Strategien (dokumentiert in
`docs/PROJEKT_ABSCHLUSS_2026_05.md`); die Fable-Exploration 2026-06-13 lief unter
eigenen IDs (H1–H6 + Runden 2–5, `research/fable_exploration/`). Neue Einträge starten
bei H-011.

---

## H-011 — Kandidat A: Regime-Gated Momentum-Quality (Mandat §3.1)

| Feld | Inhalt |
|---|---|
| H-ID | H-011 |
| Datum | 2026-07-05 |
| Hypothese | Ein monatlich rebalanciertes Top-20-Portfolio aus S&P-500-Titeln, gerankt 50/50 aus 12-1-Momentum und ROE-Quality, mit SMA200-Regime-Gate und vol-skalierten Gewichten, schlägt nach deutschen Steuern (26,375 % FIFO, Verlusttopf) und Kosten (10 bps/Seite) den thesaurierenden S&P-500-ETF-Netto-Pfad (~18,5 % effektiv) **und** die No-Signal-EW-Baseline desselben Universums. |
| Begründung | Momentum 12-1 (Jegadeesh/Titman 1993) + Quality/Profitability (Novy-Marx 2013) sind die zwei Faktoren mit 10J+-Live-Track-Record in liquiden Large Caps (SPMO/MTUM/QUAL, Mandat §2.2). Regime-Gate = dokumentiertes Time-Series-Momentum (Moskowitz/Ooi/Pedersen 2012); in Fable Round 4 als einziges robustes (Risk-)Element bestätigt. Rank-Puffer minimiert Turnover = deutschen Steuer-Edge (Mandat §2.4). |
| Parameter (fixiert VOR dem Lauf) | Universum: aktuelle S&P-500-Mitglieder mit lokal verfügbarer Historie (survivorship-biased, s. Explorativ-Flag). Ranking monatlich am Monatsultimo: Rang(12-1-Mom) 50 % + Rang(ROE, PIT via XBRL available_at) 50 %; fehlende Quality → nur Mom-Rang (dokumentiert je Lauf). Kauf bei Kombi-Rang ≤ 20, Halten bis Rang > 40. Positionsgröße ∝ 1/σ(60d), Cap 10 %. Regime: SPY-Schluss < SMA200 → Ziel-Exposure 50 % (Overlay-Logik), sonst 100 %. Katastrophen-Backstop: Exit bei Schluss < Höchstkurs − 4×ATR(20). Kosten 10 bps/Seite. Steuern: 26,375 % auf realisierte Gewinne, FIFO, Aktien-Verlusttopf (Carry-Forward). Benchmark: (a) SPY TR brutto (dekorativ), (b) ETF-Netto-Pfad 18,5 % auf Endgewinn (Beat-Kriterium), (c) EW-Baseline gleiches Universum, gleiche Steuerlogik (Survivorship-Kontrolle). |
| Pass/Fail (vorab) | PASS nur wenn ALLE: (1) Netto-Endvermögen > ETF-Netto-Pfad; (2) Netto-Sharpe > EW-Baseline-Netto-Sharpe (sonst ist es der Survivorship-Gift, Fable-Befund +0.35); (3) DSR > 0 bei kumulativem N; (4) PBO ≤ 50 % über die registrierte Variantenfamilie; (5) Teilperioden: in ≥ 3 von 4 disjunkten 2-Jahres-Fenstern nicht unter EW-Baseline. Sonst FAIL. |
| Variantenfamilie (vorab registriert, zählt als Trials) | Genau 4 Läufe: (V1) Basis wie oben; (V2) Momentum-only (Quality-Gewicht 0); (V3) ohne Regime-Gate; (V4) Rank-Puffer eng 20/25. Auswahlkriterium vorab: bestes Netto-Endvermögen, aber Verdict IMMER auf V1 (Basis) — V2–V4 sind Ablations-Diagnostik, keine Selektionsmenge für das Verdict. |
| Kumulatives N nach Test | 40 (N₀) + 4 = 44 |
| Explorativ-Flag | JA — survivorship-biased Preisuniversum (Delistings fehlen; Fable-Beweis: EW-Baseline +0.35 Sharpe Gift). NICHT verdict-fähig i. S. Mandat §2.5. Zweck: Pipeline-Aufbau + interne Signal-vs-Baseline-Differenz. Verdict-fähiger Re-Test erfordert Norgate/Sharadar (Operator-Entscheidung Hans, offen). |
| Nachtrag VOR Lauf (2026-07-05) | Datenquelle Preise: yfinance hart rate-limited (YFRateLimitError), Stooq PoW-Wall (nicht umgangen) → **Alpaca Daily Bars, adjustment=all, 2016-01→2026-07 (~10,5 J)**. Effektiver Teststart nach 252d-Momentum-Warmup ≈ 2017. Survivorship-Charakter unverändert. Keine Parameteränderung. |
| Ergebnis | **FAIL auf allen 5 Vorab-Kriterien** (2026-07-05). Details + Learnings: research/ledger.md, Ergebnisblock H-011. |

---

## H-012 — Steuer-/Turnover-optimiertes Pure-Momentum (Suchraum §4.6.4)

| Feld | Inhalt |
|---|---|
| H-ID | H-012 |
| Datum | 2026-07-05 (registriert VOR dem Lauf) |
| Hypothese | Eine turnover-minimierte 12-1-Momentum-Strategie (Kauf Top 20, Halten bis Rang > X, KEIN monatlicher Zwangs-Trade, kein Gate, kein ATR-Backstop) reduziert das deutsche FIFO-Steuerleck so weit, dass das Netto-Endvermögen den ETF-Netto-Pfad übersteigt. |
| Begründung | H-011-Learning 1 (Daten): V1 verlor 53 k an Steuern+Kosten — mehr als jede Signalwirkung; die EW-Baseline mit Minimal-Turnover schlug den ETF-Pfad. Mandat §2.4 nennt Turnover-Minimierung als Design-Priorität und §2.2 nennt Disziplin+Steueroptimierung als einzigen realen strukturellen Edge. Momentum-Persistenz: Jegadeesh/Titman 1993; Live-Evidenz SPMO/MTUM (§2.2). Frage ist NICHT „hat Momentum Alpha?" sondern „überlebt Momentum die deutsche Steuer besser als Buy-and-Hold-ETF?" |
| Parameter (fixiert VOR dem Lauf) | Wie H-011-Engine, außer: use_quality=False, use_gate=False, KEIN ATR-Backstop, Positionsgrößen vol-skaliert Cap 10 %, Kauf bei Rang ≤ 20. Familie: rank_out ∈ {60, 80, 100} (3 Läufe). Bestehende Positionen werden NICHT auf Zielgewicht nachgetrimmt (buy-and-hold bis Rang-Exit) — nur Neukäufe/Exits handeln. Kosten/Steuern identisch H-011. Gleiche Daten (Alpaca 2016–2026, explorativ). |
| Pass/Fail (vorab) | PASS nur wenn ALLE: (1) Netto-End > ETF-Netto-Pfad (373.261); (2) Netto-Sharpe > EW-Baseline (0,851); (3) DSR passes_5pct bei N=47; (4) PBO(3er-Familie) ≤ 0,5; (5) ≥ 3/5 2-J-Fenster ≥ EW-Baseline. Auswahl innerhalb Familie: bestes Netto-Endvermögen; Verdict auf der GEWÄHLTEN Variante, PBO deckt die Selektion. |
| Kumulatives N nach Test | 44 + 3 = 47 |
| Explorativ-Flag | JA — gleiche Datenlage wie H-011 (survivorship-biased, nicht verdict-fähig). Ehrliche Vorab-Erwartung: Kriterium (2) wird vermutlich scheitern (EW-Baseline enthält den Survivorship-Gift); der eigentliche Informationswert liegt in Kriterium (1): Kann IRGENDEINE aktive Umsetzung das deutsche Steuerleck vs. ETF schließen? |
| Ergebnis | **FAIL (4/5 Kriterien)** (2026-07-05). Einziges PASS: > ETF-Pfad. PBO 0,77. Details: ledger.md. |

---

## H-013 — Kandidat B: Regime-Rotation Indexebene (Mandat §3.2; NUR Backtest)

**⚠️ WARNBOX-BESTÄTIGUNG (Pflicht §3.2):** Gayed-Papier-Backtest stark, Live-Umsetzung
des Autors GESCHEITERT (RORO liquidiert Okt 2025, −2,9 %/J vs +14,6 % S&P). Volatility
Decay, Whipsaw an der 200-Tage-Linie und Steuerfolgen jedes Switches sind real. B gilt
als unbewiesen-aggressiv. Hebel-LÄUFE hier sind reine Backtests — kein Paper-Trade,
kein Live, keine Produktempfehlung ohne separate schriftliche Freigabe (Guardrail 4).

| Feld | Inhalt |
|---|---|
| H-ID | H-013 |
| Datum | 2026-07-05 (registriert VOR dem Lauf) |
| Hypothese | Eine monatlich geprüfte SMA200-Rotation auf SPY (über der Linie investiert, darunter defensiv) erzielt nach deutschen Steuern (jeder Switch realisiert FIFO-Gewinne, 26,375 %) ein höheres Endvermögen als der thesaurierende ETF-Netto-Pfad — trotz Steuerkosten der Switches. |
| Begründung | Time-Series-Momentum (Moskowitz/Ooi/Pedersen 2012); "Leverage for the Long Run" (Gayed 2016). SPY-only ⇒ **survivorship-IMMUN** — die einzige Kategorie, in der unsere Datenlage nicht strukturell schmeichelt (Fable Round 4 fand genau hier das einzige robuste Element). Deutsche Kernfrage: Überleben die Switch-Steuern? |
| Parameter (fixiert VOR dem Lauf) | Signal: SPY-Schluss vs. SMA200, geprüft NUR am Monatsultimo (Whipsaw-Reduktion §3.2), Ausführung next close. Familie (genau 3): (B1) 1x: über→100 % SPY, unter→Cash. (B2) 2x-synthetisch: über→2x-daily-SPY, unter→1x SPY. (B3) 2x-synthetisch: über→2x, unter→Cash. 2x-Modellierung: tägliche 2×SPY-Rendite − Finanzierungs-/TER-Drag 3,9 % p.a./252 (konservativ: ~3 % Financing + 0,9 % TER, SSO-Proxy). Cash verzinst 0 % (konservativ gegen die Strategie). Kosten 10 bps/Seite; Steuern 26,375 % FIFO + Verlusttopf. Start 100k. Daten: Alpaca SPY 2016–2026. |
| Pass/Fail (vorab) | PASS nur wenn ALLE: (1) Netto-End > ETF-Netto-Pfad (373.261); (2) DSR passes_5pct bei N=50; (3) MaxDD nicht schlechter als SPY brutto (−33,8 %); (4) PBO(3er-Familie) ≤ 0,5; (5) ≥ 3/5 2-J-Fenster Netto-Sharpe ≥ SPY-Buy&Hold. Auswahl: bestes Netto-Endvermögen; Verdict auf gewählter Variante. |
| Kumulatives N nach Test | 47 + 3 = 50 |
| Explorativ-Flag | TEILWEISE — survivorship-immun (SPY-only) ✓, ABER nur ~10,5 J Fenster (fast durchgehend Bullenmarkt: Regime-Gates können hier strukturell kaum glänzen — dokumentierte Erwartung) und 2x nur synthetisch modelliert. Verdict-fähig erst mit längerem Fenster (Norgate) über mehrere Bärenmärkte. |
| Ergebnis | **FAIL (4/5)** (2026-07-05). B2-Endwert-Sieg = Hebel-Beta ohne Timing-Skill (Sharpe 0,64 < SPY 0,90); Timing 2018-19/2022-23 negativ. Details: ledger.md. |

---

## Welle 2 (registriert 2026-07-06, VOR allen Läufen) — H-014 bis H-019

Gemeinsam: Daten wie H-011/012 (Alpaca 2016–2026 + PIT-XBRL; survivorship-biased →
explorativ, nicht verdict-fähig). Steuer-/Kosten-Engine identisch. Pflicht-Kontrolle
EW-Baseline (Sharpe 0,851 / Endwert 416.612). Auswahl je Familie vorab: bestes
Netto-Endvermögen; Verdict auf gewählter Variante; PBO je Familie; DSR bei kumulativem N.
Standard-Pass/Fail (wo nicht anders vermerkt): (1) > ETF-Netto-Pfad 373.261, (2) Netto-
Sharpe > EW 0,851, (3) DSR passes_5pct, (4) PBO ≤ 0,5, (5) ≥ 3/5 Fenster ≥ EW.

### H-014 — Tax-Loss-Harvesting-Overlay (§4.6.4) | 3 Läufe, N→53
Hypothese: Monatliches Ernten unrealisierter Verluste ≥ 15 % (Verkauf → Verlusttopf,
Wiedereinstieg nur über reguläre Entry-Logik ab Folgemonat; kein deutsches Wash-Sale-
Verbot, ≥1 Monat Abstand gegen §42-AO-Optik) erhöht das Netto-Endvermögen der
H-012-Familie (out60/80/100) durch frühere Topf-Nutzung. Paired-Design: Delta vs.
H-012-Läufe ist die Messgröße. Begründung: §2.4 Verlustverrechnung bewusst nutzen —
reiner Steuer-Mechanik-Effekt, kein Signal. Pass/Fail: PASS wenn Delta > 0 in ≥ 2/3
Paaren UND bestes TLH-Endvermögen > bestes H-012-Endvermögen (614.905); sonst FAIL.

### H-015 — ATR-Exit-Familie 3×/4×/5× (§4.6.3, als EINE Familie) | 3 Läufe, N→56
Hypothese: Ein weiter ATR-Trailing-Backstop (3×/4×/5× ATR20) verbessert das
Netto-Endvermögen von H-012-out60 (Krisenschutz > Steuer-/Whipsaw-Kosten).
Begründung: Mandat §3.1 verlangt Backstop; H-012 lief ohne — dies quantifiziert
seinen Preis. Pass/Fail: Standard + explizit: schlägt IRGENDEIN ATR-Level das
backstop-lose 614.905? (Wenn nein: Backstop ist auf diesen Daten reine Versicherungsprämie.)

### H-016 — Dual Momentum / GEM, Indexebene (survivorship-IMMUN) | 2 Läufe, N→58
Hypothese: Antonacci-GEM (monatlich: 12M-Rendite SPY vs. IEF; long SPY wenn
SPY-12M > IEF-12M UND > 0, sonst IEF) schlägt nach deutschen Steuern den ETF-Pfad.
Varianten: (a) GEM classic SPY/IEF, (b) Absolute-Momentum-only (SPY wenn 12M>0 sonst
IEF). Begründung: Antonacci 2014, Live-ETF-Evidenz gemischt; SPY/IEF-only ⇒ kein
Survivor-Bias. Zusatzdaten: IEF via Alpaca (Mini-Pull, dokumentiert). Pass/Fail:
Standard, aber Maßstab (2) = SPY-B&H-Sharpe 0,887 statt EW (indexbasiert).

### H-017 — Low-Volatility-Tilt (Baker/Haugen) | 1 Lauf, N→59
Hypothese: Top 20 nach niedrigster 252d-Vol (no-retrim, out60, Steuerdesign aus
H-012 übernommen — vorab fixiert) erzielt höheren Netto-Sharpe als die EW-Baseline.
Begründung: Low-Vol-Anomalie (Baker/Bradley/Wurgler 2011), turnover-arm von Natur.
Pass/Fail: Standard.

### H-018 — 52-Week-High-Momentum (George/Hwang 2004) | 1 Lauf, N→60
Hypothese: Ranking nach Nähe zum 52-Wochen-Hoch (close/max252) statt 12-1-Momentum
(sonst identisch zu H-012-out60) liefert höheres Netto-Endvermögen als H-012-out60
(Anker-Effekt persistiert; niedrigerer Turnover als klassisches Momentum).
Pass/Fail: Standard + direkter Paarvergleich vs. 614.905.

**Ergebnis Welle 2 (2026-07-06): ALLE 6 Familien FAIL** — H-014 (PBO 1,0; TLH schadet
2/3), H-015 (jeder ATR-Stop kostet, monoton), H-016 (survivorship-immune NULL),
H-017/H-018 (klar unter Baseline/Momentum), H-019 (Vol-Gate +46k, aber PBO 0,91 =
Rauschen). Details: ledger.md Welle-2-Block. N=63.

### H-019 — Regime-Gate-Familie (§4.6.2) | 3 Läufe, N→63
Hypothese: Ein Vol-Regime-Gate (SPY-realized-20d-Vol > rollierendes 80. Perzentil →
Exposure 50 %) oder die Kombination Vol×SMA200 verbessert H-012-out60 netto
(Vol-Gates reagieren schneller als Trend-Gates, geringere Whipsaw-Steuerkosten).
Varianten: (a) SMA200-Gate, (b) Vol-Gate, (c) beide (AND → 50 %, einer → 75 %).
Begründung: Moreira/Muir 2017 (volatility-managed portfolios). Pass/Fail: Standard +
Paarvergleich vs. gate-loses 614.905.

---

## Welle 3 (registriert 2026-07-07, VOR allen Läufen) — H-020 bis H-022: No-Signal-Steuer-Designs

Idee: Survivorship-Bias kürzt sich im PAAR No-Signal-vs-No-Signal heraus — diese
Familie misst NUR den deutschen Steuer-/Turnover-Effekt der Umsetzung (§4.6.4, der
"realste Edge"). Referenz: EW-Baseline monatlich (416.612 / Sharpe 0,851). Daten,
Engine, Kosten/Steuern wie gehabt (explorativ).

### H-020 — EW mit Rebalancing-Bändern | 2 Läufe, N→65
Hypothese: EW-Portfolio, das NUR rebalanced, wenn ein Gewicht sein Ziel um > B
(relativ) verlässt (B ∈ {25 %, 50 %}), schlägt die monatlich rebalancete EW-Baseline
netto (weniger Realisation = Stundung; Band-Rebalancing hält Struktur).
Pass/Fail: PASS wenn beide (oder das gewählte B) > 416.612 UND Sharpe ≥ 0,80
(nicht wesentlich unter Baseline); Auswahl: bestes Netto-End.

### H-021 — Momentum mit JÄHRLICHEM Rebalancing | 1 Lauf, N→66
Hypothese: H-012-out60, aber Signal-/Rebalance-Prüfung nur jährlich (Januar-Ultimo),
maximiert Stundung und schlägt das monatliche 614.905.
Pass/Fail: > 614.905 UND Sharpe > 0,70.

### H-022 — Buy-and-Hold-Extrem (Stundungs-Obergrenze) | 1 Lauf, N→67
Hypothese: Einmal EW kaufen (erster Monat), danach NIE handeln — quantifiziert die
theoretische Obergrenze der Steuerstundung vs. monatliche EW-Baseline.
Kein Pass/Fail im Strategie-Sinn (Diagnose-Messlatte); zählt trotzdem als Trial.

**Ergebnis Welle 3 (2026-07-07): H-020 PASS** (erster des Mandats; beide Bänder >
Referenz UND Sharpe ≥ 0,80; PBO 0,00; mechanischer Steuer-Effekt, kein Alpha-Claim,
explorativ). **H-021 FAIL** (577k < 615k). **H-022 Diagnose:** Stundungs-Obergrenze
+283k/Steuern 0 — nur relativ interpretierbar (B&H-Survivor-Universum = Hindsight).
Details: ledger.md Welle-3-Block. N=67.

---

## Welle 4 (registriert 2026-07-07, VOR allen Läufen) — H-023 bis H-025: ERSTE VERDICT-FÄHIGE TESTS

**Datenbasis (neu):** EODHD-Preise (adjusted_close) für ALLE jemals-S&P-500-Mitglieder
1996–2026 (1.166/1.202 Ticker, 418 delistete; Rest-Lücke: 36 Fails, großteils
Bankruptcy-Q-Ticker wie LEHMQ/WCOEQ → geringe Rest-Schmeichelei, dokumentiert) ×
PIT-Konstituenten-Historie (fja05680, 2.713 Snapshots). Fenster 1997–2026 (~29 J,
enthält 2000-03, 2008, 2020, 2022). Engine-Konventionen zusätzlich: handelbar am
Rebalance-Tag = aktuelles Index-Mitglied (PIT-Snapshot ≤ as_of); Nicht-mehr-Mitglieder
fallen automatisch aus dem Ranking (= Exit über rank_out-Regel); **Delisting-Handling:**
endet die Kurshistorie eines gehaltenen Titels, Zwangsverkauf zum LETZTEN verfügbaren
Kurs am Folgetag (kein Hindsight; Merger≈fair, Bankruptcy=bereits kollabierter Kurs).
Kein ATR-Backstop (keine H/L-Daten; die getesteten Designs nutzen keinen).
EW-PIT-Baseline (inkl. der Toten) ist jetzt eine EHRLICHE Baseline. SPY.US = Benchmark;
ETF-Netto-Pfad 18,5 % auf Endgewinn.

### H-023 — VERDICT-Re-Test Steuer-Momentum (H-012-Familie) | 3 Läufe, N→70
Hypothese: 12-1-Momentum Top 20, no-retrim, rank_out ∈ {60, 80, 100}, schlägt nach
deutschen Steuern den ETF-Netto-Pfad über 1997–2026 auf survivorship-freiem
PIT-Universum. Pass/Fail (VOLL): (1) > ETF-Pfad, (2) Netto-Sharpe > EW-PIT-Baseline,
(3) DSR passes bei N=70, (4) PBO ≤ 0,5, (5) ≥ 4/7 4-J-Fenster ≥ EW-PIT. DIE zentrale
Mandats-Frage unter ehrlichen Bedingungen.

### H-024 — VERDICT-Re-Test EW-Band-Rebalancing (H-020) | 2 Läufe, N→72
Hypothese: Band-Rebalancing (25 %/50 %) schlägt monatliches EW-Rebalancing netto auch
auf PIT-Universum mit Delistings (Paar-Design). Pass/Fail: beide > EW-PIT-monatlich
UND Sharpe nicht wesentlich darunter (≥ −0,05); PBO der 2er-Familie ≤ 0,5.

### H-025 — VERDICT-Re-Test Vol-Gate (H-019) | 3 Läufe, N→75
Hypothese: Vol-Gate (bzw. SMA200, bzw. beide) auf H-023-out60 verbessert Netto-Sharpe
UND reduziert MaxDD über ein Fenster MIT echten Bärenmärkten — der faire Test, den
2016–2026 nicht liefern konnte. Pass/Fail: gewählte Variante: Sharpe > gate-los UND
MaxDD ≥ 10 %p besser UND DSR passes bei N=75 UND PBO ≤ 0,5.

**Ergebnis Welle 4 (2026-07-07, nach Datenhygiene-Korrektur — Lauf 1 war kontaminiert,
Details ledger.md):** **H-023 FAIL** (1,48M < ETF 1,61M; DSR 0,94; PBO 0,83; MaxDD −93 %).
**H-024 PASS — erster verdict-fähiger PASS des Mandats** (Band-Rebalancing schlägt
Kalender-EW +24–29 % netto, PBO 0,43; Steuer-Mechanik, kein Alpha-Claim).
**H-025 FAIL** (MaxDD-Kriterium klar verfehlt: Gates verhindern das −93 %-Tal nicht;
DSR 0,97/PBO 0,31 der both-Variante nur via NEU registriertem Confirmatory-Test
weiterverfolgbar). N=75.

---

## Welle 5 (registriert 2026-07-07, VOR allen Läufen)

### H-026 — Confirmatory: gate_both auf Verdict-Daten | 6 Läufe, N→81
Herkunft (transparent): Rückschau-Beobachtung aus H-025 (gate_both: 1,94M, DSR 0,97,
PBO 0,31 — verfehlte aber sein MaxDD-Vorab-Kriterium). Confirmatory-Design: Der Effekt
(„kombiniertes SMA×Vol-Gate hebt das ENDVERMÖGEN von Verdict-Momentum über den
ETF-Pfad") gilt nur als real, wenn er PARAMETER-STÖRUNGEN überlebt — echte neue Daten
existieren nicht, also ist Nachbarschafts-Robustheit + Hälften-Konsistenz der
strengste verfügbare Test. Familie (genau 6, vorab): {Vol-Perzentil 70/80/90} ×
{SMA 150/250} kombiniert als (P70,S150), (P70,S250), (P80,S150), (P80,S250),
(P90,S150), (P90,S250) — die Original-Kombi (P80,S200) ist bewusst NICHT dabei
(sie ist die Quelle der Beobachtung, nicht ihr Test).
Pass/Fail (vorab, ALLE nötig): (1) ALLE 6 Störungen > gate-los out60 (1.214.720) im
Endvermögen; (2) ALLE 6 > ETF-Pfad (1.610.149); (3) Hälften-Konsistenz: gewählte
Variante > gate-los in BEIDEN 15-J-Hälften (Sharpe); (4) DSR passes bei N=81;
(5) PBO(6er) ≤ 0,5. Scheitert EINE Bedingung → Beobachtung war Rauschen, Thema ZU.

**Ergebnis (2026-07-07): FAIL — crit1 ✗ (4/6 unter gate-los), crit2 ✗ (6/6 unter
ETF-Pfad), crit5 ✗. Die 1,94M-Beobachtung war ein isolierter Parameter-Peak.
Thema kombinierte Gates ZU.** N=81.

### H-027 — Praxistauglichkeit des Band-Rebalancing: 50-Namen-Subset | 3 Läufe, N→84
Herkunft: H-024 (einziger verdict-fähiger PASS) nutzt ~500 Namen — operativ
unrealistisch für Retail. Hypothese: Der Band-Vorteil (Steuer-/Kosten-Mechanik)
überlebt die Verkleinerung auf ein 50-Namen-Portfolio (Top 50 nach trailing
252d-Dollar-Volumen (close×volume), jährlich im Januar bestimmt — Liquiditäts-,
kein Alpha-Kriterium). Läufe (genau 3): EW50 monatlich (Referenz), EW50 Band 25 %,
EW50 Band 50 %. Pass/Fail (vorab): beide Bänder > EW50-monatlich im Endvermögen UND
Sharpe ≥ −0,05 relativ UND PBO(2 Band-Läufe) ≤ 0,5. Paar-Design ⇒ Bankruptcy-Caveat
neutral. |

**Ergebnis (2026-07-07): formal FAIL** — final ✓✓ (+29/+35 %), Sharpe ✓, PBO 0,543 ✗.
Design-Learning: 2-Trial-PBO ist ein Münzwurf-Maß (Registrierungsfehler, keine
nachträgliche Aufweichung). Ökonomisch dritte konsistente Replikation der
Band-Richtung; Praxis-Empfehlung stützt sich allein auf H-024. N=84.

---

### Insider-Patrone (§4.6.1) — Prüfung 2026-07-07: BLEIBT AUFGESPART
Preisseite ist jetzt survivorship-frei ✓, aber das verfügbare Form-4-Universum
(data/raw/insider_congress/form4_insider_full.parquet) deckt nur **260 Symbole**
(Watchlist-Survivors; zudem Datenschmutz: transaction_date bis 2050). Die eine
erlaubte Patrone auf einem survivor-lastigen INSIDER-Universum zu verschießen,
würde den Aufspar-Zweck verfehlen. Voraussetzung für Registrierung: Form-4-Pull
über das volle jemals-Mitglieder-Universum (~1.200 CIKs × 30 J — Multi-Tage-
EDGAR-Job, als Vorbereitung notiert) + transaction_date-Sanitisierung.

### Benchmark-Notiz (kein Trial): Vorabpauschale im ETF-Netto-Pfad
Der ETF-Pfad (18,5 % auf Endgewinn, keine Vorabpauschale) ist dokumentiert
KONSERVATIV GEGEN UNS (Benchmark eher zu stark). Vorabpauschale würde den Pfad um
grob 3–6 % Endwert über 30 J senken (Niedrigzins-Jahre oft 0) — kippt KEIN
bisheriges Verdict (H-023-Bestes 1,48M bliebe unter ~1,52–1,56M). Präzisierung
lohnt erst, wenn ein Kandidat in dieser Kante landet.

---

## Modell-Erweiterung (2026-07-07, kein Trial): Dividendensteuer in der Verdict-Engine
26,375 % auf Brutto-Dividende am Ex-Tag als Cash-Abzug (adjusted_close reinvestiert
brutto — nur die Steuer wird entnommen ≈ Netto-Reinvestition). Dividenden NICHT gegen
Aktien-Verlusttopf verrechenbar; Sparerpauschbetrag ignoriert (konservativ). ETF-Pfad
unverändert (thesaurierend, Vorabpauschale weiterhin zu unseren Ungunsten ignoriert)
→ der Vergleich wird STRENGER gegen aktive Strategien. Sensitivitäts-Reruns der
Kern-Verdicts (EW-PIT, H-023-selected, H-024-Paar) ersetzen die div-freien Läufe
(Modell-Korrektur, keine neuen Trials — Konvention wie Datenhygiene-Fix).

## Welle 6 (registriert 2026-07-07, VOR Lauf) — H-028: GEM International (Original-Formulierung)

| Feld | Inhalt |
|---|---|
| H-ID | H-028 |
| Hypothese | Antonaccis Original-GEM (monatlich: 12M-Rendite SPY vs. EFA; das Bessere, wenn dessen 12M > 0, sonst IEF) schlägt nach deutschen Steuern (26,375 % FIFO je Switch + Dividendensteuer-Drag; KEINE Teilfreistellung — konservativ) den ETF-Netto-Pfad desselben Fensters. |
| Begründung | §2.2-Live-Evidenz für Momentum; H-016 testete nur die US-verkürzte Version — die Original-Formulierung (internationale Diversifikation als Rotationsmenge) ist die publizierte. Survivorship-immun (nur ETFs/Indizes). EODHD-Preise EFA ab 2001. |
| Parameter | Familie (genau 2): (a) GEM classic (SPY/EFA/IEF wie oben); (b) relative-only (immer das bessere von SPY/EFA, kein Absolut-Gate). Monatsultimo-Signal, next-close-Execution, 10 bps/Seite, Div-Steuer-Drag aktiv. Fenster: ab EFA+12M-Warmup (~2002/03) bis 2026. |
| Pass/Fail (vorab) | PASS nur wenn: (1) > ETF-Netto-Pfad gleiches Fenster; (2) DSR passes bei N=86; (3) MaxDD nicht schlechter als SPY B&H; (4) PBO(2er) informativ berichtet, nicht bindend (Lehre H-027); (5) ≥ Hälften-Konsistenz (beide Hälften Sharpe ≥ SPY B&H − 0,05). |
| Kumulatives N nach Test | 84 + 2 = 86 |
| Explorativ-Flag | NEIN (survivorship-immun, ~24 J Fenster mit 2008+2020+2022) — verdict-fähig mit der Einschränkung „nur ein Bärenmarkt-Paar im Fenster". |
| Ergebnis | **FAIL total** (2026-07-07): 506k < ETF 957k (−47 %), DSR 0,39, MaxDD ✗, Hälften ✗. Rotations-Familie über 3 Varianten tot. N=86. Details: ledger.md. |

---

## Welle 7 (registriert 2026-07-07, VOR Lauf) — H-029: 13F-Top-Manager-Konsens („Best Ideas")

| Feld | Inhalt |
|---|---|
| H-ID | H-029 |
| Hypothese | Aktien, die bei ≥ K der 100 größten 13F-Filer (PIT-Ranking je Quartal nach Portfolio-Summe) zu den Top-10-Positionen (nach Portfoliogewicht) zählen, outperformen als EW-Basket nach deutschen Steuern (inkl. Div-Steuer) den ETF-Netto-Pfad — „High-Conviction-Konsens". |
| Begründung | Cohen/Polk/Silli (2010) „Best Ideas": Manager-Top-Positionen tragen Alpha, der Rest verwässert. UNTERSCHIED zur gesperrten Congress-Copy: nicht Personen-Kopie mit Markt-Korrelation ~0,95, sondern Konzentrations-Extrakt über 100 institutionelle Portfolios; erwartete Korrelation < 0,9, prüfbar. Neuer Raum, erster 13F-Test des Projekts. |
| Parameter (fixiert) | Datenbasis: 13f_top100.parquet (52 Q, Top-100 PIT) + FTD-CUSIP-Map + Verdict-Preise (survivorship-frei) + Div-Steuer. Signal je Quartal: je Manager Top-10-Holdings nach VALUE-Gewicht; Konsens-Zähler je CUSIP→Ticker. PIT-Wirksamkeit: Monatsultimo nach der 45-Tage-Filing-Deadline (Q-Ende + 2 Monate, konservativ). Portfolio: alle Namen mit Konsens ≥ K, EW, Cap 10 %, no-retrim; Halten bis Konsens < K/2 (Turnover-Puffer). Kosten/Steuern Standard. Familie (genau 2): K ∈ {5, 10}. Fenster ~2013-08→2026 (Einschränkung: enthält 2020/2022, NICHT 2008). |
| Pass/Fail (vorab) | ALLE: (1) > ETF-Netto-Pfad gleiches Fenster; (2) Netto-Sharpe > EW-PIT-Baseline gleiches Fenster (Div-Steuer-Version); (3) DSR passes bei N=88; (4) ≥ 3/5 2-J-Fenster ≥ EW-PIT; (5) MaxDD nicht schlechter als SPY. PBO(2er) nur informativ (Lehre H-027). Zusatz-Diagnose: Korrelation zu SPY berichten (Congress-Copy-Abgrenzung). |
| Kumulatives N nach Test | 86 + 2 = 88 |
| Explorativ-Flag | TEILWEISE — Preise survivorship-frei ✓, 13F-Top-100 PIT ✓, aber nur ~13 J Fenster ohne 2008. |
| Ergebnis | **FAIL (DSR 0,73)** nach 2 Datenfixes (Perioden-Ranking; NaN→0-Bewertungsbug — Läufe 87/88 kontaminiert, ersetzt). ABER: stärkster Kandidat bisher — k10: +56 % über ETF-Pfad netto, MaxDD = SPY, Steuern 18k/13J, 4/5 Kriterien ✓. Skepsis: SPY-Korr 0,936, Fenster ohne 2008 = Growth-Beta nicht trennbar. Weiterverfolgung NUR als neu registrierter Confirmatory (Störungen + Mega-Cap-Beta-Kontrolle). N=88. |

---

## Welle 8 (registriert 2026-07-08, VOR Lauf) — H-030: Confirmatory 13F-Konsens

Herkunft (transparent): H-029-k10 (formal FAIL via DSR, aber 4/5 ✓, +56 % über
ETF-Pfad). Confirmatory nach H-026-Muster: Der Effekt gilt nur als real, wenn er
(A) Parameter-NACHBARN und (B) eine BETA-KONTROLLE überlebt.

| Feld | Inhalt |
|---|---|
| H-ID | H-030 |
| Störungsfamilie (genau 6; Original Top10/K10/M100 bewusst AUSGESCHLOSSEN) | (Top8,K10,M100) · (Top12,K10,M100) · (Top10,K7,M100) · (Top10,K13,M100) · (Top10,K10,M50) · (Top10,K10,M150) |
| Beta-Kontrolle (Benchmark, kein Trial) | VALUE-gewichteter Basket ALLER Holdings der Top-100-Manager („13F-Markt-Portfolio", gleiche Steuer-Engine, gleiche PIT-Timing-Regel) — misst, ob die Top-10-KONZENTRATION etwas addiert oder ob es nur „was Institutionen eh halten" (Mega-Cap-Beta) ist. |
| Pass/Fail (vorab, ALLE nötig) | (1) ALLE 6 Nachbarn > ETF-Netto-Pfad; (2) Median-Nachbar-Endwert > Beta-Kontrolle-Endwert; (3) PBO(6er) ≤ 0,5; (4) ≥ 4/6 Nachbarn Sharpe > EW-PIT-Fenster (0,55). DSR informativ (Familie teilt einen Effekt). Scheitert eine Bedingung → 13F-Konsens ist Growth-Beta-Rauschen, Thema zu. |
| Kumulatives N nach Test | 88 + 6 = 94 |
| Explorativ-Flag | wie H-029 (Fenster ohne 2008). |
| Ergebnis | **Formal FAIL (PBO 0,74)**, substanziell stark: ALLE 6 Nachbarn > ETF-Pfad (633–813k), Median +104 % über Beta-Kontrolle, 6/6 Sharpe > EW. Thema auf diesen Daten registry-konform ausgereizt — weiter nur via 13F-vor-2013-Parser (2008-Fenster) oder Paper-Tracking (OOS in Echtzeit). N=94. |

---

## Welle 9 (registriert 2026-07-08, VOR Lauf) — H-031 Insider-Patrone + H-032 Dividend-Tilt

### H-031 — §4.6.1-Insider-Retest (DIE EINE erlaubte Patrone) | 2 Läufe, N→96 nach H-032
Voraussetzung ERFÜLLT: Form-4-Breitpull ~komplett (28 Tranchen, ~570k+ Zeilen, volle
jemals-Mitglieder inkl. Delisted; PIT via available_at) + survivorship-freie Preise.
Hypothese (Cohen/Malloy/Pomorski 2012): OPPORTUNISTISCHE Insider-Open-Market-Käufe
(Code P, non-derivative; Insider OHNE Routine-Muster = kauft NICHT im selben
Kalendermonat in ≥3 der letzten 3 Jahre) prognostizieren 6–12M-Überrenditen; ein
monatlich gebildeter EW-Basket (Titel mit ≥1 opportunistischem Kauf in den letzten
3 Monaten, Halten 12 Monate, Cap 10 %, $1-Floor) schlägt netto (inkl. Div-Steuer)
den ETF-Pfad UND die EW-PIT-Baseline. Familie (genau 2): (a) alle opportunistischen
Käufe; (b) nur Officer/Director-Käufe ≥ 10.000 $ Volumen (role-Filter). Pass/Fail
(ALLE): (1) > ETF-Pfad gleiches Fenster; (2) Sharpe > EW-PIT-Fenster; (3) DSR passes
bei N=98; (4) ≥ 3/5 Fenster ≥ EW; (5) MaxDD nicht schlechter als SPY. Fenster:
Form-4-Daten flächig ab ~2004 → Test ~2005–2026 (~21 J, inkl. 2008!). DANACH IST
DAS INSIDER-FELD ENDGÜLTIG ZU (Mandat §4.6.1).

**Ergebnis (2026-07-08): FAIL (DSR 0,705 ✗; MaxDD −57,5 % vs SPY −55,2 % ✗) —
INSIDER-FELD ENDGÜLTIG ZU.** Substanziell zweitstärkstes Mandats-Ergebnis:
officer_10k +68 % über ETF-Pfad über 21 J inkl. 2008, 6/6 Fenster > EW, PBO 0,14,
Sharpe 0,660 > SPY 0,642. Details: ledger.md Welle 9b. N=98.

### H-032 — Dividend-Tilt: Deutsche Steuer bestraft Ausschüttung | 2 Läufe, N→96 (läuft VOR H-031)
Hypothese: Ein No-/Low-Dividend-Basket (Top 50 nach NIEDRIGSTEM trailing-12M-Yield,
inkl. Null-Zahler) schlägt einen High-Yield-Basket (Top 50 höchster Yield) nach
deutschen Steuern deutlich — weil Dividenden Zwangsrealisation ohne Stundung sind
(26,375 % sofort, kein Verlusttopf-Offset), während Kursgewinne gestundet werden.
Begründung: §4.6.4 (Steuer-Design = realer Edge); rein mechanische Steuer-These,
kein Alpha-Claim. Parameter: jährliches Signal (Januar-Ultimo, PIT: trailing-Divs /
aktueller Kurs), Kauf Top 50, Exit außerhalb Top 100 der jeweiligen Rangliste,
EW-Slots, no-retrim, Cap 10 %, $1-Floor, volle Steuer-Engine inkl. Div-Drag.
Familie = die ZWEI Pole (low_div, high_div) — Paar-Design, Survivorship im Paar
neutral. Pass/Fail (vorab): PASS wenn (1) LowDiv-Endwert > HighDiv-Endwert × 1,10
(Effekt muss ≥ 10 % betragen) UND (2) LowDiv > EW-PIT-Fenster-Endwert. DSR/PBO
informativ (mechanischer Paar-Effekt, Konvention H-020).

**Ergebnis (2026-07-08): PASS — 2. verdict-fähiger PASS des Mandats.** Ratio 4,19×
netto (1,42× brutto → Steuer-Keil ×2,95, Dekomposition im Ledger); high_div verliert
−73 % an die Steuer, low_div −21 %. N=96.

---

## Welle 10 (registriert 2026-07-09, VOR Lauf) — H-033: Congress-Trades-Re-Test

**⚠️ SPERRLISTEN-OVERRIDE (dokumentiert):** Congressional-Copy steht auf der
Sperrliste §2.3 (NANC/KRUZ ~0,95 SPY-Korrelation = teures Beta, live belegt).
**Hans hat als alleinige Entscheidungsinstanz am 2026-07-09 einen differenzierten
Re-Test angeordnet** (Begründung: Politiker-Vermögenszuwächse der Trump-Ära; neue
Infrastruktur). Warnbox bleibt: Erwartung ist SPY-Klon für das Copy-Muster; der Test
MUSS die SPY-Korrelation je Variante ausweisen. Ehrlichkeits-Fußnote: Der Präsident
selbst ist NICHT in STOCK-Act-Daten (keine PTR-Pflicht; DJT/Krypto-Vehikel nicht
PIT-handelbar) — getestet wird der KONGRESS.

| Feld | Inhalt |
|---|---|
| H-ID | H-033 |
| Hypothese | Kongress-Käufe (STOCK Act, PIT via available_at ≈ Disclosure + Lag) tragen handelbare Information, die einen 12M-Halte-Basket nach deutschen Steuern über den ETF-Pfad hebt — WENN man differenziert (Größe/Konsens) statt blind kopiert. |
| Daten | congress_trades_full.parquet: 14.593 BUYS, 311 Mitglieder, 2012–2026. Preisbasis: Verdict-Universum (Schnitt wird berichtet; Non-S&P-Picks fehlen zunächst — dokumentierte Grenze). Fenster ~2013–2026 (OHNE 2008). |
| Familie (genau 3) | (a) copy_all: alle BUYS (Kontroll-Replikation des gesperrten Musters); (b) big_buys: amount_low ≥ 50.000 $; (c) cluster: ≥ 3 Mitglieder kaufen dasselbe Symbol binnen 3 Monaten. Mechanik je: 12M-Halten (frisches Signal verlängert), EW-Slots, Cap 10 %, $1-Floor, volle Steuern + Div-Drag (identisch H-031-Engine). |
| Pass/Fail (vorab, ALLE) | (1) > ETF-Pfad gleiches Fenster; (2) Sharpe > EW-PIT-Fenster; (3) DSR passes bei N=101; (4) ≥ 3/5 Fenster ≥ EW; (5) MaxDD nicht schlechter als SPY. Pflicht-Diagnose: SPY-Korrelation je Variante. |
| Kumulatives N nach Test | 98 + 3 = 101 |
| Explorativ-Flag | TEILWEISE — Preise survivorship-frei ✓, aber Fenster ohne 2008 und Congress-Universum > Preisbasis (Schnitt-Bias möglich, wird beziffert). |
| Ergebnis | **FAIL** (2026-07-09): alle 3 unter ETF-Pfad; copy_all SPY-Korr **0,965** = Sperrlisten-Begründung mit eigenen Daten reproduziert; Differenzierung verschlechtert (big_buys −56 %). Details: ledger.md Welle 10. N=101. |

---

## Welle 10b (registriert 2026-07-09, VOR Lauf) — H-034: Congress LETZTE Patrone (volles Universum)

Von Hans freigegeben („los geht's"). H-033 verlor 36 % der Käufe (Non-S&P-Picks) —
H-034 schließt diese Lücke via EODHD-Pull der fehlenden Congress-Symbole (inkl.
Delisted soweit vorhanden) und wiederholt die IDENTISCHE 3er-Familie (copy_all,
big_buys ≥50k, cluster ≥3) mit UNVERÄNDERTEN Kriterien auf dem vollen Universum.
Pass/Fail wie H-033 (ETF-Pfad/EW/DSR@104/Fenster/MaxDD + SPY-Korr-Diagnose).
Kumulatives N nach Test: 101 + 3 = 104. **Danach ist das Congress-Feld — Override
hin oder her — endgültig zu** (zweite Patrone verbraucht; Prior nach H-033: sehr
niedrig; ehrliche Erwartung ist Bestätigung des FAIL).

**Ergebnis (2026-07-09): FAIL, deutlicher als H-033** — volles Universum macht es
SCHLECHTER (copy_all 311k vs 512k; Non-S&P-Picks sind zusätzliche Verlierer).
**CONGRESS-FELD ENDGÜLTIG ZU.** N=104. Details: ledger.md Welle 10b.

---

## Welle 11 (registriert 2026-07-09, VOR den Läufen) — Small-Cap-Suchraum

**Universum (neutral, PIT-fair):** ALLE NYSE/NASDAQ Common Stocks inkl. Delisted
(~27k Ticker, EODHD, ab 2000). Cap-Proxy OHNE Lizenz-Daten: je Jahres-Ultimo
Dollar-Volumen-Ranking aller handelbaren Titel; „Small-Cap-Band" = Perzentil 20–60
(vorab fixiert; unter P20 = untradeable Mikro, über P60 = Mid/Large). Kosten
REALISTISCH: **30 bps je Seite** (statt 10). $2-Floor. Datenhygiene wie Verdict-
Engine (Sprung-Kappung). Steuern voll inkl. Div-Drag (Dividenden für Small-Band
nachziehen soweit nötig).

### H-035 — Small-Cap-Momentum netto | 3 Läufe, N→107
Hypothese: 12-1-Momentum (Top 30 im Band, no-retrim, rank_out-Familie {90, 120, 150})
überlebt 30 bps + deutsche Steuern und schlägt den ETF-Netto-Pfad. Begründung:
Momentum-Prämie historisch in Small Caps am stärksten (Jegadeesh/Titman-Splits;
Israel/Moskowitz 2013) — aber genau dort fressen Kosten am meisten. Pass/Fail (ALLE):
(1) > ETF-Pfad; (2) Sharpe > EW-Band-Baseline (No-Signal-Kontrolle im SELBEN Band —
Survivorship-/Band-Effekte kürzen sich); (3) DSR passes bei N=107; (4) ≥ 60 %
2-J-Fenster ≥ EW-Band; (5) MaxDD nicht schlechter als EW-Band.
**Ergebnis: FAIL (4/5)** (2026-07-09, N=126): out120 final 63.115 aus 100k (CAGR −1,8 %,
MaxDD −0,911), Sharpe 0,309 < EW-Band-Small 0,413; PBO 0,886, DSR-p 0,071, Konsistenz 0,16.
Momentum-SELEKTION vernichtete >99 % ggü. EW-Band-Kontrolle auf SELBEM Universum. Small-Cap-
Momentum = Kapitalvernichter nach Kosten/Steuer/Delisting. Universum: 15.101 je-handelbar von
21.917 (survivorship-frei). Details: ledger Welle-11-Nachlauf.

### H-036 — Size-Prämie selbst (EW-Band-Mechanik) | 2 Läufe, N→109
Hypothese: Das validierte Band-Rebalancing (50 %-Band) auf dem Small-Band schlägt
dasselbe Design auf dem Large-Band (P80+) netto — trägt die Size-Prämie nach
realistischen Kosten/Steuern? Paar-Design. Pass/Fail: Small > Large × 1,10 UND
Small > ETF-Pfad; sonst FAIL (Size-Prämie tot oder von Kosten gefressen).
**Ergebnis: FORMAL PASS, substanziell FAIL (§2.5-Artefakt)** (2026-07-09, N=126): Small 5,97M
> Large 3,53M×1,10 > ETF 736k ✓ — ABER Liquiditäts-Floor-Test entlarvt es: bei ADV≥$50M (echt
liquide, 541 Namen) fällt final auf 1,90M, Überschuss-über-ETF −75 %, Sharpe 0,369 < SPY 0,545;
MaxDD −0,957…−0,982 (uninvestierbar); Micro-Cap-Impact/Bid-Ask-Bounce nicht mal modelliert.
Micro-Cap-Illiquiditäts-Artefakt, KEIN deployabler Size-Edge. Details: ledger Welle-11-Nachlauf
+ h036b/h036c robustness json.

---

## Welle 12 (registriert 2026-07-09, VOR Lauf; läuft VOR H-035/036) — H-037: Krypto-§23-Steuer-Keil

Scope-Erweiterung von Hans freigegeben („Abo-Daten nutzen"). KEIN „Krypto schlägt
Aktien"-Claim (reine Rückschau) — sondern der mechanische deutsche Steuer-Keil:
**§23 EStG: Krypto ≥ 1 Jahr Haltefrist = 0 % Steuer; < 1 Jahr = persönlicher
Einkommensteuersatz (~44 % inkl. Soli, Annahme Spitzensatz)** — der extremste
Stundungs-Fall überhaupt, in Linie mit H-024/H-032.

| Feld | Inhalt |
|---|---|
| H-ID | H-037 |
| Hypothese | Ein aktiver Krypto-Händler (monatliches SMA200-Gate long/cash — der repräsentative „Timing"-Ansatz, vorab fixiert) verliert gegenüber diszipliniertem HODL (Haltefrist ≥ 1 J → steuerfrei) einen Steuer-Keil ≥ 20 % des Endvermögens; die Steuer (nicht das Timing allein) ist ein wesentlicher Treiber (aktiv-netto < aktiv-brutto × 0,85). |
| Läufe (2 Trials) | BTC (2011–2026), ETH (2016–2026): aktiv-netto je Asset. HODL + aktiv-brutto = Benchmarks/Dekomposition (kein Selektionsdruck). Kosten 20 bps/Seite. FIFO; Gewinn nur steuerpflichtig bei Haltedauer < 365 T; Verlustverrechnung vereinfacht voll. |
| Pass/Fail (vorab) | PASS wenn BEIDE Assets: HODL > aktiv-netto × 1,20 UND aktiv-netto < aktiv-brutto × 0,85. |
| Ergebnis | **PASS** (2026-07-09): Steuer-Keil BTC −55,9 %, ETH −28,4 %; beide Kriterien ✓. 3. Mandats-PASS, wieder Stundungs-Prinzip. HINWEIS: absolute Werte reiner Hindsight, nur relativer Keil zählt. N=106. Details: ledger.md Welle 12. |
| Kumulatives N nach Test | 104 + 2 = 106 (H-035/036 belegen danach 107–111; Ledger führt die zeitliche Wahrheit) |
| Guardrails | Spot only (kein Hebel/Derivat, Guardrail 4 unberührt); reine Analyse, kein Trade. |

**Daten-Inventar-Entscheid (EODHD-Abo, 2026-07-09):** Krypto ✓ genutzt (H-037).
FX: als Daten ok, als Strategie Guardrail-4-blockiert (Retail-FX = Derivate/Hebel) —
nur ggf. künftig als Regime-Input. News/Sentiment: API verfügbar, aber §2.3-GESPERRT
als Alpha-Signal (Kollaps bei Kosten, Lookahead) — wird NICHT getestet ohne
expliziten Override. Intraday: gesperrt + empirisch beerdigt (Fable R5) — ungenutzt
bleibt hier richtig, nicht verschwendet.

---

## Welle 13 (registriert 2026-07-09, VOR Lauf) — H-038: News-Sentiment (Sperrlisten-Override Hans)

**⚠️ SPERRLISTEN-OVERRIDE:** LLM-News-/Sentiment-Signale stehen auf §2.3 (Kollaps bei
~10 bps Kosten; Lookahead). **Hans hat Nutzung explizit angeordnet** („vergiss auch
nicht sowas wie news"). Der Test ist so gebaut, dass er GENAU die Sperr-Gründe
falsifiziert: PIT-sauber (Signal Monatsultimo T, Ausführung T+1-Close — kein
Same-Day-Lookahead) und mit ansteigenden Kosten (die Sperrliste behauptet Kollaps
bei 10 bps → wir testen 5/10/20).

| Feld | Inhalt |
|---|---|
| H-ID | H-038 |
| Hypothese | Ein monatlich rebalancierter Long-Basket der Titel mit höchstem trailing-1M-Durchschnitts-Sentiment (EODHD `normalized`, PIT) schlägt netto (Kosten + deutsche Steuern inkl. Div) die EW-PIT-Baseline des survivorship-freien S&P-Universums. |
| Daten | EODHD-Sentiment (ab 2012) für Verdict-Universum; Fenster ~2013–2026. Preise survivorship-frei. |
| Familie (genau 3, = Kostenstufen) | Top-30-Sentiment-Basket, no-retrim, Halten bis Rang > 60; identisch außer Kosten ∈ {5, 10, 20} bps/Seite. |
| Pass/Fail (vorab, ALLE) | (1) bei 10 bps: > ETF-Pfad; (2) bei 10 bps: Sharpe > EW-PIT; (3) DSR passes bei N=112; (4) ≥ 60 % Fenster ≥ EW; (5) Effekt überlebt 20 bps (Sharpe@20 > EW). Falsifiziert die Sperr-Behauptung nur bei komplettem PASS. |
| Kumulatives N nach Test | +3 (Ledger führt zeitliche Reihenfolge ggü. H-035/036) |
| Explorativ-Flag | TEILWEISE — Sentiment-Coverage für Delisted lückenhaft (wird beziffert); Fenster ohne 2008. |
| Ergebnis | **FAIL total** (2026-07-09): Sharpe 0,46/0,43/0,36 @5/10/20bps — schon VOR Kosten < EW 0,59; DSR 0,166; < ETF-Pfad. **Sperrliste §2.3 mit eigenen PIT-Daten reproduziert** („Kollaps bei 10bps"). N=109. Details: ledger.md Welle 13. |

---

## Welle 14 (registriert 2026-07-09, VOR Lauf) — H-039: Geopolitik-News → Crisis-Alpha-Sektorrotation

Von Hans angeordnet („geopolitische Ebene, nicht nur Finanz-News"). Reaktiviert die
Crisis-Alpha-Vision (Hormuz→Öl). **STARKER NULL-PRIOR (dokumentiert):** Fable Round 5
testete die geopolitische TAGES-These via GPR-Index (1985–2026) → NULL (Spikes
mean-reverten; energy −1,6 %/defense −2,2 % Folgemonat; echter Move in ersten Minuten
= paid intraday). H-039 prüft, ob die ZEITNÄHEREN EODHD-News-Artikel (nicht der träge
GPR-Monatsindex) ein Signal liefern, das GPR nicht hatte.

| Feld | Inhalt |
|---|---|
| H-ID | H-039 |
| Signal | Tägliche geopolitische News-Intensität = distinkte EODHD-Artikel mit Konflikt-Tags (WAR/RUSSIA/UKRAINE/MIDDLE EAST/IRAN/ISRAEL/SANCTIONS/MILITARY/GEOPOLITICAL RISK), rolling-252T-z (past-only, PIT). Fenster ~2015–2026. |
| Crisis-Basket | EW(XLE Energie, GLD Gold, ITA Defense) via EODHD. |
| Test A (Event-Study, kein Trial) | Mittlere Überrendite Crisis-Basket − SPY über 5/20/60T NACH Intensitäts-Spike (z>1). Kein Effekt/Wrong-Sign → These tot, Test B entfällt. |
| Test B (Trial, 2 Läufe) | monatlich: z>0,5 am Ultimo → Crisis-Basket Folgemonat, sonst SPY; 10 bps, volle Steuern. Variante (b): z>0 → 50 % Crisis/50 % SPY. |
| Pass/Fail (vorab) | Test A: Überrendite > 0 mit t>2 in ≥ 2 der 3 Horizonte. Test B: > SPY-B&H UND > statische 50/50-Crisis-Baseline netto. Nur wenn A UND B. |
| Kumulatives N nach Test | +2 (Ledger führt Reihenfolge) |
| Explorativ-Flag | JA — News-Intensität ist News-VOLUMEN-Proxy (nicht verifizierte Richtung); ~11 J, kein 2008; Basket-ETFs teils jung (ITA ab 2006, GLD 2004). |
| Ergebnis | **FAIL** (2026-07-09): Event-Study Überrendite nach Spike 5T +0,13 % (t=1,35), 20T/60T →0 — kein prädiktives Signal; Rotation < ETF-Pfad + statische Baseline. Bestätigt Fable R5 (GPR) auf zeitnäheren Daten: geopolitische These bei Tagesauflösung tot. N=111. Details: ledger.md Welle 14. |

---

## Welle 15 (registriert 2026-07-09, VOR Lauf; läuft LOKAL parallel zum Small-Cap-Pull) — H-040 Low-Vol/BAB, H-041 Quality-Tilt

Evidenzgetrieben: nach 39 Hypothesen ist das EINZIGE überlebende Muster
Low-Turnover-Steuerstundung (H-024 Band, H-032 Low-Div, H-037 Krypto-§23). Welle 15
prüft die zwei robustesten Low-Turnover-Anomalien der Literatur GEGEN genau diesen
Gewinner — auf dem survivorship-freien PIT-S&P-Verdict-Universum (lokal, kein Pull).

### H-040 — Betting-Against-Beta / Low-Volatility netto-nach-Steuer | 3 Läufe
Hypothese: Ein Long-Only-Low-Vol-Buch (unterstes realized-Vol-Terzil der PIT-S&P-
Mitglieder, EW-Band-50%-Mechanik → niedriger Turnover) schlägt nach 10 bps +
deutschen Steuern (a) das volle-S&P-EW-Band-Baseline auf Sharpe UND (b) den ETF-
Netto-Pfad. Begründung: Low-Vol/BAB (Frazzini-Pedersen 2014; Baker-Bradley-Wurgler
2011) = robusteste Faktor-Anomalie; stabile Low-Vol-Namen → wenig Turnover →
steuerfreundlich, in Linie mit dem einzigen Mandats-Muster, das überlebt.
Score = −realized_vol(63T), PIT. Vol-Terzil-Familie {bottom 20 %, 33 %, 50 %}.
Pass/Fail (ALLE): (1) Sharpe > full-S&P-EW-Band; (2) final > ETF-Pfad; (3) DSR passes
bei N; (4) MaxDD besser (weniger negativ) als EW-Band (Kernversprechen Low-Vol);
(5) ≥ 60 % 2-J-Fenster Sharpe ≥ EW-Band. Kumulatives N: +3 (Ledger führt Reihenfolge).
**Ergebnis: FAIL (1/5)** (2026-07-09): lowvol_33 final 649.619 / Sharpe 0,500 / MaxDD
−0,460 vs EW-Band-Baseline 1.311.212 / 0,545 / −0,585 und ETF-Pfad 1,59M. Nur MaxDD-
Kriterium ✓; Sharpe < Baseline, final ≪ ETF, DSR 0,572, Konsistenz 34,5 %. Low-Vol
liefert Risiko- (DD-Senkung), nicht Rendite-Versprechen; unhebelbar (Guardrail 4) kein
Netto-Alpha. Feld geschlossen. N=114. Details: ledger.md Welle 15.

### H-041 — Quality-Tilt auf dem Low-Turnover-Buy-and-Hold | 2 Läufe
Hypothese: Ein PIT-Quality-Screen (oberstes ROE-Terzil aus XBRL-Fundamentals) auf dem
EW-Band-Buch schlägt das UNGESCREENTE EW-Band-Buch netto — testet, ob ein Quality-Tilt
das eine funktionierende Muster (Low-Turnover-Stundung) verbessert, ohne den Turnover-
Vorteil zu fressen. Quality-Terzil {top 33 %, top 50 %}. EW-Band 0.5, 10 bps.
Pass/Fail (ALLE): Quality-EW-Band > ungescreent-EW-Band × 1,05 (Sharpe UND final) UND
> ETF-Pfad. Sonst FAIL (Quality-Prämie netto tot oder turnover-gefressen).
Kumulatives N: +2 (Ledger führt Reihenfolge).
**Ergebnis: FAIL** (2026-07-09): quality_33 final 568.052 / Sharpe 0,506 / MaxDD −0,369
vs ungescreent EW-Band 1.311.212 / 0,545. Weder Sharpe noch final > Baseline×1,05, nicht
> ETF-Pfad. ROE-Coverage Median 552/Monat. Quality senkt DD am stärksten (−0,37), aber
kein Netto-Rendite-Alpha; die ausgeschlossenen Namen waren die größten Nach-Steuer-
Gewinner. DSR 0,584. N=116. Details: ledger.md Welle 15.

**Explorativ-Flag:** H-040 Beta≈realized-Vol-Proxy (kein CAPM-Beta gegen Markt gerechnet;
Low-Vol ist die deployable, kostenrobustere BAB-Variante). H-041 ROE-Coverage/PIT-Lag
aus XBRL wird beziffert; Terzil-Sortierung tolerant gegen Lücken.

---

## Welle 16 (registriert 2026-07-09, VOR Lauf) — Intraday + All-World (Direktauftrag Hans: „alle Abo-Daten nutzen")

**Verifizierte Plan-Entitlements (2026-07-09, /api/user + Endpoint-Probes):**
All-World-Extended EOD+Intraday, monthly paid, 100k Calls/Tag (+500 extra). NUTZBAR:
EOD alle 70 Börsen (US+Delisted ✓, Europa XETRA/LSE/Euronext volle Historie ✓),
Intraday 1m/5m/1h **erst ab ~Okt 2020** (1h; 1m/5m nur gefenstert), Dividenden, Crypto,
FX (Guardrail-4 → Regime-only), News/Sentiment. **NICHT entitled (403/404, belegt):**
Options (legacy+unicornbay), Macro-Indikatoren, EODHD-Fundamentals US+global
(Fundamentals kommen ohnehin aus SEC-EDGAR-XBRL). **Dieser Auftrag OVERRIDED die frühere
Selbst-Notiz „Intraday bleibt ungenutzt = richtig".**

### H-042 — Overnight- vs Intraday-Return-Dekomposition + deployable Overnight-Buch | 2 Läufe
Hypothese: US-Equity-Prämie akkumuliert über Nacht (close→open), Intraday (open→close)
flach/negativ (Cooper/Cliff/Gulen 2008; Bogousslavsky). Test 1 (Dekomposition, kein
Trial): SPY + Sektor-ETFs — mittlere Overnight- vs Intraday-Rendite. Test 2 (Trial):
deployable Overnight-Buch (kaufe Close, verkaufe Open) nach 26,375 % (jeder Gewinn <1 J)
+ Kosten vs ETF-Netto-Pfad. Pass/Fail: Overnight-Buch netto > ETF-Pfad UND Sharpe > SPY.
**Prior (dokumentiert, hart):** Overnight-Prämie brutto real, aber 252 Round-trips/Jahr →
Steuer+Kosten fressen sie → erwartetes FAIL netto; Test quantifiziert den Keil ehrlich.
**Ergebnis: FAIL** (2026-07-09): Dekomposition brutto glasklar (SPY overnight +2,24 bps/Tag
vs intraday +0,91; Sektoren extremer, z. B. XLU 3,37 vs −1,56) — reproduziert Cooper/
Cliff/Gulen. ABER Overnight-Buch: 6 bps Round-Trip/Tag > 2,24-bps-Prämie → 100k → 4.997 €
netto vs ETF-Pfad 436.206 €, Sharpe −0,84. Kosten allein töten es. N=119. Details: ledger
Welle 16b.

### H-043 — Crisis-Alpha „erste Minuten" INTRADAY (Lückenschluss zu H-039) | 2 Läufe
Hypothese: Der in H-039 postulierte, bei Tagesauflösung tote geopolitische Move existiert
INTRADAY in den ersten Minuten. Test A (Event-Study, intraday): Crisis-Basket (XLE/GLD/
ITA vs SPY) Reaktion in ersten 5/15/30/60 min NACH geopolitischem Intensitäts-Spike
(H-039-Spikes, Overlap-Fenster ~2020–2026). Test B (Trial): capturable netto nach
Intraday-Kosten + 26,375 %? Pass/Fail: signifikanter Move (t>2) in ≥2 Horizonten UND net
capturable. **Prior:** schwach; wenn NULL → geopolitische These bei ALLEN Auflösungen tot
= definitive Schließung. Fenster nur ~2020–2026 (Intraday-Tiefe) → explorativ-Flag.
**Ergebnis: FAIL (netto), aber Signal brutto REAL** (2026-07-09): Crisis-Basket − SPY nach
Spike (z>1, 155T): 60min +12,45 bps (t=2,70), monoton, Baseline ~0; Drop-Top-K macht es
STÄRKER (top5 t=3,32) → kein Outlier-Artefakt → H-039-Fluchtklausel empirisch bestätigt.
ABER: 4-Bein-Kosten (long 3 ETF + short SPY) fressen es (netto @10bps 2,45 bps, @14bps
negativ); nach <1J-Steuer 1,81 bps/Tag; Spike-Strategie Ann-Sharpe 0,22, **DSR 0,021 ≪
0,95 → FAIL**; regime-abhängig (2020/2022 stark, 2021/2025 NULL). Geopolitik-Feld auf
ALLEN Auflösungen geprüft (GPR→EODHD-News→5m-intraday), konsistent nicht deployable.
N=118. Details: ledger.md Welle 16a.

### H-044 — All-World-EOD Global-Momentum (Europa, survivorship-aware) | queued, 2 Läufe
Hypothese: 12-1-Momentum auf survivorship-aware europäischem Developed-Universum (XETRA/
LSE/Euronext) schlägt nach Kosten + deutscher Steuer (inkl. ausländischer Quellensteuer-
Approximation) den ETF-Pfad. **Prior:** GEM-International (H-028) FAIL; ausländische
Dividenden-Quellensteuer + FX addieren Drag → Prior FAIL, aber genuin neues Universum.
Registrierung vollständig erst mit Universum-Pull; Pass/Fail analog H-035.

**Explorativ-Flags Welle 16:** Intraday-Fenster nur 1 Regime (~2020–2026); All-World-
Delisted-Coverage bei EODHD non-US schwächer als US (wird beziffert); Foreign-Tax
vereinfacht modelliert. Kumulatives N: +2 (H-042) +2 (H-043) +2 (H-044); Ledger führt Reihenfolge.

---

## Welle 17 (registriert 2026-07-09, VOR Lauf) — H-045: Halloween / Sell-in-May (Low-Turnover-Saisonalität)

Evidenzgetrieben: das EINZIGE überlebende Muster ist Low-Turnover. Halloween ist die
rare klassische Anomalie, die NATÜRLICH low-turnover ist (2 Realisationen/Jahr) — daher
der aussichtsreichste verbleibende Klassiker im Mandats-Sinn.
Hypothese: „In-Markt Nov–Apr, Cash Mai–Okt" (Bouman/Jacobsen 2002) schlägt nach Kosten +
deutscher Steuer (jährliche Mai-Realisation, Verlusttopf) den passiven ETF-Netto-Pfad.
Läufe (2): SPY; EW-SPDR-Sektoren. Pass/Fail: Halloween-Buch netto > ETF-Pfad UND Sharpe
> Buy&Hold. **Prior (dokumentiert):** verpasst Sommer-Equity-Prämie + zieht Steuer JÄHRLICH
vor (statt ETF-Endstundung) → doppelter Gegenwind → erwartet FAIL; quantifiziert den Keil.
Kumulatives N: +1 (Ledger führt Reihenfolge).
**Ergebnis: FAIL** (2026-07-09): SPY-Halloween netto 228.150 / EW-Sektoren 208.932 —
beide ≪ ETF-Pfad (436.206 / 373.678). Sharpe im investierten Fenster höher (0,55 vs 0,42),
aber Endvermögen bricht ein: halbes Jahr Cash opfert Compounding + Jahres-Realisation zieht
Steuer vor (SPY 83k). Schärft Kernbefund: überlebendes Muster = **voll investiert bleiben +
Realisation aufschieben**; Halloween verletzt beides. N=120. Details: ledger Welle 17.

### H-044 All-World-EOD — DATEN-GATED, NICHT gelaufen
All-World-EOD (Europa XETRA/LSE/Euronext) ist im Abo verfügbar, ABER ein rigoroser
survivorship-freier Selektionstest braucht europäische PIT-Index-Membership + Delisted-
Coverage, die ich NICHT habe. Ein Test auf Current-Constituents-only wäre survivorship-
verzerrt (§2.5-Falle, die Fable killte) — wird NICHT als Verdict verkauft. Bleibt gated
bis PIT-europäische Membership beschafft ist (Operator-Entscheidung, analog Norgate).

---

## Welle 18 (registriert 2026-07-09, VOR Lauf) — H-046: Covered-Call-Overlay („Aktien vermieten", Auftrag Hans)

Von Hans angefragt: „Aktien vermieten" = Stillhalter/Covered Calls — lohnt sich das netto?
**Drei harte Realitäten vorab dokumentiert:** (1) EODHD-Optionsdaten NICHT im Plan (403/404,
diese Session belegt) → nur Black-Scholes-approximierte Prämien möglich, KEIN verdict-taugliches
Real-Options-Backtest (§2.5 explorativ). (2) Guardrail 4 (Derivate) → Covered Calls sind das
voll-besicherte, hebelfreie Ende; Ausnahme = Operator-Policy-Entscheidung Hans, nicht autonom.
(3) Steuer: Stillhalterprämie sofort 26,375 % → zerstört den Stundungs-Vorteil (das einzige
Mandats-Muster); Termingeschäft-Verlustcap-Historie → Steuerberater-Frage.

Hypothese: Ein monatliches Covered-Call-Overlay auf SPY (Stock voll investiert/gestundet +
cash-settled Short-Call-Overlay je Monat) schlägt nach Kosten + SOFORT-Steuer auf die Prämie
den passiven ETF-Netto-Pfad auf ENDVERMÖGEN. Overlay-P&L/Monat = Prämie − max(S1−K,0), positiv
sofort 26,375 % besteuert. Grid: Strike {ATM, 3 %, 5 % OTM} × IV {realized, ×1,15, ×1,3}
(bracket der Vol-Risk-Prämie, da keine echte IV). Prämie = BS-Call.
Pass/Fail: Overlay-Netto-Beitrag > 0 (schlägt reines Buy&Hold) UND kombiniert > ETF-Pfad, in
≥ 1 vernünftigen Annahme. **Prior (hart):** gekappte Oberseite in Bullen-Sample + Sofort-Steuer
→ erwartet FAIL auf Endvermögen, evtl. besserer Drawdown (wie H-040/041); Test quantifiziert
den Keil. Kumulatives N: +1 (Ledger führt Reihenfolge; explorativ wegen Modell-Prämien).
**Ergebnis: CONDITIONAL — Prior WIDERLEGT** (2026-07-09): 5/9 Grid-Zellen schlagen ETF-Pfad;
OTM(3-5%)+IV≈1,2× (historisch typische Vol-Risk-Prämie) → 5%OTM ≈ 540k vs 436k (+24 %),
Sharpe 0,73 vs 0,49, MaxDD −0,31 vs −0,52. ATM verliert immer (−283k). ERSTES nicht-
stundungs-basiertes Ergebnis das schlägt (erntet Vol-Risk-Prämie, überlebt Sofort-Steuer WENN
OTM). ABER model-basiert (keine echten Optionsdaten, 403), IV-annahme-abhängig, Skew/Frictions
untertrieben, Guardrail-4 + Steuerberater offen → NICHT als PASS gebucht. Feld bleibt OFFEN,
gated auf echte Optionsdaten. N=121. Details: ledger Welle 18.

---

## Welle 19 (registriert 2026-07-10, VOR Lauf) — H-047: Net-Share-Issuance / Buyback-Anomalie

Evidenzgetrieben (bester verbleibender low-turnover-Kandidat auf lokalen Daten). Net-Issuance
(Pontiff-Woodgate 2008; Daniel-Titman 2006) ist eine der robustesten REPLIZIERBAREN Anomalien
(überlebt in Hou-Xue-Zhang q-Faktor-Replikationen, wo die meisten sterben) UND natürlich
low-turnover (Jahressignal) → passt exakt ins einzige überlebende Mandats-Muster.
Hypothese: Long-only oberstes Terzil der Netto-Rückkäufer (grösster YoY-Rückgang der verwässerten
Aktienzahl, PIT aus XBRL `WeightedAverageNumberOfDilutedSharesOutstanding`, via `available_at`)
mit EW-Band-50% schlägt netto (10 bps, dt. Steuer inkl. Div) (a) full-S&P-EW-Band-Baseline
(Sharpe UND final ×1,05) UND (b) den ETF-Netto-Pfad; DSR passes. Score = −(shares_FY_t /
shares_FY_{t−1} − 1) (Rückgang = Kauf), Terzil-Familie {top 33 %, top 50 %} (2 Läufe).
**Prior:** vorsichtig — jede bisherige Aktien-Anomalie fiel; ABER dies ist die low-turnover-
robusteste und passt strukturell. Kumulatives N: +2 (Ledger führt Reihenfolge).
**Explorativ-Flag:** XBRL-Coverage 743 Symbole (nicht alle jemals-S&P), FY-Share-Count-Historie
kann Lücken haben (wird beziffert); WeightedAvg-Diluted ist Standard-Net-Issuance-Proxy.
**Ergebnis: FAIL** (2026-07-10, N=128): Buyback_33 final 585.904 / Sharpe 0,499 / MaxDD −0,432
vs Baseline 1,34M/0,548, ETF 1,59M; DSR-p 0,555. Coverage 571/Mo. Nur DD besser. 4. Bestätigung
des strukturellen Meta-Musters (Size/Low-Vol/Quality/Buyback): Long-only-Screen senkt DD, verliert
Endvermögen (schließt Mega-Compounder aus; hier: Tech verwässert via SBC → fällt aus Rückkäufer-
Terzil); Net-Issuance-Alpha lebt im Short-Bein (Guardrail 4 gesperrt). Details: ledger Welle 19.

---

## Welle 20 (registriert 2026-07-10, VOR Lauf) — H-048: Direct-Indexing Tax-Loss-Harvesting vs ETF

Genuin neu (H-014 war TLH auf Momentum-Buch = kontraproduktiv; hier reines Direct-Indexing).
In der ÜBERLEBENDEN Kategorie (Steuer-Mechanik, KEIN Faktor-Screen — Konsequenz des 4-fach
bestätigten Meta-Musters, dass Long-only-Screens sättigen). Deutschland-spezifisch: KEINE
Wash-Sale-Regel → Verlust ernten + sofort zurückkaufen (Exposure unverändert), Verlust in den
Aktien-Verlusttopf.
Hypothese: Ein voll-investiertes Direct-Index-EW-Buch (breites Band, low turnover) mit aggressivem
TLH schlägt netto den thesaurierenden ETF-Netto-Pfad — die geernteten Verluste senken die
effektive Steuer auf die (End-)Realisation unter die 18,5 %-Teilfreistellung.
**FAIRNESS-Fix (wichtig):** End-Liquidation auf BEIDEN Seiten — die Strategie realisiert am Ende
zu 26,375 % minus Verlusttopf (nicht mark-to-market geschenkt wie in früheren Verdicts), ETF 18,5 %.
Läufe: TLH-Schwelle {aus, 15 %, 30 %} (3). Pass/Fail: TLH-Buch final_net(post-liq) > ETF-Pfad UND
> no-TLH-Buch (isoliert die Steuer-Alpha) + DSR. **Prior: FAIL** — dt. TLH-Verluste im Bullenmarkt
begrenzt, offsetten nur Aktiengewinne, 18,5 %-Teilfreistellung + volle ETF-Stundung sind hoher
Hurdle; ABER es ist DIE reale deutsche Direct-Index-Frage, genuin neu. Kumulatives N: +3.
**Ergebnis: FAIL** (2026-07-10, N=131): TLH-Alpha REAL aber klein — TLH-15 % 1.217.851 vs no-TLH
1.155.585 = **+62.267 (~+5 %)**; beide ≪ ETF-Pfad 2.335.072. Dt. TLH ≪ US-TLH (kein Step-up, nur
Aktiengewinn-Offset, kein Satz-Rabatt); Alpha < Satz-Nachteil (26,375 vs 18,5 %). Direct-Indexing
schlägt thesaurierenden ETF in DE NICHT. **Methodisch wichtig:** erstmals End-Liquidation → no-TLH
fällt von ~1,33M (mark-to-market) auf 1,156M → alle früheren FAILs waren strategie-freundlich
gerechnet, fallen ehrlich noch klarer. OFFEN: PASS-Designs (H-024/032) mit End-Steuer neu vermessen.
Details: ledger Welle 20.

---

## Welle 21 (registriert 2026-07-10, VOR Lauf) — H-049: Konzentriertes Mega-Cap-Momentum (Copy-Trading-Archetyp)

Aus Copy-Trading-Research destilliert (Auftrag Hans): der meistkopierte/langlebigste eToro-Trader
(Jeppe Kirk Bonde, >10 J, ~25 %/J) UND C2 „US Stock Momentum" konvergieren auf DENSELBEN Archetyp —
low-turnover, voll investiert, KEIN Hebel, konzentriert in Mega-Cap-Gewinnern, minimale Handels-
frequenz. = exakt das Mandats-Muster, nur konzentrierter.
Hypothese: Ein KONZENTRIERTES Top-N-Momentum-Buch (das die Mega-Gewinner noch stärker übergewichtet
als der Cap-Index) mit niedrigem Turnover schlägt netto den breiten ETF-Pfad. Testet die Korollarie
des Meta-Befunds: wenn Cap-Weighting die Gewinner fängt → schlägt MEHR Konzentration in die Gewinner?
PIT-Momentum-Selektion (kein Hindsight — hält NICHT automatisch dieselben Namen), weites top_out
(low turnover), KEIN Hebel, terminal_liquidation (ehrliche Endsteuer), Div-Steuer aktiv.
Läufe: top_in {10, 20} × top_out {40} (2). Pass/Fail: postliq > ETF-Pfad (window-matched, end-liq)
UND Sharpe > SPY UND DSR passes. **Prior:** die EINZIGE Richtung, die MITGEHT (übergewichtet Gewinner
statt sie zu screenen) → beste verbleibende Chance; ABER Konzentrations-Drawdown + PIT-Selektion
(Momentum-Rotation ≠ Buy-Hold-der-Gewinner) + 26,375 %-Endsteuer vs ETF-18,5 % sind die Killer.
**Explorativ:** Copy-Trading-Vorbild survivorship-selektiert (offen benannt); Fenster mega-cap-lastig.
Kumulatives N: +2.
**Ergebnis: FAIL** (2026-07-10, N=133): Top-10 postliq 658.885 / Sharpe 0,373 / MaxDD −0,722;
Top-20 postliq 1,10M — beide ≪ ETF 1,61M, Sharpe < SPY 0,605, DSR-p 0,284. Mehr Konzentration =
schlechter. Kernerkenntnis: PIT-Momentum-Rotation ≠ Gewinner-Halten (Jeppe = Hindsight); Cap-Index-
Edge = Gewinner halten OHNE Selektion/Rotation/Prognose. Copy-Trading-Track-Records survivorship-
selektiert, keine Regeln. Details: ledger Welle 21.

---

## Welle 22 (registriert 2026-07-11, VOR Lauf) — H-051: §23-Tax-Free-Asset-Sleeve

Evidenzgetrieben: nach ~50 Hypothesen schlägt KEINE steuerpflichtige Aktien-Strategie den ETF
nach dt. Steuer (kein Brutto-Alpha + 26,375/18,46 %-Nachteil). Die EINZIGE Stelle, wo die Nach-
Steuer-Mathematik für etwas anderes sprechen kann: §23-EStG-Assets (Krypto, physisches Gold/
Xetra-Gold), **> 1 Jahr gehalten = 0 % Steuer** — vs Aktien-ETF 18,46 %. Struktureller Tax-Wedge.
Hypothese: Ein Portfolio Aktien-ETF + disziplinierter Buy-and-Hold-Sleeve aus §23-Assets (nie < 1 J
verkaufen → steuerfrei) schlägt 100 % Aktien-ETF nach Steuer — auf Endvermögen UND/ODER risk-adjusted
(Sharpe/MaxDD). Assets: SPY + Gold (GLD, 2004+) + Krypto (BTC/ETH, 2016+). Allokations-Sweep
{5/10/20 %}, jährlicher Rebalance (Verkauf nur steuerfreier Sleeve-Teile > 1 J; Aktien-ETF-Steuer
am Ende 18,46 %; §23-Sleeve 0 % bei > 1 J). Pass/Fail: Blend-Netto-Endwert > 100 %-ETF ODER
Sharpe-net > ETF bei nicht schlechterem MaxDD. **Prior (ehrlich):** Gold-Sleeve DRÜCKT absolute
Rendite (Gold ~0-1 % real), aber 0 % Steuer + Diversifikation könnte risk-adjusted helfen; Krypto-
Sleeve = hoher Return ABER Hindsight-verzerrt (eine Ausnahme-Historie) + extremer Drawdown →
Krypto-Ergebnis EXPLIZIT als hindsight-abhängig kennzeichnen, Return-Haircut-Sensitivität rechnen.
Kumulatives N: +2 (Gold-Blend, Krypto-Blend). Guardrail 4: nur Spot, kein Hebel/Derivat — ok.
**Ergebnis: TEIL/FAIL auf Mandats-Ziel** (2026-07-11, N=135): SPY+Gold 2005–26 verbesserte Endwert
(798k vs 767k) + Sharpe (0,64→0,80) + MaxDD (−0,55→−0,35) → SAH nach PASS aus. ABER Gold-Haircut
×0,3 (Norm-Rückkehr) → Endwert fällt UNTER SPY (599k) → Absolut-Vorteil = Gold-Regime-Artefakt.
Robust bleibt nur Risk-adjusted (Diversifikation, kein Alpha/Steuer-Edge). Krypto-Blend (90/10 BTC
31,5 % CAGR) = Hindsight, selbst ×0,5 noch riesig; Forward-Rendite unwissbar. → auf Absolut-Rendite
nach Steuer FAIL; ehrlicher Nutzen = Risk-Management. Details: ledger Welle 22.

---

## Welle 23 (registriert 2026-07-11, VOR Lauf) — H-052: Global Tax-Aware Rebalanced Portfolio (Weltmarkt + §23)

Auftrag Hans: „nicht nur S&P — Weltmarkt bedienen" + „Tax-Free-Rebalancing". Kombiniert beides:
global diversifiziertes Portfolio (US SPY + Dev-ex-US EFA + EM EEM + §23-Gold GLD [+ Krypto BTC]),
jährlich rebalanciert; Aktien-ETF-Verkäufe 18,46 % Teilfreistellung, §23-Sleeve-Verkäufe (>1 J,
FIFO-alt) 0 %. Per-Sleeve-Basis-Tracking, End-Liquidation auf allen.
Hypothese: globale Diversifikation + STEUERFREIES §23-Rebalancing schlägt den 100%-US-ETF nach
Steuer — absolut UND/ODER risk-adjusted. Vergleich: 100%-SPY-BH, Global-Equity-BH, Global+Gold
rebalanciert, +Krypto. Pass/Fail: Blend-Netto > 100%-SPY-BH (absolut) ODER Sharpe > bei besserem
MaxDD. **Prior:** US schlug 2010–26 den Weltmarkt absolut (Diversifikation drückt Absolut-Rendite);
jährliches Rebalancing realisiert Aktien-Steuer (Drag) — aber §23-Gold-Rebalancing steuerfrei hilft.
Erwartung analog H-051: FAIL absolut, evtl. besser risk-adjusted; §23-Rebalancing-Premium ist der
genuin neue Test. Kumulatives N: +4.
**Ergebnis (2026-07-11, N=139):** (1) **Weltmarkt-Geo-Diversifikation SCHADETE** — Global-Equity
60/25/15 = 605k ≪ US-only 777k (US-Dominanz-Regime 2005–26); auf Rendite half „Weltmarkt" NICHT.
(2) **§23-Rebalancing-Prämie real aber KLEIN** — US+Gold rebal 85/15 schlug SPY auf allen 3 (824k/
0,73/−0,47), aber Gold×0,3 → Absolut weg (640k), Sharpe/MaxDD nur knapp besser. Robust = etwas
bessere Risk-adjusted, KEIN Absolut-Alpha. Krypto-rebal Sharpe 1,33 aber Hindsight. Details: ledger W23.

---

## Welle 24 (registriert 2026-07-11, VOR Lauf) — H-053: §4.6.1 Insider-Patrone auf BREITEM survivorship-freiem Universum

Die aufgesparte §4.6.1-Patrone wird JETZT verschossen — Voraussetzung (survivorship-freie Broad-
Daten) ist erfüllt: EODHD-Small-Cap-Pull (15.101 handelbare Namen inkl. delisted) + EDGAR-Form-4
(29 Tranchen inkl. Pleitiers BBBY/SIVB/LEHMQ/WAMUQ). H-031 lief NUR auf S&P (Insider-Info-Gehalt ist
aber in kleineren, unbeobachteten Firmen am größten → dort blind).
Hypothese: Opportunistische Insider-Käufe (Cohen-Malloy-Pomorski, PIT: Routine = gleicher Kalender-
monat in allen 3 Vorjahren, Rest opportunistisch), v. a. CLUSTER (≥2 Insider/Titel), auf dem breiten
survivorship-freien Universum, mit HANDELBARKEITS-FLOOR (Preis ≥ $5, ADV60 ≥ $1M — gegen H-036-
Illiquiditäts-Artefakt), EW-Basket 12M-Halten, 30 bps, schlagen netto (a) SPY-ETF-Pfad UND (b) einen
No-Signal-Tradable-Kontroll-Korb (sonst nur Small-Cap-Effekt). Varianten: all-opp / cluster≥2.
Pass/Fail (ALLE): > ETF-Pfad; Sharpe > No-Signal-Kontrolle; DSR passes; ≥60 % 2-J-Fenster; MaxDD ok.
**Prior:** H-031 (S&P) FAIL, Fable H1 (survivor) nicht von Baseline trennbar — ABER dies ist der EINE
faire Test auf survivorship-freien Broad-Daten mit Liquiditäts-Floor; Info-Gehalt am höchsten bei
Small/Mid. Div-Steuer für Small Caps weggelassen (minimal, pro-Strategie-Bias benannt). Kumulatives
N: +2. Guardrail: reine Analyse.
**Ergebnis: FAIL** (2026-07-11, N=141): all-opp final 1.459.702 / Sharpe 0,655 / MaxDD −55,4 %
schlägt ETF-Pfad 773k absolut (einziger Signal-Test der das tut), ABER DSR 0,669<0,95 ✗, Sharpe nur
~=SPY 0,642 (Risiko/Beta kein Alpha), MaxDD<SPY ✗, Fenster 3/6. Vorbehalte: nur 723 Symbole Form-4
(S&P-Historie, NICHT echtes Small-Cap-Universum → §4.6.1-These „kleine Firmen" ungetestet); Opp-Filter
filterte kaum (98,8 %); mark-to-market+keine-Div = pro-Strategie. Patrone verschossen, Insider-Feld zu.
Details: ledger W24.

## Welle 25 (registriert 2026-07-11, VOR Lauf) — Portfolio-Konstruktion & Risiko: H-054 Risk-Parity, H-055 Vol-Targeting, H-056 Monte-Carlo-Robustheit

Auftrag Hans („ohne Grenzen, Portfolio-Aufstellung, Monte Carlo, mit Risiko spielen"). Ebenen-Wechsel:
nicht mehr WAS kaufen (gesättigt), sondern WIE gewichten/risikosteuern. Assets: SPY/EFA/GLD/TLT (+BTC-
Variante). Steuern: Aktien-ETF 18,46 %, Gold/BTC §23 0 % (>1J), Bond-ETF 26,375 % (keine Teilfreist.);
per-Sleeve-Basis, End-Liquidation. Kein Hebel (Guardrail 4; Risk-Parity UNLEVERED = Weights summieren 1).

### H-054 — Inverse-Vol / Risk-Parity Multi-Asset | 3 Läufe
Monatliche inverse 60T-Vol-Gewichte (Risk-Budget-Gleichverteilung, unlevered) über {SPY,EFA,GLD,TLT},
Variante {+BTC}, Variante Band-Rebalance (nur bei >20 % Drift → weniger Steuer-Events). Pass/Fail:
netto > 100%-SPY (absolut) ODER Sharpe > SPY bei besserem MaxDD; DSR informativ. Prior: unlevered
RP ist bond-lastig → Absolut-FAIL erwartet, Risk-adjusted offen (Bond-Steuer-Nachteil drückt).

### H-055 — Vol-Targeting-Overlay | 2 Läufe
Portfolio 85/15 SPY/GLD; Exposure = min(1, Ziel-Vol/realized-20T-Vol), Ziel {10 %, 15 %}, Rest Cash
(2 %); De-Risking realisiert Steuer (ehrlich). Pass/Fail: wie H-054. Prior: Vol-Targeting verbessert
Sharpe/DD brutto (dokumentiert), aber Steuer-Drag der Realisationen + verpasste Rallys → absolut offen.

### H-056 — Monte-Carlo-Pfad-Robustheit (Bewertungs-Framework, 1 Lauf-Äquivalent)
Stationärer Block-Bootstrap (E[Block]=60T) der JOINT-Tagesrenditen 2005–2026, 1.000 Pfade × volle
Länge, deterministischer Seed. Je Aufstellung {100%SPY, 70/30 SPY/GLD, 60/40 SPY/TLT, RP, 85/15+VT}:
Verteilung Netto-Endwert (Median, 5%-Quantil), P(schlägt SPY), Median-MaxDD. Ziel: welche Aufstellung
gewinnt ROBUST über Pfade statt auf dem einen historischen Pfad (Sequence-Risk). Steuer vereinfacht
terminal je Sleeve (benannt). Kumulatives N: +6 gesamt (Ledger führt).
**Ergebnis (2026-07-11, N=147):** H-054 RP: Absolut FAIL (477–490k ≪ 791k; unlevered RP bond-lastig,
Hebel Guardrail-gesperrt) aber Sharpe 0,81–0,82 vs 0,65 / MaxDD −0,24 vs −0,55. H-055 VT: **steuerlich
kaputt in DE** (De-Risking = Steuer-Event; VT15 674k vs Ref 853k bei minimalem Sharpe-Gewinn). H-056
MC (1.000 Pfade): **70/30 SPY/Gold dominiert 100 % SPY verteilungsweit** — Median +15 %, 5 %-Quantil
+41 % (Sequence-Schutz), MaxDD −0,43→−0,33, P=54 %; 60/40 SPY/TLT steuerlich dominiert (6 %). Caveat:
Gold-Sample-Mean eingebacken → Median-Teil regime-abhängig, Floor/DD-Teil robust. Kein Absolut-Alpha
auch auf Portfolio-Ebene; robuster deployable Gewinn = §23-Gold-Sleeve (Floor/DD). Details: ledger W25.

## Welle 26 (registriert 2026-07-11, VOR Lauf) — Zeit-/Zustandsdimension: H-057 Krisen-Rebalancing, H-058 Glide-Path (MC), H-059 Sparplan (MC)

Fortsetzung Portfolio-Ebene (Auftrag „weiter autonom"). Neue Dimension: WANN/zustandsabhängig
allozieren statt statisch.

### H-057 — Krisen-Rebalancing mit §23-Gold als steuerfreiem Dry-Powder | 2 Läufe
70/30 SPY/Gold; bei SPY-Drawdown ≤ −20 % → 80/20 (Gold STEUERFREI verkaufen, Aktien billig kaufen),
≤ −30 % → 90/10; bei neuem SPY-Allzeithoch zurück auf 70/30 (SPY-Verkauf 18,46 %). Variante (b):
kein Revert (nur jährliches Band). Vergleich: statisches 70/30 (jährl. Rebal) + 100 % SPY. These:
der §23-Sleeve ist als KRISEN-Kaufkraft wertvoller als als statischer Ballast — steuerfreie Quelle
für Buy-the-Dip. Pass: > statisches 70/30 netto UND > 100 % SPY. Caveat: Gold-Lots könnten bei
schnellen Folge-Krisen < 1 J alt sein (§23 dann steuerpflichtig ~44 %) — wird geprüft/benannt.

### H-058 — Glide-Path im Monte Carlo | 1 Lauf-Äquivalent
MC (1.000 Pfade, Seed 42): linearer Glide 90/10 → 50/50 SPY/Gold über den Horizont (jährliche
Shifts, SPY-Verkäufe besteuert, Basis-Tracking) vs statisch 70/30 vs 100 % SPY. Frage: kauft der
Glide-Path (Sequence-Risk-Schutz am Ende) mehr Floor als er Median kostet? Terminal-Stats.

### H-059 — Sparplan-Realität (DCA) im Monte Carlo | 1 Lauf-Äquivalent
Hans' echter Fall (Beamter, laufendes Einkommen): 1.000 €/Monat über den Horizont (statt Lump-Sum).
MC: 100 % SPY vs 70/30 SPY/Gold — Median/5 %-Quantil/P(besser) des Netto-Endwerts (Basis = Einzahlungen
je Sleeve, Terminal-Steuer). Frage: ändert laufender Zufluss das Aufstellungs-Urteil? Kumulatives N: +4.
**Ergebnis (2026-07-11, N=151):** H-057 **FAIL des cleveren Timings**: statisch-70/30-jährlich 875k/
Sharpe 0,814 SCHLÄGT crisis-rebal (849k/769k) — Revert-Steuer-Events + §23-Kurzfrist-Falle (Gold-Lots
<1J → 44 %, 5 Treffer) + Whipsaw. H-058 Glide ≈ statisch (kein Mehrwert). H-059 Sparplan: 70/30-Urteil
HÄLT (Median 924k vs 855k, Floor +25 %, P=55,5 %). Lektion: Timing-Intelligenz kostet in DE nur Steuer;
stur + selten schlägt clever. Details: ledger W26.

## Welle 27 (registriert 2026-07-11, VOR Lauf) — Verfeinerung des Deployable-Ergebnisses: H-060 robuste Gold-Quote (Maximin), H-061 Rebalancing-Kadenz

### H-060 — Robuste §23-Sleeve-Größe via Szenario-Maximin | 1 Lauf-Äquivalent (Sweep)
Gold-Quote 0–50 % (5-pp-Schritte), Gold-Rendite-Szenarien ×{1,0/0,5/0,3/0,0 (flat)} auf dem
historischen Pfad 2005–26 (jährl. Rebal, volle Steuern, End-Liq). Robuste Quote = argmax des
MINIMUM-Endwerts über Szenarien (Maximin — nicht Rückspiegel-Optimum). Zusatz: Silber (SLV, §23)
als Split 2/3-Gold-1/3-Silber — bringt das zweite Edelmetall Diversifikation oder nur Vol?
### H-061 — Steueroptimale Rebalancing-Kadenz | 1 Lauf-Äquivalent (Sweep)
70/30 SPY/Gold; Kadenz {Buy&Hold (nie), jährlich, 2-jährlich, Band-20 % (monatlich geprüft)}.
Netto/Sharpe/DD/Steuer + §23-Kurzfrist-Treffer. Frage: wie selten ist optimal? Kumulatives N: +2.
**Ergebnis (2026-07-11, N=153):** H-060 **Maximin(Endvermögen)=0 % Gold** — Gold-Sleeve ist
VERSICHERUNG (Floor/DD), kein Wealth-Optimum; Prämie beziffert (10 % Quote → −17 % im Flat-Gold-
Worst-Case); Silber NEIN (Sharpe/DD schlechter). H-061: **biennial dominiert** (890k/0,818/120k
Steuer/0 ST) > band20 > never > annual (Steuer 200k + §23-ST-Falle). Deployable-Endfassung: ETF-Kern
+ optional 10–15 % Gold (Versicherungs-Entscheidung) + 2-Jahres-Rebalancing. Details: ledger W27.

## Welle 28 (registriert 2026-07-12, VOR Lauf) — H-062 Covered Calls mit ECHTER VIX-IV, H-063 EUR-Realität, H-064 Faktor-ETFs im Wrapper

### H-062 — Covered-Call-Overlay mit realer impliziter Vol (VIX) | 1 Lauf-Äquivalent (Grid)
Upgrade von H-046 (CONDITIONAL): Prämien nicht mehr angenommen, sondern BS mit **IV = VIX zum
Monatsstart** (VIX ≈ 30-T-ATM-IV = exakt der richtige Input für Monats-Calls; Daten VIX.INDX
2000–2026 im Abo verifiziert). Struktur wie H-046 (Stock gestundet + cash-settled Overlay, Prämie
sofort 26,375 %). Grid: Strike {ATM, 3 %, 5 % OTM} × **Skew-Haircut {0/10/20 %}** (OTM-Call-IV <
ATM-IV, VIX überschätzt OTM-Prämie — ehrlich diskontiert). Pass: Overlay-Beitrag > 0 UND kombiniert
> ETF-Pfad in der konservativen Skew-Zelle. Bleibt Modell (keine echten Preise), aber IV-Annahme
ist jetzt DATEN. Kumulatives N: +1.

### H-063 — EUR-Denominierungs-Realität | 1 Lauf-Äquivalent
Blinder Fleck: alle Tests in USD, Hans versteuert aber EUR-Gewinne (FX-Komponente ist steuerbar!).
SPY/GLD → EUR via EURUSD (2002+); Kernvergleiche (100 % SPY vs 70/30, ETF-Pfad, 2-J-Rebal) in EUR
neu. Frage: kippen Verdicts durch FX? Prior: nein (FX ~ Rauschen um Drift), aber ungetestet = unklar.

### H-064 — Faktor-ETFs IM Steuer-Wrapper | 2 Läufe (Familie)
Die Faktor-Frage endlich RICHTIG: als ETF behalten Faktoren 18,46 % + interne steuerfreie
Umschichtung + keine Eigen-Kosten (statt 26,375 % + Turnover wie meine Direkt-Tests). MTUM (Momentum),
USMV (MinVol), QUAL, VLUE, SCHD/NOBL (Dividend-Quality) vs SPY, gemeinsames Fenster ab Inception,
Buy&Hold, Terminal-Steuer beidseitig gleich. Pass je ETF: final > SPY UND Sharpe > SPY UND t(Excess)
> 2. Prior: US-Faktor-ETFs underperformen SPY seit 2013 (Mega-Cap-Regime) — aber im Wrapper ist es
der faire Test. Kumulatives N: +4 gesamt.
**Ergebnis (2026-07-12, N=157):** H-062: mit realer VIX-IV + Skew-Haircut schrumpft Covered-Call auf
**grenzwertig-null** (beste realistische Zelle +39k/21,5J, schlechteste −18k; ATM-Desaster bestätigt)
→ von CONDITIONAL herabgestuft. H-063: EUR-Sicht kippt nichts, VERSTÄRKT Gold-Sleeve (70/30-EUR:
DD −0,26 vs −0,53, Gold-EUR 962k steuerfrei). H-064: Faktor-ETF-Familie FAIL (nur SPMO formal t=2,14,
aber MTUM-Widerspruch + 1 Regime + DSR + 30-J-Kreuzbeleg: Momentum brutto < SPY → Watchlist, kein
Verdict). Details: ledger W28.

## Welle 29 (registriert 2026-07-12, VOR Lauf) — H-065 Entnahmephase (SWR, MC), H-066 Rolling-Start-Robustheit

### H-065 — Sichere Entnahmerate unter deutscher Steuer | 1 Lauf-Äquivalent
MC (1.000 Pfade, Seed 42, SPY/GLD-Bootstrap): Startvermögen 500k, jährliche Entnahme {3/3,5/4/5 %}
inflationsnaiv (nominal konstant), Verkäufe anteilig besteuert (Aktien 18,46 % auf Gewinnanteil via
Basis-Tracking, Gold 0 %). Aufstellungen: 100 % SPY vs 70/30. Metrik: Ruin-Wahrscheinlichkeit
(Depot vor Pfadende leer) + Median-Restvermögen. Frage: senkt der Gold-Sleeve das Ruin-Risiko
(Sequence-Risk in der Entnahme ist DER Killer)? Prior: ja deutlich — genau dort wirkt DD-Dämpfung.

### H-066 — Rolling-Start-Robustheit des 70/30-Urteils | 1 Lauf-Äquivalent
Historisch: jeder Monats-Start 2005–2016, Horizont 10 J: 70/30 (2-J-Rebal, Steuern, End-Liq) vs
100 % SPY je Fenster. Anteil Fenster mit 70/30 ≥ SPY (netto) + Sharpe-Vergleich. Frage: hängt das
Urteil am Startpunkt? Kumulatives N: +2.
**Ergebnis (2026-07-12, N=159):** H-065: 70/30 DOMINIERT Entnahmephase (Ruin 0,0 % bis 5 %-Entnahme
vs SPY 2,0 %; Median-Rest überall höher). H-066: aber nur **32 % der 10-J-Fenster** auf Endvermögen
(44/139) — Full-Window-Vorsprung hängt an Gold-Bull-Enden. Finale Charakterisierung: Gold-Sleeve =
Versicherung (kostet meist Endvermögen, dämpft immer DD, eliminiert Ruin in Entnahme). Details: ledger W29.

## Welle 30 (registriert 2026-07-12, VOR Lauf) — H-067 BTC-Sizing (Kelly unter Unsicherheit), H-068 Krisen-Replay EUR

**SPMO-Watchlist formal GESCHLOSSEN (kein neuer Lauf):** 30-J-Kreuzbeleg h051 (Momentum-Top-20
BRUTTO CAGR 10,90 % < SPY 11,21 %) + MTUM-Fail (t=1,41, längeres Fenster desselben Faktors) →
SPMO-t=2,14 ist Regime-/Implementierungs-Pick, kein Alpha. Zu.

### H-067 — BTC-Beimischungsgröße via Kelly unter Haircut-Unsicherheit | 1 Lauf-Äquivalent
Die §23-Wette sauber dimensionieren statt raten: Kelly f*=μ/σ² aus BTC-Tagesdaten 2016–26, aber
über Forward-Rendite-Szenarien {×1,0/×0,5/×0,25/×0,1/0} (Hindsight-Entwertung). Robuste Größe =
Fractional Kelly (½) des PESSIMISTISCHEN Szenarios. Zusatz-MC: 70/30-Basis + BTC ε∈{0/2/5/10 %} ×
Szenarien → Floor/Median. Liefert Empfehlung mit ehrlicher Unsicherheits-Kennzeichnung.

### H-068 — Krisen-Replay in EUR | 1 Lauf-Äquivalent
GFC (2007-10→2009-03), COVID (2020-02→03), 2022 (01→10) auf {100 % SPY, 85/15, 70/30} in EUR:
Peak-to-Trough + Erholungsdauer. Konkrete Stress-Zahlen fürs Endportfolio. Kumulatives N: +2.
**Ergebnis (2026-07-12, N=161):** H-067: robuste BTC-Größe **≤5 %** (½-Kelly des ×0,1-Szenarios 5,7 %;
im Tot-Szenario −3…−8 % Median-Kosten, sonst Floor+Median-Beitrag). H-068 EUR-Replay: GFC −50 % (SPY)
vs **−27 % (70/30)**, Erholung 833 T vs >3 J; COVID −34/−25 %, 2022 −17,5/−9,3 %. Details: ledger W30.

## Welle 31 (registriert 2026-07-12, VOR Lauf) — H-069: Cash-Flow-Rebalancing (Null-Steuer-Rebalancing im Sparplan)

Genuin neue deutsche Optimierung: Sparraten IMMER in den untergewichteten Sleeve lenken →
Rebalancing OHNE jeden Verkauf = ohne jedes Steuer-Event. MC (1.000 Pfade, Seed 42, SPY/GLD,
1.000 €/Monat, 21,5 J), Vergleich: (a) fixe Split-Raten 70/30 ohne Rebal, (b) **Cash-Flow-Rebal**
(Rate → untergewichteter Sleeve), (c) fixe Raten + 2-J-Verkaufs-Rebal (Steuer). Metriken: Median/
Floor/Median-MaxDD + gezahlte Steuer. Pass: (b) ≥ (c) auf Endwert bei gleicher Risiko-Kontrolle
(Gewichtsdrift beziffert). Kumulatives N: +1.
**Ergebnis: PASS** (2026-07-12, N=162): Cash-Flow-Rebal 926k/Floor 500k/DD −0,231/Steuer 0 —
dominiert Verkaufs-Rebal (918k/Steuer 8k) strikt bei gleicher Risiko-Kontrolle. Ansparphase:
NIE verkaufen, Rate→Untergewicht; Verkaufs-Rebal erst ohne Zuflüsse. Details: ledger W31.

## Welle 32 (registriert 2026-07-12, VOR Lauf) — H-070: Integriertes Endportfolio (Synthese)

Synthese aller Befunde in EINEM Vergleich: (a) 100 % SPY, (b) 70/30 SPY/Gold, (c) **70/25/5
SPY/Gold/BTC** (BTC-Größe aus H-067). MC 1.000 Pfade (SPY/GLD/BTC joint, 2016+ wegen BTC, Fenster
benannt), BTC-Szenarien {×1,0 / ×0,25 (Basis-ehrlich) / ×0,0 (tot)}; Lump-Sum; Terminal-Steuer
(SPY 18,46 %, Gold/BTC §23 0 %). Metriken: Median/Floor/Median-MaxDD. Pass: (c) ≥ (b) auf Floor UND
Median im ×0,25-Basisszenario, ohne im Tot-Szenario materiell zu verlieren. Kumulatives N: +1.
**Ergebnis: PASS (knapp)** (2026-07-12, N=163): ×0,25-Basis: 70/25/5 = 407k/Floor 224k/DD −0,24 ≥
70/30 (397k/221k/−0,24); Tot-Szenario nur −4 % Median; Hindsight ×1,0 zeigt Potenzial (1,04M) UND
Preis (DD −0,46 → nie >5 % BTC). Endportfolio quantitativ fixiert. Details: ledger W32.

## Welle 33 (registriert 2026-07-12, VOR Lauf) — H-071: TECHNISCHES INDIKATOR-LABOR (Groß-Sweep, Auftrag Hans „alles, jede Variation, Indikator-Zusammenspiel")

Systematische Batterie statt Einzeltests. **Familien:** SMA-Cross (4 Param), EMA-Cross (3), MACD,
Donchian-Breakout (2), RSI (Mean-Rev 2 + Trend 1), Bollinger (Mean-Rev + Breakout), TS-Momentum
(6M/12M), Vol-Filter — PLUS **Kombinationen** (AND/OR/2-von-3-Ensembles: SMA200×Mom, MACD×RSI,
Donchian×VolFilter u. a.). **Assets:** SPY (ETF → 18,46 % Teilfreistellung auch beim Timing —
korrekt!), Gold (Xetra-Gold-Proxy: §23 >1J 0 %/<1J 44 %), BTC (§23). **Logik:** Long/Cash,
next-close-Execution, 5 bps/Seite, Round-Trip-FIFO-Steuer mit Verlusttopf, End-Liquidation.
**Metriken:** Netto-Endwert, CAGR, Sharpe, MaxDD, Trades, Steuer, OOS-Sharpe (2. Fensterhälfte).
**Pass-Kriterien (vorab, ALLE):** (1) netto > Asset-B&H-Netto-Pfad; (2) OOS-Sharpe > B&H-OOS-Sharpe;
(3) DSR passes beim NEUEN kumulativen N (~85 Configs → N≈248 — die Latte steigt ehrlich mit);
(4) Familien-PBO < 0,5. **Prior:** dokumentiert FAIL (Timing = Steuer/Whipsaw; Fable/W1-2-Echos),
aber Kombinationsraum + §23-Assets + korrekte ETF-Teilfreistellung sind teils neu. Kumulatives N:
+~85 (exakte Zahl im Ledger).

## Welle 34 (registriert 2026-07-12, VOR Lauf) — H-072: INTRADAY-Indikator-Batterie (5m SPY)

Auftrag Hans („Intraday, ob das Sinn macht, warum, wie"). SPY 5m (2020-10–2026-07, Abo-Tiefe):
SMA-Cross auf Bars, Opening-Range-Breakout (30/60 min), Intraday-RSI-Mean-Rev, Overnight-Gap-Fade/
Follow, Prev-Day-Momentum — alles day-only (kein Overnight), 4 bps/Seite, Steuer 18,46 % (ETF) auf
Jahres-Netting. **Ziel: quantifizieren WARUM Intraday (nicht) funktioniert** (Kosten-Tod vs Signal).
Pass wie W33. Prior: hart FAIL (Kosten × Frequenz). Kumulatives N: +~10.
**Ergebnis W33–35 (2026-07-12, N=1195):** H-071: SPY 0/25, GLD 0/25 (Timing zerstört §23-Frei-
stellung!), BTC 6/25 slow-trend (DSR 0,917 ✗). H-072 Intraday: **alle 7 BRUTTO negativ** (−1,7…−10,1
bps/T) — Signale existieren nicht, vor Steuern tot. H-073 Welt (950 Configs/38 Assets): **11/950 =
1,2 % < Zufallsniveau**; Regionen 0/150, EU-Aktien 4/500 verstreut, bester DSR 0,214; **ETH repliziert
BTC-Trend NICHT (0/25)** → BTC-Fußnote asset-idiosynkratisch, geschlossen. Technische Analyse über
1.032 Configs/41 Assets/2 Zeitebenen: TOT. Details: ledger W33–35.

## Welle 36 (registriert 2026-07-12, VOR Lauf) — H-074: Regime-/Makro-Konditionierung des ENDPORTFOLIOS

Letzte offene Signal-Frage: nicht Asset-Timing, sondern SLEEVE-Gewichts-Konditionierung — VIX-Regime
(reale Daten 2000+) als Schalter der GOLD-Quote {niedrig 15 %/hoch 35 % wenn VIX>P80} vs statisch
25 %. Verkäufe nur im 2-J-Raster (Steuer-schonend), §23-Uhr respektiert. Pass: > statisch auf Netto
UND Floor (MC). Prior: W26 sagt Timing verliert — dies ist der mildeste denkbare Timing-Fall (selten,
klein, steuer-arm). Kumulatives N: +2.
**Ergebnis: FAIL** (2026-07-12, N=1197): regime 826k < statisch 874k (Sharpe/DD/Steuer alle
schlechter), robust unter Gold×0,5. Timing auf JEDER Ebene geschlossen (Asset W33-35 / Zustand W26 /
Sleeve-Gewicht W36). Details: ledger W36.

## Welle 37 (registriert 2026-07-12, VOR Lauf) — H-075 Kalender-Anomalien, H-076 Sektor-Rotation im Wrapper

### H-075 — Turn-of-Month / Day-of-Week (letzte ungetestete Kalender-Klassiker) | Familie ~6
SPY 1993–2026: (a) TOM: long nur letzte 4 + erste 3 Handelstage des Monats, sonst Cash (Ariel 1987;
McConnell/Xu 2008); Varianten {4+3, 2+2}. (b) DoW: long nur Di–Fr (Montag-Effekt); long nur Mi.
ETF-Steuer 18,46 % je Round-Trip + Verlusttopf, 5 bps/Seite. Diagnose zusätzlich BRUTTO (zeigt, ob
Anomalie existiert, bevor Kosten/Steuer sie fressen). Pass wie W33 (inkl. DSR bei N≈1200).
### H-076 — Sektor-Relative-Strength-Rotation long-only im ETF-Wrapper | Familie ~4
9 SPDR-Sektoren, monatlich Top-{1,3} nach 12-1-Momentum, Halten bis Rangverlust {sofort, Puffer 5};
ETF-Steuer 18,46 % je Wechsel. Vergleich vs SPY-B&H. (Fable sector_rotation war daily-store-basiert
und REJECTED; dies ist die wrapper-korrekte Monats-Variante.) Prior: FAIL. Kumulatives N: +10.

## Welle 39 (registriert 2026-07-12, VOR Lauf) — H-077: MEGA-STRATEGIE-SUCHE (Direktauftrag Hans, alle Stränge)

**⚠️ GUARDRAIL-4-RESEARCH-OVERRIDE (Hans, 2026-07-12):** Hebel, Short, Long/Short, FX, Optionen
werden auf expliziten Auftrag GETESTET (Backtest/Modell) — KEIN Live-/Paper-Einsatz; Deployment
bliebe separate Operator-Entscheidung. Optionen weiterhin modellbasiert (VIX-IV + Skew, keine
echten Preise); Short mit Borrow-Kosten 3 %/J modelliert; Hebel mit Finanzierung 4 %/J.

**Protokoll (vorab fixiert):** Zwei Stufen. Stufe 1 = vektorisierter Monats-Screen je Strang
(EW-Basket-Forward-Returns auf survivorship-freiem Verdict-Panel bzw. Asset-Serien; Steuer-
Approximation: Jahres-Netting 26,375 % Aktien / 18,46 % ETF / 44 % Kurzfrist-§23 — als SCREEN
gekennzeichnet, kein Verdict). Stufe 2 = Überlebende (Screen-Kriterien: > SPY-B&H-Netto UND
OOS-Hälfte positiv) laufen durch die volle Trade-Engine. **Stränge (je ≥75 Configs wo Daten
tragen):** Insider (Form-4-Grid), Congress (Grid), Whale/13F (Grid), Technik (W33-35: 1.032 —
Strang erfüllt), Hebel-Grid, News-Sentiment-Grid, Geopolitik-Grid (inkl. Social-Media-Proxy),
Intraday-Grid, FX-Majors-Grid, Optionen-Grid (VIX-Modell), Short-/LS-Modi über Technik-Signale.
**Portfolio-Labor (≥50 Konstruktionen):** Gewichts-/Risikoklassen-Grid über 8 Assets, bewertet via
ECHTEM Monte Carlo (1.000 Bootstrap-Pfade, Seed 42): Median/Floor/DD/Ruin. **N-Buchung:** jede
Screen-Config zählt; N steigt auf ~2.000+ — DSR-Latte entsprechend (ehrlich ausgewiesen).
Internet-Mining (wikifolio/eToro-Archetypen) → Mapping auf Stränge, dokumentiert.
**Ergebnis Hauptlauf (2026-07-12, N=1862):** 609 Configs: Insider 162/0, Congress 108/0, News 36/0,
Geo 96/0, Short/LS 24/0, FX 48/4 (trivial ~0,6 %/J), Hebel 45/15 (alle BTC-Hindsight), **Optionen
90/32 (einziger substanzieller Cluster — Vol-Risk-Prämie, modellabhängig, daten-gated)**. H-078
Portfolio-Labor (48×MC): Top-Floor ehrlich = 65/25/5/5 SPY/Gold/BTC/ETH → Endspez. bestätigt.
Whale/13F läuft nach (CUSIP-Map). Internet-Mining: wikifolio/eToro-Archetypen (Momentum/Dividende/
Quality/AI-Themen/konz. Stockpicking) sind vollständig durch Stränge Technik/W28-Faktor/W21-
Konzentration abgedeckt — keine neue Strategieklasse gefunden (frühere wikifolio-Tiefenprüfung:
Wrapper strukturell dominiert). Details: ledger W39.
**Nachträge (2026-07-12, N=1934):** Whale/13F 60/0 (alle Event-Stränge tot). H-079 Stufe-2 Options:
**0/12 unter adversarialen Annahmen** → Stillhalter-Feld ANNAHME-GEBUNDEN (Edge liegt komplett im
Skew/Bid-Ask-Unsicherheitsband), Status: unentscheidbar-modellbasiert, daten-gated. Ledger W39c.

## Welle 40 (registriert 2026-07-12, VOR Lauf) — H-080: Rest-Dimensionen der Suche

(a) **Insider-VERKÄUFE als Negativ-Filter** (S-Code-Cluster → Titel MEIDEN im EW-Basket; 12 Configs)
— getestet als Avoidance statt Selektion (neu). (b) **Event×Technik-Kombination** (Insider-Kauf UND
über SMA200; Congress UND Momentum>0; 12 Configs) — Konfirmations-Logik. (c) **EU-Querschnitts-
Momentum** (20 EU-Blue-Chips, Top-5 nach 12-1, monatlich, 26,375 %; 6 Configs) — Basket statt
Einzeltiming. Screen-Protokoll wie W39. Prior: FAIL überall; schließt die letzten Grid-Lücken.
Kumulatives N: +~30.
**Ergebnis (2026-07-12, N=1964):** Sell-Avoidance 12/0 (Filter fügt nichts hinzu); Event×Technik 12/0
(beste 1,28M ≪ TR-SPY-Bench 2,09M); EU-Momentum 3/6 nur vs EU-EW-Bench, survivorship-verzerrt (20
heutige Blue-Chips) → exploratorisch, kein Verdict. **E-052-Artefakt-Fang:** month_panel ohne Hygiene
+ pct_change-Pad über Delisting-Lücken → 10³⁰-Fake; 2-Schicht-Fix im Framework, W39-Verdicts unberührt.
Details: ledger W40.

## Welle 41 (registriert 2026-07-12, VOR Lauf) — H-081: Stillhalter-Verdict via ECHTER CBOE-Index-Historie (Vorschlag Hans)

**Das löst die H-079-Unentscheidbarkeit:** BXM (ATM-BuyWrite), BXMD (30-Delta-OTM-BuyWrite), PUT
(Cash-Secured-Put) sind von CBOE aus ECHTEN gehandelten SPX-Optionspreisen berechnet (realer Skew,
reale Bid-Settlements, monatliche Rolle, Historie bis 1986) — kein Modell mehr.
**Plan:** (1) Daten via EODHD (BXM.INDX etc.) sonst CBOE-CSV. (2) Brutto-Vergleich vs S&P-TR über
volle Historie + Dekaden-/Regime-Tabelle (Form & Regimeabhängigkeit „mit eigenen Augen"). (3)
Deutsche-Steuer-Overlay: Monats-Dekomposition Options-P&L ≈ (Index-Ret − SPX-Ret); positive Options-
Monats-P&L sofort 26,375 % (Stillhalter §20), Aktien-Bein 18,46 % terminal (Approximation, benannt);
+ 5 bps/Mo Implementierungskosten. **Pass/Fail (vorab):** Netto-Endwert > SPY-ETF-Pfad UND Sharpe >
SPX-TR über das gemeinsame Fenster; Regime-Tabelle entscheidet Charakter (Bull-Cap vs Vol-Ernte).
Prior aus H-062/079: ATM (BXM) FAIL erwartet; **BXMD/PUT = die eigentliche Frage.** Kumulatives N: +3.
**Ergebnis: FAIL — Feld GESCHLOSSEN** (2026-07-12, N=1967): BXMD (38,4 J, echte Preise) trailt SPXTR
brutto 10,90 vs 11,52 %/J (Sharpe/DD besser = Versicherung); gewinnt NUR in der Lost Decade, verliert
Bullen 4–6 pp/J; PUT/BXM klar schlechter. DE-Steuer-Overlay: alle drei ≪ ETF (BXMD 3,56M vs 5,15M).
H-079-Band real am adversarialen Ende aufgelöst. Gold-Sleeve dominiert BuyWrite als Risiko-Werkzeug
(0 % vs 26,375 % asymmetrisch). Keine offene Alpha-Tür mehr. Details: ledger W41.

## Welle 42 (registriert 2026-07-12, VOR Lauf) — H-082: Versicherungs-Duell — Protective Put (echte CBOE-Historie) vs §23-Gold-Sleeve

Letzte offene Portfolio-Frage: die effizienteste ABSICHERUNG für den deutschen Anleger. Kandidaten:
(a) PPUT (CBOE, 5 %-OTM-Protective-Put, echte Preise 1986–2026), (b) CLL (Collar, 2008+), (c) CNDR
(Condor, Info), (d) Incumbent: 70/30-§23-Gold-Sleeve (W26/27-Zahlen). Test: (1) Brutto-Kosten der
Put-Versicherung über 38 J (CAGR-Drag vs SPXTR, DD-Schutz, Dekaden). (2) DE-Steuer-Overlay
(Monats-Dekomposition wie H-081; Termingeschäft-Behandlung vereinfacht symmetrisch mit Topf,
benannt). (3) Vergleichstabelle Netto/DD/Sharpe vs Gold-Sleeve (gleiche Fensterlogik 2005+).
Pass/Fail: Put-Versicherung schlägt Gold-Sleeve auf Netto-Endwert BEI gleichem oder besserem DD.
Prior: FAIL (Puts kosten laufend echte Prämie, Gold 0 % Steuer + positiver Erwartungswert). N: +2.
**Ergebnis: Gold-Sleeve DOMINIERT strikt** (2026-07-12, N=1969): PPUT −3,5 pp/J Versicherungskosten
über 38,4 J, Sharpe schlechter, half in der Crash-Dekade NICHT (2000er −1,39 vs SPX −0,95!); DE-Netto
545k vs Gold-Sleeve 890k (> ungesichert 767k) bei besserem DD. Collar/Condor netto tot. Versicherungs-
Frage geschlossen. Details: ledger W42.

## Welle 43 (registriert 2026-07-12) — H-083: Einheitliche OOS-Re-Evaluation ALLER Strategien (Auftrag Hans; KEINE neuen Trials)

**Ehrlichkeits-Rahmen:** Echtes OOS existiert rückwirkend nicht (Daten lagen beim Design vor).
Geliefert wird: (A) **Ernte aller gespeicherten OOS-Metriken** (2.-Hälfte-OOS-Sharpe je Config aus
allen results/*.json — W33/35/39/40-Familien, hunderte Configs) familienweise aggregiert.
(B) **Einheitliches Rezenz-Holdout 2021-07→2026-07** (2022-Bär + Bull + 2025-Vol) für ~20 kanonische
Verdict-Strategien: gleiche Metrik (CAGR/Sharpe brutto inkl. Kosten), gleicher SPY-Vergleich im
selben Fenster. PIT-sauber wo nötig (z. B. IV-Gewichte nur aus Prä-Holdout-Daten). Frage: zeigt
IRGENDEINE Strategie Edge im jüngsten Regime-Mix? Re-Evaluation → N unverändert (1969).
**Ergebnis (2026-07-12):** A: 1.112 Configs geerntet, OOS-Mediane 0,24–0,86 alle < SPY-OOS 0,78,
kein Signal-Survivor. B: **4/19 schlagen SPY im Holdout — ausschließlich gold-haltige Aufstellungen
(70/30: Sharpe 1,02, bester DD) + der bekannte BTC-Einzelfall; 0/15 Signal-Strategien.** Endspez.
= beste real gelaufene Aufstellung des Holdouts. Details: ledger W43 + h083_unified_oos.json.

## Welle 44 (registriert 2026-07-13, VOR Lauf) — H-084: Odd-Lot-Tender-Offers (kapazitätsbeschränktes Retail-Alpha)

Erste Welle der NEUEN Alpha-Landkarte („wo Fonds nicht hinkönnen"): Self-Tender (SC TO-I) mit
Odd-Lot-Klausel — Positionen ≤99 Aktien werden ohne Proration angenommen; per Definition nur für
Kleinanleger skalierbar. EDGAR-Volltextsuche verifiziert: 1.861 Filings 2015–2026.
**Stufe 1 (Event-Study, Proxy):** Filings harvesten (Ticker+Datum), Preis-Reaktion Filing→+30 BD
vs SPY (EODHD). Misst das Tender-Fenster, noch NICHT den exakten Odd-Lot-Capture (der braucht
geparste Tender-Preise = Stufe 2, nur falls Stufe 1 positiv). Pass Stufe 1: mittlere Excess-Rendite
> 0 mit t > 2 UND Hit-Rate > 60 %. Ehrlich: Proxy; Illiquidität/Spreads der Small-Caps als Risiko
benannt. Kumulatives N: +1. Parallel: Memo Abfindungswerte/Spruchverfahren (deutsches Pendant).
**Ergebnis (2026-07-13, N=1971):** Stufe 1 (Drift-Proxy) FAIL (t=1,22/−0,17/−0,24 — Markt preist
Tender effizient). Stufe 2 (geparste Tender-Preise, 124 Captures): **Mechanismus bestätigt (60,5 %
positiv, Median +3,7 %, ~6,5 Events/J), Magnitude parse-verrauscht** (Mean +65 % = Artefakte).
**Skalierungs-Wahrheit: ≤99 Aktien ⇒ ~200–600 €/J Obergrenze — real, aber Taschengeld.** Nutzen =
Forward-Scanner + manuelle Fall-Prüfung. MEMO_ABFINDUNGSWERTE.md erstellt (deutsches Pendant,
pro Fall größer: Downside ~0, Nachbesserungs-Option + 5Pp-Zins, Satelliten-Sleeve-tauglich).
Details: ledger W44.

## Welle 45 (registriert 2026-07-14) — H-085: Abfindungswerte-Watchlist (operativer Build, KEIN Trial)

Umsetzung des MEMO_ABFINDUNGSWERTE-Prozesses: (1) Inventar aktuell OFFENER Fälle (angekündigte/
beschlossene Squeeze-outs, BGAV mit Abfindung, laufende Delisting-Angebote) via Web-Recherche
(Spruchverfahren-Szene-Quellen, Bundesanzeiger-Meldungen); (2) je Fall: Abfindungshöhe vs aktueller
Kurs (EODHD XETRA) → Spread/Carry-Kandidaten; (3) wiederverwendbarer Scanner-Prozess dokumentiert.
Operativ, keine Backtest-Trials → N unverändert. Output: ABFINDUNG_WATCHLIST.md.

## Bewusst NICHT registriert (2026-07-05): H-014-alt / §4.6.1 Insider-6-12-Monats-Retest

Die EINZIGE zugelassene Insider-Neubewertung (§4.6.1, Cohen/Malloy/Pomorski) wird
BEWUSST AUFGESPART: Fable H1 hat gezeigt, dass Insider-Signale auf Survivor-Daten
nicht von der +0,35-Sharpe-Baseline trennbar sind — die eine erlaubte Patrone auf
diesen Daten zu verschießen, wäre Verschwendung des Budgets. Registrierung erfolgt
erst, wenn survivorship-freie Daten (Norgate/Sharadar — Operator-Entscheidung Hans)
verfügbar sind.

---

---

## Welle 46 (registriert 2026-08-05, VOR jedem Lauf) — Liquiditaets-Gate fuer den §4.6.1-Nachtest

**Dies ist eine GATE-Registrierung, kein Hypothesen-Schuss.** Sie legt fest, unter welchen
Bedingungen ein Nachtest der §4.6.1-These ueberhaupt gerechnet werden darf. Der Lauf selbst ist
eine separate Operator-Entscheidung (siehe „Offene Frage" unten). N unveraendert.

### Warum die Frage ueberhaupt wieder aufkommt

H-053 (Welle 24) hat die Patrone verschossen und FAIL geliefert. Der Vorbehalt im eigenen
Registry-Eintrag entwertet dieses Verdikt aber fuer genau die These, um die es ging:

> „nur 723 Symbole Form-4 (S&P-Historie, NICHT echtes Small-Cap-Universum → §4.6.1-These
> ‚kleine Firmen' ungetestet)"

Die Preisseite lag breit vor, die Signalseite nicht. Getestet wurde nochmal S&P. Seit dem
DERA-Pull (2026-08-05) liegt die Signalseite breit vor: 17.134 Emittenten, 2006–2026,
universumsunabhaengig und damit survivorship-frei fuer `as_of >= 2006-01-01`.

### Der Fehlschluss, gegen den das Gate schuetzt

H-035/H-036 sind daran gescheitert, dass „Size" ein **Illiquiditaets-Artefakt** war. Gemessen am
neuen Universum (`liquiditaet_smallcap.json`, 7.804 Namen mit plausiblem Kaufsignal und Kurs):

| Gruppe | n | Median-ADV | < 1 Mio $ | < 200k $ |
|---|---:|---:|---:|---:|
| nie im S&P-Panel (neu) | 7.182 | **2,27 Mio $** | 37 % | 16 % |
| jemals im S&P-Panel (alt) | 622 | 56,06 Mio $ | 4 % | 2 % |

Faktor 25 im Median. Ohne Gate wiederholt ein Nachtest den bekannten Fehlschluss garantiert.

### Das Gate — VOR dem Lauf festgelegt

**Primaer (entscheidend):** Kurs ≥ **5 USD** UND rollierendes **ADV60 ≥ 1 Mio USD**, geprueft
**je `as_of`**, nicht einmalig ueber die Gesamthistorie. Die Schwelle ist NICHT neu gewaehlt: sie
steht woertlich schon in den Pass/Fail-Kriterien von H-053 (Welle 24). Damit ist sie nicht an das
neue Ergebnis anpassbar.

**Sekundaer:** ADV60 ≥ 5 Mio USD. Es kann nur **verschaerfen, nie lockern** — ein Ergebnis, das
nur unter dem Primaergate haelt, gilt als FAIL. Zweck: sichtbar machen, ob ein Befund von den
duennsten Ueberlebenden des Primaergates getragen wird. Die Reihenfolge steht hier fest, damit
hinterher nicht die passendere Schwelle zur Hauptaussage erklaert wird. (Die erste Fassung nannte
es "NICHT entscheidend" — das widersprach Pass-Kriterium 5 und war nachtraeglich in beide
Richtungen lesbar.)

**PIT-Pflicht:** Ein einmalig ueber die Gesamthistorie gerechnetes Gate waehlt Titel danach aus, ob
sie SPAETER liquide wurden — ein Survivorship-artiger Lookahead. Das rollierende Fenster ist Teil
des Gates, keine Implementierungsfreiheit.

**Signal-Verfuegbarkeit:** `available_at = FILING_DATE + 1 Tag` (DERA fuehrt keine
ACCEPTANCE-Minute). Zeilen mit `datum_plausibel = False` (0,18 %) duerfen NICHT eingehen.
Zaehlungen ueber `RPTOWNERCIK.nunique()`, Stueck-/Wertsummen erst nach
`drop_duplicates('NONDERIV_TRANS_SK')` — roh sind sie 37,8 % zu hoch (E-124).

**Fenster:** 2006-01 bis 2026-03 (Beginn des DERA-Bestands), nicht 2005–2026.

### Falls gelaufen wird: Familie und Kriterien, ebenfalls vorab

Genau **zwei** Varianten, wie in H-053: (a) alle opportunistischen Kaeufe; (b) Cluster ≥ 2 Insider
je Titel. Steuerwelt PRIVAT_DE entscheidend, ZERO nur berichtet. Kumulatives N: **+2**.

Bestehen verlangt ALLE: (1) Median ueber die rollierenden Fenster > SPY-ETF-Pfad; (2) Sharpe >
No-Signal-Kontrollkorb aus demselben gegateten Universum (sonst misst man nur den Small-Cap-Effekt);
(3) DSR besteht mit **heterogen** geschaetztem V und kumuliertem N — NICHT mit Klonfamilien-Varianz
(E-077); (4) MaxDD nicht schlechter als SPY; (5) das Ergebnis kippt nicht zwischen Primaer- und
Sekundaergate.

### Stopp-Regel und Mindestgroesse — vorab, weil sie sonst nie kommen

**Stopp:** Ein FAIL schliesst das Insider-Feld **endgueltig**. Eine Wiederaufnahme verlangt eine
neue DATENQUELLE, nicht eine neue Auslegung eines alten Verdikts. Das Argument, das diesen dritten
Anlauf rechtfertigt ("H-053s Verdikt hat die These nicht getestet"), waere nach einem FAIL
unveraendert wieder verfuegbar — genau deshalb wird es hier verbraucht und steht kuenftig nicht
mehr zur Verfuegung.

**Mindestgroesse:** Bleiben nach dem Gate weniger als **300 handelbare Namen** oder weniger als
**2.000 opportunistische Kaufereignisse** uebrig, wird **kein Verdikt** gefaellt, sondern
"nicht aussagefaehig" berichtet. Ohne diese Schwelle liesse sich ein schwaches Ergebnis
nachtraeglich als Datenproblem statt als FAIL lesen — und umgekehrt ein FAIL auf zu duenner Basis
als Befund verkaufen.

### Prior — ehrlich

H-031 FAIL (S&P). H-053 FAIL (breite Preise, schmale Signale). Fable H1: auf Survivor-Daten nicht
von der Baseline trennbar. Kein Kandidat dieser Kampagne hat je die vollstaendige
Mehrfachtest-Korrektur bestanden. Die Erwartung ist FAIL.

### Offene Frage — Operator-Entscheidung

Das Insider-Feld gilt nach H-031/H-053 als **verschossen**. Dagegen steht, dass H-053s Verdikt
laut eigenem Vorbehalt die These nicht getestet hat — ein Verdikt auf ungetesteter Grundlage ist
kein Verdikt ueber diese These. Ob das ein zweiter Schuss auf dieselbe Frage ist oder der erste
echte, ist eine Entscheidung, keine Rechnung. Das Gate steht unabhaengig davon fest.

## Welle 47 (registriert 2026-08-05, VOR Lauf) — H-087: Traegt der Trendfilter auch ohne Dauerkrise?

### Die Frage, die P13 prinzipiell nicht beantworten konnte

Der SPY-Trendfilter besteht die Zielfunktion breit, ueberlebt Ausfuehrungsverzoegerung und schlaegt
60 von 60 Zufalls-Timing-Kontrollen — und scheitert an beiden Haelften der Mehrfachtest-Korrektur
(DSR 0,7838, PBO 68,6 %). Unabhaengig davon blieb ein Einwand, der im Suchfenster **nicht**
aufloesbar war: von 144 rollierenden 10-Jahres-Fenstern 1995–2016 ist **kein einziges** krisenfrei,
das mildeste hat −49,2 % Rueckgang. „Trendfolge wirkt" liess sich nicht von „Trendfolge hat
2000–2002 und 2008 umgangen" trennen.

Die kostenlose CRSP-Marktreihe (Ken French, taeglich ab 1926-07) enthaelt beides:
**338 von 1.080 Fenstern sind krisenfrei** — verteilt auf **4 disjunkte Bloecke**
(`krisenfreie_fenster.json`).

### Was gerechnet wird — genau eine Konfiguration

`preis > SMA200`, **a priori** und nicht aus einem Raster gewaehlt (Lehrbuchwert, schon in P4s
Kontrollblock). Kandidat gegen **dieselbe Reihe ohne Filter** — kein ETF-Vergleich, damit E-079
gar nicht erst greifen kann. Steuerwelt ZERO: fuer eine Mechanismusfrage ist die Steuer Rauschen.

Entscheidende Auswertung ist **nicht** der Gesamtmedian, sondern die **Aufspaltung**: Vorsprung in
Krisenfenstern gegen Vorsprung in krisenfreien Fenstern. Kumulatives N: **+1**.

### Pass/Fail — vorab

* **TRAEGT:** In den 338 krisenfreien Fenstern schlaegt der Filter die ungefilterte Reihe im Median
  UND der DD-Deckel −35 % wird in keinem Fenster gerissen. Dann misst er nicht nur Crash-Vermeidung.
* **TRAEGT NICHT:** In krisenfreien Fenstern liegt der Filter im Median gleichauf oder darunter.
  Dann ist der P13-Vorsprung genau das, was der Einwand behauptet hat — Krisenvermeidung, sonst
  Kosten.
* **Massgeblich sind die 4 disjunkten Bloecke, nicht die 338 Fenster** (E-078). Ein Ergebnis, das
  nur in einem Block auftritt, ist eine Episode, kein Mechanismus, und wird so berichtet.

### Was dieser Lauf ausdruecklich NICHT ist

**Kein Deployability-Test.** Vor den 1970ern gab es keine Indexfonds; Transaktionskosten lagen um
Groessenordnungen hoeher als die hier angesetzten `cost_bps`. Die Reihe ist ein
**Mechanismus-Labor**, kein handelbares Instrument. Ein „bestanden" hiesse: der Effekt existiert
auch ohne Dauerkrise — nicht: man haette ihn verdienen koennen.

**Kein Ersatz fuer den gescheiterten PBO.** P13 bleibt an der Mehrfachtest-Korrektur gescheitert.
Dieser Lauf beantwortet eine andere Frage und macht das Verdikt nicht rueckgaengig.

**Kein Holdout.** Das Fenster 2017-01 bis 2026-07 der Kampagnendaten bleibt versiegelt; die
CRSP-Reihe ist eine andere Datenquelle und beruehrt es nicht.

### Prior

Offen. Der Zufalls-Timing-Test (0/60, p = 0,016) spricht dafuer, dass das Timing Information
traegt; die Ereignisabhaengigkeit spricht dagegen. Das ist der Grund, den Lauf ueberhaupt zu machen.
