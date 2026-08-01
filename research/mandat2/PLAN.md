# FORSCHUNGSMANDAT II — Plan

**Status:** ENTWURF, Phase 0 begonnen · **Erstellt:** 2026-08-01 · **Auftraggeber:** Hans
**Vorgänger:** `research/mandat/FINAL_REPORT.md` (Mandat I, abgeschlossen 2026-07-12, N=1.964)

---

## 1. Auftrag

Das Mandat wird neu geöffnet, weil der **Meilenstein von Mandat I möglicherweise falsch gesetzt
war**. Mandat I maß „schlägt den S&P 500 **nach deutscher Privatanleger-Steuer** über 10-Jahres-
Fenster". Mandat II entfernt diese Bedingung und ersetzt sie durch zwei andere Steuerwelten.

**Nicht neu verhandelt wird:** dass die Antwort ehrlich sein muss. Ein PASS, das nur Suchrauschen
ist, ist schlimmer als ein FAIL.

---

## 2. Zielfunktion (GESPERRT am 2026-08-01, Entscheidung Hans)

```
maximize   Median-Endvermögen über ALLE rollierenden 10-Jahres-Fenster
s.t.       MaxDD >= -35 %   in JEDEM Fenster (nicht nur im Median)
```

- **Benchmark:** SPY Total Return, identische Fenster, identisches Steuerregime.
- Der DD-Deckel ist **bindend, nicht advisory**. Ein Kandidat, der ihn in einem einzigen Fenster
  reißt, ist raus. Grund: sonst gewinnt gehebeltes SPY trivial und die Kampagne ist wertlos.
  SPY selbst lag 2007–2009 bei ca. −55 % — der Deckel verlangt vom Kandidaten also **weniger**
  Risiko als vom Benchmark, nicht mehr.
- **Hebel ist erlaubt** und wird systematisch getestet (Phase 2) — inklusive Finanzierungskosten.
  Er muss sich unter dem DD-Deckel verdienen.

### Steuerregime (vier, parallel gerechnet)

| Regime | Kursgewinn | Verluste | Dividende | Rolle |
|---|---|---|---|---|
| `ZERO` | 0 % | voll | 0 % | Referenz: **existiert überhaupt Brutto-Alpha?** |
| `GMBH_THESAURIEREND` | ~1,5 % (§8b II KStG: 95 % frei) | **NICHT abziehbar** (§8b III) | ~30 % bei Streubesitz <10 % (§8b IV) | **Führend** |
| `GMBH_AUSSCHUETTUNG` | wie oben + 26,375 % auf die Ausschüttung | " | " | Kontrolle: nutzbares Privatvermögen |
| `PRIVAT_DE` | 26,375 %, FIFO, Verlusttopf, SPB 1.000 € | Verlusttopf | 26,375 % | Vergleichbarkeit zu Mandat I |

**Die GmbH-Asymmetrie ist die eigentliche neue Hypothese:** Turnover kostet dort fast nichts
(1,5 % statt 26,375 %) — aber Verluste sind nicht absetzbar und Dividenden werden *teurer*
(30 % statt 26,375 %). Das dreht die Optimierungsrichtung um: **Dividendenstrategien werden
schlechter, Momentum-/Turnover-Strategien deutlich besser.** Genau die Familien, die in Mandat I
an der Steuer starben, bekommen hier eine echte zweite Chance.

> **Vorbehalt:** Ich bin kein Steuerberater. Die Sätze oben sind die gängige Lesart von §8b KStG
> inkl. GewSt-Durchschlag. Bevor daraus eine Strukturentscheidung wird, gehört das fachlich
> geprüft. Für den Vergleich von Strategien untereinander ist die Modellierung belastbar genug.

---

## 3. Statistische Disziplin — der kritische Teil

Wir haben in Mandat I bereits **1.964 Trials** verbraucht. Wenn wir jetzt ohne Steuerbremse erneut
alles durchsuchen, **finden wir garantiert etwas, das SPY schlägt** — und es wird überangepasst
sein. Dagegen:

1. **Trial-Zähler läuft weiter.** N₀ = 1.964. Der DSR-Haircut wird dadurch *härter*, nicht weicher.
   Kein Reset auf 0.
2. **Registry-First bleibt Pflicht.** Jede Hypothese wird in `research/registry.md` eingetragen,
   **bevor** sie läuft — mit fixierten Parametern. Nachträgliches Anpassen = neuer Trial.
3. **Locked Holdout — gesperrt ab heute:**
   - **Suchraum:** 1995-01-03 … **2016-12-31** (22 Jahre, enthält Dotcom-Crash und GFC)
   - **Holdout:** **2017-01-01 … 2026-07-06** (9,5 Jahre, enthält COVID-Crash und 2022) —
     **wird während der Suche nicht angefasst.**
   - Der Holdout ist fast exakt ein 10-Jahres-Fenster, also genau der Zielhorizont.
   - **Ein Schuss pro finalem Kandidat.** Jeder Blick aufs Holdout wird protokolliert.
4. **CPCV/PBO** auf jedem Kandidaten, **DSR** mit dem kumulierten N.
5. Vorab-Kriterien c1–c5 wie in Mandat I (unverändert übernommen, damit Verdicts vergleichbar sind).

**Abbruchregel:** Wenn nach Phase 2 nichts überlebt, ist *das* das Ergebnis. Nicht weitersuchen,
bis etwas passt.

---

## 4. Was gegenüber Mandat I wirklich neu ist

Ohne diese Punkte wäre die Kampagne nur eine Wiederholung:

| # | Neu | Warum es kippen könnte |
|---|---|---|
| 1 | Steuerregime `ZERO` / `GMBH` | dreht die Turnover-Ökonomie um (s. o.) |
| 2 | **Hebel** — in Mandat I **nie** getestet | Kernrestriktion des Systems war „kein Leverage" |
| 3 | **Haltedauer-Sweep** Stunden → Jahre, systematisch statt ad hoc | Mandat I testete Haltedauern nur punktuell |
| 4 | **Fundamentalbewertung** | in Mandat I klar untergewichtet; 352k XBRL-Fakten liegen ungenutzt |
| 5 | **DD-Deckel als Nebenbedingung** statt Endvermögen pur | verändert, welche Kandidaten überhaupt zulässig sind |
| 6 | Externe Ideengeber (wikifolio/eToro) | Stile, auf die wir selbst nicht gekommen sind |

---

## 5. Phasen

### P0 — Fundament (läuft) · *ohne das ist nichts davon messbar*
- `TaxRegime`-Protokoll + vier Implementierungen; `TaxedPortfolio` wird regime-agnostisch.
  Rückwärtskompatibilität: `PRIVAT_DE` muss die Mandat-I-Zahlen **byte-gleich** reproduzieren.
- Margin-/Hebelmodell inkl. Finanzierungskosten (Broker-Satz, zeitvariabel) und Margin-Call-Logik.
- Haltedauer als expliziter Parameter (min/max Haltedauer, Rebalance-Trigger).
- DD-Deckel + 10-Jahres-Fenster-Auswertung in die Verdict-Engine.
- **Gate:** `PRIVAT_DE`-Regression gegen mindestens 3 Mandat-I-Ergebnisse, bevor P1 startet.

### P1 — Re-Run aller Mandat-I-Familien unter den neuen Regimen
Beantwortet direkt Hans' Frage „war der Meilenstein falsch gesetzt?". 80+ Familien, jeweils
4 Regime × neues Erfolgsmaß. Erwartung (vorab, damit ich mich nicht selbst betrüge):
die meisten bleiben FAIL, weil Grund #1 (kein Brutto-Alpha) steuerunabhängig ist.

### P2 — Hebel × Haltedauer (Sweep)
Systematisches Gitter: Haltedauer {Stunden, Tage, Wochen, Monate, 1J, 2J, 5J+} × Hebel {1,0 … 2,0}
× Finanzierungskosten. DD-Deckel bindend. Erwartung: Hebel scheitert am Deckel, außer bei
volatilitätsgesteuerter Skalierung — das ist der ehrlichste Kandidat der ganzen Kampagne.

### P3 — Fundamentalbewertung (die Lücke aus Mandat I)
XBRL as-reported, PIT-korrekt (`disclosure_date`, nicht `event_date`). Klassische Bewertungsarbeit:
Ertragskraft, Bilanzqualität, Kapitalallokation, Bewertungsniveau. Hier wird auch geprüft, ob sich
eines der **Anthropic Finance-Agent-Templates** (Statement-Auditor / Comparables) methodisch auf
unsere EODHD/XBRL-Daten adaptieren lässt — die Enterprise-Connectoren (FactSet, CapIQ) haben wir
nicht, das Skill-Muster ist aber übertragbar.

### P4 — Externe Ideengeber (wikifolio, eToro, weitere)
Öffentliche Strategien und Handelshistorien, soweit die jeweilige Plattform automatisierten Zugriff
zulässt (robots.txt, Rate-Limits, keine Umgehung von Login- oder Anti-Bot-Schranken). **Status des
Ergebnisses: Hypothese, nie Beleg** — Leaderboards sind massiv survivorship-verzerrt.

### P5 — Neue Strategiefamilien
Aus P3/P4 abgeleitet + Literatur-Sweep. Registry-First.

### P6 — Synthese + Holdout
Finale Kandidaten bekommen ihren **einen** Holdout-Schuss. Abschlussdossier wie Mandat I.

---

## 6. Was ein PASS bedeutet

Ein Kandidat ist erst dann deployable, wenn er **alle** erfüllt:
c1–c5 (Mandat-I-Kriterien) · DD-Deckel in jedem Fenster · DSR mit kumuliertem N · PBO unter
Schwelle · **und** den Holdout-Schuss besteht.

Alles darunter ist ein Zwischenergebnis, kein Ergebnis.
