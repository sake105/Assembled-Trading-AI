# Plan: Paper-Pilot auf die Mandats-Endspezifikation umstellen

**Status: ENTWURF — wartet auf Freigabe durch Hans.** Die Umsetzung berührt
`src/assembled_core/paper/` und `execution/` (Schutzzonen); ohne explizite
Freigabe wird nichts am laufenden Piloten geändert.

## Warum

Der Pilot fährt die Signal-Logik aus der Zeit vor dem Mandats-Abschluss. Die
Forschung ist beendet, und ihr Ergebnis ist eindeutig: **kein Signal hat die
Mehrfachtest-Korrektur bestanden** — das Verlässliche ist die Aufstellung
selbst plus Turnover-Disziplin. Der Pilot sollte messen, was wir tatsächlich
tun würden, nicht was widerlegt ist.

## Ist-Zustand (Review 2026-08-08)

| | |
|---|---|
| Equity | $86.723,26 (Tag 14, −0,66 % vom Start) |
| Hard-Stop | −10 %, aktuell 9,3 pp Luft |
| Trades letzte 7 T | 0 |
| Reconcile | letzter SLO-Eval: 0 Verstöße (die 12 „FAIL" im Review sind CLI-Meldungen unterhalb des Schwellen-Gates — bekanntes Verhalten, kein Halt-Kriterium) |
| Positionen | u. a. GLD/TLT (Juli adoptiert) + 5 Adoptionen vom 14.07. |

## Ziel-Zustand (Endspezifikation aus `research/mandat/FINAL_REPORT.md`)

1. **Kern 65–70 %**: Welt-Aktien-ETF (im Paper-Konto: SPY oder VT als Proxy).
2. **Stabilisierer 25 %**: Anleihen + Gold (TLT/IEF + GLD — teilweise vorhanden).
3. **Satelliten 5–10 %**: §23-Sleeve/Experimente, im Paper-Pilot zunächst Cash.
4. **Cash-Flow-Rebalancing**: keine Verkäufe zum Rebalancing; Zuflüsse (im
   Paper simuliert als monatliche Einzahlung) kaufen die untergewichtete Klasse.
5. **Verhaltensregel aus Welle 48b**: Eskalations-Nachrichten sind explizit
   KEIN Handelsgrund. Der Pilot handelt nur am Monatsanfang (Cash-Flow) —
   sonst nie.

## Umsetzungsschritte (nach Freigabe)

1. Bestehende Einzelaktien-Positionen beim nächsten Monatswechsel in die
   Zielallokation überführen (einmalige Verkäufe, dokumentiert als Umstellung).
2. Signal-Pipeline für den Piloten deaktivieren; Ersatz: monatlicher
   Cash-Flow-Rebalancer (neuer, kleiner Codepfad — Review-Kette Pflicht).
3. Baseline neu setzen (Equity am Umstellungstag), Hard-Stop-Regel unverändert.
4. Erfolgskriterien anpassen: nicht mehr „CAGR ≥ 20 %" (das war die
   Signal-Ära), sondern **Tracking gegen die theoretische Endspez-Kurve**
   (± Kosten) und **Null ungeplante Trades**.

## Was NICHT passiert

Kein Live-Geld, keine Derivate, kein Hebel (Guardrail 4 unberührt). Keine
Änderung an Kill-Switch, Reconcile oder Halt-Mechanik.

## Freigabe

- [ ] Hans: Umstellung wie oben — ja/nein/geändert
- [ ] Hans: Proxy für den Weltaktien-Kern (SPY einfach, VT näher an der Spez)
- [ ] Hans: monatliche Simulations-Einzahlung (Vorschlag: $1.000)
