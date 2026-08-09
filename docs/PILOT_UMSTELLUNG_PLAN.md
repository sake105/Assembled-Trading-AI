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
| Hard-Stop | **DD-Treppe scharf + ack-Halt −20 % (Beschluss Hans 2026-08-09, zweistufig):** (1) Treppe (`policy.yaml drawdown_policy`, gemessen vs. Höchststand seit `hwm_since` 2026-07-02, nie unter Startkapital 87.874,90): −10 % → Order-Stückzahlen ×0,5, −15 % → ×0,2 — beides **zustandslos** (erholt sich automatisch mit dem Drawdown, kein Kill-Switch, kein Token); −20 % → **persistenter Kill-Switch** (Vollstopp, Aufräumen nur mit OPERATOR_KILL_TOKEN). (2) Äußerer ack-Halt (`dd_stop_pct`) ebenfalls −20 %, aber vs. Startkapital — Backstop, greift i. d. R. nach dem Treppen-Kill. **Bewusste Redundanz mit verschiedenen Equity-Quellen** (DD2-03): Treppe misst Ledger-Mark-to-Market, ack-Halt misst Broker-Equity — bei Ledger-Broker-Drift (wie während der Scheduler-Lücke) fängt der Broker-seitige Halt, wenn die Treppe blind wäre. Drossel-Eigenschaften: seitenagnostisch (bremst auch De-Risking-Verkäufe); kumulativ mit min_notional können einzelne kleine Orders ganz entfallen. Der beim Review gefundene Fail-open-Fallback (leergefilterter Batch → ungefilterte Orders an den Broker) ist entfernt — ein leerer Batch bleibt leer. Details in `configs/app.yaml`. Hintergrund: Die Treppe war bis 2026-08-09 im Pilot **wirkungslos** (Equity nie an den Zyklus-Kontext verdrahtet), und die ursprüngliche Soft-Stufe hätte im Broker-Pfad einen Vollstopp statt einer Drossel bedeutet — beides in diesem Schritt behoben. Alt-Peak 99.036 vom 06.05. zählt per `hwm_since` nicht (Baseline-Reset-Governance) |
| Trades letzte 7 T | 0 |
| Reconcile | **Gate-Deadlock 03.–07.08. aufgelöst (2026-08-08).** Die „FAIL"-Zählung im Daily Review waren **10× `reconcile_blocked` mit Grund `reconcile_stale`**: Scheduler-Lücke 2026-07-29–08-02 → Artefakt vom 28.07. überschritt stale-hours (120 h) → ARMED-Gate blockte fail-closed jeden Zyklus 03.–07.08. (Run-Verzeichnisse leer — **der Pilot hat 5 Tage nicht gehandelt**). Recovery via `rebuild_reconcile_artifact.py`: Artefakt neu (2026-08-08T18:27Z, status=OK). **Reichweite dieses OK, präzise:** es liefen die 4 Strukturinvarianten (cash/equity/positions finite, fills↔orders) auf dem **Ledger-Stand vom 28.07.**, KEIN Broker-/Positionswert-Abgleich; das equity-Feld im Artefakt (71.533,72) ist **nur Cash** — der Rebuild bewertet Positionen strukturell mit 0 (vorbestehende Eigenheit des Tools, siehe Follow-up). **Folge der Aktion: das fail-closed Gate ist wieder durchlässig, der Pilot handelt ab dem nächsten Zyklus.** Refresht der Zyklus das Artefakt nicht selbst, re-deadlockt das Gate nach 120 h — Wiederanlauf beobachten. Separat: `reconciliation_audit.jsonl` enthält 431 historische fail-Zeilen über den Gates in identischen Wiederholungs-Tripeln — **Vermutung** (unbelegt): Testlauf-Kontamination vor dem Isolations-Fix `a7aba244`. Eine frühere Fassung dieser Zeile gab Entwarnung aus dem Gedächtnis — unbelegt (E-134); eine zweite Fassung nannte das Rebuild-Artefakt „frisch belegt" ohne diese Einschränkungen — ebenfalls zu viel (Stage-3-Fund) |
| Positionen | 2 Positionen im Ledger: GLD, TLT (Stand `ledger_state.json` 2026-07-28 — eine frühere Fassung nannte hier zusätzlich „5 Adoptionen vom 14.07."; die stehen nicht im aktuellen Ledger-Stand, Verbleib nicht nachrecherchiert) |

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
3. Baseline neu setzen (Equity am Umstellungstag). Hard-Stop-Regel: −20 %
   äußerer Halt + DD-Treppe −10/−15/−20 (Beschluss Hans 2026-08-09, bereits
   umgesetzt in `configs/app.yaml`).
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
