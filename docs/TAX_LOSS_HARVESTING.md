# Tax-Loss-Harvesting Workflow (DE) — Audit C2-064

**Status:** Operations-Doc + Detection-Script (2026-05-18).
**Audience:** Operator mit echtem German-tax-Trading-Account (Anlage KAP).
**Scope:** US-Broker (Alpaca/IBKR) + German tax filing (Abgeltungsteuer
25% + Soli 5.5% = 26.375%).

## Konzept

**Tax-Loss-Harvesting** = gezielt verlustreiche Positionen schließen, um
realisierte Verluste in den **Verlustverrechnungstopf** zu schreiben, der
gegen realisierte Gewinne im selben oder folgenden Jahr verrechnet wird.

In Deutschland gilt:
- Aktien-Verluste verrechnen **nur** mit Aktien-Gewinnen (nicht mit
  Zinsen/Dividenden) — §20 (6) EStG, "Verlustverrechnungstopf Aktien".
- Verluste **rolling forward** in zukünftige Jahre — unbegrenzt seit 2022.
- Stichtag = 31.12. (Buchungsdatum).
- **Wash-Sale-Regel:** In Deutschland gibt es **keine** Wash-Sale-Regel
  wie in den USA — rückwärtige Wiederherstellung der Position nach
  Verkauf ist erlaubt. ABER: nur kostenfrei sinnvoll, wenn Spread/Commission
  geringer als der Steuervorteil ist.

## DE-Q3-Workflow (Audit C2-064)

Der Workflow startet im 3. Quartal (Juli-September), nicht erst kurz
vor Jahresende. Grund: bis Q3 ist genug Year-to-Date P&L sichtbar, um
sinnvoll zu planen, und es bleibt genug Zeit für die Ausführung ohne
Last-Minute-Liquidität-Stress.

### Q3-Review (Juli/August)

1. **Realisierter Stand:** Run `scripts/ops/check_tax_loss_harvest.py`
   gegen das YTD-Ledger. Output: realisierte Gewinne und Verluste YTD.
2. **Unrealized-Loss-Kandidaten:** Skript listet offene Positionen mit
   negativer Total-Return seit Eröffnung, sortiert nach
   absolutem unrealized Loss.
3. **Verrechnungspotential:** Skript rechnet aus, wie viele Verluste man
   noch realisieren müsste, um YTD-realisierte Gewinne zu neutralisieren.
4. **Strategische Entscheidung (Operator, NICHT Auto-Trade):** Welche
   Verlust-Positionen verdienen es realisiert zu werden vs welche haben
   Recovery-Potential? Skript liefert nur Kandidaten + Zahlen, keine
   Order.

### Q4-Ausführung (November/Dezember)

5. Selektierte Positionen schließen via normalen Order-Workflow.
   **Keine** auto-execution durch das Tax-Loss-Skript — Risiko: blindes
   Schließen einer Position, die im Trading-System noch ein offenes Signal
   trägt.
6. Re-Entry: wenn die Strategy die Position im neuen Jahr wieder
   eröffnen will, das normale Signal-Routing nutzen (kein "tax-driven
   re-buy").

### Jahresabschluss (31.12. / Januar)

7. Run `accounting/tax_lots.py` für endgültiges FIFO-Matching.
8. `compliance/tax_report.py::summarize_closed_lots` für Anlage-KAP-Vorbereitung.

## Was das Skript NICHT macht

- **Keine Order-Generierung.** Tax-Loss-Harvesting ist eine
  Operator-Entscheidung; das Skript ist read-only-Audit.
- **Keine Wash-Sale-Erkennung.** In DE nicht nötig; bei US-Broker-Wahl
  trotzdem Auge offen halten falls man mit US-tax-Connection trades.
- **Keine Steuer-Quote-Berechnung** für Spezialfälle wie Vereinigte
  Investments / Investmentsteuergesetz (InvStG) — out of scope.
- **Keine FX-Konvertierung** auf-the-fly — nutze `accounting/tax_lots.py`
  für ECB-Reference-Rate-EUR-Wandlung.

## Audit-Cross-Reference

| Item | Bezug |
|---|---|
| C2-064 | Tax-Loss-Harvesting (dieses Doc + Detection-Script) |
| §50.1 50_COMPLIANCE_RECHT.md | German tax-lot tracking foundation |
| `src/assembled_core/accounting/tax_lots.py` | FIFO + ECB-EUR-rate |
| `src/assembled_core/compliance/tax_report.py` | Annual Anlage-KAP summary |
| C2-076 UG-Gründung | gewerblicher Trading-Status — andere Regelung |

## Limitations

Dieses Skript prüft P&L auf Trade-Ebene gegen den paper-Ledger. Für
Echtgeld-Konto muss der reale Broker-Statement-Import erst funktionieren
(siehe Audit §8.7 Fill-Modell). Ohne Real-Ledger ist das Skript ein
Trockenlauf zur Workflow-Validierung.
