# P3 Kontrolltest — Der P2-Befund ist ein Turnover-Artefakt (2026-08-01)

Rohdaten: `results/p3_kontrolle.json`. Trial-Zähler jetzt **2.008**.
Suchfenster, Holdout unberührt.

**Entscheidungsregel vor dem Lauf festgelegt:** Liegt der Momentum-Median
innerhalb der Zufallsverteilung (< 95. Perzentil), ist der Befund ein Artefakt.

---

## Das Ergebnis

Identische Mechanik, identische Parameter (`hold730 / out200 / ×1,0`),
identisches Universum. Einziger Unterschied: **der Score ist Rauschen statt
Momentum.** 20 Seeds.

| | Median-Faktor | Endwert |
|---|---|---|
| Benchmark SPY | 1,948 | 726.197 |
| **Momentum** | **2,737** | 1.090.312 |
| Zufall — Mittel über 20 Seeds | **2,692** | — |
| Zufall — 95. Perzentil | 3,237 | — |
| Zufall — Maximum | 3,309 | 1.625.185 |

**Momentum liegt im 50. Perzentil der Zufallsverteilung.** Genau die Hälfte
der Zufallsläufe war besser. Elf von zwanzig erreichten einen höheren Endwert,
der beste mit 1,63 Mio gegen Momentums 1,09 Mio.

**Verdikt: kein Auswahl-Alpha.** Was P2 gemessen hat, ist nicht die Güte der
Momentum-Auswahl, sondern die Wirkung von *20 gleichgewichteten Namen aus dem
S&P, lange gehalten* — verglichen mit dem kapitalgewichteten Index. Das Signal
war austauschbar.

## Und der Zufall ist sogar risikoärmer

| | schlimmster MaxDD |
|---|---|
| Momentum | −65,5 % |
| Zufall — Spannweite | −50,0 % bis −66,6 % |
| Zufall — typisch | ca. −55 % |

17 von 20 Zufallsläufen hatten einen **besseren** Drawdown als Momentum. Das
Signal verschlechtert das Risikoprofil, statt es zu verbessern.

## Was hier wirklich gemessen wurde

Nicht Momentum, sondern die Kombination aus:

1. **Gleichgewichtung statt Kapitalgewichtung** — bekannt und gut dokumentiert,
   kein Alpha, sondern eine andere Faktorexposition (Size/Value-nah).
2. **Konzentration auf 20 Namen** — höhere Streuung nach oben und unten; bei
   20 Zufallsläufen ist der beste zwangsläufig gut.
3. **Sehr geringer Turnover** — spart Kosten und in Steuerwelten Steuer.

Alle drei wirken unabhängig vom Signal. Deshalb bekommt Rauschen dasselbe
Ergebnis.

## Konsequenzen

**Für P2:** Befund A („Momentum schlägt den Index in allen Steuerwelten") ist
damit erledigt. Er stimmt als Beobachtung, aber die Ursache ist nicht das
Momentum. Das Dokument wird entsprechend gekennzeichnet.

**Für die Methodik der Kampagne:** Dieser Test kostete zwanzig Läufe und hat
einen Befund gekippt, der sonst in DSR/PBO und womöglich bis zum Holdout
weitergetragen worden wäre. **Ab sofort gilt: jeder Auswahl-Befund braucht die
Zufallskontrolle, bevor er als Befund gilt.** Ein Signal, das eine
Zufallsauswahl nicht schlägt, ist kein Signal.

**Für Mandat I:** Der Kernbefund („kein Brutto-Alpha") wird durch P3 nicht
widerlegt, sondern gestützt — auf einer zusätzlichen Achse. Mandat I hatte
diese Kontrolle nicht systematisch gefahren; die dort verworfenen Kandidaten
bleiben verworfen, aber die *Begründungen* wären teils andere.

## NACHTRAG: mit Risiko-Gate kippt das Verdikt

Dieselbe Kontrolle, 20 Seeds, aber mit dem SMA200-Gate aus P4:

| | Median | Perzentil in der Zufallsverteilung |
|---|---|---|
| Momentum **ohne** Gate | 2,737 | **50.** → Artefakt |
| Momentum **mit** Gate | **4,124** | **100.** → überlebt |
| Zufall mit Gate | Mittel 2,866 · P95 3,437 · Max 3,533 | — |

**Ohne Gate ist das Signal austauschbar, mit Gate nicht.** Momentum liegt dann
über allen zwanzig Zufallsläufen. Das ist eine Interaktion: der Trendfilter
entfernt die Crash-Phasen, und in den verbleibenden Risk-on-Phasen
diskriminiert das Signal offenbar tatsächlich. Details in
`BEFUND_P4_RISIKO.md`.

Das Verdikt dieses Dokuments bleibt für die **ungegatete** Variante gültig.

## Offene Frage, die daraus entsteht

Der Zufalls-Median von 2,69 gegen SPY 1,95 ist selbst bemerkenswert: eine
zufällige, gleichgewichtete 20-Namen-Auswahl aus dem S&P schlug den Index über
rollierende 10-Jahres-Fenster deutlich. Das ist **kein Fehler**, sondern der
bekannte Gleichgewichtungs-Effekt — aber es ist ein Kandidat für sich, der
nichts mit Signalen zu tun hat. Er scheitert allerdings am selben Drawdown
(−50 bis −67 %) und ist damit für die gesperrte Zielfunktion genauso
unbrauchbar. Zu prüfen wäre er trotzdem sauber: mit Kosten, in allen
Steuerwelten und gegen einen gleichgewichteten Index-Benchmark statt gegen SPY.
