# Memo: Abfindungs-/Nachbesserungswerte (Squeeze-outs & Spruchverfahren)

*Für Hans, 2026-07-13. Kontext: die deutsche Nische der neuen Alpha-Landkarte (kapazitätsbeschränkt,
Fall-für-Fall — deshalb Memo statt Backtest; ein systematischer Test ist hier prinzipiell nicht möglich.)*

## Mechanik in 5 Schritten

1. **Trigger:** Ein Großaktionär (≥90/95 %) kündigt einen **Squeeze-out** an (aktienrechtlich §327a
   AktG, verschmelzungsrechtlich, oder übernahmerechtlich §39a WpÜG) — oder einen **Beherrschungs-
   und Gewinnabführungsvertrag (BGAV)** mit Abfindungsangebot.
2. **Einstieg:** Kauf der Aktie NACH Ankündigung, typischerweise nahe/knapp über der angebotenen
   Barabfindung. Downside ab hier: minimal — die Abfindung ist der Boden (sie ist dir rechtlich
   sicher, sobald der Squeeze-out wirksam wird).
3. **Vollzug:** Eintragung ins Handelsregister → Zwangsabfindung wird ausgezahlt. Kapital zurück.
4. **Die Option:** Im **Spruchverfahren** (SpruchG) prüft das Gericht die Angemessenheit der
   Abfindung. Läuft automatisch für ALLE abgefundenen Aktionäre, wenn ein Antragsteller es anstößt
   — du musst nichts tun und trägst keine Verfahrenskosten (trägt i. d. R. die Gesellschaft).
5. **Nachbesserung:** Erhöht das Gericht (oder ein Vergleich) die Abfindung — historisch in einem
   erheblichen Teil der Verfahren, oft nach 3–8 Jahren — bekommst du die Differenz **plus Zinsen
   (5 Pp über Basiszins seit Wirksamkeit!)** nachgezahlt. Senkung ist ausgeschlossen (Verbot der
   reformatio in peius). → **Asymmetrie: Downside ~0, Upside = Gratis-Option + hohe Verzinsung.**

## Warum das echtes, nicht wegarbeitierbares Alpha ist

- **Kapazität:** Free Float nach Squeeze-out-Ankündigung ist winzig (Millionen, nicht Milliarden) —
  für Fonds irrelevant, für 5- bis 6-stellige Beträge gut zugänglich.
- **Zeithorizont:** Jahre bis zur Nachbesserung — institutionell unattraktiv, privat egal.
- **Rechtsbasis statt Markt-Timing:** Die Rendite kommt aus Bewertungs-/Verfahrensrecht
  (Ertragswert vs Börsenkurs), nicht aus Kursprognosen. Unkorreliert zum Markt.
- Dokumentierte Praktiker-Historie (Nebenwerte-Szene, z. B. Solventis-Nachbesserungsstudien:
  Mehrzahl der Verfahren endet mit Erhöhung; typ. einstellige bis niedrig zweistellige Prozente
  auf die Abfindung, plus Zinsen über die Laufzeit).

## Risiken (ehrlich)

1. **Kapitalbindung/Illiquidität:** Geld ist bis Vollzug gebunden; die Nachbesserungs-Option zahlt
   erst nach Jahren. Kein Zwischenausstieg für den Nachbesserungsanspruch (nicht handelbar; es gab
   zeitweise OTC-Käufer für „Nachbesserungsrechte", unzuverlässig).
2. **Deal-Break:** Wird der Squeeze-out abgesagt (selten nach HV-Beschluss), fällt der Kurs auf
   Stand-alone-Niveau. Einstieg NACH HV-Beschluss/Registeranmeldung minimiert das.
3. **Nullrunden:** Ein Teil der Spruchverfahren endet ohne Erhöhung → Rendite = Einstiegs-Spread
   zur Abfindung ± Zinsen. Deshalb Portfolio-Ansatz (viele kleine Positionen) statt Einzelwette.
4. **Steuer:** Abfindung + Nachbesserung = Kapitalertrag (26,375 %); Zinsen ebenso. Kein §23-Vorteil.
5. **Prozessdauer-Drift:** Verfahren können >10 J laufen (Extremfälle).

## Wie ein Prozess für dich aussähe (wenn gewünscht, baue ich Tooling)

- **Quellen:** Bundesanzeiger (Squeeze-out-Bekanntmachungen, Spruchverfahrens-Anträge),
  Unternehmensregister, HV-Einladungen; Szene-Aggregatoren (z. B. spruchverfahren-blog) als Radar.
- **Watchlist-Bot:** Scanner über Bundesanzeiger-Veröffentlichungen („Squeeze-out", „§327a",
  „Spruchverfahren eingeleitet") → Kandidatenliste mit Abfindungshöhe vs Kurs.
- **Entscheidungsregel (konservativ):** Nur nach HV-Beschluss; Einstieg ≤ ~2–3 % über Abfindung
  (Zins-Carry deckt das); Positionsgröße klein & gestreut (10–20 Fälle); Nachbesserung als
  Bonus betrachten, nicht einpreisen.
- **Buchführung:** Eigene Akte je Fall (Abfindung, Stichtag, Verfahrensstand) — die Ansprüche
  muss man über Jahre nachhalten (Depotbank schreibt Nachbesserungen automatisch gut, aber
  Kontrolle nötig).

## Einordnung ins Mandat

Kein Ersatz fürs Kern-Portfolio (Kapazität + Bindung), sondern ein **Satelliten-Sleeve** (z. B.
5–10 % des Vermögens, 10–20 Fälle). Erwartbarer Charakter: anleiheähnlicher Carry (Einstieg nahe
Abfindung + Verzinsung) mit asymmetrischer Gratis-Option auf Nachbesserung — realistisch mittlere
einstellige Jahresrendite mit geringem Beta und ECHTER, struktureller Alpha-Quelle obendrauf.
Nächster Schritt (deine Entscheidung): Bundesanzeiger-Watchlist-Scanner bauen + aktuelle offene
Fälle inventarisieren.
