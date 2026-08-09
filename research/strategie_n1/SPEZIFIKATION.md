# Strategie N1 — Multiquellen-Risiko-Score (Entwurf v0.1)

**Status: SPEZIFIKATION — nichts implementiert, nichts getestet, keine Trials gebucht.**
Auftrag Hans, 2026-08-09 (wörtlich): Zusammenspiel aus geopolitischen Nachrichten,
Finanznachrichten, sozialen Netzwerken und technischer Analyse; daraus ein
Bewertungsschema und darauf Handelsentscheidungen; eigene Risikoanalyse; Ziel
Ø ≈ +2 pp pro Trade; Exit ist der schwierigste Teil. Claude darf Stellschrauben
verbessern, **der Grundgedanke bleibt** — deshalb als eigene Strategie geführt.

---

## 1. Ehrlicher Rahmen (Pflichtlektüre vor jeder Weiterarbeit)

Unsere eigene Evidenz zu den Einzelkomponenten dieser Familie:

| Komponente einzeln | Unser Befund | Quelle |
|---|---|---|
| Geopolitik-Posts → Trades | Post-Tage = Zufalls-Tage (1.122 Ereignisse, alle Horizonte) | Welle 48/48b |
| News-Intraday-Signale | brutto real, netto tot (Kosten) | H-039..H-046 |
| Insider/Kongress/Events | keine deploybare Kante | Mandat I/II, Fable-Exploration |
| TA (Trend/Breakout/…) | 80+ Familien, keine überlebt die Mehrfachtest-Korrektur | FINAL_REPORT.md |

**Was hier trotzdem NEU ist:** die *Fusion* mehrerer schwacher Quellen zu einem
Score mit hoher Selektivität. Das haben wir nie getestet. Ehrliche Erwartung:
Einzeln tote Signale ergeben selten zusammen ein lebendiges — aber die
Filter-Richtung („nur handeln, wenn ALLES zusammenpasst, sonst nichts tun")
ist genau die Richtung, in der unsere Anti-Lektion zeigt (Eskalations-News =
Grund für Nichtstun). N1 wird deshalb primär als **Risiko-/Selektionsschema**
gebaut, nicht als Feuer-oft-Signal.

**Zum +2-pp-Ziel:** Ø +2 pp netto/Trade erzwingt mechanisch: Haltedauern von
Tagen bis Wochen (nicht 5m), sehr wenige Trades (eher 1–3/Monat), und einen
Exit, der Gewinner laufen lässt. Zum Vergleich: unsere beste je gemessene
Bruttokante lag bei ~0,6 pp/Trade. +2 pp ist das Ziel, nicht die Erwartung —
das Erfolgskriterium der Evaluation ist vorab „netto > 0 mit t > 2 gegen
Zufalls-Kontrolle", NICHT die 2 pp; sonst belügen wir uns beim ersten Lauf.

## 2a. Quellen-Stand nach Runde 2 (2026-08-09, Hans: Reddit ja, NYT-API-Key entfaellt)

**AKTIV im PIT-Sammler (`sammler.py`, 20 Quellen, Erstlauf 732 Eintraege):**
Geopolitik: Reuters (Google-News-Proxy), NYT-RSS (ohne Key), BBC, Al Jazeera,
Guardian, DW, Tagesschau, Anadolu. Finanz: FAZ, CNBC, MarketWatch + WSJ
(offizielle Dow-Jones-Feeds), Handelsblatt, n-tv, wallstreet-online, **EZB- und
Fed-Pressemitteilungen (Primaerquellen)**. Social: Reddit r/geopolitics +
r/worldnews (RSS, OHNE App/OAuth; 429 heilt sich pro Lauf), **Clash Report via
offenem Telegram-Mirror (t.me/s/)** — Hans' Wunschquelle.
Ablage: `archiv/YYYY-MM-DD.jsonl` (gitignored), jeder Eintrag mit
`fetched_utc` = PIT-Verfuegbarkeit. Betrieb: 1-2x/Std (Task-Scheduler =
Operator-Entscheidung; bis dahin manuell/Session-getrieben).

**Geprueft und (vorerst) NICHT nutzbar:** Tasnim (DNS), PressTV (kaputtes
TLS-Zertifikat), ReliefWeb-API (410), Bluesky-Such-API (403 — andere Endpoints
spaeter pruefbar), finanzen.net (403), Bloomberg (Paywall/ToS), X/Twitter
(API kostenpflichtig — Clash Report kommt stattdessen ueber Telegram).

## 2b. Urspruengliche Machbarkeitsprobe (Runde 1, HTTP-Status, keine Inhalte)

| Quelle (Hans' Liste) | Zugang | Status |
|---|---|---|
| Reuters | via Google-News-RSS-Proxy | ✅ frei (Headlines+Zeitstempel) |
| NYT | offizielles RSS (World) | ✅ frei; volle API mit Gratis-Key (Operator: Registrierung) |
| Bloomberg | Paywall + ToS verbietet Scraping | ❌ nur via Aggregator-Headlines |
| FAZ Finanzen | offizielles RSS | ✅ frei |
| finanzen.net | RSS 403 (Bot-Block) | ⚠️ ggf. mit korrektem Feed-Pfad/UA; sonst verzichtbar |
| boerse / stock3 | probierte RSS-Pfade 404 | ⚠️ Pfade recherchieren; nachrangig |
| Tasnim | DNS-Fehler bei Probe | ⚠️ erneut prüfen; iranische Staatsquelle → nur als Eskalations-Indikator, nie als Fakten-Quelle |
| Clash Report / „Tabz" | X-Accounts; X-API kostenpflichtig | ❌ ohne API-Budget; Operator-Entscheidung |
| Social allgemein | Reddit-API braucht OAuth-App (gratis) | ⚠️ Operator: App-Registrierung |
| **GDELT** (nicht auf der Liste) | freie Volltext-/Event-API, Historie seit 2015 | ✅ **wichtigste Backtest-Quelle** — einzige mit Vergangenheit |
| Truth-Social-Archiv | bereits im Haus (40.631 Posts) | ✅ vorhanden |
| Finnhub-News-Client | bereits im Repo (`data/finnhub_*`) | ⚠️ Key-Status prüfen (Operator) |
| EODHD-News | Token tot (401) | ❌ bis Abo/Token geklärt |

**Harte Konsequenz:** RSS hat keine Vergangenheit. Ein Backtest der vollen
Fusion ist unmöglich — nur GDELT (+ Truth-Archiv + Kurse) ist rückwirkend da.
Deshalb zweigleisig: **(A) Backtest** der Score-Logik auf GDELT-Historie,
**(B) Forward-Shadow**: der volle Score läuft live mit, handelt aber nichts,
bis N Monate Shadow-Daten ein vorregistriertes Kriterium erfüllen.

## 3. Bewertungsschema v0 (Vorschlag, VOR erster Datensicht zu fixieren)

Vier Teilscores, je −3…+3, PIT-gestempelt auf *Verfügbarkeitszeit*:

1. **GEO** — Eskalation/Deeskalation aus Geopolitik-Quellen: Ereignisklasse
   (Waffenstillstand, Sanktion, Angriff, Ministertreffen …) × Quellen-Breite
   (wie viele UNABHÄNGIGE Quellen binnen 6 h) × Neuheit (Novelty gegen
   30-Tage-Fenster). Wortlisten VORAB fixiert (Welle-46-Disziplin).
2. **FIN** — Finanz-News: Richtung × Spezifität (nennt konkrete Assets?) ×
   Quellen-Breite. Keine Kursbezüge im Text als Feature (GDELT-Headline-Falle:
   „ignites rally" selektiert auf das Ergebnis — verboten).
3. **SOC** — Soziale Bestätigung: Volumen-Anomalie der Erwähnungen (z-Score
   gegen eigene Historie), NICHT Stimmung. Anfangs nur Truth-Archiv + Reddit
   (falls Key), sonst Gewicht 0 mit offenem Slot.
4. **TA** — reiner *Veto*-Filter, kein Signalgeber: handle Long-Risiko-Off-
   Instrumente (Öl, Gold, Rüstung/Defence-ETF, VIX-Nachbarn — Guardrail 4:
   keine Derivate → nur ETFs/Aktien long) nur, wenn der Trend nicht dagegen
   steht (Preis > 50-Tage-Linie o. ä.). TA darf Trades nur VERHINDERN.

**Gesamtscore** = gewichtete Summe (Startgewichte 40/30/10/20, vorab fixiert,
Änderung = neue Registrierung). **Handelsregel v0:** nur |Score| ≥ Schwelle
(vorab: oberste 2 % der Shadow-Verteilung) UND eindeutiges Ziel-Asset-Mapping
(Ereignisklasse→Instrument-Tabelle, vorab fixiert) → eine Position, Größe fix.

## 4. Exit (Hans' schwierigster Teil) — v0-Vorschlag

Drei Ausgänge, der erste, der feuert, gewinnt; alles vorab fixiert:
1. **Score-Zerfall:** Ereignis-Score fällt unter die Hälfte des Entry-Scores
   (Quellen versiegen / Deeskalation) → Exit nächster Schluss.
2. **Zeit:** max. 15 Handelstage (Ereignisprämien zerfallen; Welle-48b-Lehre:
   keine Langzeit-These ohne neue Registrierung).
3. **Risiko:** −5 % vom Entry hart (eigene Risikoanalyse der Strategie,
   unabhängig vom Pilot-Rahmen), +Trailing ab +3 % mit 2 % Abstand.
Kein diskretionäres Übersteuern — das wäre der Rückweg zum Rosinenpicken.

## 5. Eigene Risikoanalyse (Hans' Anforderung)

Je Trade dokumentiert das System VOR dem Entry: Score-Zerlegung (welche Quelle
trägt wie viel), angenommenes Szenario, Invalidierung (was müsste passieren,
damit die These tot ist), Kostenannahme, Positionsgröße (fix 1 R = 0,5 % des
Buchs im Shadow). Das Protokoll IST das Produkt der ersten Monate — nicht die
Rendite.

## 6. Evaluationsprotokoll (nicht verhandelbar)

- Jede Backtest-Konfiguration = Trials im Zähler, VOR dem Lauf gebucht.
- Zufalls-Kontrolle: gleiche Exits auf Zufalls-Entry-Tagen desselben Assets.
- Erfolgskriterium vorab: netto-Ø > 0, t > 2, PF > 1, klarer Abstand zur
  Kontrolle — über ALLE Ereignisse, nicht die zitierfähigen.
- Holdout 2017–2026-07 bleibt versiegelt; GDELT-Backtest läuft auf 2015–2016 +
  Post-Holdout-Fenstern, sonst Forward-only.
- Shadow-Phase min. 3 Monate ODER 20 Score-Trigger, was später kommt.

## 7. Nächste Schritte (Reihenfolge)

1. **Operator (Hans):** ERLEDIGT teilweise (Reddit ja via RSS ohne App;
   NYT-API entfaellt, RSS reicht). OFFEN: X-Budget (optional, Telegram-Mirror
   deckt Clash Report), EODHD-Abo, Finnhub-Key-Status, Scheduler-Task fuer
   den Sammler.
2. Wortlisten + Ereignisklassen→Instrument-Tabelle schreiben (preisblind,
   Registrierung).
3. ERLEDIGT (2026-08-09): `sammler.py` archiviert 20 Quellen PIT-gestempelt.
   Offen: Task-Scheduler-Registrierung (Operator) + GDELT-Poller separat.
4. GDELT-Backtest der GEO-Komponente (erst nach 2., mit Trials-Buchung).
5. Score-Fusion + Shadow-Runner (separater Codepfad, kein Pilot-Kontakt).

*Nichts hiervon berührt den Paper-Piloten oder Schutzzonen, bis Shadow-Evidenz
vorliegt und Hans die Verdrahtung explizit freigibt.*
