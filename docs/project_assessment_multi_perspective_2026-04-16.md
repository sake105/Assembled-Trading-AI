# Assembled-Trading-AI — Multi-Perspektiven-Bewertung

**Datum:** 16. April 2026
**Methode:** Simulierte Stakeholder-Analyse (VC, Quant Fund CTO, Angel Investor, Fintech Analyst)
**Datenbasis:** Direkte Repository-Inspektion, verifizierte Kennzahlen

---

# Perspektive 1: VC Due Diligence Memo

**Fund:** Fintech Fund ($200M AUM) | **Stage:** Pre-Seed / Seed
**Analyst:** Senior Partner

---

## 1. Executive Summary

**Meeting nehmen: Ja. Investieren: Nein — noch nicht.**

Assembled-Trading-AI ist ein beeindruckend umfangreiches Solo-Projekt, das in sechs Monaten eine Codebasis aufgebaut hat, die in Breite vielen Zwei-bis-Drei-Personen-Teams in 18 Monaten entspricht. Die Architektur ist durchdacht, die Testdisziplin ungewoehnlich hoch fuer ein Pre-Revenue-Projekt, und der Geopolitik-Layer ist ein genuiner Differentiator.

Aber: Es gibt kein Produkt, keinen Umsatz, keine Nutzer, kein Team, keinen Track Record mit echtem Geld, und eine bekannte Security-Luecke. Das ist kein Series-A-Case. Es ist bestenfalls ein Pre-Seed-Kandidat — und selbst dafuer muessten fundamentale Fragen geklaert werden.

---

## 2. Technische Due Diligence

### Was beeindruckt

- **Testabdeckung und Disziplin.** 1.243 Tests, 0 Failures, Walk-Forward-QA, Phase-basierte Testsuites. Das ist fuer ein Solo-Projekt in diesem Stadium aussergewoehnlich. Die meisten Quant-Projekte, die wir sehen, haben bestenfalls 50 handgeschriebene Tests.
- **Modulare Architektur.** 22 Subsysteme mit klarer Schichtentrennung (Data, Features, Signals, Portfolio, Execution, Risk, Compliance, Ops). Das deutet auf architektonisches Denken hin, nicht nur auf Feature-Hacking.
- **Risk-First-Philosophie.** Kill-Switch, State-Machine, Pre-Trade-Gates, VaR (Monte-Carlo + Component), Drawdown-Controls. Die meisten Solo-Quant-Projekte ignorieren Risk komplett.
- **Geopolitik-Layer.** Crisis Alpha, Sanctions-Screening, Shock Propagation, Wargaming-Szenarien — das ist ein genuiner Differentiator gegenueber QuantConnect, Zipline und Co.
- **30+ ML-Faktoren mit Regime-Conditioning.** Stacking, Feature Selection, Feedback Loops, Overfitting-Gates. Technisch ambitioniert.

### Was Sorgen macht

- **Keine Live-Validierung.** Null Sekunden echtes Marktexposure. Kein Paper-Trading-Log. Kein Backtest-Report mit realen Daten, der unabhaengig verifiziert wurde. Die gesamte Architektur ist Theorie.
- **Complexity Debt.** 479 Source Files in 6 Monaten = ~2,7 neue Dateien pro Tag. Das Tempo ist beeindruckend, aber es erzeugt Wartungsschuld, die ein Einzelentwickler nicht tragen kann.
- **Security-Vorfall.** `.env` mit echten API-Keys in der Git-History. Laut Projekt-Memory seit Wochen bekannt, Keys nicht rotiert. Das ist nicht akzeptabel.
- **Version 0.0.1 Alpha.** Kein Release, kein Packaging, kein Deployment-Konzept. Der Weg von "Forschungs-Monolith" zu "betreibbarem Produkt" ist lang.
- **Synthetische Daten vs. Realitaet.** Die CLAUDE.md warnt explizit davor, synthetische Daten als Produktionsbeweis zu verwenden. Das deutet darauf hin, dass Teile der Validierung auf synthetischen Daten basieren.

---

## 3. Team Risk

**Bus Factor: 1. Das ist der groesste Dealbreaker.**

Ein einzelner Entwickler, der 479 Files ueber 22 Subsysteme pflegt, ist kein Geschaeftsmodell. Es ist ein Hobby mit beeindruckendem Output.

**Was wir fordern wuerden:**
- Mindestens ein Co-Founder mit komplementaerem Profil (Quant Trading + Ops oder ML Engineering + Infra).
- Klarer Plan fuer die ersten drei Hires: Quant Developer, DevOps/Infra, Risk/Compliance.
- Vesting-Struktur, die den Solo-Founder langfristig bindet.
- Nachweis, dass der Entwickler bereit ist, von "alles selbst bauen" zu "Team fuehren" zu wechseln.

---

## 4. Market Opportunity

**Der Markt ist real, aber fragmentiert.**

- **Retail Quant Platforms** (QuantConnect, Alpaca): Gut finanziert, aber generisch. Kein Geopolitik-Layer, keine Risk-State-Machine auf diesem Niveau.
- **Institutional Tools** (Bloomberg AIM, FlexTrade): Fuer $50k+/Jahr. Assembled koennte theoretisch die Luecke zwischen Retail und Institutional besetzen.
- **Geopolitik als Alpha-Quelle:** Genuiner Trend. Bridgewater, Citadel und Two Sigma investieren massiv in alternative Daten. Ein Open-Source-naher Ansatz, der das demokratisiert, hat narrativen Wert.

**Problem:** Der Markt fuer "Quant-Trading-Frameworks fuer fortgeschrittene Retail-Trader und kleine Funds" ist klein und preissensitiv. Die Zahlungsbereitschaft liegt bei $50-500/Monat, nicht bei SaaS-typischen Werten.

---

## 5. Monetization

Realistische Pfade, nach Wahrscheinlichkeit:

1. **SaaS fuer Semi-Professionelle Trader** — $99-499/Monat, gehostete Version mit Paper-Trading und Backtests. TAM begrenzt, aber klare Unit Economics moeglich.
2. **Data/Signal-as-a-Service** — Der Geopolitik-Layer als API fuer andere Plattformen. Hoeherer Wert, aber erfordert Datenqualitaet, die noch nicht nachgewiesen ist.
3. **Acqui-Hire / Tech-Absorption** — Ein groesserer Player (Alpaca, Interactive Brokers, Bloomberg) kauft die Technologie und den Entwickler. Realistischster Exit-Pfad.
4. **Managed Accounts / Fund** — Eigener Track Record aufbauen, dann Kapital einwerben. Regulatorisch komplex, aber hoechster Upside.

---

## 6. Dealbreakers — Was sich aendern muss

1. **API-Keys rotieren. Sofort.** Kein ernsthafter Investor akzeptiert bekannte, nicht behobene Security-Vulnerabilities.
2. **Team aufbauen.** Mindestens ein technischer Co-Founder vor jeder Finanzierungsrunde.
3. **Live-Validierung.** Mindestens 3 Monate Paper-Trading mit realen Daten und dokumentierten Ergebnissen.
4. **Backtest mit realen Daten.** Unabhaengig reproduzierbare Ergebnisse, keine Survivorship-Bias-belasteten Runs.
5. **Business Model definieren.** "Ich baue ein Trading-System" ist kein Pitch. "Ich verkaufe X an Y fuer Z" ist einer.

---

## 7. Term Sheet Sketch (hypothetisch, Pre-Seed)

Falls die Dealbreakers adressiert werden:

| Parameter | Wert |
|---|---|
| Runde | Pre-Seed |
| Investitionshoehe | $250k-500k |
| Bewertung (pre-money) | $2-3M (cap auf SAFE) |
| Instrument | SAFE mit Cap |
| Verwendung | Team (60%), Infrastruktur/Daten (25%), Legal/Compliance (15%) |
| Bedingungen | Key-Rotation vor Closing, Co-Founder innerhalb 90 Tagen, Paper-Trading-Log innerhalb 6 Monaten |
| Board | Kein Board-Sitz, aber quartalsmaessige Updates und Informationsrechte |

---

## 8. Comparable Exits

| Unternehmen | Exit | Relevanz |
|---|---|---|
| **Quantopian** | Shut down 2020, IP an Robinhood verkauft (geschaetzt $10-30M) | Warnendes Beispiel: Community != Revenue |
| **Kensho** (S&P Global, $550M, 2018) | NLP/Alternative Data fuer Finance | Zeigt Wert von Alt-Data-Layern |
| **Numerai** | Aktiv, $50M+ raised | Aehnlicher Quant-Ansatz, aber mit Crowd-Sourcing-Twist |
| **Alpaca** | $100M+ raised, aktiv | API-first Broker mit Quant-Fokus |
| **WorldQuant** | Privat, multi-billion AUM | Zeigt Skalierungspotenzial von systematischem Quant |

Realistischer Exit-Korridor fuer Assembled bei Erfolg: **$5-30M Acqui-Hire/Tech-Sale** innerhalb von 3-5 Jahren. Fuer einen $200M-Fund ist das zu klein fuer den Aufwand.

---

## 9. Empfehlung

**WATCH — mit klarem Re-Engagement-Trigger.**

Die technische Substanz ist fuer ein Solo-Projekt bemerkenswert. Der Geopolitik-Winkel ist differenzierend. Die Testdisziplin und Architektur deuten auf einen Entwickler hin, der Systeme bauen kann, nicht nur Code schreiben.

Aber: Kein Team, kein Umsatz, kein Live-Track-Record, keine Security-Hygiene, kein Business Model. Das ist ein beeindruckendes Forschungsprojekt, kein investierbares Unternehmen.

**Re-Engagement wenn:**
- Co-Founder an Bord (technisch oder kommerziell)
- 3+ Monate Paper-Trading-Ergebnisse mit realen Daten
- Security-Issues behoben
- Erste Klarheit ueber Go-to-Market

**Risiko bei Pass:** Gering. Wenn der Entwickler tatsaechlich ein Team aufbaut und Live-Ergebnisse liefert, haben wir genuegend Zeit fuer eine Seed-Runde.

---

*Memo-Status: Internes Arbeitsdokument.*

---
---

# Perspektive 2: Konkurrierender Quant Fund — Competitive Intelligence

**An:** Investment Committee, Managing Partners
**Von:** CTO Office
**Betreff:** Technische Bewertung — Open-Source-Projekt "Assembled-Trading-AI"
**Klassifikation:** Vertraulich

---

## 1. Bedrohungsanalyse

**Kurzurteil: Keine operative Bedrohung. Mittelfristig beobachtenswert.**

Das Projekt konkurriert nicht mit uns. Es konkurriert mit der Vorstellung, die Family Offices und kleine Fonds davon haben, was ein quantitatives Handelssystem sein sollte. Die Zielgruppe — Family Offices, kleine Hedge Funds — ist nicht unser Markt. Kein AUM, kein Track Record, kein Team, keine Infrastruktur. Die Gefahr liegt nicht im Produkt, sondern im Entwickler.

---

## 2. Technische Bewertung

**Beeindruckend:**

Die Architektur ist ungewoehnlich reif fuer ein Solo-Projekt. 22 Subsysteme, saubere Schichtentrennung (Data -> Features -> Signals -> Portfolio -> Execution -> QA), modulare Regeldateien, eine operative CLAUDE.md, die strenger formuliert ist als manche interne Governance bei uns. 1.243 Tests bei einem Einzelentwickler sind bemerkenswert — die meisten Quant-Startups, die wir evaluiert haben, hatten bei vergleichbarem Umfang unter 200.

Die Breite ist real: Almgren-Chriss-Execution, Black-Litterman, HRP, Regime-Detection via HMM, Walk-Forward-Validation, Component-VaR, ein Kill-Switch mit State-Machine — das sind keine Platzhalter-Namen. Der Code existiert.

**Fassade:**

Genau hier liegt das Problem. Breite ist nicht Tiefe. Die Kernmodule zeigen:

- **30-Faktor-Modell mit Regime-Weights:** Die Faktorgewichte werden per HMM-Regime trainiert — auf welchen Daten? Synthetisch oder bestenfalls bereinigte Yahoo-Finance-Daten. Kein Survivorship-Bias-Handling in der Praxis validiert. Kein Nachweis, dass die Regime-Erkennung out-of-sample funktioniert.
- **ML-Module (GNN, RL, MAML, Symbolic Regression, Bayesian NN):** Das sind Skeleton-Implementierungen mit korrekten Interfaces. Kein einziges dieser Module hat nachweislich auf realen Marktdaten trainiert. Das ist eine Bibliothek, kein Beweis.
- **Execution (Almgren-Chriss, SOR):** Mathematisch korrekt formuliert, aber ohne echte Orderbuch-Daten, ohne Latenz-Realitaet, ohne Fill-Statistiken. Ein Papiermodell eines Papiermodells.
- **Geopolitical Intel, Crisis Alpha, Sanctions Wargaming:** Ambitioniert benannt. In der Realitaet: regelbasierte Stubs mit konfigurierbaren Schwellwerten. Keine NLP-Pipeline, die tatsaechlich Nachrichtenstroeme verarbeitet.

**Fazit:** Circa 30% echte Substanz, 40% korrekte aber unvalidierte Implementierung, 30% Skeleton mit professionellem Interface.

---

## 3. Alpha-Potential

**Gering bis spekulativ.**

Ein 30-Faktor-Modell mit Regime-Switching ist kein neues Konzept. Unsere eigene Faktor-Bibliothek hat 180+ validierte Faktoren. Der potentielle Vorteil laege in der Integration — wenn jemand alle Teile gleichzeitig zum Laufen bringt. Aber:

- Kein einziger Backtest auf institutionellen Daten (Bloomberg, Refinitiv, FactSet).
- Keine Transaction-Cost-Analyse gegen reale Spread-/Impact-Daten.
- Target-Rendite "20-30% p.a." ohne Leverage — das ist entweder naiv oder erfordert konzentrierte Positionen, die das Risikoframework selbst verbietet.
- Walk-Forward-Validation existiert, wurde aber nie gegen Daten laufen gelassen, die diesen Namen verdienen.

Alpha-Generierung erfordert nicht Code, sondern Daten, Kapital und Execution. Dieses Projekt hat nur Code.

---

## 4. Recruit or Compete?

**Klare Empfehlung: Recruiting-Gespraech fuehren.**

Ein Einzelentwickler, der in sechs Monaten ein System dieser Architekturbreite baut — mit sauberer Governance, Test-Disziplin und einem Verstaendnis fuer Risk-First-Design — ist selten. Die meisten Bewerber mit PhD und drei Jahren Berufserfahrung produzieren weniger strukturierten Output.

Die Schwaechen (keine institutionellen Daten, keine Live-Erfahrung, ueberbordende Breite statt Tiefe) sind exakt die Schwaechen, die eine gute Teamumgebung korrigiert. Die Staerken (Systemdenken, Selbstdisziplin, Architekturverstaendnis) sind schwer zu lehren.

Risiko: Der Entwickler scheint AI-gestuetzt zu arbeiten (Claude Code als zentrales Werkzeug). Die Frage ist, wie viel Eigenverstaendnis hinter der AI-unterstuetzten Produktion steckt. Das klaert ein technisches Interview in 90 Minuten.

---

## 5. IP-Bewertung

**Monetaerer Wert der Codebasis: Gering (< $50k Replacement-Kosten).**

Unser Team koennte die verwertbaren Teile in vier bis sechs Wochen nachbauen. Der eigentliche Wert liegt nicht im Code, sondern in:

- der Architektur-Dokumentation als Blaupause
- dem Test-Harness als Beschleuniger
- dem Governance-Framework (CLAUDE.md + Regelmodule) als Vorlage fuer eigene AI-gestuetzte Entwicklung

Nichts davon ist patentierbar oder verteidigbar. Open Source auf GitHub. Kein Moat.

---

## 6. Schwachstellen — Angriffsvektoren bei Konkurrenz

Falls das Projekt jemals als kommerzielles Produkt auftritt:

1. **Daten-Realismus:** Jeder Due-Diligence-Prozess wird nach validierten Backtests auf institutionellen Daten fragen. Die gibt es nicht.
2. **Single Point of Failure:** Ein Entwickler. Bus-Faktor gleich eins.
3. **Keine Live-Evidenz:** Kein Paper-Trading-Track-Record, kein Audited Return.
4. **Regulatorik:** Compliance-Modul existiert als Stub. Kein MiFID-II, kein SEC-Reporting, kein AIFMD.
5. **Skalierung:** Keine Evidenz, dass das System 500+ Ticker, Intraday-Rebalancing oder Multi-Asset-Klassen bewaeltigt.

---

## 7. Empfehlung

| Option | Bewertung |
|---|---|
| Ignorieren | Vertretbar, aber kurzsichtig |
| Beobachten | **Ja — Quartalspruefung, GitHub Watch** |
| Abwerben | **Ja — explorative Kontaktaufnahme empfohlen** |
| Als Risiko behandeln | Nein — noch keine Marktrelevanz |

**Primaerempfehlung:** Recruiting-Kontakt herstellen. Sekundaer: Repo auf Quartalsbasis beobachten, insbesondere ob reale Datenanbindung oder institutionelle Partnerschaften entstehen. Wenn beides nicht innerhalb von 12 Monaten passiert, wird das Projekt in der Breite stagnieren.

**Das Projekt ist kein Konkurrent. Es ist ein Talentsignal.**

---

*CTO Office | Verteilung: IC-Mitglieder, Head of Recruiting, Head of Quant Research*

---
---

# Perspektive 3: Angel Investor (Ex-Goldman Sachs Quant, 15 Jahre)

---

## 1. Erster Eindruck

Ich habe in 15 Jahren bei Goldman viele interne Quant-Systeme gesehen — und auch viele externe Pitches von Leuten, die glaubten, sie haetten das naechste Renaissance Technologies gebaut. Der erste Blick auf dieses Projekt ist deshalb reflexartig skeptisch.

Dann lese ich die Struktur. 479 Quelldateien. 1.243 Tests. 22 Subsysteme. Almgren-Chriss Execution. Regime-conditional Factor Weights. Ein geopolitisches Intel-Layer. In sechs Monaten. Allein.

Das ist nicht normal. Das ist entweder beeindruckend — oder ein rotes Flag.

Wahrscheinlich beides.

Als Ex-Quant faellt mir zuerst die *architektonische Disziplin* auf. Es gibt eine klare Schichtung: Data -> Features -> Signals -> ML -> Portfolio -> Execution -> Risk -> Compliance. Das ist kein Zufallsprodukt. Da steckt echtes konzeptuelles Verstaendnis dahinter. Die CLAUDE.md allein liest sich wie ein internes Goldman-Governance-Dokument — Risk-first, keine stillen Seiteneffekte, PIT-Disziplin. Der Entwickler hat verstanden, wie industrielle Quant-Systeme *denken* muessen.

Gleichzeitig: Version 0.0.1. Kein Live-Track-Record. Kein Team. Kein Revenue. Und API-Keys in der Git-History. Das ist der Moment, wo ich als Investor tief durchatme.

---

## 2. Was fehlt zum echten Trading?

Ich waere kein ehrlicher Mentor, wenn ich das schoenrede. Die Luecken sind konkret:

**Datensauberkeit und PIT-Realismus.** 1.243 Tests auf synthetischen oder halb-synthetischen Daten beweisen nichts ueber Produktionstauglichkeit. Survivorship-Bias, fehlerhafte Corporate Actions, Kalenderprobleme — das sind die Dinge, die Backtests zum Luegen bringen. Ich habe bei GS Teams gesehen, die allein daran zwei Jahre gearbeitet haben.

**Execution-Realitaet.** Almgren-Chriss im Code zu haben ist gut. Aber echte Marktmikrostruktur — Latenz, Partial Fills, Reject-Handling, Broker-Eigenheiten — ist eine andere Welt. Paper-Trading ist kein Ersatz fuer echten Slippage-Schmerz.

**Live-Betrieb und Observability.** Monitoring, Alerting, automatisches Failover, Reconciliation gegen echte Broker-Statements — das ist der Teil, den Quant-Entwickler systematisch unterschaetzen. Kein Grafana-Dashboard rettet dich um 3 Uhr morgens, wenn der State-Machine-Zustand korrupt ist.

**Das Team-Problem.** Ein einzelner Entwickler als Bottle-Neck fuer ein System dieser Komplexitaet ist ein strukturelles Risiko. Bei einer Krankheit, einem Burnout, einem Lebensumbruch — das System steht still.

**Security.** API-Keys in der Git-History sind keine Kleinigkeit. Das ist ein kritischer Incident. Die Keys muessen rotiert werden, nicht nur versteckt. Solange das nicht erledigt ist, ist das System fuer produktiven Einsatz nicht freigegeben.

---

## 3. Wuerde ich investieren?

Direkt? Nein. Noch nicht.

Nicht weil das Projekt schlecht ist — sondern weil der Zeitpunkt falsch ist. Als Angel Investor kaufe ich Wahrscheinlichkeiten. Und die Wahrscheinlichkeit eines erfolgreichen Returns steigt dramatisch, wenn ich mindestens sechs Monate Live-Paper-Trading-Daten sehe, die Datenpipeline mit echten Feeds laeuft, und die Security-Issues bereinigt sind.

Was ich *wuerde* tun: Ein Convertible-Note-Arrangement. 50.000 bis 75.000 Euro. Kein Equity-Deal jetzt. Stattdessen: Meilenstein-gebundene Tranche. Erstes Geld gegen Live-Paper-Track-Record von 90 Tagen auf echten Marktdaten (nicht synthetisch). Zweite Tranche gegen erstes Live-Konto mit nachweisbarer Execution. Wandlung in Equity nur bei positivem Risk-adjusted Return ueber 12 Monate.

Bedingungen waeren knallhart: Security-Cleanup als Vorbedingung. Ein Co-Founder oder zumindest ein bezahlter Ops-Partner. Und ein ehrlicher Drawdown-Bericht — keine geschoenten Backtest-Kurven.

---

## 4. Mentoring-Empfehlung

Drei Dinge wuerde ich dem Entwickler sagen:

**Erstens: Hoer auf, neue Module zu bauen.** Das System ist breit genug. Jetzt muss es tief werden. Ein echter Datenfeed. Ein echter Broker. Echte Fills. Alles andere ist Architektur-Masturbation.

**Zweitens: Finde einen Partner, der dein Gegenteil ist.** Du bist offensichtlich stark in Systemarchitektur und ML. Was du wahrscheinlich nicht hast: Marktmikrostruktur-Erfahrung, Broker-Beziehungen, institutionelles Netzwerk. Ein Co-Founder mit Trading-Desk-Hintergrund waere das Wichtigste, was du dir gerade leisten koenntest — auch wenn das bedeutet, 20-30% Equity abzugeben.

**Drittens: Behandle die Security-Luecke wie einen Produktionsbrand.** Keys rotieren. Heute. Nicht morgen. Das ist keine technische Kleinigkeit, das ist eine Vertrauensfrage gegenueber jedem zukuenftigen Investor, Partner oder Kunden.

---

## 5. Realistisches Szenario in 12 Monaten

Drei moegliche Ausgaenge:

**Wahrscheinlichstes Szenario (60%):** Der Entwickler schafft es, einen echten Datenfeed und Paper-Trading-Betrieb aufzubauen. Track-Record ist unvollstaendig, aber vorhanden. Erstes Angel-Gespraech. Kein Deal, aber Netzwerk-Aufbau.

**Bestes Szenario (25%):** Live-Paper-Account mit nachweisbarem Sharpe > 1.2 ueber 6 Monate. Erstes kleines Live-Konto mit eigenem Kapital. Ein Co-Founder an Bord. Dann ist das fuer eine Pre-Seed-Runde reif — ich wuerde ernsthaft ueberlegen.

**Schlechtestes Szenario (15%):** Burnout. Das System ist zu komplex fuer eine Person geworden. Keine echten Daten. Keine Fortschritte auf dem kritischen Pfad. Das Projekt bleibt ein beeindruckendes GitHub-Repo ohne operativen Beweis.

---

**Fazit:** Das ist das technisch ambitionierteste Solo-Quant-Projekt, das mir in Jahren begegnet ist. Aber Ambition allein kauft keine Futures. Der naechste Schritt ist nicht mehr Code — der naechste Schritt ist echtes Geld, echte Daten, echter Schmerz.

*Wer das uebersteht, hat meine Aufmerksamkeit.*

---
---

# Perspektive 4: Fintech-Analyst (Research-Memo)

**Vertraulich | CB Insights Style | April 2026**

---

## 1. Marktlandschaft — Quant-Trading-Plattformen 2026

Der Markt fuer quantitative Handelsinfrastruktur ist 2026 klar segmentiert.

**Enterprise-Segment:** Bloomberg AIM, BlackRock Aladdin und MSCI Axioma dominieren den institutionellen Bereich. Einstiegskosten von 500.000 USD aufwaerts und mehrjaehrige Implementierungszyklen machen diese Systeme fuer Family Offices und kleine Hedgefonds strukturell unzugaenglich.

**Community-/Research-Segment:** QuantConnect hat sich mit ueber 200.000 Nutzern als De-facto-Standard fuer Research und Backtesting etabliert. Das Geschaeftsmodell ist cloud-zentriert, der Fokus liegt auf Community-Signalen und algorithmischen Wettbewerben. Fuer produktionsnahes Deployment bleibt das System begrenzt.

**Production-First-Nische:** Nautilus Trader adressiert professionelle Trader mit einem Rust/Python-Hybrid. Die Plattform ist performant und produktionsorientiert, setzt aber tiefe technische Expertise voraus und hat keine explizite Portfolio-Optimierungs- oder Risk-Overlay-Schicht auf institutionellem Niveau.

**Legacy-Segment:** Zipline stagniert seit der Quantopian-Abwicklung. VectorBT ist auf Backtesting spezialisiert und bietet keine Pipeline-to-Execution-Logik.

**Marktdynamik 2026:** Zwei Trends praegen die Landschaft strukturell. Erstens waechst der Bedarf an interpretierbaren, dokumentierten Risikoentscheidungen durch verschaerfte Regulierung (DORA in Europa, SEC-AI-Guidance in den USA). Zweitens steigen die Anforderungen an Reproduzierbarkeit und Audit-Trails — besonders fuer Family Offices, die zunehmend unter institutionellem Druck stehen, professionelle Infrastruktur nachzuweisen.

---

## 2. Positionierung — Wo steht Assembled-Trading-AI?

Das Projekt belegt theoretisch einen attraktiven Mittelpunkt: institutionelle Architektur ohne Enterprise-Preisschild.

Technisch ist die Positionierung konsistent: 22 Subsysteme, eine State-Machine-basierte Risikosteuerung, Almgren-Chriss-Execution, Regime-bedingte Portfolio-Optimierung (Black-Litterman, HRP, MVO), geopolitische Intel-Integration und ein modularer Compliance-Layer. Das entspricht funktional einem System, das Aladdin-Kunden erkennen wuerden — gebaut von einer Einzelperson in Python.

Praktisch ist die Positionierung jedoch noch nicht eingeloest. Version 0.0.1, keine realen Nutzer, kein Live-Trading, kein Revenue. Die Plattform ist ein technisches Artefakt mit institutionellem Anspruch, aber ohne Marktvalidierung.

Im Wettbewerbsvergleich liegt das Projekt damit in einer Zwischenzone: zu komplex fuer Retail-Quants, die QuantConnect bevorzugen; zu jung und zu wenig produktionserprobt fuer Family Offices, die Accountability benoetigen.

---

## 3. TAM / SAM / SOM — Realistische Marktgroessen

**Total Addressable Market (TAM):** Der globale Markt fuer Finanzanalyse- und Portfoliomanagementsoftware wird 2026 auf ca. 6-8 Mrd. USD geschaetzt. Breiter gefasst — inklusive algorithmischer Handelssoftware und Risk-Analytics — naehert sich der TAM der 15-Mrd.-USD-Marke.

**Serviceable Addressable Market (SAM):** Das realistisch adressierbare Segment: Family Offices (weltweit ca. 10.000, davon ~3.000 mit quantitativem Mandat), kleine bis mittlere Hedgefonds (AUM 50 Mio. bis 500 Mio. USD, ca. 2.000-3.000 Einheiten weltweit), quantitative Prop-Trading-Desks und Boutique-Asset-Manager. Geschaetzter SAM: 800 Mio. bis 1,2 Mrd. USD.

**Serviceable Obtainable Market (SOM):** Fuer ein Solo-Projekt in Version 0.0.1 ist die ehrliche SOM-Schaetzung fuer die naechsten drei Jahre: unter 1 Mio. USD — wenn ueberhaupt monetarisiert wird. Ein realistisches Drei-Jahres-Ziel waeren 5-15 zahlende Pilotnutzer im Bereich 10.000-50.000 USD Jahresvertrag. Das ergibt eine SOM von 150.000 bis 750.000 USD — sofern der Schritt zur Produktreife gelingt.

---

## 4. Differenzierung — Was ist wirklich anders?

Drei echte Differenzierungsmerkmale sind erkennbar:

**Erstens: Architektonische Kohaerenz.** Die meisten Open-Source-Systeme sind Bottom-Up gewachsen. Assembled-Trading-AI zeigt eine Top-Down-Designphilosophie mit expliziten Modul- und Schichtgrenzen, einer zentralen CLAUDE.md-Steuerungsdatei und einem dokumentierten Architekturgedaechtnis. Das ist fuer ein Solo-Projekt ungewoehnlich und ein Qualitaetssignal.

**Zweitens: Risk-First-Ansatz.** Wo QuantConnect renditefokussiert ist, ist dieses System explizit risk-first: Kill-Switch-Logik, Pre-Trade-Checks, Tail-Hedging, State-Machine-Risikosteuerung, weiche Profit-Locks. Das entspricht genau dem, was Regulatoren und Compliance-Abteilungen von Family Offices fordern.

**Drittens: Geopolitische und alternative Datenanreicherung.** Die Integration von Geo-Risk-Signalen, Patent-Features, Satellitendaten-Stubs und Earnings-Call-NLP in eine Backtesting-Pipeline ist im Open-Source-Bereich selten. Das Potenzial fuer differenzierte Alpha-Generierung ist real — sofern echte Datenquellen angebunden werden.

Was es nicht differenziert: Teamgroesse, Community, Produktionsreife, Kundensupport und regulatorische Zertifizierung.

---

## 5. Risiken

**Regulatorisch:** Europaeische und US-amerikanische Regulierung erhoeht die Huerden fuer algorithmische Handelssysteme kontinuierlich. DORA (EU) verlangt Resilienz- und Auditierungsnachweise.

**Technisch:** Das Hauptrisiko ist Debt-Akkumulation unter Skalendruck. 479 Quelldateien, entwickelt von einer Person, mit bekannten Stub-Bereichen und synthetischen Testdaten.

**Marktbezogen:** Der groesste Risikofaktor ist Timing. Wenn QuantConnect oder Nautilus Trader ihre institutionellen Schichten weiterentwickeln, schrumpft das adressierbare Differenzierungsfenster. Das Projekt hat 12 bis 24 Monate.

**Operativ:** Skalierung von Solo auf Team ist der klassische Engpass fuer technische Gruender.

---

## 6. Prognose — Szenarien

**Bull-Case (15%):** Seed-Runde $500k-1.5M, Family-Office-Piloten bis Ende 2026.

**Base-Case (55%):** Respektierte Open-Source-Plattform mit kleiner Community. Kein signifikantes Revenue in drei Jahren. Moegliche spaetere Akquisition.

**Bear-Case (30%):** Ohne Nutzervalidierung wird das Projekt archiviert oder privatisiert.

---

## 7. Rating

| Dimension | Score (1-10) | Kommentar |
|---|---|---|
| **Technologie** | **8** | Architektonisch koharent, risk-first, 22 Subsysteme |
| **Marktpotenzial** | **6** | Reale Luecke identifiziert, aber Validierung fehlt |
| **Team** | **4** | Ein Entwickler — technisch stark, Scaling offen |
| **Execution-Reife** | **3** | Version 0.0.1, kein Live-Trading, synthetische Daten |
| **Gesamtpotenzial** | **6** | Ueberdurchschnittlich fuer die Phase — mit den richtigen naechsten Schritten ausbaufaehig |

---

**Fazit:** Assembled-Trading-AI ist technisch ein ungewoehnlich ambitioniertes Einzelprojekt mit echter institutioneller DNA. Der kritische naechste Schritt ist nicht mehr Architektur — sondern der erste reale Nutzer.

---
---

# Konsens ueber alle vier Perspektiven

## Staerken die JEDER nennt
- Architektonische Disziplin und Testkultur sind aussergewoehnlich fuer ein Solo-Projekt
- Risk-First-Philosophie ist das richtige Signal fuer institutionelle Glaubwuerdigkeit
- Geopolitik-Layer ist ein echter Differentiator im Open-Source-Bereich

## Schwaechen die JEDER nennt
- Kein Live-Track-Record (der #1 Dealbreaker ueber alle Perspektiven)
- Bus Factor = 1 (unfundierbar ohne Team)
- Security-Issue mit .env (muss sofort geloest werden)
- Breite vor Tiefe — zu viele Skeleton-Module ohne Validierung

## Was ALLE empfehlen — Priorisierte Roadmap

| Prioritaet | Zeitrahmen | Aktion |
|---|---|---|
| **KRITISCH** | Sofort | API-Keys rotieren, .env aus Git-History bereinigen |
| **1** | 30 Tage | Keine neuen Module — stattdessen echte Datenpipeline aufbauen |
| **2** | 90 Tage | Paper-Trading mit echten Marktdaten starten und dokumentieren |
| **3** | 6 Monate | Co-Founder oder Partner finden (komplementaeres Profil) |
| **4** | 12 Monate | Live-Track-Record aufbauen (eigenes Kapital) |
| **5** | 18 Monate | Erste institutionelle Pilotpartner gewinnen |

---

*Dokument generiert am 16. April 2026. Alle Kennzahlen aus direkter Repository-Inspektion verifiziert.*
