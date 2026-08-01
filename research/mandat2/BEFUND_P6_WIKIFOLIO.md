# P6 wikifolio-Erhebung (2026-08-01)

300 wikifolios, gleichmäßig über den offiziellen Sitemap (34.647 Einträge)
gestreut, 0 Fehler. Rohdaten: `results/wikifolio_stichprobe.json`.
Erhebungscode: `wikifolio_scrape.py`.

**Rahmen:** robots.txt geprüft — wikifolio erlaubt `User-agent: *` alles außer
`/search` und Tracking-URLs; die Seiten stehen im offiziellen Sitemap. Rate-Limit
1,5 s, sequenziell, identifizierender User-Agent. **Keine personenbezogenen
Daten gespeichert** (die Seiten liefern Klarnamen mit — nicht übernommen).

**eToro wurde NICHT erhoben.** Dessen robots.txt sperrt für alle Agents
ausdrücklich `/portfolio`, `/portfolio/*`, `*/api/*`, `*/sapi*` — also genau
die Trader-Daten.

---

## Das Kernergebnis

| | Performance p. a. |
|---|---|
| **Median über 276 wikifolios mit Angabe** | **4,82 %** |
| Mittel | 4,44 % |
| Anteil über ~9 % p. a. (grob SPY-Niveau) | **26,4 %** |
| Anteil negativ | 22,8 % |

**Der typische sichtbare wikifolio liefert etwa die Hälfte der Indexrendite** —
und das ist die *optimistische* Zahl: eingestellte und gescheiterte wikifolios
verschwinden aus dem Sitemap, die Verlierer sind also gar nicht in der
Stichprobe. Nur gut ein Viertel erreicht überhaupt Indexniveau.

Dazu die Kostenstruktur: **10,0 % Performance-Fee (Median) plus 0,95 % p. a.
laufende Gebühr.** Wer über einen wikifolio investiert, zahlt das zusätzlich.

## Stil-Auswertung (schwach, aber eindeutig in eine Richtung)

| Stil | n | Median p. a. | Beta SPX |
|---|---|---|---|
| krypto | 4 | 16,59 % | 1,15 |
| langfrist | 6 | 7,94 % | 0,51 |
| value | 10 | 6,09 % | 0,57 |
| dividende | 15 | 4,50 % | 0,26 |
| trend | 21 | 3,92 % | 0,46 |
| wachstum | 9 | 2,73 % | 0,84 |
| hebel | 4 | 0,17 % | 0,20 |
| kurzfrist | 2 | −0,85 % | 0,05 |

Und deutlicher, weil auf harten Feldern statt auf Stichworten:

| | n | Median p. a. |
|---|---|---|
| **ohne** Hebelprodukte | 226 | **5,65 %** |
| **mit** Hebelprodukten | 74 | **1,62 %** |

## Was ich daraus ableite — und was nicht

**Belastbar genug für eine Aussage:** Die wikifolio-Welt als Ganzes schlägt den
Index nicht. Das stützt den Mandat-I-Befund, dass der wikifolio-*Wrapper* keine
Antwort ist — jetzt mit eigenen Zahlen statt aus zweiter Hand. Und: gehebelte
wikifolios liefern median 1,62 % gegen 5,65 % — Hebel schadet auch hier, was
zu P2-Befund C passt (Hebel verschlechterte dort systematisch das
Risikoprofil).

**Nicht belastbar:** die Stil-Tabelle. Die Stichworte stammen aus
Kurzbeschreibungen, treffen nur 2–21 von 300, und „krypto mit 16,6 %" bei n=4
ist eine Anekdote, keine Zahl. Die Reihenfolge (langfristig > kurzfristig,
ohne Hebel > mit Hebel) ist plausibel und deckt sich mit unseren eigenen
Befunden — aber sie **bestätigt** sie nicht, sie **stimmt mit ihnen überein**.
Das ist ein Unterschied.

**Als Ideenquelle mager.** Ich habe keine Strategie gefunden, die wir nicht
schon getestet hätten. Der Erhebungsweg funktioniert und ist wiederholbar; das
ist das eigentliche Ergebnis dieses Strangs.

## Eigener Fehler, genannt

Der erste Parser-Anlauf las die Performance-Kennzahlen aus den falschen
Schlüsseln (`item["label"]` statt `item["ranking"]["label"]`) und lieferte
leere Dicts. Ich hätte daraus fast geschlossen, die Daten seien nicht im
Payload. Sie waren es — 276 von 300 Einträgen. Behoben und neu erhoben.
