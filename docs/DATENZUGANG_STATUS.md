# Datenzugang — operativer Status

**Letzte Aktualisierung:** 2026-08-15
**Zweck:** Eine einzige autoritative Stelle unter `docs/`, die beantwortet: *Welche Datenquelle
liefert gerade, welche nicht, und was hängt daran?*

Dieses Dokument existiert, weil der EODHD-Zugangsverlust vom 2026-08-05 über zehn Tage lang
**ausschließlich** in einem Forschungsdokument (`research/mandat2/ABSCHLUSS.md`) vermerkt war,
während der produktive Cache-Writer `scripts/ops/refresh_daily_cache_from_eodhd.py` — die einzige
Datei im `src`-/`scripts`-Baum, die den Endpunkt überhaupt anspricht — und sämtliche
`research/mandat/pull_*.py` weiter davon abhingen. Das ist die Doku-Drift-Klasse **E-140**
(Registrierungstext beschreibt einen anderen Lauf als den gelaufenen) in ihrer operativen
Variante: der Betriebszustand stand nicht dort, wo der Betrieb nachschaut.

---

## 1. Status-Tabelle

| Quelle | Status | Seit | Belegt durch | Was daran hängt |
|---|---|---|---|---|
| **EODHD** | 🔴 **AUSGEFALLEN** | 2026-08-05 | API-Probe 2026-08-15: `/eod` → **401** für `AAPL.US` *und* `LEH.US`; Options-Marketplace → **403**; `/user` antwortet noch (`subscriptionType: monthly`, `dailyRateLimit: 20`, `apiRequestsDate: 2026-08-06`) | `scripts/ops/refresh_daily_cache_from_eodhd.py` (Primär-Writer für `output/aggregates/daily.parquet`), sämtliche `research/mandat/pull_*.py` |
| **yfinance** | 🟡 unverändert verfügbar | — | Live-Fallback in `scripts/run_live_paper.py:238-250` | Live-Preise, wenn Cache > 3 Tage alt |
| **Alpaca (Paper)** | 🟢 verfügbar | — | `src/assembled_core/execution/broker_adapter.py` | Konten, Positionen, Orders, Dividenden-Activities |
| **SEC EDGAR / DERA** | 🟢 verfügbar (frei) | — | `src/assembled_core/data/edgar_form4_ingest.py`, `fundamentals_xbrl_ingest.py` | Form 4, XBRL-Fundamentals |
| **Ken French / CRSP-Marktreihe** | 🟢 verfügbar (frei) | — | `research/mandat2/pull_gratis_quellen.py` | H-087, H-089 |
| **FRED / CBOE / BLS / World Bank** | 🟢 verfügbar (frei) | — | `src/assembled_core/data/sources/` | Makro-Faktoren |
| **Polymarket / Kalshi** | 🟢 verfügbar (frei) | — | `src/assembled_core/data/sources/polymarket_source.py` | GeoRisk-Exposure-Overlay |

---

## 2. Konsequenz des EODHD-Ausfalls

### 2.1 Der operative Preis-Cache ist eingefroren

`output/aggregates/daily.parquet` endet am **2026-08-05** — exakt am Tag des Zugangsverlusts.
Gemessen 2026-08-15: 279.013 Zeilen, 220 Symbole, letzter Bar 2026-08-05.

Der Live-Pfad bricht dadurch **nicht** hart, sondern degradiert wie vorgesehen:
`scripts/run_live_paper.py::_load_prices` erkennt den veralteten Cache (Schwelle 3 Tage) und fällt
auf yfinance zurück. Der fail-closed-Block darunter — im Quelltext markiert mit
`# --- Final fallback: BLOCK on stale cache (F-RX-7 §9.12 (e)) ---` — greift erst, wenn
*zusätzlich* yfinance ausfällt. Der Pilot handelt also weiter — auf yfinance-Preisen, nicht auf
dem bezahlten Feed.

(Verweise auf diese Funktion bewusst über Funktions- und Markernamen statt Zeilennummern: die
Datei wird häufig geändert, und ein veralteter Zeilenverweis schickt den Leser auf den falschen
Guard — was beim Erstentwurf dieses Dokuments prompt passiert ist.)

### 2.2 Was dadurch nicht mehr beschaffbar ist

- **Delisting-Kurse für 1995–2016.** Für den EOD-Endpunkt war Delisted-Coverage grundsätzlich
  verifiziert (SIVB bis 2023-03-09, BBBY), **für das Suchfenster aber nie geprüft** — und ist es
  jetzt nicht mehr. Siehe `research/mandat2/ABSCHLUSS.md` (§ „Was offen bleibt").
- **Jede Erweiterung von `research/mandat/data/prices_verdict.parquet`.** Das PIT-Panel ist am
  **2026-07-06** eingefroren.
- **Jede Erweiterung von `data/raw/intraday_1h/`** (298 Symbole, letzter Bar 2026-07-02).

### 2.3 Was *nicht* betroffen ist

Alle bereits gezogenen lokalen Bestände bleiben vollständig nutzbar — insbesondere das
PIT-Preispanel (1.167 Symbole, 1995-01-03 … 2026-07-06), das SEC-Form-4-Archiv und die
Fama-French-/CRSP-Reihe ab 1926. **Forschung auf historischen Fenstern ist nicht blockiert.**
Blockiert ist ausschließlich das *Nachziehen* neuer Daten.

---

## 3. Widersprüchliche Altdokumentation

Folgende Dokumente beschreiben einen Zugangsstand, der nicht mehr gilt. Sie werden hier bewusst
benannt statt still korrigiert, damit die Drift nachvollziehbar bleibt:

| Dokument | Behauptung | Realität |
|---|---|---|
| `autonome_weiterarbeit/20_PAID_DATEN.md` §20.1 | EODHD als „**MUST** für ernsthafte Backtests" | gekauft, genutzt, seit 2026-08-05 ausgefallen |
| `docs/DATA_PROVIDERS_COMPARISON.md:238` | Empfehlung „Twelve Data Free-Tier" | von der EODHD-Entscheidung überholt; kein Twelve-Data-Fetchmodul existiert im `src`-Baum |
| `docs/DOWNLOAD_STRATEGY.md` | yfinance-Rate-Limit-Workarounds als Hauptstrategie | beschreibt die Ära vor EODHD |
| `docs/MISSING_SYMBOLS_LIST.md` | Twelve Data als aktiver Provider (Stand 2025-12-09) | überholt |

---

## 4. Nächste Schritte (Entscheidung offen, nicht von Claude zu treffen)

1. **EODHD reaktivieren** (~20 €/Monat) — stellt den Status quo wieder her. Ohne das bleibt der
   bezahlte Feed tot und der Pilot auf yfinance.
2. **Norgate Data Platinum** (~52 $/Monat) — würde zusätzlich die Delisting-Lücke schließen
   (25.222+ delistete US-Ticker seit 1950 plus historische Index-Memberships) und den
   nicht-reproduzierbaren GitHub-Bezug von `sp500_historical_constituents.csv` ersetzen.
   Status: erwogen, nicht beschafft — siehe `KNOWN_ISSUES.md` §0.1.
3. **Nichts tun** — legitim, solange nur auf historischen Fenstern geforscht wird. Dann muss
   aber der Live-Pilot bewusst als „läuft auf yfinance" geführt werden.

---

## 5. Wie dieser Status geprüft wird

Der Zugang ist mit einem einzigen Aufruf verifizierbar:

```
GET https://eodhd.com/api/eod/AAPL.US?api_token=<TOKEN>&fmt=json&from=2026-08-10
```

- **200 + Bars** → Zugang steht, dieses Dokument ist veraltet.
- **401** → Zugang weg (aktueller Stand).

Ein Kontrollsymbol ist Pflicht: eine Probe *nur* auf einem delisteten Ticker unterscheidet nicht
zwischen „Zugang tot" und „Symbol nicht abgedeckt" (Anti-Pattern **E-113**).

---

## Verweise

- `research/mandat2/ABSCHLUSS.md` — Erstfeststellung des Zugangsverlusts
- `research/mandat2/BEFUND_DATENQUALITAET.md` — quantifizierte Datenqualität der Bestände
- `docs/PAID_ACCESS_UND_OFFENE_PUNKTE.md` — Beschaffungs-Backlog D1–D15
- `KNOWN_ISSUES.md` §0.1 — Survivorship-Status
- `docs/CLAUDE_CODING_ERRORS.md` — E-112, E-113, E-140
