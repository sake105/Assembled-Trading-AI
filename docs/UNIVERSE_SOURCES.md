# Universe- und Symbolquellen — kanonische Ordnung

**Letzte Aktualisierung:** 2026-08-15
**Zweck:** Eine einzige Antwort auf „welche Symbolliste gilt eigentlich?"

Vorher existierten drei versionierte Watchlist-Dateien im Repo-Root (plus das Output-Artefakt `missing_symbols.txt`) und 13 CSVs unter `data/universe/`
nebeneinander, ohne dass irgendwo stand, welche davon Quelle und welche Ableitung ist. Dieses
Dokument legt das fest.

---

## 1. Die kanonischen Quellen

| Rolle | Datei | Umfang | Warum sie es ist |
|---|---|---|---|
| **Operative Symbolliste** | `watchlist.txt` (Repo-Root) | 195 | 20+ Code-Consumer, git-versioniert, aktuellster Delisting-Pflegestand (`{KO, PEP}` ergänzt, `{EXAS, HOLX}` auskommentiert) |
| **Symbol-Metadaten** (Sektor/Cap) | `configs/universes/full_us_universe.yaml` | **202** | einzige Quelle mit Sektor-/Cap-Struktur; `full_us_universe.txt` wird daraus generiert (`master_universe_loader.write_flat_watchlist`) |
| **Gruppen-Metadaten** (Sektor/Region/Währung) | `configs/security_master.csv` | 32 | git-versioniert; speist Gruppen-Exposure-Limits |
| **Point-in-Time-Mitgliedschaft** | `data/universe/verdict_sp500.csv` ⚠️ **gitignored** | **1.167, davon 418 mit `end_date`** | erste Universe-Datei im Repo mit echten Ausscheide-Fenstern; erzeugt aus dem PIT-Preispanel via `pit_prices.build_pit_universe_history()` |

### Warum `verdict_sp500.csv` wichtig ist

> ⚠️ **Versionierungs-Vorbehalt.** Die Datei liegt unter `/data/` und ist damit **gitignored**.
> Erzeugbar ist sie nur aus `research/mandat/data/prices_verdict.parquet`, das am 2026-07-06
> eingefroren und wegen des EODHD-Ausfalls nicht mehr beschaffbar ist (`docs/DATENZUGANG_STATUS.md`).
> **Bei Verlust ist sie nicht rekonstruierbar.** Sie ist damit kanonisch im Sinne von
> „maßgeblich für PIT-Läufe", nicht im Sinne von „dauerhaft gesichert". Soll sie dauerhaft
> kanonisch sein, gehört sie nach `configs/universes/` (git-versioniert).

Die 13 zuvor existierenden `data/universe/*.csv` trugen zusammen **vier** `end_date`
(`EXAS` und `HOLX`, je zweimal) — das ist das
Unterscheidungsmerkmal, nicht die `status`-Spalte (`verdict_sp500.csv` ist selbst zu 100 %
`status: active`; `universe.py` wertet `status` nur dort aus, wo `end_date` NaT ist). Damit war `get_universe_members_pit()` — die PIT-sichere API — seit Monaten
faktisch wertlos: sie hatte keine Delisting-Fenster, gegen die sie hätte filtern können, und
folglich **keinen einzigen Produktions-Aufrufer**.

`verdict_sp500.csv` liefert 418 echte Ausscheide-Daten. Stichprobe gegen die Realität:

| Symbol | `end_date` | Ereignis |
|---|---|---|
| SIVB | 2023-03-10 | SVB-Kollaps, März 2023 |
| FRC | 2023-05-03 | First Republic, Zwangsübernahme Mai 2023 |
| TWTR | 2022-10-28 | Musk-Übernahme, Okt. 2022 |
| RTN | 2020-04-03 | Raytheon-UTC-Merger, April 2020 |

**Vorbehalt, der mitgelesen werden muss:** Diese Daten sind aus **Panel-Abdeckung abgeleitet**,
nicht aus Corporate Actions (DAT-006). Eine Feed-Lücke oder Umbenennung ist hier nicht von einem
echten Delisting zu unterscheiden. Bei Ticker-Recycling (E-114/E-117) markiert das `end_date`
teils die Naht zwischen zwei Unternehmen statt eines Delistings — sichtbar etwa an `BSC`
(2008-03-17 … 2011-08-26: Bear Stearns existierte ab Juni 2008 nicht mehr).

---

## 2. Was `data/universe/*.csv` wirklich ist

**Kein kuratiertes Universum, sondern ein automatisch erzeugter Cache.** Erzeugt von
`scripts/run_backtest_strategy.py`, wobei der Dateiname der *Stem der Preisdatei* ist:

```python
_universe_name = Path(_price_file).stem if _price_file else "default"
```

Deshalb heißen sie `master_universe_panel`, `price_slice`, `daily`, `backtest_crisis_test` — das
sind Namen von Preis-Parquets, keine Universen. Alle sind gitignored (`.gitignore:7`), regenerieren
sich bei Bedarf und sind gefahrlos löschbar.

**Konsequenz:** Diese Dateien sind keine Wahrheitsquelle und dürfen nicht als solche zitiert werden.

Ausnahmen mit echtem Consumer:
- `watchlist_2007_2026.csv` — argparse-Default in `scripts/forensic/survivorship_bias_check.py:359`
- `verdict_sp500.csv` — die oben beschriebene PIT-Quelle (bewusst erzeugt, kein Nebenprodukt)

---

## 3. Zurückgezogene Dateien (2026-08-15)

| Datei | Consumer | Verbleib |
|---|---|---|
| `watchlist_full.txt` (62) | **keiner** | → `archive/watchlists_2026-08-15/`; die 24 nur dort vorkommenden Symbole (ASML.AS, RHM.DE, VOW3.DE, TSM, COIN, SHOP …) wurden nach `configs/universes/international_legacy.txt` gerettet |
| `watchlist_29_backup.txt` (29) | **keiner** | → `archive/watchlists_2026-08-15/`; **symbolgleich** mit `configs/paper_track/watchlist_us_core.txt` (gleiche 29 Symbole, abweichende Reihenfolge und Header) |
| `missing_symbols.txt` (14) | **Input**, kein Output | bleibt im Root. `scripts/download_missing_symbols_sequential.ps1:57` **liest** sie (`Get-Content`, harter Abbruch bei Fehlen) und schreibt sie nie. Ein Writer existiert repo-weit nicht - woher die Datei stammt, ist ungeklaert. *(Eine fruehere Fassung behauptete das Gegenteil.)* |

---

## 4. Gemessene Äquivalenzklassen

Damit niemand erneut glaubt, es gäbe hier 17 verschiedene Universen:

| Klasse | Größe | **Symbolgleiche** Dateien |
|---|---|---|
| A | 195 | `master_universe_panel.csv` ≡ `master_universe_factor_panel.csv` |
| B | 29 | `watchlist_2020_2026.csv` ≡ `watchlist_us_core.csv` ≡ `configs/paper_track/watchlist_us_core.txt` |
| C | 2 | `daily.csv` ≡ `watchlist.csv` = {AAPL, MSFT} |

Inklusionskette:
`{AAPL,MSFT}(2) ⊂ price_slice(3) ⊂ watchlist_2007_2026(19) ⊂ watchlist_22_2020_2026(22) ⊂ Klasse B(29)`
und `top50_panel(50) ⊂ Klasse A(195) ⊂ full_us_universe.txt(200)`.

> **Nachtrag 2026-08-16 (Review-Korrekturen).**
> - `default.csv` gehoert **nicht** mehr zu Klasse B: sie hat heute **3** Zeilen
>   ({AAPL, BSC, SIVB}) statt 29 - ein `--pit-prices`-Testlauf dieser Sitzung hat sie ueber den
>   Panel-Stem `default` ueberschrieben. Kein Datenverlust, sondern der beste verfuegbare Beleg
>   fuer die These aus §2: diese CSVs sind **generierte Caches**, keine kuratierten Universen.
> - "Symbolgleich" heisst nicht "inhaltsgleich": die beiden 195er-Dateien tragen unterschiedliche
>   `start_date` (`master_universe_panel` 2021-01-04, `master_universe_factor_panel` 2025-01-02).
>   Bei einem PIT-Membership-Cache sind vier Jahre Verschiebung material.
> - `full_us_universe.yaml` hat **202** Ticker, die daraus generierte `.txt` nur **200** - es fehlen
>   genau `{KO, PEP}`. Die `.txt` wurde seit deren Aufnahme nicht regeneriert; "wird daraus
>   generiert" gilt also nur historisch.

Differenz zwischen den beiden 195er-Listen:
`watchlist.txt − master_universe_panel = {KO, PEP}` · `master_universe_panel − watchlist.txt = {EXAS, HOLX}`
→ `watchlist.txt` ist der aktuellere Stand.

---

## 5. Regeln

1. **Neue operative Symbole** kommen nach `watchlist.txt`, sonst nirgends.
2. **`data/universe/*.csv` nie von Hand pflegen** — sie werden überschrieben.
3. **Für historische Läufe** `get_universe_members_pit(as_of, universe_name="verdict_sp500")`
   verwenden, nicht `get_universe_members()` ohne `as_of` (der Fallback ist ausdrücklich
   nicht PIT-sicher und loggt das auch).
4. **Delisting-Daten aus `verdict_sp500.csv`** sind coverage-abgeleitet und dürfen nicht als
   Corporate-Action-Wahrheit weiterverwendet werden.

---

## Verweise

- `docs/DATENZUGANG_STATUS.md` — warum das PIT-Panel am 2026-07-06 eingefroren ist
- `src/assembled_core/data/pit_prices.py` — Adapter und Universe-Generator
- `KNOWN_ISSUES.md` §0.1 — Survivorship-Status
- `docs/CLAUDE_CODING_ERRORS.md` — E-114, E-117 (Ticker als Schlüssel)
- `docs/audit/07b_data_sources_ingestion.md` — DAT-006 (Delisting aus Panel-Abdeckung abgeleitet)
