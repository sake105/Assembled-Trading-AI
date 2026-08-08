# Form-4-Bestände — welcher ist autoritativ?

Im Repo existieren **drei** Form-4-Bestände mit unterschiedlicher Herkunft und
unvereinbaren Spaltennamen. Dieses Dokument legt die Autorität fest, damit kein
Konsument versehentlich den falschen liest (Follow-up aus dem Stage-3-Audit zu
`9b585eb3`, F-senior-11/F-auditor-7).

## Rangfolge

| Bestand | Pfad | Umfang | Status |
|---|---|---|---|
| **DERA (autoritativ)** | `research/mandat/data/form4_dera/` | 17.134 Emittenten, 2006Q1–2026Q1, 8,31 Mio Zeilen | **maßgeblich** für alle neuen Arbeiten |
| form4_broad (legacy) | `research/mandat/data/form4_broad/` | 921 Symbole (S&P-Historie), 2012–2026 | **nicht autoritativ** — zeitlich und universumsseitig vollständig in DERA enthalten |
| Core-Ingester-Bestand | `data/raw/insider_congress/form4_insider_full.parquet` | 180 Symbole, ab 2003 | Produktionspfad des Core-Ingesters; einzige Quelle mit **ACCEPTANCE-Minute** |

## Warum form4_broad nicht gelöscht wird

H-031/H-053 wurden darauf gerechnet; der Bestand bleibt für die
Reproduzierbarkeit dieser Verdicts liegen. Er darf aber in keine **neue**
Analyse einfließen.

## Vertragsunterschiede (deshalb scheitert ein Store-Tausch laut, nicht still)

| | DERA | form4_broad / Core |
|---|---|---|
| Stückzahl | `trans_shares` | `shares` |
| Preis | `trans_pricepershare` | `price` |
| Emittent | `ISSUERCIK` | `issuer_cik` |
| Verfügbarkeit | `available_at` = **Meldetag + 1 (UTC)**, Basis in `available_at_basis` | `available_at` = **ACCEPTANCE-Minute (UTC)** |
| Richtung | `transaction_type` ∈ {P, S, unknown} (identisch, via `classify_transaction_code`) | dito |

**Wer beide mischt**, muss auf `available_at_basis` gruppieren und darf die
gröbere Definition nur nach oben runden (E-038, F-senior-5 aus `9b585eb3`).

## Entblähung (E-124)

DERA-Zeilen sind je Meldepflichtigem dupliziert (Cluster-Signal). Zählungen
über `RPTOWNERCIK.nunique()`; Stück-/Wertsummen **erst nach**
`drop_duplicates('NONDERIV_TRANS_SK')` — roh sind sie storeweit +37,8 %
(Kaufzeilen +52,7 %) zu hoch.
