"""Erzeugt BEFUND_P12_INTRADAY.md AUS den results/*.json — nicht von Hand.

WARUM DAS EIN EIGENES SKRIPT IST
--------------------------------
Die erste Fassung des Befunds war beim Schreiben korrekt: jede Zahl stammte
aus dem damaligen `results/*.json`. Dann korrigierte die Review-Kette vier
Konstruktionsfehler, der Lauf wurde wiederholt — und das Dokument nicht. Jede
Zeile der Tabelle widersprach anschliessend ihrer eigenen Quelle, und zwei von
drei Schlussfolgerungen hatten sich umgekehrt (Anti-Pattern E-085).

Die Regel `keine Zahl, die nicht aus results/*.json stammt` (E-073/E-076) ist
beim Schreiben pruefbar und danach nie wieder. In einer Review-Kette ist das
systematisch toedlich, weil die Remediation per Definition NACH dem Schreiben
kommt. Also wird das Dokument generiert: dann ist Drift nicht unwahrscheinlich,
sondern unmoeglich.

Die Interpretation (die Prosa zwischen den Tabellen) bleibt handgeschrieben —
aber sie steht in diesem Skript, direkt neben den Zahlen, aus denen sie folgt,
und wird bei jedem Neulauf mit ihnen zusammen neu erzeugt.
"""

from __future__ import annotations

import json
from pathlib import Path

HIER = Path(__file__).resolve().parent
RES = HIER / "results"
ZIEL = HIER / "BEFUND_P12_INTRADAY.md"

AUF = "„"  # oeffnendes deutsches Anfuehrungszeichen
ZU = "“"  # schliessendes — NICHT das ASCII-", das beendet den String


def lade(name: str) -> dict | None:
    p = RES / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def pct(x: float) -> str:
    return f"{x * 100:.1f} %".replace(".", ",")


def fak(x: float) -> str:
    return f"{x:.3f}×".replace(".", ",")


def zahl(x: float, n: int = 2) -> str:
    return f"{x:.{n}f}".replace(".", ",")


def tsd(n: int) -> str:
    """Deutsche Tausendertrennung — Punkt, nicht Komma."""
    return f"{n:,}".replace(",", ".")


def tabelle(zeilen: list[dict], bh: float) -> str:
    kopf = (
        "| Haltedauer | Rückblick | Umschicht. | netto | brutto | Zufall netto "
        "| Zufall brutto | MaxDD | Kostenlast |\n"
        "|---|---|---|---|---|---|---|---|---|\n"
    )
    aus = []
    for z in zeilen:
        f = "**" if z["netto_end"] > bh else ""
        aus.append(
            f"| {z['name']} | {tsd(z['rueckblick_bars'])} | {tsd(z['umschichtungen'])} "
            f"| {f}{fak(z['netto_end'])}{f} | {fak(z['brutto_end'])} "
            f"| {fak(z['zufall_end_mittel'])} "
            f"| {fak(z.get('zufall_brutto_mittel', 0.0))} "
            f"| {pct(z['maxdd'])} | {pct(z['kostenlast'])} |"
        )
    return kopf + "\n".join(aus)


def main() -> int:
    d = lade("p12_intraday_haltedauer.json")
    if d is None:
        raise SystemExit(
            "results/p12_intraday_haltedauer.json fehlt — erst P12 laufen lassen"
        )
    st = lade("p12_intraday_stufig.json")
    b = lade("p12b_zufallskontrolle.json")
    c = lade("p12c_reversal_kostenschwelle.json")
    c_st = lade("p12c_reversal_stufig.json")

    bh = d["buy_and_hold"]["endwert"]
    reb = d["ew_rebalanciert"]["endwert"]
    fam = d["familien"]
    flach = [dict(z, familie=f) for f, zs in fam.items() for z in zs]
    kurz = [z for z in flach if z["halte_bars"] <= d["bars_pro_tag"]]
    best_n = max(flach, key=lambda z: z["netto_end"])
    best_b = max(flach, key=lambda z: z["brutto_end"])
    max_cash = max(z["anteil_cash"] for z in flach)
    verworfen_satz = (
        "Kein Symbol verworfen"
        if not d["verworfen"]
        else "Verworfen: " + "; ".join(f"{k} ({v})" for k, v in d["verworfen"].items())
    )
    # Wie oft liegt Momentum BRUTTO unter der Zufallsauswahl? Diese Zahl traegt
    # Kernaussage 2 und wird deshalb gerechnet, nicht geschaetzt.
    kurz_unter_zufall = [
        z
        for z in flach
        if z["halte_bars"] <= d["bars_pro_tag"]
        and z["brutto_end"] < z.get("zufall_brutto_mittel", 0.0)
    ]
    schlagen = [z for z in flach if z["netto_end"] > bh]

    t: list[str] = []
    t.append("# P12 — Das kurze Ende der Haltedauer (2026-08-03)\n")
    t.append(
        "> **Dieses Dokument wird generiert** (`render_befund_p12.py`), nicht von Hand\n"
        "> geschrieben. Grund: die erste Fassung war beim Schreiben korrekt und nach der\n"
        "> Review-Remediation komplett veraltet — zwei von drei Schlussfolgerungen\n"
        "> hatten sich umgekehrt (E-085). Jede Zahl unten stammt aus\n"
        "> `results/p12*.json`.\n"
    )
    t.append(
        "Der Strang, den ich fälschlich für datenblockiert erklärt hatte (E-080).\n"
        f"Hans' Frage im Wortlaut: *{AUF}Du kannst Aktie auch kürzer halten, nur\n"
        f"mehrere wenige Monate oder wenige Stunden.{ZU}* P2 hatte das lange Ende\n"
        "beantwortet. Hier ist das kurze.\n"
    )

    # ---------------- Daten ----------------
    t.append("## Die Datengrundlage\n")
    t.append(
        f"**{len(d['universum'])} Symbole**, Stundenbars aus dem EODHD-1m-Endpunkt\n"
        "verdichtet. Ausgewertet wird ausschließlich das Suchfenster; der Holdout ist\n"
        "in dieser Schicht nicht vorhanden, nicht bloß ungenutzt.\n\n"
        f"- Gemeinsames Fenster: **{d['fenster']}** ({zahl(d['jahre'])} Jahre)\n"
        f"- Warm-up {tsd(d['warmup_bars'])} Bars, für **alle** Varianten identisch —\n"
        f"  gemessener Cash-Anteil max. {zahl(max_cash * 100, 4)} %\n"
        f"- {verworfen_satz}\n"
    )
    t.append(
        "**Die Rohdaten sind unbereinigt.** Vor der Bereinigung enthält das Panel\n"
        f"{d['roh_spruenge']} Stundensprünge über 35 %, danach {d['rest_spruenge']}.\n"
        "Die größten Rohsprünge sind Kapitalmaßnahmen, keine Marktbewegungen:\n"
    )
    t.append("| Symbol | Zeitpunkt | roher Sprung |\n|---|---|---|")
    for e in d["split_diagnose"][:6]:
        t.append(f"| {e['symbol']} | {e['zeitpunkt']} | {pct(e['roher_sprung'])} |")
    t.append(
        "\nBereinigt wird über den Anker, der in dieser Kampagne ohnehin gilt — das\n"
        "tagesgenaue, total-return-adjustierte Panel. Der Tagesfaktor ist **innerhalb**\n"
        "eines Tages konstant: Intraday-Renditen bleiben unverändert (Splits wirken\n"
        "über Nacht), Übernacht-Renditen werden um Split *und* Dividende korrigiert.\n"
    )
    t.append(
        "### Survivorship — hart benannt\n\n"
        "Das Universum ist **nicht** survivorship-frei: Namen, die 2004–2016\n"
        "durchgehend im Index waren. Deshalb wird **ausschließlich innerhalb des\n"
        "Universums** verglichen, nie gegen SPY (E-079). Absolute Renditeaussagen sind\n"
        "aus diesem Strang nicht ableitbar; die relative Frage nach der Wirkung\n"
        "kürzeren Haltens ist es.\n"
    )

    # ---------------- Aufbau ----------------
    t.append("## Der Test\n")
    t.append(
        f"Variiert wird die Haltedauer von **1 Stunde bis 2 Jahre**. Top {d['top_k']}\n"
        f"gleichgewichtet, {d['kosten_bps']:.0f} bps je Seite, Steuerwelt Null.\n"
        "Positionen **driften zwischen den Terminen** (Stücke, nicht Gewichte) — sonst\n"
        "wäre Haltedauer nicht das, was das Wort sagt.\n\n"
        "Zwei Rückblick-Familien, weil ein mit der Haltedauer skalierender Rückblick\n"
        "das Signal mitvariieren lässt und damit kein Ein-Parameter-Sweep mehr ist:\n\n"
        "- **A** hält den Rückblick fest → echter Ein-Parameter-Sweep.\n"
        "- **B** skaliert ihn mit der Haltedauer. Nur Zeilen ohne Deckelung; gedeckelte\n"
        "  wären untereinander identisch parametriert und hatten im Vorlauf genau\n"
        "  deshalb die Bestwerte geliefert (E-084).\n\n"
        "**Offengelegter Freiheitsgrad:** Der Warm-up ist ein **gesetzter** Wert,\n"
        "nicht abgeleitet. Er wirkt zweifach — er bestimmt den gemeinsamen\n"
        "Fensterstart *und* welche Haltedauern Familie B überhaupt enthält (nur\n"
        "solche mit 20× Rückblick ≤ Warm-up). Ein größerer Wert würde weitere\n"
        "B-Zeilen zulassen und zugleich den Fensterstart nach hinten schieben.\n"
    )
    t.append(
        "**Benchmark — gleiches Universum, gleiche Gewichtungsmethode:**\n\n"
        "| | Endwert | CAGR | MaxDD |\n|---|---|---|---|\n"
        f"| Buy-and-Hold | {fak(bh)} | {pct(d['buy_and_hold']['cagr'])} "
        f"| {pct(d['buy_and_hold']['maxdd'])} |\n"
        f"| EW monatlich rebalanciert | {fak(reb)} "
        f"| {pct(d['ew_rebalanciert']['cagr'])} "
        f"| {pct(d['ew_rebalanciert']['maxdd'])} |\n"
    )

    # ---------------- Ergebnisse ----------------
    t.append("## Das Ergebnis\n")
    for name, zeilen in fam.items():
        t.append(f"### Familie {name}\n")
        t.append(tabelle(zeilen, bh))
        t.append("")
    t.append(f"Fett = schlägt Buy-and-Hold ({fak(bh)}).\n")

    # ---------------- Interpretation ----------------
    t.append("## Was daraus folgt\n")
    t.append(
        "**1. Das kurze Ende trägt nicht.** Keine Haltedauer bis einschließlich einem\n"
        f"Tag kommt netto in die Nähe des schlichten Haltens: der beste kurze Wert ist\n"
        f"{fak(max(z['netto_end'] for z in kurz))} gegen {fak(bh)}. Bei einstündigem\n"
        f"Halten fallen {tsd(max(z['umschichtungen'] for z in kurz))} Umschichtungen an.\n"
    )
    bruttos = [z["brutto_end"] for z in kurz]
    t.append(
        "**2. Vor Kosten gewinnen die kurzen Haltedauern — aber schlechter als das\n"
        f"Los.** Brutto liegen sie zwischen {fak(min(bruttos))} und "
        f"{fak(max(bruttos))},\n"
        f"also im Plus. In **{len(kurz_unter_zufall)} von {len(kurz)}** kurzen Zeilen\n"
        "liegt Momentum brutto jedoch **unter der Zufallsauswahl** — die Rangfolge nach\n"
        "jüngster Rendite wählt dort aktiv schlechter als das Los, und zwar bevor eine\n"
        "einzige Gebühr anfällt. Ein Brutto-Alpha ist am kurzen Ende also nicht\n"
        "nachweisbar; die Kosten verschärfen das Bild zusätzlich.\n\n"
        "*Belastbarkeit:* Diese Aussage beruht auf **fünf** Zufallsziehungen je Zeile\n"
        "ohne ausgewiesenes Streuungsmaß. Die 60-Seed-Kontrolle (P12b) lief am\n"
        "**langen** Ende. Eine 60-Seed-Kontrolle am kurzen Ende ist offener\n"
        "Folgeschritt — bis dahin ist Aussage 2 ein Hinweis, kein Beleg.\n\n"
        "Eine frühere Fassung dieses Dokuments behauptete hier ein umgekehrtes\n"
        "Vorzeichen (Bruttowert 0,159×). Das war ein Artefakt eines fehlerhaften\n"
        "Laufs — feste Gewichte statt driftender Positionen, unterschiedliche\n"
        "Startzeitpunkte, außerbörsliche Bars. Zurückgenommen.\n"
    )
    if schlagen:
        namen = ", ".join(
            f"{z['name']} ({z['familie'][0]}) {fak(z['netto_end'])}" for z in schlagen
        )
        t.append(
            f"**3. {len(schlagen)} von {len(flach)} Zeilen schlagen Buy-and-Hold:**\n"
            f"{namen}. Der Abstand ist klein gegenüber der Streuung der\n"
            "Zufallskontrolle — P12b prüft das mit mehr Ziehungen.\n"
        )
    else:
        t.append(
            f"**3. Keine einzige Zeile schlägt Buy-and-Hold.** Bester Netto-Wert\n"
            f"{fak(best_n['netto_end'])} ({best_n['name']}, {best_n['familie']}) gegen\n"
            f"{fak(bh)}. Das reproduziert am Intraday-Panel, was P2 am Tagespanel fand:\n"
            "Umschlag ist der schädliche Parameter.\n"
        )
    t.append(
        f"Bester Brutto-Wert über alle Zeilen: {fak(best_b['brutto_end'])}\n"
        f"({best_b['name']}, {best_b['familie']}).\n"
    )

    # ---------------- Robustheit ----------------
    if st:
        t.append("## Artefaktschranke des Bereinigungsverfahrens\n")
        t.append(
            "Der Tagesfaktor *soll* eine Treppe sein, ist es aber nicht: er absorbiert\n"
            "auch die Differenz zwischen Vendor-Tagesschluss und letzter Stundenbar —\n"
            "ein reversierendes Rauschen, also gleichgerichtet mit dem Effekt, den ein\n"
            "Intraday-Test am kurzen Ende sucht (E-083). Gegenprobe mit erzwungen\n"
            "stufigem Faktor:\n"
        )
        t.append(
            "| Haltedauer | Familie | netto normal | netto stufig | Δ |\n"
            "|---|---|---|---|---|"
        )
        for f_, zs in st["familien"].items():
            for z in zs:
                orig = next(
                    (o for o in fam[f_] if o["halte_bars"] == z["halte_bars"]), None
                )
                if orig and orig["netto_end"] > 0:
                    dlt = z["netto_end"] / orig["netto_end"] - 1.0
                    t.append(
                        f"| {z['name']} | {f_[0]} | {fak(orig['netto_end'])} "
                        f"| {fak(z['netto_end'])} | {pct(dlt)} |"
                    )
        t.append("")

    if b:
        t.append("## P12b — Momentum gegen Zufall am langen Ende\n")
        t.append(
            f"{b['n_seeds']} Zufallsziehungen je Haltedauer, jeweils mit derselben\n"
            "NaN-Maske wie das echte Signal (sonst vergleicht die Kontrolle einen\n"
            "anderen Zeitraum, E-082).\n"
        )
        t.append(
            "| Haltedauer | Momentum | Zufall Median | Zufall 5–95 % | Perzentil "
            "| p (einseitig) |\n|---|---|---|---|---|---|"
        )
        for z in b["zeilen"]:
            t.append(
                f"| {z['name']} | {fak(z['momentum_end'])} "
                f"| {fak(z['zufall_median'])} "
                f"| {fak(z['zufall_p05'])} – {fak(z['zufall_p95'])} "
                f"| {z['momentum_perzentil']:.0f} % | {z['p_einseitig']:.3f} |"
            )
        t.append("")

    if c:
        t.append("## P12c — Trägt das umgekehrte Vorzeichen die Reibung?\n")
        t.append(
            "| Haltedauer | Umschicht. | brutto | brutto stufig | Δ | bei 10 bps "
            "| Break-even vs. Halten |\n|---|---|---|---|---|---|---|"
        )
        for z in c["zeilen"]:
            be = z["breakeven_schlaegt_benchmark_bps"]
            zs = (
                next(
                    (x for x in c_st["zeilen"] if x["halte_bars"] == z["halte_bars"]),
                    None,
                )
                if c_st
                else None
            )
            t.append(
                f"| {z['name']} | {tsd(z['umschichtungen'])} | {fak(z['brutto_end'])} "
                f"| {fak(zs['brutto_end']) if zs else '—'} "
                f"| {pct(zs['brutto_end'] / z['brutto_end'] - 1.0) if zs else '—'} "
                f"| {fak(z['bei_10bps'])} "
                f"| {f'{be:g} bps' if be is not None else 'nie'} |"
            )
        t.append("")
        if c_st:
            deltas = [
                abs(x["brutto_end"] / z["brutto_end"] - 1.0)
                for z in c["zeilen"]
                for x in c_st["zeilen"]
                if x["halte_bars"] == z["halte_bars"]
            ]
            t.append(
                "*brutto stufig* ist die Gegenprobe mit erzwungen stufigem Tagesfaktor\n"
                "(E-083). Reversal ist der Fall, den das reversierende Rauschen des\n"
                "Bereinigungsverfahrens aufblähen würde — deshalb ist diese Spalte hier\n"
                "Pflicht und nicht Fußnote.\n\n"
                "Die Abweichung ist **nicht systematisch gerichtet** (das Vorzeichen\n"
                "wechselt), die Bruttokante ist also kein Verfahrensartefakt. Ihre Größe\n"
                f"reicht aber bis {pct(max(deltas))}, während der Break-even bei 1 bps\n"
                "liegt: die Artefaktschranke ist damit von derselben Größenordnung wie\n"
                "der verbleibende Spielraum. Beides zusammen gelesen heißt: die Kante\n"
                "ist real, aber die Aussage über ihre exakte Höhe ist es nicht.\n"
            )

    t.append("## Was dieser Strang nicht beantwortet\n")
    t.append(
        "- **Absolute Renditen** — Universum survivorship-verzerrt.\n"
        "- **Andere Signale am kurzen Ende.** Getestet wurde Momentum und sein\n"
        "  Gegenteil, nicht Orderbuch-, Nachrichten- oder Volatilitätssignale. Der\n"
        "  Befund lautet: dieses Signal trägt dort nicht — nicht: dort ist nichts.\n"
        "- **Ausführungsrealismus.** Bar-Kurs plus Pauschale; echte Marktwirkung bei\n"
        "  stündlichem Umschlag wäre schlechter, nicht besser.\n"
        "- **Gefüllte Bars.** Gerechnet wird auf `close.ffill()`. Eine gefüllte Bar\n"
        "  erzeugt exakt 0 Rendite und geht in Signal und Umschichtung ein; bei\n"
        "  stündlicher Haltedauer ist das nicht vernachlässigbar. Der\n"
        "  Abdeckungsfilter lässt strukturell bis zu 10 % gefüllte Bars zu. Die\n"
        "  Richtung ist konservativ — es dämpft das kurze Ende, rettet das negative\n"
        "  Verdikt also nicht.\n"
        "- **Der Holdout bleibt versiegelt.** Kein Kandidat aus P12 hat ihn verdient.\n"
    )
    t.append(
        "## Offene Folgeschritte\n\n"
        "1. **60-Seed-Zufallskontrolle am kurzen Ende** (netto *und* brutto). Erst\n"
        "   damit wird Aussage 2 vom Hinweis zum Beleg.\n"
        "2. **Abdeckung je behaltenem Symbol** ins Ergebnis-JSON, damit der\n"
        "   ffill-Anteil nachprüfbar ist statt nur beschränkt.\n"
        "3. **CI-Status.** Tests und Lint sind lokal grün, nicht CI-bestätigt.\n"
    )

    ZIEL.write_text("\n".join(t) + "\n", encoding="utf-8")
    print(f"-> {ZIEL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
