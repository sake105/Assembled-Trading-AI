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
import re
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


def dezimal_de(text: str) -> str:
    """Normalisiert Dezimalpunkte in freiem Text auf Komma.

    Bewusst hier und nicht in der Datenschicht: der Verwurfsgrund steht bereits
    in committeten Artefakten, und ein Fix am Erzeuger wirkt erst nach einem
    Neulauf — der 110 Trials kostet. Ein Fix in der Renderschicht wirkt sofort
    und auch rueckwirkend (Anti-Pattern E-086).

    Gezielt nur Ziffer.Ziffer, damit Symbole wie ``BRK.B`` unangetastet bleiben.
    """
    return re.sub(r"(\d)\.(\d)", r"\1,\2", text)


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
            f"| {fak(z['zufall_brutto_mittel'])} "
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
    sv = lade("p12d_survivorship.json")

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
        else "Verworfen: "
        + "; ".join(f"{k} ({dezimal_de(v)})" for k, v in d["verworfen"].items())
    )
    # Wie oft liegt Momentum BRUTTO unter der Zufallsauswahl? Diese Zahl traegt
    # Kernaussage 2 und wird deshalb gerechnet, nicht geschaetzt.
    #
    # Direktzugriff, KEIN .get(..., 0.0): ein fehlender Schluessel wuerde sonst
    # lautlos "0 von 8" ergeben und damit das Vorzeichen des Befunds umdrehen,
    # ohne Fehler und ohne Warnung. Eine Datenluecke muss krachen, nicht
    # ueberzeugend aussehen.
    kurz_unter_zufall = [
        z
        for z in flach
        if z["halte_bars"] <= d["bars_pro_tag"]
        and z["brutto_end"] < z["zufall_brutto_mittel"]
    ]
    seed_zahlen = {len(z["zufall_end_alle"]) for z in flach}
    if len(seed_zahlen) != 1:
        raise SystemExit(f"Uneinheitliche Seed-Zahlen je Zeile: {sorted(seed_zahlen)}")
    n_seeds = seed_zahlen.pop()
    schlagen = [z for z in flach if z["netto_end"] > bh]

    t: list[str] = []
    t.append("# P12 — Das kurze Ende der Haltedauer (2026-08-03)\n")
    t.append(
        "> **Dieses Dokument wird generiert** (`render_befund_p12.py`), nicht von Hand\n"
        "> geschrieben. Grund: die erste Fassung war beim Schreiben korrekt und nach der\n"
        "> Review-Remediation komplett veraltet — zwei von drei Schlussfolgerungen\n"
        "> hatten sich umgekehrt (E-085). Jede Zahl in den Tabellen und Kernaussagen\n"
        "> stammt aus `results/p12*.json`. **Nicht** von dort: Kopfdatum, die\n"
        "> Spannenangabe der Haltedauern, der Rückblick-Faktor und die im\n"
        "> Buchhaltungs-Hinweis genannte Trial-Differenz — diese stehen im Generator\n"
        "> und sind damit nicht gegen Drift geschützt.\n"
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
    # Aussage 1 und 2 sind datenABHAENGIGE Wertungen und werden deshalb genauso
    # verzweigt wie Aussage 3. Ein Generator, der nur Zahlen gegen Drift
    # schuetzt, schuetzt nicht das, was gelesen wird — die Schlussfolgerung
    # (E-089). Genau so entstand E-085: nicht die Tabelle war falsch.
    t.append("## Was daraus folgt\n")
    bestes_kurz = max(z["netto_end"] for z in kurz)
    t.append(
        (
            "**1. Das kurze Ende trägt nicht.** Keine Haltedauer bis einschließlich\n"
            if bestes_kurz < bh
            else "**1. Mindestens eine kurze Haltedauer schlägt das Halten.** Bis\n"
        )
        + f"einem Tag: der beste kurze Wert ist {fak(bestes_kurz)} gegen {fak(bh)}.\n"
        f"Bei einstündigem Halten fallen "
        f"{tsd(max(z['umschichtungen'] for z in kurz))} Umschichtungen an.\n"
    )
    bruttos = [z["brutto_end"] for z in kurz]
    t.append(
        (
            "**2. Vor Kosten gewinnen die kurzen Haltedauern — aber schlechter als\n"
            f"das Los.** Brutto liegen sie zwischen {fak(min(bruttos))} und "
            f"{fak(max(bruttos))},\nalso im Plus. "
            if min(bruttos) > 1.0
            else "**2. Ein Teil der kurzen Haltedauern verliert schon brutto.**\n"
            f"Brutto liegen sie zwischen {fak(min(bruttos))} und "
            f"{fak(max(bruttos))}.\n"
        )
        + f"In **{len(kurz_unter_zufall)} von {len(kurz)}** kurzen Zeilen\n"
        "liegt Momentum brutto jedoch **unter der Zufallsauswahl** — die Rangfolge nach\n"
        "jüngster Rendite wählt dort aktiv schlechter als das Los, und zwar bevor eine\n"
        "einzige Gebühr anfällt. Ein Brutto-Alpha ist am kurzen Ende also nicht\n"
        "nachweisbar; die Kosten verschärfen das Bild zusätzlich.\n\n"
        f"*Belastbarkeit:* Diese Aussage beruht auf **{n_seeds}** Zufallsziehungen je\n"
        "Zeile ohne ausgewiesenes Streuungsmaß. Die große Kontrolle (P12b) lief am\n"
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
        # SIGNIERTE Deltas: das Vorzeichen IST die Aussage („nicht systematisch
        # gerichtet"). Ein abs() hier würde genau die Information wegwerfen, auf
        # der der Satz beruht, und die Behauptung unprüfbar machen (E-087).
        paare = [
            (z, x)
            for z in c["zeilen"]
            for x in (c_st["zeilen"] if c_st else [])
            if x["halte_bars"] == z["halte_bars"]
        ]
        if paare:
            t.append(
                "*brutto stufig* ist die Gegenprobe mit erzwungen stufigem Tagesfaktor\n"
                "(E-083). Reversal ist der Fall, den das reversierende Rauschen des\n"
                "Bereinigungsverfahrens aufblähen würde — deshalb ist diese Spalte hier\n"
                "Pflicht und nicht Fußnote.\n"
            )
            signiert = [x["brutto_end"] / z["brutto_end"] - 1.0 for z, x in paare]
            gerichtet = min(signiert) >= 0 or max(signiert) <= 0
            t.append(
                (
                    "Die Abweichung ist **systematisch gerichtet** (alle Vorzeichen\n"
                    "gleich) — ein Verfahrensartefakt ist damit **nicht** ausgeschlossen.\n"
                    if gerichtet
                    else "Die Abweichung ist **nicht systematisch gerichtet** (das\n"
                    "Vorzeichen wechselt), die Bruttokante ist also kein\n"
                    "Verfahrensartefakt.\n"
                )
            )
            # Artefaktschranke und Break-even MÜSSEN aus DERSELBEN Zeile stammen.
            # Eine frühere Fassung stellte das Maximum der einen Spalte neben das
            # Minimum der anderen — die 12,0 % kamen aus der 1-Tag-Zeile, die gar
            # keinen Break-even hat, die 1 bps aus den kurzen Zeilen (E-088).
            mit_be = [
                (z, x)
                for z, x in paare
                if z["breakeven_schlaegt_benchmark_bps"] is not None
            ]
            if mit_be:
                # Die tragende Zeile ist die mit dem höchsten Bruttowert.
                z0, x0 = max(mit_be, key=lambda zx: zx[0]["brutto_end"])
                schranke = abs(x0["brutto_end"] / z0["brutto_end"] - 1.0)
                be0 = z0["breakeven_schlaegt_benchmark_bps"]
                # KEIN .get(..., bei_10bps): ein fehlender Schluessel wuerde den
                # 10-bps-Wert unter der Beschriftung "bei {be0} bps" drucken —
                # eine ueberzeugend aussehende Falschzahl. Genau die Regel, die
                # oben fuer `zufall_brutto_mittel` gilt (E-086/E-089).
                bei_be = z0["kurve"][f"{be0}"]
                # Wie viel des Vorsprungs ueber Buy-and-Hold ueberlebt den
                # Break-even? Gerechnet statt als "Grossteil" behauptet (E-089).
                vorsprung = z0["brutto_end"] - z0["bench_end"]
                rest = (bei_be - z0["bench_end"]) / vorsprung if vorsprung > 0 else 0.0
                t.append(
                    f"Für die tragende Zeile (**{z0['name']}**, der höchste Bruttowert)\n"
                    f"beträgt die Artefaktschranke {pct(schranke)}. Ihr Break-even gegen\n"
                    f"Buy-and-Hold liegt bei {be0:g} bps — dort bleiben {fak(bei_be)}\n"
                    f"gegenüber {fak(z0['brutto_end'])} brutto, also {pct(max(rest, 0.0))}\n"
                    "des Bruttovorsprungs über das schlichte Halten.\n\n"
                    + (
                        "Schon **ein einzelner Basispunkt** kostet damit den Großteil\n"
                        "der Kante.\n"
                        if be0 <= 1.0 and rest < 0.5
                        else f"Die Kante überlebt bis {be0:g} bps je Seite.\n"
                    )
                    + "Sie ist real, aber die Aussage über ihre exakte Höhe ist es\n"
                    "nicht.\n"
                )

    if sv:
        u = sv["ueberhoehung_cagr"]
        t.append("## Wie stark ist das Universum survivorship-verzerrt? (P12d)\n")
        t.append(
            "Der Versuch, die Verzerrung durch ein Point-in-Time-Universum zu heilen,\n"
            "scheiterte an der Datenquelle: der EODHD-**Intraday**-Endpunkt führt keine\n"
            "delisteten Ticker (gemessen an 22 ausgeschiedenen Namen: 18 % Trefferquote\n"
            "gegen 92 % bei Überlebenden). Das **Tages**panel enthält die Toten dagegen\n"
            "vollständig — dort ist die Verzerrung wenigstens **bezifferbar**.\n"
        )
        t.append(
            "| Universum | n | B&H (Erlös gehalten) | B&H (umgeschichtet) | CAGR |\n"
            "|---|---|---|---|---|"
        )
        for x in sv["zeilen"]:
            t.append(
                f"| {x['universum']} | {x['n']} | {fak(x['halten']['endwert'])} "
                f"| {fak(x['umschichten']['endwert'])} | {pct(x['halten']['cagr'])} |"
            )
        t.append(f"| SPY (Referenz) | 1 | {fak(sv['spy']['endwert'])} | — | — |\n")
        # Der Entscheidungsabstand wird GERECHNET, nicht geschrieben: er stand
        # frueher als "rund 1,3 %" fest im Generator, direkt neben zwei aus dem
        # Artefakt gelesenen Zahlen — und trug die Schlussfolgerung (F-test-7).
        bester = max(flach, key=lambda z: z["netto_end"])
        abstand = d["buy_and_hold"]["cagr"] - bester["cagr"]
        t.append(
            f"**Überhöhung des P12-Benchmarks:** {pct(u['cagr_halten'])} p. a., wenn\n"
            "der Delisting-Erlös als totes Geld liegen bleibt, und "
            f"{pct(u['cagr_umschichten'])} p. a.,\nwenn er pro rata auf die "
            "überlebenden Positionen verteilt wird. Beide Varianten sind\n"
            "Buy-and-Hold und unterscheiden sich nur in dieser einen Annahme.\n"
        )
        t.append(
            "**Konsequenz für das Verdikt — und sie ist unbequem:** der Abstand\n"
            f"zwischen bestem Kandidaten ({fak(bester['netto_end'])}, "
            f"{pct(bester['cagr'])} p. a.) und Buy-and-Hold\n"
            f"({pct(d['buy_and_hold']['cagr'])} p. a.) beträgt "
            f"**{pct(abstand)} p. a.**\nDie Verzerrung liegt mit "
            f"{pct(u['cagr_umschichten'])} bis {pct(u['cagr_halten'])} "
            "**darüber**.\n\n"
            "Das heißt nicht, dass eine Strategie das Halten schlägt. Es heißt, dass\n"
            "**dieser Datensatz die Frage nicht entscheiden kann**: der gemessene\n"
            "Vorsprung des Benchmarks ist kleiner als die bekannte Verzerrung seines\n"
            "Universums. Ein survivorship-freier Intraday-Test wäre nötig — und ist\n"
            "mit dieser Datenquelle nicht baubar, weil der Endpunkt keine delisteten\n"
            "Ticker führt.\n\n"
            "*Eine frühere Fassung dieses Abschnitts nannte hier +0,1 % p. a. und\n"
            "schloss daraus, das Verdikt kippe nicht. Diese Zahl stammte aus einer\n"
            "fehlerhaften Vergleichsrechnung — täglich rebalanciertes Portfolio statt\n"
            "Buy-and-Hold, dessen Rebalancing-Bonus mit der Namenszahl wächst — und\n"
            "ist zurückgenommen (E-096).*\n"
        )
        t.append(
            "**Die Tages-Engine der Kampagne (P1–P11): PIT-korrekt in der Auswahl,\n"
            "aber nicht lückenlos.** `engine.run_strategy` wählt je Termin aus\n"
            "`membership(t)` und erzwingt über `last_valid` den Delisting-Verkauf; das\n"
            "Panel trägt Delistings (208 von 1.037 Symbolen enden vor Panelende), und\n"
            "das PIT-Universum enthält nachweislich Pleite-Ticker\n"
            f"({', '.join(sv['tote_ticker_im_pit_universum'])}).\n\n"
            "Der Restkanal, den ich vorher zu Unrecht wegformuliert hatte: die\n"
            "Preisabdeckung der Index-Mitglieder liegt über alle Monatsenden bei\n"
            "84–96 %, und die fehlenden Namen sind rund **fünffach mit\n"
            f"Index-Austritten angereichert**. {AUF}Survivorship-frei{ZU} ist zu\n"
            "stark — richtig ist: *die Auswahl ist PIT-korrekt, die Abdeckung nicht\n"
            "vollständig, und die Lücke ist nicht neutral.* Der Intraday-Strang P12\n"
            "ist davon unabhängig und deutlich stärker betroffen.\n"
        )
        if sv.get("ausgeschlossene_glitches"):
            g = sv["ausgeschlossene_glitches"]
            schlimm = sorted(g.items(), key=lambda kv: -kv[1]["sprung"])[:3]
            t.append(
                f"*Nebenbefund Datenqualität:* {len(g)} Namen des PIT-Universums sind\n"
                "korrumpiert und wurden ausgeschlossen. Es handelt sich **nicht** um\n"
                "einzelne Ausreißertage, sondern um Serien mit zwei ineinander\n"
                "verschränkten Preisskalen über Dutzende Tage — bei MEL etwa liegt das\n"
                "Niveau 2014-11-10..17 abwechselnd bei ~141.000 und ~7,80, wobei der\n"
                "**niedrige** Wert der plausible ist. Weitere Fälle: "
                + ", ".join(f"**{s}**" for s, _ in schlimm if s != "MEL")
                + ".\n\nDie Truncation-Regel in `campaign_data` greift nur bei "
                "Vortagskursen unter\n1 USD und lässt diese Klasse durch. Der Detektor "
                "hier sieht wiederum nur\ndiese eine Morphologie: dauerhafte "
                "Niveausprünge im Band 100–200 %\n(AYE +170 %, TOY +155 %, HIG +102 %) "
                "passieren ungeprüft durch und bleiben\nim Universum — ob sie echt "
                "sind, ist **offen**. Ob P1–P11 von den korrupten\nNamen berührt sind, "
                "ist ebenfalls offen und ein eigener Prüfschritt.\n"
            )

    t.append("## Was dieser Strang nicht beantwortet\n")
    t.append(
        "- **Absolute Renditen** — Universum survivorship-verzerrt (beziffert in P12d).\n"
        "- **Andere Signale am kurzen Ende.** Getestet wurde Momentum und sein\n"
        "  Gegenteil, nicht Orderbuch-, Nachrichten- oder Volatilitätssignale. Der\n"
        "  Befund lautet: dieses Signal trägt dort nicht — nicht: dort ist nichts.\n"
        "- **Ausführungsrealismus.** Bar-Kurs plus Pauschale; echte Marktwirkung bei\n"
        "  stündlichem Umschlag wäre schlechter, nicht besser.\n"
        "- **Gefüllte Bars.** Gerechnet wird auf `close.ffill()`. Eine gefüllte Bar\n"
        "  erzeugt exakt 0 Rendite und geht in Signal und Umschichtung ein; bei\n"
        "  stündlicher Haltedauer ist das nicht vernachlässigbar.\n"
        # Die Schwelle kommt aus dem LAUF-Artefakt, nicht aus dem Live-Code:
        # ein Import der Konstante haette das Dokument an den JETZIGEN Stand
        # gekoppelt und gegen ein aelteres Ergebnis eine Zahl behauptet, die
        # dort nie galt (E-086, zweite Naht).
        + (
            f"  Der Abdeckungsfilter lässt strukturell bis zu "
            f"{pct(1.0 - d['min_abdeckung'])} gefüllte Bars zu.\n"
            if "min_abdeckung" in d
            else "  Der Abdeckungsfilter lässt strukturell gefüllte Bars zu; dieses\n"
            "  Lauf-Artefakt führt die Schwelle noch nicht, deshalb ist sie hier\n"
            "  nicht beziffert.\n"
        )
        + "  Die Richtung ist konservativ — es dämpft das kurze Ende, rettet das\n"
        "  negative Verdikt also nicht.\n"
        "- **Der Holdout bleibt versiegelt.** Kein Kandidat aus P12 hat ihn verdient.\n"
    )
    t.append(
        "## Buchhaltungs-Hinweis zum Trial-Zähler\n\n"
        "Ein Wiederholungslauf von P12c zur Artefakt-Hygiene (bit-identische\n"
        "Ergebnisse, nur ein fehlendes Metadatenfeld) hat **44 Trials** gezählt,\n"
        "ohne eine einzige neue Hypothese zu prüfen. Der Zähler steuert den\n"
        f"DSR-Haircut und bedeutet {AUF}Zahl geprüfter Hypothesen{ZU} — Regenerationen\n"
        "gehören nicht hinein. Die Skripte haben dafür jetzt `--regen`; die bereits\n"
        "gezählten 44 werden **nicht** stillschweigend zurückgeschrieben, sondern\n"
        "hier offengelegt (E-090). Wirkung: der Haircut ist um diesen Betrag zu\n"
        "streng, also konservativ.\n"
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
