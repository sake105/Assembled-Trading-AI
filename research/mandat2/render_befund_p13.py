"""Erzeugt BEFUND_SPY_TREND.md aus den P13-Artefakten.

Der Befund wird **generiert**, nicht geschrieben (E-085): sonst driftet der
Text gegen den Lauf. Jede Zahl im Dokument stammt aus einem Schluessel eines
Artefakts; steht sie dort nicht, darf sie hier nicht auftauchen (E-116).

Bewusst KEINE eigene Kennzahl, die nicht im Artefakt steht — die Bandbreiten
werden aus den gespeicherten `bestanden`-Flags abgeleitet und sind damit
jederzeit nachrechenbar.
"""

from __future__ import annotations

import json
from pathlib import Path

from research.mandat2.data_gate import TRIALS_MANDAT_I, TrialCounter
from research.mandat2.metrics import FENSTER_JAHRE

HIER = Path(__file__).resolve().parent
RES = HIER / "results"
ZIEL = HIER / "BEFUND_SPY_TREND.md"

#: Schrittweite des Fensterrasters. Aus P5 uebernommen; hier nur noetig, um
#: "zusammenhaengend" zu definieren.
SCHRITT = 20

#: Deutsche Anfuehrungszeichen als Konstanten. Das schliessende (U+201C) sieht
#: dem ASCII-Zoll zum Verwechseln aehnlich; direkt getippt beendet es den
#: f-String und der Renderer laesst sich nicht mehr laden.
AUF = "„"
ZU = "“"

#: Ab welcher lueckenlosen Kettenlaenge ein Band als "breit" gilt. Zwei
#: Drittel des Rasters (8 von 12). Als Konstante und nicht als Literal im
#: Satz, weil die Aussage "2 der 3 Definitionen tragen ein Band" sonst an
#: einer Zahl haengt, die kein Test sieht (Stage-1-Befund N5).
MINDEST_KETTE = 8


def laden(name: str) -> dict:
    p = RES / name
    if not p.exists():
        raise SystemExit(
            f"[ERROR] {p.name} fehlt. Erst den zugehoerigen Lauf ausfuehren — "
            f"ein Befund ohne Artefakt ist genau der Fehler, den E-085 meint."
        )
    return json.loads(p.read_text(encoding="utf-8"))


def laengster_block(fenster: list[int]) -> list[int]:
    """Laengste lueckenlose Kette im Raster.

    Breite UND Zusammenhang sind zwei Merkmale. Nur nach einem zu entscheiden
    war der erste Entwurf und ist die Verkuerzung aus E-117: 10 von 12 mit
    einem Loch ist nicht dasselbe wie 2 von 12.
    """
    if not fenster:
        return []
    best: list[int] = []
    lauf = [fenster[0]]
    for a, b in zip(fenster, fenster[1:]):
        if b - a == SCHRITT:
            lauf.append(b)
        else:
            best = max(best, lauf, key=len)
            lauf = [b]
    return max(best, lauf, key=len)


def bestandene(zeilen: list[dict], welt: str, defn: str) -> list[int]:
    return sorted(
        z["fenster"]
        for z in zeilen
        if z["welt"] == welt and z["definition"] == defn and z["bestanden"]
    )


def alle(zeilen: list[dict], schluessel: str) -> list[str]:
    gesehen: list[str] = []
    for z in zeilen:
        if z[schluessel] not in gesehen:
            gesehen.append(z[schluessel])
    return gesehen


def tabelle(ohne: dict, mit: dict) -> tuple[list[str], dict, int]:
    """Bandtabelle plus die Schnittmenge beider Ausfuehrungsannahmen."""
    zo, zm = ohne["zeilen"], mit["zeilen"]
    n_fenster = len({z["fenster"] for z in zo})
    zeilen = [
        "| Steuerwelt | Definition | besteht ohne Verz. | mit Verz. | in **beiden** | längste lückenlose Kette (beide) |",
        "|---|---|---:|---:|---:|---|",
    ]
    schnitt: dict[str, list[int]] = {}
    for welt in alle(zo, "welt"):
        for defn in alle(zo, "definition"):
            a, b = bestandene(zo, welt, defn), bestandene(zm, welt, defn)
            s = sorted(set(a) & set(b))
            schnitt[f"{welt}/{defn}"] = s
            kette = laengster_block(s)
            k = (
                f"{kette[0]}–{kette[-1]} ({len(kette)})"
                if len(kette) > 1
                else (str(kette[0]) if kette else "—")
            )
            zeilen.append(
                f"| {welt} | `{defn}` | {len(a)}/{n_fenster} | {len(b)}/{n_fenster} "
                f"| **{len(s)}/{n_fenster}** | {k} |"
            )
    return zeilen, schnitt, n_fenster


def ketten_je_definition(
    schnitt: dict[str, list[int]], definitionen: list[str], welt: str = "ZERO"
) -> dict[str, tuple[int, int]]:
    """Je Definition: (Zahl bestandener Fenster, Laenge der laengsten Kette).

    Genau die beiden Achsen aus E-117 — Breite und Zusammenhang — nebeneinander,
    damit der Befundsatz beide nennen muss und keine auf die andere reduziert.
    """
    return {
        d: (len(schnitt[f"{welt}/{d}"]), len(laengster_block(schnitt[f"{welt}/{d}"])))
        for d in definitionen
    }


def main() -> int:
    ohne = laden("p13_spy_trend_robustheit.json")
    mit = laden("p13b_ausfuehrungsverzoegerung.json")
    ereignis = laden("p13c_ereignisabhaengigkeit.json")
    zufall = laden("p13d_zufallstiming.json")
    korrektur = laden("p13e_dsr_pbo_spy.json")

    band, schnitt, n_fenster = tabelle(ohne, mit)
    trials = ohne["n_trials"] + mit["n_trials"]
    zero = zufall["ZERO"]
    ez = ereignis["ZERO"]
    v = ez["benchmark_dd_verteilung"]
    # Zahl der rollierenden Auswertungsfenster (144) — aus dem Artefakt, nicht
    # getippt. `n_fenster` (12) ist die Rastergroesse; die beiden zu
    # verwechseln waere in jedem Satz sofort falsch.
    n_win = ez["n_fenster"]
    # Ein Handelstag steckt in so vielen Fenstern, wie ein Fenster Monate hat
    # (Schrittweite ein Monat) — abgeleitet, nicht behauptet.
    ueberlappung = FENSTER_JAHRE * 12

    # Der schwaechste Punkt zuerst: gibt es ueberhaupt ein krisenfreies Fenster?
    ruhig = ez["ruhige_fenster"]["n"]

    t: list[str] = []
    t.append("# Befund P13 — SPY mit Trendfilter\n")
    t.append(
        "> Erzeugt von `render_befund_p13.py` aus den Artefakten in `results/`. "
        "Nicht von Hand bearbeiten.\n>\n"
        "> **Reichweite dieser Zusicherung:** Alle Ergebniszahlen dieses Strangs — "
        "Tabellen, Mediane, Drawdowns, Bandbreiten, p-Werte, DSR/PBO — stammen aus "
        "`results/p13*.json` und werden von Tests gegen die Artefakte nachgerechnet. "
        "**Zitate aus anderen Läufen** (P5, P8, Befund 6 und 7: Survivorship-Spanne, "
        "Entscheidungsmarge, Zahl recycelter Spalten, SPY-Abdeckung, die 0,9512 des "
        "Aktien-Kandidaten) sind übernommen, nicht hier neu gerechnet — sie stehen "
        f"in den dort genannten Dokumenten. Der erste Entwurf behauptete pauschal "
        f"{AUF}jede Zahl hier hat einen Schlüssel dort{ZU} und deckte damit genau "
        f"die Stelle zu, an der real gedriftet wurde (E-122).\n"
    )

    t.append("## Warum dieser Strang die Datenkritik überlebt\n")
    t.append(
        "Die Befunde 6 und 7 des Mandats haben die Datenbasis für Vergleiche gegen "
        "einen Index unbrauchbar gemacht: Survivorship 2,36–2,90 pp p. a. bei 1,5 pp "
        "Entscheidungsmarge, dazu Ticker-Recycling in 29 Spalten. Das trifft jede "
        "Strategie, die **Namen auswählt**.\n"
    )
    t.append(
        "Hier steht auf beiden Seiten derselbe Basiswert: SPY mit Filter gegen SPY "
        "ohne. Keine Auswahl, kein Survivorship, kein Recycling, keine "
        "Gewichtungsfrage. Die SPY-Serie ist geprüft sauber (99,8 % Abdeckung im "
        "Suchfenster, kein Skalenbruch, größte Lücke zwei Handelstage).\n"
    )

    t.append("## Was geprüft wurde\n")
    t.append(
        f"Dieselbe Mühle, an der der Aktien-Kandidat in P5 gescheitert ist: "
        f"{len({z['fenster'] for z in ohne['zeilen']})} Fensterwerte × "
        f"{len(alle(ohne['zeilen'], 'definition'))} Trend-Definitionen × "
        f"{len(alle(ohne['zeilen'], 'welt'))} Steuerwelten, einmal ohne und einmal "
        f"mit einem Handelstag Ausführungsverzögerung. Bestehen heißt: Median über "
        f"alle {n_win} rollierenden 10-Jahres-Fenster **über** dem von Buy-and-Hold "
        f"**und** der Drawdown-Deckel von −35 % in **keinem** Fenster gerissen.\n"
    )
    t.extend(band)
    t.append("")
    # Die Aussage wird aus der Tabelle abgeleitet, nicht behauptet: der erste
    # Entwurf schrieb „zusammenhängendes Band in allen drei Definitionen“ und
    # lag für `rendite>0` daneben (9 bestanden, längste Kette 4).
    zero_ketten = ketten_je_definition(schnitt, alle(ohne["zeilen"], "definition"))
    breit = [d for d, (_, kette) in zero_ketten.items() if kette >= MINDEST_KETTE]
    t.append(
        f"Der Aktien-Kandidat kam in P5 auf 5 und 3 von {n_fenster} Fenstern, lückig. In der "
        f"steuerfreien Welt bestehen hier "
        + ", ".join(
            f"`{d}` {n}/{n_fenster} (längste Kette {k})"
            for d, (n, k) in zero_ketten.items()
        )
        + f". {len(breit)} der {len(zero_ketten)} Definitionen tragen ein "
        f"lückenloses Band über mindestens {MINDEST_KETTE} der {n_fenster} Fensterwerte. Ein "
        f"gefundener Parameter sieht anders aus — aber `rendite>0` zeigt, dass "
        f"die Breite nicht in jeder Definition auch zusammenhängend ist.\n"
    )

    t.append("## Die Ausführungsannahme trägt das Ergebnis nicht\n")
    t.append(
        "Alle drei Gates entscheiden auf `close[t]`, und die Engine handelt am selben "
        "`close[t]` — kein Blick in die Zukunft, aber die optimistischste zulässige "
        "Annahme: der Ausstieg gelingt zu genau dem Kurs, der ihn ausgelöst hat. "
        f"Mit einem Handelstag Verzögerung bleibt das Bild bestehen (Spalte {AUF}mit "
        f"Verz.{ZU} oben); in `PRIVAT_DE`/`preis>sma` wird das Band sogar breiter. "
        "Der Vorsprung stammt also nicht aus der Ausführung.\n"
    )
    w = ereignis["warmlauf"]
    t.append(
        f"Eine Eigenheit gehört dazu genannt: Alle drei Definitionen bilden "
        f"`(a > b).astype(float)`, und NaN-Vergleiche ergeben False — die "
        f"Warmlaufphase ist **risk-off**, nicht neutral. Jede gegatete Variante "
        f"startet also in Cash, und zwar umso länger, je größer das Fenster ist: "
        f"Fenster {w['fenster_klein']} liefert ab {w['erster_gueltiger_klein']} "
        f"ein Signal, Fenster {w['fenster_gross']} erst ab "
        f"{w['erster_gueltiger_gross']} — **{w['differenz_monate']} Monate** "
        f"später. Das Fensterraster konfundiert damit Signallänge mit "
        f"anfänglicher Marktabwesenheit. Die Richtung ist konservativ (1995 ff. "
        f"war stark, wer später einsteigt, verliert), die großen Fenster sind "
        f"also eher benachteiligt.\n"
    )

    t.append(f"## Der Einwand, der bleibt: zwei Ereignisse, nicht {n_win} Fenster\n")
    t.append(
        "Der Suchzeitraum 1995–2016 enthält zwei Bärenmärkte. Ein rollierendes "
        "10-Jahres-Fenster darin startet zwischen 1995 und 2006 und trifft damit "
        "zwangsläufig mindestens einen. Gezählt statt vermutet:\n"
    )
    t.append(
        f"* Benchmark-MaxDD je Fenster: schlimmster **{v['schlimmster']:.1%}**, "
        f"Median {v['median']:.1%}, **mildester {v['mildester']:.1%}**\n"
        f"* Fenster ohne Rückgang von mindestens {abs(ereignis['schwelle']):.0%}: "
        f"**{ruhig}**\n"
        f"* Der Kandidat gewinnt {ez['krisenfenster']['gewonnen']} von "
        f"{ez['krisenfenster']['n']} Fenstern, Median-Vorsprung "
        f"{ez['krisenfenster']['median_vorsprung_pp']:+.1f} pp\n"
    )
    t.append(
        f"**Kein einziges krisenfreies Fenster.** Die Stichprobe kann {AUF}Trendfolge "
        f"wirkt{ZU} nicht von {AUF}Trendfolge hat diese beiden Abstürze umgangen{ZU} "
        f"unterscheiden. Die {n_win} Fenster überlappen zudem massiv — jeder "
        f"Handelstag steckt in bis zu {ueberlappung} von ihnen. Die effektive Stichprobe für den "
        "Mechanismus sind zwei Ereignisse.\n"
    )

    t.append("## Was das Timing wert ist (Kontrollgruppe)\n")
    t.append(
        f"Ein Gate nimmt Zeit aus dem Markt **und** wählt wann. Die Kontrolle "
        f"trennt beides: die Folge an/aus bleibt unverändert, gemischt werden nur "
        f"die **Blocklängen innerhalb ihrer Wertklasse**. Auf **Signalebene** "
        f"exakt erhalten bleiben damit der An-Anteil "
        f"({zufall['anteil_investiert']:.1%}), die Zahl der Blöcke "
        f"({zufall['n_bloecke']}, inklusive Warmlauf) und deren "
        f"Längenverteilung; verändert wird nur, **wann** die langen und kurzen "
        f"Episoden liegen. "
        f"{zufall['seeds']} Ziehungen, Parameter a priori `preis>SMA"
        f"{zufall['fenster_apriori']}` statt der besten Rasterzelle.\n"
    )
    t.append(
        f"| | Median über {n_win} Fenster |\n|---|---:|\n"
        f"| echter Filter | **{zero['echt_median']:.3f}x** |\n"
        f"| Zufalls-Timing, Median | {zero['zufall_median']:.3f}x |\n"
        f"| Zufalls-Timing, p95 | {zero['zufall_p95']:.3f}x |\n"
        f"| Zufalls-Timing, bestes von {zufall['seeds']} | {zero['zufall_bestes']:.3f}x |\n"
        f"| Buy-and-Hold | {zero['benchmark_median']:.3f}x |\n"
    )
    t.append(
        f"Auf **Portfolioebene** ist die Erhaltung nur näherungsweise, und das "
        f"ist gemessen statt angenommen: die Engine liest das Gate nur an "
        f"Monatsenden, zwischen Signal und Wirkung liegt also ein "
        f"Sampling-Schritt. Realisiert investiert war der echte Filter an "
        f"{zero['echt_investiert_anteil']:.1%} der Tage, die Zufallsläufe an "
        f"{zero['zufall_investiert_anteil_min']:.1%}–"
        f"{zero['zufall_investiert_anteil_max']:.1%} (Median "
        f"{zero['zufall_investiert_anteil_median']:.1%}). Der echte Filter ist "
        f"damit **mehr** im Markt als der typische Zufallslauf — sein Vorsprung "
        f"stammt nicht aus zusätzlicher Abwesenheit.\n"
    )
    t.append(
        f"Auf der **buchenden** Ebene ist die Kontrolle sogar im Nachteil, und "
        f"auch das ist gemessen: an den Monatsenden schaltet das echte Gate "
        f"{zero['echt_wirksame_schaltungen']}-mal, die gemischten "
        f"{zero['zufall_schaltungen_min']}- bis "
        f"{zero['zufall_schaltungen_max']}-mal (Median "
        f"{zero['zufall_schaltungen_median']:.0f}). Jede Schaltung kostet "
        f"`cost_bps`, auch in der steuerfreien Welt — die Kontrollgruppe trägt "
        f"also mehr Kostendrag als der Kandidat. Der Abstand unten ist zu groß, "
        f"als dass das ihn erklären könnte, aber das ist eine Abschätzung und "
        f"keine Bereinigung: es ist der einzige bekannte Effekt, der **zugunsten** "
        f"des Kandidaten wirkt.\n"
    )
    t.append(
        f"{zero['zufall_erreicht_echt']} von {zufall['seeds']} Zufallsläufen "
        f"erreichen den echten Filter (**p = {zero['p_wert']:.3f}**), "
        f"{zero['zufall_bestanden']}/{zufall['seeds']} bestehen die Zielfunktion. "
        f"Der Zufallsmedian liegt **unter** Buy-and-Hold: zu zufälligen Zeiten "
        f"auszusetzen kostet. Das Timing trägt also Information — innerhalb dieser "
        f"Stichprobe.\n"
    )

    t.append("## Die Mehrfachtest-Korrektur — und daran scheitert er, zweimal\n")
    dsr = korrektur["dsr"]["gewinner"]
    t.append(
        f"Familienmatrix: {korrektur['n_familie']} Varianten "
        f"({len(alle(ohne['zeilen'], 'definition'))} Definitionen × "
        f"{len({z['fenster'] for z in ohne['zeilen']})} Fenster + ungegatet), "
        f"N = {korrektur['n_kumuliert']} (kumulierter Trial-Zähler beider "
        f"Mandate). Die Entscheidungsregel stand vor dem Lauf in "
        f"`p8_dsr_heterogen.py` fest: heterogen geschätztes V, kumuliertes N, "
        f"und PBO unter 50 %.\n"
    )
    t.append(
        "| Varianzschätzer | Schwelle | p | | |\n|---|---:|---:|---|---|\n"
        + "".join(
            f"| {label} | {dsr[label]['sharpe_threshold']:.4f} | "
            f"{dsr[label]['dsr_probability']:.4f} | "
            f"{'✅' if dsr[label]['passes_5pct'] else '❌'} | "
            f"{'**Entscheidungsgrundlage**' if label == 'heterogen' else ('nicht entscheidungsfähig (E-077)' if label == 'klonfamilie' else 'konservative Gegenprobe')} |\n"
            for label in ("heterogen", "IID-Naeherung", "klonfamilie")
        )
        + f"| PBO (CSCV, 8 Blöcke, 70 Splits) | — | {korrektur['pbo']:.1%} | "
        f"{'✅' if korrektur['verdikt']['pbo_bestanden'] else '❌'} | "
        f"rangiert nach Sharpe, nicht nach dem Zielmaß |\n"
    )
    t.append(
        f"Beobachteter Sharpe {dsr['heterogen']['sharpe_observed']:.4f}. Die "
        f"Klonfamilien-Varianz ({korrektur['varianz_klonfamilie']:.3e}) ist "
        f"{korrektur['varianz_heterogen'] / korrektur['varianz_klonfamilie']:.1f}-mal "
        f"kleiner als die heterogene aus P8 "
        f"({korrektur['varianz_heterogen']:.3e}, "
        f"{korrektur['n_strategien_p8']} Strategien) — sie senkt die Schwelle von "
        f"{dsr['heterogen']['sharpe_threshold']:.4f} auf "
        f"{dsr['klonfamilie']['sharpe_threshold']:.4f} und macht aus einem "
        f"Fehlschlag ein Bestehen. Genau diese Konstruktion ist im Repo als "
        f"**E-077** protokolliert; der erste Entwurf dieses Moduls war eine "
        f"Kopie des dort verworfenen `p7_dsr_pbo.py` und hätte den Fehler in "
        f"einen neuen Befund verlängert.\n"
    )
    t.append(
        f"**Beide Korrekturen sind gerissen.** DSR p = "
        f"{dsr['heterogen']['dsr_probability']:.4f} gegen die 0,95-Schwelle, und "
        f"PBO {korrektur['pbo']:.1%}: in mehr als der Hälfte der 70 Aufteilungen "
        f"landet die in-sample beste Konfiguration out-of-sample unter dem "
        f"Median der Familie. Welches Fenster das beste ist, ist über die Zeit "
        f"nicht stabil.\n"
    )
    t.append(
        f"Drei Einordnungen, keine davon entlastend:\n\n"
        f"* **Der Vergleich mit dem Aktien-Kandidaten fällt zu Ungunsten dieses "
        f"Kandidaten aus.** Auf demselben Varianzschätzer kam jener auf 0,9512 "
        f"(bestanden, aber Münzwurf am Rand) und scheiterte erst an der "
        f"IID-Gegenprobe; dieser liegt mit "
        f"{dsr['heterogen']['dsr_probability']:.4f} deutlich darunter. Das N ist "
        f"dabei nicht dasselbe — jener wurde gegen N = 2.144 gemessen, dieser "
        f"gegen N = {korrektur['n_kumuliert']}; ein Teil der Differenz ist also "
        f"Zählerwachstum und nicht Kandidatenqualität — die 0,95-Schwelle "
        f"verfehlt dieser Kandidat aber deutlich, nicht knapp. Eine "
        f"frühere Fassung dieses Befunds stellte 0,9974 aus der Klonfamilie "
        f"neben jene 0,9512 aus der heterogenen Familie und leitete daraus eine "
        f"{AUF}Symmetrie{ZU} ab — zwei verschiedene Schätzer, verglichen "
        f"zugunsten des eigenen Kandidaten.\n"
        f"* **PBO ist hier zusätzlich nach unten verzerrt.** E-077 hält fest, "
        f"dass CSCV heterogene Spalten voraussetzt und bei Fast-Klonen zu "
        f"niedrige Werte liefert. Ein Wert von {korrektur['pbo']:.1%} unter "
        f"dieser Verzerrung ist ein deutlicherer Fehlschlag, als die Zahl "
        f"nahelegt.\n"
        f"* **In-sample-Gewinner und a-priori-Parameter fallen zusammen "
        f"(`{korrektur['gewinner']}`).** Das stützt, dass hier kein Parameter "
        f"gefunden wurde — ändert aber nichts, denn beide Korrekturen bewerten "
        f"die Suche, und gesucht wurde nachweislich "
        f"({korrektur['n_kumuliert']} Trials).\n"
    )
    t.append(
        "Die Matrix zu verkleinern oder die Klonvarianz zu behalten, bis die "
        "Zahlen passen, wäre genau die Manipulation, vor der E-077 warnt. Die "
        "Regel stand vor dem Lauf: **alle Kriterien oder keines.**\n"
    )

    t.append("## Was hier nicht behauptet wird\n")
    t.append(
        f"* **Kein Holdout.** Alles oben liegt im Suchfenster bis 2016-12-30. "
        f"Der Zeitraum 2017-01 bis 2026-07 ist unangetastet und bleibt es, bis "
        f"darüber entschieden wird.\n"
        f"* **Trials: {trials} für P13/P13b, kampagnenweit "
        f"{korrektur['n_kumuliert']}** (inklusive der {TRIALS_MANDAT_I} aus "
        f"Mandat I, die der Zähler bewusst nicht zurücksetzt). Der Wert stammt "
        f"aus dem Korrektur-Artefakt, nicht aus dem Live-Zähler — ein Dokument, "
        f"das aus Artefakten reproduzierbar sein soll, darf keinen laufenden "
        f"Zustand zitieren. P13c, P13d und P13e zählen nicht mit: Zerlegung, "
        f"Kontrollgruppe und Korrektur sind keine Suche (E-090).\n"
        f"* **Ein Markt, ein Instrument, ein Vierteljahrhundert.** Aus einem "
        f"US-Index-ETF von 1995 bis 2016 folgt nichts über andere Märkte oder "
        f"andere Regime.\n"
        f"* **Die {n_win} Fenster sind keine {n_win} Beobachtungen.** Der p-Wert der "
        f"Kontrollgruppe bezieht sich auf die Timing-Frage, nicht auf die Frage, "
        f"ob der Mechanismus außerhalb dieser Stichprobe existiert.\n"
        f"* **Bestehen heißt nicht {AUF}schlägt SPY im Endwert{ZU}.** Es heißt höherer "
        f"Median über rollierende Fenster bei eingehaltenem Drawdown-Deckel — "
        f"Buy-and-Hold reißt diesen Deckel in {n_win} von {n_win} Fenstern.\n"
    )

    t.append("## Stand: kein Holdout-Schuss\n")
    buchungen = [
        z["kaeufe"] + z["verkaeufe"] for z in list(ohne["zeilen"]) + list(mit["zeilen"])
    ]
    gmbh = [len(schnitt[f"GMBH+FK/{d}"]) for d in alle(ohne["zeilen"], "definition")]
    t.append(
        f"Der Filter übersteht die P5-Mühle als erster Kandidat der Kampagne, auf "
        f"der einzigen Datenbasis, die die Befunde 6 und 7 nicht entwerten, und "
        f"keine der billigen Widerlegungen greift. Er scheitert trotzdem an "
        f"beiden Hälften der Mehrfachtest-Korrektur — DSR "
        f"{dsr['heterogen']['dsr_probability']:.4f} und PBO "
        f"{korrektur['pbo']:.1%}. Damit hat **kein** Kandidat dieser Kampagne "
        f"die Korrektur vollständig bestanden.\n"
    )
    t.append(
        f"Zwei Dinge bleiben verwertbar:\n\n"
        f"* **Die GmbH-Frage ist beantwortet** (bestätigt Befund 3 aus anderer "
        f"Richtung): Ein Filter, der über das Raster {min(buchungen)} bis "
        f"{max(buchungen)} Buchungen erzeugt, trägt bei diesem Kapitaleinsatz "
        f"die Fixkosten der Rechtsform nicht — {min(gmbh)}/{n_fenster} bis "
        f"{max(gmbh)}/{n_fenster} bestandene Fenster.\n"
        f"* **Die Stichprobe selbst ist die Grenze.** Selbst wenn beide "
        f"Korrekturen bestanden hätten, wäre der Mechanismus nicht von zwei "
        f"Ereignissen zu trennen gewesen. Ein besserer Test bräuchte Marktdaten "
        f"vor 1995 oder andere Märkte — eine Beschaffungsfrage, keine "
        f"Forschungsfrage.\n"
    )

    ZIEL.write_text("\n".join(t), encoding="utf-8")
    print(f"-> {ZIEL}")
    for k, s in schnitt.items():
        print(f"  {k:<24} in beiden Läufen bestanden: {len(s)} {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
