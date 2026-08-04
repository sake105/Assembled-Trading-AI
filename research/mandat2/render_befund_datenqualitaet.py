"""Erzeugt BEFUND_DATENQUALITAET.md aus den Ergebnissen von P12d/P12e/P12f.

WARUM GENERIERT UND NICHT GESCHRIEBEN
-------------------------------------
Ein von Hand geschriebener Befund altert gegen seine Daten: die Zahl im Text
bleibt stehen, wenn der Lauf sich ändert. Hier hängen die **Wertungen** an
Verzweigungen — „trägt / trägt nicht", „dreht / dreht nicht" —, sodass eine
geänderte Messung die Aussage mitzieht statt sie stumm zu überleben (E-081).

Die Sätze sind deshalb bewusst so gebaut, dass keiner von ihnen ohne die
zugehörige Zahl formuliert werden kann.
"""

from __future__ import annotations

import json
from pathlib import Path

HIER = Path(__file__).resolve().parent
RES = HIER / "results"
ZIEL = HIER / "BEFUND_DATENQUALITAET.md"

#: Die Marge, um die es in der Kampagne geht.
#:
#: HERKUNFT (Stage-3-Finding F-auditor-4): dies ist die einzige handgesetzte
#: Zahl des Dokuments, und an ihr haengt die Verzweigung „traegt / traegt
#: nicht". Sie stammt aus P12: der beste Intraday-Kandidat erreichte 2.729x
#: gegen 3.138x fuer das Liegenlassen desselben Universums — ueber 10,5 Jahre
#: sind das rund 1,3 Prozentpunkte p. a.; auf 1,5 aufgerundet.
#:
#: Bei den aktuellen Zahlen ist die Verzweigung davon nicht abhaengig: die
#: Marge muesste ueber 2,36 pp steigen (+57 %), um die Aussage zu kippen. Wer
#: sie aendert, aendert eine Wertung — deshalb steht sie hier und nicht inline.
ENTSCHEIDUNGSMARGE_PP = 1.5

AUF = "„"
ZU = "“"


def dez(x: float, n: int = 2) -> str:
    return f"{x:,.{n}f}".replace(",", "#").replace(".", ",").replace("#", ".")


def pp(x: float, n: int = 2) -> str:
    return dez(x * 100.0, n)


def lade(name: str) -> dict | None:
    p = RES / f"{name}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


#: Felder, die ein Artefakt tragen MUSS, damit es zur aktuellen Codegeneration
#: gehoert. Jeder Eintrag stammt aus einer Detektor-Reparatur, nach der die
#: alten Artefakte stillschweigend weiterverwendet wurden.
PFLICHTFELDER = {
    "p12e": ("korrumpierte_namen", "unaufloesbar_grund"),
    "p12f": ("gegenprobe", None),
}


def pruefe_artefakt_aktuell(e: dict, f: dict) -> list[str]:
    """Stammen die Artefakte aus DERSELBEN Codegeneration?

    Ein generiertes Dokument garantiert Konsistenz zwischen Zahl und Satz —
    nicht zwischen Artefakt und Code. Als ``korruptions_spannen`` repariert
    wurde, blieb das Ergebnis-JSON des zweiten Konsumenten stehen: Abschnitt 2
    des Befunds kam aus dem alten Detektor (246 Uebergangstage), Abschnitt 4 aus
    dem neuen. Ein Dokument, zwei Detektorgenerationen, beide Zahlen plausibel
    (Stage-2-Finding F-senior-1).

    Geprueft wird deshalb strukturell, nicht ueber Zeitstempel: traegt jeder
    Eintrag die Felder, die der aktuelle Detektor erzeugt? Fehlt eines, ist das
    Artefakt veraltet — dann wird abgebrochen statt gerendert.
    """
    maengel: list[str] = []
    namen = e.get("korrumpierte_namen") or {}
    if not namen:
        maengel.append("p12e: keine korrumpierten Namen im Artefakt")
    else:
        beispiel = next(iter(namen.values()))
        for feld in ("unaufloesbar", "unaufloesbar_grund", "n_tage_falsch"):
            if feld not in beispiel:
                maengel.append(
                    f"p12e: Feld '{feld}' fehlt — Artefakt stammt von einer "
                    "aelteren Fassung von korruptions_spannen"
                )
    fehlend = set(f.get("unaufloesbar") or []) - set(namen)
    if fehlend:
        maengel.append(
            f"p12f kennt unaufloesbare Namen, die p12e nicht meldet: "
            f"{sorted(fehlend)} — die beiden Laeufe sahen verschiedene Panels"
        )
    if "gegenprobe" not in f:
        maengel.append("p12f: 'gegenprobe' fehlt — Artefakt veraltet")
    return maengel


def kipp_abstand(f: dict) -> tuple[dict, float]:
    """Wie weit war der Ausgang vom Kippen entfernt — und wie stark wirkt die
    Bereinigung ueberhaupt?

    Ein Test, dessen Ergebnis per Konstruktion nicht kippen kann, ist keine
    Entwarnung; er misst nichts. Genau das lag hier vor: KEINE der 24
    Parametrisierungen haelt den Drawdown-Deckel, in keinem Panel und keiner
    Steuerwelt. „Dreht nicht" war deshalb richtig, „robust" aber unbelegt
    (Stage-3-Finding F-auditor-1).

    Belastbar wird die Aussage durch zwei Zahlen: den kleinsten Abstand zum
    Deckel ueber alle Zeilen, und die groesste Verschiebung, die die Bereinigung
    ueberhaupt bewirkt. Deren Verhaeltnis sagt, um welchen Faktor die Reparatur
    haette staerker sein muessen, damit sich etwas aendert.

    Rueckgabe: (Kennzahlen zum Abstand, groesste |Delta MaxDD|).
    """
    deckel = float(f.get("dd_deckel", -0.35))
    bester_dd = max(
        z["schlimmster_maxdd"]
        for v in f["welten"].values()
        for p in ("original", "bereinigt")
        for z in v[p]["zeilen"]
    )
    wirkung = 0.0
    for v in f["welten"].values():
        schluessel = ("haltetage", "rank_out", "hebel")
        orig = {tuple(z[k] for k in schluessel): z for z in v["original"]["zeilen"]}
        for z in v["bereinigt"]["zeilen"]:
            o = orig.get(tuple(z[k] for k in schluessel))
            if o is not None:
                wirkung = max(
                    wirkung, abs(z["schlimmster_maxdd"] - o["schlimmster_maxdd"])
                )
    # Betrag: der beste Kandidat liegt UNTER dem Deckel, der Abstand ist die
    # Strecke dorthin. Ohne abs() kaeme die Zahl negativ heraus und der Faktor
    # gleich mit — eine Kennzahl, die die eigene Aussage umdreht.
    abstand_pp = abs(bester_dd - deckel) * 100.0
    return (
        {
            "bester_dd": bester_dd,
            "deckel": deckel,
            "abstand_pp": abstand_pp,
            "faktor": abstand_pp / (wirkung * 100.0) if wirkung > 0 else float("inf"),
        },
        wirkung,
    )


def zeilen_mit_wechsel(f: dict, feld: str) -> dict[str, int]:
    """Wie viele EINZELNE Parametrisierungen wechseln ihren Status?

    `schlaegt_dreht` war als 0-gegen-nicht-0 definiert und konnte bei
    Startwerten von 6/2/2 nie feuern — der zugehoerige Absatz blieb stumm,
    obwohl das schwaechere Kriterium sehr wohl reagiert (PRIVAT_DE 2 -> 4).
    Auf Zeilenebene gezaehlt wird die Empfindlichkeit sichtbar
    (Stage-3-Finding F-auditor-2).
    """
    aus: dict[str, int] = {}
    schluessel = ("haltetage", "rank_out", "hebel")
    for welt, v in f["welten"].items():
        orig = {tuple(z[k] for k in schluessel): z for z in v["original"]["zeilen"]}
        n = 0
        for z in v["bereinigt"]["zeilen"]:
            o = orig.get(tuple(z[k] for k in schluessel))
            if o is not None and bool(o[feld]) != bool(z[feld]):
                n += 1
        if n:
            aus[welt] = n
    return aus


def abschnitt_g(g: dict) -> list[str]:
    """Kann der Endpunkt die Lücke überhaupt schließen? — API-Probe, kein ``ls``.

    Das Verdikt kommt aus ``p12g_pull_bilanz.verdikt`` — derselben Funktion, die
    auch das Skript benutzt. Vorher entschieden beide getrennt, und beide
    keyten auf „mindestens ein stummer Name" statt auf „Evidenz vollständig":
    bei lauter fehlgeschlagenen Abfragen meldete das eine „der Weg ist gangbar",
    bei einem stummen von sechs das andere „die Ausscheider sind nicht zu
    haben" (Stage-3-Finding F-auditor-1).
    """
    from research.mandat2.p12g_pull_bilanz import verdikt

    p = g.get("api_probe")
    b = g.get("bilanz") or {}
    v = verdikt(p)
    zeilen = [
        "## 5. Kann der Intraday-Endpunkt die Lücke überhaupt schließen?",
        "",
        "Nach Abschnitt 1 lag der Schluss nahe, einfach mehr Symbole zu ziehen.",
        f"Genau das ist geschehen — {tsd(b.get('n_dateien', 0))} Dateien liegen",
        "inzwischen vor. Ob das zum Ziel führt, beantwortet aber nicht das",
        "Dateiverzeichnis, sondern nur eine Abfrage.",
        "",
    ]
    if v["status"] == "keine_probe":
        return zeilen + [
            "**Ohne API-Probe ist hier nichts auszusagen.** Ein fehlender",
            "Datensatz auf der Platte kann bedeuten: nie angefragt, angefragt",
            "und leer, oder Anfrage fehlgeschlagen — drei Zustände, die das",
            "Verzeichnis nicht unterscheidet (E-112).",
        ]
    if v["status"] == "unvollstaendig":
        return zeilen + [
            "**Die Probe ist unvollständig — es wird kein Befund berichtet.**",
            f"Fehlgeschlagene Abfragen: {v['fehler'] or 'keine'}; Kontrollgruppe",
            f"liefert {v['n_kontrolle_lebt']} von {v['n_kontrolle']}. Ein",
            "Negativbefund wäre hier nicht von einem kaputten Aufruf zu",
            "unterscheiden, und ein Fehlerstring ist keine Messung.",
        ]

    aus = p["ausscheider"]
    kontrolle = p["kontrolle"]
    lebt = [n for n in kontrolle.values() if isinstance(n, int) and n > 0]
    zeilen += [
        "Geprüft wurden Ausscheider des Suchfensters — jeweils unter dem Symbol,",
        "unter dem sie **damals im Index standen**, und zusätzlich unter dem",
        "Post-Insolvenz-Ticker. Das ist nicht dasselbe: die Q-Ticker entstehen",
        "erst mit dem Chapter-11-Handel, ein Negativbefund auf ihnen ist fast",
        "garantiert und beweist nichts (E-113). Gezählt werden **Minutenbars**.",
        "",
        "| Name | Symbol damals | Bars | Q-Ticker | Bars |",
        "|---|---|---:|---|---:|",
    ]
    for sym, x in aus.items():
        qt = x["q_ticker"] or "—"
        qb = "—" if x["bars_q_ticker"] is None else tsd(x["bars_q_ticker"])
        zeilen.append(
            f"| {x['name']} | {sym} | {tsd(x['bars_mitgliedschaftssymbol'])} | "
            f"{qt} | {qb} |"
        )
    zeilen += [
        "",
        f"**Kontrollgruppe:** dieselbe Abfrage für Überlebende — "
        f"{v['n_kontrolle_lebt']} von {v['n_kontrolle']} liefern Bars "
        f"({tsd(min(lebt))}–{tsd(max(lebt))} im Probefenster).",
        "Der Aufruf funktioniert also; das Schweigen bei den Ausscheidern ist",
        "kein Fehler der Abfrage. Die Kontrolle liegt zudem im **früheren**",
        "Fenster als alle Ausscheider — eine reine Datumsgrenze scheidet damit",
        "als Erklärung aus.",
        "",
    ]
    if v["status"] == "weg_zu":
        zeilen += [
            f"> **Alle {v['n_ausscheider']} geprüften Ausscheider liefern keine",
            "> einzige Bar** — auch nicht unter ihrem Handelssymbol vor der",
            "> Insolvenz. Für die Survivorship-Korrektur ist dieser Weg zu.",
            "",
            "Mehr Anfragen erhöhen also die **Abdeckung** des Universums (viele",
            "Überlebende sind schlicht noch nicht gezogen), aber nicht seine",
            "**Unverzerrtheit**: die geprüften Ausscheider sind bei dieser Quelle",
            "nicht zu haben. Dafür braucht es Tagesdaten mit Delisting-Kursen.",
        ]
    elif v["status"] == "teilweise":
        zeilen += [
            f"**{v['n_stumm']} von {v['n_ausscheider']} geprüften Ausscheidern**",
            "liefern nichts — die übrigen schon. Die Quelle ist also nicht",
            "pauschal blind; welche Namen sie führt, ist einzeln zu prüfen. Ein",
            "verallgemeinernder Schluss wäre hier nicht gedeckt.",
        ]
    else:
        zeilen += [
            "Alle geprüften Ausscheider liefern Bars — der Weg über mehr",
            "Anfragen ist gangbar.",
        ]
    zeilen += [
        "",
        f"*Der Stand des bisherigen Pulls ({pp(b.get('abdeckung', 0.0), 1)} % der",
        "PIT-Mitglieder) beschreibt die Zusammensetzung der bisherigen",
        "Anfrageliste, nicht den Endpunkt — er wird hier bewusst nicht als",
        "Verzerrungsmaß ausgewiesen (Stage-2-Findings F-senior-1/7).*",
    ]
    return zeilen


def ueberhoehung(d: dict) -> tuple[float, float]:
    """Die Survivorship-Ueberhoehung als SPANNE in Prozentpunkten p. a.

    P12d rechnet zwei Delisting-Behandlungen (halten / umschichten) und weist
    beide aus, weil eine einzelne Zahl hier Scheinpraezision waere (E-078).
    Der Befund muss dieselbe Spanne fuehren.

    Entschieden wird am **unteren** Rand: liegt schon der guenstigste Fall
    ueber der Marge, ist die Aussage eindeutig. Eine fruehere Fassung nutzte an
    einer Stelle `max`, an der anderen nur `cagr_halten` — dasselbe Dokument
    konnte dadurch in der Kurzfassung entwarnen und in Abschnitt 1 Alarm
    schlagen (Stage-1-Finding F8).
    """
    u = d["ueberhoehung_cagr"]
    a = u["cagr_halten"] * 100.0
    b = u["cagr_umschichten"] * 100.0
    return (min(a, b), max(a, b))


def abschnitt_d(d: dict) -> list[str]:
    # Das INTRADAY-Universum ist die verzerrte Stichprobe; verglichen wird gegen
    # das PIT-Universum, das Ausscheider mitfuehrt. Beide werden mit demselben
    # Verfahren gerechnet — nur so misst die Differenz die Auswahl und nicht
    # nebenbei Gewichtung oder Indexkonstruktion.
    z = next(x for x in d["zeilen"] if x["universum"] == "intraday_p12")
    pit = next((x for x in d["zeilen"] if x["universum"].startswith("pit")), None)
    lo, hi = ueberhoehung(d)
    traegt = lo < ENTSCHEIDUNGSMARGE_PP
    bezug = pit["universum"] if pit else "PIT-Universum"
    zeilen = [
        "## 1. Wie viel Rendite kommt allein aus der Auswahl der Namen?",
        "",
        "Das Intraday-Universum von P12 besteht aus Namen, die **heute** noch",
        "handelbar sind. Wer 2006 in genau diese Namen investierte, wusste 2006",
        "nicht, dass sie überleben würden. Die Frage ist, wie groß dieser",
        "Vorteil ist — nicht, ob es ihn gibt.",
        "",
        f"Gemessen über {dez(d['jahre'], 1)} Jahre ({d['fenster']}), "
        "gleichgewichtet, gleiches Verfahren für alle Zeilen:",
        "",
        "| Universum | Namen | Endwert (halten) | CAGR (halten) | CAGR (umschichten) | MaxDD |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for x in d["zeilen"]:
        zeilen.append(
            f"| {x['universum']} | {x['n']} | {dez(x['halten']['endwert'])}× | "
            f"{pp(x['halten']['cagr'])} % | {pp(x['umschichten']['cagr'])} % | "
            f"{pp(x['halten']['maxdd'], 1)} % |"
        )
    zeilen += [
        f"| SPY (Referenz) | 1 | {dez(d['spy']['endwert'])}× | "
        f"{pp(d['spy']['cagr'])} % | — | {pp(d['spy']['maxdd'], 1)} % |",
        "",
        f"**Überhöhung: {dez(lo)} bis {dez(hi)} Prozentpunkte p. a.** gegenüber",
        f"dem survivorship-freien **{bezug}** — allein aus der Zusammensetzung",
        "des Universums, ohne jede Strategie, ohne jedes Signal, bei reinem",
        "Liegenlassen. Die Spanne kommt daher, dass P12d zwei Delisting-",
        "Behandlungen rechnet; eine einzelne Zahl wäre hier Scheinpräzision.",
        "",
        f"Gegen **SPY** wäre die Zahl mit "
        f"{dez((z['halten']['cagr'] - d['spy']['cagr']) * 100.0)} pp noch größer.",
        "Der Vergleich gegen das PIT-Universum ist aber der ehrlichere: er",
        "isoliert die Auswahl, während gegen SPY zusätzlich Gewichtung und",
        "Indexkonstruktion mitgemessen würden (vgl. E-079).",
        "",
    ]
    if traegt:
        zeilen += [
            f"Das liegt **unter** der Entscheidungsmarge von rund "
            f"{dez(ENTSCHEIDUNGSMARGE_PP, 1)} pp p. a., um die es in der",
            "Kampagne geht. Die Verzerrung ist damit nicht groß genug, um die",
            "Verdikte allein zu erklären.",
        ]
    else:
        zeilen += [
            f"Die Kampagne entscheidet Fragen im Bereich von rund "
            f"{dez(ENTSCHEIDUNGSMARGE_PP, 1)} Prozentpunkten p. a.",
            f"Schon der **untere** Rand der Spanne liegt mit {dez(lo)} pp",
            "**über der Marge, um die gestritten wird.**",
            "",
            "> Damit ist die entscheidende Aussage dieses Dokuments erreicht:",
            f"> **Der Datensatz kann die Frage nicht entscheiden.** Nicht {AUF}die",
            f"> Strategie verliert{ZU} und nicht {AUF}die Strategie gewinnt{ZU} — die",
            "> Datengrundlage trägt das Urteil in dieser Größenordnung nicht.",
            "",
            "Das ist keine Formalie. Ein Ergebnis, dessen Vorzeichen von einer",
            "Verzerrung abhängt, die größer ist als der gemessene Effekt, ist",
            "kein Ergebnis.",
        ]
    tote = d.get("tote_ticker_im_pit_universum") or []
    if tote:
        zeilen += [
            "",
            f"Zur Einordnung: das PIT-Universum enthält Insolvenzticker "
            f"({', '.join(tote)}),",
            "das Intraday-Universum keinen einzigen. Der Unterschied ist keine",
            "Feinheit der Stichprobe, sondern ihre Konstruktion.",
        ]
    return zeilen


def abschnitt_e_kanaele(e: dict) -> list[str]:
    halte = e["halte_kanal"]
    ausw = e["auswahl_kanal"]
    n_pl = e["auswahlplaetze_kanal_b"]
    anteil = e["anteil_plaetze_kanal_b"]
    zeilen = [
        "## 2. Sind die kaputten Kurse in die Ergebnisse eingegangen?",
        "",
        f"Im Panel liegen **{len(e['korrumpierte_namen'])} Namen** mit",
        f"Skalenbrüchen: {tsd(e['uebergangstage_gesamt'])} Übergangstage,",
        f"{tsd(e['tage_auf_falscher_skala_gesamt'])} Tage auf einer falschen",
        "Skala. Die Frage ist nicht, ob es sie gibt, sondern ob die Strategie",
        "sie **angefasst** hat. Dafür zwei getrennte Kanäle:",
        "",
        "**Kanal A — gehalten über einen Übergangstag.** Gemessen am echten",
        "Bestand der Engine, nicht an der Auswahl: die Turnover-Bremse hält",
        "Namen über die Auswahl hinaus, ein Auswahl-Proxy hätte hier",
        "entwarnt, wo keine Entwarnung war (E-102).",
        "",
        "| Name | Tage | größte Wirkung auf die Tagesrendite | Rang unter allen Tagen |",
        "|---|---:|---:|---:|",
    ]
    for sym, v in sorted(
        halte.items(), key=lambda kv: -kv[1]["groesste_wirkung_betrag"]
    ):
        zeilen.append(
            f"| {sym} | {v['n_tage']} | {pp(v['groesste_wirkung'], 2)} % "
            f"({v['groesste_wirkung_tag']}) | "
            f"{tsd(v['rang_unter_allen_tagen'])} von {tsd(v['n_handelstage'])} |"
        )
    # Nach dem BETRAG der Wirkung, nicht nach dem Rang: `-rang` haette immer den
    # kleinsten Rang gewaehlt und damit strukturell nur positive Extremtage
    # erreichen koennen — einseitige Auswahl in einem Dokument, das anderswo auf
    # Zweiseitigkeit besteht (Stage-1-Finding F11).
    s_sym, s_v = max(halte.items(), key=lambda kv: kv[1]["groesste_wirkung_betrag"])
    # „Extrem" heisst hier: nah an einem der beiden Enden der Rangliste.
    s_rang = min(
        s_v["rang_unter_allen_tagen"],
        s_v["n_handelstage"] - s_v["rang_unter_allen_tagen"] + 1,
    )
    zeilen += [
        "",
        f"{s_sym} lag an seinem Übergangstag auf Rang "
        f"{tsd(s_v['rang_unter_allen_tagen'])} von "
        f"{tsd(s_v['n_handelstage'])} — das ist der "
        f"**{tsd(s_rang)}-extremste Tag** der ganzen Kampagne, gemessen vom",
        f"näheren Ende der Rangliste. Der Tag mit "
        f"{pp(s_v['groesste_wirkung'], 2)} % Portfolio-Rendite ist also nicht",
        "irgendein Tag. Ein Vendor-Fehler an dieser Stelle ist kein Rauschen.",
        "",
        "**Kanal B — mit kontaminiertem Momentum-Score gewählt.** Der Score ist",
        f"`close.shift({e['momentum_fenster_handelstage'][0]}) / "
        f"close.shift({e['momentum_fenster_handelstage'][1]})` — ein Quotient",
        "aus **zwei** Stützstellen, kein Fenster. Kontaminiert ist er genau",
        "dann, wenn die beiden Beine auf **verschiedenen** Skalen liegen; liegen",
        "beide auf derselben falschen Skala, kürzt sich der Faktor heraus",
        "(E-104).",
        "",
        f"Betroffen: **{len(ausw)} Namen, {n_pl} von "
        f"{tsd(e['auswahlplaetze_gesamt'])} Auswahlplätzen "
        f"({pp(anteil, 2)} %).**",
        "",
    ]
    for sym, tage in sorted(ausw.items()):
        zeilen.append(f"* {sym}: {len(tage)} Termine ({tage[0]} … {tage[-1]})")
    zeilen += [
        "",
        f"Beide Kanäle sind klein. {AUF}Klein{ZU} ist aber keine Antwort auf",
        f"{AUF}dreht es ein Verdikt?{ZU} — diese Frage beantwortet nur ein",
        "Neulauf (Abschnitt 4).",
    ]
    return zeilen


def tsd(n: int) -> str:
    return f"{int(n):,}".replace(",", ".")


def abschnitt_e_abdeckung(e: dict) -> list[str]:
    ab = e["abdeckung"]
    an = e["austritts_anreicherung"]
    faktor = an["anreicherungsfaktor"]
    zeilen = [
        "## 3. Was fehlt im Panel — und fehlt es zufällig?",
        "",
        f"Von den Indexmitgliedern haben im Median nur **{pp(ab['median'], 1)} %**",
        f"überhaupt eine Preisspalte (Spanne {pp(ab['min'], 1)} % bis "
        f"{pp(ab['max'], 1)} %). Fehlende Spalten wären harmlos, wenn sie",
        "zufällig fehlten. Sie fehlen nicht zufällig:",
        "",
        f"Von den {tsd(an['n_mit_spalte'])} Mitgliedern **mit** Preisspalte am",
        f"{an['referenz_start']} sind am {an['referenz_ende']} noch",
        f"**{pp(an['ueberlebensquote_mit_spalte'], 1)} %** im Index. Von den",
        f"{tsd(an['n_ohne_spalte'])} Mitgliedern **ohne** Preisspalte nur",
        f"**{pp(an['ueberlebensquote_ohne_spalte'], 1)} %**.",
        "",
        f"> **Anreicherungsfaktor {dez(faktor)}×.** Das Fehlen einer Preisspalte",
        "> ist kein technischer Zufall — es sagt voraus, dass der Name",
        "> ausscheidet. Das Panel verliert also bevorzugt die Verlierer.",
        "",
        "Diese Verzerrung wirkt in dieselbe Richtung wie die aus Abschnitt 1",
        "und ist von ihr **nicht unabhängig**: beide entstehen daraus, dass",
        "Ausscheider schlechter dokumentiert sind als Überlebende. Die",
        "Größenordnungen dürfen deshalb nicht addiert werden — wohl aber",
        "gilt: die Schranke aus Abschnitt 1 ist eher eine Unter- als eine",
        "Obergrenze.",
    ]
    return zeilen


def abschnitt_f(f: dict) -> list[str]:
    dreht = [w for w, v in f["welten"].items() if v["verdikt_dreht"]]
    wandert = [w for w, v in f["welten"].items() if v["optimum_wandert"]]
    unaufl = f.get("unaufloesbar") or []
    zeilen = [
        "## 4. Dreht ein Verdikt, wenn man die Brüche repariert?",
        "",
        "Repariert wird durch **Spleißen**: innerhalb einer Korruptionsspanne",
        "liegen die Kurse um einen konstanten Faktor daneben, geteilt durch",
        "diesen Faktor liegen sie wieder auf der Basisskala. Je Spanne werden",
        "damit genau zwei Renditen ersetzt — die an den beiden Rändern —, alle",
        "übrigen bleiben bis auf Maschinengenauigkeit erhalten, weil sich ein konstanter Faktor",
        "im Quotienten herauskürzt.",
        "",
        f"Bereinigt: **{f['n_symbole_bereinigt']} Namen, {f['n_spannen']} Spannen.**",
    ]
    if unaufl:
        # Der Grund gehoert dazu: „verschraenkte Skalen" stimmte fuer 12 von 13,
        # WFT ist ausschliesslich ueber den Vendor-Sentinel unauflösbar. Das
        # GENERIERTE Dokument war hier ungenauer als das handgeschriebene
        # (Stage-2-Finding F-senior-6).
        gruende = f.get("unaufloesbar_grund") or {}
        verschr = sorted(n for n in unaufl if "verschraenkt" in gruende.get(n, ""))
        # PRAEVALENZ, nicht Partition: drei Namen tragen Sentinel UND
        # Verschraenkung. Wer nur  zaehlt, weist einen
        # statt vier aus und untertreibt die Klasse um Faktor 4 — in einem
        # Absatz, der Ehrlichkeit ueber Teilreparaturen einfordert
        # (Stage-3-Finding F-auditor-3).
        sent = sorted(n for n in unaufl if "sentinel" in gruende.get(n, ""))
        nur_sent = sorted(n for n in unaufl if gruende.get(n, "") == "sentinel")
        zeilen += [
            f"Nicht bereinigt: **{len(unaufl)} Namen** ({', '.join(unaufl)}).",
        ]
        if verschr:
            zeilen += [
                f"Bei {len(verschr)} davon sind die Skalen **verschränkt** — nach",
                "einem Sprung folgt ein weiterer, ohne dass die Rückkehr zum",
                "ersten passt; dort ist nicht bestimmbar, welcher Kurs auf welcher",
                "Skala liegt.",
            ]
        if sent:
            zusatz = (
                f" — {len(sent) - len(nur_sent)} davon zusätzlich verschränkt"
                if len(sent) > len(nur_sent)
                else ""
            )
            zeilen += [
                f"**{len(sent)} Namen tragen den Sättigungswert des",
                f"Datenlieferanten** (999.999,9999): {', '.join(sent)}{zusatz}.",
                "Das ist kein Kurs, und Konstante geteilt durch Konstante ist kein",
                "Spleiß.",
            ]
        zeilen += [
            "Alle bleiben unberührt und stehen im Protokoll. Eine Bereinigung,",
            "die einen Teil repariert und das sagt, ist ehrlicher als eine, die",
            "alles zu reparieren behauptet (E-107).",
        ]
    gp = f.get("gegenprobe")
    if gp:
        anteil = gp["beseitigt"] / max(gp["auffaellig_original"], 1)
        zeilen += [
            "",
            "**Gegenprobe in beide Richtungen.** Eine Reparatur, die ihre eigenen",
            "Nebenwirkungen nicht misst, ist eine zweite, unbeobachtete",
            "Datenquelle — und zwar genau dort, wo die Frage entschieden wird:",
            "",
            f"| auffällige Tage im Original | {tsd(gp['auffaellig_original'])} |",
            "|---|---:|",
            f"| davon beseitigt | {tsd(gp['beseitigt'])} |",
            f"| **neu entstanden** | **{tsd(gp['neu_entstanden'])}** |",
            f"| bleiben auffällig | {tsd(gp['auffaellig_bereinigt'])} |",
            "",
        ]
        if gp["neu_entstanden"]:
            zeilen += [
                f"**{tsd(gp['neu_entstanden'])} neue Ausreißer entstanden durch die",
                "Bereinigung selbst.** Das Ergebnis dieses Abschnitts ist damit",
                "nicht verwendbar.",
            ]
        else:
            zeilen += [
                "Kein einziger Ausreißer ist durch die Bereinigung entstanden —",
                "geprüft nach oben (>+100 %) **und** nach unten (<−50 %). Ein",
                "einseitiger Wächter hätte die halbe Fehlerklasse durchgelassen.",
                "",
                f"Beseitigt werden damit {pp(anteil, 0)} % der auffälligen Tage;",
                f"{tsd(gp['auffaellig_bereinigt'])} bleiben stehen. Die Bereinigung",
                f"ist eine **Untergrenze**, kein sauberes Panel — {AUF}bereinigt{ZU}",
                "heißt hier: die eindeutig auflösbaren Skalenbrüche sind weg.",
            ]
    abw = f.get("dividendenrendite_max_abweichung")
    if abw is not None:
        zeilen += [
            "",
            "Die Dividenden wurden mitskaliert; die Invariante *Dividende je",
            f"Kurseinheit* ändert sich um maximal {abw:.1e} — Rundungsrauschen.",
            "Ohne diese Mitskalierung stiege die implizite Dividendenrendite in",
            "der Spanne um genau den Spleißfaktor (bei WIN von 26 % auf 274 %),",
            "und der Vergleich zweier Panels wäre unfair auf genau der Achse, um",
            "die es in der GmbH-Frage geht.",
        ]
    zeilen += [
        "",
        "Gerechnet wird **dasselbe Parametergitter wie in P2** — kein neuer",
        "Parameter, keine neue Suche, der Trial-Zähler bleibt unverändert",
        "(E-090).",
        "",
        "Gemessen wird die **Zielfunktion der Kampagne**, nicht der Endwert:",
        "Median über alle rollierenden 10-Jahres-Fenster gegen den Benchmark,",
        f"unter der bindenden Nebenbedingung MaxDD ≥ {pp(f.get('dd_deckel', -0.35), 0)} %",
        "in *jedem* Fenster. Ein Endwertvergleich hätte hier eine andere Frage",
        "beantwortet: P2 hielt ausdrücklich fest, dass der beste Kandidat den",
        "Index **bei der Rendite schlägt** und an der Nebenbedingung scheitert.",
        "",
        "| Steuerwelt | Panel | Median bester | Median Bench | schlimmster DD | schlägt | **besteht** |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for welt, v in f["welten"].items():
        for panel in ("original", "bereinigt"):
            e = v[panel]
            n = len(e["zeilen"])
            b = e["bester"]
            zeilen.append(
                f"| {welt} | {panel} | {dez(b['median_kandidat'], 3)} | "
                f"{dez(b['median_benchmark'], 3)} | "
                f"{pp(b['schlimmster_maxdd'], 1)} % | "
                f"{e['n_schlagen_bench']}/{n} | **{e['n_bestanden']}/{n}** |"
            )
    zeilen.append("")
    if dreht:
        zeilen += [
            f"**Das Verdikt dreht in {', '.join(dreht)}.** Ob eine",
            "Parametrisierung Zielfunktion und Drawdown-Deckel gemeinsam besteht,",
            "hängt damit an Vendor-Preisfehlern. Alle betroffenen Phasen sind neu",
            "zu bewerten.",
        ]
    else:
        # WARUM NICHT EINFACH „ROBUST" (Stage-3-Finding F-auditor-1)
        # ---------------------------------------------------------
        # Hier stand: „ist gegen die Preisfehler robust". Das war aus dem Lauf
        # nicht belegt, sondern gesaettigt — KEINE Parametrisierung haelt den
        # Deckel, in keinem Panel. Ein Test, dessen Ergebnis per Konstruktion
        # nicht kippen kann, entwarnt nicht; er misst nichts. Genau der Vorwurf,
        # den dieses Dokument unter E-102 an anderer Stelle erhebt.
        # Belastbar wird die Aussage erst mit dem ABSTAND zum Kipppunkt.
        abstand, wirkung = kipp_abstand(f)
        zeilen += [
            "**Das Verdikt dreht in keiner Steuerwelt.** Das ist allerdings keine",
            "Robustheitsaussage, solange nicht dabeisteht, wie weit der Ausgang vom",
            "Kippen entfernt war:",
            "",
            "* Der **beste** schlimmste Drawdown über alle Parametrisierungen und",
            f"  beide Panels liegt bei {pp(abstand['bester_dd'], 1)} % — der Deckel",
            f"  fordert {pp(f.get('dd_deckel', -0.35), 0)} %. Der beste Kandidat",
            f"  verfehlt ihn also um **{dez(abstand['abstand_pp'], 1)} Prozentpunkte.**",
            "* Die Bereinigung verschiebt den schlimmsten Drawdown um höchstens",
            f"  **{dez(wirkung * 100.0, 2)} Prozentpunkte.**",
            "",
            f"Sie hätte also rund **{dez(abstand['faktor'], 1)}-mal stärker** wirken",
            "müssen, um auch nur eine einzige Zeile über den Deckel zu heben — und",
            "das in die richtige Richtung. Der Ausgang",
            "konnte an dieser Stelle nicht kippen — das ist die ehrliche Fassung",
            f"von {AUF}robust{ZU}, und sie ist stärker, weil sie die Auflösung",
            "des Tests",
            "mitliefert.",
            "",
            "Das ist eine Aussage über die **Preisfehler**, nicht über die",
            "Datenqualität insgesamt. Die Survivorship-Schranke aus Abschnitt 1",
            "bleibt davon vollständig unberührt — sie ist der größere Posten",
            "und durch keine Bereinigung erreichbar.",
        ]
    # Auf ZEILENEBENE, nicht als 0-gegen-nicht-0: der alte Schalter konnte bei
    # Startwerten von 6/2/2 nie feuern, und der Absatz blieb stumm, obwohl das
    # schwaechere Kriterium messbar reagiert (Stage-3-Finding F-auditor-2).
    wechsel = zeilen_mit_wechsel(f, "schlaegt_bench")
    if wechsel:
        gesamt = sum(wechsel.values())
        je = ", ".join(f"{w}: {n}" for w, n in sorted(wechsel.items()))
        zeilen += [
            "",
            "**Das schwächere Kriterium reagiert dagegen sehr wohl.** Lässt man den",
            "Drawdown-Deckel weg und fragt nur, ob der Benchmark bei der Rendite",
            f"geschlagen wird, wechseln **{gesamt} einzelne Parametrisierungen**",
            f"ihren Status ({je}) — ohne dass sich an der Gesamtaussage etwas",
            "ändert, weil keine von ihnen den Deckel hält.",
            "",
            "Das ist der eigentliche Beleg dafür, dass die Preisfehler wirken: sie",
            "verschieben die Rangfolge messbar. Sie verschieben sie nur nicht weit",
            "genug, um an der bindenden Nebenbedingung etwas zu ändern.",
        ]
    if wandert:
        # WELCHE Dimension wandert? „Das Optimum wandert" ist zu grob: P2 stuetzte
        # seinen Schluss auf Haltedauer und rank_out und wies die Hebelwahl
        # ausdruecklich als unterschiedlich aus. Wandert nur der Hebel, ist der
        # Schluss unberuehrt — wandert die Haltedauer, ist er es nicht.
        dims: set[str] = set()
        for w in wandert:
            o = f["welten"][w]["original"]["bester"]
            b = f["welten"][w]["bereinigt"]["bester"]
            dims |= {k for k in ("haltetage", "rank_out", "hebel") if o[k] != b[k]}
        kern = dims & {"haltetage", "rank_out"}
        namen = {
            "haltetage": "Mindesthaltedauer",
            "rank_out": "`rank_out`",
            "hebel": "Hebel",
        }
        zeilen += [
            "",
            f"Das Optimum wandert in: {', '.join(wandert)} — und zwar",
            f"ausschließlich in der Dimension "
            f"{', '.join(sorted(namen[x] for x in dims))}.",
        ]
        if kern:
            zeilen += [
                "Das betrifft **die Handelsweise selbst**. P2 hatte aus dem",
                "unbewegten Optimum geschlossen, dass die Steuer nicht die",
                "bindende Restriktion ist — dieser Schluss steht damit auf",
                "tönernen Füßen.",
            ]
        else:
            # Wie eng war das Rennen im Original? Eine Rangfolge, die an einem
            # halben Prozent haengt, ist kein Befund, der kippen KANN.
            enge = []
            for w in wandert:
                z = sorted(
                    f["welten"][w]["original"]["zeilen"],
                    key=lambda r: -r["median_kandidat"],
                )
                if len(z) > 1 and z[1]["median_kandidat"] > 0:
                    enge.append(z[0]["median_kandidat"] / z[1]["median_kandidat"] - 1.0)
            # Der Satz behauptete das frueher unbedingt — er stand im else-Zweig,
            # ohne je berechnet worden zu sein (Stage-1-Finding F9). Jetzt wird
            # ueber ALLE Welten und BEIDE Panels geprueft, nicht nur ueber die
            # wandernden.
            kombis = {
                (v[p]["bester"]["haltetage"], v[p]["bester"]["rank_out"])
                for v in f["welten"].values()
                for p in ("original", "bereinigt")
            }
            if len(kombis) == 1:
                halt, out = next(iter(kombis))
                zeilen += [
                    f"**Mindesthaltedauer ({halt} Tage) und `rank_out` ({out}) sind",
                    "in allen Steuerwelten und in beiden Panels identisch.** Genau",
                    "darauf stützte P2 den Schluss, dass nicht die Steuer, sondern",
                    "der Turnover die bindende Restriktion ist — dieser Schluss",
                    "überlebt die Bereinigung unverändert.",
                ]
            else:
                zeilen += [
                    "Mindesthaltedauer und `rank_out` sind **nicht** über alle",
                    f"Steuerwelten identisch — gefunden wurden {len(kombis)}",
                    f"verschiedene Kombinationen: {sorted(kombis)}. Der P2-Schluss,",
                    "dass nicht die Steuer, sondern der Turnover bindet, ist damit",
                    "auf dieser Datenbasis nicht mehr gestützt.",
                ]
            if enge:
                zeilen += [
                    "",
                    "Die Hebelwahl war schon im Original kein belastbarer Befund:",
                    f"Erst- und Zweitplatzierter trennten dort {pp(max(enge), 2)} %.",
                    "Eine Rangfolge an dieser Marge kippt bei jeder Störung — sie",
                    "sagt nichts über den Hebel, sondern über die Auflösung der",
                    "Messung.",
                ]
    else:
        zeilen += [
            "",
            "Das Optimum wandert in keiner Welt. Der Schluss aus P2 — die Steuer",
            "ist nicht die bindende Restriktion — überlebt die Bereinigung.",
        ]
    return zeilen


def main() -> int:
    d, e, f = (
        lade("p12d_survivorship"),
        lade("p12e_panel_hygiene"),
        lade("p12f_neulauf_bereinigt"),
    )
    g = lade("p12g_pull_bilanz")  # optional: Bilanz des breiteren Pulls
    fehlt = [n for n, v in (("p12d", d), ("p12e", e), ("p12f", f)) if v is None]
    if fehlt:
        print(f"[ERROR] Ergebnisse fehlen: {', '.join(fehlt)} — erst die Läufe.")
        return 1

    maengel = pruefe_artefakt_aktuell(e, f)
    if maengel:
        print("[ERROR] Artefakte stammen aus verschiedenen Codegenerationen:")
        for m in maengel:
            print(f"        - {m}")
        print("        Erst die Läufe wiederholen, dann rendern — sonst mischt")
        print("        das Dokument zwei Detektorgenerationen (E-110).")
        return 1

    lo, hi = ueberhoehung(d)
    traegt_nicht = lo >= ENTSCHEIDUNGSMARGE_PP

    kopf = [
        "# Befund — Trägt der Datensatz die Verdikte?",
        "",
        "*Erzeugt von `render_befund_datenqualitaet.py` aus "
        "`results/p12d_*.json`, `p12e_*.json`, `p12f_*.json`, `p12g_*.json`. "
        "Nicht von Hand "
        "bearbeiten — Änderungen gehen beim nächsten Lauf verloren.*",
        "",
        "---",
        "",
        "## Kurzfassung",
        "",
    ]
    if traegt_nicht:
        kopf += [
            "**Nein — nicht in der Größenordnung, um die gestritten wird.** Die",
            f"Auswahl des Universums allein liefert {dez(lo)} bis {dez(hi)}",
            "Prozentpunkte p. a. gegenüber einem survivorship-freien Universum,",
            "ohne jede Strategie. Die Kampagne entscheidet Fragen im Bereich von",
            f"rund {dez(ENTSCHEIDUNGSMARGE_PP, 1)} pp p. a. Die Verzerrung ist",
            "größer als der zu messende Effekt.",
            "",
        ]
        # Diese beiden Saetze standen frueher fest im Zweig — unabhaengig davon,
        # was p12e und p12f gemessen hatten. Bei einer drehenden Welt haette die
        # Kurzfassung „drehen kein Verdikt" behauptet, waehrend Abschnitt 4
        # dasselbe Dokument widerlegt (Stage-2-Finding F-senior-3). Und die
        # Kurzfassung ist die Stelle, die gelesen wird.
        eingegangen = bool(e.get("kontaminiert"))
        dreht = [w for w, v in f["welten"].items() if v["verdikt_dreht"]]
        kopf.append(
            "Die Preisfehler im Panel sind dagegen **nachweislich** in die"
            if eingegangen
            else "Preisfehler sind in den gemessenen Kanälen **nicht** in die"
        )
        kopf.append("Ergebnisse eingegangen (Abschnitt 2)")
        if dreht:
            kopf += [
                f"— und sie **drehen das Verdikt** in {', '.join(dreht)}",
                "(Abschnitt 4). Damit hängen die betroffenen Aussagen der",
                "Kampagne an Vendor-Fehlern und sind neu zu bewerten.",
            ]
        else:
            kopf += [
                "und drehen dennoch kein Verdikt (Abschnitt 4). Der begrenzende",
                "Faktor ist nicht die Sauberkeit der Kurse, sondern die Auswahl",
                "der Namen — und die ist durch keine Bereinigung reparierbar,",
                "nur durch andere Daten.",
            ]
    else:
        kopf += [
            "**Ja, in dieser Größenordnung.** Die Auswahl des Universums",
            f"liefert {dez(lo)} bis {dez(hi)} Prozentpunkte p. a. — der untere",
            f"Rand liegt unter der Entscheidungsmarge von "
            f"{dez(ENTSCHEIDUNGSMARGE_PP, 1)} pp.",
        ]
    kopf += ["", "---", ""]

    text = "\n".join(
        kopf
        + abschnitt_d(d)
        + ["", "---", ""]
        + abschnitt_e_kanaele(e)
        + ["", "---", ""]
        + abschnitt_e_abdeckung(e)
        + ["", "---", ""]
        + abschnitt_f(f)
        + (["", "---", ""] + abschnitt_g(g) if g else [])
        + [
            "",
            "---",
            "",
            "## Was daraus folgt",
            "",
            "1. Die Intraday-Ergebnisse aus P12 (keine Haltedauer von 1 Stunde",
            "   bis 2 Jahre schlägt das Liegenlassen desselben Universums)",
            "   bleiben **innerhalb** des Universums gültig — der Vergleich ist",
            "   dort ceteris paribus, weil beide Seiten dieselbe Verzerrung",
            "   tragen.",
            "2. Jeder Vergleich **gegen SPY** oder gegen einen passiven ETF ist",
            "   auf dieser Datenbasis nicht belastbar.",
            "3. Der nächste Schritt ist kein weiterer Backtest, sondern ein",
            "   Panel, das Ausscheider mitführt (PIT-Universum mit",
            "   Delisting-Kursen).",
            "",
        ]
    )
    ZIEL.write_text(text + "\n", encoding="utf-8")
    print(f"[OK] {ZIEL} ({len(text.splitlines())} Zeilen)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
