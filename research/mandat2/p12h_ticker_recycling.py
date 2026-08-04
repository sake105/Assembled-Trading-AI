"""P12h — Enthalten Panel-Spalten die Kurse ZWEIER Unternehmen?

DER ANLASS
----------
P12g hat gezeigt, dass der Intraday-Endpunkt Ausscheider nicht führt. Beim
Versuch, dieselbe Lücke über Tagesdaten zu schließen, fiel etwas anderes auf:
99 PIT-Mitglieder, die in der Rohdatei liegen, haben dort Kurse **nur außerhalb
des Suchfensters**. Stichprobe:

* ABI (Anheuser-Busch, übernommen 2008) → Serie beginnt 2025-06-26
* ABS (Albertsons, übernommen 2006) → Serie beginnt 2018-08-28
* ALTR (Altera, übernommen 2015) → Serie beginnt 2017-11-01

Der Vendor liefert unter dem Symbol die **heutige** Firma. Das ist E-113 auf
der Tagesseite: der Ticker ist ein Zeitreihen-Attribut, kein Schlüssel.

WARUM DAS SCHLIMMER IST ALS EINE LÜCKE
--------------------------------------
Solange die zweite Firma erst nach dem Suchfenster handelt, entsteht nur eine
Lücke. Liegt die Neuvergabe aber **innerhalb** des Fensters, steht in einer
Spalte die Historie von Unternehmen A, dann nichts, dann Unternehmen B — und
jede Rechnung darüber hinweg hält zwei Firmen für eine.

Genau das prüft dieses Skript. Es ist **Diagnostik, kein Backtest**: gemessen
wird, wie viele Spalten betroffen sind und ob die Strategie sie angefasst hat.
Der Trial-Zähler bleibt unberührt.

WIE ERKANNT WIRD — UND WAS DAS NICHT IST
----------------------------------------
Signatur: eine Lücke von mindestens ``MIN_LUECKE`` Handelstagen, nach der die
Serie weiterläuft. Ein delistetes Unternehmen hört auf — es kommt nicht nach
Jahren zurück.

Das ist eine **Signatur, keine Ursache**. Dieselbe Signatur tragen auch
Vendor-Datenlöcher: CCE (Coca-Cola Enterprises) hat sechs jährliche Lücken und
existierte durchgehend. Die Trennung in ``panel_getrennt`` benutzt deshalb eine
höhere Schwelle (500) und weist die wahrscheinlichen Fehltreffer aus. Wer die
Ausgabe dieses Skripts zitiert, sollte von „Spalten mit Lücke ≥ X" sprechen,
nicht von „Spalten mit zwei Unternehmen" (E-115).

Jeder Treffer wird mit Datum und Kursen davor und danach ausgewiesen, damit die
Einordnung nachprüfbar bleibt.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd  # noqa: E402

from research.mandat2.campaign_data import load_campaign  # noqa: E402

HIER = Path(__file__).resolve().parent
OUT = HIER / "results"

#: Die ungefensterte Quelle. Wird NUR fuer `rohdaten_lage` gelesen — die Frage
#: dort ist ja gerade, was VOR dem Fenster-Gate in der Datei steht. Jede
#: Rechnung geht weiterhin ueber `load_campaign`.
ROH = HIER.parents[1] / "research" / "mandat" / "data" / "prices_verdict.parquet"

#: Ab wie vielen Handelstagen ohne Kurs gilt eine Serie als unterbrochen?
#: 120 ≈ ein halbes Jahr. Kürzere Lücken kommen bei Handelsaussetzungen und
#: dünn gehandelten Namen real vor; ein Unternehmen, das ein halbes Jahr nicht
#: handelt und dann zurückkehrt, ist praktisch immer ein neuer Emittent unter
#: demselben Symbol.
MIN_LUECKE = 120

#: Mindestlänge einer Serie, damit sie überhaupt betrachtet wird.
MIN_PUNKTE = 40


def unterbrechungen(close: pd.DataFrame, min_luecke: int = MIN_LUECKE) -> list[dict]:
    """Spalten mit einer Lücke, nach der die Serie weiterläuft.

    Gemessen wird in **Handelstagen** des Panel-Index, nicht in Kalendertagen:
    Wochenenden und Feiertage sind keine Unterbrechung, und über 20 Jahre
    summieren sie sich auf ein Drittel der Kalenderzeit.
    """
    aus: list[dict] = []
    for sym in close.columns:
        ser = close[sym].dropna()
        if len(ser) < MIN_PUNKTE:
            continue
        pos = pd.Series(close.index.get_indexer(ser.index))
        luecke = pos.diff()
        if pd.isna(luecke.max()) or luecke.max() < min_luecke:
            continue
        i = int(luecke.idxmax())
        vor, nach = float(ser.iloc[i - 1]), float(ser.iloc[i])
        aus.append(
            {
                "symbol": sym,
                "luecke_handelstage": int(luecke.max()),
                "letzter_kurs_am": f"{ser.index[i - 1]:%Y-%m-%d}",
                "naechster_kurs_am": f"{ser.index[i]:%Y-%m-%d}",
                "kurs_vor": vor,
                "kurs_nach": nach,
                "faktor": nach / vor if vor else None,
                "n_punkte_vor": i,
                "n_punkte_nach": len(ser) - i,
            }
        )
    aus.sort(key=lambda x: -x["luecke_handelstage"])
    return aus


def betroffene_auswahl(membership: pd.Series, symbole: set[str]) -> dict:
    """Wurden die betroffenen Namen überhaupt je Index-Mitglied?

    Ein recycelter Ticker, der nie im Universum stand, ist harmlos. Einer, der
    zu BEIDEN Zeiten Mitglied war, ist der schlimmste Fall: dann hat die
    Strategie beide Firmen als denselben Namen gesehen.
    """
    aus: dict[str, dict] = {}
    for sym in sorted(symbole):
        termine = [t for t, m in membership.items() if sym in m]
        if termine:
            aus[sym] = {
                "n_termine_mitglied": len(termine),
                "erstes": f"{termine[0]:%Y-%m-%d}",
                "letztes": f"{termine[-1]:%Y-%m-%d}",
            }
    return aus


def rohdaten_lage(membership, close, roh_pfad) -> dict:
    """Warum fehlen PIT-Mitglieder im Panel — und was liegt in der Rohdatei?

    Diese Zahlen standen zuerst nur im Chat: „133 fehlen, 99 liegen in der
    Rohdatei, ABI beginnt 2025-06-26". Sie stammten aus einer Ad-hoc-Sonde und
    waren damit nicht reproduzierbar — genau die Drift, gegen die diese Kampagne
    ihre Befunde generiert statt schreibt (Stage-2-Finding F-senior-4, E-085).

    Der Direktzugriff auf ``prices_verdict.parquet`` ist hier **bewusst** und
    zulaessig: gemessen wird, was VOR dem Fenster-Gate in der Quelle steht.
    Genau das ist die Frage. Fuer jede Rechnung gilt weiterhin
    ``load_campaign``.
    """
    import pandas as pd

    roh = pd.read_parquet(roh_pfad)
    pivot = roh.pivot(index="timestamp", columns="symbol", values="close")
    alle: set[str] = set()
    for m in membership:
        alle |= set(m)
    im_panel = set(close.columns)
    in_roh = set(pivot.columns)
    fehlt = alle - im_panel
    nur_roh = sorted((alle & in_roh) - im_panel)
    von, bis = close.index[0], close.index[-1]
    # ALLE messen, nur die ersten anzeigen (Stage-3-Finding F-auditor-2). Die
    # erste Fassung rechnete nur ueber `nur_roh[:40]` — und wegen `sorted()`
    # ueber die alphabetisch ersten, also keine Stichprobe. Der Befund sagte
    # trotzdem „ausschliesslich".
    beispiele = {}
    n_erst_nach_fenster = 0
    n_ohne_punkte_im_fenster = 0
    for i, sym in enumerate(nur_roh):
        ser = pivot[sym].dropna()
        if not len(ser):
            continue
        im_fenster = int(ser.loc[(ser.index >= von) & (ser.index <= bis)].notna().sum())
        if im_fenster == 0:
            n_ohne_punkte_im_fenster += 1
            if ser.index[0] > bis:
                n_erst_nach_fenster += 1
        if i < 40:
            beispiele[sym] = {
                "erste_zeile": f"{ser.index[0]:%Y-%m-%d}",
                "letzte_zeile": f"{ser.index[-1]:%Y-%m-%d}",
                "n": int(len(ser)),
                "im_suchfenster": im_fenster,
            }
    return {
        "n_pit_mitglieder": len(alle),
        "n_im_panel": len(alle & im_panel),
        "n_fehlt_im_panel": len(fehlt),
        "n_fehlt_aber_in_rohdatei": len(nur_roh),
        "n_fehlt_ueberall": len(fehlt - in_roh),
        # „Keine Kurse IM Fenster" ist TAUTOLOGISCH: `campaign_data` wirft per
        # `dropna(axis=1, how="all")` genau die Spalten raus, die im Fenster
        # keinen Kurs haben. Informativ ist nur, ob die Serie ERST DANACH
        # beginnt — das ist der Beleg fuer Ticker-Recycling (F-auditor-2).
        "n_ohne_punkte_im_fenster": n_ohne_punkte_im_fenster,
        "n_serie_beginnt_erst_nach_dem_fenster": n_erst_nach_fenster,
        "hinweis_tautologie": (
            "n_ohne_punkte_im_fenster wiederholt die Panel-Filterregel "
            "(dropna how=all) und belegt sie nicht. Aussagekraeftig ist "
            "n_serie_beginnt_erst_nach_dem_fenster."
        ),
        "beispiele": beispiele,
    }


# MAJOR-2 (Stage 1):  stand hier ein zweites Mal — eine Kopie
# aus p12e mit einem Waechter, den das Original nicht hatte. Zwei Fassungen
# derselben Messung sind eine zweite Wahrheit (Rule 50), und gehaertet war
# ausgerechnet die juengere. Jetzt gibt es genau eine, in p12e, mit Waechter.
from research.mandat2.p12e_panel_hygiene import gehaltene_namen  # noqa: E402


def wirkung(treffer: list[dict], bestand: dict[str, set[str]], equity) -> dict:
    """Wurde ueber die Luecke hinweg gehalten — und was kostete das?

    Der Wiedereinstiegstag ist der kritische: dort bewertet die Engine eine
    Position, die sie in Firma A gekauft hat, erstmals mit dem Kurs von Firma B.
    ``close_ff`` haelt sie bis dahin auf dem letzten Kurs von A eingefroren; die
    Delisting-Regel greift nicht, weil ``last_valid`` am Ende der Serie von B
    liegt.

    Gemessen wird deshalb: lag der Name am Tag VOR dem Wiedereinstieg im
    Bestand, und wie gross war die Portfolio-Tagesrendite am Wiedereinstiegstag?

    UNTERGRENZE (Stage-2-Finding F-senior-11): ``unterbrechungen`` meldet je
    Symbol nur die GROESSTE Luecke. Bei mehrfach vergebenen Tickern bleiben die
    uebrigen Wiedereinstiege ungezaehlt — die Zahl ist eine untere Schranke,
    kein vollstaendiger Messwert.
    """
    r = equity.pct_change(fill_method=None)
    rang = r.rank(ascending=False)
    n_tage = int(r.notna().sum())
    aus: dict[str, dict] = {}
    for x in treffer:
        sym, tag = x["symbol"], x["naechster_kurs_am"]
        # Der letzte Handelstag VOR dem Wiedereinstieg, an dem Bestand bekannt ist
        vortage = sorted(d for d in bestand if d < tag)
        if not vortage:
            # Kein Bestandstag vor dem Wiedereinstieg — der Name faellt aus der
            # Messung, und das ist von „nicht betroffen" nicht zu unterscheiden.
            # Deshalb sichtbar machen statt still ueberspringen (MINOR-4).
            aus[sym] = {
                "gehalten_am_vortag": None,
                "hinweis": "kein Bestandstag vor dem Wiedereinstieg",
            }
            continue
        if sym not in bestand.get(vortage[-1], ()):
            continue
        ts = pd.Timestamp(tag, tz="UTC")
        if ts not in r.index or not pd.notna(r.loc[ts]):
            # Fail-loud: ein gehaltener Name ohne messbare Wirkung waere die
            # beruhigende Antwort aus einem Verdrahtungsfehler (E-103).
            aus[sym] = {
                "gehalten_am_vortag": True,
                "rendite": None,
                "hinweis": "keine Portfolio-Rendite am Wiedereinstiegstag",
            }
            continue
        aus[sym] = {
            "gehalten_am_vortag": True,
            "wiedereinstieg": tag,
            "kurssprung": x["faktor"],
            "portfolio_tagesrendite": float(r.loc[ts]),
            "rang_unter_allen_tagen": int(rang.loc[ts]),
            "n_handelstage": n_tage,
        }
    return aus


def tote_haltedauer(treffer: list[dict], bestand: dict[str, set[str]]) -> dict:
    """Wie lange wurde eine bereits delistete Firma weitergehalten?

    Das ist der eigentliche Schaden — nicht der Kurssprung am Wiedereinstieg,
    sondern der Ausfall der Delisting-Hygiene. Die Engine verkauft zwangsweise,
    wenn ``last_valid < t`` und die Serie mehr als 10 Tage vor Panelende endet.
    Bei einem recycelten Ticker liegt ``last_valid`` aber am Ende der Serie von
    Firma **B** — die Bedingung ist nie erfuellt, und die Position in der toten
    Firma A laeuft weiter, bewertet auf ihrem letzten Kurs (``close_ff``).

    Gezaehlt wird deshalb: Handelstage zwischen dem letzten Kurs von A und dem
    Wiedereinstieg von B, an denen der Name im Bestand lag.

    Auch das ist eine **Untergrenze**: gezaehlt wird nur die groesste Luecke je
    Symbol (F-senior-11).
    """
    aus: dict[str, dict] = {}
    for x in treffer:
        sym = x["symbol"]
        tage = [
            d
            for d, namen in bestand.items()
            if x["letzter_kurs_am"] < d < x["naechster_kurs_am"] and sym in namen
        ]
        if tage:
            aus[sym] = {
                "tote_haltetage": len(tage),
                "von": min(tage),
                "bis": max(tage),
                "letzter_echter_kurs": x["letzter_kurs_am"],
            }
    return aus


def main() -> int:
    OUT.mkdir(exist_ok=True)
    d = load_campaign()
    print(d, flush=True)
    print("Diagnostik, kein Backtest: Trial-Zaehler bleibt unberuehrt.\n")

    lage = rohdaten_lage(d.membership, d.close, ROH)
    print("ROHDATEN-LAGE (vor dem Fenster-Gate)")
    print(f"  PIT-Mitglieder            : {lage['n_pit_mitglieder']}")
    print(f"  davon im Kampagnen-Panel  : {lage['n_im_panel']}")
    print(f"  fehlen im Panel           : {lage['n_fehlt_im_panel']}")
    print(f"  davon in der Rohdatei DA  : {lage['n_fehlt_aber_in_rohdatei']}")
    print(f"  fehlen ueberall           : {lage['n_fehlt_ueberall']}")
    for sym, v in list(lage["beispiele"].items())[:4]:
        print(
            f"     {sym:<7}Serie {v['erste_zeile']}..{v['letzte_zeile']}, "
            f"{v['im_suchfenster']} Punkte im Suchfenster"
        )
    print("", flush=True)

    treffer = unterbrechungen(d.close)
    syms = {t["symbol"] for t in treffer}
    mitglied = betroffene_auswahl(d.membership, syms)

    print(f"Panel-Spalten gesamt            : {d.close.shape[1]}")
    print(f"mit Luecke >= {MIN_LUECKE} Handelstagen : {len(treffer)}")
    print(f"davon je Index-Mitglied         : {len(mitglied)}")
    print(
        f"\n{'Sym':<7}{'Luecke':>7}  {'letzter':<12}{'naechster':<12}"
        f"{'vor':>11}{'nach':>11}{'Mitglied':>10}"
    )
    for t in treffer[:20]:
        m = mitglied.get(t["symbol"])
        print(
            f"{t['symbol']:<7}{t['luecke_handelstage']:>7}  "
            f"{t['letzter_kurs_am']:<12}{t['naechster_kurs_am']:<12}"
            f"{t['kurs_vor']:>11,.2f}{t['kurs_nach']:>11,.2f}"
            f"{(str(m['n_termine_mitglied']) if m else '-'):>10}"
        )

    print("")
    print("WIRKUNG — echte Engine, echter Bestand (12-1-Momentum, top20)")
    bestand, equity = gehaltene_namen(d, top_in=20)
    getroffen = wirkung(treffer, bestand, equity)
    if getroffen:
        kopf = f"  {'Sym':<7}{'Wiedereinstieg':<16}{'Sprung':>11}"
        print(kopf + f"{'Portfolio-Tag':>15}{'Rang':>14}")
        for sym, v in sorted(
            getroffen.items(),
            key=lambda kv: -abs(kv[1].get("portfolio_tagesrendite") or 0.0),
        ):
            rt = v.get("portfolio_tagesrendite")
            f_ = v.get("kurssprung")
            sprung = f"{f_:,.2f}x" if f_ else "-"
            tag = f"{rt:+.2%}" if rt is not None else "n/a"
            rang = (
                f"{v['rang_unter_allen_tagen']}/{v['n_handelstage']}"
                if rt is not None
                else "-"
            )
            print(
                f"  {sym:<7}{v.get('wiedereinstieg', '-'):<16}{sprung:>11}"
                f"{tag:>15}{rang:>14}"
            )
    else:
        print("  Kein betroffener Name lag am Vortag des Wiedereinstiegs im Bestand.")

    tot = tote_haltedauer(treffer, bestand)
    print("")
    print("DELISTING-HYGIENE — wie lange lief eine tote Firma weiter?")
    if tot:
        gesamt = sum(v["tote_haltetage"] for v in tot.values())
        print(f"  {len(tot)} Namen, zusammen {gesamt} Handelstage im Bestand OHNE")
        print("  echten Kurs — die Zwangsverkaufs-Regel greift bei ihnen nie,")
        print("  weil last_valid am Ende der Serie von Firma B liegt.")
        for sym, v in sorted(tot.items(), key=lambda kv: -kv[1]["tote_haltetage"])[:8]:
            print(
                f"    {sym:<7}{v['tote_haltetage']:>5} Tage   "
                f"letzter echter Kurs {v['letzter_echter_kurs']}"
            )
    else:
        print("  Kein betroffener Name lag waehrend seiner Luecke im Bestand.")

    from research.mandat2.panel_getrennt import MIN_LUECKE as TRENN_SCHWELLE

    ab_trennschwelle = unterbrechungen(d.close, min_luecke=TRENN_SCHWELLE)
    print(
        f"  davon >= {TRENN_SCHWELLE} Handelstage (Trennschwelle): "
        f"{len(ab_trennschwelle)}"
    )

    ergebnis = {
        "fenster": d.fenster,
        "trennschwelle": TRENN_SCHWELLE,
        "n_ab_trennschwelle": len(ab_trennschwelle),
        "rohdaten_lage": lage,
        "wirkung": getroffen,
        "tote_haltedauer": tot,
        "min_luecke_handelstage": MIN_LUECKE,
        "n_spalten": int(d.close.shape[1]),
        "n_unterbrochen": len(treffer),
        "n_unterbrochen_und_mitglied": len(mitglied),
        "treffer": treffer,
        "mitgliedschaft": mitglied,
    }
    (OUT / "p12h_ticker_recycling.json").write_text(
        json.dumps(ergebnis, indent=2), encoding="utf-8"
    )
    print(f"\n-> {OUT / 'p12h_ticker_recycling.json'}")

    print("\n" + "=" * 72)
    if not treffer:
        print("BEFUND: keine unterbrochenen Serien — jede Spalte fuehrt genau")
        print("        ein Unternehmen.")
    else:
        print(
            f"BEFUND: {len(treffer)} Panel-Spalten sind unterbrochen "
            f"(>= {MIN_LUECKE} Handelstage);"
        )
        print(f"        {len(mitglied)} davon gehoerten dem Index an. Ab der")
        print(f"        Trennschwelle {TRENN_SCHWELLE}: {len(ab_trennschwelle)}.")
        print("        Wo die Neuvergabe INNERHALB des Fensters liegt, stehen zwei")
        print("        Unternehmen in einer Spalte — jede Rechnung darueber hinweg")
        print("        haelt sie fuer eines. Der Ticker ist ein Zeitreihen-Attribut,")
        print("        kein Schluessel. ACHTUNG: die Luecke ist eine SIGNATUR;")
        print("        Vendor-Datenloecher tragen sie auch (E-115).")
    print("=" * 72, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
