"""Die Dividendenskalierung hatte null Tests — und den groessten Hebel.

Sie beruehrt JEDE Dividende der Kampagne und wirkt genau auf die Achse, die
gemessen werden soll (GmbH: Dividende 29,83 % gegen Kursgewinn 1,49 %). Der
einzige „Beleg" war ein Ad-hoc-Blick auf ein Toleranzband, das breiter war als
der Fehler, den es finden sollte (E-073). Deshalb hier: gegen analytisch
bekannte Groessen pinnen, nicht gegen ein Erwartungsband.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.mandat2_daten_guard import braucht_kampagnendaten

from research.mandat2.dividenden import (
    auf_panel_skalieren,
    implizite_jahresrendite,
    rohpfad,
)

pytestmark = pytest.mark.fast


def _synthetische_tr_reihe(
    n: int = 260, tages_rendite: float = 0.0002, div_je_quartal: float = 0.5
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Baut ROH-Pfad und daraus den TR-adjustierten Pfad — Wahrheit bekannt.

    Der Rohkurs waechst mit fester Rendite und faellt am Ex-Tag um die
    Dividende. Der TR-Pfad reinvestiert sie. Rueckwaerts normalisiert, damit
    beide am letzten Punkt zusammenfallen — genau die Konvention der Quelle.
    """
    idx = pd.bdate_range("2000-01-03", periods=n)
    ex_tage = set(range(60, n, 60))
    raw = np.empty(n)
    tr = np.empty(n)
    raw[0] = 100.0
    tr[0] = 100.0
    divs = np.zeros(n)
    for i in range(1, n):
        d = div_je_quartal if i in ex_tage else 0.0
        divs[i] = d
        raw[i] = raw[i - 1] * (1 + tages_rendite) - d
        tr[i] = tr[i - 1] * (raw[i] + d) / raw[i - 1]
    # Rueckwaerts normieren: adj(T) == raw(T)
    adj = pd.Series(tr * (raw[-1] / tr[-1]), index=idx)
    return pd.Series(raw, index=idx), adj, pd.Series(divs, index=idx)


def test_rohpfad_trifft_die_bekannte_wahrheit():
    raw_soll, adj, divs = _synthetische_tr_reihe()
    raw_ist = rohpfad(adj, divs)
    # Ueber die ganze Reihe, nicht nur am Anker.
    rel = ((raw_ist - raw_soll).abs() / raw_soll).max()
    assert rel < 1e-9, f"groesster relativer Fehler {rel:.2e}"


def test_anker_liegt_am_letzten_punkt():
    _, adj, divs = _synthetische_tr_reihe()
    raw = rohpfad(adj, divs)
    assert raw.iloc[-1] == pytest.approx(adj.iloc[-1])


def test_abgeschnittener_index_verschiebt_den_anker_und_verfaelscht_die_skala():
    """DER Fehler, der als BLOCKER gefunden wurde (E-074).

    Die Rekursion verankert am letzten Kurs der uebergebenen Reihe. Wird sie
    auf ein GEFENSTERTES Panel angewandt, dessen Adjustierung ueber die volle
    Historie normiert ist, liegt der Anker daneben — und der Fehler waechst
    monoton zum Fensterrand. Dieser Test haelt fest, dass die Abweichung real
    ist, damit niemand die Funktion erneut auf gefensterte Daten wirft.
    """
    raw_soll, adj, divs = _synthetische_tr_reihe(n=260)
    schnitt = 200
    raw_voll = rohpfad(adj, divs)
    raw_geschnitten = rohpfad(adj.iloc[:schnitt], divs.iloc[:schnitt])

    # Auf der vollen Reihe stimmt es.
    assert raw_voll.iloc[0] == pytest.approx(raw_soll.iloc[0], rel=1e-9)
    # Auf der geschnittenen NICHT — und zwar messbar daneben.
    assert raw_geschnitten.iloc[0] != pytest.approx(raw_soll.iloc[0], rel=1e-3)


def test_skalierung_macht_die_dividende_kleiner_nicht_groesser():
    """adj < raw in der Fruehhistorie -> die Panel-Dividende ist kleiner."""
    _, adj, divs = _synthetische_tr_reihe()
    close = pd.DataFrame({"AAA": adj})
    div_panel = pd.DataFrame({"AAA": divs})
    skaliert = auf_panel_skalieren(close, div_panel)
    frueh = divs[divs > 0].index[0]
    assert 0 < skaliert.loc[frueh, "AAA"] < divs.loc[frueh]


def test_implizite_rendite_trifft_die_konstruierte_rendite():
    """Der Sanity-Check muss den KONSTRUIERTEN Wert treffen, nicht ein Band."""
    _, adj, divs = _synthetische_tr_reihe(n=520, div_je_quartal=0.5)
    close = pd.DataFrame({"AAA": adj})
    skaliert = auf_panel_skalieren(close, pd.DataFrame({"AAA": divs}))
    r = implizite_jahresrendite(close, skaliert, "AAA")
    # Konstruktion: ~4 Ex-Tage/Jahr a 0,50 auf einem Kurs um 100 -> ~2 %.
    assert 0.015 < r.iloc[0] < 0.025


def test_symbole_ohne_dividende_bleiben_unveraendert():
    _, adj, divs = _synthetische_tr_reihe()
    close = pd.DataFrame({"AAA": adj, "BBB": adj})
    div_panel = pd.DataFrame({"AAA": divs, "BBB": np.zeros(len(divs))}, index=adj.index)
    skaliert = auf_panel_skalieren(close, div_panel)
    assert (skaliert["BBB"] == 0).all()


def test_leere_reihe_bricht_nicht():
    leer = pd.Series(dtype=float)
    assert rohpfad(leer, leer).empty


@braucht_kampagnendaten
@pytest.mark.parametrize("symbol,jahr,band", [("SPY", 1995, (0.020, 0.028))])
def test_echte_spy_rendite_liegt_im_engen_band(symbol, jahr, band):
    """Regression gegen die ECHTEN Daten — enges Band, nicht 1,3-3,5 %.

    Das alte Band haette den 20-%-Skalenfehler durchgelassen; beide
    Zahlenreihen lagen darin. SPY 1995 lag real bei ~2,3 %.
    """
    from research.mandat2.campaign_data import load_campaign

    d = load_campaign()
    r = implizite_jahresrendite(d.close, d.div_panel, symbol)
    assert jahr in r.index
    assert band[0] < r.loc[jahr] < band[1], f"{symbol} {jahr}: {r.loc[jahr]:.4%}"
