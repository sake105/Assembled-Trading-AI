"""Skip-Guard für Tests, die die echten Kampagnendaten brauchen.

WARUM DAS NÖTIG IST
-------------------
``research/mandat/data/`` ist gitignored (~4,1 GB Kursdaten, jederzeit über
``research/mandat/pull_eodhd_verdict.py`` nachziehbar). In CI existiert das
Verzeichnis deshalb nicht — und jeder Test, der ``load_campaign()`` aufruft,
scheitert dort mit einem Dateifehler.

Genau das ist am 2026-08-01 passiert und **zwei Tage lang unbemerkt geblieben**:
CI war ab Commit ``b9656969`` rot (``CI`` und ``Backend CI``), fünf Commits
lang, weil die Mandat-II-Arbeit ausschließlich in ``research/`` stattfand und
der CI-Status nie geprüft wurde. Lokal war alles grün — die Daten liegen hier.

DIE UNTERSCHEIDUNG, AUF DIE ES ANKOMMT
--------------------------------------
Das ist **kein** Weichspülen eines fehlschlagenden Tests. Es ist die
Unterscheidung, die Rule 40 verlangt: ein *erwarteter Skip* wegen fehlender
Umgebungsvoraussetzung ist etwas anderes als ein Fehler. Das Repo nutzt für
optionale Pakete bereits ``pytest.importorskip``; dies ist das Gegenstück für
optionale **Daten**.

Der Skip ist bewusst eng: er greift nur, wenn das Panel wirklich fehlt. Wo die
Daten liegen — lokal, und bei jedem, der sie gezogen hat — laufen die Tests
vollständig. Die Skip-Begründung nennt den Grund im Klartext, damit ein
CI-Report nicht wie ein sauberer Durchlauf aussieht.
"""

from __future__ import annotations

from pathlib import Path

import pytest

KAMPAGNEN_DATEN = Path(__file__).resolve().parents[1] / "research" / "mandat" / "data"
PANEL = KAMPAGNEN_DATEN / "prices_verdict.parquet"

braucht_kampagnendaten = pytest.mark.skipif(
    not PANEL.exists(),
    reason=(
        "SKIP (nicht geprueft): Kampagnen-Kursdaten fehlen — "
        f"{PANEL.relative_to(Path(__file__).resolve().parents[1])} ist gitignored "
        "(~4,1 GB) und in CI nicht vorhanden. Nachziehbar via "
        "research/mandat/pull_eodhd_verdict.py."
    ),
)
