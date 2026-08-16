from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path

import pandas as pd
from common.io_utils import http_get_text, to_parquet

# SDW CSV Example (EUR->USD Daily):
# https://sdw.ecb.europa.eu/quickviewexport.do?trans=N&node=2018794&SERIES_KEY=120.EXR.D.USD.EUR.SP00.A&type=csv
# Wir nutzen die generische SDW API (CSV), hier Demo: EURUSD, EURGBP.


SERIES = {
    "EURUSD": "120.EXR.D.USD.EUR.SP00.A",
    "EURGBP": "120.EXR.D.GBP.EUR.SP00.A",
}


BASE = "https://sdw.ecb.europa.eu/quickviewexport.do?trans=N&node=2018794&SERIES_KEY={series}&type=csv"


def fetch(series_key: str) -> pd.DataFrame:
    url = BASE.format(series=series_key)
    csv_txt = http_get_text(url)
    # CSV hat Headerzeilen mit Metadaten -> ab der letzten Headerzeile einlesen
    lines = [  # noqa: F841
        ln
        for ln in csv_txt.splitlines()
        if ";" in ln and ln.split(";")[0].strip().isdigit() is False or ln[0].isdigit()
    ]
    # Simplifizierter Parser: wir suchen die Data-Section -- als Fallback lesen wir alles und filtern spaeter
    df = pd.read_csv(StringIO(csv_txt), sep=",")
    # Heuristik: Spaltennamen finden
    cand_date = [
        c
        for c in df.columns
        if c.lower().startswith("time")
        or c.lower().startswith("period")
        or c.lower().startswith("date")
    ]
    cand_val = [
        c
        for c in df.columns
        if c.lower().startswith("obs") or c.lower().startswith("value")
    ]
    if not cand_date or not cand_val:
        # alternativer Versuch: direkt die letzten 2 Spalten
        cand_date = [df.columns[0]]
        cand_val = [df.columns[-1]]
    df = df[[cand_date[0], cand_val[0]]].rename(
        columns={cand_date[0]: "timestamp", cand_val[0]: "close"}
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["open"] = df["high"] = df["low"] = df["close"]
    df["volume"] = 0.0
    return df


def main():
    if len(sys.argv) < 3:
        print("Usage: python pull_ecb_fx.py <pairs_csv> <out_dir>")
        sys.exit(2)
    pairs = sys.argv[1].split(",")
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    # E-112: one request protocol per run, unmapped and empty pairs included.
    plog = None
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from src.assembled_core.data.pull_log import STATUS_SKIPPED, PullLog

        plog = PullLog(source="ecb_fx")
        skipped_status = STATUS_SKIPPED
    except Exception as exc:  # pragma: no cover - must not block a pull
        print(f"[FX] WARN pull_log unavailable: {exc}")

    dfs = []
    n_fail = 0
    failed_all = False
    # try/finally, nicht einfach write() am Ende: http_get_text raist bei einem
    # Anbieterausfall durch main() hindurch. Stand der Schreibvorgang hinter dem
    # Erfolgspfad, existierte das Protokoll in jedem Lauf AUSSER dem einen, der
    # zaehlt - und der Nachweis "wir haben gefragt" fehlte weiterhin (E-112).
    try:
        for p in pairs:
            key = SERIES.get(p.upper())
            if not key:
                print(f"WARN: Pair {p} nicht im Demo-Mapping, überspringe.")
                # A pair we never asked about must still leave a trace, else
                # 'no file' later reads as 'the provider has no data'.
                if plog is not None:
                    plog.record(
                        p.upper(),
                        status=skipped_status,
                        n_rows=0,
                        skipped_reason="not in SERIES mapping - never requested",
                    )
                continue
            # Per-key try: the transport raises on a provider outage, and the
            # caller is the only layer that knows WHICH key it was asking for.
            # Without this the protocol survives the outage but records nothing
            # — "0 requested" instead of "EURUSD asked, HTTP 503".
            try:
                df = fetch(key)
            except Exception as exc:  # noqa: BLE001
                print(f"[FX] ERR {p}: {exc}")
                n_fail += 1
                if plog is not None:
                    plog.record(
                        p.upper(),
                        http_status=getattr(exc, "code", None),
                        error=f"{type(exc).__name__}: {str(exc)[:200]}",
                    )
                continue
            df["symbol"] = p.upper()
            df["provider"] = "ecb_sdw"
            df = (
                df[
                    [
                        "timestamp",
                        "symbol",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "provider",
                    ]
                ]
                .sort_values("timestamp")
                .reset_index(drop=True)
            )
            to_parquet(df, out_dir / f"{p.upper()}.parquet")
            # Parsed row count, recorded where it is actually known.
            if plog is not None:
                plog.record(p.upper(), n_rows=len(df))
            dfs.append(df)
        if dfs:
            big = pd.concat(dfs, ignore_index=True)
            to_parquet(big, out_dir / "fx_ref.parquet")

        # E-147/Exit-Code: das per-Symbol-except oben faengt die Exception, die
        # vorher durch main() raiste und Exit-Code 1 erzeugte. Ein Protokoll zu
        # gewinnen und dafuer den Exit-Code zu verlieren ist kein Netto-Gewinn:
        # der Exit-Code ist die einzige Ebene, die ein Scheduler liest.
        # try/finally fuer die Evidenz, expliziter Exit-Code fuer das Ergebnis.
        if n_fail and not dfs:
            print(f"[FX] ERR alle {n_fail} Paare failten", file=sys.stderr)
            failed_all = True

    finally:
        if plog is not None:
            plog.write()

    if failed_all:
        sys.exit(2)


if __name__ == "__main__":
    main()
