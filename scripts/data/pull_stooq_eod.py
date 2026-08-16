from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path

import pandas as pd
from common.io_utils import http_get_text, normalize_ohlc, to_parquet

"""
Stooq CSV EoD Download.
URL-Format (US): https://stooq.com/q/d/l/?s=aapl.us&i=d
Hinweis: Symbole je nach Börse suffixen (z. B. .us, .de). Für Demo nutzen wir .us.
"""


BASE = "https://stooq.com/q/d/l/?i=d&s={symbol}"


def fetch(symbol: str) -> pd.DataFrame:
    url = BASE.format(symbol=symbol)
    txt = http_get_text(url)
    df = pd.read_csv(StringIO(txt))
    # stooq columns: Date,Open,High,Low,Close,Volume
    df = df.rename(
        columns={
            "Date": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    return df


def main():
    if len(sys.argv) < 4:
        print("Usage: python pull_stooq_eod.py <symbols_csv> <out_dir> <suffix>")
        print("Example: python pull_stooq_eod.py AAPL,MSFT data/raw/eod/stooq .us")
        sys.exit(2)
    symbols = sys.argv[1].split(",")
    out_dir = Path(sys.argv[2])
    suffix = sys.argv[3]
    out_dir.mkdir(parents=True, exist_ok=True)
    # E-112: request protocol, written in finally so it survives the run that
    # matters most - the one where the provider is down.
    plog = None
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from src.assembled_core.data.pull_log import PullLog

        plog = PullLog(source="stooq_eod")
    except Exception as exc:  # pragma: no cover - must not block a pull
        print(f"[EOD] WARN pull_log unavailable: {exc}")

    all_dfs = []
    n_fail = 0
    failed_all = False
    try:
        for s in symbols:
            sym = f"{s}{suffix}"
            try:
                # Parse UND Write gehoeren in denselben Guard wie der Fetch:
                # lag normalize_ohlc/to_parquet ausserhalb, entkam ein
                # ValueError der Schleife, die restlichen Symbole wurden nie
                # angefragt und das Protokoll war leer (E-147, zweite Stufe).
                df = fetch(sym)
                df = normalize_ohlc(df, s, provider="stooq")
                to_parquet(df, out_dir / f"{s}.parquet")
                if plog is not None:
                    plog.record(sym, n_rows=len(df))
                all_dfs.append(df)
            except Exception as exc:  # noqa: BLE001
                print(f"[EOD] ERR {s}: {exc}")
                n_fail += 1
                if plog is not None:
                    # sym (mit Suffix) ist das, was tatsaechlich angefragt wurde -
                    # genau das muss das Protokoll beantworten koennen.
                    plog.record(
                        sym,
                        http_status=getattr(exc, "code", None),
                        error=f"{type(exc).__name__}: {str(exc)[:200]}",
                    )
                continue
        if all_dfs:
            big = pd.concat(all_dfs, ignore_index=True)
            to_parquet(big, out_dir / "eod_all.parquet")

        # E-147/Exit-Code: das per-Symbol-except oben faengt die Exception, die
        # vorher durch main() raiste und Exit-Code 1 erzeugte. Ein Protokoll zu
        # gewinnen und dafuer den Exit-Code zu verlieren ist kein Netto-Gewinn:
        # der Exit-Code ist die einzige Ebene, die ein Scheduler liest.
        # try/finally fuer die Evidenz, expliziter Exit-Code fuer das Ergebnis.
        if n_fail and not all_dfs:
            print(f"[EOD] ERR alle {n_fail} Symbole failten", file=sys.stderr)
            failed_all = True
    finally:
        if plog is not None:
            plog.write()

    if failed_all:
        sys.exit(2)


if __name__ == "__main__":
    main()
