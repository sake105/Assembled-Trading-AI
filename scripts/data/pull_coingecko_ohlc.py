from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from common.io_utils import http_get_json, normalize_ohlc, to_parquet

# OHLC endpoint: /coins/{id}/ohlc?vs_currency=usd&days=1/7/30/90/180/365/max
BASE = "https://api.coingecko.com/api/v3/coins/{cid}/ohlc?vs_currency={ccy}&days={days}"


CID_MAP = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
}


def pull_one(symbol: str, days: int, ccy: str = "usd", pull_log=None) -> pd.DataFrame:
    cid = CID_MAP.get(symbol.upper())
    if not cid:
        # ValueError statt SystemExit: SystemExit erbt von BaseException und
        # wird von `except Exception` im Aufrufer NICHT gefangen - der Lauf
        # staerbe beim ersten unbekannten Symbol, und die restlichen wuerden
        # weder angefragt noch protokolliert.
        raise ValueError(f"Unbekanntes Symbol für Demo: {symbol}")
    url = BASE.format(cid=cid, ccy=ccy, days=days)
    # E-112: hand the protocol down so this request is recorded with its
    # symbol and window - also when it comes back empty.
    data = http_get_json(
        url,
        # KEIN log_key: sonst protokollieren Transport UND Caller-except
        # denselben Key bei jedem Fehler (duplicate_keys waere dauerhaft
        # nicht leer). Genau eine Ebene pro Key - hier der Caller, weil nur
        # er den Parse-Ausgang kennt.
        pull_log=None,
        # window bleibt leer: window_start/-end sind ZEITpunkte. Dauer und
        # Waehrung gehoeren als Extra-Felder an den Recorder, nicht als
        # window-Bestandteile - und NICHT an http_get_json: dessen Signatur
        # kennt kein **extra, `quote_ccy=` dort war ein TypeError bei jedem
        # Symbol (E-149).
        log_window=None,
    )
    # Antwort: [[ts, open, high, low, close], ...] (ms-epoch)
    df = pd.DataFrame(
        data, columns=["ts_ms", "open", "high", "low", "close"]
    )  # kein Volume hier
    df["timestamp"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    df["volume"] = 0.0
    df = df.drop(columns=["ts_ms"])
    df = normalize_ohlc(df, symbol, provider="coingecko")
    return df


def main():
    if len(sys.argv) < 4:
        print("Usage: python pull_coingecko_ohlc.py <symbols_csv> <days> <out_dir>")
        sys.exit(2)
    symbols = sys.argv[1].split(",")
    try:
        days = int(sys.argv[2])
    except ValueError:
        print(f"Error: <days> must be an integer, got {sys.argv[2]!r}")
        sys.exit(2)
    out_dir = Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)
    # E-112: one request protocol per run, empty results included.
    plog = None
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from src.assembled_core.data.pull_log import PullLog

        plog = PullLog(source="coingecko_ohlc")
    except Exception as exc:  # pragma: no cover - must not block a pull
        print(f"[CG] WARN pull_log unavailable: {exc}")

    dfs = []
    n_fail = 0
    failed_all = False
    # try/finally: http_get_json raist last_ex nach dem record() durch pull_one
    # hindurch. Ohne finally verlöre genau der Ausfall-Lauf sein Protokoll -
    # der einzige, für den es gebaut wurde (E-112).
    try:
        for s in symbols:
            # Per-symbol try: http_get_json records the failure itself (it knows
            # the key via log_key), but the loop must not die on the first bad
            # symbol — otherwise the remaining symbols are never even requested
            # and the protocol cannot say whether they were covered.
            try:
                df = pull_one(s, days, pull_log=plog)
            except Exception as exc:  # noqa: BLE001
                print(f"[CG] ERR {s}: {exc}")
                n_fail += 1
                # Auch hier protokollieren: http_get_json faengt nur HTTPError,
                # URLError, TimeoutError und JSONDecodeError. Alles andere
                # (BadGzipFile, UnicodeDecodeError, ConnectionResetError, und
                # das ValueError oben) entkaeme sonst spurlos aus dem Protokoll.
                # duplicate_keys macht eine etwaige Doppelung sichtbar.
                if plog is not None:
                    plog.record(
                        s.upper(),
                        http_status=getattr(exc, "code", None),
                        error=f"{type(exc).__name__}: {str(exc)[:200]}",
                    )
                continue
            to_parquet(df, out_dir / f"{s}_ohlc.parquet")
            # Parse-Ergebnis + Kontext am Recorder (der nimmt **extra).
            if plog is not None:
                plog.record(
                    s.upper(),
                    n_rows=len(df),
                    lookback_days=days,
                    quote_ccy="usd",
                )
            dfs.append(df)
        if dfs:
            big = pd.concat(dfs, ignore_index=True)
            to_parquet(big, out_dir / "crypto_ohlc_all.parquet")

        # E-147/Exit-Code: das per-Symbol-except oben faengt die Exception, die
        # vorher durch main() raiste und Exit-Code 1 erzeugte. Ein Protokoll zu
        # gewinnen und dafuer den Exit-Code zu verlieren ist kein Netto-Gewinn:
        # der Exit-Code ist die einzige Ebene, die ein Scheduler liest.
        # try/finally fuer die Evidenz, expliziter Exit-Code fuer das Ergebnis.
        if n_fail and not dfs:
            print(f"[CG] ERR alle {n_fail} Symbole failten", file=sys.stderr)
            failed_all = True
    finally:
        if plog is not None:
            plog.write()

    if failed_all:
        sys.exit(2)


if __name__ == "__main__":
    main()
