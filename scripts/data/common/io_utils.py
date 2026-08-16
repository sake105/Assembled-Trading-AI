from __future__ import annotations

import gzip
import json
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

USER_AGENT = "AssembledTradingAI/1.0"


class RateBucket:
    """Simple token-bucket rate limiter."""

    def __init__(self, calls_per_minute: int = 5):
        self._interval = 60.0 / max(calls_per_minute, 1)
        self._last = 0.0

    def consume(self):
        now = time.monotonic()
        wait = self._interval - (now - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# --- HTTP (stdlib, ohne requests) ---
#
# Beide Getter akzeptieren optional einen PullLog (assembled_core.data.pull_log).
# Wird einer uebergeben, schreibt der Aufruf eine Protokollzeile - auch und
# gerade bei Leerergebnis oder Fehler. Das ist die Lehre aus E-112: ohne
# Anfrage-Protokoll ist "keine Datei" nicht von "nie angefragt" zu unterscheiden,
# und jede spaetere Coverage-Aussage ist geraten.
#
# Ohne pull_log verhalten sich beide Funktionen exakt wie zuvor.


def _record_pull(pull_log, key, *, window, http_status, n_rows, error):
    """Protokollzeile schreiben, falls ein PullLog uebergeben wurde.

    Nie-raise: Buchhaltung darf einen Ingest-Lauf nicht toeten.
    """
    if pull_log is None:
        return
    # Kein try/except: PullLog.record ist per Vertrag nie-raise
    # (pull_log.py), ein zusaetzlicher stiller Schlucker wuerde einen echten
    # Defekt dort verbergen statt ihn zu zeigen.
    pull_log.record(
        key,
        window=window,
        http_status=http_status,
        n_rows=n_rows,
        error=error,
    )


def _status_of(ex) -> int | None:
    """HTTP-Code aus einer Exception ziehen, soweit vorhanden."""
    return getattr(ex, "code", None)


def http_get_json(
    url: str,
    headers: dict | None = None,
    retries: int = 3,
    backoff: float = 0.8,
    *,
    pull_log=None,
    log_key: str | None = None,
    log_window=None,
):
    h = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip"}
    if headers:
        h.update(headers)
    key = log_key or url
    last_ex = None
    for i in range(retries):
        try:
            req = Request(url, headers=h)
            with urlopen(req, timeout=30) as resp:
                data = resp.read()
                if resp.headers.get("Content-Encoding") == "gzip":
                    data = gzip.decompress(data)
                payload = json.loads(data.decode("utf-8"))
                # n_rows must mean ROWS, not "length of whatever came back".
                # An earlier version used len(payload) for any sized object, so
                # a dict response like {"code": "ok", "data": []} reported
                # n_rows=2 (its top-level keys) and the protocol recorded an
                # EMPTY result as "ok" — reinstating exactly the confusion E-112
                # exists to prevent. Only a list is countable here; for anything
                # else the caller knows the row count and must pass it.
                _record_pull(
                    pull_log,
                    key,
                    window=log_window,
                    http_status=getattr(resp, "status", 200),
                    n_rows=len(payload) if isinstance(payload, list) else None,
                    error=None,
                )
                return payload
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as ex:
            last_ex = ex
            time.sleep((i + 1) * backoff)
    _record_pull(
        pull_log,
        key,
        window=log_window,
        http_status=_status_of(last_ex),
        n_rows=0,
        error=f"{type(last_ex).__name__}: {last_ex}",
    )
    raise last_ex


def http_get_text(
    url: str,
    headers: dict | None = None,
    *,
    pull_log=None,
    log_key: str | None = None,
    log_window=None,
):
    h = {"User-Agent": USER_AGENT}
    if headers:
        h.update(headers)
    key = log_key or url
    req = Request(url, headers=h)
    try:
        with urlopen(req, timeout=30) as resp:
            txt = resp.read().decode("utf-8", errors="replace")
            status = getattr(resp, "status", 200)
    except (HTTPError, URLError, TimeoutError) as ex:
        _record_pull(
            pull_log,
            key,
            window=log_window,
            http_status=_status_of(ex),
            n_rows=0,
            error=f"{type(ex).__name__}: {ex}",
        )
        raise
    # NOT len(txt): that is a CHARACTER count, so the four-byte body "null"
    # would be recorded as n_rows=4 and therefore "ok". A text endpoint has no
    # row count the transport layer can know — leave it unknown and let the
    # caller record the parsed row count.
    _record_pull(
        pull_log,
        key,
        window=log_window,
        http_status=status,
        n_rows=None,
        error=None,
    )
    return txt


# --- Parquet/CSV helpers ---
SCHEMA_EQ = [
    "timestamp",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "provider",
]


def to_parquet(df: pd.DataFrame, out_path: Path):
    ensure_dir(out_path.parent)
    df.to_parquet(out_path, index=False)


# Harmonisierung für OHLC Frames


def normalize_ohlc(
    df: pd.DataFrame, symbol: str, provider: str, tz="UTC"
) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}  # noqa: F841
    rename = {}
    for k in ["open", "high", "low", "close", "volume"]:
        for c in list(df.columns):
            if c.lower() == k:
                rename[c] = k
                break
    if "timestamp" not in [c.lower() for c in df.columns]:
        # haeufige Varianten
        cand = ["time", "date", "datetime", "timestamp"]
        for c in df.columns:
            if c.lower() in cand:
                rename[c] = "timestamp"
                break
    df = df.rename(columns=rename)
    if "timestamp" not in df.columns:
        raise ValueError("timestamp column not found after rename")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["symbol"] = symbol
    df["provider"] = provider
    # minimale Sortierung & Spaltenreihenfolge
    keep = [c for c in SCHEMA_EQ if c in df.columns]
    df = df[keep].sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    return df
