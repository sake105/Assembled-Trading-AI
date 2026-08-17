"""QuantConnect REST-API-v2-Client für das Forschungsmandat.

Zweck: verdict-fähige Backtests auf QuantConnects survivorship-bias-FREIEN
PIT-Daten (US Equities inkl. Delistings, historische Index-Zusammensetzung) —
löst die Mandat-§2.5-Blockade ohne Norgate-Abo. Free-Tier: 1 Backtest-Node.

Auth: HMAC-timestamped (sha256(token:timestamp), Basic userId:hash) aus .env
(QUANTCONNECT_USER_ID / QUANTCONNECT_API_TOKEN). Werte werden nie geloggt.

Kern-Endpoints: projects/create|read, files/create|update|read,
compile/create|read, backtests/create|read.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
API = "https://www.quantconnect.com/api/v2"


def _creds() -> tuple[str, str]:
    if "QUANTCONNECT_USER_ID" not in os.environ:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    return os.environ["QUANTCONNECT_USER_ID"], os.environ["QUANTCONNECT_API_TOKEN"]


def call(endpoint: str, payload: dict | None = None, *, retries: int = 3) -> dict:
    uid, token = _creds()
    last: Exception | None = None
    for attempt in range(retries):
        ts = str(int(time.time()))
        hashed = hashlib.sha256(f"{token}:{ts}".encode()).hexdigest()
        auth = base64.b64encode(f"{uid}:{hashed}".encode()).decode()
        req = urllib.request.Request(
            f"{API}/{endpoint}",
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={
                "Authorization": f"Basic {auth}",
                "Timestamp": ts,
                "Content-Type": "application/json",
            },
            method="POST" if payload is not None else "GET",
        )
        try:
            out: dict = json.loads(
                urllib.request.urlopen(req, timeout=60).read().decode()
            )
            return out
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"QC API {endpoint} failed after {retries} tries: {last}")


# --------------------------------------------------------------- convenience
def create_project(name: str, language: str = "Py") -> int:
    r = call("projects/create", {"name": name, "language": language})
    if not r.get("success"):
        raise RuntimeError(f"create_project: {r}")
    return int(r["projects"][0]["projectId"])


def list_projects() -> list[dict]:
    r = call("projects/read")
    return list(r.get("projects", []))


def upsert_file(project_id: int, name: str, content: str) -> None:
    r = call(
        "files/create", {"projectId": project_id, "name": name, "content": content}
    )
    if not r.get("success"):
        r = call(
            "files/update",
            {"projectId": project_id, "name": name, "content": content},
        )
        if not r.get("success"):
            raise RuntimeError(f"upsert_file {name}: {r}")


def compile_project(project_id: int, *, timeout_s: int = 120) -> str:
    r = call("compile/create", {"projectId": project_id})
    if not r.get("success"):
        raise RuntimeError(f"compile/create: {r}")
    cid = r["compileId"]
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        st = call("compile/read", {"projectId": project_id, "compileId": cid})
        if st.get("state") == "BuildSuccess":
            return cid
        if st.get("state") == "BuildError":
            raise RuntimeError(f"BuildError: {st.get('logs')}")
        time.sleep(3)
    raise TimeoutError("compile timeout")


def run_backtest(project_id: int, compile_id: str, name: str) -> str:
    r = call(
        "backtests/create",
        {"projectId": project_id, "compileId": compile_id, "backtestName": name},
    )
    if not r.get("success"):
        raise RuntimeError(f"backtests/create: {r}")
    return str(r["backtest"]["backtestId"])


def read_backtest(project_id: int, backtest_id: str) -> dict:
    return call("backtests/read", {"projectId": project_id, "backtestId": backtest_id})


def wait_backtest(
    project_id: int, backtest_id: str, *, timeout_s: int = 1800, poll_s: int = 15
) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        r = read_backtest(project_id, backtest_id)
        bt = r.get("backtest") or {}
        if bt.get("completed"):
            return r
        time.sleep(poll_s)
    raise TimeoutError("backtest timeout")


if __name__ == "__main__":
    print(json.dumps(call("authenticate"), indent=2))
    print(f"projects: {[p.get('name') for p in list_projects()]}")
