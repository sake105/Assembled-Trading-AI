"""Set GitHub Actions secrets via REST API using stored git credential.

Usage:
    python scripts/set_gh_secrets.py --env-file .env --names ALPACA_API_KEY,ALPACA_API_SECRET

Reads names from an env-style file (KEY=VALUE per line), fetches the repo's
actions public key, sealed-box-encrypts each value, PUTs to the repo.

Never prints secret material. Token is pulled from `git credential fill`.
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

from nacl import encoding, public

REPO = "sake105/Assembled-Trading-AI"


def _get_token() -> str:
    proc = subprocess.run(
        ["git", "credential", "fill"],
        input="url=https://github.com\n\n",
        capture_output=True,
        text=True,
        check=True,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("password="):
            return line.split("=", 1)[1]
    raise RuntimeError("no password in git credential response")


def _api(method: str, path: str, token: str, body: dict | None = None) -> dict:
    url = f"https://api.github.com{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req) as resp:
        raw = resp.read()
        try:
            return json.loads(raw) if raw else {}
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"GitHub API returned non-JSON response: {raw[:200]}"
            ) from exc


def _encrypt(public_key_b64: str, secret: str) -> str:
    pub = public.PublicKey(public_key_b64.encode(), encoding.Base64Encoder())
    sealed = public.SealedBox(pub).encrypt(secret.encode())
    return base64.b64encode(sealed).decode()


def _parse_env(path: Path, names: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip("'\"")
        if key in names:
            out[key] = val
    missing = [n for n in names if n not in out]
    if missing:
        raise SystemExit(f"missing in {path}: {missing}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-file", required=True)
    ap.add_argument("--names", required=True, help="comma-separated secret names")
    args = ap.parse_args()

    names = [n.strip() for n in args.names.split(",") if n.strip()]
    values = _parse_env(Path(args.env_file), names)

    token = _get_token()
    pk = _api("GET", f"/repos/{REPO}/actions/secrets/public-key", token)
    key_id = pk["key_id"]
    public_key = pk["key"]

    for name in names:
        enc = _encrypt(public_key, values[name])
        _api(
            "PUT",
            f"/repos/{REPO}/actions/secrets/{name}",
            token,
            {"encrypted_value": enc, "key_id": key_id},
        )
        print(f"[OK] set secret {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
