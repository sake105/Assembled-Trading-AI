"""One-shot Telegram alert setup helper.

Run AFTER you (1) created a bot via @BotFather, (2) put TELEGRAM_BOT_TOKEN=<token>
into .env, and (3) sent any message to your bot in Telegram.

This script: loads .env, reads the bot token, asks Telegram getUpdates for your
chat id, writes TELEGRAM_CHAT_ID into .env, and sends a confirmation message.
It NEVER prints the bot token.

    python scripts/setup_telegram.py
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ENV = REPO / ".env"


def _load_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(ENV)
    except Exception:
        # minimal fallback parser (never prints values)
        if ENV.exists():
            for line in ENV.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "setup/1.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read().decode())


def _upsert_env(key: str, value: str) -> None:
    """Write key=value into .env, replacing an existing line or appending. Never prints token."""
    lines = ENV.read_text(encoding="utf-8").splitlines() if ENV.exists() else []
    out, found = [], False
    for ln in lines:
        if ln.strip().startswith(f"{key}="):
            out.append(f"{key}={value}")
            found = True
        else:
            out.append(ln)
    if not found:
        out.append(f"{key}={value}")
    ENV.write_text("\n".join(out) + "\n", encoding="utf-8")


def main() -> int:
    _load_env()
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        print("[setup] TELEGRAM_BOT_TOKEN is not set in .env.")
        print(
            "        Add a line  TELEGRAM_BOT_TOKEN=<the token from @BotFather>  to .env, then re-run."
        )
        return 2

    print("[setup] token found (not shown). Asking Telegram who messaged the bot…")
    try:
        upd = _get(f"https://api.telegram.org/bot{token}/getUpdates")
    except Exception as exc:
        print(
            f"[setup] getUpdates failed: {exc}. Check the token is correct + you have internet."
        )
        return 1

    if not upd.get("ok"):
        print(
            f"[setup] Telegram rejected the token (ok=false): {upd.get('description', upd)}"
        )
        print(
            "        The token is probably wrong — re-copy it from @BotFather into .env."
        )
        return 1

    results = upd.get("result", [])
    chat = None
    for u in reversed(results):  # most recent first
        msg = u.get("message") or u.get("my_chat_member") or {}
        c = msg.get("chat") or {}
        if c.get("id"):
            chat = c
            break
    if not chat:
        print(
            "[setup] No messages found yet. Open your bot in Telegram and send it any message"
        )
        print("        (e.g. 'hi' or tap START), then re-run this script.")
        return 3

    chat_id = str(chat["id"])
    who = chat.get("first_name") or chat.get("title") or chat.get("username") or "?"
    print(f"[setup] Found your chat: '{who}' (chat_id={chat_id}).")

    cur = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if cur and cur != chat_id:
        print(
            f"[setup] NOTE: .env already had TELEGRAM_CHAT_ID={cur}; updating to {chat_id}."
        )
    _upsert_env("TELEGRAM_CHAT_ID", chat_id)
    os.environ["TELEGRAM_CHAT_ID"] = chat_id
    print("[setup] wrote TELEGRAM_CHAT_ID to .env.")

    # round-trip confirmation
    try:
        payload = json.dumps(
            {
                "chat_id": chat_id,
                "text": "✅ Assembled-Trading-AI alerts are wired up. You'll get pilot halt/heartbeat/drawdown alerts here.",
            }
        ).encode()
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=15) as r:
            ok = json.loads(r.read().decode()).get("ok")
        print(
            "[setup] SUCCESS — sent a test message to your Telegram."
            if ok
            else "[setup] chat_id saved, but the test send was rejected — check the bot isn't blocked."
        )
    except Exception as exc:
        print(f"[setup] chat_id saved, but test send failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
