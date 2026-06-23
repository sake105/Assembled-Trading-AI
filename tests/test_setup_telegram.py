"""Unit tests for setup_telegram._upsert_env — the .env credential-file mutator.
Only the pure file logic is tested; the interactive network path (getUpdates/sendMessage)
is not (it needs a real token + would send a Telegram message)."""

import importlib.util
import pathlib

_spec = importlib.util.spec_from_file_location(
    "setup_telegram",
    pathlib.Path(__file__).resolve().parent.parent / "scripts" / "setup_telegram.py",
)
st = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(st)


def test_upsert_appends_when_absent(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("ALPACA_KEY=abc\n", encoding="utf-8")
    monkeypatch.setattr(st, "ENV", env)
    st._upsert_env("TELEGRAM_CHAT_ID", "999")
    txt = env.read_text(encoding="utf-8")
    assert "ALPACA_KEY=abc" in txt  # unrelated key preserved
    assert "TELEGRAM_CHAT_ID=999" in txt


def test_upsert_replaces_existing(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("TELEGRAM_CHAT_ID=old\nOTHER=1\n", encoding="utf-8")
    monkeypatch.setattr(st, "ENV", env)
    st._upsert_env("TELEGRAM_CHAT_ID", "new")
    txt = env.read_text(encoding="utf-8")
    assert "TELEGRAM_CHAT_ID=new" in txt
    assert "TELEGRAM_CHAT_ID=old" not in txt
    assert txt.count("TELEGRAM_CHAT_ID=") == 1  # no duplicate
    assert "OTHER=1" in txt


def test_upsert_matches_spaces_around_eq(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("TELEGRAM_CHAT_ID = old\n", encoding="utf-8")
    monkeypatch.setattr(st, "ENV", env)
    st._upsert_env("TELEGRAM_CHAT_ID", "new")
    txt = env.read_text(encoding="utf-8")
    assert txt.count("TELEGRAM_CHAT_ID") == 1  # matched + replaced, not duplicated
    assert "new" in txt


def test_upsert_matches_export_prefix(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("export TELEGRAM_CHAT_ID=old\n", encoding="utf-8")
    monkeypatch.setattr(st, "ENV", env)
    st._upsert_env("TELEGRAM_CHAT_ID", "new")
    txt = env.read_text(encoding="utf-8")
    assert txt.count("TELEGRAM_CHAT_ID") == 1
    assert "TELEGRAM_CHAT_ID=new" in txt


def test_upsert_preserves_comments(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# my creds\n#TELEGRAM_CHAT_ID=commented\n", encoding="utf-8")
    monkeypatch.setattr(st, "ENV", env)
    st._upsert_env("TELEGRAM_CHAT_ID", "5")
    txt = env.read_text(encoding="utf-8")
    assert "# my creds" in txt
    assert "#TELEGRAM_CHAT_ID=commented" in txt  # commented line untouched
    assert "TELEGRAM_CHAT_ID=5" in txt  # real value appended separately


def test_upsert_creates_file_when_absent(tmp_path, monkeypatch):
    env = tmp_path / ".env"  # does not exist
    monkeypatch.setattr(st, "ENV", env)
    st._upsert_env("TELEGRAM_CHAT_ID", "42")
    assert env.exists()
    assert "TELEGRAM_CHAT_ID=42" in env.read_text(encoding="utf-8")
