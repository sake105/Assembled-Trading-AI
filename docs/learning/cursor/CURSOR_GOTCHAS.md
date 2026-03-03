Cursor Gotchas — Typische Fallen
================================

Liste häufiger Stolpersteine bei der Arbeit mit Cursor in diesem Repo.

## 1) Statefulness / Nebenwirkungen

- Shell/Interpreter sind **zustandsbehaftet**:
  - Nach einem fehlgeschlagenen Command können Pfad/Env verändert sein.
  - Immer explizit ins Projekt-Root wechseln (`cd` mit absolutem Pfad) bevor kritische Commands laufen.

## 2) Pfade & OS-Eigenheiten

- Unter Windows: Backslashes, Laufwerksbuchstaben und UTF-8 Pfade beachten.
- In Skripten möglichst **`Path`-APIs** (Python) oder `Join-Path` (PowerShell) statt manueller String-Konkatenation nutzen.

## 3) Zeitzonen & Timestamps

- Backtests nutzen typischerweise **UTC**-Timestamps.
- Lokale Zeitzonen / `datetime.now()` ohne TZ können deterministische Backtests brechen.
- Immer `tz="UTC"` / `tz_localize` / `tz_convert` bewusst setzen.

## 4) Deterministische Backtests

- Keine versteckten Random-Quellen ohne Fixierung (Seeds).
- Keine Zeitabhängigkeit (z.B. „heute minus X Tage“) in Core-Backtests; stattdessen explizite Zeiträume.

## 5) Pfad-Filter / .cursorignore

- Manche Output-Pfade sind in `.cursorignore` ausgeblendet.
- Cursor kann sie nicht direkt via Code-Tools sehen; für Analysen in diesen Verzeichnissen Powershell/Python-Commands nutzen.

