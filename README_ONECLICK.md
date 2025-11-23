# One-Click Autopilot Add-on (v24)

Ziel: **Minimale Klicks**, maximale Sicherheit. Dieses Paket ergänzt dein bestehendes
`TradingSystem_Python_v24_skeleton` um **Ein-Kommando**-Skripte, Preflight-Checks und Logging.

## Nutzung (Windows, PowerShell)
1. Entpacke **dieses ZIP** **in dieselbe Ebene** wie dein Skeleton (oder direkt *in das Skeleton-Verzeichnis*).
2. PowerShell **als Administrator** öffnen (nur fürs erste Mal, wegen ExecutionPolicy).
3. `oneclick\run_all.ps1` ausführen:
   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   .\oneclick\run_all.ps1
   ```
   Das Skript erledigt: venv anlegen, Packages installieren, **Preflight-Doctor**, **Bootstrap**, **Backfill**, **Screener**,
   **UI-Start** (Streamlit). Logs: `logs/app.log` (JSON), `logs/app.err.log`.

## Nutzung (macOS/Linux, Bash)
```bash
chmod +x oneclick/run_all.sh
oneclick/run_all.sh
```

## Was wird installiert/ausgeführt?
- `.venv` (lokale virtuelle Umgebung)
- `pip install -r requirements.txt` (aus dem Skeleton)
- `python tools/preflight_doctor.py`
- `python tools/bootstrap.py`
- `python scripts/run_all.py`  (führt Backfill → Screener → UI-Start in Folge aus)

## Anpassungen
- API-Keys: `.env` im Projekt-Verzeichnis (aus `.env.example` kopieren)
- Watchlist: `config/watchlist.txt`
- Logging-Konfig: `config/logging.json`

## Sicherungen
- **Preflight-Doctor** bricht mit klarer Fehlermeldung ab (roter Code + Hinweis)
- **Idempotenz**: Bootstrap und Backfill sind wiederholbar
- **JSON-Logs**: maschinenlesbar + rotierend (Tagesrotation)

Viel Erfolg — du musst im Idealfall nur *ein Skript* starten.

