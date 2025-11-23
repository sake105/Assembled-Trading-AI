# Run-All Orchestrator (Sprint 8–10)

## Voller Durchlauf (Seed + Rehydrate + Features + Cost + Backtest + CostGrid + Portfolio + Git)
pwsh -File .\scripts\run_all_sprint10.ps1 `
  -Seed -Rehydrate -Features -Cost -Backtest -CostGrid -Portfolio -Sync `
  -Freq 5min -StartCapital 10000 -Exposure 1 -MaxLeverage 1 `
  -CommissionBps 0.5 -SpreadW 1 -ImpactW 1

## Ohne Seed/Sync (nur Datenfluss)
pwsh -File .\scripts\run_all_sprint10.ps1 `
  -Rehydrate -Features -Cost -Backtest -CostGrid -Portfolio `
  -Freq 5min -StartCapital 10000 -CommissionBps 0.5

## Mit Discord-Notify (Webhook per ENV)
$env:DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/…"
pwsh -File .\scripts\run_all_sprint10.ps1 `
  -Rehydrate -Features -Cost -Backtest -CostGrid -Portfolio -Notify `
  -Freq 5min -StartCapital 10000 -CommissionBps 0.5

## Mit Discord-Notify (Webhook einmalig per Parameter)
pwsh -File .\scripts\run_all_sprint10.ps1 `
  -Rehydrate -Features -Cost -Backtest -CostGrid -Portfolio -Notify `
  -DiscordWebhookUrl "https://discord.com/api/webhooks/…" `
  -Freq 5min -StartCapital 10000

## Quick/QuickDays explizit steuern (Sprint8)
pwsh -File .\scripts\run_all_sprint10.ps1 `
  -Rehydrate -Features `
  -Freq 5min -Symbols "AAPL,MSFT" -Quick -QuickDays 180
# Quick entfernen → komplette Historie:
pwsh -File .\scripts\run_all_sprint10.ps1 -Rehydrate -Features -Quick:$false

---

## Troubleshooting (Kurz)
- **Python nicht gefunden:** `tools\activate_python.ps1` wird automatisch geladen. Danach sollte `Using Python: …\.venv\Scripts\python.exe` erscheinen.
- **Seed fehlt:** Warnung ist ok, wenn `scripts\seed_demo_data.py` nicht existiert.
- **Discord 404 / leerer Hook:** Entweder `-DiscordWebhookUrl` setzen oder `$env:DISCORD_WEBHOOK_URL`.
- **Pandas FutureWarnings:** harmlos, wir haben bereits `groupby(..., group_keys=False)` o.ä. im Code angepasst.

