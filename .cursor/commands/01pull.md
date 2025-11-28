---
name: Pull Intraday (Yahoo)
description: Lädt 1m-Daten für Symbole nach data/raw/1min (UTC).
arguments:
  - name: symbols
    description: CSV der Ticker
    default: "AAPL,MSFT"
  - name: days
    description: Anzahl Tage rückwärts
    default: 2
run: |
  pwsh -NoProfile -ExecutionPolicy Bypass -File .\scripts\live\pull_intraday.ps1 -Symbols {{symbols}} -Days {{days}}
---

Write your command content here.

This command will be available in chat with /01pull
