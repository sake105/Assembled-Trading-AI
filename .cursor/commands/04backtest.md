---
name: Backtest
description: Führt den Backtest aus und schreibt Equity/Report.
arguments:
  - name: freq
    default: "5min"
    description: 5min oder 1d
  - name: start_capital
    default: 10000
    description: Startkapital
run: |
  .\.venv\Scripts\python.exe .\scripts\sprint9_backtest.py --freq {{freq}} --start-capital {{start_capital}}
---

Write your command content here.

This command will be available in chat with /04backtest
