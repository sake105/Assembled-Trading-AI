---
name: Portfolio Run
description: Bewertet Portfolio inkl. Kostenannahmen.
arguments:
  - name: freq
    default: "5min"
  - name: start_capital
    default: 10000
  - name: commission_bps
    default: 0.5
  - name: spread_w
    default: 0.5
  - name: impact_w
    default: 1.0
run: |
  .\.venv\Scripts\python.exe .\scripts\sprint10_portfolio.py --freq {{freq}} --start-capital {{start_capital}} --commission-bps {{commission_bps}} --spread-w {{spread_w}} --impact-w {{impact_w}}
---

Write your command content here.

This command will be available in chat with /05portfolio
