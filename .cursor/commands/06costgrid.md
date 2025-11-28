---
name: Cost Grid (5min)
description: Sweep über Kommission/Spread/Impact für 5min-Strategie.
arguments:
  - name: commission
    default: "0 0.5 1.0"
    description: Space-separierte Werte (bps)
  - name: spread
    default: "0.25 0.5 1.0"
    description: Space-separierte Gewichte
  - name: impact
    default: "0.5 1.0 2.0"
    description: Space-separierte Gewichte
  - name: ema_fast
    default: 20
  - name: ema_slow
    default: 60
  - name: notional
    default: 10000
run: |
  .\.venv\Scripts\python.exe .\scripts\sprint9_cost_grid.py --freq 5min --commission-bps {{commission}} --spread-w {{spread}} --impact-w {{impact}} --ema-fast {{ema_fast}} --ema-slow {{ema_slow}} --notional {{notional}}
---

Write your command content here.

This command will be available in chat with /06costgrid
