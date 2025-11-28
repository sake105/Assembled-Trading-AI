---
name: Execute EMA Orders (5min)
description: Erzeugt Orders mittels EMA( fast, slow ) auf 5min-Preisen.
arguments:
  - name: ema_fast
    default: 20
    description: Schneller EMA
  - name: ema_slow
    default: 60
    description: Langsamer EMA
  - name: price_file
    default: ".\\output\\aggregates\\5min.parquet"
    description: Pfad zu 5min-Preisen
run: |
  .\.venv\Scripts\python.exe .\scripts\sprint9_execute.py --freq 5min --ema-fast {{ema_fast}} --ema-slow {{ema_slow}} --price-file "{{price_file}}"
---

Write your command content here.

This command will be available in chat with /03execute
