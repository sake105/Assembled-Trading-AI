---
name: Resample 1m -> 5m
description: Resampled data/raw/1min zu output/aggregates/5min.parquet.
arguments:
  - name: input_dir
    description: Ordner mit 1m-Parquets
    default: ".\\data\\raw\\1min"
  - name: out_file
    description: Ziel-Parquet (5min)
    default: ".\\output\\aggregates\\5min.parquet"
run: |
  .\.venv\Scripts\python.exe .\scripts\dev\resample_1m_to_5m.py --input {{input_dir}} --output {{out_file}}
---

Write your command content here.

This command will be available in chat with /02resample
