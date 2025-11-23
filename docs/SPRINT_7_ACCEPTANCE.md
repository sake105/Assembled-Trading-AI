# SPRINT 7 - ACCEPTANCE (Resample + QC)

**Datum (Berlin):** 2025-10-05 13:26:13 +02:00

## Ergebnis
- **Status:** PASS

## Gate-Parameter
- MinRowsPerInterval: **50**
- MaxGapMinutes: **480**
- MaxTotalGaps: **100000**

## Resample-Report
- Intervall **5min**: Zeilen **2595** -> Datei: scripts\..\output\aggregates\assembled_intraday_5min.parquet
- Intervall **15min**: Zeilen **867** -> Datei: scripts\..\output\aggregates\assembled_intraday_15min.parquet
- Intervall **30min**: Zeilen **435** -> Datei: scripts\..\output\aggregates\assembled_intraday_30min.parquet
- Intervall **60min**: Zeilen **219** -> Datei: scripts\..\output\aggregates\assembled_intraday_60min.parquet

## QC Gaps (Kurzfassung)
- total_gaps: **0**
- max_gap_minutes_overall: **0**

## Artefakte
output/aggregates/resample_report.json
output/qc/intraday_gaps.csv
output/qc/intraday_gaps_summary.json


