# Archivierte Legacy-Skripte (2026-08-16, Audit-Plan 6.1)

- `assemble_eod_daily.py` — aus `scripts/data/`. Selbst als LEGACY markiert;
  GEFAEHRLICH: ein Aufruf haette den 197-Symbol-Live-Cache
  (`output/aggregates/daily.parquet`) mit einem 2-Symbol-Restbestand
  ueberschrieben. Kein Aufrufer im Repo (Nutzungsaudit
  `docs/DATEN_UND_NUTZUNGSAUDIT.md` §3). Archiviert statt geloescht
  (Projektkonvention).
