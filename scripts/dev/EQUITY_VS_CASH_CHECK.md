# Equity vs. Cash – 60-Sekunden-Check und MTM-Verifikation

Wenn du **Fills** hast, aber **0 % Return** oder **konstante/negative Equity** siehst, liegt oft daran, dass die Engine-Kurve **nur Cash** abbildet (Startkapital + Cash-Deltas), nicht **Cash + Mark-to-Market-Positionswert**.

## 1) 60-Sekunden-Check: Ist equity_curve_*.csv nur Cash?

```powershell
$R = "output\system_run\debug_relaxed_1y"
py -3 -c "import pandas as pd; import pathlib; r=pathlib.Path(r'output/system_run/debug_relaxed_1y'); eq=pd.read_csv(r/'equity_curve_1d.csv'); print('equity rows',len(eq)); m=eq.loc[eq['equity'].idxmin()]; M=eq.loc[eq['equity'].idxmax()]; print('min',m['equity'],'at',m.get('timestamp',m.get('date'))); print('max',M['equity'],'at',M.get('timestamp',M.get('date')))"
```

- **Min nahe 0 oder negativ, Max = Startkapital** → sehr wahrscheinlich **nur Cash** (Käufe ziehen Cash ab; Positionswert fehlt).
- **Echte Equity** = Cash + (Holdings × Preise) und sollte nicht ohne Grund negativ werden.

## 2) Harte Verifikation: Cash + Mark-to-Market selbst rechnen

Skript: `scripts/dev/verify_equity_mtm.py`  
(Verwendet `trades_1d.csv` und `price_slice.parquet`; wenn der Parquet nicht im Run-Dir liegt, wird `benchmark/trend_baseline/1y/price_slice.parquet` genutzt.)

```powershell
py -3 scripts/dev/verify_equity_mtm.py
```

Erwartung:
- `cash_min` / `cash_max`: Cash kann negativ werden (Käufe + Kosten).
- `equity_mtm_min` / `equity_mtm_max`: Sollte sinnvoll schwanken und nicht konstant 0 oder nur Cash sein.

Wenn **equity_mtm** plausibel ist, die **equity_curve_1d.csv** der Engine aber flach/negativ ist → die Engine schreibt aktuell **Cash als Equity** (Mark-to-Market fehlt).

## 3) Warum Cash negativ werden kann

- Cash-Gate prüft oft **pro Order** gegen `available_cash`, nicht **kumulativ** über mehrere BUYs und inkl. Kosten.
- Equal-Weight mit 3 BUYs kann fast 100 % Cash verbrauchen; mit Kosten rutscht Cash leicht ins Minus.

Mögliche No-Regrets-Anpassungen (ohne Strategie-Konzept zu ändern):
- Cash-Gate **kumulativ**: BUYs nach Zeit sortieren, laufend Notional + geschätzte Kosten vom verfügbaren Cash abziehen.
- Oder **Cash-Puffer** (z. B. nur 99,5 % investierbar) bei Sizing/Order-Gen.

## 4) Wo die Engine „Equity“ baut

In `src/assembled_core/pipeline/portfolio.py` ist die Kurve aktuell:
**equity = start_capital + cumsum(cash_delta)**  
→ reines **Cash**, kein Mark-to-Market.  
Für echte Performance-Metriken müsste **equity = cash + (holdings × prices)** pro Bar ergänzt werden.
