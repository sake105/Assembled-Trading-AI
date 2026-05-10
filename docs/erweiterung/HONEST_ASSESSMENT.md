# Ehrliche Selbsteinschätzung der ERWEITERUNG

Diese Datei dient der technischen Ehrlichkeit (CLAUDE.md §2.5, §15) und dokumentiert, was diese Erweiterung **kann** und was sie **nicht kann**.

## Was die Erweiterung liefert

✅ **38 neue Module** in 8 thematischen Bereichen, alle mit Docstrings, Type-Hints, akademischen Referenzen.

✅ **91 grüne Unit-Tests** (lokal, offline) — `pytest tests/erweiterung` exit 0.

✅ **End-to-End-Demo-Skript** (`scripts/erweiterung/run_demo_backtest.py`), läuft auf synthetischen Daten ohne Netzwerk.

✅ **Rigorose Statistik-Werkzeuge:**
- Combinatorial Purged CV (Lopez de Prado 2018)
- Deflated Sharpe Ratio (Bailey/Lopez 2014)
- White's Reality Check (2000) + Hansen's SPA-Test (2005)
- Probabilistic Sharpe Ratio
- Stationary Bootstrap (Politis/Romano 1994)

✅ **PIT-Sicherheit explizit codiert** für jede altdata-Quelle.

✅ **Fail-soft-Design:** keine Crashes bei API-Ausfall, fehlenden Optional-Packages oder leeren Inputs.

✅ **Dokumentation:** README, ARCHITECTURE, SIGNALS_REFERENCE, PAID_DATA_WISHLIST.

## Was die Erweiterung NICHT liefert

❌ **Keine Garantie, dass diese Signale auf echten Daten Alpha liefern.** Das ist empirisch zu prüfen — und genau dafür sind White-Reality-Check + Deflated-Sharpe da.

❌ **Keine CI-Verifikation** auf Ubuntu/Windows. Tests laufen lokal grün, aber `.github/workflows/` ist nicht angepasst (das wäre eine separate Diskussion mit dem Maintainer).

❌ **Keine Integration mit `assembled_core`-Pipeline.** Erweiterung ist isoliert; vollständige Integration wäre ein separates Projekt.

❌ **Keine Live-API-Tests.** Wir haben Logik getestet, nicht den live Aufruf der externen Dienste — diese sind rate-limitiert, instabil und ändern sich.

❌ **Keine Backtest-Ergebnisse auf Echtdaten.** Demo-Skript zeigt die Pipeline funktioniert, aber Renditezahlen sind aus synthetischen Daten und damit ohne ökonomische Aussage.

❌ **Keine Reddit-Pushshift-Garantie.** Pushshift wurde 2023 weitgehend abgeschaltet; arctic-shift ist Best-Effort. Bei Ausfall liefert das Modul leeres Ergebnis statt Crash.

❌ **Keine OptionMetrics-Quality-Skew-Daten.** Yahoo-Options sind Snapshot-only und für historische Backtests **nicht** brauchbar — dafür braucht es paid-data (siehe PAID_DATA_WISHLIST.md).

❌ **Keine Frontend-, Plattform-, oder Live-Trading-Komponenten.** Das ist Backend-Forschungslayer.

## Was zu beachten ist

⚠️ **Synthetic-Demo zeigt negative Sharpe** — das ist **gewollt** und **richtig**. Auf zufällig generierten Renditen darf kein Signal Edge haben, und Reality-Check + SPA bestätigen genau das. Wäre Sharpe positiv signifikant, wäre der Code kaputt.

⚠️ **Ein Pull-Request, keine Übernahmeforderung.** Der Maintainer entscheidet über Integration. Diese Erweiterung lebt auf Branch `ERWEITERUNG` und erweitert vorerst nichts am Mainline-Verhalten.

⚠️ **Survivorship-Bias** ist ein bekanntes Problem für freie Datenquellen. Echte Backtests benötigen CRSP / WRDS — dokumentiert in `PAID_DATA_WISHLIST.md`.

⚠️ **Optional-Dependencies** (scipy, sklearn, pytrends, yfinance, vaderSentiment) sind nicht in den Standard-Requirements. Bei Aktivierung einzelner Features müssen diese ggf. nachinstalliert werden.

## Was als nächste Schritte sinnvoll wäre (für den Maintainer)

1. **CI-Workflow erweitern** um `tests/erweiterung` zu laufen.
2. **Live-Test mit echten Daten** — z. B. SP500-Universum, 2018-2025, mit yfinance + EDGAR-Insider, Validierung der Reality-Check-Ergebnisse.
3. **Selektiver Pull** einzelner Module in den Mainline (z. B. nur `risk/tail_risk_evt.py` oder `backtest/deflated_sharpe.py`), wenn diese in Pipelines genutzt werden sollen.
4. **Erweiterung der Tests** um network-marked Tests, die in CI optional laufen.

## Überprüfung dieser Aussagen

```bash
# Tests prüfen
.venv/Scripts/python.exe -m pytest tests/erweiterung -v   # erwartet 91 passed

# Demo prüfen
.venv/Scripts/python.exe scripts/erweiterung/run_demo_backtest.py --n-days 750
# erwartet Reality-Check p-value > 0.05 für synthetische Daten

# Imports prüfen
.venv/Scripts/python.exe -c "import erweiterung; print(erweiterung.__version__)"
```
