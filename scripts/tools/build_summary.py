# scripts/tools/build_summary.py
from pathlib import Path
import datetime as dt

ROOT = Path(__file__).resolve().parents[2]
OUT  = ROOT / "output"
SUMMARY = OUT / "summary.md"

def safe_read(p: Path) -> str:
    if p.exists():
        try: return p.read_text(encoding="utf-8").strip()
        except: return f"_Konnte {p.name} nicht lesen_"
    return f"_{p.name} nicht gefunden_"

def section(title: str, body: str) -> str:
    return f"## {title}\n\n{body}\n"

def main():
    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    parts = [f"# Run Summary\n\n_Generiert: {ts}_\n"]

    parts.append(section("Backtest Performance (Sprint 9)",
        safe_read(OUT / "performance_report.md")))

    parts.append(section("Cost Grid (Sprint 9)",
        safe_read(OUT / "cost_grid_report.md")))

    parts.append(section("Portfolio Report (Sprint 10)",
        safe_read(OUT / "portfolio_report.md")))

    # Optional: Param-Sweep
    parts.append(section("Param Sweep (Sprint 10)",
        safe_read(OUT / "param_sweep_report.md")))

    # Optionale Links
    links = []
    for p in ["equity_curve_5min.csv", "portfolio_equity_5min.csv", "portfolio_trades.csv", "orders.csv"]:
        fp = OUT / p
        if fp.exists():
            links.append(f"- `{p}`")
    if links:
        parts.append(section("Dateien", "\n".join(links)))

    SUMMARY.write_text("\n".join(parts), encoding="utf-8")
    print(f"[SUMMARY] wrote {SUMMARY}")

if __name__ == "__main__":
    main()
