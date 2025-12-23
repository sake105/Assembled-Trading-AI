# scripts/tools/param_sweep_report.py
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output"
CSV = OUT / "portfolio_sweep.csv"
MD = OUT / "param_sweep_report.md"


def main():
    if not CSV.exists():
        MD.write_text("_portfolio_sweep.csv nicht gefunden_", encoding="utf-8")
        return
    df = pd.read_csv(CSV)
    # Erwartete Spalten: exp, comm, final
    # Robust sort: höchstes final zuerst
    df = df.sort_values(by="final", ascending=False)
    top = df.head(10)

    lines = [
        "| Rank | Exposure | Commission(bps) | Final Equity |",
        "|-----:|---------:|----------------:|------------:|",
    ]
    for i, row in enumerate(top.itertuples(index=False), start=1):
        lines.append(f"| {i} | {row.exp:.2f} | {row.comm:.2f} | {row.final:,.2f} |")

    MD.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
