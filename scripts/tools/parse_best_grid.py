# scripts/tools/parse_best_grid.py
# DEPRECATED: Grid searches are now handled via batch_run (scripts/cli.py batch_run).
# This tool may still be useful for parsing legacy grid outputs,
# but new grid searches should use the batch runner with its built-in summary artifacts.
# See: docs/BATCH_RUNNER_P4.md

import json
import re
import math
import sys
from pathlib import Path

REPORT = Path(__file__).resolve().parents[2] / "output" / "cost_grid_report.md"


def parse_table(md: str):
    # Markdown-Tabelle erwarten: | col1 | col2 | ... |
    rows = []
    lines = [ln.strip() for ln in md.splitlines() if ln.strip()]
    # finde erste Separator-Zeile (---) und starte danach
    header = None
    for i, ln in enumerate(lines):
        if re.match(r"^\|\s*[-:]+", ln):
            # header ist Zeile vorher
            header = [h.strip() for h in lines[i - 1].strip("|").split("|")]
            start = i + 1
            break
    if not header:
        return rows, []

    for ln in lines[start:]:
        if not ln.startswith("|"):
            continue
        cols = [c.strip() for c in ln.strip("|").split("|")]
        if len(cols) != len(header):
            continue
        rows.append(dict(zip(header, cols)))
    return rows, header


def to_float(s):
    s = s.strip().replace(",", ".")
    s = re.sub(r"[^0-9\.\-eE]", "", s)
    if s == "":
        return float("nan")
    try:
        return float(s)
    except (ValueError, TypeError):
        return float("nan")


def main():
    if not REPORT.exists():
        print(json.dumps({"error": f"{REPORT} not found"}))
        sys.exit(0)
    md = REPORT.read_text(encoding="utf-8")
    rows, header = parse_table(md)
    if not rows:
        print(json.dumps({"error": "no rows parsed"}))
        sys.exit(0)

    # Spaltennamen tolerant finden
    def get(row, *keys):
        for k in keys:
            if k in row:
                return row[k]
        return ""

    enriched = []
    for r in rows:
        enriched.append(
            {
                "pf": to_float(get(r, "PF", "ProfitFactor", "Profit-Factor")),
                "sharpe": to_float(get(r, "Sharpe")),
                "commission_bps": to_float(
                    get(r, "commission_bps", "comm_bps", "commission")
                ),
                "spread_w": to_float(get(r, "spread_w", "spread")),
                "impact_w": to_float(get(r, "impact_w", "impact")),
                "trades": to_float(get(r, "trades", "Trades")),
                "equity_final": to_float(get(r, "equity", "Equity", "equity_final")),
                "_raw": r,
            }
        )

    # Ranking: max PF, Tiebreak max Sharpe
    valid = [r for r in enriched if not math.isnan(r["pf"])]
    if not valid:
        print(json.dumps({"error": "no numeric PF"}))
        sys.exit(0)

    valid.sort(key=lambda x: (x["pf"], x["sharpe"]), reverse=True)
    best = valid[0]
    print(
        json.dumps(
            {
                "commission_bps": best["commission_bps"],
                "spread_w": best["spread_w"],
                "impact_w": best["impact_w"],
                "pf": best["pf"],
                "sharpe": best["sharpe"],
                "trades": best["trades"],
                "equity_final": best["equity_final"],
            }
        )
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
