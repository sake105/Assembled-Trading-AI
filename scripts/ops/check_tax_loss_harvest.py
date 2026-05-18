"""Tax-Loss-Harvesting Detection (audit C2-064).

Reads a trade ledger CSV (with realized P&L) and an open-positions CSV
(with unrealized P&L) and produces a Tax-Loss-Harvesting analysis:

1. YTD realized gains / losses (split positive vs negative).
2. Unrealized-loss candidates in open positions, sorted by absolute loss.
3. Verrechnungspotential: how much realized loss is needed to offset
   YTD realized gains.

This is a READ-ONLY audit/decision-support script — it does NOT generate
orders. See `docs/TAX_LOSS_HARVESTING.md` for the full DE-Q3-Workflow.

Input schema (CSV):
- ledger.csv: columns at minimum `symbol`, `close_date`, `pnl_eur`
  (negative for losses, positive for gains, in EUR).
- positions.csv: columns `symbol`, `qty`, `entry_price`, `current_price`
  (in trading currency; EUR conversion applied via simple ratio if
  ``--fx-rate-usd-eur`` is provided).

Usage::

    python scripts/ops/check_tax_loss_harvest.py
    python scripts/ops/check_tax_loss_harvest.py \\
        --ledger output/ledger_realized.csv \\
        --positions output/open_positions.csv \\
        --tax-year 2025

Output: JSON + Markdown under ``output/ops/tax_loss_harvest.{json,md}``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------


def analyse_realized_pnl(ledger: pd.DataFrame, tax_year: int) -> dict[str, Any]:
    """Split realized P&L into gains and losses for the given tax year.

    Args:
        ledger: DataFrame with columns ``close_date`` (parseable to datetime),
            ``pnl_eur`` (signed), ``symbol``.
        tax_year: Integer year (e.g. 2025).

    Returns:
        Dict with ``total_gains_eur``, ``total_losses_eur``,
        ``net_pnl_eur``, ``n_winning_trades``, ``n_losing_trades``,
        plus per-symbol breakdown.
    """
    if ledger.empty:
        return {
            "total_gains_eur": 0.0,
            "total_losses_eur": 0.0,
            "net_pnl_eur": 0.0,
            "n_winning_trades": 0,
            "n_losing_trades": 0,
            "by_symbol": {},
        }
    if "close_date" not in ledger.columns or "pnl_eur" not in ledger.columns:
        raise ValueError("ledger must have 'close_date' and 'pnl_eur' columns")
    df = ledger.copy()
    df["close_date"] = pd.to_datetime(df["close_date"], errors="coerce", utc=True)
    df = df[df["close_date"].dt.year == tax_year]
    df["pnl_eur"] = pd.to_numeric(df["pnl_eur"], errors="coerce")
    df = df.dropna(subset=["pnl_eur"])
    gains_mask = df["pnl_eur"] > 0
    losses_mask = df["pnl_eur"] < 0
    total_gains = float(df.loc[gains_mask, "pnl_eur"].sum())
    total_losses = float(df.loc[losses_mask, "pnl_eur"].sum())  # negative
    by_symbol: dict[str, float] = {}
    if "symbol" in df.columns:
        by_symbol = (
            df.groupby("symbol")["pnl_eur"].sum().sort_values().round(2).to_dict()
        )
    return {
        "tax_year": int(tax_year),
        "total_gains_eur": round(total_gains, 2),
        "total_losses_eur": round(total_losses, 2),
        "net_pnl_eur": round(total_gains + total_losses, 2),
        "n_winning_trades": int(gains_mask.sum()),
        "n_losing_trades": int(losses_mask.sum()),
        "by_symbol": by_symbol,
    }


def find_harvesting_candidates(
    positions: pd.DataFrame,
    fx_rate_usd_eur: float = 1.0,
) -> list[dict[str, Any]]:
    """Identify open positions with unrealized losses.

    Args:
        positions: DataFrame with ``symbol``, ``qty``, ``entry_price``,
            ``current_price``. unrealized PnL = (current - entry) * qty.
            If quotes are USD, supply ``fx_rate_usd_eur`` to convert to EUR.
        fx_rate_usd_eur: Multiplier USD → EUR. Default 1.0 (no conversion).

    Returns:
        List of dicts, sorted by unrealized_eur ascending (worst loss first).
        Only candidates with unrealized_eur < 0 are included.
    """
    if positions.empty:
        return []
    required = {"symbol", "qty", "entry_price", "current_price"}
    missing = required - set(positions.columns)
    if missing:
        raise ValueError(f"positions missing columns: {missing}")
    df = positions.copy()
    for col in ("qty", "entry_price", "current_price"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=list(required - {"symbol"}))
    df["unrealized_quote_ccy"] = (df["current_price"] - df["entry_price"]) * df["qty"]
    df["unrealized_eur"] = df["unrealized_quote_ccy"] * float(fx_rate_usd_eur)
    losers = df[df["unrealized_eur"] < 0].sort_values("unrealized_eur")
    candidates: list[dict[str, Any]] = []
    for _, row in losers.iterrows():
        candidates.append(
            {
                "symbol": str(row["symbol"]),
                "qty": float(row["qty"]),
                "entry_price": float(row["entry_price"]),
                "current_price": float(row["current_price"]),
                "unrealized_eur": round(float(row["unrealized_eur"]), 2),
                # F-senior-1: entry_price=0 (corp-action artefact / bad CSV row)
                # would raise ZeroDivisionError. Keep candidate in list (the
                # unrealized_eur is still well-defined from qty * current_price)
                # but report pct_loss as NaN to signal the data quality issue.
                "pct_loss": (
                    round(
                        float(
                            (row["current_price"] / row["entry_price"] - 1.0) * 100.0
                        ),
                        2,
                    )
                    if row["entry_price"]
                    else float("nan")
                ),
            }
        )
    return candidates


def compute_offset_potential(
    realized: dict[str, Any], candidates: list[dict[str, Any]]
) -> dict[str, Any]:
    """How much unrealized loss would be needed to offset realized gains."""
    realized_gains = float(realized.get("total_gains_eur", 0.0))
    realized_losses = float(realized.get("total_losses_eur", 0.0))
    net = realized_gains + realized_losses  # losses are negative
    # If net > 0, we still have unoffset gains; targeting up to `net` in
    # unrealized losses to harvest would zero out the taxable basis.
    target_offset_eur = max(net, 0.0)
    # Cumulative sum of candidate losses (sorted ascending = worst first)
    cumulative: list[dict[str, Any]] = []
    running = 0.0
    needed = (
        -target_offset_eur
    )  # negative — we want SUM(candidate.unrealized) ≤ -target
    for c in candidates:
        running += c["unrealized_eur"]
        meets = running <= needed
        cumulative.append(
            {
                "symbol": c["symbol"],
                "unrealized_eur": c["unrealized_eur"],
                "cumulative_loss_eur": round(running, 2),
                "meets_target": bool(meets),
            }
        )
    # Find the smallest prefix of losers that meets the target
    enough_idx = next((i for i, c in enumerate(cumulative) if c["meets_target"]), None)
    return {
        "realized_net_pnl_eur": round(net, 2),
        "target_loss_to_offset_gains_eur": round(target_offset_eur, 2),
        "harvest_cumulative_path": cumulative,
        "min_n_positions_to_neutralise": (
            enough_idx + 1 if enough_idx is not None else None
        ),
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_tax_loss_check(
    ledger_path: Path,
    positions_path: Path,
    tax_year: int,
    fx_rate_usd_eur: float = 1.0,
) -> dict[str, Any]:
    """Full pipeline: read both CSVs, run all 3 analyses, aggregate."""
    if not ledger_path.exists():
        ledger = pd.DataFrame(columns=["symbol", "close_date", "pnl_eur"])
        ledger_status = f"missing: {ledger_path}"
    else:
        ledger = pd.read_csv(ledger_path)
        ledger_status = f"ok: {len(ledger)} rows"
    if not positions_path.exists():
        positions = pd.DataFrame(
            columns=["symbol", "qty", "entry_price", "current_price"]
        )
        positions_status = f"missing: {positions_path}"
    else:
        positions = pd.read_csv(positions_path)
        positions_status = f"ok: {len(positions)} rows"

    realized = analyse_realized_pnl(ledger, tax_year)
    candidates = find_harvesting_candidates(positions, fx_rate_usd_eur)
    offset = compute_offset_potential(realized, candidates)

    return {
        "tax_year": int(tax_year),
        "ledger_path": str(ledger_path),
        "ledger_status": ledger_status,
        "positions_path": str(positions_path),
        "positions_status": positions_status,
        "fx_rate_usd_eur": float(fx_rate_usd_eur),
        "realized_pnl": realized,
        "harvesting_candidates": candidates,
        "offset_potential": offset,
        "limitations": (
            "Read-only audit. NO orders generated. Operator must select "
            "and execute harvesting trades manually via normal order "
            "workflow. Wash-sale not enforced (DE has no wash-sale rule). "
            "FX conversion is a simple ratio multiplier — for production, "
            "use accounting/tax_lots.py with ECB reference rates per trade."
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    r = report
    realized = r["realized_pnl"]
    offset = r["offset_potential"]
    lines = [
        f"# Tax-Loss-Harvesting Report (tax year {r['tax_year']})",
        "",
        f"**Ledger:** `{r['ledger_path']}` — {r['ledger_status']}",
        f"**Positions:** `{r['positions_path']}` — {r['positions_status']}",
        f"**FX rate USD→EUR:** {r['fx_rate_usd_eur']}",
        "",
        "## Realised P&L (YTD)",
        f"- **Total gains:** €{realized['total_gains_eur']}",
        f"- **Total losses:** €{realized['total_losses_eur']}",
        f"- **Net:** €{realized['net_pnl_eur']}",
        f"- **Winning trades:** {realized['n_winning_trades']}",
        f"- **Losing trades:** {realized['n_losing_trades']}",
        "",
        "## Offset Potential",
        f"- **Realized net (target to neutralise):** €{offset['target_loss_to_offset_gains_eur']}",
        f"- **Minimum positions to neutralise via harvesting:** "
        f"{offset['min_n_positions_to_neutralise']}",
        "",
        f"## Harvesting Candidates ({len(r['harvesting_candidates'])} positions with unrealized loss)",
    ]
    if r["harvesting_candidates"]:
        lines.append("")
        lines.append("| Symbol | Qty | Entry | Current | Unrealized EUR | Loss % |")
        lines.append("|:-------|----:|------:|--------:|---------------:|-------:|")
        for c in r["harvesting_candidates"][:20]:  # top 20 worst losses
            lines.append(
                f"| {c['symbol']} | {c['qty']} | {c['entry_price']} | "
                f"{c['current_price']} | {c['unrealized_eur']} | {c['pct_loss']}% |"
            )
        if len(r["harvesting_candidates"]) > 20:
            lines.append(
                f"\n_({len(r['harvesting_candidates']) - 20} more candidates omitted)_"
            )
    else:
        lines.append("")
        lines.append("(no open positions with unrealized loss)")
    lines.append("")
    lines.append("## Limitations")
    lines.append(r["limitations"])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ledger",
        type=Path,
        default=Path("output/ledger_realized.csv"),
    )
    parser.add_argument(
        "--positions",
        type=Path,
        default=Path("output/open_positions.csv"),
    )
    parser.add_argument("--tax-year", type=int, default=datetime.now().year)
    parser.add_argument("--fx-rate-usd-eur", type=float, default=1.0)
    parser.add_argument("--out", type=Path, default=Path("output/ops"))
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    report = run_tax_loss_check(
        ledger_path=args.ledger,
        positions_path=args.positions,
        tax_year=args.tax_year,
        fx_rate_usd_eur=args.fx_rate_usd_eur,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    json_path = args.out / "tax_loss_harvest.json"
    md_path = args.out / "tax_loss_harvest.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    logger.info("[tax_loss_harvest] JSON: %s", json_path)
    logger.info("[tax_loss_harvest] Markdown: %s", md_path)
    logger.info(
        "[tax_loss_harvest] tax_year=%d realized_net=%.2f harvesting_candidates=%d",
        report["tax_year"],
        report["realized_pnl"]["net_pnl_eur"],
        len(report["harvesting_candidates"]),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
