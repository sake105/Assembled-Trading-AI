#!/usr/bin/env python3
"""Full system run: dataset discovery, multi-horizon backtests, EMA sweep, report.

ASCII-only CLI output. No new dependencies. Produces:
- output/system_run/dataset_inventory.json
- output/system_run/runs/<run_id>/ (metrics_summary.json, metrics_summary.csv)
- output/system_run/sweep/sweep_results.csv, sweep_results.json
- output/system_run/SYSTEM_RUN_REPORT.md

Reproduce:
  py -3 scripts/dev/run_full_system_backtests.py --output-root output/system_run
  py -3 scripts/dev/run_full_system_backtests.py --output-root output/system_run --dataset path/to.parquet
  py -3 scripts/dev/run_full_system_backtests.py --output-root output/system_run --include-synthetic
  py -3 scripts/dev/run_full_system_backtests.py --output-root output/system_run --synthetic-only  # smoke / CI
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


# ---------- Part B: Dataset discovery (stdlib + optional pandas for schema) ----------
def _scan_parquets_and_panels(base_dirs: list[Path]) -> list[Path]:
    """Return list of .parquet paths and panel/prices dirs (first .parquet inside)."""
    found: list[Path] = []
    seen: set[Path] = set()
    for base in base_dirs:
        if not base.exists():
            continue
        try:
            for p in base.rglob("*.parquet"):
                if p.is_file() and p not in seen:
                    seen.add(p)
                    found.append(p)
            for name in ("panel", "panels", "prices", "aggregates"):
                d = base / name
                if d.is_dir():
                    for q in d.rglob("*.parquet"):
                        if q.is_file() and q not in seen:
                            seen.add(q)
                            found.append(q)
        except OSError:
            pass
    return sorted(found, key=lambda p: str(p))


def _schema_hint(path: Path) -> list[str] | str:
    """Return column list if fast to read, else 'unknown'."""
    try:
        import pandas as pd

        df = pd.read_parquet(path, columns=None)
        return list(df.columns)
    except Exception:
        return "unknown"


def run_dataset_discovery(output_root: Path) -> list[dict]:
    """Scan for local price datasets; write dataset_inventory.json. Return inventory list."""
    search_dirs = [
        ROOT / "output",
        ROOT / "data",
        ROOT,
    ]
    paths = _scan_parquets_and_panels(search_dirs)
    inventory = []
    for p in paths:
        try:
            st = p.stat()
            mtime = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()
        except OSError:
            mtime = ""
            st = None
        size_bytes = st.st_size if st else 0
        schema = _schema_hint(p) if size_bytes < 50_000_000 else "unknown"
        rel = p.relative_to(ROOT) if ROOT in p.parents else p
        inventory.append(
            {
                "path": str(rel.as_posix()),
                "path_absolute": str(p.resolve()),
                "size_bytes": size_bytes,
                "modified_time": mtime,
                "schema_hint": schema,
            }
        )
    out_file = output_root / "dataset_inventory.json"
    output_root.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(inventory, f, indent=2, sort_keys=True)
    return inventory


def _pick_dataset(
    inventory: list[dict], dataset_path: str | None
) -> tuple[str | None, bool]:
    """Return (absolute path or None, is_synthetic). Prefer real price panel over synthetic."""
    if dataset_path:
        p = Path(dataset_path)
        if not p.is_absolute():
            p = ROOT / p
        if p.exists():
            return str(p.resolve()), False
        return None, True

    # Prefer paths that look like price panels (eod, panel, aggregates, smoke, sample)
    def score(item: dict) -> int:
        path = (item.get("path") or "").lower()
        path_abs = (item.get("path_absolute") or "").lower()
        combined = path + " " + path_abs
        if "factor" in combined and "year=" in combined:
            return 0
        if any(
            x in combined
            for x in ("eod", "aggregates", "panel", "smoke", "sample", "price")
        ):
            return 2
        return 1

    candidates = []
    for item in inventory:
        path_abs = item.get("path_absolute")
        if not path_abs or not Path(path_abs).exists():
            continue
        cols = item.get("schema_hint")
        if cols != "unknown" and isinstance(cols, list):
            if "timestamp" in cols and "symbol" in cols and "close" in cols:
                candidates.append((score(item), path_abs))
        elif cols == "unknown":
            candidates.append((score(item), path_abs))
    if not candidates:
        return None, True
    candidates.sort(key=lambda x: (-x[0], x[1]))
    return candidates[0][1], False


def _generate_synthetic_parquet(out_path: Path, start: str, end: str) -> None:
    """Write minimal synthetic EOD parquet (same semantics as smoke_backtest_local)."""
    import pandas as pd

    dates = pd.date_range(start=start, end=end, freq="B", tz="UTC")
    symbols = ["AAPL", "MSFT", "GOOGL"]
    rows = []
    for sym in symbols:
        for i, d in enumerate(dates):
            close = 100.0 + i * 0.05 + (i % 20) * 0.5
            rows.append(
                {
                    "timestamp": d,
                    "symbol": sym,
                    "open": close * 0.99,
                    "high": close * 1.01,
                    "low": close * 0.98,
                    "close": close,
                    "volume": 1_000_000.0,
                }
            )
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)


def _load_and_slice_prices(
    path: str, start_date: str | None, end_date: str | None
) -> "pd.DataFrame":  # noqa: F821
    import pandas as pd

    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if start_date:
        df = df[df["timestamp"] >= pd.Timestamp(start_date, tz="UTC")]
    if end_date:
        df = df[df["timestamp"] <= pd.Timestamp(end_date, tz="UTC")]
    return df


def _run_backtest_subprocess(
    price_file: Path,
    run_dir: Path,
    freq: str,
) -> int:
    """Run run_backtest_strategy.py; return exit code."""
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_backtest_strategy.py"),
        "--freq",
        freq,
        "--price-file",
        str(price_file),
        "--strategy",
        "trend_baseline",
        "--start-capital",
        "10000",
        "--out",
        str(run_dir),
        "--no-ledger",
    ]
    r = subprocess.run(cmd, cwd=str(ROOT), timeout=600, capture_output=True, text=True)
    if r.returncode != 0 and r.stderr:
        sys.stderr.write(r.stderr[:2000] + "\n")
    return r.returncode


def _run_analysis(run_dir: Path, summary_dir: Path, freq: str) -> int:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "dev" / "analyze_backtest_results.py"),
        "--out",
        str(run_dir),
        "--summary-dir",
        str(summary_dir),
        "--freq",
        freq,
    ]
    r = subprocess.run(cmd, cwd=str(ROOT), timeout=60)
    return r.returncode


def _compute_horizons(
    dataset_path: str, is_synthetic: bool
) -> list[tuple[str, str, str]]:
    """Return list of (run_id, start_date, end_date)."""
    import pandas as pd

    df = pd.read_parquet(dataset_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    ts = df["timestamp"]
    min_ts = ts.min()
    max_ts = ts.max()
    if pd.isna(min_ts) or pd.isna(max_ts) or len(ts) == 0:
        return [("full", "2020-01-01", "2023-12-31")]
    min_d = min_ts.date().isoformat()
    max_d = max_ts.date().isoformat()
    trading_dates = sorted(ts.dt.date.unique())
    n_days = len(trading_dates)
    horizons = []
    # 1y: last 252 trading days or all if fewer
    want_1y = min(252, n_days)
    start_1y = trading_dates[-want_1y] if want_1y else trading_dates[-1]
    end_1y = trading_dates[-1]
    horizons.append(("1y", start_1y.isoformat(), end_1y.isoformat()))
    # 3y if enough data (>= 750 trading days)
    if n_days >= 750:
        want_3y = min(750, n_days)
        start_3y = trading_dates[-want_3y]
        horizons.append(("3y", start_3y.isoformat(), end_1y.isoformat()))
    horizons.append(("full", min_d, max_d))
    return horizons


# ---------- Part C & D: Backtest and sweep ----------
def run_backtests(
    output_root: Path,
    dataset_path: str | None,
    is_synthetic: bool,
    freq: str,
    start_date: str | None,
    end_date: str | None,
    include_synthetic: bool,
) -> tuple[list[dict], str]:
    """Run 1y, 3y (if data), full backtests. Return list of run_info dicts and dataset_label."""
    runs_root = output_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    if not dataset_path or not Path(dataset_path).exists():
        if not include_synthetic:
            print(
                "No dataset found and --include-synthetic not set; skipping backtests.",
                file=sys.stderr,
            )
            return [], "none"
        synth_path = output_root / "synthetic" / "eod_synthetic.parquet"
        _generate_synthetic_parquet(synth_path, "2020-01-01", "2023-12-31")
        dataset_path = str(synth_path.resolve())
        is_synthetic = True
    dataset_label = "synthetic" if is_synthetic else Path(dataset_path).name

    horizons = _compute_horizons(dataset_path, is_synthetic)
    if is_synthetic:
        horizons = horizons[:1]  # smoke: one horizon keeps test within timeout
    run_infos = []
    for run_id, start_d, end_d in horizons:
        if start_date:
            start_d = start_date
        if end_date:
            end_d = end_date
        run_dir = runs_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        slice_path = run_dir / "price_slice.parquet"
        try:
            prices = _load_and_slice_prices(dataset_path, start_d, end_d)
            if prices.empty:
                print(
                    f"[{run_id}] No data in range {start_d} to {end_d}; skip.",
                    file=sys.stderr,
                )
                continue
            prices.to_parquet(slice_path, index=False)
        except Exception as e:
            print(f"[{run_id}] Slice failed: {e}", file=sys.stderr)
            continue
        print(f"[{run_id}] Running backtest {start_d} to {end_d}...")
        code = _run_backtest_subprocess(slice_path, run_dir, freq)
        if code != 0:
            print(f"[{run_id}] Backtest exit code {code}", file=sys.stderr)
            run_infos.append(
                {"run_id": run_id, "start": start_d, "end": end_d, "exit_code": code}
            )
            continue
        code = _run_analysis(run_dir, run_dir, freq)
        if code != 0:
            print(f"[{run_id}] Analysis exit code {code}", file=sys.stderr)
        metrics_path = run_dir / "metrics_summary.json"
        run_infos.append(
            {
                "run_id": run_id,
                "start": start_d,
                "end": end_d,
                "exit_code": 0,
                "metrics_path": str(metrics_path) if metrics_path.exists() else None,
            }
        )
    return run_infos, dataset_label


def _create_trend_signal_fn(ma_fast: int, ma_slow: int):
    """Build trend signal function (mirrors run_backtest_strategy.create_trend_baseline_signal_fn)."""
    from src.assembled_core.signals.rules_trend import (
        generate_trend_signals_from_prices,
    )

    def signal_fn(prices_df):
        return generate_trend_signals_from_prices(
            prices_df, ma_fast=ma_fast, ma_slow=ma_slow
        )

    return signal_fn


def run_sweep(
    output_root: Path,
    dataset_path: str | None,
    freq: str,
    run_infos: list[dict],
) -> list[dict]:
    """EMA parameter sweep on 1y slice only. Return list of sweep rows."""
    import pandas as pd
    from src.assembled_core.portfolio.position_sizing import (
        compute_target_positions_from_trend_signals,
    )
    from src.assembled_core.qa.backtest_engine import run_portfolio_backtest
    from src.assembled_core.qa.metrics import compute_all_metrics

    sweep_dir = output_root / "sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    price_slice = output_root / "runs" / "1y" / "price_slice.parquet"
    if not price_slice.exists():
        print(
            "Sweep skipped: no 1y price slice (run 1y backtest first).", file=sys.stderr
        )
        return []
    prices = pd.read_parquet(price_slice)
    prices["timestamp"] = pd.to_datetime(prices["timestamp"], utc=True)
    prices = prices.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    ema_fast_list = [10, 20, 30]
    ema_slow_list = [50, 60, 80, 100]
    grid = [(f, s) for f in ema_fast_list for s in ema_slow_list if f < s]
    results = []
    start_capital = 10000.0
    for i, (ma_fast, ma_slow) in enumerate(grid):
        print(f"Sweep {i + 1}/{len(grid)}: ma_fast={ma_fast} ma_slow={ma_slow}")
        signal_fn = _create_trend_signal_fn(ma_fast=ma_fast, ma_slow=ma_slow)

        def _sizing_fn(sig, cap):
            return compute_target_positions_from_trend_signals(sig, cap)

        try:
            result = run_portfolio_backtest(
                prices=prices,
                signal_fn=signal_fn,
                position_sizing_fn=_sizing_fn,
                start_capital=start_capital,
                include_costs=True,
                include_trades=True,
                rebalance_freq=freq,
                include_ledger=False,
                compute_features=True,
            )
            metrics = compute_all_metrics(
                equity=result.equity,
                trades=result.trades,
                start_capital=start_capital,
                freq=freq,
                risk_free_rate=0.0,
            )
            row = {
                "start": str(prices["timestamp"].min().date()),
                "end": str(prices["timestamp"].max().date()),
                "freq": freq,
                "ma_fast": ma_fast,
                "ma_slow": ma_slow,
                "total_return": metrics.total_return,
                "cagr": metrics.cagr,
                "sharpe_ratio": metrics.sharpe_ratio,
                "max_drawdown_pct": metrics.max_drawdown_pct,
                "trades_count": metrics.total_trades or 0,
                "turnover": metrics.turnover,
            }
            results.append(row)
        except Exception as e:
            print(f"  Error: {e}", file=sys.stderr)
            results.append(
                {
                    "start": "",
                    "end": "",
                    "freq": freq,
                    "ma_fast": ma_fast,
                    "ma_slow": ma_slow,
                    "total_return": None,
                    "cagr": None,
                    "sharpe_ratio": None,
                    "max_drawdown_pct": None,
                    "trades_count": 0,
                    "turnover": None,
                    "error": str(e),
                }
            )

    with (sweep_dir / "sweep_results.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as f:
        json.dump(results, f, indent=2, sort_keys=True)
    if results:
        with (sweep_dir / "sweep_results.csv").open(
            "w", encoding="utf-8", newline="\n"
        ) as f:
            w = csv.DictWriter(f, fieldnames=sorted(results[0].keys()))
            w.writeheader()
            w.writerows(results)
    return results


def _load_run_metrics(run_dir: Path) -> dict | None:
    p = run_dir / "metrics_summary.json"
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


# ---------- Part E: Report ----------
def write_system_run_report(
    output_root: Path,
    dataset_label: str,
    is_synthetic: bool,
    run_infos: list[dict],
    sweep_results: list[dict],
) -> None:
    """Write output/system_run/SYSTEM_RUN_REPORT.md."""
    _runs_root = output_root / "runs"
    lines = [
        "# System Run Report",
        "",
        "Generated by scripts/dev/run_full_system_backtests.py",
        "",
        "## Dataset",
        "",
        "- **Used:** "
        + ("synthetic (labeled)" if is_synthetic else "real")
        + f" - {dataset_label}",
        "",
        "## Strategy summary",
        "",
        "See docs/STRATEGY_REVIEW_AUTOGENERATED.md. Trend-baseline: EMA crossover, long-only, equal-weight sizing.",
        "",
        "## Baseline results",
        "",
    ]
    for r in run_infos:
        run_id = r.get("run_id", "")
        start = r.get("start", "")
        end = r.get("end", "")
        metrics_path = r.get("metrics_path")
        if metrics_path and Path(metrics_path).exists():
            with open(metrics_path, "r", encoding="utf-8") as f:
                m = json.load(f)
            tr = m.get("total_return")
            cagr = m.get("cagr")
            sharpe = m.get("sharpe_ratio")
            dd = m.get("max_drawdown_pct")
            trades = m.get("total_trades")
            lines.append(f"### {run_id} ({start} to {end})")
            lines.append("")
            lines.append(f"- total_return: {tr}")
            lines.append(f"- cagr: {cagr}")
            lines.append(f"- sharpe_ratio: {sharpe}")
            lines.append(f"- max_drawdown_pct: {dd}")
            lines.append(f"- total_trades: {trades}")
            lines.append("")
        else:
            lines.append(f"### {run_id} - no metrics (exit_code={r.get('exit_code')})")
            lines.append("")

    lines.append("## Sweep top 10 (by total_return)")
    lines.append("")
    valid = [s for s in sweep_results if s.get("total_return") is not None]
    for row in sorted(valid, key=lambda x: (x.get("total_return") or 0), reverse=True)[
        :10
    ]:
        lines.append(
            f"- ma_fast={row.get('ma_fast')} ma_slow={row.get('ma_slow')}: return={row.get('total_return')} sharpe={row.get('sharpe_ratio')} max_dd={row.get('max_drawdown_pct')} trades={row.get('trades_count')}"
        )
    lines.append("")
    lines.append("## Sweep top 10 (by sharpe_ratio)")
    lines.append("")
    for row in sorted(
        valid, key=lambda x: (x.get("sharpe_ratio") or -999), reverse=True
    )[:10]:
        lines.append(
            f"- ma_fast={row.get('ma_fast')} ma_slow={row.get('ma_slow')}: sharpe={row.get('sharpe_ratio')} return={row.get('total_return')} max_dd={row.get('max_drawdown_pct')}"
        )
    lines.append("")
    lines.append("## Drawdown-adjusted (return / |max_dd| if dd < 0)")
    lines.append("")

    def dd_adj(r):
        tr = r.get("total_return") or 0
        dd = r.get("max_drawdown_pct")
        if dd is not None and dd < 0 and dd != 0:
            return tr / abs(dd)
        return tr if tr else -999

    for row in sorted(valid, key=dd_adj, reverse=True)[:10]:
        lines.append(
            f"- ma_fast={row.get('ma_fast')} ma_slow={row.get('ma_slow')}: dd_adj={dd_adj(row):.4f} return={row.get('total_return')} max_dd={row.get('max_drawdown_pct')}"
        )
    lines.append("")
    lines.append("## Recommendations (no-concept-change)")
    lines.append("")
    lines.append(
        "- If trades are too few: expand universe, check signal trigger (EMA windows), verify price panel quality."
    )
    lines.append(
        "- If turnover is high: adjust costs (commission_bps, spread_w), consider longer EMA slow."
    )
    lines.append(
        "- If results sensitive to params: use default 20/60 and add guardrails (max turnover, min bars)."
    )
    lines.append("")
    lines.append("## Anomalies to check")
    lines.append("")
    lines.append("- Zero trades: may indicate no crossover in range or QC blocking.")
    lines.append("- Constant equity curve: no orders executed.")
    lines.append("- Missing/NaN prices: data quality; run QC.")
    lines.append(
        "- Lookahead: use PIT checks (disclosure_date <= as_of) for event data."
    )
    lines.append("")

    report_path = output_root / "SYSTEM_RUN_REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {report_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Full system backtests and sweep.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output/system_run"),
        help="Output root",
    )
    parser.add_argument("--freq", type=str, default="1d")
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument(
        "--dataset", type=str, default=None, help="Path to price parquet"
    )
    parser.add_argument(
        "--include-synthetic", action="store_true", help="Use synthetic if no dataset"
    )
    parser.add_argument(
        "--synthetic-only",
        action="store_true",
        help="Force synthetic data only (no discovery; for smoke tests)",
    )
    parser.add_argument(
        "--skip-sweep", action="store_true", help="Skip EMA parameter sweep"
    )
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    if not output_root.is_absolute():
        output_root = (ROOT / output_root).resolve()

    if args.synthetic_only:
        inventory = []
        dataset_path = None
        is_synthetic = True
        (output_root / "synthetic").mkdir(parents=True, exist_ok=True)
        synth_path = output_root / "synthetic" / "eod_synthetic.parquet"
        # Short window so smoke test stays well within 420s timeout (1 horizon only)
        _generate_synthetic_parquet(synth_path, "2022-06-01", "2023-06-01")
        dataset_path = str(synth_path.resolve())
        dataset_label = "synthetic"
        output_root.mkdir(parents=True, exist_ok=True)
        with (output_root / "dataset_inventory.json").open(
            "w", encoding="utf-8", newline="\n"
        ) as f:
            json.dump(
                [
                    {
                        "path": "synthetic/eod_synthetic.parquet",
                        "path_absolute": dataset_path,
                        "size_bytes": (
                            synth_path.stat().st_size if synth_path.exists() else 0
                        ),
                        "modified_time": datetime.now(timezone.utc).isoformat(),
                        "schema_hint": "synthetic",
                    }
                ],
                f,
                indent=2,
                sort_keys=True,
            )
        print("Using synthetic-only dataset.")
    else:
        print("Dataset discovery...")
        inventory = run_dataset_discovery(output_root)
        dataset_path, is_synthetic = _pick_dataset(inventory, args.dataset)
        if dataset_path:
            print(f"Using dataset: {dataset_path} (synthetic={is_synthetic})")
        else:
            print("No dataset selected.")
        dataset_label = Path(dataset_path).name if dataset_path else "none"

    print("Running backtests...")
    run_infos, dataset_label_out = run_backtests(
        output_root=output_root,
        dataset_path=dataset_path,
        is_synthetic=is_synthetic,
        freq=args.freq,
        start_date=args.start_date,
        end_date=args.end_date,
        include_synthetic=args.include_synthetic,
    )
    if not args.synthetic_only:
        dataset_label = dataset_label_out

    sweep_results: list[dict] = []
    if not args.skip_sweep and (
        dataset_path or args.include_synthetic or args.synthetic_only
    ):
        print("Running parameter sweep...")
        sweep_results = run_sweep(output_root, dataset_path, args.freq, run_infos)

    print("Writing SYSTEM_RUN_REPORT.md...")
    write_system_run_report(
        output_root, dataset_label, is_synthetic, run_infos, sweep_results
    )

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
