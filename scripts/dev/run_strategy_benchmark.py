#!/usr/bin/env python3
"""Strategy benchmark: run multiple variants from benchmark_variants.json, ablation, OOS, scoreboard.

Uses existing backtest entrypoints. ASCII-only CLI. No new dependencies.

Outputs:
- output/system_run/benchmark/<variant_id>/<horizon>/ (metrics_summary.json, etc.)
- output/system_run/benchmark/scoreboard.csv, scoreboard.json
- output/system_run/benchmark/ablation_summary.csv
- output/system_run/benchmark/oos_report.md (if --oos)
- output/system_run/benchmark/oos_sweep_report.md (if --oos-sweep; skipped if too few trading days)
- output/system_run/benchmark/BENCHMARK_REPORT.md

Reproduce:
  py -3 scripts/dev/run_strategy_benchmark.py --output-root output/system_run
  py -3 scripts/dev/run_strategy_benchmark.py --output-root output/system_run --dataset path/to.parquet
  py -3 scripts/dev/run_strategy_benchmark.py --output-root output/system_run --synthetic-only --quick
  py -3 scripts/dev/run_strategy_benchmark.py --output-root output/system_run --oos --max-variants 3
  py -3 scripts/dev/run_strategy_benchmark.py --output-root output/system_run --include-synthetic
  py -3 scripts/dev/run_strategy_benchmark.py --output-root output/system_run --quick --max-variants 6 --include-synthetic
  py -3 scripts/dev/run_strategy_benchmark.py --output-root output/system_run --quick --include-synthetic --sweep-filters

OOS-sweep debug (PowerShell): scripts/dev/run_oos_sweep_debug.ps1 -ParquetPath "<path>"
  Checks parquet history, runs --oos-sweep without --quick, then shows oos_sweep_report.md or run_inputs.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Default commission bps for cost scaling (engine default often 0)
DEFAULT_COMMISSION_BPS = 5.0


def _scan_parquets(base_dirs: list[Path]) -> list[Path]:
    found = []
    seen = set()
    for base in base_dirs:
        if not base.exists():
            continue
        try:
            for p in base.rglob("*.parquet"):
                if p.is_file() and p not in seen:
                    seen.add(p)
                    found.append(p)
        except OSError:
            pass
    return sorted(found, key=lambda p: str(p))


def _schema_ok(path: Path) -> bool:
    try:
        import pandas as pd

        df = pd.read_parquet(path, columns=None)
        return (
            "timestamp" in df.columns
            and "symbol" in df.columns
            and "close" in df.columns
        )
    except Exception:
        return False


def _pick_dataset(
    output_root: Path, dataset_path: str | None
) -> tuple[str | None, bool]:
    if dataset_path:
        p = Path(dataset_path)
        if not p.is_absolute():
            p = ROOT / p
        if p.exists():
            return str(p.resolve()), False
        return None, True
    for base in [ROOT / "output", ROOT / "data"]:
        for p in _scan_parquets([base]):
            if "factor" in str(p) and "year=" in str(p):
                continue
            if _schema_ok(p):
                return str(p.resolve()), False
    return None, True


def _generate_synthetic(out_path: Path, start: str, end: str) -> None:
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


def _load_and_slice(path: str, start: str, end: str):
    import pandas as pd

    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df[
        (df["timestamp"] >= pd.Timestamp(start, tz="UTC"))
        & (df["timestamp"] <= pd.Timestamp(end, tz="UTC"))
    ]
    return df


def _ensure_ohlcv(df):
    """Ensure dataframe has open, high, low, close, volume so QC and execution do not block.
    Returns (df, ohlcv_info): ohlcv_info has columns_synthesized, synthetic_ohlcv, synthetic_volume_value (ASCII, deterministic).
    """
    synthesized = []
    vol_val = None
    if "close" not in df.columns:
        return df, {
            "columns_synthesized": [],
            "synthetic_ohlcv": False,
            "synthetic_volume_value": None,
        }
    close = df["close"].astype(float)
    if "open" not in df.columns:
        df["open"] = close
        synthesized.append("open")
    if "high" not in df.columns:
        df["high"] = close
        synthesized.append("high")
    if "low" not in df.columns:
        df["low"] = close
        synthesized.append("low")
    if "volume" not in df.columns:
        df["volume"] = 1.0e6
        synthesized.append("volume")
        vol_val = 1.0e6
    return df, {
        "columns_synthesized": sorted(synthesized),
        "synthetic_ohlcv": len(synthesized) > 0,
        "synthetic_volume_value": vol_val if vol_val is not None else None,
    }


def _compute_horizons(path: str):
    import pandas as pd

    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    ts = df["timestamp"]
    trading_dates = sorted(ts.dt.date.unique())
    n = len(trading_dates)
    if n == 0:
        return [("full", "2020-01-01", "2023-12-31")]
    min_d = trading_dates[0].isoformat()
    max_d = trading_dates[-1].isoformat()
    out = [("1y", trading_dates[-min(252, n)].isoformat(), max_d)]
    if n >= 750:
        out.append(("3y", trading_dates[-750].isoformat(), max_d))
    out.append(("full", min_d, max_d))
    return out


SUPPORTED_STRATEGIES = (
    "trend_baseline",
    "event_insider_shipping",
    "multifactor_long_short",
    "trend_baseline_rsi_filter",
    "trend_baseline_vol_filter",
    "trend_baseline_regime_gate",
    "trend_baseline_realized_vol_filter",
    "trend_baseline_liquidity_filter",
    "trend_baseline_rsi_vol_combo_filter",
)


def _validate_variant_wiring(variant: dict) -> list[dict]:
    """Validate variant strategy and params. Return list of anomaly dicts (empty if ok)."""
    issues = []
    vid = variant.get("id", "unknown")
    strategy = variant.get("strategy", "trend_baseline")
    if strategy not in SUPPORTED_STRATEGIES:
        issues.append(
            {
                "variant_id": vid,
                "horizon": None,
                "type": "unsupported_strategy",
                "message": f"Strategy '{strategy}' not in {list(SUPPORTED_STRATEGIES)}",
            }
        )
    if strategy == "multifactor_long_short":
        bundle = (variant.get("params") or {}).get("bundle_path")
        if not bundle:
            issues.append(
                {
                    "variant_id": vid,
                    "horizon": None,
                    "type": "missing_bundle_path",
                    "message": "multifactor_long_short requires params.bundle_path",
                }
            )
        else:
            p = Path(bundle)
            if not p.is_absolute():
                p = ROOT / bundle
            if not p.exists():
                issues.append(
                    {
                        "variant_id": vid,
                        "horizon": None,
                        "type": "bundle_not_found",
                        "message": f"Bundle not found: {bundle}",
                    }
                )
    return issues


def _run_backtest(
    price_file: Path,
    run_dir: Path,
    strategy: str,
    freq: str,
    commission_bps: float | None,
    bundle_path: str | None = None,
    params: dict | None = None,
    no_strict_session_gate: bool = False,
) -> int:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_backtest_strategy.py"),
        "--freq",
        freq,
        "--price-file",
        str(price_file),
        "--strategy",
        strategy,
        "--start-capital",
        "10000",
        "--out",
        str(run_dir),
        "--no-ledger",
    ]
    if no_strict_session_gate:
        cmd.append("--no-strict-session-gate")
    if (params or {}).get("rebalance") == "weekly":
        cmd.extend(["--rebalance", "weekly"])
    if commission_bps is not None:
        cmd.extend(["--commission-bps", str(round(commission_bps, 2))])
    if strategy == "multifactor_long_short" and bundle_path:
        bp = Path(bundle_path)
        if not bp.is_absolute():
            bp = ROOT / bundle_path
        if bp.exists():
            cmd.extend(["--bundle-path", str(bp.resolve())])
    params = params or {}
    if strategy == "trend_baseline_rsi_filter":
        cmd.extend(["--rsi-entry", str(float(params.get("rsi_entry", 55)))])
        cmd.extend(["--rsi-overbought", str(float(params.get("rsi_overbought", 80)))])
    if strategy == "trend_baseline_vol_filter":
        cmd.extend(["--vol-cap", str(float(params.get("vol_cap", 0.30)))])
    if strategy == "trend_baseline_regime_gate":
        cmd.extend(
            ["--risk-on-threshold", str(float(params.get("risk_on_threshold", 0)))]
        )
    if strategy == "trend_baseline_realized_vol_filter":
        cmd.extend(["--rv-cap", str(float(params.get("rv_cap", 0.30)))])
        cmd.extend(["--rv-window", str(int(params.get("rv_window", 20)))])
    if strategy == "trend_baseline_liquidity_filter":
        cmd.extend(["--liquidity-min", str(float(params.get("liquidity_min", 0)))])
    if strategy == "trend_baseline_rsi_vol_combo_filter":
        cmd.extend(["--rsi-entry", str(float(params.get("rsi_entry", 55)))])
        cmd.extend(["--rsi-overbought", str(float(params.get("rsi_overbought", 80)))])
        cmd.extend(["--vol-cap", str(float(params.get("vol_cap", 0.30)))])
    r = subprocess.run(cmd, cwd=str(ROOT), timeout=300, capture_output=True, text=True)
    if r.returncode != 0 and r.stderr:
        sys.stderr.write(r.stderr[:1500] + "\n")
    return r.returncode


def _run_analysis(run_dir: Path, freq: str) -> int:
    r = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "dev" / "analyze_backtest_results.py"),
            "--out",
            str(run_dir),
            "--summary-dir",
            str(run_dir),
            "--freq",
            freq,
        ],
        cwd=str(ROOT),
        timeout=60,
    )
    return r.returncode


def _enhance_metrics_from_equity(run_dir: Path, freq: str, row: dict) -> dict:
    """Add worst_month, best_month, drawdown_duration, pct_profitable_months, median_monthly_return from equity."""
    import pandas as pd

    eq_path = run_dir / f"equity_curve_{freq}.csv"
    if not eq_path.exists():
        return row
    try:
        eq = pd.read_csv(eq_path)
        if "timestamp" not in eq.columns or "equity" not in eq.columns:
            return row
        eq["timestamp"] = pd.to_datetime(eq["timestamp"], utc=True)
        eq = eq.set_index("timestamp").sort_index()
        eq = eq["equity"].astype(float)
        # Monthly returns
        monthly = eq.resample("ME").last().pct_change(fill_method=None).dropna()
        if len(monthly) > 0:
            row["best_month_pct"] = round(float(monthly.max() * 100), 4)
            row["worst_month_pct"] = round(float(monthly.min() * 100), 4)
            row["pct_profitable_months"] = round(
                float((monthly > 0).sum() / len(monthly) * 100), 4
            )
            row["median_monthly_return"] = round(float(monthly.median() * 100), 4)
        # Drawdown duration approx
        rolling_max = eq.expanding().max()
        dd = eq - rolling_max
        in_dd = dd < 0
        grp = (~in_dd).cumsum()
        if in_dd.any():
            dur = in_dd.groupby(grp).sum()
            row["drawdown_duration_max_periods"] = int(dur.max())
        return row
    except Exception:
        return row


def _extend_metrics_from_equity_and_trades(run_dir: Path, freq: str, row: dict) -> dict:
    """Add pct_days_in_market, median_holding_period, avg_holding_period, win_streak_max, loss_streak_max, var_95, es_95, total_cost_pct, cost_per_turnover from equity + trades."""
    import pandas as pd

    for k in (
        "var_95",
        "es_95",
        "pct_days_in_market",
        "median_holding_period",
        "avg_holding_period",
        "win_streak_max",
        "loss_streak_max",
        "total_cost_pct",
        "cost_per_turnover",
    ):
        if k not in row:
            row[k] = None
    eq_path = run_dir / f"equity_curve_{freq}.csv"
    if not eq_path.exists():
        return row
    try:
        eq = pd.read_csv(eq_path)
        if "timestamp" not in eq.columns or "equity" not in eq.columns:
            return row
        eq["timestamp"] = pd.to_datetime(eq["timestamp"], utc=True)
        eq = eq.set_index("timestamp").sort_index()
        eq = eq["equity"].astype(float)
        ret = eq.pct_change(fill_method=None).dropna()
        if len(ret) < 2:
            return row
        total_days = len(ret)
        # VaR 95 and ES 95 (historical)
        var_95 = float(ret.quantile(0.05))
        below = ret[ret <= var_95]
        es_95 = float(below.mean()) if len(below) > 0 else var_95
        row["var_95"] = round(var_95, 6)
        row["es_95"] = round(es_95, 6)
        # Win/loss streaks
        pos = (ret > 0).astype(int)
        neg = (ret < 0).astype(int)

        def max_streak(series):
            grp = (series != series.shift()).cumsum()
            return int(series.groupby(grp).sum().max()) if series.any() else 0

        row["win_streak_max"] = max_streak(pos)
        row["loss_streak_max"] = max_streak(neg)
        # pct_days_in_market: from trades if available else approximate from non-zero return days
        trades_df = _load_trades_df(run_dir, freq)
        if (
            trades_df is not None
            and not trades_df.empty
            and "_date" in trades_df.columns
        ):
            active_days = trades_df["_date"].nunique()
            row["pct_days_in_market"] = (
                round(min(1.0, active_days / total_days) * 100, 4)
                if total_days
                else None
            )
            # Holding period proxy: per symbol, (max date - min date).days; then median/avg across symbols
            sym_dates = trades_df.groupby("symbol")["_date"].agg(["min", "max"])
            sym_dates["days"] = (sym_dates["max"] - sym_dates["min"]).dt.days
            row["median_holding_period"] = (
                round(float(sym_dates["days"].median()), 2) if len(sym_dates) else None
            )
            row["avg_holding_period"] = (
                round(float(sym_dates["days"].mean()), 2) if len(sym_dates) else None
            )
            # Cost diagnostics
            start_equity = 10000.0
            if "total_cost_cash" in trades_df.columns:
                total_cost = float(trades_df["total_cost_cash"].sum())
                row["total_cost_pct"] = (
                    round(total_cost / start_equity * 100, 4) if start_equity else None
                )
                to = row.get("turnover")
                row["cost_per_turnover"] = (
                    round(total_cost / to, 6) if (to and to != 0) else None
                )
            elif "commission_cash" in trades_df.columns:
                total_cost = float(trades_df["commission_cash"].sum())
                row["total_cost_pct"] = (
                    round(total_cost / start_equity * 100, 4) if start_equity else None
                )
                to = row.get("turnover")
                row["cost_per_turnover"] = (
                    round(total_cost / to, 6) if (to and to != 0) else None
                )
        else:
            row["pct_days_in_market"] = (
                round((ret != 0).sum() / total_days * 100, 4) if total_days else None
            )
            row["median_holding_period"] = None
            row["avg_holding_period"] = None
        return row
    except Exception:
        return row


def _load_variants(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_benchmark(
    output_root: Path,
    dataset_path: str | None,
    is_synthetic: bool,
    freq: str,
    variants: list[dict],
    baseline_id: str,
    ablation_ids: list[str],
    max_variants: int | None,
    quick: bool,
    no_strict_session_gate: bool = False,
) -> tuple[list[dict], str]:
    """Run each variant on 1y (and 3y/full if not quick). Return all scoreboard rows and dataset_label."""
    bench_root = output_root / "benchmark"
    bench_root.mkdir(parents=True, exist_ok=True)
    if not dataset_path or not Path(dataset_path).exists():
        synth = bench_root / "synthetic" / "eod.parquet"
        _generate_synthetic(synth, "2020-01-01", "2023-12-31")
        dataset_path = str(synth.resolve())
        is_synthetic = True
    dataset_label = "synthetic" if is_synthetic else Path(dataset_path).name
    horizons = _compute_horizons(dataset_path)
    if quick:
        horizons = [h for h in horizons if h[0] == "1y"]
        # Limit quick-mode window to 90 trading days so smoke tests stay fast
        if horizons:
            import pandas as _pd
            _h = horizons[0]
            _trading_dts = _pd.bdate_range(_h[1], _h[2])
            if len(_trading_dts) > 90:
                horizons = [(_h[0], _trading_dts[-90].strftime("%Y-%m-%d"), _h[2])]
    MA_SLOW_DEFAULT = 60
    MIN_BARS_WARMUP_BUFFER = 30
    to_run = variants[:max_variants] if max_variants else variants
    wiring_anomalies = []
    bar_anomalies = []
    for v in to_run:
        wiring_anomalies.extend(_validate_variant_wiring(v))
    skipped_variants = {a["variant_id"] for a in wiring_anomalies}
    all_rows = []
    for v in to_run:
        vid = v.get("id", "unknown")
        if vid in skipped_variants:
            for horizon_id, _, _ in horizons:
                all_rows.append(
                    {
                        "variant_id": vid,
                        "horizon": horizon_id,
                        "total_return": None,
                        "error": "variant_wiring_invalid",
                    }
                )
            continue
        strategy = v.get("strategy", "trend_baseline")
        params = v.get("params") or {}
        scale = params.get("commission_scale")
        commission_bps = None
        if scale is not None:
            commission_bps = DEFAULT_COMMISSION_BPS * scale
        bundle_path = params.get("bundle_path")
        for horizon_id, start_d, end_d in horizons:
            run_dir = bench_root / vid / horizon_id
            run_dir.mkdir(parents=True, exist_ok=True)
            slice_path = run_dir / "price_slice.parquet"
            try:
                prices = _load_and_slice(dataset_path, start_d, end_d)
                if prices.empty:
                    continue
                prices, ohlcv_info = _ensure_ohlcv(prices)
                prices.to_parquet(slice_path, index=False)
                run_inputs = {
                    "columns": list(prices.columns),
                    "columns_synthesized": ohlcv_info.get("columns_synthesized", []),
                    "end_date": end_d,
                    "slice_path": str(slice_path),
                    "source_dataset": dataset_path,
                    "start_date": start_d,
                    "synthetic_ohlcv": ohlcv_info.get("synthetic_ohlcv", False),
                    "synthetic_volume_value": ohlcv_info.get("synthetic_volume_value"),
                }
                with (run_dir / "run_inputs.json").open(
                    "w", encoding="utf-8", newline="\n"
                ) as f:
                    json.dump(run_inputs, f, indent=2, sort_keys=True)
                    f.write("\n")
                if quick and horizon_id == "1y" and "timestamp" in prices.columns:
                    ma_slow = int(params.get("ma_slow", MA_SLOW_DEFAULT))
                    min_bars = ma_slow + MIN_BARS_WARMUP_BUFFER
                    n_dates = int(prices["timestamp"].dt.date.nunique())
                    if n_dates < min_bars:
                        msg = f"quick 1y slice has {n_dates} trading days < {min_bars} (ma_slow={ma_slow}+buffer={MIN_BARS_WARMUP_BUFFER}), warmup may reduce valid signals"
                        bar_anomalies.append(
                            {
                                "variant_id": vid,
                                "horizon": horizon_id,
                                "type": "too_few_bars_for_warmup",
                                "message": msg,
                            }
                        )
                        print(f"[{vid}/{horizon_id}] WARNING: {msg}", file=sys.stderr)
            except Exception as e:
                print(f"[{vid}/{horizon_id}] Slice failed: {e}", file=sys.stderr)
                continue
            print(f"[{vid}/{horizon_id}] Running backtest...")
            code = _run_backtest(
                slice_path,
                run_dir,
                strategy,
                freq,
                commission_bps,
                bundle_path,
                params,
                no_strict_session_gate=no_strict_session_gate,
            )
            if code != 0:
                all_rows.append(
                    {
                        "variant_id": vid,
                        "horizon": horizon_id,
                        "total_return": None,
                        "error": "backtest_failed",
                    }
                )
                continue
            _run_analysis(run_dir, freq)
            metrics_path = run_dir / "metrics_summary.json"
            if not metrics_path.exists():
                all_rows.append(
                    {"variant_id": vid, "horizon": horizon_id, "total_return": None}
                )
                continue
            with metrics_path.open("r", encoding="utf-8") as f:
                m = json.load(f)
            row = {
                "variant_id": vid,
                "horizon": horizon_id,
                "total_return": m.get("total_return"),
                "cagr": m.get("cagr"),
                "volatility": m.get("volatility"),
                "sharpe_ratio": m.get("sharpe_ratio"),
                "sortino_ratio": m.get("sortino_ratio"),
                "calmar_ratio": None,
                "max_drawdown_pct": m.get("max_drawdown_pct"),
                "total_trades": m.get("total_trades"),
                "hit_rate": m.get("hit_rate"),
                "profit_factor": m.get("profit_factor"),
                "avg_win": m.get("avg_win"),
                "avg_loss": m.get("avg_loss"),
                "turnover": m.get("turnover"),
                "start_date": m.get("start_date"),
                "end_date": m.get("end_date"),
            }
            if (
                m.get("cagr")
                and m.get("max_drawdown_pct")
                and m.get("max_drawdown_pct") != 0
            ):
                row["calmar_ratio"] = (
                    m["cagr"] / abs(m["max_drawdown_pct"])
                    if m["max_drawdown_pct"]
                    else None
                )
            row = _enhance_metrics_from_equity(run_dir, freq, row)
            row = _extend_metrics_from_equity_and_trades(run_dir, freq, row)
            all_rows.append(row)
    return all_rows, dataset_label, wiring_anomalies, bar_anomalies


def write_scoreboard(bench_root: Path, all_rows: list[dict]) -> None:
    if not all_rows:
        return
    keys = sorted(set().union(*(set(r.keys()) for r in all_rows)))
    with (bench_root / "scoreboard.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as f:
        json.dump(all_rows, f, indent=2, sort_keys=True)
        f.write("\n")
    with (bench_root / "scoreboard.csv").open("w", encoding="utf-8", newline="\n") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)
    print(f"Wrote {bench_root / 'scoreboard.csv'} and .json")


def write_indicator_exposure_summary(
    bench_root: Path,
    all_rows: list[dict],
    variants: list[dict],
    baseline_id: str = "trend_baseline",
) -> None:
    """Write indicator_exposure_summary: pct_days_filtered_out (vs baseline), pct_entries_blocked, mean/median on entry if backtest exports them."""
    vid_to_components = {
        v.get("id"): v.get("enabled_components") or [] for v in variants
    }
    base_trades_by_horizon = {}
    base_pct_days_by_horizon = {}
    for r in all_rows:
        if r.get("variant_id") == baseline_id and r.get("horizon"):
            base_trades_by_horizon[r["horizon"]] = r.get("total_trades")
            pct = r.get("pct_days_in_market")
            if pct is not None:
                base_pct_days_by_horizon[r["horizon"]] = float(pct)
    out = []
    for r in all_rows:
        vid, hor = r.get("variant_id"), r.get("horizon")
        comps = vid_to_components.get(vid) or []
        indicator_type = "baseline"
        if "regime_gate" in comps:
            indicator_type = "regime_gate"
        elif "realized_vol_filter" in comps:
            indicator_type = "realized_vol_filter"
        elif "liquidity_filter" in comps:
            indicator_type = "liquidity_filter"
        elif "rsi_filter" in comps and "vol_filter" in comps:
            indicator_type = "rsi_vol_combo"
        elif "rsi_filter" in comps:
            indicator_type = "rsi_filter"
        elif "vol_filter" in comps:
            indicator_type = "vol_filter"
        base_trades = base_trades_by_horizon.get(hor) if hor else None
        var_trades = r.get("total_trades")
        if (
            base_trades is not None
            and var_trades is not None
            and base_trades > 0
            and indicator_type != "baseline"
        ):
            pct_blocked = round((1.0 - float(var_trades) / float(base_trades)) * 100, 2)
        else:
            pct_blocked = None
        base_pct_days = base_pct_days_by_horizon.get(hor) if hor else None
        var_pct_days = r.get("pct_days_in_market")
        if (
            base_pct_days is not None
            and base_pct_days > 0
            and var_pct_days is not None
            and indicator_type != "baseline"
        ):
            pct_days_filtered_out = round(
                100.0 - (float(var_pct_days) / base_pct_days * 100), 2
            )
        else:
            pct_days_filtered_out = None
        run_dir = bench_root / vid / hor if vid and hor else None
        mean_on_entry = median_on_entry = None
        if run_dir and (run_dir / "metrics_summary.json").exists():
            try:
                m = json.loads(
                    (run_dir / "metrics_summary.json").read_text(encoding="utf-8")
                )
                mean_on_entry = (
                    m.get("mean_indicator_on_entry")
                    or m.get("mean_rsi_on_entry")
                    or m.get("mean_atr_pct_on_entry")
                )
                median_on_entry = (
                    m.get("median_indicator_on_entry")
                    or m.get("median_rsi_on_entry")
                    or m.get("median_atr_pct_on_entry")
                )
                if mean_on_entry is not None:
                    mean_on_entry = round(float(mean_on_entry), 4)
                if median_on_entry is not None:
                    median_on_entry = round(float(median_on_entry), 4)
            except Exception:
                pass
        out.append(
            {
                "variant_id": vid,
                "horizon": hor,
                "indicator_type": indicator_type,
                "total_trades": var_trades,
                "turnover": r.get("turnover"),
                "pct_days_in_market": r.get("pct_days_in_market"),
                "pct_days_filtered_out": pct_days_filtered_out,
                "mean_indicator_on_entry": mean_on_entry,
                "median_indicator_on_entry": median_on_entry,
                "pct_entries_blocked": pct_blocked,
            }
        )
    if not out:
        return
    keys = sorted(out[0].keys())
    with (bench_root / "indicator_exposure_summary.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as f:
        json.dump(out, f, indent=2, sort_keys=True)
        f.write("\n")
    with (bench_root / "indicator_exposure_summary.csv").open(
        "w", encoding="utf-8", newline="\n"
    ) as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(out)
    print(f"Wrote {bench_root / 'indicator_exposure_summary.csv'} and .json")


def write_metrics_extended(bench_root: Path, all_rows: list[dict]) -> None:
    """Write optional metrics_extended.csv/json with extended columns only."""
    ext_keys = [
        "variant_id",
        "horizon",
        "total_return",
        "sharpe_ratio",
        "max_drawdown_pct",
        "turnover",
        "pct_days_in_market",
        "median_holding_period",
        "avg_holding_period",
        "win_streak_max",
        "loss_streak_max",
        "var_95",
        "es_95",
        "total_cost_pct",
        "cost_per_turnover",
        "cost_share_of_return",
        "stability_score",
        "stability_score_v2",
        "robustness_score",
        "return_per_turnover",
        "pct_profitable_months",
        "median_monthly_return",
        "worst_month_pct",
    ]
    rows = [{k: r.get(k) for k in ext_keys if k in r} for r in all_rows]
    if not rows:
        return
    keys_used = sorted(set().union(*(set(row.keys()) for row in rows)))
    with (bench_root / "metrics_extended.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as f:
        json.dump(rows, f, indent=2, sort_keys=True)
        f.write("\n")
    with (bench_root / "metrics_extended.csv").open(
        "w", encoding="utf-8", newline="\n"
    ) as f:
        w = csv.DictWriter(f, fieldnames=keys_used, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {bench_root / 'metrics_extended.csv'} and .json")


def write_ablation(
    bench_root: Path, all_rows: list[dict], baseline_id: str, ablation_ids: list[str]
) -> None:
    """Compute deltas vs baseline per horizon; write ablation_summary.csv."""
    rows_by_key = {}
    for r in all_rows:
        key = (r.get("variant_id"), r.get("horizon"))
        rows_by_key[key] = r
    out = []
    for horizon in ["1y", "3y", "full"]:
        base = rows_by_key.get((baseline_id, horizon))
        if not base:
            continue
        for aid in ablation_ids:
            alt = rows_by_key.get((aid, horizon))
            if not alt:
                continue

            def f(k):
                b, a = base.get(k), alt.get(k)
                if b is None or a is None:
                    return None
                try:
                    return round(float(a) - float(b), 6)
                except (TypeError, ValueError):
                    return None

            out.append(
                {
                    "horizon": horizon,
                    "baseline": baseline_id,
                    "variant": aid,
                    "delta_return": f("total_return"),
                    "delta_sharpe": f("sharpe_ratio"),
                    "delta_max_dd_pct": f("max_drawdown_pct"),
                    "delta_turnover": f("turnover"),
                    "delta_trades": (
                        None
                        if base.get("total_trades") is None
                        or alt.get("total_trades") is None
                        else (
                            int(alt.get("total_trades") or 0)
                            - int(base.get("total_trades") or 0)
                        )
                    ),
                }
            )
    if not out:
        return
    with (bench_root / "ablation_summary.csv").open(
        "w", encoding="utf-8", newline="\n"
    ) as f:
        w = csv.DictWriter(f, fieldnames=sorted(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(f"Wrote {bench_root / 'ablation_summary.csv'}")


def _load_equity_series(run_dir: Path, freq: str):
    """Return (eq series, returns series) or (None, None)."""
    import pandas as pd

    eq_path = run_dir / f"equity_curve_{freq}.csv"
    if not eq_path.exists():
        return None, None
    try:
        eq = pd.read_csv(eq_path)
        if "timestamp" not in eq.columns or "equity" not in eq.columns:
            return None, None
        eq["timestamp"] = pd.to_datetime(eq["timestamp"], utc=True)
        eq = eq.set_index("timestamp").sort_index()
        eq = eq["equity"].astype(float)
        ret = eq.pct_change(fill_method=None).dropna()
        return eq, ret
    except Exception:
        return None, None


def _load_trades_df(run_dir: Path, freq: str):
    """Load trades CSV if present; return DataFrame with timestamp, symbol or None."""
    import pandas as pd

    p = run_dir / f"trades_{freq}.csv"
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
        if "timestamp" not in df.columns or "symbol" not in df.columns:
            return None
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["_date"] = df["timestamp"].dt.date
        return df
    except Exception:
        return None


def write_regime_metrics(
    bench_root: Path,
    variant_horizon_dirs: list[tuple[str, str, Path]],
    freq: str,
    roll_win: int = 20,
) -> None:
    """Compute metrics per regime (volatility/trend/drawdown) from equity; write regime_metrics.json/csv."""
    import pandas as pd

    out_rows = []
    for vid, horizon_id, run_dir in variant_horizon_dirs:
        eq, ret = _load_equity_series(run_dir, freq)
        if eq is None or ret is None or len(ret) < roll_win:
            continue
        # Volatility regime: rolling std of returns -> tertiles
        vol = ret.rolling(roll_win, min_periods=roll_win).std()
        vol_qt = vol.quantile([1 / 3, 2 / 3])
        q1, q2 = vol_qt.iloc[0], vol_qt.iloc[1]
        vol_regime = pd.Series("med", index=ret.index)
        vol_regime[vol <= q1] = "low"
        vol_regime[vol > q2] = "high"
        # Trend regime: rolling slope of equity (normalized) over roll_win
        eq_norm = eq / eq.expanding().min()
        slope = eq_norm.rolling(roll_win, min_periods=roll_win).apply(
            lambda x: (
                float((x.iloc[-1] - x.iloc[0]) / roll_win)
                if len(x) == roll_win
                else None
            ),
            raw=False,
        )
        sl_qt = slope.quantile([1 / 3, 2 / 3]).dropna()
        if len(sl_qt) >= 2:
            sl1, sl2 = sl_qt.iloc[0], sl_qt.iloc[1]
            trend_regime = pd.Series("neutral", index=ret.index)
            trend_regime[slope <= sl1] = "down"
            trend_regime[slope > sl2] = "up"
        else:
            trend_regime = pd.Series("neutral", index=ret.index)
        # Drawdown regime: equity below prior peak (align to ret.index so all regimes share same index)
        peak = eq.expanding().max()
        dd_regime = (
            (eq < peak)
            .astype(str)
            .replace({"True": "in_drawdown", "False": "recovery"})
            .reindex(ret.index)
            .ffill()
        )
        eq_on_ret = eq.reindex(ret.index).ffill()
        for reg_name, reg_series in [
            ("volatility", vol_regime),
            ("trend", trend_regime),
            ("drawdown", dd_regime),
        ]:
            for label in reg_series.dropna().unique():
                mask = reg_series == label
                if mask.sum() < 2:
                    continue
                r = ret[mask]
                e = eq_on_ret[mask]
                tr = (e.iloc[-1] / e.iloc[0] - 1.0) if e.iloc[0] != 0 else None
                sh = (r.mean() / r.std() * (252**0.5)) if r.std() > 0 else None
                peak_r = e.expanding().max()
                dd_r = (e - peak_r).min()
                max_dd_pct = (dd_r / peak_r.max() * 100) if peak_r.max() != 0 else None
                out_rows.append(
                    {
                        "variant_id": vid,
                        "horizon": horizon_id,
                        "regime_type": reg_name,
                        "regime_label": str(label),
                        "total_return": round(float(tr), 6) if tr is not None else None,
                        "sharpe_ratio": round(float(sh), 6) if sh is not None else None,
                        "max_drawdown_pct": (
                            round(float(max_dd_pct), 4)
                            if max_dd_pct is not None
                            else None
                        ),
                        "bars_count": int(mask.sum()),
                    }
                )
    fieldnames = [
        "variant_id",
        "horizon",
        "regime_type",
        "regime_label",
        "total_return",
        "sharpe_ratio",
        "max_drawdown_pct",
        "bars_count",
    ]
    with (bench_root / "regime_metrics.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as f:
        json.dump(out_rows, f, indent=2, sort_keys=True)
    with (bench_root / "regime_metrics.csv").open(
        "w", encoding="utf-8", newline="\n"
    ) as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(out_rows)
    print(f"Wrote {bench_root / 'regime_metrics.csv'} and .json")


def write_attribution_summary(
    bench_root: Path,
    baseline_id: str,
    variant_ids: list[str],
    horizons: list[str],
    freq: str,
) -> None:
    """Trade overlap vs baseline + incremental return/turnover; write attribution_summary.csv/json."""
    out_rows = []
    for horizon_id in horizons:
        base_dir = bench_root / baseline_id / horizon_id
        base_trades = _load_trades_df(base_dir, freq)
        if base_trades is None or base_trades.empty:
            base_set = set()
        else:
            base_set = set(zip(base_trades["symbol"].astype(str), base_trades["_date"]))
        for vid in variant_ids:
            if vid == baseline_id:
                continue
            var_dir = bench_root / vid / horizon_id
            var_trades = _load_trades_df(var_dir, freq)
            if var_trades is None or var_trades.empty:
                out_rows.append(
                    {
                        "horizon": horizon_id,
                        "baseline": baseline_id,
                        "variant": vid,
                        "overlap_pct": None,
                        "incremental_trades": None,
                        "incremental_return": None,
                        "incremental_turnover": None,
                        "incremental_return_per_turnover": None,
                    }
                )
                continue
            n_var = len(var_trades)
            from datetime import timedelta

            overlap = 0
            for _, r in var_trades.iterrows():
                sym, d = str(r["symbol"]), r["_date"]
                matched = (sym, d) in base_set
                if not matched:
                    for delta in [-1, 1]:
                        try:
                            d2 = d + timedelta(days=delta)
                            if (sym, d2) in base_set:
                                matched = True
                                break
                        except Exception:
                            pass
                if matched:
                    overlap += 1
            overlap_pct = round(overlap / n_var * 100, 2) if n_var else None
            inc_trades = n_var - overlap
            # Incremental return/turnover: use metrics_summary deltas as proxy
            m_base = bench_root / baseline_id / horizon_id / "metrics_summary.json"
            m_var = bench_root / vid / horizon_id / "metrics_summary.json"
            inc_ret = None
            inc_turn = None
            if m_base.exists() and m_var.exists():
                with m_base.open("r", encoding="utf-8") as f:
                    b = json.load(f)
                with m_var.open("r", encoding="utf-8") as f:
                    v = json.load(f)
                tr_b, tr_v = b.get("total_return"), v.get("total_return")
                to_b, to_v = b.get("turnover"), v.get("turnover")
                if tr_b is not None and tr_v is not None:
                    inc_ret = round(float(tr_v) - float(tr_b), 6)
                if to_b is not None and to_v is not None:
                    inc_turn = round(float(to_v) - float(to_b), 4)
            inc_ret_per_turn = (
                (inc_ret / inc_turn) if inc_turn and inc_turn != 0 else None
            )
            out_rows.append(
                {
                    "horizon": horizon_id,
                    "baseline": baseline_id,
                    "variant": vid,
                    "overlap_pct": overlap_pct,
                    "incremental_trades": inc_trades,
                    "incremental_return": inc_ret,
                    "incremental_turnover": inc_turn,
                    "incremental_return_per_turnover": (
                        round(inc_ret_per_turn, 6)
                        if inc_ret_per_turn is not None
                        else None
                    ),
                }
            )
    attr_fieldnames = [
        "horizon",
        "baseline",
        "variant",
        "overlap_pct",
        "incremental_trades",
        "incremental_return",
        "incremental_turnover",
        "incremental_return_per_turnover",
    ]
    with (bench_root / "attribution_summary.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as f:
        json.dump(out_rows, f, indent=2, sort_keys=True)
    with (bench_root / "attribution_summary.csv").open(
        "w", encoding="utf-8", newline="\n"
    ) as f:
        w = csv.DictWriter(f, fieldnames=attr_fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(out_rows)
    print(f"Wrote {bench_root / 'attribution_summary.csv'} and .json")


def _robustness_score_formula() -> str:
    return (
        "robustness_score = 0.20*norm(total_return) + 0.20*norm(sharpe) + 0.15*(1 - norm(|max_dd|)) "
        "+ 0.10*(1 - norm(turnover)) + 0.10*norm(profit_factor) + 0.10*norm(hit_rate) "
        "+ 0.05*pct_profitable_months/100 + 0.05*norm(median_monthly_return) + 0.05*(1 - norm(|worst_month|)) "
        "- cost_sensitivity_penalty (0 if no cost variants, else penalty when return drops a lot at 2x cost)"
    )


def add_robustness_columns(
    all_rows: list[dict],
    variants: list[dict],
    baseline_id: str,
) -> list[dict]:
    """Add robustness_score, return_per_turnover; ensure pct_profitable_months, median_monthly_return, worst_month_pct; cost_sensitivity_penalty for 2x cost."""
    rows_by_key = {(r.get("variant_id"), r.get("horizon")): r for r in all_rows}
    base_1y = rows_by_key.get((baseline_id, "1y")) or rows_by_key.get(
        (baseline_id, "full")
    )
    base_return_1y = float(base_1y.get("total_return") or 0) if base_1y else 0

    def _norm(x, low=0, high=1):
        if x is None:
            return 0.0
        try:
            v = float(x)
            if high == low:
                return 0.5
            return max(0, min(1, (v - low) / (high - low)))
        except (TypeError, ValueError):
            return 0.0

    valid = [r for r in all_rows if r.get("total_return") is not None]
    if not valid:
        return all_rows
    tr_vals = [r.get("total_return") for r in valid]
    sh_vals = [
        r.get("sharpe_ratio") for r in valid if r.get("sharpe_ratio") is not None
    ]
    dd_vals = [
        r.get("max_drawdown_pct")
        for r in valid
        if r.get("max_drawdown_pct") is not None
    ]
    to_vals = [r.get("turnover") for r in valid if r.get("turnover") is not None]
    pf_vals = [
        r.get("profit_factor") for r in valid if r.get("profit_factor") is not None
    ]
    hr_vals = [r.get("hit_rate") for r in valid if r.get("hit_rate") is not None]
    med_vals = [
        r.get("median_monthly_return")
        for r in all_rows
        if r.get("median_monthly_return") is not None
    ]
    tr_min, tr_max = min(tr_vals), max(tr_vals)
    sh_min, sh_max = (min(sh_vals), max(sh_vals)) if sh_vals else (0, 1)
    dd_abs = [abs(x) for x in dd_vals if x is not None]
    to_min, to_max = (min(to_vals), max(to_vals)) if to_vals else (0, 1)
    pf_min, pf_max = (min(pf_vals), max(pf_vals)) if pf_vals else (0, 1)
    hr_min, hr_max = (min(hr_vals), max(hr_vals)) if hr_vals else (0, 1)
    med_min = min(med_vals) if med_vals else 0
    med_max = max(med_vals) if med_vals else 100

    def _rank_pct(vals: list, key_fn, higher_better: bool = True) -> dict:
        """Return dict mapping index -> percentile rank (0-1). None -> 0.5."""
        non_none = [
            (i, key_fn(vals[i]))
            for i in range(len(vals))
            if key_fn(vals[i]) is not None
        ]
        non_none.sort(key=lambda x: x[1] if higher_better else -x[1])
        out_map = {i: 0.5 for i in range(len(vals))}
        for rank, (idx, _) in enumerate(non_none):
            out_map[idx] = (rank + 1) / (len(non_none) + 1) if non_none else 0.5
        return out_map

    tr_rank = _rank_pct(all_rows, lambda r: r.get("total_return"), higher_better=True)
    sh_rank = _rank_pct(all_rows, lambda r: r.get("sharpe_ratio"), higher_better=True)
    dd_rank = _rank_pct(
        all_rows, lambda r: abs(r.get("max_drawdown_pct") or 0), higher_better=False
    )
    ext_defaults = (
        "var_95",
        "es_95",
        "pct_days_in_market",
        "median_holding_period",
        "avg_holding_period",
        "win_streak_max",
        "loss_streak_max",
        "total_cost_pct",
        "cost_per_turnover",
    )
    out = []
    for i, r in enumerate(all_rows):
        r = dict(r)
        for k in ext_defaults:
            if k not in r:
                r[k] = None
        tr = r.get("total_return")
        sh = r.get("sharpe_ratio")
        dd = r.get("max_drawdown_pct")
        to = r.get("turnover")
        pf = r.get("profit_factor")
        hr = r.get("hit_rate")
        r["return_per_turnover"] = (
            round(tr / to, 6)
            if (tr is not None and to is not None and to != 0)
            else None
        )
        tc_pct = r.get("total_cost_pct")
        if tr is not None and tc_pct is not None and abs(tr) > 1e-9:
            r["cost_share_of_return"] = round(float(tc_pct) / (abs(float(tr)) * 100), 6)
        else:
            r["cost_share_of_return"] = None
        r["net_total_return"] = tr
        if tr is not None and tc_pct is not None:
            r["gross_total_return_est"] = round(float(tr) + float(tc_pct) / 100.0, 6)
        else:
            r["gross_total_return_est"] = None
        # Units: total_return = decimal (0.12 = 12%), total_cost_pct = percent (1.5 = 1.5%). gross_est = total_return + total_cost_pct/100.
        penalty = 0
        if (
            "cost_2x" in str(r.get("variant_id", ""))
            and base_1y
            and r.get("horizon") == "1y"
        ):
            if tr is not None and base_return_1y is not None:
                drop = base_return_1y - float(tr)
                penalty = max(0, min(1, drop))
        score = (
            0.20 * _norm(tr, tr_min, tr_max)
            + 0.20 * _norm(sh, sh_min, sh_max)
            + 0.15 * (1 - _norm(abs(dd) if dd else 0, 0, max(dd_abs) if dd_abs else 1))
            + 0.10 * (1 - _norm(to, to_min, to_max))
            + 0.10 * _norm(pf, pf_min, pf_max)
            + 0.10 * _norm(hr, hr_min, hr_max)
            + 0.05 * (r.get("pct_profitable_months") or 0) / 100.0
            + 0.05 * _norm(r.get("median_monthly_return"), med_min, med_max)
            + 0.05 * (1 - _norm(abs(r.get("worst_month_pct") or 0), 0, 100))
            - penalty
        )
        r["robustness_score"] = round(max(0, min(1, score)), 6)
        r["stability_score"] = round(
            (tr_rank.get(i, 0.5) + sh_rank.get(i, 0.5) + dd_rank.get(i, 0.5)) / 3.0, 6
        )
        out.append(r)
    rows_by_variant = {}
    for r in out:
        vid = r.get("variant_id")
        if vid not in rows_by_variant:
            rows_by_variant[vid] = []
        rows_by_variant[vid].append(r)

    def _stability_v2(r: dict) -> float:
        vid = r.get("variant_id")
        same = rows_by_variant.get(vid, [r])
        returns = [
            x.get("total_return") for x in same if x.get("total_return") is not None
        ]
        if len(returns) < 2:
            return r.get("stability_score") or 0.5
        import statistics

        try:
            std = statistics.stdev(returns)
            mean_ret = statistics.mean(returns)
            cv = std / abs(mean_ret) if mean_ret != 0 else 1.0
            return max(0, min(1, 1.0 - min(cv, 2.0) / 2.0))
        except Exception:
            return r.get("stability_score") or 0.5

    for r in out:
        r["stability_score_v2"] = round(_stability_v2(r), 6)
    return out


def write_anomalies(
    bench_root: Path,
    all_rows: list[dict],
    variant_horizon_dirs: list[tuple[str, str, Path]],
    freq: str,
    wiring_anomalies: list[dict] | None = None,
) -> None:
    """Detect zero trades, constant equity, NaNs, extreme turnover, OOS collapse; write anomalies.json."""
    issues = list(wiring_anomalies) if wiring_anomalies else []
    for r in all_rows:
        vid, hor = r.get("variant_id"), r.get("horizon")
        if r.get("total_trades") == 0:
            issues.append(
                {
                    "variant_id": vid,
                    "horizon": hor,
                    "type": "zero_trades",
                    "message": "No trades executed",
                }
            )
        if (
            r.get("total_return") is not None
            and r.get("total_return") == 0
            and (r.get("total_trades") or 0) == 0
        ):
            issues.append(
                {
                    "variant_id": vid,
                    "horizon": hor,
                    "type": "constant_equity",
                    "message": "Zero return and zero trades",
                }
            )
        tr, to = r.get("total_return"), r.get("turnover")
        if tr is not None and to is not None and to > 0:
            if abs(tr) < 0.001 and to > 100:
                issues.append(
                    {
                        "variant_id": vid,
                        "horizon": hor,
                        "type": "turnover_extreme_vs_return",
                        "message": f"Turnover {to} very high vs return {tr}",
                    }
                )
        for k in ["total_return", "sharpe_ratio", "max_drawdown_pct"]:
            v = r.get(k)
            if v is not None and isinstance(v, float) and (v != v):
                issues.append(
                    {
                        "variant_id": vid,
                        "horizon": hor,
                        "type": "nan_metric",
                        "message": f"NaN in {k}",
                    }
                )
        dd_dur = r.get("drawdown_duration_max_periods")
        if dd_dur is not None and int(dd_dur) > 200:
            issues.append(
                {
                    "variant_id": vid,
                    "horizon": hor,
                    "type": "extreme_drawdown_duration",
                    "message": f"Drawdown duration {dd_dur} periods very high",
                }
            )
    for vid, hor, run_dir in variant_horizon_dirs:
        eq_path = run_dir / f"equity_curve_{freq}.csv"
        if eq_path.exists():
            try:
                import pandas as pd

                df = pd.read_csv(eq_path)
                if "equity" in df.columns and df["equity"].nunique() == 1:
                    issues.append(
                        {
                            "variant_id": vid,
                            "horizon": hor,
                            "type": "constant_equity",
                            "message": "Equity curve is constant",
                        }
                    )
                if df["equity"].isna().any():
                    issues.append(
                        {
                            "variant_id": vid,
                            "horizon": hor,
                            "type": "nan_equity",
                            "message": "NaN in equity curve",
                        }
                    )
                ts = pd.to_datetime(df["timestamp"], utc=True)
                if not ts.is_monotonic_increasing and len(ts) > 1:
                    issues.append(
                        {
                            "variant_id": vid,
                            "horizon": hor,
                            "type": "inconsistent_timestamps",
                            "message": "Equity timestamps not monotonic",
                        }
                    )
            except Exception:
                pass
        req = [run_dir / f"equity_curve_{freq}.csv", run_dir / "metrics_summary.json"]
        for p in req:
            if not p.exists():
                issues.append(
                    {
                        "variant_id": vid,
                        "horizon": hor,
                        "type": "missing_required_file",
                        "message": f"Missing {p.name}",
                    }
                )
        qc_path = run_dir / "qc_report.json"
        if qc_path.exists():
            try:
                qc = json.loads(qc_path.read_text(encoding="utf-8"))
                if (
                    qc.get("ok") is False
                    and qc.get("summary", {}).get("fail_count", 0) > 0
                ):
                    fail_count = qc["summary"]["fail_count"]
                    by_check = qc.get("summary", {}).get("issues_by_check", {})
                    msg = f"DATA_QC_FAIL: {fail_count} fail(s) - {by_check} (trading blocked)"
                    issues.append(
                        {
                            "variant_id": vid,
                            "horizon": hor,
                            "type": "data_qc_fail",
                            "message": msg,
                        }
                    )
            except Exception:
                pass
        run_inputs_path = run_dir / "run_inputs.json"
        if run_inputs_path.exists():
            try:
                ri = json.loads(run_inputs_path.read_text(encoding="utf-8"))
                if ri.get("synthetic_ohlcv") is True:
                    cols = ri.get("columns_synthesized") or []
                    msg = f"OHLCV columns synthesized for QC/execution: {cols}"
                    issues.append(
                        {
                            "variant_id": vid,
                            "horizon": hor,
                            "type": "synthetic_ohlcv",
                            "message": msg,
                        }
                    )
            except Exception:
                pass
    oos_path = bench_root / "oos_report.md"
    if oos_path.exists():
        try:
            text = oos_path.read_text(encoding="utf-8")
            if "large drop = likely overfit" in text:
                issues.append(
                    {
                        "variant_id": "OOS",
                        "horizon": "test",
                        "type": "oos_collapse",
                        "message": "OOS report flags large train-to-test drop (possible overfit)",
                    }
                )
        except Exception:
            pass
    with (bench_root / "anomalies.json").open("w", encoding="utf-8", newline="\n") as f:
        json.dump(issues, f, indent=2, sort_keys=True)
    print(f"Wrote {bench_root / 'anomalies.json'}")


def write_data_quality_summary(
    bench_root: Path,
    all_rows: list[dict],
    variant_horizon_dirs: list[tuple[str, str, Path]],
    freq: str,
) -> None:
    """Write data_quality_summary.json: missing values, duplicate dates, non-monotonic, symbol coverage, min/max dates. Deterministic, ASCII-only."""
    import pandas as pd

    summary = {
        "runs_checked": 0,
        "missing_values": {},
        "duplicate_dates": 0,
        "non_monotonic_count": 0,
        "symbol_coverage": [],
        "date_min": None,
        "date_max": None,
    }
    for vid, hor, run_dir in variant_horizon_dirs:
        eq_path = run_dir / f"equity_curve_{freq}.csv"
        if not eq_path.exists():
            continue
        try:
            df = pd.read_csv(eq_path)
            if "timestamp" not in df.columns or "equity" not in df.columns:
                continue
            summary["runs_checked"] += 1
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            if df["equity"].isna().any():
                summary["missing_values"][f"{vid}/{hor}"] = int(
                    df["equity"].isna().sum()
                )
            dup = df["timestamp"].duplicated().sum()
            if dup > 0:
                summary["duplicate_dates"] += int(dup)
            if not df["timestamp"].is_monotonic_increasing and len(df) > 1:
                summary["non_monotonic_count"] += 1
            if summary["date_min"] is None or df["timestamp"].min() < pd.Timestamp(
                summary["date_min"], tz="UTC"
            ):
                summary["date_min"] = df["timestamp"].min().strftime("%Y-%m-%d")
            if summary["date_max"] is None or df["timestamp"].max() > pd.Timestamp(
                summary["date_max"], tz="UTC"
            ):
                summary["date_max"] = df["timestamp"].max().strftime("%Y-%m-%d")
            trades_df = _load_trades_df(run_dir, freq)
            if trades_df is not None and "symbol" in trades_df.columns:
                n_sym = trades_df["symbol"].nunique()
                summary["symbol_coverage"].append(
                    {"variant_id": vid, "horizon": hor, "symbol_count": int(n_sym)}
                )
        except Exception:
            pass
    summary["symbol_coverage"] = summary["symbol_coverage"][:50]
    with (bench_root / "data_quality_summary.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"Wrote {bench_root / 'data_quality_summary.json'}")


def run_oos(
    output_root: Path,
    dataset_path: str,
    freq: str,
    variants: list[dict],
    max_sweep: int,
    no_strict_session_gate: bool = False,
) -> None:
    """Train 70% / Test 30%; run limited variants on train; evaluate best on test; write oos_report.md."""
    import pandas as pd

    bench_root = output_root / "benchmark"
    df = pd.read_parquet(dataset_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    dates = sorted(df["timestamp"].dt.date.unique())
    n = len(dates)
    if n < 100:
        lines = ["# OOS Report", "", "Insufficient data for train/test split.", ""]
        (bench_root / "oos_report.md").write_text("\n".join(lines), encoding="utf-8")
        return
    split_idx = int(n * 0.7)
    train_end = dates[split_idx - 1].isoformat()
    test_start = dates[split_idx].isoformat()
    test_end = dates[-1].isoformat()
    # Run variants on train only (reuse run_benchmark with date filter)
    train_path = bench_root / "oos_train.parquet"
    test_path = bench_root / "oos_test.parquet"
    train_df = df[(df["timestamp"].dt.date <= dates[split_idx - 1])]
    test_df = df[(df["timestamp"].dt.date >= dates[split_idx])]
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)
    to_run = variants[:max_sweep]
    train_results = []
    for v in to_run:
        vid = v.get("id", "unknown")
        strategy = v.get("strategy", "trend_baseline")
        bundle_path = (v.get("params") or {}).get("bundle_path")
        run_dir = bench_root / "oos_train" / vid
        run_dir.mkdir(parents=True, exist_ok=True)
        code = _run_backtest(
            train_path,
            run_dir,
            strategy,
            freq,
            None,
            bundle_path,
            v.get("params"),
            no_strict_session_gate=no_strict_session_gate,
        )
        if code != 0:
            continue
        _run_analysis(run_dir, freq)
        mp = run_dir / "metrics_summary.json"
        if mp.exists():
            with mp.open("r", encoding="utf-8") as f:
                m = json.load(f)
            train_results.append(
                {
                    "variant_id": vid,
                    "total_return": m.get("total_return"),
                    "sharpe_ratio": m.get("sharpe_ratio"),
                }
            )
    # Evaluate top K on test
    top = sorted(
        [r for r in train_results if r.get("total_return") is not None],
        key=lambda x: (x.get("total_return") or 0),
        reverse=True,
    )[:5]
    test_results = []
    for r in top:
        vid = r["variant_id"]
        v = next((x for x in variants if x.get("id") == vid), None)
        if not v:
            continue
        run_dir = bench_root / "oos_test" / vid
        run_dir.mkdir(parents=True, exist_ok=True)
        code = _run_backtest(
            test_path,
            run_dir,
            v.get("strategy", "trend_baseline"),
            freq,
            None,
            (v.get("params") or {}).get("bundle_path"),
            v.get("params"),
            no_strict_session_gate=no_strict_session_gate,
        )
        if code != 0:
            test_results.append(
                {
                    "variant_id": vid,
                    "train_return": r.get("total_return"),
                    "test_return": None,
                    "note": "test_run_failed",
                }
            )
            continue
        _run_analysis(run_dir, freq)
        mp = run_dir / "metrics_summary.json"
        if mp.exists():
            with mp.open("r", encoding="utf-8") as f:
                m = json.load(f)
            test_results.append(
                {
                    "variant_id": vid,
                    "train_return": r.get("total_return"),
                    "test_return": m.get("total_return"),
                    "test_sharpe": m.get("sharpe_ratio"),
                }
            )
    lines = [
        "# Out-of-Sample Report",
        "",
        f"Train: up to {train_end} (70%). Test: {test_start} to {test_end} (30%).",
        "",
        "## Top configs on Train",
        "",
    ]
    for r in top:
        lines.append(
            f"- {r.get('variant_id')}: return={r.get('total_return')} sharpe={r.get('sharpe_ratio')}"
        )
    lines.append("")
    lines.append("## Performance on Test")
    lines.append("")
    for r in test_results:
        tr = r.get("test_return")
        train_ret = r.get("train_return")
        drop = (
            (float(train_ret) - float(tr))
            if (train_ret is not None and tr is not None)
            else None
        )
        stability = (
            " (large drop = likely overfit)" if drop is not None and drop > 0.1 else ""
        )
        lines.append(
            f"- {r.get('variant_id')}: test_return={tr} test_sharpe={r.get('test_sharpe')}{stability}"
        )
    lines.append("")
    (bench_root / "oos_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {bench_root / 'oos_report.md'}")


def write_benchmark_report(
    bench_root: Path,
    dataset_label: str,
    is_synthetic: bool,
    all_rows: list[dict],
    baseline_id: str,
    ablation_ids: list[str],
) -> None:
    synth_count = 0
    run_inputs_count = 0
    for r in all_rows:
        vid, hor = r.get("variant_id"), r.get("horizon")
        if not vid or not hor:
            continue
        ri_path = bench_root / vid / hor / "run_inputs.json"
        if ri_path.exists():
            run_inputs_count += 1
            try:
                ri = json.loads(ri_path.read_text(encoding="utf-8"))
                if ri.get("synthetic_ohlcv") is True:
                    synth_count += 1
            except Exception:
                pass
    lines = [
        "# Benchmark Report",
        "",
        "Generated by scripts/dev/run_strategy_benchmark.py",
        "",
    ]
    if run_inputs_count > 0:
        vol_note = " (volume set to 1e6 when missing)" if synth_count else ""
        lines.append(
            f"Data note: OHLCV synthesized for {synth_count}/{run_inputs_count} runs{vol_note}."
        )
        lines.append("")
    lines.extend(
        [
            "## Dataset",
            "",
            f"Used: {'synthetic' if is_synthetic else 'real'} - {dataset_label}",
            "",
            "## Indicator inventory summary",
            "",
            "See docs/INDICATOR_INVENTORY_AUTOGENERATED.md. Wired: trend (EMA), event (insider/shipping), multifactor (bundle).",
            "",
            "## Baseline results",
            "",
        ]
    )
    for r in all_rows:
        if r.get("variant_id") == baseline_id:
            lines.append(
                f"- {r.get('horizon')}: return={r.get('total_return')} sharpe={r.get('sharpe_ratio')} max_dd={r.get('max_drawdown_pct')} trades={r.get('total_trades')}"
            )
    lines.append("")
    lines.append("## Scoreboard top 10 (by total_return)")
    lines.append("")
    valid = [r for r in all_rows if r.get("total_return") is not None]
    for r in sorted(valid, key=lambda x: (x.get("total_return") or 0), reverse=True)[
        :10
    ]:
        lines.append(
            f"- {r.get('variant_id')} {r.get('horizon')}: return={r.get('total_return')} sharpe={r.get('sharpe_ratio')} max_dd={r.get('max_drawdown_pct')}"
        )
    lines.append("")
    lines.append("## Ablation highlights")
    lines.append("")
    ab_path = bench_root / "ablation_summary.csv"
    if ab_path.exists():
        with ab_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                lines.append(
                    f"- {row.get('variant')} vs {row.get('baseline')} ({row.get('horizon')}): delta_return={row.get('delta_return')} delta_sharpe={row.get('delta_sharpe')}"
                )
    lines.append("")
    lines.append("## OOS robustness")
    lines.append("")
    if (bench_root / "oos_report.md").exists():
        lines.append("See benchmark/oos_report.md.")
    else:
        lines.append("OOS not run (use --oos).")
    oos_sweep_path = bench_root / "oos_sweep_report.md"
    if oos_sweep_path.exists():
        try:
            sweep_text = oos_sweep_path.read_text(encoding="utf-8")
            if "Skipped:" in sweep_text:
                lines.append(
                    "OOS sweep skipped (insufficient history or split); see benchmark/oos_sweep_report.md."
                )
            else:
                lines.append(
                    "See benchmark/oos_sweep_report.md for filter-sweep OOS (best config on test)."
                )
        except Exception:
            lines.append("See benchmark/oos_sweep_report.md.")
    lines.append("")
    lines.append("## Regime summary")
    lines.append("")
    if (bench_root / "regime_metrics.csv").exists():
        lines.append(
            "See benchmark/regime_metrics.csv. Metrics per volatility/trend/drawdown regime per variant/horizon."
        )
    else:
        lines.append("Regime metrics not run (use --regime).")
    lines.append("")
    lines.append("## Robustness score")
    lines.append("")
    lines.append("Formula: " + _robustness_score_formula())
    lines.append("")
    lines.append("## Anomalies")
    lines.append("")
    anom_path = bench_root / "anomalies.json"
    if anom_path.exists():
        try:
            anom = json.loads(anom_path.read_text(encoding="utf-8"))
            if anom:
                for a in anom[:20]:
                    lines.append(
                        f"- [{a.get('variant_id')} {a.get('horizon')}] {a.get('type')}: {a.get('message')}"
                    )
                if len(anom) > 20:
                    lines.append(
                        f"- ... and {len(anom) - 20} more (see anomalies.json)"
                    )
            else:
                lines.append("None detected.")
        except Exception:
            lines.append("See anomalies.json.")
    else:
        lines.append("Anomaly check not run.")
    lines.append("")
    lines.append("## Profit drivers")
    lines.append("")
    lines.append(
        "Units: total_return as decimal (0.12 = 12%), total_cost_pct in percent (1.5 = 1.5%). gross_total_return_est = total_return + total_cost_pct/100 (estimate). cost_drag_pct = total_cost_pct. return_per_turnover when turnover>0."
    )
    lines.append("")
    valid_rows = [r for r in all_rows if r.get("total_return") is not None]
    if valid_rows:
        by_ret = sorted(
            valid_rows, key=lambda x: (x.get("total_return") or 0), reverse=True
        )[:5]
        by_sharpe = sorted(
            [r for r in valid_rows if r.get("sharpe_ratio") is not None],
            key=lambda x: (x.get("sharpe_ratio") or 0),
            reverse=True,
        )[:5]
        by_stability = sorted(
            valid_rows, key=lambda x: (x.get("stability_score") or 0), reverse=True
        )[:5]
        lines.append(
            "Top by return: "
            + ", ".join(f"{r.get('variant_id')}/{r.get('horizon')}" for r in by_ret)
        )
        lines.append(
            "Top by sharpe: "
            + ", ".join(f"{r.get('variant_id')}/{r.get('horizon')}" for r in by_sharpe)
        )
        lines.append(
            "Top by stability_score: "
            + ", ".join(
                f"{r.get('variant_id')}/{r.get('horizon')}" for r in by_stability
            )
        )
        lines.append(
            "- Cost sensitivity: see trend_baseline_cost_0.5x vs cost_2x; prefer variants that hold up at 2x."
        )
        by_stability_v2 = sorted(
            valid_rows, key=lambda x: (x.get("stability_score_v2") or 0), reverse=True
        )[:5]
        lines.append(
            "Top by stability_score_v2: "
            + ", ".join(
                f"{r.get('variant_id')}/{r.get('horizon')}" for r in by_stability_v2
            )
        )
    else:
        lines.append("No valid runs to summarize.")
    lines.append("")
    lines.append("## Indicator filter impact")
    lines.append("")
    base_rows = {
        r.get("horizon"): r for r in all_rows if r.get("variant_id") == baseline_id
    }
    for r in all_rows:
        vid, hor = r.get("variant_id"), r.get("horizon")
        if vid in (
            "trend_baseline_rsi_filter",
            "trend_baseline_vol_filter",
        ) and base_rows.get(hor):
            b = base_rows[hor]
            dt = (r.get("total_trades") or 0) - (b.get("total_trades") or 0)
            dto = (r.get("turnover") or 0) - (b.get("turnover") or 0)
            ddd = (r.get("max_drawdown_pct") or 0) - (b.get("max_drawdown_pct") or 0)
            dcost = (r.get("cost_share_of_return") or 0) - (
                b.get("cost_share_of_return") or 0
            )
            lines.append(
                f"- {vid} ({hor}): delta_trades={dt} delta_turnover={dto} delta_max_dd={ddd} delta_cost_share={dcost}"
            )
    if not any(
        r.get("variant_id")
        in ("trend_baseline_rsi_filter", "trend_baseline_vol_filter")
        for r in all_rows
    ):
        lines.append("No filter variants in this run.")
    lines.append("")
    lines.append("## Filter effectiveness")
    lines.append("")
    lines.append(
        "Metrics: pct_days_filtered_out (share of baseline in-market days blocked), pct_entries_blocked (trades), mean/median_indicator_on_entry when backtest exports them."
    )
    lines.append("")
    exp_path = bench_root / "indicator_exposure_summary.json"
    if exp_path.exists():
        try:
            exp = json.loads(exp_path.read_text(encoding="utf-8"))
            for row in exp:
                if row.get("indicator_type") != "baseline":
                    parts = [
                        f"{row.get('variant_id')} ({row.get('horizon')}): indicator={row.get('indicator_type')}"
                    ]
                    if row.get("pct_entries_blocked") is not None:
                        parts.append(
                            f"pct_entries_blocked={row.get('pct_entries_blocked')}"
                        )
                    if row.get("pct_days_filtered_out") is not None:
                        parts.append(
                            f"pct_days_filtered_out={row.get('pct_days_filtered_out')}"
                        )
                    parts.append(f"total_trades={row.get('total_trades')}")
                    lines.append("- " + " ".join(parts))
            if not any(r.get("indicator_type") != "baseline" for r in exp):
                lines.append(
                    "See indicator_exposure_summary.csv for pct_days_filtered_out, pct_entries_blocked, mean/median on entry."
                )
        except Exception:
            lines.append("See indicator_exposure_summary.csv.")
    else:
        lines.append("Indicator exposure summary not run.")
    lines.append("")
    lines.append("## Regime win/loss")
    lines.append("")
    if (bench_root / "regime_metrics.csv").exists():
        lines.append(
            "See benchmark/regime_metrics.csv for performance by volatility/trend/drawdown regime. Compare risk_off drawdown across variants."
        )
    else:
        lines.append("Regime metrics not run (use --regime).")
    lines.append("")
    lines.append("## Failure modes")
    lines.append("")
    if anom_path.exists():
        try:
            anom_list = json.loads(anom_path.read_text(encoding="utf-8"))
            if anom_list:
                types = {}
                for a in anom_list:
                    t = a.get("type", "other")
                    types[t] = types.get(t, 0) + 1
                lines.append(
                    "Anomaly summary: "
                    + ", ".join(f"{t}={c}" for t, c in sorted(types.items()))
                )
                lines.append(
                    "Next actions: fix zero_trades (universe/signal), constant_equity (data/strategy), turnover_extreme (costs/rebalance), oos_collapse (avoid overfit)."
                )
            else:
                lines.append("None detected.")
        except Exception:
            lines.append("See anomalies.json.")
    else:
        lines.append("Anomaly check not run.")
    lines.append("")
    lines.append("## No-concept-change recommendations")
    lines.append("")
    lines.append(
        "- Use scoreboard and ablation to pick stable variants; prefer lower cost sensitivity if similar return."
    )
    lines.append("- Prefer strategies with stable Train vs Test when OOS is run.")
    lines.append(
        "- Default trend_baseline remains production default; event/multifactor for diversification experiments."
    )
    lines.append("")
    if (bench_root / "filter_sweep_results.json").exists():
        try:
            sweep = json.loads(
                (bench_root / "filter_sweep_results.json").read_text(encoding="utf-8")
            )
            lines.append("## Filter sweep top 10")
            lines.append("")
            for r in sorted(
                sweep, key=lambda x: (x.get("total_return") or 0), reverse=True
            )[:10]:
                lines.append(
                    f"- {r.get('variant_id')} {r.get('param_name')}={r.get('param_value')}: return={r.get('total_return')} sharpe={r.get('sharpe_ratio')}"
                )
            lines.append("")
        except Exception:
            lines.append("See filter_sweep_results.csv.")
            lines.append("")
    (bench_root / "BENCHMARK_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {bench_root / 'BENCHMARK_REPORT.md'}")


def _filter_sweep_configs() -> list[tuple]:
    """Shared sweep grid: rsi_entry, vol_cap, rv_cap, combo (deterministic order)."""
    return [
        (
            "trend_baseline_rsi_filter",
            {"rsi_entry": 50, "rsi_overbought": 80},
            "rsi_entry",
            50,
        ),
        (
            "trend_baseline_rsi_filter",
            {"rsi_entry": 55, "rsi_overbought": 80},
            "rsi_entry",
            55,
        ),
        (
            "trend_baseline_rsi_filter",
            {"rsi_entry": 60, "rsi_overbought": 80},
            "rsi_entry",
            60,
        ),
        ("trend_baseline_vol_filter", {"vol_cap": 0.20}, "vol_cap", 0.20),
        ("trend_baseline_vol_filter", {"vol_cap": 0.30}, "vol_cap", 0.30),
        (
            "trend_baseline_realized_vol_filter",
            {"rv_cap": 0.20, "rv_window": 20},
            "rv_cap",
            0.20,
        ),
        (
            "trend_baseline_realized_vol_filter",
            {"rv_cap": 0.30, "rv_window": 20},
            "rv_cap",
            0.30,
        ),
        (
            "trend_baseline_realized_vol_filter",
            {"rv_cap": 0.40, "rv_window": 20},
            "rv_cap",
            0.40,
        ),
        (
            "trend_baseline_rsi_vol_combo_filter",
            {"rsi_entry": 55, "rsi_overbought": 80, "vol_cap": 0.20},
            "combo",
            "55_0.20",
        ),
        (
            "trend_baseline_rsi_vol_combo_filter",
            {"rsi_entry": 55, "rsi_overbought": 80, "vol_cap": 0.30},
            "combo",
            "55_0.30",
        ),
        (
            "trend_baseline_rsi_vol_combo_filter",
            {"rsi_entry": 60, "rsi_overbought": 80, "vol_cap": 0.30},
            "combo",
            "60_0.30",
        ),
    ]


def _run_sweep_on_slice(
    slice_path: Path,
    sweep_base_dir: Path,
    freq: str,
    configs: list[tuple],
    horizon_label: str = "1y",
    no_strict_session_gate: bool = False,
) -> list[dict]:
    """Run all configs on slice; return list of metric dicts (variant_id = strategy_param_value)."""
    results = []
    for strategy, params, param_name, param_value in configs:
        run_dir = sweep_base_dir / f"{strategy}_{param_name}_{param_value}"
        run_dir.mkdir(parents=True, exist_ok=True)
        code = _run_backtest(
            str(slice_path),
            run_dir,
            strategy,
            freq,
            DEFAULT_COMMISSION_BPS,
            None,
            params,
            no_strict_session_gate=no_strict_session_gate,
        )
        if code != 0:
            continue
        _run_analysis(run_dir, freq)
        mp = run_dir / "metrics_summary.json"
        if not mp.exists():
            continue
        with mp.open("r", encoding="utf-8") as f:
            m = json.load(f)
        results.append(
            {
                "variant_id": f"{strategy}_{param_name}_{param_value}",
                "param_name": param_name,
                "param_value": param_value,
                "horizon": horizon_label,
                "total_return": m.get("total_return"),
                "sharpe_ratio": m.get("sharpe_ratio"),
                "max_drawdown_pct": m.get("max_drawdown_pct"),
                "turnover": m.get("turnover"),
                "total_trades": m.get("total_trades"),
                "_sweep_strategy": strategy,
                "_sweep_params": params,
            }
        )
    return results


def run_filter_sweep(
    bench_root: Path,
    dataset_path: str,
    freq: str,
    no_strict_session_gate: bool = False,
) -> None:
    """Run filter param sweep on 1y only; write filter_sweep_results.csv/json."""
    horizons = _compute_horizons(dataset_path)
    one_y = [h for h in horizons if h[0] == "1y"]
    if not one_y:
        return
    _, start_d, end_d = one_y[0]
    try:
        prices = _load_and_slice(dataset_path, start_d, end_d)
        if prices.empty:
            return
    except Exception:
        return
    sweep_dir = bench_root / "filter_sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    slice_path = sweep_dir / "price_slice_1y.parquet"
    prices, _ = _ensure_ohlcv(prices)
    prices.to_parquet(slice_path, index=False)
    configs = _filter_sweep_configs()
    results = _run_sweep_on_slice(
        slice_path,
        sweep_dir,
        freq,
        configs,
        "1y",
        no_strict_session_gate=no_strict_session_gate,
    )
    for r in results:
        r["variant_id"] = r.pop("_sweep_strategy", r.get("variant_id", ""))
        r.pop("_sweep_params", None)
    if not results:
        return
    keys = sorted(set().union(*(set(r.keys()) for r in results)))
    with (bench_root / "filter_sweep_results.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as f:
        json.dump(results, f, indent=2, sort_keys=True)
        f.write("\n")
    with (bench_root / "filter_sweep_results.csv").open(
        "w", encoding="utf-8", newline="\n"
    ) as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"Wrote {bench_root / 'filter_sweep_results.csv'} and .json")


MIN_TRADING_DAYS_FOR_OOS_SPLIT = 60


def run_oos_sweep(
    bench_root: Path,
    dataset_path: str,
    freq: str,
    baseline_id: str = "trend_baseline",
    train_frac: float = 0.70,
    no_strict_session_gate: bool = False,
) -> None:
    """70/30 train/test: run filter sweep on train, pick best by stability_score_v2, run on test.
    Writes benchmark/oos_sweep_report.md. (Use --oos-sweep; --oos writes oos_report.md instead.)
    Skips with a written report if history too short (start_date==end_date or < MIN_TRADING_DAYS_FOR_OOS_SPLIT).
    """
    import pandas as pd

    horizons = _compute_horizons(dataset_path)
    one_y = [h for h in horizons if h[0] == "1y"]
    if not one_y:
        msg = "OOS sweep: no 1y horizon."
        (bench_root / "oos_sweep_report.md").write_text(
            "# OOS filter sweep report\n\nSkipped: "
            + msg
            + " Run with dataset that has 1y horizon.\n",
            encoding="utf-8",
        )
        print(msg, file=sys.stderr)
        return
    _, start_d, end_d = one_y[0]
    try:
        prices = _load_and_slice(dataset_path, start_d, end_d)
        if prices.empty or len(prices) < 100:
            msg = "OOS sweep: not enough data (empty or < 100 rows)."
            (bench_root / "oos_sweep_report.md").write_text(
                "# OOS filter sweep report\n\nSkipped: "
                + msg
                + " Use dataset with more history or run without --quick.\n",
                encoding="utf-8",
            )
            print(msg, file=sys.stderr)
            return
    except Exception as e:
        (bench_root / "oos_sweep_report.md").write_text(
            f"# OOS filter sweep report\n\nSkipped: load failed ({e}).\n",
            encoding="utf-8",
        )
        print(f"OOS sweep: load failed {e}", file=sys.stderr)
        return
    trading_dates = sorted(pd.to_datetime(prices["timestamp"]).dt.date.unique())
    n = len(trading_dates)
    if n < MIN_TRADING_DAYS_FOR_OOS_SPLIT:
        (bench_root / "oos_sweep_report.md").write_text(
            f"# OOS filter sweep report\n\n"
            f"Skipped: insufficient history ({n} trading days). Need at least {MIN_TRADING_DAYS_FOR_OOS_SPLIT} for 70/30 split and warmup.\n"
            f"Check run_inputs.json (start_date/end_date). Use --dataset with more history or run without --quick.\n",
            encoding="utf-8",
        )
        print(
            f"OOS sweep: {n} trading days < {MIN_TRADING_DAYS_FOR_OOS_SPLIT}; report written, sweep skipped.",
            file=sys.stderr,
        )
        return
    split_idx = max(1, int(n * train_frac))
    train_end_d = trading_dates[split_idx - 1].isoformat()
    test_start_d = trading_dates[split_idx].isoformat()
    train_df = prices[
        pd.to_datetime(prices["timestamp"]).dt.date <= trading_dates[split_idx - 1]
    ]
    test_df = prices[
        pd.to_datetime(prices["timestamp"]).dt.date >= trading_dates[split_idx]
    ]
    if train_df.empty or test_df.empty:
        (bench_root / "oos_sweep_report.md").write_text(
            f"# OOS filter sweep report\n\nSkipped: 70/30 split left empty set (train or test). "
            f"Trading days: {n}, split_idx: {split_idx}. Use dataset with more history or run without --quick.\n",
            encoding="utf-8",
        )
        print("OOS sweep: split left empty set.", file=sys.stderr)
        return
    oos_dir = bench_root / "filter_sweep" / "oos"
    oos_dir.mkdir(parents=True, exist_ok=True)
    train_path = oos_dir / "train.parquet"
    test_path = oos_dir / "test.parquet"
    train_df, _ = _ensure_ohlcv(train_df)
    test_df, _ = _ensure_ohlcv(test_df)
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)
    configs = _filter_sweep_configs()
    train_sweep_dir = oos_dir / "train_runs"
    train_sweep_dir.mkdir(parents=True, exist_ok=True)
    baseline_run = oos_dir / "train_baseline"
    baseline_run.mkdir(parents=True, exist_ok=True)
    code = _run_backtest(
        str(train_path),
        baseline_run,
        baseline_id,
        freq,
        DEFAULT_COMMISSION_BPS,
        None,
        {},
        no_strict_session_gate=no_strict_session_gate,
    )
    baseline_row = {
        "variant_id": baseline_id,
        "horizon": "train",
        "total_return": None,
        "sharpe_ratio": None,
        "max_drawdown_pct": None,
        "turnover": None,
        "total_trades": None,
    }
    if code == 0:
        _run_analysis(baseline_run, freq)
        mp = baseline_run / "metrics_summary.json"
        if mp.exists():
            with mp.open("r", encoding="utf-8") as f:
                m = json.load(f)
            baseline_row.update(
                total_return=m.get("total_return"),
                sharpe_ratio=m.get("sharpe_ratio"),
                max_drawdown_pct=m.get("max_drawdown_pct"),
                turnover=m.get("turnover"),
                total_trades=m.get("total_trades"),
            )
    sweep_results = _run_sweep_on_slice(
        train_path,
        train_sweep_dir,
        freq,
        configs,
        "train",
        no_strict_session_gate=no_strict_session_gate,
    )
    train_rows = [baseline_row] + [dict(r) for r in sweep_results]
    train_rows = add_robustness_columns(train_rows, [], baseline_id)
    sweep_with_scores = [
        r
        for r in train_rows
        if r.get("variant_id") != baseline_id and r.get("_sweep_strategy")
    ]
    if not sweep_with_scores:
        (bench_root / "oos_sweep_report.md").write_text(
            "# OOS filter sweep report\n\nSkipped: no sweep configs succeeded on train. Check backtest/QC logs.\n",
            encoding="utf-8",
        )
        print("OOS sweep: no sweep configs succeeded.", file=sys.stderr)
        return
    best = max(sweep_with_scores, key=lambda x: float(x.get("stability_score_v2") or 0))
    best_strategy = best.get("_sweep_strategy")
    best_params = best.get("_sweep_params") or {}
    best_dir = oos_dir / "test_best"
    best_dir.mkdir(parents=True, exist_ok=True)
    code_test = _run_backtest(
        str(test_path),
        best_dir,
        best_strategy,
        freq,
        DEFAULT_COMMISSION_BPS,
        None,
        best_params,
        no_strict_session_gate=no_strict_session_gate,
    )
    test_return = test_sharpe = test_dd = None
    if code_test == 0:
        _run_analysis(best_dir, freq)
        mp = best_dir / "metrics_summary.json"
        if mp.exists():
            with mp.open("r", encoding="utf-8") as f:
                m = json.load(f)
            test_return = m.get("total_return")
            test_sharpe = m.get("sharpe_ratio")
            test_dd = m.get("max_drawdown_pct")
    lines = [
        "# OOS filter sweep report",
        "",
        f"Train: {start_d} to {train_end_d} ({split_idx} days). Test: {test_start_d} to {end_d} ({n - split_idx} days).",
        "",
        "## Best config (by stability_score_v2 on train)",
        f"- strategy: {best_strategy}",
        f"- params: {best_params}",
        f"- train stability_score_v2: {best.get('stability_score_v2')}",
        f"- train total_return: {best.get('total_return')}",
        f"- train sharpe_ratio: {best.get('sharpe_ratio')}",
        f"- train max_drawdown_pct: {best.get('max_drawdown_pct')}",
        "",
        "## Test metrics",
        f"- total_return: {test_return}",
        f"- sharpe_ratio: {test_sharpe}",
        f"- max_drawdown_pct: {test_dd}",
        "",
    ]
    (bench_root / "oos_sweep_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {bench_root / 'oos_sweep_report.md'}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Strategy benchmark runner.")
    ap.add_argument("--output-root", type=Path, default=Path("output/system_run"))
    ap.add_argument("--dataset", type=str, default=None)
    ap.add_argument(
        "--synthetic-only", action="store_true", help="Force synthetic data"
    )
    ap.add_argument(
        "--include-synthetic",
        action="store_true",
        help="Use real data if found; else generate synthetic and mark as fallback in report",
    )
    ap.add_argument("--quick", action="store_true", help="1y only, fewer horizons")
    ap.add_argument("--oos", action="store_true", help="Run OOS train/test split")
    ap.add_argument(
        "--regime",
        action="store_true",
        default=True,
        help="Compute regime metrics (default on)",
    )
    ap.add_argument(
        "--no-regime", action="store_false", dest="regime", help="Skip regime analysis"
    )
    ap.add_argument(
        "--attribution",
        action="store_true",
        default=True,
        help="Compute attribution vs baseline (default on)",
    )
    ap.add_argument(
        "--no-attribution",
        action="store_false",
        dest="attribution",
        help="Skip attribution",
    )
    ap.add_argument("--max-variants", type=int, default=None)
    ap.add_argument(
        "--max-sweep", type=int, default=5, help="Max variants for OOS sweep"
    )
    ap.add_argument(
        "--sweep-filters",
        action="store_true",
        dest="sweep_filters",
        help="Run filter param sweep (1y only): RSI/vol/rv/combo grid",
    )
    ap.add_argument(
        "--oos-sweep",
        action="store_true",
        dest="oos_sweep",
        help="70/30 train/test: pick best filter by stability_score_v2 on train, report on test",
    )
    ap.add_argument(
        "--no-strict-session-gate",
        action="store_true",
        dest="no_strict_session_gate",
        help="Pass --no-strict-session-gate to backtest (recommended for 1d EOD data with 00:00 UTC timestamps)",
    )
    ap.add_argument(
        "--variants-config",
        type=Path,
        default=ROOT / "scripts" / "dev" / "benchmark_variants.json",
    )
    args = ap.parse_args()
    output_root = args.output_root.resolve()
    if not output_root.is_absolute():
        output_root = (ROOT / output_root).resolve()

    config = _load_variants(args.variants_config)
    variants = config.get("variants", [])
    baseline_id = config.get("baseline_id", "trend_baseline")
    ablation_ids = config.get("ablation_component_ids", [])

    dataset_path, is_synthetic = _pick_dataset(output_root, args.dataset)
    synthetic_fallback = False
    if args.synthetic_only:
        bench_root = output_root / "benchmark"
        bench_root.mkdir(parents=True, exist_ok=True)
        synth = bench_root / "synthetic" / "eod.parquet"
        _generate_synthetic(synth, "2020-01-01", "2023-12-31")
        dataset_path = str(synth.resolve())
        is_synthetic = True
    elif not dataset_path and args.include_synthetic:
        bench_root = output_root / "benchmark"
        bench_root.mkdir(parents=True, exist_ok=True)
        synth = bench_root / "synthetic" / "eod.parquet"
        _generate_synthetic(synth, "2020-01-01", "2023-12-31")
        dataset_path = str(synth.resolve())
        is_synthetic = True
        synthetic_fallback = True
        print("No real dataset found; using synthetic (fallback).", file=sys.stderr)
    if not dataset_path:
        print(
            "No dataset and not --synthetic-only / --include-synthetic; abort.",
            file=sys.stderr,
        )
        return 1

    print("Running benchmark...")
    all_rows, dataset_label, wiring_anomalies, bar_anomalies = run_benchmark(
        output_root=output_root,
        dataset_path=dataset_path,
        is_synthetic=is_synthetic,
        freq="1d",
        variants=variants,
        baseline_id=baseline_id,
        ablation_ids=ablation_ids,
        max_variants=args.max_variants,
        quick=args.quick,
        no_strict_session_gate=getattr(args, "no_strict_session_gate", False),
    )

    bench_root = output_root / "benchmark"
    variant_horizon_dirs = [
        (r["variant_id"], r["horizon"], bench_root / r["variant_id"] / r["horizon"])
        for r in all_rows
        if r.get("variant_id") and r.get("horizon")
    ]
    all_rows = add_robustness_columns(all_rows, variants, baseline_id)
    write_scoreboard(bench_root, all_rows)
    write_metrics_extended(bench_root, all_rows)
    write_indicator_exposure_summary(bench_root, all_rows, variants, baseline_id)
    write_ablation(bench_root, all_rows, baseline_id, ablation_ids)
    if args.regime and variant_horizon_dirs:
        write_regime_metrics(bench_root, variant_horizon_dirs, "1d")
    variant_ids = [v.get("id") for v in variants if v.get("id")]
    horizons_used = sorted({r["horizon"] for r in all_rows if r.get("horizon")})
    if args.attribution and variant_ids and horizons_used:
        write_attribution_summary(
            bench_root, baseline_id, variant_ids, horizons_used, "1d"
        )
    write_anomalies(
        bench_root,
        all_rows,
        variant_horizon_dirs,
        "1d",
        list(wiring_anomalies) + list(bar_anomalies),
    )
    write_data_quality_summary(bench_root, all_rows, variant_horizon_dirs, "1d")
    if getattr(args, "sweep_filters", False):
        run_filter_sweep(
            bench_root,
            dataset_path,
            "1d",
            no_strict_session_gate=getattr(args, "no_strict_session_gate", False),
        )
    if getattr(args, "oos_sweep", False) and dataset_path and not is_synthetic:
        run_oos_sweep(
            bench_root,
            dataset_path,
            "1d",
            baseline_id=baseline_id,
            no_strict_session_gate=getattr(args, "no_strict_session_gate", False),
        )
    if args.oos and dataset_path and not is_synthetic:
        run_oos(
            output_root,
            dataset_path,
            "1d",
            variants,
            args.max_sweep,
            no_strict_session_gate=getattr(args, "no_strict_session_gate", False),
        )
    report_label = (
        "synthetic (fallback - no real data)" if synthetic_fallback else dataset_label
    )
    write_benchmark_report(
        bench_root, report_label, is_synthetic, all_rows, baseline_id, ablation_ids
    )
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
