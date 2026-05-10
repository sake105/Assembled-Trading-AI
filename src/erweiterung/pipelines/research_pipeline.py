"""End-to-End Research Pipeline.

Stages
------
1. **Data**           : load OHLCV panel
2. **Signal**         : compute multi-factor signals
3. **Strategy**       : convert signals to positions
4. **Backtest**       : compute returns + metrics
5. **Validation**     : Reality-Check, DSR, IC-decay
6. **Stress-Test**    : historical replay + Monte-Carlo
7. **Report**         : HTML report + JSON metrics

Verwendung als Library oder per Command-Line-Script.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ResearchPipelineConfig:
    output_dir: str = "output/erweiterung_research"
    transaction_cost_bps: float = 5.0
    quantile: float = 0.2
    n_bootstrap: int = 2000
    seed: int = 42


@dataclass
class ResearchResult:
    config: ResearchPipelineConfig
    strategies: dict[str, pd.Series] = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    reality_check: dict = field(default_factory=dict)
    spa: dict = field(default_factory=dict)
    stress: dict = field(default_factory=dict)


def run_research_pipeline(
    panel: pd.DataFrame,
    market_returns: pd.Series,
    signal_builder: Callable[[pd.DataFrame, pd.Series], pd.DataFrame],
    strategy_runners: dict[str, Callable[[pd.DataFrame], pd.Series]],
    config: ResearchPipelineConfig | None = None,
    benchmark: Optional[pd.Series] = None,
) -> ResearchResult:
    """Run the full research pipeline.

    Args:
        panel: long-format DataFrame [date, symbol, open, high, low, close, volume, return].
        market_returns: market series.
        signal_builder: callable that adds signal columns to panel.
        strategy_runners: dict of {name: callable(panel_with_signals) -> Series}.
        config: pipeline config.
        benchmark: optional benchmark return-series.

    Returns:
        ResearchResult.
    """
    cfg = config or ResearchPipelineConfig()
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Building signals ...")
    enriched = signal_builder(panel, market_returns)

    logger.info("Running strategies ...")
    strategies = {}
    for name, runner in strategy_runners.items():
        try:
            strategies[name] = runner(enriched)
        except Exception as e:  # noqa: BLE001
            logger.warning("Strategy %s failed: %s", name, e)
            continue

    if benchmark is None:
        benchmark = panel.groupby("date")["return"].mean()
    if not isinstance(benchmark.index, pd.DatetimeIndex):
        benchmark.index = pd.to_datetime(benchmark.index)
    if benchmark.index.tz is None:
        benchmark.index = benchmark.index.tz_localize("UTC")
    strategies["benchmark"] = benchmark

    logger.info("Computing metrics ...")
    from erweiterung.backtest.deflated_sharpe import deflated_sharpe_ratio
    from erweiterung.backtest.performance_metrics import all_metrics
    from erweiterung.backtest.white_reality_check import (
        hansen_spa_test,
        whites_reality_check,
    )
    from erweiterung.risk_metrics.advanced_metrics import comprehensive_metrics

    metrics: dict[str, dict] = {}
    for name, ret in strategies.items():
        m = all_metrics(ret, benchmark=benchmark)
        m.update(comprehensive_metrics(ret, benchmark=benchmark))
        dsr = deflated_sharpe_ratio(ret, n_trials=len(strategies))
        m["dsr_z"] = dsr.get("dsr_z", float("nan"))
        m["dsr_p"] = dsr.get("dsr_p", float("nan"))
        metrics[name] = m

    excess = (
        pd.DataFrame(strategies)
        .fillna(0)
        .subtract(strategies["benchmark"], axis=0)
        .drop(columns=["benchmark"])
    )
    rc = whites_reality_check(excess, n_bootstrap=cfg.n_bootstrap, seed=cfg.seed)
    spa = hansen_spa_test(excess, n_bootstrap=cfg.n_bootstrap, seed=cfg.seed)

    logger.info("Stress-testing ...")
    from erweiterung.stress_test.historical_replay import (
        replay_all_crises,
        stress_score,
    )
    from erweiterung.stress_test.monte_carlo import (
        MCConfig,
        path_metrics,
        simulate_paths,
    )

    stress: dict = {}
    main_strat = next(iter(strategies))
    main_returns = strategies[main_strat]
    if len(main_returns.dropna()) > 100:
        replay = replay_all_crises(main_returns)
        stress["historical_replay"] = replay.to_dict("records")
        stress["stress_score"] = stress_score(replay)
        try:
            paths = simulate_paths(
                main_returns,
                MCConfig(n_paths=500, horizon=126, seed=cfg.seed, method="stationary"),
            )
            stress["monte_carlo"] = path_metrics(paths)
        except Exception as e:  # noqa: BLE001
            stress["monte_carlo_error"] = str(e)

    # Save outputs
    json_path = out_dir / "metrics.json"

    def _convert(o):
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, pd.Timestamp):
            return o.isoformat()
        try:
            if pd.isna(o):
                return None
        except (TypeError, ValueError):
            pass
        return o

    def _walk(o):
        if isinstance(o, dict):
            return {str(k): _walk(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_walk(v) for v in o]
        return _convert(o)

    json_path.write_text(
        json.dumps(
            _walk(
                {
                    "metrics": metrics,
                    "reality_check": rc,
                    "spa": spa,
                    "stress": stress,
                }
            ),
            indent=2,
            default=str,
        )
    )

    # Equity curves CSV
    eq_csv = pd.DataFrame({k: (1 + v).cumprod() for k, v in strategies.items()})
    eq_csv.to_csv(out_dir / "equity_curves.csv")

    # HTML Report
    try:
        from erweiterung.report.html_report import generate_full_report

        html = generate_full_report(
            title="Erweiterung Research Pipeline Report",
            strategy_returns=strategies,
            metrics=metrics,
            output_path=str(out_dir / "report.html"),
            main_strategy=main_strat,
        )
        logger.info("HTML report: %s", html)
    except Exception as e:  # noqa: BLE001
        logger.warning("HTML report failed: %s", e)

    return ResearchResult(
        config=cfg,
        strategies=strategies,
        metrics=metrics,
        reality_check=rc,
        spa=spa,
        stress=stress,
    )


__all__ = ["ResearchPipelineConfig", "ResearchResult", "run_research_pipeline"]
