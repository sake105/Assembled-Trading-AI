BACKTEST B1 – Unified Backtest Pipeline and Deterministic Seeds
===============================================================

1. Overview and Scope
---------------------

This document describes the design and implementation status of the
unified backtest pipeline (B1) for the Assembled Trading AI backend.

The main goals are:

- Provide a single, well-defined API for portfolio-level backtests
  that can be used consistently from:
  - research scripts and playbooks
  - CLI tools (single backtest runs)
  - EOD / batch jobs
  - profiling and QA tooling
- Ensure deterministic and reproducible backtests via a central
  seeding mechanism.
- Reuse the optimized backtest engine (P3) and batch runner (P4)
  instead of duplicating orchestration logic.

Scope:

- B1 focuses on orchestration and configuration, not on changing any
  core financial logic or strategy semantics.
- B1 is designed to be backwards compatible with existing entry
  points such as ``scripts/run_backtest_strategy.py`` and the
  ``run_backtest`` CLI subcommand.


2. Current State (before B1)
----------------------------

Before B1 the system already contained:

- A portfolio-level backtest engine
  (``src/assembled_core/qa/backtest_engine.py``) with vectorization
  and optional Numba acceleration (P3).
- CLI tooling to run a single backtest
  (``scripts/run_backtest_strategy.py`` and
  ``scripts/cli.py run_backtest``).
- EOD orchestration for execute → backtest → portfolio → QA
  (``scripts/run_eod_pipeline.py`` and
  ``src/assembled_core/pipeline/orchestrator.py``).
- A batch runner for many backtests from YAML/JSON configs (P4)
  (``scripts/batch_backtest.py`` and ``cli.py batch_backtest``).

However, the orchestration logic was still fragmented:

- Single backtests, EOD jobs and batch runs each built their own
  parameter sets and called the engine in slightly different ways.
- Determinism and seeding were not centralised and were often
  handled ad hoc (for example by directly calling
  ``np.random.seed`` inside tests).


3. Target Architecture
----------------------

The unified backtest pipeline introduces three core concepts:

3.1 PortfolioBacktestConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Backtests are described by a single configuration object with fields
such as:

- strategy identifier (for example ``multifactor_long_short``)
- frequency (``freq``) and rebalancing frequency
- universe or symbol list (universe file, symbols file, explicit
  list)
- factor bundle path and optional additional panel paths
- start date, end date
- start capital and risk parameters
- cost model flags (include costs, commission, spread, impact)
- output directory and reporting flags
- optional ``seed`` for deterministic runs

The config is a pure data container. It does not contain any logic
or side effects.

3.2 Central Backtest Runner
~~~~~~~~~~~~~~~~~~~~~~~~~~~

All higher-level tools should converge on a single call pattern:

.. code-block:: python

    from src.assembled_core.qa.backtest_engine import run_portfolio_backtest

    result = run_portfolio_backtest(
        prices=prices,
        signal_fn=signal_fn,
        position_sizing_fn=position_sizing_fn,
        start_capital=cfg.start_capital,
        include_costs=cfg.include_costs,
        include_trades=True,
    )

where:

- input prices and strategy wiring (signal and sizing functions)
  are derived from ``PortfolioBacktestConfig`` by thin adapter
  layers (for example in ``run_backtest_strategy.py`` or EOD
  orchestrators),
- the core engine implementation stays in
  ``qa/backtest_engine.py`` and is shared across all callers.

3.3 Seed Management
~~~~~~~~~~~~~~~~~~~

Deterministic behaviour is handled by a dedicated utility module:

- ``src/assembled_core/utils/random_state.py`` provides:
  - ``set_global_seed(seed: int) -> None``
  - ``seed_context(seed: int)``
- ``set_global_seed`` sets:
  - ``PYTHONHASHSEED`` via ``os.environ``
  - ``random.seed``
  - ``numpy.random.seed``

The backtest pipeline uses this utility whenever a seed is relevant
for a run. Tests and research code are encouraged to call
``set_global_seed`` or ``seed_context`` instead of ad hoc seeding.


4. Data Contracts
-----------------

B1 reuses existing data contracts defined elsewhere in the project:

- Input price panels
  (parquet/csv, ``timestamp`` in UTC, ``symbol``, ``close`` and
  optional OHLCV columns).
- Factor bundles (YAML files under ``config/factor_bundles``)
  specifying factors, weights and options.
- Backtest outputs:
  - ``equity_curve_*.csv`` / ``equity_curve_*.parquet``
  - ``trades.csv`` / ``trades.parquet``
  - performance reports, risk reports, TCA reports.

B1 does not change these contracts; it only standardises how they
are wired into and out of the backtest engine.


5. Testing Strategy
-------------------

The unified pipeline is covered by several layers of tests:

- Engine-level regression tests
  (``tests/test_qa_backtest_engine.py``,
  ``tests/test_qa_backtest_engine_numba.py``,
  ``tests/test_performance_backtest_engine_regression.py``)
  ensure numerical stability and performance.
- CLI and orchestration tests
  (for example ``tests/test_cli_run_backtest_strategy.py``,
  ``tests/test_cli_batch_backtest.py``,
  ``tests/test_research_playbook_ai_tech.py``) verify that the
  various entry points still behave consistently.
- Seed and determinism tests
  (``tests/test_utils_random_state.py`` and
  ``tests/test_backtest_determinism.py``) confirm that:
  - global seeding via ``set_global_seed`` affects numpy and
    ``random`` as expected,
  - repeated backtests with the same config and seed yield
    identical equity curves and trade logs,
  - different seeds can lead to different paths, but remain
    deterministic for each individual seed.


6. Implementation Status
------------------------

Status: B1 is implemented and integrated.

Key elements that are now in place:

- Optimised backtest engine with vectorisation and optional Numba
  acceleration (P3) is the single source of truth for portfolio
  backtests.
- CLI entry points and batch runners delegate to this engine
  instead of implementing their own backtest logic.
- Central seed management is implemented in
  ``src/assembled_core/utils/random_state.py`` and used by tests
  and research tooling.
- Dedicated determinism tests for backtests are available in
  ``tests/test_backtest_determinism.py``.

Relationship to other phases:

- P1 (profiling) and P3 (engine optimisation) provide the runtime
  foundation that B1 relies on.
- P4 (batch backtests) uses the same engine and configuration
  principles for many runs.
- D2/E4 (risk and TCA workflows) consume backtest outputs that are
  produced via the unified pipeline.


7. Future Work
--------------

Potential follow-up improvements include:

- A concrete ``PortfolioBacktestConfig`` dataclass in the QA layer
  that can be serialised to and from YAML/JSON and used directly
  by EOD and batch jobs.
- Helpers to persist the effective backtest configuration next to
  each run (for example as ``backtest_config.yaml`` under the
  backtest output directory) to make audits and reproduction even
  easier.
- Additional integration tests comparing EOD / batch / research
  entry points for strict numerical equivalence under the same
  configuration.


