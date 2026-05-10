"""Rigorose Backtest-Validierung.

Inhalt
------
- ``cpcv``                  : Combinatorial Purged CV (Lopez de Prado).
- ``deflated_sharpe``       : Bailey & Lopez de Prado (2014) Deflated Sharpe.
- ``white_reality_check``   : White (2000), Hansen (2005) SPA-Test.
- ``stationary_bootstrap``  : Politis-Romano (1994) für Time-Series-Bootstrap.
- ``performance_metrics``   : Sharpe, Sortino, Calmar, ... mit corrected SE.
- ``run_full_backtest``     : End-to-End Walk-Forward-Backtest.
"""
