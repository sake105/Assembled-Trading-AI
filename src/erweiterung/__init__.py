"""Assembled-Trading-AI — ERWEITERUNG.

Forschungs-Erweiterungspaket auf Basis des bestehenden Cores.

Das Paket ist intentional **isoliert** vom Hauptcode: Kein Import aus
`erweiterung` in `assembled_core` (eine Richtung), damit der Mainline-Branch
unbeeinflusst bleibt.

Submodule
---------
- altdata    : kostenlose alternative Datenquellen
- signals    : neuartige Signale auf altdata + Preisen
- ml         : moderne ML-Methoden (Conformal, Stacking, Patch-TS)
- portfolio  : HRP, Black-Litterman, CVaR-Optimierung
- risk       : Tail-Risk, dynamische DD-Kontrolle, EVT
- backtest   : Combinatorial Purged CV, Deflated Sharpe, White Reality Check
- execution  : adaptive Slippage, TWAP/VWAP-Ausführung
- meta       : Strategie-Orchestrierung, Regime-Routing
"""

__version__ = "0.1.0"
__all__ = [
    "altdata",
    "signals",
    "ml",
    "portfolio",
    "risk",
    "backtest",
    "execution",
    "meta",
]
