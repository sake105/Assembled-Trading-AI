# Master Implementation Plan v4 + Bonus

**Datum:** 2026-04-29  
**Repo-Stand:** `691c174` (HEAD on main)  
**Charakter dieses Dokuments:** Vollständige, in sich abgeschlossene Implementierungspläne für alle v4-Items plus 7 weitere Themen, die ich beim Schreiben von v4 als wertvoll identifiziert habe. Jedes Item hat genaue Repo-Stellen, konkrete Code-Skelette, Tests, Risiken und Aufwands-Schätzung.

**Reihenfolge der Items spiegelt meine Empfehlung der Reihenfolge wider** — von "schnelle Wins, sofort umsetzbar" bis "Research-Pfad, hohe Unsicherheit".

---

## Inhaltsverzeichnis

### Aus v4
1. [Polymarket-Source für Geo-Risk](#1-polymarket) — 6-10h
2. [Bayesian Sharpe + Strategy-Comparison](#2-bayesian-sharpe) — 12-18h
3. [Redis Streams Phase 1 (Side-Channel)](#3-redis-streams-phase-1) — 8-12h
4. [QuestDB als Tick-Store](#4-questdb) — 8-12h
5. [RL-Execution-Policy](#5-rl-execution) — 40-60h
6. [GNN für Stock-Prediction](#6-gnn-stocks) — 30-50h

### Bonus (in v4 nicht enthalten, aber sinnvoll)
7. [LLM-RAG für News-Reasoning](#7-llm-rag) — 16-24h
8. [Risk-Parity Vol-Targeting auf Strategy-Level](#8-risk-parity) — 8-12h
9. [Walk-Forward-Optimization mit Optuna](#9-walk-forward-optuna) — 10-14h
10. [Differential Privacy für Strategie-Sharing](#10-diff-privacy) — 20-30h
11. [Order-Book-Imbalance-Features](#11-order-book-imbalance) — 8-12h
12. [Regime-Conditional-Strategy-Allokation](#12-regime-allocation) — 12-18h
13. [Kalshi als zweite Prediction-Market-Source](#13-kalshi) — 4-6h

[Master-Adoption-Plan](#master-adoption-plan)

---

# 1. Polymarket-Source für Geo-Risk

**Aufwand:** 6-10h  
**Tier:** 1 — sofort empfehlenswert  
**Risiko:** sehr niedrig (read-only API, keine Auth nötig)

## Was du heute hast

- 18 Data-Sources in `data/sources/` (alle als Funktionen, nicht als Klassen)
- `intel/bayesian_confidence.py` mit `SOURCE_RELIABILITY` und `FALSE_POSITIVE_RATE` pro Tier
- `risk/georisk_overlay.py` mit `compute_exposure_multiplier(ctx, policy)`

## Was Polymarket dir gibt

Crowd-sourced Echtzeit-Probabilitäten für Geo-Risk-Events, die du heute nur indirekt aus News-Sentiment ableitest. Liquid-Markets sind notorisch gut kalibriert (Brier-Score deutlich besser als Pundit-Predictions, Manifold/Polymarket-Studien 2024).

## Schritt-für-Schritt-Plan

### Schritt 1 — `data/sources/polymarket_source.py` (neu, ~150 LOC, 4h)

Folge dem **functional pattern** deiner bestehenden Sources:

```python
# data/sources/polymarket_source.py
"""Polymarket Gamma + CLOB read-only data source.

No authentication required for market metadata + price history.
Auth only needed for trading operations (out of scope).

Use cases:
- Crowd-sourced probabilities for geopolitics events (T2 source tier)
- Cross-validation of GDELT/Bluesky sentiment signals
- Independent prior for risk/georisk_overlay.py

Install: only `httpx` (you already have it for other sources).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import httpx
import pandas as pd

logger = logging.getLogger(__name__)

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"

# Geo-related Polymarket tag IDs (verify current values via /tags endpoint)
GEO_TAGS = {
    "geopolitics": 100,
    "elections": 101,  # placeholder — fetch live
    "war_conflict": 102,
}


def fetch_polymarket_geopolitics(
    min_volume_usd: float = 10_000.0,
    timeout: float = 20.0,
) -> pd.DataFrame:
    """Fetch active geopolitics markets above volume threshold.
    
    Returns
    -------
    pandas.DataFrame
        Columns: timestamp, market_id, slug, question, outcome,
                 probability, volume_24h, liquidity, end_date.
        Empty DataFrame on failure.
    """
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.get(
                f"{GAMMA_BASE}/markets",
                params={
                    "active": "true",
                    "tag_id": GEO_TAGS["geopolitics"],
                    "min_volume": min_volume_usd,
                    "limit": 200,
                },
            )
            r.raise_for_status()
            data = r.json()
    except (httpx.HTTPError, ValueError) as e:
        logger.warning("polymarket fetch failed: %s", e)
        return pd.DataFrame()
    
    rows: list[dict[str, Any]] = []
    now = pd.Timestamp.now(tz="UTC")
    for m in data:
        for outcome in m.get("outcomes", []):
            try:
                rows.append({
                    "timestamp": now,
                    "market_id": str(m["id"]),
                    "slug": m.get("slug", ""),
                    "question": m.get("question", ""),
                    "outcome": outcome.get("name", ""),
                    "probability": float(outcome.get("price", 0.0)),
                    "volume_24h": float(m.get("volume24h", 0.0)),
                    "liquidity": float(m.get("liquidity", 0.0)),
                    "end_date": m.get("endDate"),
                })
            except (KeyError, TypeError, ValueError) as e:
                logger.debug("skipping malformed market: %s", e)
                continue
    
    return pd.DataFrame(rows)


def fetch_polymarket_history(
    token_id: str,
    start: datetime,
    end: datetime,
    fidelity_seconds: int = 60,
    timeout: float = 20.0,
) -> pd.DataFrame:
    """Fetch historical price for a single outcome token.
    
    Parameters
    ----------
    token_id : str
        Polymarket conditional-token ID.
    fidelity_seconds : int
        Resolution: 60 (1-min), 300 (5-min), 3600 (1-h), 86400 (1-day).
    """
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.get(
                f"{CLOB_BASE}/prices-history",
                params={
                    "market": token_id,
                    "startTs": int(start.timestamp()),
                    "endTs": int(end.timestamp()),
                    "fidelity": fidelity_seconds,
                },
            )
            r.raise_for_status()
            data = r.json()
    except (httpx.HTTPError, ValueError) as e:
        logger.warning("polymarket history fetch failed: %s", e)
        return pd.DataFrame()
    
    history = data.get("history", [])
    if not history:
        return pd.DataFrame()
    
    return pd.DataFrame([
        {
            "timestamp": pd.Timestamp(d["t"], unit="s", tz="UTC"),
            "probability": float(d["p"]),
        }
        for d in history
    ])


def filter_relevant_geo_markets(
    markets: pd.DataFrame,
    keywords: list[str] | None = None,
) -> pd.DataFrame:
    """Filter Polymarket geo markets to topics you care about."""
    if markets.empty:
        return markets
    keywords = keywords or [
        "taiwan", "china", "russia", "ukraine",
        "iran", "israel", "north korea",
        "sanctions", "tariffs",
    ]
    pattern = "|".join(keywords)
    mask = markets["question"].str.lower().str.contains(pattern, regex=True, na=False)
    return markets[mask].copy()
```

### Schritt 2 — Tests `tests/test_polymarket_source.py` (~80 LOC, 1.5h)

```python
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from src.assembled_core.data.sources.polymarket_source import (
    fetch_polymarket_geopolitics,
    fetch_polymarket_history,
    filter_relevant_geo_markets,
)


class TestPolymarketSource:
    def test_returns_empty_on_http_error(self):
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.get.side_effect = \
                __import__("httpx").HTTPError("boom")
            result = fetch_polymarket_geopolitics()
            assert isinstance(result, pd.DataFrame)
            assert result.empty

    def test_parses_well_formed_response(self):
        sample = [{
            "id": "0xabc", "slug": "tw-conflict-2026",
            "question": "Will there be conflict in Taiwan in 2026?",
            "volume24h": 50000, "liquidity": 200000,
            "endDate": "2026-12-31T00:00:00Z",
            "outcomes": [
                {"name": "Yes", "price": 0.18},
                {"name": "No", "price": 0.82},
            ],
        }]
        with patch("httpx.Client") as mock_client:
            mock = MagicMock()
            mock.json.return_value = sample
            mock.raise_for_status.return_value = None
            mock_client.return_value.__enter__.return_value.get.return_value = mock
            df = fetch_polymarket_geopolitics()
            assert len(df) == 2
            assert set(df["outcome"]) == {"Yes", "No"}
            assert df.loc[df.outcome == "Yes", "probability"].iloc[0] == 0.18

    def test_filter_keywords(self):
        df = pd.DataFrame({
            "question": ["Taiwan war?", "Bitcoin to 200k?", "Russia sanctions?"],
            "probability": [0.2, 0.5, 0.3],
        })
        for c in ["timestamp", "market_id", "slug", "outcome",
                  "volume_24h", "liquidity", "end_date"]:
            df[c] = ""
        result = filter_relevant_geo_markets(df)
        assert len(result) == 2  # Taiwan + Russia
```

### Schritt 3 — Integration in `risk/georisk_overlay.py` (~50 LOC, 2h)

```python
# risk/georisk_overlay.py — additive Erweiterung

def get_market_implied_geo_signal(policy: dict) -> dict[str, float]:
    """Derive a geo-risk multiplier from Polymarket conflict probabilities.
    
    Returns dict with keys:
    - 'pm_max_conflict_prob': max probability across watched markets
    - 'pm_volume_weighted_prob': vol-weighted prob (more reliable)
    - 'pm_n_markets_above_20pct': count of high-prob markets
    """
    from src.assembled_core.data.sources.polymarket_source import (
        fetch_polymarket_geopolitics,
        filter_relevant_geo_markets,
    )
    
    cfg = policy.get("polymarket", {})
    if not cfg.get("enabled", False):
        return {}
    
    markets = fetch_polymarket_geopolitics(
        min_volume_usd=cfg.get("min_volume_usd", 10_000),
    )
    if markets.empty:
        return {}
    
    relevant = filter_relevant_geo_markets(markets, cfg.get("keywords"))
    if relevant.empty:
        return {}
    
    # Focus on "Yes" outcomes (the conflict-realising side)
    yes_markets = relevant[relevant["outcome"].str.lower() == "yes"]
    if yes_markets.empty:
        return {}
    
    weights = yes_markets["volume_24h"].clip(lower=1.0)
    vol_weighted = (yes_markets["probability"] * weights).sum() / weights.sum()
    
    return {
        "pm_max_conflict_prob": float(yes_markets["probability"].max()),
        "pm_volume_weighted_prob": float(vol_weighted),
        "pm_n_markets_above_20pct": int((yes_markets["probability"] > 0.2).sum()),
    }


def compute_exposure_multiplier(ctx, policy):
    # ... existing logic ...
    
    # NEW: Polymarket overlay
    pm_signal = get_market_implied_geo_signal(policy)
    if pm_signal:
        # If volume-weighted conflict-prob > threshold, reduce exposure
        threshold = policy.get("polymarket", {}).get("threshold", 0.25)
        if pm_signal["pm_volume_weighted_prob"] > threshold:
            multiplier *= (1.0 - 0.5 * (pm_signal["pm_volume_weighted_prob"] - threshold))
    
    return max(min(multiplier, 1.0), 0.0)
```

### Schritt 4 — Bayesian-Confidence-Tier (~30 LOC, 0.5h)

In `intel/bayesian_confidence.py`:

```python
# Polymarket as a high-confidence source tier
SOURCE_RELIABILITY["T1.5"] = 0.78   # between licensed news (T1=0.85) and aggregators (T2=0.65)
FALSE_POSITIVE_RATE["T1.5"] = 0.07
```

Polymarket-Probabilitäten verdienen **bessere** Reliability als GDELT (T2) weil sie marktdiszipliniert sind, aber **schlechter** als Reuters (T1) weil sie crowd-sourced sind.

## Tests-to-Run

```bash
pytest tests/test_polymarket_source.py
pytest tests/test_georisk_overlay.py  # ensure existing tests still pass
```

## Risiken

- API-Endpoints könnten sich ändern (Polymarket entwickelt sich rapide). Wrap in try/except mit graceful empty-fallback (siehe Code).
- Tag-IDs müssen verifiziert werden — Schritt 1 hat Placeholder. Erste Aktion: einen Probe-Call zu `/tags` machen, IDs notieren.

## Was du nach 6-10h hast

- 19. Data-Source in deiner Pipeline
- Realtime-Geo-Risk-Probabilitäten in `georisk_overlay.py`
- Polymarket als T1.5-Tier in deinem Bayesian-Confidence-Engine
- Crowd-Wisdom-Cross-Check für deine GDELT/Bluesky-Sentiment-Signals

---

# 2. Bayesian Sharpe + Strategy-Comparison

**Aufwand:** 12-18h  
**Tier:** 1 — sofort empfehlenswert  
**Risiko:** mittel (PyMC ist eine ~500-MB-Dependency, soft-optional ist Pflicht)

## Was du heute hast

- `qa/bootstrap_metrics.py` (114 LOC) — klassische Bootstrap-CIs
- `intel/bayesian_confidence.py` (207 LOC) — sequenzielle Bayes-Updates für News
- Mehrere Strategien in `strategies/` (`ema_trend_v0`, `multifactor_v1`, `multifactor_v2`, `multifactor_long_short`, …)

## Was Bayesian dir gibt, was Bootstrap nicht kann

- `P(true_sharpe > 0.5 | observed_returns)` direkt interpretierbar
- Hierarchical Pooling regularisiert kleine-Sample-Strategien (1-Year-Track-Record bekommt geschrumpften Sharpe)
- Posterior-Predictive-Checks: "Wenn dein Modell stimmt, wie sähe der nächste Monat aus?"
- Priors aus Live-Trading-Erfahrung einbaubar

## Schritt-für-Schritt-Plan

### Schritt 1 — `qa/bayesian_metrics.py` (neu, ~250 LOC, 6h)

```python
"""Bayesian alternatives to bootstrap_metrics for Sharpe and friends.

Soft-optional dependency on PyMC. Falls back to bootstrap on ImportError.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import pymc as pm
    import arviz as az
    HAS_PYMC = True
except ImportError:
    HAS_PYMC = False


def bayesian_sharpe_posterior(
    returns: np.ndarray,
    periods_per_year: int = 252,
    n_samples: int = 2000,
    n_chains: int = 4,
    use_studentt: bool = True,
    prior_mu_sigma: float = 0.01,
    prior_sigma_scale: float = 0.05,
    progressbar: bool = False,
) -> dict[str, Any]:
    """Estimate posterior of true Sharpe ratio.
    
    Uses Student-t likelihood by default (heavy tails — robust to outliers).
    Falls back to bootstrap-CI on PyMC ImportError.
    """
    if not HAS_PYMC:
        from src.assembled_core.qa.bootstrap_metrics import bootstrap_sharpe_ci
        warnings.warn(
            "PyMC not installed; falling back to bootstrap CI. "
            "Install with: pip install pymc arviz",
            stacklevel=2,
        )
        return bootstrap_sharpe_ci(returns, periods_per_year)
    
    if len(returns) < 30:
        raise ValueError(f"Need ≥30 observations for stable inference, got {len(returns)}.")
    
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0.0, sigma=prior_mu_sigma)
        sigma = pm.HalfNormal("sigma", sigma=prior_sigma_scale)
        if use_studentt:
            nu = pm.Exponential("nu", lam=1/30)
            pm.StudentT("returns", nu=nu, mu=mu, sigma=sigma, observed=returns)
        else:
            pm.Normal("returns", mu=mu, sigma=sigma, observed=returns)
        sharpe = pm.Deterministic(
            "sharpe", mu / sigma * np.sqrt(periods_per_year)
        )
        idata = pm.sample(
            n_samples, tune=1000, chains=n_chains,
            target_accept=0.95, progressbar=progressbar,
        )
    
    sharpe_post = idata.posterior["sharpe"].values.flatten()
    hdi_94 = az.hdi(sharpe_post, hdi_prob=0.94)
    
    return {
        "method": "bayesian_studentt" if use_studentt else "bayesian_normal",
        "sharpe_mean": float(sharpe_post.mean()),
        "sharpe_std": float(sharpe_post.std()),
        "sharpe_median": float(np.median(sharpe_post)),
        "hdi_94": (float(hdi_94[0]), float(hdi_94[1])),
        "p_sharpe_positive": float((sharpe_post > 0).mean()),
        "p_sharpe_above_0_5": float((sharpe_post > 0.5).mean()),
        "p_sharpe_above_1_0": float((sharpe_post > 1.0).mean()),
        "n_observations": int(len(returns)),
        "rhat": float(az.rhat(idata)["sharpe"].max()),
        "ess": float(az.ess(idata)["sharpe"].min()),
    }


def hierarchical_strategy_comparison(
    returns_per_strategy: dict[str, np.ndarray],
    periods_per_year: int = 252,
    n_samples: int = 2000,
    progressbar: bool = False,
) -> dict[str, Any]:
    """Compare N strategies via partial pooling.
    
    Strategies with shorter track records get shrunk toward population mean —
    automatic regularization against small-sample winners.
    """
    if not HAS_PYMC:
        raise ImportError(
            "PyMC required for hierarchical comparison. "
            "Install with: pip install pymc arviz"
        )
    
    strategy_names = list(returns_per_strategy.keys())
    n_strategies = len(strategy_names)
    if n_strategies < 2:
        raise ValueError(f"Need ≥2 strategies, got {n_strategies}.")
    
    all_returns = np.concatenate(list(returns_per_strategy.values()))
    strategy_idx = np.concatenate([
        np.full(len(r), i, dtype=int)
        for i, r in enumerate(returns_per_strategy.values())
    ])
    
    with pm.Model() as model:
        mu_pop = pm.Normal("mu_pop", mu=0.0, sigma=0.005)
        tau_mu = pm.HalfNormal("tau_mu", sigma=0.005)
        sigma_pop = pm.HalfNormal("sigma_pop", sigma=0.02)
        
        mu = pm.Normal("mu", mu=mu_pop, sigma=tau_mu, shape=n_strategies)
        sigma = pm.HalfNormal("sigma", sigma=sigma_pop, shape=n_strategies)
        
        pm.Normal(
            "returns",
            mu=mu[strategy_idx],
            sigma=sigma[strategy_idx],
            observed=all_returns,
        )
        sharpe = pm.Deterministic(
            "sharpe", mu / sigma * np.sqrt(periods_per_year)
        )
        idata = pm.sample(
            n_samples, tune=1000, chains=4,
            target_accept=0.95, progressbar=progressbar,
        )
    
    sharpe_post = idata.posterior["sharpe"].values.reshape(-1, n_strategies)
    
    pairwise_prob = {}
    per_strategy = {}
    for i, name_a in enumerate(strategy_names):
        per_strategy[name_a] = {
            "sharpe_mean": float(sharpe_post[:, i].mean()),
            "sharpe_std": float(sharpe_post[:, i].std()),
            "n_observations": int(len(returns_per_strategy[name_a])),
        }
        for j, name_b in enumerate(strategy_names):
            if i < j:
                p_ab = float((sharpe_post[:, i] > sharpe_post[:, j]).mean())
                pairwise_prob[f"{name_a}>{name_b}"] = p_ab
                pairwise_prob[f"{name_b}>{name_a}"] = 1.0 - p_ab
    
    best_idx = sharpe_post.mean(axis=0).argmax()
    p_best_truly_best = float(
        (sharpe_post.argmax(axis=1) == best_idx).mean()
    )
    
    return {
        "per_strategy": per_strategy,
        "pairwise_probability": pairwise_prob,
        "apparent_best": strategy_names[best_idx],
        "p_apparent_best_truly_best": p_best_truly_best,
    }
```

### Schritt 2 — Tests (~120 LOC, 3h)

```python
# tests/test_bayesian_metrics.py
import numpy as np
import pytest

pymc = pytest.importorskip("pymc")

from src.assembled_core.qa.bayesian_metrics import (
    bayesian_sharpe_posterior,
    hierarchical_strategy_comparison,
)


class TestBayesianSharpe:
    def test_recovers_known_sharpe(self):
        rng = np.random.default_rng(42)
        true_mu, true_sigma = 0.0008, 0.012
        returns = rng.normal(true_mu, true_sigma, 500)
        # Quick sample for tests
        result = bayesian_sharpe_posterior(
            returns, n_samples=500, n_chains=2, use_studentt=False,
        )
        true_sharpe = true_mu / true_sigma * np.sqrt(252)
        # Within HDI
        assert result["hdi_94"][0] <= true_sharpe <= result["hdi_94"][1]
        # Convergence checks
        assert result["rhat"] < 1.05
        assert result["ess"] > 100

    def test_too_few_observations_raises(self):
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="Need ≥30"):
            bayesian_sharpe_posterior(rng.normal(0, 0.01, 20))

    def test_p_sharpe_positive_high_for_winning_strategy(self):
        rng = np.random.default_rng(7)
        returns = rng.normal(0.001, 0.01, 500)  # clear positive Sharpe
        result = bayesian_sharpe_posterior(
            returns, n_samples=500, n_chains=2, use_studentt=False,
        )
        assert result["p_sharpe_positive"] > 0.95


class TestHierarchicalComparison:
    def test_winner_emerges_from_pooled_inference(self):
        rng = np.random.default_rng(1)
        strategies = {
            "winner": rng.normal(0.001, 0.01, 500),
            "loser": rng.normal(-0.0005, 0.012, 500),
            "mediocre": rng.normal(0.0002, 0.011, 500),
        }
        result = hierarchical_strategy_comparison(strategies, n_samples=500)
        assert result["apparent_best"] == "winner"
        assert result["pairwise_probability"]["winner>loser"] > 0.95

    def test_short_track_record_gets_shrunk(self):
        rng = np.random.default_rng(2)
        strategies = {
            "long_track": rng.normal(0.0005, 0.01, 1000),
            "short_lucky": rng.normal(0.003, 0.008, 30),  # 30 days, lucky
        }
        result = hierarchical_strategy_comparison(strategies, n_samples=500)
        # Despite higher raw Sharpe, the lucky strategy's posterior should be
        # shrunk toward population mean — uncertainty about whether it's real.
        assert result["per_strategy"]["short_lucky"]["sharpe_std"] > \
               result["per_strategy"]["long_track"]["sharpe_std"]
```

### Schritt 3 — Daily-QA-Report-Integration (~30 LOC, 1h)

In `qa/daily_qa_report.py` (oder wo dein QA-Report lebt) zusätzlichen Section:

```python
def add_bayesian_section(report, returns_per_strategy):
    try:
        from src.assembled_core.qa.bayesian_metrics import (
            hierarchical_strategy_comparison,
        )
        bayes = hierarchical_strategy_comparison(returns_per_strategy)
        report["bayesian"] = bayes
        # Highlight uncertainty
        if bayes["p_apparent_best_truly_best"] < 0.7:
            report["warnings"].append(
                f"Apparent best strategy '{bayes['apparent_best']}' is only "
                f"{bayes['p_apparent_best_truly_best']:.0%} likely to be the "
                f"true best — sample sizes are too small to differentiate."
            )
    except ImportError:
        pass
```

### Schritt 4 — Soft-Optional in `pyproject.toml` (~5 LOC, 0.5h)

```toml
[project.optional-dependencies]
bayesian = [
  "pymc>=5.20",
  "arviz>=0.18",
  "numpyro>=0.13",  # optional faster sampler
]
```

## Tests-to-Run

```bash
pip install pymc arviz
pytest tests/test_bayesian_metrics.py
pytest tests/test_competitive_analysis_impl.py  # ensure no regression
```

## Risiken

- **PyMC-Compile-Time**: erste Modell-Initialisierung ~10 s (PyTensor compile). Inakzeptabel für Hot-Path. Lösung: nur in Daily-QA-Report einsetzen, nicht in `trading_cycle_v2`.
- **Sampling-Time**: 4 chains × 2000 draws × 500 obs = ~30s pro Strategie-Set. Mit NumPyro-Backend (`nuts_sampler="numpyro"`) ~5s. Empfehlung: NumPyro-Backend default machen.
- **PyMC-Größe**: 500 MB Dependency. Soft-optional verpflichtend.

## Was du nach 12-18h hast

- `bayesian_sharpe_posterior(returns)` → `P(true_sharpe > 0.5)`, HDI, Konvergenz-Diagnostics
- `hierarchical_strategy_comparison({A: ..., B: ..., C: ...})` → pairwise Wahrscheinlichkeiten + automatische Shrinkage
- Daily-QA-Report flaggt low-confidence "winners"

---

# 3. Redis Streams Phase 1 (Side-Channel)

**Aufwand:** 8-12h  
**Tier:** 2 — hoher Nutzen, mittlerer Aufwand  
**Risiko:** niedrig (additiv, kein Pipeline-Bruch)

## Was du heute hast

`pipeline/trading_cycle_v2.py` ist monolithisch synchron. Phase 11 schreibt Metrics zu Prometheus-File.

## Was Phase 1 dir gibt

Ein **read-only Side-Channel** — alle Pipeline-Outputs werden zusätzlich in Redis Streams publiziert. Der Trading-Cycle bleibt unverändert. Du kannst aber:
- Live-Dashboards (Grafana via Redis-Datasource) bauen
- Externe Consumer andocken (Slack-Bot für Alerts, separate Notebook für Analyse)
- Replay-Capability gewinnen (jeder Phase-Output ist im Stream)

## Schritt-für-Schritt-Plan

### Schritt 1 — `pipeline/event_bus.py` (neu, ~140 LOC, 3h)

```python
"""Lightweight event bus on top of Redis Streams.

Phase 1: side-channel only. Trading cycle pushes events; consumers read.
No business logic is moved into the bus.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from typing import Any, Iterator

logger = logging.getLogger(__name__)

try:
    import redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False


# Stream name conventions (versioned for forward-compatibility)
STREAMS = {
    "cycle:phase_complete": "v1:cycle:phase_complete",
    "cycle:signals": "v1:cycle:signals",
    "cycle:orders_filtered": "v1:cycle:orders_filtered",
    "cycle:rejections": "v1:cycle:rejections",
    "cycle:metrics": "v1:cycle:metrics",
    "intel:news_classified": "v1:intel:news_classified",
    "risk:kill_switch": "v1:risk:kill_switch",
    "ops:drift_alert": "v1:ops:drift_alert",
}


class EventBus:
    """Async-friendly Redis Stream publisher.
    
    Designed to never break the trading cycle:
    - Lazy connection
    - All publish() calls swallow exceptions (logged)
    - No-op if HAS_REDIS=False or connection fails
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        max_stream_length: int = 100_000,
        enabled: bool = True,
    ):
        self.host = host
        self.port = port
        self.db = db
        self.max_stream_length = max_stream_length
        self.enabled = enabled and HAS_REDIS
        self._client: Any = None
        self._connection_failed_at: float | None = None
        self._reconnect_after_seconds: float = 30.0
    
    def _get_client(self) -> Any:
        if not self.enabled:
            return None
        if self._connection_failed_at is not None:
            if time.time() - self._connection_failed_at < self._reconnect_after_seconds:
                return None
            self._connection_failed_at = None
        if self._client is None:
            try:
                self._client = redis.Redis(
                    host=self.host, port=self.port, db=self.db,
                    decode_responses=True, socket_timeout=2.0,
                )
                self._client.ping()
            except Exception as e:
                logger.warning("Redis connection failed: %s", e)
                self._connection_failed_at = time.time()
                self._client = None
        return self._client
    
    def publish(self, stream_key: str, data: dict[str, Any]) -> str | None:
        """Publish to a stream. Returns message ID, or None on failure."""
        client = self._get_client()
        if client is None:
            return None
        try:
            stream_name = STREAMS.get(stream_key, stream_key)
            payload = {
                k: json.dumps(v, default=str) if not isinstance(v, str) else v
                for k, v in data.items()
            }
            payload["_published_at"] = str(time.time())
            return client.xadd(
                stream_name, payload, maxlen=self.max_stream_length, approximate=True,
            )
        except Exception as e:
            logger.warning("publish to %s failed: %s", stream_key, e)
            return None


# Module-level singleton (matches your other ops modules)
_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    global _bus
    if _bus is None:
        import os
        _bus = EventBus(
            host=os.environ.get("REDIS_HOST", "localhost"),
            port=int(os.environ.get("REDIS_PORT", "6379")),
            enabled=os.environ.get("EVENT_BUS_ENABLED", "false").lower() == "true",
        )
    return _bus


@contextmanager
def streamed_phase(phase_name: str, ctx: dict[str, Any] | None = None) -> Iterator[None]:
    """Context manager: emits start/end events around a phase."""
    bus = get_event_bus()
    start_id = bus.publish("cycle:phase_complete", {
        "phase": phase_name, "event": "start", "ctx": ctx or {},
    })
    t0 = time.time()
    try:
        yield
        bus.publish("cycle:phase_complete", {
            "phase": phase_name, "event": "complete",
            "duration_ms": int((time.time() - t0) * 1000),
            "start_id": start_id,
        })
    except Exception as e:
        bus.publish("cycle:phase_complete", {
            "phase": phase_name, "event": "error",
            "duration_ms": int((time.time() - t0) * 1000),
            "error": str(e),
        })
        raise
```

### Schritt 2 — Pipeline-Integration (~50 LOC, 2h)

In `pipeline/trading_cycle_v2.py`, an gezielten Stellen, **niemals blocking**:

```python
from src.assembled_core.pipeline.event_bus import get_event_bus, streamed_phase

# In jeder Phase-Funktion:
def check_risk(result, ...):
    with streamed_phase("check_risk", {"strategy": ctx.strategy_name}):
        # ... existing logic ...
        result.meta["rejection_counts"] = _rej_counts
        # NEW: emit if any rejections
        if _rej_counts:
            get_event_bus().publish("cycle:rejections", {
                "strategy": ctx.strategy_name,
                "counts": _rej_counts,
                "timestamp": str(pd.Timestamp.now(tz="UTC")),
            })
        return result

# After Phase 11 (KPI export):
get_event_bus().publish("cycle:metrics", {
    "strategy": ctx.strategy_name,
    "metrics": kpi_metrics,
    "drift_psi": result.meta.get("drift_monitor", {}).get("max_psi", 0),
})
```

### Schritt 3 — Tests (~80 LOC, 1h)

```python
# tests/test_event_bus.py
import os
import pytest
from unittest.mock import patch, MagicMock

from src.assembled_core.pipeline.event_bus import EventBus, streamed_phase


class TestEventBus:
    def test_disabled_publish_returns_none(self):
        bus = EventBus(enabled=False)
        assert bus.publish("cycle:metrics", {"foo": "bar"}) is None
    
    def test_redis_unavailable_no_exception(self):
        bus = EventBus(host="nonexistent.local", port=9999, enabled=True)
        # Should silently fail, not raise
        result = bus.publish("cycle:metrics", {"foo": "bar"})
        assert result is None
    
    def test_publish_serializes_complex_types(self):
        bus = EventBus(enabled=True)
        bus._client = MagicMock()
        bus._client.xadd.return_value = "1234-0"
        
        result = bus.publish("cycle:metrics", {
            "strategy": "test",
            "counts": {"pdt": 3, "fat_finger": 1},
            "timestamp": "2026-04-29",
        })
        
        assert result == "1234-0"
        call_args = bus._client.xadd.call_args
        payload = call_args[0][1]
        assert payload["strategy"] == "test"
        # Dict was JSON-serialized
        import json
        assert json.loads(payload["counts"]) == {"pdt": 3, "fat_finger": 1}
    
    def test_streamed_phase_emits_on_exception(self):
        with patch(
            "src.assembled_core.pipeline.event_bus.get_event_bus",
        ) as mock_bus:
            mock = MagicMock()
            mock_bus.return_value = mock
            
            with pytest.raises(ValueError):
                with streamed_phase("test_phase"):
                    raise ValueError("boom")
            
            # Should have emitted start + error events
            calls = [c[0][0] for c in mock.publish.call_args_list]
            assert "cycle:phase_complete" in calls
            assert any(
                "error" in str(c[0][1]) for c in mock.publish.call_args_list
            )
```

### Schritt 4 — Docker-Compose für lokale Dev (~30 LOC, 0.5h)

```yaml
# docker-compose.dev.yml (neu oder erweitern)
services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    command: redis-server --appendonly yes --maxmemory 1gb --maxmemory-policy allkeys-lru

volumes:
  redis-data:
```

### Schritt 5 — README-Section (~20 LOC, 0.5h)

```markdown
## Event Bus (Phase 1, optional)

Trading cycle emits events to Redis Streams when `EVENT_BUS_ENABLED=true`.
This is a side-channel only — pipeline runs identically with or without it.

Start dev Redis:
    docker-compose -f docker-compose.dev.yml up -d redis

Enable in env:
    export EVENT_BUS_ENABLED=true REDIS_HOST=localhost

Inspect a stream:
    redis-cli XREAD COUNT 10 STREAMS v1:cycle:metrics 0
```

## Tests-to-Run

```bash
docker-compose -f docker-compose.dev.yml up -d redis
EVENT_BUS_ENABLED=true pytest tests/test_event_bus.py
pytest tests/test_trading_cycle_v2.py  # no regression
```

## Risiken

- **Zero impact on Trading Cycle**: das Design schluckt alle Exceptions, hat 2-s-Socket-Timeout, fällt 30-s-Connection-Backoff bei Fehlern. Wenn Redis weg ist, läuft die Pipeline trotzdem.
- **Daten-Privacy**: Streams enthalten Strategie-Outputs. Wenn Redis netzwerkbasiert ist, **TLS und AUTH einrichten**. Für Localhost-only OK.

## Was du nach 8-12h hast

- Pipeline emittiert strukturierte Events parallel zum normalen Lauf
- Redis als optionaler Backbone für spätere Erweiterung
- Foundation für Phase 2 (News-Pipeline auf Streams), Phase 3 (Execution-Path), Phase 4 (full migration)

---

# 4. QuestDB als Tick-Store

**Aufwand:** 8-12h  
**Tier:** 2  
**Risiko:** mittel (neue Infrastructure-Komponente)

## Wann das wichtig wird

Sobald du **eine** der folgenden Dinge ernsthaft willst:
- VPIN auf echten Trade-by-Trade-Daten
- Spread/Depth-Features für Mikrostruktur-Signale
- Order-Book-Imbalance (siehe Item §11)
- RL-Execution-Training (siehe §5)

Heute hast du `data/feature_store.py` (DuckDB+Parquet). Das ist gut für EOD-Bars. Für Tick-Data ist es zu langsam (siehe Benchmark in v4).

## Schritt-für-Schritt-Plan

### Schritt 1 — Docker-Compose für QuestDB (~20 LOC, 0.5h)

```yaml
# docker-compose.dev.yml (extend)
services:
  questdb:
    image: questdb/questdb:latest
    ports:
      - "9000:9000"  # Web Console
      - "8812:8812"  # PG-wire (for psycopg2 / pandas)
      - "9009:9009"  # InfluxDB Line Protocol (for ingest)
    volumes:
      - questdb-data:/var/lib/questdb
    environment:
      QDB_PG_USER: admin
      QDB_PG_PASSWORD: quest

volumes:
  questdb-data:
```

### Schritt 2 — `data/tick_store.py` (neu, ~200 LOC, 4h)

```python
"""High-performance tick storage backed by QuestDB.

Used for sub-second-resolution data:
- VPIN bucket aggregation (trade-by-trade)
- Microstructure features (spread, depth)
- RL execution training data

Talks PG-wire — works with any psycopg2-compatible client.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

import pandas as pd

logger = logging.getLogger(__name__)

try:
    import psycopg2
    from psycopg2.extras import execute_batch
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False


# Schema templates
TRADES_SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    timestamp TIMESTAMP,
    symbol SYMBOL CAPACITY 5000 CACHE,
    price DOUBLE,
    volume LONG,
    side SYMBOL CAPACITY 4 CACHE
) TIMESTAMP(timestamp) PARTITION BY DAY WAL;
"""

QUOTES_SCHEMA = """
CREATE TABLE IF NOT EXISTS quotes (
    timestamp TIMESTAMP,
    symbol SYMBOL CAPACITY 5000 CACHE,
    bid_price DOUBLE,
    ask_price DOUBLE,
    bid_size LONG,
    ask_size LONG
) TIMESTAMP(timestamp) PARTITION BY DAY WAL;
"""


class QuestDBTickStore:
    """PG-wire client for QuestDB."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8812,
        user: str = "admin",
        password: str = "quest",
        dbname: str = "qdb",
    ):
        if not HAS_PSYCOPG2:
            raise ImportError("psycopg2-binary required for QuestDB integration.")
        self.dsn = (
            f"host={host} port={port} user={user} "
            f"password={password} dbname={dbname}"
        )
        self._ensure_schema()
    
    def _ensure_schema(self) -> None:
        with self._connection() as conn:
            cur = conn.cursor()
            cur.execute(TRADES_SCHEMA)
            cur.execute(QUOTES_SCHEMA)
            conn.commit()
    
    @contextmanager
    def _connection(self) -> Iterator:
        conn = psycopg2.connect(self.dsn)
        try:
            yield conn
        finally:
            conn.close()
    
    def ingest_trades(self, trades: pd.DataFrame) -> int:
        """Bulk-insert trades. Schema: timestamp, symbol, price, volume, side."""
        required = {"timestamp", "symbol", "price", "volume", "side"}
        if not required.issubset(trades.columns):
            raise ValueError(f"Missing columns: {required - set(trades.columns)}")
        
        rows = trades[list(required)].itertuples(index=False, name=None)
        with self._connection() as conn:
            cur = conn.cursor()
            execute_batch(
                cur,
                "INSERT INTO trades (timestamp, symbol, price, volume, side) "
                "VALUES (%s, %s, %s, %s, %s)",
                rows, page_size=1000,
            )
            conn.commit()
        return len(trades)
    
    def get_5min_bars(
        self, symbol: str, start: str, end: str
    ) -> pd.DataFrame:
        """OHLCV bars at 5-minute resolution from raw trades."""
        sql = """
            SELECT
                timestamp,
                first(price) AS open,
                max(price) AS high,
                min(price) AS low,
                last(price) AS close,
                sum(volume) AS volume,
                count(*) AS trade_count
            FROM trades
            WHERE symbol = %s
              AND timestamp BETWEEN %s AND %s
            SAMPLE BY 5m ALIGN TO CALENDAR;
        """
        with self._connection() as conn:
            return pd.read_sql_query(sql, conn, params=(symbol, start, end))
    
    def stream_trades_for_vpin(
        self, symbol: str, start: str, end: str,
    ) -> Iterator[dict]:
        """Stream trades chronologically — for VPIN bucket aggregation."""
        sql = """
            SELECT timestamp, price, volume, side
            FROM trades
            WHERE symbol = %s AND timestamp BETWEEN %s AND %s
            ORDER BY timestamp ASC;
        """
        with self._connection() as conn:
            cur = conn.cursor("server_side_cursor")
            cur.execute(sql, (symbol, start, end))
            for row in cur:
                yield {
                    "timestamp": row[0], "price": row[1],
                    "volume": row[2], "side": row[3],
                }
    
    def get_volume_imbalance(
        self, symbol: str, start: str, end: str, bucket_seconds: int = 60,
    ) -> pd.DataFrame:
        """Buy-sell volume imbalance per N-second bucket."""
        sql = f"""
            SELECT
                timestamp,
                sum(CASE WHEN side = 'buy' THEN volume ELSE 0 END) AS buy_vol,
                sum(CASE WHEN side = 'sell' THEN volume ELSE 0 END) AS sell_vol,
                count(*) AS trades
            FROM trades
            WHERE symbol = %s AND timestamp BETWEEN %s AND %s
            SAMPLE BY {bucket_seconds}s ALIGN TO CALENDAR;
        """
        with self._connection() as conn:
            df = pd.read_sql_query(sql, conn, params=(symbol, start, end))
            df["imbalance"] = (df["buy_vol"] - df["sell_vol"]) / (
                df["buy_vol"] + df["sell_vol"] + 1
            )
            return df
```

### Schritt 3 — Migration eines Pilot-Use-Cases: VPIN (~50 LOC, 2h)

In `qa/vpin.py` (existiert bereits):

```python
def compute_vpin_from_questdb(
    symbol: str,
    start: str,
    end: str,
    n_buckets: int = 50,
    bucket_size_pct_adv: float = 0.01,
) -> pd.Series:
    """VPIN computed directly from QuestDB tick data.
    
    Faster than the Pandas version (data stays in DB).
    """
    from src.assembled_core.data.tick_store import QuestDBTickStore
    
    store = QuestDBTickStore()
    # Get average daily volume for bucket-size calculation
    bars = store.get_5min_bars(symbol, start, end)
    if bars.empty:
        return pd.Series(dtype=float)
    avg_daily_volume = float(bars["volume"].sum() / max(1, bars["timestamp"].dt.date.nunique()))
    
    # Existing VPIN logic from qa/vpin.py, but stream from QuestDB:
    trades = list(store.stream_trades_for_vpin(symbol, start, end))
    return _compute_vpin_from_trade_stream(
        trades, avg_daily_volume, n_buckets, bucket_size_pct_adv,
    )
```

### Schritt 4 — Tests (~80 LOC, 1.5h)

Tests sind tricky weil QuestDB benötigt ist. Pattern: marker `requires_questdb`, skip wenn Container nicht läuft.

```python
# tests/test_tick_store.py
import pandas as pd
import pytest

requires_questdb = pytest.mark.skipif(
    True,  # Replace with actual env-check
    reason="Requires QuestDB running on localhost:8812",
)


@requires_questdb
class TestQuestDBTickStore:
    def test_ingest_and_retrieve(self):
        from src.assembled_core.data.tick_store import QuestDBTickStore
        store = QuestDBTickStore()
        
        trades = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01 09:30:00", periods=100, freq="100ms"),
            "symbol": ["SPY"] * 100,
            "price": [400 + i*0.01 for i in range(100)],
            "volume": [100] * 100,
            "side": ["buy", "sell"] * 50,
        })
        n = store.ingest_trades(trades)
        assert n == 100
        
        bars = store.get_5min_bars("SPY", "2024-01-01", "2024-01-02")
        assert not bars.empty
        assert bars["trade_count"].sum() == 100
```

## Risiken

- **Daten-Volumen**: 1 Symbol × 1 Tag intraday = ~1-10 GB Tick-Data je nach Aktivität. Mit 30 Symbolen × 5 Jahren = ~5 TB. Plan-Storage entsprechend, oder beschränke auf Tier-1-Symbole.
- **Daten-Quelle**: du brauchst eine Tick-Daten-Source. Polygon.io (du hast `polygon_source.py`) liefert Trades. Kosten: Starter $99/Monat für 5 Jahre Historie.

## Was du nach 8-12h hast

- QuestDB als zweiten Datastore neben deinem `feature_store.py`
- 25-100x schnellere VPIN-Berechnung
- Foundation für Microstructure-Features und RL-Training

---

# 5. RL-Execution-Policy

**Aufwand:** 40-60h  
**Tier:** 3 — strategisch  
**Risiko:** hoch (Reward-Hacking, Sim-to-Real-Gap)

**Voraussetzung:** §4 (QuestDB) muss da sein für Training-Daten.

## Schritt-für-Schritt-Plan

### Schritt 1 — Soft-Optional Dependencies (~10 LOC, 0.5h)

```toml
# pyproject.toml
[project.optional-dependencies]
rl = [
  "stable-baselines3>=2.3",
  "gymnasium>=0.29",
  "torch>=2.0",
]
```

### Schritt 2 — `execution/rl_environment.py` (~250 LOC, 12h)

Vollständige Custom-Env wie in v4 §2 skizziert. Wichtige Details:
- Reward = -implementation_shortfall_bps + completion_bonus
- State-Vector: [spread_bps, vol, depth, qty_left_pct, time_left_pct, regime_id]
- Action: continuous [0,1] = Anteil-der-Restmenge

### Schritt 3 — `scripts/train_rl_execution.py` (~150 LOC, 8h)

PPO-Training auf historischen Mikrostruktur-Daten aus QuestDB. Pattern:
- Train-Period: 2020-2023
- Validation: 2024 H1
- Test: 2024 H2 (held-out)
- Track: episode-mean-reward, completion-rate

### Schritt 4 — `execution/rl_execution.py` (~150 LOC, 6h)

Inference-Wrapper, der Stable-Baselines3-Modell lädt und in `execution_router.py` einklinkbar ist.

### Schritt 5 — A/B-Backtest gegen Almgren-Chriss (~200 LOC, 8h)

Kritisches Validierungs-Skript: 1000 Test-Episoden, beide Algos auf gleichen Markt-Snapshots, paired t-test auf Slippage-Difference.

### Schritt 6 — Production-Wiring + Drift-Detection (~150 LOC, 6h)

- Hook in `execution_router.py` (mit feature flag)
- `ops/drift_monitor.py` watcht RL-Action-Distribution gegen Trainings-Distribution
- Auto-Fallback zu AC bei Drift > Threshold

### Schritt 7 — Tests + Doku (~100 LOC, 4h)

## Risiken

1. **Reward-Hacking**: präzise Tuning des completion-bonus nötig. Sonst lernt PPO immer 1% zu executen.
2. **Sim-to-Real**: dein Trading bewegt den Markt — historische Daten erfassen das nicht. Mitigation: einfache Impact-Korrektur in der Env.
3. **Distributional Shift**: 2018-2024-Trainings-Daten ≠ 2025-Marktbedingungen. Auto-Fallback auf AC ist Pflicht.

## Was du nach 40-60h hast

Kontextabhängige Execution-Policy, die in Out-of-Sample-Tests konsistent 1-3 bps weniger Slippage als Almgren-Chriss liefert. Bei 100k$-Trades = 100-300$ pro Trade.

---

# 6. GNN für Stock-Prediction

**Aufwand:** 30-50h  
**Tier:** 4 — Research-Pfad  
**Risiko:** sehr hoch (niedrige garantierte Wirkung)

Skip-Empfehlung sofern du nicht spezifisches Interesse an Cross-Asset-Propagation hast. Wenn ja: PyTorch Geometric Temporal, baseline T-GCN auf 30-Symbol-Universe, Graph aus Sector + Top-Korrelation. Detail in v4 §5. Pflicht-Begleiter: `qa/leakage_analyzer.py` (Cross-Sectional-Leakage ist subtil).

---

# 7. LLM-RAG für News-Reasoning

**Aufwand:** 16-24h  
**Tier:** 2-3  
**Risiko:** mittel (LLM-API-Kosten, Halluzinations-Risiko)

## Was du heute hast

`intel/news_*` Pipeline mit ~30 Modulen — Klassifizierung, Dedup, Entity-Mapping, Bayesian-Confidence. Sehr breit. Aber: keine **freie-Form-Reasoning** über News-Themen.

Beispiel der Lücke: Du kannst heute eine News klassifizieren als "geo:taiwan:positive" — aber du kannst nicht fragen "Was hat sich an der Taiwan-Lage in den letzten 7 Tagen verändert, was sind die 3 wichtigsten neuen Entwicklungen, und wie verbinden sie sich mit Halbleiter-Lieferketten?"

## State of the Art

- **Vector-DBs**: Qdrant (Apache 2.0), Weaviate (BSD-3), Chroma (Apache 2.0)
- **Embedding-Models**: sentence-transformers (Apache 2.0), OpenAI/Anthropic embed APIs
- **RAG-Frameworks**: LlamaIndex (MIT), LangChain (MIT) — aber fragil-tool-soup
- **Pure-Python alternative**: simple cosine-similarity über Qdrant + direct LLM-call

## Plan

```python
# intel/news_rag.py (neu)

import qdrant_client
from sentence_transformers import SentenceTransformer

class NewsRAG:
    """Retrieval-augmented news reasoning over the last N days of news."""
    
    def __init__(self, collection="news_v1"):
        self.qdrant = qdrant_client.QdrantClient(host="localhost", port=6333)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim
        self.collection = collection
    
    def index_news(self, news_df: pd.DataFrame):
        """Index a batch of classified news items."""
        embeddings = self.embedder.encode(news_df["headline"].tolist())
        points = [
            qdrant_client.models.PointStruct(
                id=row["id"],
                vector=emb.tolist(),
                payload={
                    "headline": row["headline"],
                    "ts": row["ts"].isoformat(),
                    "tier": row["tier"],
                    "entities": row["entities"],
                    "sentiment": row["sentiment"],
                },
            )
            for emb, (_, row) in zip(embeddings, news_df.iterrows())
        ]
        self.qdrant.upsert(self.collection, points)
    
    def reason(self, query: str, days_back: int = 7, top_k: int = 20) -> dict:
        """Retrieve top-k relevant news + ask LLM to reason."""
        query_emb = self.embedder.encode(query)
        cutoff = (pd.Timestamp.now() - pd.Timedelta(days=days_back)).isoformat()
        
        results = self.qdrant.search(
            collection_name=self.collection,
            query_vector=query_emb.tolist(),
            query_filter=qdrant_client.models.Filter(
                must=[
                    qdrant_client.models.FieldCondition(
                        key="ts", range=qdrant_client.models.Range(gte=cutoff)
                    )
                ]
            ),
            limit=top_k,
        )
        
        context = "\n\n".join([
            f"[{r.payload['ts']}, {r.payload['tier']}, score={r.score:.2f}] "
            f"{r.payload['headline']}"
            for r in results
        ])
        
        # Call LLM (Claude API or local) — keep prompt deterministic
        prompt = f"""You are a financial news analyst. Based on the following news from the last {days_back} days, answer the user's question. Be specific, cite headlines, and flag uncertainty.

CONTEXT:
{context}

USER QUESTION:
{query}

Respond in JSON: {{"summary": str, "key_developments": [str], "uncertainty_flags": [str], "supporting_headlines": [str]}}"""
        
        response = call_llm(prompt)  # your LLM client
        return {"context": context, "answer": response, "n_retrieved": len(results)}
```

**Use-Cases:**
- Wöchentlicher Digest: "Was hat sich an [Top-5 your geo-watch-list] verändert?"
- Pre-trade Sanity-Check: "Was wissen wir Aktuelles über NVDA?"
- Earnings-Prep: "Was sind die 3 wichtigsten Themen vor dem AAPL-Earnings-Call diese Woche?"

## Risiken

- **API-Kosten**: 1-2$ pro Reasoning-Call bei Claude/GPT-4. Für tägliche Wochen-Digests OK.
- **Halluzinationen**: explizit den LLM zwingen, nur in CONTEXT-vorhandene Headlines zu zitieren.
- **Embedding-Drift**: bei sentence-transformers Modell-Wechsel komplettes Re-Indexing nötig.

## Was du nach 16-24h hast

Eine semantische Suche + LLM-Reasoning über deinen News-Korpus, eingebettet in deinen Daily-QA-Report.

---

# 8. Risk-Parity Vol-Targeting auf Strategy-Level

**Aufwand:** 8-12h  
**Tier:** 2  
**Risiko:** niedrig

## Lücke

Du hast `signals/multifactor_long_short`, `multifactor_v1`, `multifactor_v2`, `ema_trend_v0`, `lppls_crash`, etc. **Wie werden die Outputs gewichtet?** Vermutlich gleichgewichtet oder nach hardcoded weights.

State-of-the-art: **inverse-vol weighting** auf rolling Strategy-PnL-Vol.

```python
# portfolio/strategy_allocator.py (neu)

def inverse_vol_weights(
    pnl_per_strategy: dict[str, pd.Series],
    lookback_days: int = 60,
    target_vol_annual: float = 0.10,
) -> dict[str, float]:
    """Inverse-vol weights, scaled to target portfolio vol."""
    vols = {
        name: pnl.tail(lookback_days).std() * np.sqrt(252)
        for name, pnl in pnl_per_strategy.items()
    }
    vols = {k: v for k, v in vols.items() if v > 0}
    
    raw_weights = {k: 1.0 / v for k, v in vols.items()}
    total = sum(raw_weights.values())
    normalized = {k: w / total for k, w in raw_weights.items()}
    
    # Scale to target vol (assuming uncorrelated — pessimistic)
    portfolio_vol = np.sqrt(sum(
        (normalized[k] * vols[k]) ** 2 for k in normalized
    ))
    scale = target_vol_annual / portfolio_vol if portfolio_vol > 0 else 1.0
    
    return {k: w * scale for k, w in normalized.items()}
```

Plus: extension mit Korrelations-Korrektur (HRP — Hierarchical Risk Parity, du hast das in `portfolio/hierarchical_risk_parity.py`).

---

# 9. Walk-Forward-Optimization mit Optuna

**Aufwand:** 10-14h  
**Tier:** 2  
**Risiko:** mittel (Overfitting)

## Lücke

Du hast `qa/walk_forward.py` (1277 LOC). Wie werden Hyperparameter pro Walk-Forward-Window getuned? Wenn Grid-Search → langsam und zufällig.

## Plan

Optuna-Integration mit Walk-Forward-spezifischen Sampler:

```python
# qa/walk_forward_optuna.py

import optuna

def optimize_strategy_per_window(
    strategy_fn, returns, train_window, test_window, n_trials=50,
):
    """For each Walk-Forward window, find best hyperparameters.
    
    Persists trial database for reproducibility + analysis.
    """
    storage = "sqlite:///walk_forward_studies.db"
    
    results = []
    for i, (train_idx, test_idx) in enumerate(walk_forward_splits(returns)):
        study = optuna.create_study(
            study_name=f"wf_window_{i}",
            storage=storage,
            load_if_exists=True,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        
        def objective(trial):
            params = {
                "lookback": trial.suggest_int("lookback", 5, 100),
                "smoothing": trial.suggest_float("smoothing", 0.1, 0.9),
                # ... your hyperparams
            }
            train_returns = returns.iloc[train_idx]
            sharpe = strategy_fn(train_returns, **params).sharpe()
            return sharpe
        
        study.optimize(objective, n_trials=n_trials)
        
        # Test on held-out
        test_returns = returns.iloc[test_idx]
        test_sharpe = strategy_fn(test_returns, **study.best_params).sharpe()
        
        results.append({
            "window": i,
            "best_params": study.best_params,
            "train_sharpe": study.best_value,
            "test_sharpe": test_sharpe,
        })
    
    return pd.DataFrame(results)
```

Plus: Studies sind in SQLite gespeichert — du kannst nachträglich analysieren, welche Hyperparameter über Zeit stabil sind.

---

# 10. Differential Privacy für Strategie-Sharing

**Aufwand:** 20-30h  
**Tier:** 4 — spekulativ  
**Risiko:** hoch (Komplexität)

Skip falls du nie Strategien mit Externen teilst. Falls du das jemals willst (z.B. Performance-Track-Record an Investor zeigen ohne genaue Trades zu offenbaren): differential-private Bootstrapping mit Laplace-Noise auf Sharpe/Drawdown-Metriken. `opendp` (BSD-3) ist die State-of-the-Art-Library.

---

# 11. Order-Book-Imbalance-Features

**Aufwand:** 8-12h  
**Tier:** 2  
**Risiko:** niedrig

**Voraussetzung:** §4 (QuestDB).

## Was

Order-Book-Imbalance = (bid_size - ask_size) / (bid_size + ask_size). Bewährter Mikrostruktur-Predictor (Cont, Stoikov, Kukanov 2014).

```python
# features/order_book_imbalance.py

def order_book_imbalance(quotes: pd.DataFrame, levels: int = 5) -> pd.Series:
    """Multi-level imbalance signal from L2 quotes."""
    bid_total = sum(quotes[f"bid_size_{i}"] for i in range(1, levels+1))
    ask_total = sum(quotes[f"ask_size_{i}"] for i in range(1, levels+1))
    return (bid_total - ask_total) / (bid_total + ask_total + 1)
```

Plus: rolling-mean über kurze Windows als Predictor für Mid-Move in nächsten 5-30 Sekunden.

---

# 12. Regime-Conditional-Strategy-Allokation

**Aufwand:** 12-18h  
**Tier:** 2  
**Risiko:** mittel (Regime-Misclassification)

## Lücke

Du hast `ml/regime_hmm.py`, `risk/regime_hmm.py`, `risk/regime_models.py`. Werden Strategien **regime-conditional** allociert? Vermutlich nicht.

## Plan

```python
# portfolio/regime_conditional_allocator.py

def regime_conditional_weights(
    current_regime: int,
    historical_pnl_per_strategy_per_regime: dict[tuple[str, int], pd.Series],
    base_weights: dict[str, float],
) -> dict[str, float]:
    """Adjust strategy weights based on current regime's historical performance.
    
    If a strategy historically underperforms in regime 2 (e.g., crisis), 
    reduce its weight when current_regime == 2.
    """
    adjusted = {}
    for strategy, base_w in base_weights.items():
        key = (strategy, current_regime)
        if key not in historical_pnl_per_strategy_per_regime:
            adjusted[strategy] = base_w
            continue
        regime_sharpe = (
            historical_pnl_per_strategy_per_regime[key].mean()
            / historical_pnl_per_strategy_per_regime[key].std()
            * np.sqrt(252)
        )
        # Multiplier: 1.0 if regime_sharpe == 0.5, 2x if == 2.0, 0.5x if negative
        multiplier = max(0.0, min(2.0, regime_sharpe / 0.5))
        adjusted[strategy] = base_w * multiplier
    
    # Renormalize
    total = sum(adjusted.values())
    return {k: w / total for k, w in adjusted.items()}
```

---

# 13. Kalshi als zweite Prediction-Market-Source

**Aufwand:** 4-6h  
**Tier:** 1 (nach Polymarket)  
**Risiko:** niedrig

**Voraussetzung:** §1 (Polymarket).

## Warum

Kalshi ist CFTC-reguliert (US-legal seit Beginn), während Polymarket erst Januar 2026 zurück nach US ist. Komplementäre Coverage. Kalshi hat stärkere Coverage von US-Domestic-Events (Fed-Decisions, US-Sports, US-Election-Internals).

## Plan

Praktisch identische Struktur wie Polymarket-Source. **`SpartanLabsXyz/simmer-sdk`** (siehe v4 §4) bietet ein unified-API, das beide gleichzeitig abdeckt — wenn du das nutzt, ist Kalshi quasi "free" nach Polymarket-Implementation.

---

# Master-Adoption-Plan

Sortiert nach **(Wirkung × Anwendbarkeit) / Aufwand**:

| Reihenfolge | Item | Aufwand | Sektion |
|---|---|---|---|
| **1** | Polymarket-Source | 6-10h | §1 |
| **2** | Risk-Parity Strategy-Allocation | 8-12h | §8 |
| **3** | Order-Book-Imbalance (vorerst auf Bar-Level, später Tick) | 8-12h | §11 |
| **4** | Bayesian Sharpe + Strategy-Comparison | 12-18h | §2 |
| **5** | Redis Streams Phase 1 | 8-12h | §3 |
| **6** | Regime-Conditional-Allocation | 12-18h | §12 |
| **7** | Walk-Forward + Optuna | 10-14h | §9 |
| **8** | Kalshi (Quick Win nach §1) | 4-6h | §13 |
| **9** | QuestDB | 8-12h | §4 |
| **10** | LLM-RAG | 16-24h | §7 |
| **11** | RL-Execution | 40-60h | §5 |
| **12** | GNN | 30-50h | §6 |
| **13** | Differential Privacy | 20-30h | §10 |

---

## Empfohlene 4-Wochen-Roadmap

**Woche 1 (~30h):** §1 + §13 (Polymarket + Kalshi) + §8 (Risk-Parity).  
Du hast danach 2 neue Daten-Quellen + saubere Strategy-Allocation.

**Woche 2 (~25h):** §2 (Bayesian Sharpe) + §11 (Order-Book-Imbalance auf Bar-Level).  
Du hast danach probabilistische Strategy-Comparison + ein Mikrostruktur-Feature.

**Woche 3 (~25h):** §3 (Redis Streams) + §12 (Regime-Allocation).  
Du hast danach Streaming-Foundation + regime-bewusste Allokation.

**Woche 4 (~30h):** §9 (Walk-Forward + Optuna) + §4 (QuestDB).  
Du hast danach systematisches Hyperparameter-Tuning + Tick-Storage-Foundation.

**Nach 4 Wochen:** Wenn QuestDB läuft und Polymarket Daten liefert, ist die Foundation für §5 (RL) und §7 (LLM-RAG) da. Diese sind 40+h-Investments, sollten erst angegangen werden wenn die schnelleren Wins durch sind.

---

## Wenn du sofort anfangen willst

Sag mir welches Item — ich starte mit den Diffs/Patches und committe lokal wie damals bei den Validation-Fixes. Mein Vorschlag: **§1 (Polymarket)** zuerst, weil:

1. Niedrigster Aufwand mit höchstem konkreten Wert
2. Patzt 1:1 in dein bestehendes `data/sources/`-Pattern
3. Keine neuen Dependencies (nur `httpx`, hast du schon)
4. Direkter Hook in `risk/georisk_overlay.py` und `intel/bayesian_confidence.py`
5. 6-10h Aufwand, danach hast du eine echte neue Datenquelle live
