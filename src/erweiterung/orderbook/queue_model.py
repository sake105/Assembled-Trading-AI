"""Queue-Position-Modell für Optimal-Limit-Order-Placement.

Theorie
-------
Beim Plazieren einer Limit-Order in einer Queue (am Bid oder Ask) ist Reihenfolge
entscheidend: erste Order in Queue wird zuerst gefüllt.

Cont, Kukanov & Stoikov (2014) entwickeln Modell:
    Fill-Probability ≈ P(Volume vor mir wird abgebaut bevor Preis-Level cancelt)

Approximationen
---------------
1. **Naïve**: fill_prob = my_size / queue_total.
2. **Cont-Kukanov-Stoikov**: berücksichtigt arrival/cancel-rates.

Hier: einfache Modell für edukativ + Strategy-Benchmark.

Reference
---------
Cont, R., Kukanov, A. & Stoikov, S. (2014). The price impact of order book
events. *J. Financial Econometrics* 12.
"""

from __future__ import annotations


def naive_fill_probability(my_position_ahead: float, queue_total: float) -> float:
    """Estimate fill-prob from queue-position.

    Args:
        my_position_ahead: volume ahead of my order.
        queue_total: total queue volume at level.

    Returns:
        Approximate fill probability before level cancels / moves.
    """
    if queue_total <= 0:
        return 1.0  # alone in queue
    return float(max(0.0, 1.0 - my_position_ahead / queue_total))


def expected_fill_time(
    my_position_ahead: float, arrival_rate: float, cancel_rate: float
) -> float:
    """Expected time-to-fill given Poisson-arrival of trades + cancels.

    Args:
        my_position_ahead: queued volume ahead of order.
        arrival_rate: trades-arrivals λ_t (volume per time unit).
        cancel_rate: cancels λ_c (volume per time unit).

    Returns:
        Expected wait-time before fill.
    """
    net_rate = arrival_rate + cancel_rate
    if net_rate <= 0:
        return float("inf")
    return my_position_ahead / net_rate


def optimal_placement_choice(
    spread: float,
    my_size: float,
    queue_bid: float,
    queue_ask: float,
    tick_size: float = 0.01,
) -> str:
    """Wähle Order-Placement: Aggressiv (cross spread) vs Patient (join bid/ask).

    Heuristik
    ---------
    - Spread < 2 ticks + niedrige Queue: Aggressiv (Market-Order kostet wenig)
    - Spread ≥ 2 ticks: Patient (Limit-Order)
    - sehr lange Queue: skip-out (innerhalb spread)

    Returns:
        ``'aggressive'`` | ``'patient_join'`` | ``'patient_inside'``.
    """
    if spread <= 2 * tick_size and (queue_bid + queue_ask) / 2 > 10 * my_size:
        return "aggressive"
    if (queue_bid + queue_ask) / 2 > 100 * my_size:
        return "patient_inside"
    return "patient_join"


def adverse_selection_cost(fill_price: float, future_mid: float, side: int) -> float:
    """Cost = sign × (filled-price − future-mid) — wenn Mid danach gegen uns läuft.

    Side: +1 buy / -1 sell.
    """
    return float(side * (fill_price - future_mid))


__all__ = [
    "naive_fill_probability",
    "expected_fill_time",
    "optimal_placement_choice",
    "adverse_selection_cost",
]
