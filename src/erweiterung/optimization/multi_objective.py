"""Multi-Objective-Optimization: Pareto-Frontier + NSGA-II-Lite.

Theorie
-------
In Portfolio-Optimization gibt es selten EINE optimale Lösung. Multi-Objective
(z.B. Maximize-Return + Minimize-Risk + Minimize-Turnover) liefert eine
Pareto-Frontier nicht-dominierter Lösungen.

Pareto-Dominance: A dominiert B wenn A in jedem Objective ≥ B und in mindestens
einem > B.

Implementation
--------------
- Pareto-Frontier-Extraction in O(n²) (für n < 1000 OK).
- NSGA-II-Lite: Non-dominated-Sort + Crowding-Distance.

Reference
---------
- Deb, K. et al. (2002). A fast and elitist multiobjective genetic algorithm:
  NSGA-II. *IEEE Trans. Evol. Comp.* 6.
"""

from __future__ import annotations

import numpy as np


def is_dominated(point: np.ndarray, points: np.ndarray, maximize: bool = True) -> bool:
    """True wenn ``point`` von mindestens einem in ``points`` dominiert wird.

    Args:
        point: 1-D objective-array.
        points: (n, m) array of other objectives.
        maximize: ob höhere Werte besser sind.

    Returns:
        bool.
    """
    if maximize:
        dominated = (points >= point).all(axis=1) & (points > point).any(axis=1)
    else:
        dominated = (points <= point).all(axis=1) & (points < point).any(axis=1)
    return bool(dominated.any())


def pareto_front_indices(objectives: np.ndarray, maximize: bool = True) -> np.ndarray:
    """Indices of Pareto-non-dominated points.

    Args:
        objectives: (n, m) array — n candidates, m objectives.
        maximize: für alle Objectives gleich.

    Returns:
        Array of indices forming Pareto-front.
    """
    n = len(objectives)
    if n == 0:
        return np.array([], dtype=int)
    keep = []
    for i in range(n):
        others = np.delete(objectives, i, axis=0)
        if not is_dominated(objectives[i], others, maximize=maximize):
            keep.append(i)
    return np.array(keep)


def non_dominated_sort(
    objectives: np.ndarray, maximize: bool = True
) -> list[list[int]]:
    """NSGA-II non-dominated sort: split into Pareto-fronts.

    Returns:
        List of fronts (lists of indices). Front 0 = best (non-dominated by any),
        Front 1 = dominated only by Front-0, etc.
    """
    n = len(objectives)
    if n == 0:
        return []
    domination_count = np.zeros(n, dtype=int)
    dominated_solutions: dict[int, list[int]] = {i: [] for i in range(n)}

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if maximize:
                # i dominates j?
                if (objectives[i] >= objectives[j]).all() and (
                    objectives[i] > objectives[j]
                ).any():
                    dominated_solutions[i].append(j)
                elif (objectives[j] >= objectives[i]).all() and (
                    objectives[j] > objectives[i]
                ).any():
                    domination_count[i] += 1
            else:
                if (objectives[i] <= objectives[j]).all() and (
                    objectives[i] < objectives[j]
                ).any():
                    dominated_solutions[i].append(j)
                elif (objectives[j] <= objectives[i]).all() and (
                    objectives[j] < objectives[i]
                ).any():
                    domination_count[i] += 1

    fronts: list[list[int]] = []
    current = [i for i in range(n) if domination_count[i] == 0]
    fronts.append(current)
    while current:
        next_front: list[int] = []
        for i in current:
            for j in dominated_solutions[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        if not next_front:
            break
        fronts.append(next_front)
        current = next_front
    return fronts


def crowding_distance(objectives: np.ndarray, indices: list[int]) -> np.ndarray:
    """NSGA-II Crowding Distance — Distanz zu Nachbarn in Objective-Space.

    Höhere Distanz = besser (diverser).
    """
    n = len(indices)
    if n <= 2:
        return np.full(n, np.inf)
    obj = objectives[indices]
    m = obj.shape[1]
    dist = np.zeros(n)
    for d in range(m):
        order = np.argsort(obj[:, d])
        dist[order[0]] = np.inf
        dist[order[-1]] = np.inf
        max_d = obj[order[-1], d]
        min_d = obj[order[0], d]
        if max_d - min_d == 0:
            continue
        for k in range(1, n - 1):
            dist[order[k]] += (obj[order[k + 1], d] - obj[order[k - 1], d]) / (
                max_d - min_d
            )
    return dist


__all__ = [
    "is_dominated",
    "pareto_front_indices",
    "non_dominated_sort",
    "crowding_distance",
]
