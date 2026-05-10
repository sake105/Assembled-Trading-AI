"""Genetic-Programming für Signal-Discovery.

Idee
----
Generiere zufällige Formeln aus elementaren Operatoren (`+`, `-`, `*`, `/`,
`log`, `lag`, `mean`, `std`) auf Basis-Features (returns, volume, OHLC).
Selektiere die mit höchstem out-of-sample Information-Coefficient. Mutiere/
Kreuze die besten — wiederhole.

Reference
---------
- Schmidt, M. & Lipson, H. (2009). Distilling Free-Form Natural Laws.
  *Science* 324.
- Symbolic regression for finance: Allen/Karjalainen (1999), genetic
  programming for trading rules.

**Hinweis**: Aggressive GP-Search ist hochanfällig für Backtest-Overfitting.
Daher OOS-Validation + Reality-Check zwingend.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Elementare Operatoren — als Strings, später kompiliert
PRIMITIVES_BINARY = ["+", "-", "*", "div", "min", "max"]
PRIMITIVES_UNARY = [
    "neg",
    "abs",
    "sign",
    "tanh",
    "log1p",
    "lag1",
    "lag5",
    "ma5",
    "std10",
]


@dataclass
class GPNode:
    op: str  # "leaf" or operator name
    children: list["GPNode"] = field(default_factory=list)
    leaf_name: Optional[str] = None  # for leaf nodes: feature name

    def to_str(self) -> str:
        if self.op == "leaf":
            return str(self.leaf_name)
        if len(self.children) == 1:
            return f"{self.op}({self.children[0].to_str()})"
        return f"({self.children[0].to_str()} {self.op} {self.children[1].to_str()})"


def random_tree(depth: int, features: list[str], rng: random.Random) -> GPNode:
    if depth == 0 or rng.random() < 0.3:
        return GPNode(op="leaf", leaf_name=rng.choice(features))
    # internal node
    if rng.random() < 0.5:
        op = rng.choice(PRIMITIVES_UNARY)
        return GPNode(op=op, children=[random_tree(depth - 1, features, rng)])
    op = rng.choice(PRIMITIVES_BINARY)
    return GPNode(
        op=op,
        children=[
            random_tree(depth - 1, features, rng),
            random_tree(depth - 1, features, rng),
        ],
    )


def evaluate_tree(node: GPNode, df: pd.DataFrame) -> pd.Series:
    """Evaluiere Tree auf Feature-DataFrame. Liefert Series."""
    if node.op == "leaf":
        return df[node.leaf_name].astype(float)  # type: ignore
    a = evaluate_tree(node.children[0], df)
    if len(node.children) == 1:
        if node.op == "neg":
            return -a
        if node.op == "abs":
            return a.abs()
        if node.op == "sign":
            return np.sign(a)
        if node.op == "tanh":
            return np.tanh(a)
        if node.op == "log1p":
            return np.log1p(a.clip(lower=-0.999))
        if node.op == "lag1":
            return a.shift(1)
        if node.op == "lag5":
            return a.shift(5)
        if node.op == "ma5":
            return a.rolling(5, min_periods=2).mean()
        if node.op == "std10":
            return a.rolling(10, min_periods=3).std()
    b = evaluate_tree(node.children[1], df)
    if node.op == "+":
        return a + b
    if node.op == "-":
        return a - b
    if node.op == "*":
        return a * b
    if node.op == "div":
        return a / b.replace(0, np.nan)
    if node.op == "min":
        return pd.concat([a, b], axis=1).min(axis=1)
    if node.op == "max":
        return pd.concat([a, b], axis=1).max(axis=1)
    raise ValueError(f"unknown op: {node.op}")


def mutate(node: GPNode, features: list[str], rng: random.Random) -> GPNode:
    """Replace random subtree with new random subtree."""
    if rng.random() < 0.2 or node.op == "leaf":
        return random_tree(2, features, rng)
    out = GPNode(op=node.op, children=[mutate(c, features, rng) for c in node.children])
    return out


def crossover(a: GPNode, b: GPNode, rng: random.Random) -> GPNode:
    """Swap subtrees."""
    if rng.random() < 0.3 or a.op == "leaf":
        return b
    if rng.random() < 0.3 or b.op == "leaf":
        return a
    return GPNode(
        op=a.op,
        children=[crossover(ac, bc, rng) for ac, bc in zip(a.children, b.children)],
    )


def fitness_ic(signal: pd.Series, target: pd.Series) -> float:
    """Spearman-IC fitness — robust to outliers."""
    df = pd.concat([signal, target], axis=1).dropna()
    if len(df) < 30:
        return 0.0
    ic = df.iloc[:, 0].corr(df.iloc[:, 1], method="spearman")
    if pd.isna(ic):
        return 0.0
    return float(abs(ic))  # take abs so negative correlations also evolve


def gp_search(
    features_df: pd.DataFrame,
    target: pd.Series,
    n_generations: int = 30,
    population_size: int = 80,
    tree_depth: int = 3,
    elite_frac: float = 0.2,
    seed: int = 42,
) -> list[tuple[GPNode, float]]:
    """Run GP-Search, return ranked list of (tree, fitness)."""
    rng = random.Random(seed)
    feats = list(features_df.columns)
    population = [random_tree(tree_depth, feats, rng) for _ in range(population_size)]

    for _ in range(n_generations):
        scored = []
        for tree in population:
            try:
                sig = evaluate_tree(tree, features_df)
            except Exception:  # noqa: BLE001
                scored.append((tree, 0.0))
                continue
            scored.append((tree, fitness_ic(sig, target)))
        scored.sort(key=lambda x: x[1], reverse=True)
        n_elite = max(1, int(population_size * elite_frac))
        elite = [t for t, _ in scored[:n_elite]]

        # next generation: elite + mutated/crossover
        new_pop = list(elite)
        while len(new_pop) < population_size:
            if rng.random() < 0.5 and len(elite) >= 2:
                a, b = rng.sample(elite, 2)
                child = crossover(a, b, rng)
            else:
                a = rng.choice(elite)
                child = mutate(a, feats, rng)
            new_pop.append(child)
        population = new_pop

    final = []
    for tree in population:
        try:
            sig = evaluate_tree(tree, features_df)
            f = fitness_ic(sig, target)
        except Exception:  # noqa: BLE001
            f = 0.0
        final.append((tree, f))
    final.sort(key=lambda x: x[1], reverse=True)
    return final


__all__ = [
    "GPNode",
    "random_tree",
    "evaluate_tree",
    "mutate",
    "crossover",
    "fitness_ic",
    "gp_search",
]
