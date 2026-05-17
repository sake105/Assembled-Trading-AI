"""Classification accuracy script for news sentiment validation (Level A).

Usage:
  python scripts/news_validation/level_a_classification.py \
      --dataset tests/news_gold/fpb_allagree.jsonl \
      --predictions tests/news_gold/my_predictions.jsonl

Predictions file: JSONL, each line {text, label} (predicted labels).
Gold dataset file: JSONL, each line {text, label} (ground-truth labels).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO / "src"))

from src.assembled_core.signals.news_validation import (
    check_level_a,
    classification_metrics,
    load_gold_dataset,
)


def load_predictions(path: Path) -> list[str]:
    labels = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    labels.append(json.loads(line)["label"].lower())
                except (json.JSONDecodeError, KeyError):
                    continue
    return labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Gold dataset JSONL")
    parser.add_argument("--predictions", required=True, help="Predictions JSONL")
    parser.add_argument(
        "--name", default="", help="Dataset name for threshold selection"
    )
    args = parser.parse_args()

    _, y_true = load_gold_dataset(Path(args.dataset))
    y_pred = load_predictions(Path(args.predictions))

    if len(y_true) != len(y_pred):
        print(f"ERROR: Gold has {len(y_true)} samples, predictions have {len(y_pred)}")
        sys.exit(1)

    metrics = classification_metrics(y_true, y_pred)
    gate = check_level_a(metrics, dataset_name=args.name or args.dataset)

    print(f"Dataset: {args.dataset}")
    print(f"N: {metrics['n']}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Macro-F1: {metrics['macro_f1']:.3f}")
    print(f"Weighted-F1: {metrics['weighted_f1']:.3f}")
    print(f"Directional error rate: {metrics['directional_error_rate']:.3f}")
    print()
    print("Per-class F1:")
    for label, f1 in metrics["per_class_f1"].items():
        print(f"  {label}: {f1:.3f}")
    print()
    print("Level A Gate:")
    for k, v in gate.items():
        mark = "PASS" if v else "FAIL"
        print(f"  [{mark}] {k}")
    all_pass = all(gate.values())
    print()
    print("Overall:", "PASS" if all_pass else "FAIL")


if __name__ == "__main__":
    main()
