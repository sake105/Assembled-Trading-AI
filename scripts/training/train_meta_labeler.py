"""Trainiert MetaLabeler aus learning_store.jsonl (Lopez de Prado Meta-Labeling).

Verwendung:
    python scripts/training/train_meta_labeler.py \\
        --learning-store output/ops/learning_store.jsonl \\
        --out models/meta_labeler_v1.joblib \\
        --as-of 2025-12-31

PIT-Invariante:
    Nur Records mit closed_at <= as-of werden genutzt (realisierte Returns).
    Minimum 100 Records erforderlich.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Trainiere MetaLabeler aus learning_store")
    parser.add_argument(
        "--learning-store",
        type=Path,
        default=Path("output/ops/learning_store.jsonl"),
        help="Pfad zur JSONL learning_store Datei",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("models/meta_labeler_v1.joblib"),
        help="Ausgabepfad für gespeichertes MetaLabeler-Modell",
    )
    parser.add_argument(
        "--as-of",
        type=str,
        default=None,
        help="PIT-Cutoff ISO-Datum (default: gestern)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.55,
        help="Konfidenz-Threshold für scale_position (default: 0.55)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Nur Daten laden, kein Training",
    )
    args = parser.parse_args()

    if args.as_of is None:
        as_of = pd.Timestamp.now(tz="UTC").floor("D") - pd.Timedelta(days=1)
    else:
        as_of = pd.Timestamp(args.as_of)
    logger.info("PIT as-of: %s", as_of.date())

    if not args.learning_store.exists():
        logger.error("learning-store nicht gefunden: %s", args.learning_store)
        return 1

    try:
        from src.assembled_core.ml.meta_labeling import MetaLabeler
    except ImportError as exc:
        logger.error("Import fehlgeschlagen: %s", exc)
        return 1

    if args.dry_run:
        from src.assembled_core.ml.meta_labeling import _load_records_from_store
        records = _load_records_from_store(args.learning_store, as_of=as_of)
        logger.info("--dry-run: %d Records verfügbar", len(records))
        return 0

    try:
        labeler, report = MetaLabeler.from_learning_store(
            learning_store_path=args.learning_store,
            as_of=as_of,
            confidence_threshold=args.threshold,
        )
    except ValueError as exc:
        logger.error("%s", exc)
        return 1

    logger.info(
        "[OK] MetaLabeler trainiert: n_records=%d hit_rate=%.1f%% threshold=%.2f",
        report["n_records"],
        report["hit_rate"] * 100,
        report["confidence_threshold"],
    )

    labeler.save(args.out)

    report_path = args.out.with_suffix(".report.json")
    report_path.write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    logger.info("Report: %s", report_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
