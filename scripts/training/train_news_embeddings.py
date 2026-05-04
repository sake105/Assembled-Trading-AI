"""Trainiert PCA auf FinBERT-Embeddings aus der News-Archive.

Verwendung:
    python scripts/training/train_news_embeddings.py \\
        --news-store output/intel/news_event_store.jsonl \\
        --as-of 2025-12-31 \\
        --n-components 32 \\
        --out models/news_embedding_pca.joblib

PIT-Invariante:
    Nur Einträge mit published_at <= as-of werden genutzt.
    --as-of default: gestern (letzter voller Handelstag).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_texts_from_store(
    store_path: Path,
    as_of: pd.Timestamp,
    text_fields: list[str],
) -> list[str]:
    """Liest Texte aus JSONL news_event_store, gefiltert auf as_of."""
    texts: list[str] = []
    skipped = 0
    with store_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                ts_raw = (
                    rec.get("published_at")
                    or rec.get("timestamp")
                    or rec.get("created_at")
                )
                if ts_raw:
                    ts = pd.Timestamp(ts_raw)
                    if ts.tz is None:
                        ts = ts.tz_localize("UTC")
                    as_of_tz = as_of.tz_localize("UTC") if as_of.tz is None else as_of
                    if ts > as_of_tz:
                        skipped += 1
                        continue
                for field in text_fields:
                    val = rec.get(field)
                    if val and isinstance(val, str) and val.strip():
                        texts.append(val.strip())
                        break
            except Exception as _exc:
                logger.debug("[train_news_embeddings] line skipped: %s", _exc)
                continue
    logger.info(
        "Geladen: %d Texte (%d übersprungen wegen PIT-Guard)", len(texts), skipped
    )
    return texts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Trainiere PCA auf FinBERT-Embeddings aus News-Archive"
    )
    parser.add_argument(
        "--news-store",
        type=Path,
        default=Path("output/intel/news_event_store.jsonl"),
        help="Pfad zur JSONL-News-Archive",
    )
    parser.add_argument(
        "--as-of",
        type=str,
        default=None,
        help="PIT-Cutoff ISO-Datum (default: gestern)",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=32,
        help="Anzahl PCA-Komponenten (default: 32)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("models/news_embedding_pca.joblib"),
        help="Ausgabepfad für PCA-Modell",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="FinBERT-Inferenz-Batch-Größe",
    )
    parser.add_argument(
        "--text-fields",
        nargs="+",
        default=["title", "headline", "summary"],
        help="Felder für Text-Extraktion (Prioritätsreihenfolge)",
    )
    parser.add_argument(
        "--min-texts",
        type=int,
        default=100,
        help="Minimum Texte für Training (default: 100)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Nur Daten laden, kein PCA fitten",
    )
    args = parser.parse_args()

    if args.as_of is None:
        as_of = pd.Timestamp.now(tz="UTC").floor("D") - pd.Timedelta(days=1)
    else:
        as_of = pd.Timestamp(args.as_of)
    logger.info("PIT as-of: %s", as_of.date())

    if not args.news_store.exists():
        logger.error("news-store nicht gefunden: %s", args.news_store)
        return 1

    texts = _load_texts_from_store(args.news_store, as_of, args.text_fields)

    if len(texts) < args.min_texts:
        logger.error(
            "Nur %d Texte geladen — Minimum %d. "
            "Mehr News sammeln oder --min-texts reduzieren.",
            len(texts),
            args.min_texts,
        )
        return 1

    if args.dry_run:
        logger.info("--dry-run: %d Texte verfügbar, kein PCA-Fit", len(texts))
        return 0

    try:
        from src.assembled_core.ml.news_ml_bridge import (
            extract_finbert_embeddings,
            fit_embedding_pca,
        )
    except ImportError as exc:
        logger.error("Import fehlgeschlagen: %s", exc)
        return 1

    logger.info("Extrahiere FinBERT-Embeddings für %d Texte …", len(texts))
    embeddings = extract_finbert_embeddings(texts, batch_size=args.batch_size)

    if embeddings.shape[0] == 0 or np.all(embeddings == 0):
        logger.error("Embeddings leer oder alle Zero — transformers/torch installiert?")
        return 1

    logger.info("Embeddings-Shape: %s", embeddings.shape)

    pca = fit_embedding_pca(
        embeddings,
        n_components=args.n_components,
        save_path=args.out,
        as_of=as_of,
    )

    var_explained = float(pca.explained_variance_ratio_.sum()) * 100
    logger.info(
        "[OK] PCA(%d) gespeichert: %s  |  %.1f%% Varianz erklärt",
        args.n_components,
        args.out,
        var_explained,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
