"""LDA-Topic-Modeling für Headlines (Blei/Ng/Jordan 2003).

Anwendung
---------
- Cluster News-Themen (z. B. "Earnings", "M&A", "Geopolitik", "Crypto")
- Topic-Distribution als Feature
- Topic-Drift Detection (neue Themen tauchen auf)

Implementation: scikit-learn LatentDirichletAllocation (online-VEM).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class LDAModel:
    n_topics: int
    topic_words: list[list[tuple[str, float]]]
    feature_names: list[str]
    sklearn_lda: object  # type: ignore
    sklearn_vectorizer: object  # type: ignore


def fit_lda(
    documents: list[str],
    n_topics: int = 10,
    max_features: int = 5000,
    min_df: int = 5,
    max_df: float = 0.7,
    n_top_words: int = 10,
    random_state: int = 42,
) -> LDAModel:
    """Fit LDA-Topic-Model on a corpus of documents.

    Args:
        documents: list of strings.
        n_topics: number of topics.
        max_features: vocab cap.
        min_df, max_df: document frequency filters.
        n_top_words: top-N tokens per topic to expose.

    Returns:
        ``LDAModel`` mit topic_words pro Topic.
    """
    try:
        from sklearn.decomposition import LatentDirichletAllocation
        from sklearn.feature_extraction.text import CountVectorizer
    except ImportError as e:
        raise RuntimeError("scikit-learn required for LDA") from e

    docs = [d for d in documents if isinstance(d, str) and len(d.split()) > 2]
    if len(docs) < 50:
        raise ValueError("need >= 50 documents")

    vec = CountVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        stop_words="english",
    )
    X = vec.fit_transform(docs)
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        learning_method="online",
        random_state=random_state,
        max_iter=20,
    )
    lda.fit(X)

    feature_names = list(vec.get_feature_names_out())
    topic_words = []
    for k in range(n_topics):
        top = np.argsort(lda.components_[k])[-n_top_words:][::-1]
        topic_words.append(
            [(feature_names[i], float(lda.components_[k, i])) for i in top]
        )

    return LDAModel(
        n_topics=n_topics,
        topic_words=topic_words,
        feature_names=feature_names,
        sklearn_lda=lda,
        sklearn_vectorizer=vec,
    )


def topic_distribution(model: LDAModel, documents: list[str]) -> pd.DataFrame:
    """Topic-Distribution je Dokument."""
    if not documents:
        return pd.DataFrame()
    X = model.sklearn_vectorizer.transform(documents)  # type: ignore
    dist = model.sklearn_lda.transform(X)  # type: ignore
    return pd.DataFrame(dist, columns=[f"topic_{k}" for k in range(model.n_topics)])


__all__ = ["LDAModel", "fit_lda", "topic_distribution"]
