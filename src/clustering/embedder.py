"""
ReviewPulse — Embeddings & Clustering

Generates sentence-transformer embeddings and clusters reviews with KMeans.
Falls back to TF-IDF + TruncatedSVD embeddings when sentence-transformers
are not available.

Also provides PCA-based 2D projection for cluster visualisation.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from configs.settings import (
    CLUSTERING_N_CLUSTERS,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL,
)

logger = logging.getLogger(__name__)

_sentence_model = None


def _get_sentence_model():
    global _sentence_model
    if _sentence_model is not None:
        return _sentence_model
    try:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
        _sentence_model = SentenceTransformer(EMBEDDING_MODEL)
        return _sentence_model
    except Exception as exc:
        logger.warning("Could not load SentenceTransformer (%s): %s", EMBEDDING_MODEL, exc)
        return None


def _tfidf_embeddings(texts: pd.Series, n_components: int = 64) -> np.ndarray:
    """TF-IDF + SVD fallback embeddings."""
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", sublinear_tf=True)
    mat = vectorizer.fit_transform(texts.astype(str))
    svd = TruncatedSVD(n_components=min(n_components, mat.shape[1] - 1), random_state=42)
    embeddings = svd.fit_transform(mat)
    return normalize(embeddings)


def compute_embeddings(texts: pd.Series) -> tuple[np.ndarray, str]:
    """
    Compute document embeddings.

    Returns
    -------
    embeddings : ndarray (n_docs, dim)
    method     : 'sentence-transformers' or 'tfidf-svd'
    """
    model = _get_sentence_model()
    if model is not None:
        try:
            embeddings = model.encode(
                texts.astype(str).tolist(),
                batch_size=EMBEDDING_BATCH_SIZE,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return embeddings, "sentence-transformers"
        except Exception as exc:
            logger.warning("Sentence encoding failed (%s). Falling back to TF-IDF.", exc)

    embeddings = _tfidf_embeddings(texts)
    return embeddings, "tfidf-svd"


def cluster_reviews(
    embeddings: np.ndarray,
    n_clusters: int = CLUSTERING_N_CLUSTERS,
) -> np.ndarray:
    """Run KMeans clustering on embeddings. Returns cluster labels array."""
    n_clusters = min(n_clusters, len(embeddings))
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    return km.fit_predict(embeddings)


def reduce_to_2d(embeddings: np.ndarray) -> np.ndarray:
    """Reduce embeddings to 2D via PCA for visualisation."""
    n_components = min(2, embeddings.shape[1], embeddings.shape[0] - 1)
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(embeddings)


def run_clustering_pipeline(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, str]:
    """
    Compute embeddings, run clustering, add PCA coordinates.

    Returns
    -------
    df_out     : df with cluster, embed_x, embed_y columns
    embeddings : raw embedding matrix (for search)
    method     : embedding method string
    """
    text_col = "cleaned_review" if "cleaned_review" in df.columns else "original_review"
    texts = df[text_col]

    embeddings, method = compute_embeddings(texts)
    labels = cluster_reviews(embeddings)
    xy = reduce_to_2d(embeddings)

    df = df.copy()
    df["cluster"] = labels.astype(int)
    df["embed_x"] = xy[:, 0]
    df["embed_y"] = xy[:, 1]

    logger.info("Clustered %d reviews into %d clusters using %s.", len(df), len(set(labels)), method)
    return df, embeddings, method
