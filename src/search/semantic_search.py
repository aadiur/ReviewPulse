"""
ReviewPulse — Semantic Search

Given a pre-computed embedding matrix, find the top-k reviews most similar
to a natural-language query using cosine similarity.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from configs.settings import SEARCH_TOP_K

logger = logging.getLogger(__name__)


class SemanticSearch:
    """Cosine-similarity semantic search over review embeddings."""

    def __init__(self, embeddings: np.ndarray, df: pd.DataFrame, embedding_method: str = ""):
        self.embeddings = embeddings
        self.df = df
        self.embedding_method = embedding_method
        self._query_encoder = None

    def _get_encoder(self):
        if self._query_encoder is not None:
            return self._query_encoder
        try:
            from sentence_transformers import SentenceTransformer
            from configs.settings import EMBEDDING_MODEL
            self._query_encoder = SentenceTransformer(EMBEDDING_MODEL)
            return self._query_encoder
        except Exception:
            return None

    def _tfidf_query_embed(self, query: str) -> np.ndarray:
        """
        Very basic fallback: represent query as a mean of word indicator
        vectors aligned to the corpus embedding space.
        Not as good as transformer query encoding, but functional.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        from sklearn.preprocessing import normalize

        texts = self.df.get("cleaned_review", self.df.get("original_review", pd.Series([""]))).astype(str).tolist()
        vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", sublinear_tf=True)
        mat = vectorizer.fit_transform(texts)
        n_comp = min(64, mat.shape[1] - 1, self.embeddings.shape[1])
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        svd.fit(mat)

        q_vec = vectorizer.transform([query])
        q_embed = svd.transform(q_vec)
        return normalize(q_embed)

    def search(self, query: str, top_k: int = SEARCH_TOP_K) -> pd.DataFrame:
        """
        Return top_k reviews most similar to query.

        Returns DataFrame with columns from df plus `similarity_score`.
        """
        if len(self.embeddings) == 0:
            return pd.DataFrame()

        encoder = self._get_encoder()
        if encoder is not None:
            try:
                q_embed = encoder.encode([query], convert_to_numpy=True)
            except Exception as exc:
                logger.warning("Query encoding failed: %s. Using TF-IDF fallback.", exc)
                q_embed = self._tfidf_query_embed(query)
        else:
            q_embed = self._tfidf_query_embed(query)

        # ensure compatible shapes
        if q_embed.shape[1] != self.embeddings.shape[1]:
            logger.warning(
                "Query embedding dim %d ≠ corpus embedding dim %d. Truncating.",
                q_embed.shape[1], self.embeddings.shape[1],
            )
            min_dim = min(q_embed.shape[1], self.embeddings.shape[1])
            q_embed = q_embed[:, :min_dim]
            corpus = self.embeddings[:, :min_dim]
        else:
            corpus = self.embeddings

        sims = cosine_similarity(q_embed, corpus).ravel()
        top_indices = sims.argsort()[::-1][:top_k]

        result = self.df.iloc[top_indices].copy()
        result["similarity_score"] = np.round(sims[top_indices], 4)
        result = result.sort_values("similarity_score", ascending=False).reset_index(drop=True)
        return result
