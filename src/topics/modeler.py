"""
ReviewPulse — Topic Modelling

Primary:  Latent Dirichlet Allocation (LDA) via scikit-learn — reliable, CPU-fast.
Enhanced: BERTopic when sentence-transformers are available and n_docs >= threshold.

Outputs per review: topic_id, topic_label, topic_keywords
Outputs global:     topic summary DataFrame
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from configs.settings import (
    TOPIC_MIN_REVIEWS_FOR_BERTOPIC,
    TOPIC_MIN_TOPIC_SIZE,
    TOPIC_N_COMPONENTS,
)

logger = logging.getLogger(__name__)


# ── LDA (always available) ────────────────────────────────────────────────────

class LDATopicModel:
    """Thin wrapper around sklearn LDA with human-readable topic labels."""

    def __init__(self, n_topics: int = TOPIC_N_COMPONENTS, max_features: int = 3000):
        self.n_topics = n_topics
        self.max_features = max_features
        self._vectorizer: Optional[CountVectorizer] = None
        self._model: Optional[LatentDirichletAllocation] = None
        self.topic_keywords: dict[int, list[str]] = {}
        self.topic_labels: dict[int, str] = {}

    def fit_transform(self, texts: pd.Series) -> np.ndarray:
        """Fit LDA and return per-document topic index array."""
        self._vectorizer = CountVectorizer(
            max_features=self.max_features,
            stop_words="english",
            max_df=0.95,
            min_df=2,
            ngram_range=(1, 2),
        )
        dtm = self._vectorizer.fit_transform(texts.astype(str))
        self._model = LatentDirichletAllocation(
            n_components=self.n_topics,
            random_state=42,
            max_iter=20,
        )
        doc_topic = self._model.fit_transform(dtm)
        topic_ids = doc_topic.argmax(axis=1)

        # build keyword lists
        vocab = self._vectorizer.get_feature_names_out()
        for t_idx, component in enumerate(self._model.components_):
            top_words = [vocab[i] for i in component.argsort()[:-16:-1]]
            self.topic_keywords[t_idx] = top_words
            self.topic_labels[t_idx] = " / ".join(top_words[:3]).title()

        return topic_ids

    def get_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return summary DataFrame: topic_id | label | keywords | count | representative."""
        if "topic_id" not in df.columns:
            return pd.DataFrame()
        rows = []
        for tid, label in self.topic_labels.items():
            mask = df["topic_id"] == tid
            count = mask.sum()
            rep_reviews = df.loc[mask, "original_review"].head(3).tolist()
            rows.append({
                "topic_id":       tid,
                "topic_label":    label,
                "keywords":       ", ".join(self.topic_keywords.get(tid, [])),
                "count":          int(count),
                "representative": " | ".join(str(r)[:120] for r in rep_reviews),
            })
        return pd.DataFrame(rows).sort_values("count", ascending=False)


# ── BERTopic (optional) ───────────────────────────────────────────────────────

def _try_bertopic(texts: pd.Series) -> Optional[tuple[np.ndarray, object]]:
    """Try to fit BERTopic. Returns (topic_ids_array, model) or None."""
    try:
        from bertopic import BERTopic
        from sentence_transformers import SentenceTransformer

        logger.info("Fitting BERTopic on %d documents…", len(texts))
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        topic_model = BERTopic(
            embedding_model=embedding_model,
            min_topic_size=TOPIC_MIN_TOPIC_SIZE,
            nr_topics="auto",
            calculate_probabilities=False,
            verbose=False,
        )
        topics, _ = topic_model.fit_transform(texts.tolist())
        return np.array(topics), topic_model
    except Exception as exc:
        logger.warning("BERTopic failed (%s). Using LDA fallback.", exc)
        return None


# ── Unified entry point ────────────────────────────────────────────────────────

def run_topic_pipeline(
    df: pd.DataFrame,
    prefer_bertopic: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, object]:
    """
    Run topic modelling on df["cleaned_review"].

    Returns
    -------
    df_out        : original df with topic_id, topic_label, topic_keywords columns
    topic_summary : per-topic summary DataFrame
    model         : fitted model object (LDATopicModel or BERTopic)
    """
    text_col = "cleaned_review" if "cleaned_review" in df.columns else "original_review"
    texts = df[text_col].astype(str)

    bertopic_result = None
    lda_model = LDATopicModel(n_topics=TOPIC_N_COMPONENTS)

    if prefer_bertopic and len(df) >= TOPIC_MIN_REVIEWS_FOR_BERTOPIC:
        bertopic_result = _try_bertopic(texts)

    if bertopic_result is not None:
        raw_topics, bt_model = bertopic_result
        # BERTopic uses -1 for outliers; remap to 0 for display simplicity
        topic_ids = np.where(raw_topics == -1, 0, raw_topics)

        topic_info = bt_model.get_topic_info()
        label_map: dict[int, str] = {}
        keyword_map: dict[int, str] = {}
        for _, row in topic_info.iterrows():
            tid = int(row["Topic"])
            if tid == -1:
                tid = 0
            # BERTopic Name is like "-1_word1_word2_word3"
            name_parts = str(row.get("Name", "")).split("_")[1:]
            label_map[tid] = " / ".join(p.title() for p in name_parts[:3]) or f"Topic {tid}"
            # keywords from representation
            try:
                kw_list = bt_model.get_topic(row["Topic"]) or []
                keyword_map[tid] = ", ".join(w for w, _ in kw_list[:8])
            except Exception:
                keyword_map[tid] = ""

        df = df.copy()
        df["topic_id"]       = topic_ids
        df["topic_label"]    = [label_map.get(int(t), f"Topic {t}") for t in topic_ids]
        df["topic_keywords"] = [keyword_map.get(int(t), "") for t in topic_ids]

        # build summary
        summary_rows = []
        for tid in sorted(set(topic_ids)):
            mask = df["topic_id"] == tid
            count = int(mask.sum())
            rep = df.loc[mask, "original_review"].head(3).tolist()
            summary_rows.append({
                "topic_id":       tid,
                "topic_label":    label_map.get(tid, f"Topic {tid}"),
                "keywords":       keyword_map.get(tid, ""),
                "count":          count,
                "representative": " | ".join(str(r)[:120] for r in rep),
            })
        topic_summary = pd.DataFrame(summary_rows).sort_values("count", ascending=False)
        return df, topic_summary, bt_model

    else:
        # LDA path
        topic_ids = lda_model.fit_transform(texts)
        df = df.copy()
        df["topic_id"]       = topic_ids
        df["topic_label"]    = [lda_model.topic_labels.get(int(t), f"Topic {t}") for t in topic_ids]
        df["topic_keywords"] = [", ".join(lda_model.topic_keywords.get(int(t), [])) for t in topic_ids]

        topic_summary = lda_model.get_summary(df)
        return df, topic_summary, lda_model
