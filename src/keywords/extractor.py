"""
ReviewPulse — Keyword & Keyphrase Extraction

Uses TF-IDF with n-gram support as the primary method.
KeyBERT is used as an optional enhancement when sentence-transformers are available.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from configs.settings import KEYWORD_NGRAM_RANGE, KEYWORD_TOP_N

logger = logging.getLogger(__name__)


# ── TF-IDF corpus keywords ────────────────────────────────────────────────────

def extract_corpus_keywords(
    texts: pd.Series,
    top_n: int = KEYWORD_TOP_N,
    ngram_range: tuple = KEYWORD_NGRAM_RANGE,
) -> pd.DataFrame:
    """
    Extract top keywords/keyphrases from the whole corpus using TF-IDF.

    Returns DataFrame: keyword | tfidf_score
    """
    n_docs = len(texts)
    min_df = 1 if n_docs < 10 else 2
    max_df = 1.0 if n_docs < 10 else 0.95

    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=ngram_range,
        max_df=max_df,
        min_df=min_df,
        sublinear_tf=True,
    )
    try:
        tfidf_matrix = vectorizer.fit_transform(texts.astype(str))
    except Exception as exc:
        logger.warning("TF-IDF failed: %s", exc)
        return pd.DataFrame(columns=["keyword", "tfidf_score"])

    mean_scores = np.asarray(tfidf_matrix.mean(axis=0)).ravel()
    vocab = vectorizer.get_feature_names_out()
    indices = mean_scores.argsort()[::-1][:top_n]

    return pd.DataFrame({
        "keyword":     [vocab[i] for i in indices],
        "tfidf_score": [round(float(mean_scores[i]), 5) for i in indices],
    })


def extract_review_keywords(
    text: str,
    top_n: int = 10,
    ngram_range: tuple = (1, 2),
) -> list[tuple[str, float]]:
    """Extract keywords from a single review string. Returns list of (word, score)."""
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=ngram_range,
        max_features=200,
        sublinear_tf=True,
    )
    try:
        mat = vectorizer.fit_transform([text])
        vocab = vectorizer.get_feature_names_out()
        scores = mat.toarray().ravel()
        indices = scores.argsort()[::-1][:top_n]
        return [(vocab[i], round(float(scores[i]), 4)) for i in indices if scores[i] > 0]
    except Exception:
        return []


# ── KeyBERT (optional) ────────────────────────────────────────────────────────

def extract_keybert_keywords(
    texts: pd.Series,
    top_n: int = KEYWORD_TOP_N,
    ngram_range: tuple = KEYWORD_NGRAM_RANGE,
) -> Optional[pd.DataFrame]:
    """
    Use KeyBERT if available. Returns None if not installed.
    Returns DataFrame: keyword | score
    """
    try:
        from keybert import KeyBERT
        kw_model = KeyBERT()
        combined_text = " ".join(texts.astype(str).tolist())[:50000]
        keywords = kw_model.extract_keywords(
            combined_text,
            keyphrase_ngram_range=ngram_range,
            top_n=top_n,
            stop_words="english",
        )
        return pd.DataFrame(keywords, columns=["keyword", "score"])
    except ImportError:
        return None
    except Exception as exc:
        logger.warning("KeyBERT extraction failed: %s", exc)
        return None


# ── Unified entry point ────────────────────────────────────────────────────────

def run_keyword_pipeline(
    df: pd.DataFrame,
    use_keybert: bool = True,
) -> pd.DataFrame:
    """
    Extract corpus keywords. Tries KeyBERT first, falls back to TF-IDF.

    Returns DataFrame: keyword | score | method
    """
    text_col = "cleaned_review" if "cleaned_review" in df.columns else "original_review"
    texts = df[text_col]

    if use_keybert:
        kb_result = extract_keybert_keywords(texts)
        if kb_result is not None:
            kb_result["method"] = "KeyBERT"
            return kb_result

    tfidf_result = extract_corpus_keywords(texts)
    tfidf_result = tfidf_result.rename(columns={"tfidf_score": "score"})
    tfidf_result["method"] = "TF-IDF"
    return tfidf_result
