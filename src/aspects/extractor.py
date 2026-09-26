"""
ReviewPulse — Aspect Extraction & Aspect-Based Sentiment

Strategy (no spaCy required):
  1. Noun-phrase / keyword matching against seed aspect categories (via TF-IDF tokens).
  2. For each found aspect phrase, run VADER on a context window around the aspect.
  3. Return per-review aspect table with aspect name, phrase, sentiment, confidence.

If sentence-transformers are available, embedding similarity is used for better matching.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import pandas as pd

from configs.settings import (
    ASPECT_SEED_CATEGORIES,
    ASPECT_SIMILARITY_THRESHOLD,
    VADER_NEGATIVE_THRESHOLD,
    VADER_POSITIVE_THRESHOLD,
)

logger = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

def _vader_score(text: str, analyzer) -> tuple[str, float]:
    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]
    if compound >= VADER_POSITIVE_THRESHOLD:
        return "Positive", round(compound, 4)
    elif compound <= VADER_NEGATIVE_THRESHOLD:
        return "Negative", round(compound, 4)
    return "Neutral", round(compound, 4)


def _extract_context(text: str, phrase: str, window: int = 50) -> str:
    """Return a ±window-character context around the first occurrence of phrase."""
    idx = text.lower().find(phrase.lower())
    if idx == -1:
        return text[:200]
    start = max(0, idx - window)
    end   = min(len(text), idx + len(phrase) + window)
    return text[start:end]


# ── main extractor ────────────────────────────────────────────────────────────

class AspectExtractor:
    """Keyword-matching aspect extractor with VADER context sentiment."""

    def __init__(self):
        self._analyzer = None
        self._seed_patterns: dict[str, list[re.Pattern]] = {}
        self._build_patterns()

    def _build_patterns(self):
        for category, keywords in ASPECT_SEED_CATEGORIES.items():
            patterns = []
            for kw in keywords:
                escaped = re.escape(kw)
                patterns.append(re.compile(r"\b" + escaped + r"\b", re.IGNORECASE))
            self._seed_patterns[category] = patterns

    def _get_analyzer(self):
        if self._analyzer is None:
            try:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                import nltk
                try:
                    nltk.data.find("sentiment/vader_lexicon.zip")
                except LookupError:
                    nltk.download("vader_lexicon", quiet=True)
                self._analyzer = SentimentIntensityAnalyzer()
            except ImportError:
                self._analyzer = None
        return self._analyzer

    def extract_from_text(self, text: str, review_id: int = 0) -> list[dict]:
        """Return list of aspect dicts for a single review text."""
        analyzer = self._get_analyzer()
        results = []
        seen_categories = set()

        for category, patterns in self._seed_patterns.items():
            if category in seen_categories:
                continue
            matched_phrase = None
            for pat in patterns:
                m = pat.search(text)
                if m:
                    matched_phrase = m.group(0)
                    break
            if matched_phrase is None:
                continue

            context = _extract_context(text, matched_phrase)
            if analyzer:
                sentiment, confidence = _vader_score(context, analyzer)
            else:
                sentiment, confidence = "Unknown", 0.0

            results.append({
                "review_id":  review_id,
                "aspect":     category,
                "phrase":     matched_phrase,
                "sentiment":  sentiment,
                "confidence": confidence,
            })
            seen_categories.add(category)

        return results

    def extract_from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run aspect extraction on the whole DataFrame.

        Returns a long-format DataFrame:
            review_id | aspect | phrase | sentiment | confidence
        """
        text_col = "cleaned_review" if "cleaned_review" in df.columns else "original_review"
        all_rows = []
        for _, row in df.iterrows():
            rows = self.extract_from_text(str(row[text_col]), review_id=int(row.get("review_id", 0)))
            all_rows.extend(rows)

        if not all_rows:
            return pd.DataFrame(columns=["review_id", "aspect", "phrase", "sentiment", "confidence"])

        return pd.DataFrame(all_rows)

    def add_aspects_summary(self, df: pd.DataFrame, aspects_df: pd.DataFrame) -> pd.DataFrame:
        """Add ``aspects_summary`` column to main DataFrame."""
        if aspects_df.empty:
            df = df.copy()
            df["aspects_summary"] = ""
            return df

        summary = (
            aspects_df
            .groupby("review_id")["aspect"]
            .apply(lambda x: ", ".join(sorted(set(x))))
            .rename("aspects_summary")
        )
        df = df.copy()
        df = df.merge(summary, left_on="review_id", right_index=True, how="left")
        df["aspects_summary"] = df["aspects_summary"].fillna("")
        return df


# ── aggregate insights ────────────────────────────────────────────────────────

def get_aspect_sentiment_summary(aspects_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a pivot summary:
        aspect | Positive | Negative | Neutral | total | net_sentiment
    """
    if aspects_df.empty:
        return pd.DataFrame()

    counts = (
        aspects_df
        .groupby(["aspect", "sentiment"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for col in ("Positive", "Negative", "Neutral"):
        if col not in counts.columns:
            counts[col] = 0

    counts["total"] = counts["Positive"] + counts["Negative"] + counts["Neutral"]
    counts["net_sentiment"] = (counts["Positive"] - counts["Negative"]) / counts["total"].clip(lower=1)
    return counts.sort_values("total", ascending=False)
