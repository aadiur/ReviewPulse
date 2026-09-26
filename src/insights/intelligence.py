"""
ReviewPulse — Product Intelligence Engine

Computes high-level insights from processed review data.
All insights are derived from actual computed values — nothing is fabricated.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd


def generate_insights(
    df: pd.DataFrame,
    aspects_df: Optional[pd.DataFrame] = None,
    topic_summary: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Generate a structured insights dictionary from processed review data.

    All values are computed from real data. No metrics are invented.
    Returns a dict that can be rendered directly in the dashboard.
    """
    insights: dict = {}

    # ── basic counts ───────────────────────────────────────────────────────────
    insights["total_reviews"] = int(len(df))

    if "rating" in df.columns:
        valid_ratings = df["rating"].dropna()
        if len(valid_ratings) > 0:
            insights["average_rating"] = round(float(valid_ratings.mean()), 2)
            insights["rating_distribution"] = valid_ratings.value_counts().sort_index().to_dict()
        else:
            insights["average_rating"] = None
            insights["rating_distribution"] = {}
    else:
        insights["average_rating"] = None
        insights["rating_distribution"] = {}

    # ── sentiment ──────────────────────────────────────────────────────────────
    if "sentiment" in df.columns:
        sent_counts = df["sentiment"].value_counts()
        total = len(df)
        insights["sentiment_counts"] = sent_counts.to_dict()
        insights["positive_pct"] = round(100 * sent_counts.get("Positive", 0) / total, 1)
        insights["negative_pct"] = round(100 * sent_counts.get("Negative", 0) / total, 1)
        insights["neutral_pct"]  = round(100 * sent_counts.get("Neutral",  0) / total, 1)
    else:
        insights["sentiment_counts"] = {}
        insights["positive_pct"] = None
        insights["negative_pct"] = None
        insights["neutral_pct"]  = None

    # ── uncertainty ────────────────────────────────────────────────────────────
    if "uncertainty_flag" in df.columns:
        n_uncertain = int(df["uncertainty_flag"].sum())
        insights["uncertain_count"] = n_uncertain
        insights["uncertain_pct"]   = round(100 * n_uncertain / len(df), 1)
    else:
        insights["uncertain_count"] = 0
        insights["uncertain_pct"]   = 0.0

    # ── emotion distribution ───────────────────────────────────────────────────
    if "emotion" in df.columns:
        insights["emotion_distribution"] = df["emotion"].value_counts().to_dict()
        most_common_emotion = df["emotion"].mode()
        insights["dominant_emotion"] = most_common_emotion.iloc[0] if len(most_common_emotion) > 0 else None
    else:
        insights["emotion_distribution"] = {}
        insights["dominant_emotion"] = None

    # ── aspects ────────────────────────────────────────────────────────────────
    if aspects_df is not None and not aspects_df.empty:
        pos_aspects = (
            aspects_df[aspects_df["sentiment"] == "Positive"]
            .groupby("aspect").size()
            .sort_values(ascending=False)
        )
        neg_aspects = (
            aspects_df[aspects_df["sentiment"] == "Negative"]
            .groupby("aspect").size()
            .sort_values(ascending=False)
        )
        insights["most_praised_aspects"]   = pos_aspects.head(5).to_dict()
        insights["most_criticized_aspects"] = neg_aspects.head(5).to_dict()
        insights["aspect_frequency"] = aspects_df["aspect"].value_counts().to_dict()
    else:
        insights["most_praised_aspects"]   = {}
        insights["most_criticized_aspects"] = {}
        insights["aspect_frequency"] = {}

    # ── topics ────────────────────────────────────────────────────────────────
    if topic_summary is not None and not topic_summary.empty:
        top_topics = topic_summary.head(5)
        insights["top_topics"] = top_topics[["topic_label", "count"]].to_dict("records")
    elif "topic_label" in df.columns:
        tc = df["topic_label"].value_counts().head(5).reset_index()
        tc.columns = ["topic_label", "count"]
        insights["top_topics"] = tc.to_dict("records")
    else:
        insights["top_topics"] = []

    # ── time-based insights (only if real date column exists) ─────────────────
    if "date" in df.columns and df["date"].notna().sum() > 10:
        try:
            df_time = df[df["date"].notna()].copy()
            df_time["month"] = df_time["date"].dt.to_period("M").astype(str)
            if "sentiment" in df.columns:
                monthly = (
                    df_time.groupby("month")["sentiment"]
                    .value_counts()
                    .unstack(fill_value=0)
                    .reset_index()
                )
                insights["monthly_sentiment"] = monthly.to_dict("records")
            else:
                insights["monthly_sentiment"] = []
        except Exception:
            insights["monthly_sentiment"] = []
    else:
        insights["monthly_sentiment"] = []

    # ── review length stats ───────────────────────────────────────────────────
    text_col = "original_review" if "original_review" in df.columns else None
    if text_col:
        lengths = df[text_col].str.len()
        insights["avg_review_length"]    = round(float(lengths.mean()), 1)
        insights["median_review_length"] = round(float(lengths.median()), 1)
    else:
        insights["avg_review_length"]    = None
        insights["median_review_length"] = None

    return insights
