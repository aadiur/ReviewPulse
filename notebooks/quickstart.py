"""
ReviewPulse — Quick-start example notebook (script form).

Run this to see the pipeline in action on the sample dataset:
    python notebooks/quickstart.py
"""

import sys
from pathlib import Path

# ensure imports work from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.data.loader import load_reviews
from src.preprocessing.cleaner import preprocess_dataframe
from src.sentiment.analyzer import run_sentiment_pipeline
from src.aspects.extractor import AspectExtractor, get_aspect_sentiment_summary
from src.topics.modeler import run_topic_pipeline
from src.keywords.extractor import run_keyword_pipeline
from src.insights.intelligence import generate_insights


def main():
    sample_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "sample_reviews.csv"
    print(f"Loading sample dataset from: {sample_path}")

    # Load
    df = load_reviews(str(sample_path))
    print(f"Loaded {len(df)} reviews.\n")

    # Preprocess
    df = preprocess_dataframe(df)
    print("Preprocessing complete.")

    # Sentiment (VADER only for speed)
    df = run_sentiment_pipeline(df, use_transformer=False)
    sent_counts = df["sentiment"].value_counts()
    print("\nSentiment distribution:")
    for k, v in sent_counts.items():
        print(f"  {k}: {v}")

    # Aspects
    extractor = AspectExtractor()
    aspects_df = extractor.extract_from_dataframe(df)
    df = extractor.add_aspects_summary(df, aspects_df)
    print(f"\nAspect mentions: {len(aspects_df)}")
    if not aspects_df.empty:
        print("Top aspects:")
        print(aspects_df["aspect"].value_counts().head(5).to_string())

    # Topics (LDA fallback — fast)
    df, topic_summary, _ = run_topic_pipeline(df, prefer_bertopic=False)
    print(f"\nTopics discovered: {len(topic_summary)}")
    print(topic_summary[["topic_label", "count"]].head(5).to_string(index=False))

    # Keywords
    keywords_df = run_keyword_pipeline(df, use_keybert=False)
    print(f"\nTop keywords ({keywords_df['method'].iloc[0]}):")
    print(keywords_df[["keyword", "score"]].head(10).to_string(index=False))

    # Insights
    insights = generate_insights(df, aspects_df=aspects_df, topic_summary=topic_summary)
    print(f"\n--- Product Intelligence ---")
    print(f"Total reviews: {insights['total_reviews']}")
    if insights.get('positive_pct') is not None:
        print(f"Positive: {insights['positive_pct']}% | Negative: {insights['negative_pct']}% | Neutral: {insights['neutral_pct']}%")
    if insights.get('average_rating'):
        print(f"Average rating: {insights['average_rating']}")

    print("\nQuick-start complete! All modules ran successfully.")


if __name__ == "__main__":
    main()
