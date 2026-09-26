"""
ReviewPulse — End-to-End Pipeline Runner

Run from the project root:
    python run_pipeline.py --input data/raw/sample_reviews.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# ── Windows UTF-8 console fix ─────────────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ── ensure src is importable ──────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from configs.settings import PROC_DIR
from src.data.loader import DataLoadError, load_reviews
from src.preprocessing.cleaner import preprocess_dataframe
from src.sentiment.analyzer import run_sentiment_pipeline
from src.emotions.detector import run_emotion_pipeline
from src.aspects.extractor import AspectExtractor
from src.topics.modeler import run_topic_pipeline
from src.keywords.extractor import run_keyword_pipeline
from src.clustering.embedder import run_clustering_pipeline
from src.insights.intelligence import generate_insights

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("reviewpulse.pipeline")


def run(
    input_path: str,
    output_dir: str = str(PROC_DIR),
    sample_n: int = None,
    use_transformer: bool = True,
    use_emotions: bool = True,
    use_bertopic: bool = True,
    use_keybert: bool = True,
) -> dict:
    t0 = time.perf_counter()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load ────────────────────────────────────────────────────────────────
    logger.info("Step 1/8 — Loading data from: %s", input_path)
    try:
        df = load_reviews(input_path, sample_n=sample_n)
    except DataLoadError as exc:
        logger.error("Data loading failed: %s", exc)
        return {"error": str(exc)}

    logger.info("  → %d reviews loaded.", len(df))

    # ── 2. Preprocess ──────────────────────────────────────────────────────────
    logger.info("Step 2/8 — Preprocessing text…")
    df = preprocess_dataframe(df)

    # ── 3. Sentiment ───────────────────────────────────────────────────────────
    logger.info("Step 3/8 — Sentiment analysis…")
    df = run_sentiment_pipeline(df, use_transformer=use_transformer)

    # ── 4. Emotions ────────────────────────────────────────────────────────────
    if use_emotions:
        logger.info("Step 4/8 — Emotion detection…")
        df = run_emotion_pipeline(df)
    else:
        logger.info("Step 4/8 — Emotion detection skipped.")

    # ── 5. Aspects ────────────────────────────────────────────────────────────
    logger.info("Step 5/8 — Aspect extraction…")
    extractor = AspectExtractor()
    aspects_df = extractor.extract_from_dataframe(df)
    df = extractor.add_aspects_summary(df, aspects_df)
    logger.info("  → %d aspect mentions found.", len(aspects_df))

    # ── 6. Topics ─────────────────────────────────────────────────────────────
    logger.info("Step 6/8 — Topic modelling…")
    df, topic_summary, topic_model = run_topic_pipeline(df, prefer_bertopic=use_bertopic)
    logger.info("  → %d topics discovered.", len(topic_summary))

    # ── 7. Keywords ───────────────────────────────────────────────────────────
    logger.info("Step 7/8 — Keyword extraction…")
    keywords_df = run_keyword_pipeline(df, use_keybert=use_keybert)

    # ── 8. Clustering + Embeddings ─────────────────────────────────────────────
    logger.info("Step 8/8 — Embeddings & clustering…")
    df, embeddings, embed_method = run_clustering_pipeline(df)

    # ── Insights ───────────────────────────────────────────────────────────────
    insights = generate_insights(df, aspects_df=aspects_df, topic_summary=topic_summary)

    # ── Save outputs ───────────────────────────────────────────────────────────
    reviews_out = out_dir / "reviews_processed.csv"
    aspects_out = out_dir / "aspects.csv"
    topics_out  = out_dir / "topics.csv"
    keywords_out = out_dir / "keywords.csv"

    # Drop embed columns from CSV (large floats, not needed in CSV)
    save_cols = [c for c in df.columns if c not in ("embed_x", "embed_y")]
    df[save_cols].to_csv(reviews_out, index=False)
    aspects_df.to_csv(aspects_out, index=False)
    topic_summary.to_csv(topics_out, index=False)
    keywords_df.to_csv(keywords_out, index=False)

    elapsed = round(time.perf_counter() - t0, 1)
    logger.info("Pipeline complete in %.1fs. Outputs saved to %s", elapsed, out_dir)
    logger.info("  Reviews:  %s", reviews_out)
    logger.info("  Aspects:  %s", aspects_out)
    logger.info("  Topics:   %s", topics_out)
    logger.info("  Keywords: %s", keywords_out)

    # print a brief summary
    print("\n" + "=" * 60)
    print("ReviewPulse Pipeline Summary")
    print("=" * 60)
    print(f"  Total reviews analysed : {insights['total_reviews']}")
    if insights.get("positive_pct") is not None:
        print(f"  Positive               : {insights['positive_pct']}%")
        print(f"  Negative               : {insights['negative_pct']}%")
        print(f"  Neutral                : {insights['neutral_pct']}%")
    if insights.get("average_rating"):
        print(f"  Avg rating             : {insights['average_rating']}")
    print(f"  Aspect mentions        : {len(aspects_df)}")
    print(f"  Topics discovered      : {len(topic_summary)}")
    print(f"  Embedding method       : {embed_method}")
    print(f"  Time elapsed           : {elapsed}s")
    print("=" * 60 + "\n")

    return {
        "df": df,
        "aspects_df": aspects_df,
        "topic_summary": topic_summary,
        "keywords_df": keywords_df,
        "embeddings": embeddings,
        "insights": insights,
        "embed_method": embed_method,
    }


def main():
    parser = argparse.ArgumentParser(
        description="ReviewPulse NLP Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to input CSV file containing reviews.",
    )
    parser.add_argument(
        "--output", "-o",
        default=str(PROC_DIR),
        help="Directory to save processed outputs.",
    )
    parser.add_argument(
        "--sample", "-n",
        type=int,
        default=None,
        help="Randomly sample N reviews (for testing).",
    )
    parser.add_argument(
        "--no-transformer",
        action="store_true",
        help="Skip transformer sentiment model (VADER only).",
    )
    parser.add_argument(
        "--no-emotions",
        action="store_true",
        help="Skip emotion detection.",
    )
    parser.add_argument(
        "--no-bertopic",
        action="store_true",
        help="Use LDA topic modelling instead of BERTopic.",
    )
    parser.add_argument(
        "--no-keybert",
        action="store_true",
        help="Use TF-IDF keyword extraction instead of KeyBERT.",
    )

    args = parser.parse_args()

    result = run(
        input_path=args.input,
        output_dir=args.output,
        sample_n=args.sample,
        use_transformer=not args.no_transformer,
        use_emotions=not args.no_emotions,
        use_bertopic=not args.no_bertopic,
        use_keybert=not args.no_keybert,
    )
    if "error" in result:
        sys.exit(1)


if __name__ == "__main__":
    main()
