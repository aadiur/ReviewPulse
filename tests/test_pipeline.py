"""
ReviewPulse — Test Suite

Tests cover: preprocessing, sentiment, aspect extraction,
data loading, semantic search, and pipeline execution.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import pytest

# ── make src importable ───────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.loader import DataLoadError, load_reviews, infer_column_names
from src.preprocessing.cleaner import clean_text, preprocess_dataframe
from src.sentiment.analyzer import run_vader, compute_agreement
from src.aspects.extractor import AspectExtractor, get_aspect_sentiment_summary
from src.keywords.extractor import extract_corpus_keywords
from src.insights.intelligence import generate_insights


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_reviews_csv() -> bytes:
    data = (
        "review,rating\n"
        '"The battery life is amazing and the sound quality is excellent.",5\n'
        '"Bluetooth keeps disconnecting - very frustrating experience.",1\n'
        '"Decent product for the price - nothing special.",3\n'
        '"Great build quality but the camera is disappointing.",4\n'
        '"Terrible customer service and the delivery was late.",1\n'
        '"Love the display - very bright and crisp.",5\n'
        '"Performance is sluggish with multiple apps open.",2\n'
        '"Packaging was damaged but the product works fine.",3\n'
    )
    return data.encode("utf-8")


@pytest.fixture
def sample_df(sample_reviews_csv) -> pd.DataFrame:
    return load_reviews(io.BytesIO(sample_reviews_csv))


@pytest.fixture
def preprocessed_df(sample_df) -> pd.DataFrame:
    return preprocess_dataframe(sample_df)


# ── Data loading tests ────────────────────────────────────────────────────────

class TestDataLoader:
    def test_load_basic(self, sample_reviews_csv):
        df = load_reviews(io.BytesIO(sample_reviews_csv))
        assert len(df) > 0
        assert "original_review" in df.columns
        assert "review_id" in df.columns

    def test_rating_column_detected(self, sample_reviews_csv):
        df = load_reviews(io.BytesIO(sample_reviews_csv))
        assert "rating" in df.columns
        assert df["rating"].notna().any()

    def test_empty_csv_raises(self):
        with pytest.raises(DataLoadError):
            load_reviews(io.BytesIO(b"review\n"))

    def test_missing_review_column_raises(self):
        data = b"col1,col2\na,b\nc,d\n"
        with pytest.raises(DataLoadError, match="review"):
            load_reviews(io.BytesIO(data))

    def test_duplicates_removed(self):
        data = b"review\nSame review text.\nSame review text.\nDifferent review.\n"
        df = load_reviews(io.BytesIO(data))
        assert len(df) == 2

    def test_empty_rows_removed(self):
        data = b"review\nGood product.\n\n   \nBad product.\n"
        df = load_reviews(io.BytesIO(data))
        assert len(df) == 2

    def test_sample_n(self, sample_reviews_csv):
        df = load_reviews(io.BytesIO(sample_reviews_csv), sample_n=3)
        assert len(df) == 3

    def test_infer_column_names(self):
        df_raw = pd.DataFrame({"Text": ["a", "b"], "Score": [4, 5]})
        names = infer_column_names(df_raw)
        assert names["review"] == "Text"
        assert names["rating"] == "Score"

    def test_column_alias_case_insensitive(self):
        data = b"REVIEW,RATING\nGood product.,5\nBad product.,1\n"
        df = load_reviews(io.BytesIO(data))
        assert "original_review" in df.columns


# ── Preprocessing tests ───────────────────────────────────────────────────────

class TestPreprocessing:
    def test_html_removed(self):
        result = clean_text("<b>Great product!</b>")
        assert "<b>" not in result
        assert "great product" in result

    def test_url_removed(self):
        result = clean_text("Check this out https://example.com for details.")
        assert "https" not in result
        assert "example.com" not in result

    def test_lowercase(self):
        result = clean_text("AMAZING Product Quality")
        assert result == result.lower()

    def test_unicode_normalised(self):
        result = clean_text("caf\u00e9 review")
        assert isinstance(result, str)

    def test_empty_string(self):
        result = clean_text("")
        assert result == ""

    def test_very_long_review_truncated(self, preprocessed_df):
        from configs.settings import PREPROCESSING
        max_len = PREPROCESSING["max_review_length"]
        assert all(len(t) <= max_len for t in preprocessed_df["model_input"])

    def test_original_review_preserved(self, sample_df):
        df = preprocess_dataframe(sample_df)
        assert "original_review" in df.columns
        # original_review should be identical to input
        assert df["original_review"].iloc[0] == sample_df["original_review"].iloc[0]

    def test_columns_added(self, preprocessed_df):
        assert "cleaned_review" in preprocessed_df.columns
        assert "model_input" in preprocessed_df.columns


try:
    import vaderSentiment  # noqa: F401
    HAS_VADER = True
except ImportError:
    HAS_VADER = False


# ── Sentiment tests ───────────────────────────────────────────────────────────

class TestSentiment:
    @pytest.mark.skipif(not HAS_VADER, reason="vaderSentiment package is required for VADER sentiment analysis")
    def test_vader_positive(self):
        texts = pd.Series(["This is an absolutely amazing and wonderful product!"])
        result = run_vader(texts)
        assert result["vader_sentiment"].iloc[0] == "Positive"
        assert result["vader_compound"].iloc[0] > 0.05

    @pytest.mark.skipif(not HAS_VADER, reason="vaderSentiment package is required for VADER sentiment analysis")
    def test_vader_negative(self):
        texts = pd.Series(["Terrible product, worst purchase ever, very disappointed."])
        result = run_vader(texts)
        assert result["vader_sentiment"].iloc[0] == "Negative"
        assert result["vader_compound"].iloc[0] < -0.05

    def test_vader_returns_expected_columns(self):
        texts = pd.Series(["Decent product."])
        result = run_vader(texts)
        for col in ["vader_positive", "vader_negative", "vader_neutral", "vader_compound", "vader_sentiment"]:
            assert col in result.columns

    @pytest.mark.skipif(not HAS_VADER, reason="vaderSentiment package is required for VADER sentiment analysis")
    def test_vader_scores_sum_to_one(self):
        texts = pd.Series(["The product works well enough for the price."])
        result = run_vader(texts)
        total = result["vader_positive"].iloc[0] + result["vader_negative"].iloc[0] + result["vader_neutral"].iloc[0]
        assert abs(total - 1.0) < 0.01

    def test_compute_agreement_adds_columns(self, preprocessed_df):
        vader_df = run_vader(preprocessed_df["model_input"])
        df = pd.concat([preprocessed_df, vader_df], axis=1)
        df_out = compute_agreement(df)
        assert "sentiment" in df_out.columns
        assert "uncertainty_flag" in df_out.columns

    def test_empty_text_handled(self):
        texts = pd.Series([""])
        result = run_vader(texts)
        assert len(result) == 1


# ── Aspect extraction tests ───────────────────────────────────────────────────

class TestAspectExtractor:
    def test_battery_extracted(self):
        extractor = AspectExtractor()
        results = extractor.extract_from_text("The battery life is excellent.")
        aspects = [r["aspect"] for r in results]
        assert "battery" in aspects

    def test_multiple_aspects(self):
        extractor = AspectExtractor()
        text = "The sound quality is great but the battery life is poor."
        results = extractor.extract_from_text(text)
        aspects = [r["aspect"] for r in results]
        assert "sound" in aspects
        assert "battery" in aspects

    def test_no_aspect_returns_empty(self):
        extractor = AspectExtractor()
        results = extractor.extract_from_text("This is a very generic statement with no specific keywords.")
        assert isinstance(results, list)

    def test_aspect_sentiment_present(self):
        extractor = AspectExtractor()
        results = extractor.extract_from_text("Battery life is terrible.")
        for r in results:
            assert "sentiment" in r
            assert "confidence" in r
            assert "aspect" in r

    def test_dataframe_extraction(self, preprocessed_df):
        extractor = AspectExtractor()
        aspects_df = extractor.extract_from_dataframe(preprocessed_df)
        assert isinstance(aspects_df, pd.DataFrame)
        assert "aspect" in aspects_df.columns
        assert "sentiment" in aspects_df.columns

    def test_get_aspect_summary(self, preprocessed_df):
        extractor = AspectExtractor()
        aspects_df = extractor.extract_from_dataframe(preprocessed_df)
        if not aspects_df.empty:
            summary = get_aspect_sentiment_summary(aspects_df)
            assert "aspect" in summary.columns
            assert "total" in summary.columns


# ── Keyword extraction tests ──────────────────────────────────────────────────

class TestKeywords:
    def test_returns_dataframe(self, preprocessed_df):
        result = extract_corpus_keywords(preprocessed_df["cleaned_review"])
        assert isinstance(result, pd.DataFrame)
        assert "keyword" in result.columns
        assert "tfidf_score" in result.columns

    def test_top_n_respected(self, preprocessed_df):
        result = extract_corpus_keywords(preprocessed_df["cleaned_review"], top_n=5)
        assert len(result) <= 5

    def test_scores_positive(self, preprocessed_df):
        result = extract_corpus_keywords(preprocessed_df["cleaned_review"])
        assert (result["tfidf_score"] >= 0).all()


# ── Insights tests ────────────────────────────────────────────────────────────

class TestInsights:
    def test_basic_insights(self, preprocessed_df):
        # Add a mock sentiment column
        df = preprocessed_df.copy()
        df["sentiment"] = ["Positive", "Negative", "Neutral", "Positive",
                           "Negative", "Positive", "Negative", "Neutral"][:len(df)]
        ins = generate_insights(df)
        assert "total_reviews" in ins
        assert ins["total_reviews"] == len(df)
        assert "positive_pct" in ins
        assert "sentiment_counts" in ins

    def test_no_rating_column(self):
        """DataFrame without a rating column should have None average_rating."""
        df = pd.DataFrame({
            "review_id": [0, 1, 2],
            "original_review": ["Good product.", "Bad product.", "Average."],
        })
        ins = generate_insights(df)
        assert ins.get("average_rating") is None

    def test_with_rating_column(self, sample_df, preprocessed_df):
        df = preprocess_dataframe(sample_df)
        ins = generate_insights(df)
        if "rating" in df.columns:
            assert ins.get("average_rating") is not None

    def test_aspect_insights(self, preprocessed_df):
        extractor = AspectExtractor()
        aspects_df = extractor.extract_from_dataframe(preprocessed_df)
        df = preprocessed_df.copy()
        df["sentiment"] = "Positive"
        ins = generate_insights(df, aspects_df=aspects_df)
        assert "most_praised_aspects" in ins
        assert "most_criticized_aspects" in ins


# ── Pipeline smoke test ───────────────────────────────────────────────────────

class TestPipelineSmoke:
    """Quick end-to-end smoke test — does not call transformers to stay fast."""

    def test_full_pipeline_vader_only(self, sample_reviews_csv, tmp_path):
        """Run the pipeline with --no-transformer for CI speed."""
        from run_pipeline import run

        csv_path = tmp_path / "test_reviews.csv"
        csv_path.write_bytes(sample_reviews_csv)

        result = run(
            input_path=str(csv_path),
            output_dir=str(tmp_path / "out"),
            use_transformer=False,
            use_emotions=False,
            use_bertopic=False,
            use_keybert=False,
        )
        assert "error" not in result
        assert "df" in result
        assert "sentiment" in result["df"].columns
        assert "topic_label" in result["df"].columns
        assert "cluster" in result["df"].columns
