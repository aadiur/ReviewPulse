"""
ReviewPulse — Sentiment Analysis

Two-layer approach:
  1. VADER (rule-based, always available)
  2. HuggingFace DistilBERT (transformer, downloaded on first use)

Both results are stored; an agreement / uncertainty flag is computed.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from configs.settings import (
    SENTIMENT_BATCH_SIZE,
    SENTIMENT_MAX_LENGTH,
    SENTIMENT_MODEL,
    UNCERTAINTY_CONFIDENCE_THRESHOLD,
    VADER_NEGATIVE_THRESHOLD,
    VADER_POSITIVE_THRESHOLD,
)

logger = logging.getLogger(__name__)


# ── VADER ─────────────────────────────────────────────────────────────────────

def _vader_label(compound: float) -> str:
    if compound >= VADER_POSITIVE_THRESHOLD:
        return "Positive"
    elif compound <= VADER_NEGATIVE_THRESHOLD:
        return "Negative"
    return "Neutral"


def run_vader(texts: pd.Series) -> pd.DataFrame:
    """Return DataFrame with vader_positive, vader_negative, vader_neutral, vader_compound, vader_sentiment."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        import nltk
        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
        analyzer = SentimentIntensityAnalyzer()
    except ImportError:
        logger.warning("vaderSentiment not installed. VADER scores will be NaN.")
        n = len(texts)
        return pd.DataFrame({
            "vader_positive": [None] * n,
            "vader_negative": [None] * n,
            "vader_neutral":  [None] * n,
            "vader_compound": [None] * n,
            "vader_sentiment":["Unknown"] * n,
        })

    records = []
    for text in texts:
        scores = analyzer.polarity_scores(str(text))
        records.append({
            "vader_positive": round(scores["pos"], 4),
            "vader_negative": round(scores["neg"], 4),
            "vader_neutral":  round(scores["neu"], 4),
            "vader_compound": round(scores["compound"], 4),
            "vader_sentiment": _vader_label(scores["compound"]),
        })
    return pd.DataFrame(records, index=texts.index)


# ── Transformer ───────────────────────────────────────────────────────────────

_transformer_pipeline = None   # cached globally within a session


def _get_transformer_pipeline():
    global _transformer_pipeline
    if _transformer_pipeline is not None:
        return _transformer_pipeline

    try:
        from transformers import pipeline as hf_pipeline
        logger.info("Loading transformer sentiment model: %s", SENTIMENT_MODEL)
        _transformer_pipeline = hf_pipeline(
            "sentiment-analysis",
            model=SENTIMENT_MODEL,
            truncation=True,
            max_length=SENTIMENT_MAX_LENGTH,
            batch_size=SENTIMENT_BATCH_SIZE,
            device=-1,   # CPU
        )
        return _transformer_pipeline
    except Exception as exc:
        logger.warning("Could not load transformer model (%s): %s", SENTIMENT_MODEL, exc)
        return None


def _hf_label_to_standard(label: str) -> str:
    label = label.upper()
    if "POS" in label or label == "LABEL_1":
        return "Positive"
    elif "NEG" in label or label == "LABEL_0":
        return "Negative"
    return "Neutral"


def run_transformer(texts: pd.Series) -> pd.DataFrame:
    """Return DataFrame with transformer_sentiment and transformer_confidence."""
    pipe = _get_transformer_pipeline()
    n = len(texts)
    if pipe is None:
        return pd.DataFrame({
            "transformer_sentiment":   ["Unavailable"] * n,
            "transformer_confidence":  [None] * n,
        }, index=texts.index)

    results = []
    text_list = texts.tolist()
    try:
        outputs = pipe(text_list)
    except Exception as exc:
        logger.warning("Transformer inference failed: %s", exc)
        outputs = [{"label": "Unknown", "score": 0.0}] * n

    for out in outputs:
        results.append({
            "transformer_sentiment":  _hf_label_to_standard(out["label"]),
            "transformer_confidence": round(float(out["score"]), 4),
        })

    return pd.DataFrame(results, index=texts.index)


# ── Uncertainty / agreement ───────────────────────────────────────────────────

def compute_agreement(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ``sentiment``, ``sentiment_confidence``, and ``uncertainty_flag`` columns.

    Logic:
      - If transformer is available: use transformer label as primary sentiment.
      - If transformer confidence < threshold OR labels disagree → flag uncertain.
      - Fallback to VADER if transformer unavailable.
    """
    df = df.copy()

    has_transformer = (
        "transformer_sentiment" in df.columns
        and df["transformer_sentiment"].ne("Unavailable").any()
    )

    if has_transformer:
        df["sentiment"] = df["transformer_sentiment"]
        df["sentiment_confidence"] = df["transformer_confidence"]

        low_conf = df["transformer_confidence"] < UNCERTAINTY_CONFIDENCE_THRESHOLD
        disagree = df["vader_sentiment"] != df["transformer_sentiment"]
        # treat neutrals specially: neutral VADER ≠ polarised transformer is OK
        actual_disagree = disagree & ~(
            (df["vader_sentiment"] == "Neutral") & (df["transformer_confidence"] > 0.80)
        )
        df["uncertainty_flag"] = (low_conf | actual_disagree).astype(bool)
    else:
        # fallback to VADER only
        df["sentiment"] = df.get("vader_sentiment", "Unknown")
        df["sentiment_confidence"] = df.get("vader_compound", None)
        df["uncertainty_flag"] = False

    return df


# ── Supervised evaluation (if ground truth exists) ───────────────────────────

def evaluate_predictions(
    y_true: pd.Series,
    y_pred: pd.Series,
    labels: Optional[list] = None,
) -> dict:
    """
    Compute accuracy, precision, recall, F1, and confusion matrix.
    Only call when real ground-truth labels are available.
    Returns a dict with all metrics.
    """
    try:
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
        )
    except ImportError:
        return {"error": "scikit-learn not installed"}

    labels = labels or sorted(y_true.unique().tolist())
    mask = y_true.notna() & y_pred.notna()
    yt = y_true[mask]
    yp = y_pred[mask]

    if len(yt) == 0:
        return {"error": "No valid ground-truth samples found."}

    report = classification_report(yt, yp, labels=labels, output_dict=True, zero_division=0)
    return {
        "accuracy":        round(accuracy_score(yt, yp), 4),
        "macro_precision": round(precision_score(yt, yp, average="macro", zero_division=0), 4),
        "macro_recall":    round(recall_score(yt, yp, average="macro", zero_division=0), 4),
        "macro_f1":        round(f1_score(yt, yp, average="macro", zero_division=0), 4),
        "per_class":       report,
        "confusion_matrix": confusion_matrix(yt, yp, labels=labels).tolist(),
        "labels":          labels,
        "n_samples":       int(len(yt)),
    }


# ── Full pipeline entry point ─────────────────────────────────────────────────

def run_sentiment_pipeline(df: pd.DataFrame, use_transformer: bool = True) -> pd.DataFrame:
    """
    Run full sentiment pipeline on a DataFrame with ``model_input`` column.

    Adds columns:
        vader_positive, vader_negative, vader_neutral, vader_compound, vader_sentiment
        transformer_sentiment, transformer_confidence   (if use_transformer)
        sentiment, sentiment_confidence, uncertainty_flag
    """
    text_col = "model_input" if "model_input" in df.columns else "cleaned_review"
    if text_col not in df.columns:
        text_col = "original_review"

    logger.info("Running VADER on %d reviews…", len(df))
    vader_df = run_vader(df[text_col])
    df = pd.concat([df, vader_df], axis=1)

    if use_transformer:
        logger.info("Running transformer sentiment on %d reviews…", len(df))
        trans_df = run_transformer(df[text_col])
        df = pd.concat([df, trans_df], axis=1)

    df = compute_agreement(df)
    return df
