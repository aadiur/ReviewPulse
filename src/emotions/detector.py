"""
ReviewPulse — Emotion Detection

Uses the HuggingFace model j-hartmann/emotion-english-distilroberta-base
which returns: anger, disgust, fear, joy, neutral, sadness, surprise.

Falls back to a rule-based VADER-emotion mapping when the model is unavailable.
"""

from __future__ import annotations

import logging

import pandas as pd

from configs.settings import EMOTION_BATCH_SIZE, EMOTION_MODEL

logger = logging.getLogger(__name__)

_emotion_pipeline = None


def _get_emotion_pipeline():
    global _emotion_pipeline
    if _emotion_pipeline is not None:
        return _emotion_pipeline
    try:
        from transformers import pipeline as hf_pipeline
        logger.info("Loading emotion model: %s", EMOTION_MODEL)
        _emotion_pipeline = hf_pipeline(
            "text-classification",
            model=EMOTION_MODEL,
            truncation=True,
            max_length=512,
            batch_size=EMOTION_BATCH_SIZE,
            device=-1,
            top_k=1,
        )
        return _emotion_pipeline
    except Exception as exc:
        logger.warning("Could not load emotion model: %s", exc)
        return None


# ── VADER-based fallback ──────────────────────────────────────────────────────

def _vader_emotion(compound: float) -> tuple[str, float]:
    if compound >= 0.5:
        return "joy", round(compound, 4)
    elif compound >= 0.05:
        return "joy", round(compound, 4)
    elif compound <= -0.5:
        return "anger", round(abs(compound), 4)
    elif compound <= -0.05:
        return "sadness", round(abs(compound), 4)
    return "neutral", 0.5


def _run_fallback(texts: pd.Series, vader_compounds: pd.Series) -> pd.DataFrame:
    records = []
    for compound in vader_compounds:
        emo, conf = _vader_emotion(float(compound) if compound is not None else 0.0)
        records.append({"emotion": emo, "emotion_confidence": conf})
    return pd.DataFrame(records, index=texts.index)


# ── main ──────────────────────────────────────────────────────────────────────

def _normalise_label(label: str) -> str:
    return label.lower().strip()


def run_emotion_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ``emotion`` and ``emotion_confidence`` columns to df.

    Uses the transformer model when available, otherwise falls back
    to VADER-compound-based heuristic emotion mapping.
    """
    text_col = "model_input" if "model_input" in df.columns else "cleaned_review"
    if text_col not in df.columns:
        text_col = "original_review"

    pipe = _get_emotion_pipeline()

    if pipe is None:
        logger.info("Using VADER fallback for emotion detection.")
        if "vader_compound" in df.columns:
            emo_df = _run_fallback(df[text_col], df["vader_compound"])
        else:
            n = len(df)
            emo_df = pd.DataFrame({
                "emotion": ["neutral"] * n,
                "emotion_confidence": [0.5] * n,
            }, index=df.index)
        return pd.concat([df, emo_df], axis=1)

    logger.info("Running emotion detection on %d reviews…", len(df))
    texts = df[text_col].tolist()
    try:
        outputs = pipe(texts)
    except Exception as exc:
        logger.warning("Emotion inference failed: %s", exc)
        outputs = [[{"label": "neutral", "score": 0.5}]] * len(texts)

    emotions = []
    confidences = []
    for out in outputs:
        # top_k=1 returns list of list
        if isinstance(out, list):
            top = out[0]
        else:
            top = out
        emotions.append(_normalise_label(top["label"]))
        confidences.append(round(float(top["score"]), 4))

    emo_df = pd.DataFrame({
        "emotion":            emotions,
        "emotion_confidence": confidences,
    }, index=df.index)

    return pd.concat([df, emo_df], axis=1)
