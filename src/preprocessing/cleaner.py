"""
ReviewPulse — Text Preprocessing Pipeline

Produces three columns:
    original_review  — unchanged raw text
    cleaned_review   — suitable for NLP display / keyword analysis
    model_input      — short, stripped version suitable as model input
"""

from __future__ import annotations

import html
import logging
import re
import unicodedata
from typing import Optional

import pandas as pd

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    for _res in ("stopwords", "wordnet", "omw-1.4", "punkt", "averaged_perceptron_tagger"):
        try:
            nltk.data.find(f"corpora/{_res}" if _res not in ("punkt", "averaged_perceptron_tagger") else f"tokenizers/{_res}")
        except LookupError:
            nltk.download(_res, quiet=True)

    _STOPWORDS = set(stopwords.words("english"))
    _LEMMATIZER = WordNetLemmatizer()
    _NLTK_AVAILABLE = True
except Exception:
    _STOPWORDS = set()
    _LEMMATIZER = None
    _NLTK_AVAILABLE = False

from configs.settings import PREPROCESSING

logger = logging.getLogger(__name__)


# ── URL / HTML / emoji patterns ───────────────────────────────────────────────
_RE_URL         = re.compile(r"https?://\S+|www\.\S+")
_RE_HTML        = re.compile(r"<[^>]+>")
_RE_WHITESPACE  = re.compile(r"\s+")
_RE_NONASCII    = re.compile(r"[^\x00-\x7F]+")


def preprocess_series(
    series: pd.Series,
    lowercase: bool = True,
    remove_stopwords: bool = False,
    lemmatize: bool = False,
    max_length: Optional[int] = None,
) -> tuple[pd.Series, pd.Series]:
    """
    Preprocess a Series of raw review strings.

    Returns
    -------
    cleaned_series  : cleaned text (for display, keyword analysis)
    model_series    : model-ready text (truncated to max_length chars)
    """
    max_length = max_length or PREPROCESSING["max_review_length"]

    cleaned = series.astype(str).apply(
        lambda t: _clean(
            t,
            lowercase=lowercase,
            remove_stopwords=remove_stopwords,
            lemmatize=lemmatize,
        )
    )
    model_input = cleaned.str[:max_length]
    return cleaned, model_input


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ``cleaned_review`` and ``model_input`` columns to *df*.
    Preserves ``original_review`` unchanged.
    """
    if "original_review" not in df.columns:
        raise ValueError("DataFrame must contain an 'original_review' column.")

    cfg = PREPROCESSING
    cleaned, model_input = preprocess_series(
        df["original_review"],
        lowercase=cfg["lowercase"],
        remove_stopwords=False,   # keep stopwords for sentiment / topic context
        lemmatize=False,          # lemmatisation happens inside specific modules
        max_length=cfg["max_review_length"],
    )
    df = df.copy()
    df["cleaned_review"] = cleaned
    df["model_input"]    = model_input
    return df


# ── single-text clean ─────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Clean a single review string. Useful for interactive / API usage."""
    return _clean(text)


def _clean(
    text: str,
    lowercase: bool = True,
    remove_stopwords: bool = False,
    lemmatize: bool = False,
) -> str:
    cfg = PREPROCESSING

    # 1. Unicode normalise
    if cfg["normalize_unicode"]:
        text = unicodedata.normalize("NFKD", text)
        text = text.encode("ascii", "ignore").decode("ascii")

    # 2. HTML unescape + remove tags
    if cfg["remove_html"]:
        text = html.unescape(text)
        text = _RE_HTML.sub(" ", text)

    # 3. URLs
    if cfg["remove_urls"]:
        text = _RE_URL.sub(" ", text)

    # 4. Case
    if lowercase:
        text = text.lower()

    # 5. Whitespace
    if cfg["remove_extra_whitespace"]:
        text = _RE_WHITESPACE.sub(" ", text).strip()

    # 6. Optional stopword removal + lemmatisation (for keyword tasks)
    if _NLTK_AVAILABLE and (remove_stopwords or lemmatize):
        tokens = text.split()
        if remove_stopwords:
            tokens = [t for t in tokens if t not in _STOPWORDS]
        if lemmatize and _LEMMATIZER:
            tokens = [_LEMMATIZER.lemmatize(t) for t in tokens]
        text = " ".join(tokens)

    return text.strip()
