"""
ReviewPulse — Data Loader
Handles CSV loading, column detection, validation, and basic cleaning.
Works both from Streamlit file-upload objects and from file paths.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from configs.settings import (
    DATE_COLUMN_ALIASES,
    RATING_COLUMN_ALIASES,
    REVIEW_COLUMN_ALIASES,
)

logger = logging.getLogger(__name__)


# ── public API ────────────────────────────────────────────────────────────────

class DataLoadError(Exception):
    """Raised when the CSV cannot be loaded or validated."""


def load_reviews(
    source: Union[str, Path, io.BytesIO],
    review_col: Optional[str] = None,
    rating_col: Optional[str] = None,
    date_col: Optional[str] = None,
    sample_n: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load a CSV of reviews and return a normalised DataFrame.

    Parameters
    ----------
    source      : file path, Path object, or a Streamlit UploadedFile / BytesIO.
    review_col  : override automatic review-column detection.
    rating_col  : override automatic rating-column detection.
    date_col    : override automatic date-column detection.
    sample_n    : if given, randomly sample this many rows (reproducible seed=42).

    Returns
    -------
    DataFrame with at minimum:
        review_id        int
        original_review  str
    Optionally:
        rating           float
        date             datetime
    """
    df = _read_csv(source)
    _check_not_empty(df)

    # ── detect / validate columns ─────────────────────────────────────────────
    review_col = review_col or _detect_column(df.columns, REVIEW_COLUMN_ALIASES, "review")
    rating_col = rating_col or _detect_column(df.columns, RATING_COLUMN_ALIASES, "rating", required=False)
    date_col   = date_col   or _detect_column(df.columns, DATE_COLUMN_ALIASES,   "date",   required=False)

    # ── select and rename ─────────────────────────────────────────────────────
    out = pd.DataFrame()
    out["original_review"] = df[review_col].astype(str)

    if rating_col:
        out["rating"] = pd.to_numeric(df[rating_col], errors="coerce")

    if date_col:
        out["date"] = pd.to_datetime(df[date_col], errors="coerce")

    # preserve any extra columns users might want
    extra_cols = [c for c in df.columns
                  if c not in (review_col, rating_col, date_col)
                  and c not in out.columns]
    for c in extra_cols:
        out[c] = df[c].values

    # ── clean ─────────────────────────────────────────────────────────────────
    before = len(out)
    out = out.dropna(subset=["original_review"])
    out = out[out["original_review"].str.strip() != ""]
    out = out.drop_duplicates(subset=["original_review"])
    after = len(out)
    if before != after:
        logger.info("Dropped %d rows (missing/duplicate reviews).", before - after)

    _check_not_empty(out, msg="All rows were dropped during cleaning (empty/duplicate reviews).")

    # ── sample ────────────────────────────────────────────────────────────────
    if sample_n and sample_n < len(out):
        out = out.sample(n=sample_n, random_state=42)
        logger.info("Sampled %d rows from %d.", sample_n, after)

    out = out.reset_index(drop=True)
    out.insert(0, "review_id", out.index)

    logger.info("Loaded %d reviews. Columns: %s", len(out), list(out.columns))
    return out


def infer_column_names(df: pd.DataFrame) -> dict:
    """Return detected column names without raising if not found."""
    return {
        "review": _detect_column(df.columns, REVIEW_COLUMN_ALIASES, "review", required=False),
        "rating": _detect_column(df.columns, RATING_COLUMN_ALIASES, "rating", required=False),
        "date":   _detect_column(df.columns, DATE_COLUMN_ALIASES,   "date",   required=False),
    }


# ── helpers ───────────────────────────────────────────────────────────────────

def _read_csv(source: Union[str, Path, io.BytesIO]) -> pd.DataFrame:
    try:
        if isinstance(source, str) and source.startswith(("http://", "https://")):
            return pd.read_csv(source, low_memory=False)
        elif isinstance(source, (str, Path)):
            p = Path(source)
            if not p.exists():
                raise DataLoadError(f"File not found: {p}")
            return pd.read_csv(p, low_memory=False)
        else:
            # Streamlit UploadedFile / BytesIO
            return pd.read_csv(source, low_memory=False)
    except DataLoadError:
        raise
    except Exception as exc:
        raise DataLoadError(f"Could not read CSV: {exc}") from exc


def _check_not_empty(df: pd.DataFrame, msg: str = "The CSV file is empty.") -> None:
    if df is None or len(df) == 0:
        raise DataLoadError(msg)


def _detect_column(
    columns: pd.Index,
    aliases: list[str],
    field_name: str,
    required: bool = True,
) -> Optional[str]:
    col_lower = {c.lower(): c for c in columns}
    for alias in aliases:
        if alias in columns:
            return alias
        if alias.lower() in col_lower:
            return col_lower[alias.lower()]
    if required:
        raise DataLoadError(
            f"Could not find a '{field_name}' column. "
            f"Expected one of: {aliases}. "
            f"Found columns: {list(columns)}. "
            "Please rename your column or specify it manually."
        )
    return None
