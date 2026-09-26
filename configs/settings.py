"""
ReviewPulse — Central Configuration
All model names, thresholds, column aliases, and default paths live here.
Downstream modules import from this file; nothing is hardcoded elsewhere.
"""

import os
from pathlib import Path

# ── Project root ─────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[1]      # reviewpulse/
DATA_DIR = ROOT_DIR / "data"
RAW_DIR  = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
MODEL_DIR = ROOT_DIR / "models"

for _d in (RAW_DIR, PROC_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Column aliases ────────────────────────────────────────────────────────────
REVIEW_COLUMN_ALIASES = [
    "review", "Review", "text", "Text", "review_text",
    "ReviewText", "content", "Content", "body", "Body",
    "comment", "Comment",
]

RATING_COLUMN_ALIASES = [
    "rating", "Rating", "score", "Score", "stars", "Stars",
    "overall", "Overall",
]

DATE_COLUMN_ALIASES = [
    "date", "Date", "time", "Time", "timestamp", "Timestamp",
    "created_at", "CreatedAt", "review_date",
]

# ── Preprocessing ─────────────────────────────────────────────────────────────
PREPROCESSING = {
    "lowercase": True,
    "remove_html": True,
    "remove_urls": True,
    "normalize_unicode": True,
    "remove_extra_whitespace": True,
    "max_review_length": 5000,       # characters; truncate if longer
}

# ── Sentiment ─────────────────────────────────────────────────────────────────
VADER_POSITIVE_THRESHOLD  =  0.05
VADER_NEGATIVE_THRESHOLD  = -0.05

# Transformer model for sentiment (CPU-friendly distilbert)
SENTIMENT_MODEL = os.getenv(
    "REVIEWPULSE_SENTIMENT_MODEL",
    "distilbert-base-uncased-finetuned-sst-2-english",
)
SENTIMENT_BATCH_SIZE = int(os.getenv("REVIEWPULSE_SENTIMENT_BATCH", "32"))
SENTIMENT_MAX_LENGTH = 512

# ── Uncertainty / disagreement ────────────────────────────────────────────────
UNCERTAINTY_CONFIDENCE_THRESHOLD = 0.65   # transformer prob below this → low-conf

# ── Emotion ──────────────────────────────────────────────────────────────────
EMOTION_MODEL = os.getenv(
    "REVIEWPULSE_EMOTION_MODEL",
    "j-hartmann/emotion-english-distilroberta-base",
)
EMOTION_BATCH_SIZE = int(os.getenv("REVIEWPULSE_EMOTION_BATCH", "32"))

# ── Topic modelling ───────────────────────────────────────────────────────────
TOPIC_MIN_REVIEWS_FOR_BERTOPIC = 50       # fall back to TF-IDF LDA if fewer
TOPIC_N_COMPONENTS = 8                    # LDA fallback topics
TOPIC_MIN_TOPIC_SIZE = 5                  # BERTopic min_topic_size

# ── Embeddings / clustering / search ─────────────────────────────────────────
EMBEDDING_MODEL = os.getenv(
    "REVIEWPULSE_EMBEDDING_MODEL",
    "all-MiniLM-L6-v2",
)
EMBEDDING_BATCH_SIZE = int(os.getenv("REVIEWPULSE_EMBED_BATCH", "64"))

CLUSTERING_N_CLUSTERS = int(os.getenv("REVIEWPULSE_CLUSTERS", "6"))
SEARCH_TOP_K = int(os.getenv("REVIEWPULSE_SEARCH_TOPK", "10"))

# ── Keyword extraction ────────────────────────────────────────────────────────
KEYWORD_TOP_N = 20
KEYWORD_NGRAM_RANGE = (1, 2)

# ── Aspect extraction ─────────────────────────────────────────────────────────
ASPECT_SEED_CATEGORIES = {
    "battery":          ["battery", "battery life", "charge", "charging", "power"],
    "sound":            ["sound", "audio", "bass", "volume", "noise", "speaker", "headphone"],
    "display":          ["display", "screen", "resolution", "brightness", "visual"],
    "camera":           ["camera", "photo", "picture", "image", "lens", "zoom"],
    "performance":      ["performance", "speed", "fast", "slow", "lag", "processor", "cpu"],
    "price":            ["price", "cost", "value", "expensive", "cheap", "affordable", "worth"],
    "build_quality":    ["build", "quality", "durable", "sturdy", "material", "design", "feel"],
    "software":         ["software", "app", "update", "bug", "crash", "interface", "ui", "os"],
    "delivery":         ["delivery", "shipping", "arrived", "packaging", "box", "transit"],
    "customer_service": ["service", "support", "return", "refund", "warranty", "customer care"],
    "size_weight":      ["size", "weight", "heavy", "light", "compact", "portable"],
    "connectivity":     ["wifi", "bluetooth", "connection", "network", "signal", "wireless"],
}
ASPECT_TOP_N = 10           # top aspects to show per review (max)
ASPECT_SIMILARITY_THRESHOLD = 0.30   # cosine sim threshold for embedding matching

# ── Dashboard ─────────────────────────────────────────────────────────────────
SAMPLE_DATASET_PATH = RAW_DIR / "sample_reviews.csv"
SAMPLE_DATASET_URL = (
    "https://raw.githubusercontent.com/datasciencedojo/datasets/master/"
    "Womens%20Clothing%20E-Commerce%20Reviews.csv"
)

# Columns shown in the Review Explorer
REVIEW_EXPLORER_COLUMNS = [
    "review_id", "original_review", "rating", "date",
    "sentiment", "sentiment_confidence", "emotion",
    "aspects_summary", "topic_label", "cluster", "uncertainty_flag",
]
