"""
ReviewPulse — AI-Powered Product Review Intelligence Platform
Professional Enterprise Interface (app_professional.py)

Run with:
    streamlit run app_professional.py
"""

from __future__ import annotations

import html
import io
import sys
from pathlib import Path
from typing import Optional

# Ensure project root is in sys.path
_CURRENT_DIR = Path(__file__).resolve().parent
if (_CURRENT_DIR / "src").exists():
    _REPO = _CURRENT_DIR
elif (_CURRENT_DIR.parent / "src").exists():
    _REPO = _CURRENT_DIR.parent
else:
    _REPO = _CURRENT_DIR

if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit import config as _st_config

# Programmatically enforce light, professional theme for this interface
try:
    _st_config.set_option("theme.base", "light")
    _st_config.set_option("theme.primaryColor", "#C25E3E")
    _st_config.set_option("theme.backgroundColor", "#F8F7F5")
    _st_config.set_option("theme.secondaryBackgroundColor", "#FFF8F3")
    _st_config.set_option("theme.textColor", "#1F2937")
except Exception:
    pass

# Backend pipeline imports (reused directly from src)
from src.data.loader import DataLoadError, load_reviews, infer_column_names
from src.preprocessing.cleaner import preprocess_dataframe
from src.sentiment.analyzer import run_sentiment_pipeline
from src.emotions.detector import run_emotion_pipeline
from src.aspects.extractor import AspectExtractor, get_aspect_sentiment_summary
from src.topics.modeler import run_topic_pipeline
from src.keywords.extractor import run_keyword_pipeline
from src.clustering.embedder import run_clustering_pipeline
from src.insights.intelligence import generate_insights
from src.search.semantic_search import SemanticSearch

# Configure Streamlit page layout (strictly zero emojis)
st.set_page_config(
    page_title="ReviewPulse | Product Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Color System: Light, Warm, Professional Analytics Theme ───────────────────
# Application Background: #F8F7F5 (subtle warm neutral, not stark white)
# Sidebar: #FFF8F3 (light warm neutral / soft cream)
# Content Surfaces & Chart Containers: #FFFFFF (crisp white)
# Borders: #EAE4DC / #E5E0D8 (soft warm gray)
# Primary Text: #1F2937 (dark charcoal, WCAG AAA compliant)
# Secondary Text: #4B5563 (medium gray, high contrast)
# Primary Accent: #C25E3E (warm muted terracotta/coral)
# Active Nav Tint: #FDF2EC

COLOR_PRIMARY = "#C25E3E"
COLOR_PRIMARY_HOVER = "#A94E31"
COLOR_PRIMARY_TINT = "#FDF2EC"
COLOR_PRIMARY_BORDER = "#F5D5C6"

COLOR_APP_BG = "#F8F7F5"
COLOR_SIDEBAR_BG = "#FFF8F3"
COLOR_SIDEBAR_BORDER = "#EAE4DC"
COLOR_SURFACE = "#FFFFFF"
COLOR_BORDER = "#EAE4DC"

COLOR_TEXT_PRIMARY = "#1F2937"
COLOR_TEXT_MUTED = "#4B5563"

# Semantic Colors (Restrained & Muted)
COLOR_POSITIVE = "#16A34A"
COLOR_NEGATIVE = "#DC2626"
COLOR_NEUTRAL = "#6B7280"
COLOR_WARNING = "#D97706"

PALETTE_SENTIMENT = {
    "Positive": COLOR_POSITIVE,
    "Negative": COLOR_NEGATIVE,
    "Neutral": COLOR_NEUTRAL,
    "Unknown": "#9CA3AF",
    "Unavailable": "#9CA3AF",
}

PALETTE_EMOTIONS = {
    "joy": "#D97706",
    "anger": "#DC2626",
    "sadness": "#4B6B94",
    "fear": "#8C6D97",
    "surprise": "#0D9488",
    "neutral": "#6B7280",
    "disgust": "#B45309",
}

PALETTE_TOPICS = [
    "#C25E3E", "#4B6B94", "#5B8E7D", "#8C6D97",
    "#D97706", "#64748B", "#A0522D", "#3B82F6",
]

# ── Global Light Enterprise CSS ───────────────────────────────────────────────
ENTERPRISE_CSS = """
<style>
/* Modern Typography and Base */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root, [data-testid="stAppViewContainer"], .stApp, body, html {
    --background-color: #F8F7F5 !important;
    --secondary-background-color: #FFF8F3 !important;
    --text-color: #1F2937 !important;
    --primary-color: #C25E3E !important;
    background-color: #F8F7F5 !important;
    color: #1F2937 !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

/* Page Layout & Vertical Spacing — Prevents Header Clipping */
.block-container {
    padding-top: 2.2rem !important;
    padding-bottom: 3.5rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 1320px !important;
}

/* Header bar */
[data-testid="stHeader"] {
    background-color: #F8F7F5 !important;
    border-bottom: 1px solid #EAE4DC !important;
}

/* Sidebar: Warm Light Cream Theme */
[data-testid="stSidebar"] {
    background-color: #FFF8F3 !important;
    color: #1F2937 !important;
    border-right: 1px solid #EAE4DC !important;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div {
    color: #1F2937;
}

[data-testid="stSidebar"] hr {
    border-color: #EAE4DC !important;
}

/* Sidebar Navigation Items */
[data-testid="stSidebar"] [data-testid="stRadio"] > div {
    gap: 4px;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    background: transparent;
    padding: 8px 12px;
    border-radius: 6px;
    border: 1px solid transparent;
    transition: all 0.15s ease-in-out;
    cursor: pointer;
    color: #374151 !important;
    font-weight: 500;
    margin-bottom: 2px;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background-color: #F8EFE8 !important;
    color: #1F2937 !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label[data-checked="true"],
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
    background-color: #FDF2EC !important;
    border-left: 3px solid #C25E3E !important;
    color: #9E4226 !important;
    font-weight: 600 !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] input[type="radio"]:checked {
    accent-color: #C25E3E !important;
}

/* Analytical Container Surfaces (st.container with border) */
[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #FFFFFF !important;
    border: 1px solid #EAE4DC !important;
    border-radius: 8px !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02) !important;
    margin-bottom: 14px !important;
}

[data-testid="stVerticalBlockBorderWrapper"] > div {
    background-color: #FFFFFF !important;
    border-radius: 8px !important;
    padding: 16px 20px !important;
}

/* Form Controls & Inputs */
.stTextInput input,
.stSelectbox div[data-baseweb="select"] > div,
.stMultiSelect div[data-baseweb="select"] > div,
.stSlider {
    background-color: #FFFFFF !important;
    border-color: #D1D5DB !important;
    color: #1F2937 !important;
}

.stTextInput input:focus,
.stSelectbox div[data-baseweb="select"]:focus-within {
    border-color: #C25E3E !important;
    box-shadow: 0 0 0 1px #C25E3E !important;
}

.stFileUploader section {
    border: 1px dashed #D1D5DB !important;
    background-color: #FFFFFF !important;
}

/* Buttons */
button[kind="primary"],
.stButton > button[type="primary"] {
    background-color: #C25E3E !important;
    color: #FFFFFF !important;
    border: none !important;
    font-weight: 600 !important;
}

button[kind="primary"]:hover,
.stButton > button[type="primary"]:hover {
    background-color: #A94E31 !important;
}

button[kind="secondary"],
.stButton > button:not([type="primary"]),
.stDownloadButton > button {
    background-color: #FFFFFF !important;
    color: #374151 !important;
    border: 1px solid #D1D5DB !important;
}

button[kind="secondary"]:hover,
.stButton > button:not([type="primary"]):hover,
.stDownloadButton > button:hover {
    background-color: #FAF8F5 !important;
    border-color: #9CA3AF !important;
}

/* Compact Analytical Metric Blocks */
.rp-metric-block {
    background: #FFFFFF;
    border: 1px solid #EAE4DC;
    border-radius: 6px;
    padding: 12px 16px;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02);
    min-height: 84px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.rp-metric-label {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #6B7280;
    margin-bottom: 4px;
}

.rp-metric-value {
    font-size: 22px;
    font-weight: 700;
    color: #111827;
    line-height: 1.1;
}

.rp-metric-sub {
    font-size: 11px;
    color: #6B7280;
    margin-top: 3px;
}

/* Section Headers */
.rp-section-header {
    margin-top: 4px;
    margin-bottom: 16px;
}

.rp-section-title {
    font-size: 20px;
    font-weight: 700;
    color: #111827;
    margin-bottom: 3px;
    letter-spacing: -0.01em;
}

.rp-section-subtitle {
    font-size: 13px;
    color: #4B5563;
}

/* Chart Container Header Labels */
.rp-chart-header {
    margin-bottom: 10px;
}

.rp-chart-title {
    font-size: 14px;
    font-weight: 700;
    color: #111827;
    margin-bottom: 2px;
}

.rp-chart-subtitle {
    font-size: 12px;
    color: #6B7280;
}

/* Callout Banners */
.rp-callout {
    background: #FFFFFF;
    border-left: 4px solid #C25E3E;
    border-radius: 4px 6px 6px 4px;
    padding: 14px 18px;
    margin-bottom: 14px;
    border-top: 1px solid #EAE4DC;
    border-right: 1px solid #EAE4DC;
    border-bottom: 1px solid #EAE4DC;
}

.rp-callout-warning {
    border-left-color: #D97706;
    background: #FFFBEB;
    border-color: #FDE68A;
}

.rp-callout-success {
    border-left-color: #16A34A;
    background: #F0FDF4;
    border-color: #BBF7D0;
}

/* Badges */
.rp-badge {
    display: inline-block;
    padding: 2px 7px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

.rp-badge-positive { background-color: #DCFCE7; color: #166534; border: 1px solid #BBF7D0; }
.rp-badge-negative { background-color: #FEE2E2; color: #991B1B; border: 1px solid #FECACA; }
.rp-badge-neutral  { background-color: #F3F4F6; color: #374151; border: 1px solid #E5E7EB; }
.rp-badge-warning  { background-color: #FEF3C7; color: #92400E; border: 1px solid #FDE68A; }
.rp-badge-primary  { background-color: #FDF2EC; color: #9E4226; border: 1px solid #F5D5C6; }
</style>
"""

st.markdown(ENTERPRISE_CSS, unsafe_allow_html=True)


def apply_chart_theme(
    fig: go.Figure,
    height: Optional[int] = 320,
    x_title: Optional[str] = None,
    y_title: Optional[str] = None,
) -> go.Figure:
    """
    Applies a clean, high-contrast, light professional chart theme.
    Guarantees readable dark charcoal labels, visible axes, clear titles,
    and crisp legends with zero low-contrast text.
    """
    xaxis_config = dict(
        showgrid=True,
        gridcolor="#F0ECE6",
        gridwidth=1,
        zeroline=True,
        zerolinecolor="#E5E0D8",
        zerolinewidth=1,
        showline=True,
        linecolor="#9CA3AF",
        linewidth=1.5,
        tickfont=dict(color="#1F2937", size=11, family="Inter, sans-serif"),
        ticks="outside",
        tickcolor="#9CA3AF",
        ticklen=4,
    )
    if x_title:
        xaxis_config["title"] = dict(
            text=x_title,
            font=dict(color="#111827", size=12, family="Inter, sans-serif"),
            standoff=10,
        )

    yaxis_config = dict(
        showgrid=True,
        gridcolor="#F0ECE6",
        gridwidth=1,
        zeroline=True,
        zerolinecolor="#E5E0D8",
        zerolinewidth=1,
        showline=True,
        linecolor="#9CA3AF",
        linewidth=1.5,
        tickfont=dict(color="#1F2937", size=11, family="Inter, sans-serif"),
        ticks="outside",
        tickcolor="#9CA3AF",
        ticklen=4,
    )
    if y_title:
        yaxis_config["title"] = dict(
            text=y_title,
            font=dict(color="#111827", size=12, family="Inter, sans-serif"),
            standoff=10,
        )

    fig.update_layout(
        font=dict(
            family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif",
            size=11,
            color="#1F2937",
        ),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        margin=dict(l=20, r=20, t=25, b=25),
        height=height,
        hoverlabel=dict(
            bgcolor="#1F2937",
            font_size=12,
            font_family="Inter, sans-serif",
            font_color="#FFFFFF",
            bordercolor="#374151",
        ),
        xaxis=xaxis_config,
        yaxis=yaxis_config,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#1F2937", size=11, family="Inter, sans-serif"),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#EAE4DC",
            borderwidth=1,
        ),
    )
    return fig


def render_professional_table(df: pd.DataFrame, max_height: Optional[int] = 450) -> None:
    """
    Renders a high-contrast, beautiful light professional table with clear headers,
    readable body text, subtle borders, alternating row tints, and sticky header scrolling.
    Guarantees 100% visibility of all data rows and column headers regardless of Streamlit theme settings.
    """
    if df.empty:
        st.markdown(
            "<div style='padding: 16px; color: #6B7280; font-size: 13px; text-align: center;'>No records to display.</div>",
            unsafe_allow_html=True,
        )
        return

    col_labels = {
        "review_id": "ID",
        "original_review": "Customer Review",
        "cleaned_review": "Cleaned Text",
        "rating": "Rating",
        "sentiment": "Sentiment",
        "sentiment_confidence": "Confidence",
        "vader_sentiment": "VADER",
        "transformer_sentiment": "Transformer",
        "transformer_confidence": "Confidence",
        "vader_compound": "VADER Score",
        "aspect": "Aspect",
        "phrase": "Extracted Phrase",
        "confidence": "Confidence",
        "emotion": "Emotion",
        "aspects_summary": "Detected Aspects",
        "topic_label": "Topic",
        "cluster": "Cluster",
        "uncertainty_flag": "QA Status",
    }

    headers_html = "".join(
        f"<th style='padding: 10px 14px; text-align: left; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: #1F2937; background-color: #FAF7F2; border-bottom: 2px solid #EAE4DC; position: sticky; top: 0; z-index: 2; white-space: nowrap;'>{html.escape(col_labels.get(c, c.replace('_', ' ').title()))}</th>"
        for c in df.columns
    )

    rows_html = []
    for idx, row in df.iterrows():
        bg_color = "#FFFFFF" if len(rows_html) % 2 == 0 else "#FAF8F5"
        tds = []
        for col in df.columns:
            val = row[col]
            if pd.isna(val):
                cell_content = "<span style='color: #9CA3AF;'>—</span>"
            elif col in ("sentiment", "vader_sentiment", "transformer_sentiment"):
                s = str(val)
                if s == "Positive":
                    cell_content = "<span class='rp-badge rp-badge-positive'>Positive</span>"
                elif s == "Negative":
                    cell_content = "<span class='rp-badge rp-badge-negative'>Negative</span>"
                else:
                    cell_content = f"<span class='rp-badge rp-badge-neutral'>{html.escape(s)}</span>"
            elif col == "uncertainty_flag":
                cell_content = "<span class='rp-badge rp-badge-warning'>Uncertain</span>" if val else "<span style='color: #4B5563; font-size: 12px;'>Confirmed</span>"
            elif col == "rating" and pd.notna(val):
                cell_content = f"<span style='font-weight: 600; color: #111827;'>{val}</span> <span style='color: #6B7280; font-size: 11px;'>/ 5.0</span>"
            elif isinstance(val, (float, np.floating)):
                cell_content = f"<span style='font-family: monospace; font-size: 12px; color: #1F2937;'>{val:.3f}</span>"
            elif col == "original_review":
                review_str = str(val)
                truncated = review_str[:220] + ("..." if len(review_str) > 220 else "")
                cell_content = f"<div style='max-width: 480px; min-width: 260px; line-height: 1.45; color: #1F2937; word-wrap: break-word;'>{html.escape(truncated)}</div>"
            elif col == "phrase":
                cell_content = f"<div style='max-width: 220px; line-height: 1.4; color: #1F2937; font-style: italic;'>{html.escape(str(val))}</div>"
            else:
                cell_content = f"<span style='color: #1F2937; font-size: 12px;'>{html.escape(str(val))}</span>"

            tds.append(f"<td style='padding: 9px 14px; border-bottom: 1px solid #F0ECE6; font-size: 13px; color: #1F2937; vertical-align: middle;'>{cell_content}</td>")

        rows_html.append(f"<tr style='background-color: {bg_color};'>{''.join(tds)}</tr>")

    max_h_style = f"max-height: {max_height}px;" if max_height else ""
    table_wrapper = f"""
    <div style="{max_h_style} overflow-y: auto; overflow-x: auto; border: 1px solid #EAE4DC; border-radius: 6px; background: #FFFFFF; box-shadow: 0 1px 2px rgba(0,0,0,0.02); margin-top: 6px; margin-bottom: 14px;">
        <table style="width: 100%; border-collapse: collapse; text-align: left; font-family: Inter, -apple-system, sans-serif;">
            <thead>
                <tr>{headers_html}</tr>
            </thead>
            <tbody>
                {''.join(rows_html)}
            </tbody>
        </table>
    </div>
    """
    st.markdown(table_wrapper, unsafe_allow_html=True)


# Session state keys
SS_KEYS = [
    "df", "aspects_df", "aspect_summary", "topic_summary",
    "keywords_df", "embeddings", "embed_method",
    "insights", "searcher", "pipeline_done",
    "use_transformer", "use_emotions",
]


def reset_session():
    """Clear all computed analysis state."""
    for k in SS_KEYS:
        if k in st.session_state:
            del st.session_state[k]


@st.cache_data(show_spinner=False)
def load_cached_bytes(source_bytes: bytes, filename: str) -> pd.DataFrame:
    """Load reviews from uploaded CSV bytes."""
    return load_reviews(io.BytesIO(source_bytes))


@st.cache_data(show_spinner=False)
def load_cached_path(path_str: str) -> pd.DataFrame:
    """Load reviews from local filesystem path."""
    return load_reviews(path_str)


# ── Pipeline Execution ────────────────────────────────────────────────────────

def run_analysis_pipeline(
    uploaded_file,
    local_path: str,
    data_source: str,
    use_transformer: bool,
    use_emotions: bool,
):
    """Executes the full backend NLP analysis workflow."""
    with st.spinner("Ingesting dataset..."):
        try:
            if data_source == "Upload CSV" and uploaded_file is not None:
                file_bytes = uploaded_file.read()
                df = load_cached_bytes(file_bytes, uploaded_file.name)
            elif data_source == "Local file path" and local_path.strip():
                df = load_cached_path(local_path.strip())
            elif data_source == "Sample Dataset" or local_path == "__sample__":
                sample_file = _REPO / "data" / "raw" / "sample_reviews.csv"
                df = load_cached_path(str(sample_file))
            else:
                st.error("Please configure a valid dataset source before running.")
                return
        except DataLoadError as exc:
            st.error(f"Data ingestion error: {exc}")
            return
        except Exception as exc:
            st.error(f"Unexpected data ingestion failure: {exc}")
            return

    progress_bar = st.progress(0, text="Preprocessing text corpus...")

    # Preprocessing
    df = preprocess_dataframe(df)
    progress_bar.progress(15, text="Running sentiment analysis models...")

    # Sentiment analysis
    df = run_sentiment_pipeline(df, use_transformer=use_transformer)
    progress_bar.progress(35, text="Detecting customer emotional states...")

    # Emotion detection
    if use_emotions:
        df = run_emotion_pipeline(df)
    progress_bar.progress(50, text="Extracting domain aspect mentions...")

    # Aspect extraction
    extractor = AspectExtractor()
    aspects_df = extractor.extract_from_dataframe(df)
    df = extractor.add_aspects_summary(df, aspects_df)
    aspect_summary = get_aspect_sentiment_summary(aspects_df)
    progress_bar.progress(65, text="Discovering semantic topic structures...")

    # Topic modeling
    df, topic_summary, _ = run_topic_pipeline(df)
    progress_bar.progress(75, text="Extracting salient keywords and n-grams...")

    # Keyword extraction
    keywords_df = run_keyword_pipeline(df)
    progress_bar.progress(85, text="Generating embeddings and vector clusters...")

    # Clustering and embeddings
    df, embeddings, embed_method = run_clustering_pipeline(df)
    progress_bar.progress(95, text="Synthesizing executive intelligence insights...")

    # Intelligence synthesis
    insights = generate_insights(df, aspects_df=aspects_df, topic_summary=topic_summary)
    searcher = SemanticSearch(embeddings, df, embed_method)

    progress_bar.progress(100, text="Analysis complete.")

    # Store in session state
    st.session_state["df"] = df
    st.session_state["aspects_df"] = aspects_df
    st.session_state["aspect_summary"] = aspect_summary
    st.session_state["topic_summary"] = topic_summary
    st.session_state["keywords_df"] = keywords_df
    st.session_state["embeddings"] = embeddings
    st.session_state["embed_method"] = embed_method
    st.session_state["insights"] = insights
    st.session_state["searcher"] = searcher
    st.session_state["pipeline_done"] = True
    st.session_state["use_transformer"] = use_transformer
    st.session_state["use_emotions"] = use_emotions

    st.rerun()


# ── Sidebar Interface ─────────────────────────────────────────────────────────

def render_sidebar():
    """Renders the executive sidebar configuration and navigation."""
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 4px 0 14px 0;">
                <div style="font-size: 18px; font-weight: 700; color: #1F2937; letter-spacing: -0.01em;">ReviewPulse</div>
                <div style="font-size: 12px; font-weight: 500; color: #6B7280; margin-top: 2px;">Enterprise Product Intelligence</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown(
            "<div style='font-size: 11px; font-weight: 600; text-transform: uppercase; color: #6B7280; letter-spacing: 0.05em;'>Data Source</div>",
            unsafe_allow_html=True,
        )

        data_source = st.radio(
            "Data Source",
            ["Sample Dataset", "Upload CSV", "Local file path"],
            index=0,
            label_visibility="collapsed",
        )

        uploaded_file = None
        local_path = ""

        if data_source == "Upload CSV":
            uploaded_file = st.file_uploader(
                "Upload CSV File",
                type=["csv"],
                label_visibility="collapsed",
            )
        elif data_source == "Local file path":
            local_path = st.text_input(
                "File Path",
                value="data/raw/sample_reviews.csv",
                placeholder="data/raw/sample_reviews.csv",
                label_visibility="collapsed",
            )
        else:
            local_path = "__sample__"
            st.caption("Default: 50 curated consumer product reviews.")

        st.markdown("---")
        st.markdown(
            "<div style='font-size: 11px; font-weight: 600; text-transform: uppercase; color: #6B7280; letter-spacing: 0.05em;'>Pipeline Configuration</div>",
            unsafe_allow_html=True,
        )

        use_transformer = st.checkbox("Transformer Sentiment Model", value=True)
        use_emotions = st.checkbox("Emotion Classification", value=True)

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

        col_run, col_reset = st.columns(2)
        with col_run:
            run_btn = st.button("Run Analysis", use_container_width=True, type="primary")
        with col_reset:
            reset_btn = st.button("Reset", use_container_width=True)

        if reset_btn:
            reset_session()
            st.rerun()

        st.markdown("---")
        st.markdown(
            "<div style='font-size: 11px; font-weight: 600; text-transform: uppercase; color: #6B7280; letter-spacing: 0.05em;'>Navigation</div>",
            unsafe_allow_html=True,
        )

        nav_page = st.radio(
            "Navigation Menu",
            [
                "Overview",
                "Sentiment Intelligence",
                "Aspect Intelligence",
                "Topic Explorer",
                "Semantic Explorer",
                "Review Explorer",
                "Product Insights",
            ],
            index=0,
            label_visibility="collapsed",
        )

        st.markdown("---")
        pipeline_done = st.session_state.get("pipeline_done", False)
        if pipeline_done:
            df = st.session_state["df"]
            st.markdown(
                f"""
                <div style='background: #FFFFFF; border: 1px solid #EAE4DC; border-radius: 6px; padding: 12px 14px;'>
                    <div style='font-size: 10px; text-transform: uppercase; color: #6B7280; font-weight: 600; letter-spacing: 0.05em;'>Active Corpus</div>
                    <div style='font-size: 15px; font-weight: 700; color: #1F2937; margin-top: 2px;'>{len(df):,} Reviews</div>
                    <div style='font-size: 11px; font-weight: 500; color: #16A34A; margin-top: 2px;'>Pipeline Active</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div style='background: #FFFFFF; border: 1px solid #EAE4DC; border-radius: 6px; padding: 12px 14px;'>
                    <div style='font-size: 10px; text-transform: uppercase; color: #6B7280; font-weight: 600; letter-spacing: 0.05em;'>Pipeline Status</div>
                    <div style='font-size: 13px; font-weight: 500; color: #4B5563; margin-top: 2px;'>Awaiting Execution</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    return uploaded_file, local_path, data_source, use_transformer, use_emotions, run_btn, nav_page


# ── Blank Slate / Pre-run Landing ─────────────────────────────────────────────

def render_empty_state():
    """Renders the professional landing state when no analysis is active."""
    st.markdown(
        """
        <div style="background: #FFFFFF; border: 1px solid #EAE4DC; border-radius: 8px; padding: 28px 32px; margin-top: 4px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.02);">
            <div style="font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #C25E3E;">Enterprise Product Intelligence</div>
            <div style="font-size: 26px; font-weight: 700; color: #111827; margin-top: 4px;">ReviewPulse Analytics Platform</div>
            <div style="font-size: 14px; color: #4B5563; margin-top: 6px; max-width: 820px; line-height: 1.5;">
                Extract actionable product signals, track customer sentiment distribution, surface granular aspect performance, and isolate high-value issues across unstructured customer reviews.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="rp-section-header">
            <div class="rp-section-title">Platform Capabilities</div>
            <div class="rp-section-subtitle">Automated natural language intelligence pipeline ready for execution.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            """
            <div class="rp-metric-block" style="min-height: 130px; justify-content: flex-start;">
                <div class="rp-metric-label">Sentiment Engine</div>
                <div style="font-size: 14px; font-weight: 700; color: #111827; margin-top: 2px;">Dual-Layer Detection</div>
                <div style="font-size: 12px; color: #4B5563; margin-top: 5px; line-height: 1.4;">
                    Combines rule-based VADER lexicon with contextual DistilBERT transformers, tracking inter-model agreement and flagging uncertainty.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="rp-metric-block" style="min-height: 130px; justify-content: flex-start;">
                <div class="rp-metric-label">Aspect Extraction</div>
                <div style="font-size: 14px; font-weight: 700; color: #111827; margin-top: 2px;">Feature-Level Insights</div>
                <div style="font-size: 12px; color: #4B5563; margin-top: 5px; line-height: 1.4;">
                    Isolates mentions of battery, build, performance, software, pricing, and service with localized sentiment mapping.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div class="rp-metric-block" style="min-height: 130px; justify-content: flex-start;">
                <div class="rp-metric-label">Topic Discovery</div>
                <div style="font-size: 14px; font-weight: 700; color: #111827; margin-top: 2px;">Unsupervised Clusters</div>
                <div style="font-size: 12px; color: #4B5563; margin-top: 5px; line-height: 1.4;">
                    Groups feedback into coherent operational topics with 2D embedding space projections and TF-IDF keyphrase extraction.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            """
            <div class="rp-metric-block" style="min-height: 130px; justify-content: flex-start;">
                <div class="rp-metric-label">Semantic Discovery</div>
                <div style="font-size: 14px; font-weight: 700; color: #111827; margin-top: 2px;">Neural Search Engine</div>
                <div style="font-size: 12px; color: #4B5563; margin-top: 5px; line-height: 1.4;">
                    Enables cosine-similarity natural language queries across the entire review collection to pinpoint specific feedback.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="rp-callout">
            <div style="font-size: 13px; font-weight: 700; color: #9E4226;">Getting Started</div>
            <div style="font-size: 13px; color: #374151; margin-top: 4px;">
                Select <b>Sample Dataset</b> in the sidebar and click <b>Run Analysis</b>, or upload your own CSV containing customer reviews.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Page 1: Overview ──────────────────────────────────────────────────────────

def render_overview_page(
    df: pd.DataFrame,
    insights: dict,
    topic_summary: pd.DataFrame,
    aspect_summary: pd.DataFrame,
):
    """Renders the executive summary overview page."""
    st.markdown(
        """
        <div class="rp-section-header">
            <div class="rp-section-title">Executive Overview</div>
            <div class="rp-section-subtitle">Real-time aggregate product health metrics and sentiment indicators across customer reviews.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    total_reviews = insights.get("total_reviews", len(df))
    avg_rating = insights.get("average_rating")
    pos_pct = insights.get("positive_pct", 0.0)
    neg_pct = insights.get("negative_pct", 0.0)
    unc_count = insights.get("uncertain_count", 0)
    unc_pct = insights.get("uncertain_pct", 0.0)

    # Compact Analytical KPI Row
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(
            f"""
            <div class="rp-metric-block">
                <div class="rp-metric-label">TOTAL REVIEWS</div>
                <div class="rp-metric-value">{total_reviews:,}</div>
                <div class="rp-metric-sub">Processed corpus</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        rating_str = f"{avg_rating:.2f} / 5" if avg_rating is not None else "N/A"
        st.markdown(
            f"""
            <div class="rp-metric-block">
                <div class="rp-metric-label">AVERAGE RATING</div>
                <div class="rp-metric-value">{rating_str}</div>
                <div class="rp-metric-sub">Mean customer score</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="rp-metric-block">
                <div class="rp-metric-label">POSITIVE REVIEWS</div>
                <div class="rp-metric-value" style="color: #16A34A;">{pos_pct}%</div>
                <div class="rp-metric-sub">Favorable feedback</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div class="rp-metric-block">
                <div class="rp-metric-label">NEGATIVE REVIEWS</div>
                <div class="rp-metric-value" style="color: #DC2626;">{neg_pct}%</div>
                <div class="rp-metric-sub">Unfavorable feedback</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c5:
        st.markdown(
            f"""
            <div class="rp-metric-block">
                <div class="rp-metric-label">UNCERTAIN REVIEWS</div>
                <div class="rp-metric-value" style="color: #D97706;">{unc_pct}%</div>
                <div class="rp-metric-sub">{unc_count:,} reviews flagged</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    if unc_count > 0:
        st.markdown(
            f"""
            <div class="rp-callout rp-callout-warning">
                <div style="font-weight: 700; color: #92400E; font-size: 13px;">Attention Required: Model Divergence Detected</div>
                <div style="font-size: 12px; color: #78350F; margin-top: 3px;">
                    {unc_count:,} reviews ({unc_pct}%) exhibit model disagreement or low confidence thresholds. These are segregated for manual QA inspection in the Review Explorer.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Chart Row 1: Sentiment Distribution & Top Topics
    r1_col1, r1_col2 = st.columns(2)

    with r1_col1:
        with st.container(border=True):
            st.markdown(
                """
                <div class="rp-chart-header">
                    <div class="rp-chart-title">Sentiment Distribution</div>
                    <div class="rp-chart-subtitle">Breakdown of positive, negative, and neutral customer feedback.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if "sentiment" in df.columns:
                sent_counts = df["sentiment"].value_counts().reset_index()
                sent_counts.columns = ["Sentiment", "Count"]
                fig_pie = px.pie(
                    sent_counts,
                    names="Sentiment",
                    values="Count",
                    color="Sentiment",
                    color_discrete_map=PALETTE_SENTIMENT,
                    hole=0.45,
                )
                fig_pie.update_traces(
                    textposition="inside",
                    textinfo="percent+label",
                    textfont=dict(color="#FFFFFF", size=12, family="Inter, sans-serif"),
                    marker=dict(line=dict(color="#FFFFFF", width=2)),
                )
                fig_pie = apply_chart_theme(fig_pie, height=280)
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("Sentiment data unavailable.")

    with r1_col2:
        with st.container(border=True):
            st.markdown(
                """
                <div class="rp-chart-header">
                    <div class="rp-chart-title">Leading Discussion Topics</div>
                    <div class="rp-chart-subtitle">Volume of reviews grouped by unsupervised topic discovery.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if topic_summary is not None and not topic_summary.empty:
                top_t = topic_summary.head(7).sort_values("count", ascending=True)
                fig_top = px.bar(
                    top_t,
                    x="count",
                    y="topic_label",
                    orientation="h",
                    color_discrete_sequence=[COLOR_PRIMARY],
                    labels={"count": "Review Count", "topic_label": "Topic"},
                )
                fig_top = apply_chart_theme(fig_top, height=280, x_title="Review Count", y_title="Topic")
                fig_top.update_layout(showlegend=False)
                st.plotly_chart(fig_top, use_container_width=True)
            else:
                st.info("Topic distribution unavailable.")

    # Chart Row 2: Aspect Sentiment Overview & Emotion Breakdown
    r2_col1, r2_col2 = st.columns(2)

    with r2_col1:
        with st.container(border=True):
            st.markdown(
                """
                <div class="rp-chart-header">
                    <div class="rp-chart-title">Aspect Sentiment Breakdown</div>
                    <div class="rp-chart-subtitle">Customer perception mapped across extracted product components.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if aspect_summary is not None and not aspect_summary.empty:
                melt_cols = [c for c in ["Positive", "Negative", "Neutral"] if c in aspect_summary.columns]
                aspect_plot = aspect_summary.head(8).melt(
                    id_vars=["aspect"],
                    value_vars=melt_cols,
                    var_name="Sentiment",
                    value_name="Count",
                )
                fig_aspect = px.bar(
                    aspect_plot,
                    x="Count",
                    y="aspect",
                    color="Sentiment",
                    color_discrete_map=PALETTE_SENTIMENT,
                    orientation="h",
                    barmode="stack",
                    labels={"Count": "Mentions", "aspect": "Product Aspect"},
                )
                fig_aspect = apply_chart_theme(fig_aspect, height=290, x_title="Mention Count", y_title="Product Aspect")
                fig_aspect.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig_aspect, use_container_width=True)
            else:
                st.info("No domain aspects extracted from corpus.")

    with r2_col2:
        with st.container(border=True):
            st.markdown(
                """
                <div class="rp-chart-header">
                    <div class="rp-chart-title">Customer Emotion Spectrum</div>
                    <div class="rp-chart-subtitle">Emotional tones identified across customer reviews.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if "emotion" in df.columns:
                emo_counts = df["emotion"].value_counts().reset_index()
                emo_counts.columns = ["Emotion", "Count"]
                fig_emo = px.bar(
                    emo_counts,
                    x="Emotion",
                    y="Count",
                    color="Emotion",
                    color_discrete_map=PALETTE_EMOTIONS,
                    labels={"Count": "Reviews", "Emotion": "Detected Emotion"},
                )
                fig_emo = apply_chart_theme(fig_emo, height=290, x_title="Detected Emotion", y_title="Reviews")
                fig_emo.update_layout(showlegend=False)
                st.plotly_chart(fig_emo, use_container_width=True)
            else:
                st.info("Emotion classification data unavailable.")

    # Rating Breakdown if present
    if "rating" in df.columns and df["rating"].notna().sum() > 0:
        with st.container(border=True):
            st.markdown(
                """
                <div class="rp-chart-header">
                    <div class="rp-chart-title">Customer Rating Distribution</div>
                    <div class="rp-chart-subtitle">Frequency of verified star ratings in active dataset.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            rd_df = df["rating"].dropna().value_counts().reset_index()
            rd_df.columns = ["Rating", "Count"]
            rd_df["Rating"] = rd_df["Rating"].astype(str) + " Stars"
            fig_r = px.bar(
                rd_df.sort_values("Rating"),
                x="Rating",
                y="Count",
                color_discrete_sequence=[COLOR_PRIMARY],
                labels={"Count": "Reviews", "Rating": "Star Rating"},
            )
            fig_r = apply_chart_theme(fig_r, height=250, x_title="Star Rating", y_title="Reviews")
            fig_r.update_layout(showlegend=False)
            st.plotly_chart(fig_r, use_container_width=True)


# ── Page 2: Sentiment Intelligence ───────────────────────────────────────────

def render_sentiment_page(df: pd.DataFrame):
    """Renders deep-dive dual model sentiment analysis and agreement audit."""
    st.markdown(
        """
        <div class="rp-section-header">
            <div class="rp-section-title">Sentiment Intelligence</div>
            <div class="rp-section-subtitle">Multi-model audit comparing deterministic VADER lexicon and transformer neural classification.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    has_vader = "vader_sentiment" in df.columns
    has_transformer = "transformer_sentiment" in df.columns and df["transformer_sentiment"].ne("Unavailable").any()

    # Section 1: Side by side model distribution
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown(
                """
                <div class="rp-chart-header">
                    <div class="rp-chart-title">VADER Lexicon Model</div>
                    <div class="rp-chart-subtitle">Deterministic rule-based sentiment mapping.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if has_vader:
                vc = df["vader_sentiment"].value_counts().reset_index()
                vc.columns = ["Sentiment", "Count"]
                fig_v = px.bar(
                    vc,
                    x="Sentiment",
                    y="Count",
                    color="Sentiment",
                    color_discrete_map=PALETTE_SENTIMENT,
                    labels={"Count": "Reviews", "Sentiment": "Sentiment"},
                )
                fig_v = apply_chart_theme(fig_v, height=260, x_title="Sentiment Classification", y_title="Reviews")
                fig_v.update_layout(showlegend=False)
                st.plotly_chart(fig_v, use_container_width=True)
            else:
                st.info("VADER sentiment data unavailable.")

    with col2:
        with st.container(border=True):
            st.markdown(
                """
                <div class="rp-chart-header">
                    <div class="rp-chart-title">Transformer Model</div>
                    <div class="rp-chart-subtitle">Contextual neural representation (DistilBERT).</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if has_transformer:
                tc = df["transformer_sentiment"].value_counts().reset_index()
                tc.columns = ["Sentiment", "Count"]
                fig_t = px.bar(
                    tc,
                    x="Sentiment",
                    y="Count",
                    color="Sentiment",
                    color_discrete_map=PALETTE_SENTIMENT,
                    labels={"Count": "Reviews", "Sentiment": "Sentiment"},
                )
                fig_t = apply_chart_theme(fig_t, height=260, x_title="Sentiment Classification", y_title="Reviews")
                fig_t.update_layout(showlegend=False)
                st.plotly_chart(fig_t, use_container_width=True)
            else:
                st.info("Transformer model execution was skipped or unavailable.")

    # Section 2: Model Agreement & Disagreement Audit
    with st.container(border=True):
        st.markdown(
            """
            <div class="rp-chart-header">
                <div class="rp-chart-title">Model Agreement & Divergence Analysis</div>
                <div class="rp-chart-subtitle">Evaluating classification consensus between lexicon and transformer systems.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if has_vader and has_transformer:
            agree_mask = df["vader_sentiment"] == df["transformer_sentiment"]
            n_agree = int(agree_mask.sum())
            n_disagree = len(df) - n_agree
            pct_agree = round(100 * n_agree / len(df), 1)
            pct_disagree = round(100 - pct_agree, 1)

            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(
                    f"""
                    <div class="rp-metric-block">
                        <div class="rp-metric-label">MODEL AGREEMENT</div>
                        <div class="rp-metric-value" style="color: #16A34A;">{pct_agree}%</div>
                        <div class="rp-metric-sub">{n_agree:,} matching classifications</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with m2:
                st.markdown(
                    f"""
                    <div class="rp-metric-block">
                        <div class="rp-metric-label">MODEL DISAGREEMENT</div>
                        <div class="rp-metric-value" style="color: #DC2626;">{pct_disagree}%</div>
                        <div class="rp-metric-sub">{n_disagree:,} divergent classifications</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with m3:
                unc_total = int(df["uncertainty_flag"].sum()) if "uncertainty_flag" in df.columns else 0
                st.markdown(
                    f"""
                    <div class="rp-metric-block">
                        <div class="rp-metric-label">FLAGGED UNCERTAIN</div>
                        <div class="rp-metric-value" style="color: #D97706;">{unc_total:,}</div>
                        <div class="rp-metric-sub">Low confidence or disagreement</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if n_disagree > 0:
                st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
                st.markdown(
                    f"<div style='font-size: 13px; font-weight: 700; color: #111827; margin-bottom: 6px;'>Model Disagreement Instances ({n_disagree} reviews)</div>",
                    unsafe_allow_html=True,
                )
                disagree_cols = [
                    c for c in ["review_id", "original_review", "vader_sentiment", "transformer_sentiment", "transformer_confidence"]
                    if c in df.columns
                ]
                render_professional_table(df[~agree_mask][disagree_cols].head(25), max_height=320)
        else:
            st.info("Both VADER and Transformer must be active to compute agreement metrics.")

    # Section 3: Confidence Distribution & Score Analysis
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        with st.container(border=True):
            st.markdown(
                """
                <div class="rp-chart-header">
                    <div class="rp-chart-title">Prediction Confidence Spectrum</div>
                    <div class="rp-chart-subtitle">Distribution of model confidence scores across all reviews.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if has_transformer and "transformer_confidence" in df.columns:
                fig_conf = px.histogram(
                    df,
                    x="transformer_confidence",
                    nbins=35,
                    color_discrete_sequence=[COLOR_PRIMARY],
                    labels={"transformer_confidence": "Confidence Score"},
                )
                fig_conf.add_vline(
                    x=0.65,
                    line_dash="dash",
                    line_color=COLOR_NEGATIVE,
                    annotation_text="Confidence Cutoff (0.65)",
                    annotation_position="top left",
                    annotation_font=dict(color="#991B1B", size=11, family="Inter, sans-serif"),
                )
                fig_conf = apply_chart_theme(fig_conf, height=270, x_title="Confidence Score", y_title="Review Count")
                st.plotly_chart(fig_conf, use_container_width=True)
            elif has_vader and "vader_compound" in df.columns:
                fig_vscore = px.histogram(
                    df,
                    x="vader_compound",
                    nbins=35,
                    color_discrete_sequence=[COLOR_PRIMARY],
                    labels={"vader_compound": "VADER Compound Score"},
                )
                fig_vscore = apply_chart_theme(fig_vscore, height=270, x_title="VADER Compound Score", y_title="Review Count")
                st.plotly_chart(fig_vscore, use_container_width=True)

    with col_c2:
        with st.container(border=True):
            st.markdown(
                """
                <div class="rp-chart-header">
                    <div class="rp-chart-title">Review Length vs Sentiment</div>
                    <div class="rp-chart-subtitle">Character count distribution across sentiment classes.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            text_col = "original_review" if "original_review" in df.columns else "cleaned_review"
            if text_col in df.columns:
                df_len = df.copy()
                df_len["char_length"] = df_len[text_col].astype(str).str.len()
                fig_len = px.histogram(
                    df_len,
                    x="char_length",
                    color="sentiment" if "sentiment" in df_len.columns else None,
                    color_discrete_map=PALETTE_SENTIMENT,
                    nbins=35,
                    barmode="overlay",
                    labels={"char_length": "Character Count"},
                )
                fig_len = apply_chart_theme(fig_len, height=270, x_title="Character Count", y_title="Review Count")
                st.plotly_chart(fig_len, use_container_width=True)

    # Section 4: Rating vs Sentiment & Temporal Trends
    if "date" in df.columns and df["date"].notna().sum() > 5 and "sentiment" in df.columns:
        with st.container(border=True):
            st.markdown(
                """
                <div class="rp-chart-header">
                    <div class="rp-chart-title">Sentiment Movement Over Time</div>
                    <div class="rp-chart-subtitle">Monthly volume trends segmented by sentiment classification.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            try:
                time_df = df[df["date"].notna()].copy()
                time_df["month"] = time_df["date"].dt.to_period("M").dt.to_timestamp()
                monthly_counts = time_df.groupby(["month", "sentiment"]).size().reset_index(name="count")
                fig_time = px.line(
                    monthly_counts,
                    x="month",
                    y="count",
                    color="sentiment",
                    color_discrete_map=PALETTE_SENTIMENT,
                    markers=True,
                    labels={"count": "Monthly Reviews", "month": "Timeline"},
                )
                fig_time = apply_chart_theme(fig_time, height=280, x_title="Timeline", y_title="Monthly Reviews")
                st.plotly_chart(fig_time, use_container_width=True)
            except Exception:
                pass


# ── Page 3: Aspect Intelligence ───────────────────────────────────────────────

def render_aspect_page(df: pd.DataFrame, aspects_df: pd.DataFrame, aspect_summary: pd.DataFrame):
    """Renders granular aspect-based sentiment extraction analysis."""
    st.markdown(
        """
        <div class="rp-section-header">
            <div class="rp-section-title">Aspect Intelligence</div>
            <div class="rp-section-subtitle">Granular extraction and sentiment attribution for specific product components and features.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if aspects_df.empty:
        st.markdown(
            """
            <div class="rp-callout rp-callout-warning">
                <div style="font-weight: 700; color: #92400E; font-size: 13px;">No Domain Aspects Detected</div>
                <div style="font-size: 12px; color: #78350F; margin-top: 2px;">
                    The aspect extractor searched for keywords related to battery, display, audio, build quality, software, pricing, and service, but found zero matches in this corpus.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Aspect KPIs
    total_mentions = len(aspects_df)
    unique_aspects = aspects_df["aspect"].nunique()
    reviews_with_aspect = aspects_df["review_id"].nunique()
    coverage_pct = round(100 * reviews_with_aspect / len(df), 1)

    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown(
            f"""
            <div class="rp-metric-block">
                <div class="rp-metric-label">ASPECT MENTIONS</div>
                <div class="rp-metric-value">{total_mentions:,}</div>
                <div class="rp-metric-sub">Extracted phrase instances</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"""
            <div class="rp-metric-block">
                <div class="rp-metric-label">FEATURE CLASSES</div>
                <div class="rp-metric-value">{unique_aspects}</div>
                <div class="rp-metric-sub">Categorized components</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f"""
            <div class="rp-metric-block">
                <div class="rp-metric-label">REVIEW COVERAGE</div>
                <div class="rp-metric-value">{coverage_pct}%</div>
                <div class="rp-metric-sub">{reviews_with_aspect:,} reviews contain aspects</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # Net Sentiment Score Chart
    with st.container(border=True):
        st.markdown(
            """
            <div class="rp-chart-header">
                <div class="rp-chart-title">Net Aspect Sentiment Score</div>
                <div class="rp-chart-subtitle">Calculated as (Positive Mentions - Negative Mentions) / Total Mentions. Normalized from -1.00 to +1.00.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if "net_sentiment" in aspect_summary.columns:
            ns_df = aspect_summary.sort_values("net_sentiment", ascending=True).copy()
            ns_df["color"] = ns_df["net_sentiment"].apply(lambda v: COLOR_POSITIVE if v >= 0 else COLOR_NEGATIVE)
            fig_net = px.bar(
                ns_df,
                x="net_sentiment",
                y="aspect",
                orientation="h",
                color="color",
                color_discrete_map="identity",
                labels={"net_sentiment": "Net Sentiment Score", "aspect": "Feature Aspect"},
            )
            fig_net.add_vline(x=0, line_dash="solid", line_color="#4B5563", line_width=1.5)
            fig_net = apply_chart_theme(fig_net, height=290, x_title="Net Sentiment Score (-1.00 to +1.00)", y_title="Feature Aspect")
            fig_net.update_layout(showlegend=False)
            st.plotly_chart(fig_net, use_container_width=True)

    # Praised vs Criticized Breakdown
    p_col, c_col = st.columns(2)
    with p_col:
        with st.container(border=True):
            st.markdown(
                """
                <div class="rp-chart-header">
                    <div class="rp-chart-title">Top Praised Product Features</div>
                    <div class="rp-chart-subtitle">Aspects receiving the highest frequency of positive customer feedback.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            pos_aspects = (
                aspects_df[aspects_df["sentiment"] == "Positive"]
                .groupby("aspect")
                .size()
                .sort_values(ascending=True)
                .reset_index(name="Count")
                .tail(7)
            )
            if not pos_aspects.empty:
                fig_p = px.bar(
                    pos_aspects,
                    x="Count",
                    y="aspect",
                    orientation="h",
                    color_discrete_sequence=[COLOR_POSITIVE],
                    labels={"Count": "Positive Mentions", "aspect": "Feature Aspect"},
                )
                fig_p = apply_chart_theme(fig_p, height=260, x_title="Positive Mentions", y_title="Feature Aspect")
                fig_p.update_layout(showlegend=False)
                st.plotly_chart(fig_p, use_container_width=True)
            else:
                st.info("No positive aspect mentions detected.")

    with c_col:
        with st.container(border=True):
            st.markdown(
                """
                <div class="rp-chart-header">
                    <div class="rp-chart-title">Top Criticized Product Issues</div>
                    <div class="rp-chart-subtitle">Aspects receiving the highest frequency of negative customer feedback.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            neg_aspects = (
                aspects_df[aspects_df["sentiment"] == "Negative"]
                .groupby("aspect")
                .size()
                .sort_values(ascending=True)
                .reset_index(name="Count")
                .tail(7)
            )
            if not neg_aspects.empty:
                fig_c = px.bar(
                    neg_aspects,
                    x="Count",
                    y="aspect",
                    orientation="h",
                    color_discrete_sequence=[COLOR_NEGATIVE],
                    labels={"Count": "Negative Mentions", "aspect": "Feature Aspect"},
                )
                fig_c = apply_chart_theme(fig_c, height=260, x_title="Negative Mentions", y_title="Feature Aspect")
                fig_c.update_layout(showlegend=False)
                st.plotly_chart(fig_c, use_container_width=True)
            else:
                st.info("No negative aspect mentions detected.")

    # Detailed Aspect Table
    with st.container(border=True):
        st.markdown(
            """
            <div class="rp-chart-header">
                <div class="rp-chart-title">Aspect Mentions Audit Table</div>
                <div class="rp-chart-subtitle">Individual feature extractions, assigned sentiment, extracted context, and confidence.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        cols_to_show = [c for c in ["review_id", "aspect", "sentiment", "phrase", "confidence"] if c in aspects_df.columns]
        render_professional_table(aspects_df[cols_to_show].head(150), max_height=380)


# ── Page 4: Topic Explorer ────────────────────────────────────────────────────

def render_topic_page(df: pd.DataFrame, topic_summary: pd.DataFrame, keywords_df: pd.DataFrame):
    """Renders unsupervised topic discovery, 2D semantic projections, and keywords."""
    st.markdown(
        """
        <div class="rp-section-header">
            <div class="rp-section-title">Topic Explorer</div>
            <div class="rp-section-subtitle">Unsupervised clustering and semantic topic representations across the customer review base.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if topic_summary is None or topic_summary.empty:
        st.info("Topic modeling pipeline produced no clustered results.")
        return

    # Section 1: Topic Frequency
    with st.container(border=True):
        st.markdown(
            """
            <div class="rp-chart-header">
                <div class="rp-chart-title">Discovered Topic Volume</div>
                <div class="rp-chart-subtitle">Review count distribution across automatically discovered semantic topics.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        t_plot = topic_summary.head(10).sort_values("count", ascending=True)
        fig_tf = px.bar(
            t_plot,
            x="count",
            y="topic_label",
            orientation="h",
            color_discrete_sequence=[COLOR_PRIMARY],
            labels={"count": "Review Count", "topic_label": "Topic"},
        )
        fig_tf = apply_chart_theme(fig_tf, height=290, x_title="Review Count", y_title="Topic")
        fig_tf.update_layout(showlegend=False)
        st.plotly_chart(fig_tf, use_container_width=True)

    # Section 2: 2D Projection PCA Cluster Scatter
    if "embed_x" in df.columns and "embed_y" in df.columns:
        with st.container(border=True):
            st.markdown(
                """
                <div class="rp-chart-header">
                    <div class="rp-chart-title">Semantic Review Space (PCA 2D Projection)</div>
                    <div class="rp-chart-subtitle">Dimensionality reduction projecting high-dimensional text embeddings into a 2D coordinate plane.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            scatter_df = df.copy()
            color_column = "topic_label" if "topic_label" in scatter_df.columns else "cluster"
            if "original_review" in scatter_df.columns:
                scatter_df["snippet"] = scatter_df["original_review"].astype(str).str[:140] + "..."

            hover_data = ["snippet"]
            if "sentiment" in scatter_df.columns:
                hover_data.append("sentiment")

            fig_pca = px.scatter(
                scatter_df,
                x="embed_x",
                y="embed_y",
                color=color_column,
                color_discrete_sequence=PALETTE_TOPICS,
                hover_data=hover_data,
                opacity=0.8,
                labels={"embed_x": "Principal Component 1", "embed_y": "Principal Component 2"},
            )
            fig_pca.update_traces(marker=dict(size=7, line=dict(width=0.75, color="#FFFFFF")))
            fig_pca = apply_chart_theme(fig_pca, height=360, x_title="Principal Component 1", y_title="Principal Component 2")
            st.plotly_chart(fig_pca, use_container_width=True)

    # Section 3: Topic Details & Keywords
    col_t_detail, col_kw = st.columns([3, 2])

    with col_t_detail:
        with st.container(border=True):
            st.markdown(
                """
                <div class="rp-chart-header">
                    <div class="rp-chart-title">Topic Representation & Snippets</div>
                    <div class="rp-chart-subtitle">Characteristic key phrases and representative customer quotes.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            for _, row in topic_summary.head(5).iterrows():
                label = row.get("topic_label", "Topic")
                count = int(row.get("count", 0))
                kws = row.get("keywords", "N/A")
                rep_text = row.get("representative", "")

                with st.expander(f"{label} ({count:,} reviews)"):
                    st.markdown(f"**Key Terms:** `{kws}`")
                    if rep_text:
                        st.markdown("**Representative Review Snippets:**")
                        snippets = str(rep_text).split(" | ")
                        for snip in snippets[:2]:
                            if snip.strip():
                                st.markdown(f"> *{snip.strip()[:200]}*")

    with col_kw:
        with st.container(border=True):
            st.markdown(
                """
                <div class="rp-chart-header">
                    <div class="rp-chart-title">Top Distinctive Keywords (TF-IDF)</div>
                    <div class="rp-chart-subtitle">Statistically salient terms across dataset.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if keywords_df is not None and not keywords_df.empty:
                score_col = "score" if "score" in keywords_df.columns else "tfidf_score"
                kw_head = keywords_df.head(12).sort_values(score_col, ascending=True)
                fig_kw = px.bar(
                    kw_head,
                    x=score_col,
                    y="keyword",
                    orientation="h",
                    color_discrete_sequence=[COLOR_PRIMARY],
                    labels={score_col: "Relevance Score", "keyword": "Keyword"},
                )
                fig_kw = apply_chart_theme(fig_kw, height=310, x_title="Relevance Score", y_title="Keyword")
                fig_kw.update_layout(showlegend=False)
                st.plotly_chart(fig_kw, use_container_width=True)
            else:
                st.info("Keyword extraction data unavailable.")


# ── Page 5: Semantic Explorer ─────────────────────────────────────────────────

def render_semantic_explorer_page(df: pd.DataFrame, searcher: SemanticSearch):
    """Renders high-precision neural semantic search interface."""
    st.markdown(
        """
        <div class="rp-section-header">
            <div class="rp-section-title">Semantic Explorer</div>
            <div class="rp-section-subtitle">Vector-based conceptual search across customer reviews using cosine similarity over embeddings.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        search_col, slider_col = st.columns([4, 1])
        with search_col:
            query = st.text_input(
                "Semantic Query",
                placeholder="Type customer intent (e.g., poor battery after update, exceptional build quality, delayed delivery)",
                label_visibility="collapsed",
            )
        with slider_col:
            top_k = st.slider("Top Results", min_value=3, max_value=30, value=8)

    if not query.strip():
        st.markdown(
            """
            <div class="rp-callout">
                <div style="font-size: 13px; font-weight: 700; color: #9E4226;">Semantic Search Capability</div>
                <div style="font-size: 12px; color: #374151; margin-top: 4px;">
                    This interface evaluates conceptual semantic similarity rather than exact keyword matching. Try querying customer pain points:
                </div>
                <ul style="font-size: 12px; color: #4B5563; margin-top: 6px; margin-bottom: 0;">
                    <li><i>software freezes and connectivity drops</i></li>
                    <li><i>outstanding battery longevity</i></li>
                    <li><i>packaging was severely damaged upon arrival</i></li>
                    <li><i>great value for money compared to competitors</i></li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    with st.spinner("Searching semantic vector space..."):
        results = searcher.search(query.strip(), top_k=top_k)

    if results.empty:
        st.warning("No matches found above the similarity threshold.")
        return

    st.markdown(
        f"""
        <div style="margin-top: 10px; margin-bottom: 12px; font-size: 13px; font-weight: 500; color: #4B5563;">
            Surfaced <span style="font-weight: 700; color: #111827;">{len(results)}</span> semantic matches for query: <i>"{query}"</i>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for idx, row in results.iterrows():
        sim_score = float(row.get("similarity_score", 0.0))
        sentiment = row.get("sentiment", "Neutral")
        emotion = row.get("emotion", "N/A")
        topic = row.get("topic_label", "N/A")
        rating = row.get("rating")
        review_text = str(row.get("original_review", row.get("cleaned_review", "")))
        aspects_val = row.get("aspects_summary", "")

        if sentiment == "Positive":
            badge_class = "rp-badge-positive"
        elif sentiment == "Negative":
            badge_class = "rp-badge-negative"
        else:
            badge_class = "rp-badge-neutral"

        rating_tag = f"Rating: {rating} / 5.0 | " if pd.notna(rating) else ""
        topic_tag = f"Topic: {topic} | " if topic and topic != "N/A" else ""
        emotion_tag = f"Emotion: {emotion} | " if emotion and emotion != "N/A" else ""

        st.markdown(
            f"""
            <div style="background: #FFFFFF; border: 1px solid #EAE4DC; border-radius: 6px; padding: 14px 18px; margin-bottom: 10px; box-shadow: 0 1px 2px rgba(0,0,0,0.02);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <div>
                        <span class="rp-badge {badge_class}">{sentiment}</span>
                        <span style="font-size: 12px; color: #4B5563; margin-left: 8px;">{rating_tag}{topic_tag}{emotion_tag}Cosine Similarity: <b>{sim_score:.4f}</b></span>
                    </div>
                </div>
                <div style="font-size: 13px; color: #1F2937; line-height: 1.5; border-left: 3px solid #D6D0C7; padding-left: 12px; margin-top: 6px;">
                    {html.escape(review_text)}
                </div>
                {"<div style='font-size: 11px; color: #6B7280; margin-top: 8px;'>Aspects: " + html.escape(str(aspects_val)) + "</div>" if aspects_val and str(aspects_val).strip() else ""}
            </div>
            """,
            unsafe_allow_html=True,
        )


# ── Page 6: Review Explorer ───────────────────────────────────────────────────

def render_review_explorer_page(df: pd.DataFrame, aspects_df: pd.DataFrame):
    """Renders full tabular inspection, multi-criteria filtering, and data export."""
    st.markdown(
        """
        <div class="rp-section-header">
            <div class="rp-section-title">Review Explorer</div>
            <div class="rp-section-subtitle">Multi-parameter filter controls, record audit, and CSV export functionality.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    filtered = df.copy()

    # Filter Controls Block inside an analytical container
    with st.container(border=True):
        st.markdown(
            """
            <div class="rp-chart-header">
                <div class="rp-chart-title">Filter Parameters</div>
                <div class="rp-chart-subtitle">Refine review cohort by sentiment, emotion, topic, uncertainty, and rating.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        f_row1_c1, f_row1_c2, f_row1_c3, f_row1_c4 = st.columns(4)

        with f_row1_c1:
            if "sentiment" in df.columns:
                sent_opts = ["All"] + sorted(df["sentiment"].dropna().unique().tolist())
                sel_sent = st.selectbox("Sentiment Filter", sent_opts)
                if sel_sent != "All":
                    filtered = filtered[filtered["sentiment"] == sel_sent]

        with f_row1_c2:
            if "emotion" in df.columns:
                emo_opts = ["All"] + sorted(df["emotion"].dropna().unique().tolist())
                sel_emo = st.selectbox("Emotion Filter", emo_opts)
                if sel_emo != "All":
                    filtered = filtered[filtered["emotion"] == sel_emo]

        with f_row1_c3:
            if "topic_label" in df.columns:
                top_opts = ["All"] + sorted(df["topic_label"].dropna().unique().tolist())
                sel_topic = st.selectbox("Topic Filter", top_opts)
                if sel_topic != "All":
                    filtered = filtered[filtered["topic_label"] == sel_topic]

        with f_row1_c4:
            if "uncertainty_flag" in df.columns:
                unc_opts = ["All", "Uncertain Only", "Certain Only"]
                sel_unc = st.selectbox("Uncertainty Status", unc_opts)
                if sel_unc == "Uncertain Only":
                    filtered = filtered[filtered["uncertainty_flag"] == True]
                elif sel_unc == "Certain Only":
                    filtered = filtered[filtered["uncertainty_flag"] == False]

        f_row2_c1, f_row2_c2 = st.columns(2)
        with f_row2_c1:
            if "rating" in df.columns and df["rating"].notna().sum() > 0:
                min_r = float(df["rating"].min())
                max_r = float(df["rating"].max())
                if min_r < max_r:
                    r_range = st.slider("Rating Score Range", min_r, max_r, (min_r, max_r))
                    filtered = filtered[
                        (filtered["rating"].isna()) |
                        ((filtered["rating"] >= r_range[0]) & (filtered["rating"] <= r_range[1]))
                    ]

        with f_row2_c2:
            search_query = st.text_input("Review Text Search", placeholder="Substring or keyword match...")
            if search_query.strip():
                t_col = "original_review" if "original_review" in filtered.columns else "cleaned_review"
                filtered = filtered[filtered[t_col].astype(str).str.contains(search_query.strip(), case=False, na=False)]

    # Result count indicator
    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 10px; margin-bottom: 8px;">
            <div style="font-size: 13px; font-weight: 700; color: #111827;">
                Showing {len(filtered):,} of {len(df):,} reviews matching criteria
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Column selection
    candidate_cols = [
        "review_id",
        "original_review",
        "rating",
        "sentiment",
        "sentiment_confidence",
        "emotion",
        "aspects_summary",
        "topic_label",
        "uncertainty_flag",
    ]
    show_cols = [c for c in candidate_cols if c in filtered.columns]
    if not show_cols:
        show_cols = list(filtered.columns[:7])

    # Visible Light Table
    with st.container(border=True):
        render_professional_table(filtered[show_cols].head(300), max_height=520)

    # Download button
    csv_bytes = filtered[show_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Export Filtered Reviews (CSV)",
        data=csv_bytes,
        file_name="reviewpulse_filtered_reviews.csv",
        mime="text/csv",
    )


# ── Page 7: Product Insights ──────────────────────────────────────────────────

def render_insights_page(
    insights: dict,
    df: pd.DataFrame,
    aspects_df: pd.DataFrame,
    topic_summary: pd.DataFrame,
):
    """Renders the executive briefing and strategic synthesis report."""
    st.markdown(
        """
        <div class="rp-section-header">
            <div class="rp-section-title">Product Intelligence Insights</div>
            <div class="rp-section-subtitle">Automated strategic executive summary derived strictly from empirical review data.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    pos_pct = insights.get("positive_pct")
    neg_pct = insights.get("negative_pct")

    # Executive Verdict Banner
    if pos_pct is not None and neg_pct is not None:
        if pos_pct >= 60:
            st.markdown(
                f"""
                <div class="rp-callout rp-callout-success">
                    <div style="font-size: 14px; font-weight: 700; color: #166534;">Strategic Verdict: Strong Positive Market Reception</div>
                    <div style="font-size: 13px; color: #14532D; margin-top: 3px;">
                        Customer sentiment is predominantly positive at {pos_pct}% of total processed reviews. Key product features demonstrate solid customer alignment.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif neg_pct >= 35:
            st.markdown(
                f"""
                <div class="rp-callout rp-callout-warning">
                    <div style="font-size: 14px; font-weight: 700; color: #991B1B;">Strategic Verdict: Critical Issue Resolution Recommended</div>
                    <div style="font-size: 13px; color: #7F1D1D; margin-top: 3px;">
                        Elevated negative sentiment observed ({neg_pct}% of total reviews). Focused intervention required on frequently criticized product aspects.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="rp-callout">
                    <div style="font-size: 14px; font-weight: 700; color: #9E4226;">Strategic Verdict: Balanced Product Reception</div>
                    <div style="font-size: 13px; color: #374151; margin-top: 3px;">
                        Customer sentiment exhibits parity ({pos_pct}% positive vs {neg_pct}% negative). Priority should be placed on mitigating identified failure points.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Executive Grid
    col_str, col_weak = st.columns(2)

    with col_str:
        with st.container(border=True):
            st.markdown(
                """
                <div style="font-size: 11px; font-weight: 700; text-transform: uppercase; color: #16A34A; letter-spacing: 0.06em;">Product Strengths</div>
                <div style="font-size: 16px; font-weight: 700; color: #111827; margin-top: 2px; margin-bottom: 12px;">Most Praised Aspects</div>
                """,
                unsafe_allow_html=True,
            )
            praised = insights.get("most_praised_aspects", {})
            if praised:
                for asp, count in list(praised.items())[:5]:
                    name = asp.replace("_", " ").title()
                    st.markdown(
                        f"""
                        <div style="display: flex; justify-content: space-between; padding: 7px 0; border-bottom: 1px solid #F0ECE6; font-size: 13px;">
                            <span style="font-weight: 500; color: #1F2937;">{name}</span>
                            <span class="rp-badge rp-badge-positive">{count:,} positive mentions</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("No praised aspects recorded.")

    with col_weak:
        with st.container(border=True):
            st.markdown(
                """
                <div style="font-size: 11px; font-weight: 700; text-transform: uppercase; color: #DC2626; letter-spacing: 0.06em;">Improvement Priorities</div>
                <div style="font-size: 16px; font-weight: 700; color: #111827; margin-top: 2px; margin-bottom: 12px;">Most Criticized Aspects</div>
                """,
                unsafe_allow_html=True,
            )
            criticized = insights.get("most_criticized_aspects", {})
            if criticized:
                for asp, count in list(criticized.items())[:5]:
                    name = asp.replace("_", " ").title()
                    st.markdown(
                        f"""
                        <div style="display: flex; justify-content: space-between; padding: 7px 0; border-bottom: 1px solid #F0ECE6; font-size: 13px;">
                            <span style="font-weight: 500; color: #1F2937;">{name}</span>
                            <span class="rp-badge rp-badge-negative">{count:,} negative mentions</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("No criticized aspects recorded.")

    # Secondary Insights Grid: Topics & Uncertainties
    c_top, c_unc = st.columns(2)

    with c_top:
        with st.container(border=True):
            st.markdown(
                """
                <div style="font-size: 11px; font-weight: 700; text-transform: uppercase; color: #C25E3E; letter-spacing: 0.06em;">Discussion Themes</div>
                <div style="font-size: 16px; font-weight: 700; color: #111827; margin-top: 2px; margin-bottom: 12px;">Top Discovered Topics</div>
                """,
                unsafe_allow_html=True,
            )
            top_topics = insights.get("top_topics", [])
            if top_topics:
                for item in top_topics[:5]:
                    lbl = item.get("topic_label", "Unknown")
                    cnt = item.get("count", 0)
                    st.markdown(
                        f"""
                        <div style="display: flex; justify-content: space-between; padding: 7px 0; border-bottom: 1px solid #F0ECE6; font-size: 13px;">
                            <span style="font-weight: 500; color: #1F2937;">{lbl}</span>
                            <span style="color: #4B5563; font-weight: 600;">{cnt:,} reviews</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("Topic summaries unavailable.")

    with c_unc:
        with st.container(border=True):
            unc_cnt = insights.get("uncertain_count", 0)
            unc_pct = insights.get("uncertain_pct", 0.0)
            st.markdown(
                f"""
                <div style="font-size: 11px; font-weight: 700; text-transform: uppercase; color: #D97706; letter-spacing: 0.06em;">Quality Assurance</div>
                <div style="font-size: 16px; font-weight: 700; color: #111827; margin-top: 2px; margin-bottom: 8px;">Uncertainty & Model Disagreement</div>
                <div style="font-size: 13px; color: #374151; line-height: 1.5; margin-bottom: 8px;">
                    <b>{unc_cnt:,} reviews ({unc_pct}%)</b> flagged due to disagreement between VADER and DistilBERT or sub-threshold confidence (< 0.65).
                </div>
                """,
                unsafe_allow_html=True,
            )
            if unc_cnt > 0 and "uncertainty_flag" in df.columns:
                u_cols = [c for c in ["review_id", "original_review", "vader_sentiment", "transformer_sentiment"] if c in df.columns]
                render_professional_table(df[df["uncertainty_flag"] == True][u_cols].head(8), max_height=260)


# ── Application Dispatcher ────────────────────────────────────────────────────

def main():
    """Application entrypoint."""
    uploaded_file, local_path, data_source, use_transformer, use_emotions, run_btn, nav_page = render_sidebar()

    if run_btn:
        reset_session()
        run_analysis_pipeline(uploaded_file, local_path, data_source, use_transformer, use_emotions)

    pipeline_done = st.session_state.get("pipeline_done", False)

    if not pipeline_done:
        render_empty_state()
        return

    # Extract session data
    df = st.session_state["df"]
    aspects_df = st.session_state["aspects_df"]
    aspect_summary = st.session_state["aspect_summary"]
    topic_summary = st.session_state["topic_summary"]
    keywords_df = st.session_state["keywords_df"]
    insights = st.session_state["insights"]
    searcher = st.session_state["searcher"]

    # Dispatch to chosen section
    if nav_page == "Overview":
        render_overview_page(df, insights, topic_summary, aspect_summary)
    elif nav_page == "Sentiment Intelligence":
        render_sentiment_page(df)
    elif nav_page == "Aspect Intelligence":
        render_aspect_page(df, aspects_df, aspect_summary)
    elif nav_page == "Topic Explorer":
        render_topic_page(df, topic_summary, keywords_df)
    elif nav_page == "Semantic Explorer":
        render_semantic_explorer_page(df, searcher)
    elif nav_page == "Review Explorer":
        render_review_explorer_page(df, aspects_df)
    elif nav_page == "Product Insights":
        render_insights_page(insights, df, aspects_df, topic_summary)


if __name__ == "__main__":
    main()
