"""
ReviewPulse — Streamlit Application Entry Point

Run with:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

# ── Dynamic project root resolution ───────────────────────────────────────────
_CURRENT_DIR = Path(__file__).resolve().parent
if (_CURRENT_DIR / "src").exists():
    _REPO = _CURRENT_DIR
elif (_CURRENT_DIR.parent / "src").exists():
    _REPO = _CURRENT_DIR.parent
else:
    _REPO = _CURRENT_DIR

if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import streamlit as st

st.set_page_config(
    page_title="ReviewPulse",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── imports after sys.path is set ─────────────────────────────────────────────
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

# ── page imports using importlib ──────────────────────────────────────────────
import importlib.util as _ilu

def _load_page(name: str):
    spec = _ilu.spec_from_file_location(name, _REPO / "app" / "pages" / f"{name}.py")
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

overview        = _load_page("overview")
sentiment       = _load_page("sentiment")
aspects         = _load_page("aspects")
topics          = _load_page("topics")
semantic_search = _load_page("semantic_search")
reviews_page    = _load_page("reviews")
insights_page   = _load_page("insights")

# ── session-state keys ────────────────────────────────────────────────────────
SS_KEYS = [
    "df", "aspects_df", "aspect_summary", "topic_summary",
    "keywords_df", "embeddings", "embed_method",
    "insights", "searcher", "pipeline_done",
    "use_transformer", "use_emotions",
]


def _reset_session():
    for k in SS_KEYS:
        if k in st.session_state:
            del st.session_state[k]


# ── sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar() -> tuple:
    st.sidebar.image(
        "https://img.shields.io/badge/ReviewPulse-AI%20Intelligence-blueviolet?style=for-the-badge",
        use_container_width=True,
    )
    st.sidebar.title("🧠 ReviewPulse")
    st.sidebar.caption("From customer reviews to actionable product intelligence.")
    st.sidebar.divider()

    st.sidebar.subheader("📂 Data Source")
    data_source = st.sidebar.radio(
        "Choose source",
        ["Use sample dataset", "Upload CSV", "Local file path"],
        index=0,
        label_visibility="collapsed",
    )

    uploaded_file = None
    local_path = None

    if data_source == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader(
            "Upload reviews CSV", type=["csv"],
            label_visibility="collapsed",
        )
    elif data_source == "Local file path":
        local_path = st.sidebar.text_input(
            "Path to CSV",
            placeholder="data/raw/sample_reviews.csv",
        )
    else:
        st.sidebar.info("Using built-in sample dataset (50 curated product reviews with ratings & dates).")
        local_path = "__sample__"

    st.sidebar.subheader("⚙️ Pipeline Options")
    use_transformer = st.sidebar.checkbox("Transformer Sentiment", value=True)
    use_emotions    = st.sidebar.checkbox("Emotion Detection",     value=True)

    col1, col2 = st.sidebar.columns(2)
    with col1:
        run_btn = st.sidebar.button("▶ Run Analysis", use_container_width=True, type="primary")
    with col2:
        reset_btn = st.sidebar.button("🔄 Reset",      use_container_width=True)

    if reset_btn:
        _reset_session()
        st.rerun()

    st.sidebar.divider()
    st.sidebar.subheader("📑 Navigation")
    page = st.sidebar.radio(
        "Page",
        ["Overview", "Sentiment", "Aspects", "Topics", "Semantic Search", "Review Explorer", "Insights"],
        label_visibility="collapsed",
    )

    return uploaded_file, local_path, data_source, use_transformer, use_emotions, run_btn, page


# ── pipeline runner ────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_cached(source_bytes: bytes, filename: str):
    """Cache-wrapped CSV loader."""
    return load_reviews(io.BytesIO(source_bytes))


@st.cache_data(show_spinner=False)
def _load_from_path(path: str):
    return load_reviews(path)


def run_pipeline(
    uploaded_file,
    local_path: str,
    data_source: str,
    use_transformer: bool,
    use_emotions: bool,
):
    """Run the full NLP pipeline and store results in session_state."""

    with st.spinner("📥 Loading data…"):
        try:
            if data_source == "Upload CSV" and uploaded_file is not None:
                file_bytes = uploaded_file.read()
                df = _load_cached(file_bytes, uploaded_file.name)
            elif data_source == "Local file path" and local_path:
                df = _load_from_path(local_path)
            elif data_source == "Use sample dataset" or local_path == "__sample__":
                sample_file = _REPO / "data" / "raw" / "sample_reviews.csv"
                df = _load_from_path(str(sample_file))
            else:
                st.warning("Please select a data source first.")
                return
        except DataLoadError as exc:
            st.error(f"❌ Data loading failed: {exc}")
            return
        except Exception as exc:
            st.error(f"❌ Unexpected error: {exc}")
            return

    progress = st.progress(0, "Preprocessing…")

    with st.spinner("🧹 Preprocessing text…"):
        df = preprocess_dataframe(df)
    progress.progress(15, "Sentiment analysis…")

    with st.spinner("🔍 Analysing sentiment…"):
        df = run_sentiment_pipeline(df, use_transformer=use_transformer)
    progress.progress(35, "Detecting emotions…")

    if use_emotions:
        with st.spinner("😊 Detecting emotions…"):
            df = run_emotion_pipeline(df)
    progress.progress(50, "Extracting aspects…")

    with st.spinner("🎯 Extracting aspects…"):
        extractor = AspectExtractor()
        aspects_df = extractor.extract_from_dataframe(df)
        df = extractor.add_aspects_summary(df, aspects_df)
        aspect_summary = get_aspect_sentiment_summary(aspects_df)
    progress.progress(65, "Modelling topics…")

    with st.spinner("🧠 Modelling topics…"):
        df, topic_summary, _ = run_topic_pipeline(df)
    progress.progress(75, "Extracting keywords…")

    with st.spinner("🔑 Extracting keywords…"):
        keywords_df = run_keyword_pipeline(df)
    progress.progress(85, "Computing embeddings & clusters…")

    with st.spinner("🔵 Computing embeddings & clusters…"):
        df, embeddings, embed_method = run_clustering_pipeline(df)
    progress.progress(95, "Generating insights…")

    with st.spinner("💡 Generating insights…"):
        insights = generate_insights(df, aspects_df=aspects_df, topic_summary=topic_summary)
        searcher = SemanticSearch(embeddings, df, embed_method)
    progress.progress(100, "Done!")

    # Store in session
    st.session_state["df"]             = df
    st.session_state["aspects_df"]     = aspects_df
    st.session_state["aspect_summary"] = aspect_summary
    st.session_state["topic_summary"]  = topic_summary
    st.session_state["keywords_df"]    = keywords_df
    st.session_state["embeddings"]     = embeddings
    st.session_state["embed_method"]   = embed_method
    st.session_state["insights"]       = insights
    st.session_state["searcher"]       = searcher
    st.session_state["pipeline_done"]  = True
    st.session_state["use_transformer"] = use_transformer
    st.session_state["use_emotions"]   = use_emotions

    st.success(f"✅ Analysis complete! {len(df):,} reviews processed.")
    st.rerun()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    uploaded_file, local_path, data_source, use_transformer, use_emotions, run_btn, page = render_sidebar()

    if run_btn:
        _reset_session()
        run_pipeline(uploaded_file, local_path, data_source, use_transformer, use_emotions)

    pipeline_done = st.session_state.get("pipeline_done", False)

    if not pipeline_done:
        st.title("🧠 ReviewPulse")
        st.subheader("AI-Powered Product Review Intelligence Platform")
        st.divider()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("**Step 1**\n\nUpload your reviews CSV or click ▶ Run Analysis with the sample dataset.")
        with col2:
            st.info("**Step 2**\n\nConfigure pipeline options in the sidebar.")
        with col3:
            st.info("**Step 3**\n\nClick ▶ Run Analysis to start the NLP pipeline.")

        st.divider()
        st.markdown("### What ReviewPulse analyses:")
        cols = st.columns(4)
        with cols[0]:
            st.markdown("🔍 **Sentiment**\n\nVADER + Transformer with agreement detection")
        with cols[1]:
            st.markdown("🎯 **Aspects**\n\nAspect-based sentiment for product features")
        with cols[2]:
            st.markdown("🧠 **Topics**\n\nUnsupervised topic discovery (BERTopic / LDA)")
        with cols[3]:
            st.markdown("🔍 **Semantic Search**\n\nNatural language search over reviews")
        return

    # ── dispatch to page ──────────────────────────────────────────────────────
    df             = st.session_state["df"]
    aspects_df     = st.session_state["aspects_df"]
    aspect_summary = st.session_state["aspect_summary"]
    topic_summary  = st.session_state["topic_summary"]
    keywords_df    = st.session_state["keywords_df"]
    insights       = st.session_state["insights"]
    searcher       = st.session_state["searcher"]

    if page == "Overview":
        overview.render(df, insights, topic_summary, aspect_summary)
    elif page == "Sentiment":
        sentiment.render(df)
    elif page == "Aspects":
        aspects.render(df, aspects_df, aspect_summary)
    elif page == "Topics":
        topics.render(df, topic_summary, keywords_df)
    elif page == "Semantic Search":
        semantic_search.render(df, searcher)
    elif page == "Review Explorer":
        reviews_page.render(df, aspects_df)
    elif page == "Insights":
        insights_page.render(insights, df, aspects_df, topic_summary)


if __name__ == "__main__":
    main()
