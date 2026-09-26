"""ReviewPulse — Semantic Search Page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.search.semantic_search import SemanticSearch


def render(df: pd.DataFrame, searcher: SemanticSearch):
    st.title("🔎 Semantic Review Search")
    st.caption("Search reviews using natural language — powered by cosine similarity over embeddings.")
    st.divider()

    # ── Search box ─────────────────────────────────────────────────────────────
    query = st.text_input(
        "Enter a natural-language query",
        placeholder="e.g. customers complaining about battery life",
    )
    top_k = st.slider("Number of results", min_value=3, max_value=50, value=10)

    if not query.strip():
        st.info("Enter a query above to search.")
        with st.expander("Example queries"):
            st.markdown(
                "- customers praising sound quality\n"
                "- bad experience with delivery\n"
                "- great value for money\n"
                "- complaints about software bugs\n"
                "- positive experience with customer service"
            )
        return

    with st.spinner("Searching…"):
        results = searcher.search(query, top_k=top_k)

    if results.empty:
        st.warning("No results found.")
        return

    st.success(f"Found **{len(results)}** results for: *{query}*")
    st.divider()

    color_map = {"Positive": "🟢", "Negative": "🔴", "Neutral": "⚪"}

    for i, row in results.iterrows():
        sim = float(row.get("similarity_score", 0))
        sentiment = row.get("sentiment", "—")
        emotion   = row.get("emotion", "—")
        topic     = row.get("topic_label", "—")
        cluster   = row.get("cluster", "—")
        aspects   = row.get("aspects_summary", "—") or "—"
        review_id = row.get("review_id", i)

        badge = color_map.get(sentiment, "⚪")
        st.markdown(f"#### {badge} Result {i+1}  — Similarity: `{sim:.4f}`")

        cols = st.columns([4, 1])
        with cols[0]:
            review_text = str(row.get("original_review", row.get("cleaned_review", "N/A")))
            st.markdown(f"> {review_text[:600]}")
        with cols[1]:
            st.markdown(f"**Sentiment:** {sentiment}")
            if emotion and emotion != "—":
                st.markdown(f"**Emotion:** {emotion}")
            if topic and topic != "—":
                st.markdown(f"**Topic:** {topic}")
            if cluster != "—":
                st.markdown(f"**Cluster:** {cluster}")
            if aspects and aspects != "—":
                st.markdown(f"**Aspects:** {aspects}")
            if "rating" in row and pd.notna(row.get("rating")):
                st.markdown(f"**Rating:** ⭐ {row['rating']}")

        st.divider()
