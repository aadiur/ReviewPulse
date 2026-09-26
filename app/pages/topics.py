"""ReviewPulse — Topics Explorer Page."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st


def render(df: pd.DataFrame, topic_summary: pd.DataFrame, keywords_df: pd.DataFrame):
    st.title("🧠 Topic Explorer")
    st.divider()

    if topic_summary is None or topic_summary.empty:
        st.warning("Topic model produced no results.")
        return

    # ── Topic frequency ────────────────────────────────────────────────────────
    st.subheader("Topic Frequency")
    fig = px.bar(
        topic_summary.head(12),
        x="count", y="topic_label",
        orientation="h", color="count",
        color_continuous_scale="Viridis",
        labels={"count": "Reviews", "topic_label": "Topic"},
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"},
                      coloraxis_showscale=False, margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Topic detail cards ─────────────────────────────────────────────────────
    st.subheader("Topic Details")
    for _, row in topic_summary.head(10).iterrows():
        with st.expander(f"📌 {row['topic_label']}  —  {int(row['count']):,} reviews"):
            st.markdown(f"**Keywords:** `{row.get('keywords', 'N/A')}`")
            if row.get("representative"):
                st.markdown("**Representative reviews:**")
                for snippet in str(row["representative"]).split(" | "):
                    if snippet.strip():
                        st.markdown(f"> {snippet.strip()[:200]}")

    st.divider()

    # ── Cluster scatter (embed_x, embed_y) ────────────────────────────────────
    if "embed_x" in df.columns and "embed_y" in df.columns:
        st.subheader("🔵 Review Cluster Map (PCA 2D)")
        plot_df = df.copy()
        color_col = "topic_label" if "topic_label" in plot_df.columns else "cluster"
        hover_cols = ["original_review"] if "original_review" in plot_df.columns else []
        if "sentiment" in plot_df.columns:
            hover_cols.append("sentiment")

        # truncate long reviews for hover
        if "original_review" in plot_df.columns:
            plot_df["review_snippet"] = plot_df["original_review"].str[:120]
            hover_cols = ["review_snippet"] + [c for c in hover_cols if c != "original_review"]

        fig = px.scatter(
            plot_df,
            x="embed_x",
            y="embed_y",
            color=color_col,
            hover_data=hover_cols,
            opacity=0.6,
            labels={"embed_x": "PC1", "embed_y": "PC2"},
        )
        fig.update_traces(marker=dict(size=5))
        fig.update_layout(margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Each point is a review projected into 2D via PCA. Colours indicate topics/clusters.")

    st.divider()

    # ── Keywords ───────────────────────────────────────────────────────────────
    st.subheader("🔑 Top Keywords & Keyphrases")
    if keywords_df is not None and not keywords_df.empty:
        method = keywords_df["method"].iloc[0] if "method" in keywords_df.columns else "TF-IDF"
        st.caption(f"Method: **{method}**")
        score_col = "score" if "score" in keywords_df.columns else "tfidf_score"
        fig = px.bar(
            keywords_df.head(20),
            x=score_col, y="keyword", orientation="h",
            color=score_col, color_continuous_scale="Oranges",
            labels={score_col: "Score", "keyword": "Keyword"},
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"},
                          coloraxis_showscale=False, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Keyword data not available.")
