"""
ReviewPulse — Overview Page

Shows key metrics, sentiment distribution, top topics, and top aspects.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def render(df: pd.DataFrame, insights: dict, topic_summary: pd.DataFrame, aspect_summary: pd.DataFrame):
    st.title("📊 Overview")
    st.caption(f"Analysed **{insights['total_reviews']:,}** reviews")
    st.divider()

    # ── KPI row ────────────────────────────────────────────────────────────────
    cols = st.columns(5)
    cols[0].metric("Total Reviews", f"{insights['total_reviews']:,}")

    if insights.get("average_rating") is not None:
        cols[1].metric("Avg Rating", f"⭐ {insights['average_rating']}")
    else:
        cols[1].metric("Avg Rating", "N/A")

    pos = insights.get("positive_pct")
    neg = insights.get("negative_pct")
    if pos is not None:
        cols[2].metric("Positive", f"{pos}%", delta=f"{pos - neg:.1f}% vs Negative")
        cols[3].metric("Negative", f"{neg}%")
        cols[4].metric("Neutral",  f"{insights['neutral_pct']}%")
    else:
        cols[2].metric("Positive", "N/A")
        cols[3].metric("Negative", "N/A")
        cols[4].metric("Neutral",  "N/A")

    st.divider()

    # ── Sentiment + Topics row ─────────────────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Sentiment Distribution")
        if "sentiment" in df.columns:
            sent_counts = df["sentiment"].value_counts().reset_index()
            sent_counts.columns = ["Sentiment", "Count"]
            color_map = {"Positive": "#2ecc71", "Negative": "#e74c3c", "Neutral": "#95a5a6",
                         "Unknown": "#bdc3c7", "Unavailable": "#bdc3c7"}
            fig = px.pie(
                sent_counts,
                names="Sentiment",
                values="Count",
                color="Sentiment",
                color_discrete_map=color_map,
                hole=0.4,
            )
            fig.update_layout(margin=dict(t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Sentiment data not available.")

    with col_right:
        st.subheader("Top Topics")
        if topic_summary is not None and not topic_summary.empty:
            top5 = topic_summary.head(8)
            fig = px.bar(
                top5,
                x="count",
                y="topic_label",
                orientation="h",
                color="count",
                color_continuous_scale="Blues",
                labels={"count": "Reviews", "topic_label": "Topic"},
            )
            fig.update_layout(margin=dict(t=20, b=20), coloraxis_showscale=False, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Topic data not available.")

    st.divider()

    # ── Aspects + Emotion row ──────────────────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Aspect Sentiment Overview")
        if aspect_summary is not None and not aspect_summary.empty:
            plot_data = aspect_summary.head(10).melt(
                id_vars=["aspect"],
                value_vars=["Positive", "Negative", "Neutral"],
                var_name="Sentiment",
                value_name="Count",
            )
            color_map = {"Positive": "#2ecc71", "Negative": "#e74c3c", "Neutral": "#95a5a6"}
            fig = px.bar(
                plot_data,
                x="Count",
                y="aspect",
                color="Sentiment",
                orientation="h",
                color_discrete_map=color_map,
                barmode="stack",
            )
            fig.update_layout(margin=dict(t=20, b=20), yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No aspect data found. Aspect extraction found no matching keywords.")

    with col_b:
        st.subheader("Emotion Distribution")
        if "emotion" in df.columns:
            emo_counts = df["emotion"].value_counts().reset_index()
            emo_counts.columns = ["Emotion", "Count"]
            emo_colors = {
                "joy": "#f1c40f", "anger": "#e74c3c", "sadness": "#3498db",
                "fear": "#9b59b6", "surprise": "#1abc9c", "neutral": "#95a5a6",
                "disgust": "#e67e22",
            }
            fig = px.bar(
                emo_counts,
                x="Emotion",
                y="Count",
                color="Emotion",
                color_discrete_map=emo_colors,
            )
            fig.update_layout(margin=dict(t=20, b=20), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Emotion data not available.")

    # ── Uncertainty flag summary ────────────────────────────────────────────────
    st.divider()
    if "uncertainty_flag" in df.columns:
        n_uncertain = int(df["uncertainty_flag"].sum())
        pct = round(100 * n_uncertain / len(df), 1)
        if n_uncertain > 0:
            st.warning(
                f"⚠️ **{n_uncertain:,} reviews ({pct}%)** flagged as uncertain "
                "(VADER and transformer disagree or low confidence). "
                "These are worth manual review."
            )
        else:
            st.success("✅ VADER and transformer models agree on all reviews.")

    # ── Rating distribution (if available) ────────────────────────────────────
    if "rating" in df.columns and df["rating"].notna().sum() > 0:
        st.subheader("Rating Distribution")
        rating_df = df["rating"].dropna().astype(float)
        fig = px.histogram(
            rating_df,
            x=rating_df,
            nbins=20,
            labels={"x": "Rating"},
            color_discrete_sequence=["#3498db"],
        )
        fig.update_layout(bargap=0.1, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)
