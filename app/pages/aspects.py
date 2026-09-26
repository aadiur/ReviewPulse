"""ReviewPulse — Aspects Intelligence Page."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st


def render(df: pd.DataFrame, aspects_df: pd.DataFrame, aspect_summary: pd.DataFrame):
    st.title("🎯 Aspect Intelligence")
    st.divider()

    if aspects_df.empty:
        st.warning(
            "No aspect mentions were found in the dataset. "
            "The aspect extractor looks for keywords related to: "
            "battery, sound, display, camera, performance, price, build quality, "
            "software, delivery, customer service, size/weight, and connectivity."
        )
        return

    # ── KPIs ──────────────────────────────────────────────────────────────────
    total_mentions = len(aspects_df)
    unique_aspects = aspects_df["aspect"].nunique()
    n_reviews_with_aspect = aspects_df["review_id"].nunique()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Aspect Mentions", f"{total_mentions:,}")
    c2.metric("Unique Aspects Found", unique_aspects)
    c3.metric("Reviews with Aspects", f"{n_reviews_with_aspect:,}")
    st.divider()

    # ── Stacked sentiment bar ──────────────────────────────────────────────────
    st.subheader("Aspect Sentiment Breakdown")
    color_map = {"Positive": "#2ecc71", "Negative": "#e74c3c", "Neutral": "#95a5a6"}
    if not aspect_summary.empty:
        plot_cols = [c for c in ["Positive", "Negative", "Neutral"] if c in aspect_summary.columns]
        melt = aspect_summary.head(15).melt(
            id_vars=["aspect"], value_vars=plot_cols,
            var_name="Sentiment", value_name="Count",
        )
        fig = px.bar(
            melt, x="Count", y="aspect", color="Sentiment",
            color_discrete_map=color_map, orientation="h", barmode="stack",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Most praised / criticized ──────────────────────────────────────────────
    col_p, col_n = st.columns(2)

    with col_p:
        st.subheader("👍 Most Praised Aspects")
        pos_df = (
            aspects_df[aspects_df["sentiment"] == "Positive"]
            .groupby("aspect").size()
            .sort_values(ascending=False)
            .reset_index(name="Positive Mentions")
            .head(8)
        )
        if not pos_df.empty:
            fig = px.bar(pos_df, x="Positive Mentions", y="aspect", orientation="h",
                         color_discrete_sequence=["#2ecc71"])
            fig.update_layout(yaxis={"categoryorder": "total ascending"}, margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No positive aspect mentions found.")

    with col_n:
        st.subheader("👎 Most Criticised Aspects")
        neg_df = (
            aspects_df[aspects_df["sentiment"] == "Negative"]
            .groupby("aspect").size()
            .sort_values(ascending=False)
            .reset_index(name="Negative Mentions")
            .head(8)
        )
        if not neg_df.empty:
            fig = px.bar(neg_df, x="Negative Mentions", y="aspect", orientation="h",
                         color_discrete_sequence=["#e74c3c"])
            fig.update_layout(yaxis={"categoryorder": "total ascending"}, margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No negative aspect mentions found.")

    st.divider()

    # ── Net sentiment score ────────────────────────────────────────────────────
    st.subheader("📊 Net Aspect Sentiment Score")
    if "net_sentiment" in aspect_summary.columns:
        ns_df = aspect_summary[["aspect", "net_sentiment", "total"]].sort_values("net_sentiment", ascending=True)
        ns_df["color"] = ns_df["net_sentiment"].apply(lambda x: "#2ecc71" if x >= 0 else "#e74c3c")
        fig = px.bar(
            ns_df, x="net_sentiment", y="aspect", orientation="h",
            color="color", color_discrete_map="identity",
            hover_data=["total"],
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, margin=dict(t=20, b=20),
                          showlegend=False)
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Net sentiment = (Positive – Negative) / Total mentions. Range: –1 to +1.")

    st.divider()

    # ── Detailed table ─────────────────────────────────────────────────────────
    st.subheader("📋 Aspect Detail Table")
    show_cols = [c for c in ["aspect", "sentiment", "phrase", "confidence", "review_id"]
                 if c in aspects_df.columns]
    st.dataframe(aspects_df[show_cols].head(200), use_container_width=True, height=400)
