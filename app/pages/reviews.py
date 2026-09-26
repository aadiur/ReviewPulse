"""ReviewPulse — Review Explorer Page."""

from __future__ import annotations

import pandas as pd
import streamlit as st


def render(df: pd.DataFrame, aspects_df: pd.DataFrame):
    st.title("📋 Review Explorer")
    st.caption("Filter, search, and inspect individual reviews.")
    st.divider()

    filtered = df.copy()

    # ── Filter controls ────────────────────────────────────────────────────────
    with st.expander("🔧 Filters", expanded=True):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if "sentiment" in df.columns:
                sentiments = ["All"] + sorted(df["sentiment"].dropna().unique().tolist())
                sel_sentiment = st.selectbox("Sentiment", sentiments)
                if sel_sentiment != "All":
                    filtered = filtered[filtered["sentiment"] == sel_sentiment]

        with col2:
            if "emotion" in df.columns:
                emotions = ["All"] + sorted(df["emotion"].dropna().unique().tolist())
                sel_emotion = st.selectbox("Emotion", emotions)
                if sel_emotion != "All":
                    filtered = filtered[filtered["emotion"] == sel_emotion]

        with col3:
            if "cluster" in df.columns:
                clusters = ["All"] + sorted(df["cluster"].dropna().unique().tolist())
                sel_cluster = st.selectbox("Cluster", clusters)
                if sel_cluster != "All":
                    filtered = filtered[filtered["cluster"] == sel_cluster]

        with col4:
            if "uncertainty_flag" in df.columns:
                unc_options = ["All", "Uncertain only", "Certain only"]
                sel_unc = st.selectbox("Uncertainty", unc_options)
                if sel_unc == "Uncertain only":
                    filtered = filtered[filtered["uncertainty_flag"] == True]
                elif sel_unc == "Certain only":
                    filtered = filtered[filtered["uncertainty_flag"] == False]

        col5, col6 = st.columns(2)
        with col5:
            if "rating" in df.columns and df["rating"].notna().sum() > 0:
                min_r = float(df["rating"].min())
                max_r = float(df["rating"].max())
                if min_r < max_r:
                    r_range = st.slider("Rating range", min_r, max_r, (min_r, max_r))
                    filtered = filtered[
                        (filtered["rating"].isna()) |
                        ((filtered["rating"] >= r_range[0]) & (filtered["rating"] <= r_range[1]))
                    ]

        with col6:
            search_text = st.text_input("Search in review text", placeholder="keyword or phrase…")
            if search_text.strip():
                text_col = "original_review" if "original_review" in filtered.columns else "cleaned_review"
                filtered = filtered[
                    filtered[text_col].str.contains(search_text.strip(), case=False, na=False)
                ]

    # ── Aspect filter ──────────────────────────────────────────────────────────
    if not aspects_df.empty and "aspects_summary" in filtered.columns:
        aspect_options = ["All"] + sorted(aspects_df["aspect"].unique().tolist())
        sel_aspect = st.selectbox("Filter by Aspect", aspect_options)
        if sel_aspect != "All":
            filtered = filtered[
                filtered["aspects_summary"].str.contains(sel_aspect, na=False)
            ]

    # ── Topic filter ───────────────────────────────────────────────────────────
    if "topic_label" in filtered.columns:
        topic_options = ["All"] + sorted(filtered["topic_label"].dropna().unique().tolist())
        sel_topic = st.selectbox("Filter by Topic", topic_options)
        if sel_topic != "All":
            filtered = filtered[filtered["topic_label"] == sel_topic]

    st.divider()
    st.markdown(f"**{len(filtered):,}** reviews match the current filters.")

    # ── Display columns ────────────────────────────────────────────────────────
    display_cols = []
    for col in ["review_id", "original_review", "rating", "sentiment", "sentiment_confidence",
                "emotion", "aspects_summary", "topic_label", "cluster", "uncertainty_flag"]:
        if col in filtered.columns:
            display_cols.append(col)

    if not display_cols:
        display_cols = list(filtered.columns[:8])

    st.dataframe(
        filtered[display_cols].head(500),
        use_container_width=True,
        height=500,
        column_config={
            "original_review": st.column_config.TextColumn("Review", width="large"),
            "sentiment_confidence": st.column_config.ProgressColumn(
                "Confidence", min_value=0.0, max_value=1.0, format="%.2f"
            ) if "sentiment_confidence" in display_cols else None,
        },
    )

    # ── Download ───────────────────────────────────────────────────────────────
    csv = filtered[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download filtered reviews (CSV)",
        data=csv,
        file_name="reviewpulse_filtered.csv",
        mime="text/csv",
    )
