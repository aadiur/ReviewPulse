"""ReviewPulse — Sentiment Intelligence Page."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def render(df: pd.DataFrame):
    st.title("🔍 Sentiment Intelligence")
    st.divider()

    # ── VADER vs Transformer comparison ───────────────────────────────────────
    has_vader       = "vader_sentiment" in df.columns
    has_transformer = "transformer_sentiment" in df.columns and df["transformer_sentiment"].ne("Unavailable").any()

    col1, col2 = st.columns(2)
    color_map = {"Positive": "#2ecc71", "Negative": "#e74c3c", "Neutral": "#95a5a6"}

    with col1:
        st.subheader("VADER Results")
        if has_vader:
            vc = df["vader_sentiment"].value_counts().reset_index()
            vc.columns = ["Sentiment", "Count"]
            fig = px.bar(vc, x="Sentiment", y="Count", color="Sentiment",
                         color_discrete_map=color_map)
            fig.update_layout(showlegend=False, margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Rule-based lexicon sentiment.")
        else:
            st.info("VADER results not available.")

    with col2:
        st.subheader("Transformer Results")
        if has_transformer:
            tc = df["transformer_sentiment"].value_counts().reset_index()
            tc.columns = ["Sentiment", "Count"]
            fig = px.bar(tc, x="Sentiment", y="Count", color="Sentiment",
                         color_discrete_map=color_map)
            fig.update_layout(showlegend=False, margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
            avg_conf = df["transformer_confidence"].mean()
            st.caption(f"DistilBERT transformer. Avg confidence: **{avg_conf:.3f}**")
        else:
            st.info("Transformer not available or disabled.")

    st.divider()

    # ── Agreement analysis ─────────────────────────────────────────────────────
    st.subheader("🤝 Model Agreement Analysis")
    if has_vader and has_transformer:
        agree_mask = df["vader_sentiment"] == df["transformer_sentiment"]
        n_agree   = int(agree_mask.sum())
        n_disagree = len(df) - n_agree
        pct_agree = round(100 * n_agree / len(df), 1)

        a_col, b_col, c_col = st.columns(3)
        a_col.metric("Agreement",    f"{n_agree:,}",   f"{pct_agree}%")
        b_col.metric("Disagreement", f"{n_disagree:,}", f"{100-pct_agree:.1f}%")
        if "uncertainty_flag" in df.columns:
            c_col.metric("Uncertain", f"{int(df['uncertainty_flag'].sum()):,}")

        if n_disagree > 0:
            disagree_df = df[~agree_mask][["original_review", "vader_sentiment", "transformer_sentiment",
                                           "transformer_confidence", "uncertainty_flag"]].head(20)
            with st.expander(f"View {min(20, n_disagree)} disagreement examples"):
                st.dataframe(disagree_df, use_container_width=True)
    else:
        st.info("Need both VADER and transformer results for comparison.")

    st.divider()

    # ── Confidence distribution ────────────────────────────────────────────────
    st.subheader("📊 Confidence Distribution")
    if has_transformer and "transformer_confidence" in df.columns:
        fig = px.histogram(
            df, x="transformer_confidence", nbins=40,
            color_discrete_sequence=["#3498db"],
            labels={"transformer_confidence": "Confidence"},
        )
        fig.add_vline(x=0.65, line_dash="dash", line_color="red",
                      annotation_text="Uncertainty threshold (0.65)")
        fig.update_layout(margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)
    elif has_vader and "vader_compound" in df.columns:
        fig = px.histogram(
            df, x="vader_compound", nbins=40,
            color_discrete_sequence=["#3498db"],
            labels={"vader_compound": "VADER Compound Score"},
        )
        fig.add_vline(x=0.05, line_dash="dash", line_color="green",
                      annotation_text="Positive threshold")
        fig.add_vline(x=-0.05, line_dash="dash", line_color="red",
                      annotation_text="Negative threshold")
        fig.update_layout(margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    # ── Sentiment over time ────────────────────────────────────────────────────
    if "date" in df.columns and df["date"].notna().sum() > 10 and "sentiment" in df.columns:
        st.divider()
        st.subheader("📅 Sentiment Trends Over Time")
        try:
            df_time = df[df["date"].notna()].copy()
            df_time["month"] = df_time["date"].dt.to_period("M").dt.to_timestamp()
            monthly = (
                df_time.groupby(["month", "sentiment"])
                .size()
                .reset_index(name="count")
            )
            fig = px.line(
                monthly, x="month", y="count", color="sentiment",
                color_discrete_map=color_map,
                markers=True,
            )
            fig.update_layout(margin=dict(t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:
            st.warning(f"Could not render time chart: {exc}")

    # ── Rating vs Sentiment ────────────────────────────────────────────────────
    if "rating" in df.columns and df["rating"].notna().sum() > 10 and "sentiment" in df.columns:
        st.divider()
        st.subheader("⭐ Rating vs Sentiment")
        rating_sent = (
            df[df["rating"].notna()]
            .groupby(["rating", "sentiment"])
            .size()
            .reset_index(name="count")
        )
        fig = px.bar(
            rating_sent, x="rating", y="count", color="sentiment",
            color_discrete_map=color_map,
            barmode="stack",
            labels={"rating": "Rating", "count": "Reviews"},
        )
        fig.update_layout(margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    # ── Review length distribution ─────────────────────────────────────────────
    st.divider()
    st.subheader("📏 Review Length Distribution")
    text_col = "original_review" if "original_review" in df.columns else None
    if text_col:
        length_df = df.copy()
        length_df["length"] = length_df[text_col].str.len()
        fig = px.histogram(
            length_df, x="length", color="sentiment" if "sentiment" in df.columns else None,
            nbins=50, color_discrete_map=color_map,
            labels={"length": "Review Length (chars)"},
            barmode="overlay",
        )
        fig.update_layout(margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)
