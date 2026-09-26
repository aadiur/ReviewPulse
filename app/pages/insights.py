"""ReviewPulse — Product Intelligence Insights Page."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st


def render(insights: dict, df: pd.DataFrame, aspects_df: pd.DataFrame, topic_summary: pd.DataFrame):
    st.title("💡 Product Intelligence Insights")
    st.caption("Automatically generated insights from the NLP analysis — all derived from real data.")
    st.divider()

    # ── Sentiment summary ──────────────────────────────────────────────────────
    st.subheader("📊 Sentiment Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Reviews", f"{insights['total_reviews']:,}")

    pos = insights.get("positive_pct")
    neg = insights.get("negative_pct")
    neu = insights.get("neutral_pct")

    if pos is not None:
        c2.metric("Positive", f"{pos}%",  delta=f"Net: {pos - neg:.1f}%")
        c3.metric("Negative", f"{neg}%")
        c4.metric("Neutral",  f"{neu}%")

        # Verdict
        if pos >= 60:
            st.success(f"✅ Overall product reception is **positive** ({pos}% positive reviews).")
        elif neg >= 40:
            st.error(f"⚠️ Product has significant negative feedback ({neg}% negative reviews).")
        else:
            st.info("ℹ️ Mixed reception — substantial positive and negative feedback.")

    st.divider()

    # ── Aspect insights ────────────────────────────────────────────────────────
    if insights.get("most_praised_aspects") or insights.get("most_criticized_aspects"):
        st.subheader("🎯 Aspect Intelligence")
        col_p, col_n = st.columns(2)

        with col_p:
            st.markdown("**👍 Most Praised Aspects**")
            praised = insights.get("most_praised_aspects", {})
            if praised:
                for asp, count in list(praised.items())[:5]:
                    st.markdown(f"- **{asp.replace('_', ' ').title()}**: {count:,} positive mentions")
            else:
                st.info("No praised aspects found.")

        with col_n:
            st.markdown("**👎 Most Criticised Aspects**")
            criticised = insights.get("most_criticized_aspects", {})
            if criticised:
                for asp, count in list(criticised.items())[:5]:
                    st.markdown(f"- **{asp.replace('_', ' ').title()}**: {count:,} negative mentions")
            else:
                st.info("No criticised aspects found.")

        st.divider()

    # ── Emotion insights ───────────────────────────────────────────────────────
    emo_dist = insights.get("emotion_distribution", {})
    if emo_dist:
        st.subheader("😊 Dominant Emotions")
        dom_emotion = insights.get("dominant_emotion")
        if dom_emotion:
            st.info(f"Most common emotion expressed in reviews: **{dom_emotion.title()}**")

        emo_df = pd.DataFrame(list(emo_dist.items()), columns=["Emotion", "Count"])
        emo_df = emo_df.sort_values("Count", ascending=False)
        emo_colors = {
            "joy": "#f1c40f", "anger": "#e74c3c", "sadness": "#3498db",
            "fear": "#9b59b6", "surprise": "#1abc9c", "neutral": "#95a5a6",
            "disgust": "#e67e22",
        }
        fig = px.bar(
            emo_df, x="Count", y="Emotion", orientation="h",
            color="Emotion", color_discrete_map=emo_colors,
        )
        fig.update_layout(showlegend=False, margin=dict(t=10, b=10),
                          yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)
        st.divider()

    # ── Topic insights ─────────────────────────────────────────────────────────
    top_topics = insights.get("top_topics", [])
    if top_topics:
        st.subheader("🧠 Most Discussed Topics")
        for t in top_topics[:5]:
            label = t.get("topic_label", "Unknown")
            count = t.get("count", 0)
            st.markdown(f"- **{label}**: {count:,} reviews")
        st.divider()

    # ── Uncertainty ────────────────────────────────────────────────────────────
    unc_count = insights.get("uncertain_count", 0)
    unc_pct   = insights.get("uncertain_pct", 0)
    st.subheader("⚠️ Uncertainty Analysis")
    if unc_count > 0:
        st.warning(
            f"**{unc_count:,} reviews ({unc_pct}%)** where VADER and the transformer model disagree "
            "or the transformer confidence is low. These reviews are worth manual inspection."
        )
        if "uncertainty_flag" in df.columns:
            uncertain_df = df[df["uncertainty_flag"] == True][
                [c for c in ["original_review", "vader_sentiment", "transformer_sentiment",
                              "transformer_confidence"] if c in df.columns]
            ].head(10)
            with st.expander("Show uncertain reviews (first 10)"):
                st.dataframe(uncertain_df, use_container_width=True)
    else:
        st.success("All model predictions are consistent and high-confidence.")

    # ── Time trends ────────────────────────────────────────────────────────────
    monthly = insights.get("monthly_sentiment", [])
    if monthly:
        st.divider()
        st.subheader("📅 Sentiment Trends Over Time")
        monthly_df = pd.DataFrame(monthly)
        if not monthly_df.empty and "month" in monthly_df.columns:
            sent_cols = [c for c in monthly_df.columns if c != "month"]
            melt = monthly_df.melt(id_vars="month", value_vars=sent_cols,
                                   var_name="Sentiment", value_name="Count")
            color_map = {"Positive": "#2ecc71", "Negative": "#e74c3c", "Neutral": "#95a5a6"}
            fig = px.line(melt, x="month", y="Count", color="Sentiment",
                          color_discrete_map=color_map, markers=True)
            fig.update_layout(margin=dict(t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)

    # ── Review stats ───────────────────────────────────────────────────────────
    avg_len = insights.get("avg_review_length")
    if avg_len:
        st.divider()
        st.subheader("📝 Review Statistics")
        s1, s2 = st.columns(2)
        s1.metric("Avg Review Length", f"{avg_len:.0f} chars")
        s2.metric("Median Review Length", f"{insights.get('median_review_length', 0):.0f} chars")

    # ── Rating distribution ────────────────────────────────────────────────────
    rating_dist = insights.get("rating_distribution", {})
    if rating_dist:
        st.divider()
        st.subheader("⭐ Rating Distribution")
        avg_rating = insights.get("average_rating")
        if avg_rating:
            st.metric("Average Rating", f"⭐ {avg_rating}")
        rd_df = pd.DataFrame(list(rating_dist.items()), columns=["Rating", "Count"])
        rd_df = rd_df.sort_values("Rating")
        fig = px.bar(rd_df, x="Rating", y="Count", color_discrete_sequence=["#f39c12"])
        fig.update_layout(margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)
