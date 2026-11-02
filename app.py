"""
The Data-Driven Social Engagement Initiative
Evidence-Based Social Content Analytics & Strategy Platform

Main Streamlit Application Dashboard.
Suitable for BE CSE / Data Science Major Project Demonstration.
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Setup Path
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from config import (
    COMMENTS_DATA_PATH,
    CONTENT_TYPES,
    HOOK_TYPES,
    MODELS_DIR,
    PLATFORMS,
    POSTS_DATA_PATH,
    RELATABLE_MODEL_PATH,
    REPORT_OUTPUT_PATH,
    SENTIMENT_COLORS,
    TFIDF_VECTORIZER_PATH,
    THEME_COLORS,
    TOPICS,
    VIRAL_COLORS,
    VIRAL_MODEL_PATH,
    VIRAL_THRESHOLDS,
    VIRAL_WEIGHTS,
)
from src.ab_testing import (
    compare_durations,
    compare_hook_types,
    compare_posting_times,
    run_statistical_test,
)
from src.data_loader import generate_demo_data, load_data, normalize_column_names
from src.feature_engineering import (
    calculate_single_viral_score,
    engineer_features,
    get_normalization_bounds,
)
from src.nlp_model import (
    analyze_comment_sample,
    enrich_comments_dataset,
    load_or_train_nlp_model,
)
from src.preprocessing import clean_comments, clean_posts
from src.recommender import generate_content_recommendations
from src.trend_analysis import compute_topic_trends
from src.viral_model import (
    load_or_train_viral_model,
    predict_virality,
)
from src.visualizations import (
    plot_ab_comparison,
    plot_confusion_matrix,
    plot_engagement_by_content_type,
    plot_engagement_by_hook,
    plot_engagement_by_platform,
    plot_engagement_by_topic,
    plot_engagement_distribution,
    plot_feature_importances,
    plot_likes_vs_shares,
    plot_posting_hour_performance,
    plot_relatable_breakdown,
    plot_relatability_by_topic,
    plot_retention_vs_engagement,
    plot_save_to_share_ratio,
    plot_sentiment_breakdown,
    plot_shares_vs_saves,
    plot_topic_follower_growth,
)
from utils.helpers import format_duration, format_metric, format_percent, save_insights_report

# Page Configuration
st.set_page_config(
    page_title="Data-Driven Social Engagement Initiative",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Modern, Polished Aesthetics
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Metric Card Styling */
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: rgba(99, 102, 241, 0.4);
        transform: translateY(-2px);
    }
    .metric-label {
        font-size: 0.82rem;
        font-weight: 500;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #F8FAFC;
        line-height: 1.2;
    }
    .metric-delta {
        font-size: 0.8rem;
        font-weight: 500;
        margin-top: 4px;
    }
    
    /* Strategy Recommendation Card */
    .strategy-card {
        background: linear-gradient(135deg, #1E1B4B 0%, #0F172A 100%);
        border: 1px solid #4F46E5;
        border-radius: 14px;
        padding: 24px;
        box-shadow: 0 8px 24px rgba(79, 70, 229, 0.2);
        margin-bottom: 20px;
    }
    .strategy-title {
        color: #818CF8;
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 8px;
    }
    
    /* Badge styling */
    .badge-demo {
        background-color: rgba(245, 158, 11, 0.15);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.3);
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-imported {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    /* Subheaders */
    .section-header {
        font-size: 1.35rem;
        font-weight: 700;
        color: #F1F5F9;
        margin-top: 15px;
        margin-bottom: 12px;
        border-left: 4px solid #6366F1;
        padding-left: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Cached Data and Model Pipeline
@st.cache_data(show_spinner="Ingesting and processing datasets...")
def get_processed_data() -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """Load, clean, and feature engineer datasets."""
    posts_raw, comments_raw, source_label = load_data()
    posts_clean = clean_posts(posts_raw)
    comments_clean = clean_comments(comments_raw)
    posts_engineered = engineer_features(posts_clean)
    return posts_engineered, comments_clean, source_label


@st.cache_resource(show_spinner="Initializing Machine Learning and NLP models...")
def get_models(
    posts_df: pd.DataFrame, comments_df: pd.DataFrame
) -> Tuple[Any, Dict, Any, Any, Dict]:
    """Train or load viral prediction model and NLP relatability classifier."""
    viral_pipe, viral_metrics = load_or_train_viral_model(posts_df)
    tfidf_vec, nlp_model, nlp_metrics = load_or_train_nlp_model(comments_df)
    return viral_pipe, viral_metrics, tfidf_vec, nlp_model, nlp_metrics


# Initialize Data and Models
try:
    posts_df_all, comments_df_all, data_source_label = get_processed_data()
    viral_pipe, viral_metrics, tfidf_vec, nlp_model, nlp_metrics = get_models(
        posts_df_all, comments_df_all
    )
    comments_df_all = enrich_comments_dataset(
        comments_df_all, posts_df_all, tfidf_vec, nlp_model
    )
except Exception as exc:
    st.error(f"System initialization error: {exc}")
    st.stop()


# SIDEBAR CONTROLS & FILTERS
with st.sidebar:
    st.title("📊 Control Center")
    
    # Data Source Indicator Badge
    if "Demo" in data_source_label:
        st.markdown(f'<span class="badge-demo">● {data_source_label}</span>', unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="badge-imported">● {data_source_label}</span>', unsafe_allow_html=True)
        
    st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)

    # Dataset Source Selector
    with st.expander("📂 Dataset Management", expanded=False):
        st.caption("Load custom CSV or regenerate synthetic demo corpus.")
        uploaded_posts = st.file_uploader("Upload Posts CSV", type=["csv"], key="up_posts")
        uploaded_comments = st.file_uploader("Upload Comments CSV", type=["csv"], key="up_comm")

        if uploaded_posts is not None:
            try:
                custom_p = pd.read_csv(uploaded_posts)
                custom_p = normalize_column_names(custom_p)
                custom_p.to_csv(POSTS_DATA_PATH, index=False)
                st.success("Custom posts CSV uploaded! Click below to reload.")
                if st.button("Reload with Uploaded Posts"):
                    st.cache_data.clear()
                    st.rerun()
            except Exception as e:
                st.error(f"Upload error: {e}")

        if st.button("Regenerate Demo Dataset", help="Create a fresh synthetic corpus with 1,200 posts & 3,500 comments"):
            demo_p, demo_c = generate_demo_data()
            demo_p.to_csv(POSTS_DATA_PATH, index=False)
            demo_c.to_csv(COMMENTS_DATA_PATH, index=False)
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Demo dataset refreshed!")
            st.rerun()

    st.markdown("---")
    st.subheader("Global Filters")

    # Platform Filter
    available_platforms = sorted(posts_df_all["platform"].dropna().unique().tolist())
    selected_platforms = st.multiselect(
        "Platform",
        options=available_platforms,
        default=available_platforms,
    )

    # Topic Filter
    available_topics = sorted(posts_df_all["topic"].dropna().unique().tolist())
    selected_topics = st.multiselect(
        "Topic",
        options=available_topics,
        default=available_topics,
    )

    # Content Type Filter
    available_types = sorted(posts_df_all["content_type"].dropna().unique().tolist())
    selected_types = st.multiselect(
        "Content Format",
        options=available_types,
        default=available_types,
    )

    # Hook Type Filter
    available_hooks = sorted(posts_df_all["hook_type"].dropna().unique().tolist())
    selected_hooks = st.multiselect(
        "Hook Strategy",
        options=available_hooks,
        default=available_hooks,
    )

    # Reset Filters Button
    if st.button("Reset All Filters", use_container_width=True):
        st.rerun()

    st.markdown("---")
    # Export Report Action
    if st.button("📥 Generate Text Report", use_container_width=True):
        recs = generate_content_recommendations(posts_df_all)
        top_t = posts_df_all.groupby("topic").agg(
            total_engagement=("total_engagement", "sum"),
            avg_viral_score=("viral_score", "mean")
        ).reset_index().to_dict(orient="records")
        
        sample_insights = [
            f"Analyzed {len(posts_df_all):,} posts across {len(available_platforms)} platforms.",
            f"Highest viral potential identified in topic '{recs['topic']}'.",
            f"Prescribed format is {recs['content_type']} with {recs['hook_type']} hook.",
            f"Audience relatability classification rate stands at {((comments_df_all['relatability'] == 'RELATABLE').mean()*100):.1f}%.",
        ]
        summary_m = {
            "total_posts": len(posts_df_all),
            "total_engagement": posts_df_all["total_engagement"].sum(),
            "avg_engagement_rate": posts_df_all["engagement_rate"].mean(),
            "avg_viral_score": posts_df_all["viral_score"].mean(),
            "high_viral_pct": (posts_df_all["viral_category"] == "High").mean() * 100.0,
            "total_comments": len(comments_df_all),
            "relatable_pct": (comments_df_all["relatability"] == "RELATABLE").mean() * 100.0,
            "data_source": data_source_label,
        }
        saved = save_insights_report(sample_insights, summary_m, top_t, recs, REPORT_OUTPUT_PATH)
        if saved:
            st.success("Report written to reports/generated_insights.txt!")
            with open(REPORT_OUTPUT_PATH, "r", encoding="utf-8") as rf:
                st.download_button(
                    "Download Insights Report",
                    data=rf.read(),
                    file_name="generated_insights.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

# Apply Active Filtering
filtered_posts = posts_df_all[
    (posts_df_all["platform"].isin(selected_platforms))
    & (posts_df_all["topic"].isin(selected_topics))
    & (posts_df_all["content_type"].isin(selected_types))
    & (posts_df_all["hook_type"].isin(selected_hooks))
].copy()

# Linked comments filter
if "post_id" in filtered_posts.columns and not filtered_posts.empty:
    filtered_post_ids = set(filtered_posts["post_id"])
    filtered_comments = comments_df_all[comments_df_all["post_id"].isin(filtered_post_ids)].copy()
else:
    filtered_comments = comments_df_all.copy()


# Top Navigation Tabs
st.title("THE DATA-DRIVEN SOCIAL ENGAGEMENT INITIATIVE")
st.caption("Evidence-Based Social Content Analytics, Predictive Virality, and Prescriptive Content Strategy")

tabs = st.tabs([
    "1. Executive Overview",
    "2. Content Analytics",
    "3. Virality Prediction",
    "4. Audience NLP",
    "5. A/B Testing",
    "6. Trend Analysis",
    "7. Strategy Recommendations",
    "8. Dataset Explorer",
    "9. Model Performance",
    "10. About Project",
])


# ==============================================================================
# TAB 1: EXECUTIVE OVERVIEW
# ==============================================================================
with tabs[0]:
    if filtered_posts.empty:
        st.warning("⚠️ No data matches the selected filters. Please adjust your filters or click 'Reset All Filters'.")
    else:
        # KPI Row
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        tot_posts = len(filtered_posts)
        tot_eng = filtered_posts["total_engagement"].sum()
        avg_eng_rate = filtered_posts["engagement_rate"].mean()
        avg_viral = filtered_posts["viral_score"].mean()
        high_viral_pct = (filtered_posts["viral_category"] == "High").mean() * 100.0
        relatable_pct = (
            (filtered_comments["relatability"] == "RELATABLE").mean() * 100.0
            if not filtered_comments.empty
            else 0.0
        )

        with col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Total Posts</div>
                    <div class="metric-value">{tot_posts:,}</div>
                    <div class="metric-delta" style="color: #38BDF8;">Filtered Sample</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Total Engagement</div>
                    <div class="metric-value">{format_metric(tot_eng)}</div>
                    <div class="metric-delta" style="color: #34D399;">Likes+Shares+Saves+Comments</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Avg Engagement Rate</div>
                    <div class="metric-value">{avg_eng_rate:.2f}%</div>
                    <div class="metric-delta" style="color: #A78BFA;">Per Post Exposure</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Avg Viral Score</div>
                    <div class="metric-value">{avg_viral:.1f}</div>
                    <div class="metric-delta" style="color: #FBBF24;">Scale 0 – 100 Index</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col5:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">High-Viral Content</div>
                    <div class="metric-value">{high_viral_pct:.1f}%</div>
                    <div class="metric-delta" style="color: #10B981;">Score ≥ 70</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col6:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Relatable Comments</div>
                    <div class="metric-value">{relatable_pct:.1f}%</div>
                    <div class="metric-delta" style="color: #EC4899;">NLP Resonance</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Dynamic Key Insights
        st.markdown('<div class="section-header">Key Analytical Insights</div>', unsafe_allow_html=True)
        
        top_topic = filtered_posts.groupby("topic")["total_engagement"].mean().idxmax()
        top_topic_eng = filtered_posts.groupby("topic")["total_engagement"].mean().max()
        top_format = filtered_posts.groupby("content_type")["viral_score"].mean().idxmax()
        top_hook = filtered_posts.groupby("hook_type")["total_engagement"].mean().idxmax()
        save_share_leader = filtered_posts.groupby("topic")["save_share_ratio"].mean().idxmax()
        peak_hour = int(filtered_posts.groupby("hour")["total_engagement"].mean().idxmax())

        insight_cols = st.columns(2)
        with insight_cols[0]:
            st.info(
                f"📌 **Topic Resonance:** **'{top_topic}'** leads engagement across this cohort, "
                f"averaging **{top_topic_eng:,.0f}** interactions per post."
            )
            st.success(
                f"📌 **High-Converting Format:** **'{top_format}'** posts exhibit the strongest average Viral Score, "
                f"supported by elevated audience retention."
            )
        with insight_cols[1]:
            st.warning(
                f"📌 **Opening Attention:** Posts leveraging **'{top_hook}'** hooks generated the highest interaction velocity."
            )
            st.info(
                f"📌 **Temporal Velocity:** Historical activity peaks around **{peak_hour:02d}:00**, highlighting an optimal publishing window."
            )

        st.markdown("---")

        # Visualizations Grid
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.plotly_chart(plot_engagement_by_topic(filtered_posts), use_container_width=True)
            st.plotly_chart(plot_retention_vs_engagement(filtered_posts), use_container_width=True)
        with chart_col2:
            st.plotly_chart(plot_engagement_by_content_type(filtered_posts), use_container_width=True)
            st.plotly_chart(plot_posting_hour_performance(filtered_posts), use_container_width=True)

        chart_col3, chart_col4 = st.columns(2)
        with chart_col3:
            st.plotly_chart(plot_shares_vs_saves(filtered_posts), use_container_width=True)
        with chart_col4:
            st.plotly_chart(plot_engagement_by_platform(filtered_posts), use_container_width=True)


# ==============================================================================
# TAB 2: CONTENT ANALYTICS
# ==============================================================================
with tabs[1]:
    st.markdown('<div class="section-header">In-Depth Content Performance Analysis</div>', unsafe_allow_html=True)
    if filtered_posts.empty:
        st.warning("⚠️ No data available under current filters.")
    else:
        # Top Performing Posts
        st.subheader("Top Performing Posts by Viral Score")
        top_posts = filtered_posts.sort_values(by="viral_score", ascending=False).head(15)[
            [
                "post_id",
                "topic",
                "content_type",
                "platform",
                "hook_type",
                "viral_score",
                "viral_category",
                "total_engagement",
                "likes",
                "shares",
                "saves",
                "comments_count",
                "retention_rate",
                "followers_gained",
            ]
        ]
        st.dataframe(
            top_posts.style.format({
                "viral_score": "{:.1f}",
                "total_engagement": "{:,.0f}",
                "likes": "{:,.0f}",
                "shares": "{:,.0f}",
                "saves": "{:,.0f}",
                "comments_count": "{:,.0f}",
                "retention_rate": "{:.1f}%",
                "followers_gained": "{:,.0f}",
            }),
            use_container_width=True,
        )

        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_engagement_by_hook(filtered_posts), use_container_width=True)
            st.plotly_chart(plot_save_to_share_ratio(filtered_posts), use_container_width=True)
        with c2:
            st.plotly_chart(plot_likes_vs_shares(filtered_posts), use_container_width=True)
            st.plotly_chart(plot_topic_follower_growth(filtered_posts), use_container_width=True)

        st.markdown("---")
        # Download Filtered CSV
        csv_data = filtered_posts.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Filtered Content Analytics CSV",
            data=csv_data,
            file_name=f"content_analytics_filtered_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )


# ==============================================================================
# TAB 3: VIRALITY PREDICTION
# ==============================================================================
with tabs[2]:
    st.markdown('<div class="section-header">Predictive Virality Modeling & Interactive Calculator</div>', unsafe_allow_html=True)
    
    # Formula Explainer Expander
    with st.expander("📐 Transparent Viral Score Formulation", expanded=True):
        st.markdown(
            r"""
            The **Viral Score** is a project-defined normalized index ($0 \text{ to } 100$) reflecting active engagement velocity,
            saving utility, sharing resonance, audience retention, and audience conversion:
            
            $$\text{Viral Score} = 0.15 \tilde{L} + 0.25 \tilde{S} + 0.25 \tilde{V} + 0.20 \tilde{R} + 0.10 \tilde{C} + 0.05 \tilde{F}$$
            
            Where:
            * $\tilde{L} = \text{Normalized Likes}$ (15% weight)
            * $\tilde{S} = \text{Normalized Shares}$ (25% weight - primary propagation driver)
            * $\tilde{V} = \text{Normalized Saves}$ (25% weight - utility/bookmarking value)
            * $\tilde{R} = \text{Normalized Retention Rate}$ (20% weight - watch completion ratio)
            * $\tilde{C} = \text{Normalized Comments}$ (10% weight - conversational stimulus)
            * $\tilde{F} = \text{Normalized Followers Gained}$ (5% weight - account acquisition index)

            **Classification Thresholds:**
            * 🟢 **High Virality:** $\text{Viral Score} \ge 70$
            * 🟡 **Medium Virality:** $40 \le \text{Viral Score} < 70$
            * 🔴 **Low Virality:** $\text{Viral Score} < 40$
            """
        )

    vcol1, vcol2 = st.columns([1.2, 1.8])
    with vcol1:
        st.subheader("Simulate & Predict Content Performance")
        with st.form("virality_prediction_form"):
            in_topic = st.selectbox("Topic", TOPICS, index=0)
            in_content = st.selectbox("Content Type", CONTENT_TYPES, index=0)
            in_hook = st.selectbox("Hook Type", HOOK_TYPES, index=0)
            in_platform = st.selectbox("Platform", PLATFORMS, index=0)
            
            fcol1, fcol2 = st.columns(2)
            with fcol1:
                in_duration = st.slider("Duration (seconds)", 0, 300, 35)
                in_hour = st.slider("Posting Hour (0-23)", 0, 23, 19)
                in_likes = st.number_input("Estimated Likes", min_value=0, max_value=500000, value=850)
                in_shares = st.number_input("Estimated Shares", min_value=0, max_value=50000, value=110)
                in_saves = st.number_input("Estimated Saves", min_value=0, max_value=50000, value=140)
            with fcol2:
                in_comments = st.number_input("Estimated Comments", min_value=0, max_value=25000, value=45)
                in_retention = st.slider("Retention Rate (%)", 0.0, 100.0, 68.0)
                in_followers = st.number_input("Followers Gained", min_value=0, max_value=10000, value=35)
                in_caption_len = st.slider("Caption Length (chars)", 10, 500, 140)
                in_hashtags = st.slider("Hashtag Count", 0, 30, 6)

            predict_submitted = st.form_submit_button("⚡ PREDICT VIRALITY", use_container_width=True)

        if predict_submitted:
            # 1. Feature pipeline simulation input
            sim_input = {
                "likes": in_likes,
                "shares": in_shares,
                "saves": in_saves,
                "comments_count": in_comments,
                "retention_rate": in_retention,
                "followers_gained": in_followers,
                "duration": in_duration,
                "caption_length": in_caption_len,
                "hashtag_count": in_hashtags,
                "hour": in_hour,
                "content_type": in_content,
                "topic": in_topic,
                "hook_type": in_hook,
                "platform": in_platform,
            }

            # 2. Formulaic Score
            norm_bounds = get_normalization_bounds(posts_df_all)
            calculated_score, calc_cat = calculate_single_viral_score(
                in_likes, in_shares, in_saves, in_comments, in_retention, in_followers, norm_bounds
            )

            # 3. Machine Learning RandomForest Prediction
            ml_result = predict_virality(viral_pipe, sim_input)
            pred_tier = ml_result["predicted_category"]
            confidence = ml_result["confidence"]

            st.markdown("### Prediction Results")
            color_hex = VIRAL_COLORS.get(pred_tier, "#4F46E5")
            st.markdown(
                f"""
                <div style="background: rgba(30, 41, 59, 0.9); border: 2px solid {color_hex}; border-radius: 12px; padding: 20px;">
                    <div style="font-size: 0.9rem; color: #94A3B8;">Predicted Virality Tier (RandomForest)</div>
                    <div style="font-size: 2.2rem; font-weight: 800; color: {color_hex};">{pred_tier.upper()}</div>
                    <div style="font-size: 0.95rem; color: #E2E8F0; margin-top: 6px;">
                        Model Confidence: <strong>{confidence*100:.1f}%</strong> | Analytical Viral Score: <strong>{calculated_score:.1f} / 100</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.caption("Class Probability Distribution:")
            prob_df = pd.DataFrame([ml_result["probabilities"]]).T.reset_index()
            prob_df.columns = ["Tier", "Probability"]
            st.dataframe(prob_df.style.format({"Probability": "{:.2%}"}), use_container_width=True)

    with vcol2:
        st.subheader("Viral Performance Distributions")
        st.plotly_chart(plot_engagement_distribution(filtered_posts), use_container_width=True)
        
        # Breakdown of categories
        tier_counts = filtered_posts["viral_category"].value_counts().reset_index()
        tier_counts.columns = ["Tier", "Count"]
        tier_counts["Percentage"] = (tier_counts["Count"] / len(filtered_posts)) * 100.0
        
        fig_pie = px.pie(
            tier_counts,
            names="Tier",
            values="Count",
            color="Tier",
            color_discrete_map=VIRAL_COLORS,
            hole=0.4,
            title="<b>Viral Category Distribution in Filtered Dataset</b>",
        )
        st.plotly_chart(fig_pie, use_container_width=True)


# ==============================================================================
# TAB 4: AUDIENCE NLP
# ==============================================================================
with tabs[3]:
    st.markdown('<div class="section-header">Audience Natural Language Processing & Sentiment Intelligence</div>', unsafe_allow_html=True)

    st.markdown(
        """
        > **Methodological Clarification:** **Relatability** and **Sentiment** represent distinct analytical dimensions.
        > A comment such as *"I am so overwhelmed and stressed out, this is literally me"* is **highly relatable**,
        > but exhibits **negative sentiment polarity**.
        """
    )

    # Comment Stats
    ncol1, ncol2, ncol3, ncol4, ncol5 = st.columns(5)
    total_c = len(filtered_comments)
    rel_pct = (filtered_comments["relatability"] == "RELATABLE").mean() * 100.0 if total_c > 0 else 0
    neut_c_pct = 100.0 - rel_pct
    pos_pct = (filtered_comments["sentiment"] == "Positive").mean() * 100.0 if total_c > 0 else 0
    neg_pct = (filtered_comments["sentiment"] == "Negative").mean() * 100.0 if total_c > 0 else 0

    with ncol1:
        st.metric("Total Comments", f"{total_c:,}")
    with ncol2:
        st.metric("Relatable Rate", f"{rel_pct:.1f}%")
    with ncol3:
        st.metric("Neutral Relatability", f"{neut_c_pct:.1f}%")
    with ncol4:
        st.metric("Positive Sentiment", f"{pos_pct:.1f}%")
    with ncol5:
        st.metric("Negative Sentiment", f"{neg_pct:.1f}%")

    st.markdown("---")

    # Interactive Comment Tester
    st.subheader("Interactive Comment NLP Classifier")
    user_comment = st.text_input(
        "Enter audience comment to classify:",
        value="This is literally me every single Monday morning!",
    )
    if st.button("🔍 ANALYZE COMMENT", use_container_width=True):
        res = analyze_comment_sample(user_comment, tfidf_vec, nlp_model)
        
        c_res1, c_res2, c_res3 = st.columns(3)
        with c_res1:
            rel_color = "#10B981" if res["relatability"] == "RELATABLE" else "#64748B"
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Relatability Classification</div>
                    <div class="metric-value" style="color: {rel_color};">{res['relatability']}</div>
                    <div class="metric-delta">Confidence: {res['relatability_confidence']*100:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c_res2:
            s_color = SENTIMENT_COLORS.get(res["sentiment"], "#94A3B8")
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Sentiment Polarity</div>
                    <div class="metric-value" style="color: {s_color};">{res['sentiment']}</div>
                    <div class="metric-delta">Score: {res['sentiment_polarity']:+.2f} (-1 to +1)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c_res3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Preprocessed Tokens</div>
                    <div style="font-size: 0.95rem; color: #F1F5F9; margin-top: 6px;">
                        <code>{res['cleaned_text']}</code>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")
    # NLP Plots
    pcol1, pcol2 = st.columns(2)
    with pcol1:
        st.plotly_chart(plot_relatable_breakdown(filtered_comments), use_container_width=True)
        st.plotly_chart(plot_relatability_by_topic(filtered_comments), use_container_width=True)
    with pcol2:
        st.plotly_chart(plot_sentiment_breakdown(filtered_comments), use_container_width=True)
        # Sentiment by topic
        if "topic" in filtered_comments.columns:
            sent_topic = (
                pd.crosstab(filtered_comments["topic"], filtered_comments["sentiment"], normalize="index") * 100.0
            ).reset_index()
            fig_st = px.bar(
                sent_topic,
                x="topic",
                y=["Positive", "Neutral", "Negative"],
                title="<b>Sentiment Polarity Breakdown by Topic</b>",
                color_discrete_map=SENTIMENT_COLORS,
                barmode="stack",
            )
            st.plotly_chart(fig_st, use_container_width=True)


# ==============================================================================
# TAB 5: A/B TESTING
# ==============================================================================
with tabs[4]:
    st.markdown('<div class="section-header">Statistical A/B Testing & Content Experimentation</div>', unsafe_allow_html=True)
    
    st.info(
        "💡 **Methodological Integrity Note:** Observational A/B comparisons evaluate empirical statistical association. "
        "They do not fulfill randomized controlled trial (RCT) criteria and must not be interpreted as causal proofs."
    )

    ab_tabs = st.tabs([
        "Test A: Hook Type (Visual vs Text)",
        "Test B: Content Duration (Short vs Long)",
        "Test C: Posting Window (Morning vs Evening)",
        "Custom A/B Experiment",
    ])

    def render_test_results(result: Dict[str, Any], series_a: pd.Series, series_b: pd.Series):
        if "error" in result:
            st.warning(result["error"])
            return

        r1, r2, r3, r4 = st.columns(4)
        with r1:
            st.metric(f"{result['group_a_label']} Mean", f"{result['mean_a']:,.1f}", f"n = {result['count_a']}")
        with r2:
            st.metric(f"{result['group_b_label']} Mean", f"{result['mean_b']:,.1f}", f"n = {result['count_b']}")
        with r3:
            st.metric("Mean Difference", f"{result['difference']:+,.1f}", f"{result['percent_difference']:+.1f}%")
        with r4:
            sig_color = "#10B981" if result["is_significant"] else "#EF4444"
            st.markdown(
                f"""
                <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid {sig_color}; border-radius: 8px; padding: 10px; text-align: center;">
                    <div style="font-size: 0.75rem; color: #94A3B8;">SIGNIFICANCE (α = 0.05)</div>
                    <div style="font-size: 1.05rem; font-weight: 700; color: {sig_color};">{result['significance_conclusion']}</div>
                    <div style="font-size: 0.75rem; color: #CBD5E1;">p = {result['welch_p_value']:.4f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(f"**Statistical Interpretation:** {result['narrative']}")
        with st.expander("Technical Test Statistics"):
            st.write({
                "Welch t-statistic": result["welch_t_statistic"],
                "Welch p-value": result["welch_p_value"],
                "Mann-Whitney U statistic": result["mann_whitney_u"],
                "Mann-Whitney p-value": result["mann_whitney_p_value"],
                "Group A Std Dev": result["std_a"],
                "Group B Std Dev": result["std_b"],
            })

        st.plotly_chart(
            plot_ab_comparison(series_a, series_b, result["group_a_label"], result["group_b_label"], result["metric_name"]),
            use_container_width=True,
        )

    with ab_tabs[0]:
        st.subheader("Hypothesis: Visual Opening Hooks Generate Greater Total Engagement than Text Hooks")
        hook_res = compare_hook_types(posts_df_all, "total_engagement", "Visual", "Text")
        ga = posts_df_all[posts_df_all["hook_type"] == "Visual"]["total_engagement"]
        gb = posts_df_all[posts_df_all["hook_type"] == "Text"]["total_engagement"]
        render_test_results(hook_res, ga, gb)

    with ab_tabs[1]:
        st.subheader("Hypothesis: Short Content (<60s) Outperforms Long Format (≥60s) in Audience Retention")
        dur_res = compare_durations(posts_df_all, "retention_rate", threshold_seconds=60)
        da = posts_df_all[posts_df_all["duration"] < 60]["retention_rate"]
        db = posts_df_all[posts_df_all["duration"] >= 60]["retention_rate"]
        render_test_results(dur_res, da, db)

    with ab_tabs[2]:
        st.subheader("Hypothesis: Evening Publishing (17:00–23:00) Captures Higher Engagement than Morning (06:00–12:00)")
        time_res = compare_posting_times(posts_df_all, "total_engagement")
        ta = posts_df_all[(posts_df_all["hour"] >= 6) & (posts_df_all["hour"] < 12)]["total_engagement"]
        tb = posts_df_all[(posts_df_all["hour"] >= 17) & (posts_df_all["hour"] <= 23)]["total_engagement"]
        render_test_results(time_res, ta, tb)

    with ab_tabs[3]:
        st.subheader("Custom Cohort Statistical Comparison")
        c_split_col = st.selectbox("Cohort Variable", ["topic", "platform", "content_type", "hook_type"])
        available_vals = sorted(posts_df_all[c_split_col].dropna().unique().tolist())
        if len(available_vals) >= 2:
            ccol1, ccol2, ccol3 = st.columns(3)
            with ccol1:
                val_a = st.selectbox("Group A Value", available_vals, index=0)
            with ccol2:
                val_b = st.selectbox("Group B Value", available_vals, index=1 if len(available_vals) > 1 else 0)
            with ccol3:
                target_m = st.selectbox("Metric to Compare", ["total_engagement", "viral_score", "shares", "saves", "retention_rate"])

            if val_a == val_b:
                st.warning("Please select two distinct group values.")
            else:
                sa = posts_df_all[posts_df_all[c_split_col] == val_a][target_m]
                sb = posts_df_all[posts_df_all[c_split_col] == val_b][target_m]
                c_res = run_statistical_test(sa, sb, f"{c_split_col}: {val_a}", f"{c_split_col}: {val_b}", target_m)
                render_test_results(c_res, sa, sb)


# ==============================================================================
# TAB 6: TREND ANALYSIS
# ==============================================================================
with tabs[5]:
    st.markdown('<div class="section-header">Trend Opportunity & Topic Momentum Analysis</div>', unsafe_allow_html=True)
    
    trend_df, trend_meta = compute_topic_trends(filtered_posts)
    
    st.caption(f"**Analysis Mode:** {trend_meta['analysis_type']}")
    st.markdown(
        f"""
        > ℹ️ *{trend_meta['note']}*
        """
    )

    if not trend_df.empty:
        # Opportunity Bar Chart
        fig_trend = px.bar(
            trend_df,
            x="trend_score",
            y="topic",
            orientation="h",
            color="momentum_tier",
            title="<b>Trend Opportunity Score by Thematic Category</b>",
            labels={"trend_score": "Trend Opportunity Score (0-100)", "topic": "Category"},
            color_discrete_map={
                "Rising Star (High Opportunity)": "#10B981",
                "Evergreen Consistent": "#38BDF8",
                "Low Momentum / Saturated": "#94A3B8",
            },
            text=trend_df["trend_score"].apply(lambda x: f"{x:.1f}"),
        )
        fig_trend.update_traces(textposition="outside")
        fig_trend.update_layout(height=450, margin=dict(l=40, r=40, t=50, b=40))
        st.plotly_chart(fig_trend, use_container_width=True)

        st.subheader("Comprehensive Topic Opportunity Matrix")
        st.dataframe(
            trend_df[
                [
                    "topic",
                    "trend_score",
                    "momentum_tier",
                    "post_count",
                    "avg_engagement",
                    "avg_viral_score",
                    "avg_shares",
                    "avg_saves",
                    "avg_retention",
                    "growth_rate_pct",
                ]
            ].style.format({
                "trend_score": "{:.1f}",
                "avg_engagement": "{:,.0f}",
                "avg_viral_score": "{:.1f}",
                "avg_shares": "{:,.0f}",
                "avg_saves": "{:,.0f}",
                "avg_retention": "{:.1f}%",
                "growth_rate_pct": "{:+.1f}%",
            }),
            use_container_width=True,
        )


# ==============================================================================
# TAB 7: STRATEGY RECOMMENDATIONS
# ==============================================================================
with tabs[6]:
    st.markdown('<div class="section-header">Prescriptive Content Strategy Engine</div>', unsafe_allow_html=True)
    
    recommendation = generate_content_recommendations(filtered_posts)

    st.markdown(
        f"""
        <div class="strategy-card">
            <div class="strategy-title">Optimal Content Strategy Blueprint</div>
            <h2 style="color: #F8FAFC; margin-top: 0; margin-bottom: 18px;">
                Topic Focus: <span style="color: #38BDF8;">{recommendation['topic']}</span>
            </h2>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 20px;">
                <div style="background: rgba(15, 23, 42, 0.6); padding: 12px; border-radius: 8px;">
                    <div style="color: #94A3B8; font-size: 0.75rem; text-transform: uppercase;">Recommended Platform</div>
                    <div style="color: #F8FAFC; font-size: 1.15rem; font-weight: 700;">{recommendation['platform']}</div>
                </div>
                <div style="background: rgba(15, 23, 42, 0.6); padding: 12px; border-radius: 8px;">
                    <div style="color: #94A3B8; font-size: 0.75rem; text-transform: uppercase;">Content Format</div>
                    <div style="color: #F8FAFC; font-size: 1.15rem; font-weight: 700;">{recommendation['content_type']}</div>
                </div>
                <div style="background: rgba(15, 23, 42, 0.6); padding: 12px; border-radius: 8px;">
                    <div style="color: #94A3B8; font-size: 0.75rem; text-transform: uppercase;">Hook Strategy</div>
                    <div style="color: #F8FAFC; font-size: 1.15rem; font-weight: 700;">{recommendation['hook_type']}</div>
                </div>
                <div style="background: rgba(15, 23, 42, 0.6); padding: 12px; border-radius: 8px;">
                    <div style="color: #94A3B8; font-size: 0.75rem; text-transform: uppercase;">Target Duration</div>
                    <div style="color: #F8FAFC; font-size: 1.15rem; font-weight: 700;">{recommendation['duration_range']}</div>
                </div>
                <div style="background: rgba(15, 23, 42, 0.6); padding: 12px; border-radius: 8px;">
                    <div style="color: #94A3B8; font-size: 0.75rem; text-transform: uppercase;">Publishing Window</div>
                    <div style="color: #F8FAFC; font-size: 1.15rem; font-weight: 700;">{recommendation['posting_window']}</div>
                </div>
                <div style="background: rgba(15, 23, 42, 0.6); padding: 12px; border-radius: 8px;">
                    <div style="color: #94A3B8; font-size: 0.75rem; text-transform: uppercase;">Expected Performance</div>
                    <div style="color: #10B981; font-size: 1.15rem; font-weight: 700;">{recommendation['expected_performance']}</div>
                </div>
            </div>
            <div style="background: rgba(15, 23, 42, 0.6); padding: 12px; border-radius: 8px;">
                <div style="color: #94A3B8; font-size: 0.75rem; text-transform: uppercase;">Caption & Hashtags</div>
                <div style="color: #E2E8F0; font-size: 0.95rem;">{recommendation['caption_style']}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Why This Strategy Is Recommended (Empirical Justifications)")
    for i, reason in enumerate(recommendation["reasons"], 1):
        st.success(f"**Reason {i}:** {reason}")


# ==============================================================================
# TAB 8: DATASET EXPLORER
# ==============================================================================
with tabs[7]:
    st.markdown('<div class="section-header">Dataset Explorer & Schema Inspector</div>', unsafe_allow_html=True)
    
    exp_tabs = st.tabs(["Cleaned Posts Dataset", "Audience Comments Dataset", "Corpus Statistics"])
    
    with exp_tabs[0]:
        st.caption(f"Showing {len(filtered_posts)} active post records.")
        st.dataframe(filtered_posts, use_container_width=True)
        csv_p = filtered_posts.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Cleaned Posts CSV",
            data=csv_p,
            file_name="cleaned_posts.csv",
            mime="text/csv",
        )

    with exp_tabs[1]:
        st.caption(f"Showing {len(filtered_comments)} audience comments.")
        st.dataframe(filtered_comments, use_container_width=True)
        csv_c = filtered_comments.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Processed Comments CSV",
            data=csv_c,
            file_name="processed_comments.csv",
            mime="text/csv",
        )

    with exp_tabs[2]:
        st.subheader("Statistical Summary of Post Features")
        st.dataframe(filtered_posts.describe(), use_container_width=True)


# ==============================================================================
# TAB 9: MODEL PERFORMANCE
# ==============================================================================
with tabs[8]:
    st.markdown('<div class="section-header">Machine Learning & NLP Model Evaluation</div>', unsafe_allow_html=True)

    st.caption("All metrics derived from independent stratified holdout test sets (80/20 train-test split).")

    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.subheader("Virality Classifier (RandomForest)")
        st.metric("Test Accuracy", f"{viral_metrics['accuracy']*100:.2f}%")
        st.metric("Weighted F1-Score", f"{viral_metrics['f1_score']:.4f}")
        st.metric("Weighted Precision", f"{viral_metrics['precision']:.4f}")
        st.metric("Weighted Recall", f"{viral_metrics['recall']:.4f}")

        st.plotly_chart(
            plot_confusion_matrix(
                viral_metrics["confusion_matrix"],
                viral_metrics["classes"],
                title="RandomForest Virality Confusion Matrix",
            ),
            use_container_width=True,
        )

        st.plotly_chart(
            plot_feature_importances(viral_metrics["feature_importances"]),
            use_container_width=True,
        )

    with m_col2:
        st.subheader("Audience Relatability Model (TF-IDF + Logistic Regression)")
        st.metric("Test Accuracy", f"{nlp_metrics['accuracy']*100:.2f}%")
        st.metric("Relatable F1-Score", f"{nlp_metrics['f1_score']:.4f}")
        st.metric("Precision", f"{nlp_metrics['precision']:.4f}")
        st.metric("Recall", f"{nlp_metrics['recall']:.4f}")

        st.plotly_chart(
            plot_confusion_matrix(
                nlp_metrics["confusion_matrix"],
                nlp_metrics["classes"],
                title="NLP Relatability Confusion Matrix",
            ),
            use_container_width=True,
        )

        st.subheader("Top Indicative Relatable Vocabulary Tokens")
        top_tokens = pd.DataFrame({
            "Relatable Indicative N-grams": nlp_metrics["top_relatable_terms"][:10],
            "Neutral Indicative N-grams": nlp_metrics["top_neutral_terms"][:10],
        })
        st.dataframe(top_tokens, use_container_width=True)


# ==============================================================================
# TAB 10: ABOUT PROJECT
# ==============================================================================
with tabs[9]:
    st.markdown('<div class="section-header">About The Major Project</div>', unsafe_allow_html=True)

    st.markdown(
        """
        ### THE DATA-DRIVEN SOCIAL ENGAGEMENT INITIATIVE
        **Evidence-Based Social Content Analytics & Intelligent Strategy Platform**
        *A BE CSE / Data Science Major Project Demonstration*

        ---

        #### 1. Problem Statement
        In modern digital marketing and social media ecosystems, content creation is often guided by subjective
        intuition rather than rigorous data analysis. Content creators and brands struggle to reliably quantify
        content engagement, understand audience resonance, and discover high-probability strategic choices.

        #### 2. Technical Architecture & Methodology
        This platform integrates classical machine learning, natural language processing, and statistical hypothesis
        testing into a unified decision support pipeline:
        * **Data Ingestion & Invariant Preprocessing:** Normalizes heterogeneous social schemas with alias mapping and range validations.
        * **Analytical Viral Scoring:** Transparent mathematical weighting of normalized metrics ($0-100$).
        * **Predictive Virality Classification:** Scikit-Learn `ColumnTransformer` + `RandomForestClassifier` with stratified evaluation.
        * **Dual-Dimension NLP:** TF-IDF with Logistic Regression for audience relatability classification, paired with TextBlob sentiment polarity.
        * **Statistical A/B Testing:** Welch's t-test and non-parametric Mann-Whitney U test with alpha = 0.05.
        * **Prescriptive Optimization:** Data-driven recommendation engine prescribing format, duration, hook, and posting window.

        ---

        #### 3. Honest Project Limitations
        1. **Observational Association vs Causality:** Observational A/B comparisons do not meet the criteria of randomized controlled trials (RCTs).
        2. **Synthetic Data Transparency:** Synthetic demo data is clearly labelled and used only when live user CSVs are unprovided.
        3. **Relatability Complexity:** Relatability is nuanced and relies on weak supervision heuristics in the absence of human-annotated labels.
        4. **Viral Score Index:** The Viral Score is an analytical benchmark index rather than an immutable law of nature.

        ---

        #### 4. Future Enhancements
        * Integration with official Meta Graph API and YouTube Data API v3.
        * Transformer-based LLM sentiment analysis (BERT / RoBERTa).
        * Multilingual and cross-platform trend scraping firehose.
        * Time-series forecasting for long-term audience retention using ARIMA / Prophet.
        """
    )
