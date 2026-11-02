"""
Interactive Plotly visualizations module for The Data-Driven Social Engagement Initiative.
Provides high-fidelity, responsive data visualizations for EDA, virality metrics,
NLP relatability/sentiment, and statistical A/B distributions.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config import SENTIMENT_COLORS, THEME_COLORS, VIRAL_COLORS

# Default visual layout template for consistent professional aesthetics
LAYOUT_DEFAULTS = dict(
    font=dict(family="Inter, Roboto, sans-serif", size=12, color="#E2E8F0"),
    paper_bgcolor="rgba(15, 23, 42, 0.0)",  # Transparent for Streamlit dark/light adaptation
    plot_bgcolor="rgba(30, 41, 59, 0.3)",
    margin=dict(l=40, r=30, t=50, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)


def plot_engagement_distribution(df: pd.DataFrame) -> go.Figure:
    """Plot distribution of total engagement with median and mean markers."""
    fig = px.histogram(
        df,
        x="total_engagement",
        color="viral_category",
        color_discrete_map=VIRAL_COLORS,
        nbins=40,
        marginal="box",
        title="<b>Total Engagement Distribution across Content Tiers</b>",
        labels={"total_engagement": "Total Engagement (Likes + Shares + Saves + Comments)", "viral_category": "Viral Tier"},
    )
    fig.update_layout(**LAYOUT_DEFAULTS)
    fig.update_xaxes(showgrid=True, gridcolor="#334155")
    fig.update_yaxes(showgrid=True, gridcolor="#334155")
    return fig


def plot_engagement_by_topic(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart showing average total engagement and viral score by topic."""
    agg = (
        df.groupby("topic")
        .agg(
            avg_engagement=("total_engagement", "mean"),
            avg_viral_score=("viral_score", "mean"),
            post_count=("post_id", "count"),
        )
        .reset_index()
        .sort_values(by="avg_engagement", ascending=True)
    )

    fig = px.bar(
        agg,
        x="avg_engagement",
        y="topic",
        orientation="h",
        color="avg_viral_score",
        color_continuous_scale="Viridis",
        title="<b>Mean Total Engagement & Viral Score by Topic</b>",
        labels={"avg_engagement": "Average Total Engagement", "topic": "Thematic Category", "avg_viral_score": "Avg Viral Score"},
        text=agg["avg_engagement"].apply(lambda x: f"{x:,.0f}"),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(**LAYOUT_DEFAULTS)
    fig.update_xaxes(showgrid=True, gridcolor="#334155")
    return fig


def plot_engagement_by_content_type(df: pd.DataFrame) -> go.Figure:
    """Bar chart comparing engagement across content formats with error bars / spread."""
    agg = (
        df.groupby("content_type")
        .agg(
            avg_engagement=("total_engagement", "mean"),
            avg_retention=("retention_rate", "mean"),
        )
        .reset_index()
        .sort_values(by="avg_engagement", ascending=False)
    )

    fig = px.bar(
        agg,
        x="content_type",
        y="avg_engagement",
        color="content_type",
        title="<b>Performance by Content Format</b>",
        labels={"content_type": "Content Format", "avg_engagement": "Average Engagement"},
        color_discrete_sequence=px.colors.qualitative.Prism,
        text=agg["avg_engagement"].apply(lambda x: f"{x:,.0f}"),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(**LAYOUT_DEFAULTS, showlegend=False)
    fig.update_yaxes(showgrid=True, gridcolor="#334155")
    return fig


def plot_engagement_by_hook(df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart comparing hook strategies across viral tiers."""
    agg = (
        df.groupby(["hook_type", "viral_category"])["post_id"]
        .count()
        .reset_index(name="post_count")
    )

    fig = px.bar(
        agg,
        x="hook_type",
        y="post_count",
        color="viral_category",
        barmode="group",
        color_discrete_map=VIRAL_COLORS,
        title="<b>Hook Strategy Distribution by Virality Tier</b>",
        labels={"hook_type": "Opening Hook Type", "post_count": "Post Volume", "viral_category": "Viral Tier"},
    )
    fig.update_layout(**LAYOUT_DEFAULTS)
    fig.update_yaxes(showgrid=True, gridcolor="#334155")
    return fig


def plot_engagement_by_platform(df: pd.DataFrame) -> go.Figure:
    """Donut chart illustrating total engagement share across platforms."""
    agg = df.groupby("platform")["total_engagement"].sum().reset_index()

    fig = px.pie(
        agg,
        names="platform",
        values="total_engagement",
        hole=0.45,
        title="<b>Audience Engagement Share by Social Platform</b>",
        color_discrete_sequence=[THEME_COLORS["primary"], THEME_COLORS["secondary"], THEME_COLORS["accent"]],
    )
    fig.update_traces(textinfo="percent+label", pull=[0.05, 0, 0])
    fig.update_layout(**LAYOUT_DEFAULTS)
    return fig


def plot_posting_hour_performance(df: pd.DataFrame) -> go.Figure:
    """Line and area chart showing hourly engagement fluctuations across 24-hour cycle."""
    agg = (
        df.groupby("hour")
        .agg(
            avg_engagement=("total_engagement", "mean"),
            post_count=("post_id", "count"),
        )
        .reindex(range(24), fill_value=0)
        .reset_index()
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=agg["hour"],
            y=agg["avg_engagement"],
            mode="lines+markers",
            name="Avg Engagement",
            line=dict(color=THEME_COLORS["secondary"], width=3),
            marker=dict(size=7, color=THEME_COLORS["primary"]),
            fill="tozeroy",
            fillcolor="rgba(6, 182, 212, 0.15)",
        )
    )

    # Highlight prime evening window 18-21
    fig.add_vrect(
        x0=17.5,
        x1=21.5,
        fillcolor="rgba(79, 70, 229, 0.18)",
        layer="below",
        line_width=0,
        annotation_text="Prime Traffic Window (18:00 - 21:00)",
        annotation_position="top left",
    )

    fig.update_layout(
        title="<b>24-Hour Chronological Engagement Velocity</b>",
        xaxis=dict(title="Hour of Day (00:00 - 23:00)", tickmode="linear", tick0=0, dtick=2, showgrid=True, gridcolor="#334155"),
        yaxis=dict(title="Average Interactions", showgrid=True, gridcolor="#334155"),
        **LAYOUT_DEFAULTS,
    )
    return fig


def plot_retention_vs_engagement(df: pd.DataFrame) -> go.Figure:
    """Scatter plot mapping audience retention rate against total engagement."""
    fig = px.scatter(
        df,
        x="retention_rate",
        y="total_engagement",
        color="viral_category",
        color_discrete_map=VIRAL_COLORS,
        size="shares",
        hover_data=["post_id", "topic", "content_type", "hook_type"],
        title="<b>Audience Retention Rate vs. Total Engagement</b>",
        labels={"retention_rate": "Audience Retention Rate (%)", "total_engagement": "Total Engagement", "viral_category": "Viral Tier"},
    )
    fig.update_layout(**LAYOUT_DEFAULTS)
    fig.update_xaxes(showgrid=True, gridcolor="#334155")
    fig.update_yaxes(showgrid=True, gridcolor="#334155")
    return fig


def plot_shares_vs_saves(df: pd.DataFrame) -> go.Figure:
    """Quadrant scatter plot comparing content utility (saves) vs content resonance (shares)."""
    fig = px.scatter(
        df,
        x="shares",
        y="saves",
        color="topic",
        hover_data=["post_id", "content_type", "viral_score"],
        title="<b>Content Utility vs. Resonance: Saves vs. Shares Matrix</b>",
        labels={"shares": "Content Shares (Resonance)", "saves": "Content Saves (Utility)", "topic": "Thematic Topic"},
    )
    # Add quadrant reference lines at medians
    med_shares = df["shares"].median()
    med_saves = df["saves"].median()

    fig.add_vline(x=med_shares, line_dash="dash", line_color="#94A3B8", opacity=0.6)
    fig.add_hline(y=med_saves, line_dash="dash", line_color="#94A3B8", opacity=0.6)

    fig.update_layout(**LAYOUT_DEFAULTS)
    fig.update_xaxes(showgrid=True, gridcolor="#334155")
    fig.update_yaxes(showgrid=True, gridcolor="#334155")
    return fig


def plot_likes_vs_shares(df: pd.DataFrame) -> go.Figure:
    """Scatter plot illustrating correlation between raw likes and active shares."""
    fig = px.scatter(
        df,
        x="likes",
        y="shares",
        color="platform",
        title="<b>Likes vs. Shares Correlation</b>",
        labels={"likes": "Likes Count", "shares": "Shares Count"},
    )
    fig.update_layout(**LAYOUT_DEFAULTS)
    fig.update_xaxes(showgrid=True, gridcolor="#334155")
    fig.update_yaxes(showgrid=True, gridcolor="#334155")
    return fig


def plot_topic_follower_growth(df: pd.DataFrame) -> go.Figure:
    """Bar chart ranking topics by average net follower gain per post."""
    agg = (
        df.groupby("topic")["followers_gained"]
        .mean()
        .reset_index()
        .sort_values(by="followers_gained", ascending=True)
    )

    fig = px.bar(
        agg,
        x="followers_gained",
        y="topic",
        orientation="h",
        color="followers_gained",
        color_continuous_scale="Tealgrn",
        title="<b>Net Follower Acquisition by Thematic Topic</b>",
        labels={"followers_gained": "Avg Followers Gained / Post", "topic": "Topic"},
        text=agg["followers_gained"].apply(lambda x: f"{x:,.0f}"),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(**LAYOUT_DEFAULTS)
    fig.update_xaxes(showgrid=True, gridcolor="#334155")
    return fig


def plot_save_to_share_ratio(df: pd.DataFrame) -> go.Figure:
    """Bar chart displaying save-to-share ratio by topic (actionable bookmarking index)."""
    agg = (
        df.groupby("topic")["save_share_ratio"]
        .mean()
        .reset_index()
        .sort_values(by="save_share_ratio", ascending=True)
    )

    fig = px.bar(
        agg,
        x="save_share_ratio",
        y="topic",
        orientation="h",
        color="save_share_ratio",
        color_continuous_scale="Purp",
        title="<b>Actionable Bookmark Index: Save-to-Share Ratio by Topic</b>",
        labels={"save_share_ratio": "Average Save-to-Share Ratio", "topic": "Topic"},
        text=agg["save_share_ratio"].apply(lambda x: f"{x:.2f}"),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(**LAYOUT_DEFAULTS)
    fig.update_xaxes(showgrid=True, gridcolor="#334155")
    return fig


def plot_confusion_matrix(cm: List[List[int]], classes: List[str], title: str = "Confusion Matrix") -> go.Figure:
    """Plotly interactive confusion matrix heatmap."""
    z = np.array(cm)
    fig = px.imshow(
        z,
        x=classes,
        y=classes,
        color_continuous_scale="Blues",
        labels=dict(x="Predicted Class", y="Ground Truth Class", color="Count"),
        title=f"<b>{title}</b>",
        text_auto=True,
    )
    fig.update_layout(**LAYOUT_DEFAULTS)
    return fig


def plot_feature_importances(feature_data: List[Dict[str, Any]]) -> go.Figure:
    """Horizontal bar chart showing top Gini feature importances from RandomForest."""
    df_feat = pd.DataFrame(feature_data).sort_values(by="importance", ascending=True)

    fig = px.bar(
        df_feat,
        x="importance",
        y="feature",
        orientation="h",
        title="<b>RandomForest Gini Feature Importances for Virality Prediction</b>",
        labels={"importance": "Importance Metric", "feature": "Predictor Feature"},
        color="importance",
        color_continuous_scale="Plotly3",
        text=df_feat["importance"].apply(lambda x: f"{x:.3f}"),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(**LAYOUT_DEFAULTS)
    fig.update_xaxes(showgrid=True, gridcolor="#334155")
    return fig


def plot_relatable_breakdown(comments_df: pd.DataFrame) -> go.Figure:
    """Donut chart illustrating Relatable vs Neutral comment volume."""
    agg = comments_df["relatability"].value_counts().reset_index()
    agg.columns = ["relatability", "count"]

    fig = px.pie(
        agg,
        names="relatability",
        values="count",
        hole=0.45,
        title="<b>Audience Relatability Classification Distribution</b>",
        color="relatability",
        color_discrete_map={"RELATABLE": THEME_COLORS["primary"], "NEUTRAL": THEME_COLORS["neutral"]},
    )
    fig.update_traces(textinfo="percent+label", pull=[0.05, 0])
    fig.update_layout(**LAYOUT_DEFAULTS)
    return fig


def plot_sentiment_breakdown(comments_df: pd.DataFrame) -> go.Figure:
    """Donut chart of sentiment polarity classes."""
    agg = comments_df["sentiment"].value_counts().reset_index()
    agg.columns = ["sentiment", "count"]

    fig = px.pie(
        agg,
        names="sentiment",
        values="count",
        hole=0.45,
        title="<b>Audience Comment Sentiment Polarity</b>",
        color="sentiment",
        color_discrete_map=SENTIMENT_COLORS,
    )
    fig.update_traces(textinfo="percent+label")
    fig.update_layout(**LAYOUT_DEFAULTS)
    return fig


def plot_relatability_by_topic(comments_df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart showing relatability percentage across topics."""
    if "topic" not in comments_df.columns:
        return go.Figure()

    grouped = (
        pd.crosstab(comments_df["topic"], comments_df["relatability"], normalize="index") * 100.0
    ).reset_index()

    relatable_col = "RELATABLE" if "RELATABLE" in grouped.columns else None
    if relatable_col:
        grouped = grouped.sort_values(by=relatable_col, ascending=True)

    fig = px.bar(
        grouped,
        x=relatable_col if relatable_col else "topic",
        y="topic",
        orientation="h",
        title="<b>Percentage of Relatable Comments by Topic</b>",
        labels={relatable_col: "% Relatable Comments", "topic": "Thematic Category"},
        color=relatable_col,
        color_continuous_scale="Purples",
        text=grouped[relatable_col].apply(lambda x: f"{x:.1f}%") if relatable_col else None,
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(**LAYOUT_DEFAULTS)
    fig.update_xaxes(showgrid=True, gridcolor="#334155")
    return fig


def plot_ab_comparison(
    group_a: pd.Series,
    group_b: pd.Series,
    group_a_name: str,
    group_b_name: str,
    metric_name: str,
) -> go.Figure:
    """Box and violin comparison plot for A/B testing cohorts."""
    df_plot = pd.concat([
        pd.DataFrame({"Cohort": group_a_name, "Value": pd.to_numeric(group_a, errors="coerce").dropna()}),
        pd.DataFrame({"Cohort": group_b_name, "Value": pd.to_numeric(group_b, errors="coerce").dropna()}),
    ]).reset_index(drop=True)

    fig = px.box(
        df_plot,
        x="Cohort",
        y="Value",
        color="Cohort",
        points="all",
        title=f"<b>Statistical Distribution Comparison: {metric_name}</b>",
        labels={"Value": metric_name, "Cohort": "Test Group"},
        color_discrete_sequence=[THEME_COLORS["primary"], THEME_COLORS["accent"]],
    )
    fig.update_layout(**LAYOUT_DEFAULTS, showlegend=False)
    fig.update_yaxes(showgrid=True, gridcolor="#334155")
    return fig
