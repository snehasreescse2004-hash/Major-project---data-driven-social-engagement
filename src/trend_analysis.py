"""
Trend analysis module for The Data-Driven Social Engagement Initiative.
Evaluates historical content velocity, engagement momentum, and derives
a standardized Trend Opportunity Score across topics.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

TREND_NOTE = (
    "Analytical Note: This module models retrospective velocity and engagement momentum "
    "from available historical dataset records. It reflects empirical dataset opportunity "
    "and should not be conflated with an external real-time live social API trend firehose."
)


def compute_topic_trends(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Compute comprehensive topic performance metrics and Trend Opportunity Scores.
    If posting_time exists, calculates recent-vs-earlier chronological growth.
    """
    if df.empty or "topic" not in df.columns:
        return pd.DataFrame(), {"analysis_type": "None", "note": TREND_NOTE}

    has_dates = "posting_time" in df.columns and pd.to_datetime(df["posting_time"], errors="coerce").notna().any()

    # Base aggregate metrics by topic
    topic_agg = (
        df.groupby("topic")
        .agg(
            post_count=("post_id", "count"),
            avg_engagement=("total_engagement", "mean"),
            avg_viral_score=("viral_score", "mean"),
            avg_shares=("shares", "mean"),
            avg_saves=("saves", "mean"),
            avg_retention=("retention_rate", "mean"),
            avg_followers_gained=("followers_gained", "mean"),
            avg_save_share_ratio=("save_share_ratio", "mean"),
        )
        .reset_index()
    )

    # Calculate recent-vs-earlier growth if temporal data is valid
    if has_dates:
        analysis_type = "Historical Temporal Momentum Analysis"
        df_sorted = df.sort_values("posting_time")
        split_idx = int(len(df_sorted) * 0.5)
        earlier_df = df_sorted.iloc[:split_idx]
        recent_df = df_sorted.iloc[split_idx:]

        earlier_eng = earlier_df.groupby("topic")["total_engagement"].mean()
        recent_eng = recent_df.groupby("topic")["total_engagement"].mean()

        growth_rates = {}
        for topic in topic_agg["topic"]:
            e_val = earlier_eng.get(topic, np.nan)
            r_val = recent_eng.get(topic, np.nan)
            if pd.notna(e_val) and pd.notna(r_val) and e_val > 0:
                growth_rates[topic] = ((r_val - e_val) / e_val) * 100.0
            else:
                growth_rates[topic] = 0.0

        topic_agg["growth_rate_pct"] = topic_agg["topic"].map(growth_rates).fillna(0.0)
    else:
        analysis_type = "Trend Opportunity Analysis (Static Historical)"
        topic_agg["growth_rate_pct"] = 0.0

    # Min-max normalize components for trend score
    def normalize(s: pd.Series) -> pd.Series:
        rng = s.max() - s.min()
        return (s - s.min()) / (rng + 1e-9)

    norm_growth = normalize(topic_agg["growth_rate_pct"])
    norm_eng = normalize(topic_agg["avg_engagement"])
    norm_freq = normalize(topic_agg["post_count"])

    # Trend Opportunity Score Formula:
    # 0.40 * norm_growth + 0.30 * norm_eng + 0.30 * norm_freq
    if has_dates and topic_agg["growth_rate_pct"].abs().sum() > 0:
        raw_trend = 0.40 * norm_growth + 0.30 * norm_eng + 0.30 * norm_freq
    else:
        # Fallback if no date variation: 0.60 * norm_eng + 0.40 * norm_freq
        raw_trend = 0.60 * norm_eng + 0.40 * norm_freq

    topic_agg["trend_score"] = np.round(raw_trend * 100.0, 1)
    topic_agg = topic_agg.sort_values(by="trend_score", ascending=False).reset_index(drop=True)

    # Classify Momentum Tier
    def classify_momentum(row: pd.Series) -> str:
        score = row["trend_score"]
        if score >= 70:
            return "Rising Star (High Opportunity)"
        elif score >= 45:
            return "Evergreen Consistent"
        else:
            return "Low Momentum / Saturated"

    topic_agg["momentum_tier"] = topic_agg.apply(classify_momentum, axis=1)

    metadata = {
        "analysis_type": analysis_type,
        "note": TREND_NOTE,
        "top_topic": topic_agg.iloc[0]["topic"] if not topic_agg.empty else "N/A",
        "top_trend_score": topic_agg.iloc[0]["trend_score"] if not topic_agg.empty else 0.0,
    }

    return topic_agg, metadata
