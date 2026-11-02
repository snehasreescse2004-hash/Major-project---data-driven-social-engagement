"""
Prescriptive Recommendation Engine for The Data-Driven Social Engagement Initiative.
Analyzes historical performance clusters in the active dataset slice to dynamically prescribe
the optimal content strategy, format, hook, timing, and empirical justification.
"""

from typing import Any, Dict, List
import numpy as np
import pandas as pd


def generate_content_recommendations(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate prescriptive strategy recommendations dynamically from the provided dataset slice.
    """
    if df.empty or len(df) < 5:
        return {
            "topic": "Mental Wellness",
            "platform": "Instagram",
            "content_type": "Reel",
            "hook_type": "Visual",
            "duration_range": "30–45 seconds",
            "posting_window": "18:00–21:00",
            "caption_style": "120–160 characters with 4–6 hashtags",
            "expected_performance": "Medium to High",
            "reasons": [
                "Baseline recommendation generated from general engagement heuristics.",
                "Short video formats reliably maintain high completion rates across diverse cohorts.",
                "Visual hooks capture immediate initial attention within the first 3 seconds.",
            ],
        }

    # 1. Optimal Topic (weighted by mean Viral Score and engagement)
    topic_summary = (
        df.groupby("topic")
        .agg(
            viral_score=("viral_score", "mean"),
            engagement=("total_engagement", "mean"),
            saves=("saves", "mean"),
            shares=("shares", "mean"),
            count=("post_id", "count"),
        )
        .reset_index()
    )
    # Require at least 2 posts in topic if possible
    valid_topics = topic_summary[topic_summary["count"] >= 2]
    if valid_topics.empty:
        valid_topics = topic_summary

    best_topic_row = valid_topics.sort_values(by="viral_score", ascending=False).iloc[0]
    best_topic = str(best_topic_row["topic"])
    best_topic_viral = float(best_topic_row["viral_score"])
    best_topic_saves = float(best_topic_row["saves"])

    # Filter to top topic or entire slice for nuanced format selection
    topic_slice = df[df["topic"] == best_topic]
    sub_df = topic_slice if len(topic_slice) >= 3 else df

    # 2. Optimal Platform
    platform_summary = sub_df.groupby("platform")["viral_score"].mean()
    best_platform = str(platform_summary.idxmax()) if not platform_summary.empty else "Instagram"

    # 3. Optimal Content Type
    content_summary = sub_df.groupby("content_type").agg(
        viral_score=("viral_score", "mean"),
        retention=("retention_rate", "mean"),
    )
    best_content_type = str(content_summary["viral_score"].idxmax()) if not content_summary.empty else "Reel"
    best_content_retention = float(content_summary.loc[best_content_type, "retention"]) if best_content_type in content_summary.index else 60.0

    # 4. Optimal Hook Type
    hook_summary = sub_df.groupby("hook_type")["total_engagement"].mean()
    best_hook = str(hook_summary.idxmax()) if not hook_summary.empty else "Visual"
    best_hook_eng = float(hook_summary.max()) if not hook_summary.empty else 0.0

    # 5. Optimal Duration Range
    duration_slice = sub_df[sub_df["duration"] > 0]["duration"]
    if duration_slice.empty:
        duration_slice = df[df["duration"] > 0]["duration"]

    if not duration_slice.empty:
        q25 = int(np.percentile(duration_slice, 25))
        q75 = int(np.percentile(duration_slice, 75))
        q25 = max(15, q25)
        q75 = max(q25 + 10, q75)
        duration_range = f"{q25}–{q75} seconds"
    else:
        duration_range = "30–60 seconds"

    # 6. Optimal Posting Window
    hour_summary = df.groupby("hour")["total_engagement"].mean()
    if not hour_summary.empty:
        peak_hour = int(hour_summary.idxmax())
        start_h = max(0, peak_hour - 1)
        end_h = min(23, peak_hour + 2)
        posting_window = f"{start_h:02d}:00–{end_h:02d}:00"
    else:
        posting_window = "18:00–21:00"

    # 7. Caption & Hashtag Style
    top_performers = df[df["viral_category"] == "High"]
    if len(top_performers) < 3:
        top_performers = df.sort_values(by="viral_score", ascending=False).head(max(5, int(len(df) * 0.2)))

    avg_caption = int(top_performers["caption_length"].mean()) if not top_performers.empty else 135
    avg_tags = int(top_performers["hashtag_count"].mean()) if not top_performers.empty else 5
    caption_style = f"{max(40, avg_caption - 25)}–{avg_caption + 25} characters with {max(2, avg_tags - 1)}–{avg_tags + 2} hashtags"

    # 8. Expected Performance
    overall_mean_viral = df["viral_score"].mean()
    expected_performance = "High (Viral Score > 70)" if best_topic_viral >= 65.0 else "Medium (Viral Score 50–70)"

    # 9. Dynamic Data-Backed Rationale Generation
    reasons = [
        f"Topic '{best_topic}' demonstrated dominant performance with an average Viral Score of {best_topic_viral:.1f} and {best_topic_saves:,.0f} average saves.",
        f"Format '{best_content_type}' achieved an impressive {best_content_retention:.1f}% average audience retention rate within this thematic cluster.",
        f"Posts incorporating '{best_hook}' hooks captured the highest interaction velocity (averaging {best_hook_eng:,.0f} total engagements).",
        f"The {posting_window} posting window aligns with empirical historical peak traffic hours across {best_platform}.",
        f"Content calibrated with {caption_style} closely mirrors high-tier benchmark posts.",
    ]

    return {
        "topic": best_topic,
        "platform": best_platform,
        "content_type": best_content_type,
        "hook_type": best_hook,
        "duration_range": duration_range,
        "posting_window": posting_window,
        "caption_style": caption_style,
        "expected_performance": expected_performance,
        "reasons": reasons,
    }
