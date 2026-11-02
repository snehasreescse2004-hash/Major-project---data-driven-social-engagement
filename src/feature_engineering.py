"""
Feature engineering module for The Data-Driven Social Engagement Initiative.
Derives engagement metrics, interaction ratios, min-max normalized components,
and calculates the transparent project-defined Viral Score and categorization.
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd

from config import VIRAL_THRESHOLDS, VIRAL_WEIGHTS


def min_max_normalize(series: pd.Series, lower_quantile: float = 0.01, upper_quantile: float = 0.99) -> pd.Series:
    """
    Min-max normalize a numeric series between 0.0 and 1.0 using robust quantile capping
    to prevent extreme outliers from crushing the variance of typical posts.
    """
    min_val = series.quantile(lower_quantile)
    max_val = series.quantile(upper_quantile)

    if max_val <= min_val or pd.isna(min_val) or pd.isna(max_val):
        return pd.Series(0.5, index=series.index)

    clipped = series.clip(lower=min_val, upper=max_val)
    return (clipped - min_val) / (max_val - min_val)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute domain-specific interaction metrics, ratios, rates, and the Viral Score index.
    """
    df = df.copy()

    # 1. Total Engagement
    df["total_engagement"] = (
        df["likes"] + df["comments_count"] + df["shares"] + df["saves"]
    )

    # 2. Engagement Rate Calculation
    # If explicit 'reach' or 'impressions' column exists in imported data, use total_engagement / reach * 100.
    # Otherwise, compute a standardized benchmark engagement rate based on mean pool exposure.
    if "reach" in df.columns and (df["reach"] > 0).any():
        reach_safe = df["reach"].replace(0, np.nan)
        df["engagement_rate"] = (df["total_engagement"] / reach_safe) * 100.0
        df["engagement_rate"] = df["engagement_rate"].fillna(df["engagement_rate"].median()).clip(0.0, 100.0)
    else:
        # Standardized analytical proxy: engagement relative to expected exposure baseline
        baseline = df["total_engagement"].median()
        baseline = baseline if (pd.notna(baseline) and baseline > 0) else 100.0
        # Scaled to a realistic engagement percentage between 1.5% and 18%
        df["engagement_rate"] = np.clip((df["total_engagement"] / (baseline * 8.0)) * 100.0, 0.5, 25.0)

    # 3. Micro-Interaction Rates & Ratios
    # Adding +1 smoothing to avoid zero-division
    denom = df["total_engagement"] + 1.0

    df["save_share_ratio"] = df["saves"] / (df["shares"] + 1.0)
    df["comment_rate"] = (df["comments_count"] / denom) * 100.0
    df["share_rate"] = (df["shares"] / denom) * 100.0
    df["save_rate"] = (df["saves"] / denom) * 100.0
    df["like_rate"] = (df["likes"] / denom) * 100.0
    df["follower_growth_rate"] = (df["followers_gained"] / denom) * 100.0

    # 4. Normalized Components for Transparent Viral Score
    norm_likes = min_max_normalize(df["likes"])
    norm_shares = min_max_normalize(df["shares"])
    norm_saves = min_max_normalize(df["saves"])
    norm_retention = df["retention_rate"] / 100.0  # Already in 0-100 scale
    norm_comments = min_max_normalize(df["comments_count"])
    norm_followers = min_max_normalize(df["followers_gained"])

    # 5. Viral Score Formula
    # viral_score = 0.15*L + 0.25*S + 0.25*V + 0.20*R + 0.10*C + 0.05*F, scaled 0 to 100
    raw_viral = (
        VIRAL_WEIGHTS["likes"] * norm_likes
        + VIRAL_WEIGHTS["shares"] * norm_shares
        + VIRAL_WEIGHTS["saves"] * norm_saves
        + VIRAL_WEIGHTS["retention"] * norm_retention
        + VIRAL_WEIGHTS["comments"] * norm_comments
        + VIRAL_WEIGHTS["followers_gained"] * norm_followers
    )

    df["viral_score"] = np.round(raw_viral * 100.0, 2)
    df["viral_score"] = np.clip(df["viral_score"], 0.0, 100.0)

    # 6. Viral Category Classification
    def classify_virality(score: float) -> str:
        if score < VIRAL_THRESHOLDS["low_max"]:
            return "Low"
        elif score < VIRAL_THRESHOLDS["medium_max"]:
            return "Medium"
        else:
            return "High"

    df["viral_category"] = df["viral_score"].apply(classify_virality)

    return df


def calculate_single_viral_score(
    likes: float,
    shares: float,
    saves: float,
    comments: float,
    retention: float,
    followers_gained: float,
    normalization_reference: Dict[str, Tuple[float, float]],
) -> Tuple[float, str]:
    """
    Calculate Viral Score and category for an individual post or simulated input.
    Uses reference min-max bounds derived from the historical dataset.
    """
    def scale(val: float, key: str) -> float:
        min_v, max_v = normalization_reference.get(key, (0.0, 1000.0))
        if max_v <= min_v:
            return 0.5
        clamped = max(min_v, min(max_v, val))
        return (clamped - min_v) / (max_v - min_v)

    norm_l = scale(likes, "likes")
    norm_s = scale(shares, "shares")
    norm_v = scale(saves, "saves")
    norm_r = max(0.0, min(100.0, retention)) / 100.0
    norm_c = scale(comments, "comments_count")
    norm_f = scale(followers_gained, "followers_gained")

    score = (
        VIRAL_WEIGHTS["likes"] * norm_l
        + VIRAL_WEIGHTS["shares"] * norm_s
        + VIRAL_WEIGHTS["saves"] * norm_v
        + VIRAL_WEIGHTS["retention"] * norm_r
        + VIRAL_WEIGHTS["comments"] * norm_c
        + VIRAL_WEIGHTS["followers_gained"] * norm_f
    ) * 100.0

    score = round(float(np.clip(score, 0.0, 100.0)), 2)

    if score < VIRAL_THRESHOLDS["low_max"]:
        category = "Low"
    elif score < VIRAL_THRESHOLDS["medium_max"]:
        category = "Medium"
    else:
        category = "High"

    return score, category


def get_normalization_bounds(df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    """
    Extract reference quantile boundaries for single-observation simulation.
    """
    bounds = {}
    for col in ["likes", "shares", "saves", "comments_count", "followers_gained"]:
        if col in df.columns:
            low = float(df[col].quantile(0.01))
            high = float(df[col].quantile(0.99))
            bounds[col] = (low, high if high > low else low + 100.0)
        else:
            bounds[col] = (0.0, 1000.0)
    return bounds
