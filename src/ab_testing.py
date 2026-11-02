"""
Statistical A/B Testing module for The Data-Driven Social Engagement Initiative.
Executes rigorous parametric (Welch's t-test) and non-parametric (Mann-Whitney U)
statistical evaluations across content strategies, with formal methodological caveats.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats

from config import SIGNIFICANCE_ALPHA

METHODOLOGICAL_DISCLAIMER = (
    "Methodological Note: This analysis evaluates empirical statistical association within "
    "the available observational dataset. Observational comparisons do not satisfy the criteria "
    "of a randomized controlled trial (RCT) and must not be interpreted as definitive proof of causality."
)


def run_statistical_test(
    group_a: pd.Series,
    group_b: pd.Series,
    group_a_label: str = "Group A",
    group_b_label: str = "Group B",
    metric_name: str = "Engagement",
    alpha: float = SIGNIFICANCE_ALPHA,
) -> Dict[str, Any]:
    """
    Perform dual statistical evaluation (Welch's t-test & Mann-Whitney U) between two cohort samples.
    """
    clean_a = pd.to_numeric(group_a, errors="coerce").dropna()
    clean_b = pd.to_numeric(group_b, errors="coerce").dropna()

    count_a = len(clean_a)
    count_b = len(clean_b)

    if count_a < 3 or count_b < 3:
        return {
            "error": f"Insufficient sample size for statistical test (Group A: {count_a}, Group B: {count_b}, min required: 3).",
            "sample_size_a": count_a,
            "sample_size_b": count_b,
        }

    mean_a = float(clean_a.mean())
    mean_b = float(clean_b.mean())
    std_a = float(clean_a.std(ddof=1)) if count_a > 1 else 0.0
    std_b = float(clean_b.std(ddof=1)) if count_b > 1 else 0.0
    median_a = float(clean_a.median())
    median_b = float(clean_b.median())

    diff = mean_a - mean_b
    pct_diff = ((mean_a - mean_b) / (mean_b + 1e-9)) * 100.0

    # 1. Welch's t-test (unequal variance assumed)
    t_stat, p_val_t = stats.ttest_ind(clean_a, clean_b, equal_var=False, nan_policy="omit")
    t_stat = float(t_stat) if pd.notna(t_stat) else 0.0
    p_val_t = float(p_val_t) if pd.notna(p_val_t) else 1.0

    # 2. Mann-Whitney U (Non-parametric alternative, robust to skew)
    try:
        u_stat, p_val_u = stats.mannwhitneyu(clean_a, clean_b, alternative="two-sided")
        u_stat = float(u_stat)
        p_val_u = float(p_val_u)
    except Exception:
        u_stat, p_val_u = 0.0, 1.0

    is_significant = p_val_t < alpha

    if is_significant:
        significance_tag = "Statistically significant"
        winner = group_a_label if diff > 0 else group_b_label
        narrative = (
            f"The difference in {metric_name} between {group_a_label} and {group_b_label} "
            f"is statistically significant (p = {p_val_t:.4f} < {alpha}). "
            f"{winner} demonstrated an advantage of {abs(pct_diff):.1f}%."
        )
    else:
        significance_tag = "Not statistically significant"
        narrative = (
            f"The observed difference in {metric_name} between {group_a_label} and {group_b_label} "
            f"is not statistically significant (p = {p_val_t:.4f} >= {alpha}). "
            f"The variation may likely be attributed to random observational noise."
        )

    return {
        "metric_name": metric_name,
        "group_a_label": group_a_label,
        "group_b_label": group_b_label,
        "count_a": count_a,
        "count_b": count_b,
        "mean_a": round(mean_a, 2),
        "mean_b": round(mean_b, 2),
        "std_a": round(std_a, 2),
        "std_b": round(std_b, 2),
        "median_a": round(median_a, 2),
        "median_b": round(median_b, 2),
        "difference": round(diff, 2),
        "percent_difference": round(pct_diff, 2),
        "welch_t_statistic": round(t_stat, 4),
        "welch_p_value": round(p_val_t, 5),
        "mann_whitney_u": round(u_stat, 2),
        "mann_whitney_p_value": round(p_val_u, 5),
        "is_significant": is_significant,
        "significance_conclusion": significance_tag,
        "narrative": narrative,
        "disclaimer": METHODOLOGICAL_DISCLAIMER,
    }


def compare_hook_types(
    df: pd.DataFrame,
    metric: str = "total_engagement",
    hook_a: str = "Visual",
    hook_b: str = "Text",
) -> Dict[str, Any]:
    """Test differences between two Hook strategies (e.g. Visual vs Text)."""
    ga = df[df["hook_type"] == hook_a][metric]
    gb = df[df["hook_type"] == hook_b][metric]
    return run_statistical_test(ga, gb, f"Hook: {hook_a}", f"Hook: {hook_b}", metric_name=metric)


def compare_durations(
    df: pd.DataFrame,
    metric: str = "total_engagement",
    threshold_seconds: int = 60,
) -> Dict[str, Any]:
    """Test difference between Short (<60s) and Long (>=60s) content."""
    ga = df[df["duration"] < threshold_seconds][metric]
    gb = df[df["duration"] >= threshold_seconds][metric]
    return run_statistical_test(
        ga, gb, f"Short (<{threshold_seconds}s)", f"Long (>={threshold_seconds}s)", metric_name=metric
    )


def compare_posting_times(
    df: pd.DataFrame,
    metric: str = "total_engagement",
) -> Dict[str, Any]:
    """Test difference between Morning (06:00-12:00) vs Evening (17:00-23:00) posting windows."""
    ga = df[(df["hour"] >= 6) & (df["hour"] < 12)][metric]
    gb = df[(df["hour"] >= 17) & (df["hour"] <= 23)][metric]
    return run_statistical_test(ga, gb, "Morning (06:00-12:00)", "Evening (17:00-23:00)", metric_name=metric)
