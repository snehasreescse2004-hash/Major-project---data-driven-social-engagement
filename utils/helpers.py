"""
Helper utilities for formatting, reporting, logging, and data conversions.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd


def format_metric(val: Any, decimals: int = 1, prefix: str = "", suffix: str = "") -> str:
    """Format numerical values cleanly (e.g. 1.2K, 3.4M, or standard float)."""
    if val is None or pd.isna(val):
        return "N/A"
    try:
        num = float(val)
        abs_num = abs(num)
        if abs_num >= 1_000_000:
            formatted = f"{num / 1_000_000:.{decimals}f}M"
        elif abs_num >= 1_000:
            formatted = f"{num / 1_000:.{decimals}f}K"
        elif abs_num < 1 and abs_num > 0:
            formatted = f"{num:.{decimals + 1}f}"
        else:
            formatted = f"{num:,.{decimals}f}".rstrip("0").rstrip(".") if decimals > 0 else f"{int(round(num)):,}"
        return f"{prefix}{formatted}{suffix}"
    except (ValueError, TypeError):
        return str(val)


def format_percent(val: Any, decimals: int = 1) -> str:
    """Format percentage value."""
    if val is None or pd.isna(val):
        return "0.0%"
    try:
        return f"{float(val):.{decimals}f}%"
    except (ValueError, TypeError):
        return "0.0%"


def format_duration(seconds: Any) -> str:
    """Format duration in seconds into human-readable minutes/seconds."""
    if seconds is None or pd.isna(seconds):
        return "0s"
    try:
        sec = int(round(float(seconds)))
        if sec < 60:
            return f"{sec}s"
        mins = sec // 60
        rem_sec = sec % 60
        return f"{mins}m {rem_sec}s" if rem_sec > 0 else f"{mins}m"
    except (ValueError, TypeError):
        return f"{seconds}s"


def save_insights_report(
    insights: List[str],
    summary_metrics: Dict[str, Any],
    top_topics: List[Dict[str, Any]],
    recommendation: Dict[str, Any],
    file_path: Path,
) -> bool:
    """
    Export analytical findings and recommendations into a formatted text report.
    """
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        lines = [
            "=" * 70,
            "THE DATA-DRIVEN SOCIAL ENGAGEMENT INITIATIVE - EXECUTIVE REPORT",
            "=" * 70,
            f"Generated: {timestamp}",
            f"Data Source: {summary_metrics.get('data_source', 'Social Media Analytics')}",
            "",
            "1. CORE METRICS SUMMARY",
            "-" * 40,
            f"Total Posts Analyzed       : {summary_metrics.get('total_posts', 0):,}",
            f"Total Audience Engagement  : {summary_metrics.get('total_engagement', 0):,}",
            f"Average Engagement Rate    : {summary_metrics.get('avg_engagement_rate', 0):.2f}%",
            f"Average Viral Score        : {summary_metrics.get('avg_viral_score', 0):.1f} / 100",
            f"High-Viral Posts Share     : {summary_metrics.get('high_viral_pct', 0):.1f}%",
            f"Total Comments Processed   : {summary_metrics.get('total_comments', 0):,}",
            f"Relatable Audience Rate    : {summary_metrics.get('relatable_pct', 0):.1f}%",
            "",
            "2. KEY ANALYTICAL INSIGHTS",
            "-" * 40,
        ]
        
        for i, insight in enumerate(insights, 1):
            lines.append(f"[{i}] {insight}")
            
        lines.extend([
            "",
            "3. TOP PERFORMING TOPICS",
            "-" * 40,
        ])
        
        for topic in top_topics[:5]:
            name = topic.get("topic", "Unknown")
            eng = topic.get("total_engagement", 0)
            score = topic.get("avg_viral_score", 0.0)
            lines.append(f" - {name:<22}: Avg Viral Score = {score:.1f} | Total Eng = {eng:,.0f}")
            
        lines.extend([
            "",
            "4. PRESCRIPTIVE STRATEGY RECOMMENDATIONS",
            "-" * 40,
            f"Optimal Topic          : {recommendation.get('topic', 'N/A')}",
            f"Optimal Platform       : {recommendation.get('platform', 'N/A')}",
            f"Optimal Content Type   : {recommendation.get('content_type', 'N/A')}",
            f"Optimal Hook Type      : {recommendation.get('hook_type', 'N/A')}",
            f"Recommended Duration   : {recommendation.get('duration_range', 'N/A')}",
            f"Recommended Posting    : {recommendation.get('posting_window', 'N/A')}",
            f"Expected Virality Tier : {recommendation.get('expected_performance', 'N/A')}",
            "",
            "Strategic Rationale:",
        ])
        
        for reason in recommendation.get("reasons", []):
            lines.append(f" * {reason}")
            
        lines.extend([
            "",
            "=" * 70,
            "End of Report - The Data-Driven Social Engagement Initiative",
            "=" * 70,
        ])
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return True
    except Exception as exc:
        print(f"Error saving report: {exc}")
        return False
