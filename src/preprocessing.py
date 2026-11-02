"""
Data preprocessing and cleaning module for posts and audience comments.
Handles missing values, range validations, date-time extractions, and text normalization.
"""

import re
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from config import (
    CONTENT_TYPES,
    HOOK_TYPES,
    PLATFORMS,
    TOPICS,
)


def clean_posts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate social media posts dataset.
    Removes duplicates, enforces valid numeric bounds, imputes missing values,
    and extracts temporal components.
    """
    df = df.copy()

    # 1. Deduplication
    if "post_id" in df.columns:
        df = df.drop_duplicates(subset=["post_id"])
    else:
        df = df.drop_duplicates()
        df["post_id"] = [f"POST_{i:05d}" for i in range(1, len(df) + 1)]

    # 2. Categorical Imputation and Formatting
    categorical_defaults = {
        "topic": "Mental Wellness",
        "content_type": "Reel",
        "hook_type": "Visual",
        "platform": "Instagram",
    }

    for col, default_val in categorical_defaults.items():
        if col not in df.columns:
            df[col] = default_val
        else:
            df[col] = df[col].fillna(default_val).astype(str).str.strip()
            # If empty string, replace with default
            df[col] = df[col].replace("", default_val)

    # 3. Numeric Conversions and Imputations
    numeric_columns = [
        "likes",
        "shares",
        "saves",
        "comments_count",
        "retention_rate",
        "followers_gained",
        "duration",
        "caption_length",
        "hashtag_count",
    ]

    for col in numeric_columns:
        if col not in df.columns:
            # Provide sensible fallback defaults if completely missing from imported CSV
            if col == "retention_rate":
                df[col] = 50.0
            elif col == "duration":
                df[col] = 30
            elif col == "caption_length":
                df[col] = 120
            elif col == "hashtag_count":
                df[col] = 5
            else:
                df[col] = 0
        else:
            # Coerce string numbers (e.g., '1,200', '1.5k')
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

            # Impute missing with median (or 0 if all missing)
            median_val = df[col].median()
            median_val = median_val if pd.notna(median_val) else 0.0
            df[col] = df[col].fillna(median_val)

    # 4. Numeric Range Validation & Bounds Enforcement
    df["likes"] = np.maximum(0, df["likes"].astype(float))
    df["shares"] = np.maximum(0, df["shares"].astype(float))
    df["saves"] = np.maximum(0, df["saves"].astype(float))
    df["comments_count"] = np.maximum(0, df["comments_count"].astype(float))
    df["followers_gained"] = np.maximum(0, df["followers_gained"].astype(float))
    df["duration"] = np.maximum(0, df["duration"].astype(float))
    df["caption_length"] = np.maximum(0, df["caption_length"].astype(float))
    df["hashtag_count"] = np.maximum(0, df["hashtag_count"].astype(float))

    # Retention rate bounded to [0.0, 100.0]
    df["retention_rate"] = np.clip(df["retention_rate"].astype(float), 0.0, 100.0)

    # 5. Temporal Features Extraction
    if "posting_time" not in df.columns:
        # Default to synthetic recent timestamps
        base_time = pd.Timestamp.now()
        df["posting_time"] = [base_time - pd.Timedelta(hours=i) for i in range(len(df))]

    df["posting_time"] = pd.to_datetime(df["posting_time"], errors="coerce")
    
    # Fill unparseable dates with current time
    df["posting_time"] = df["posting_time"].fillna(pd.Timestamp.now())

    df["hour"] = df["posting_time"].dt.hour
    df["day_of_week"] = df["posting_time"].dt.day_name()
    df["day_of_week_num"] = df["posting_time"].dt.dayofweek
    df["is_weekend"] = df["day_of_week_num"].apply(lambda x: 1 if x >= 5 else 0)
    df["posting_date"] = df["posting_time"].dt.date

    # Ensure data_source is present
    if "data_source" not in df.columns:
        df["data_source"] = "Imported Dataset"

    return df.reset_index(drop=True)


def clean_comment_text(text: str) -> str:
    """
    Normalize comment text:
    - Lowercase
    - Strip URLs
    - Remove superfluous punctuation while keeping words intact
    - Normalize excessive whitespace
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Strip URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    
    # 3. Strip HTML tags if any
    text = re.sub(r"<.*?>", "", text)
    
    # 4. Remove emojis / non-ascii or normalize them
    text = re.sub(r"[^\w\s\']", " ", text)
    
    # 5. Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_comments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean audience comments dataset.
    Removes duplicates, cleans text, and drops empty comments.
    """
    df = df.copy()

    # Ensure required columns
    if "comment" not in df.columns:
        # Check if text column exists
        for alt in ["text", "body", "comment_text"]:
            if alt in df.columns:
                df["comment"] = df[alt]
                break
        else:
            df["comment"] = "Great content"

    if "post_id" not in df.columns:
        df["post_id"] = "POST_00001"

    if "comment_id" not in df.columns:
        df["comment_id"] = [f"COMM_{i:06d}" for i in range(1, len(df) + 1)]

    # Deduplicate
    df = df.drop_duplicates(subset=["comment_id"])
    
    # Text normalization
    df["cleaned_comment"] = df["comment"].astype(str).apply(clean_comment_text)

    # Filter out empty or whitespace-only comments
    df = df[df["cleaned_comment"].str.len() > 1].copy()

    return df.reset_index(drop=True)
