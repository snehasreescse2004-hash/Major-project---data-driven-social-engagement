"""
Data loading and synthesis module for The Data-Driven Social Engagement Initiative.
Provides resilient ingestion with column-synonym normalization and realistic fallback
demo dataset generation.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from config import (
    COLUMN_SYNONYMS,
    COMMENTS_DATA_PATH,
    CONTENT_TYPES,
    HOOK_TYPES,
    NEUTRAL_KEYWORDS,
    PLATFORMS,
    POST_STANDARD_COLUMNS,
    POSTS_DATA_PATH,
    RANDOM_STATE,
    RELATABLE_KEYWORDS,
    TOPICS,
)


def normalize_column_names(df: pd.DataFrame, synonym_dict: Dict[str, List[str]] = COLUMN_SYNONYMS) -> pd.DataFrame:
    """
    Standardize DataFrame column names by stripping whitespace, converting to lowercase,
    and mapping recognized synonyms to project canonical names.
    """
    df = df.copy()
    cleaned_cols = {col: str(col).strip().lower().replace(" ", "_") for col in df.columns}
    df.rename(columns=cleaned_cols, inplace=True)

    # Invert mapping to find targets
    alias_to_standard = {}
    for standard_col, aliases in synonym_dict.items():
        for alias in aliases:
            alias_to_standard[alias.lower()] = standard_col

    rename_map = {}
    for col in df.columns:
        if col in alias_to_standard:
            rename_map[col] = alias_to_standard[col]
        else:
            # Check without punctuation
            simplified = col.replace("-", "_").replace(".", "_")
            if simplified in alias_to_standard:
                rename_map[col] = alias_to_standard[simplified]

    df.rename(columns=rename_map, inplace=True)
    return df


def generate_demo_data(
    num_posts: int = 1200,
    num_comments: int = 3500,
    seed: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate realistic, correlated synthetic social media datasets for posts and audience comments.
    Incorporates domain-specific distributions (e.g., Mental Wellness and Financial Stress
    yield higher saves, Dating and College Life yield higher shares, Story hooks boost retention).
    """
    rng = np.random.default_rng(seed)

    # Realistic topic bias matrices for saves vs shares
    topic_weights = {
        "Social Anxiety": {"save_mult": 1.4, "share_mult": 1.3, "relatable_bias": 0.75},
        "Dating": {"save_mult": 0.9, "share_mult": 1.5, "relatable_bias": 0.65},
        "Career Stress": {"save_mult": 1.5, "share_mult": 1.1, "relatable_bias": 0.70},
        "College Life": {"save_mult": 0.8, "share_mult": 1.4, "relatable_bias": 0.80},
        "Family Pressure": {"save_mult": 1.2, "share_mult": 1.3, "relatable_bias": 0.72},
        "Friendship": {"save_mult": 0.9, "share_mult": 1.2, "relatable_bias": 0.60},
        "Self Confidence": {"save_mult": 1.3, "share_mult": 1.1, "relatable_bias": 0.68},
        "Financial Stress": {"save_mult": 1.7, "share_mult": 1.2, "relatable_bias": 0.70},
        "Work Life": {"save_mult": 1.3, "share_mult": 1.4, "relatable_bias": 0.75},
        "Mental Wellness": {"save_mult": 1.8, "share_mult": 1.2, "relatable_bias": 0.78},
    }

    content_duration_map = {
        "Image": (0, 0),
        "Reel": (15, 60),
        "Carousel": (30, 90),
        "Short Video": (20, 60),
        "Long Video": (180, 600),
    }

    start_date = datetime.now() - timedelta(days=180)
    posts: List[Dict] = []

    for i in range(1, num_posts + 1):
        post_id = f"POST_{i:05d}"
        topic = rng.choice(TOPICS)
        content_type = rng.choice(CONTENT_TYPES, p=[0.35, 0.20, 0.15, 0.20, 0.10])
        hook_type = rng.choice(HOOK_TYPES, p=[0.35, 0.25, 0.25, 0.15])
        platform = rng.choice(PLATFORMS, p=[0.45, 0.30, 0.25])

        # Durations based on format
        min_dur, max_dur = content_duration_map[content_type]
        duration = int(rng.integers(min_dur, max_dur + 1)) if max_dur > 0 else 0

        # Posting timestamp (realistic hour spread with peak in evening 18-21 & afternoon 12-14)
        days_offset = rng.integers(0, 180)
        hour_weights = np.array([
            0.01, 0.01, 0.01, 0.01, 0.01, 0.02,  # 0-5
            0.03, 0.04, 0.05, 0.05, 0.04, 0.05,  # 6-11
            0.08, 0.07, 0.05, 0.04, 0.06, 0.08,  # 12-17
            0.10, 0.12, 0.10, 0.08, 0.05, 0.03   # 18-23
        ], dtype=float)
        hour_weights = hour_weights / hour_weights.sum()
        hour = rng.choice(range(24), p=hour_weights)
        minute = rng.integers(0, 60)
        post_time = start_date + timedelta(days=int(days_offset), hours=int(hour), minutes=int(minute))

        # Base virality factor (log-normal distribution)
        base_virality = rng.lognormal(mean=6.5, sigma=0.9)

        # Topic multiplier
        t_meta = topic_weights.get(topic, {"save_mult": 1.0, "share_mult": 1.0})

        # Hook influence
        hook_retention_boost = 10 if hook_type == "Story" else (5 if hook_type == "Question" else 0)
        hook_comment_boost = 1.35 if hook_type == "Question" else 1.0

        # Correlated metrics
        likes = max(10, int(base_virality * rng.uniform(0.8, 1.3)))
        comments_count = max(1, int((likes * rng.uniform(0.02, 0.08)) * hook_comment_boost))
        shares = max(0, int((likes * rng.uniform(0.04, 0.14)) * t_meta["share_mult"]))
        saves = max(0, int((likes * rng.uniform(0.05, 0.18)) * t_meta["save_mult"]))

        # Retention rate (0-100%) - influenced by duration and hook
        base_retention = rng.normal(loc=55, scale=15) + hook_retention_boost
        if duration > 180:
            base_retention -= 12
        elif 0 < duration <= 45:
            base_retention += 8
        retention_rate = float(np.clip(base_retention, 12.0, 96.5))

        # Follower gains correlated with shares and saves
        followers_gained = max(0, int((shares * 0.18 + saves * 0.22 + likes * 0.01) * rng.uniform(0.6, 1.4)))

        # Metadata
        caption_length = int(rng.normal(loc=140, scale=45))
        caption_length = max(20, min(500, caption_length))
        hashtag_count = int(rng.choice(range(2, 16)))

        posts.append({
            "post_id": post_id,
            "topic": topic,
            "content_type": content_type,
            "duration": duration,
            "posting_time": post_time.strftime("%Y-%m-%d %H:%M:%S"),
            "likes": likes,
            "shares": shares,
            "saves": saves,
            "comments_count": comments_count,
            "retention_rate": round(retention_rate, 2),
            "followers_gained": followers_gained,
            "hook_type": hook_type,
            "caption_length": caption_length,
            "hashtag_count": hashtag_count,
            "platform": platform,
            "data_source": "Demo Synthetic Dataset",
        })

    posts_df = pd.DataFrame(posts)

    # -------------------------------------------------------------
    # Audience Comments Generation
    # -------------------------------------------------------------
    comments: List[Dict] = []
    
    # Real-world authentic sentence templates
    relatable_templates = [
        "This is literally me every single day.",
        "I feel this on a personal level, so true!",
        "Same here, honestly thought I was the only one going through this.",
        "Story of my life right here.",
        "You didn't have to call me out like that!",
        "Exactly what I experienced all week.",
        "I relate to this way too much.",
        "This describes my entire situation right now.",
        "Me too, especially when it happens unexpectedly.",
        "Felt this in my soul. Thank you for putting it into words.",
        "Hits home so hard. Relatable content!",
        "Literally going through this exact thing right now.",
        "Why is this so accurate? It hurts how true this is.",
        "My life in a nutshell.",
        "Target audience reached, I feel seen.",
    ]

    neutral_templates = [
        "Great video, thanks for sharing!",
        "Very informative and concise explanation.",
        "Nice tips, appreciate the insight.",
        "Good perspective on this topic.",
        "Cool animation and editing style.",
        "Thanks for the breakdown, well presented.",
        "Interesting thoughts, bookmarking for later.",
        "Awesome breakdown, good to know.",
        "Informative content, keep it up!",
        "Clear and helpful advice.",
        "Very well articulated summary.",
        "Good points mentioned here.",
        "Appreciate the clarity on this.",
    ]

    topic_specific_comments = {
        "Social Anxiety": [
            "My social battery completely empties after 20 minutes, so true.",
            "The overthinking after every simple conversation is real.",
            "I literally rehearse phone calls before dialing. Same!",
        ],
        "Career Stress": [
            "Sunday evening dread is real. This is literally me.",
            "Workload is through the roof, I feel this so much.",
            "Burnout is tough to explain until you feel it exactly like this.",
        ],
        "Financial Stress": [
            "Checking my bank account gives me chills, literally me.",
            "Saving money in this economy is so hard, felt this.",
            "Budgeting advice is nice but the cost of living is wild.",
        ],
        "Dating": [
            "Modern dating is exhausting, story of my life.",
            "Dating apps are so draining, I relate to every word.",
            "Communication is everything, glad you highlighted this.",
        ],
        "Mental Wellness": [
            "Taking a mental health day changed everything for me.",
            "Prioritizing peace of mind over pleasing everyone. So true.",
            "Deep breathing and grounding really help when anxious.",
        ],
    }

    all_post_ids = posts_df["post_id"].tolist()
    topic_by_post = dict(zip(posts_df["post_id"], posts_df["topic"]))

    for j in range(1, num_comments + 1):
        comment_id = f"COMM_{j:06d}"
        chosen_post_id = rng.choice(all_post_ids)
        post_topic = topic_by_post[chosen_post_id]
        t_meta = topic_weights.get(post_topic, {"relatable_bias": 0.65})

        is_relatable = rng.random() < t_meta["relatable_bias"]

        if is_relatable:
            if post_topic in topic_specific_comments and rng.random() < 0.45:
                text = rng.choice(topic_specific_comments[post_topic])
            else:
                text = rng.choice(relatable_templates)
        else:
            text = rng.choice(neutral_templates)

        comments.append({
            "comment_id": comment_id,
            "post_id": chosen_post_id,
            "comment": text,
        })

    comments_df = pd.DataFrame(comments)
    return posts_df, comments_df


def load_data(
    posts_path: Optional[Path] = None,
    comments_path: Optional[Path] = None,
    auto_generate: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Load posts and comments data from CSV files.
    If files are absent, empty, or invalid, automatically falls back to generating
    a complete demo dataset and persists it.

    Returns:
        posts_df (pd.DataFrame): Normalized posts DataFrame
        comments_df (pd.DataFrame): Normalized comments DataFrame
        data_source (str): 'Imported Dataset' or 'Demo Synthetic Dataset'
    """
    p_path = Path(posts_path) if posts_path else POSTS_DATA_PATH
    c_path = Path(comments_path) if comments_path else COMMENTS_DATA_PATH

    posts_loaded = False
    comments_loaded = False
    posts_df = pd.DataFrame()
    comments_df = pd.DataFrame()
    source_label = "Demo Synthetic Dataset"

    # Attempt to load posts
    if p_path.exists() and p_path.is_file():
        try:
            raw_posts = pd.read_csv(p_path)
            if not raw_posts.empty and len(raw_posts) > 5:
                posts_df = normalize_column_names(raw_posts)
                posts_loaded = True
                source_label = "Imported Dataset"
        except Exception as e:
            print(f"Warning: Failed to load posts from {p_path}: {e}")

    # Attempt to load comments
    if c_path.exists() and c_path.is_file():
        try:
            raw_comments = pd.read_csv(c_path)
            if not raw_comments.empty and len(raw_comments) > 5:
                comments_df = normalize_column_names(raw_comments)
                comments_loaded = True
        except Exception as e:
            print(f"Warning: Failed to load comments from {c_path}: {e}")

    # If either is missing or invalid, generate fallback demo data
    if (not posts_loaded or not comments_loaded) and auto_generate:
        print("Generating comprehensive fallback demo dataset (1200 posts, 3500 comments)...")
        demo_posts, demo_comments = generate_demo_data()
        
        # Ensure directories
        POSTS_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        demo_posts.to_csv(POSTS_DATA_PATH, index=False)
        demo_comments.to_csv(COMMENTS_DATA_PATH, index=False)

        posts_df = demo_posts if not posts_loaded else posts_df
        comments_df = demo_comments if not comments_loaded else comments_df
        source_label = "Demo Synthetic Dataset" if not posts_loaded else "Imported Posts + Demo Comments"

    # Ensure data_source column is populated
    if "data_source" not in posts_df.columns:
        posts_df["data_source"] = source_label

    return posts_df, comments_df, source_label
