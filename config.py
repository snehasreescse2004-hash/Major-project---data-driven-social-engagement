"""
Configuration and constants for The Data-Driven Social Engagement Initiative.
Defines project paths, standard schema names, alias dictionaries for external CSVs,
Viral Score weights, category thresholds, and visual design themes.
"""

from pathlib import Path

# Base Paths (Supports running from either project subfolder or repo root)
CONFIG_DIR = Path(__file__).resolve().parent
BASE_DIR = CONFIG_DIR

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
SRC_DIR = BASE_DIR / "src"
UTILS_DIR = BASE_DIR / "utils"

# Ensure directories exist
for directory in [DATA_DIR, MODELS_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

POSTS_DATA_PATH = DATA_DIR / "posts.csv"
COMMENTS_DATA_PATH = DATA_DIR / "comments.csv"

VIRAL_MODEL_PATH = MODELS_DIR / "viral_model.pkl"
RELATABLE_MODEL_PATH = MODELS_DIR / "relatable_model.pkl"
TFIDF_VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
REPORT_OUTPUT_PATH = REPORTS_DIR / "generated_insights.txt"

# Random Seed for Reproducibility
RANDOM_STATE = 42

# Standard Project Schema Fields
POST_STANDARD_COLUMNS = [
    "post_id",
    "topic",
    "content_type",
    "duration",
    "posting_time",
    "likes",
    "shares",
    "saves",
    "comments_count",
    "retention_rate",
    "followers_gained",
    "hook_type",
    "caption_length",
    "hashtag_count",
    "platform",
    "data_source",
]

COMMENT_STANDARD_COLUMNS = [
    "comment_id",
    "post_id",
    "comment",
]

# External Dataset Column Alias Normalization Dictionary
COLUMN_SYNONYMS = {
    "likes": ["likes", "like_count", "likes_count", "favorite_count", "num_likes", "like"],
    "comments_count": ["comments_count", "comments", "comment_count", "num_comments", "comments_num", "reply_count"],
    "shares": ["shares", "share_count", "shares_count", "num_shares", "retweets", "reposts", "share"],
    "saves": ["saves", "save_count", "saves_count", "num_saves", "bookmarks", "bookmark_count", "save"],
    "retention_rate": ["retention_rate", "retention", "avg_retention", "completion_rate", "watch_ratio", "watch_time_pct"],
    "followers_gained": ["followers_gained", "followers", "follower_count", "new_followers", "net_followers", "followers_added"],
    "duration": ["duration", "video_duration", "length", "video_length", "duration_sec", "seconds"],
    "content_type": ["content_type", "media_type", "type", "post_type", "format"],
    "topic": ["topic", "category", "niche", "theme", "subject", "genre"],
    "hook_type": ["hook_type", "hook", "intro_type", "opening_type"],
    "caption_length": ["caption_length", "caption_len", "caption_chars", "text_length", "caption_word_count"],
    "hashtag_count": ["hashtag_count", "hashtags", "num_hashtags", "tag_count", "tags_count"],
    "platform": ["platform", "channel", "network", "social_network", "source_platform"],
    "posting_time": ["posting_time", "posted_at", "created_at", "timestamp", "publish_time", "post_date", "datetime"],
    "post_id": ["post_id", "id", "postid", "content_id", "item_id"],
    "comment_id": ["comment_id", "commentid", "cid", "reply_id"],
    "comment": ["comment", "comment_text", "text", "body", "content", "message"]
}

# Categorical Domains
TOPICS = [
    "Social Anxiety",
    "Dating",
    "Career Stress",
    "College Life",
    "Family Pressure",
    "Friendship",
    "Self Confidence",
    "Financial Stress",
    "Work Life",
    "Mental Wellness",
]

CONTENT_TYPES = [
    "Reel",
    "Carousel",
    "Image",
    "Short Video",
    "Long Video",
]

HOOK_TYPES = [
    "Visual",
    "Text",
    "Question",
    "Story",
]

PLATFORMS = [
    "Instagram",
    "YouTube",
    "TikTok",
]

# Viral Score Analytical Formulation Weights
# viral_score = 0.15*norm_likes + 0.25*norm_shares + 0.25*norm_saves + 0.20*norm_retention + 0.10*norm_comments + 0.05*norm_followers_gained
VIRAL_WEIGHTS = {
    "likes": 0.15,
    "shares": 0.25,
    "saves": 0.25,
    "retention": 0.20,
    "comments": 0.10,
    "followers_gained": 0.05,
}

# Virality Categories
VIRAL_THRESHOLDS = {
    "low_max": 40.0,
    "medium_max": 70.0,
}

# Relatability Linguistic Keywords (Heuristic Weak-Supervision)
RELATABLE_KEYWORDS = [
    "me too", "same", "literally me", "this is me", "exactly",
    "so true", "i feel this", "i relate", "my life", "story of my life",
    "only one", "going through this", "this describes me", "relatable",
    "called me out", "felt that", "felt this", "same here", "real",
    "me every day", "target audience", "hits home", "it hurts how true"
]

NEUTRAL_KEYWORDS = [
    "nice", "good video", "great", "cool", "thanks", "informative",
    "awesome", "love this", "helpful", "well done", "interesting",
    "wow", "ok", "okay", "good to know", "first", "hello"
]

# A/B Testing Default Alpha
SIGNIFICANCE_ALPHA = 0.05

# Color Themes (Plotly & Streamlit Styling)
THEME_COLORS = {
    "primary": "#4F46E5",       # Indigo
    "secondary": "#06B6D4",     # Cyan
    "accent": "#EC4899",        # Pink
    "success": "#10B981",       # Emerald
    "warning": "#F59E0B",       # Amber
    "danger": "#EF4444",        # Rose
    "neutral": "#64748B",       # Slate
    "dark_bg": "#0F172A",       # Slate 900
    "card_bg": "#1E293B",       # Slate 800
    "card_border": "#334155",   # Slate 700
    "light_bg": "#F8FAFC",
}

VIRAL_COLORS = {
    "High": "#10B981",    # Green
    "Medium": "#F59E0B",  # Yellow/Orange
    "Low": "#EF4444",     # Red
}

SENTIMENT_COLORS = {
    "Positive": "#10B981",
    "Neutral": "#64748B",
    "Negative": "#EF4444",
}
