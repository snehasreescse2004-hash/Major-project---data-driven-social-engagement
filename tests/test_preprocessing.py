"""
Unit tests for data preprocessing and cleaning functions.
"""

import unittest
import pandas as pd
import numpy as np
from src.data_loader import normalize_column_names
from src.preprocessing import clean_posts, clean_comments, clean_comment_text


class TestPreprocessing(unittest.TestCase):
    def test_normalize_column_names(self):
        df = pd.DataFrame({
            "Like_Count": [10, 20],
            "Media_Type": ["Reel", "Image"],
            "Bookmark_Count": [5, 2],
            "Caption_Len": [100, 150],
        })
        normalized = normalize_column_names(df)
        self.assertIn("likes", normalized.columns)
        self.assertIn("content_type", normalized.columns)
        self.assertIn("saves", normalized.columns)
        self.assertIn("caption_length", normalized.columns)

    def test_clean_posts_bounds_and_imputation(self):
        raw_df = pd.DataFrame({
            "post_id": ["P1", "P1", "P2"],  # duplicate P1
            "likes": ["1,000", np.nan, -50],
            "retention_rate": [120.0, -10.0, 50.0],
            "duration": [30, np.nan, 45],
            "posting_time": ["2026-01-15 18:30:00", "2026-01-16 09:15:00", "invalid_date"],
        })
        cleaned = clean_posts(raw_df)

        # Deduplication check
        self.assertEqual(len(cleaned), 2)
        # Bounds check
        self.assertTrue((cleaned["retention_rate"] <= 100.0).all())
        self.assertTrue((cleaned["retention_rate"] >= 0.0).all())
        self.assertTrue((cleaned["likes"] >= 0).all())
        # Datetime check
        self.assertIn("hour", cleaned.columns)
        self.assertIn("is_weekend", cleaned.columns)
        self.assertEqual(cleaned.loc[cleaned["post_id"] == "P1", "hour"].iloc[0], 18)

    def test_clean_comment_text(self):
        raw_text = "Check this out https://example.com/video !!! This is SO true   "
        cleaned = clean_comment_text(raw_text)
        self.assertNotIn("https", cleaned)
        self.assertNotIn("example.com", cleaned)
        self.assertEqual(cleaned, "check this out this is so true")

    def test_clean_comments_filtering(self):
        comments_df = pd.DataFrame({
            "comment_id": ["C1", "C2", "C3", "C4", "C4"],
            "post_id": ["P1", "P1", "P2", "P2", "P2"],
            "comment": ["Literally me", "   ", "http://spam.com", "Same here", "Duplicate ID"],
        })
        cleaned = clean_comments(comments_df)
        # "   " and "http://spam.com" dropped due to empty text, duplicate "C4" dropped
        self.assertEqual(len(cleaned), 2)
        self.assertIn("cleaned_comment", cleaned.columns)


if __name__ == "__main__":
    unittest.main()
