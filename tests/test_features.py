"""
Unit tests for feature engineering and Viral Score calculations.
"""

import unittest
import pandas as pd
import numpy as np
from src.feature_engineering import (
    engineer_features,
    calculate_single_viral_score,
    get_normalization_bounds,
)


class TestFeatures(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "post_id": [f"P_{i}" for i in range(1, 21)],
            "topic": ["Mental Wellness"] * 20,
            "content_type": ["Reel"] * 20,
            "hook_type": ["Visual"] * 20,
            "platform": ["Instagram"] * 20,
            "likes": [100 * i for i in range(1, 21)],
            "shares": [10 * i for i in range(1, 21)],
            "saves": [15 * i for i in range(1, 21)],
            "comments_count": [5 * i for i in range(1, 21)],
            "retention_rate": [30.0 + 2.0 * i for i in range(1, 21)],
            "followers_gained": [2 * i for i in range(1, 21)],
            "duration": [30] * 20,
            "caption_length": [120] * 20,
            "hashtag_count": [5] * 20,
            "hour": [18] * 20,
        })

    def test_engineer_features_columns(self):
        engineered = engineer_features(self.df)
        expected_cols = [
            "total_engagement",
            "engagement_rate",
            "save_share_ratio",
            "viral_score",
            "viral_category",
        ]
        for col in expected_cols:
            self.assertIn(col, engineered.columns)

        # Check total_engagement arithmetic
        first_row = engineered.iloc[0]
        expected_total = (
            first_row["likes"]
            + first_row["comments_count"]
            + first_row["shares"]
            + first_row["saves"]
        )
        self.assertEqual(first_row["total_engagement"], expected_total)

        # Viral score bounded
        self.assertTrue((engineered["viral_score"] >= 0.0).all())
        self.assertTrue((engineered["viral_score"] <= 100.0).all())
        self.assertTrue(set(engineered["viral_category"]).issubset({"Low", "Medium", "High"}))

    def test_calculate_single_viral_score(self):
        bounds = get_normalization_bounds(self.df)
        
        # High performance simulation
        high_score, high_cat = calculate_single_viral_score(
            likes=2500,
            shares=300,
            saves=400,
            comments=150,
            retention=95.0,
            followers_gained=80,
            normalization_reference=bounds,
        )
        self.assertEqual(high_cat, "High")
        self.assertGreaterEqual(high_score, 70.0)

        # Low performance simulation
        low_score, low_cat = calculate_single_viral_score(
            likes=10,
            shares=0,
            saves=0,
            comments=0,
            retention=15.0,
            followers_gained=0,
            normalization_reference=bounds,
        )
        self.assertEqual(low_cat, "Low")
        self.assertLess(low_score, 40.0)


if __name__ == "__main__":
    unittest.main()
