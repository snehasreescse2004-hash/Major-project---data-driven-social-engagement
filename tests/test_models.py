"""
Unit tests for Machine Learning, NLP, and Statistical testing modules.
"""

import unittest
import pandas as pd
import numpy as np
from src.viral_model import train_viral_model, predict_virality
from src.nlp_model import train_relatability_model, analyze_comment_sample, compute_sentiment_polarity
from src.ab_testing import run_statistical_test
from src.data_loader import generate_demo_data
from src.preprocessing import clean_posts, clean_comments
from src.feature_engineering import engineer_features


class TestModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Generate lightweight test corpus
        posts_raw, comments_raw = generate_demo_data(num_posts=120, num_comments=250, seed=123)
        cls.posts_df = engineer_features(clean_posts(posts_raw))
        cls.comments_df = clean_comments(comments_raw)

    def test_viral_model_training_and_inference(self):
        pipeline, metrics = train_viral_model(self.posts_df)
        self.assertIn("accuracy", metrics)
        self.assertIn("confusion_matrix", metrics)
        self.assertGreater(metrics["accuracy"], 0.3)  # Meaningfully better than random 3-class guess

        # Test single-post inference
        sample_post = {
            "likes": 1200,
            "shares": 150,
            "saves": 200,
            "comments_count": 50,
            "retention_rate": 72.0,
            "followers_gained": 30,
            "duration": 40,
            "caption_length": 140,
            "hashtag_count": 5,
            "hour": 19,
            "content_type": "Reel",
            "topic": "Mental Wellness",
            "hook_type": "Visual",
            "platform": "Instagram",
        }
        pred = predict_virality(pipeline, sample_post)
        self.assertIn("predicted_category", pred)
        self.assertIn(pred["predicted_category"], ["Low", "Medium", "High"])
        self.assertIn("probabilities", pred)

    def test_nlp_relatability_and_sentiment(self):
        vec, model, metrics = train_relatability_model(self.comments_df)
        self.assertIn("accuracy", metrics)
        self.assertIn("confusion_matrix", metrics)

        # Test relatability prediction on sample text
        res_rel = analyze_comment_sample("This is literally me every single day, so true!", vec, model)
        self.assertEqual(res_rel["relatability"], "RELATABLE")

        res_neut = analyze_comment_sample("Great video, thanks for the informative breakdown.", vec, model)
        self.assertEqual(res_neut["relatability"], "NEUTRAL")

        # Test Sentiment Polarity
        pos_cat, pos_pol = compute_sentiment_polarity("I absolutely love this, very helpful and wonderful!")
        self.assertEqual(pos_cat, "Positive")
        self.assertGreater(pos_pol, 0.05)

        neg_cat, neg_pol = compute_sentiment_polarity("This was terrible, frustrating and completely awful.")
        self.assertEqual(neg_cat, "Negative")
        self.assertLess(neg_pol, -0.05)

    def test_statistical_ab_test(self):
        group_a = pd.Series(np.random.normal(loc=120, scale=15, size=40))
        group_b = pd.Series(np.random.normal(loc=80, scale=15, size=40))
        res = run_statistical_test(group_a, group_b, "Strategy A", "Strategy B", "Engagement")

        self.assertTrue(res["is_significant"])
        self.assertEqual(res["significance_conclusion"], "Statistically significant")
        self.assertLess(res["welch_p_value"], 0.05)
        self.assertIn("disclaimer", res)


if __name__ == "__main__":
    unittest.main()
