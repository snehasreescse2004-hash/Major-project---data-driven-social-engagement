"""
Natural Language Processing module for audience comments.
Implements:
1. Relatability Classification ('RELATABLE' vs 'NEUTRAL') via TF-IDF Vectorizer + Logistic Regression,
   supported by transparent weak-supervision heuristics for unlabeled corpora.
2. Sentiment Polarity Analysis using TextBlob ('Positive', 'Neutral', 'Negative').
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from textblob import TextBlob

from config import (
    RANDOM_STATE,
    RELATABLE_KEYWORDS,
    RELATABLE_MODEL_PATH,
    TFIDF_VECTORIZER_PATH,
)
from src.preprocessing import clean_comment_text


def apply_heuristic_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Project-defined weak-supervision labeling for audience comments.
    Labels a comment as 'RELATABLE' if it exhibits linguistic resonance tokens
    (e.g., 'literally me', 'so true', 'same here'), otherwise 'NEUTRAL'.
    Transparently documented as heuristic weak supervision.
    """
    df = df.copy()

    # If genuine ground-truth label column exists, respect it
    if "relatability" in df.columns:
        df["label"] = df["relatability"].str.upper()
        return df

    def label_text(comment_str: str) -> str:
        lower = comment_str.lower()
        for kw in RELATABLE_KEYWORDS:
            if kw in lower:
                return "RELATABLE"
        return "NEUTRAL"

    source_col = "cleaned_comment" if "cleaned_comment" in df.columns else "comment"
    df["label"] = df[source_col].astype(str).apply(label_text)
    return df


def compute_sentiment_polarity(text: str) -> Tuple[str, float]:
    """
    Compute sentiment polarity using TextBlob.
    - Polarity > 0.05  => Positive
    - Polarity < -0.05 => Negative
    - Otherwise        => Neutral
    """
    cleaned = clean_comment_text(text)
    if not cleaned:
        return "Neutral", 0.0
    
    blob = TextBlob(cleaned)
    polarity = float(blob.sentiment.polarity)

    if polarity > 0.05:
        category = "Positive"
    elif polarity < -0.05:
        category = "Negative"
    else:
        category = "Neutral"

    return category, round(polarity, 3)


def train_relatability_model(
    df: pd.DataFrame,
    vectorizer_path: Optional[Path] = None,
    model_path: Optional[Path] = None,
) -> Tuple[TfidfVectorizer, LogisticRegression, Dict[str, Any]]:
    """
    Train TF-IDF Vectorizer and Logistic Regression classifier for relatability detection.
    Persists artifacts and outputs comprehensive evaluation metrics.
    """
    v_path = Path(vectorizer_path) if vectorizer_path else TFIDF_VECTORIZER_PATH
    m_path = Path(model_path) if model_path else RELATABLE_MODEL_PATH
    v_path.parent.mkdir(parents=True, exist_ok=True)

    df_labeled = apply_heuristic_labels(df)
    text_col = "cleaned_comment" if "cleaned_comment" in df_labeled.columns else "comment"

    # Train / Test split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df_labeled[text_col].astype(str),
        df_labeled["label"],
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df_labeled["label"],
    )

    vectorizer = TfidfVectorizer(
        max_features=2500,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
    )

    X_train_vec = vectorizer.fit_transform(X_train_raw)
    X_test_vec = vectorizer.transform(X_test_raw)

    classifier = LogisticRegression(
        max_iter=1000,
        C=1.5,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )
    classifier.fit(X_train_vec, y_train)

    y_pred = classifier.predict(X_test_vec)
    classes = ["NEUTRAL", "RELATABLE"]

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, pos_label="RELATABLE", zero_division=0))
    rec = float(recall_score(y_test, y_pred, pos_label="RELATABLE", zero_division=0))
    f1 = float(f1_score(y_test, y_pred, pos_label="RELATABLE", zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=classes)
    report_dict = classification_report(y_test, y_pred, labels=classes, output_dict=True, zero_division=0)

    # Top indicative vocabulary features for 'RELATABLE'
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = classifier.coef_[0]
    top_relatable_indices = np.argsort(coefs)[-15:][::-1]
    top_neutral_indices = np.argsort(coefs)[:15]

    metrics = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": cm.tolist(),
        "classes": classes,
        "classification_report": report_dict,
        "top_relatable_terms": feature_names[top_relatable_indices].tolist(),
        "top_neutral_terms": feature_names[top_neutral_indices].tolist(),
        "test_sample_count": len(y_test),
    }

    # Persist artifacts
    joblib.dump(vectorizer, v_path)
    joblib.dump({"model": classifier, "metrics": metrics}, m_path)

    return vectorizer, classifier, metrics


def load_or_train_nlp_model(
    df: pd.DataFrame,
    force_retrain: bool = False,
    vectorizer_path: Optional[Path] = None,
    model_path: Optional[Path] = None,
) -> Tuple[TfidfVectorizer, LogisticRegression, Dict[str, Any]]:
    """
    Load existing NLP vectorizer and classifier from disk, or train fresh if absent.
    """
    v_path = Path(vectorizer_path) if vectorizer_path else TFIDF_VECTORIZER_PATH
    m_path = Path(model_path) if model_path else RELATABLE_MODEL_PATH

    if not force_retrain and v_path.exists() and m_path.exists():
        try:
            vectorizer = joblib.load(v_path)
            bundle = joblib.load(m_path)
            if isinstance(bundle, dict) and "model" in bundle and "metrics" in bundle:
                return vectorizer, bundle["model"], bundle["metrics"]
        except Exception as e:
            print(f"Notice: Existing NLP model could not be loaded ({e}). Retraining...")

    return train_relatability_model(df, v_path, m_path)


def analyze_comment_sample(
    text: str,
    vectorizer: TfidfVectorizer,
    classifier: LogisticRegression,
) -> Dict[str, Any]:
    """
    Full dual-dimension inference for a single comment:
    1. Relatability Classification (RELATABLE vs NEUTRAL) with prediction probability.
    2. Sentiment Polarity (Positive, Neutral, Negative) with numeric polarity score.
    """
    cleaned = clean_comment_text(text)
    
    # NLP Relatability Prediction
    vec = vectorizer.transform([cleaned])
    pred_label = str(classifier.predict(vec)[0])
    
    probs = {}
    if hasattr(classifier, "predict_proba"):
        p = classifier.predict_proba(vec)[0]
        for cls_name, prob in zip(classifier.classes_, p):
            probs[str(cls_name)] = round(float(prob), 4)

    # Sentiment Polarity
    sentiment_cat, polarity_score = compute_sentiment_polarity(text)

    return {
        "input_text": text,
        "cleaned_text": cleaned,
        "relatability": pred_label,
        "relatability_confidence": probs.get(pred_label, 1.0),
        "relatability_probabilities": probs,
        "sentiment": sentiment_cat,
        "sentiment_polarity": polarity_score,
    }


def enrich_comments_dataset(
    comments_df: pd.DataFrame,
    posts_df: Optional[pd.DataFrame] = None,
    vectorizer: Optional[TfidfVectorizer] = None,
    classifier: Optional[LogisticRegression] = None,
) -> pd.DataFrame:
    """
    Process full comments dataset:
    - Runs relatability classification
    - Runs sentiment polarity scoring
    - Merges associated post metadata (e.g. topic, content_type) for cross-tabulation.
    """
    df = comments_df.copy()

    # Predict relatability if models provided, else fallback to heuristic
    if vectorizer is not None and classifier is not None:
        source_text = df["cleaned_comment"] if "cleaned_comment" in df.columns else df["comment"]
        vecs = vectorizer.transform(source_text.astype(str))
        df["relatability"] = classifier.predict(vecs)
    else:
        df = apply_heuristic_labels(df)
        df["relatability"] = df["label"]

    # Sentiment Polarity calculation
    sentiments = []
    polarities = []
    for txt in df["comment"].astype(str):
        cat, pol = compute_sentiment_polarity(txt)
        sentiments.append(cat)
        polarities.append(pol)

    df["sentiment"] = sentiments
    df["sentiment_polarity"] = polarities

    # Merge post topic and content_type if posts_df available
    if posts_df is not None and "post_id" in posts_df.columns:
        meta = posts_df[["post_id", "topic", "content_type", "platform"]].drop_duplicates(subset=["post_id"])
        df = df.merge(meta, on="post_id", how="left")
        df["topic"] = df["topic"].fillna("General")
        df["content_type"] = df["content_type"].fillna("Standard")

    return df
