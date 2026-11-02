"""
Machine Learning module for Virality Classification.
Constructs a Scikit-Learn Pipeline combining ColumnTransformer (StandardScaler + OneHotEncoder)
with a RandomForestClassifier to predict viral performance tier ('Low', 'Medium', 'High').
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import RANDOM_STATE, VIRAL_MODEL_PATH

NUMERICAL_FEATURES = [
    "likes",
    "shares",
    "saves",
    "comments_count",
    "retention_rate",
    "followers_gained",
    "duration",
    "caption_length",
    "hashtag_count",
    "hour",
]

CATEGORICAL_FEATURES = [
    "content_type",
    "topic",
    "hook_type",
    "platform",
]

TARGET_COL = "viral_category"
CLASS_ORDER = ["Low", "Medium", "High"]


def build_pipeline() -> Pipeline:
    """Build preprocessing and RandomForest classification pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=12,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    return pipeline


def train_viral_model(
    df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Train the virality classification model, evaluate on stratified holdout test split,
    and persist the trained model pipeline to disk.
    """
    save_file = Path(save_path) if save_path else VIRAL_MODEL_PATH
    save_file.parent.mkdir(parents=True, exist_ok=True)

    # Validate dataset features
    required_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + [TARGET_COL]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for model training: {missing}")

    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_COL]

    # Stratified Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # Predictions & Evaluation
    y_pred = pipeline.predict(X_test)
    labels = [c for c in CLASS_ORDER if c in np.unique(y_test)]

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    rec = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    report_dict = classification_report(y_test, y_pred, labels=labels, output_dict=True, zero_division=0)

    # Extract Feature Importances
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    
    cat_encoder = preprocessor.named_transformers_["cat"]
    encoded_cat_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    all_feature_names = NUMERICAL_FEATURES + encoded_cat_names
    importances = classifier.feature_importances_

    feature_imp_df = (
        pd.DataFrame({"feature": all_feature_names, "importance": importances})
        .sort_values(by="importance", ascending=False)
        .reset_index(drop=True)
    )

    metrics = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": cm.tolist(),
        "classes": labels,
        "classification_report": report_dict,
        "feature_importances": feature_imp_df.head(15).to_dict(orient="records"),
        "test_sample_count": len(y_test),
        "train_sample_count": len(y_train),
    }

    # Save artifact
    bundle = {
        "pipeline": pipeline,
        "metrics": metrics,
        "feature_names": NUMERICAL_FEATURES + CATEGORICAL_FEATURES,
    }
    joblib.dump(bundle, save_file)
    return pipeline, metrics


def load_or_train_viral_model(
    df: pd.DataFrame,
    force_retrain: bool = False,
    model_path: Optional[Path] = None,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Load an existing trained model bundle or automatically train and save a new one.
    """
    target_path = Path(model_path) if model_path else VIRAL_MODEL_PATH

    if not force_retrain and target_path.exists():
        try:
            bundle = joblib.load(target_path)
            if isinstance(bundle, dict) and "pipeline" in bundle and "metrics" in bundle:
                return bundle["pipeline"], bundle["metrics"]
        except Exception as e:
            print(f"Notice: Existing viral model could not be loaded ({e}). Retraining...")

    return train_viral_model(df, save_path=target_path)


def predict_virality(
    pipeline: Pipeline,
    input_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Perform inference on a single post's metadata and simulated metrics.
    Returns predicted tier and probability distribution.
    """
    row_df = pd.DataFrame([input_data])

    # Ensure all expected columns are present
    for num_col in NUMERICAL_FEATURES:
        if num_col not in row_df.columns:
            row_df[num_col] = 0.0
        else:
            row_df[num_col] = float(row_df[num_col].iloc[0])

    for cat_col in CATEGORICAL_FEATURES:
        if cat_col not in row_df.columns:
            row_df[cat_col] = "Unknown"
        else:
            row_df[cat_col] = str(row_df[cat_col].iloc[0])

    X_in = row_df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    pred_category = pipeline.predict(X_in)[0]
    
    probabilities = {}
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(X_in)[0]
        classes = pipeline.classes_
        for cls_name, prob in zip(classes, probs):
            probabilities[str(cls_name)] = round(float(prob), 4)

    return {
        "predicted_category": str(pred_category),
        "confidence": probabilities.get(str(pred_category), 1.0),
        "probabilities": probabilities,
    }
