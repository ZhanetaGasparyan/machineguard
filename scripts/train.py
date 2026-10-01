import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "ai4i2020.csv"
MODELS_DIR = PROJECT_ROOT / "models"

TARGET_COLUMN = "Machine failure"

CATEGORICAL_FEATURES = ["Type"]

NUMERICAL_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERICAL_FEATURES


def load_data():
    """Load the raw dataset and select non-leaking features."""

    df = pd.read_csv(DATA_PATH)

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    return X, y


def create_preprocessor():
    """Create preprocessing for categorical and numerical features."""

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, NUMERICAL_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def create_models():
    """Create the baseline and nonlinear model pipelines."""

    logistic_regression = Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )

    random_forest = Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return {
        "logistic_regression": logistic_regression,
        "random_forest": random_forest,
    }


def calculate_metrics(model, X, y):
    """Calculate classification metrics using a 0.5 threshold."""

    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "pr_auc": float(average_precision_score(y, probabilities)),
    }


def main():
    """Train, compare, evaluate, and save the best model."""

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_data()

    # Keep 20% completely untouched for final testing.
    X_train_validation, X_test, y_train_validation, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # Split the remaining 80% into 60% training and 20% validation.
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train_validation,
        y_train_validation,
        test_size=0.25,
        random_state=42,
        stratify=y_train_validation,
    )

    print("Dataset split:")
    print(f"  Training rows:   {len(X_train):,}")
    print(f"  Validation rows: {len(X_validation):,}")
    print(f"  Test rows:       {len(X_test):,}")

    models = create_models()
    validation_results = {}

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")

        model.fit(X_train, y_train)

        metrics = calculate_metrics(
            model,
            X_validation,
            y_validation,
        )

        validation_results[model_name] = metrics

        print(pd.Series(metrics).round(4))

    # Select using validation PR-AUC because failures are uncommon.
    best_model_name = max(
        validation_results,
        key=lambda name: validation_results[name]["pr_auc"],
    )

    best_model = models[best_model_name]

    # Retrain the selected model using training and validation data.
    best_model.fit(X_train_validation, y_train_validation)

    # Evaluate once on the untouched test set.
    test_metrics = calculate_metrics(best_model, X_test, y_test)

    model_path = MODELS_DIR / "model.joblib"
    metadata_path = MODELS_DIR / "metadata.json"

    joblib.dump(best_model, model_path)

    metadata = {
        "model_name": best_model_name,
        "model_version": "1.0.0",
        "decision_threshold": 0.5,
        "features": FEATURE_COLUMNS,
        "validation_results": validation_results,
        "test_metrics": test_metrics,
        "dataset": {
            "name": "AI4I 2020 Predictive Maintenance Dataset",
            "synthetic": True,
            "total_rows": len(X),
            "training_rows": len(X_train_validation),
            "test_rows": len(X_test),
        },
    }

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print(f"\nSelected model: {best_model_name}")
    print("\nFinal test metrics:")
    print(pd.Series(test_metrics).round(4))
    print(f"\nSaved model to: {model_path}")
    print(f"Saved metadata to: {metadata_path}")


if __name__ == "__main__":
    main()