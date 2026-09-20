"""Reusable workflow for the dating-app behavior analysis."""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


FEATURES = [
    "app_usage_time_min",
    "swipe_right_ratio",
    "likes_received",
    "message_sent_count",
    "emoji_usage_rate",
    "profile_pics_count",
    "bio_length",
]
TARGET = "mutual_matches"
REQUIRED_COLUMNS = FEATURES + [TARGET, "swipe_right_label", "app_usage_time_label"]


def load_data(path):
    """Load a non-empty CSV file and validate the columns used by the workflow."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    data = pd.read_csv(path)
    if data.empty:
        raise ValueError("Dataset is empty.")

    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    return data


def preprocess_data(data):
    """Remove duplicate rows and rows missing required modeling values."""
    cleaned = data.drop_duplicates().dropna(subset=REQUIRED_COLUMNS).copy()
    if cleaned.empty:
        raise ValueError("No usable rows remain after preprocessing.")
    return cleaned


def summarize_matches(data, group_column):
    """Summarize matching outcomes for a categorical behavior variable."""
    if group_column not in data.columns:
        raise KeyError(f"Unknown grouping column: {group_column}")

    return (
        data.groupby(group_column, observed=True)[TARGET]
        .agg(
            average_matches="mean",
            median_matches="median",
            number_of_users="count",
        )
        .sort_index()
    )


def train_and_evaluate(data, test_size=0.2, random_state=42, n_estimators=30):
    """Train the Random Forest and return the model, predictions, and metrics."""
    if len(data) < 5:
        raise ValueError("At least five rows are required to train and evaluate the model.")

    X = data[FEATURES]
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    model = RandomForestRegressor(
        n_estimators=n_estimators, random_state=random_state, n_jobs=-1
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = {
        "mae": mean_absolute_error(y_test, predictions),
        "r2": r2_score(y_test, predictions),
    }
    return model, y_test, predictions, metrics


def run_pipeline(path, n_estimators=30):
    """Run data loading, preprocessing, summaries, training, and evaluation."""
    data = preprocess_data(load_data(path))
    model, y_test, predictions, metrics = train_and_evaluate(
        data, n_estimators=n_estimators
    )
    return {
        "row_count": len(data),
        "swipe_summary": summarize_matches(data, "swipe_right_label"),
        "usage_summary": summarize_matches(data, "app_usage_time_label"),
        "model": model,
        "y_test": y_test,
        "predictions": predictions,
        "metrics": metrics,
    }


if __name__ == "__main__":
    results = run_pipeline("dating_app_behavior_dataset.csv", n_estimators=100)
    print("Rows analyzed:", results["row_count"])
    print("Mean Absolute Error:", results["metrics"]["mae"])
    print("R-squared:", results["metrics"]["r2"])
