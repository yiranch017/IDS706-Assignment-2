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


def detect_outliers(data, column):
    """Return a Boolean mask identifying IQR outliers in a numeric column."""
    if column not in data.columns:
        raise KeyError(f"Unknown column: {column}")
    if not pd.api.types.is_numeric_dtype(data[column]):
        raise TypeError(f"Outlier detection requires a numeric column: {column}")

    first_quartile = data[column].quantile(0.25)
    third_quartile = data[column].quantile(0.75)
    iqr = third_quartile - first_quartile
    lower_bound = first_quartile - 1.5 * iqr
    upper_bound = third_quartile + 1.5 * iqr
    return (data[column] < lower_bound) | (data[column] > upper_bound)


def summarize_outliers(data, columns=None):
    """Summarize IQR-based outliers across selected numeric columns."""
    if columns is None:
        columns = FEATURES + [TARGET]

    records = []

    for column in columns:
        outlier_mask = detect_outliers(data, column)

        records.append(
            {
                "column": column,
                "outlier_count": int(outlier_mask.sum()),
                "outlier_percentage": round(outlier_mask.mean() * 100, 2),
            }
        )

    return (
        pd.DataFrame(records)
        .sort_values(
            ["outlier_count", "column"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )


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


def summarize_swipe_deciles(data, bins=10):
    """Group users by swipe-right ratio and summarize matching outcomes."""
    if bins < 2:
        raise ValueError("At least two bins are required.")

    subset = data[["swipe_right_ratio", TARGET]].dropna().copy()

    if subset.empty:
        raise ValueError("No usable rows available for swipe analysis.")

    subset["swipe_ratio_bin"] = pd.qcut(
        subset["swipe_right_ratio"],
        q=bins,
        duplicates="drop",
    )

    return (
        subset.groupby("swipe_ratio_bin", observed=True)
        .agg(
            average_swipe_ratio=("swipe_right_ratio", "mean"),
            average_matches=(TARGET, "mean"),
            median_matches=(TARGET, "median"),
            number_of_users=(TARGET, "count"),
        )
        .reset_index()
    )


def train_and_evaluate(data, test_size=0.2, random_state=42, n_estimators=30):
    """Train the Random Forest and return the model, predictions, and metrics."""
    if len(data) < 5:
        raise ValueError(
            "At least five rows are required to train and evaluate the model."
        )

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
        "outlier_summary": summarize_outliers(data),
        "swipe_summary": summarize_matches(data, "swipe_right_label"),
        "usage_summary": summarize_matches(data, "app_usage_time_label"),
        "swipe_decile_summary": summarize_swipe_deciles(data),
        "model": model,
        "y_test": y_test,
        "predictions": predictions,
        "metrics": metrics,
    }


if __name__ == "__main__":
    results = run_pipeline(
        "dating_app_behavior_dataset.csv",
        n_estimators=100,
    )

    print("Rows analyzed:", results["row_count"])

    print("\nOutlier Summary:")
    print(results["outlier_summary"])

    print("\nSwipe Behavior Summary:")
    print(results["swipe_summary"])

    print("\nSwipe Ratio Decile Summary:")
    print(results["swipe_decile_summary"])

    print("\nModel Performance:")
    print("Mean Absolute Error:", results["metrics"]["mae"])
    print("R-squared:", results["metrics"]["r2"])
