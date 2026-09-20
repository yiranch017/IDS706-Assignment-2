"""Unit and integration tests for the dating-app analysis workflow."""

import numpy as np
import pandas as pd
import pytest

from analysis import (
    FEATURES,
    load_data,
    preprocess_data,
    run_pipeline,
    summarize_matches,
    train_and_evaluate,
)


@pytest.fixture
def sample_data():
    """Create deterministic data that is small enough for fast tests."""
    rng = np.random.default_rng(42)
    rows = 40
    data = pd.DataFrame(
        {
            "app_usage_time_min": rng.integers(10, 180, rows),
            "swipe_right_ratio": rng.uniform(0.1, 0.9, rows),
            "likes_received": rng.integers(1, 100, rows),
            "message_sent_count": rng.integers(0, 50, rows),
            "emoji_usage_rate": rng.uniform(0, 1, rows),
            "profile_pics_count": rng.integers(1, 7, rows),
            "bio_length": rng.integers(20, 300, rows),
            "mutual_matches": rng.integers(0, 30, rows),
            "swipe_right_label": ["Choosy", "Balanced"] * 20,
            "app_usage_time_label": ["Low", "High"] * 20,
        }
    )
    return data


def test_load_data_reads_valid_csv(tmp_path, sample_data):
    csv_path = tmp_path / "sample.csv"
    sample_data.to_csv(csv_path, index=False)

    loaded = load_data(csv_path)

    assert loaded.shape == sample_data.shape
    assert set(FEATURES).issubset(loaded.columns)


def test_load_data_rejects_missing_required_column(tmp_path, sample_data):
    csv_path = tmp_path / "missing.csv"
    sample_data.drop(columns=["mutual_matches"]).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="Missing required columns"):
        load_data(csv_path)


def test_preprocess_removes_duplicates_and_missing_values(sample_data):
    dirty = pd.concat([sample_data, sample_data.iloc[[0]]], ignore_index=True)
    dirty.loc[1, "likes_received"] = np.nan

    cleaned = preprocess_data(dirty)

    assert len(cleaned) == len(sample_data) - 1
    assert cleaned.duplicated().sum() == 0
    assert cleaned[FEATURES].isna().sum().sum() == 0


def test_summary_returns_correct_counts_and_means(sample_data):
    summary = summarize_matches(sample_data, "swipe_right_label")

    expected = sample_data.groupby("swipe_right_label")["mutual_matches"].mean()
    assert summary["number_of_users"].sum() == len(sample_data)
    pd.testing.assert_series_equal(
        summary["average_matches"], expected.sort_index(), check_names=False
    )


def test_model_training_returns_predictions_and_metrics(sample_data):
    model, y_test, predictions, metrics = train_and_evaluate(
        sample_data, n_estimators=5
    )

    assert model.n_features_in_ == len(FEATURES)
    assert len(predictions) == len(y_test)
    assert np.isfinite(metrics["mae"])
    assert np.isfinite(metrics["r2"])
    assert metrics["mae"] >= 0


def test_end_to_end_pipeline(tmp_path, sample_data):
    """System test: validate the complete CSV-to-model workflow."""
    csv_path = tmp_path / "sample.csv"
    sample_data.to_csv(csv_path, index=False)

    results = run_pipeline(csv_path, n_estimators=5)

    assert results["row_count"] == len(sample_data)
    assert len(results["predictions"]) == len(results["y_test"])
    assert results["swipe_summary"]["number_of_users"].sum() == len(sample_data)
    assert results["usage_summary"]["number_of_users"].sum() == len(sample_data)
