# IDS706 Dating App Behavior Analysis

[![Tests](https://github.com/yiranch017/IDS706-Assignment-2/actions/workflows/tests.yml/badge.svg)](https://github.com/yiranch017/IDS706-Assignment-2/actions/workflows/tests.yml)

## Project Overview

This project explores how dating-app behavior relates to matching outcomes using a synthetic Kaggle dataset. The workflow loads and validates the data, removes unusable observations, summarizes matching outcomes across behavioral groups, and trains a Random Forest regressor to predict `mutual_matches`. The project also compares Pandas and Polars and now includes automated testing and continuous integration.

## Motivation and Goal

Dating apps often encourage continued engagement by suggesting that greater activity may improve matching outcomes. This analysis asks whether swiping behavior and app usage are meaningfully related to mutual matches and whether a set of profile and behavioral features can predict the number of mutual matches.

Because the dataset is synthetic, the results are treated as a programming and modeling exercise rather than evidence about real users.

## Data

The [Dating App Behavior Dataset](https://www.kaggle.com/datasets/keyushnisar/dating-app-behavior-dataset/data) contains 50,000 observations and 19 variables. The model uses:

- `app_usage_time_min`
- `swipe_right_ratio`
- `likes_received`
- `message_sent_count`
- `emoji_usage_rate`
- `profile_pics_count`
- `bio_length`

The prediction target is `mutual_matches`.

## Repository Structure

- `Assignment2_PythonCode` — original exploratory analysis, visualizations, and Pandas/Polars comparison
- `analysis.py` — reusable functions for loading, preprocessing, grouping, modeling, and the complete pipeline
- `tests/test_analysis.py` — unit, edge-case, and end-to-end tests
- `.github/workflows/tests.yml` — GitHub Actions CI workflow
- `dating_app_behavior_dataset.csv` — project dataset
- `requirements.txt` — Python dependencies
- `rust_vs_python_intro.ipynb` — Rust ownership exercises

## Setup and Usage

```bash
git clone git@github.com:yiranch017/IDS706-Assignment-2.git
cd IDS706-Assignment-2
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python analysis.py
```

The same commands are also available as `make install`, `make test`, and `make run`.

## Testing

Run the full test suite locally with:

```bash
python -m pytest -v
```

The suite contains five unit/edge-case tests and one system test:

- valid CSV loading;
- rejection of a dataset missing a required column;
- removal of duplicate and missing observations;
- correctness of grouped counts and averages;
- model training, prediction, and evaluation outputs; and
- an end-to-end test from CSV loading through model evaluation.

GitHub Actions automatically installs the dependencies and runs all tests on every push and pull request. The badge at the top of this README reports the latest workflow result.

Workflow history is available on the repository's **Actions** tab for CI evidence and screenshots.

## Main Results

- The original dataset contains no missing values or duplicate rows.
- Matching outcomes varied only slightly across swiping-behavior and app-usage groups.
- The Random Forest achieved an MAE of approximately 7.22 and an R² of approximately 0.10, so its predictive performance was limited.
- `likes_received` had the highest feature importance in the fitted model, but feature importance should not be interpreted causally.
- In the exploratory comparison, Polars executed the tested filtering and grouping operations faster, while Pandas provided a more polished Jupyter display.

## Limitations

The data is synthetic and the model has low explanatory power. Findings should not be generalized to real dating-app users, and the observed relationships do not establish causality.
