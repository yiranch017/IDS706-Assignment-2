# IDS706 Dating App Behavior Analysis

[![Tests and Code Quality](https://github.com/yiranch017/IDS706-Assignment-2/actions/workflows/tests.yml/badge.svg)](https://github.com/yiranch017/IDS706-Assignment-2/actions/workflows/tests.yml)

## Project Overview

This project explores how dating-app behavior relates to matching outcomes using a synthetic dataset of 50,000 users. The analysis focuses on whether swiping behavior and app usage are associated with `mutual_matches`, and whether behavioral and profile variables can predict matching outcomes.

The project includes exploratory analysis, Random Forest regression, Pandas/Polars comparison, automated testing, continuous integration, code-quality checks, and Docker containerization.

Because the dataset is synthetic, the results are treated as a programming and modeling exercise rather than evidence about real dating-app users.

## Main Findings

- Matching outcomes vary only slightly across different swipe-behavior groups.
- Dividing users across the full `swipe_right_ratio` distribution also shows no clear monotonic increase in matches as users swipe right more frequently.
- The Random Forest model achieved an MAE of approximately **7.22** and an R² of approximately **0.10**, indicating limited predictive power.
- `likes_received` had the highest Random Forest feature importance, although feature importance should not be interpreted causally.
- Polars was faster than Pandas for the filtering and grouping operations tested, while Pandas provided a more polished interactive table display.

## Data Quality

The dataset contains no missing values or duplicate observations.

IQR-based outlier detection identified **311 observations (0.62%)** in `emoji_usage_rate`. These observations ranged from **0.74 to 0.94**, which remains within the valid 0-to-1 range for a rate.

Because these observations are statistically unusual but still plausible values, they were retained rather than automatically removed.

## Analysis

The analysis includes:

- distribution of swipe-right behavior;
- comparison of matching outcomes across predefined swipe groups;
- swipe-ratio quantile analysis across the full behavior distribution;
- comparison of matching outcomes across app-usage groups;
- Random Forest regression;
- actual-versus-predicted visualization;
- feature-importance analysis; and
- equivalent filtering/grouping tasks in Pandas and Polars.

The primary reusable workflow is implemented in `analysis.py`, while `exploratory_data_analysis.py` contains the interpretation and visualization workflow.

## Repository Structure

```text
.
├── .github/workflows/tests.yml
├── tests/test_analysis.py
├── analysis.py
├── exploratory_data_analysis.py
├── dating_app_behavior_dataset.csv
├── Dockerfile
├── .dockerignore
├── .flake8
├── Makefile
├── requirements.txt
└── README.md
```

## Setup and Usage

Clone the repository and install the dependencies:

```bash
git clone https://github.com/yiranch017/IDS706-Assignment-2.git
cd IDS706-Assignment-2

python -m venv .venv
source .venv/bin/activate

python -m pip install -r requirements.txt
```

Run the reproducible analysis pipeline:

```bash
python analysis.py
```

Or use the Makefile:

```bash
make install
make run
```

## Testing

The project contains **9 automated tests** covering typical behavior, edge cases, and the full workflow.

The tests include:

- valid CSV loading;
- rejection of missing required columns;
- removal of duplicate and missing observations;
- IQR outlier detection;
- outlier-summary validation;
- swipe-ratio group validation;
- grouped summary calculations;
- model training and evaluation; and
- an end-to-end CSV-to-model pipeline test.

Run all tests with:

```bash
make test
```

or:

```bash
python -m pytest -v
```

## Continuous Integration

GitHub Actions automatically verifies the project on pushes, pull requests, manual runs, and scheduled daily runs.

The CI workflow includes:

- a **Python 3.11 / 3.12 matrix**;
- automated pytest execution;
- Black formatting checks;
- Flake8 linting;
- Python syntax compilation checks; and
- a scheduled daily workflow.

Local formatting and code-quality checks can also be run with:

```bash
make format
make quality
```

### CI Evidence

Workflow history and successful matrix runs are available in the repository's **Actions** tab.

<!-- Add the screenshot showing Code Quality, Python 3.11, and Python 3.12 all passing here. -->

## Refactoring and Code Quality

The original exploratory analysis was written as one long procedural script. It was refactored into smaller functions with distinct responsibilities for dataset inspection, data-quality assessment, swipe-behavior analysis, app-usage analysis, model evaluation, visualization, and Pandas/Polars comparison.

The refactor also reduced duplicated logic by reusing functions from `analysis.py`, including model training, outlier detection, and grouped summaries. Important analytical comments and interpretations were retained so that the code documents not only **what** is being done, but also **why** particular analytical decisions were made.

The project was verified after refactoring using Black, Flake8, and the automated test suite.

### Refactoring Evidence

<!-- Add the GitHub commit-diff screenshot for the refactoring commit here. -->

## Docker and Containerization

The project can also run inside a reproducible Docker environment.

Build the image:

```bash
docker build -t dating-app-analysis .
```

or:

```bash
make docker-build
```

Run the analysis inside a container:

```bash
docker run --rm dating-app-analysis
```

or:

```bash
make docker-run
```

The container uses `python:3.12-slim`, installs the project dependencies, copies the analysis code and dataset, and runs `analysis.py`.

Containerization ensures that the analysis runs with the same Python version and dependencies regardless of the host machine.

### Docker Evidence

**Successful image build**

<!-- Add the Docker build screenshot here. -->

**Successful container execution**

<!-- Add the Docker run screenshot here. -->

## Limitations

The dataset is synthetic, so the observed patterns should not be generalized to real dating-app users. The Random Forest also has low explanatory power, and neither feature importance nor the exploratory associations establish causal relationships.
