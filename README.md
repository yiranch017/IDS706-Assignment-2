# IDS706 Dating App Behavior Analysis

[![Tests and Code Quality](https://github.com/yiranch017/IDS706-Assignment-2/actions/workflows/tests.yml/badge.svg)](https://github.com/yiranch017/IDS706-Assignment-2/actions/workflows/tests.yml)

## Project Overview

This project explores how users' behavior on dating apps relates to their matching outcomes using a synthetic dataset of 50,000 users.

The main question I wanted to look at was: **does swiping right more actually lead to more mutual matches?**

I first explored different patterns in swiping and app usage, then trained a Random Forest model to see how well a group of behavioral and profile variables could predict `mutual_matches`. I also compared Pandas and Polars for similar data manipulation tasks.

Over the past few assignments, I gradually improved this project by adding reusable functions, tests, CI, code-quality checks, and Docker containerization.

Because the dataset is synthetic, I treat the results mainly as a data analysis and engineering exercise rather than evidence about real dating-app users.

## Main Findings

One of the main findings is that swiping right more does not seem to have a clear relationship with getting more matches in this dataset.

When I compared the predefined swipe-behavior groups, their average number of mutual matches was very similar. I also divided users into groups across the full `swipe_right_ratio` distribution, and the average number of matches did not consistently increase as swipe-right behavior increased.

The Random Forest model also had relatively limited predictive power:

- **MAE:** about 7.22 mutual matches
- **R²:** about 0.10

This means the variables included in the model only explain a small amount of the variation in matching outcomes.

`likes_received` had the highest feature importance in the Random Forest model. However, I treat this only as information about how the model makes predictions, not as evidence that receiving more likes directly causes more mutual matches.

## Data Quality

I checked the dataset for missing values, duplicate rows, and outliers before doing the main analysis.

There were no missing values or duplicate observations.

Using the IQR method, I found **311 outliers (0.62% of the dataset)** in `emoji_usage_rate`. These observations ranged from **0.74 to 0.94**.

I decided to keep them because `emoji_usage_rate` is a rate between 0 and 1, so these values are still valid. They are unusual compared with most users, but unusual behavior does not necessarily mean incorrect data.

## Analysis

The project includes:

- inspection of the dataset and variable types
- missing-value and duplicate checks
- IQR-based outlier detection
- distribution of `swipe_right_ratio`
- comparison of matching outcomes across swipe-behavior groups
- comparison across swipe-right ratio quantiles
- comparison across app-usage groups
- Random Forest regression
- actual vs. predicted values
- Random Forest feature importance
- Pandas vs. Polars comparison

The main reusable analysis functions are in `analysis.py`.

`exploratory_data_analysis.py` contains the exploratory analysis, visualizations, and my interpretations of the results.

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

Clone the repository:

```bash
git clone https://github.com/yiranch017/IDS706-Assignment-2.git
cd IDS706-Assignment-2
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the main analysis:

```bash
python analysis.py
```

The same steps can also be run with:

```bash
make install
make run
```

## Testing

The project currently includes **9 automated tests**.

They check:

- loading a valid dataset
- rejecting a dataset with missing required columns
- removing duplicate and missing observations
- detecting an extreme outlier
- creating the outlier summary correctly
- creating swipe-ratio groups correctly
- calculating grouped summaries
- training and evaluating the Random Forest model
- running the full pipeline from CSV loading to model evaluation

Run the tests with:

```bash
python -m pytest -v
```

or:

```bash
make test
```

## Continuous Integration

I expanded the original GitHub Actions workflow so that it now checks more than whether the tests simply pass.

The workflow runs on:

- pushes
- pull requests
- manual runs
- a daily scheduled run

It also uses a matrix to test the project with both:

- Python 3.11
- Python 3.12

Before running the tests, the workflow also checks:

- Black formatting
- Flake8 linting
- Python syntax

This helps make sure the project is not only working, but also stays consistently formatted and readable.

Locally, I can run the same code-quality checks with:

```bash
make quality
```

### CI Evidence

<!-- Add screenshot showing Code Quality, Python 3.11, and Python 3.12 passing -->

## Refactoring and Code Quality

The original version of `exploratory_data_analysis.py` was mostly one long script. It worked, but as the project became larger it was harder to read and there was repeated logic between the exploratory analysis and `analysis.py`.

I refactored the file into smaller functions for different parts of the workflow, including:

- dataset inspection
- data-quality checks
- swipe-behavior analysis
- app-usage analysis
- model evaluation
- visualization
- Pandas/Polars comparison

I also reused functions already defined in `analysis.py` instead of repeating the same model training, grouping, and outlier-detection logic.

I kept the comments that explain my analytical decisions and interpretations because I wanted the code to show not only what I did, but also why I made certain choices.

After refactoring, I ran Black, Flake8, and the full test suite to make sure the project still worked.

### Refactoring Evidence

<!-- Add GitHub commit-diff screenshot here -->

## Docker and Containerization

I also containerized the analysis so that it can run in a consistent environment without depending on the Python setup on my own computer.

The Docker image uses `python:3.12-slim`, installs the dependencies from `requirements.txt`, copies the necessary project files, and runs `analysis.py`.

Build the image with:

```bash
docker build -t dating-app-analysis .
```

or:

```bash
make docker-build
```

Run it with:

```bash
docker run --rm dating-app-analysis
```

or:

```bash
make docker-run
```

The container produced the same main analysis output as running the project locally, including the outlier summary, swipe-behavior analysis, and Random Forest evaluation.

### Docker Evidence

**Successful image build**

<!-- Add Docker build screenshot here -->

**Successful container run**

<!-- Add Docker run screenshot here -->

## Pandas vs. Polars

I ran similar filtering and grouping operations using both Pandas and Polars.

Polars was faster for the operations I tested and also displays the data type directly under each column in its default output.

At the same time, I found Pandas easier to work with interactively because its tables integrate more naturally with Jupyter Notebook and are easier to visually inspect.

For this dataset, I think the difference is less about one library being universally better and more about the tradeoff between performance and the workflow I am using.

## Limitations

The dataset is synthetic, so these findings should not be generalized to real dating-app users.

The Random Forest model also has relatively low explanatory power, which suggests that the variables included here are not enough to explain most of the variation in matching outcomes.

Finally, the relationships found in the exploratory analysis and the Random Forest feature importance should not be interpreted as causal effects.
