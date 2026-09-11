# IDS706-Assignment-2

## Project Overview
This project explores a synthetic dating-app behavior dataset using Python. The data-analysis workflow includes:

- loading and inspecting a dataset;
- summarizing numerical and categorical variables;
- filtering and grouping observations;
- creating exploratory visualizations;
- building and evaluating a machine-learning model
- comparing Pandas and Polars for similar data-processing tasks; and
- experimenting with Rust ownership in a Jupyter notebook.

## Data Source
The dataset used in this project is the Dating App Behavior Dataset from Kaggle:https://www.kaggle.com/datasets/keyushnisar/dating-app-behavior-dataset/data. It is uploaded as dating_app_behavior_dataset in the repository.

The dataset contains:
- 50,000 observations
- 19 variables

The variables include demographic information, profile characteristics, app-usage behavior, swiping behavior, communication activity, and matching outcomes.

Examples of variables used in the analysis include:

- `app_usage_time_min`
- `swipe_right_ratio`
- `likes_received`
- `message_sent_count`
- `emoji_usage_rate`
- `profile_pics_count`
- `bio_length`
- `mutual_matches`
- `swipe_right_label`
- `app_usage_time_label`
- `match_outcome`

## Repository Contents

The repository contains the main files required for the assignment:

- `Assignment2_PythonCode.py` — main Python data-analysis script
- `dating_app_behavior_dataset.csv` — dating-app dataset
- `rust_vs_python_intro.ipynb` — modified Rust Jupyter notebook with ownership experiments
- `README.md` — project documentation

## Setup Instructions

### 1. Clone the repository

git clone git@github.com:yiranch017/IDS706-Assignment-2.git
cd IDS706-Assignment-2

### 2. Create or activate a Python environment
A Python 3 environment can be used for this project.

### 3. Install the required packages
pip install pandas polars matplotlib scikit-learn

## Data Analysis
### 1. Import and Inspection
The dataset was first loaded using Pandas.

Several methods were used to understand the structure and quality of the data:
.head() to preview observations;
.shape to determine the dataset dimensions;
.columns to inspect variable names;
.info() to examine data types and non-null observations;
.describe() to summarize numerical variables;
.describe(include="object") to summarize categorical variables;
.isnull().sum() to check for missing values; and
.duplicated().sum() to check for duplicate rows.

The dataset contains 50,000 rows and 19 columns.
There were 0 missing values and 0 duplicated rows
Therefore, extensive data cleaning was not necessary before beginning the exploratory analysis.

The dataset contains both numerical and categorical variables. Categorical variables were further inspected to identify the categories represented in each variable.

One variable, interest_tags, contains a very large number of unique combinations, so it was not used directly in the main analysis.

### 2. Filtering and Grouping
Filtering was used to examine specific behavioral subsets of the dataset.
This allowed the analysis to focus on users with relatively high right-swipe rates.

Grouping and aggregation were then used to compare matching outcomes across behavioral categories.
For example, users were grouped by swipe_right_label and their mutual matches were summarized using mean, median, and number of users. A similar analysis was then conducted using app_usage_time_label.

### 3. Data Visualization
Two histograms were created to examine the distributions of `swipe_right_ratio` and `mutual_matches`.

A histogram was appropriate for `swipe_right_ratio` because it is a continuous numerical variable. The plot shows how users are distributed across different swipe-right rates and helps identify the overall shape of the variable.

`mutual_matches` was also visualized because it later serves as the target variable in the machine-learning model. Examining its distribution first provides context for understanding the prediction results.

---

## Machine Learning: Random Forest Regression

A Random Forest Regressor was used to predict `mutual_matches` using:
- `app_usage_time_min`
- `swipe_right_ratio`
- `likes_received`
- `message_sent_count`
- `emoji_usage_rate`
- `profile_pics_count`
- `bio_length`

Random Forest was chosen because it can capture nonlinear relationships and interactions between predictors without requiring a linear relationship between each feature and the outcome.

The data was divided into 80% training data and 20% testing data.

### Model Results

The model produced approximately:
- Mean Absolute Error (MAE): 7.22
- R²: 0.10
The MAE indicates that predictions differed from the actual number of mutual matches by about 7.2 matches on average. The R² shows that the model explained only about 10% of the variation in the outcome, indicating relatively weak predictive performance.

An actual-versus-predicted scatter plot was used to visualize model performance. A scatter plot is suitable because both values are numerical, while the diagonal reference line represents perfect predictions. Most predictions clustered near the middle of the range instead of closely following this line, which is consistent with the low R².

## Feature Importance

The Random Forest identified `likes_received` as the most important feature, followed by `bio_length` and `app_usage_time_min`. `profile_pics_count` had the lowest importance. A horizontal bar chart was used for visualization because it allows the importance scores of multiple named features to be compared clearly while keeping the variable names readable.

---

## Pandas vs. Polars

Several filtering and grouping operations were repeated using Polars to compare it with Pandas.

Both libraries produced the same analytical results, but I noticed some differences in usage and presentation. Polars displays each column's data type directly in its table output, making variable types easy to identify. Pandas, however, integrates more naturally with Jupyter's HTML-style display and was visually easier to read. For the filtering and grouping operations tested in this project, Polars also had a shorter execution time than Pandas.

---

## Rust Ownership Experiment

The modified Rust Jupyter notebook explores:

- immutable and mutable variables;
- `let` versus `let mut`;
- ownership transfer;
- compiler errors caused by moved values; and
- `.clone()`.

---

## Main Findings

- The dataset contains 50,000 observations with no missing values or duplicate rows.
- The Random Forest model had weak predictive performance, with an R² of approximately 0.10.
- `likes_received` had the highest feature importance within the fitted model.
- Predictions tended to concentrate near the middle of the outcome range rather than accurately predicting extreme values.
- Polars completed the tested filtering and grouping operations faster than Pandas.
- Polars made column data types more visible, while Pandas provided a more polished Jupyter display.

---

## Limitations

The dataset is synthetic and intended for educational use. Therefore, patterns found in the analysis should not be generalized to real dating-app users, and feature importance should not be interpreted causally.

