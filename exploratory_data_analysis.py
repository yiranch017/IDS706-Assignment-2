"""Exploratory analysis for the dating-app behavior dataset."""

from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl

from analysis import (
    FEATURES,
    detect_outliers,
    load_data,
    summarize_matches,
    summarize_outliers,
    summarize_swipe_deciles,
    train_and_evaluate,
)

DATA_PATH = Path("dating_app_behavior_dataset.csv")


def inspect_dataset(data):
    """Inspect the structure, data types, and variables in the dataset."""

    print("\nFirst five rows:")
    print(data.head())

    print("\nDataset shape:")
    print(data.shape)

    # The dataset has 50,000 rows and 19 columns.
    # The variables can broadly be grouped into:
    # 1. Demographic characteristics
    # 2. App behavior and profile characteristics
    # 3. Matching outcomes

    print("\nData types:")

    # I use a separate table instead of only calling df.info()
    # because it clearly displays each variable alongside its data type.
    dtype_table = data.dtypes.reset_index()
    dtype_table.columns = ["Column", "Data Type"]
    print(dtype_table)

    print("\nSummary statistics:")
    print(data.describe())

    # Inspect the categories contained in categorical variables.
    categorical_columns = data.select_dtypes(include="object").columns

    print("\nCategorical Variables:")

    for column in categorical_columns:
        print(f"\n{column}:")
        print(data[column].unique())

    # Most categorical variables contain a relatively small number of
    # categories. interest_tags contains many combinations of three
    # interests, so I do not include it in the main analysis in order
    # to keep the project focused.


def assess_data_quality(data):
    """Check missing values, duplicates, and potential outliers."""

    print("\nMissing values:")
    print(data.isnull().sum())

    print("\nDuplicated rows:")
    print(data.duplicated().sum())

    # Missing values, duplicate observations, and statistical outliers
    # are considered separately. An unusual observation is not
    # automatically an invalid observation.
    outlier_report = summarize_outliers(data)

    print("\nOutlier Summary:")
    print(outlier_report)

    # emoji_usage_rate is the only analyzed numerical variable for which
    # the IQR method identifies outliers. Before deciding whether to remove
    # them, I check whether those values fall outside the valid range.
    emoji_outlier_mask = detect_outliers(data, "emoji_usage_rate")
    emoji_outliers = data[emoji_outlier_mask]

    print("\nEmoji usage rate range:")
    print("Overall minimum:", data["emoji_usage_rate"].min())
    print("Overall maximum:", data["emoji_usage_rate"].max())

    print("\nEmoji usage outlier range:")
    print("Minimum outlier:", emoji_outliers["emoji_usage_rate"].min())
    print("Maximum outlier:", emoji_outliers["emoji_usage_rate"].max())

    outlier_count = int(emoji_outlier_mask.sum())
    outlier_percentage = emoji_outlier_mask.mean() * 100

    print(
        "\nOutlier decision:"
        f" {outlier_count} observations "
        f"({outlier_percentage:.2f}% of the dataset) are flagged as "
        "emoji_usage_rate outliers."
    )

    # Interpretation:
    # The IQR method identifies statistically unusual observations, but
    # unusual does not necessarily mean incorrect. The flagged
    # emoji_usage_rate values remain within the valid 0-to-1 range for
    # a rate. Therefore, they are treated as plausible high-usage
    # behavior and retained rather than automatically removed.


def plot_swipe_distribution(data):
    """Visualize the overall distribution of swipe-right behavior."""

    # swipe_right_ratio is a continuous numerical variable ranging from
    # 0 to 1, so a histogram is appropriate for examining its distribution.
    # Dividing the values into intervals helps show whether users cluster
    # around particular swipe-right rates and whether the distribution is
    # skewed or contains unusually common ranges.

    plt.figure(figsize=(8, 5))

    plt.hist(
        data["swipe_right_ratio"],
        bins=20,
        edgecolor="black",
    )

    plt.xlabel("Swipe Right Ratio")
    plt.ylabel("Number of Users")
    plt.title("Distribution of Swipe Right Ratio")
    plt.tight_layout()

    plt.show()


def analyze_swipe_behavior(data):
    """Analyze how swipe-right behavior relates to mutual matches."""

    # Key research question:
    # How does swiping behavior relate to matching outcomes?
    #
    # Many dating apps use subscription-based business models and may
    # encourage users to remain active on the platform. This makes it
    # useful to examine whether greater swipe-right activity is actually
    # associated with more mutual matches in this dataset.
    #
    # Because this dataset is synthetic, the findings should be treated
    # as a programming and modeling exercise rather than evidence about
    # real dating-app users.

    # First, use a simple threshold-based comparison. Users who swipe
    # right more than 70% of the time are classified here as high-swiping
    # users. This gives an intuitive comparison with the overall sample.
    high_swipe_users = data[data["swipe_right_ratio"] > 0.70]

    high_swipe_mean = high_swipe_users["mutual_matches"].mean()
    overall_mean = data["mutual_matches"].mean()

    print("\nHigh-Swipe Users:")
    print("Number of users:", len(high_swipe_users))
    print(
        "Difference from overall average matches:",
        high_swipe_mean - overall_mean,
    )

    # The high-swipe group represents about 15% of users in this dataset,
    # and its average number of matches is only slightly different from
    # the overall average. The threshold is useful for an intuitive
    # comparison, but 70% is still an arbitrary cutoff.

    # The dataset also contains predefined swipe behavior labels.
    # These provide a broader categorical comparison of matching outcomes.
    swipe_summary = summarize_matches(
        data,
        "swipe_right_label",
    )

    print("\nSwipe Behavior Summary:")
    print(swipe_summary)

    # To avoid relying only on an arbitrary 70% threshold, I also divide
    # users into approximately equal-sized groups according to their actual
    # swipe_right_ratio. This provides a more detailed view across the full
    # distribution of swipe behavior.
    swipe_decile_summary = summarize_swipe_deciles(data)

    print("\nMatching Outcomes Across Swipe-Right Ratio Groups:")
    print(swipe_decile_summary.to_string(index=False))

    # A line plot is appropriate here because the groups are ordered from
    # lower to higher swipe-right behavior. This makes it easier to assess
    # whether average mutual matches consistently increase as users swipe
    # right more frequently.
    plt.figure(figsize=(8, 5))

    plt.plot(
        swipe_decile_summary["average_swipe_ratio"],
        swipe_decile_summary["average_matches"],
        marker="o",
    )

    plt.xlabel("Average Swipe Right Ratio")
    plt.ylabel("Average Mutual Matches")
    plt.title("Swipe-Right Behavior and Average Mutual Matches")
    plt.tight_layout()

    plt.show()

    # Interpretation:
    # Average mutual matches remain relatively stable across swipe-right
    # ratio groups instead of steadily increasing as swipe-right behavior
    # increases. Therefore, higher swipe-right activity does not show a
    # clear monotonic relationship with matching outcomes in this
    # synthetic dataset.


def analyze_app_usage(data):
    """Compare matching outcomes across app-usage groups."""

    # Swiping is not the only behavior that may relate to matching
    # outcomes. App usage time provides another measure of user activity,
    # so I compare mutual matches across the predefined usage groups.
    usage_summary = summarize_matches(
        data,
        "app_usage_time_label",
    )

    print("\nApp Usage Summary:")
    print(usage_summary)

    # Similar to the swipe behavior results, differences in average
    # matching outcomes across app-usage groups are relatively small.


def plot_model_predictions(y_test, predictions):
    """Plot actual mutual matches against predicted values."""

    # Both actual and predicted mutual matches are numerical variables,
    # so a scatter plot provides a direct visual assessment of predictive
    # accuracy. The diagonal reference line represents perfect prediction.
    # Points farther from the line correspond to larger prediction errors.

    plt.figure(figsize=(8, 6))

    plt.scatter(
        y_test,
        predictions,
        alpha=0.3,
    )

    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        linestyle="--",
    )

    plt.xlabel("Actual Mutual Matches")
    plt.ylabel("Predicted Mutual Matches")
    plt.title("Random Forest: Actual vs. Predicted Mutual Matches")
    plt.tight_layout()

    plt.show()


def plot_feature_importance(model):
    """Display and visualize Random Forest feature importance."""

    feature_importance = pd.DataFrame(
        {
            "Feature": FEATURES,
            "Importance": model.feature_importances_,
        }
    ).sort_values(
        "Importance",
        ascending=False,
    )

    print("\nFeature Importance:")
    print(feature_importance)

    # Feature importance assigns one numerical score to each predictor.
    # A horizontal bar chart makes it easy to compare their relative
    # contributions and keeps longer feature names readable.
    plt.figure(figsize=(8, 5))

    plt.barh(
        feature_importance["Feature"],
        feature_importance["Importance"],
    )

    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.title("Random Forest Feature Importance")

    # Put the feature with the largest importance at the top.
    plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.show()

    # Interpretation:
    # likes_received is the most important feature in this fitted model,
    # followed by variables such as bio_length and app_usage_time_min.
    #
    # Feature importance describes how much the trained model relies on
    # each variable for prediction. It should not be interpreted as proof
    # that these variables causally increase mutual matches.


def analyze_model(data):
    """Train and evaluate the Random Forest regression model."""

    # I use Random Forest regression because the target, mutual_matches,
    # is numerical and the behavioral/profile predictors may relate to
    # matching outcomes in nonlinear ways. Random Forest can capture such
    # patterns without requiring a strictly linear relationship.
    model, y_test, predictions, metrics = train_and_evaluate(
        data,
        n_estimators=100,
    )

    print("\nModel Performance:")
    print("Mean Absolute Error:", metrics["mae"])
    print("R-squared:", metrics["r2"])

    # Interpretation:
    # The MAE is approximately 7.22, meaning that predictions differ from
    # observed mutual matches by roughly seven matches on average.
    #
    # The R-squared value is approximately 0.10, meaning the selected
    # predictors explain only a small portion of the variation in mutual
    # matches. The model therefore has limited predictive power.
    #
    # This is consistent with the relatively weak patterns observed in
    # the exploratory analysis and with the synthetic nature of the data.

    comparison = pd.DataFrame(
        {
            "Actual": y_test,
            "Predicted": predictions,
        }
    )

    print("\nSample Predictions:")
    print(comparison.head(10))

    plot_model_predictions(
        y_test,
        predictions,
    )

    plot_feature_importance(model)


def compare_pandas_and_polars(path):
    """Compare equivalent operations using Pandas and Polars."""

    pandas_data = pd.read_csv(path)
    polars_data = pl.read_csv(path)

    # Run equivalent filtering and grouping tasks in each library so that
    # their syntax, outputs, and execution times can be compared.

    pandas_start = perf_counter()

    pandas_high_swipe = pandas_data[pandas_data["swipe_right_ratio"] > 0.70]

    pandas_swipe_summary = (
        pandas_data.groupby("swipe_right_label")["mutual_matches"]
        .agg(
            average_matches="mean",
            median_matches="median",
            number_of_users="count",
        )
        .sort_index()
    )

    pandas_usage_summary = (
        pandas_data.groupby("app_usage_time_label")["mutual_matches"]
        .agg(
            average_matches="mean",
            median_matches="median",
            number_of_users="count",
        )
        .sort_index()
    )

    pandas_time = perf_counter() - pandas_start

    polars_start = perf_counter()

    polars_high_swipe = polars_data.filter(pl.col("swipe_right_ratio") > 0.70)

    polars_swipe_summary = (
        polars_data.group_by("swipe_right_label")
        .agg(
            pl.col("mutual_matches").mean().alias("average_matches"),
            pl.col("mutual_matches").median().alias("median_matches"),
            pl.col("mutual_matches").count().alias("number_of_users"),
        )
        .sort("swipe_right_label")
    )

    polars_usage_summary = (
        polars_data.group_by("app_usage_time_label")
        .agg(
            pl.col("mutual_matches").mean().alias("average_matches"),
            pl.col("mutual_matches").median().alias("median_matches"),
            pl.col("mutual_matches").count().alias("number_of_users"),
        )
        .sort("app_usage_time_label")
    )

    polars_time = perf_counter() - polars_start

    print("\nPandas High-Swipe Users:")
    print(
        pandas_high_swipe[
            [
                "swipe_right_ratio",
                "likes_received",
                "mutual_matches",
            ]
        ].head()
    )

    print("\nPandas Swipe Summary:")
    print(pandas_swipe_summary)

    print("\nPandas Usage Summary:")
    print(pandas_usage_summary)

    print("\nPolars High-Swipe Users:")
    print(
        polars_high_swipe.select(
            [
                "swipe_right_ratio",
                "likes_received",
                "mutual_matches",
            ]
        ).head()
    )

    print("\nPolars Swipe Summary:")
    print(polars_swipe_summary)

    print("\nPolars Usage Summary:")
    print(polars_usage_summary)

    print("\nPerformance Comparison:")
    print(f"Pandas execution time: {pandas_time:.6f} seconds")
    print(f"Polars execution time: {polars_time:.6f} seconds")

    # Reflection:
    # After performing equivalent manipulation and aggregation tasks in
    # Pandas and Polars, I noticed several differences.
    #
    # Polars displays the data type directly beneath each column name in
    # its default table output, making the structure immediately visible.
    # Its output is more compact and text-based, while Pandas integrates
    # naturally with Jupyter Notebook's HTML-style tables and is visually
    # easier to inspect interactively.
    #
    # Execution time is measured above rather than assuming one library is
    # always faster. In my previous run on this dataset, Polars completed
    # the tested filtering and grouping operations more quickly.


def main():
    """Run the complete exploratory analysis."""

    data = load_data(DATA_PATH)

    # Part 1: Understand the dataset and evaluate data quality.
    inspect_dataset(data)
    assess_data_quality(data)

    # Part 2: Explore the main research question about swipe behavior.
    plot_swipe_distribution(data)
    analyze_swipe_behavior(data)

    # Part 3: Examine another measure of user engagement.
    analyze_app_usage(data)

    # Part 4: Evaluate whether multiple behavioral/profile variables
    # can jointly predict matching outcomes.
    analyze_model(data)

    # Part 5: Compare two dataframe libraries used for similar tasks.
    compare_pandas_and_polars(DATA_PATH)


if __name__ == "__main__":
    main()
