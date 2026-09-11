
#This project will use packages of pandas, matplotlib, scikit-learn, and polars.
#If it's not installed, run "pip install" + package_name in the terminal to install it.

#Import libraries
import pandas as pd
import matplotlib.pyplot as plt


# Import and read the Dataset
df = pd.read_csv("dating_app_behavior_dataset.csv")

# Inspection
# Display the first five rows of the dataset
df.head()

# Display the shape of the dataset
df.shape
# This dataset has 50000 rows and 19 columns.

# Check missing values
df.isnull().sum()
# Check duplicated values
df.duplicated().sum()
# There's no missing or duplicated values to be cleaned. We can start to look into the contents of variables.

# Display the data types of each column in the dataset
# I chose this method instead of df.info because this table specifies the data type of each column correspondingly.
dtype_table = df.dtypes.reset_index()
dtype_table.columns = ['Column', 'Data Type']
print(dtype_table)
# We can understand the variables by grouping them into 3 categories: 1) Demographic characteristics for columns 0-5, 2) App behavior for columns 6-10 and 12-17, and 3) Matching outcome variables for columns 11 & 18.

# Display the summary statistics of numerical variables in the dataset
df.describe()

# Inspect categorical variables
# Group them together first
categorical_cols = df.select_dtypes(include="object").columns

# Display the categories contained in each variable
for col in categorical_cols:
    print(f"\n{col}:")
    print(df[col].unique())
# Most of them don't have unique values, except for the interest_tags (which contains a number of combinations of three interest categories). To keep the analysis focused, we will not include this variable in the analysis for now.

# After inspection, now we can start the real analysis.
# Key Question: How does swiping behavior relate to matching outcomes?
# Most of the dating apps use subscription-based business models. They provide users with a limited number of swipes per day, and if they want to swipe more, they have to pay for a subscription. 
# This means that companies need to sell the idea that swiping more will increase the chances of matching with someone to users to keep them engaged and willing to pay for a subscription.
# Therefore, it is essential to understand if the relationship actually holds.
# Note: I planned to look into demographic charactersitics as well, but since this dataset is not a real-world dataset but a simulated one, I decided to drop this part in the analysis onforth.

# Plot the distribution of swipe_right_ratio to see how users behave in general
# Justification:
# swipe_right_ratio is a continuous numerical variable ranging from 0 to 1, so a histogram is appropriate for examining its overall distribution. 
# Dividing the values into intervals allows me to see whether users are concentrated around particular swipe-right rates, whether the distribution is skewed, and whether there are unusually common or uncommon ranges.

plt.figure(figsize=(8, 5))

plt.hist(
    df["swipe_right_ratio"],
    bins=20,
    edgecolor="black"
)

plt.xlabel("Swipe Right Ratio")
plt.ylabel("Number of Users")
plt.title("Distribution of Swipe Right Ratio")

plt.show()

# Define the high_swipe_users by filtering users who swipe right more than 70% of the time
high_swipe_users = df[df["swipe_right_ratio"] > 0.70]

# Display the shape of the filtered dataset (i.e., the number of high_swipe users)
high_swipe_users.shape
# 7700 in total, which is 15.4% of the total users.

# Calculate the mean of mutual_matches for high swipe users and compare it with the overall mean
high_swipe_mean = high_swipe_users["mutual_matches"].mean()
overall_mean = df["mutual_matches"].mean()
print(high_swipe_mean - overall_mean)
# The mean of mutual_matches for high swipe users is 0.026 higher than the overall mean, which is a very small difference. 

# We can also use the prexisting labels of swipe_right_label to group the data and get a full picture of the matching outcomes across groups with different swipe behaviors.
swipe_summary = (
    df.groupby("swipe_right_label")["mutual_matches"]
    .agg(
        average_matches="mean",
        median_matches="median",
        number_of_users="count"
    )
)
swipe_summary
# There isn't a big variance in the mutual matches outcome within different swipe behavior groups, besides choosy swipers have a lower mean (13.65) than other groups (13.87, 13.90, 13.88).


# Swiping is not the only behavior that can affect matching outcomes. App usage time is another important factor to consider. Similarly, we can use the prexisting labels of app_usage_time_label to group the data. 
# Compare matching outcomes across app usage levels
usage_summary = (
    df.groupby("app_usage_time_label")["mutual_matches"]
    .agg(
        average_matches="mean",
        median_matches="median",
        number_of_users="count"
    )
)
usage_summary


# Machine Learning Model - Random Forest Regression
# I chose a Random Forest regression model to predict the number of mutual matches. 
# Random Forest is suitable for this task because the dataset contains several numerical behavioral and profile variables that may relate to matching outcomes in nonlinear ways.
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Select numerical predictor variables
features = [
    "app_usage_time_min",
    "swipe_right_ratio",
    "likes_received",
    "message_sent_count",
    "emoji_usage_rate",
    "profile_pics_count",
    "bio_length"
]

# X contains the predictor variables
X = df[features]

# y is the outcome we want to predict
y = df["mutual_matches"]

# Split the data into training and testing sets
# 80% of the data is used to train the model
# 20% is used to evaluate its performance
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create the Random Forest regression model
# n_estimators=100 saying that the model builds 100 decision trees
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train the model using the training data
rf_model.fit(X_train, y_train)

# Use the trained model to predict mutual matches
y_pred_rf = rf_model.predict(X_test)

# Evaluate model performance
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("Random Forest Mean Absolute Error:", mae_rf)
print("Random Forest R-squared:", r2_rf)
# Interpretation: The mean absolute error (MAE) is 7.22, which means that on average, the model's predictions are off by about 7.22 mutual matches. 
# The R-squared value is 0.1, indicating that the model explains about 10% of the variance in mutual matches. 
# The explanatory power of this model is pretty limited, there are likely other factors influencing mutual matches that are not included in the model.
# This is consistent with the weak relationships observed in the exploratory analysis and the synthetic nature of the dataset.

# Compare actual and predicted values
comparison = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred_rf
})

print("\nSample predictions:")
print(comparison.head(10))


# Visualization 1: Actual vs. Predicted Values
# Justification:
# A scatter plot is appropriate because both the actual and predicted numbers of mutual matches are numerical variables. 
# Plotting each observation's actual value against its prediction allows me to visually assess model accuracy. 
# The diagonal reference line represents perfect predictions: points close to the line indicate accurate predictions, while points far from it indicate larger prediction errors.
plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred_rf,
    alpha=0.3
)

# Add a reference line showing perfect predictions
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)

plt.xlabel("Actual Mutual Matches")
plt.ylabel("Predicted Mutual Matches")
plt.title("Random Forest: Actual vs. Predicted Mutual Matches")

plt.show()


# Feature Importance

feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": rf_model.feature_importances_
}).sort_values(
    "Importance",
    ascending=False
)

print("\nFeature Importance:")
print(feature_importance)
#Interpretation: Likes received is the most important feature in predicting mutual matches, followed by bio length and app usage time. 
# This suggests that users who receive more likes, has longer biography, and spend more time on the app are more likely to have mutual matches.

# Visualization 2: Random Forest Feature Importance
# Justification:
# Feature importance consists of one numerical importance score for each categorical feature name, so a horizontal bar chart makes it easy to compare the relative contribution of predictors.

plt.figure(figsize=(8, 5))

plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Random Forest Feature Importance")

# Put the most important feature at the top
plt.gca().invert_yaxis()

plt.show()


# ---------------------------------------------
# Use of Polars compared to Pandas 
import polars as pl

# Load dataset
pl_df = pl.read_csv("dating_app_behavior_dataset.csv")

# Inspect the dataset in the same way as we did with pandas
print(pl_df.head())
print(pl_df.shape)

print("\nData types:")
print(pl_df.schema)

print("\nSummary statistics:")
print(pl_df.describe())

print("\nMissing values:")
print(pl_df.null_count())

# I noticed that the tables produced by Polars are more compact and in a text/terminal style than those produced by Pandas.

# Filtering high swipe users using Polars
high_swipe_users_pl = pl_df.filter(
    pl.col("swipe_right_ratio") > 0.70
)

print(
    high_swipe_users_pl.select(
        [
            "swipe_right_ratio",
            "likes_received",
            "mutual_matches"
        ]
    ).head()
)

# Grouping
swipe_summary_pl = (
    pl_df
    .group_by("swipe_right_label")
    .agg(
        pl.col("mutual_matches")
        .mean()
        .alias("average_matches"),

        pl.col("mutual_matches")
        .median()
        .alias("median_matches"),

        pl.col("mutual_matches")
        .count()
        .alias("number_of_users")
    )
    .sort("swipe_right_label")
)

print(swipe_summary_pl)

usage_summary_pl = (
    pl_df
    .group_by("app_usage_time_label")
    .agg(
        pl.col("mutual_matches")
        .mean()
        .alias("average_matches"),

        pl.col("mutual_matches")
        .median()
        .alias("median_matches"),

        pl.col("mutual_matches")
        .count()
        .alias("number_of_users")
    )
    .sort("app_usage_time_label")
)

print(usage_summary_pl)

# Reflection on Pandas vs. Polars:
# After performing similar data manipulation and aggregation tasks in both Pandas and Polars, I noticed several differences between the two libraries. 
# Polars consistently displays the data type of each variable directly under the column name in its table output, which makes the structure of the data immediately visible. 
# However, its default table display is more text-based, while Pandas integrates more naturally with Jupyter Notebook's HTML-style table display and is visually easier to read. 
# In my performance comparison, Polars also had a shorter execution time than Pandas for the same filtering and grouping operations. 