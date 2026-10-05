import os
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
# 1. SETTINGS

TRAIN_FILE = "train-test.csv"
VALIDATION_FILE = "validation.csv"
DECEMBER_FILE = "december-chart-inputs.csv"

VALIDATION_OUTPUT = "validation_predictions.csv"
DECEMBER_OUTPUT = "december-chart-inputs.csv"

RANDOM_STATE = 42


# 2. LOAD DATA

print("=" * 70)
print("LOADING DATA")
print("=" * 70)

df = pd.read_csv(TRAIN_FILE)

print("Training shape:", df.shape)
print("Columns:", df.columns.tolist())


# 3. BASIC DATA QUALITY CHECK

print("\n" + "=" * 70)
print("DATA QUALITY CHECK")
print("=" * 70)

print("Duplicate load IDs:", df["load_id"].duplicated().sum())
print("Duplicate rows:", df.duplicated().sum())

print("\nMissing values:")
print(df.isnull().sum())

# Remove duplicate rows if they exist
if df.duplicated().sum() > 0:
    df = df.drop_duplicates().reset_index(drop=True)

# Remove duplicate load IDs if they exist
if df["load_id"].duplicated().sum() > 0:
    df = df.drop_duplicates(subset=["load_id"]).reset_index(drop=True)


# 4. DATE PROCESSING


df["date"] = pd.to_datetime(df["date"])

print("\nDate range:")
print(df["date"].min(), "to", df["date"].max())


# 5. FEATURE ENGINEERING

def create_features(data):

    data = data.copy()

    # Date
    data["date"] = pd.to_datetime(data["date"])

    data["year"] = data["date"].dt.year
    data["month"] = data["date"].dt.month
    data["day_of_week"] = data["date"].dt.dayofweek
    data["day_of_month"] = data["date"].dt.day
    data["week_of_year"] = data["date"].dt.isocalendar().week.astype(int)
    data["day_of_year"] = data["date"].dt.dayofyear
    data["is_weekend"] = (
        data["day_of_week"] >= 5
    ).astype(int)

    # Distance transformation
    data["distance_log"] = np.log1p(
        data["distance"].clip(lower=0)
    )

    # Route feature
    data["route"] = (
        data["pickup"].astype(str)
        + " -> "
        + data["delivery"].astype(str)
    )

    return data


df = create_features(df)


# ============================================================
# 6. FEATURES
#
# IMPORTANT:
# December does not contain:
# market_index
# quote_signal
# pickup_lat
# pickup_lon
# delivery_lat
# delivery_lon
# ============================================================

target = "posted_rate"

categorical_features = [
    "pickup",
    "delivery",
    "equipment",
    "route"
]

numeric_features = [
    "distance",
    "distance_log",
    "weight",
    "year",
    "month",
    "day_of_week",
    "day_of_month",
    "week_of_year",
    "day_of_year",
    "is_weekend"
]

feature_columns = categorical_features + numeric_features


X = df[feature_columns]
y = df[target]

# 7. TIME-BASED TRAIN / VALIDATION SPLIT
#
# January - September = training
# October = validation

print("\n" + "=" * 70)
print("TIME-BASED VALIDATION")
print("=" * 70)

train_mask = df["date"] < "2025-10-01"
valid_mask = df["date"] >= "2025-10-01"

X_train = df.loc[train_mask, feature_columns]
y_train = df.loc[train_mask, target]

X_valid = df.loc[valid_mask, feature_columns]
y_valid = df.loc[valid_mask, target]

print("Training rows:", len(X_train))
print("Validation rows:", len(X_valid))

print(
    "Training period:",
    df.loc[train_mask, "date"].min(),
    "to",
    df.loc[train_mask, "date"].max()
)

print(
    "Validation period:",
    df.loc[valid_mask, "date"].min(),
    "to",
    df.loc[valid_mask, "date"].max()
)


# 8. PREPROCESSING

numeric_transformer = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    )
])

categorical_transformer = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "onehot",
        OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    )
])

preprocessor = ColumnTransformer([
    (
        "num",
        numeric_transformer,
        numeric_features
    ),
    (
        "cat",
        categorical_transformer,
        categorical_features
    )
])



# 9. MODELS TO COMPARE

models = {

    "HGB_1": HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=RANDOM_STATE
    ),

    "HGB_2": HistGradientBoostingRegressor(
        max_iter=400,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=RANDOM_STATE
    ),

    "HGB_3": HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.03,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=RANDOM_STATE
    ),

    "HGB_4": HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=63,
        l2_regularization=1.0,
        random_state=RANDOM_STATE
    ),

    "HGB_5": HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=3.0,
        random_state=RANDOM_STATE
    )
}

# 10. TRAIN AND COMPARE MODELS

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

results = []
trained_pipelines = {}

for name, model in models.items():

    print("\n" + "-" * 70)
    print("Training:", name)
    print("-" * 70)

    # IMPORTANT:
    # Create a new Pipeline for each model.
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    # Train
    pipeline.fit(
        X_train,
        y_train
    )

    # Predict
    predictions = pipeline.predict(X_valid)

    # Make sure predictions are positive
    predictions = np.maximum(
        predictions,
        1
    )

    # Metrics
    mae = mean_absolute_error(
        y_valid,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_valid,
            predictions
        )
    )

    r2 = r2_score(
        y_valid,
        predictions
    )

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    trained_pipelines[name] = pipeline

    print(f"MAE : ${mae:,.2f}")
    print(f"RMSE: ${rmse:,.2f}")
    print(f"R²  : {r2:.4f}")


# 11. MODEL COMPARISON TABLE

results_df = pd.DataFrame(results)

# Sort by MAE
results_df = results_df.sort_values(
    by="MAE",
    ascending=True
).reset_index(drop=True)

print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        formatters={
            "MAE": "{:,.2f}".format,
            "RMSE": "{:,.2f}".format,
            "R2": "{:.4f}".format
        }
    )
)


# Save comparison
results_df.to_csv(
    "model_comparison.csv",
    index=False
)

print("\nSaved: model_comparison.csv")

# 12. SELECT BEST MODEL

best_model_name = results_df.iloc[0]["Model"]

best_pipeline = trained_pipelines[
    best_model_name
]

print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)

print("Selected model:", best_model_name)

best_predictions = best_pipeline.predict(
    X_valid
)

best_predictions = np.maximum(
    best_predictions,
    1
)

best_mae = mean_absolute_error(
    y_valid,
    best_predictions
)

best_rmse = np.sqrt(
    mean_squared_error(
        y_valid,
        best_predictions
    )
)

best_r2 = r2_score(
    y_valid,
    best_predictions
)

print(f"MAE : ${best_mae:,.2f}")
print(f"RMSE: ${best_rmse:,.2f}")
print(f"R²  : {best_r2:.4f}")


# 13. ERROR ANALYSIS


print("\n" + "=" * 70)
print("LARGEST VALIDATION ERRORS")
print("=" * 70)

error_analysis = pd.DataFrame({
    "actual": y_valid.values,
    "predicted": best_predictions
})

error_analysis["error"] = (
    error_analysis["actual"]
    - error_analysis["predicted"]
)

error_analysis["absolute_error"] = (
    error_analysis["error"].abs()
)

error_analysis["percentage_error"] = (
    error_analysis["absolute_error"]
    / error_analysis["actual"].replace(0, np.nan)
    * 100
)

print(
    error_analysis
    .sort_values(
        "absolute_error",
        ascending=False
    )
    .head(20)
    .to_string(index=False)
)

error_analysis.to_csv(
    "validation_error_analysis.csv",
    index=False
)

print("\nSaved: validation_error_analysis.csv")


# 14. TRAIN FINAL MODEL ON ALL DEVELOPMENT DATA

print("\n" + "=" * 70)
print("TRAINING FINAL MODEL ON ALL DEVELOPMENT DATA")
print("=" * 70)

X_all = df[feature_columns]
y_all = df[target]

# Create a completely new model using the best hyperparameters
final_model = models[best_model_name]

final_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", final_model)
])

final_pipeline.fit(
    X_all,
    y_all
)

print("Final model trained on:", len(X_all), "rows")


# 15. LOAD VALIDATION DATA

print("\n" + "=" * 70)
print("PREDICTING VALIDATION DATA")
print("=" * 70)

validation = pd.read_csv(
    VALIDATION_FILE
)

print("Validation shape:", validation.shape)

# Create same features
validation_features = create_features(
    validation
)

X_test = validation_features[
    feature_columns
]

# Predict
validation_predictions = final_pipeline.predict(
    X_test
)

validation_predictions = np.maximum(
    validation_predictions,
    1
)


# 16. CREATE validation_predictions.csv

validation_output = pd.DataFrame({
    "load_id": validation["load_id"],
    "predicted_rate": validation_predictions
})

# Ensure correct order
validation_output = validation_output[
    ["load_id", "predicted_rate"]
]

validation_output.to_csv(
    VALIDATION_OUTPUT,
    index=False
)

print(
    f"Saved {VALIDATION_OUTPUT}"
)

print(
    "Rows:",
    len(validation_output)
)

print("\nSample validation predictions:")
print(
    validation_output.head(10)
)
# 17. CHECK VALIDATION OUTPUT

print("\n" + "=" * 70)
print("VALIDATION OUTPUT CHECK")
print("=" * 70)

print(
    "Expected rows:",
    len(validation)
)

print(
    "Output rows:",
    len(validation_output)
)

print(
    "Duplicate IDs:",
    validation_output["load_id"].duplicated().sum()
)

print(
    "Missing predictions:",
    validation_output["predicted_rate"].isna().sum()
)

print(
    "Non-positive predictions:",
    (validation_output["predicted_rate"] <= 0).sum()
)

print(
    "Minimum prediction:",
    validation_output["predicted_rate"].min()
)

print(
    "Maximum prediction:",
    validation_output["predicted_rate"].max()
)



# 18. DECEMBER PREDICTIONS


print("\n" + "=" * 70)
print("PREDICTING DECEMBER DATA")
print("=" * 70)

december = pd.read_csv(
    DECEMBER_FILE
)

print("December shape:", december.shape)

print("\nDecember columns:")
print(december.columns.tolist())

# Create features
december_features = create_features(
    december
)

X_december = december_features[
    feature_columns
]

# Predict
december_predictions = final_pipeline.predict(
    X_december
)

december_predictions = np.maximum(
    december_predictions,
    1
)


# 19. UPDATE DECEMBER FILE

december_output = december.copy()

december_output["predicted_rate"] = (
    december_predictions
)

# Required column order
december_columns = [
    "pickup",
    "delivery",
    "distance",
    "equipment",
    "weight",
    "date",
    "predicted_rate"
]

december_output = december_output[
    december_columns
]

december_output.to_csv(
    DECEMBER_OUTPUT,
    index=False
)

print(
    f"\nSaved December predictions to:"
    f" {DECEMBER_OUTPUT}"
)

print(
    "December rows:",
    len(december_output)
)

print("\nDecember predictions:")
print(
    december_output[
        ["date", "predicted_rate"]
    ].to_string(index=False)
)

# 20. FINAL CHECKS

print("\n" + "=" * 70)
print("FINAL CHECKS")
print("=" * 70)

print("\nValidation predictions:")
print(
    validation_output.head()
)

print("\nDecember predictions:")
print(
    december_output.head()
)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

print(
    f"\nBest model: {best_model_name}"
)

print(
    f"Validation MAE: ${best_mae:,.2f}"
)

print(
    f"Validation RMSE: ${best_rmse:,.2f}"
)

print(
    f"Validation R²: {best_r2:.4f}"
)

print(
    "\nFiles created:"
)

print("1. validation_predictions.csv")
print("2. december_chart_inputs.csv")
print("3. model_comparison.csv")
print("4. validation_error_analysis.csv")